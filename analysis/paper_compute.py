"""Paper-facing exact-support profiling and symbolic FLOP accounting.

The runtime profiler advances the checkpoint with the canonical production
minimal kernels.  It only observes address-stage selection margins and
gate weights, before the high-dimensional read/write application.  This
keeps the measured support definition aligned with the exact conditional
execution boundary without claiming that the current dense kernel skips work.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import analysis_model_module
from analysis.operator_interpretability.benchmark_schema import canonical_hash


ROUTES = ("q", "k", "qk_union", "v", "rst")
DEFAULT_EPSILON_THRESHOLDS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)


def threshold_id(index: int, epsilon_thresholds: Sequence[float]) -> str:
    if int(index) == 0:
        return "exact_margin_gt_0"
    return f"gate_weight_gt_{float(epsilon_thresholds[index - 1]):.0e}"


def threshold_record(
        index: int, epsilon_thresholds: Sequence[float]) -> dict[str, Any]:
    if int(index) == 0:
        return {
            "threshold_id": threshold_id(index, epsilon_thresholds),
            "kind": "exact",
            "criterion": "selection_margin > 0",
            "epsilon": None,
            "output_equivalence": "mathematically exact",
        }
    epsilon = float(epsilon_thresholds[index - 1])
    return {
        "threshold_id": threshold_id(index, epsilon_thresholds),
        "kind": "approximate",
        "criterion": f"gate_weight > {epsilon:.0e}",
        "epsilon": epsilon,
        "output_equivalence": "not asserted; approximate support threshold",
    }


def _transition_output(result: Any, *, name: str, expected_ndim: int):
    output = result[0] if isinstance(result, tuple) else result
    if not hasattr(output, "ndim") or int(output.ndim) != int(expected_ndim):
        raise ValueError(
            f"{name} returned transition shape={getattr(output, 'shape', None)}; "
            f"expected rank {expected_ndim}")
    return output


def _address_support(
        operator_query, operator_keys, raw_tau, *, model_module,
        execution_kwargs: Mapping[str, Any]):
    """Return production-precision selection margin and gate weight."""
    kwargs = dict(execution_kwargs)
    for key in (
            "gate_den_power", "gate_den_power_qk",
            "gate_den_power_v", "gate_den_power_rst",
            "admission_den_power", "admission_den_power_qk",
            "admission_den_power_v", "admission_den_power_rst"):
        kwargs.pop(key, None)
    query_direction = model_module._forward_unit_direction(
        operator_query.astype(jnp.bfloat16).astype(jnp.float32)).astype(
            jnp.bfloat16)
    key_directions = model_module._forward_unit_direction(
        operator_keys.astype(jnp.bfloat16).astype(jnp.float32)).astype(
            jnp.bfloat16)
    score = (query_direction @ key_directions.T).astype(jnp.float32)
    tau = model_module._tau_from_param(raw_tau).astype(jnp.float32)
    temperature = kwargs.pop("soft_gate_temperature")
    boundary_power = kwargs.pop("soft_gate_boundary_power")
    effective_active_eps = kwargs.pop("soft_gate_effective_active_eps")
    if hasattr(model_module, "_compute_gate_weight"):
        selection_margin, gate_weight, _, _ = (
            model_module._compute_gate_weight(
                score,
                tau,
                **kwargs,
            ))
    else:
        selection_margin, gate_weight, _, _, _ = (
            model_module._compute_admission_drive(
                score,
                tau,
                temperature,
                boundary_power=boundary_power,
                effective_active_eps=effective_active_eps,
                **kwargs,
            ))
    return selection_margin, gate_weight


def _legacy_kernel_tail(model_module, gate_kwargs):
    """Supply the retired prune scalar only to older module signatures."""
    if hasattr(model_module, "_compute_gate_weight"):
        return ()
    return (gate_kwargs.get("execution_prune_eps", 0.0),)


def _support_counts(
        selection_margin, gate_weight, epsilon_thresholds):
    exact = (selection_margin > jnp.float32(0.0)).sum(
        axis=-1, dtype=jnp.int32)
    eps = jnp.asarray(epsilon_thresholds, dtype=jnp.float32)
    approximate = jax.vmap(
        lambda value: (gate_weight > value).sum(axis=-1, dtype=jnp.int32)
    )(eps)
    return jnp.concatenate((exact[jnp.newaxis, ...], approximate), axis=0)


def _union_support_counts(
        q_margin, q_gate_weight, k_margin, k_gate_weight,
        epsilon_thresholds):
    exact = jnp.logical_or(
        q_margin > jnp.float32(0.0),
        k_margin > jnp.float32(0.0),
    ).sum(axis=-1, dtype=jnp.int32)
    eps = jnp.asarray(epsilon_thresholds, dtype=jnp.float32)
    approximate = jax.vmap(
        lambda value: jnp.logical_or(
            q_gate_weight > value, k_gate_weight > value
        ).sum(axis=-1, dtype=jnp.int32)
    )(eps)
    return jnp.concatenate((exact[jnp.newaxis, ...], approximate), axis=0)


def _count_histograms(counts, pool_size: int):
    width = int(pool_size) + 1
    return jax.vmap(
        lambda values: jnp.bincount(
            values.reshape(-1), length=width).astype(jnp.int32)
    )(counts)


def support_profile_forward(
        params, model_cfg: Mapping[str, Any], input_ids, *,
        epsilon_thresholds: Sequence[float] = DEFAULT_EPSILON_THRESHOLDS,
        production_srw_fns: Mapping[str, Any]):
    """Advance one packed batch and return per-layer support histograms.

    Histogram bins are active-operator counts.  No per-token gate, margin, or
    operator-id vector is returned or persisted.
    """
    model_module = analysis_model_module(dict(model_cfg))
    required = {
        "attn_qk_paired_minimal", "attn_v_single_minimal",
        "rst_single_minimal",
    }
    if required - set(production_srw_fns):
        raise ValueError(
            "support profiling requires canonical production minimal kernels: "
            + ",".join(sorted(required)))
    if not epsilon_thresholds:
        raise ValueError("at least one approximate epsilon threshold is required")
    if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in epsilon_thresholds):
        raise ValueError("epsilon thresholds must be finite and positive")
    if tuple(sorted(map(float, epsilon_thresholds))) != tuple(
            map(float, epsilon_thresholds)):
        raise ValueError("epsilon thresholds must be sorted ascending")

    params = model_module._squeeze_params(params)
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    batch_size, seq_len = input_ids.shape
    d_model = int(model_cfg["d_model"])
    n_layers = int(model_cfg["n_layers"])
    n_heads = int(model_cfg["n_heads"])
    d_head = d_model // n_heads
    n_qk = int(model_cfg["n_qk"])
    n_v = int(model_cfg["n_v"])
    n_rst = int(model_cfg.get("n_rst", model_cfg.get("n_know")))

    gate_kwargs_builder = getattr(
        model_module, "_angular_gate_kwargs_from_model_cfg", None)
    if gate_kwargs_builder is None:
        gate_kwargs_builder = getattr(
            model_module, "_angular_execution_kwargs_from_model_cfg")
    execution_kwargs = gate_kwargs_builder(model_cfg)
    den_prefix = (
        "gate_den_power"
        if "gate_den_power" in execution_kwargs else "admission_den_power")
    default_power = float(execution_kwargs.get(den_prefix, 1.0))
    powers = {
        route: float(execution_kwargs.get(
            f"{den_prefix}_{route}", default_power))
        for route in ("qk", "v", "rst")
    }

    def execution_for(temperature_key: str) -> dict[str, Any]:
        value = dict(execution_kwargs)
        temperature = model_cfg.get(temperature_key)
        if temperature is not None:
            value["soft_gate_temperature"] = float(temperature)
        return value

    execution_qk = execution_for("soft_gate_T_qk")
    execution_v = execution_for("soft_gate_T_v")
    execution_rst = execution_for("soft_gate_T_rst")

    pool = model_module._pool_params_with_operator_keys(
        params["neuron_pool"], model_cfg.get("operator_key_mode"))
    router = params["router"]
    qk_scale, v_scale, rst_scale = (
        model_module._effective_pool_output_scales(
            pool, d_model, n_layers))
    positions = jnp.arange(seq_len)[jnp.newaxis, :]
    residual_state = (
        params["token_emb"]["embedding"][input_ids]
        + params["pos_emb"]["embedding"][positions])
    histograms = {route: [] for route in ROUTES}

    for layer_idx in range(n_layers):
        block = params[f"block_{layer_idx}"]
        normed = model_module._layer_norm(
            residual_state,
            block["norm1"]["scale"],
            block["norm1"]["bias"],
        )
        operator_queries = (
            normed @ router["proj_attn"]["kernel"]
            + router["proj_attn"]["bias"])
        query_q, query_k, query_v = jnp.split(
            operator_queries, 3, axis=-1)
        query_adapter = getattr(
            model_module, "_read_write_attn_operator_queries", None)
        if query_adapter is not None:
            query_q, query_k, query_v = query_adapter(
                router, normed, query_q, query_k, query_v)
        tau_all = (
            normed @ router["raw_tau_attn"]["kernel"]
            + router["raw_tau_attn"]["bias"])

        q_margin, q_gate_weight = _address_support(
            query_q, pool["attn_qk_op_key"], tau_all[:, :, 0:1],
            model_module=model_module, execution_kwargs=execution_qk)
        k_margin, k_gate_weight = _address_support(
            query_k, pool["attn_qk_op_key"], tau_all[:, :, 1:2],
            model_module=model_module, execution_kwargs=execution_qk)
        v_margin, v_gate_weight = _address_support(
            query_v, pool["attn_v_op_key"], tau_all[:, :, 2:3],
            model_module=model_module, execution_kwargs=execution_v)
        histograms["q"].append(_count_histograms(
            _support_counts(
                q_margin, q_gate_weight, epsilon_thresholds), n_qk))
        histograms["k"].append(_count_histograms(
            _support_counts(
                k_margin, k_gate_weight, epsilon_thresholds), n_qk))
        histograms["qk_union"].append(_count_histograms(
            _union_support_counts(
                q_margin, q_gate_weight, k_margin, k_gate_weight,
                epsilon_thresholds),
            n_qk))
        histograms["v"].append(_count_histograms(
            _support_counts(
                v_margin, v_gate_weight, epsilon_thresholds), n_v))

        paired_queries = jnp.stack((query_q, query_k), axis=2)
        paired_tau = jnp.stack(
            (tau_all[:, :, 0:1], tau_all[:, :, 1:2]), axis=2)
        qk_transitions = _transition_output(
            production_srw_fns["attn_qk_paired_minimal"](
                normed,
                paired_queries,
                pool["attn_qk_op_key"],
                paired_tau,
                pool["attn_qk_read"],
                pool["attn_qk_write"],
                execution_qk["soft_gate_temperature"],
                model_cfg.get(
                    "soft_gate_t_final",
                    execution_qk["soft_gate_temperature"]),
                execution_qk["soft_gate_boundary_power"],
                model_cfg.get(
                    "soft_gate_boundary_power_final",
                    execution_qk["soft_gate_boundary_power"]),
                *_legacy_kernel_tail(model_module, execution_qk),
            ),
            name="attn_qk_paired_minimal",
            expected_ndim=4,
        )
        v_transition = _transition_output(
            production_srw_fns["attn_v_single_minimal"](
                normed,
                query_v,
                pool["attn_v_op_key"],
                tau_all[:, :, 2:3],
                pool["attn_v_read"],
                pool["attn_v_write"],
                execution_v["soft_gate_temperature"],
                model_cfg.get(
                    "soft_gate_t_final",
                    execution_v["soft_gate_temperature"]),
                execution_v["soft_gate_boundary_power"],
                model_cfg.get(
                    "soft_gate_boundary_power_final",
                    execution_v["soft_gate_boundary_power"]),
                *_legacy_kernel_tail(model_module, execution_v),
            ),
            name="attn_v_single_minimal",
            expected_ndim=3,
        )
        attention_q = qk_transitions[:, :, 0, :] * qk_scale
        attention_k = qk_transitions[:, :, 1, :] * qk_scale
        attention_v = v_transition * v_scale
        attention_q = attention_q.reshape(
            batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_k = attention_k.reshape(
            batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_v = attention_v.reshape(
            batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = jnp.einsum(
            "bhsd,bhtd->bhst", attention_q, attention_k
        ) / jnp.sqrt(jnp.float32(d_head))
        causal = jnp.tril(jnp.ones(
            (seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(
            causal, scores, jnp.finfo(scores.dtype).min)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attention_output = jnp.einsum(
            "bhst,bhtd->bhsd", attention_weights, attention_v)
        attention_output = attention_output.transpose(
            0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        residual_state = residual_state + (
            attention_output @ block["attn"]["expand_O"]["kernel"])

        normed = model_module._layer_norm(
            residual_state,
            block["norm2"]["scale"],
            block["norm2"]["bias"],
        )
        query_rst = (
            normed @ router["proj_rst"]["kernel"]
            + router["proj_rst"]["bias"])
        rst_adapter = getattr(
            model_module, "_read_write_rst_operator_query", None)
        if rst_adapter is not None:
            query_rst = rst_adapter(
                router, normed, query_rst)
        tau_rst = (
            normed @ router["raw_tau_rst"]["kernel"]
            + router["raw_tau_rst"]["bias"])
        rst_margin, rst_gate_weight = _address_support(
            query_rst, pool["rst_op_key"], tau_rst,
            model_module=model_module, execution_kwargs=execution_rst)
        histograms["rst"].append(_count_histograms(
            _support_counts(
                rst_margin, rst_gate_weight, epsilon_thresholds), n_rst))
        rst_transition = _transition_output(
            production_srw_fns["rst_single_minimal"](
                normed,
                query_rst,
                pool["rst_op_key"],
                tau_rst,
                pool["rst_read"],
                pool["rst_write"],
                execution_rst["soft_gate_temperature"],
                model_cfg.get(
                    "soft_gate_t_final",
                    execution_rst["soft_gate_temperature"]),
                execution_rst["soft_gate_boundary_power"],
                model_cfg.get(
                    "soft_gate_boundary_power_final",
                    execution_rst["soft_gate_boundary_power"]),
                *_legacy_kernel_tail(model_module, execution_rst),
            ),
            name="rst_single_minimal",
            expected_ndim=3,
        ) * rst_scale
        residual_state = residual_state + rst_transition

    return {
        route: jnp.stack(values, axis=0)
        for route, values in histograms.items()
    }


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> int:
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("cannot compute a quantile from an empty histogram")
    target = max(1, int(math.ceil(float(quantile) * total)))
    return int(np.searchsorted(np.cumsum(histogram), target, side="left"))


def summarize_one_histogram(
        histogram: np.ndarray, pool_size: int) -> dict[str, Any]:
    histogram = np.asarray(histogram, dtype=np.int64)
    if histogram.ndim != 1 or histogram.size != int(pool_size) + 1:
        raise ValueError(
            f"invalid histogram shape={histogram.shape} pool_size={pool_size}")
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("support histogram has no observations")
    occupied = np.flatnonzero(histogram)
    values = np.arange(histogram.size, dtype=np.float64)
    mean = float(np.dot(values, histogram.astype(np.float64)) / total)
    return {
        "observations": total,
        "mean": mean,
        "median": _histogram_quantile(histogram, 0.50),
        "p90": _histogram_quantile(histogram, 0.90),
        "p99": _histogram_quantile(histogram, 0.99),
        "minimum": int(occupied[0]),
        "maximum": int(occupied[-1]),
        "pool_size": int(pool_size),
        "mean_pool_fraction": mean / float(pool_size),
        "no_active_fraction": float(histogram[0]) / total,
    }


def summarize_support_histograms(
        histograms: Mapping[str, np.ndarray], *,
        epsilon_thresholds: Sequence[float] = DEFAULT_EPSILON_THRESHOLDS,
        pool_sizes: Mapping[str, int]) -> dict[str, Any]:
    n_thresholds = 1 + len(epsilon_thresholds)
    layer_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    n_layers = None
    for route in ROUTES:
        values = np.asarray(histograms[route], dtype=np.int64)
        if values.ndim != 3 or values.shape[1] != n_thresholds:
            raise ValueError(
                f"{route} histogram must be [layers,{n_thresholds},bins], "
                f"got {values.shape}")
        n_layers = values.shape[0] if n_layers is None else n_layers
        if values.shape[0] != n_layers:
            raise ValueError("all route histograms must have the same layers")
        pool_size = int(pool_sizes[route])
        if values.shape[2] != pool_size + 1:
            raise ValueError(
                f"{route} histogram bins={values.shape[2]} "
                f"but pool_size={pool_size}")
        for threshold_index in range(n_thresholds):
            threshold = threshold_record(
                threshold_index, epsilon_thresholds)
            aggregate_rows.append({
                "scope": "all_layers",
                "layer": None,
                "route": route,
                **threshold,
                **summarize_one_histogram(
                    values[:, threshold_index, :].sum(axis=0),
                    pool_size),
            })
            for layer in range(values.shape[0]):
                layer_rows.append({
                    "scope": "layer",
                    "layer": int(layer),
                    "route": route,
                    **threshold,
                    **summarize_one_histogram(
                        values[layer, threshold_index, :],
                        pool_size),
                })
    payload = {
        "schema_version": 1,
        "support_definition": {
            "exact": "selection_margin > 0",
            "approximate": "gate_weight > epsilon",
            "epsilon_thresholds": [
                float(value) for value in epsilon_thresholds],
            "qk_union": (
                "operator active for Q or K at the same layer and token"),
        },
        "n_layers": int(n_layers or 0),
        "routes": list(ROUTES),
        "aggregate_rows": aggregate_rows,
        "layer_rows": layer_rows,
        "raw_per_token_vectors_persisted": False,
        "raw_histograms_persisted": False,
    }
    payload["summary_hash"] = canonical_hash(payload)
    return payload


def _config_dimensions(config: Mapping[str, Any]) -> dict[str, int]:
    model = config.get("model", {})
    required = ("d_model", "n_layers", "n_heads")
    missing = [key for key in required if model.get(key) is None]
    if missing:
        raise ValueError(
            "model config lacks FLOP dimensions: " + ",".join(missing))
    return {
        "d_model": int(model["d_model"]),
        "d_ff": int(model.get("d_ff", 4 * int(model["d_model"]))),
        "n_layers": int(model["n_layers"]),
        "n_heads": int(model["n_heads"]),
        "vocab_size": int(model.get("logical_vocab_size",
                                    model.get("vocab_size", 30522))),
        "d_route": int(model.get("d_route",
                                 model.get("d_bottleneck", 0))),
        "n_qk": int(model.get("n_qk", 0)),
        "n_v": int(model.get("n_v", 0)),
        "n_rst": int(model.get("n_rst",
                               model.get("n_know", 0))),
    }


def _exact_layer_support(
        support_summary: Mapping[str, Any], n_layers: int
) -> list[dict[str, float]]:
    exact_rows = {
        (int(row["layer"]), str(row["route"])): row
        for row in support_summary.get("layer_rows", ())
        if row.get("threshold_id") == "exact_margin_gt_0"
        and row.get("layer") is not None
    }
    required_routes = ("q", "k", "qk_union", "v", "rst")
    result = []
    for layer in range(int(n_layers)):
        missing = [
            route for route in required_routes
            if (layer, route) not in exact_rows]
        if missing:
            raise ValueError(
                f"support summary layer={layer} lacks "
                + ",".join(missing))
        result.append({
            route: float(exact_rows[(layer, route)]["mean"])
            for route in required_routes
        })
    return result


def compute_flop_accounting(
        dawn_config: Mapping[str, Any],
        baseline_config: Mapping[str, Any],
        support_summary: Mapping[str, Any], *,
        batch_size: int,
        sequence_length: int,
        dawn_parameter_count: int | None = None,
        baseline_parameter_count: int | None = None,
) -> dict[str, Any]:
    """Count major forward matmul/einsum MACs and FLOPs.

    One MAC is reported as two FLOPs.  Softmax, normalization, gate elementwise
    arithmetic, embedding lookup, residual adds, and collectives are omitted.
    """
    batch_size = int(batch_size)
    sequence_length = int(sequence_length)
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    dawn = _config_dimensions(dawn_config)
    baseline = _config_dimensions(baseline_config)
    if dawn["d_route"] <= 0 or min(
            dawn["n_qk"], dawn["n_v"], dawn["n_rst"]) <= 0:
        raise ValueError("DAWN config lacks route or operator-pool dimensions")
    if baseline["d_ff"] <= 0:
        raise ValueError("baseline config lacks a positive d_ff")

    tokens = batch_size * sequence_length
    d_model = dawn["d_model"]
    d_route = dawn["d_route"]
    n_qk, n_v, n_rst = (
        dawn["n_qk"], dawn["n_v"], dawn["n_rst"])
    layers = dawn["n_layers"]
    exact_support = _exact_layer_support(support_summary, layers)

    route_projection_per_layer = tokens * d_model * (4 * d_route + 4)
    address_scoring_per_layer = (
        tokens * d_route * (2 * n_qk + n_v + n_rst))
    attention_per_layer = (
        2 * batch_size * sequence_length * sequence_length * d_model)
    output_projection_per_layer = tokens * d_model * d_model
    dense_rw_per_layer = (
        tokens * d_model * (3 * n_qk + 2 * n_v + 2 * n_rst))
    exact_rw_layers = [
        tokens * d_model * (
            row["qk_union"] + row["q"] + row["k"]
            + 2 * row["v"] + 2 * row["rst"])
        for row in exact_support
    ]

    dawn_current_macs = {
        "route_and_tau_projections": route_projection_per_layer * layers,
        "full_pool_address_scoring": address_scoring_per_layer * layers,
        "full_pool_rw_application": dense_rw_per_layer * layers,
        "causal_attention": attention_per_layer * layers,
        "attention_output_projection": output_projection_per_layer * layers,
    }
    dawn_exact_macs = {
        "route_and_tau_projections": route_projection_per_layer * layers,
        "full_pool_address_scoring": address_scoring_per_layer * layers,
        "exact_support_rw_application": float(sum(exact_rw_layers)),
        "causal_attention": attention_per_layer * layers,
        "attention_output_projection": output_projection_per_layer * layers,
    }

    base_d = baseline["d_model"]
    base_ff = baseline["d_ff"]
    base_layers = baseline["n_layers"]
    baseline_macs = {
        "qkv_and_output_projections": (
            4 * tokens * base_d * base_d * base_layers),
        "ffn": 2 * tokens * base_d * base_ff * base_layers,
        "causal_attention": (
            2 * batch_size * sequence_length * sequence_length
            * base_d * base_layers),
    }

    def column(macs: Mapping[str, float]) -> dict[str, Any]:
        mac_values = {key: float(value) for key, value in macs.items()}
        flop_values = {
            key: 2.0 * value for key, value in mac_values.items()}
        return {
            "components_macs": mac_values,
            "components_flops": flop_values,
            "total_macs": float(sum(mac_values.values())),
            "total_flops": float(sum(flop_values.values())),
        }

    current = column(dawn_current_macs)
    exact = column(dawn_exact_macs)
    dense = column(baseline_macs)
    key_materialization_macs = (
        2 * (n_qk + n_v + n_rst) * d_model * d_route)
    dawn_lm_head_macs = tokens * d_model * dawn["vocab_size"]
    baseline_lm_head_macs = (
        tokens * base_d * baseline["vocab_size"])
    fixed_exact_macs = (
        route_projection_per_layer
        + address_scoring_per_layer
        + attention_per_layer
        + output_projection_per_layer
    ) * layers

    payload = {
        "schema_version": 1,
        "counting_convention": {
            "mac_to_flops": 2,
            "direction": "forward_only",
            "counted": "major matmul/einsum operations",
            "omitted": [
                "softmax",
                "normalization",
                "gate elementwise arithmetic",
                "embedding lookup",
                "residual additions",
                "communication collectives",
            ],
            "current_dawn_execution": (
                "all addresses and all high-dimensional RW operators execute"),
            "exact_support_execution": (
                "full dense address scan plus only margin>0 RW application"),
            "exact_support_output_equivalence": "mathematically exact",
            "indexed_addressing_included": False,
            "measured_latency_claimed": False,
        },
        "input": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "tokens": tokens,
            "support_summary_hash": support_summary.get("summary_hash"),
            "support_threshold_id": "exact_margin_gt_0",
        },
        "dimensions": {
            "dawn": dawn,
            "baseline": baseline,
        },
        "parameter_counts": {
            "dawn": (
                int(dawn_parameter_count)
                if dawn_parameter_count is not None else None),
            "dense_baseline": (
                int(baseline_parameter_count)
                if baseline_parameter_count is not None else None),
        },
        "columns": {
            "dawn_current_dense_execution": current,
            "dawn_exact_support_estimate": exact,
            "dense_transformer": dense,
        },
        "extras": {
            "dawn_operator_key_materialization_per_model_call": {
                "macs": float(key_materialization_macs),
                "flops": float(2 * key_materialization_macs),
                "amortized_flops_per_sequence": float(
                    2 * key_materialization_macs / batch_size),
                "current_inference_cache_used": False,
                "cacheability_claim": "possible future optimization only",
            },
            "dawn_lm_head": {
                "macs": float(dawn_lm_head_macs),
                "flops": float(2 * dawn_lm_head_macs),
            },
            "dense_baseline_lm_head": {
                "macs": float(baseline_lm_head_macs),
                "flops": float(2 * baseline_lm_head_macs),
            },
        },
        "exact_support_layer_rw_macs": [
            {
                "layer": layer,
                "support_means": exact_support[layer],
                "rw_macs": float(exact_rw_layers[layer]),
                "rw_flops": float(2 * exact_rw_layers[layer]),
            }
            for layer in range(layers)
        ],
        "comparisons": {
            "current_dawn_vs_dense_transformer": (
                current["total_flops"] / dense["total_flops"]),
            "exact_support_vs_current_dawn": (
                exact["total_flops"] / current["total_flops"]),
            "exact_support_vs_dense_transformer": (
                exact["total_flops"] / dense["total_flops"]),
            "exact_support_fixed_cost_only_vs_dense_transformer": (
                (2.0 * fixed_exact_macs) / dense["total_flops"]),
        },
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload
