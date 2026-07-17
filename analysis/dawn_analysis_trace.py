"""Production-faithful sparse contribution capture for train_analysis_pool."""

from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from analysis.dawn_analysis_common import (
    V417X_MODEL_VERSIONS,
    analysis_model_module,
)


TRACE_POOLS = ("q", "k", "v", "rst")
TRACE_FIELDS = ("top_idx", "top_val", "captured_mass")


def _operator_contribution_topk(
        state, operator_query, operator_keys, raw_tau,
        read_vectors, write_vectors, *, model_module, topk: int,
        execution_kwargs: Dict[str, Any], admission_den_power: float,
        target_positions=None) -> dict[str, jax.Array]:
    """Rank pre-cancellation production-precision contribution norms."""
    kwargs = dict(execution_kwargs)
    for key in (
            "admission_den_power", "admission_den_power_qk",
            "admission_den_power_v", "admission_den_power_rst"):
        kwargs.pop(key, None)
    query_direction = model_module._forward_unit_direction(
        operator_query.astype(jnp.bfloat16).astype(jnp.float32)).astype(
            jnp.bfloat16)
    key_directions = model_module._forward_unit_direction(
        operator_keys.astype(jnp.bfloat16).astype(jnp.float32)).astype(
            jnp.bfloat16)
    rho = (query_direction @ key_directions.T).astype(jnp.float32)
    tau = model_module._tau_from_param(raw_tau).astype(jnp.float32)
    drive_kwargs = dict(kwargs)
    temperature = drive_kwargs.pop("soft_gate_temperature")
    boundary_power = drive_kwargs.pop("soft_gate_boundary_power")
    effective_active_eps = drive_kwargs.pop(
        "soft_gate_effective_active_eps")
    _, admission, _, execution_weight, _ = (
        model_module._compute_admission_drive(
            rho, tau, temperature,
            boundary_power=boundary_power,
            effective_active_eps=effective_active_eps,
            **drive_kwargs,
        ))
    read_directions = model_module._forward_unit_direction(
        read_vectors.astype(jnp.bfloat16).astype(jnp.float32)).astype(
            jnp.bfloat16)
    write_directions = model_module._forward_unit_direction(
        write_vectors.astype(jnp.bfloat16).astype(jnp.float32)).astype(
            jnp.bfloat16)
    read_activations = state.astype(jnp.bfloat16) @ read_directions.T
    coefficient = execution_weight * read_activations
    admission_mass = admission.sum(axis=-1, keepdims=True)
    composition_den = getattr(model_module, "_composition_den", None)
    if composition_den is None:
        denominator = jnp.power(
            jnp.maximum(admission_mass, 1.0),
            jnp.asarray(admission_den_power, dtype=jnp.float32),
        )
    elif hasattr(model_module, "DEFAULT_SRW_COMPOSITION_MODE"):
        denominator = composition_den(
            admission_mass,
            admission_den_power,
            kwargs.get(
                "srw_composition_mode",
                model_module.DEFAULT_SRW_COMPOSITION_MODE),
        )
    else:
        denominator = composition_den(admission_mass, admission_den_power)
    production_coefficient = (
        coefficient.astype(jnp.bfloat16).astype(jnp.float32)
        / jnp.maximum(denominator, jnp.float32(1.0e-8)))
    contribution_norm = (
        jnp.abs(production_coefficient)
        * jnp.linalg.norm(write_directions.astype(jnp.float32), axis=-1))
    if target_positions is not None:
        contribution_norm = contribution_norm[
            jnp.arange(contribution_norm.shape[0], dtype=jnp.int32),
            jnp.asarray(target_positions, dtype=jnp.int32),
        ]
    width = min(int(topk), int(contribution_norm.shape[-1]))
    top_val, top_idx = jax.lax.top_k(contribution_norm, width)
    total_mass = contribution_norm.sum(axis=-1)
    return {
        "top_idx": top_idx.astype(jnp.int32),
        "top_val": top_val.astype(jnp.float32),
        "captured_mass": (
            top_val.sum(axis=-1)
            / jnp.maximum(total_mass, jnp.float32(1.0e-8))).astype(
                jnp.float32),
    }


def topk_trace_forward(
        params, model_cfg: Dict[str, Any], input_ids, *, topk: int = 8,
        topk_qk: Optional[int] = None, topk_v: Optional[int] = None,
        topk_rst: Optional[int] = None,
        execution_prune_eps: Optional[float] = None,
        target_positions=None, production_srw_fns=None):
    """Capture sparse contribution mass while advancing production kernels."""
    model_module = analysis_model_module(model_cfg)
    model_version = str(model_cfg.get("model_version", ""))
    if model_version not in V417X_MODEL_VERSIONS:
        raise ValueError(
            f"train_analysis_pool trace does not support {model_version!r}")
    required = {
        "attn_qk_paired_minimal", "attn_v_single_minimal",
        "rst_single_minimal",
    }
    if (not isinstance(production_srw_fns, dict)
            or required - set(production_srw_fns)):
        raise ValueError(
            "production contribution capture requires canonical minimal kernels: "
            + ",".join(sorted(required)))

    params = model_module._squeeze_params(params)
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    batch_size, seq_len = input_ids.shape
    if target_positions is not None:
        target_positions = jnp.asarray(target_positions, dtype=jnp.int32)
        if target_positions.shape != (batch_size,):
            raise ValueError(
                "target_positions must have shape "
                f"({batch_size},), got {target_positions.shape}")
    d_model = int(model_cfg["d_model"])
    n_layers = int(model_cfg["n_layers"])
    n_heads = int(model_cfg["n_heads"])
    d_head = d_model // n_heads
    execution_kwargs = (
        model_module._angular_execution_kwargs_from_model_cfg(model_cfg))
    if execution_prune_eps is not None:
        execution_kwargs["execution_prune_eps"] = float(execution_prune_eps)
    default_power = float(execution_kwargs.get("admission_den_power", 1.0))
    powers = {
        route: float(execution_kwargs.get(
            f"admission_den_power_{route}", default_power))
        for route in ("qk", "v", "rst")
    }

    def execution_for(temperature_key: str) -> Dict[str, Any]:
        value = dict(execution_kwargs)
        temperature = model_cfg.get(temperature_key)
        if temperature is not None:
            value["soft_gate_temperature"] = float(temperature)
        return value

    execution_qk = execution_for("soft_gate_T_qk")
    execution_v = execution_for("soft_gate_T_v")
    execution_rst = execution_for("soft_gate_T_rst")
    topk_qk = int(topk if topk_qk is None else topk_qk)
    topk_v = int(topk if topk_v is None else topk_v)
    topk_rst = int(topk if topk_rst is None else topk_rst)

    pool = model_module._pool_params_with_operator_keys(
        params["neuron_pool"], model_cfg.get("operator_key_mode"))
    router = params["router"]
    qk_scale, v_scale, rst_scale = model_module._effective_pool_output_scales(
        pool, d_model, n_layers)
    positions = jnp.arange(seq_len)[jnp.newaxis, :]
    residual_state = (
        params["token_emb"]["embedding"][input_ids]
        + params["pos_emb"]["embedding"][positions])
    traces = {
        f"{route}_{field}": []
        for route in TRACE_POOLS for field in TRACE_FIELDS
    }

    for layer_idx in range(n_layers):
        block = params[f"block_{layer_idx}"]
        normed = model_module._layer_norm(
            residual_state, block["norm1"]["scale"],
            block["norm1"]["bias"])
        operator_queries = (
            normed @ router["proj_attn"]["kernel"]
            + router["proj_attn"]["bias"])
        query_q, query_k, query_v = jnp.split(operator_queries, 3, axis=-1)
        query_adapter = getattr(
            model_module, "_read_write_attn_operator_queries", None)
        if query_adapter is not None:
            query_q, query_k, query_v = query_adapter(
                router, normed, query_q, query_k, query_v)
        tau_all = (
            normed @ router["raw_tau_attn"]["kernel"]
            + router["raw_tau_attn"]["bias"])
        q_stats = _operator_contribution_topk(
            normed, query_q, pool["attn_qk_op_key"], tau_all[:, :, 0:1],
            pool["attn_qk_read"], pool["attn_qk_write"],
            model_module=model_module, topk=topk_qk,
            execution_kwargs=execution_qk,
            admission_den_power=powers["qk"],
            target_positions=target_positions)
        k_stats = _operator_contribution_topk(
            normed, query_k, pool["attn_qk_op_key"], tau_all[:, :, 1:2],
            pool["attn_qk_read"], pool["attn_qk_write"],
            model_module=model_module, topk=topk_qk,
            execution_kwargs=execution_qk,
            admission_den_power=powers["qk"],
            target_positions=target_positions)
        v_stats = _operator_contribution_topk(
            normed, query_v, pool["attn_v_op_key"], tau_all[:, :, 2:3],
            pool["attn_v_read"], pool["attn_v_write"],
            model_module=model_module, topk=topk_v,
            execution_kwargs=execution_v,
            admission_den_power=powers["v"],
            target_positions=target_positions)

        paired_queries = jnp.stack((query_q, query_k), axis=2)
        paired_tau = jnp.stack(
            (tau_all[:, :, 0:1], tau_all[:, :, 1:2]), axis=2)
        qk_transitions = production_srw_fns["attn_qk_paired_minimal"](
            normed, paired_queries, pool["attn_qk_op_key"], paired_tau,
            pool["attn_qk_read"], pool["attn_qk_write"],
            execution_qk["soft_gate_temperature"],
            model_cfg.get(
                "soft_gate_t_final", execution_qk["soft_gate_temperature"]),
            execution_qk["soft_gate_boundary_power"],
            model_cfg.get(
                "soft_gate_boundary_power_final",
                execution_qk["soft_gate_boundary_power"]),
            execution_qk.get("execution_prune_eps", 0.0),
        )[0]
        v_transition = production_srw_fns["attn_v_single_minimal"](
            normed, query_v, pool["attn_v_op_key"], tau_all[:, :, 2:3],
            pool["attn_v_read"], pool["attn_v_write"],
            execution_v["soft_gate_temperature"],
            model_cfg.get(
                "soft_gate_t_final", execution_v["soft_gate_temperature"]),
            execution_v["soft_gate_boundary_power"],
            model_cfg.get(
                "soft_gate_boundary_power_final",
                execution_v["soft_gate_boundary_power"]),
            execution_v.get("execution_prune_eps", 0.0),
        )[0]
        attention_q = qk_transitions[:, :, 0, :] * qk_scale
        attention_k = qk_transitions[:, :, 1, :] * qk_scale
        attention_v = v_transition * v_scale
        for stats, scale in (
                (q_stats, qk_scale), (k_stats, qk_scale),
                (v_stats, v_scale)):
            stats["top_val"] = stats["top_val"] * jnp.abs(scale)

        attention_q = attention_q.reshape(
            batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_k = attention_k.reshape(
            batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_v = attention_v.reshape(
            batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = jnp.einsum(
            "bhsd,bhtd->bhst", attention_q, attention_k
        ) / jnp.sqrt(jnp.float32(d_head))
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attention_output = jnp.einsum(
            "bhst,bhtd->bhsd", attention_weights, attention_v)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, d_model)
        residual_state = residual_state + (
            attention_output @ block["attn"]["expand_O"]["kernel"])

        normed = model_module._layer_norm(
            residual_state, block["norm2"]["scale"],
            block["norm2"]["bias"])
        query_rst = (
            normed @ router["proj_rst"]["kernel"]
            + router["proj_rst"]["bias"])
        rst_adapter = getattr(
            model_module, "_read_write_rst_operator_query", None)
        if rst_adapter is not None:
            query_rst = rst_adapter(router, normed, query_rst)
        tau_rst = (
            normed @ router["raw_tau_rst"]["kernel"]
            + router["raw_tau_rst"]["bias"])
        rst_stats = _operator_contribution_topk(
            normed, query_rst, pool["rst_op_key"], tau_rst,
            pool["rst_read"], pool["rst_write"],
            model_module=model_module, topk=topk_rst,
            execution_kwargs=execution_rst,
            admission_den_power=powers["rst"],
            target_positions=target_positions)
        rst_transition = production_srw_fns["rst_single_minimal"](
            normed, query_rst, pool["rst_op_key"], tau_rst,
            pool["rst_read"], pool["rst_write"],
            execution_rst["soft_gate_temperature"],
            model_cfg.get(
                "soft_gate_t_final", execution_rst["soft_gate_temperature"]),
            execution_rst["soft_gate_boundary_power"],
            model_cfg.get(
                "soft_gate_boundary_power_final",
                execution_rst["soft_gate_boundary_power"]),
            execution_rst.get("execution_prune_eps", 0.0),
        )[0] * rst_scale
        rst_stats["top_val"] = rst_stats["top_val"] * jnp.abs(rst_scale)
        residual_state = residual_state + rst_transition

        for route, stats in (
                ("q", q_stats), ("k", k_stats),
                ("v", v_stats), ("rst", rst_stats)):
            for field in TRACE_FIELDS:
                traces[f"{route}_{field}"].append(stats[field])

    return {
        key: jnp.stack(values, axis=0)
        for key, values in traces.items()
    }
