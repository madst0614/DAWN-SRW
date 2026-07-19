"""Exact production-path behavior, retention, suppression, and interchange."""

from __future__ import annotations

import copy
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import (
    materialize_global_array,
    materialize_global_tree,
)
from analysis.operator_interpretability.benchmark_schema import (
    BenchmarkExample,
    canonical_hash,
)
from analysis.operator_interpretability.circuit import (
    bootstrap_faithfulness_ci,
    normalized_faithfulness,
)
from analysis.operator_interpretability.program import (
    PROGRAM_MODES,
    PROGRAM_ROUTES,
    OperatorProgramSchedule,
    compactness_metrics,
    deterministic_mismatch_mapping,
    random_program_schedule,
    reindex_program_schedule,
)
from analysis.operator_interpretability.protocol import ProtocolConfig
from analysis.operator_interpretability.statistics import (
    bootstrap_mean_ci,
    paired_permutation_test,
)
from analysis.operator_interpretability.units import (
    OperatorCircuit,
    OperatorSpaceShape,
    ROUTE_INDEX,
)


def _runtime_kwargs(
        ctx: Any, *, kernel_profile: str = "production") -> dict[str, Any]:
    cfg = ctx.model_cfg
    kernel_profile = str(kernel_profile).strip().lower()
    if kernel_profile == "production":
        sharded_fns = ctx.sharded_fns
    else:
        profiles = getattr(ctx, "sharded_fns_analysis", None)
        if isinstance(profiles, Mapping) and kernel_profile in profiles:
            sharded_fns = profiles[kernel_profile]
        elif not profiles:
            # Lightweight unit harnesses may provide one already-selected
            # analysis closure instead of a full production context.
            sharded_fns = ctx.sharded_fns
            current_profile = (
                sharded_fns.get("_v4171_kernel_profile")
                if isinstance(sharded_fns, Mapping) else None)
            if (current_profile is not None
                    and str(current_profile) != kernel_profile):
                raise ValueError(
                    "analysis context kernel profile mismatch: "
                    f"requested={kernel_profile!r} "
                    f"available={current_profile!r}")
        else:
            raise ValueError(
                f"analysis context lacks kernel profile {kernel_profile!r}")
    temperature = float(cfg["soft_gate_temperature"])
    boundary = float(cfg["soft_gate_boundary_power"])
    # QK/V/RST denominator powers are immutable constructor fields on the
    # restored v417x model. The forward accepts only the legacy scalar and
    # internally resolves each production pool from those constructor fields.
    return {
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(0)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "soft_gate_temperature": temperature,
        "soft_gate_t_final": float(cfg.get("soft_gate_t_final", temperature)),
        "soft_gate_T_qk": float(cfg.get("soft_gate_T_qk", temperature)),
        "soft_gate_T_v": float(cfg.get("soft_gate_T_v", temperature)),
        "soft_gate_T_rst": float(cfg.get("soft_gate_T_rst", temperature)),
        "soft_gate_boundary_power": boundary,
        "soft_gate_boundary_power_final": float(
            cfg.get("soft_gate_boundary_power_final", boundary)),
        "admission_den_power": float(cfg["admission_den_power"]),
        "srw_composition_mode": str(cfg["srw_composition_mode"]),
        "heat_kernel_beta": float(cfg["heat_kernel_beta"]),
        "execution_prune_eps": jnp.float32(
            float(cfg.get("execution_prune_eps", 0.0) or 0.0)),
        "compute_accuracy": False,
    }


def validate_operator_interchange_request(
        model_cfg: Mapping[str, Any], examples: Sequence[BenchmarkExample], *,
        layer: int, route: str,
        operator_ids: Sequence[int]) -> tuple[int, ...]:
    """Host contract required before staging the production JAX hook."""
    if not examples:
        raise ValueError("interchange evaluation has no examples")
    if route not in ROUTE_INDEX:
        raise ValueError(f"unknown interchange route: {route}")
    if not operator_ids:
        raise ValueError("interchange operator family is empty")
    shape = OperatorSpaceShape.from_model_cfg(model_cfg)
    if (isinstance(layer, bool) or not isinstance(layer, numbers.Integral)
            or not 0 <= int(layer) < shape.n_layers):
        raise ValueError(f"interchange layer is out of range: {layer!r}")
    normalized_operator_ids: list[int] = []
    for value in operator_ids:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                f"interchange operator id must be an integer: {value!r}")
        operator_id = int(value)
        if not 0 <= operator_id < shape.pool_size(route):
            raise ValueError(
                "interchange operator id is out of range for route "
                f"{route}: {operator_id}")
        normalized_operator_ids.append(operator_id)
    if len(set(normalized_operator_ids)) != len(normalized_operator_ids):
        raise ValueError("interchange operator family contains duplicate ids")
    for example in examples:
        example.validate()
    return tuple(normalized_operator_ids)


def _sequence_arrays(examples: Sequence[BenchmarkExample], *,
                     prompt_side: str, answer_side: str,
                     pad_token_id: int, multiple: int) -> tuple[np.ndarray, np.ndarray, int]:
    if prompt_side not in {"base", "source"}:
        raise ValueError("prompt_side must be base or source")
    answer_fields = {
        "positive": "positive_ids",
        "negative": "negative_ids",
        "source_positive": "source_positive_ids",
        "source_negative": "source_negative_ids",
        "intervention_positive": "intervention_positive_ids",
        "intervention_negative": "intervention_negative_ids",
    }
    if answer_side not in answer_fields:
        raise ValueError(f"unknown answer side: {answer_side}")
    rows: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for example in examples:
        prompt = (
            example.input_ids_base if prompt_side == "base"
            else example.input_ids_source)
        answer = tuple(getattr(example, answer_fields[answer_side]))
        if not answer:
            raise ValueError(f"{example.example_id}: empty {answer_side}")
        rows.append((tuple(prompt), answer))
    length = max(len(prompt) + len(answer) for prompt, answer in rows)
    batch_size = ((len(rows) + multiple - 1) // multiple) * multiple
    input_ids = np.full((batch_size, length), int(pad_token_id), dtype=np.int32)
    labels = np.full((batch_size, length), -100, dtype=np.int32)
    for index, (prompt, answer) in enumerate(rows):
        sequence = np.asarray((*prompt, *answer), dtype=np.int32)
        input_ids[index, :len(sequence)] = sequence
        labels[index, len(prompt):len(sequence)] = np.asarray(answer, dtype=np.int32)
    for index in range(len(rows), batch_size):
        input_ids[index] = input_ids[0]
        labels[index] = labels[0]
    return input_ids, labels, len(rows)


def _device_batch(ctx: Any, input_ids: np.ndarray, labels: np.ndarray):
    return (
        jax.device_put(
            jnp.asarray(input_ids), NamedSharding(ctx.mesh, P("data", None))),
        jax.device_put(
            jnp.asarray(labels), NamedSharding(ctx.mesh, P("data", None))),
    )


def _score_from_result(
        result: Mapping[str, Any], normalization: str) -> jax.Array:
    valid = result["valid_mask"].astype(jnp.float32)
    token_logp = -result["per_token_ce"].astype(jnp.float32) * valid
    total = token_logp.sum(axis=-1)
    if normalization == "sum_log_probability":
        return total
    if normalization == "mean_log_probability_per_token":
        return total / jnp.maximum(valid.sum(axis=-1), jnp.float32(1.0))
    raise ValueError(f"unknown candidate score normalization: {normalization}")


def _plain_score_executable(
        ctx: Any, normalization: str = "sum_log_probability"):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = ("plain_score", "production", normalization)
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx, kernel_profile="production")

    @jax.jit
    def score(params, input_ids, labels):
        # The v417x core applies its causal mask internally and currently does
        # not consume attention_mask. Right-padding is excluded from scoring by
        # labels=-100, so use the same all-ones placeholder as every analysis
        # route instead of implying that token id 0 is model-level padding.
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            minimal_train=True, **kwargs)
        return _score_from_result(result, normalization)

    cache[key] = score
    return score


def _candidate_score_normalization(
        examples: Sequence[BenchmarkExample]) -> str:
    values = {
        str(example.metadata.get("candidate_score_normalization")
            or "sum_log_probability")
        for example in examples
    }
    if len(values) != 1:
        raise ValueError(
            "candidate score normalization differs within one evaluation batch")
    normalization = next(iter(values))
    if normalization not in {
            "sum_log_probability", "mean_log_probability_per_token"}:
        raise ValueError(
            f"unknown candidate score normalization: {normalization}")
    return normalization


def _retention_mode_code(mode: str) -> int:
    modes = {
        "conditional_execution_sufficiency": 1,
        "autonomous_subcircuit_sufficiency": 2,
    }
    if mode not in modes:
        raise ValueError(f"unknown circuit retention mode: {mode}")
    return modes[mode]


def _retention_score_executable(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = ("retention_score", "retention", "dynamic_mode")
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx, kernel_profile="retention")

    @jax.jit
    def score(params, input_ids, labels, keep_qk, keep_v, keep_rst,
              retention_mode):
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            minimal_train=True,
            analysis_keep_qk=keep_qk,
            analysis_keep_v=keep_v,
            analysis_keep_rst=keep_rst,
            analysis_position_mask=jnp.ones_like(
                input_ids, dtype=jnp.bool_),
            analysis_retention_mode=jnp.asarray(
                retention_mode, dtype=jnp.int32),
            analysis_return_residual=False,
            **kwargs)
        return _score_from_result(result, "sum_log_probability")

    cache[key] = score
    return score


def evaluate_behavior(ctx: Any, examples: Sequence[BenchmarkExample], *,
                      pad_token_id: int) -> dict[str, Any]:
    if not examples:
        raise ValueError("behavior evaluation has no examples")
    multiple = max(1, int(ctx.mesh.shape["data"]))
    score_normalization = _candidate_score_normalization(examples)
    score = _plain_score_executable(ctx, score_normalization)

    def run(prompt_side: str, answer_side: str) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side=prompt_side, answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        return materialize_global_array(
            score(ctx.params, ids, labels))[:arrays[2]].astype(np.float64)

    base_positive = run("base", "positive")
    base_negative = run("base", "negative")
    source_base_positive = run("source", "positive")
    source_base_negative = run("source", "negative")
    source_scored = np.asarray([
        example.source_behavior_required for example in examples],
        dtype=np.bool_)
    source_own_margin = np.full((len(examples),), np.nan, dtype=np.float64)
    if np.any(source_scored):
        scored_examples = [
            example for example in examples if example.source_behavior_required]

        def run_source(answer_side: str) -> np.ndarray:
            arrays = _sequence_arrays(
                scored_examples, prompt_side="source", answer_side=answer_side,
                pad_token_id=pad_token_id, multiple=multiple)
            ids, labels = _device_batch(ctx, arrays[0], arrays[1])
            return materialize_global_array(
                score(ctx.params, ids, labels))[:arrays[2]].astype(np.float64)

        source_own_margin[source_scored] = (
            run_source("source_positive") - run_source("source_negative"))
    base_margin = base_positive - base_negative
    # Official MIB faithfulness keeps the clean-label orientation on the
    # corrupted/source prompt.  source_own_margin separately audits whether
    # the checkpoint solves the source member of the counterfactual pair.
    corrupted_margin = source_base_positive - source_base_negative
    base_known_correct = base_margin > 0.0
    # The official RAVEL Wikipedia source is a context prompt, not a labeled
    # behavioral query.  It is therefore not a second behavioral prerequisite;
    # preserve that fact explicitly instead of inventing a source label.
    source_known_correct = (~source_scored) | (source_own_margin > 0.0)
    pair_known_correct = base_known_correct & source_known_correct
    source_margin_output = [
        float(value) if scored else None
        for value, scored in zip(source_own_margin, source_scored)
    ]
    scored_source_correct = source_known_correct[source_scored]
    return {
        "status": "ready",
        "phase": examples[0].phase,
        "candidate_score_normalization": score_normalization,
        "example_ids": [example.example_id for example in examples],
        "base_positive_logp": base_positive.tolist(),
        "base_negative_logp": base_negative.tolist(),
        "base_margin": base_margin.tolist(),
        "corrupted_margin": corrupted_margin.tolist(),
        "source_own_margin": source_margin_output,
        "source_behavior_scored": source_scored.tolist(),
        "base_known_correct": base_known_correct.tolist(),
        "source_known_correct": source_known_correct.tolist(),
        "known_correct": pair_known_correct.tolist(),
        "known_correct_count": int(np.sum(pair_known_correct)),
        "base_known_correct_count": int(np.sum(base_known_correct)),
        "source_known_correct_count": int(np.sum(scored_source_correct)),
        "source_behavior_scored_count": int(np.sum(source_scored)),
        "accuracy": float(np.mean(base_known_correct)),
        "source_accuracy": (
            float(np.mean(scored_source_correct))
            if scored_source_correct.size else None),
        "pair_accuracy": float(np.mean(pair_known_correct)),
        "mean_margin": float(np.mean(base_margin)),
        "mean_corrupted_margin": float(np.mean(corrupted_margin)),
        "mean_source_own_margin": (
            float(np.mean(source_own_margin[source_scored]))
            if np.any(source_scored) else None),
        "corrupted_margin_orientation": (
            "base_positive_minus_base_negative_on_source_prompt"),
        "mechanistic_eligibility": (
            "base_known_correct_and_labeled_source_known_correct_when_defined"),
    }


def evaluate_circuit_retention(
        ctx: Any, examples: Sequence[BenchmarkExample], circuit: OperatorCircuit,
        *, shape: OperatorSpaceShape, mode: str,
        pad_token_id: int) -> dict[str, Any]:
    multiple = max(1, int(ctx.mesh.shape["data"]))
    masks = circuit.dense_masks(shape)
    all_operators_retained = all(
        bool(np.all(mask)) for mask in masks.values())
    score = (
        _plain_score_executable(ctx)
        if all_operators_retained
        else _retention_score_executable(ctx))
    retention_mode = jnp.int32(_retention_mode_code(mode))

    def run(answer_side: str) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side="base", answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        if all_operators_retained:
            # The registered 1.00 circuit is the production endpoint.  Route
            # it through the production executable itself so exact no-op is a
            # semantic guarantee rather than a tolerance-based observation.
            value = score(ctx.params, ids, labels)
        else:
            value = score(
                ctx.params, ids, labels,
                jnp.asarray(masks["qk"]), jnp.asarray(masks["v"]),
                jnp.asarray(masks["rst"]), retention_mode)
        return materialize_global_array(value)[:arrays[2]].astype(np.float64)

    positive = run("positive")
    negative = run("negative")
    margin = positive - negative
    return {
        "status": "ready",
        "phase": examples[0].phase,
        "mode": mode,
        "execution_path": (
            "production_exact_noop"
            if all_operators_retained else "circuit_retention"),
        "site_count": circuit.site_count,
        "circuit_hash": circuit.circuit_hash,
        "margin": margin.tolist(),
        "mean_margin": float(np.mean(margin)),
        "accuracy": float(np.mean(margin > 0.0)),
    }


def evaluate_circuit_necessity(
        ctx: Any, examples: Sequence[BenchmarkExample], circuit: OperatorCircuit,
        *, shape: OperatorSpaceShape, pad_token_id: int) -> dict[str, Any]:
    """Suppress the complete circuit numerator while preserving full admission."""
    masks = circuit.dense_masks(shape)
    if any(value.dtype != np.bool_ for value in masks.values()):
        raise TypeError("circuit necessity masks must have bool dtype")
    complement = {key: ~value for key, value in masks.items()}
    # Invoke the dense masks directly; materializing millions of complement
    # OperatorSite objects would add no scientific information.
    multiple = max(1, int(ctx.mesh.shape["data"]))
    score = _retention_score_executable(ctx)
    retention_mode = jnp.int32(_retention_mode_code(
        "conditional_execution_sufficiency"))

    def run(answer_side: str) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side="base", answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        value = score(
            ctx.params, ids, labels,
            jnp.asarray(complement["qk"]), jnp.asarray(complement["v"]),
            jnp.asarray(complement["rst"]), retention_mode)
        return materialize_global_array(value)[:arrays[2]].astype(np.float64)

    positive = run("positive")
    negative = run("negative")
    return {
        "status": "ready",
        "phase": examples[0].phase,
        "intervention": "circuit_wide_execution_numerator_suppression",
        "admission_denominator": "full_production_denominator",
        "suppressed_site_count": circuit.site_count,
        "margin": (positive - negative).tolist(),
        "mean_margin": float(np.mean(positive - negative)),
    }


def all_ones_retention_parity(ctx: Any, examples: Sequence[BenchmarkExample], *,
                              shape: OperatorSpaceShape,
                              pad_token_id: int, mode: str) -> dict[str, Any]:
    multiple = max(1, int(ctx.mesh.shape["data"]))
    arrays = _sequence_arrays(
        examples, prompt_side="base", answer_side="positive",
        pad_token_id=pad_token_id, multiple=multiple)
    ids, labels = _device_batch(ctx, arrays[0], arrays[1])
    score = _retention_score_executable(ctx)
    ones_qk = jnp.ones((shape.n_layers, 2, shape.n_qk), dtype=jnp.bool_)
    ones_v = jnp.ones((shape.n_layers, shape.n_v), dtype=jnp.bool_)
    ones_rst = jnp.ones((shape.n_layers, shape.n_rst), dtype=jnp.bool_)
    # Both references are separate invocations of the same compiled forward.
    # The mode is a dynamic scalar argument, so TPU lowering and reduction
    # order cannot differ between the production-mode and retained-mode runs.
    before = materialize_global_array(score(
        ctx.params, ids, labels, ones_qk, ones_v, ones_rst, jnp.int32(0)))
    after = materialize_global_array(score(
        ctx.params, ids, labels, ones_qk, ones_v, ones_rst,
        jnp.int32(_retention_mode_code(mode))))
    exact = bool(np.array_equal(before, after))
    if not exact:
        unequal = np.not_equal(before, after)
        first_index = tuple(int(value) for value in np.argwhere(unequal)[0])
        delta = np.abs(
            before.astype(np.float64) - after.astype(np.float64))
        finite_delta = delta[np.isfinite(delta)]
        max_absolute_delta = (
            float(np.max(finite_delta)) if finite_delta.size else None)
        raise RuntimeError(
            "all-ones circuit retention parity is not machine-exact: "
            f"mismatch_count={int(np.count_nonzero(unequal))} "
            f"first_index={first_index} "
            f"production={before[first_index]!r} "
            f"retained={after[first_index]!r} "
            f"max_absolute_delta={max_absolute_delta!r}")
    return {
        "status": "passed",
        "machine_exact": True,
        "parity_scope": "single_dynamic_mode_executable",
        "mode": mode,
        "example_count": arrays[2],
    }


def _prompt_arrays(examples: Sequence[BenchmarkExample], *, side: str,
                   pad_token_id: int, multiple: int):
    prompts = [
        example.input_ids_base if side == "base" else example.input_ids_source
        for example in examples
    ]
    length = max(map(len, prompts))
    batch = ((len(prompts) + multiple - 1) // multiple) * multiple
    ids = np.full((batch, length), int(pad_token_id), dtype=np.int32)
    positions = np.zeros((batch,), dtype=np.int32)
    for index, (example, prompt) in enumerate(zip(examples, prompts)):
        ids[index, :len(prompt)] = np.asarray(prompt, dtype=np.int32)
        positions[index] = (
            example.trace_position_base if side == "base"
            else example.trace_position_source)
    for index in range(len(prompts), batch):
        ids[index] = ids[0]
        positions[index] = positions[0]
    return ids, positions, len(prompts)


def evaluate_operator_interchange(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        layer: int, route: str, operator_ids: Sequence[int],
        pad_token_id: int) -> list[dict[str, Any]]:
    normalized_operator_ids = validate_operator_interchange_request(
        ctx.model_cfg, examples, layer=layer, route=route,
        operator_ids=operator_ids)
    multiple = max(1, int(ctx.mesh.shape["data"]))
    score_normalization = _candidate_score_normalization(examples)
    source_arrays = _prompt_arrays(
        examples, side="source", pad_token_id=pad_token_id,
        multiple=multiple)
    source_ids = jax.device_put(
        jnp.asarray(source_arrays[0]), NamedSharding(ctx.mesh, P("data", None)))
    source_positions = jax.device_put(
        jnp.asarray(source_arrays[1]), NamedSharding(ctx.mesh, P("data")))
    selected = np.tile(
        np.asarray(normalized_operator_ids, dtype=np.int32)[None, :],
        (source_arrays[0].shape[0], 1))
    selected_ids = jax.device_put(
        jnp.asarray(selected), NamedSharding(ctx.mesh, P("data", None)))
    kwargs = _runtime_kwargs(ctx, kernel_profile="suppression")
    route_index = ROUTE_INDEX[route]

    @jax.jit
    def capture(params, ids, positions, selected_group):
        result = ctx.model.apply(
            {"params": params}, ids, selected_group, jnp.int32(layer),
            positions, jnp.int32(route_index), labels=ids,
            attention_mask=jnp.ones_like(ids), return_residual=False,
            method=ctx.model.analysis_capture_operator_group_contribution,
            **kwargs)
        values = result["operator_route_contributions"][route]
        return values[jnp.int32(layer)]

    source_contribution = capture(
        ctx.params, source_ids, source_positions, selected_ids)
    expected_source_shape = (
        source_arrays[0].shape[0], int(ctx.model_cfg["d_model"]))
    if source_contribution.shape != expected_source_shape:
        raise RuntimeError(
            "captured interchange contribution shape mismatch: "
            f"expected={expected_source_shape} "
            f"actual={source_contribution.shape}")
    if source_contribution.dtype != jnp.float32:
        raise RuntimeError(
            "captured interchange contribution must be post-scale float32, "
            f"got {source_contribution.dtype}")

    @jax.jit
    def patched_score(params, ids, labels, positions, selected_group, source):
        result = ctx.model.apply(
            {"params": params}, ids, selected_group, jnp.int32(layer),
            positions, jnp.int32(route_index), source,
            labels=labels, attention_mask=jnp.ones_like(ids),
            return_residual=False,
            method=ctx.model.analysis_forward_with_operator_interchange,
            **kwargs)
        return _score_from_result(result, score_normalization)

    plain = _plain_score_executable(ctx, score_normalization)

    def scores(prompt_side: str, answer_side: str, *, patch: bool) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side=prompt_side, answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        if patch:
            position_values = np.asarray([
                example.trace_position_base for example in examples
            ], dtype=np.int32)
            padded_positions = np.resize(position_values, arrays[0].shape[0])
            if arrays[0].shape[0] > len(examples):
                padded_positions[len(examples):] = position_values[0]
            positions = jax.device_put(
                jnp.asarray(padded_positions),
                NamedSharding(ctx.mesh, P("data")))
            value = patched_score(
                ctx.params, ids, labels, positions, selected_ids,
                source_contribution)
        else:
            value = plain(ctx.params, ids, labels)
        return materialize_global_array(value)[:arrays[2]].astype(np.float64)

    base_base = scores("base", "positive", patch=False) - scores(
        "base", "negative", patch=False)
    base_intervention = scores(
        "base", "intervention_positive", patch=False) - scores(
            "base", "intervention_negative", patch=False)
    patched_intervention = scores(
        "base", "intervention_positive", patch=True) - scores(
            "base", "intervention_negative", patch=True)
    patched_base = scores("base", "positive", patch=True) - scores(
        "base", "negative", patch=True)
    return [
        {
            "example_id": example.example_id,
            "pair_type": example.pair_type,
            "causal_variable": example.causal_variable,
            "layer": int(layer),
            "route": route,
            "operator_ids": list(normalized_operator_ids),
            "position_kind": example.metadata.get("position_kind"),
            "candidate_score_normalization": score_normalization,
            "base_base_margin": float(base_base[index]),
            "base_intervention_margin": float(base_intervention[index]),
            "patched_intervention_margin": float(patched_intervention[index]),
            "patched_base_margin": float(patched_base[index]),
        }
        for index, example in enumerate(examples)
    ]


def _pad_program_schedule(
        schedule: OperatorProgramSchedule, batch_size: int) -> tuple[
            dict[str, np.ndarray], dict[str, np.ndarray]]:
    if batch_size < schedule.batch_size:
        raise ValueError("program batch is smaller than its real examples")
    ids = {}
    valid = {}
    for route in PROGRAM_ROUTES:
        source_ids = np.asarray(schedule.ids[route], dtype=np.int32)
        source_valid = np.asarray(schedule.valid[route], dtype=np.bool_)
        ids[route] = np.zeros(
            (source_ids.shape[0], batch_size, source_ids.shape[2]),
            dtype=np.int32)
        valid[route] = np.zeros_like(ids[route], dtype=np.bool_)
        ids[route][:, :schedule.batch_size] = source_ids
        valid[route][:, :schedule.batch_size] = source_valid
        if batch_size > schedule.batch_size:
            ids[route][:, schedule.batch_size:] = source_ids[:, :1]
            valid[route][:, schedule.batch_size:] = source_valid[:, :1]
    return ids, valid


def _pad_program_source(
        source: Mapping[str, np.ndarray] | None, *, n_layers: int,
        real_count: int, batch_size: int, d_model: int) -> dict[str, np.ndarray]:
    output = {}
    for route in PROGRAM_ROUTES:
        if source is None:
            values = np.zeros(
                (n_layers, real_count, d_model), dtype=np.float32)
        else:
            values = np.asarray(source[route])
            expected = (n_layers, real_count, d_model)
            if values.dtype != np.float32 or values.shape != expected:
                raise ValueError(
                    f"program source contribution {route} must be float32 "
                    f"with shape {expected}, got {values.dtype}/{values.shape}")
        padded = np.zeros((n_layers, batch_size, d_model), dtype=np.float32)
        padded[:, :real_count] = values
        if batch_size > real_count:
            padded[:, real_count:] = values[:, :1]
        output[route] = padded
    return output


def _program_device_arrays(
        ctx: Any, schedule: OperatorProgramSchedule, *, batch_size: int,
        source: Mapping[str, np.ndarray] | None = None):
    ids, valid = _pad_program_schedule(schedule, batch_size)
    source_values = _pad_program_source(
        source,
        n_layers=int(ctx.model_cfg["n_layers"]),
        real_count=schedule.batch_size,
        batch_size=batch_size,
        d_model=int(ctx.model_cfg["d_model"]),
    )
    ids_device = {
        route: jax.device_put(
            jnp.asarray(ids[route]),
            NamedSharding(ctx.mesh, P(None, "data", None)))
        for route in PROGRAM_ROUTES
    }
    valid_device = {
        route: jax.device_put(
            jnp.asarray(valid[route]),
            NamedSharding(ctx.mesh, P(None, "data", None)))
        for route in PROGRAM_ROUTES
    }
    source_device = {
        route: jax.device_put(
            jnp.asarray(source_values[route]),
            NamedSharding(ctx.mesh, P(None, "data", None)))
        for route in PROGRAM_ROUTES
    }
    return ids_device, valid_device, source_device


def _program_target_positions(
        ctx: Any, examples: Sequence[BenchmarkExample], *, prompt_side: str,
        batch_size: int):
    values = np.asarray([
        example.trace_position_base if prompt_side == "base"
        else example.trace_position_source
        for example in examples], dtype=np.int32)
    padded = np.empty((batch_size,), dtype=np.int32)
    padded[:len(examples)] = values
    if batch_size > len(examples):
        padded[len(examples):] = values[0]
    return jax.device_put(
        jnp.asarray(padded), NamedSharding(ctx.mesh, P("data")))


def _program_score_executable(
        ctx: Any, normalization: str = "sum_log_probability"):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = ("native_program_score", "suppression", normalization)
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx, kernel_profile="suppression")

    @jax.jit
    def score(
            params, input_ids, labels, target_positions,
            ids_q, ids_k, ids_v, ids_rst,
            valid_q, valid_k, valid_v, valid_rst,
            source_q, source_k, source_v, source_rst, program_mode):
        result = ctx.model.apply(
            {"params": params}, input_ids,
            selected_ids_q=ids_q,
            selected_ids_k=ids_k,
            selected_ids_v=ids_v,
            selected_ids_rst=ids_rst,
            selected_valid_q=valid_q,
            selected_valid_k=valid_k,
            selected_valid_v=valid_v,
            selected_valid_rst=valid_rst,
            target_positions=target_positions,
            program_mode=program_mode,
            source_contribution_q=source_q,
            source_contribution_k=source_k,
            source_contribution_v=source_v,
            source_contribution_rst=source_rst,
            labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=False,
            method=ctx.model.analysis_forward_with_operator_program,
            **kwargs)
        return _score_from_result(result, normalization)

    cache[key] = score
    return score


def _program_capture_executable(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = ("native_program_capture", "suppression")
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx, kernel_profile="suppression")

    @jax.jit
    def capture(
            params, input_ids, target_positions,
            ids_q, ids_k, ids_v, ids_rst,
            valid_q, valid_k, valid_v, valid_rst):
        result = ctx.model.apply(
            {"params": params}, input_ids,
            selected_ids_q=ids_q,
            selected_ids_k=ids_k,
            selected_ids_v=ids_v,
            selected_ids_rst=ids_rst,
            selected_valid_q=valid_q,
            selected_valid_k=valid_k,
            selected_valid_v=valid_v,
            selected_valid_rst=valid_rst,
            target_positions=target_positions,
            labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            method=ctx.model.analysis_capture_operator_program_contributions,
            **kwargs)
        return result["operator_program_contributions"]

    cache[key] = capture
    return capture


def capture_operator_program_contributions(
        ctx: Any, examples: Sequence[BenchmarkExample],
        schedule: OperatorProgramSchedule, *, prompt_side: str,
        pad_token_id: int) -> dict[str, np.ndarray]:
    """Capture post-denominator selected transitions on an unpatched path."""
    if not examples or len(examples) != schedule.batch_size:
        raise ValueError("program contribution examples and schedule differ")
    multiple = max(1, int(ctx.mesh.shape["data"]))
    arrays = _prompt_arrays(
        examples, side=prompt_side, pad_token_id=pad_token_id,
        multiple=multiple)
    input_ids = jax.device_put(
        jnp.asarray(arrays[0]), NamedSharding(ctx.mesh, P("data", None)))
    positions = _program_target_positions(
        ctx, examples, prompt_side=prompt_side,
        batch_size=arrays[0].shape[0])
    ids, valid, _ = _program_device_arrays(
        ctx, schedule, batch_size=arrays[0].shape[0])
    result = materialize_global_tree(_program_capture_executable(ctx)(
        ctx.params, input_ids, positions,
        ids["q"], ids["k"], ids["v"], ids["rst"],
        valid["q"], valid["k"], valid["v"], valid["rst"]))
    output = {}
    for route in PROGRAM_ROUTES:
        values = np.asarray(result[route])[:, :arrays[2]].astype(
            np.float32, copy=False)
        expected = (
            int(ctx.model_cfg["n_layers"]), arrays[2],
            int(ctx.model_cfg["d_model"]))
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"captured {route} program contribution is invalid: "
                f"expected={expected} actual={values.shape}")
        output[route] = values
    return output


def _score_operator_program(
        ctx: Any, examples: Sequence[BenchmarkExample],
        schedule: OperatorProgramSchedule, *, prompt_side: str,
        answer_side: str, pad_token_id: int, program_mode: int,
        source: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    multiple = max(1, int(ctx.mesh.shape["data"]))
    arrays = _sequence_arrays(
        examples, prompt_side=prompt_side, answer_side=answer_side,
        pad_token_id=pad_token_id, multiple=multiple)
    ids_batch, labels = _device_batch(ctx, arrays[0], arrays[1])
    positions = _program_target_positions(
        ctx, examples, prompt_side=prompt_side,
        batch_size=arrays[0].shape[0])
    ids, valid, source_values = _program_device_arrays(
        ctx, schedule, batch_size=arrays[0].shape[0], source=source)
    score = _program_score_executable(
        ctx, _candidate_score_normalization(examples))
    values = score(
        ctx.params, ids_batch, labels, positions,
        ids["q"], ids["k"], ids["v"], ids["rst"],
        valid["q"], valid["k"], valid["v"], valid["rst"],
        source_values["q"], source_values["k"],
        source_values["v"], source_values["rst"],
        jnp.asarray(program_mode, dtype=jnp.int32))
    return materialize_global_array(values)[:arrays[2]].astype(np.float64)


def _program_margin(
        ctx: Any, examples: Sequence[BenchmarkExample],
        schedule: OperatorProgramSchedule, *, prompt_side: str,
        positive_side: str, negative_side: str, pad_token_id: int,
        program_mode: int,
        source: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    positive = _score_operator_program(
        ctx, examples, schedule,
        prompt_side=prompt_side, answer_side=positive_side,
        pad_token_id=pad_token_id, program_mode=program_mode,
        source=source)
    negative = _score_operator_program(
        ctx, examples, schedule,
        prompt_side=prompt_side, answer_side=negative_side,
        pad_token_id=pad_token_id, program_mode=program_mode,
        source=source)
    return positive - negative


def _ablation_metrics(
        full_margin: np.ndarray, ablated_margin: np.ndarray, *,
        config: ProtocolConfig, seed: int) -> dict[str, Any]:
    drop = np.asarray(full_margin) - np.asarray(ablated_margin)
    return {
        "mean_margin": float(np.mean(ablated_margin)),
        "mean_margin_drop": float(np.mean(drop)),
        "median_margin_drop": float(np.median(drop)),
        "positive_drop_fraction": float(np.mean(drop > 0.0)),
        "margin_drop_ci": bootstrap_mean_ci(
            drop, samples=config.bootstrap_samples, alpha=config.alpha,
            seed=seed),
        "permutation": paired_permutation_test(
            full_margin, ablated_margin,
            samples=config.permutation_samples, seed=seed + 1),
    }


def _paired_advantage_metrics(
        primary: np.ndarray, control: np.ndarray, *,
        config: ProtocolConfig, seed: int) -> dict[str, Any]:
    """Summarize the paired advantage of one intervention over a control."""
    effect = np.asarray(primary) - np.asarray(control)
    return {
        "mean_effect": float(np.mean(effect)),
        "median_effect": float(np.median(effect)),
        "positive_effect_fraction": float(np.mean(effect > 0.0)),
        "effect_ci": bootstrap_mean_ci(
            effect, samples=config.bootstrap_samples, alpha=config.alpha,
            seed=seed),
        "permutation": paired_permutation_test(
            primary, control, samples=config.permutation_samples,
            seed=seed + 1),
    }


def _random_overlap_metrics(
        schedule: OperatorProgramSchedule) -> dict[str, Any]:
    fractions = np.asarray([
        float(record["random_reference_overlap_fraction"])
        for record in schedule.records
    ], dtype=np.float64)
    return {
        "per_example_overlap_fraction": fractions.tolist(),
        "mean_overlap_fraction": float(np.mean(fractions)),
        "max_overlap_fraction": float(np.max(fractions)),
        "complement_only_fraction": float(np.mean([
            bool(record["random_complement_only"])
            for record in schedule.records
        ])),
    }


def _direction_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    improvement = np.asarray(after) - np.asarray(before)
    flips = (np.asarray(before) <= 0.0) & (np.asarray(after) > 0.0)
    return {
        "counterfactual_margin_before_mean": float(np.mean(before)),
        "counterfactual_margin_after_mean": float(np.mean(after)),
        "margin_improvement_mean": float(np.mean(improvement)),
        "answer_flip_fraction": float(np.mean(flips)),
    }


@dataclass(frozen=True)
class NativeProgramPhaseBaselines:
    """Mass-independent native-program margins frozen for one data phase."""

    full_base_margin: np.ndarray
    corrupted_margin: np.ndarray
    before_base_to_source: np.ndarray
    before_source_to_base: np.ndarray
    example_ids: tuple[str, ...] = ()
    phase: str = ""

    def __post_init__(self) -> None:
        expected_shape: tuple[int, ...] | None = None
        for name in (
                "full_base_margin", "corrupted_margin",
                "before_base_to_source", "before_source_to_base"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.ndim != 1 or value.size == 0:
                raise ValueError(
                    f"native program phase baseline {name} must be nonempty 1D")
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"native program phase baseline {name} is non-finite")
            if expected_shape is None:
                expected_shape = value.shape
            elif value.shape != expected_shape:
                raise ValueError("native program phase baseline shapes differ")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        example_ids = tuple(str(value) for value in self.example_ids)
        if example_ids and len(example_ids) != int(expected_shape[0]):
            raise ValueError(
                "native program phase baseline example ids are not aligned")
        object.__setattr__(self, "example_ids", example_ids)
        object.__setattr__(self, "phase", str(self.phase))


def _validate_native_operator_program_phase(
        examples: Sequence[BenchmarkExample], *,
        base_schedule: OperatorProgramSchedule,
        source_schedule: OperatorProgramSchedule,
        shape: OperatorSpaceShape) -> tuple[str, ...]:
    if not examples or any(
            example.benchmark_id != "mib_ioi"
            or example.pair_type != "s2_io_flip_counterfactual"
            for example in examples):
        raise ValueError(
            "native operator programs require official paired IOI examples")
    example_ids = tuple(example.example_id for example in examples)
    if example_ids != base_schedule.example_ids or base_schedule.example_ids != (
            source_schedule.example_ids):
        raise ValueError("base/source programs are not example-aligned")
    phases = {str(example.phase) for example in examples}
    if len(phases) != 1:
        raise ValueError("native operator program examples span multiple phases")
    for example in examples:
        orientation = (
            tuple(example.intervention_positive_ids)
                == tuple(example.negative_ids),
            tuple(example.intervention_negative_ids)
                == tuple(example.positive_ids),
            tuple(example.source_negative_ids) == tuple(example.positive_ids),
            tuple(example.source_positive_ids) == tuple(example.negative_ids),
        )
        if not all(orientation):
            raise ValueError(
                f"{example.example_id}: IOI answer orientation contract failed")
    if base_schedule.prompt_side != "base":
        raise ValueError("native operator base schedule has the wrong side")
    if source_schedule.prompt_side != "source":
        raise ValueError("native operator source schedule has the wrong side")
    base_schedule.validate(shape)
    source_schedule.validate(shape)
    if base_schedule.program_mass != source_schedule.program_mass:
        raise ValueError("base/source program masses differ")
    if _candidate_score_normalization(examples) != "sum_log_probability":
        raise ValueError(
            "IOI native program requires sum_log_probability scoring")
    return example_ids


def _validate_native_program_phase_baselines(
        baselines: NativeProgramPhaseBaselines,
        examples: Sequence[BenchmarkExample]) -> None:
    if not isinstance(baselines, NativeProgramPhaseBaselines):
        raise TypeError("native program phase baselines have the wrong type")
    example_ids = tuple(example.example_id for example in examples)
    if baselines.example_ids and baselines.example_ids != example_ids:
        raise ValueError("native program phase baselines use different examples")
    if baselines.phase and baselines.phase != str(examples[0].phase):
        raise ValueError("native program phase baselines use a different phase")
    if baselines.full_base_margin.shape != (len(examples),):
        raise ValueError("native program phase baselines have the wrong length")


def _native_program_replay_summary(
        replay_margin: np.ndarray, baselines: NativeProgramPhaseBaselines, *,
        config: ProtocolConfig, seed: int) -> dict[str, Any]:
    full_base_margin = baselines.full_base_margin
    corrupted_margin = baselines.corrupted_margin
    faithfulness = normalized_faithfulness(
        float(np.mean(replay_margin)), float(np.mean(full_base_margin)),
        float(np.mean(corrupted_margin)))
    if faithfulness is None:
        raise RuntimeError("native program faithfulness endpoint is undefined")
    return {
        "full_base_margin_mean": float(np.mean(full_base_margin)),
        "replay_base_margin_mean": float(np.mean(replay_margin)),
        "answer_agreement_with_full": float(np.mean(
            (replay_margin > 0.0) == (full_base_margin > 0.0))),
        "accuracy": float(np.mean(replay_margin > 0.0)),
        "normalized_faithfulness": float(faithfulness),
        "faithfulness_ci": bootstrap_faithfulness_ci(
            replay_margin, full_base_margin, corrupted_margin,
            samples=config.bootstrap_samples, alpha=config.alpha,
            seed=seed + 307),
        "faithfulness_endpoint": config.program_faithfulness_endpoint,
    }


def evaluate_native_operator_program_phase_baselines(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        base_schedule: OperatorProgramSchedule,
        source_schedule: OperatorProgramSchedule,
        shape: OperatorSpaceShape, pad_token_id: int,
) -> NativeProgramPhaseBaselines:
    """Evaluate the four production margins exactly once for a phase."""
    example_ids = _validate_native_operator_program_phase(
        examples, base_schedule=base_schedule, source_schedule=source_schedule,
        shape=shape)
    mode0 = PROGRAM_MODES["production"]
    full_base_margin = _program_margin(
        ctx, examples, base_schedule, prompt_side="base",
        positive_side="positive", negative_side="negative",
        pad_token_id=pad_token_id, program_mode=mode0)
    corrupted_margin = _program_margin(
        ctx, examples, source_schedule, prompt_side="source",
        positive_side="positive", negative_side="negative",
        pad_token_id=pad_token_id, program_mode=mode0)
    before_base_to_source = _program_margin(
        ctx, examples, base_schedule, prompt_side="base",
        positive_side="intervention_positive",
        negative_side="intervention_negative",
        pad_token_id=pad_token_id, program_mode=mode0)
    before_source_to_base = _program_margin(
        ctx, examples, source_schedule, prompt_side="source",
        positive_side="source_negative", negative_side="source_positive",
        pad_token_id=pad_token_id, program_mode=mode0)
    return NativeProgramPhaseBaselines(
        full_base_margin=full_base_margin,
        corrupted_margin=corrupted_margin,
        before_base_to_source=before_base_to_source,
        before_source_to_base=before_source_to_base,
        example_ids=example_ids,
        phase=str(examples[0].phase),
    )


def evaluate_native_operator_program_selection_candidate(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        base_schedule: OperatorProgramSchedule,
        source_schedule: OperatorProgramSchedule,
        baselines: NativeProgramPhaseBaselines,
        shape: OperatorSpaceShape, pad_token_id: int,
        config: ProtocolConfig, seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate only the replay and compactness gates used to select mass."""
    _validate_native_operator_program_phase(
        examples, base_schedule=base_schedule, source_schedule=source_schedule,
        shape=shape)
    _validate_native_program_phase_baselines(baselines, examples)
    replay_margin = _program_margin(
        ctx, examples, base_schedule, prompt_side="base",
        positive_side="positive", negative_side="negative",
        pad_token_id=pad_token_id,
        program_mode=PROGRAM_MODES["own_id_replay"])
    replay_margin = np.asarray(replay_margin, dtype=np.float64)
    replay = _native_program_replay_summary(
        replay_margin, baselines, config=config, seed=seed)
    compactness = compactness_metrics(
        base_schedule, shape=shape, paired_schedule=source_schedule)
    return {
        "status": "ready",
        "phase": str(examples[0].phase),
        "program_mass": float(base_schedule.program_mass),
        "example_count": len(examples),
        "decision_scope": "final_ioi_decision_at_answer_position",
        "program_position_scope": config.program_position_scope,
        "program_routes": list(config.program_routes),
        "program_denominator_policy": config.program_denominator_policy,
        "candidate_score_normalization": "sum_log_probability",
        "selection_only": True,
        "causal_diagnostics_evaluated": False,
        "compactness": compactness,
        "replay": replay,
    }, replay_margin


def evaluate_native_operator_program_causal_diagnostics(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        base_schedule: OperatorProgramSchedule,
        source_schedule: OperatorProgramSchedule,
        baselines: NativeProgramPhaseBaselines,
        selection_candidate: Mapping[str, Any],
        replay_margin: np.ndarray,
        shape: OperatorSpaceShape, pad_token_id: int,
        config: ProtocolConfig, seed: int,
) -> tuple[dict[str, Any], dict[str, OperatorProgramSchedule]]:
    """Run controls and causal diagnostics for one already-frozen mass."""
    _validate_native_operator_program_phase(
        examples, base_schedule=base_schedule, source_schedule=source_schedule,
        shape=shape)
    _validate_native_program_phase_baselines(baselines, examples)
    if float(selection_candidate.get("program_mass", -1.0)) != float(
            base_schedule.program_mass):
        raise ValueError("selection replay and diagnostic program masses differ")
    if str(selection_candidate.get("phase")) != str(examples[0].phase):
        raise ValueError("selection replay and diagnostic phases differ")
    if int(selection_candidate.get("example_count", -1)) != len(examples):
        raise ValueError("selection replay and diagnostic examples differ")
    if selection_candidate.get("selection_only") is not True:
        raise ValueError("causal diagnostics require a selection-only replay")
    if selection_candidate.get("causal_diagnostics_evaluated") is not False:
        raise ValueError("selection replay already contains causal diagnostics")
    replay_margin = np.asarray(replay_margin, dtype=np.float64)
    if replay_margin.shape != (len(examples),):
        raise ValueError("selection replay margin has the wrong shape")
    expected_replay = _native_program_replay_summary(
        replay_margin, baselines, config=config, seed=seed)
    if canonical_hash(expected_replay) != canonical_hash(
            selection_candidate.get("replay")):
        raise ValueError("selection replay metrics do not match replay margins")
    replay = copy.deepcopy(dict(selection_candidate["replay"]))
    full_base_margin = baselines.full_base_margin
    corrupted_margin = baselines.corrupted_margin
    before_b2s = baselines.before_base_to_source
    before_s2b = baselines.before_source_to_base

    mismatch_source_mapping = deterministic_mismatch_mapping(
        examples, source_schedule, seed=seed + 101)
    mismatch_base_mapping = deterministic_mismatch_mapping(
        examples, base_schedule, seed=seed + 113)
    source_donor_indices = mismatch_source_mapping["donor_indices"]
    base_donor_indices = mismatch_base_mapping["donor_indices"]
    recipient_ids = [example.example_id for example in examples]
    mismatch_source = reindex_program_schedule(
        source_schedule, source_donor_indices,
        recipient_example_ids=recipient_ids,
        prompt_side="mismatched_source", shape=shape)
    mismatch_base = reindex_program_schedule(
        base_schedule, base_donor_indices,
        recipient_example_ids=recipient_ids,
        prompt_side="mismatched_base", shape=shape)
    random_source = random_program_schedule(
        source_schedule, shape=shape, seed=seed + 211)
    random_base = random_program_schedule(
        base_schedule, shape=shape, seed=seed + 223)
    source_donor_examples = [
        examples[int(index)] for index in source_donor_indices]
    base_donor_examples = [
        examples[int(index)] for index in base_donor_indices]

    paired_source_contribution = capture_operator_program_contributions(
        ctx, examples, source_schedule, prompt_side="source",
        pad_token_id=pad_token_id)
    paired_base_contribution = capture_operator_program_contributions(
        ctx, examples, base_schedule, prompt_side="base",
        pad_token_id=pad_token_id)
    mismatch_source_contribution = capture_operator_program_contributions(
        ctx, source_donor_examples, mismatch_source, prompt_side="source",
        pad_token_id=pad_token_id)
    mismatch_base_contribution = capture_operator_program_contributions(
        ctx, base_donor_examples, mismatch_base, prompt_side="base",
        pad_token_id=pad_token_id)
    random_source_contribution = capture_operator_program_contributions(
        ctx, examples, random_source, prompt_side="source",
        pad_token_id=pad_token_id)
    random_base_contribution = capture_operator_program_contributions(
        ctx, examples, random_base, prompt_side="base",
        pad_token_id=pad_token_id)

    own_ablated = _program_margin(
        ctx, examples, base_schedule, prompt_side="base",
        positive_side="positive", negative_side="negative",
        pad_token_id=pad_token_id,
        program_mode=PROGRAM_MODES["own_id_ablation"])
    mismatch_ablated = _program_margin(
        ctx, examples, mismatch_base, prompt_side="base",
        positive_side="positive", negative_side="negative",
        pad_token_id=pad_token_id,
        program_mode=PROGRAM_MODES["own_id_ablation"])
    random_ablated = _program_margin(
        ctx, examples, random_base, prompt_side="base",
        positive_side="positive", negative_side="negative",
        pad_token_id=pad_token_id,
        program_mode=PROGRAM_MODES["own_id_ablation"])
    own_drop = full_base_margin - own_ablated
    mismatch_drop = full_base_margin - mismatch_ablated
    random_drop = full_base_margin - random_ablated
    ablation = {
        "control_schedule_side": "base",
        "mismatched_schedule_hash": mismatch_base.schedule_hash,
        "random_schedule_hash": random_base.schedule_hash,
        "own_program": _ablation_metrics(
            full_base_margin, own_ablated, config=config, seed=seed + 401),
        "mismatched_program": _ablation_metrics(
            full_base_margin, mismatch_ablated,
            config=config, seed=seed + 419),
        "random_program": _ablation_metrics(
            full_base_margin, random_ablated,
            config=config, seed=seed + 431),
        "specificity": {
            "own_vs_mismatched": _paired_advantage_metrics(
                own_drop, mismatch_drop, config=config, seed=seed + 443),
            "own_vs_random": _paired_advantage_metrics(
                own_drop, random_drop, config=config, seed=seed + 457),
        },
    }

    def direction(
            schedule, prompt_side, positive_side, negative_side, mode,
            source=None):
        return _program_margin(
            ctx, examples, schedule, prompt_side=prompt_side,
            positive_side=positive_side, negative_side=negative_side,
            pad_token_id=pad_token_id, program_mode=mode, source=source)

    paired_id_b2s = direction(
        source_schedule, "base", "intervention_positive",
        "intervention_negative", PROGRAM_MODES["source_id_replay"])
    paired_id_s2b = direction(
        base_schedule, "source", "source_negative", "source_positive",
        PROGRAM_MODES["source_id_replay"])
    mismatch_id_b2s = direction(
        mismatch_source, "base", "intervention_positive",
        "intervention_negative", PROGRAM_MODES["source_id_replay"])
    mismatch_id_s2b = direction(
        mismatch_base, "source", "source_negative", "source_positive",
        PROGRAM_MODES["source_id_replay"])
    random_id_b2s = direction(
        random_source, "base", "intervention_positive",
        "intervention_negative", PROGRAM_MODES["source_id_replay"])
    random_id_s2b = direction(
        random_base, "source", "source_negative", "source_positive",
        PROGRAM_MODES["source_id_replay"])

    paired_id_improvement = 0.5 * (
        (paired_id_b2s - before_b2s) + (paired_id_s2b - before_s2b))
    mismatch_id_improvement = 0.5 * (
        (mismatch_id_b2s - before_b2s)
        + (mismatch_id_s2b - before_s2b))
    random_id_improvement = 0.5 * (
        (random_id_b2s - before_b2s) + (random_id_s2b - before_s2b))
    paired_id_vs_mismatch = paired_id_improvement - mismatch_id_improvement
    paired_id_vs_random = paired_id_improvement - random_id_improvement
    paired_id_flip_b2s = (before_b2s <= 0.0) & (paired_id_b2s > 0.0)
    paired_id_flip_s2b = (before_s2b <= 0.0) & (paired_id_s2b > 0.0)

    paired_b2s = direction(
        source_schedule, "base", "intervention_positive",
        "intervention_negative",
        PROGRAM_MODES["source_contribution_transplant"],
        paired_source_contribution)
    paired_s2b = direction(
        base_schedule, "source", "source_negative", "source_positive",
        PROGRAM_MODES["source_contribution_transplant"],
        paired_base_contribution)
    mismatch_b2s = direction(
        mismatch_source, "base", "intervention_positive",
        "intervention_negative",
        PROGRAM_MODES["source_contribution_transplant"],
        mismatch_source_contribution)
    mismatch_s2b = direction(
        mismatch_base, "source", "source_negative", "source_positive",
        PROGRAM_MODES["source_contribution_transplant"],
        mismatch_base_contribution)
    random_b2s = direction(
        random_source, "base", "intervention_positive",
        "intervention_negative",
        PROGRAM_MODES["source_contribution_transplant"],
        random_source_contribution)
    random_s2b = direction(
        random_base, "source", "source_negative", "source_positive",
        PROGRAM_MODES["source_contribution_transplant"],
        random_base_contribution)

    paired_improvement = 0.5 * (
        (paired_b2s - before_b2s) + (paired_s2b - before_s2b))
    mismatch_improvement = 0.5 * (
        (mismatch_b2s - before_b2s) + (mismatch_s2b - before_s2b))
    random_improvement = 0.5 * (
        (random_b2s - before_b2s) + (random_s2b - before_s2b))
    paired_vs_mismatch = paired_improvement - mismatch_improvement
    paired_vs_random = paired_improvement - random_improvement
    paired_flip_b2s = (before_b2s <= 0.0) & (paired_b2s > 0.0)
    paired_flip_s2b = (before_s2b <= 0.0) & (paired_s2b > 0.0)
    source_id = {
        "paired": {
            "base_to_source": _direction_metrics(before_b2s, paired_id_b2s),
            "source_to_base": _direction_metrics(before_s2b, paired_id_s2b),
            "mean_bidirectional_improvement": float(
                np.mean(paired_id_improvement)),
        },
        "mismatched": {
            "base_to_source": _direction_metrics(
                before_b2s, mismatch_id_b2s),
            "source_to_base": _direction_metrics(
                before_s2b, mismatch_id_s2b),
        },
        "random": {
            "base_to_source": _direction_metrics(before_b2s, random_id_b2s),
            "source_to_base": _direction_metrics(before_s2b, random_id_s2b),
        },
        "paired_vs_mismatch": _paired_advantage_metrics(
            paired_id_improvement, mismatch_id_improvement,
            config=config, seed=seed + 509),
        "paired_vs_random": _paired_advantage_metrics(
            paired_id_improvement, random_id_improvement,
            config=config, seed=seed + 523),
        "bidirectional_answer_flip_fraction": float(np.mean(
            paired_id_flip_b2s & paired_id_flip_s2b)),
    }
    transplant = {
        "paired": {
            "base_to_source": _direction_metrics(before_b2s, paired_b2s),
            "source_to_base": _direction_metrics(before_s2b, paired_s2b),
            "mean_bidirectional_improvement": float(
                np.mean(paired_improvement)),
        },
        "mismatched": {
            "base_to_source": _direction_metrics(before_b2s, mismatch_b2s),
            "source_to_base": _direction_metrics(before_s2b, mismatch_s2b),
            "mean_bidirectional_improvement": float(
                np.mean(mismatch_improvement)),
        },
        "random": {
            "base_to_source": _direction_metrics(before_b2s, random_b2s),
            "source_to_base": _direction_metrics(before_s2b, random_s2b),
            "mean_bidirectional_improvement": float(
                np.mean(random_improvement)),
        },
        "paired_vs_mismatch": {
            **_paired_advantage_metrics(
                paired_improvement, mismatch_improvement,
                config=config, seed=seed + 607),
        },
        "paired_vs_random": {
            **_paired_advantage_metrics(
                paired_improvement, random_improvement,
                config=config, seed=seed + 631),
        },
        "bidirectional_answer_flip_fraction": float(np.mean(
            paired_flip_b2s & paired_flip_s2b)),
    }
    compactness = compactness_metrics(
        base_schedule, shape=shape, paired_schedule=source_schedule,
        mismatched_schedule=mismatch_base)
    selection_compactness = selection_candidate.get("compactness")
    if not isinstance(selection_compactness, Mapping):
        raise ValueError("selection replay is missing compactness metrics")
    selection_keys = (
        "scope", "decision_position_site_fraction",
        "median_decision_position_site_fraction",
        "mean_decision_position_site_fraction",
        "per_route_decision_position_site_fraction",
        "union_decision_position_fraction_vs_example_count",
        "same_pair_route_overlap",
    )
    if canonical_hash({
            key: selection_compactness.get(key) for key in selection_keys
    }) != canonical_hash({key: compactness.get(key) for key in selection_keys}):
        raise ValueError(
            "selection replay and causal diagnostic compactness differ")
    result = {
        "status": "ready",
        "phase": examples[0].phase,
        "program_mass": float(base_schedule.program_mass),
        "example_count": len(examples),
        "decision_scope": "final_ioi_decision_at_answer_position",
        "program_position_scope": config.program_position_scope,
        "program_routes": list(config.program_routes),
        "program_denominator_policy": config.program_denominator_policy,
        "candidate_score_normalization": "sum_log_probability",
        "selection_replay_reused": True,
        "causal_diagnostics_evaluated": True,
        "compactness": compactness,
        "replay": replay,
        "ablation": ablation,
        "source_id_replay": source_id,
        "transplant": transplant,
        "mismatch_source_mapping": mismatch_source_mapping,
        "mismatch_base_mapping": mismatch_base_mapping,
        "random_control": {
            "seed_source": seed + 211,
            "seed_base": seed + 223,
            "sampling_policy": config.program_random_sampling,
            "ablation_reference_side": "base",
            "count_preserved_per_example_layer_route": True,
            "without_replacement": True,
            "source": _random_overlap_metrics(random_source),
            "base": _random_overlap_metrics(random_base),
        },
        "primary_effect_vectors_persisted_in_item_json": False,
        "_effect_vectors": {
            "example_ids": np.asarray(
                [example.example_id for example in examples]),
            "full_base_margin": full_base_margin,
            "corrupted_endpoint_margin": corrupted_margin,
            "replay_base_margin": replay_margin,
            "own_ablated_margin": own_ablated,
            "mismatched_ablated_margin": mismatch_ablated,
            "random_ablated_margin": random_ablated,
            "own_program_margin_drop": own_drop,
            "mismatched_program_margin_drop": mismatch_drop,
            "random_program_margin_drop": random_drop,
            "own_vs_mismatched_margin_drop": own_drop - mismatch_drop,
            "own_vs_random_margin_drop": own_drop - random_drop,
            "counterfactual_before_base_to_source": before_b2s,
            "counterfactual_before_source_to_base": before_s2b,
            "source_id_paired_base_to_source": paired_id_b2s,
            "source_id_paired_source_to_base": paired_id_s2b,
            "source_id_mismatched_base_to_source": mismatch_id_b2s,
            "source_id_mismatched_source_to_base": mismatch_id_s2b,
            "source_id_random_base_to_source": random_id_b2s,
            "source_id_random_source_to_base": random_id_s2b,
            "source_id_paired_improvement": paired_id_improvement,
            "source_id_mismatched_improvement": mismatch_id_improvement,
            "source_id_random_improvement": random_id_improvement,
            "source_id_paired_vs_mismatch_improvement": (
                paired_id_vs_mismatch),
            "source_id_paired_vs_random_improvement": paired_id_vs_random,
            "source_id_bidirectional_pair_success": (
                paired_id_flip_b2s & paired_id_flip_s2b),
            "transplant_paired_base_to_source": paired_b2s,
            "transplant_paired_source_to_base": paired_s2b,
            "transplant_mismatched_base_to_source": mismatch_b2s,
            "transplant_mismatched_source_to_base": mismatch_s2b,
            "transplant_random_base_to_source": random_b2s,
            "transplant_random_source_to_base": random_s2b,
            "paired_vs_mismatch_improvement": paired_vs_mismatch,
            "paired_vs_random_improvement": paired_vs_random,
            "bidirectional_pair_success": (
                paired_flip_b2s & paired_flip_s2b),
        },
    }
    controls = {
        "mismatch_source": mismatch_source,
        "mismatch_base": mismatch_base,
        "random_source": random_source,
        "random_base": random_base,
    }
    return result, controls


def evaluate_native_operator_program_candidate(
        ctx: Any, examples: Sequence[BenchmarkExample], *,
        base_schedule: OperatorProgramSchedule,
        source_schedule: OperatorProgramSchedule,
        shape: OperatorSpaceShape, pad_token_id: int,
        config: ProtocolConfig, seed: int,
) -> tuple[dict[str, Any], dict[str, OperatorProgramSchedule]]:
    """Compatibility entrypoint for a single full native-program evaluation."""
    baselines = evaluate_native_operator_program_phase_baselines(
        ctx, examples, base_schedule=base_schedule,
        source_schedule=source_schedule, shape=shape,
        pad_token_id=pad_token_id)
    selection_candidate, replay_margin = (
        evaluate_native_operator_program_selection_candidate(
            ctx, examples, base_schedule=base_schedule,
            source_schedule=source_schedule, baselines=baselines,
            shape=shape, pad_token_id=pad_token_id,
            config=config, seed=seed))
    return evaluate_native_operator_program_causal_diagnostics(
        ctx, examples, base_schedule=base_schedule,
        source_schedule=source_schedule, baselines=baselines,
        selection_candidate=selection_candidate,
        replay_margin=replay_margin, shape=shape,
        pad_token_id=pad_token_id, config=config, seed=seed)
