"""Exact production-path behavior, retention, suppression, and interchange."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.operator_interpretability.benchmark_schema import BenchmarkExample
from analysis.operator_interpretability.units import (
    OperatorCircuit,
    OperatorSpaceShape,
    ROUTE_INDEX,
)


def _runtime_kwargs(ctx: Any) -> dict[str, Any]:
    cfg = ctx.model_cfg
    temperature = float(cfg["soft_gate_temperature"])
    boundary = float(cfg["soft_gate_boundary_power"])
    return {
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(0)},
        "sharded_fns": ctx.sharded_fns,
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


def _score_from_result(result: Mapping[str, Any]) -> jax.Array:
    valid = result["valid_mask"].astype(jnp.float32)
    return (-result["per_token_ce"].astype(jnp.float32) * valid).sum(axis=-1)


def _plain_score_executable(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    if "plain_score" in cache:
        return cache["plain_score"]
    kwargs = _runtime_kwargs(ctx)

    @jax.jit
    def score(params, input_ids, labels):
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            attention_mask=(input_ids >= 0).astype(jnp.int32),
            minimal_train=True, **kwargs)
        return _score_from_result(result)

    cache["plain_score"] = score
    return score


def _retention_score_executable(ctx: Any, mode: str):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = f"retention_score:{mode}"
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx)

    @jax.jit
    def score(params, input_ids, labels, keep_qk, keep_v, keep_rst):
        result = ctx.model.apply(
            {"params": params}, input_ids, keep_qk, keep_v, keep_rst,
            mode=mode,
            position_mask=jnp.ones_like(input_ids, dtype=jnp.bool_),
            labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=False,
            method=ctx.model.analysis_forward_with_circuit_retention,
            **kwargs)
        return _score_from_result(result)

    cache[key] = score
    return score


def evaluate_behavior(ctx: Any, examples: Sequence[BenchmarkExample], *,
                      pad_token_id: int) -> dict[str, Any]:
    if not examples:
        raise ValueError("behavior evaluation has no examples")
    multiple = max(1, int(ctx.mesh.shape["data"]))
    score = _plain_score_executable(ctx)

    def run(prompt_side: str, answer_side: str) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side=prompt_side, answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        return np.asarray(jax.device_get(score(ctx.params, ids, labels)))[
            :arrays[2]].astype(np.float64)

    base_positive = run("base", "positive")
    base_negative = run("base", "negative")
    source_base_positive = run("source", "positive")
    source_base_negative = run("source", "negative")
    source_own_positive = run("source", "source_positive")
    source_own_negative = run("source", "source_negative")
    base_margin = base_positive - base_negative
    # Official MIB faithfulness keeps the clean-label orientation on the
    # corrupted/source prompt.  source_own_margin separately audits whether
    # the checkpoint solves the source member of the counterfactual pair.
    corrupted_margin = source_base_positive - source_base_negative
    source_own_margin = source_own_positive - source_own_negative
    base_known_correct = base_margin > 0.0
    source_known_correct = source_own_margin > 0.0
    pair_known_correct = base_known_correct & source_known_correct
    return {
        "status": "ready",
        "phase": examples[0].phase,
        "example_ids": [example.example_id for example in examples],
        "base_positive_logp": base_positive.tolist(),
        "base_negative_logp": base_negative.tolist(),
        "base_margin": base_margin.tolist(),
        "corrupted_margin": corrupted_margin.tolist(),
        "source_own_margin": source_own_margin.tolist(),
        "base_known_correct": base_known_correct.tolist(),
        "source_known_correct": source_known_correct.tolist(),
        "known_correct": pair_known_correct.tolist(),
        "known_correct_count": int(np.sum(pair_known_correct)),
        "base_known_correct_count": int(np.sum(base_known_correct)),
        "source_known_correct_count": int(np.sum(source_known_correct)),
        "accuracy": float(np.mean(base_known_correct)),
        "pair_accuracy": float(np.mean(pair_known_correct)),
        "mean_margin": float(np.mean(base_margin)),
        "mean_corrupted_margin": float(np.mean(corrupted_margin)),
        "mean_source_own_margin": float(np.mean(source_own_margin)),
        "corrupted_margin_orientation": (
            "base_positive_minus_base_negative_on_source_prompt"),
        "mechanistic_eligibility": "base_and_source_both_known_correct",
    }


def evaluate_circuit_retention(
        ctx: Any, examples: Sequence[BenchmarkExample], circuit: OperatorCircuit,
        *, shape: OperatorSpaceShape, mode: str,
        pad_token_id: int) -> dict[str, Any]:
    multiple = max(1, int(ctx.mesh.shape["data"]))
    masks = circuit.dense_masks(shape)
    score = _retention_score_executable(ctx, mode)

    def run(answer_side: str) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side="base", answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        value = score(
            ctx.params, ids, labels,
            jnp.asarray(masks["qk"]), jnp.asarray(masks["v"]),
            jnp.asarray(masks["rst"]))
        return np.asarray(jax.device_get(value))[:arrays[2]].astype(np.float64)

    positive = run("positive")
    negative = run("negative")
    margin = positive - negative
    return {
        "status": "ready",
        "phase": examples[0].phase,
        "mode": mode,
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
    complement = {key: ~value for key, value in masks.items()}
    # Invoke the dense masks directly; materializing millions of complement
    # OperatorSite objects would add no scientific information.
    multiple = max(1, int(ctx.mesh.shape["data"]))
    score = _retention_score_executable(
        ctx, "conditional_execution_sufficiency")

    def run(answer_side: str) -> np.ndarray:
        arrays = _sequence_arrays(
            examples, prompt_side="base", answer_side=answer_side,
            pad_token_id=pad_token_id, multiple=multiple)
        ids, labels = _device_batch(ctx, arrays[0], arrays[1])
        value = score(
            ctx.params, ids, labels,
            jnp.asarray(complement["qk"]), jnp.asarray(complement["v"]),
            jnp.asarray(complement["rst"]))
        return np.asarray(jax.device_get(value))[:arrays[2]].astype(np.float64)

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
    plain = _plain_score_executable(ctx)
    retained = _retention_score_executable(ctx, mode)
    ones_qk = jnp.ones((shape.n_layers, 2, shape.n_qk), dtype=jnp.bool_)
    ones_v = jnp.ones((shape.n_layers, shape.n_v), dtype=jnp.bool_)
    ones_rst = jnp.ones((shape.n_layers, shape.n_rst), dtype=jnp.bool_)
    before = jax.device_get(plain(ctx.params, ids, labels))
    after = jax.device_get(retained(
        ctx.params, ids, labels, ones_qk, ones_v, ones_rst))
    exact = bool(np.array_equal(np.asarray(before), np.asarray(after)))
    if not exact:
        raise RuntimeError("all-ones circuit retention parity is not machine-exact")
    return {
        "status": "passed",
        "machine_exact": True,
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
    if route not in ROUTE_INDEX:
        raise ValueError(f"unknown interchange route: {route}")
    if not operator_ids:
        raise ValueError("interchange operator family is empty")
    multiple = max(1, int(ctx.mesh.shape["data"]))
    source_arrays = _prompt_arrays(
        examples, side="source", pad_token_id=pad_token_id,
        multiple=multiple)
    source_ids = jax.device_put(
        jnp.asarray(source_arrays[0]), NamedSharding(ctx.mesh, P("data", None)))
    source_positions = jax.device_put(
        jnp.asarray(source_arrays[1]), NamedSharding(ctx.mesh, P("data")))
    width = len(operator_ids)
    selected = np.tile(
        np.asarray(operator_ids, dtype=np.int32)[None, :],
        (source_arrays[0].shape[0], 1))
    selected_ids = jax.device_put(
        jnp.asarray(selected), NamedSharding(ctx.mesh, P("data", None)))
    kwargs = _runtime_kwargs(ctx)
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

    @jax.jit
    def patched_score(params, ids, labels, positions, selected_group, source):
        result = ctx.model.apply(
            {"params": params}, ids, selected_group, jnp.int32(layer),
            positions, jnp.int32(route_index), source,
            labels=labels, attention_mask=jnp.ones_like(ids),
            return_residual=False,
            method=ctx.model.analysis_forward_with_operator_interchange,
            **kwargs)
        return _score_from_result(result)

    plain = _plain_score_executable(ctx)

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
        return np.asarray(jax.device_get(value))[:arrays[2]].astype(np.float64)

    base_base = scores("base", "positive", patch=False) - scores(
        "base", "negative", patch=False)
    base_source = scores("base", "source_positive", patch=False) - scores(
        "base", "source_negative", patch=False)
    source_source = scores("source", "source_positive", patch=False) - scores(
        "source", "source_negative", patch=False)
    patched_source = scores("base", "source_positive", patch=True) - scores(
        "base", "source_negative", patch=True)
    patched_base = scores("base", "positive", patch=True) - scores(
        "base", "negative", patch=True)
    return [
        {
            "example_id": example.example_id,
            "pair_type": example.pair_type,
            "causal_variable": example.causal_variable,
            "layer": int(layer),
            "route": route,
            "operator_ids": [int(value) for value in operator_ids],
            "position_kind": example.metadata.get("position_kind"),
            "base_base_margin": float(base_base[index]),
            "base_source_margin": float(base_source[index]),
            "source_source_margin": float(source_source[index]),
            "patched_source_margin": float(patched_source[index]),
            "patched_base_margin": float(patched_base[index]),
        }
        for index, example in enumerate(examples)
    ]
