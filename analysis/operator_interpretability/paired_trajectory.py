"""Exact production-bound IOI S2 paired operator trajectory analysis."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import (
    analysis_model_module,
    create_or_reuse_sharded_fns,
    materialize_global_array,
    materialize_global_tree,
)
from analysis.dawn_analysis_storage import write_json_atomic, write_npz_atomic
from analysis.operator_interpretability.artifacts import sha256_path
from analysis.operator_interpretability.benchmark_schema import (
    BenchmarkExample,
    canonical_hash,
)
from analysis.operator_interpretability.benchmarks.mib_ioi import (
    _render_semantic_prompt,
    _semantic_spans_from_rendered_prompt,
)
from analysis.operator_interpretability.eligibility import (
    _token_span_from_char_span,
)
from analysis.operator_interpretability.intervention import (
    _candidate_score_normalization,
    _runtime_kwargs,
    _score_from_result,
)
from analysis.operator_interpretability.protocol import (
    PAIRED_TRAJECTORY_ALGORITHM_VERSION,
    ProtocolConfig,
)
from analysis.operator_interpretability.statistics import (
    bootstrap_mean_ci,
    paired_permutation_test,
)


ROUTES = ("q", "k", "v", "rst")
ROUTE_INDEX = {route: index for index, route in enumerate(ROUTES)}
PATCH_STAGE = {
    "q": 0,
    "k": 1,
    "v": 2,
    "rst": 3,
    "residual_input": 4,
    "post_attention": 5,
    "post_rst": 6,
}
STATE_STAGES = (
    "residual_input", "post_attention", "post_rst",
)
TRACE_OPERATOR_FIELDS = (
    "operator_id",
    "operator_valid",
    "read_scalar_bf16_bits",
    "prewrite_amplitude_bf16_bits",
    "execution_weight",
    "admission",
    "margin",
    "rho",
)


@dataclass(frozen=True)
class IOISemanticRecord:
    example_id: str
    s2_start: int
    s2_end: int
    answer_position: int
    positions: tuple[int, ...]
    position_roles: tuple[str, ...]
    base_prompt_tokens: int
    source_prompt_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "s2_start": self.s2_start,
            "s2_end": self.s2_end,
            "answer_position": self.answer_position,
            "positions": list(self.positions),
            "position_roles": list(self.position_roles),
            "base_prompt_tokens": self.base_prompt_tokens,
            "source_prompt_tokens": self.source_prompt_tokens,
        }


def _semantic_char_spans(example: BenchmarkExample) -> Mapping[str, Any]:
    metadata = dict(example.metadata)
    existing = metadata.get("semantic_char_spans")
    if isinstance(existing, Mapping):
        return existing
    template = str(metadata.get("template") or "")
    subject = str(metadata.get("subject") or "")
    indirect_object = str(metadata.get("indirect_object") or "")
    if not template or not subject or not indirect_object:
        raise ValueError(
            f"{example.example_id}: IOI semantic template metadata missing")
    if metadata.get("place") and metadata.get("object"):
        base_prompt, base = _render_semantic_prompt(
            template, metadata, s2_name=subject)
        source_prompt, source = _render_semantic_prompt(
            template, metadata, s2_name=indirect_object)
        if (base_prompt != example.base_prompt
                or source_prompt != example.source_prompt):
            raise ValueError(
                f"{example.example_id}: IOI template rendering drift")
    else:
        base = _semantic_spans_from_rendered_prompt(
            template, example.base_prompt, {
                "name_A": subject,
                "name_B": indirect_object,
                "name_C": subject,
            })
        source = _semantic_spans_from_rendered_prompt(
            template, example.source_prompt, {
                "name_A": subject,
                "name_B": indirect_object,
                "name_C": indirect_object,
            })
    return {
        "base": {
            "first_name_a": base["name_A"],
            "first_name_b": base["name_B"],
            "s2_counterfactual": base["name_C"],
        },
        "source": {
            "first_name_a": source["name_A"],
            "first_name_b": source["name_B"],
            "s2_counterfactual": source["name_C"],
        },
    }


def ioi_semantic_record(
        example: BenchmarkExample, tokenizer: Any) -> IOISemanticRecord:
    """Resolve S2 from official template roles, including old build rows."""
    example.validate()
    if (tuple(example.source_positive_ids) != tuple(example.negative_ids)
            or tuple(example.source_negative_ids)
            != tuple(example.positive_ids)
            or tuple(example.intervention_positive_ids)
            != tuple(example.negative_ids)
            or tuple(example.intervention_negative_ids)
            != tuple(example.positive_ids)):
        raise ValueError(
            f"{example.example_id}: IOI candidate orientation is invalid")
    token_spans = example.metadata.get("semantic_token_spans")
    if isinstance(token_spans, Mapping):
        base_spans = token_spans.get("base")
        source_spans = token_spans.get("source")
        if not isinstance(base_spans, Mapping) or not isinstance(
                source_spans, Mapping):
            raise ValueError(
                f"{example.example_id}: malformed semantic token spans")
        base_s2 = tuple(int(value) for value in (
            base_spans.get("s2_counterfactual") or ()))
        source_s2 = tuple(int(value) for value in (
            source_spans.get("s2_counterfactual") or ()))
    else:
        char_spans = _semantic_char_spans(example)
        base_s2 = _token_span_from_char_span(
            tokenizer, example.base_prompt,
            char_spans["base"]["s2_counterfactual"],
            role="base/s2_counterfactual")
        source_s2 = _token_span_from_char_span(
            tokenizer, example.source_prompt,
            char_spans["source"]["s2_counterfactual"],
            role="source/s2_counterfactual")
    if len(base_s2) != 2 or len(source_s2) != 2:
        raise ValueError(f"{example.example_id}: invalid S2 token span")
    if base_s2 != source_s2:
        raise ValueError(
            f"{example.example_id}: base/source S2 spans are not aligned")
    if len(example.input_ids_base) != len(example.input_ids_source):
        raise ValueError(
            f"{example.example_id}: paired trajectory requires aligned prompts")
    s2_start, s2_end = base_s2
    if tuple(example.input_ids_base[:s2_start]) != tuple(
            example.input_ids_source[:s2_start]):
        raise ValueError(
            f"{example.example_id}: base/source diverge before S2")
    if (example.trace_position_base != example.trace_position_source
            or example.trace_position_base < s2_end):
        raise ValueError(
            f"{example.example_id}: answer position is not S2-aligned")
    answer_position = int(example.trace_position_base)
    positions = tuple(range(int(s2_start), answer_position + 1))
    roles = []
    for position in positions:
        if position < s2_end:
            s2_offset = position - s2_start
            role = "s2" if s2_offset == 0 else f"s2_subtoken_{s2_offset}"
        elif position == answer_position:
            role = "answer_position"
        elif position == s2_end:
            role = "post_s2"
        else:
            role = f"post_s2_offset_{position - s2_end}"
        roles.append(role)
    return IOISemanticRecord(
        example_id=example.example_id,
        s2_start=int(s2_start),
        s2_end=int(s2_end),
        answer_position=answer_position,
        positions=positions,
        position_roles=tuple(roles),
        base_prompt_tokens=len(example.input_ids_base),
        source_prompt_tokens=len(example.input_ids_source),
    )


def deterministic_phase_selection(
        examples: Sequence[BenchmarkExample], *, limit: int,
        seed: int) -> list[BenchmarkExample]:
    if limit <= 0:
        raise ValueError("trajectory phase limit must be positive")
    ranked = sorted(examples, key=lambda example: (
        hashlib.sha256(
            f"{seed}:{example.example_id}".encode("utf-8")).hexdigest(),
        example.example_id,
    ))
    return ranked[:min(limit, len(ranked))]


def deterministic_deep_selection(
        examples: Sequence[BenchmarkExample], *, limit: int,
        seed: int) -> list[BenchmarkExample]:
    """Select a seeded deep cohort with an in-cohort mismatch donor."""
    ranked = deterministic_phase_selection(
        examples, limit=len(examples), seed=seed)
    selected: list[BenchmarkExample] = []
    selected_ids: set[str] = set()
    for recipient in ranked:
        if recipient.example_id in selected_ids:
            continue
        recipient_answers = {
            tuple(recipient.positive_ids), tuple(recipient.negative_ids)}
        eligible = [
            donor for donor in ranked
            if donor.example_id not in selected_ids
            and donor.example_id != recipient.example_id
            and str(donor.metadata.get("template") or "")
            == str(recipient.metadata.get("template") or "")
            and not recipient_answers.intersection({
                tuple(donor.positive_ids), tuple(donor.negative_ids)})
        ]
        if not eligible:
            continue
        donor = eligible[0]
        for value in (recipient, donor):
            if len(selected) >= limit:
                break
            selected.append(value)
            selected_ids.add(value.example_id)
        if len(selected) >= limit:
            break
    if len(selected) < 2:
        raise ValueError(
            "deep trajectory cohort has no answer-disjoint same-template "
            "mismatch pair")
    return selected[:limit]


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _pad_example_batch(
        examples: Sequence[BenchmarkExample], target_count: int
) -> list[BenchmarkExample]:
    values = list(examples)
    if not values:
        raise ValueError("cannot pad an empty trajectory example batch")
    if len(values) > int(target_count):
        raise ValueError("trajectory example batch exceeds its fixed width")
    return values + [values[0]] * (int(target_count) - len(values))


def _candidate_variants(example: BenchmarkExample):
    return (
        ("base", "positive", example.input_ids_base, example.positive_ids),
        ("base", "negative", example.input_ids_base, example.negative_ids),
        ("source", "positive", example.input_ids_source,
         example.source_positive_ids),
        ("source", "negative", example.input_ids_source,
         example.source_negative_ids),
    )


def build_deep_trace_batch(
        examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *,
        pad_token_id: int, data_multiple: int,
        fixed_sequence_length: int | None = None,
        fixed_trace_width: int | None = None) -> dict[str, Any]:
    if not examples:
        raise ValueError("deep trajectory cohort is empty")
    rows = []
    for example_index, example in enumerate(examples):
        record = semantic[example.example_id]
        for side, candidate, prompt, answer in _candidate_variants(example):
            rows.append({
                "example_index": example_index,
                "side": side,
                "candidate": candidate,
                "tokens": tuple(prompt) + tuple(answer),
                "prompt_length": len(prompt),
                "positions": record.positions,
            })
    real_count = len(rows)
    batch_size = _round_up(real_count, max(1, data_multiple))
    required_sequence_length = max(len(row["tokens"]) for row in rows)
    required_trace_width = max(len(row["positions"]) for row in rows)
    sequence_length = int(
        fixed_sequence_length or required_sequence_length)
    trace_width = int(fixed_trace_width or required_trace_width)
    if sequence_length < required_sequence_length:
        raise ValueError("fixed deep trace sequence length is too small")
    if trace_width < required_trace_width:
        raise ValueError("fixed deep trace position width is too small")
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    labels = np.full(
        (batch_size, sequence_length), -100, dtype=np.int32)
    positions = np.zeros((batch_size, trace_width), dtype=np.int32)
    position_valid = np.zeros((batch_size, trace_width), dtype=np.bool_)
    example_index = np.zeros((batch_size,), dtype=np.int32)
    side_code = np.zeros((batch_size,), dtype=np.int8)
    candidate_code = np.zeros((batch_size,), dtype=np.int8)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        tokens = np.asarray(row["tokens"], dtype=np.int32)
        input_ids[row_index, :len(tokens)] = tokens
        prompt_length = int(row["prompt_length"])
        labels[row_index, prompt_length:len(tokens)] = tokens[prompt_length:]
        row_positions = np.asarray(row["positions"], dtype=np.int32)
        positions[row_index, :len(row_positions)] = row_positions
        position_valid[row_index, :len(row_positions)] = True
        example_index[row_index] = int(row["example_index"])
        side_code[row_index] = 0 if row["side"] == "base" else 1
        candidate_code[row_index] = (
            0 if row["candidate"] == "positive" else 1)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": positions,
        "position_valid": position_valid,
        "example_index": example_index,
        "side_code": side_code,
        "candidate_code": candidate_code,
        "real_count": real_count,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "trace_width": trace_width,
    }


def _device(ctx: Any, value: np.ndarray, spec: P):
    return jax.device_put(
        jnp.asarray(value), NamedSharding(ctx.mesh, spec))


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    words = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return words.view(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / den) if den > 0.0 else 0.0


def _mean_or_zero(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(array.mean()) if array.size else 0.0


def _artifact_record(path: str, **fields: Any) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_path(path), **fields}


def _trajectory_executable(
        ctx: Any, sharded_fns: Mapping[str, Any], *,
        widths: Mapping[str, int], compact_replay_output: bool = False):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = (
        "paired_operator_trajectory",
        int(widths["qk"]), int(widths["v"]), int(widths["rst"]),
        bool(compact_replay_output),
    )
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx)
    kwargs["sharded_fns"] = sharded_fns

    @jax.jit
    def execute(
            params, input_ids, labels, positions, position_valid,
            ids_q, ids_k, ids_v, ids_rst,
            valid_q, valid_k, valid_v, valid_rst,
            replay_enabled, patch_layers, patch_positions, patch_stages,
            patch_enabled, patch_values):
        result = ctx.model.apply(
            {"params": params}, input_ids,
            trajectory_positions=positions,
            trajectory_position_valid=position_valid,
            selected_ids_q=ids_q,
            selected_ids_k=ids_k,
            selected_ids_v=ids_v,
            selected_ids_rst=ids_rst,
            selected_valid_q=valid_q,
            selected_valid_k=valid_k,
            selected_valid_v=valid_v,
            selected_valid_rst=valid_rst,
            replay_full_active=replay_enabled,
            patch_layers=patch_layers,
            patch_positions=patch_positions,
            patch_stages=patch_stages,
            patch_enabled=patch_enabled,
            patch_values=patch_values,
            labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=True,
            return_logits=False,
            method=(
                ctx.model.analysis_forward_with_paired_operator_trajectory),
            **kwargs)
        trace = result["operator_trajectory_trace"]
        if compact_replay_output:
            trace = {
                "routes": {
                    route: {
                        field: trace["routes"][route][field]
                        for field in (
                            "production_output",
                            "selected_replay_output",
                            "production_precast_output",
                            "selected_replay_precast_output",
                        )
                    }
                    for route in ROUTES
                },
            }
        return (
            _score_from_result(result, "sum_log_probability"),
            result["final_residual"], trace)

    cache[key] = execute
    return execute


def _trajectory_device_inputs(
        ctx: Any, batch: Mapping[str, Any], *, n_layers: int,
        d_model: int, widths: Mapping[str, int], patch_slots: int,
        selected: Mapping[str, np.ndarray] | None = None,
        selected_valid: Mapping[str, np.ndarray] | None = None
) -> dict[str, Any]:
    B = int(batch["batch_size"])
    T = int(batch["trace_width"])
    if selected is None:
        selected = {
            route: np.zeros(
                (n_layers, B, T,
                 int(widths["qk"] if route in {"q", "k"} else widths[route])),
                dtype=np.int32)
            for route in ROUTES
        }
    if selected_valid is None:
        selected_valid = {
            route: np.zeros_like(selected[route], dtype=np.bool_)
            for route in ROUTES
        }
    patch_shape = (B, int(patch_slots))
    return {
        "input_ids": _device(ctx, batch["input_ids"], P("data", None)),
        "labels": _device(ctx, batch["labels"], P("data", None)),
        "positions": _device(ctx, batch["positions"], P("data", None)),
        "position_valid": _device(
            ctx, batch["position_valid"], P("data", None)),
        **{
            f"ids_{route}": _device(
                ctx, np.asarray(selected[route], dtype=np.int32),
                P(None, "data", None, None))
            for route in ROUTES
        },
        **{
            f"valid_{route}": _device(
                ctx, np.asarray(selected_valid[route], dtype=np.bool_),
                P(None, "data", None, None))
            for route in ROUTES
        },
        "patch_layers": _device(
            ctx, np.zeros(patch_shape, dtype=np.int32), P("data", None)),
        "patch_positions": _device(
            ctx, np.zeros(patch_shape, dtype=np.int32), P("data", None)),
        "patch_stages": _device(
            ctx, np.zeros(patch_shape, dtype=np.int32), P("data", None)),
        "patch_enabled": _device(
            ctx, np.zeros(patch_shape, dtype=np.bool_), P("data", None)),
        "patch_values": _device(
            ctx, np.zeros(patch_shape + (d_model,), dtype=np.float32),
            P("data", None, None)),
    }


def _call_trajectory(
        executable: Any, ctx: Any, inputs: Mapping[str, Any], *,
        replay: bool) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    result = executable(
        ctx.params, inputs["input_ids"], inputs["labels"],
        inputs["positions"], inputs["position_valid"],
        inputs["ids_q"], inputs["ids_k"],
        inputs["ids_v"], inputs["ids_rst"],
        inputs["valid_q"], inputs["valid_k"],
        inputs["valid_v"], inputs["valid_rst"],
        jnp.asarray(replay, dtype=jnp.bool_),
        inputs["patch_layers"], inputs["patch_positions"],
        inputs["patch_stages"], inputs["patch_enabled"],
        inputs["patch_values"])
    score, residual, trace = materialize_global_tree(result)
    return np.asarray(score), np.asarray(residual), trace


def _trace_width_overflow(
        trace: Mapping[str, Any], *, real_count: int,
        widths: Mapping[str, int]) -> dict[str, int]:
    overflow = {}
    for route in ROUTES:
        counts = np.asarray(
            trace["routes"][route]["numerator_active_count"])
        maximum = int(np.max(counts[:, :real_count]))
        width = int(widths["qk"] if route in {"q", "k"} else widths[route])
        if maximum > width:
            pool = "qk" if route in {"q", "k"} else route
            overflow[pool] = max(overflow.get(pool, 0), maximum)
    return overflow


def _expand_trajectory_widths(
        widths: Mapping[str, int], pool_sizes: Mapping[str, int],
        overflow: Mapping[str, int]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Grow only overflowing pools while preserving every captured ID."""
    updated_widths = {key: int(value) for key, value in widths.items()}
    changed = {}
    for pool, observed_value in overflow.items():
        if pool not in updated_widths or pool not in pool_sizes:
            raise ValueError(f"unknown trajectory overflow pool: {pool}")
        observed = int(observed_value)
        previous = int(updated_widths[pool])
        updated = min(int(pool_sizes[pool]), max(previous * 2, observed))
        if updated <= previous:
            raise RuntimeError(
                f"trajectory {pool} active count exceeds pool-size width")
        updated_widths[pool] = updated
        changed[pool] = {
            "previous_width": previous,
            "observed_active_count": observed,
            "updated_width": updated,
        }
    return updated_widths, changed


def _validate_complete_trace(
        trace: Mapping[str, Any], *, real_count: int) -> dict[str, Any]:
    route_summary = {}
    for route in ROUTES:
        values = trace["routes"][route]
        count = np.asarray(values["numerator_active_count"])[:, :real_count]
        valid = np.asarray(values["operator_valid"])[:, :real_count]
        ids = np.asarray(values["operator_id"])[:, :real_count]
        captured = valid.sum(axis=-1, dtype=np.int64)
        omitted = count.astype(np.int64) - captured
        if np.any(omitted != 0):
            raise RuntimeError(
                f"trajectory route={route} omitted active operators")
        if np.any(ids[valid] < 0):
            raise RuntimeError(
                f"trajectory route={route} retained a sentinel operator id")
        route_summary[route] = {
            "row_count": int(count.size),
            "numerator_active_count_min": int(count.min()),
            "numerator_active_count_max": int(count.max()),
            "numerator_active_count_mean": float(count.mean()),
            "denominator_active_count_mean": float(np.asarray(
                values["denominator_active_count"]
            )[:, :real_count].mean()),
            "captured_active_count": int(captured.sum()),
            "omitted_active_count": int(omitted.sum()),
            "sentinel_id_count": 0,
        }
    return route_summary


def _closure_metrics(
        capture_score: np.ndarray, capture_residual: np.ndarray,
        capture_trace: Mapping[str, Any], replay_score: np.ndarray,
        replay_residual: np.ndarray, replay_trace: Mapping[str, Any], *,
        real_count: int, atol: float, rtol: float) -> dict[str, Any]:
    score_error = float(np.max(np.abs(
        capture_score[:real_count] - replay_score[:real_count])))
    residual_error = float(np.max(np.abs(
        capture_residual[:real_count] - replay_residual[:real_count])))
    route_metrics = {}
    authoritative_pass = True
    for route in ROUTES:
        production = np.asarray(
            capture_trace["routes"][route]["production_output"]
        )[:, :real_count]
        replay_production = np.asarray(
            replay_trace["routes"][route]["production_output"]
        )[:, :real_count]
        selected_replay = np.asarray(
            replay_trace["routes"][route]["selected_replay_output"]
        )[:, :real_count]
        production_error = float(np.max(np.abs(
            production - replay_production)))
        selected_error = float(np.max(np.abs(
            replay_production - selected_replay)))
        pre_cast = np.asarray(
            replay_trace["routes"][route]["production_precast_output"]
        )[:, :real_count]
        selected_pre_cast = np.asarray(
            replay_trace["routes"][route][
                "selected_replay_precast_output"]
        )[:, :real_count]
        pre_cast_error = float(np.max(np.abs(
            pre_cast - selected_pre_cast)))
        route_pass = bool(
            np.allclose(production, replay_production, atol=atol, rtol=rtol)
            and np.allclose(
                replay_production, selected_replay, atol=atol, rtol=rtol))
        authoritative_pass &= route_pass
        route_metrics[route] = {
            "production_vs_replay_max_abs": production_error,
            "canonical_selected_replay_max_abs": selected_error,
            "pre_cast_descriptive_max_abs": pre_cast_error,
            "authoritative_parity_passed": route_pass,
        }
    final_pass = bool(
        np.allclose(
            capture_score[:real_count], replay_score[:real_count],
            atol=atol, rtol=rtol)
        and np.allclose(
            capture_residual[:real_count], replay_residual[:real_count],
            atol=atol, rtol=rtol))
    authoritative_pass &= final_pass
    return {
        "status": (
            "passed" if authoritative_pass
            else "full_active_replay_parity_failed"),
        "authoritative_parity_passed": bool(authoritative_pass),
        "candidate_log_probability_max_abs": score_error,
        "final_residual_max_abs": residual_error,
        "atol": float(atol),
        "rtol": float(rtol),
        "routes": route_metrics,
    }


def capture_full_active_trajectory(
        ctx: Any, examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *,
        pad_token_id: int, config: ProtocolConfig,
        initial_widths: Mapping[str, int] | None = None,
        fixed_sequence_length: int | None = None,
        fixed_trace_width: int | None = None,
        progress: Any | None = None) -> dict[str, Any]:
    batch = build_deep_trace_batch(
        examples, semantic, pad_token_id=pad_token_id,
        data_multiple=max(1, int(ctx.mesh.shape["data"])),
        fixed_sequence_length=fixed_sequence_length,
        fixed_trace_width=fixed_trace_width)
    pool_sizes = {
        "qk": int(ctx.model_cfg["n_qk"]),
        "v": int(ctx.model_cfg["n_v"]),
        "rst": int(
            ctx.model_cfg["n_rst"]
            if "n_rst" in ctx.model_cfg else ctx.model_cfg["n_know"]),
    }
    configured_widths = {
        "qk": min(config.trajectory_capture_topk_qk, pool_sizes["qk"]),
        "v": min(config.trajectory_capture_topk_v, pool_sizes["v"]),
        "rst": min(config.trajectory_capture_topk_rst, pool_sizes["rst"]),
    }
    if initial_widths is None:
        widths = dict(configured_widths)
    else:
        if set(initial_widths) != set(pool_sizes):
            raise ValueError(
                "trajectory initial widths must define qk, v, and rst")
        widths = {key: int(initial_widths[key]) for key in pool_sizes}
        invalid = [
            key for key, value in widths.items()
            if value <= 0 or value > pool_sizes[key]]
        if invalid:
            raise ValueError(
                "trajectory initial widths are outside their pools: "
                + ",".join(invalid))
    starting_widths = dict(widths)
    retries = []
    forward_call_count = 0
    started = time.monotonic()
    while True:
        if progress is not None:
            progress(
                "stage=full_active_trace phase=discovery "
                f"example_count={len(examples)} candidate_count=0 "
                "intervention_variants=0 compile=reuse_or_compile "
                f"elapsed_s={time.monotonic() - started:.1f} "
                f"capture_width={widths} retry_count={len(retries)}")
        sharded_fns = create_or_reuse_sharded_fns(
            ctx.config, ctx.mesh, analysis=False,
            trajectory_widths=widths)
        executable = _trajectory_executable(
            ctx, sharded_fns, widths=widths,
            compact_replay_output=False)
        inputs = _trajectory_device_inputs(
            ctx, batch, n_layers=int(ctx.model_cfg["n_layers"]),
            d_model=int(ctx.model_cfg["d_model"]), widths=widths,
            patch_slots=config.trajectory_max_patch_sites_per_variant)
        capture_score, capture_residual, capture_trace = _call_trajectory(
            executable, ctx, inputs, replay=False)
        forward_call_count += 1
        overflow = _trace_width_overflow(
            capture_trace, real_count=int(batch["real_count"]),
            widths=widths)
        if not overflow:
            break
        widths, changed = _expand_trajectory_widths(
            widths, pool_sizes, overflow)
        retries.append({
            "retry_index": len(retries) + 1,
            "affected_routes": changed,
            "retry_scope": (
                "single_fused_deep_trace_batch_with_only_affected_widths_"
                "changed"),
            "production_atlas_repeated": False,
            "unaffected_route_widths_changed": False,
        })
        del inputs, executable, sharded_fns
        del capture_score, capture_residual, capture_trace
        gc.collect()
    completeness = _validate_complete_trace(
        capture_trace, real_count=int(batch["real_count"]))
    selected = {
        route: np.asarray(
            capture_trace["routes"][route]["operator_id"],
            dtype=np.int32)
        for route in ROUTES
    }
    selected_valid = {
        route: np.asarray(
            capture_trace["routes"][route]["operator_valid"],
            dtype=np.bool_)
        for route in ROUTES
    }
    replay_inputs = _trajectory_device_inputs(
        ctx, batch, n_layers=int(ctx.model_cfg["n_layers"]),
        d_model=int(ctx.model_cfg["d_model"]), widths=widths,
        patch_slots=config.trajectory_max_patch_sites_per_variant,
        selected=selected, selected_valid=selected_valid)
    replay_executable = _trajectory_executable(
        ctx, sharded_fns, widths=widths,
        compact_replay_output=True)
    replay_score, replay_residual, replay_trace = _call_trajectory(
        replay_executable, ctx, replay_inputs, replay=True)
    forward_call_count += 1
    closure = _closure_metrics(
        capture_score, capture_residual, capture_trace,
        replay_score, replay_residual, replay_trace,
        real_count=int(batch["real_count"]),
        atol=config.trajectory_replay_atol,
        rtol=config.trajectory_replay_rtol)
    trace_output_bytes = int(sum(
        np.asarray(leaf).nbytes
        for leaf in jax.tree_util.tree_leaves(capture_trace)))
    replay_trace_output_bytes = int(sum(
        np.asarray(leaf).nbytes
        for leaf in jax.tree_util.tree_leaves(replay_trace)))
    capture_score_residual_bytes = int(
        np.asarray(capture_score).nbytes
        + np.asarray(capture_residual).nbytes)
    replay_score_residual_bytes = int(
        np.asarray(replay_score).nbytes
        + np.asarray(replay_residual).nbytes)
    peak_materialized_output_bytes = int(
        trace_output_bytes + replay_trace_output_bytes
        + capture_score_residual_bytes + replay_score_residual_bytes)
    if not closure["authoritative_parity_passed"]:
        return {
            "status": "full_active_replay_parity_failed",
            "batch": batch,
            "initial_widths": starting_widths,
            "widths": widths,
            "retries": retries,
            "completeness": completeness,
            "closure": closure,
            "trace_output_bytes": trace_output_bytes,
            "replay_trace_output_bytes": replay_trace_output_bytes,
            "capture_score_residual_bytes": capture_score_residual_bytes,
            "replay_score_residual_bytes": replay_score_residual_bytes,
            "peak_materialized_output_bytes": (
                peak_materialized_output_bytes),
            "forward_call_count": forward_call_count,
            "capture_score": capture_score,
            "capture_trace": capture_trace,
        }
    return {
        "status": "ready",
        "batch": batch,
        "initial_widths": starting_widths,
        "widths": widths,
        "retries": retries,
        "completeness": completeness,
        "closure": closure,
        "trace_output_bytes": trace_output_bytes,
        "replay_trace_output_bytes": replay_trace_output_bytes,
        "capture_score_residual_bytes": capture_score_residual_bytes,
        "replay_score_residual_bytes": replay_score_residual_bytes,
        "peak_materialized_output_bytes": peak_materialized_output_bytes,
        "forward_call_count": forward_call_count,
        "capture_score": capture_score,
        "capture_residual": capture_residual,
        "capture_trace": capture_trace,
    }


def merge_trajectory_batch_summaries(
        batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("trajectory summary merge has no batches")
    completeness = {}
    closure_routes = {}
    for route in ROUTES:
        rows = [batch["completeness"][route] for batch in batches]
        total_rows = sum(int(row["row_count"]) for row in rows)
        completeness[route] = {
            "row_count": total_rows,
            "numerator_active_count_min": min(
                int(row["numerator_active_count_min"]) for row in rows),
            "numerator_active_count_max": max(
                int(row["numerator_active_count_max"]) for row in rows),
            "numerator_active_count_mean": float(sum(
                float(row["numerator_active_count_mean"])
                * int(row["row_count"]) for row in rows) / total_rows),
            "denominator_active_count_mean": float(sum(
                float(row["denominator_active_count_mean"])
                * int(row["row_count"]) for row in rows) / total_rows),
            "captured_active_count": sum(
                int(row["captured_active_count"]) for row in rows),
            "omitted_active_count": sum(
                int(row["omitted_active_count"]) for row in rows),
            "sentinel_id_count": sum(
                int(row["sentinel_id_count"]) for row in rows),
        }
        route_closures = [batch["closure"]["routes"][route]
                          for batch in batches]
        closure_routes[route] = {
            "production_vs_replay_max_abs": max(float(
                row["production_vs_replay_max_abs"])
                for row in route_closures),
            "canonical_selected_replay_max_abs": max(float(
                row["canonical_selected_replay_max_abs"])
                for row in route_closures),
            "pre_cast_descriptive_max_abs": max(float(
                row["pre_cast_descriptive_max_abs"])
                for row in route_closures),
            "authoritative_parity_passed": all(bool(
                row["authoritative_parity_passed"])
                for row in route_closures),
        }
    retries = []
    for batch in batches:
        for retry in batch["retries"]:
            retries.append({
                **dict(retry),
                "example_id": str(batch["example_id"]),
                "global_retry_index": len(retries) + 1,
            })
    authoritative = all(bool(
        batch["closure"]["authoritative_parity_passed"])
        for batch in batches)
    closure = {
        "status": (
            "passed" if authoritative
            else "full_active_replay_parity_failed"),
        "authoritative_parity_passed": authoritative,
        "candidate_log_probability_max_abs": max(float(
            batch["closure"]["candidate_log_probability_max_abs"])
            for batch in batches),
        "final_residual_max_abs": max(float(
            batch["closure"]["final_residual_max_abs"])
            for batch in batches),
        "atol": float(batches[0]["closure"]["atol"]),
        "rtol": float(batches[0]["closure"]["rtol"]),
        "routes": closure_routes,
    }
    width_inheritance_monotonic = all(
        all(
            int(batches[index].get(
                "initial_widths", batches[index]["widths"])[pool])
            >= int(batches[index - 1]["widths"][pool])
            for pool in ("qk", "v", "rst"))
        for index in range(1, len(batches)))
    return {
        "status": "ready" if authoritative else (
            "full_active_replay_parity_failed"),
        "batch_count": len(batches),
        "widths": {
            pool: max(int(batch["widths"][pool]) for batch in batches)
            for pool in ("qk", "v", "rst")
        },
        "initial_widths_by_batch": [
            {
                "example_id": str(batch["example_id"]),
                "widths": {
                    pool: int(batch.get(
                        "initial_widths", batch["widths"])[pool])
                    for pool in ("qk", "v", "rst")
                },
            }
            for batch in batches
        ],
        "widths_carried_forward_between_deep_examples": True,
        "width_inheritance_monotonic": width_inheritance_monotonic,
        "retries": retries,
        "completeness": completeness,
        "closure": closure,
        "trace_output_bytes": max(
            int(batch["trace_output_bytes"]) for batch in batches),
        "replay_trace_output_bytes": max(
            int(batch["replay_trace_output_bytes"])
            for batch in batches),
        "capture_score_residual_bytes": max(
            int(batch["capture_score_residual_bytes"])
            for batch in batches),
        "replay_score_residual_bytes": max(
            int(batch["replay_score_residual_bytes"])
            for batch in batches),
        "peak_materialized_output_bytes": max(
            int(batch["peak_materialized_output_bytes"])
            for batch in batches),
        "forward_call_count": sum(
            int(batch["forward_call_count"]) for batch in batches),
        "total_streamed_trace_output_bytes": sum(
            int(batch["trace_output_bytes"]) for batch in batches),
    }


def _row_field(
        route_values: Mapping[str, Any], field: str, *,
        layer: int, batch_index: int, position_slot: int) -> np.ndarray:
    value = np.asarray(route_values[field])
    if field == "scale":
        return np.asarray(value[layer])
    return np.asarray(value[layer, batch_index, position_slot])


def write_deep_trace_shards(
        store: Any, trajectory: Mapping[str, Any],
        examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *,
        protocol_hash: str, shard_index_offset: int = 0,
        write_intermediate_manifest: bool = True) -> dict[str, Any]:
    """Stream one example-side CSR-like shard at a time."""
    if not store.is_primary:
        return {
            "status": "primary_host_writes_only",
            "shard_count": 2 * len(examples),
        }
    trace = trajectory["capture_trace"]
    batch = trajectory["batch"]
    shards = []
    n_layers = int(np.asarray(
        trace["states"]["residual_input"]).shape[0])
    for example_index, example in enumerate(examples):
        shard_index = int(shard_index_offset) + example_index
        record = semantic[example.example_id]
        for side, variant_offset in (("base", 0), ("source", 2)):
            batch_index = example_index * 4 + variant_offset
            row_ptr = [0]
            row_layer = []
            row_position = []
            row_position_offset = []
            row_route = []
            row_example_index = []
            numerator_count = []
            denominator_count = []
            captured_count = []
            omitted_count = []
            denominator = []
            admission_mass = []
            pool_scale = []
            query = []
            tau = []
            production_output = []
            production_precast_output = []
            operator_id = []
            read_bits = []
            amplitude_bits = []
            execution_weight = []
            admission = []
            margin = []
            rho = []
            for layer in range(n_layers):
                for position_slot, position in enumerate(record.positions):
                    for route in ROUTES:
                        route_values = trace["routes"][route]
                        valid = np.asarray(_row_field(
                            route_values, "operator_valid", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot), dtype=np.bool_)
                        ids = np.asarray(_row_field(
                            route_values, "operator_id", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot), dtype=np.int32)[valid]
                        if ids.size and np.any(ids < 0):
                            raise RuntimeError(
                                "sentinel ID survived trajectory CSR compaction")
                        num = int(_row_field(
                            route_values, "numerator_active_count",
                            layer=layer, batch_index=batch_index,
                            position_slot=position_slot))
                        if ids.size != num:
                            raise RuntimeError(
                                "trajectory CSR row is not full-active")
                        row_layer.append(layer)
                        row_position.append(position)
                        row_position_offset.append(position - record.s2_start)
                        row_route.append(ROUTE_INDEX[route])
                        row_example_index.append(example_index)
                        numerator_count.append(num)
                        denominator_count.append(int(_row_field(
                            route_values, "denominator_active_count",
                            layer=layer, batch_index=batch_index,
                            position_slot=position_slot)))
                        captured_count.append(int(ids.size))
                        omitted_count.append(0)
                        denominator.append(np.asarray(_row_field(
                            route_values, "denominator", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot)).reshape(-1)[0])
                        admission_mass.append(np.asarray(_row_field(
                            route_values, "admission_mass", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot)).reshape(-1)[0])
                        pool_scale.append(float(_row_field(
                            route_values, "scale", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot)))
                        query.append(_row_field(
                            route_values, "query", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot))
                        tau.append(np.asarray(_row_field(
                            route_values, "tau", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot)).reshape(-1)[0])
                        production_output.append(_row_field(
                            route_values, "production_output", layer=layer,
                            batch_index=batch_index,
                            position_slot=position_slot))
                        production_precast_output.append(_row_field(
                            route_values, "production_precast_output",
                            layer=layer, batch_index=batch_index,
                            position_slot=position_slot))
                        operator_id.extend(ids.tolist())
                        for field, target in (
                                ("read_scalar_bf16_bits", read_bits),
                                ("prewrite_amplitude_bf16_bits",
                                 amplitude_bits),
                                ("execution_weight", execution_weight),
                                ("admission", admission),
                                ("margin", margin),
                                ("rho", rho)):
                            values = np.asarray(_row_field(
                                route_values, field, layer=layer,
                                batch_index=batch_index,
                                position_slot=position_slot))[valid]
                            target.extend(values.tolist())
                        row_ptr.append(len(operator_id))
            states = np.stack([
                np.asarray(trace["states"][stage])[
                    :, batch_index, :len(record.positions), :]
                for stage in STATE_STAGES
            ], axis=0)
            path = store.path(
                "trajectory", "deep_trace",
                f"shard_{shard_index:03d}_{side}.npz")
            write_npz_atomic(
                path,
                protocol_hash=np.asarray(protocol_hash),
                algorithm_version=np.asarray(
                    PAIRED_TRAJECTORY_ALGORITHM_VERSION),
                example_id=np.asarray(example.example_id),
                side=np.asarray(side),
                semantic_positions=np.asarray(
                    record.positions, dtype=np.int32),
                semantic_roles=np.asarray(record.position_roles),
                row_ptr=np.asarray(row_ptr, dtype=np.int64),
                row_layer=np.asarray(row_layer, dtype=np.int16),
                row_position=np.asarray(row_position, dtype=np.int32),
                row_position_offset=np.asarray(
                    row_position_offset, dtype=np.int16),
                row_route=np.asarray(row_route, dtype=np.int8),
                row_example_index=np.asarray(
                    [shard_index] * len(row_example_index), dtype=np.int16),
                numerator_active_count=np.asarray(
                    numerator_count, dtype=np.int32),
                denominator_active_count=np.asarray(
                    denominator_count, dtype=np.int32),
                captured_active_count=np.asarray(
                    captured_count, dtype=np.int32),
                omitted_active_count=np.asarray(
                    omitted_count, dtype=np.int32),
                denominator=np.asarray(denominator, dtype=np.float32),
                admission_mass=np.asarray(
                    admission_mass, dtype=np.float32),
                pool_scale=np.asarray(pool_scale, dtype=np.float32),
                query=np.asarray(query, dtype=np.float16),
                tau=np.asarray(tau, dtype=np.float32),
                production_output=np.asarray(
                    production_output, dtype=np.float16),
                production_precast_output=np.asarray(
                    production_precast_output, dtype=np.float16),
                operator_id=np.asarray(operator_id, dtype=np.int32),
                read_scalar_bf16_bits=np.asarray(
                    read_bits, dtype=np.uint16),
                prewrite_amplitude_bf16_bits=np.asarray(
                    amplitude_bits, dtype=np.uint16),
                execution_weight=np.asarray(
                    execution_weight, dtype=np.float32),
                admission=np.asarray(admission, dtype=np.float16),
                margin=np.asarray(margin, dtype=np.float16),
                rho=np.asarray(rho, dtype=np.float16),
                state_stage=np.asarray(STATE_STAGES),
                state_snapshot=np.asarray(states, dtype=np.float16),
            )
            shards.append(_artifact_record(
                path, example_id=example.example_id, side=side,
                row_count=len(row_layer), operator_count=len(operator_id),
                sentinel_id_count=0, omitted_active_count=0))
            del states
            gc.collect()
    manifest = {
        "status": "complete",
        "algorithm_version": PAIRED_TRAJECTORY_ALGORITHM_VERSION,
        "protocol_hash": protocol_hash,
        "format": "csr_like_npz_per_example_side",
        "shards": shards,
        "dtype_schema": {
            "operator_id": "int32",
            "row_ptr": "int64",
            "read_scalar_bf16_bits": "uint16_exact_bfloat16_bits",
            "prewrite_amplitude_bf16_bits": (
                "uint16_exact_bfloat16_bits"),
            "execution_weight": "float32",
            "admission_margin_rho": "float16_descriptive",
            "query": "float16_persisted_float32_metrics",
            "state_snapshot": "float16_persisted_float32_metrics",
            "aggregate_metrics": "float64_host",
        },
        "padding_policy": "removed_before_persistence",
        "static_operator_vectors_repeated": False,
        "authoritative_replay_uses_serialized_quantized_values": False,
    }
    result = {
        "status": "complete",
        "shards": shards,
        "shard_count": len(shards),
    }
    if write_intermediate_manifest:
        manifest_path = store.path("trajectory", "manifest.json")
        write_json_atomic(manifest_path, manifest)
        result["manifest"] = _artifact_record(
            manifest_path, shard_count=len(shards))
    return result


def _hash_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(name).encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def operator_parameter_provenance(ctx: Any) -> tuple[
        dict[str, Any], dict[str, np.ndarray]]:
    module = analysis_model_module(ctx.model_cfg)
    pool = ctx.params["neuron_pool"]
    parameter_schema = {
        str(name): {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
        }
        for name, array in pool.items()
    }
    pool_with_keys = module._pool_params_with_operator_keys(
        pool, str(ctx.model_cfg["operator_key_mode"]))
    key_names = {
        "qk": "attn_qk_op_key",
        "v": "attn_v_op_key",
        "rst": "rst_op_key",
    }
    read_write_names = (
        "attn_qk_read", "attn_qk_write", "attn_v_read", "attn_v_write",
        "rst_read", "rst_write",
    )
    keys = {}
    key_digest = hashlib.sha256()
    for route, name in key_names.items():
        value = materialize_global_array(pool_with_keys[name])
        keys[route] = np.asarray(value, dtype=np.float32)
        _hash_array(key_digest, route, value)
    read_write_digest = hashlib.sha256()
    for name in read_write_names:
        value = materialize_global_array(pool[name])
        _hash_array(read_write_digest, name, value)
        del value
    return ({
        "operator_key_mode": str(ctx.model_cfg["operator_key_mode"]),
        "parameter_schema_hash": canonical_hash(parameter_schema),
        "parameter_schema_entry_count": len(parameter_schema),
        "operator_key_digest": key_digest.hexdigest(),
        "read_write_parameter_digest": read_write_digest.hexdigest(),
        "static_vectors_reconstructable_from_checkpoint": True,
        "static_vectors_repeated_per_occurrence": False,
        "host_materialization_scope": (
            "one_checkpoint_parameter_table_per_pool_not_per_occurrence"),
    }, keys)


def _operator_row(
        route_values: Mapping[str, Any], *, layer: int, batch_index: int,
        position_slot: int) -> dict[str, Any]:
    valid = np.asarray(
        route_values["operator_valid"][
            layer, batch_index, position_slot], dtype=np.bool_)
    ids = np.asarray(
        route_values["operator_id"][
            layer, batch_index, position_slot], dtype=np.int32)[valid]
    amplitude = _bf16_bits_to_float32(np.asarray(
        route_values["prewrite_amplitude_bf16_bits"][
            layer, batch_index, position_slot], dtype=np.uint16)[valid])
    read_scalar = _bf16_bits_to_float32(np.asarray(
        route_values["read_scalar_bf16_bits"][
            layer, batch_index, position_slot], dtype=np.uint16)[valid])
    return {
        "ids": ids,
        "amplitude": amplitude,
        "read_scalar": read_scalar,
        "execution_weight": np.asarray(
            route_values["execution_weight"][
                layer, batch_index, position_slot], dtype=np.float32)[valid],
        "query": np.asarray(
            route_values["query"][
                layer, batch_index, position_slot], dtype=np.float32),
        "tau": float(np.asarray(
            route_values["tau"][
                layer, batch_index, position_slot]).reshape(-1)[0]),
        "denominator": float(np.asarray(
            route_values["denominator"][
                layer, batch_index, position_slot]).reshape(-1)[0]),
        "scale": float(np.asarray(route_values["scale"])[layer]),
        "output": np.asarray(
            route_values["production_output"][
                layer, batch_index, position_slot], dtype=np.float32),
    }


def _weighted_centroid(
        keys: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if ids.size == 0:
        return np.zeros((keys.shape[1],), dtype=np.float64)
    weight = np.abs(np.asarray(weights, dtype=np.float64))
    total = float(weight.sum())
    if total == 0.0:
        weight = np.ones_like(weight)
        total = float(weight.size)
    return np.sum(
        np.asarray(keys[ids], dtype=np.float64) * weight[:, None],
        axis=0) / total


def build_divergence_atlas(
        trajectory: Mapping[str, Any],
        examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *,
        operator_keys: Mapping[str, np.ndarray], epsilon: float
) -> dict[str, Any]:
    trace = trajectory["capture_trace"]
    site_values: dict[tuple[int, str, str], list[dict[str, Any]]] = (
        defaultdict(list))
    state_values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    per_example_sites: dict[str, dict[str, Any]] = {}
    n_layers = int(np.asarray(
        trace["states"]["residual_input"]).shape[0])
    for example_index, example in enumerate(examples):
        record = semantic[example.example_id]
        base_index = example_index * 4
        source_index = base_index + 2
        example_sites = {}
        for layer in range(n_layers):
            for position_slot, (position, role) in enumerate(zip(
                    record.positions, record.position_roles)):
                stage_delta = {}
                for stage in STATE_STAGES:
                    base_state = np.asarray(trace["states"][stage])[
                        layer, base_index, position_slot].astype(np.float32)
                    source_state = np.asarray(trace["states"][stage])[
                        layer, source_index, position_slot].astype(np.float32)
                    delta_norm = float(np.linalg.norm(base_state - source_state))
                    stage_delta[stage] = delta_norm
                    state_values[(layer, role, stage)].append(delta_norm)
                residual_growth = (
                    stage_delta["post_rst"] - stage_delta["residual_input"])
                for route in ROUTES:
                    values = trace["routes"][route]
                    base = _operator_row(
                        values, layer=layer, batch_index=base_index,
                        position_slot=position_slot)
                    source = _operator_row(
                        values, layer=layer, batch_index=source_index,
                        position_slot=position_slot)
                    base_map = {
                        int(operator_id): index for index, operator_id
                        in enumerate(base["ids"])}
                    source_map = {
                        int(operator_id): index for index, operator_id
                        in enumerate(source["ids"])}
                    base_ids = set(base_map)
                    source_ids = set(source_map)
                    common = sorted(base_ids & source_ids)
                    base_only = sorted(base_ids - source_ids)
                    source_only = sorted(source_ids - base_ids)
                    union = base_ids | source_ids
                    common_amp_delta = _mean_or_zero(
                        abs(float(base["amplitude"][base_map[operator_id]])
                            - float(source["amplitude"][
                                source_map[operator_id]]))
                        for operator_id in common)
                    common_read_delta = _mean_or_zero(
                        abs(float(base["read_scalar"][base_map[operator_id]])
                            - float(source["read_scalar"][
                                source_map[operator_id]]))
                        for operator_id in common)
                    common_execution_delta = _mean_or_zero(
                        abs(float(base["execution_weight"][
                            base_map[operator_id]])
                            - float(source["execution_weight"][
                                source_map[operator_id]]))
                        for operator_id in common)
                    base_only_support = float(sum(
                        abs(float(base["amplitude"][base_map[value]]))
                        for value in base_only))
                    source_only_support = float(sum(
                        abs(float(source["amplitude"][source_map[value]]))
                        for value in source_only))
                    key_pool = "qk" if route in {"q", "k"} else route
                    base_centroid = _weighted_centroid(
                        operator_keys[key_pool], base["ids"],
                        base["amplitude"])
                    source_centroid = _weighted_centroid(
                        operator_keys[key_pool], source["ids"],
                        source["amplitude"])
                    base_output = base["output"] * base["scale"]
                    source_output = source["output"] * source["scale"]
                    query_cosine = _cosine(base["query"], source["query"])
                    row = {
                        "example_id": example.example_id,
                        "layer": layer,
                        "position": int(position),
                        "position_offset_from_s2": (
                            int(position - record.s2_start)),
                        "semantic_role": role,
                        "route": route,
                        "base_active_count": len(base_ids),
                        "source_active_count": len(source_ids),
                        "common_count": len(common),
                        "base_only_count": len(base_only),
                        "source_only_count": len(source_only),
                        "jaccard_overlap": (
                            len(common) / len(union) if union else 1.0),
                        "common_fraction": (
                            len(common) / max(1, min(
                                len(base_ids), len(source_ids)))),
                        "support_turnover": (
                            (len(base_only) + len(source_only))
                            / max(1, len(union))),
                        "query_angular_displacement": float(
                            math.acos(float(np.clip(
                                query_cosine, -1.0, 1.0)))),
                        "tau_difference": abs(base["tau"] - source["tau"]),
                        "weighted_operator_key_centroid_displacement": float(
                            np.linalg.norm(base_centroid - source_centroid)),
                        "route_output_difference_norm": float(
                            np.linalg.norm(base_output - source_output)),
                        "route_output_cosine": _cosine(
                            base_output, source_output),
                        "denominator_difference": abs(
                            base["denominator"] - source["denominator"]),
                        "common_read_scalar_delta_mean": common_read_delta,
                        "common_execution_weight_delta_mean": (
                            common_execution_delta),
                        "common_executed_amplitude_delta_mean": (
                            common_amp_delta),
                        "base_only_support_amplitude": base_only_support,
                        "source_only_support_amplitude": source_only_support,
                        "block_residual_divergence_growth_context": (
                            residual_growth),
                        "nontrivial_divergence": bool(
                            np.linalg.norm(base_output - source_output)
                            > epsilon
                            or base_ids != source_ids
                            or query_cosine < 1.0 - epsilon),
                        "common_ids": np.asarray(common, dtype=np.int32),
                        "base_only_ids": np.asarray(
                            base_only, dtype=np.int32),
                        "source_only_ids": np.asarray(
                            source_only, dtype=np.int32),
                    }
                    site_values[(layer, role, route)].append(row)
                    example_sites[f"route:{layer}:{role}:{route}"] = {
                        "common_count": len(common),
                        "base_only_count": len(base_only),
                        "source_only_count": len(source_only),
                        "common_fraction": row["common_fraction"],
                        "common_ids": row["common_ids"],
                        "base_only_ids": row["base_only_ids"],
                        "source_only_ids": row["source_only_ids"],
                    }
        per_example_sites[example.example_id] = example_sites
    aggregate_rows = []
    scalar_fields = (
        "base_active_count", "source_active_count", "common_count",
        "base_only_count", "source_only_count", "jaccard_overlap",
        "common_fraction", "support_turnover",
        "query_angular_displacement", "tau_difference",
        "weighted_operator_key_centroid_displacement",
        "route_output_difference_norm", "route_output_cosine",
        "denominator_difference", "common_read_scalar_delta_mean",
        "common_execution_weight_delta_mean",
        "common_executed_amplitude_delta_mean",
        "base_only_support_amplitude", "source_only_support_amplitude",
        "block_residual_divergence_growth_context",
    )
    for (layer, role, route), rows in site_values.items():
        aggregate = {
            "layer": int(layer),
            "semantic_role": role,
            "route": route,
            "example_count": len(rows),
            "position_offset_from_s2_mean": float(np.mean([
                row["position_offset_from_s2"] for row in rows])),
            "nontrivial_fraction": float(np.mean([
                row["nontrivial_divergence"] for row in rows])),
        }
        for field in scalar_fields:
            values = np.asarray([row[field] for row in rows], dtype=np.float64)
            aggregate[f"{field}_mean"] = float(values.mean())
            aggregate[f"{field}_max"] = float(values.max())
        aggregate_rows.append(aggregate)
    aggregate_rows.sort(key=lambda row: (
        int(row["layer"]), _semantic_role_order(row["semantic_role"]),
        ROUTE_INDEX[row["route"]]))
    state_rows = []
    for (layer, role, stage), values in state_values.items():
        array = np.asarray(values, dtype=np.float64)
        state_rows.append({
            "layer": int(layer), "semantic_role": role, "stage": stage,
            "example_count": int(array.size),
            "delta_norm_mean": float(array.mean()),
            "delta_norm_max": float(array.max()),
        })
    state_rows.sort(key=lambda row: (
        row["layer"], _semantic_role_order(row["semantic_role"]),
        STATE_STAGES.index(row["stage"])))
    return {
        "status": "ready",
        "site_count": len(aggregate_rows),
        "state_site_count": len(state_rows),
        "site_rows": aggregate_rows,
        "state_rows": state_rows,
        "decomposition_interpretation": "descriptive_accounting_not_exact_additivity",
        "streamed_example_count": len(examples),
        "_per_example_sites": per_example_sites,
    }


def merge_divergence_atlases(
        atlases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge streamed one-example divergence accumulators exactly by count."""
    if not atlases:
        raise ValueError("divergence atlas merge has no inputs")
    site_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list))
    state_groups: dict[tuple[int, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list))
    per_example = {}
    for atlas in atlases:
        for row in atlas["site_rows"]:
            site_groups[(
                int(row["layer"]), str(row["semantic_role"]),
                str(row["route"]))].append(row)
        for row in atlas["state_rows"]:
            state_groups[(
                int(row["layer"]), str(row["semantic_role"]),
                str(row["stage"]))].append(row)
        per_example.update(dict(atlas.get("_per_example_sites") or {}))
    site_rows = []
    for (layer, role, route), rows in site_groups.items():
        counts = np.asarray([
            int(row["example_count"]) for row in rows], dtype=np.float64)
        total = float(counts.sum())
        merged = {
            "layer": layer,
            "semantic_role": role,
            "route": route,
            "example_count": int(total),
            "position_offset_from_s2_mean": float(sum(
                float(row["position_offset_from_s2_mean"]) * count
                for row, count in zip(rows, counts)) / total),
            "nontrivial_fraction": float(sum(
                float(row["nontrivial_fraction"]) * count
                for row, count in zip(rows, counts)) / total),
        }
        numeric_keys = sorted({
            key for row in rows for key, value in row.items()
            if isinstance(value, (int, float))
            and key.endswith(("_mean", "_max"))
            and key not in merged
        })
        for key in numeric_keys:
            if key.endswith("_max"):
                merged[key] = float(max(float(row[key]) for row in rows))
            else:
                merged[key] = float(sum(
                    float(row[key]) * count
                    for row, count in zip(rows, counts)) / total)
        site_rows.append(merged)
    site_rows.sort(key=lambda row: (
        int(row["layer"]), _semantic_role_order(row["semantic_role"]),
        ROUTE_INDEX[row["route"]]))
    state_rows = []
    for (layer, role, stage), rows in state_groups.items():
        counts = np.asarray([
            int(row.get("example_count", 1)) for row in rows],
            dtype=np.float64)
        total = float(counts.sum())
        state_rows.append({
            "layer": layer,
            "semantic_role": role,
            "stage": stage,
            "example_count": int(total),
            "delta_norm_mean": float(sum(
                float(row["delta_norm_mean"]) * count
                for row, count in zip(rows, counts)) / total),
            "delta_norm_max": float(max(
                row["delta_norm_max"] for row in rows)),
        })
    state_rows.sort(key=lambda row: (
        row["layer"], _semantic_role_order(row["semantic_role"]),
        STATE_STAGES.index(row["stage"])))
    return {
        "status": "ready",
        "site_count": len(site_rows),
        "state_site_count": len(state_rows),
        "site_rows": site_rows,
        "state_rows": state_rows,
        "decomposition_interpretation": (
            "descriptive_accounting_not_exact_additivity"),
        "streamed_example_count": sum(int(
            atlas.get(
                "streamed_example_count",
                len(atlas.get("_per_example_sites") or {})))
            for atlas in atlases),
        "_per_example_sites": per_example,
    }


def _semantic_role_order(role: str) -> int:
    if role == "s2":
        return 0
    if role.startswith("s2_subtoken_"):
        try:
            return int(role.rsplit("_", 1)[1])
        except ValueError:
            return 999
    if role == "post_s2":
        return 10_000
    if role.startswith("post_s2_offset_"):
        try:
            return 10_000 + int(role.rsplit("_", 1)[1])
        except ValueError:
            return 50_000
    if role == "answer_position":
        return 100_000
    return 50_000


def select_discovery_candidates(
        atlas: Mapping[str, Any], *, config: ProtocolConfig) -> dict[str, Any]:
    rows = [dict(row) for row in atlas["site_rows"]]
    candidates = []
    selection_rules = {}
    for route in ROUTES:
        route_rows = [row for row in rows if row["route"] == route]
        if not route_rows:
            continue
        chronological = sorted(route_rows, key=lambda row: (
            int(row["layer"]), _semantic_role_order(row["semantic_role"]),
            -float(row["route_output_difference_norm_mean"])))
        nontrivial = [
            row for row in chronological
            if float(row["nontrivial_fraction"]) > 0.0]
        ranked_route = sorted(route_rows, key=lambda row: (
            -float(row["route_output_difference_norm_mean"]),
            int(row["layer"]), _semantic_role_order(row["semantic_role"])))
        ranked_query = sorted(route_rows, key=lambda row: (
            -float(row["query_angular_displacement_mean"]),
            int(row["layer"]), _semantic_role_order(row["semantic_role"])))
        ranked_turnover = sorted(route_rows, key=lambda row: (
            -float(row["support_turnover_mean"]),
            int(row["layer"]), _semantic_role_order(row["semantic_role"])))
        s2_rows = [row for row in ranked_route if row["semantic_role"] == "s2"]
        answer_rows = [
            row for row in ranked_route
            if row["semantic_role"] == "answer_position"]
        proposed = []

        def propose(rule: str, values: Sequence[Mapping[str, Any]], count=1):
            for row in values[:count]:
                proposed.append((rule, dict(row)))

        propose("first_nontrivial_divergence", nontrivial or chronological)
        propose("top_route_output_difference", ranked_route, 2)
        propose("top_query_displacement", ranked_query)
        propose("top_support_turnover", ranked_turnover)
        propose("mandatory_s2", s2_rows or ranked_route)
        propose("mandatory_answer_position", answer_rows or ranked_route)
        selected = []
        seen = set()
        for rule, row in proposed + [
                ("route_difference_fill", row) for row in ranked_route]:
            key = (
                int(row["layer"]), str(row["semantic_role"]), route)
            if key in seen:
                continue
            seen.add(key)
            selected.append({
                "layer": key[0],
                "semantic_role": key[1],
                "route": route,
                "native_stage": route,
                "selection_rule": rule,
                "position_offset_from_s2_mean": float(
                    row["position_offset_from_s2_mean"]),
                "discovery_metrics": {
                    "route_output_difference_norm_mean": float(
                        row["route_output_difference_norm_mean"]),
                    "query_angular_displacement_mean": float(
                        row["query_angular_displacement_mean"]),
                    "support_turnover_mean": float(
                        row["support_turnover_mean"]),
                    "block_residual_divergence_growth_context_mean": float(
                        row[
                            "block_residual_divergence_growth_context_mean"]),
                    "nontrivial_fraction": float(
                        row["nontrivial_fraction"]),
                },
            })
            if len(selected) >= config.trajectory_max_candidates_per_route:
                break
        selection_rules[route] = [
            row["selection_rule"] for row in selected]
        candidates.extend(selected)
    candidates.sort(key=lambda row: (
        int(row["layer"]),
        0 if row["route"] in {"q", "k", "v"} else 1,
        _semantic_role_order(row["semantic_role"]),
        ROUTE_INDEX[row["route"]],
    ))
    candidates = candidates[:config.trajectory_max_candidate_sites]
    for candidate_index, candidate in enumerate(candidates):
        candidate["candidate_index"] = int(candidate_index)
    freeze_record = {
        "algorithm_version": PAIRED_TRAJECTORY_ALGORITHM_VERSION,
        "selection_phase": "discovery",
        "causal_intervention_results_used": False,
        "validation_results_used": False,
        "test_results_used": False,
        "candidate_count": len(candidates),
        "per_route_cap": config.trajectory_max_candidates_per_route,
        "global_cap": config.trajectory_max_candidate_sites,
        "candidates": candidates,
        "selection_rules": selection_rules,
    }
    return {
        **freeze_record,
        "selection_record_hash": canonical_hash(freeze_record),
        "frozen": True,
    }


def _phase_sequence_length(examples: Sequence[BenchmarkExample]) -> int:
    return max(
        len(prompt) + len(answer)
        for example in examples
        for _, _, prompt, answer in _candidate_variants(example)
    )


def build_candidate_batch(
        examples: Sequence[BenchmarkExample], *, pad_token_id: int,
        data_multiple: int, sequence_length: int) -> dict[str, Any]:
    rows = []
    for example_index, example in enumerate(examples):
        for side, candidate, prompt, answer in _candidate_variants(example):
            tokens = tuple(prompt) + tuple(answer)
            if len(tokens) > sequence_length:
                raise ValueError("candidate batch sequence length is too small")
            rows.append({
                "example_index": example_index,
                "side": side,
                "candidate": candidate,
                "prompt_length": len(prompt),
                "tokens": tokens,
            })
    real_count = len(rows)
    batch_size = _round_up(real_count, max(1, data_multiple))
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    labels = np.full(
        (batch_size, sequence_length), -100, dtype=np.int32)
    prompt_lengths = np.zeros((batch_size,), dtype=np.int32)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        tokens = np.asarray(row["tokens"], dtype=np.int32)
        prompt_length = int(row["prompt_length"])
        input_ids[row_index, :len(tokens)] = tokens
        labels[row_index, prompt_length:len(tokens)] = tokens[prompt_length:]
        prompt_lengths[row_index] = prompt_length
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_lengths": prompt_lengths,
        "real_count": real_count,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def build_paired_state_batch(
        examples: Sequence[BenchmarkExample], *, pad_token_id: int,
        data_multiple: int, sequence_length: int) -> dict[str, Any]:
    """Build one causal-equivalent scoring row per base/source prompt."""
    rows = []
    for example_index, example in enumerate(examples):
        for side, prompt, answer in (
                ("base", example.input_ids_base, example.positive_ids),
                ("source", example.input_ids_source,
                 example.source_positive_ids)):
            tokens = tuple(prompt) + tuple(answer)
            if len(tokens) > sequence_length:
                raise ValueError("state batch sequence length is too small")
            rows.append({
                "example_index": example_index,
                "side": side,
                "prompt_length": len(prompt),
                "tokens": tokens,
            })
    real_count = len(rows)
    batch_size = _round_up(real_count, max(1, data_multiple))
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    labels = np.full(
        (batch_size, sequence_length), -100, dtype=np.int32)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        tokens = np.asarray(row["tokens"], dtype=np.int32)
        prompt_length = int(row["prompt_length"])
        input_ids[row_index, :len(tokens)] = tokens
        labels[row_index, prompt_length:len(tokens)] = tokens[prompt_length:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "real_count": real_count,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def _production_atlas_executable(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = "paired_trajectory_production_atlas"
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx)

    @jax.jit
    def execute(params, input_ids, labels):
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            minimal_train=True, analysis_parity_debug=True,
            analysis_return_residual=False,
            **kwargs)
        debug = result["parity_debug"]
        return (
            _score_from_result(result, "sum_log_probability"),
            {
                "residual_input": debug["residual_input"],
                "post_attention": debug["post_attention"],
                "post_layer_residual": debug["post_layer_residual"],
            },
        )

    cache[key] = execute
    return execute


def _candidate_site_executable(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = "paired_trajectory_candidate_semantic_sites"
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx)

    @jax.jit
    def execute(params, input_ids, labels, positions, position_valid):
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            minimal_train=True, analysis_parity_debug=True,
            analysis_return_residual=False,
            **kwargs)
        debug = result["parity_debug"]
        batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)[:, None]
        positions = jnp.clip(
            jnp.asarray(positions, dtype=jnp.int32),
            0, input_ids.shape[1] - 1)
        valid = jnp.asarray(position_valid, dtype=jnp.bool_)

        def gather(value):
            gathered = value[:, batch, positions, :]
            return jnp.where(
                valid[None, :, :, None], gathered,
                jnp.zeros_like(gathered))

        return {
            name: gather(debug[name])
            for name in (
                "residual_input", "q", "k", "v", "rst")
        }

    cache[key] = execute
    return execute


def _semantic_role_at(record: IOISemanticRecord, position: int) -> str:
    if position < record.s2_start:
        return "prefix"
    if position in record.positions:
        return record.position_roles[
            record.positions.index(position)]
    return "prompt_other"


def _answer_directions(
        ctx: Any, examples: Sequence[BenchmarkExample]) -> dict[str, np.ndarray]:
    embedding = ctx.params["token_emb"]["embedding"]
    unique_ids = sorted({
        int(token)
        for example in examples
        for ids in (example.positive_ids, example.negative_ids)
        for token in ids
    })
    selected = materialize_global_array(
        embedding[jnp.asarray(unique_ids, dtype=jnp.int32)])
    lookup = {
        token: np.asarray(selected[index], dtype=np.float32)
        for index, token in enumerate(unique_ids)
    }
    output = {}
    for example in examples:
        positive = np.mean([
            lookup[int(token)] for token in example.positive_ids], axis=0)
        negative = np.mean([
            lookup[int(token)] for token in example.negative_ids], axis=0)
        output[example.example_id] = positive - negative
    return output


def capture_production_atlas(
        ctx: Any, examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *,
        pad_token_id: int, config: ProtocolConfig,
        phase: str, progress: Any | None = None) -> dict[str, Any]:
    if not examples:
        raise ValueError(f"trajectory {phase} cohort is empty")
    if _candidate_score_normalization(examples) != "sum_log_probability":
        raise ValueError("IOI trajectory requires sum-log-probability scoring")
    executable = _production_atlas_executable(ctx)
    data_multiple = max(1, int(ctx.mesh.shape["data"]))
    # One example expands to four fused candidate rows.  Using one example
    # per data shard keeps every device busy without a broad-cohort state dump.
    examples_per_batch = max(1, data_multiple)
    sequence_length = _phase_sequence_length(examples)
    directions = _answer_directions(ctx, examples)
    metric_fields = (
        "base_norm", "source_norm", "delta_norm", "relative_delta_norm",
        "cosine", "base_answer_projection", "source_answer_projection",
        "delta_answer_projection",
    )
    accumulators: dict[tuple[int, str, str], dict[str, float]] = {}
    base_margin = np.empty((len(examples),), dtype=np.float64)
    source_margin = np.empty((len(examples),), dtype=np.float64)
    prefix_max_abs = 0.0
    forward_call_count = 0
    started = time.monotonic()
    for batch_index, start in enumerate(
            range(0, len(examples), examples_per_batch)):
        chunk = list(examples[start:start + examples_per_batch])
        execution_chunk = _pad_example_batch(chunk, examples_per_batch)
        built = build_candidate_batch(
            execution_chunk, pad_token_id=pad_token_id,
            data_multiple=data_multiple, sequence_length=sequence_length)
        ids = _device(ctx, built["input_ids"], P("data", None))
        labels = _device(ctx, built["labels"], P("data", None))
        score, debug = materialize_global_tree(executable(
            ctx.params, ids, labels))
        forward_call_count += 1
        score = np.asarray(score)
        debug = {name: np.asarray(value) for name, value in debug.items()}
        if progress is not None:
            progress(
                f"stage=production_atlas phase={phase} "
                f"batch_index={batch_index} example_count={len(chunk)} "
                "candidate_count=0 intervention_variants=4 "
                "compile=reuse elapsed_s="
                f"{time.monotonic() - started:.1f} capture_width=none "
                "retry_count=0")
        for local_index, example in enumerate(chunk):
            global_index = start + local_index
            base_row = local_index * 4
            source_row = base_row + 2
            base_margin[global_index] = (
                score[base_row] - score[base_row + 1])
            source_margin[global_index] = (
                score[source_row] - score[source_row + 1])
            record = semantic[example.example_id]
            direction = directions[example.example_id]
            for layer in range(int(ctx.model_cfg["n_layers"])):
                for position in range(record.base_prompt_tokens):
                    role = _semantic_role_at(record, position)
                    for stage, debug_name in (
                            ("residual_input", "residual_input"),
                            ("post_attention", "post_attention"),
                            ("post_rst", "post_layer_residual")):
                        base_state = debug[debug_name][
                            layer, base_row, position].astype(np.float32)
                        source_state = debug[debug_name][
                            layer, source_row, position].astype(np.float32)
                        delta = base_state - source_state
                        if position < record.s2_start:
                            prefix_max_abs = max(
                                prefix_max_abs,
                                float(np.max(np.abs(delta))))
                        base_norm = float(np.linalg.norm(base_state))
                        delta_norm = float(np.linalg.norm(delta))
                        metrics = {
                            "base_norm": base_norm,
                            "source_norm": float(
                                np.linalg.norm(source_state)),
                            "delta_norm": delta_norm,
                            "relative_delta_norm": float(
                                delta_norm / max(base_norm, 1.0e-12)),
                            "cosine": _cosine(base_state, source_state),
                            "base_answer_projection": float(
                                np.dot(base_state, direction)),
                            "source_answer_projection": float(
                                np.dot(source_state, direction)),
                            "delta_answer_projection": float(
                                np.dot(delta, direction)),
                        }
                        key = (layer, role, stage)
                        bucket = accumulators.get(key)
                        if bucket is None:
                            bucket = {"count": 0.0}
                            for field in metric_fields:
                                bucket[f"{field}_sum"] = 0.0
                                bucket[f"{field}_max"] = -math.inf
                            accumulators[key] = bucket
                        bucket["count"] += 1.0
                        for field, value in metrics.items():
                            bucket[f"{field}_sum"] += float(value)
                            bucket[f"{field}_max"] = max(
                                bucket[f"{field}_max"], float(value))
        del score, debug, ids, labels, built, execution_chunk
        gc.collect()
    rows = []
    for (layer, role, stage), values in accumulators.items():
        count = int(values["count"])
        row = {
            "layer": int(layer), "semantic_role": role, "stage": stage,
            "observation_count": count,
        }
        for field in metric_fields:
            row[f"{field}_mean"] = float(
                values[f"{field}_sum"] / count)
            row[f"{field}_max"] = float(values[f"{field}_max"])
        rows.append(row)
    rows.sort(key=lambda row: (
        row["layer"], _semantic_role_order(row["semantic_role"]),
        STATE_STAGES.index(row["stage"])))
    prefix_passed = bool(
        prefix_max_abs <= config.trajectory_state_identity_atol)
    return {
        "status": "ready" if prefix_passed else "s2_prefix_identity_failed",
        "phase": phase,
        "example_count": len(examples),
        "base_margin_mean": float(base_margin.mean()),
        "source_own_margin_mean": float(source_margin.mean()),
        "base_accuracy": float(np.mean(base_margin > 0.0)),
        "source_accuracy": float(np.mean(source_margin > 0.0)),
        "s2_prefix_state_identity": {
            "max_abs_error": float(prefix_max_abs),
            "atol": float(config.trajectory_state_identity_atol),
            "passed": prefix_passed,
        },
        "metric_rows": rows,
        "answer_projection_interpretation": (
            "descriptive_logit_lens_not_intermediate_causal_effect"),
        "forward_call_count": forward_call_count,
        "_base_margin": base_margin,
        "_source_margin": source_margin,
    }


def write_atlas_metric_artifact(
        store: Any, atlas: Mapping[str, Any], *, phase: str,
        protocol_hash: str) -> dict[str, Any] | None:
    if not store.is_primary:
        return None
    rows = list(atlas["metric_rows"])
    scalar_keys = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float))
    })
    path = store.path(
        "trajectory", "atlas", f"{phase}_metrics.npz")
    arrays = {
        "protocol_hash": np.asarray(protocol_hash),
        "algorithm_version": np.asarray(
            PAIRED_TRAJECTORY_ALGORITHM_VERSION),
        "layer": np.asarray([row["layer"] for row in rows], np.int16),
        "semantic_role": np.asarray([
            row["semantic_role"] for row in rows]),
        "stage": np.asarray([row["stage"] for row in rows]),
    }
    for key in scalar_keys:
        arrays[key] = np.asarray([
            row.get(key, np.nan) for row in rows], dtype=np.float64)
    write_npz_atomic(path, **arrays)
    return _artifact_record(
        path, phase=phase, row_count=len(rows), raw_vectors_persisted=False)


def _position_for_role(record: IOISemanticRecord, role: str) -> int:
    matches = [
        position for position, value in zip(
            record.positions, record.position_roles)
        if value == role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{record.example_id}: semantic role {role!r} is not unique")
    return int(matches[0])


def capture_candidate_site_values(
        ctx: Any, examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord],
        candidates: Sequence[Mapping[str, Any]], *,
        pad_token_id: int, phase: str, progress: Any | None = None
) -> dict[str, Any]:
    """Recompute production and retain only exact frozen-site vectors."""
    executable = _candidate_site_executable(ctx)
    data_multiple = max(1, int(ctx.mesh.shape["data"]))
    examples_per_batch = max(1, data_multiple)
    sequence_length = max(
        len(prompt) + len(answer)
        for example in examples
        for prompt, answer in (
            (example.input_ids_base, example.positive_ids),
            (example.input_ids_source, example.source_positive_ids),
        ))
    count = len(examples)
    candidate_count = len(candidates)
    trace_width = max(
        len(semantic[example.example_id].positions)
        for example in examples)
    d_model = int(ctx.model_cfg["d_model"])
    output = {
        side: {
            "route": np.zeros(
                (count, candidate_count, d_model), dtype=np.float32),
            "residual": np.zeros(
                (count, candidate_count, d_model), dtype=np.float32),
            "position": np.zeros(
                (count, candidate_count), dtype=np.int32),
        }
        for side in ("base", "source")
    }
    started = time.monotonic()
    forward_call_count = 0
    for batch_index, start in enumerate(
            range(0, count, examples_per_batch)):
        chunk = list(examples[start:start + examples_per_batch])
        execution_chunk = _pad_example_batch(chunk, examples_per_batch)
        built = build_paired_state_batch(
            execution_chunk, pad_token_id=pad_token_id,
            data_multiple=data_multiple, sequence_length=sequence_length)
        trace_positions = np.zeros(
            (built["batch_size"], trace_width), dtype=np.int32)
        trace_valid = np.zeros_like(trace_positions, dtype=np.bool_)
        for row_index in range(built["batch_size"]):
            local_index = (
                row_index // 2 if row_index < built["real_count"] else 0)
            record = semantic[execution_chunk[local_index].example_id]
            width = len(record.positions)
            trace_positions[row_index, :width] = np.asarray(
                record.positions, dtype=np.int32)
            trace_valid[row_index, :width] = True
        debug = materialize_global_tree(executable(
            ctx.params,
            _device(ctx, built["input_ids"], P("data", None)),
            _device(ctx, built["labels"], P("data", None)),
            _device(ctx, trace_positions, P("data", None)),
            _device(ctx, trace_valid, P("data", None))))
        forward_call_count += 1
        debug = {name: np.asarray(value) for name, value in debug.items()}
        for local_index, example in enumerate(chunk):
            global_index = start + local_index
            record = semantic[example.example_id]
            side_rows = {"base": local_index * 2,
                         "source": local_index * 2 + 1}
            for candidate_index, candidate in enumerate(candidates):
                layer = int(candidate["layer"])
                role = str(candidate["semantic_role"])
                route = str(candidate["route"])
                position = _position_for_role(record, role)
                position_slot = record.positions.index(position)
                for side, row in side_rows.items():
                    output[side]["route"][
                        global_index, candidate_index] = debug[route][
                            layer, row, position_slot].astype(np.float32)
                    output[side]["residual"][
                        global_index, candidate_index] = debug[
                            "residual_input"][
                                layer, row, position_slot].astype(np.float32)
                    output[side]["position"][
                        global_index, candidate_index] = position
        if progress is not None:
            progress(
                f"stage=candidate_value_capture phase={phase} "
                f"batch_index={batch_index} example_count={len(chunk)} "
                f"candidate_count={candidate_count} intervention_variants=0 "
                f"compile=reuse elapsed_s={time.monotonic() - started:.1f} "
                "capture_width=none retry_count=0")
        del debug, trace_positions, trace_valid, built, execution_chunk
        gc.collect()
    output["sequence_length"] = sequence_length
    output["forward_call_count"] = forward_call_count
    return output


def deterministic_mismatch_mapping(
        examples: Sequence[BenchmarkExample], *, seed: int) -> dict[str, Any]:
    if len(examples) < 2:
        raise ValueError("mismatched-pair control needs at least two examples")
    donors = []
    for recipient_index, recipient in enumerate(examples):
        recipient_answers = {
            tuple(recipient.positive_ids), tuple(recipient.negative_ids)}
        template = str(recipient.metadata.get("template") or "")
        eligible = []
        for donor_index, donor in enumerate(examples):
            if donor_index == recipient_index:
                continue
            if str(donor.metadata.get("template") or "") != template:
                continue
            donor_answers = {tuple(donor.positive_ids), tuple(donor.negative_ids)}
            if recipient_answers & donor_answers:
                continue
            eligible.append(donor_index)
        if not eligible:
            raise ValueError(
                f"{recipient.example_id}: no answer-disjoint same-template donor")
        eligible.sort(key=lambda donor_index: hashlib.sha256(
            f"{seed}:{recipient.example_id}:"
            f"{examples[donor_index].example_id}".encode("utf-8")
        ).hexdigest())
        donors.append(int(eligible[0]))
    record = {
        "seed": int(seed),
        "matching": "same_template_semantic_role_answer_disjoint",
        "recipient_example_ids": [example.example_id for example in examples],
        "donor_indices": donors,
        "donor_example_ids": [examples[index].example_id for index in donors],
    }
    return {**record, "mapping_hash": canonical_hash(record)}


def _trajectory_patch_executable(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = "paired_trajectory_fixed_patch_score"
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx)

    @jax.jit
    def execute(
            params, input_ids, labels, target_positions,
            patch_layers, patch_positions, patch_stages,
            patch_enabled, patch_values):
        result = ctx.model.apply(
            {"params": params}, input_ids,
            patch_layers=patch_layers,
            patch_positions=patch_positions,
            patch_stages=patch_stages,
            patch_enabled=patch_enabled,
            patch_values=patch_values,
            labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=True,
            return_logits=False,
            method=ctx.model.analysis_forward_with_trajectory_patches,
            **kwargs)
        batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
        target = result["final_residual"][batch, target_positions, :]
        return _score_from_result(
            result, "sum_log_probability"), target

    cache[key] = execute
    return execute


def _batched_intervention_rows(
        examples: Sequence[BenchmarkExample], *, direction: str,
        sequence_length: int, pad_token_id: int, data_multiple: int,
        patch_slots: int, d_model: int,
        tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fuse several candidate sites into one fixed patch executable call."""
    variant_names = ("paired", "mismatched", "self", "disabled")
    rows = []
    for task_index, task in enumerate(tasks):
        for example_index, example in enumerate(examples):
            if direction == "source_to_base":
                prompt = example.input_ids_source
                answers = (example.positive_ids, example.negative_ids)
                target_position = example.trace_position_source
            else:
                prompt = example.input_ids_base
                answers = (example.negative_ids, example.positive_ids)
                target_position = example.trace_position_base
            values = (
                task["paired_values"][example_index],
                task["mismatch_values"][example_index],
                task["self_values"][example_index],
                np.zeros((d_model,), dtype=np.float32),
            )
            for variant_index, variant in enumerate(variant_names):
                for answer_index, answer in enumerate(answers):
                    rows.append({
                        "task_index": task_index,
                        "example_index": example_index,
                        "variant_index": variant_index,
                        "answer_index": answer_index,
                        "tokens": tuple(prompt) + tuple(answer),
                        "prompt_length": len(prompt),
                        "target_position": int(target_position),
                        "patch_position": int(
                            task["positions"][example_index]),
                        "patch_value": values[variant_index],
                        "enabled": variant != "disabled",
                        "layer": int(task["layer"]),
                        "stage": int(task["stage"]),
                    })
    real_count = len(rows)
    batch_size = _round_up(real_count, data_multiple)
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    labels = np.full(
        (batch_size, sequence_length), -100, dtype=np.int32)
    target_positions = np.zeros((batch_size,), dtype=np.int32)
    patch_shape = (batch_size, patch_slots)
    patch_layers = np.zeros(patch_shape, dtype=np.int32)
    patch_positions = np.zeros(patch_shape, dtype=np.int32)
    patch_stages = np.zeros(patch_shape, dtype=np.int32)
    patch_enabled = np.zeros(patch_shape, dtype=np.bool_)
    patch_values = np.zeros(
        patch_shape + (d_model,), dtype=np.float32)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        tokens = np.asarray(row["tokens"], dtype=np.int32)
        prompt_length = int(row["prompt_length"])
        input_ids[row_index, :tokens.size] = tokens
        labels[row_index, prompt_length:tokens.size] = tokens[prompt_length:]
        target_positions[row_index] = int(row["target_position"])
        patch_layers[row_index, 0] = int(row["layer"])
        patch_positions[row_index, 0] = int(row["patch_position"])
        patch_stages[row_index, 0] = int(row["stage"])
        patch_enabled[row_index, 0] = bool(row["enabled"])
        patch_values[row_index, 0] = np.asarray(
            row["patch_value"], dtype=np.float32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "target_positions": target_positions,
        "patch_layers": patch_layers,
        "patch_positions": patch_positions,
        "patch_stages": patch_stages,
        "patch_enabled": patch_enabled,
        "patch_values": patch_values,
        "real_count": real_count,
        "task_count": len(tasks),
        "example_count": len(examples),
    }


def _execute_patch_rows(
        ctx: Any, executable: Any, rows: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    score, residual = materialize_global_tree(executable(
        ctx.params,
        _device(ctx, rows["input_ids"], P("data", None)),
        _device(ctx, rows["labels"], P("data", None)),
        _device(ctx, rows["target_positions"], P("data")),
        _device(ctx, rows["patch_layers"], P("data", None)),
        _device(ctx, rows["patch_positions"], P("data", None)),
        _device(ctx, rows["patch_stages"], P("data", None)),
        _device(ctx, rows["patch_enabled"], P("data", None)),
        _device(ctx, rows["patch_values"], P("data", None, None)),
    ))
    real_count = int(rows["real_count"])
    return np.asarray(score)[:real_count], np.asarray(residual)[:real_count]


def _summarize_patch_direction(
        margins: np.ndarray, residual: np.ndarray, *,
        before: np.ndarray, self_atol: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = margins.shape[0]
    if margins.shape != (count, 4):
        raise ValueError("trajectory patch margin shape mismatch")
    paired = margins[:, 0]
    mismatched = margins[:, 1]
    self_margin = margins[:, 2]
    disabled = margins[:, 3]
    self_error = np.abs(self_margin - before)
    disabled_error = np.abs(disabled - before)
    paired_effect = paired - before
    mismatch_effect = mismatched - before
    target_residual = residual.reshape(count, 4, -1)
    downstream = np.linalg.norm(
        target_residual[:, 0] - target_residual[:, 3], axis=-1)
    summary = {
        "before_margin_mean": float(before.mean()),
        "paired_after_margin_mean": float(paired.mean()),
        "paired_margin_shift_mean": float(paired_effect.mean()),
        "paired_answer_flip_fraction": float(np.mean(
            (before <= 0.0) & (paired > 0.0))),
        "mismatched_after_margin_mean": float(mismatched.mean()),
        "mismatched_margin_shift_mean": float(mismatch_effect.mean()),
        "paired_minus_mismatched_effect_mean": float(np.mean(
            paired_effect - mismatch_effect)),
        "self_reconstruction_max_abs": float(self_error.max()),
        "self_reconstruction_passed": bool(
            np.all(self_error <= self_atol)),
        "disabled_noop_max_abs": float(disabled_error.max()),
        "disabled_noop_passed": bool(
            np.all(disabled_error <= self_atol)),
        "downstream_residual_divergence_mean": float(downstream.mean()),
    }
    vectors = {
        "before": before,
        "paired": paired,
        "mismatched": mismatched,
        "self": self_margin,
        "disabled": disabled,
        "paired_effect": paired_effect,
        "mismatched_effect": mismatch_effect,
        "downstream_residual_divergence": downstream,
    }
    return summary, vectors


def _evaluate_coarse_site_patches_once(
        ctx: Any, examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord],
        candidates: Sequence[Mapping[str, Any]],
        site_values: Mapping[str, Any], mismatch: Mapping[str, Any], *,
        production_atlas: Mapping[str, Any], pad_token_id: int,
        config: ProtocolConfig, phase: str, task_batch_size: int,
        patch_kinds: Sequence[str],
        progress: Any | None = None) -> dict[str, Any]:
    executable = _trajectory_patch_executable(ctx)
    data_multiple = max(1, int(ctx.mesh.shape["data"]))
    # Four intervention variants x two answer candidates per example.
    examples_per_batch = max(1, data_multiple)
    sequence_length = int(site_values["sequence_length"])
    donor_indices = np.asarray(mismatch["donor_indices"], dtype=np.int32)
    before_by_direction = {
        "source_to_base": -np.asarray(
            production_atlas["_source_margin"], dtype=np.float64),
        "base_to_source": -np.asarray(
            production_atlas["_base_margin"], dtype=np.float64),
    }
    definitions = []
    for candidate_index, candidate in enumerate(candidates):
        value_index = int(candidate.get("candidate_index", candidate_index))
        for patch_kind in patch_kinds:
            stage_name = (
                str(candidate["route"])
                if patch_kind == "route" else "residual_input")
            definitions.append({
                "candidate_index": value_index,
                "candidate": dict(candidate),
                "patch_kind": patch_kind,
                "stage_name": stage_name,
                "stage": PATCH_STAGE[stage_name],
            })
    direction_results: dict[int, dict[str, Any]] = {
        index: {} for index in range(len(definitions))
    }
    forward_call_count = 0
    started = time.monotonic()
    for definition_start in range(
            0, len(definitions), task_batch_size):
        definition_stop = min(
            len(definitions), definition_start + task_batch_size)
        definition_batch = definitions[definition_start:definition_stop]
        for direction in ("source_to_base", "base_to_source"):
            donor_side, recipient_side = (
                ("base", "source") if direction == "source_to_base"
                else ("source", "base"))
            task_margins: list[list[np.ndarray]] = [
                [] for _ in definition_batch]
            task_residuals: list[list[np.ndarray]] = [
                [] for _ in definition_batch]
            for batch_index, start in enumerate(
                    range(0, len(examples), examples_per_batch)):
                stop = min(len(examples), start + examples_per_batch)
                chunk = list(examples[start:stop])
                indices = np.arange(start, stop)
                execution_chunk = _pad_example_batch(
                    chunk, examples_per_batch)
                execution_indices = np.pad(
                    indices,
                    (0, examples_per_batch - len(indices)),
                    mode="edge")
                tasks = []
                for definition in definition_batch:
                    value_index = int(definition["candidate_index"])
                    patch_kind = str(definition["patch_kind"])
                    tasks.append({
                        "layer": int(definition["candidate"]["layer"]),
                        "stage": int(definition["stage"]),
                        "positions": np.asarray(
                            site_values[recipient_side]["position"]
                        )[execution_indices, value_index],
                        "paired_values": np.asarray(
                            site_values[donor_side][patch_kind]
                        )[execution_indices, value_index],
                        "mismatch_values": np.asarray(
                            site_values[donor_side][patch_kind]
                        )[donor_indices[execution_indices], value_index],
                        "self_values": np.asarray(
                            site_values[recipient_side][patch_kind]
                        )[execution_indices, value_index],
                    })
                tasks.extend(
                    dict(tasks[0])
                    for _ in range(task_batch_size - len(tasks)))
                rows = _batched_intervention_rows(
                    execution_chunk, direction=direction,
                    sequence_length=sequence_length,
                    pad_token_id=pad_token_id,
                    data_multiple=data_multiple,
                    patch_slots=(
                        config.trajectory_max_patch_sites_per_variant),
                    d_model=int(ctx.model_cfg["d_model"]),
                    tasks=tasks)
                score, residual = _execute_patch_rows(
                    ctx, executable, rows)
                forward_call_count += 1
                score = score.reshape(
                    task_batch_size, examples_per_batch, 4, 2
                )[:len(definition_batch), :len(chunk)]
                residual = residual.reshape(
                    task_batch_size, examples_per_batch, 4, 2, -1
                )[:len(definition_batch), :len(chunk)]
                for local_task in range(len(definition_batch)):
                    task_margins[local_task].append(
                        score[local_task, :, :, 0]
                        - score[local_task, :, :, 1])
                    task_residuals[local_task].append(
                        residual[local_task, :, :, 0])
                if progress is not None:
                    progress(
                        f"stage=coarse_patch phase={phase} "
                        f"batch_index={batch_index} "
                        f"example_count={len(chunk)} "
                        f"candidate_count={definition_stop}/"
                        f"{len(definitions)} intervention_variants=4 "
                        f"interventions_fused={len(definition_batch)} "
                        "compile=reuse elapsed_s="
                        f"{time.monotonic() - started:.1f} "
                        "capture_width=none retry_count=0")
                del execution_chunk, execution_indices, rows
            for local_task in range(len(definition_batch)):
                margins = np.concatenate(
                    task_margins[local_task], axis=0)
                residual = np.concatenate(
                    task_residuals[local_task], axis=0)
                summary, vectors = _summarize_patch_direction(
                    margins, residual,
                    before=before_by_direction[direction],
                    self_atol=config.trajectory_replay_atol)
                direction_results[definition_start + local_task][direction] = {
                    "summary": summary,
                    "vectors": vectors,
                }
    summaries = []
    vector_records = []
    for definition_index, definition in enumerate(definitions):
        direction_summaries = {
            direction: values["summary"]
            for direction, values in direction_results[
                definition_index].items()
        }
        direction_vectors = {
            direction: values["vectors"]
            for direction, values in direction_results[
                definition_index].items()
        }
        paired_effect = np.mean([
            value["paired_margin_shift_mean"]
            for value in direction_summaries.values()])
        mismatch_effect = np.mean([
            value["mismatched_margin_shift_mean"]
            for value in direction_summaries.values()])
        candidate = definition["candidate"]
        summaries.append({
            "candidate_index": int(definition["candidate_index"]),
            "layer": int(candidate["layer"]),
            "semantic_role": str(candidate["semantic_role"]),
            "route": str(candidate["route"]),
            "patch_kind": str(definition["patch_kind"]),
            "stage": str(definition["stage_name"]),
            "directions": direction_summaries,
            "bidirectional_paired_effect_mean": float(paired_effect),
            "bidirectional_mismatched_effect_mean": float(mismatch_effect),
            "bidirectional_specific_effect_mean": float(
                paired_effect - mismatch_effect),
        })
        vector_records.append({
            "candidate_index": int(definition["candidate_index"]),
            "patch_kind": str(definition["patch_kind"]),
            "directions": direction_vectors,
        })
    return {
        "status": "ready",
        "phase": phase,
        "candidate_count": len(candidates),
        "patch_kinds_evaluated": list(patch_kinds),
        "evaluated_patch_count": len(summaries),
        "intervention_batch_size": int(task_batch_size),
        "candidate_interventions_fused_per_forward": (
            int(task_batch_size)),
        "fixed_patch_slots": config.trajectory_max_patch_sites_per_variant,
        "candidate_specific_jit_compilation": False,
        "params_replicated_per_variant": False,
        "candidate_answer_variants_fused_in_batch": True,
        "last_candidate_chunk_padded_to_fixed_shape": True,
        "last_example_chunk_padded_to_fixed_shape": True,
        "phase_baseline_reused": True,
        "forward_call_count": forward_call_count,
        "mismatch_mapping_hash": mismatch["mapping_hash"],
        "site_summaries": summaries,
        "_vectors": vector_records,
    }


def evaluate_coarse_site_patches(
        ctx: Any, examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord],
        candidates: Sequence[Mapping[str, Any]],
        site_values: Mapping[str, Any], mismatch: Mapping[str, Any], *,
        production_atlas: Mapping[str, Any], pad_token_id: int,
        config: ProtocolConfig, phase: str,
        patch_kinds: Sequence[str] = ("route", "residual"),
        progress: Any | None = None) -> dict[str, Any]:
    patch_kinds = tuple(str(value) for value in patch_kinds)
    if (not patch_kinds or len(set(patch_kinds)) != len(patch_kinds)
            or any(value not in {"route", "residual"}
                   for value in patch_kinds)):
        raise ValueError(
            "coarse trajectory patch kinds must be unique route/residual "
            "values")
    initial = int(config.trajectory_intervention_batch_size)
    effective = initial
    retries = []
    while True:
        try:
            result = _evaluate_coarse_site_patches_once(
                ctx, examples, semantic, candidates, site_values, mismatch,
                production_atlas=production_atlas,
                pad_token_id=pad_token_id, config=config, phase=phase,
                task_batch_size=effective, patch_kinds=patch_kinds,
                progress=progress)
            result["initial_intervention_batch_size"] = initial
            result["effective_intervention_batch_size"] = effective
            result["resource_retry_count"] = len(retries)
            result["resource_retries"] = retries
            return result
        except Exception as exc:
            message = str(exc).lower()
            exhausted = isinstance(exc, MemoryError) or any(
                value in message for value in (
                    "resource_exhausted", "out of memory", "oom",
                    "allocation failed"))
            if not exhausted or effective <= 1:
                raise
            updated = max(1, effective // 2)
            retries.append({
                "retry_index": len(retries) + 1,
                "previous_intervention_batch_size": effective,
                "updated_intervention_batch_size": updated,
                "failed_stage": f"coarse_patch_{phase}",
                "active_operator_truncation_applied": False,
            })
            gc.collect()
            effective = updated


def deduplicate_residual_candidates(
        candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep one residual diagnostic per layer/semantic position."""
    selected = []
    seen = set()
    for candidate in candidates:
        key = (int(candidate["layer"]), str(candidate["semantic_role"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(candidate))
    return selected


def merge_staged_coarse_patch_results(
        route_result: Mapping[str, Any],
        residual_result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Combine route results and layer/role-deduplicated residual diagnostics."""
    if tuple(route_result.get("patch_kinds_evaluated") or ()) != ("route",):
        raise ValueError("staged coarse evaluation requires a route-only first pass")
    output = dict(route_result)
    route_summaries = list(route_result.get("site_summaries") or ())
    route_vectors = list(route_result.get("_vectors") or ())
    stages = {
        "route": {
            "status": str(route_result.get("status")),
            "candidate_count": int(route_result.get("candidate_count", 0)),
            "evaluated_patch_count": int(
                route_result.get("evaluated_patch_count", 0)),
            "forward_call_count": int(
                route_result.get("forward_call_count", 0)),
            "effective_intervention_batch_size": int(
                route_result.get("effective_intervention_batch_size", 0)),
            "resource_retry_count": int(
                route_result.get("resource_retry_count", 0)),
        },
    }
    if residual_result is None:
        residual_summaries = []
        residual_vectors = []
        stages["residual"] = {
            "status": "not_evaluated_no_causal_route_sites",
            "candidate_count": 0,
            "evaluated_patch_count": 0,
            "forward_call_count": 0,
            "effective_intervention_batch_size": 0,
            "resource_retry_count": 0,
        }
    else:
        if tuple(residual_result.get("patch_kinds_evaluated") or ()) != (
                "residual",):
            raise ValueError(
                "staged coarse evaluation requires a residual-only second pass")
        if residual_result.get("phase") != route_result.get("phase"):
            raise ValueError("staged coarse phases disagree")
        if residual_result.get("mismatch_mapping_hash") != (
                route_result.get("mismatch_mapping_hash")):
            raise ValueError("staged coarse mismatch mappings disagree")
        residual_summaries = list(
            residual_result.get("site_summaries") or ())
        residual_vectors = list(residual_result.get("_vectors") or ())
        stages["residual"] = {
            "status": str(residual_result.get("status")),
            "candidate_count": int(
                residual_result.get("candidate_count", 0)),
            "evaluated_patch_count": int(
                residual_result.get("evaluated_patch_count", 0)),
            "forward_call_count": int(
                residual_result.get("forward_call_count", 0)),
            "effective_intervention_batch_size": int(
                residual_result.get("effective_intervention_batch_size", 0)),
            "resource_retry_count": int(
                residual_result.get("resource_retry_count", 0)),
        }
    combined_summaries = route_summaries + residual_summaries
    combined_vectors = route_vectors + residual_vectors
    seen = set()
    for row in combined_summaries:
        key = (int(row["candidate_index"]), str(row["patch_kind"]))
        if key in seen:
            raise ValueError("staged coarse evaluation duplicated a patch")
        seen.add(key)
    combined_summaries.sort(key=lambda row: (
        int(row["candidate_index"]),
        0 if row["patch_kind"] == "route" else 1))
    combined_vectors.sort(key=lambda row: (
        int(row["candidate_index"]),
        0 if row["patch_kind"] == "route" else 1))
    route_positive = [
        int(row["candidate_index"])
        for row in route_summaries
        if _patch_row_passes_per_direction(row, "route")
    ]
    residual_positive = [
        {
            "representative_candidate_index": int(row["candidate_index"]),
            "layer": int(row["layer"]),
            "semantic_role": str(row["semantic_role"]),
        }
        for row in residual_summaries
        if _patch_row_passes_per_direction(row, "residual")
    ]
    resource_retries = list(route_result.get("resource_retries") or ())
    if residual_result is not None:
        resource_retries.extend(
            residual_result.get("resource_retries") or ())
    output.update({
        "patch_kinds_evaluated": ["route", "residual"],
        "evaluated_patch_count": len(combined_summaries),
        "site_summaries": combined_summaries,
        "_vectors": combined_vectors,
        "forward_call_count": sum(
            int(stage["forward_call_count"]) for stage in stages.values()),
        "resource_retry_count": sum(
            int(stage["resource_retry_count"]) for stage in stages.values()),
        "resource_retries": resource_retries,
        "route_first_staged_evaluation": True,
        "residual_diagnostics_selection_basis": (
            "all_candidates_deduplicated_by_layer_and_semantic_role"),
        "positive_sites": {
            "route_candidate_indices": route_positive,
            "residual_layer_roles": residual_positive,
            "operator_followup_basis": "route_positive_only",
            "causal_path_basis": "route_positive_only",
            "state_mediated_path_status": "exploratory_not_selected",
        },
        "stages": stages,
    })
    return output


def _patch_row_passes_per_direction(
        row: Mapping[str, Any], patch_kind: str) -> bool:
    if row.get("patch_kind") != patch_kind:
        return False
    directions = row.get("directions") or {}
    required = ("source_to_base", "base_to_source")
    if any(direction not in directions for direction in required):
        return False
    for direction in required:
        values = directions[direction]
        if float(values["paired_margin_shift_mean"]) <= 0.0:
            return False
        if float(values["paired_minus_mismatched_effect_mean"]) <= 0.0:
            return False
        if not (values["self_reconstruction_passed"]
                and values["disabled_noop_passed"]):
            return False
    return True


def _route_row_passes_per_direction(row: Mapping[str, Any]) -> bool:
    return _patch_row_passes_per_direction(row, "route")


def freeze_operator_followup_sites(
        coarse: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]], *,
        config: ProtocolConfig) -> dict[str, Any]:
    candidate_by_index = {
        int(row.get("candidate_index", index)): dict(row)
        for index, row in enumerate(candidates)
    }
    eligible = []
    for row in coarse["site_summaries"]:
        if not _route_row_passes_per_direction(row):
            continue
        eligible.append(dict(row))
    eligible.sort(key=lambda row: (
        -float(row["bidirectional_specific_effect_mean"]),
        -float(row["bidirectional_paired_effect_mean"]),
        int(row["candidate_index"]),
    ))
    sites = []
    for row in eligible[:config.trajectory_max_operator_followup_sites]:
        candidate_index = int(row["candidate_index"])
        candidate = candidate_by_index[candidate_index]
        sites.append({
            "candidate_index": candidate_index,
            "layer": int(candidate["layer"]),
            "semantic_role": str(candidate["semantic_role"]),
            "route": str(candidate["route"]),
            "coarse_bidirectional_paired_effect_mean": float(
                row["bidirectional_paired_effect_mean"]),
            "coarse_bidirectional_specific_effect_mean": float(
                row["bidirectional_specific_effect_mean"]),
        })
    record = {
        "status": (
            "ready" if sites else "no_causally_relevant_route_sites"),
        "algorithm_version": PAIRED_TRAJECTORY_ALGORITHM_VERSION,
        "selection_phase": "discovery",
        "selection_rule": (
            "positive_paired_and_specific_complete_route_effect_in_each_"
            "direction_with_self_and_disabled_controls"),
        "per_direction_gate_required": True,
        "validation_results_used": False,
        "test_results_used": False,
        "max_sites": config.trajectory_max_operator_followup_sites,
        "site_count": len(sites),
        "individual_operator_followup_limit": (
            config.trajectory_individual_operator_followup_limit),
        "sites": sites,
    }
    return {
        **record,
        "selection_record_hash": canonical_hash(record),
        "frozen": True,
    }


def _operator_group_ids(
        divergence: Mapping[str, Any], example_id: str,
        site: Mapping[str, Any], group_kind: str) -> np.ndarray:
    key = (
        f"route:{int(site['layer'])}:{site['semantic_role']}:"
        f"{site['route']}")
    row = divergence["_per_example_sites"][example_id][key]
    common = np.asarray(row["common_ids"], dtype=np.int32)
    support = np.union1d(
        np.asarray(row["base_only_ids"], dtype=np.int32),
        np.asarray(row["source_only_ids"], dtype=np.int32)).astype(
            np.int32, copy=False)
    if group_kind == "common_id_realized_contribution_swap":
        return common
    if group_kind == "support_only_swap":
        return support
    if group_kind == "full_local_differential_patch":
        return np.union1d(common, support).astype(np.int32, copy=False)
    raise ValueError(f"unknown trajectory operator group={group_kind}")


def _operator_group_executables(ctx: Any):
    cache = getattr(ctx, "_operator_interpretability_executables", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_operator_interpretability_executables", cache)
    key = "paired_trajectory_operator_group"
    if key in cache:
        return cache[key]
    kwargs = _runtime_kwargs(ctx)

    @jax.jit
    def capture(params, input_ids, positions, selected_ids, layer, route):
        result = ctx.model.apply(
            {"params": params}, input_ids, selected_ids, layer,
            positions, route, labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=False,
            method=ctx.model.analysis_capture_operator_group_contribution,
            **kwargs)
        values = jnp.stack([
            result["operator_route_contributions"][name]
            for name in ROUTES
        ], axis=0)
        return values[route, layer]

    @jax.jit
    def patch_score(
            params, input_ids, labels, positions, selected_ids,
            source_contribution, layer, route):
        result = ctx.model.apply(
            {"params": params}, input_ids, selected_ids, layer,
            positions, route, source_contribution, labels=labels,
            attention_mask=jnp.ones_like(input_ids),
            return_residual=True,
            method=ctx.model.analysis_forward_with_operator_interchange,
            **kwargs)
        batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
        target = result["final_residual"][batch, positions, :]
        return _score_from_result(
            result, "sum_log_probability"), target

    cache[key] = (capture, patch_score)
    return cache[key]


def _padded_group_ids(
        values: Sequence[np.ndarray], *, width: int) -> np.ndarray:
    output = np.full((len(values), width), -1, dtype=np.int32)
    for index, group in enumerate(values):
        normalized = np.unique(np.asarray(group, dtype=np.int32))
        if normalized.size > width:
            raise ValueError("operator group exceeds its frozen slot width")
        output[index, :normalized.size] = normalized
    return output


def _operator_group_capture_rows(
        examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *, direction: str,
        site: Mapping[str, Any], groups: Sequence[np.ndarray],
        mismatch_indices: np.ndarray, group_width: int,
        pad_token_id: int, data_multiple: int) -> dict[str, Any]:
    donor_side, recipient_side = (
        ("base", "source") if direction == "source_to_base"
        else ("source", "base"))
    rows = []
    for example_index, example in enumerate(examples):
        mismatch_index = int(mismatch_indices[example_index])
        variants = (
            ("paired", example_index, donor_side, example_index),
            ("same_group_mismatched_state", mismatch_index, donor_side,
             example_index),
            ("donor_own_group_mismatched_program", mismatch_index,
             donor_side, mismatch_index),
            ("self", example_index, recipient_side, example_index),
        )
        for variant, donor_index, side, group_index in variants:
            donor = examples[donor_index]
            prompt = (
                donor.input_ids_base if side == "base"
                else donor.input_ids_source)
            position = _position_for_role(
                semantic[donor.example_id], str(site["semantic_role"]))
            rows.append({
                "variant": variant,
                "prompt": tuple(prompt),
                "position": position,
                "group": groups[group_index],
            })
    real_count = len(rows)
    batch_size = _round_up(real_count, data_multiple)
    sequence_length = max(len(row["prompt"]) for row in rows)
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    positions = np.zeros((batch_size,), dtype=np.int32)
    selected = np.full((batch_size, group_width), -1, dtype=np.int32)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        prompt = np.asarray(row["prompt"], dtype=np.int32)
        input_ids[row_index, :prompt.size] = prompt
        positions[row_index] = int(row["position"])
        group = np.asarray(row["group"], dtype=np.int32)
        selected[row_index, :group.size] = group
    return {
        "input_ids": input_ids,
        "positions": positions,
        "selected_ids": selected,
        "real_count": real_count,
    }


def _operator_group_patch_rows(
        examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord], *, direction: str,
        site: Mapping[str, Any], groups: Sequence[np.ndarray],
        mismatch_indices: np.ndarray, contributions: np.ndarray,
        group_width: int,
        sequence_length: int, pad_token_id: int,
        data_multiple: int) -> dict[str, Any]:
    recipient_side = "source" if direction == "source_to_base" else "base"
    variants = (
        "paired", "same_group_mismatched_state",
        "donor_own_group_mismatched_program", "self", "suppressed")
    rows = []
    for example_index, example in enumerate(examples):
        if direction == "source_to_base":
            prompt = example.input_ids_source
            answers = (example.positive_ids, example.negative_ids)
        else:
            prompt = example.input_ids_base
            answers = (example.negative_ids, example.positive_ids)
        position = _position_for_role(
            semantic[example.example_id], str(site["semantic_role"]))
        for variant_index, variant in enumerate(variants):
            group_index = (
                int(mismatch_indices[example_index])
                if variant == "donor_own_group_mismatched_program"
                else example_index)
            source = (
                contributions[example_index, variant_index]
                if variant_index < 4 else
                np.zeros((contributions.shape[-1],), dtype=np.float32))
            for answer in answers:
                rows.append({
                    "tokens": tuple(prompt) + tuple(answer),
                    "prompt_length": len(prompt),
                    "position": position,
                    "group": groups[group_index],
                    "source": source,
                    "variant": variant,
                    "recipient_side": recipient_side,
                })
    real_count = len(rows)
    batch_size = _round_up(real_count, data_multiple)
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    labels = np.full(
        (batch_size, sequence_length), -100, dtype=np.int32)
    positions = np.zeros((batch_size,), dtype=np.int32)
    selected = np.full((batch_size, group_width), -1, dtype=np.int32)
    source = np.zeros(
        (batch_size, contributions.shape[-1]), dtype=np.float32)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        tokens = np.asarray(row["tokens"], dtype=np.int32)
        prompt_length = int(row["prompt_length"])
        input_ids[row_index, :tokens.size] = tokens
        labels[row_index, prompt_length:tokens.size] = tokens[prompt_length:]
        positions[row_index] = int(row["position"])
        group = np.asarray(row["group"], dtype=np.int32)
        selected[row_index, :group.size] = group
        source[row_index] = np.asarray(row["source"], dtype=np.float32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": positions,
        "selected_ids": selected,
        "source_contribution": source,
        "real_count": real_count,
    }


def _summarize_operator_group_direction(
        margins: np.ndarray, residual: np.ndarray, *, before: np.ndarray,
        self_atol: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if margins.shape != (len(before), 5):
        raise ValueError(
            "operator group margins must contain paired, both mismatch "
            "controls, self, and suppression variants")
    (paired, same_group_mismatched_state,
     donor_own_group_mismatched_program, self_margin, suppressed) = (
        margins[:, index] for index in range(5))
    paired_effect = paired - before
    same_group_mismatched_state_effect = (
        same_group_mismatched_state - before)
    donor_own_group_mismatched_program_effect = (
        donor_own_group_mismatched_program - before)
    self_error = np.abs(self_margin - before)
    target_residual = residual.reshape(len(before), 5, -1)
    downstream = np.linalg.norm(
        target_residual[:, 0] - target_residual[:, 3], axis=-1)
    summary = {
        "before_margin_mean": float(before.mean()),
        "paired_after_margin_mean": float(paired.mean()),
        "paired_margin_shift_mean": float(paired_effect.mean()),
        "paired_answer_flip_fraction": float(np.mean(
            (before <= 0.0) & (paired > 0.0))),
        "same_group_mismatched_state_margin_shift_mean": float(
            same_group_mismatched_state_effect.mean()),
        "donor_own_group_mismatched_program_margin_shift_mean": float(
            donor_own_group_mismatched_program_effect.mean()),
        # The donor-own program is the primary specificity control.  Keep the
        # generic fields aligned to it so downstream causal gates remain
        # conservative and backward-readable.
        "mismatched_margin_shift_mean": float(
            donor_own_group_mismatched_program_effect.mean()),
        "paired_minus_mismatched_effect_mean": float(np.mean(
            paired_effect - donor_own_group_mismatched_program_effect)),
        "paired_minus_same_group_mismatched_state_effect_mean": float(
            np.mean(paired_effect - same_group_mismatched_state_effect)),
        "paired_minus_donor_own_group_mismatched_program_effect_mean": float(
            np.mean(
                paired_effect
                - donor_own_group_mismatched_program_effect)),
        "suppression_margin_shift_mean": float(
            np.mean(suppressed - before)),
        "self_reconstruction_max_abs": float(self_error.max()),
        "self_reconstruction_passed": bool(
            np.all(self_error <= self_atol)),
        "downstream_residual_divergence_mean": float(downstream.mean()),
    }
    return summary, {
        "before": before,
        "paired": paired,
        "same_group_mismatched_state": same_group_mismatched_state,
        "donor_own_group_mismatched_program": (
            donor_own_group_mismatched_program),
        "mismatched": donor_own_group_mismatched_program,
        "self": self_margin,
        "suppressed": suppressed,
        "paired_effect": paired_effect,
        "same_group_mismatched_state_effect": (
            same_group_mismatched_state_effect),
        "donor_own_group_mismatched_program_effect": (
            donor_own_group_mismatched_program_effect),
        "mismatched_effect": donor_own_group_mismatched_program_effect,
        "downstream_residual_divergence": downstream,
    }


def evaluate_operator_group_patches(
        ctx: Any, examples: Sequence[BenchmarkExample],
        semantic: Mapping[str, IOISemanticRecord],
        divergence: Mapping[str, Any],
        followup: Mapping[str, Any], mismatch: Mapping[str, Any], *,
        production_atlas: Mapping[str, Any], sequence_length: int,
        pad_token_id: int, config: ProtocolConfig,
        progress: Any | None = None) -> dict[str, Any]:
    sites = list(followup["sites"])
    if not sites:
        return {
            "status": "no_causally_relevant_route_sites",
            "selection_record_hash": followup["selection_record_hash"],
            "mismatch_group_policy": "dual_control",
            "primary_mismatch_control": (
                "donor_own_group_mismatched_program"),
            "secondary_mismatch_control": "same_group_mismatched_state",
            "site_summaries": [], "_vectors": [],
            "forward_call_count": 0,
        }
    kinds = (
        "support_only_swap", "common_id_realized_contribution_swap",
        "full_local_differential_patch",
    )
    group_cache = {}
    maximum = 1
    for site_index, site in enumerate(sites):
        for kind in kinds:
            groups = [
                _operator_group_ids(
                    divergence, example.example_id, site, kind)
                for example in examples
            ]
            group_cache[(site_index, kind)] = groups
            maximum = max(maximum, *(len(group) for group in groups))
    group_width = int(maximum)
    capture, patch_score = _operator_group_executables(ctx)
    data_multiple = max(1, int(ctx.mesh.shape["data"]))
    mismatch_indices = np.asarray(
        mismatch["donor_indices"], dtype=np.int32)
    before_by_direction = {
        "source_to_base": -np.asarray(
            production_atlas["_source_margin"], dtype=np.float64),
        "base_to_source": -np.asarray(
            production_atlas["_base_margin"], dtype=np.float64),
    }
    summaries = []
    vectors = []
    forward_call_count = 0
    started = time.monotonic()
    for site_index, site in enumerate(sites):
        layer = jnp.asarray(int(site["layer"]), dtype=jnp.int32)
        route = jnp.asarray(
            ROUTE_INDEX[str(site["route"])], dtype=jnp.int32)
        site_key = (
            f"route:{int(site['layer'])}:{site['semantic_role']}:"
            f"{site['route']}")
        descriptive_rows = [
            divergence["_per_example_sites"][example.example_id][site_key]
            for example in examples
        ]
        descriptive_support = {
            "common_active_fraction_mean": float(np.mean([
                row["common_fraction"] for row in descriptive_rows])),
            "common_active_count_mean": float(np.mean([
                row["common_count"] for row in descriptive_rows])),
            "base_only_count_mean": float(np.mean([
                row["base_only_count"] for row in descriptive_rows])),
            "source_only_count_mean": float(np.mean([
                row["source_only_count"] for row in descriptive_rows])),
        }
        for kind in kinds:
            groups = group_cache[(site_index, kind)]
            direction_summaries = {}
            direction_vectors = {}
            for direction in ("source_to_base", "base_to_source"):
                capture_rows = _operator_group_capture_rows(
                    examples, semantic, direction=direction, site=site,
                    groups=groups, mismatch_indices=mismatch_indices,
                    group_width=group_width, pad_token_id=pad_token_id,
                    data_multiple=data_multiple)
                contribution = materialize_global_array(capture(
                    ctx.params,
                    _device(ctx, capture_rows["input_ids"], P("data", None)),
                    _device(ctx, capture_rows["positions"], P("data")),
                    _device(
                        ctx, capture_rows["selected_ids"], P("data", None)),
                    layer, route,
                ))[:capture_rows["real_count"]]
                forward_call_count += 1
                contribution = np.asarray(
                    contribution, dtype=np.float32).reshape(
                        len(examples), 4, -1)
                patch_rows = _operator_group_patch_rows(
                    examples, semantic, direction=direction, site=site,
                    groups=groups, mismatch_indices=mismatch_indices,
                    contributions=contribution,
                    group_width=group_width,
                    sequence_length=sequence_length,
                    pad_token_id=pad_token_id,
                    data_multiple=data_multiple)
                score, residual = materialize_global_tree(patch_score(
                    ctx.params,
                    _device(ctx, patch_rows["input_ids"], P("data", None)),
                    _device(ctx, patch_rows["labels"], P("data", None)),
                    _device(ctx, patch_rows["positions"], P("data")),
                    _device(
                        ctx, patch_rows["selected_ids"], P("data", None)),
                    _device(
                        ctx, patch_rows["source_contribution"],
                        P("data", None)),
                    layer, route,
                ))
                forward_call_count += 1
                real_count = int(patch_rows["real_count"])
                scores = np.asarray(score)[:real_count].reshape(
                    len(examples), 5, 2)
                margins = scores[:, :, 0] - scores[:, :, 1]
                residual_values = np.asarray(residual)[:real_count].reshape(
                    len(examples), 5, 2, -1)[:, :, 0]
                summary, vector = _summarize_operator_group_direction(
                    margins, residual_values,
                    before=before_by_direction[direction],
                    self_atol=config.trajectory_replay_atol)
                direction_summaries[direction] = summary
                direction_vectors[direction] = vector
            group_counts = np.asarray(
                [len(group) for group in groups], dtype=np.int32)
            paired_effect = float(np.mean([
                value["paired_margin_shift_mean"]
                for value in direction_summaries.values()
            ]))
            mismatch_effect = float(np.mean([
                value["mismatched_margin_shift_mean"]
                for value in direction_summaries.values()
            ]))
            summaries.append({
                "candidate_index": int(site["candidate_index"]),
                "layer": int(site["layer"]),
                "semantic_role": str(site["semantic_role"]),
                "route": str(site["route"]),
                "group_kind": kind,
                "group_nonempty_fraction": float(np.mean(group_counts > 0)),
                "group_size_mean": float(group_counts.mean()),
                "group_size_max": int(group_counts.max()),
                **descriptive_support,
                "directions": direction_summaries,
                "bidirectional_paired_effect_mean": paired_effect,
                "bidirectional_mismatched_effect_mean": mismatch_effect,
                "bidirectional_specific_effect_mean": (
                    paired_effect - mismatch_effect),
            })
            vectors.append({
                "candidate_index": int(site["candidate_index"]),
                "group_kind": kind,
                "directions": direction_vectors,
            })
            if progress is not None:
                progress(
                    "stage=operator_group_patch phase=discovery "
                    f"example_count={len(examples)} "
                    f"candidate_count={site_index + 1}/{len(sites)} "
                    "intervention_variants=5 compile=reuse elapsed_s="
                    f"{time.monotonic() - started:.1f} "
                    "capture_width=full_active_group retry_count=0")
    return {
        "status": "ready",
        "phase": "discovery",
        "selection_record_hash": followup["selection_record_hash"],
        "site_count": len(sites),
        "group_kinds": list(kinds),
        "fixed_group_slots": group_width,
        "individual_operator_forward_count": 0,
        "candidate_specific_jit_compilation": False,
        "params_replicated_per_variant": False,
        "canonical_group_suppression_transplant_kernel": True,
        "forward_call_count": forward_call_count,
        "mismatch_mapping_hash": mismatch["mapping_hash"],
        "mismatch_group_policy": "dual_control",
        "primary_mismatch_control": (
            "donor_own_group_mismatched_program"),
        "secondary_mismatch_control": "same_group_mismatched_state",
        "common_group_semantics": (
            "common_id_realized_contribution_swap_transplants_the_realized_"
            "group_contribution_not_coefficients_alone"),
        "site_summaries": summaries,
        "_vectors": vectors,
    }


def freeze_chronological_path(
        coarse: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *,
        config: ProtocolConfig) -> dict[str, Any]:
    eligible_rows = [
        row for row in coarse["site_summaries"]
        if _route_row_passes_per_direction(row)
    ]
    ranked = sorted(eligible_rows, key=lambda row: (
        -float(row["bidirectional_specific_effect_mean"]),
        -float(row["bidirectional_paired_effect_mean"]),
        int(row["candidate_index"]),
    ))
    chosen = ranked[:config.trajectory_max_path_sites]
    sites = []
    for row in chosen:
        candidate_index = int(row["candidate_index"])
        candidate = dict(candidates[candidate_index])
        sites.append({
            "candidate_index": candidate_index,
            "layer": int(candidate["layer"]),
            "semantic_role": str(candidate["semantic_role"]),
            "route": str(candidate["route"]),
            "stage": str(candidate["route"]),
            "discovery_paired_effect": float(
                row["bidirectional_paired_effect_mean"]),
            "discovery_specific_effect": float(
                row["bidirectional_specific_effect_mean"]),
        })
    sites.sort(key=lambda row: (
        int(row["layer"]),
        ROUTE_INDEX[row["route"]],
        _semantic_role_order(row["semantic_role"]),
    ))
    record = {
        "status": "ready" if sites else "no_causal_path",
        "algorithm_version": PAIRED_TRAJECTORY_ALGORITHM_VERSION,
        "selection_phase": "discovery",
        "selection_metric": (
            "positive_per_direction_paired_and_specific_effect_then_"
            "chronological_order"),
        "per_direction_gate_required": True,
        "validation_results_used": False,
        "test_results_used": False,
        "path_length": len(sites),
        "causal_path_supported": bool(sites),
        "validation_path_evaluated": False,
        "max_path_length": config.trajectory_max_path_sites,
        "sites": sites,
    }
    return {**record, "path_record_hash": canonical_hash(record),
            "frozen": True}


def _path_intervention_rows(
        examples: Sequence[BenchmarkExample], *, direction: str,
        global_indices: np.ndarray,
        sequence_length: int, pad_token_id: int, data_multiple: int,
        patch_slots: int, d_model: int,
        path_prefixes: Sequence[Sequence[Mapping[str, Any]]],
        site_values: Mapping[str, Any], donor_indices: np.ndarray
) -> dict[str, Any]:
    donor_side, recipient_side = (
        ("base", "source") if direction == "source_to_base"
        else ("source", "base"))
    variant_names = ("paired", "mismatched", "self", "disabled")
    rows = []
    if np.asarray(global_indices).shape != (len(examples),):
        raise ValueError("path global example indices are misaligned")
    if not path_prefixes:
        raise ValueError("path prefix batch is empty")
    if any(not prefix or len(prefix) > patch_slots
           for prefix in path_prefixes):
        raise ValueError("path prefix exceeds or omits fixed patch slots")
    for task_index, _ in enumerate(path_prefixes):
        for local_index, example in enumerate(examples):
            example_index = int(global_indices[local_index])
            if direction == "source_to_base":
                prompt = example.input_ids_source
                answers = (example.positive_ids, example.negative_ids)
                target_position = example.trace_position_source
            else:
                prompt = example.input_ids_base
                answers = (example.negative_ids, example.positive_ids)
                target_position = example.trace_position_base
            for variant_index, variant in enumerate(variant_names):
                for answer_index, answer in enumerate(answers):
                    rows.append({
                        "task_index": task_index,
                        "example_index": example_index,
                        "variant_index": variant_index,
                        "answer_index": answer_index,
                        "tokens": tuple(prompt) + tuple(answer),
                        "prompt_length": len(prompt),
                        "target_position": int(target_position),
                        "variant": variant,
                    })
    real_count = len(rows)
    batch_size = _round_up(real_count, data_multiple)
    input_ids = np.full(
        (batch_size, sequence_length), int(pad_token_id), dtype=np.int32)
    labels = np.full(
        (batch_size, sequence_length), -100, dtype=np.int32)
    target_positions = np.zeros((batch_size,), dtype=np.int32)
    patch_shape = (batch_size, patch_slots)
    patch_layers = np.zeros(patch_shape, dtype=np.int32)
    patch_positions = np.zeros(patch_shape, dtype=np.int32)
    patch_stages = np.zeros(patch_shape, dtype=np.int32)
    patch_enabled = np.zeros(patch_shape, dtype=np.bool_)
    patch_values = np.zeros(
        patch_shape + (d_model,), dtype=np.float32)
    for row_index in range(batch_size):
        row = rows[row_index if row_index < real_count else 0]
        example_index = int(row["example_index"])
        tokens = np.asarray(row["tokens"], dtype=np.int32)
        prompt_length = int(row["prompt_length"])
        input_ids[row_index, :len(tokens)] = tokens
        labels[row_index, prompt_length:len(tokens)] = tokens[prompt_length:]
        target_positions[row_index] = int(row["target_position"])
        path_sites = path_prefixes[int(row["task_index"])]
        for slot, site in enumerate(path_sites):
            candidate_index = int(site["candidate_index"])
            patch_layers[row_index, slot] = int(site["layer"])
            patch_positions[row_index, slot] = int(np.asarray(
                site_values[recipient_side]["position"]
            )[example_index, candidate_index])
            patch_stages[row_index, slot] = PATCH_STAGE[str(site["route"])]
            variant = str(row["variant"])
            if variant == "paired":
                value_index = example_index
                value_side = donor_side
            elif variant == "mismatched":
                value_index = int(donor_indices[example_index])
                value_side = donor_side
            elif variant == "self":
                value_index = example_index
                value_side = recipient_side
            else:
                value_index = example_index
                value_side = recipient_side
            patch_enabled[row_index, slot] = variant != "disabled"
            if variant != "disabled":
                patch_values[row_index, slot] = np.asarray(
                    site_values[value_side]["route"]
                )[value_index, candidate_index]
    return {
        "input_ids": input_ids, "labels": labels,
        "target_positions": target_positions,
        "patch_layers": patch_layers,
        "patch_positions": patch_positions,
        "patch_stages": patch_stages,
        "patch_enabled": patch_enabled,
        "patch_values": patch_values,
        "real_count": real_count,
        "task_count": len(path_prefixes),
    }


def _evaluate_cumulative_path_once(
        ctx: Any, examples: Sequence[BenchmarkExample],
        path_record: Mapping[str, Any], site_values: Mapping[str, Any],
        mismatch: Mapping[str, Any], *, production_atlas: Mapping[str, Any],
        pad_token_id: int, config: ProtocolConfig, phase: str,
        evaluate_prefix_curve: bool, prefix_batch_size: int,
        progress: Any | None = None
) -> dict[str, Any]:
    sites = list(path_record["sites"])
    if not sites:
        return {
            "status": "no_causal_path", "phase": phase,
            "path_record_hash": path_record["path_record_hash"],
            "path_length": 0,
            "path_evaluated": False,
            "validation_path_evaluated": False,
            "causal_path_supported": False,
            "prefixes": [], "_vectors": [], "forward_call_count": 0,
        }
    executable = _trajectory_patch_executable(ctx)
    data_multiple = max(1, int(ctx.mesh.shape["data"]))
    examples_per_batch = max(1, data_multiple)
    donor_indices = np.asarray(mismatch["donor_indices"], dtype=np.int32)
    before_by_direction = {
        "source_to_base": -np.asarray(
            production_atlas["_source_margin"], dtype=np.float64),
        "base_to_source": -np.asarray(
            production_atlas["_base_margin"], dtype=np.float64),
    }
    lengths = list(
        range(1, len(sites) + 1) if evaluate_prefix_curve
        else (len(sites),))
    prefix_batch_size = (
        min(max(1, int(prefix_batch_size)), len(lengths))
        if evaluate_prefix_curve else 1)
    prefixes = []
    vector_records = []
    forward_call_count = 0
    started = time.monotonic()
    for prefix_start in range(0, len(lengths), prefix_batch_size):
        active_lengths = lengths[
            prefix_start:prefix_start + prefix_batch_size]
        execution_lengths = list(active_lengths)
        execution_lengths.extend(
            [active_lengths[-1]]
            * (prefix_batch_size - len(execution_lengths)))
        path_prefixes = [sites[:path_length]
                         for path_length in execution_lengths]
        summaries_by_task = [dict() for _ in active_lengths]
        vectors_by_task = [dict() for _ in active_lengths]
        for direction in ("source_to_base", "base_to_source"):
            margins_by_task = [[] for _ in active_lengths]
            residual_by_task = [[] for _ in active_lengths]
            for batch_index, start in enumerate(
                    range(0, len(examples), examples_per_batch)):
                stop = min(len(examples), start + examples_per_batch)
                chunk = list(examples[start:stop])
                global_indices = np.arange(start, stop, dtype=np.int32)
                execution_chunk = _pad_example_batch(
                    chunk, examples_per_batch)
                execution_indices = np.pad(
                    global_indices,
                    (0, examples_per_batch - len(global_indices)),
                    mode="edge")
                rows = _path_intervention_rows(
                    execution_chunk, direction=direction,
                    global_indices=execution_indices,
                    sequence_length=int(site_values["sequence_length"]),
                    pad_token_id=pad_token_id,
                    data_multiple=data_multiple,
                    patch_slots=(
                        config.trajectory_max_patch_sites_per_variant),
                    d_model=int(ctx.model_cfg["d_model"]),
                    path_prefixes=path_prefixes,
                    site_values=site_values,
                    donor_indices=donor_indices)
                score, residual = _execute_patch_rows(ctx, executable, rows)
                forward_call_count += 1
                score = score.reshape(
                    prefix_batch_size, examples_per_batch, 4, 2)
                residual = residual.reshape(
                    prefix_batch_size, examples_per_batch, 4, 2, -1)
                for local_task in range(len(active_lengths)):
                    margins_by_task[local_task].append(
                        score[local_task, :len(chunk), :, 0]
                        - score[local_task, :len(chunk), :, 1])
                    residual_by_task[local_task].append(
                        residual[local_task, :len(chunk), :, 0])
                if progress is not None:
                    progress(
                        f"stage=cumulative_path phase={phase} "
                        f"batch_index={batch_index} "
                        f"example_count={len(chunk)} "
                        f"candidate_count={active_lengths[-1]} "
                        f"prefixes_fused={len(active_lengths)} "
                        "intervention_variants=4 compile=reuse elapsed_s="
                        f"{time.monotonic() - started:.1f} "
                        "capture_width=none retry_count=0")
                del execution_chunk, execution_indices, rows
            for local_task in range(len(active_lengths)):
                margins = np.concatenate(
                    margins_by_task[local_task], axis=0)
                residual = np.concatenate(
                    residual_by_task[local_task], axis=0)
                summary, vectors = _summarize_patch_direction(
                    margins, residual, before=before_by_direction[direction],
                    self_atol=config.trajectory_replay_atol)
                summaries_by_task[local_task][direction] = summary
                vectors_by_task[local_task][direction] = vectors
        for local_task, path_length in enumerate(active_lengths):
            direction_summaries = summaries_by_task[local_task]
            prefixes.append({
                "path_length": int(path_length),
                "directions": direction_summaries,
                "bidirectional_paired_effect_mean": float(np.mean([
                    value["paired_margin_shift_mean"]
                    for value in direction_summaries.values()])),
                "bidirectional_mismatched_effect_mean": float(np.mean([
                    value["mismatched_margin_shift_mean"]
                    for value in direction_summaries.values()])),
                "bidirectional_flip_fraction": float(np.mean([
                    value["paired_answer_flip_fraction"]
                    for value in direction_summaries.values()])),
            })
            vector_records.append({
                "path_length": int(path_length),
                "directions": vectors_by_task[local_task],
            })
    result = {
        "status": "ready", "phase": phase,
        "path_record_hash": path_record["path_record_hash"],
        "forward_call_count": forward_call_count,
        "last_example_chunk_padded_to_fixed_shape": True,
        "prefix_batch_fusion_enabled": bool(evaluate_prefix_curve),
        "prefixes_fused_per_forward": int(prefix_batch_size),
        "prefixes": prefixes,
        "_vectors": vector_records,
    }
    if phase == "validation":
        result["final_frozen_path_uncertainty"] = (
            summarize_frozen_path_uncertainty(
                vector_records[-1], config=config,
                seed=config.trajectory_seed + 211))
    return result


def evaluate_cumulative_path(
        ctx: Any, examples: Sequence[BenchmarkExample],
        path_record: Mapping[str, Any], site_values: Mapping[str, Any],
        mismatch: Mapping[str, Any], *, production_atlas: Mapping[str, Any],
        pad_token_id: int, config: ProtocolConfig, phase: str,
        evaluate_prefix_curve: bool, progress: Any | None = None
) -> dict[str, Any]:
    sites = list(path_record["sites"])
    if not sites:
        return {
            "status": "no_causal_path", "phase": phase,
            "path_record_hash": path_record["path_record_hash"],
            "path_length": 0,
            "path_evaluated": False,
            "validation_path_evaluated": False,
            "causal_path_supported": False,
            "prefixes": [], "_vectors": [], "forward_call_count": 0,
            "initial_prefix_batch_size": 0,
            "effective_prefix_batch_size": 0,
            "resource_retry_count": 0,
            "resource_retries": [],
        }
    initial = (
        min(int(config.trajectory_path_prefix_batch_size), len(sites))
        if evaluate_prefix_curve else 1)
    effective = max(1, initial)
    retries = []
    while True:
        try:
            result = _evaluate_cumulative_path_once(
                ctx, examples, path_record, site_values, mismatch,
                production_atlas=production_atlas,
                pad_token_id=pad_token_id, config=config, phase=phase,
                evaluate_prefix_curve=evaluate_prefix_curve,
                prefix_batch_size=effective, progress=progress)
            result["initial_prefix_batch_size"] = initial
            result["effective_prefix_batch_size"] = effective
            result["resource_retry_count"] = len(retries)
            result["resource_retries"] = retries
            result["path_length"] = len(sites)
            result["path_evaluated"] = True
            result["validation_path_evaluated"] = phase == "validation"
            result["causal_path_supported"] = True
            return result
        except Exception as exc:
            message = str(exc).lower()
            exhausted = isinstance(exc, MemoryError) or any(
                value in message for value in (
                    "resource_exhausted", "out of memory", "oom",
                    "allocation failed"))
            if not exhausted or effective <= 1:
                raise
            updated = max(1, effective // 2)
            retries.append({
                "retry_index": len(retries) + 1,
                "previous_prefix_batch_size": effective,
                "updated_prefix_batch_size": updated,
                "failed_stage": f"cumulative_path_{phase}",
                "scientific_path_changed": False,
            })
            gc.collect()
            effective = updated


def summarize_frozen_path_uncertainty(
        vector_record: Mapping[str, Any], *, config: ProtocolConfig,
        seed: int) -> dict[str, Any]:
    """Evaluate the discovery-frozen final path without reselection."""
    directions = vector_record.get("directions") or {}
    required = ("source_to_base", "base_to_source")
    if any(direction not in directions for direction in required):
        raise ValueError(
            "frozen path uncertainty requires both intervention directions")
    paired_by_direction = []
    mismatch_by_direction = []
    for direction in required:
        values = directions[direction]
        paired = np.asarray(values["paired_effect"], dtype=np.float64)
        mismatched = np.asarray(
            values["mismatched_effect"], dtype=np.float64)
        if paired.ndim != 1 or paired.shape != mismatched.shape:
            raise ValueError(
                "frozen path effect vectors must be aligned one-dimensional "
                "arrays")
        paired_by_direction.append(paired)
        mismatch_by_direction.append(mismatched)
    if paired_by_direction[0].shape != paired_by_direction[1].shape:
        raise ValueError(
            "frozen path intervention directions have different cohorts")
    paired_effect = np.mean(np.stack(paired_by_direction, axis=0), axis=0)
    mismatched_effect = np.mean(
        np.stack(mismatch_by_direction, axis=0), axis=0)
    specific_effect = paired_effect - mismatched_effect
    paired_ci = bootstrap_mean_ci(
        paired_effect, samples=config.bootstrap_samples,
        alpha=config.alpha, seed=seed)
    specific_ci = bootstrap_mean_ci(
        specific_effect, samples=config.bootstrap_samples,
        alpha=config.alpha, seed=seed + 1)
    paired_permutation = paired_permutation_test(
        paired_effect, np.zeros_like(paired_effect),
        samples=config.permutation_samples, seed=seed + 2)
    specific_permutation = paired_permutation_test(
        paired_effect, mismatched_effect,
        samples=config.permutation_samples, seed=seed + 3)
    paired_supported = bool(
        paired_ci["ci_low"] is not None
        and float(paired_ci["ci_low"]) > 0.0
        and float(paired_permutation["p_value_two_sided"]) < config.alpha)
    specific_supported = bool(
        specific_ci["ci_low"] is not None
        and float(specific_ci["ci_low"]) > 0.0
        and float(specific_permutation["p_value_two_sided"]) < config.alpha)
    return {
        "evaluation_scope": (
            "discovery_frozen_final_path_on_preregistered_validation_split"),
        "path_length": int(vector_record["path_length"]),
        "bidirectional_effect_aggregation": (
            "per_example_mean_of_source_to_base_and_base_to_source"),
        "paired_effect_ci": paired_ci,
        "paired_minus_mismatched_effect_ci": specific_ci,
        "paired_effect_paired_permutation": paired_permutation,
        "paired_minus_mismatched_paired_permutation": specific_permutation,
        "alpha": float(config.alpha),
        "paired_effect_supported": paired_supported,
        "pair_specific_effect_supported": specific_supported,
        "causal_pair_specific_validation_passed": bool(
            paired_supported and specific_supported),
        "validation_used_for_path_selection": False,
        "test_used": False,
    }


def divergence_extrema(atlas: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in atlas["site_rows"]]
    if not rows:
        return {}

    def compact(row: Mapping[str, Any], field: str) -> dict[str, Any]:
        return {
            "layer": int(row["layer"]),
            "semantic_role": str(row["semantic_role"]),
            "route": str(row["route"]),
            "position_offset_from_s2": float(
                row["position_offset_from_s2_mean"]),
            "effect_size": float(row[field]),
            "metric": field,
        }

    chronological = sorted(rows, key=lambda row: (
        int(row["layer"]),
        _semantic_role_order(str(row["semantic_role"])),
        ROUTE_INDEX[str(row["route"])],
    ))
    first = next(
        (row for row in chronological
         if float(row["nontrivial_fraction"]) > 0.0),
        chronological[0])
    fields = {
        "largest_block_residual_divergence_growth_context": (
            "block_residual_divergence_growth_context_mean"),
        "largest_query_displacement": "query_angular_displacement_mean",
        "largest_support_turnover": "support_turnover_mean",
        "largest_route_difference": "route_output_difference_norm_mean",
    }
    output = {
        "first_divergence": compact(first, "nontrivial_fraction"),
    }
    for name, field in fields.items():
        output[name] = compact(max(
            rows, key=lambda row: (
                float(row[field]), -int(row["layer"]),
                -_semantic_role_order(str(row["semantic_role"]))
            )), field)
    answer_rows = [
        row for row in rows
        if row["semantic_role"] == "answer_position"]
    if answer_rows:
        output["answer_position_largest_route_difference"] = compact(
            max(answer_rows, key=lambda row: float(
                row["route_output_difference_norm_mean"])),
            "route_output_difference_norm_mean")
    return output


def write_causal_vector_artifact(
        store: Any, relative_name: str, result: Mapping[str, Any], *,
        protocol_hash: str) -> dict[str, Any] | None:
    if not store.is_primary:
        return None
    records = list(result.get("_vectors") or ())
    arrays: dict[str, Any] = {
        "protocol_hash": np.asarray(protocol_hash),
        "algorithm_version": np.asarray(
            PAIRED_TRAJECTORY_ALGORITHM_VERSION),
    }
    metadata = []
    for record_index, record in enumerate(records):
        metadata.append({
            str(key): value for key, value in record.items()
            if key != "directions"
        })
        for direction, values in record.get("directions", {}).items():
            for metric, value in values.items():
                key = f"r{record_index:03d}_{direction}_{metric}"
                arrays[key] = np.asarray(value)
    arrays["record_metadata_json"] = np.asarray(json.dumps(
        metadata, sort_keys=True, ensure_ascii=False))
    path = store.path("trajectory", "causal", relative_name)
    write_npz_atomic(path, **arrays)
    return _artifact_record(
        path, record_count=len(records),
        vector_array_count=len(arrays) - 3,
        raw_vectors_embedded_in_item_json=False)


def write_trajectory_graph(
        store: Any, candidates: Sequence[Mapping[str, Any]],
        path_record: Mapping[str, Any],
        operator_followup: Mapping[str, Any], *,
        protocol_hash: str) -> dict[str, Any] | None:
    if not store.is_primary:
        return None
    path_indices = {
        int(row["candidate_index"]): index
        for index, row in enumerate(path_record.get("sites") or ())
    }
    followup_indices = {
        int(row["candidate_index"])
        for row in operator_followup.get("sites") or ()
    }
    nodes = []
    for local_index, candidate in enumerate(candidates):
        candidate_index = int(candidate.get("candidate_index", local_index))
        nodes.append({
            "id": f"site_{candidate_index:03d}",
            "candidate_index": candidate_index,
            "layer": int(candidate["layer"]),
            "semantic_role": str(candidate["semantic_role"]),
            "route": str(candidate["route"]),
            "selection_rule": str(candidate["selection_rule"]),
            "in_frozen_path": candidate_index in path_indices,
            "path_order": path_indices.get(candidate_index),
            "operator_followup": candidate_index in followup_indices,
        })
    edges = []
    path_sites = list(path_record.get("sites") or ())
    for source, target in zip(path_sites, path_sites[1:]):
        edges.append({
            "source": f"site_{int(source['candidate_index']):03d}",
            "target": f"site_{int(target['candidate_index']):03d}",
            "kind": "chronological_live_state_path",
        })
    graph = {
        "algorithm_version": PAIRED_TRAJECTORY_ALGORITHM_VERSION,
        "protocol_hash": protocol_hash,
        "graph_semantics": (
            "causal_site_trajectory_not_token_to_token_attention_flow"),
        "attention_edge_weights_captured": False,
        "selection_phase": "discovery",
        "validation_used_for_graph_selection": False,
        "test_used": False,
        "candidate_selection_hash": canonical_hash(list(candidates)),
        "path_record_hash": path_record.get("path_record_hash"),
        "operator_followup_selection_hash": operator_followup.get(
            "selection_record_hash"),
        "nodes": nodes,
        "edges": edges,
    }
    encoded = (json.dumps(
        graph, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8")
    limit = 2 * 1024 * 1024
    threshold_exceeded = len(encoded) > limit
    if threshold_exceeded:
        warnings.warn(
            "trajectory graph exceeds the advisory compact-artifact "
            f"threshold: bytes={len(encoded)} warning_threshold={limit}; "
            "writing the graph so a completed analysis is not discarded",
            RuntimeWarning,
            stacklevel=2,
        )
    path = store.path("trajectory", "graph", "trajectory_graph.json")
    write_json_atomic(path, graph)
    return _artifact_record(
        path, encoded_bytes=len(encoded), node_count=len(nodes),
        edge_count=len(edges), warning_threshold_bytes=limit,
        advisory_threshold_exceeded=threshold_exceeded,
        write_continued=True, artifact_discarded=False)


def write_trajectory_manifest(
        store: Any, *, protocol_hash: str,
        deep_trace: Mapping[str, Any], artifacts: Mapping[str, Any],
        trace_output_bytes: int, replay_trace_output_bytes: int,
        peak_materialized_output_bytes: int,
        operator_provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    if not store.is_primary:
        return None
    manifest = {
        "status": "complete",
        "algorithm_version": PAIRED_TRAJECTORY_ALGORITHM_VERSION,
        "protocol_hash": protocol_hash,
        "format": "binary_shards_plus_compact_json_index",
        "deep_trace_shards": list(deep_trace.get("shards") or ()),
        "artifacts": dict(artifacts),
        "trace_output_bytes_per_materialized_deep_batch": int(
            trace_output_bytes),
        "compact_replay_output_bytes_per_deep_batch": int(
            replay_trace_output_bytes),
        "estimated_peak_trace_outputs_bytes": int(
            trace_output_bytes + replay_trace_output_bytes),
        "estimated_peak_materialized_output_bytes": int(
            peak_materialized_output_bytes),
        "operator_parameter_provenance": dict(operator_provenance),
        "dtype_schema": {
            "operator_id": "int32",
            "row_ptr": "int64",
            "exact_bfloat16_scalars": "uint16_bits",
            "execution_weight_denominator_tau": "float32",
            "admission_margin_rho": "float16_descriptive",
            "query_and_state_snapshots": "float16_persisted_float32_metrics",
            "aggregate_statistics": "float64_host",
        },
        "padding_policy": "removed_before_sparse_persistence",
        "raw_arrays_in_item_json": False,
        "authoritative_replay_uses_serialized_quantized_values": False,
        "streaming_contract": (
            "one_example_side_compacted_written_hashed_then_released"),
    }
    path = store.path("trajectory", "manifest.json")
    write_json_atomic(path, manifest)
    return _artifact_record(
        path, shard_count=len(manifest["deep_trace_shards"]),
        artifact_count=len(manifest["artifacts"]))
