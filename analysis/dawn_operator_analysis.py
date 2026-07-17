"""Actual checkpoint evaluation for prepared v417x operator datasets.

The prepared-data contract lives in :mod:`analysis.dawn_operator_datasets`.
This module performs production model forwards, target-only sparse traces,
route comparisons, and production-core execution-suppression interventions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import (
    AnalysisContext,
    analysis_model_module,
    sync_hosts,
)
from analysis.dawn_analysis_storage import (
    exists,
    join_path,
    list_paths,
    makedirs,
    read_json,
    read_npz,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_npz_atomic,
    write_text_atomic,
)
from analysis.dawn_analysis_trace import (
    TRACE_POOLS,
    TRANSITION_CANDIDATE_STRATEGIES,
    topk_trace_forward,
)
from analysis.dawn_operator_datasets import (
    DATASET_IDS,
    DEFAULT_OPERATOR_CACHE_DIR,
    OperatorDatasetBuild,
    canonical_hash,
    iter_operator_rows,
    load_subset,
    resolve_operator_dataset_build,
    validate_operator_dataset_build,
)


ANALYSIS_VERSION = "dawn_operator_analysis_v2"
INTERVENTION_TYPE = "production_core_execution_suppression"
CAPTURE_THRESHOLD = 0.95
DATASET_ITEM = {
    "ioi": "ioi_operator_circuit",
    "blimp": "blimp_operator_grammar",
    "ravel": "ravel_operator_disentanglement",
    "synthetic": "synthetic_binding_sanity",
    "lama": "lama_counterfact_factual_recall",
    "counterfact": "lama_counterfact_factual_recall",
}
OPERATOR_ITEMS = {
    "operator_dataset_manifest", "operator_behavior_eval",
    "ravel_operator_disentanglement", "ioi_operator_circuit",
    "blimp_operator_grammar", "lama_counterfact_factual_recall",
    "synthetic_binding_sanity", "operator_function_reuse",
    "operator_route_specificity", "operator_causal_specificity",
    "operator_analysis_summary",
}


class CausalParityError(RuntimeError):
    """Production-core suppression invariants failed; evidence is invalid."""


def _safe_id(value: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def _analysis_job_key(
    config_hash: str, dataset_id: str, stage: str, example_id: str,
) -> str:
    """Stable completed-job key; the global config already pins checkpoint/runtime."""
    return canonical_hash({
        "config_hash": config_hash,
        "dataset": dataset_id,
        "stage": stage,
        "example_id": example_id,
    })


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    rows = _finite(values)
    return float(np.mean(rows)) if rows else None


def _cosine(left: Any, right: Any) -> Optional[float]:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size:
        return None
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1.0e-12 else None


def _bootstrap_ci(values: Sequence[float], seed: int, samples: int = 1000) -> List[Optional[float]]:
    data = np.asarray(_finite(values), dtype=np.float64)
    if not data.size:
        return [None, None]
    rng = np.random.default_rng(seed)
    if data.size > 4096:
        data = rng.choice(data, size=4096, replace=False)
    estimates = np.empty((samples,), dtype=np.float64)
    for index in range(samples):
        estimates[index] = float(np.mean(rng.choice(data, size=data.size, replace=True)))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _profile_config(args: Any) -> Dict[str, Any]:
    profile = str(getattr(args, "operator_analysis_profile", "monitor") or "monitor")
    if profile not in {"smoke", "monitor", "full"}:
        raise ValueError(f"Unknown operator analysis profile: {profile}")
    defaults = {
        "smoke": {"behavior_subset": "smoke", "trace_subset": "smoke", "causal_subset": "smoke", "trace_max": 8, "causal_max": 2},
        "monitor": {"behavior_subset": "monitor", "trace_subset": "trace", "causal_subset": "causal", "trace_max": 64, "causal_max": 24},
        "full": {"behavior_subset": None, "trace_subset": "trace", "causal_subset": "causal", "trace_max": 128, "causal_max": 48},
    }[profile]
    out = {"profile": profile, **defaults}
    for target, attr in (
        ("behavior_max", "operator_behavior_max_examples"),
        ("trace_max", "operator_trace_max_examples"),
        ("causal_max", "operator_causal_max_examples"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            out[target] = max(0, int(value))
    out["trace_per_group"] = getattr(args, "operator_trace_per_group", None)
    out["causal_per_group"] = getattr(args, "operator_causal_per_group", None)
    return out


def _selected_datasets(value: Any) -> List[str]:
    raw = [part.strip().lower() for part in str(value or "all").split(",") if part.strip()]
    if not raw or raw == ["all"]:
        return list(DATASET_IDS)
    unknown = sorted(set(raw) - set(DATASET_IDS))
    if unknown:
        raise ValueError(f"Unknown operator datasets: {unknown}")
    return list(dict.fromkeys(raw))


def _runtime_kwargs(ctx: AnalysisContext, sharded_fns: Any) -> Dict[str, Any]:
    cfg = ctx.model_cfg
    return {
        "attention_mask": None,
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(0)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "minimal_train": True,
        "soft_gate_temperature": float(cfg["soft_gate_temperature"]),
        "soft_gate_t_final": float(cfg.get("soft_gate_t_final", cfg["soft_gate_temperature"])),
        "soft_gate_T_qk": float(cfg.get("soft_gate_T_qk", cfg["soft_gate_temperature"])),
        "soft_gate_T_v": float(cfg.get("soft_gate_T_v", cfg["soft_gate_temperature"])),
        "soft_gate_T_rst": float(cfg.get("soft_gate_T_rst", cfg["soft_gate_temperature"])),
        "soft_gate_boundary_power": float(cfg["soft_gate_boundary_power"]),
        "soft_gate_boundary_power_final": float(cfg.get("soft_gate_boundary_power_final", cfg["soft_gate_boundary_power"])),
        "admission_den_power": float(cfg["admission_den_power"]),
        "srw_composition_mode": str(cfg["srw_composition_mode"]),
        "heat_kernel_beta": float(cfg["heat_kernel_beta"]),
        "execution_prune_eps": jnp.float32(
            float(cfg.get("execution_prune_eps", 0.0) or 0.0)),
        "compute_accuracy": False,
    }


def _make_behavior_score_step(ctx: AnalysisContext):
    kwargs = _runtime_kwargs(ctx, ctx.sharded_fns)

    @jax.jit
    def score(params, input_ids, labels):
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            **{**kwargs, "attention_mask": jnp.ones_like(input_ids)})
        valid = result["valid_mask"].astype(jnp.float32)
        token_score = -result["per_token_ce"] * valid
        count = jnp.maximum(valid.sum(axis=-1), 1.0)
        return token_score.sum(axis=-1), token_score.sum(axis=-1) / count

    return score


def _pad_sequence(ids: Sequence[int], labels: Sequence[int], seq_len: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(ids) > seq_len:
        raise ValueError(f"Teacher-forced sequence exceeds model length: {len(ids)}>{seq_len}")
    x = np.full((seq_len,), pad_id, dtype=np.int32)
    y = np.full((seq_len,), -100, dtype=np.int32)
    x[:len(ids)] = np.asarray(ids, dtype=np.int32)
    y[:len(labels)] = np.asarray(labels, dtype=np.int32)
    return x, y


def _row_array(row: Mapping[str, Any], name: str) -> List[int]:
    arrays = row["arrays"]
    if name.startswith("context_ids_"):
        suffix = name[-1]
        length = int(arrays[f"context_len_{suffix}"])
    else:
        prefix, _, suffix = name.partition("_ids_")
        length = int(arrays[f"{prefix}_len_{suffix}"])
    return [int(value) for value in np.asarray(arrays[name])[:length]]


def _sequence_spec(ids: Sequence[int], *, candidate_start: int, seq_len: int, pad_id: int, tag: str) -> Dict[str, Any]:
    labels = [-100] * len(ids)
    if candidate_start <= 0:
        labels = list(ids)
        if labels:
            labels[0] = -100
    else:
        labels[candidate_start:] = list(ids[candidate_start:])
    x, y = _pad_sequence(ids, labels, seq_len, pad_id)
    return {"tag": tag, "input_ids": x, "labels": y, "length": len(ids)}


def _behavior_specs(row: Mapping[str, Any], seq_len: int, pad_id: int) -> List[Dict[str, Any]]:
    mode = str(row["metadata"]["score_mode"])
    context_a = _row_array(row, "context_ids_a")
    context_b = _row_array(row, "context_ids_b")
    if mode == "paired_sequence_logprob":
        return [
            _sequence_spec(context_a, candidate_start=0, seq_len=seq_len, pad_id=pad_id, tag="sequence_a"),
            _sequence_spec(context_b, candidate_start=0, seq_len=seq_len, pad_id=pad_id, tag="sequence_b"),
        ]
    specs: List[Dict[str, Any]] = []
    for side, context in (("a", context_a), ("b", context_b)):
        if not context:
            continue
        for polarity in ("positive", "negative"):
            candidate = _row_array(row, f"{polarity}_ids_{side}")
            if not candidate:
                continue
            specs.append(_sequence_spec(
                context + candidate, candidate_start=len(context), seq_len=seq_len,
                pad_id=pad_id, tag=f"{polarity}_{side}"))
    return specs


def _behavior_result(metadata: Mapping[str, Any], scores: Mapping[str, Tuple[float, float]]) -> Dict[str, Any]:
    mode = str(metadata["score_mode"])
    row: Dict[str, Any] = {
        key: metadata.get(key) for key in (
            "example_id", "pair_id", "dataset", "split", "phenomenon",
            "relation", "group_id", "score_mode")}
    if mode == "paired_sequence_logprob":
        sum_a, mean_a = scores["sequence_a"]
        sum_b, mean_b = scores["sequence_b"]
        row.update({
            "logprob_a": sum_a, "logprob_b": sum_b,
            "mean_token_logprob_a": mean_a, "mean_token_logprob_b": mean_b,
            "margin": sum_a - sum_b,
            "length_normalized_margin": mean_a - mean_b,
            "correct": sum_a > sum_b,
        })
    else:
        margins: List[float] = []
        for side in ("a", "b"):
            positive = scores.get(f"positive_{side}")
            negative = scores.get(f"negative_{side}")
            if positive is None or negative is None:
                continue
            margin = positive[0] - negative[0]
            row.update({
                f"positive_logprob_{side}": positive[0],
                f"negative_logprob_{side}": negative[0],
                f"margin_{side}": margin,
                f"correct_{side}": margin > 0.0,
            })
            margins.append(margin)
        row["margin"] = float(np.mean(margins)) if margins else None
        row["correct"] = bool(margins and all(value > 0.0 for value in margins))
        if mode == "true_new_object_margin":
            row["known_true"] = bool(margins and margins[0] > 0.0)
            row["true_object_logprob"] = row.get("positive_logprob_a")
            row["new_object_logprob"] = row.get("negative_logprob_a")
            row["true_minus_new_margin"] = row.get("margin_a")
    return row


def _bucket_summary(bucket: Mapping[str, Sequence[Any]], seed: int) -> Dict[str, Any]:
    margins = _finite(bucket.get("margins") or [])
    correct = [bool(value) for value in (bucket.get("correct") or [])]
    return {
        "n": len(correct),
        "accuracy": float(np.mean(correct)) if correct else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "median_margin": float(np.median(margins)) if margins else None,
        "bootstrap_ci95": _bootstrap_ci(margins, seed),
        "known_correct": sum(correct),
    }


def _dataset_behavior_details(dataset_id: str, detail: Mapping[str, Any]) -> Dict[str, Any]:
    if dataset_id == "ioi":
        clean = _finite(detail.get("clean") or [])
        corrupt = _finite(detail.get("corrupt") or [])
        return {
            "clean_accuracy": float(np.mean([value > 0.0 for value in clean])) if clean else None,
            "clean_mean_margin": _mean(clean),
            "corrupt_accuracy": float(np.mean([value > 0.0 for value in corrupt])) if corrupt else None,
            "corrupt_mean_margin": _mean(corrupt),
            "clean_to_corrupt_margin_change": (
                float(np.mean(np.asarray(corrupt) - np.asarray(clean)))
                if clean and len(clean) == len(corrupt) else None),
            "correct_examples": int(detail.get("correct_examples", 0)),
            "incorrect_examples": int(detail.get("incorrect_examples", 0)),
        }
    if dataset_id == "blimp":
        return {
            "headline": "raw_sequence_logprob_preference",
            "phenomena_observed": len(detail.get("phenomena") or []),
            "mean_length_normalized_margin": _mean(detail.get("length_margins") or []),
        }
    if dataset_id in {"lama", "counterfact"}:
        return {
            "known_correct": int(detail.get("known_correct", 0)),
            "unknown_or_rewrite_favored": int(detail.get("unknown", 0)),
            "causal_interpretation_subset": "known_correct",
        }
    if dataset_id == "ravel":
        return {"headline": "attribute_value_margin", "comparison_families": sorted(detail.get("families") or [])}
    if dataset_id == "synthetic":
        return {
            "headline": "controlled_binding_margin",
            "pair_families": sorted(detail.get("families") or []),
            "claims": {
                "binding_invariance": ["same_operation_different_entity", "distractor_swap"],
                "attribute_specificity": ["same_entity_different_attribute", "attribute_swap"],
                "entity_specificity": ["entity_swap"],
                "order_robustness": ["order_permutation"],
                "irrelevant_context_robustness": ["irrelevant_fact_addition"],
            },
        }
    return {}


def _run_behavior(
    ctx: AnalysisContext, build: OperatorDatasetBuild, dataset_id: str,
    profile: Mapping[str, Any], output_root: str, pad_id: int, cache_dir: str,
    *, config_hash: str, resume: bool,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    started = time.time()
    score_step = _make_behavior_score_step(ctx)
    seq_len = int(ctx.model_cfg["max_seq_len"])
    data_replicas = max(1, int(ctx.mesh.shape["data"]))
    logical_batch = max(1, data_replicas)
    rows_iter = iter_operator_rows(
        build, dataset_id, subset=profile.get("behavior_subset"),
        max_examples=profile.get("behavior_max"), cache_dir=cache_dir)
    overall: Dict[str, List[Any]] = {"margins": [], "correct": []}
    groups: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {"margins": [], "correct": []})
    lengths: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {"margins": [], "correct": []})
    detail: Dict[str, Any] = {
        "clean": [], "corrupt": [], "phenomena": set(),
        "length_margins": [], "families": set(),
        "known_correct": 0, "unknown": 0,
        "correct_examples": 0, "incorrect_examples": 0,
    }
    by_example: Dict[str, Dict[str, Any]] = {}
    behavior_needed = {
        str(entry["example_id"])
        for subset_id in (profile["causal_subset"], profile["trace_subset"])
        for entry in load_subset(build, dataset_id, subset_id)
    }
    part_paths: List[str] = []
    part_index = 0
    pending: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []
    job_root = join_path(output_root, "behavior", "jobs")
    makedirs(job_root)
    # Snapshot before any host writes new jobs.  Multi-host workers must make
    # identical execute/reuse decisions or production collectives can diverge.
    sync_hosts(f"operator-behavior-{dataset_id}-resume-snapshot")
    reusable_jobs = set(list_paths(job_root, "*.json")) if resume else set()
    resumed_jobs = 0

    def write_part(force: bool = False) -> None:
        nonlocal part_index
        while len(artifact_rows) >= 1024 or (force and artifact_rows):
            size = min(1024, len(artifact_rows))
            rows = artifact_rows[:size]
            del artifact_rows[:size]
            if ctx.is_primary:
                path = join_path(
                    output_root, "behavior", "parts", f"part-{part_index:05d}.jsonl")
                write_jsonl_atomic(path, rows)
                part_paths.append(path)
            part_index += 1

    def consume_result(result: Dict[str, Any]) -> None:
        margin = result.get("margin")
        if margin is not None:
            overall["margins"].append(float(margin))
            groups[str(result.get("group_id"))]["margins"].append(float(margin))
            lengths[str(result["length_group"])]["margins"].append(float(margin))
        correct = bool(result.get("correct"))
        overall["correct"].append(correct)
        groups[str(result.get("group_id"))]["correct"].append(correct)
        lengths[str(result["length_group"])]["correct"].append(correct)
        example_id = str(result["example_id"])
        if example_id in behavior_needed:
            by_example[example_id] = result
        detail["families"].add(str(result.get("phenomenon")))
        if dataset_id == "ioi":
            if result.get("margin_a") is not None:
                detail["clean"].append(float(result["margin_a"]))
            if result.get("margin_b") is not None:
                detail["corrupt"].append(float(result["margin_b"]))
            detail["correct_examples" if correct else "incorrect_examples"] += 1
        elif dataset_id == "blimp":
            detail["phenomena"].add(str(result.get("phenomenon")))
            if result.get("length_normalized_margin") is not None:
                detail["length_margins"].append(
                    float(result["length_normalized_margin"]))
        elif dataset_id in {"lama", "counterfact"}:
            detail["known_correct" if correct else "unknown"] += 1
        artifact_rows.append(result)
        write_part()

    def flush(batch: List[Dict[str, Any]]) -> None:
        real_count = len(batch)
        specs_by_row = [_behavior_specs(row, seq_len, pad_id) for row in batch]
        per_row = len(specs_by_row[0])
        if any(len(specs) != per_row for specs in specs_by_row):
            raise ValueError(f"Inconsistent behavior sequence arity in {dataset_id}")
        while len(batch) < logical_batch:
            batch.append(batch[-1])
            specs_by_row.append(specs_by_row[-1])
        specs = [spec for row_specs in specs_by_row for spec in row_specs]
        inputs = np.stack([spec["input_ids"] for spec in specs])
        labels = np.stack([spec["labels"] for spec in specs])
        if inputs.shape[0] % data_replicas:
            raise ValueError("Behavior batch is not divisible by the data mesh")
        score_sum, score_mean = jax.device_get(score_step(
            ctx.params, jax.device_put(jnp.asarray(inputs), ctx.data_sharding),
            jax.device_put(jnp.asarray(labels), ctx.data_sharding)))
        score_sum = np.asarray(score_sum)
        score_mean = np.asarray(score_mean)
        for row_index, row in enumerate(batch[:real_count]):
            offset = row_index * per_row
            scores = {
                spec["tag"]: (float(score_sum[offset + spec_index]),
                              float(score_mean[offset + spec_index]))
                for spec_index, spec in enumerate(specs_by_row[row_index])}
            result = _behavior_result(row["metadata"], scores)
            token_counts = {
                spec["tag"]: int(np.count_nonzero(spec["labels"] != -100))
                for spec in specs_by_row[row_index]
            }
            result["scored_token_counts"] = token_counts
            result["length_group"] = str(max(token_counts.values(), default=0))
            example_id = str(result["example_id"])
            job_key = _analysis_job_key(
                config_hash, dataset_id, "behavior", example_id)
            result["job_key"] = job_key
            consume_result(result)
            if ctx.is_primary:
                write_json_atomic(join_path(job_root, f"{job_key}.json"), {
                    "status": "complete", "stage": "behavior",
                    "dataset": dataset_id, "example_id": example_id,
                    "config_hash": config_hash, "job_key": job_key,
                    "result": result,
                })

    for source_row in rows_iter:
        example_id = str(source_row["metadata"]["example_id"])
        job_key = _analysis_job_key(
            config_hash, dataset_id, "behavior", example_id)
        job_path = join_path(job_root, f"{job_key}.json")
        if job_path in reusable_jobs:
            cached = read_json(job_path, {})
            if (
                cached.get("status") == "complete"
                and cached.get("config_hash") == config_hash
                and cached.get("job_key") == job_key
                and isinstance(cached.get("result"), Mapping)
            ):
                if pending:
                    flush(pending)
                    pending = []
                consume_result(dict(cached["result"]))
                resumed_jobs += 1
                continue
        pending.append(source_row)
        if len(pending) == logical_batch:
            flush(pending)
            pending = []
    if pending:
        flush(pending)
    write_part(force=True)
    summary = {
        "status": "ready" if overall["correct"] else "insufficient_evidence",
        **_bucket_summary(overall, int(getattr(ctx.args, "operator_analysis_seed", 4171))),
        "dataset": dataset_id,
        "groups": {key: _bucket_summary(value, 4171 + index)
                   for index, (key, value) in enumerate(sorted(groups.items()))},
        "length_groups": {key: _bucket_summary(value, 8171 + index)
                          for index, (key, value) in enumerate(sorted(lengths.items()))},
        "parts": part_paths,
        "artifact_part_rows": 1024,
        "completed_job_key_fields": [
            "config_hash", "dataset", "stage", "example_id"],
        "resumed_jobs": resumed_jobs,
        "causal_behavior_lookup_rows": len(by_example),
        "dataset_details": _dataset_behavior_details(dataset_id, detail),
        "sec": time.time() - started,
    }
    if ctx.is_primary:
        write_json_atomic(join_path(output_root, "behavior", "behavior_summary.json"), summary)
    return summary, by_example


def _weighted_jaccard(ids_a: Any, weights_a: Any, ids_b: Any, weights_b: Any) -> Optional[float]:
    left: Dict[int, float] = defaultdict(float)
    right: Dict[int, float] = defaultdict(float)
    for operator_id, weight in zip(np.asarray(ids_a).reshape(-1), np.asarray(weights_a).reshape(-1)):
        left[int(operator_id)] += max(0.0, float(weight))
    for operator_id, weight in zip(np.asarray(ids_b).reshape(-1), np.asarray(weights_b).reshape(-1)):
        right[int(operator_id)] += max(0.0, float(weight))
    keys = set(left) | set(right)
    den = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return (sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys) / den
            if den > 1.0e-12 else None)


def _trace_inputs(row: Mapping[str, Any], seq_len: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ids_a = _row_array(row, "context_ids_a")
    ids_b = _row_array(row, "context_ids_b")
    conditions = ["a", "b"] if ids_b else ["a", "a_duplicate"]
    ids = [ids_a, ids_b if ids_b else ids_a]
    arrays = [_pad_sequence(value, [-100] * len(value), seq_len, pad_id)[0] for value in ids]
    positions = [int(row["arrays"]["trace_position_a"]),
                 int(row["arrays"]["trace_position_b"] if ids_b else row["arrays"]["trace_position_a"])]
    return np.stack(arrays), np.asarray(positions, dtype=np.int32), conditions


def _execute_trace(
    ctx: AnalysisContext, trace_fn: Any, inputs: np.ndarray, positions: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run each logical condition on a full data mesh and keep one replica."""
    data_replicas = max(1, int(ctx.mesh.shape["data"]))
    condition_traces: List[Dict[str, np.ndarray]] = []
    for input_ids, position in zip(inputs, positions):
        batch = np.repeat(input_ids[None, :], data_replicas, axis=0)
        targets = np.full((data_replicas,), int(position), dtype=np.int32)
        traced = jax.device_get(trace_fn(
            ctx.params, jax.device_put(jnp.asarray(batch), ctx.data_sharding),
            jax.device_put(jnp.asarray(targets), ctx.data_sharding)))
        condition_traces.append({key: np.asarray(value) for key, value in traced.items()})
    out: Dict[str, np.ndarray] = {}
    for key in condition_traces[0]:
        values = [trace[key] for trace in condition_traces]
        if values[0].ndim >= 2 and values[0].shape[1] == data_replicas:
            out[key] = np.concatenate([value[:, :1] for value in values], axis=1)
        else:
            out[key] = values[0]
    return out


def _surface_key(metadata: Mapping[str, Any], condition: str) -> Optional[str]:
    extension = metadata.get("metadata") or {}
    dataset = str(metadata.get("dataset"))
    value: Any = None
    if dataset == "ravel":
        value = extension.get(f"entity_{condition}")
        if isinstance(value, Mapping):
            value = value.get("City") or value.get("ID")
    elif dataset == "synthetic":
        value = extension.get(f"entity_{condition}") or extension.get("entity")
    elif dataset in {"lama", "counterfact"}:
        value = metadata.get("subject") or extension.get("subject")
    elif dataset == "ioi":
        value = "|".join(sorted(filter(None, (
            str(metadata.get("name_a") or ""), str(metadata.get("name_b") or "")))))
    elif dataset == "blimp":
        value = extension.get("field") or extension.get("linguistics_term")
    return str(value) if value not in (None, "") else None


def _trace_signature(
    trace: Mapping[str, np.ndarray], condition_index: int,
    metadata: Mapping[str, Any], row: Mapping[str, Any], condition: str,
    behavior_correct: Optional[bool],
) -> Dict[str, Any]:
    example_id = f"{metadata['example_id']}:{condition}"
    signature: Dict[str, Any] = {
        "group": str(metadata["group_id"]), "example_id": example_id,
        "logical_example_id": str(metadata["example_id"]),
        "phenomenon": str(metadata.get("phenomenon")),
        "relation": str(metadata.get("relation")),
        "condition": condition,
        "context_length": int(row["arrays"][f"context_len_{condition}"]),
        "candidate_token_count": max(
            int(row["arrays"][f"positive_len_{condition}"]),
            int(row["arrays"][f"negative_len_{condition}"])),
        "surface_key": _surface_key(metadata, condition),
        "behavior_correct": behavior_correct,
        "pools": {},
    }
    transition_key = {"q": "srw_feature_q", "k": "srw_feature_k", "v": "srw_feature_v", "rst": "delta_rst"}
    state_key = {"q": "residual_before_router", "k": "residual_before_router", "v": "residual_before_router", "rst": "residual_after_attention"}
    query_key = {"q": "query_q", "k": "query_k", "v": "query_v", "rst": "query_rst"}
    for pool in TRACE_POOLS:
        layers = []
        for layer in range(np.asarray(trace[f"{pool}_top_idx"]).shape[0]):
            layers.append({
                "ids": np.asarray(trace[f"{pool}_top_idx"])[layer, condition_index, :64],
                "weights": np.asarray(trace[f"{pool}_top_val"])[layer, condition_index, :64],
                "captured": float(np.asarray(trace[f"{pool}_captured_mass"])[layer, condition_index]),
                "transition": np.asarray(trace[transition_key[pool]])[layer, condition_index],
                "state": np.asarray(trace[state_key[pool]])[layer, condition_index],
                "query": np.asarray(trace[query_key[pool]])[layer, condition_index],
            })
        signature["pools"][pool] = layers
    return signature


def _signature_metrics(left: Mapping[str, Any], right: Mapping[str, Any], pair_type: str, is_null: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    logical_pair_id = "|".join((
        pair_type,
        *sorted((str(left["example_id"]), str(right["example_id"]))),
    ))
    for pool in TRACE_POOLS:
        for layer, (a, b) in enumerate(zip(left["pools"][pool], right["pools"][pool])):
            routing = _weighted_jaccard(a["ids"], a["weights"], b["ids"], b["weights"])
            transition = _cosine(a["transition"], b["transition"])
            state = _cosine(a["state"], b["state"])
            query = _cosine(a["query"], b["query"])
            valid = a["captured"] >= CAPTURE_THRESHOLD and b["captured"] >= CAPTURE_THRESHOLD
            rows.append({
                "logical_pair_id": logical_pair_id,
                "example_a": left["example_id"], "example_b": right["example_id"],
                "logical_example_a": left["logical_example_id"],
                "logical_example_b": right["logical_example_id"],
                "group_a": left["group"], "group_b": right["group"],
                "pair_type": pair_type, "is_random_null": is_null,
                "pool": pool, "layer": layer,
                "state_similarity": state,
                "query_similarity": query,
                "routing_similarity": routing if valid else None,
                "transition_similarity": transition if valid else None,
                "path_similarity": (
                    float(np.mean([routing, transition]))
                    if valid and routing is not None and transition is not None else None),
                "captured_mass_a": a["captured"], "captured_mass_b": b["captured"],
                "context_length_a": left["context_length"],
                "context_length_b": right["context_length"],
                "candidate_token_count_a": left["candidate_token_count"],
                "candidate_token_count_b": right["candidate_token_count"],
                "surface_key_a": left.get("surface_key"),
                "surface_key_b": right.get("surface_key"),
                "behavior_correct_a": left.get("behavior_correct"),
                "behavior_correct_b": right.get("behavior_correct"),
                "metric_valid": bool(valid and routing is not None and transition is not None),
                "invalid_reason": None if valid else "low_captured_mass",
            })
    return rows


def _logical_route_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("metric_valid"):
            grouped[str(row["logical_pair_id"])].append(row)
    output: List[Dict[str, Any]] = []
    for pair_id, values in grouped.items():
        first = values[0]
        output.append({
            "logical_pair_id": pair_id,
            "pair_type": first["pair_type"],
            "is_random_null": bool(first["is_random_null"]),
            "group_a": first["group_a"], "group_b": first["group_b"],
            "behavior_correct_a": first.get("behavior_correct_a"),
            "behavior_correct_b": first.get("behavior_correct_b"),
            "state_similarity": _mean(row.get("state_similarity") for row in values),
            "query_similarity": _mean(row.get("query_similarity") for row in values),
            "routing_similarity": _mean(row.get("routing_similarity") for row in values),
            "transition_similarity": _mean(row.get("transition_similarity") for row in values),
            "path_similarity": _mean(row.get("path_similarity") for row in values),
            "pool_layer_rows": len(values),
        })
    return output


def _effect_bootstrap_ci(
    actual: Sequence[float], null: Sequence[float], seed: int, samples: int = 1000,
) -> List[Optional[float]]:
    left = np.asarray(_finite(actual), dtype=np.float64)
    right = np.asarray(_finite(null), dtype=np.float64)
    if not left.size or not right.size:
        return [None, None]
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples,), dtype=np.float64)
    for index in range(samples):
        estimates[index] = (
            float(np.mean(rng.choice(left, size=left.size, replace=True)))
            - float(np.mean(rng.choice(right, size=right.size, replace=True))))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _enrichment_with_permutation(
    signatures: Sequence[Mapping[str, Any]], seed: int, permutations: int = 128,
) -> Tuple[Dict[str, Any], Dict[str, Counter[Tuple[str, int, int]]]]:
    observed: Dict[str, Counter[Tuple[str, int, int]]] = defaultdict(Counter)
    keys_by_signature: List[List[Tuple[str, int, int]]] = []
    labels: List[str] = []
    for signature in signatures:
        keys: List[Tuple[str, int, int]] = []
        for pool in TRACE_POOLS:
            for layer, row in enumerate(signature["pools"][pool]):
                ids = np.asarray(row["ids"])
                if ids.size:
                    keys.append((pool, layer, int(ids[0])))
        keys_by_signature.append(keys)
        label = str(signature["group"])
        labels.append(label)
        observed[label].update(keys)
    targets = {(group, key) for group, counter in observed.items() for key in counter}
    sums: Counter[Tuple[str, Tuple[str, int, int]]] = Counter()
    squares: Counter[Tuple[str, Tuple[str, int, int]]] = Counter()
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels, dtype=object)
    for _ in range(permutations):
        shuffled = rng.permutation(labels_array)
        perm_counts: Counter[Tuple[str, Tuple[str, int, int]]] = Counter()
        for assigned, keys in zip(shuffled.tolist(), keys_by_signature):
            for key in keys:
                pair = (str(assigned), key)
                if pair in targets:
                    perm_counts[pair] += 1
        for pair in targets:
            value = float(perm_counts[pair])
            sums[pair] += value
            squares[pair] += value * value
    groups: Dict[str, List[Dict[str, Any]]] = {}
    expected_rates: List[float] = []
    for group, counter in sorted(observed.items()):
        entries = []
        group_n = max(1, sum(label == group for label in labels))
        for (pool, layer, operator_id), count in counter.items():
            pair = (group, (pool, layer, operator_id))
            expected = float(sums[pair]) / max(permutations, 1)
            variance = max(
                0.0, float(squares[pair]) / max(permutations, 1) - expected * expected)
            std = math.sqrt(variance)
            expected_rates.append(expected / group_n)
            entries.append({
                "pool": pool, "layer": layer, "operator_id": operator_id,
                "observed_count": int(count), "observed_rate": float(count) / group_n,
                "permutation_mean_count": expected,
                "permutation_std_count": std,
                "enrichment_over_null": float(count) - expected,
                "z_score": (float(count) - expected) / std if std > 1.0e-12 else None,
            })
        groups[group] = sorted(
            entries,
            key=lambda row: (-row["enrichment_over_null"], -row["observed_count"]),
        )[:50]
    return ({
        "status": "ready" if signatures else "insufficient_evidence",
        "permutations": permutations, "seed": seed,
        "permutation_baseline_mean_rate": _mean(expected_rates),
        "groups": groups,
    }, observed)


def _stratified_rows(rows: Iterable[Dict[str, Any]], limit: int, per_group: Optional[int]) -> Iterable[Dict[str, Any]]:
    if int(limit) <= 0:
        return
    counts: Counter[str] = Counter()
    emitted = 0
    for row in rows:
        group = str(row["metadata"].get("group_id"))
        if per_group is not None and counts[group] >= int(per_group):
            continue
        counts[group] += 1
        yield row
        emitted += 1
        if emitted >= limit:
            return


def _run_trace(
    ctx: AnalysisContext, build: OperatorDatasetBuild, dataset_id: str,
    profile: Mapping[str, Any], output_root: str, pad_id: int, cache_dir: str,
    behavior_by_example: Mapping[str, Mapping[str, Any]],
    *, config_hash: str, resume: bool,
) -> Tuple[Dict[str, Any], Dict[str, Counter[Tuple[str, int, int]]]]:
    started = time.time()
    seq_len = int(ctx.model_cfg["max_seq_len"])
    topk_qk = int(getattr(ctx.args, "transition_topk_qk", 512))
    topk_v = int(getattr(ctx.args, "transition_topk_v", 2048))
    topk_rst = int(getattr(ctx.args, "transition_topk_rst", 4096))
    seed = int(getattr(ctx.args, "operator_analysis_seed", 4171))
    trace_fn = jax.jit(lambda p, x, t: topk_trace_forward(
        p, ctx.model_cfg, x, topk_qk=topk_qk, topk_v=topk_v,
        topk_rst=topk_rst, target_positions=t, candidate_seed=seed,
        production_srw_fns=ctx.sharded_fns))
    source = iter_operator_rows(
        build, dataset_id, subset=profile["trace_subset"],
        cache_dir=cache_dir)
    source = _stratified_rows(
        source, int(profile["trace_max"]), profile.get("trace_per_group"))
    signatures: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    captured: List[float] = []
    artifacts: List[str] = []
    job_root = join_path(output_root, "trace", "jobs")
    makedirs(job_root)
    sync_hosts(f"operator-trace-{dataset_id}-resume-snapshot")
    reusable_jobs = set(list_paths(job_root, "*.json")) if resume else set()
    resumed_jobs = 0
    for index, row in enumerate(source):
        metadata = row["metadata"]
        example_id = str(metadata["example_id"])
        job_key = _analysis_job_key(config_hash, dataset_id, "trace", example_id)
        trace_path = join_path(
            output_root, "trace", f"trace-{job_key[:16]}-{_safe_id(example_id)}.npz")
        job_path = join_path(job_root, f"{job_key}.json")
        cached = read_json(job_path, {}) if job_path in reusable_jobs else {}
        if (
            cached.get("status") == "complete"
            and cached.get("config_hash") == config_hash
            and cached.get("job_key") == job_key
            and cached.get("artifact") == trace_path
            and exists(trace_path)
        ):
            trace_np = read_npz(trace_path)
            inputs, positions, conditions = _trace_inputs(row, seq_len, pad_id)
            resumed_jobs += 1
        else:
            inputs, positions, conditions = _trace_inputs(row, seq_len, pad_id)
            trace_np = _execute_trace(ctx, trace_fn, inputs, positions)
            if ctx.is_primary:
                essential = {
                    key: value for key, value in trace_np.items()
                    if any(token in key for token in (
                        "top_", "candidate_", "captured_mass", "active_count",
                        "effective_count", "query_", "srw_feature_", "delta_",
                        "residual_"))}
                write_npz_atomic(trace_path, **essential)
                write_json_atomic(job_path, {
                    "status": "complete", "stage": "trace",
                    "dataset": dataset_id, "example_id": example_id,
                    "config_hash": config_hash, "job_key": job_key,
                    "artifact": trace_path,
                })
        behavior_correct = behavior_by_example.get(
            example_id, {}).get("correct")
        sig_a = _trace_signature(
            trace_np, 0, metadata, row, "a", behavior_correct)
        signatures.append(sig_a)
        if conditions[1] == "b":
            sig_b = _trace_signature(
                trace_np, 1, metadata, row, "b", behavior_correct)
            signatures.append(sig_b)
            pair_rows.extend(_signature_metrics(
                sig_a, sig_b, f"condition_pair:{metadata['phenomenon']}", False))
        for pool in TRACE_POOLS:
            masses = np.asarray(trace_np[f"{pool}_captured_mass"])
            for layer in range(masses.shape[0]):
                for condition_index in range(2 if conditions[1] == "b" else 1):
                    captured.append(float(masses[layer, condition_index]))
        artifacts.append(trace_path)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for signature in signatures:
        by_group[signature["group"]].append(signature)
    seen_pairs: set[Tuple[str, str]] = set()
    for group, values in sorted(by_group.items()):
        values = sorted(values, key=lambda row: (
            row["logical_example_id"], row["condition"], row["example_id"]))
        used: set[str] = set()
        for left in values:
            if left["example_id"] in used:
                continue
            candidates = [
                right for right in values
                if right["example_id"] not in used
                and right["example_id"] != left["example_id"]
                and right["logical_example_id"] != left["logical_example_id"]
            ]
            candidates.sort(key=lambda right: (
                right.get("surface_key") == left.get("surface_key"),
                abs(int(right["context_length"]) - int(left["context_length"])),
                right["example_id"],
            ))
            if not candidates:
                continue
            right = candidates[0]
            pair_rows.extend(_signature_metrics(
                left, right, "same_function_different_surface", False))
            seen_pairs.add(tuple(sorted((left["example_id"], right["example_id"]))))
            used.update((left["example_id"], right["example_id"]))

    # Explicit same-surface/entity, different-function comparisons where the
    # source contract exposes a stable surface key.
    by_surface: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for signature in signatures:
        if signature.get("surface_key"):
            by_surface[str(signature["surface_key"])].append(signature)
    for values in by_surface.values():
        values = sorted(values, key=lambda row: (row["group"], row["example_id"]))
        for index, left in enumerate(values):
            right = next((
                candidate for candidate in values[index + 1:]
                if candidate["group"] != left["group"]
                and tuple(sorted((left["example_id"], candidate["example_id"]))) not in seen_pairs
            ), None)
            if right is not None:
                pair_rows.extend(_signature_metrics(
                    left, right, "same_surface_different_function", False))
                seen_pairs.add(tuple(sorted((left["example_id"], right["example_id"]))))

    # Fixed-seed, no-duplicate random null. Candidate-token count is exact;
    # context length is the nearest available match from another group.
    ordered = sorted(signatures, key=lambda row: (row["group"], row["example_id"]))
    rng = np.random.default_rng(seed)
    tie_break = {row["example_id"]: float(rng.random()) for row in ordered}
    for left in ordered:
        candidates = [
            right for right in ordered
            if right["group"] != left["group"]
            and right["logical_example_id"] != left["logical_example_id"]
            and right["candidate_token_count"] == left["candidate_token_count"]
            and tuple(sorted((left["example_id"], right["example_id"]))) not in seen_pairs
        ]
        candidates.sort(key=lambda right: (
            abs(int(right["context_length"]) - int(left["context_length"])),
            right["phenomenon"] == left["phenomenon"],
            tie_break[right["example_id"]], right["example_id"],
        ))
        right = candidates[0] if candidates else None
        if right is not None:
            pair_rows.extend(_signature_metrics(
                left, right, "random_length_candidate_matched_null", True))
            seen_pairs.add(tuple(sorted((left["example_id"], right["example_id"]))))
    valid = [row for row in pair_rows if row["metric_valid"]]
    logical_pairs = _logical_route_pairs(pair_rows)
    nonnull = [row for row in logical_pairs if not row["is_random_null"]]
    actual = [
        row for row in nonnull
        if row["pair_type"] == "same_function_different_surface"]
    within = actual
    null = [row for row in logical_pairs if row["is_random_null"]]
    actual_path = _mean(row["path_similarity"] for row in actual)
    null_path = _mean(row["path_similarity"] for row in null)
    enrichment_payload, enrichment = _enrichment_with_permutation(
        signatures, seed)
    operator_entropy: Dict[str, float] = {}
    for group, counter in enrichment.items():
        counts = np.asarray(list(counter.values()), dtype=np.float64)
        probabilities = counts / max(float(counts.sum()), 1.0)
        operator_entropy[group] = float(-np.sum(
            probabilities * np.log(np.maximum(probabilities, 1.0e-12))))
    comparison_pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in logical_pairs:
        comparison_pairs[str(pair["pair_type"])].append(pair)
    summary = {
        "status": "ready" if actual and null else "insufficient_evidence",
        "dataset": dataset_id, "examples": len(signatures),
        "pair_rows": len(pair_rows), "valid_pair_rows": len(valid),
        "valid_logical_pairs": len(logical_pairs),
        "excluded_low_capture": sum(row["invalid_reason"] == "low_captured_mass" for row in pair_rows),
        "captured_mass_mean": _mean(captured),
        "captured_mass_min": min(captured) if captured else None,
        "actual_path_similarity": actual_path,
        "random_null_path_similarity": null_path,
        "within_group_overlap": _mean(row["routing_similarity"] for row in within),
        "between_group_overlap": _mean(row["routing_similarity"] for row in null),
        "within_group_transition_similarity": _mean(row["transition_similarity"] for row in within),
        "between_group_transition_similarity": _mean(row["transition_similarity"] for row in null),
        "within_group_state_similarity": _mean(row["state_similarity"] for row in within),
        "between_group_state_similarity": _mean(row["state_similarity"] for row in null),
        "within_group_query_similarity": _mean(row["query_similarity"] for row in within),
        "between_group_query_similarity": _mean(row["query_similarity"] for row in null),
        "permutation_baseline": enrichment_payload["permutation_baseline_mean_rate"],
        "operator_entropy": operator_entropy,
        "top_enriched_operators": enrichment_payload["groups"],
        "effect_vs_null": (
            actual_path - null_path if actual_path is not None and null_path is not None else None),
        "bootstrap_ci95": _effect_bootstrap_ci(
            [float(row["path_similarity"]) for row in actual],
            [float(row["path_similarity"]) for row in null], seed),
        "comparison_groups": {
            pair_type: {
                "n": len(values),
                "path_similarity": _mean(row["path_similarity"] for row in values),
                "routing_similarity": _mean(row["routing_similarity"] for row in values),
                "transition_similarity": _mean(row["transition_similarity"] for row in values),
            }
            for pair_type, values in sorted(comparison_pairs.items())
        },
        "model_correct_path_similarity": _mean(
            row["path_similarity"] for row in actual
            if row.get("behavior_correct_a") is True
            and row.get("behavior_correct_b") is True),
        "model_incorrect_path_similarity": _mean(
            row["path_similarity"] for row in actual
            if row.get("behavior_correct_a") is False
            or row.get("behavior_correct_b") is False),
        "topk": {"qk": topk_qk, "v": topk_v, "rst": topk_rst},
        "capture_threshold": CAPTURE_THRESHOLD,
        "completed_job_key_fields": [
            "config_hash", "dataset", "stage", "example_id"],
        "resumed_jobs": resumed_jobs,
        "trace_artifacts": artifacts,
        "sec": time.time() - started,
    }
    if ctx.is_primary:
        write_csv_atomic(join_path(output_root, "trace", "route_pair_metrics.csv"), pair_rows)
        write_json_atomic(join_path(output_root, "trace", "route_summary.json"), summary)
        write_json_atomic(
            join_path(output_root, "trace", "operator_enrichment.json"),
            enrichment_payload)
    return summary, enrichment


def _dense_sharded_fns(ctx: AnalysisContext) -> Any:
    if not isinstance(ctx.sharded_fns, dict):
        return ctx.sharded_fns
    return {
        key: value for key, value in ctx.sharded_fns.items()
        if key not in {"vocab_parallel_embedding", "vocab_ce", "vocab_eval_stats", "vocab_argmax"}}


def _make_causal_compare_step(ctx: AnalysisContext):
    sharded = _dense_sharded_fns(ctx)
    kwargs = _runtime_kwargs(ctx, sharded)

    @jax.jit
    def compare(
            params, input_ids, labels, positions, layer, route,
            selected_operator_ids, apply_suppression):
        common = {
            key: value for key, value in kwargs.items()
            if key not in {"minimal_train", "attention_mask"}}
        baseline = ctx.model.apply(
            {"params": params}, input_ids, selected_operator_ids, layer,
            positions, route,
            method=ctx.model.analysis_forward_with_operator_suppression,
            apply_suppression=False, return_residual=True,
            **common, attention_mask=jnp.ones_like(input_ids))
        changed = ctx.model.apply(
            {"params": params}, input_ids, selected_operator_ids, layer,
            positions, route,
            method=ctx.model.analysis_forward_with_operator_suppression,
            apply_suppression=apply_suppression, return_residual=True,
            **common, attention_mask=jnp.ones_like(input_ids))
        base_logits = baseline["logits"][:, :-1].astype(jnp.float32)
        new_logits = changed["logits"][:, :-1].astype(jnp.float32)
        shifted_labels = labels[:, 1:]
        valid = shifted_labels != -100
        safe_labels = jnp.maximum(shifted_labels, 0)
        base_logp = jax.nn.log_softmax(base_logits, axis=-1)
        new_logp = jax.nn.log_softmax(new_logits, axis=-1)
        base_selected = jnp.take_along_axis(base_logp, safe_labels[..., None], axis=-1)[..., 0]
        new_selected = jnp.take_along_axis(new_logp, safe_labels[..., None], axis=-1)[..., 0]
        valid_f = valid.astype(jnp.float32)
        score_before = (base_selected * valid_f).sum(axis=-1)
        score_after = (new_selected * valid_f).sum(axis=-1)
        kl_token = jnp.sum(jnp.exp(base_logp) * (base_logp - new_logp), axis=-1)
        kl = (kl_token * valid_f).sum(axis=-1) / jnp.maximum(valid_f.sum(axis=-1), 1.0)
        top_changed = (
            (jnp.argmax(base_logits, axis=-1) != jnp.argmax(new_logits, axis=-1))
            & valid).sum(axis=-1) / jnp.maximum(valid.sum(axis=-1), 1)
        gather = lambda value: value[jnp.arange(value.shape[0]), positions]
        base_residual = gather(baseline["final_residual"].astype(jnp.float32))
        new_residual = gather(changed["final_residual"].astype(jnp.float32))
        rel = jnp.linalg.norm(new_residual - base_residual, axis=-1) / jnp.maximum(
            jnp.linalg.norm(base_residual, axis=-1), 1.0e-12)
        return score_before, score_after, kl, top_changed, rel

    return compare


def _make_causal_score_step(ctx: AnalysisContext):
    """Task score on the exact production vocab/sharding path."""
    kwargs = _runtime_kwargs(ctx, ctx.sharded_fns)

    @jax.jit
    def score(
            params, input_ids, labels, positions, layer, route,
            selected_operator_ids, apply_suppression):
        common = {
            key: value for key, value in kwargs.items()
            if key not in {"minimal_train", "attention_mask"}}
        normal = ctx.model.apply(
            {"params": params}, input_ids, selected_operator_ids, layer,
            positions, route, labels=labels,
            method=ctx.model.analysis_forward_with_operator_suppression,
            apply_suppression=False, return_residual=True,
            **common, attention_mask=jnp.ones_like(input_ids))
        changed = ctx.model.apply(
            {"params": params}, input_ids, selected_operator_ids, layer,
            positions, route,
            labels=labels,
            method=ctx.model.analysis_forward_with_operator_suppression,
            apply_suppression=apply_suppression, return_residual=True,
            **common, attention_mask=jnp.ones_like(input_ids))
        valid = normal["valid_mask"].astype(jnp.float32)
        before = (-normal["per_token_ce"] * valid).sum(axis=-1)
        after = (-changed["per_token_ce"] * valid).sum(axis=-1)
        gather = lambda value: value[jnp.arange(value.shape[0]), positions]
        residual_before = gather(normal["final_residual"].astype(jnp.float32))
        residual_after = gather(changed["final_residual"].astype(jnp.float32))
        relative_change = jnp.linalg.norm(
            residual_after - residual_before, axis=-1) / jnp.maximum(
                jnp.linalg.norm(residual_before, axis=-1), 1.0e-12)
        return (
            before, after, relative_change,
            jnp.all(normal["per_token_ce"] == changed["per_token_ce"]),
            jnp.all(normal["final_residual"] == changed["final_residual"]),
        )

    return score


def _make_parity_step(ctx: AnalysisContext):
    kwargs = _runtime_kwargs(ctx, ctx.sharded_fns)

    @jax.jit
    def parity(params, input_ids, labels, positions):
        normal = ctx.model.apply(
            {"params": params}, input_ids, labels=labels,
            analysis_return_residual=True,
            **{**kwargs, "attention_mask": jnp.ones_like(input_ids)})
        hooked = ctx.model.apply(
            {"params": params}, input_ids,
            jnp.zeros((input_ids.shape[0],), jnp.int32),
            jnp.int32(0), positions, jnp.int32(0), labels=labels,
            method=ctx.model.analysis_forward_with_operator_suppression,
            apply_suppression=False, return_residual=True,
            **{key: value for key, value in kwargs.items()
               if key not in {"minimal_train", "attention_mask"}},
            attention_mask=jnp.ones_like(input_ids))
        return (
            jnp.max(jnp.abs(normal["per_token_ce"] - hooked["per_token_ce"])),
            jnp.mean(jnp.abs(normal["per_token_ce"] - hooked["per_token_ce"])),
            jnp.max(jnp.abs(normal["final_residual"] - hooked["final_residual"])),
            jnp.all(normal["per_token_ce"] == hooked["per_token_ce"]),
            jnp.all(normal["final_residual"] == hooked["final_residual"]),
        )

    return parity


def _make_dense_parity_step(ctx: AnalysisContext):
    sharded = _dense_sharded_fns(ctx)
    kwargs = _runtime_kwargs(ctx, sharded)

    @jax.jit
    def parity(params, input_ids, positions):
        normal = ctx.model.apply(
            {"params": params}, input_ids, analysis_return_residual=True,
            **{**kwargs, "attention_mask": jnp.ones_like(input_ids)})
        hooked = ctx.model.apply(
            {"params": params}, input_ids,
            jnp.zeros((input_ids.shape[0],), jnp.int32),
            jnp.int32(0), positions, jnp.int32(0),
            method=ctx.model.analysis_forward_with_operator_suppression,
            apply_suppression=False, return_residual=True,
            **{key: value for key, value in kwargs.items()
               if key not in {"minimal_train", "attention_mask"}},
            attention_mask=jnp.ones_like(input_ids))
        difference = jnp.abs(
            normal["logits"].astype(jnp.float32)
            - hooked["logits"].astype(jnp.float32))
        return (
            jnp.mean(difference), jnp.max(difference),
            jnp.mean((jnp.argmax(normal["logits"], axis=-1)
                      == jnp.argmax(hooked["logits"], axis=-1)).astype(jnp.float32)),
            jnp.all(normal["logits"] == hooked["logits"]),
        )

    return parity


def _validate_global_operator_id(
        ctx: AnalysisContext, pool: str, operator_id: int) -> int:
    count_key = {"q": "n_qk", "k": "n_qk", "v": "n_v", "rst": "n_rst"}[pool]
    count = ctx.model_cfg.get(count_key)
    if count is None and pool == "rst":
        count = ctx.model_cfg.get("n_know")
    count = int(count)
    operator_id = int(operator_id)
    if operator_id < 0 or operator_id >= count:
        raise ValueError(
            f"global operator id {operator_id} is outside {pool} pool [0, {count})")
    return operator_id


def _candidate_rows(
    trace: Mapping[str, np.ndarray], pool: str, condition: int,
    cross_operator_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ids = np.asarray(trace[f"{pool}_candidate_ids"])[:, condition]
    valid = np.asarray(trace[f"{pool}_candidate_valid"])[:, condition].astype(bool)
    execution = np.asarray(trace[f"{pool}_candidate_execution"])[:, condition]
    admission = np.asarray(trace[f"{pool}_candidate_admission"])[:, condition]
    absolute = np.asarray(trace[f"{pool}_candidate_abs_coefficient"])[:, condition]
    signed = np.asarray(trace[f"{pool}_candidate_coefficient"])[:, condition]
    strategy_index = {name: index for index, name in enumerate(TRANSITION_CANDIDATE_STRATEGIES)}
    contribution_layer = int(np.argmax(absolute[:, strategy_index["top_contribution"]]))
    gate_layer = int(np.argmax(execution[:, strategy_index["top_gate"]]))
    rows = []
    for strategy in TRANSITION_CANDIDATE_STRATEGIES:
        index = strategy_index[strategy]
        layer = gate_layer if strategy == "top_gate" else contribution_layer
        is_valid = bool(valid[layer, index]) and int(ids[layer, index]) >= 0
        rows.append({
            "strategy": strategy, "layer": layer,
            "operator_id": int(ids[layer, index]), "candidate_valid": is_valid,
            "execution": float(execution[layer, index]),
            "admission": float(admission[layer, index]),
            "sidecar_estimated_post_denominator_coefficient": float(
                signed[layer, index]),
            "sidecar_estimated_abs_post_denominator_coefficient": float(
                absolute[layer, index]),
            "skip_reason": None if is_valid else f"no_{strategy}_candidate",
        })
    cross_row = {
        "strategy": "cross_function_control", "layer": contribution_layer,
        "operator_id": -1, "candidate_valid": False, "execution": 0.0,
        "admission": 0.0,
        "sidecar_estimated_post_denominator_coefficient": 0.0,
        "sidecar_estimated_abs_post_denominator_coefficient": 0.0,
        "skip_reason": "no_cross_group_enriched_operator",
    }
    if cross_operator_id is not None:
        top_ids = np.asarray(trace[f"{pool}_top_idx"])[:, condition]
        matches = np.argwhere(top_ids == int(cross_operator_id))
        if matches.size:
            coefficients = np.asarray(trace[f"{pool}_top_coefficient"])[:, condition]
            best = max(matches.tolist(), key=lambda pos: abs(float(coefficients[pos[0], pos[1]])))
            layer, rank = int(best[0]), int(best[1])
            cross_row = {
                "strategy": "cross_function_control", "layer": layer,
                "operator_id": int(cross_operator_id), "candidate_valid": True,
                "execution": float(np.asarray(trace[f"{pool}_top_val"])[layer, condition, rank]),
                "admission": float(np.asarray(trace[f"{pool}_top_admission"])[layer, condition, rank]),
                "sidecar_estimated_post_denominator_coefficient": float(
                    coefficients[layer, rank]),
                "sidecar_estimated_abs_post_denominator_coefficient": abs(
                    float(coefficients[layer, rank])),
                "skip_reason": None,
            }
        else:
            cross_row["operator_id"] = int(cross_operator_id)
            cross_row["skip_reason"] = "cross_group_operator_not_in_captured_current_route"
    rows.append(cross_row)
    return rows


def _cross_group_operator(
    enrichment: Mapping[str, Counter[Tuple[str, int, int]]],
    current_group: str, pool: str, excluded_operator_ids: Optional[set[int]] = None,
) -> Optional[int]:
    excluded_operator_ids = excluded_operator_ids or set()
    candidates: Counter[int] = Counter()
    for group, counter in enrichment.items():
        if str(group) == str(current_group):
            continue
        for (entry_pool, _layer, operator_id), count in counter.items():
            if entry_pool == pool and int(operator_id) not in excluded_operator_ids:
                candidates[int(operator_id)] += int(count)
    return candidates.most_common(1)[0][0] if candidates else None


def _causal_sequences(row: Mapping[str, Any], condition: str, seq_len: int, pad_id: int) -> Tuple[Dict[str, Any], Dict[str, Any], bool, bool]:
    mode = str(row["metadata"]["score_mode"])
    if mode == "paired_sequence_logprob":
        a = _sequence_spec(_row_array(row, "context_ids_a"), candidate_start=0, seq_len=seq_len, pad_id=pad_id, tag="a")
        b = _sequence_spec(_row_array(row, "context_ids_b"), candidate_start=0, seq_len=seq_len, pad_id=pad_id, tag="b")
        return a, b, condition == "a", condition == "b"
    context = _row_array(row, f"context_ids_{condition}")
    positive = _row_array(row, f"positive_ids_{condition}")
    negative = _row_array(row, f"negative_ids_{condition}")
    return (
        _sequence_spec(context + positive, candidate_start=len(context), seq_len=seq_len, pad_id=pad_id, tag="positive"),
        _sequence_spec(context + negative, candidate_start=len(context), seq_len=seq_len, pad_id=pad_id, tag="negative"),
        True, True,
    )


def _logical_causal_summary(
    rows: Sequence[Mapping[str, Any]], seed: int,
) -> Dict[str, Any]:
    by_example: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get("task_margin_drop")
        if value is not None and math.isfinite(float(value)):
            by_example[str(row["example_id"])].append(float(value))
    logical = [float(np.mean(values)) for values in by_example.values() if values]
    return {
        "n_examples": len(logical), "n_jobs": sum(len(values) for values in by_example.values()),
        "mean_task_margin_drop": _mean(logical),
        "bootstrap_ci95": _bootstrap_ci(logical, seed),
    }


def _paired_selected_control_effect(
    rows: Sequence[Mapping[str, Any]], seed: int,
) -> Dict[str, Any]:
    selected: Dict[str, List[float]] = defaultdict(list)
    controls: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get("task_margin_drop")
        if value is None:
            continue
        target = (
            selected if row.get("strategy") in {"top_contribution", "top_gate"}
            else controls if row.get("strategy") in {
                "matched_active", "active_random", "inactive_random"}
            else None)
        if target is not None:
            target[str(row["example_id"])].append(float(value))
    differences = [
        float(np.mean(selected[example_id]) - np.mean(controls[example_id]))
        for example_id in sorted(set(selected) & set(controls))
        if selected[example_id] and controls[example_id]
    ]
    return {
        "n_examples": len(differences),
        "mean_selected_minus_control": _mean(differences),
        "bootstrap_ci95": _bootstrap_ci(differences, seed),
    }


def _run_causal(
    ctx: AnalysisContext, build: OperatorDatasetBuild, dataset_id: str,
    profile: Mapping[str, Any], output_root: str, pad_id: int, cache_dir: str,
    behavior_by_example: Mapping[str, Mapping[str, Any]],
    enrichment: Mapping[str, Counter[Tuple[str, int, int]]],
    *, config_hash: str, resume: bool,
) -> Dict[str, Any]:
    started = time.time()
    seq_len = int(ctx.model_cfg["max_seq_len"])
    data_replicas = max(1, int(ctx.mesh.shape["data"]))
    seed = int(getattr(ctx.args, "operator_analysis_seed", 4171))
    trace_fn = jax.jit(lambda p, x, t: topk_trace_forward(
        p, ctx.model_cfg, x,
        topk_qk=int(getattr(ctx.args, "transition_topk_qk", 512)),
        topk_v=int(getattr(ctx.args, "transition_topk_v", 2048)),
        topk_rst=int(getattr(ctx.args, "transition_topk_rst", 4096)),
        target_positions=t, candidate_seed=seed,
        production_srw_fns=ctx.sharded_fns))
    compare = _make_causal_compare_step(ctx)
    score_exact = _make_causal_score_step(ctx)
    parity_step = _make_parity_step(ctx)
    dense_parity_step = _make_dense_parity_step(ctx)
    causal_candidates = list(iter_operator_rows(
        build, dataset_id, subset=profile["causal_subset"], cache_dir=cache_dir))
    causal_candidates.sort(key=lambda row: (
        not bool(behavior_by_example.get(str(row["metadata"]["example_id"]), {}).get("correct")),
        str(row["metadata"]["group_id"]), str(row["metadata"]["example_id"])))
    rows = list(_stratified_rows(
        causal_candidates, int(profile["causal_max"]),
        profile.get("causal_per_group")))
    parity_summary: Dict[str, Any] = {"status": "not_run", "machine_exact": False}
    jobs: List[Dict[str, Any]] = []
    job_root = join_path(output_root, "causal", "jobs")
    makedirs(job_root)
    sync_hosts(f"operator-causal-{dataset_id}-resume-snapshot")
    reusable_jobs = set(list_paths(job_root, "*.json")) if resume else set()
    resumed_jobs = 0
    for row_index, row in enumerate(rows):
        example_id = str(row["metadata"]["example_id"])
        job_key = _analysis_job_key(config_hash, dataset_id, "causal", example_id)
        job_path = join_path(job_root, f"{job_key}.json")
        cached = read_json(job_path, {}) if job_path in reusable_jobs else {}
        if (
            cached.get("status") == "complete"
            and cached.get("config_hash") == config_hash
            and cached.get("job_key") == job_key
            and isinstance(cached.get("jobs"), list)
            and isinstance(cached.get("causal_parity"), Mapping)
        ):
            cached_parity = dict(cached["causal_parity"])
            if not cached_parity.get("machine_exact"):
                raise CausalParityError(
                    "cached production-core zero-suppression parity failed")
            if parity_summary.get("status") != "ready":
                parity_summary = cached_parity
            jobs.extend(dict(value) for value in cached["jobs"])
            resumed_jobs += 1
            continue
        row_job_start = len(jobs)
        inputs, positions, conditions = _trace_inputs(row, seq_len, pad_id)
        trace_np = _execute_trace(ctx, trace_fn, inputs, positions)
        real_conditions = ["a", "b"] if conditions[1] == "b" else ["a"]
        for condition_index, condition in enumerate(real_conditions):
            positive, negative, apply_positive, apply_negative = _causal_sequences(
                row, condition, seq_len, pad_id)
            if row_index == 0 and condition_index == 0:
                p_input = np.repeat(positive["input_ids"][None, :], data_replicas, axis=0)
                p_labels = np.repeat(positive["labels"][None, :], data_replicas, axis=0)
                parity_values = jax.device_get(parity_step(
                    ctx.params, jax.device_put(jnp.asarray(p_input), ctx.data_sharding),
                    jax.device_put(jnp.asarray(p_labels), ctx.data_sharding),
                    jax.device_put(jnp.full((data_replicas,), positions[condition_index], jnp.int32), ctx.data_sharding)))
                dense_parity = jax.device_get(dense_parity_step(
                    ctx.params, jax.device_put(jnp.asarray(p_input), ctx.data_sharding),
                    jax.device_put(jnp.full((data_replicas,), positions[condition_index], jnp.int32), ctx.data_sharding)))
                parity_summary = {
                    "status": "ready",
                    "machine_exact": (
                        bool(parity_values[3]) and bool(parity_values[4])
                        and bool(dense_parity[3])),
                    "ce_max_abs_diff": float(parity_values[0]),
                    "ce_mean_abs_diff": float(parity_values[1]),
                    "mean_logit_abs_diff": float(dense_parity[0]),
                    "max_logit_abs_diff": float(dense_parity[1]),
                    "top1_agreement": float(dense_parity[2]),
                    "final_residual_max_abs_diff": float(parity_values[2]),
                    "intervention_type": INTERVENTION_TYPE,
                    "baseline_path": "production_core_suppression_disabled",
                    "intervention_path": "production_core_suppression_enabled",
                    "logit_check_path": (
                        "production minimal SRW path with dense vocab projection; "
                        "CE/residual check retains production vocab sharding"),
                }
                if not parity_summary["machine_exact"]:
                    raise CausalParityError(
                        "production-core zero-suppression parity failed")
            for pool in TRACE_POOLS:
                selected_ids = {
                    int(value) for value in np.asarray(
                        trace_np[f"{pool}_candidate_ids"]
                    )[:, condition_index, :].reshape(-1)
                    if int(value) >= 0
                }
                cross_operator = _cross_group_operator(
                    enrichment, str(row["metadata"]["group_id"]), pool,
                    selected_ids)
                for candidate in _candidate_rows(
                    trace_np, pool, condition_index, cross_operator):
                    base = {
                        "example_id": example_id,
                        "job_key": job_key,
                        "pair_id": row["metadata"]["pair_id"],
                        "dataset": dataset_id, "group": row["metadata"]["group_id"],
                        "phenomenon": row["metadata"]["phenomenon"],
                        "relation": row["metadata"]["relation"],
                        "condition": condition, "pool": pool,
                        "target_position": int(positions[condition_index]),
                        "behavior_correct": bool(behavior_by_example.get(
                            example_id, {}).get("correct")),
                        "intervention_type": INTERVENTION_TYPE,
                        "candidate_selection_source": "sidecar_trace",
                        "intervention_execution_source": "production_core",
                        "canonical_unpruned_admission_denominator": True,
                        **candidate,
                    }
                    if not candidate["candidate_valid"]:
                        jobs.append({**base, "status": "skipped"})
                        continue
                    operator_id = _validate_global_operator_id(
                        ctx, pool, candidate["operator_id"])
                    route_code = {"q": 0, "k": 1, "v": 2, "rst": 3}[pool]
                    selected_operator_ids = jax.device_put(
                        jnp.full((data_replicas,), operator_id, jnp.int32),
                        ctx.data_sharding)
                    def evaluate(
                            spec: Mapping[str, Any], enabled: bool,
                    ) -> Tuple[float, float, float, float, float, bool, bool]:
                        x = np.repeat(spec["input_ids"][None, :], data_replicas, axis=0)
                        y = np.repeat(spec["labels"][None, :], data_replicas, axis=0)
                        exact = jax.device_get(score_exact(
                            ctx.params, jax.device_put(jnp.asarray(x), ctx.data_sharding),
                            jax.device_put(jnp.asarray(y), ctx.data_sharding),
                            jax.device_put(jnp.full((data_replicas,), positions[condition_index], jnp.int32), ctx.data_sharding),
                            jnp.int32(candidate["layer"]), jnp.int32(route_code),
                            selected_operator_ids, jnp.bool_(enabled)))
                        dense = jax.device_get(compare(
                            ctx.params, jax.device_put(jnp.asarray(x), ctx.data_sharding),
                            jax.device_put(jnp.asarray(y), ctx.data_sharding),
                            jax.device_put(jnp.full((data_replicas,), positions[condition_index], jnp.int32), ctx.data_sharding),
                            jnp.int32(candidate["layer"]), jnp.int32(route_code),
                            selected_operator_ids, jnp.bool_(enabled)))
                        return (
                            float(np.asarray(exact[0])[0]),
                            float(np.asarray(exact[1])[0]),
                            float(np.asarray(dense[2])[0]),
                            float(np.asarray(dense[3])[0]),
                            float(np.asarray(exact[2])[0]),
                            bool(np.asarray(exact[3])),
                            bool(np.asarray(exact[4])),
                        )
                    (pos_before, pos_after, pos_kl, pos_top, pos_rel,
                     pos_ce_exact, pos_residual_exact) = evaluate(
                        positive, apply_positive)
                    (neg_before, neg_after, neg_kl, neg_top, neg_rel,
                     neg_ce_exact, neg_residual_exact) = evaluate(
                        negative, apply_negative)
                    if (
                        float(candidate["execution"]) == 0.0
                        and (
                            (apply_positive
                             and not (pos_ce_exact and pos_residual_exact))
                            or (apply_negative
                                and not (neg_ce_exact and neg_residual_exact)))
                    ):
                        raise CausalParityError(
                            "inactive operator suppression changed production output")
                    margin_before = pos_before - neg_before
                    margin_after = pos_after - neg_after
                    jobs.append({
                        **base, "status": "ready",
                        "behavior_margin_before": margin_before,
                        "behavior_margin_after": margin_after,
                        "positive_logprob_delta": pos_after - pos_before,
                        "negative_logprob_delta": neg_after - neg_before,
                        "margin_delta": margin_after - margin_before,
                        "task_margin_drop": margin_before - margin_after,
                        "kl": float(np.mean([pos_kl, neg_kl])),
                        "kl_and_top_prediction_source": (
                            "device_aggregated_dense_logits; task margin uses exact production vocab path"),
                        "top_prediction_change_fraction": float(np.mean([pos_top, neg_top])),
                        "target_residual_relative_change": float(np.mean([pos_rel, neg_rel])),
                        "positive_machine_exact_noop": (
                            pos_ce_exact and pos_residual_exact),
                        "negative_machine_exact_noop": (
                            neg_ce_exact and neg_residual_exact),
                    })
        if ctx.is_primary:
            write_json_atomic(job_path, {
                "status": "complete", "stage": "causal",
                "dataset": dataset_id, "example_id": example_id,
                "config_hash": config_hash, "job_key": job_key,
                "causal_parity": parity_summary,
                "jobs": jobs[row_job_start:],
            })
    valid = [row for row in jobs if row.get("status") == "ready"]
    headline_correct_only = dataset_id in {"blimp", "lama", "counterfact"}
    headline_valid = (
        [row for row in valid if row.get("behavior_correct") is True]
        if headline_correct_only else valid)
    by_strategy: Dict[str, Dict[str, Any]] = {}
    for strategy in (
        "top_contribution", "top_gate", "matched_active", "active_random",
        "inactive_random", "cross_function_control"):
        row_summary = _logical_causal_summary(
            [row for row in headline_valid if row["strategy"] == strategy],
            seed + len(by_strategy))
        by_strategy[strategy] = {"n": row_summary["n_examples"], **row_summary}
    selected_control = _paired_selected_control_effect(headline_valid, seed + 101)

    def by_dimension(field: str, offset: int) -> Dict[str, Any]:
        values: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for causal_row in headline_valid:
            values[str(causal_row.get(field))].append(causal_row)
        return {
            key: _logical_causal_summary(rows_for_key, seed + offset + index)
            for index, (key, rows_for_key) in enumerate(sorted(values.items()))
        }

    pool_layer_strategy: List[Dict[str, Any]] = []
    grouped_jobs: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for causal_row in headline_valid:
        grouped_jobs[(
            str(causal_row["pool"]), int(causal_row["layer"]),
            str(causal_row["strategy"]),
        )].append(causal_row)
    for index, ((pool, layer, strategy), grouped) in enumerate(sorted(grouped_jobs.items())):
        pool_layer_strategy.append({
            "pool": pool, "layer": layer, "strategy": strategy,
            **_logical_causal_summary(grouped, seed + 500 + index),
        })

    has_selected = any(by_strategy[name]["n_examples"] for name in (
        "top_contribution", "top_gate"))
    has_control = any(by_strategy[name]["n_examples"] for name in (
        "matched_active", "active_random", "inactive_random"))
    summary = {
        "status": (
            "ready" if headline_valid and parity_summary.get("machine_exact")
            and has_selected and has_control and selected_control["n_examples"]
            else "insufficient_evidence"),
        "dataset": dataset_id, "intervention_type": INTERVENTION_TYPE,
        "causal_parity": parity_summary,
        "examples": len(rows), "jobs": len(headline_valid),
        "all_ready_jobs": len(valid), "skipped_jobs": len(jobs) - len(valid),
        "behavior_correct_examples": sum(bool(behavior_by_example.get(str(row["metadata"]["example_id"]), {}).get("correct")) for row in rows),
        "by_strategy": by_strategy,
        "selected_minus_control_effect": selected_control["mean_selected_minus_control"],
        "selected_minus_control": selected_control,
        "by_pool": by_dimension("pool", 200),
        "by_layer": by_dimension("layer", 250),
        "by_group": by_dimension("group", 300),
        "by_phenomenon": by_dimension("phenomenon", 350),
        "by_relation": by_dimension("relation", 400),
        "by_pool_layer_strategy": pool_layer_strategy,
        "skip_reasons": dict(Counter(
            str(row.get("skip_reason") or "unspecified")
            for row in jobs if row.get("status") == "skipped")),
        "bootstrap_unit": "logical_example",
        "completed_job_key_fields": [
            "config_hash", "dataset", "stage", "example_id"],
        "resumed_jobs": resumed_jobs,
        "known_correct_interpretation_only": dataset_id in {"lama", "counterfact"},
        "headline_model_correct_only": headline_correct_only,
        "sec": time.time() - started,
    }
    if ctx.is_primary:
        write_jsonl_atomic(join_path(output_root, "causal", "causal_jobs.jsonl"), jobs)
        write_json_atomic(join_path(output_root, "causal", "causal_summary.json"), summary)
    return summary


def _analysis_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Operator Analysis v2", "",
        f"Status: `{summary.get('status')}`", "",
        "## 1. Behavior competence", "",
    ]
    for dataset, row in sorted((summary.get("datasets") or {}).items()):
        behavior = row.get("behavior") or {}
        lines.append(
            f"- **{dataset}**: n={behavior.get('n', 0)}, "
            f"accuracy={behavior.get('accuracy')}, margin={behavior.get('mean_margin')}")
    lines.extend(["", "## 2. Same-function path reuse vs random null", ""])
    for dataset, row in sorted((summary.get("function_reuse") or {}).items()):
        if dataset == "overall":
            continue
        lines.append(
            f"- **{dataset}**: actual={row.get('actual_path')}, "
            f"null={row.get('random_null')}, effect={row.get('effect')}, "
            f"CI95={row.get('ci95')}")
    lines.extend(["", "## 3. Same-surface, different-function divergence", ""])
    for dataset, row in sorted((summary.get("function_reuse") or {}).items()):
        if dataset == "overall":
            continue
        comparison = (row.get("comparisons") or {}).get(
            "same_surface_different_function") or {}
        lines.append(
            f"- **{dataset}**: n={comparison.get('n', 0)}, "
            f"path_similarity={comparison.get('path_similarity')}")
    lines.extend(["", "## 4. Selected vs control causal effect", ""])
    for dataset, row in sorted((summary.get("causal_specificity") or {}).items()):
        if dataset == "overall":
            continue
        lines.append(
            f"- **{dataset}**: status={row.get('status')}, "
            f"selected_minus_control={row.get('selected_minus_control_effect')}")
    lines.extend(["", "## 5. Pool roles", ""])
    for dataset, row in sorted((summary.get("causal_specificity") or {}).items()):
        if dataset == "overall":
            continue
        pool_effects = {
            pool: value.get("mean_task_margin_drop")
            for pool, value in (row.get("by_pool") or {}).items()
        }
        lines.append(f"- **{dataset}**: `{json.dumps(pool_effects, sort_keys=True)}`")
    lines.extend([
        "", "## 6. Validity, sample counts, and limitations", "",
        f"- Build: `{summary.get('build_id')}`",
        f"- Status: `{summary.get('status')}`",
    ])
    for limitation in summary.get("limitations") or []:
        lines.append(f"- Limitation: {limitation}")
    lines.append("")
    return "\n".join(lines)


def _print_console_summary(result: Mapping[str, Any]) -> None:
    validation = result.get("dataset_manifest") or {}
    print("OPERATOR DATASET BUILD:", flush=True)
    print(
        f"  root={validation.get('root')} build_id={result.get('build_id')} "
        f"schema={validation.get('schema')} tokenizer="
        f"{(validation.get('tokenizer') or {}).get('tokenizer_name')}", flush=True)
    for dataset, row in sorted((validation.get("datasets") or {}).items()):
        print(
            f"  dataset={dataset} rows={row.get('rows')} shards={row.get('shards')} "
            f"checksummed={row.get('checksummed_shards')}", flush=True)
    for dataset, row in sorted((result.get("datasets") or {}).items()):
        behavior = row.get("behavior") or {}
        print(
            "BEHAVIOR: "
            f"dataset={dataset} n={behavior.get('n', 0)} "
            f"accuracy={behavior.get('accuracy')} margin={behavior.get('mean_margin')} "
            f"known_correct={behavior.get('known_correct', 0)}", flush=True)
        route = row.get("route") or {}
        print(
            "FUNCTION REUSE: "
            f"dataset={dataset} actual={route.get('actual_path_similarity')} "
            f"random_null={route.get('random_null_path_similarity')} "
            f"effect={route.get('effect_vs_null')} ci95={route.get('bootstrap_ci95')} "
            "same_surface_different_function="
            f"{(route.get('comparison_groups') or {}).get('same_surface_different_function')}",
            flush=True)
        print(
            "ROUTE SPECIFICITY: "
            f"dataset={dataset} within={route.get('within_group_overlap')} "
            f"between={route.get('between_group_overlap')} "
            f"specificity={route.get('effect_vs_null')}", flush=True)
        causal = row.get("causal") or {}
        strategies = causal.get("by_strategy") or {}
        print(
            "CAUSAL SPECIFICITY: "
            f"dataset={dataset} selected={((strategies.get('top_contribution') or {}).get('mean_task_margin_drop'))} "
            f"matched={((strategies.get('matched_active') or {}).get('mean_task_margin_drop'))} "
            f"active_random={((strategies.get('active_random') or {}).get('mean_task_margin_drop'))} "
            f"inactive_random={((strategies.get('inactive_random') or {}).get('mean_task_margin_drop'))} "
            f"effect={causal.get('selected_minus_control_effect')} "
            f"jobs={causal.get('jobs', 0)} skipped={causal.get('skipped_jobs', 0)}",
            flush=True)
        print(
            "CAUSAL POOL ROLES: "
            f"dataset={dataset} by_pool={causal.get('by_pool', {})}",
            flush=True)
        print(
            "DATASET DETAILS: "
            f"dataset={dataset} status={row.get('status')} "
            f"details={(behavior.get('dataset_details') or {})}", flush=True)
        print(
            "VALIDITY: "
            f"dataset={dataset} captured_mean={route.get('captured_mass_mean')} "
            f"captured_min={route.get('captured_mass_min')} "
            f"excluded_low_capture={route.get('excluded_low_capture', 0)} "
            f"behavior_correct={causal.get('behavior_correct_examples', 0)} "
            f"causal_parity={(causal.get('causal_parity') or {}).get('machine_exact')} "
            f"failed={row.get('status') == 'failed'} skipped={causal.get('skipped_jobs', 0)}",
            flush=True)
    print(
        "VALIDITY: "
        f"status={result.get('status')} limitations={result.get('limitations') or []}",
        flush=True)


def run_operator_analysis(ctx: AnalysisContext, items: Sequence[str]) -> Dict[str, Any]:
    """Run selected operator items and return artifact-backed item statuses."""
    selected_items = set(items) & OPERATOR_ITEMS
    if not selected_items:
        return {}
    profile = _profile_config(ctx.args)
    datasets = _selected_datasets(getattr(ctx.args, "operator_datasets", "all"))
    cache_dir = str(getattr(ctx.args, "operator_cache_dir", DEFAULT_OPERATOR_CACHE_DIR))
    build = resolve_operator_dataset_build(getattr(ctx.args, "operator_dataset_root", None))
    validation = validate_operator_dataset_build(
        build.root, required_datasets=datasets, cache_dir=cache_dir,
        checksum_scope="all")
    prepared_vocab = int((validation.get("tokenizer") or {}).get("vocab_size", -1))
    model_vocab = int(ctx.model_cfg.get("logical_vocab_size", ctx.model_cfg.get("vocab_size", -1)))
    if prepared_vocab != model_vocab:
        raise ValueError(
            f"Prepared tokenizer/model vocabulary mismatch: prepared={prepared_vocab} model={model_vocab}")
    code_digest = hashlib.sha256()
    model_module = analysis_model_module(ctx.model_cfg)
    model_source_paths = [Path(model_module.__file__).resolve()]
    shared_core = getattr(model_module, "_core", None)
    shared_core_path = getattr(shared_core, "__file__", None)
    if shared_core_path is not None:
        shared_core_path = Path(shared_core_path).resolve()
        if shared_core_path not in model_source_paths:
            model_source_paths.append(shared_core_path)
    for path in (
        Path(__file__),
        Path(__file__).with_name("dawn_operator_datasets.py"),
        Path(__file__).with_name("dawn_analysis_trace.py"),
        *model_source_paths,
    ):
        code_digest.update(path.name.encode("utf-8"))
        code_digest.update(path.read_bytes())
    code_hash = code_digest.hexdigest()
    config = {
        "checkpoint_step": ctx.checkpoint_step,
        "checkpoint_path": ctx.checkpoint_path,
        "model_version": ctx.model_cfg.get("model_version"),
        "dataset_build_id": build.build_id,
        "dataset_manifest_hash": build.manifest_hash,
        "datasets": datasets, "items": sorted(selected_items), "profile": profile,
        "analysis_code_hash": code_hash,
        "runtime_gate_state": {
            key: ctx.model_cfg.get(key)
            for key in (
                "soft_gate_temperature", "soft_gate_t_final",
                "soft_gate_T_qk", "soft_gate_T_v", "soft_gate_T_rst",
                "soft_gate_boundary_power", "soft_gate_boundary_power_final",
                "admission_den_power", "admission_den_power_qk",
                "admission_den_power_v", "admission_den_power_rst",
                "srw_composition_mode",
                "heat_kernel_beta", "execution_prune_eps",
            )
        },
        "topk": {
            "qk": int(getattr(ctx.args, "transition_topk_qk", 512)),
            "v": int(getattr(ctx.args, "transition_topk_v", 2048)),
            "rst": int(getattr(ctx.args, "transition_topk_rst", 4096))},
        "capture_threshold": CAPTURE_THRESHOLD,
        "causal_strategy": list(TRANSITION_CANDIDATE_STRATEGIES) + ["cross_function_control"],
        "intervention_type": INTERVENTION_TYPE,
        "seed": int(getattr(ctx.args, "operator_analysis_seed", 4171)),
    }
    config_hash = canonical_hash(config)
    output_root = ctx.store.path(
        "operator_analysis_v2", profile["profile"], f"build_{build.build_id}",
        f"config_{config_hash[:16]}")
    manifest_path = join_path(output_root, "analysis_manifest.json")
    resume = bool(getattr(ctx.args, "operator_analysis_resume", True))
    if resume and exists(manifest_path):
        existing = read_json(manifest_path, {})
        if existing.get("status") == "complete" and existing.get("config_hash") == config_hash:
            result_path = join_path(output_root, "summaries", "operator_analysis_summary.json")
            resumed = read_json(result_path, {})
            resumed["resumed"] = True
            if ctx.is_primary:
                _print_console_summary(resumed)
            return resumed
    for rel in ("behavior", "trace", "causal", "summaries", "errors"):
        makedirs(join_path(output_root, rel))
    analysis_manifest = {
        "status": "running", "created_at": utc_now(), "config": config,
        "config_hash": config_hash, "output_root": output_root,
        "dataset_validation": validation, "jobs": {},
    }
    if ctx.is_primary:
        write_json_atomic(manifest_path, analysis_manifest)
    pad_id = int((build.manifest.get("tokenizer") or {}).get("pad_token_id") or 0)
    dataset_results: Dict[str, Any] = {}
    needs_model_analysis = bool(selected_items - {"operator_dataset_manifest"})
    if needs_model_analysis:
        for dataset_id in datasets:
            dataset_root = join_path(output_root, dataset_id)
            dataset_summary_path = join_path(dataset_root, "dataset_summary.json")
            for rel in ("behavior/parts", "trace", "causal"):
                makedirs(join_path(dataset_root, rel))
            if resume and exists(dataset_summary_path):
                cached = read_json(dataset_summary_path, {})
                if cached.get("config_hash") == config_hash and cached.get("status") in {
                    "ready", "partial", "insufficient_evidence"}:
                    dataset_results[dataset_id] = cached
                    continue
            try:
                behavior, behavior_rows = _run_behavior(
                    ctx, build, dataset_id, profile, dataset_root, pad_id, cache_dir,
                    config_hash=config_hash, resume=resume)
                route, enrichment = _run_trace(
                    ctx, build, dataset_id, profile, dataset_root, pad_id,
                    cache_dir, behavior_rows,
                    config_hash=config_hash, resume=resume)
                causal = _run_causal(
                    ctx, build, dataset_id, profile, dataset_root, pad_id,
                    cache_dir, behavior_rows, enrichment,
                    config_hash=config_hash, resume=resume)
                status = (
                    "ready" if behavior["status"] == route["status"] == causal["status"] == "ready"
                    else "partial" if behavior["status"] == "ready" else "insufficient_evidence")
                dataset_results[dataset_id] = {
                    "status": status, "behavior": behavior, "route": route,
                    "causal": causal,
                    "config_hash": config_hash,
                    "artifacts": {
                        "behavior": join_path(dataset_root, "behavior", "behavior_summary.json"),
                        "trace": join_path(dataset_root, "trace", "route_summary.json"),
                        "causal": join_path(dataset_root, "causal", "causal_summary.json"),
                        "errors": join_path(dataset_root, "errors.jsonl"),
                    },
                }
                if ctx.is_primary:
                    write_jsonl_atomic(join_path(dataset_root, "errors.jsonl"), [])
                    write_json_atomic(dataset_summary_path, dataset_results[dataset_id])
            except Exception as exc:
                if isinstance(exc, CausalParityError):
                    if ctx.is_primary:
                        write_jsonl_atomic(join_path(dataset_root, "errors.jsonl"), [{
                            "dataset": dataset_id, "error_type": type(exc).__name__,
                            "error": str(exc), "at": utc_now(),
                            "blocking": True,
                        }])
                        analysis_manifest.update({
                            "status": "failed", "failed_at": utc_now(),
                            "blocking_error": f"{type(exc).__name__}: {exc}",
                        })
                        write_json_atomic(manifest_path, analysis_manifest)
                    raise
                dataset_results[dataset_id] = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                if ctx.is_primary:
                    write_jsonl_atomic(join_path(dataset_root, "errors.jsonl"), [{
                        "dataset": dataset_id, "error_type": type(exc).__name__,
                        "error": str(exc), "at": utc_now()}])
                if bool(getattr(ctx.args, "fail_fast", False)):
                    raise
    behavior_eval = {
        dataset: {
            "status": row.get("behavior", {}).get("status") or row.get("status"),
            "n": row.get("behavior", {}).get("n"),
            "accuracy": row.get("behavior", {}).get("accuracy"),
            "mean_margin": row.get("behavior", {}).get("mean_margin"),
            "known_correct": row.get("behavior", {}).get("known_correct"),
        } for dataset, row in dataset_results.items()}
    function_reuse = {
        dataset: {
            "actual_path": row.get("route", {}).get("actual_path_similarity"),
            "random_null": row.get("route", {}).get("random_null_path_similarity"),
            "effect": row.get("route", {}).get("effect_vs_null"),
            "ci95": row.get("route", {}).get("bootstrap_ci95"),
            "routing_within": row.get("route", {}).get("within_group_overlap"),
            "routing_null": row.get("route", {}).get("between_group_overlap"),
            "transition_within": row.get("route", {}).get("within_group_transition_similarity"),
            "transition_null": row.get("route", {}).get("between_group_transition_similarity"),
            "state_within": row.get("route", {}).get("within_group_state_similarity"),
            "query_within": row.get("route", {}).get("within_group_query_similarity"),
            "comparisons": row.get("route", {}).get("comparison_groups"),
            "n": row.get("route", {}).get("valid_logical_pairs"),
        } for dataset, row in dataset_results.items()}
    route_specificity = {
        dataset: {
            "within": row.get("route", {}).get("actual_path_similarity"),
            "between_null": row.get("route", {}).get("random_null_path_similarity"),
            "specificity_gap": row.get("route", {}).get("effect_vs_null"),
            "captured_mass": row.get("route", {}).get("captured_mass_mean"),
            "permutation_baseline": row.get("route", {}).get("permutation_baseline"),
            "operator_entropy": row.get("route", {}).get("operator_entropy"),
            "top_enriched_operators": row.get("route", {}).get("top_enriched_operators"),
        } for dataset, row in dataset_results.items()}
    causal_specificity = {
        dataset: {
            "status": row.get("causal", {}).get("status"),
            "strategies": row.get("causal", {}).get("by_strategy"),
            "selected_minus_control_effect": row.get("causal", {}).get("selected_minus_control_effect"),
            "selected_minus_control": row.get("causal", {}).get("selected_minus_control"),
            "by_pool": row.get("causal", {}).get("by_pool"),
            "by_layer": row.get("causal", {}).get("by_layer"),
            "by_group": row.get("causal", {}).get("by_group"),
            "by_phenomenon": row.get("causal", {}).get("by_phenomenon"),
            "by_relation": row.get("causal", {}).get("by_relation"),
            "by_pool_layer_strategy": row.get("causal", {}).get("by_pool_layer_strategy"),
            "parity": row.get("causal", {}).get("causal_parity"),
        } for dataset, row in dataset_results.items()}

    behavior_rows = list(behavior_eval.values())
    behavior_n = sum(int(row.get("n") or 0) for row in behavior_rows)
    behavior_accuracy_n = sum(
        int(row.get("n") or 0) for row in behavior_rows
        if row.get("accuracy") is not None)
    behavior_margin_n = sum(
        int(row.get("n") or 0) for row in behavior_rows
        if row.get("mean_margin") is not None)
    behavior_eval["overall"] = {
        "status": (
            "ready" if behavior_rows and all(row.get("status") == "ready" for row in behavior_rows)
            else "partial" if behavior_rows else "missing_dataset"),
        "n": behavior_n,
        "accuracy": (
            sum(float(row["accuracy"]) * int(row.get("n") or 0)
                for row in behavior_rows if row.get("accuracy") is not None)
            / behavior_accuracy_n if behavior_accuracy_n else None),
        "mean_margin": (
            sum(float(row["mean_margin"]) * int(row.get("n") or 0)
                for row in behavior_rows if row.get("mean_margin") is not None)
            / behavior_margin_n if behavior_margin_n else None),
        "known_correct": sum(int(row.get("known_correct") or 0) for row in behavior_rows),
    }
    function_dataset_rows = list(function_reuse.values())
    function_reuse["overall"] = {
        "actual_path": _mean(row.get("actual_path") for row in function_dataset_rows),
        "random_null": _mean(row.get("random_null") for row in function_dataset_rows),
        "effect": _mean(row.get("effect") for row in function_dataset_rows),
        "ci95": _bootstrap_ci(
            _finite(row.get("effect") for row in function_dataset_rows),
            int(getattr(ctx.args, "operator_analysis_seed", 4171)) + 900),
        "datasets": len(function_dataset_rows),
    }
    route_dataset_rows = list(route_specificity.values())
    route_specificity["overall"] = {
        "within": _mean(row.get("within") for row in route_dataset_rows),
        "between_null": _mean(row.get("between_null") for row in route_dataset_rows),
        "specificity_gap": _mean(row.get("specificity_gap") for row in route_dataset_rows),
        "captured_mass": _mean(row.get("captured_mass") for row in route_dataset_rows),
        "datasets": len(route_dataset_rows),
    }
    causal_dataset_rows = list(causal_specificity.values())
    causal_specificity["overall"] = {
        "status": (
            "ready" if causal_dataset_rows and all(row.get("status") == "ready" for row in causal_dataset_rows)
            else "partial" if causal_dataset_rows else "missing_dataset"),
        "selected_minus_control_effect": _mean(
            row.get("selected_minus_control_effect") for row in causal_dataset_rows),
        "datasets": len(causal_dataset_rows),
    }
    item_results: Dict[str, Any] = {
        "operator_dataset_manifest": {
            "status": "ready", "artifact": build.manifest_path,
            "sample_count": sum(row["rows"] for row in validation["datasets"].values())},
        "operator_behavior_eval": {
            "status": behavior_eval["overall"]["status"],
            "artifact": join_path(output_root, "summaries", "operator_behavior_eval.json"),
            "sample_count": behavior_eval["overall"]["n"]},
        "operator_function_reuse": {
            "status": "ready" if any(row.get("effect") is not None for row in function_dataset_rows) else "insufficient_evidence",
            "artifact": join_path(output_root, "summaries", "operator_function_reuse.json"),
            "sample_count": sum(int(dataset_results[d].get("route", {}).get("valid_logical_pairs") or 0) for d in dataset_results)},
        "operator_route_specificity": {
            "status": "ready" if any(row.get("specificity_gap") is not None for row in route_dataset_rows) else "insufficient_evidence",
            "artifact": join_path(output_root, "summaries", "operator_route_specificity.json"),
            "sample_count": sum(int(dataset_results[d].get("route", {}).get("valid_logical_pairs") or 0) for d in dataset_results)},
        "operator_causal_specificity": {
            "status": causal_specificity["overall"]["status"],
            "artifact": join_path(output_root, "summaries", "operator_causal_specificity.json"),
            "sample_count": sum(int(dataset_results[d].get("causal", {}).get("jobs") or 0) for d in dataset_results)},
    }
    for dataset_id, item in DATASET_ITEM.items():
        if item not in item_results:
            relevant = [row for key, row in dataset_results.items() if DATASET_ITEM[key] == item]
            item_results[item] = {
                "status": (
                    "failed" if any(row.get("status") == "failed" for row in relevant)
                    else "ready" if relevant and all(row.get("status") == "ready" for row in relevant)
                    else "partial" if relevant else "missing_dataset"),
                "behavior_artifact": [row.get("artifacts", {}).get("behavior") for row in relevant],
                "trace_artifact": [row.get("artifacts", {}).get("trace") for row in relevant],
                "causal_artifact": [row.get("artifacts", {}).get("causal") for row in relevant],
                "sample_count": sum(int(row.get("behavior", {}).get("n") or 0) for row in relevant),
            }
    overall_status = (
        "ready" if dataset_results and all(row.get("status") == "ready" for row in dataset_results.values())
        else "partial" if dataset_results else "ready")
    result = {
        "status": overall_status, "analysis_version": ANALYSIS_VERSION,
        "profile": profile["profile"], "build_id": build.build_id,
        "build_hash": build.manifest_hash, "config_hash": config_hash,
        "output_root": output_root, "dataset_manifest": validation,
        "datasets": dataset_results, "behavior_eval": behavior_eval,
        "function_reuse": function_reuse, "route_specificity": route_specificity,
        "causal_specificity": causal_specificity, "items": item_results,
        "limitations": [
            "cross_function_control is skipped when the current trace has no signed coefficient for the cross-group operator",
            "bootstrap resamples logical examples and caps very large datasets at a deterministic 4096-example bootstrap frame",
        ],
    }
    item_results["operator_analysis_summary"] = {
        "status": overall_status,
        "artifact": join_path(output_root, "summaries", "operator_analysis_summary.json"),
        "sample_count": sum(int(row.get("behavior", {}).get("n") or 0) for row in dataset_results.values())}
    if ctx.is_primary:
        summary_root = join_path(output_root, "summaries")
        write_json_atomic(join_path(summary_root, "operator_behavior_eval.json"), behavior_eval)
        write_json_atomic(join_path(summary_root, "operator_function_reuse.json"), function_reuse)
        write_json_atomic(join_path(summary_root, "operator_route_specificity.json"), route_specificity)
        write_json_atomic(join_path(summary_root, "operator_causal_specificity.json"), causal_specificity)
        write_json_atomic(join_path(summary_root, "operator_analysis_summary.json"), result)
        write_text_atomic(join_path(summary_root, "operator_analysis_summary.md"), _analysis_markdown(result))
        metric_rows = [
            {"dataset": dataset, "metric": "behavior_accuracy", "value": row.get("accuracy")}
            for dataset, row in behavior_eval.items()]
        write_jsonl_atomic(join_path(output_root, "metrics.jsonl"), metric_rows)
        analysis_manifest.update({
            "status": "complete", "completed_at": utc_now(),
            "result": join_path(summary_root, "operator_analysis_summary.json"),
            "items": item_results,
            "jobs": {
                dataset: {"status": row.get("status"),
                          "sample_count": (row.get("behavior") or {}).get("n", 0)}
                for dataset, row in dataset_results.items()},
        })
        write_json_atomic(manifest_path, analysis_manifest)
        _print_console_summary(result)
    return result
