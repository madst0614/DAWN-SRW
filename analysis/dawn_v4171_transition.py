"""Item-driven v4171 transition analysis built on the existing train-analysis path.

The module deliberately keeps model execution in analysis-only JITs.  It never
changes training outputs or checkpoint structure, and it only transfers target
token vectors plus sparse top-k operator records to the host.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import (
    AnalysisContext,
    git_info,
    maybe_load_tokenizer,
)
from analysis.dawn_analysis_storage import (
    exists,
    open_path,
    read_json,
    read_jsonl,
    read_npz,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_npz_atomic,
)
from analysis.dawn_analysis_trace import (
    TRACE_FIELDS,
    TRACE_POOLS,
    TRANSITION_CANDIDATE_FIELDS,
    TRANSITION_CANDIDATE_STRATEGIES,
    topk_trace_forward,
)
V4171_MODEL_VERSION = "spatial-r1-v4.1.7.1"
ANALYSIS_SCHEMA_VERSION = 2
ANALYSIS_CODE_SCHEMA_HASH = hashlib.sha256(
    b"v4171-operator-family-analysis-schema-2").hexdigest()
DEFAULT_TRANSITION_PROMPT_SET = str(
    Path(__file__).resolve().parent / "prompts" / "v4171_transition_pairs.jsonl"
)
CORE_TRANSITION_ITEMS = (
    "global_router_audit",
    "trajectory_trace",
    "context_divergence",
    "state_transition_decoupling",
    "causal_intervention",
    "causal_recovery_trace",
    "operator_functional_graph",
    "group_causal_intervention",
    "causal_ranking_calibration",
)
ITEM_DEPENDENCIES = {
    "causal_recovery_trace": ("causal_intervention",),
    "operator_functional_graph": ("trajectory_trace",),
    "group_causal_intervention": (
        "operator_functional_graph", "causal_intervention"),
    "causal_ranking_calibration": (
        "causal_intervention", "causal_recovery_trace"),
}
PAIR_CAPTURE_THRESHOLD = 0.95
ADAPTIVE_CAPTURE_TIERS = {
    "qk": (512, 1024, 2048),
    "v": (2048, 4096, 8192),
    "rst": (4096, 8192),
}


def _unit_rows(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1.0e-12)


def rw_functional_similarity(
        read_left: Any, write_left: Any,
        read_right: Any, write_right: Any) -> np.ndarray:
    """Frobenius cosine of normalized rank-1 read/write operators."""
    read_cos = np.sum(
        _unit_rows(read_left) * _unit_rows(read_right), axis=-1)
    write_cos = np.sum(
        _unit_rows(write_left) * _unit_rows(write_right), axis=-1)
    return np.asarray(read_cos * write_cos)


def classify_function_address_pairs(
        functional_similarity: Sequence[float],
        address_similarity: Sequence[float],
        *, high_quantile: float = 0.90,
        low_quantile: float = 0.50) -> Dict[str, Any]:
    """Classify pair similarities with thresholds learned from observations."""
    functional = np.asarray(functional_similarity, dtype=np.float64)
    address = np.asarray(address_similarity, dtype=np.float64)
    if functional.shape != address.shape:
        raise ValueError("functional and address similarities must align")
    valid = np.isfinite(functional) & np.isfinite(address)
    if not np.any(valid):
        return {
            "thresholds": {}, "labels": [None] * int(functional.size),
            "counts": {},
        }
    f_high = float(np.quantile(functional[valid], high_quantile))
    f_low = float(np.quantile(functional[valid], low_quantile))
    a_high = float(np.quantile(address[valid], high_quantile))
    a_low = float(np.quantile(address[valid], low_quantile))
    labels: List[Optional[str]] = []
    counts: Dict[str, int] = defaultdict(int)
    for f_value, a_value in zip(functional.ravel(), address.ravel()):
        if not (math.isfinite(float(f_value)) and math.isfinite(float(a_value))):
            labels.append(None)
            continue
        function_band = (
            "high" if float(f_value) >= f_high else
            "low" if float(f_value) <= f_low else None)
        address_band = (
            "high" if float(a_value) >= a_high else
            "low" if float(a_value) <= a_low else None)
        if function_band is None or address_band is None:
            labels.append(None)
            counts["mid_similarity_unclassified"] += 1
            continue
        label = (
            f"function_{function_band}_address_{address_band}")
        labels.append(label)
        counts[label] += 1
    return {
        "thresholds": {
            "high_quantile": float(high_quantile),
            "low_quantile": float(low_quantile),
            "functional_high": f_high,
            "functional_low": f_low,
            "address_high": a_high,
            "address_low": a_low,
        },
        "labels": labels,
        "counts": dict(counts),
    }


def mutual_neighbor_families(
        neighbor_ids: Mapping[int, Sequence[Any]],
        qualified_edges: Sequence[Tuple[int, int]] = ()) -> List[List[int]]:
    """Build deterministic union-find components from conservative edges."""
    nodes = sorted(int(node) for node in neighbor_ids)
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        if left not in parent or right not in parent:
            return
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            low, high = sorted((root_left, root_right))
            parent[high] = low

    normalized = {
        node: {
            int(value.get("operator_id") if isinstance(value, Mapping) else value)
            for value in values
        }
        for node, values in neighbor_ids.items()
    }
    for left in nodes:
        for right in normalized.get(left, set()):
            if right in normalized and left in normalized[right]:
                union(left, right)
    for left, right in qualified_edges:
        union(int(left), int(right))
    groups: Dict[int, List[int]] = defaultdict(list)
    for node in nodes:
        groups[find(node)].append(node)
    return sorted((sorted(group) for group in groups.values()), key=lambda x: x[0])


def _layer_vector(trace: Mapping[str, Any], key: str) -> np.ndarray:
    value = np.asarray(trace[key], dtype=np.float64)
    if value.ndim == 3:
        value = value[:, 0, :]
    if value.ndim != 2:
        raise ValueError(f"trace {key} must have shape [L,D] or [L,B,D]")
    return value


def compute_causal_recovery_metrics(
        baseline_trace: Mapping[str, Any],
        intervention_trace: Mapping[str, Any], *, route: str,
        target_layer: int, baseline_final_residual: Any,
        intervention_final_residual: Any, baseline_logits: Any,
        intervention_logits: Any, target_position: int) -> Dict[str, Any]:
    """Measure immediate damage and downstream recovery on target vectors."""
    immediate_key = (
        "post_layer_residual" if str(route) == "rst"
        else "post_attention_residual")
    baseline_immediate = _layer_vector(baseline_trace, immediate_key)
    after_immediate = _layer_vector(intervention_trace, immediate_key)
    baseline_post = _layer_vector(baseline_trace, "post_layer_residual")
    after_post = _layer_vector(intervention_trace, "post_layer_residual")
    layer = int(target_layer)
    immediate_delta = float(np.linalg.norm(
        after_immediate[layer] - baseline_immediate[layer]))
    immediate_relative = float(
        immediate_delta
        / max(float(np.linalg.norm(baseline_immediate[layer])), 1.0e-12))
    per_norm = np.linalg.norm(after_post - baseline_post, axis=-1)
    per_relative = per_norm / np.maximum(
        np.linalg.norm(baseline_post, axis=-1), 1.0e-12)
    per_cosine = [
        _cosine(baseline_post[index], after_post[index])
        for index in range(baseline_post.shape[0])
    ]
    baseline_final = np.asarray(baseline_final_residual, dtype=np.float64)
    after_final = np.asarray(intervention_final_residual, dtype=np.float64)
    if baseline_final.ndim > 1:
        baseline_final = baseline_final.reshape(-1, baseline_final.shape[-1])[0]
        after_final = after_final.reshape(-1, after_final.shape[-1])[0]
    final_delta = float(np.linalg.norm(after_final - baseline_final))
    final_relative = float(
        final_delta / max(float(np.linalg.norm(baseline_final)), 1.0e-12))
    denominator = max(immediate_delta, 1.0e-12)
    remaining = per_norm / denominator
    downstream = remaining[layer:]
    maximum_index = layer + int(np.argmax(downstream))
    minimum_index = layer + int(np.argmin(downstream))
    half_indices = np.flatnonzero(downstream <= 0.5)
    base_logits = np.asarray(baseline_logits, dtype=np.float64)
    after_logits = np.asarray(intervention_logits, dtype=np.float64)
    if base_logits.ndim == 3:
        base_logits = base_logits[0]
        after_logits = after_logits[0]
    final_logit_delta = float(np.linalg.norm(
        after_logits[int(target_position)] - base_logits[int(target_position)]))
    return {
        "immediate_state": immediate_key,
        "immediate_delta_norm": immediate_delta,
        "immediate_relative_delta": immediate_relative,
        "per_layer_post_residual_delta_norm": per_norm.tolist(),
        "per_layer_post_residual_delta_relative": per_relative.tolist(),
        "per_layer_post_residual_cosine": per_cosine,
        "final_delta_norm": final_delta,
        "final_relative_delta": final_relative,
        "recovery_ratio_final": final_delta / denominator,
        "maximum_amplification_ratio": float(remaining[maximum_index]),
        "maximum_amplification_layer": maximum_index,
        "minimum_remaining_ratio": float(remaining[minimum_index]),
        "minimum_remaining_layer": minimum_index,
        "first_half_recovery_layer": (
            layer + int(half_indices[0]) if half_indices.size else None),
        "final_logit_delta": final_logit_delta,
        "non_monotonic_path": bool(
            downstream.size >= 3
            and np.any(np.diff(downstream) < 0.0)
            and np.any(np.diff(downstream) > 0.0)),
    }


def pad_group_operator_ids(
        operator_ids: Sequence[int], max_width: int) -> np.ndarray:
    ids = [int(value) for value in operator_ids]
    if int(max_width) <= 0 or len(ids) > int(max_width):
        raise ValueError("operator group exceeds fixed max width")
    if len(set(ids)) != len(ids) or any(value < 0 for value in ids):
        raise ValueError("operator group ids must be unique nonnegative values")
    out = np.full((int(max_width),), -1, dtype=np.int32)
    out[:len(ids)] = ids
    return out


def group_operator_membership_mask(
        global_operator_ids: Any, selected_operator_ids: Any) -> np.ndarray:
    global_ids = np.asarray(global_operator_ids, dtype=np.int32)
    selected = np.asarray(selected_operator_ids, dtype=np.int32)
    if selected.ndim == 1:
        selected = selected[None, :]
    return np.any(
        global_ids[None, :, None] == selected[:, None, :], axis=-1)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    left = np.asarray(xs, dtype=np.float64)
    right = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(valid)) < 2:
        return None
    left_rank, right_rank = _rankdata(left[valid]), _rankdata(right[valid])
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def pairwise_win_rate(
        rows: Sequence[Mapping[str, Any]], left_strategy: str,
        right_strategy: str, value_key: str) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        value = _json_float(row.get(value_key))
        if value is None:
            continue
        grouped[(str(row.get("prompt_id")), str(row.get("pool")))][
            str(row.get("strategy"))].append(abs(value))
    outcomes = []
    for strategies in grouped.values():
        if left_strategy in strategies and right_strategy in strategies:
            outcomes.append(
                float(np.mean(strategies[left_strategy]))
                > float(np.mean(strategies[right_strategy])))
    return {
        "n": len(outcomes),
        "win_rate": float(np.mean(outcomes)) if outcomes else None,
    }


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "row"


def _json_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _analysis_provenance(
        ctx: AnalysisContext, prompt_hash: Optional[str],
        parity: Optional[Mapping[str, Any]] = None,
        cross_graph: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    git_commit = git_info().get("git_commit") or "unknown"
    analysis_config = {
        "transition_max_prompts": getattr(
            ctx.args, "transition_max_prompts", None),
        "trace_seq_len": int(getattr(ctx.args, "trace_seq_len", 128) or 128),
        "transition_topk_qk": int(getattr(
            ctx.args, "transition_topk_qk", 512) or 512),
        "transition_topk_v": int(getattr(
            ctx.args, "transition_topk_v", 2048) or 2048),
        "transition_topk_rst": int(getattr(
            ctx.args, "transition_topk_rst", 4096) or 4096),
        "transition_capture_threshold": float(getattr(
            ctx.args, "transition_capture_threshold", PAIR_CAPTURE_THRESHOLD)
            or PAIR_CAPTURE_THRESHOLD),
        "transition_adaptive_capture": bool(getattr(
            ctx.args, "transition_adaptive_capture", True)),
        "causal_max_prompts": int(getattr(
            ctx.args, "causal_max_prompts", 6) or 6),
        "functional_graph_max_operators": {
            pool: int(getattr(
                ctx.args, f"functional_graph_max_operators_{pool}", 2048)
                or 2048)
            for pool in ("qk", "v", "rst")
        },
        "functional_graph_neighbor_k": int(getattr(
            ctx.args, "functional_graph_neighbor_k", 16) or 16),
        "group_causal_sizes": str(getattr(
            ctx.args, "group_causal_sizes", "1,2,4,8")),
        "group_causal_max_width": int(getattr(
            ctx.args, "group_causal_max_width", 8) or 8),
        "group_causal_max_prompts": getattr(
            ctx.args, "group_causal_max_prompts", None),
    }
    analysis_config_hash = hashlib.sha256(json.dumps(
        analysis_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "model_version": str(ctx.model_cfg.get("model_version", "unknown")),
        "checkpoint_step": ctx.checkpoint_step,
        "checkpoint_path": ctx.checkpoint_path,
        "git_commit": git_commit,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "code_schema_hash": ANALYSIS_CODE_SCHEMA_HASH,
        "analysis_config": analysis_config,
        "analysis_config_hash": analysis_config_hash,
        "prompt_hash": prompt_hash,
        "composition_mode": str(ctx.model_cfg.get("srw_composition_mode")),
        "admission_den_power": _json_float(
            ctx.model_cfg.get("admission_den_power")),
        "d_route": int(ctx.model_cfg.get("d_route", 0)),
        "pool_sizes": {
            "qk": int(ctx.model_cfg.get("n_qk", 0)),
            "v": int(ctx.model_cfg.get("n_v", 0)),
            "rst": int(ctx.model_cfg.get(
                "n_rst", ctx.model_cfg.get("n_know", 0))),
        },
        "canonical_parity_machine_exact": (
            bool(parity.get("machine_exact")) if parity else None),
        "cross_graph_audit_machine_exact": (
            bool(cross_graph.get("machine_exact")) if cross_graph else None),
    }


def _resume_summary_matches(
        ctx: AnalysisContext, summary: Mapping[str, Any],
        prompt_hash: Optional[str]) -> bool:
    expected = _analysis_provenance(ctx, prompt_hash)
    exact_keys = (
        "model_version", "checkpoint_step", "checkpoint_path",
        "analysis_schema_version",
        "code_schema_hash", "analysis_config_hash", "prompt_hash", "composition_mode",
        "admission_den_power", "d_route", "pool_sizes",
    )
    return all(summary.get(key) == expected.get(key) for key in exact_keys)


def _load_resumable_summary(
        ctx: AnalysisContext, filename: str, prompt_hash: Optional[str],
        *, required_keys: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    if not bool(getattr(ctx.args, "resume", True)):
        return None
    path = ctx.store.path(filename)
    if not exists(path):
        return None
    summary = read_json(path, {})
    if not isinstance(summary, dict) or not _resume_summary_matches(
            ctx, summary, prompt_hash):
        return None
    if any(key not in summary for key in required_keys):
        return None
    if str(summary.get("status")) not in ("ready", "partial"):
        return None
    summary["resumed"] = True
    return summary


def _cosine(a: Any, b: Any) -> Optional[float]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or aa.shape != bb.shape:
        return None
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if not math.isfinite(den) or den <= 1.0e-12:
        return None
    value = float(np.dot(aa, bb) / den)
    return value if math.isfinite(value) else None


def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _quantile(values: Iterable[Any], q: float) -> Optional[float]:
    arr = np.asarray(
        [float(v) for v in values if v is not None and math.isfinite(float(v))],
        dtype=np.float64,
    )
    return float(np.quantile(arr, q)) if arr.size else None


def _bootstrap_mean_ci(values: Sequence[float], seed: int, draws: int = 500) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if not arr.size:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(int(seed))
    if arr.size == 1:
        ci = [float(arr[0]), float(arr[0])]
    else:
        means = np.asarray([
            np.mean(rng.choice(arr, size=arr.size, replace=True))
            for _ in range(int(draws))
        ])
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {"mean": float(np.mean(arr)), "ci95": ci, "n": int(arr.size)}


def load_transition_prompt_rows(path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load and validate the controlled JSONL without requiring a tokenizer."""
    rows: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    seen = set()
    required = (
        "prompt_id", "pair_id", "phenomenon", "relation", "text",
        "target_text", "continuation", "control_group",
    )
    with open_path(path, "r") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.encode("utf-8") if isinstance(line, str) else bytes(line)
            digest.update(raw)
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"Invalid transition JSONL at {path}:{line_no}: {exc}") from exc
            missing = [key for key in required if row.get(key) in (None, "")]
            if missing:
                raise ValueError(
                    f"Transition row {path}:{line_no} missing {','.join(missing)}")
            prompt_id = str(row["prompt_id"])
            if prompt_id in seen:
                raise ValueError(f"Duplicate transition prompt_id {prompt_id!r}")
            seen.add(prompt_id)
            row["target_occurrence"] = int(row.get("target_occurrence", 0))
            rows.append(row)
    if not rows:
        raise ValueError(f"Transition prompt set is empty: {path}")
    pair_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        pair_counts[str(row["pair_id"])] += 1
    bad_pairs = sorted(pair_id for pair_id, count in pair_counts.items() if count < 2)
    if bad_pairs:
        raise ValueError(
            "Every transition pair_id needs at least two rows; bad="
            + ",".join(bad_pairs))
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    for pair_id, pair_rows in by_pair.items():
        texts = [str(row["text"]) for row in pair_rows]
        if len(texts) != len(set(texts)):
            raise ValueError(
                f"Transition pair_id={pair_id!r} contains identical text rows")
    return rows, digest.hexdigest()


def _target_char_span(row: Mapping[str, Any]) -> Tuple[int, int]:
    text = str(row["text"])
    target = str(row["target_text"])
    occurrence = int(row.get("target_occurrence", 0))
    starts = [match.start() for match in re.finditer(re.escape(target), text)]
    if occurrence < 0 or occurrence >= len(starts):
        raise ValueError(
            f"prompt_id={row.get('prompt_id')} target={target!r} occurrence={occurrence} "
            f"found={len(starts)}")
    start = int(starts[occurrence])
    return start, start + len(target)


def _tokenize_transition_row(
    tokenizer: Any,
    row: Mapping[str, Any],
    seq_len: int,
) -> Dict[str, Any]:
    start, end = _target_char_span(row)
    encoded = tokenizer(
        str(row["text"]),
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=int(seq_len),
    )
    ids = [int(v) for v in encoded.get("input_ids", [])]
    offsets = [(int(a), int(b)) for a, b in encoded.get("offset_mapping", [])]
    if not ids or len(ids) != len(offsets):
        raise ValueError(
            f"prompt_id={row.get('prompt_id')} tokenizer did not return aligned offsets")
    token_indices = [
        idx for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start and tok_start < end
    ]
    if not token_indices:
        raise ValueError(
            f"prompt_id={row.get('prompt_id')} target span [{start},{end}) did not map to tokens")
    pad_id = int(tokenizer.pad_token_id or 0)
    input_array = np.full((int(seq_len),), pad_id, dtype=np.int32)
    input_array[: len(ids)] = np.asarray(ids, dtype=np.int32)
    continuation_ids = tokenizer.encode(
        str(row.get("continuation") or ""), add_special_tokens=False)
    out = dict(row)
    out.update({
        "target_char_span": [start, end],
        "target_token_indices": token_indices,
        "target_token_ids": [ids[idx] for idx in token_indices],
        "token_ids": ids,
        "offset_mapping": offsets,
        "input_array": input_array,
        "length": len(ids),
        "continuation_token_ids": [int(v) for v in continuation_ids],
    })
    return out


def _validate_tokenized_pairs(prompts: Sequence[Mapping[str, Any]]) -> None:
    by_pair: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        by_pair[str(prompt["pair_id"])].append(prompt)
    for pair_id, pair_rows in by_pair.items():
        for left_idx, left in enumerate(pair_rows):
            for right in pair_rows[left_idx + 1:]:
                if list(left["token_ids"]) == list(right["token_ids"]):
                    raise ValueError(
                        "Transition pair has identical tokenized inputs: "
                        f"pair_id={pair_id} prompt_a={left['prompt_id']} "
                        f"prompt_b={right['prompt_id']} "
                        f"target_a={left['target_token_indices']} "
                        f"target_b={right['target_token_indices']}")


def _flatten_param_paths(tree: Any, prefix: Tuple[str, ...] = ()) -> List[str]:
    if isinstance(tree, Mapping):
        out: List[str] = []
        for key, value in tree.items():
            out.extend(_flatten_param_paths(value, prefix + (str(key),)))
        return out
    return ["/".join(prefix)]


def run_global_router_audit(ctx: AnalysisContext) -> Dict[str, Any]:
    if str(ctx.model_cfg.get("model_version")) != V4171_MODEL_VERSION:
        return {
            "status": "unsupported_for_v4171",
            "reason": f"model_version={ctx.model_cfg.get('model_version')}",
        }
    paths = sorted(_flatten_param_paths(ctx.params))
    router_paths = [path for path in paths if path.startswith("router/")]
    pool_paths = [path for path in paths if path.startswith("neuron_pool/")]
    hidden = [
        path for path in paths
        if path.startswith("block_")
        and any(part in path.lower() for part in ("router", "query", "op_key"))
    ]
    required_router = (
        "router/proj_attn/kernel",
        "router/proj_rst/kernel",
        "router/raw_tau_attn/kernel",
        "router/raw_tau_rst/kernel",
    )
    required_pool = (
        "neuron_pool/attn_qk_op_key",
        "neuron_pool/attn_qk_read",
        "neuron_pool/attn_qk_write",
        "neuron_pool/attn_v_op_key",
        "neuron_pool/attn_v_read",
        "neuron_pool/attn_v_write",
        "neuron_pool/rst_op_key",
        "neuron_pool/rst_read",
        "neuron_pool/rst_write",
    )
    missing = [path for path in required_router + required_pool if path not in paths]
    if hidden or missing:
        raise RuntimeError(
            "v4171 global router audit failed: "
            f"hidden_layer_router_params={hidden} missing={missing}")
    mcfg = ctx.config.get("model", {})
    result = {
        "status": "ready",
        "router_param_paths": router_paths,
        "router_param_count": len(router_paths),
        "shared_across_layers": True,
        "hidden_layer_router_params": hidden,
        "operator_pool_param_paths": pool_paths,
        "operator_keys_shared": True,
        "operator_rw_shared": True,
        "query_types": {
            "q": "LN1(residual)->global router/proj_attn slice 0",
            "k": "LN1(residual)->global router/proj_attn slice 1",
            "v": "LN1(residual)->global router/proj_attn slice 2",
            "rst": "LN2(residual_after_attention)->global router/proj_rst",
        },
        "pool_sizes": {
            "qk": int(mcfg.get("n_qk", 0)),
            "v": int(mcfg.get("n_v", 0)),
            "rst": int(mcfg.get("n_rst", mcfg.get("n_know", 0))),
        },
        "d_route": int(mcfg.get("d_route", 0)),
        "d_model": int(mcfg.get("d_model", 0)),
        "n_layers": int(mcfg.get("n_layers", 0)),
        "composition_mode": str(mcfg.get("srw_composition_mode", "linear_angular")),
        "admission_den_power": float(mcfg.get("admission_den_power", 1.0)),
    }
    if ctx.is_primary:
        write_json_atomic(ctx.store.path("global_router_audit.json"), result)
    return result


def _trace_pool_arrays(trace: Mapping[str, np.ndarray], pool: str) -> Dict[str, np.ndarray]:
    return {
        field: np.asarray(trace[f"{pool}_{field}"])[:, 0]
        for field in TRACE_FIELDS + TRANSITION_CANDIDATE_FIELDS
        if f"{pool}_{field}" in trace
    }


def _layer_trace_rows(
    prompt: Mapping[str, Any],
    token_index: int,
    trace: Mapping[str, np.ndarray],
    console_topk: int = 8,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_layers = int(np.asarray(trace["residual_before_router"]).shape[0])
    output_key = {
        "q": "srw_feature_q",
        "k": "srw_feature_k",
        "v": "srw_feature_v",
        "rst": "delta_rst",
    }
    output_kind = {
        "q": "attention_srw_feature",
        "k": "attention_srw_feature",
        "v": "attention_srw_feature",
        "rst": "residual_update",
    }
    for layer in range(n_layers):
        state = np.asarray(trace["residual_before_router"])[layer, 0]
        prev_state = (
            np.asarray(trace["residual_before_router"])[layer - 1, 0]
            if layer > 0 else state
        )
        layer_row: Dict[str, Any] = {
            "prompt_id": prompt["prompt_id"],
            "pair_id": prompt["pair_id"],
            "phenomenon": prompt["phenomenon"],
            "relation": prompt["relation"],
            "target_char_span": prompt["target_char_span"],
            "target_token_indices": prompt["target_token_indices"],
            "target_token_index": int(token_index),
            "target_token_id": int(prompt["token_ids"][token_index]),
            "layer": layer,
            "state_norm": float(np.linalg.norm(state)),
            "state_step_norm": float(np.linalg.norm(state - prev_state)),
            "state_cosine_to_previous": _cosine(state, prev_state),
            "attention_update_norm": float(
                np.linalg.norm(np.asarray(trace["delta_attention"])[layer, 0])),
            "rst_update_norm": float(
                np.linalg.norm(np.asarray(trace["delta_rst"])[layer, 0])),
            "pools": {},
            "srw_feature_cosines": {
                "q_k": _cosine(
                    np.asarray(trace["srw_feature_q"])[layer, 0],
                    np.asarray(trace["srw_feature_k"])[layer, 0]),
                "q_v": _cosine(
                    np.asarray(trace["srw_feature_q"])[layer, 0],
                    np.asarray(trace["srw_feature_v"])[layer, 0]),
                "k_v": _cosine(
                    np.asarray(trace["srw_feature_k"])[layer, 0],
                    np.asarray(trace["srw_feature_v"])[layer, 0]),
            },
            "residual_update_cosine_attention_rst": _cosine(
                np.asarray(trace["delta_attention"])[layer, 0],
                np.asarray(trace["delta_rst"])[layer, 0]),
        }
        for pool in TRACE_POOLS:
            pdata = _trace_pool_arrays(trace, pool)
            ids = np.asarray(pdata["top_idx"])[layer]
            vals = np.asarray(pdata["top_val"])[layer]
            query_key = f"query_{pool}"
            query = np.asarray(trace[query_key])[layer, 0]
            prev_query = np.asarray(trace[query_key])[layer - 1, 0] if layer > 0 else query
            output = np.asarray(trace[output_key[pool]])[layer, 0]
            prev_output = (
                np.asarray(trace[output_key[pool]])[layer - 1, 0]
                if layer > 0 else output)
            prev_ids = (
                set(int(v) for v in np.asarray(pdata["top_idx"])[layer - 1])
                if layer > 0 else set(int(v) for v in ids))
            current_ids = set(int(v) for v in ids)
            overlap_union = len(prev_ids | current_ids)
            layer_row["pools"][pool] = {
                "output_kind": output_kind[pool],
                "query_norm": float(np.linalg.norm(query)),
                "query_cosine_to_previous": _cosine(query, prev_query),
                "tau": _json_float(np.asarray(pdata["tau"])[layer]),
                "active_count": int(np.asarray(pdata["active_count"])[layer]),
                "effective_count": _json_float(np.asarray(pdata["effective_count"])[layer]),
                "gate_mass": _json_float(np.asarray(pdata["mass"])[layer]),
                "captured_gate_mass": _json_float(np.asarray(pdata["captured_mass"])[layer]),
                "top1_mass_fraction": _json_float(np.asarray(pdata["top1_frac"])[layer]),
                "output_norm": float(np.linalg.norm(output)),
                "output_residual_norm_ratio": float(
                    np.linalg.norm(output) / max(float(np.linalg.norm(state)), 1.0e-12)),
                "output_cosine_to_previous": _cosine(output, prev_output),
                "top_operator_overlap_to_previous": (
                    float(len(prev_ids & current_ids) / overlap_union)
                    if overlap_union else None),
                "top_operator_ids": [int(v) for v in ids[:console_topk]],
                "execution_weights": [float(v) for v in vals[:console_topk]],
                "admission_weights": [
                    float(v) for v in np.asarray(pdata["top_admission"])[layer][:console_topk]
                ],
                "rho": [float(v) for v in np.asarray(pdata["top_rho"])[layer][:console_topk]],
                "read_response": [
                    float(v) for v in np.asarray(pdata["top_read"])[layer][:console_topk]
                ],
                "post_denominator_coefficient": [
                    float(v) for v in np.asarray(pdata["top_coefficient"])[layer][:console_topk]
                ],
            }
        rows.append(layer_row)
    return rows


def _trace_internal_record(
    prompt: Dict[str, Any],
    token_index: int,
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    trace_np = {key: np.asarray(value) for key, value in trace.items()}
    return {
        "prompt": prompt,
        "target_token_index": int(token_index),
        "trace": trace_np,
        "rows": _layer_trace_rows(prompt, token_index, trace_np),
    }


def run_transition_trace_cache(ctx: AnalysisContext) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    args = ctx.args
    prompt_set = str(
        getattr(args, "transition_prompt_set", None) or DEFAULT_TRANSITION_PROMPT_SET)
    rows, prompt_hash = load_transition_prompt_rows(prompt_set)
    max_prompts = getattr(args, "transition_max_prompts", None)
    if max_prompts is not None:
        rows = rows[: max(1, int(max_prompts))]
    tokenizer = maybe_load_tokenizer(local_only=True)
    if tokenizer is None:
        raise RuntimeError(
            "v4171 transition analysis requires the cached bert-base-uncased fast tokenizer")
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("v4171 transition analysis requires a fast tokenizer with offsets")
    seq_len = int(getattr(args, "trace_seq_len", 128) or 128)
    topk_qk = int(getattr(args, "transition_topk_qk", 512) or 512)
    topk_v = int(getattr(args, "transition_topk_v", 2048) or 2048)
    topk_rst = int(getattr(args, "transition_topk_rst", 4096) or 4096)
    capture_threshold = float(getattr(
        args, "transition_capture_threshold", PAIR_CAPTURE_THRESHOLD)
        or PAIR_CAPTURE_THRESHOLD)
    adaptive_capture = bool(getattr(
        args, "transition_adaptive_capture", True))
    prompts = [_tokenize_transition_row(tokenizer, row, seq_len) for row in rows]
    _validate_tokenized_pairs(prompts)
    resumed_summary = _load_resumable_summary(
        ctx, "transition_trace_summary.json", prompt_hash,
        required_keys=("num_prompts", "captured_mass_by_pool"))
    if resumed_summary is not None:
        resumed_records = _load_transition_trace_records(
            ctx, prompts, resumed_summary, capture_threshold)
        if resumed_records is not None:
            if ctx.is_primary:
                print(
                    "TRANSITION_TRACE RESUME "
                    f"prompts={len(prompts)} schema={ANALYSIS_SCHEMA_VERSION}",
                    flush=True)
            return resumed_summary, resumed_records
    candidate_seed = int(ctx.config.get("seed", 0))
    data_replicas = max(1, int(ctx.mesh.shape["data"]))

    trace_fns: Dict[Tuple[int, int, int], Any] = {}

    def trace_function(tops: Tuple[int, int, int]):
        if tops not in trace_fns:
            qk_value, v_value, rst_value = tops

            def trace_step(p, x, t):
                trace = topk_trace_forward(
                    p,
                    ctx.model_cfg,
                    x,
                    topk_qk=qk_value,
                    topk_v=v_value,
                    topk_rst=rst_value,
                    target_positions=t,
                    candidate_seed=candidate_seed,
                    production_srw_fns=ctx.sharded_fns,
                )
                return jax.tree.map(
                    lambda value: (
                        value[:, :1]
                        if value.ndim >= 2 and value.shape[1] == data_replicas
                        else value),
                    trace,
                )

            trace_fns[tops] = jax.jit(trace_step)
        return trace_fns[tops]
    internal: List[Dict[str, Any]] = []
    jsonl_rows: List[Dict[str, Any]] = []
    npz_payload: Dict[str, np.ndarray] = {}
    started = time.time()
    vector_keys = (
        "residual_before_router", "query_q", "query_k", "query_v",
        "query_rst", "srw_feature_q", "srw_feature_k", "srw_feature_v",
        "delta_attention", "delta_rst", "residual_after_update",
    )
    for prompt_idx, prompt in enumerate(prompts):
        prompt_records: List[Dict[str, Any]] = []
        for subtoken_idx, token_index in enumerate(prompt["target_token_indices"]):
            input_ids = jax.device_put(jnp.asarray(np.repeat(
                prompt["input_array"][None, :],
                data_replicas,
                axis=0,
            ), dtype=jnp.int32), ctx.data_sharding)
            target = jax.device_put(
                jnp.full(
                    (data_replicas,), int(token_index), dtype=jnp.int32),
                NamedSharding(ctx.mesh, P("data")),
            )
            initial_tops = (topk_qk, topk_v, topk_rst)
            trace = jax.device_get(
                trace_function(initial_tops)(ctx.params, input_ids, target))
            capture: Dict[str, Dict[str, Any]] = {}
            for pool in TRACE_POOLS:
                before = np.asarray(
                    trace[f"{pool}_captured_mass"])[:, 0].astype(np.float64)
                capture[pool] = {
                    "topk": topk_qk if pool in ("q", "k") else (
                        topk_v if pool == "v" else topk_rst),
                    "retry_count": 0,
                    "before": before.copy(),
                    "after": before.copy(),
                }
            if adaptive_capture:
                family_routes = {
                    "qk": ("q", "k"), "v": ("v",), "rst": ("rst",),
                }
                current_tops = {
                    "qk": topk_qk, "v": topk_v, "rst": topk_rst,
                }
                for family, routes in family_routes.items():
                    if all(np.all(capture[route]["after"] >= capture_threshold)
                           for route in routes):
                        continue
                    for next_topk in _capture_tiers(
                            current_tops[family], family)[1:]:
                        retry_tops = (
                            next_topk if family == "qk" else current_tops["qk"],
                            next_topk if family == "v" else current_tops["v"],
                            next_topk if family == "rst" else current_tops["rst"],
                        )
                        retry = jax.device_get(
                            trace_function(retry_tops)(
                                ctx.params, input_ids, target))
                        accepted = False
                        for route in routes:
                            old_capture = np.asarray(
                                capture[route]["after"], dtype=np.float64)
                            new_capture = np.asarray(
                                retry[f"{route}_captured_mass"]
                            )[:, 0].astype(np.float64)
                            improves = bool(
                                np.all(new_capture >= old_capture)
                                and np.any(new_capture > old_capture))
                            if not improves:
                                continue
                            for field in TRACE_FIELDS:
                                key = f"{route}_{field}"
                                trace[key] = retry[key]
                            capture[route]["topk"] = int(next_topk)
                            capture[route]["retry_count"] += 1
                            capture[route]["after"] = new_capture
                            accepted = True
                        current_tops[family] = int(next_topk)
                        if all(np.all(capture[route]["after"] >= capture_threshold)
                               for route in routes):
                            break
                        if not accepted:
                            break
            record = _trace_internal_record(prompt, int(token_index), trace)
            _decorate_capture_rows(record, capture, capture_threshold)
            internal.append(record)
            prompt_records.append(record)
            for row in record["rows"]:
                row["record_type"] = "subtoken"
                row["span_subtoken_index"] = subtoken_idx
                row["span_subtoken_count"] = len(prompt["target_token_indices"])
                row["is_span_last_token"] = (
                    subtoken_idx == len(prompt["target_token_indices"]) - 1)
                jsonl_rows.append(row)
            prefix = f"p{prompt_idx:04d}_t{subtoken_idx:02d}"
            for key, value in record["trace"].items():
                npz_payload[f"{prefix}__trace__{key}"] = np.asarray(value)
            for pool, metadata in record["capture"].items():
                capture_prefix = f"{prefix}__capture__{pool}__"
                npz_payload[f"{capture_prefix}before"] = np.asarray(
                    metadata["before"])
                npz_payload[f"{capture_prefix}after"] = np.asarray(
                    metadata["after"])
                npz_payload[f"{capture_prefix}topk"] = np.asarray(
                    metadata["topk"], dtype=np.int32)
                npz_payload[f"{capture_prefix}retry_count"] = np.asarray(
                    metadata["retry_count"], dtype=np.int32)
            for key in vector_keys:
                npz_payload[f"{prefix}_{key}"] = np.asarray(record["trace"][key])[:, 0]
        for key in vector_keys:
            span_values = np.stack([
                np.asarray(record["trace"][key])[:, 0]
                for record in prompt_records
            ], axis=0)
            npz_payload[f"p{prompt_idx:04d}_span_mean_{key}"] = np.mean(
                span_values, axis=0)
            npz_payload[f"p{prompt_idx:04d}_span_last_{key}"] = span_values[-1]
        if ctx.is_primary:
            print(
                "TRANSITION_TRACE "
                f"prompt={prompt_idx + 1:03d}/{len(prompts):03d} "
                f"id={prompt['prompt_id']} tokens={prompt['target_token_indices']}",
                flush=True,
            )
    captured_by_pool: Dict[str, List[float]] = {pool: [] for pool in TRACE_POOLS}
    for record in internal:
        for pool in TRACE_POOLS:
            captured_by_pool[pool].extend(
                np.asarray(record["trace"][f"{pool}_captured_mass"])[:, 0].tolist())
    captured = [value for values in captured_by_pool.values() for value in values]
    retry_rows = 0
    recovered_rows = 0
    remaining_low_capture_rows = 0
    for record in internal:
        for pool in TRACE_POOLS:
            metadata = record["capture"][pool]
            before = np.asarray(metadata["before"])
            after = np.asarray(metadata["after"])
            if int(metadata["retry_count"]) > 0:
                retry_rows += int(np.sum(before < capture_threshold))
            recovered_rows += int(np.sum(
                (before < capture_threshold) & (after >= capture_threshold)))
            remaining_low_capture_rows += int(np.sum(after < capture_threshold))
    captured_summary = {
        pool: {
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "p10": float(np.quantile(values, 0.10)) if values else None,
        }
        for pool, values in captured_by_pool.items()
    }
    summary = {
        "status": "partial" if remaining_low_capture_rows else "ready",
        "prompt_set": prompt_set,
        "prompt_set_hash": prompt_hash,
        "num_prompts": len(prompts),
        "num_target_subtokens": len(internal),
        "trace_topk": {"qk": topk_qk, "v": topk_v, "rst": topk_rst},
        "span_aggregation": "per-subtoken JSONL plus span_mean/span_last vectors in NPZ",
        "captured_mass": {
            "mean": float(np.mean(captured)) if captured else None,
            "min": float(np.min(captured)) if captured else None,
            "p10": float(np.quantile(captured, 0.10)) if captured else None,
        },
        "captured_mass_by_pool": captured_summary,
        "captured_mass_valid_threshold": capture_threshold,
        "adaptive_capture": {
            "enabled": adaptive_capture,
            "threshold": capture_threshold,
            "tiers": {
                key: _capture_tiers(
                    {"qk": topk_qk, "v": topk_v, "rst": topk_rst}[key],
                    key)
                for key in ("qk", "v", "rst")
            },
            "retry_rows": retry_rows,
            "recovered_rows": recovered_rows,
            "remaining_low_capture_rows": remaining_low_capture_rows,
            "final_capture_by_pool": captured_summary,
        },
        "sec": time.time() - started,
        "artifacts": {
            "trajectory_traces": ctx.store.path("trajectory_traces.jsonl"),
            "transition_trace_cache": ctx.store.path("transition_trace_cache.npz"),
        },
    }
    if (summary["captured_mass"]["min"] is not None
            and summary["captured_mass"]["min"] < capture_threshold):
        summary["captured_mass_warning"] = (
            "Low-capture rows are marked partial and excluded from pair metrics")
    if ctx.is_primary:
        write_jsonl_atomic(ctx.store.path("trajectory_traces.jsonl"), jsonl_rows)
        write_npz_atomic(ctx.store.path("transition_trace_cache.npz"), **npz_payload)
        write_json_atomic(ctx.store.path("transition_trace_summary.json"), summary)
    return summary, internal


def _primary_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use the last WordPiece for pair-level comparisons, while retaining all traces on disk."""
    by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_prompt[str(record["prompt"]["prompt_id"])].append(record)
    return [
        max(rows, key=lambda row: int(row["target_token_index"]))
        for _, rows in sorted(by_prompt.items())
    ]


def _sparse_similarity(
    ids_a: np.ndarray,
    weights_a: np.ndarray,
    ids_b: np.ndarray,
    weights_b: np.ndarray,
) -> Dict[str, Any]:
    a: Dict[int, float] = defaultdict(float)
    b: Dict[int, float] = defaultdict(float)
    for idx, weight in zip(np.asarray(ids_a).reshape(-1), np.asarray(weights_a).reshape(-1)):
        a[int(idx)] += max(0.0, float(weight))
    for idx, weight in zip(np.asarray(ids_b).reshape(-1), np.asarray(weights_b).reshape(-1)):
        b[int(idx)] += max(0.0, float(weight))
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    keys = sorted(keys_a | keys_b)
    if not keys:
        return {
            "gate_cosine": None,
            "weighted_jaccard": None,
            "intersection": 0,
            "union": 0,
        }
    va = np.asarray([a.get(key, 0.0) for key in keys], dtype=np.float64)
    vb = np.asarray([b.get(key, 0.0) for key in keys], dtype=np.float64)
    den = float(np.sum(np.maximum(va, vb)))
    return {
        "gate_cosine": _cosine(va, vb),
        "weighted_jaccard": (
            float(np.sum(np.minimum(va, vb)) / den) if den > 1.0e-12 else None),
        "intersection": len(keys_a & keys_b),
        "union": len(keys_a | keys_b),
    }


def _pair_layer_rows(
    record_a: Dict[str, Any],
    record_b: Dict[str, Any],
    *,
    pair_type: Optional[str] = None,
    is_null: bool = False,
) -> List[Dict[str, Any]]:
    prompt_a = record_a["prompt"]
    prompt_b = record_b["prompt"]
    trace_a = record_a["trace"]
    trace_b = record_b["trace"]
    n_layers = int(np.asarray(trace_a["residual_before_router"]).shape[0])
    rows: List[Dict[str, Any]] = []
    query_key = {pool: f"query_{pool}" for pool in TRACE_POOLS}
    update_key = {
        "q": "srw_feature_q",
        "k": "srw_feature_k",
        "v": "srw_feature_v",
        "rst": "delta_rst",
    }
    update_kind = {
        "q": "srw_feature",
        "k": "srw_feature",
        "v": "srw_feature",
        "rst": "residual_update",
    }
    for layer in range(n_layers):
        state_a = np.asarray(trace_a["residual_before_router"])[layer, 0]
        state_b = np.asarray(trace_b["residual_before_router"])[layer, 0]
        state_cos = _cosine(state_a, state_b)
        for pool in TRACE_POOLS:
            ids_a = np.asarray(trace_a[f"{pool}_top_idx"])[layer, 0]
            ids_b = np.asarray(trace_b[f"{pool}_top_idx"])[layer, 0]
            weights_a = np.asarray(trace_a[f"{pool}_top_val"])[layer, 0]
            weights_b = np.asarray(trace_b[f"{pool}_top_val"])[layer, 0]
            sparse = _sparse_similarity(ids_a, weights_a, ids_b, weights_b)
            query_cos = _cosine(
                np.asarray(trace_a[query_key[pool]])[layer, 0],
                np.asarray(trace_b[query_key[pool]])[layer, 0],
            )
            update_a = np.asarray(trace_a[update_key[pool]])[layer, 0]
            update_b = np.asarray(trace_b[update_key[pool]])[layer, 0]
            delta_cos = _cosine(update_a, update_b)
            delta_den = max(float(np.linalg.norm(update_a)), 1.0e-12)
            delta_rel = float(np.linalg.norm(update_a - update_b) / delta_den)
            captured_a = float(np.asarray(trace_a[f"{pool}_captured_mass"])[layer, 0])
            captured_b = float(np.asarray(trace_b[f"{pool}_captured_mass"])[layer, 0])
            routing_similarity = sparse["weighted_jaccard"]
            capture_threshold = max(
                float(record_a.get(
                    "capture_threshold", PAIR_CAPTURE_THRESHOLD)),
                float(record_b.get(
                    "capture_threshold", PAIR_CAPTURE_THRESHOLD)))
            capture_valid = (
                captured_a >= capture_threshold
                and captured_b >= capture_threshold)
            metric_valid = bool(
                capture_valid
                and routing_similarity is not None
                and delta_cos is not None)
            invalid_reason = None
            if not capture_valid:
                invalid_reason = "low_captured_mass"
            elif routing_similarity is None:
                invalid_reason = "missing_routing_similarity"
            elif delta_cos is None:
                invalid_reason = "missing_transition_similarity"
            path_similarity = (
                float(np.mean([routing_similarity, delta_cos]))
                if metric_valid else None)
            rows.append({
                "pair_id": str(prompt_a.get("pair_id")),
                "prompt_a": str(prompt_a.get("prompt_id")),
                "prompt_b": str(prompt_b.get("prompt_id")),
                "pair_type": str(pair_type or prompt_a.get("relation")),
                "phenomenon": str(prompt_a.get("phenomenon")),
                "is_random_null": bool(is_null),
                "layer": layer,
                "pool": pool,
                "state_similarity": state_cos,
                "query_similarity": query_cos,
                "gate_similarity": sparse["gate_cosine"],
                "weighted_jaccard": sparse["weighted_jaccard"],
                "routing_similarity": routing_similarity,
                "active_intersection": sparse["intersection"],
                "active_union": sparse["union"],
                "delta_similarity": delta_cos,
                "transition_similarity": delta_cos,
                "delta_relative_error": delta_rel,
                "path_similarity": path_similarity,
                "trajectory_similarity": path_similarity,
                "captured_mass_a": captured_a,
                "captured_mass_b": captured_b,
                "metric_valid": metric_valid,
                "invalid_reason": invalid_reason,
                "gate_similarity_exact": False,
                "update_kind": update_kind[pool],
            })
    return rows


def _context_divergence_summary(
    pair_rows: Sequence[Dict[str, Any]],
    null_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    contextual_relations = {
        "same_surface_different_function",
        "same_lexical_different_function",
    }
    for row in pair_rows:
        if row.get("is_random_null") or row.get("pair_type") not in contextual_relations:
            continue
        groups[(str(row["pair_id"]), str(row["pool"]))].append(row)
    null_by_pool_layer: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    null_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in null_rows:
        null_groups[(str(row["pair_id"]), str(row["pool"]))].append(row)
    for rows in null_groups.values():
        valid = sorted(
            (row for row in rows if row.get("metric_valid")),
            key=lambda row: int(row["layer"]),
        )
        early = [row for row in valid if int(row["layer"]) in (0, 1, 2)]
        baseline = (
            float(np.mean([float(row["routing_similarity"]) for row in early]))
            if early else None)
        if baseline is None:
            continue
        for row in valid:
            null_by_pool_layer[(str(row["pool"]), int(row["layer"]))].append(
                baseline - float(row["routing_similarity"]))

    def consecutive_start(rows: Sequence[Dict[str, Any]], key: str) -> Optional[int]:
        for left, right in zip(rows, rows[1:]):
            if (
                bool(left.get(key))
                and bool(right.get(key))
                and int(right["layer"]) == int(left["layer"]) + 1
            ):
                return int(left["layer"])
        return None

    summaries = []
    for (pair_id, pool), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: int(row["layer"]))
        valid = [row for row in rows if row.get("metric_valid")]
        early = [row for row in valid if int(row["layer"]) in (0, 1, 2)]
        baseline = (
            float(np.mean([float(row["routing_similarity"]) for row in early]))
            if early else None)
        evidence: List[Dict[str, Any]] = []
        if baseline is not None:
            for row in valid:
                layer = int(row["layer"])
                null_dist = null_by_pool_layer.get((pool, layer), [])
                drop = baseline - float(row["routing_similarity"])
                null_mean = (
                    float(np.mean(null_dist)) if null_dist else None)
                null_p95 = _quantile(null_dist, 0.95)
                null_p75 = _quantile(null_dist, 0.75)
                evidence.append({
                    "layer": layer,
                    "routing_similarity": float(row["routing_similarity"]),
                    "actual_drop": drop,
                    "null_mean": null_mean,
                    "null_p95": null_p95,
                    "null_p75": null_p75,
                    "above_null_p95": (
                        null_p95 is not None and drop > float(null_p95)),
                    "below_null_p75": (
                        null_p75 is not None and drop < float(null_p75)),
                })
        first_layer = consecutive_start(evidence, "above_null_p95")
        supported = [
            row for row in evidence if row.get("null_mean") is not None]
        max_row = (
            max(
                supported,
                key=lambda row: float(row["actual_drop"]) - float(row["null_mean"]),
            )
            if supported else None)
        n_layers = max((int(row["layer"]) for row in rows), default=-1) + 1
        late_start = max(
            int(math.ceil(2.0 * n_layers / 3.0)),
            (first_layer + 1) if first_layer is not None else n_layers,
        )
        late = [row for row in evidence if int(row["layer"]) >= late_start]
        reconvergence_layer = (
            consecutive_start(late, "below_null_p75")
            if first_layer is not None else None)
        evidence_status = (
            "significant_divergence" if first_layer is not None else
            "no_significant_divergence" if supported else
            "insufficient_evidence"
        )
        summaries.append({
            "pair_id": pair_id,
            "pool": pool,
            "status": evidence_status,
            "early_routing_baseline_layers": [0, 1, 2],
            "early_routing_baseline": baseline,
            "first_divergence_layer": first_layer,
            "maximum_divergence_layer": int(max_row["layer"]) if max_row else None,
            "maximum_divergence_evidence": (
                float(max_row["actual_drop"] - max_row["null_mean"])
                if max_row else None),
            "late_reconvergence": reconvergence_layer is not None,
            "late_reconvergence_layer": reconvergence_layer,
            "null_rule": {
                "first": "actual_drop_gt_null_p95_for_2_consecutive_valid_layers",
                "maximum": "max_actual_drop_minus_null_mean",
                "late_reconvergence": "actual_drop_lt_null_p75_for_2_consecutive_late_layers",
            },
            "mean_state_similarity": _json_float(np.mean([
                row["state_similarity"] for row in valid if row.get("state_similarity") is not None
            ])) if valid else None,
            "mean_query_similarity": _json_float(np.mean([
                row["query_similarity"] for row in valid if row.get("query_similarity") is not None
            ])) if valid else None,
            "mean_gate_similarity": _json_float(np.mean([
                row["routing_similarity"] for row in valid if row.get("routing_similarity") is not None
            ])) if valid else None,
            "mean_delta_similarity": _json_float(np.mean([
                row["transition_similarity"] for row in valid if row.get("transition_similarity") is not None
            ])) if valid else None,
            "min_captured_mass": min(
                min(float(row["captured_mass_a"]), float(row["captured_mass_b"]))
                for row in rows),
            "valid_layers": len(valid),
            "excluded_low_captured_layers": sum(
                row.get("invalid_reason") == "low_captured_mass" for row in rows),
            "layer_evidence": evidence,
        })
    statuses = {row["status"] for row in summaries}
    valid_layer_count = sum(int(row["valid_layers"]) for row in summaries)
    excluded_layer_count = sum(
        int(row["excluded_low_captured_layers"]) for row in summaries)
    overall_status = (
        "insufficient_evidence" if valid_layer_count == 0 else
        "partial" if excluded_layer_count > 0 else
        "ready" if "significant_divergence" in statuses else
        "no_significant_divergence"
    )
    return {
        "status": overall_status,
        "pairs": summaries,
        "num_pairs": len({row["pair_id"] for row in summaries}),
        "num_significant_pool_pairs": sum(
            row["status"] == "significant_divergence" for row in summaries),
        "num_no_significant_pool_pairs": sum(
            row["status"] == "no_significant_divergence" for row in summaries),
        "valid_layer_rows": valid_layer_count,
        "excluded_low_capture_rows": excluded_layer_count,
        "gate_metric": "sparse_topk_weighted_jaccard_capture_validated",
    }


def _state_transition_summary(
    pair_rows: Sequence[Dict[str, Any]],
    null_rows: Sequence[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    actual_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    null_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        actual_groups[(str(row["pair_id"]), str(row["pool"]))].append(row)
    for row in null_rows:
        null_groups[(str(row["pair_id"]), str(row["pool"]))].append(row)

    def aggregate(groups: Mapping[Tuple[str, str], List[Dict[str, Any]]], is_null: bool):
        out = []
        for (pair_id, pool), rows in sorted(groups.items()):
            valid_rows = [row for row in rows if row.get("metric_valid")]
            if not valid_rows:
                continue
            def mean_key(key: str) -> Optional[float]:
                vals = [
                    float(row[key]) for row in valid_rows
                    if row.get(key) is not None]
                return float(np.mean(vals)) if vals else None
            out.append({
                "pair_id": pair_id,
                "pair_type": valid_rows[0].get("pair_type"),
                "phenomenon": valid_rows[0].get("phenomenon"),
                "pool": pool,
                "is_random_null": is_null,
                "valid_layers": len(valid_rows),
                "state_cos": mean_key("state_similarity"),
                "query_cos": mean_key("query_similarity"),
                "gate_sim": mean_key("routing_similarity"),
                "delta_cos": mean_key("transition_similarity"),
                "path_sim": mean_key("path_similarity"),
            })
        return out

    rows = aggregate(actual_groups, False)
    null = aggregate(null_groups, True)
    state_threshold = _quantile([row["state_cos"] for row in rows], 0.25)
    transition_threshold = _quantile([row["path_sim"] for row in null], 0.75)
    if transition_threshold is None:
        transition_threshold = _quantile([row["path_sim"] for row in rows], 0.75)
    null_path = np.asarray(
        [row["path_sim"] for row in null if row.get("path_sim") is not None],
        dtype=np.float64,
    )
    quadrants: Dict[str, int] = defaultdict(int)
    for row in rows:
        state = row.get("state_cos")
        path = row.get("path_sim")
        if state is None or path is None or state_threshold is None or transition_threshold is None:
            row["quadrant"] = "insufficient"
            row["null_percentile"] = None
            continue
        state_low = float(state) <= float(state_threshold)
        path_high = float(path) >= float(transition_threshold)
        quadrant = (
            "low_state_high_transition" if state_low and path_high else
            "high_state_low_transition" if (not state_low and not path_high) else
            "high_high" if path_high else "low_low"
        )
        row["quadrant"] = quadrant
        quadrants[quadrant] += 1
        row["null_percentile"] = (
            float(np.mean(null_path <= float(path))) if null_path.size else None)
    correlations = {}
    for pool in TRACE_POOLS:
        pool_rows = [row for row in rows if row["pool"] == pool]
        correlations[pool] = {
            "corr_state_gate": _corr(
                [row["state_cos"] for row in pool_rows],
                [row["gate_sim"] for row in pool_rows]),
            "corr_state_delta": _corr(
                [row["state_cos"] for row in pool_rows],
                [row["delta_cos"] for row in pool_rows]),
            "corr_query_gate": _corr(
                [row["query_cos"] for row in pool_rows],
                [row["gate_sim"] for row in pool_rows]),
            "corr_gate_delta": _corr(
                [row["gate_sim"] for row in pool_rows],
                [row["delta_cos"] for row in pool_rows]),
        }
    def unique_pair_paths(values: Sequence[Dict[str, Any]]) -> List[float]:
        by_pair: Dict[str, List[float]] = defaultdict(list)
        for row in values:
            if row.get("path_sim") is not None:
                by_pair[str(row["pair_id"])].append(float(row["path_sim"]))
        return [float(np.mean(group)) for group in by_pair.values() if group]

    actual_paths = unique_pair_paths(rows)
    null_paths = unique_pair_paths(null)
    effect = (
        float(np.mean(actual_paths) - np.mean(null_paths))
        if actual_paths and null_paths else None)
    quadrant_unique_pairs = {
        quadrant: len({
            str(row["pair_id"]) for row in rows
            if row.get("quadrant") == quadrant
        })
        for quadrant in (
            "low_state_high_transition", "high_state_low_transition",
            "high_high", "low_low",
        )
    }
    actual_unique = len({str(row["pair_id"]) for row in pair_rows})
    null_unique = len({str(row["pair_id"]) for row in null_rows})
    expected_actual_pool_pairs = actual_unique * len(TRACE_POOLS)
    expected_null_pool_pairs = null_unique * len(TRACE_POOLS)
    excluded_low_capture = sum(
        row.get("invalid_reason") == "low_captured_mass"
        for row in list(pair_rows) + list(null_rows))
    status = (
        "insufficient_evidence" if not rows else
        "partial" if (
            not null
            or len(rows) < expected_actual_pool_pairs
            or len(null) < expected_null_pool_pairs
            or excluded_low_capture > 0
        ) else
        "ready"
    )
    return {
        "status": status,
        "state_low_threshold_data_q25": state_threshold,
        "transition_high_threshold_random_null_q75": transition_threshold,
        "quadrants": dict(quadrants),
        "quadrant_unique_pairs": quadrant_unique_pairs,
        "correlations": correlations,
        "path_similarity_effect_vs_random": effect,
        "path_similarity": _bootstrap_mean_ci(actual_paths, seed),
        "random_null_path_similarity": _bootstrap_mean_ci(null_paths, seed + 1),
        "counts": {
            "actual_unique_pairs": actual_unique,
            "null_unique_pairs": null_unique,
            "actual_valid_pool_pairs": len(rows),
            "null_valid_pool_pairs": len(null),
            "actual_expected_pool_pairs": expected_actual_pool_pairs,
            "null_expected_pool_pairs": expected_null_pool_pairs,
            "valid_pair_metric_row_fraction": (
                float(sum(row.get("metric_valid", False) for row in pair_rows))
                / len(pair_rows) if pair_rows else None),
            "excluded_low_capture_rows": excluded_low_capture,
        },
        "path_definition": "mean(routing_similarity, transition_similarity); query excluded",
        "rows": rows,
    }


def build_pair_analyses(
    ctx: AnalysisContext,
    records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    primary = _primary_records(records)
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in primary:
        by_pair[str(record["prompt"]["pair_id"])].append(record)
    actual_rows: List[Dict[str, Any]] = []
    for pair_id, pair_records in sorted(by_pair.items()):
        if len(pair_records) < 2:
            continue
        actual_rows.extend(_pair_layer_rows(pair_records[0], pair_records[1]))
    null_rows: List[Dict[str, Any]] = []
    if len(primary) >= 4:
        rng = np.random.default_rng(int(ctx.config.get("seed", 0)))
        used_prompt_combinations = set()
        for pair_id, pair_records in sorted(by_pair.items()):
            if len(pair_records) < 2:
                continue
            record_a = pair_records[int(rng.integers(0, len(pair_records)))]
            prompt_a = record_a["prompt"]
            candidates = [
                row for row in primary
                if row["prompt"]["pair_id"] != prompt_a["pair_id"]
                and row["prompt"].get("control_group") != prompt_a.get("control_group")
                and tuple(sorted((
                    str(prompt_a["prompt_id"]),
                    str(row["prompt"]["prompt_id"]),
                ))) not in used_prompt_combinations
            ]
            if not candidates:
                continue
            preferred = [
                row for row in candidates
                if row["prompt"].get("phenomenon") != prompt_a.get("phenomenon")]
            if preferred:
                candidates = preferred
            length_a = int(prompt_a["length"])
            length_deltas = np.asarray([
                abs(int(row["prompt"]["length"]) - length_a)
                for row in candidates
            ], dtype=np.int32)
            closest = np.flatnonzero(length_deltas == int(length_deltas.min()))
            record_b = candidates[int(rng.choice(closest))]
            combination = tuple(sorted((
                str(prompt_a["prompt_id"]),
                str(record_b["prompt"]["prompt_id"]),
            )))
            used_prompt_combinations.add(combination)
            null_pair_id = f"random-null-{pair_id}"
            rows = _pair_layer_rows(
                record_a,
                record_b,
                pair_type="random_length_matched_control",
                is_null=True,
            )
            for row in rows:
                row["pair_id"] = null_pair_id
            null_rows.extend(rows)
    context = _context_divergence_summary(actual_rows, null_rows)
    decoupling = _state_transition_summary(
        actual_rows,
        null_rows,
        int(ctx.config.get("seed", 0)),
    )
    csv_rows = actual_rows + null_rows
    if ctx.is_primary:
        write_csv_atomic(ctx.store.path("pair_metrics.csv"), csv_rows)
        write_json_atomic(ctx.store.path("context_divergence.json"), context)
        write_json_atomic(
            ctx.store.path("state_transition_decoupling.json"), decoupling)
    return context, decoupling, csv_rows


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _sequence_ce(logits: np.ndarray, token_ids: Sequence[int], length: int) -> Optional[float]:
    if int(length) < 2:
        return None
    logp = _log_softmax_np(np.asarray(logits)[0, : int(length) - 1])
    labels = np.asarray(token_ids[1:int(length)], dtype=np.int64)
    return float(-np.mean(logp[np.arange(labels.size), labels]))


def _intervention_candidates(
    record: Dict[str, Any],
    pool: str,
) -> List[Dict[str, Any]]:
    """Read exact on-device candidates captured before sparse top-k transfer."""
    trace = record["trace"]
    ids = np.asarray(trace[f"{pool}_candidate_ids"])[:, 0, :]
    valid = np.asarray(trace[f"{pool}_candidate_valid"])[:, 0, :].astype(bool)
    execution = np.asarray(trace[f"{pool}_candidate_execution"])[:, 0, :]
    admission = np.asarray(trace[f"{pool}_candidate_admission"])[:, 0, :]
    coefficient = np.asarray(
        trace[f"{pool}_candidate_abs_coefficient"])[:, 0, :]
    signed_coefficient = np.asarray(
        trace[f"{pool}_candidate_coefficient"])[:, 0, :]
    strategy_index = {
        strategy: index
        for index, strategy in enumerate(TRANSITION_CANDIDATE_STRATEGIES)
    }
    contribution_index = strategy_index["top_contribution"]
    gate_index = strategy_index["top_gate"]
    contribution_layer = int(np.argmax(coefficient[:, contribution_index]))
    gate_layer = int(np.argmax(execution[:, gate_index]))
    rows: List[Dict[str, Any]] = []
    for strategy in TRANSITION_CANDIDATE_STRATEGIES:
        index = strategy_index[strategy]
        layer = gate_layer if strategy == "top_gate" else contribution_layer
        candidate_valid = bool(valid[layer, index]) and int(ids[layer, index]) >= 0
        row = {
            "strategy": strategy,
            "layer": layer,
            "operator_id": int(ids[layer, index]),
            "candidate_valid": candidate_valid,
            "candidate_execution": float(execution[layer, index]),
            "candidate_admission": float(admission[layer, index]),
            "sidecar_estimated_abs_post_denominator_coefficient": float(
                coefficient[layer, index]),
            "sidecar_estimated_post_denominator_coefficient": float(
                signed_coefficient[layer, index]),
            "candidate_source": "sidecar_trace",
        }
        if not candidate_valid:
            row["status"] = (
                "skipped_no_inactive_candidate"
                if strategy == "inactive_random" else
                f"skipped_no_{strategy}_candidate"
            )
        rows.append(row)
    return rows


def _select_causal_records(
    records: Sequence[Dict[str, Any]],
    max_prompts: int,
) -> List[Dict[str, Any]]:
    """Select exactly one complete pair for each requested phenomenon."""
    if int(max_prompts) < 6:
        raise ValueError(
            "causal_intervention requires --causal-max-prompts >= 6 for the "
            "fixed ambiguity/negation/agreement stratification")
    primary = _primary_records(records)
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in primary:
        by_pair[str(record["prompt"]["pair_id"])].append(record)
    selected: List[Dict[str, Any]] = []
    for phenomenon in (
        "lexical_ambiguity", "negation", "subject_verb_agreement"):
        pair_candidates = [
            pair_records
            for _, pair_records in sorted(by_pair.items())
            if len(pair_records) >= 2
            and all(
                str(record["prompt"].get("phenomenon")) == phenomenon
                for record in pair_records[:2])
        ]
        if not pair_candidates:
            raise ValueError(
                f"causal_intervention missing complete {phenomenon!r} pair")
        selected.extend(sorted(
            pair_candidates[0][:2],
            key=lambda record: str(record["prompt"]["prompt_id"]),
        ))
    return selected


def _dense_minimal_sharded_fns(ctx: AnalysisContext):
    sharded_fns = ctx.sharded_fns
    if not isinstance(sharded_fns, dict):
        return sharded_fns
    return {
        key: value for key, value in sharded_fns.items()
        if key not in {
            "vocab_parallel_embedding", "vocab_ce", "vocab_eval_stats",
            "vocab_argmax",
        }
    }


def _capture_tiers(first: int, pool: str) -> List[int]:
    tiers = [int(first)]
    tiers.extend(
        value for value in ADAPTIVE_CAPTURE_TIERS[pool]
        if int(value) > int(first))
    return list(dict.fromkeys(tiers))


def _decorate_capture_rows(
        record: Dict[str, Any], capture: Mapping[str, Mapping[str, Any]],
        threshold: float) -> None:
    record["capture"] = capture
    record["capture_threshold"] = float(threshold)
    for layer, row in enumerate(record["rows"]):
        for pool in TRACE_POOLS:
            metadata = capture[pool]
            row["pools"][pool].update({
                "capture_topk_used": int(metadata["topk"]),
                "capture_retry_count": int(metadata["retry_count"]),
                "capture_threshold_pass": bool(
                    float(metadata["after"][layer]) >= float(threshold)),
                "capture_before": float(metadata["before"][layer]),
                "capture_after": float(metadata["after"][layer]),
            })
        for field in (
                "capture_topk_used", "capture_retry_count",
                "capture_threshold_pass", "capture_before", "capture_after"):
            row[field] = {
                pool: row["pools"][pool][field] for pool in TRACE_POOLS}


def _load_transition_trace_records(
        ctx: AnalysisContext, prompts: Sequence[Dict[str, Any]],
        summary: Mapping[str, Any], capture_threshold: float,
        ) -> Optional[List[Dict[str, Any]]]:
    path = ctx.store.path("transition_trace_cache.npz")
    if not exists(path):
        return None
    payload = read_npz(path)
    records: List[Dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for subtoken_index, token_index in enumerate(prompt["target_token_indices"]):
            prefix = f"p{prompt_index:04d}_t{subtoken_index:02d}"
            trace_prefix = f"{prefix}__trace__"
            trace = {
                key[len(trace_prefix):]: value
                for key, value in payload.items()
                if key.startswith(trace_prefix)
            }
            if not trace:
                return None
            record = _trace_internal_record(prompt, int(token_index), trace)
            capture: Dict[str, Dict[str, Any]] = {}
            for pool in TRACE_POOLS:
                capture_prefix = f"{prefix}__capture__{pool}__"
                before = payload.get(f"{capture_prefix}before")
                after = payload.get(f"{capture_prefix}after")
                if before is None or after is None:
                    after = np.asarray(trace[f"{pool}_captured_mass"])[:, 0]
                    before = after.copy()
                capture[pool] = {
                    "topk": int(np.asarray(payload.get(
                        f"{capture_prefix}topk",
                        np.asarray(trace[f"{pool}_top_idx"]).shape[-1])).item()),
                    "retry_count": int(np.asarray(payload.get(
                        f"{capture_prefix}retry_count", 0)).item()),
                    "before": np.asarray(before),
                    "after": np.asarray(after),
                }
            _decorate_capture_rows(record, capture, capture_threshold)
            records.append(record)
    return records


def _first_difference(left: np.ndarray, right: np.ndarray):
    indices = np.argwhere(np.asarray(left) != np.asarray(right))
    return tuple(int(value) for value in indices[0]) if indices.size else None


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not left.size:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _device_get_process_local_debug(tree):
    def get_leaf(value):
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            shards = value.addressable_shards
            if not shards:
                raise RuntimeError(
                    "parity debug array has no process-local addressable shards")
            return np.asarray(jax.device_get(shards[0].data))
        return np.asarray(jax.device_get(value))

    return jax.tree.map(get_leaf, tree)


def _parity_failure_diagnostics(
    ctx: AnalysisContext, input_ids, baseline_controls, comparison_controls,
    baseline_logits: np.ndarray, comparison_logits: np.ndarray,
    baseline_residual: np.ndarray, comparison_residual: np.ndarray,
) -> Dict[str, Any]:
    """Inspect tensors emitted by the actual minimal ``lax.scan``."""
    debug_forward = _canonical_causal_debug_forward(ctx)
    baseline_global = debug_forward(
        ctx.params, input_ids, *baseline_controls)
    comparison_global = debug_forward(
        ctx.params, input_ids, *comparison_controls)
    baseline_debug = _device_get_process_local_debug(baseline_global)
    comparison_debug = _device_get_process_local_debug(comparison_global)

    debug_keys = ("q", "k", "v", "attention_update", "rst",
                  "post_layer_residual")
    first_layer = next((
        layer_index
        for layer_index in range(int(ctx.model_cfg["n_layers"]))
        if any(
            not np.array_equal(
                np.asarray(baseline_debug[key][layer_index]),
                np.asarray(comparison_debug[key][layer_index]))
            for key in debug_keys)
    ), None)
    difference_slice = slice(None) if first_layer is None else first_layer
    differences = {
        key: _max_abs_difference(
            baseline_debug[key][difference_slice],
            comparison_debug[key][difference_slice])
        for key in debug_keys
    }
    route_names = ("q", "k", "v", "rst")
    first_route = next(
        (route for route in route_names if differences[route] != 0.0),
        "attention" if differences["attention_update"] != 0.0 else None,
    )
    residual_index = _first_difference(
        baseline_debug["post_layer_residual"][difference_slice],
        comparison_debug["post_layer_residual"][difference_slice])
    component_index = None
    for key in ("q", "k", "v", "attention_update", "rst"):
        component_index = _first_difference(
            baseline_debug[key][difference_slice],
            comparison_debug[key][difference_slice])
        if component_index is not None:
            break
    if residual_index is None:
        residual_index = component_index
    if residual_index is None:
        residual_index = _first_difference(
            baseline_residual, comparison_residual)
    logit_index = _first_difference(baseline_logits, comparison_logits)
    route_indices = {"q": 0, "k": 1, "v": 2, "rst": 3}
    return {
        "first_differing_layer": first_layer,
        "first_differing_route": first_route,
        "first_differing_route_index": route_indices.get(first_route),
        "first_differing_token": (
            residual_index[-2] if residual_index is not None else
            (logit_index[-2] if logit_index is not None else None)),
        "first_differing_residual_dimension": (
            residual_index[-1] if residual_index is not None else None),
        "first_differing_vocab_index": (
            logit_index[-1] if logit_index is not None else None),
        "max_q_output_difference": differences["q"],
        "max_k_output_difference": differences["k"],
        "max_v_output_difference": differences["v"],
        "max_attention_update_difference": differences["attention_update"],
        "max_rst_output_difference": differences["rst"],
        "max_post_layer_residual_difference": differences[
            "post_layer_residual"],
        "diagnostic_execution": "actual_minimal_lax_scan",
        "debug_layers_compared": (
            int(ctx.model_cfg["n_layers"])
            if first_layer is None else first_layer + 1),
    }


def _minimal_forward_kwargs(ctx: AnalysisContext) -> Dict[str, Any]:
    """Runtime settings shared by every dense minimal analysis forward."""
    mcfg = ctx.model_cfg
    return {
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(0)},
        "sharded_fns": _dense_minimal_sharded_fns(ctx),
        "analysis": False,
        "minimal_train": True,
        "soft_gate_temperature": float(mcfg["soft_gate_temperature"]),
        "soft_gate_t_final": float(mcfg.get(
            "soft_gate_t_final", mcfg["soft_gate_temperature"])),
        "soft_gate_T_qk": float(mcfg.get(
            "soft_gate_T_qk", mcfg["soft_gate_temperature"])),
        "soft_gate_T_v": float(mcfg.get(
            "soft_gate_T_v", mcfg["soft_gate_temperature"])),
        "soft_gate_T_rst": float(mcfg.get(
            "soft_gate_T_rst", mcfg["soft_gate_temperature"])),
        "soft_gate_boundary_power": float(mcfg["soft_gate_boundary_power"]),
        "soft_gate_boundary_power_final": float(mcfg.get(
            "soft_gate_boundary_power_final",
            mcfg["soft_gate_boundary_power"])),
        "admission_den_power": float(mcfg["admission_den_power"]),
        "srw_composition_mode": str(mcfg["srw_composition_mode"]),
        "heat_kernel_beta": float(mcfg["heat_kernel_beta"]),
        "execution_prune_eps": jnp.float32(
            float(mcfg.get("execution_prune_eps", 0.0) or 0.0)),
        "compute_accuracy": False,
        "analysis_return_residual": True,
        "analysis_return_logits": True,
    }


def _normal_production_logits_forward(ctx: AnalysisContext):
    """Build the ordinary production entry point for nonblocking audit."""
    forward_kwargs = _minimal_forward_kwargs(ctx)

    @jax.jit
    def forward(params, input_ids):
        result = ctx.model.apply(
            {"params": params}, input_ids, labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            **forward_kwargs)
        return (result["logits"][:1], result["per_token_ce"][:1],
                result["final_residual"][:1])

    return forward


def _canonical_causal_logits_forward(ctx: AnalysisContext):
    """Build the one canonical full-forward executable for causal analysis."""
    forward_kwargs = _minimal_forward_kwargs(ctx)

    @jax.jit
    def forward(
            params, input_ids, target_positions, target_layer, target_route,
            selected_operator_ids, apply_suppression):
        result = ctx.model.apply(
            {"params": params},
            input_ids,
            labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            analysis_contribution=selected_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=target_route,
            analysis_intervention_enabled=apply_suppression,
            **forward_kwargs,
        )
        return (result["logits"][:1], result["per_token_ce"][:1],
                result["final_residual"][:1])

    return forward


def _canonical_causal_trace_forward(ctx: AnalysisContext):
    """Canonical causal executable returning target-only layer traces."""
    forward_kwargs = _minimal_forward_kwargs(ctx)

    @jax.jit
    def forward(
            params, input_ids, target_positions, target_layer, target_route,
            selected_operator_ids, apply_suppression):
        result = ctx.model.apply(
            {"params": params},
            input_ids,
            labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            analysis_contribution=selected_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=target_route,
            analysis_intervention_enabled=apply_suppression,
            analysis_causal_trace=True,
            **forward_kwargs,
        )
        return (
            result["logits"][:1], result["per_token_ce"][:1],
            result["final_residual"][:1],
            jax.tree.map(lambda value: value[:, :1], result["causal_trace"]),
        )

    return forward


def _canonical_group_causal_trace_forward(ctx: AnalysisContext, max_width: int):
    """Fixed-width group executable shared by sizes 0/1/2/4/8."""
    if int(max_width) <= 0:
        raise ValueError("group causal max width must be positive")
    forward_kwargs = _minimal_forward_kwargs(ctx)

    @jax.jit
    def forward(
            params, input_ids, target_positions, target_layer, target_route,
            selected_operator_ids):
        if selected_operator_ids.shape[1] != int(max_width):
            raise ValueError("group causal ids must use the configured width")
        result = ctx.model.apply(
            {"params": params},
            input_ids,
            labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            analysis_contribution=selected_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=target_route,
            analysis_intervention_enabled=jnp.bool_(True),
            analysis_causal_trace=True,
            **forward_kwargs,
        )
        return (
            result["logits"][:1], result["per_token_ce"][:1],
            result["final_residual"][:1],
            jax.tree.map(lambda value: value[:, :1], result["causal_trace"]),
        )

    return forward


def _canonical_causal_debug_forward(ctx: AnalysisContext):
    """Failure-only canonical forward with actual-scan layer outputs."""
    forward_kwargs = _minimal_forward_kwargs(ctx)

    @jax.jit
    def forward(
            params, input_ids, target_positions, target_layer, target_route,
            selected_operator_ids, apply_suppression):
        result = ctx.model.apply(
            {"params": params},
            input_ids,
            labels=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            analysis_contribution=selected_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=target_route,
            analysis_intervention_enabled=apply_suppression,
            analysis_parity_debug=True,
            **forward_kwargs,
        )
        return result["parity_debug"]

    return forward


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


def _neutral_causal_controls(batch_size: int):
    return (
        jnp.full((batch_size,), -1, dtype=jnp.int32),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.full((batch_size,), -1, dtype=jnp.int32),
        jnp.bool_(False),
    )


def _forward_comparison_row(
    prompt: Mapping[str, Any], target_position: int, reference, candidate,
    comparison: str,
) -> Dict[str, Any]:
    reference_logits, reference_per_token_ce, reference_residual = (
        np.asarray(value) for value in reference)
    candidate_logits, candidate_per_token_ce, candidate_residual = (
        np.asarray(value) for value in candidate)
    logit_abs_diff = np.abs(
        reference_logits.astype(np.float64)
        - candidate_logits.astype(np.float64))
    ce_abs_diff = np.abs(
        reference_per_token_ce - candidate_per_token_ce)
    return {
        "prompt_id": prompt["prompt_id"],
        "comparison": comparison,
        "ce_abs_diff": (
            float(np.max(ce_abs_diff)) if ce_abs_diff.size else 0.0),
        "mean_logit_abs_diff": float(np.mean(logit_abs_diff)),
        "max_logit_abs_diff": float(np.max(logit_abs_diff)),
        "top1_agreement": float(np.mean(
            np.argmax(reference_logits, axis=-1)
            == np.argmax(candidate_logits, axis=-1))),
        "final_residual_cosine": _cosine(
            reference_residual[0, target_position],
            candidate_residual[0, target_position]),
        "logits_machine_exact": bool(np.array_equal(
            reference_logits, candidate_logits)),
        "per_token_ce_machine_exact": bool(np.array_equal(
            reference_per_token_ce, candidate_per_token_ce)),
        "final_residual_machine_exact": bool(np.array_equal(
            reference_residual, candidate_residual)),
        "final_residual_max_abs_diff": _max_abs_difference(
            reference_residual, candidate_residual),
    }


def _intervention_forward_parity(
    ctx: AnalysisContext,
    records: Sequence[Dict[str, Any]],
    canonical_forward: Any,
) -> Dict[str, Any]:
    """Require machine-exact no-ops within one canonical causal executable."""
    rows = []
    first_failure = None
    for record in records:
        prompt = record["prompt"]
        data_replicas = max(1, int(ctx.mesh.shape["data"]))
        input_ids = jax.device_put(jnp.asarray(np.repeat(
            np.asarray(prompt["input_array"], dtype=np.int32)[None, :],
            data_replicas, axis=0)), ctx.data_sharding)
        batch_size = input_ids.shape[0]
        target_position = int(record["target_token_index"])
        target_positions = jnp.full(
            (batch_size,), target_position, dtype=jnp.int32)
        selected_zero = jnp.zeros((batch_size,), dtype=jnp.int32)
        selected_inactive = jnp.full(
            (batch_size,), -1, dtype=jnp.int32)
        baseline_controls = _neutral_causal_controls(batch_size)
        baseline = jax.device_get(canonical_forward(
            ctx.params, input_ids, *baseline_controls))
        comparisons = (
            ("repeated_canonical_baseline", baseline_controls),
            ("suppression_disabled", (
                target_positions, jnp.int32(0), jnp.int32(0),
                selected_zero, jnp.bool_(False))),
            ("non_target_layer_suppression", (
                target_positions, jnp.int32(ctx.model_cfg["n_layers"]),
                jnp.int32(0), selected_zero, jnp.bool_(True))),
            ("non_target_route_suppression", (
                target_positions, jnp.int32(0), jnp.int32(4),
                selected_zero, jnp.bool_(True))),
            ("inactive_operator_suppression", (
                target_positions, jnp.int32(0), jnp.int32(0),
                selected_inactive, jnp.bool_(True))),
        )
        for comparison_name, controls in comparisons:
            candidate = jax.device_get(canonical_forward(
                ctx.params, input_ids, *controls))
            row = _forward_comparison_row(
                prompt, target_position, baseline, candidate,
                comparison_name)
            rows.append(row)
            if first_failure is None and not (
                    row["logits_machine_exact"]
                    and row["per_token_ce_machine_exact"]
                    and row["final_residual_machine_exact"]):
                first_failure = {
                    "input_ids": input_ids,
                    "baseline_controls": baseline_controls,
                    "comparison_controls": controls,
                    "baseline_logits": np.asarray(baseline[0]),
                    "comparison_logits": np.asarray(candidate[0]),
                    "baseline_residual": np.asarray(baseline[2]),
                    "comparison_residual": np.asarray(candidate[2]),
                }
    summary = {
        "status": "ready",
        "blocking": True,
        "intervention_type": "canonical_causal_execution_suppression",
        "purpose": "machine-exact no-ops within one canonical causal forward",
        "num_prompts": len(records),
        "num_comparisons": len(rows),
        "ce_abs_diff": max(float(row["ce_abs_diff"]) for row in rows),
        "mean_logit_abs_diff": float(np.mean([
            float(row["mean_logit_abs_diff"]) for row in rows])),
        "max_logit_abs_diff": max(
            float(row["max_logit_abs_diff"]) for row in rows),
        "final_residual_max_abs_diff": max(
            float(row["final_residual_max_abs_diff"]) for row in rows),
        "top1_agreement": float(np.mean([
            float(row["top1_agreement"]) for row in rows])),
        "final_residual_cosine": min(
            float(row["final_residual_cosine"]) for row in rows),
        "machine_exact": all(
            row["logits_machine_exact"]
            and row["per_token_ce_machine_exact"]
            and row["final_residual_machine_exact"]
            for row in rows),
        "rows": rows,
    }
    if ctx.is_primary:
        write_json_atomic(
            ctx.store.path("intervention_forward_parity.json"), summary)
    if not summary["machine_exact"]:
        try:
            summary.update(_parity_failure_diagnostics(
                ctx,
                first_failure["input_ids"],
                first_failure["baseline_controls"],
                first_failure["comparison_controls"],
                first_failure["baseline_logits"],
                first_failure["comparison_logits"],
                first_failure["baseline_residual"],
                first_failure["comparison_residual"],
            ))
            summary["failure_diagnostics_status"] = "ready"
        except Exception as exc:
            summary["failure_diagnostics_status"] = "failed"
            summary["failure_diagnostics_error"] = (
                f"{type(exc).__name__}: {exc}")
        if ctx.is_primary:
            write_json_atomic(
                ctx.store.path("intervention_forward_parity.json"), summary)
        raise RuntimeError("canonical causal zero-suppression parity failed")
    return summary


def _normal_production_cross_graph_audit(
    ctx: AnalysisContext, records: Sequence[Dict[str, Any]],
    canonical_forward: Any,
) -> Dict[str, Any]:
    """Record, but never block on, ordinary production vs canonical neutral."""
    production_forward = _normal_production_logits_forward(ctx)
    rows = []
    for record in records:
        prompt = record["prompt"]
        data_replicas = max(1, int(ctx.mesh.shape["data"]))
        input_ids = jax.device_put(jnp.asarray(np.repeat(
            np.asarray(prompt["input_array"], dtype=np.int32)[None, :],
            data_replicas, axis=0)), ctx.data_sharding)
        target_position = int(record["target_token_index"])
        production = jax.device_get(production_forward(ctx.params, input_ids))
        canonical = jax.device_get(canonical_forward(
            ctx.params, input_ids, *_neutral_causal_controls(
                input_ids.shape[0])))
        rows.append(_forward_comparison_row(
            prompt, target_position, production, canonical,
            "normal_production_vs_canonical_neutral"))
    summary = {
        "status": "cross_graph_audit",
        "blocking": False,
        "machine_exact": all(
            row["logits_machine_exact"]
            and row["per_token_ce_machine_exact"]
            and row["final_residual_machine_exact"]
            for row in rows),
        "max_logit_abs_diff": max(
            float(row["max_logit_abs_diff"]) for row in rows),
        "mean_logit_abs_diff": float(np.mean([
            float(row["mean_logit_abs_diff"]) for row in rows])),
        "per_token_ce_max_abs_diff": max(
            float(row["ce_abs_diff"]) for row in rows),
        "final_residual_max_abs_diff": max(
            float(row["final_residual_max_abs_diff"]) for row in rows),
        "top1_agreement": float(np.mean([
            float(row["top1_agreement"]) for row in rows])),
        "final_residual_cosine": min(
            float(row["final_residual_cosine"]) for row in rows),
        "rows": rows,
    }
    if ctx.is_primary:
        write_json_atomic(
            ctx.store.path("normal_production_vs_canonical_neutral.json"),
            summary)
        print(
            "NORMAL_PRODUCTION_CROSS_GRAPH_AUDIT "
            f"machine_exact={summary['machine_exact']} blocking=False",
            flush=True,
        )
    return summary


def _print_parity_success(ctx: AnalysisContext, parity: Mapping[str, Any]) -> None:
    if ctx.is_primary:
        print(
            "CANONICAL_CAUSAL_ZERO_PARITY_OK "
            f"machine_exact={parity['machine_exact']} "
            f"max_logit_abs_diff={parity['max_logit_abs_diff']:g} "
            f"ce_abs_diff={parity['ce_abs_diff']:g} "
            "final_residual_max_abs_diff="
            f"{parity['final_residual_max_abs_diff']:g}",
            flush=True,
        )


def run_v4171_parity_only_smoke(ctx: AnalysisContext) -> Dict[str, Any]:
    """Run blocking machine-exact parity without building transition traces."""
    if str(ctx.model_cfg.get("model_version")) != V4171_MODEL_VERSION:
        raise ValueError(
            "v4171 parity-only smoke requires model_version="
            f"{V4171_MODEL_VERSION}")
    args = ctx.args
    prompt_set = str(
        getattr(args, "transition_prompt_set", None)
        or DEFAULT_TRANSITION_PROMPT_SET)
    rows, prompt_hash = load_transition_prompt_rows(prompt_set)
    max_prompts = getattr(args, "transition_max_prompts", None)
    if max_prompts is None:
        max_prompts = max(1, int(getattr(
            args, "causal_max_prompts", 6) or 6))
    rows = rows[:max(1, int(max_prompts))]
    tokenizer = maybe_load_tokenizer(local_only=True)
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "v4171 parity-only smoke requires the cached bert-base-uncased "
            "fast tokenizer")
    seq_len = int(getattr(args, "trace_seq_len", 128) or 128)
    prompts = [_tokenize_transition_row(tokenizer, row, seq_len) for row in rows]
    _validate_tokenized_pairs(prompts)
    records = [
        {
            "prompt": prompt,
            "target_token_index": int(max(prompt["target_token_indices"])),
        }
        for prompt in prompts
    ]
    canonical_forward = _canonical_causal_logits_forward(ctx)
    cross_graph = _normal_production_cross_graph_audit(
        ctx, records, canonical_forward)
    parity = _intervention_forward_parity(
        ctx, records, canonical_forward)
    provenance = _analysis_provenance(
        ctx, prompt_hash, parity, cross_graph)
    parity.update(provenance)
    cross_graph.update(provenance)
    cross_graph["blocking"] = False
    if ctx.is_primary:
        write_json_atomic(
            ctx.store.path("intervention_forward_parity.json"), parity)
        write_json_atomic(
            ctx.store.path("normal_production_vs_canonical_neutral.json"),
            cross_graph)
    _print_parity_success(ctx, parity)
    return parity


def _causal_effect_summary(
    rows: Sequence[Dict[str, Any]], seed: int,
) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("status") == "ready"
        and row.get("target_logprob_delta") is not None]
    abs_delta = [abs(float(row["target_logprob_delta"])) for row in valid]
    effect = _bootstrap_mean_ci(abs_delta, seed)
    return {
        "n": len(valid),
        "mean_abs_target_logprob_delta": effect["mean"],
        "bootstrap_ci95": effect["ci95"],
        "mean_kl": (
            float(np.mean([float(row["full_output_kl"]) for row in valid]))
            if valid else None),
        "top_prediction_changed_fraction": (
            float(np.mean([bool(row["top_prediction_changed"]) for row in valid]))
            if valid else None),
    }


def run_causal_intervention(
    ctx: AnalysisContext,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Suppress selected execution numerators in the shared production core."""
    max_prompts = max(1, int(getattr(ctx.args, "causal_max_prompts", 6) or 6))
    primary = _select_causal_records(records, max_prompts)
    pool_codes = {"q": 0, "k": 1, "v": 2, "rst": 3}
    canonical_forward = _canonical_causal_logits_forward(ctx)
    causal_trace_forward = _canonical_causal_trace_forward(ctx)
    cross_graph_audit = _normal_production_cross_graph_audit(
        ctx, primary, canonical_forward)
    parity = _intervention_forward_parity(
        ctx, primary, canonical_forward)
    _print_parity_success(ctx, parity)

    result_rows: List[Dict[str, Any]] = []
    started = time.time()
    for prompt_idx, record in enumerate(primary):
        prompt = record["prompt"]
        data_replicas = max(1, int(ctx.mesh.shape["data"]))
        input_ids = jax.device_put(jnp.asarray(np.repeat(
            np.asarray(prompt["input_array"], dtype=np.int32)[None, :],
            data_replicas, axis=0)), ctx.data_sharding)
        target_position = int(record["target_token_index"])
        positions = jnp.full(
            (input_ids.shape[0],), target_position, dtype=jnp.int32)
        (baseline_logits, _, baseline_residual,
         baseline_trace) = jax.device_get(
            causal_trace_forward(
                ctx.params, input_ids, positions, jnp.int32(0), jnp.int32(0),
                jnp.zeros((input_ids.shape[0],), dtype=jnp.int32),
                jnp.bool_(False)))
        baseline_logits = np.asarray(baseline_logits)
        baseline_residual = np.asarray(baseline_residual)
        length = int(prompt["length"])
        baseline_ce = _sequence_ce(
            baseline_logits, prompt["token_ids"], length)
        baseline_logp = _log_softmax_np(baseline_logits[0, :length])
        for pool in TRACE_POOLS:
            for candidate in _intervention_candidates(record, pool):
                common = {
                    "prompt_id": prompt["prompt_id"],
                    "phenomenon": prompt["phenomenon"],
                    "pool": pool,
                    **candidate,
                    "intervention_type": "production_core_execution_suppression",
                    "candidate_selection_source": "sidecar_trace",
                    "intervention_execution_source": "production_core",
                    "canonical_unpruned_admission_denominator": True,
                    "effect_reference": "canonical_suppression_disabled",
                }
                if not candidate["candidate_valid"]:
                    result_rows.append(common)
                    continue
                operator_id = _validate_global_operator_id(
                    ctx, pool, int(candidate["operator_id"]))
                selected_operator_ids = jnp.full(
                    (input_ids.shape[0],), operator_id, jnp.int32)
                logits, _, residual, intervention_trace = jax.device_get(
                    causal_trace_forward(
                    ctx.params, input_ids, positions,
                    jnp.int32(candidate["layer"]), jnp.int32(pool_codes[pool]),
                    selected_operator_ids, jnp.bool_(True)))
                logits = np.asarray(logits)
                residual = np.asarray(residual)
                after_ce = _sequence_ce(logits, prompt["token_ids"], length)
                after_logp = _log_softmax_np(logits[0, :length])
                base_prob = np.exp(baseline_logp)
                full_output_kl = float(np.mean(np.sum(
                    base_prob * (baseline_logp - after_logp), axis=-1)))
                target_base = baseline_residual[0, target_position]
                target_after = residual[0, target_position]
                behavior_before = None if baseline_ce is None else -float(baseline_ce)
                behavior_after = None if after_ce is None else -float(after_ce)
                inactive_exact_noop = None
                if float(candidate["candidate_execution"]) == 0.0:
                    inactive_exact_noop = bool(
                        np.array_equal(baseline_logits, logits)
                        and np.array_equal(baseline_residual, residual))
                    if not inactive_exact_noop:
                        raise RuntimeError(
                            "inactive operator suppression changed production output")
                recovery = compute_causal_recovery_metrics(
                    baseline_trace,
                    intervention_trace,
                    route=pool,
                    target_layer=int(candidate["layer"]),
                    baseline_final_residual=baseline_residual[
                        0, target_position],
                    intervention_final_residual=residual[0, target_position],
                    baseline_logits=baseline_logits,
                    intervention_logits=logits,
                    target_position=target_position,
                )
                result_rows.append({
                    **common,
                    "status": "ready",
                    "removed_operator_count": 1,
                    "behavior_score_before": behavior_before,
                    "behavior_score_after": behavior_after,
                    "behavior_score_drop": (
                        None if behavior_before is None or behavior_after is None
                        else behavior_before - behavior_after),
                    "target_logprob_delta": (
                        None if behavior_before is None or behavior_after is None
                        else behavior_after - behavior_before),
                    "full_output_kl": full_output_kl,
                    "next_token_kl": float(np.sum(
                        base_prob[-1] * (baseline_logp[-1] - after_logp[-1]))),
                    "top_prediction_changed": bool(
                        np.argmax(baseline_logp[-1]) != np.argmax(after_logp[-1])),
                    "final_residual_cosine": _cosine(target_base, target_after),
                    "final_residual_relative_error": float(
                        np.linalg.norm(target_after - target_base)
                        / max(float(np.linalg.norm(target_base)), 1.0e-12)),
                    "inactive_machine_exact_noop": inactive_exact_noop,
                    **recovery,
                })
        if ctx.is_primary:
            print(
                "CAUSAL_PRODUCTION_CORE_SUPPRESSION "
                f"prompt={prompt_idx + 1:02d}/{len(primary):02d} "
                f"id={prompt['prompt_id']}", flush=True)

    valid_rows = [row for row in result_rows if row.get("status") == "ready"]
    seed = int(ctx.config.get("seed", 0))
    def grouped(key: str, offset: int) -> Dict[str, Any]:
        return {
            value: _causal_effect_summary(
                [row for row in valid_rows if str(row[key]) == value],
                seed + offset + index)
            for index, value in enumerate(sorted({str(row[key]) for row in valid_rows}))
        }
    selected = [
        float(row["behavior_score_drop"]) for row in valid_rows
        if row["strategy"] in ("top_contribution", "top_gate")
        and row.get("behavior_score_drop") is not None]
    controls = [
        float(row["behavior_score_drop"]) for row in valid_rows
        if row["strategy"] in ("inactive_random", "active_random", "matched_active")
        and row.get("behavior_score_drop") is not None]
    summary = {
        "status": "ready" if valid_rows else "partial",
        "intervention_type": "production_core_execution_suppression",
        "canonical_unpruned_admission_denominator": True,
        "zero_suppression_parity": parity,
        "normal_production_cross_graph_audit": cross_graph_audit,
        "causal_baseline": "canonical_suppression_disabled",
        "canonical_baseline_source": "canonical_suppression_disabled",
        "effect_reference": "canonical_suppression_disabled",
        "canonical_parity_machine_exact": bool(parity["machine_exact"]),
        "cross_graph_audit_blocking": False,
        "num_prompts": len(primary),
        "num_interventions": len(valid_rows),
        "num_skipped": len(result_rows) - len(valid_rows),
        "effects": {
            "overall": _causal_effect_summary(valid_rows, seed),
            "by_strategy": grouped("strategy", 100),
            "by_pool": grouped("pool", 200),
            "by_phenomenon": grouped("phenomenon", 300),
        },
        "selected_behavior_score_drop": _bootstrap_mean_ci(selected, seed),
        "control_behavior_score_drop": _bootstrap_mean_ci(controls, seed + 1),
        "selected_minus_control_effect": (
            float(np.mean(selected) - np.mean(controls))
            if selected and controls else None),
        "sec": time.time() - started,
        "artifact": ctx.store.path("interventions.jsonl"),
        "limitations": [
            "transition prompt item reports sequence-score effects; dataset items report task margins",
        ],
    }
    if ctx.is_primary:
        write_jsonl_atomic(ctx.store.path("interventions.jsonl"), result_rows)
        write_json_atomic(ctx.store.path("causal_intervention_summary.json"), summary)
    summary["_result_rows"] = result_rows
    return summary


def _classify_recovery_rows(
        rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("status") == "ready"
        and _json_float(row.get("immediate_delta_norm")) is not None
        and _json_float(row.get("recovery_ratio_final")) is not None
    ]
    immediate = np.asarray([
        float(row["immediate_delta_norm"]) for row in valid], dtype=np.float64)
    ratios = np.asarray([
        float(row["recovery_ratio_final"]) for row in valid], dtype=np.float64)
    basis = {
        "immediate_q25": float(np.quantile(immediate, 0.25)) if valid else None,
        "immediate_median": float(np.quantile(immediate, 0.50)) if valid else None,
        "recovery_ratio_q25": float(np.quantile(ratios, 0.25)) if valid else None,
        "recovery_ratio_median": float(np.quantile(ratios, 0.50)) if valid else None,
        "recovery_ratio_q75": float(np.quantile(ratios, 0.75)) if valid else None,
        "method": "observed_distribution_quantiles",
    }
    for row in valid:
        immediate_value = float(row["immediate_delta_norm"])
        ratio = float(row["recovery_ratio_final"])
        if immediate_value <= float(basis["immediate_q25"]):
            phenomenon = "immediate_small"
        elif (immediate_value >= float(basis["immediate_median"])
              and ratio <= float(basis["recovery_ratio_q25"])):
            phenomenon = "downstream_recovery"
        elif ratio >= float(basis["recovery_ratio_q75"]):
            phenomenon = "downstream_amplification"
        elif bool(row.get("non_monotonic_path")):
            phenomenon = "non_monotonic"
        else:
            phenomenon = "approximately_preserved"
        row["recovery_phenomenon"] = phenomenon
    return basis


def _recovery_group_summary(
        rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ready"]
    ratios = [
        float(row["recovery_ratio_final"]) for row in valid
        if _json_float(row.get("recovery_ratio_final")) is not None]
    return {
        "n": len(valid),
        "immediate_delta_mean": (
            float(np.mean([float(row["immediate_delta_norm"]) for row in valid]))
            if valid else None),
        "final_delta_mean": (
            float(np.mean([float(row["final_delta_norm"]) for row in valid]))
            if valid else None),
        "recovery_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "recovery_ratio_ci95": _bootstrap_mean_ci(ratios, seed)["ci95"],
        "recovery_fraction": (
            float(np.mean([
                row.get("recovery_phenomenon") == "downstream_recovery"
                for row in valid])) if valid else None),
        "amplification_fraction": (
            float(np.mean([
                row.get("recovery_phenomenon") == "downstream_amplification"
                for row in valid])) if valid else None),
    }


def run_causal_recovery_trace(
        ctx: AnalysisContext, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize traces returned by the canonical intervention forwards."""
    recovery_rows = [dict(row) for row in rows]
    basis = _classify_recovery_rows(recovery_rows)
    n_layers = max(1, int(ctx.model_cfg["n_layers"]))
    for row in recovery_rows:
        layer = int(row.get("layer", 0))
        row["layer_bucket"] = (
            "early" if layer < n_layers / 3.0 else
            "middle" if layer < 2.0 * n_layers / 3.0 else "late")
    seed = int(ctx.config.get("seed", 0))

    def grouped(key: str, offset: int) -> Dict[str, Any]:
        values = sorted({
            str(row.get(key)) for row in recovery_rows
            if row.get(key) is not None})
        return {
            value: _recovery_group_summary(
                [row for row in recovery_rows if str(row.get(key)) == value],
                seed + offset + index)
            for index, value in enumerate(values)
        }

    summary = {
        "status": "ready" if any(
            row.get("status") == "ready" for row in recovery_rows) else "partial",
        "classification_basis": basis,
        "overall": _recovery_group_summary(recovery_rows, seed),
        "by_pool": grouped("pool", 100),
        "by_strategy": grouped("strategy", 200),
        "by_phenomenon": grouped("phenomenon", 300),
        "by_recovery_phenomenon": grouped("recovery_phenomenon", 350),
        "by_layer_bucket": grouped("layer_bucket", 400),
        "artifacts": {
            "rows": ctx.store.path("causal_recovery_traces.jsonl"),
            "summary": ctx.store.path("causal_recovery_summary.json"),
        },
    }
    if ctx.is_primary:
        write_jsonl_atomic(
            ctx.store.path("causal_recovery_traces.jsonl"), recovery_rows)
        write_json_atomic(
            ctx.store.path("causal_recovery_summary.json"), summary)
    summary["_rows"] = recovery_rows
    return summary


def _pool_parameter_keys(pool: str) -> Tuple[str, str, str]:
    prefix = {"qk": "attn_qk", "v": "attn_v", "rst": "rst"}[pool]
    return f"{prefix}_op_key", f"{prefix}_read", f"{prefix}_write"


def _candidate_pool_vectors(
        ctx: AnalysisContext, pool: str, operator_ids: Sequence[int]
        ) -> Dict[str, np.ndarray]:
    """Gather only candidate embeddings and replicate the small result."""
    address_key, read_key, write_key = _pool_parameter_keys(pool)
    replicated = NamedSharding(ctx.mesh, P())
    ids = jax.device_put(
        jnp.asarray(operator_ids, dtype=jnp.int32), replicated)

    @partial(jax.jit, out_shardings={
        "address": replicated, "read": replicated, "write": replicated})
    def select(params, selected):
        pool_params = params["neuron_pool"]
        return {
            "address": pool_params[address_key][selected],
            "read": pool_params[read_key][selected],
            "write": pool_params[write_key][selected],
        }

    selected_host = _device_get_process_local_debug(select(ctx.params, ids))
    return {key: np.asarray(value) for key, value in selected_host.items()}


def _sparse_profile_cosine(
        left: Mapping[int, float], right: Mapping[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(float(value) * float(right.get(key, 0.0))
              for key, value in left.items())
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    return float(dot / max(left_norm * right_norm, 1.0e-12))


def _sparse_profile_neighbors(
        signatures: Mapping[int, Mapping[int, float]],
        operator_ids: Sequence[int], k: int) -> Dict[int, List[Dict[str, Any]]]:
    """Shortlist with a sketch, then score sparse profiles exactly."""
    ids = [int(value) for value in operator_ids]
    n = len(ids)
    sketch_dim = 64
    sketch = np.zeros((n, sketch_dim), dtype=np.float32)
    for index, operator_id in enumerate(ids):
        for observation, value in signatures.get(operator_id, {}).items():
            hashed = (int(observation) * 2654435761 + 1013904223) & 0xFFFFFFFF
            bucket = hashed % sketch_dim
            sign = 1.0 if (hashed & 0x80000000) else -1.0
            sketch[index, bucket] += sign * float(value)
    sketch /= np.maximum(
        np.linalg.norm(sketch, axis=-1, keepdims=True), 1.0e-12)
    shortlist_count = min(max(int(k) * 4, 64), max(0, n - 1))
    shortlists: List[List[int]] = [[] for _ in ids]
    if shortlist_count:
        block = 128
        for start in range(0, n, block):
            end = min(n, start + block)
            similarity = sketch[start:end] @ sketch.T
            similarity[np.arange(end - start), np.arange(start, end)] = -np.inf
            chosen = np.argpartition(
                -similarity, shortlist_count - 1, axis=1)[:, :shortlist_count]
            for local, row_indices in enumerate(chosen):
                row_indices = row_indices[np.lexsort((
                    np.asarray(ids)[row_indices],
                    -similarity[local, row_indices]))]
                shortlists[start + local] = row_indices.tolist()
    out: Dict[int, List[Dict[str, Any]]] = {}
    for index, operator_id in enumerate(ids):
        candidates = [ids[other] for other in shortlists[index]]
        scored = [
            {
                "operator_id": other,
                "similarity": _sparse_profile_cosine(
                    signatures.get(operator_id, {}),
                    signatures.get(other, {})),
            }
            for other in candidates
        ]
        scored.sort(key=lambda row: (-float(row["similarity"]), row["operator_id"]))
        out[operator_id] = scored[:int(k)]
    return out


def _parameter_neighbors(
        vectors: Mapping[str, np.ndarray], operator_ids: Sequence[int],
        k: int, seed: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    ids = np.asarray(operator_ids, dtype=np.int64)
    address = _unit_rows(vectors["address"])
    read = _unit_rows(vectors["read"])
    write = _unit_rows(vectors["write"])
    n = int(ids.size)
    result = {
        int(operator_id): {
            "functional": [], "address": [],
        } for operator_id in ids
    }
    block = 128
    neighbor_count = min(int(k), max(0, n - 1))
    if neighbor_count:
        n_blocks = (n + block - 1) // block
        pad_rows = n_blocks * block - n

        @jax.jit
        def top_parameter_neighbors(address_all, read_all, write_all):
            address_padded = jnp.pad(address_all, ((0, pad_rows), (0, 0)))
            read_padded = jnp.pad(read_all, ((0, pad_rows), (0, 0)))
            write_padded = jnp.pad(write_all, ((0, pad_rows), (0, 0)))
            columns = jnp.arange(n, dtype=jnp.int32)[None, :]

            def one_block(block_index):
                start = block_index * block
                address_block = jax.lax.dynamic_slice_in_dim(
                    address_padded, start, block, axis=0)
                read_block = jax.lax.dynamic_slice_in_dim(
                    read_padded, start, block, axis=0)
                write_block = jax.lax.dynamic_slice_in_dim(
                    write_padded, start, block, axis=0)
                functional = (
                    (read_block @ read_all.T)
                    * (write_block @ write_all.T))
                address_cosine = address_block @ address_all.T
                rows = start + jnp.arange(block, dtype=jnp.int32)[:, None]
                self_mask = rows == columns
                functional = jnp.where(self_mask, -jnp.inf, functional)
                address_cosine = jnp.where(
                    self_mask, -jnp.inf, address_cosine)
                functional_values, functional_indices = jax.lax.top_k(
                    functional, neighbor_count)
                address_values, address_indices = jax.lax.top_k(
                    address_cosine, neighbor_count)
                return (functional_values, functional_indices,
                        address_values, address_indices)

            return jax.lax.map(
                one_block, jnp.arange(n_blocks, dtype=jnp.int32))

        (functional_values, functional_indices,
         address_values, address_indices) = (
            np.asarray(value).reshape(n_blocks * block, neighbor_count)[:n]
            for value in jax.device_get(top_parameter_neighbors(
                jnp.asarray(address, dtype=jnp.float32),
                jnp.asarray(read, dtype=jnp.float32),
                jnp.asarray(write, dtype=jnp.float32))))
        for index, operator_id in enumerate(ids):
            for name, values, indices in (
                    ("functional", functional_values, functional_indices),
                    ("address", address_values, address_indices)):
                order = np.lexsort((ids[indices[index]], -values[index]))
                rows = []
                for offset in order:
                    other = int(indices[index, offset])
                    row = {
                        "operator_id": int(ids[other]),
                        "similarity": float(values[index, offset]),
                    }
                    if name == "functional":
                        row.update({
                            "read_cosine": float(np.dot(read[index], read[other])),
                            "write_cosine": float(np.dot(write[index], write[other])),
                        })
                    rows.append(row)
                result[int(operator_id)][name] = rows
    pair_count = n * (n - 1) // 2
    sample_count = min(pair_count, 100000)
    rng = np.random.default_rng(int(seed))
    sampled_pairs = set()
    if pair_count <= sample_count:
        sampled_pairs = {(left, right) for left in range(n)
                         for right in range(left + 1, n)}
    else:
        while len(sampled_pairs) < sample_count:
            left = int(rng.integers(0, n))
            right = int(rng.integers(0, n - 1))
            if right >= left:
                right += 1
            if left > right:
                left, right = right, left
            sampled_pairs.add((left, right))
    pair_rows = []
    for left, right in sorted(sampled_pairs):
        pair_rows.append({
            "left": int(ids[left]),
            "right": int(ids[right]),
            "functional": float(
                np.dot(read[left], read[right])
                * np.dot(write[left], write[right])),
            "address": float(np.dot(address[left], address[right])),
        })
    return result, {
        "pair_count": pair_count,
        "sampled_pair_count": len(pair_rows),
        "pairs": pair_rows,
    }


def _functional_candidate_universe(
        records: Sequence[Dict[str, Any]], causal_rows: Sequence[Mapping[str, Any]],
        pool: str, pool_size: int, max_operators: int, seed: int,
        capture_threshold: float) -> Tuple[List[int], Dict[str, Any]]:
    routes = ("q", "k") if pool == "qk" else (pool,)
    frequency: Dict[int, int] = defaultdict(int)
    qualified_observations = 0
    for record in _primary_records(records):
        trace = record["trace"]
        for route in routes:
            captures = np.asarray(trace[f"{route}_captured_mass"])[:, 0]
            ids = np.asarray(trace[f"{route}_top_idx"])[:, 0, :]
            for layer in range(ids.shape[0]):
                if float(captures[layer]) < float(capture_threshold):
                    continue
                qualified_observations += 1
                for operator_id in np.unique(ids[layer]):
                    if int(operator_id) >= 0:
                        frequency[int(operator_id)] += 1
    causal_ids = sorted({
        int(row["operator_id"]) for row in causal_rows
        if row.get("candidate_valid") and (
            str(row.get("pool")) in routes)
    })
    repeated = sorted(
        (operator_id for operator_id, count in frequency.items() if count >= 2),
        key=lambda operator_id: (-frequency[operator_id], operator_id))
    selected: List[int] = []
    for operator_id in causal_ids + repeated:
        if operator_id not in selected and len(selected) < int(max_operators):
            selected.append(operator_id)
    rng = np.random.default_rng(int(seed))
    for operator_id in rng.permutation(int(pool_size)).tolist():
        if len(selected) >= int(max_operators):
            break
        if operator_id not in selected:
            selected.append(int(operator_id))
    selected.sort()
    return selected, {
        "qualified_observation_count": qualified_observations,
        "trace_union_count": len(frequency),
        "trace_union_coverage": len(frequency) / max(int(pool_size), 1),
        "causal_candidate_count": len(causal_ids),
    }


def _functional_signatures(
        records: Sequence[Dict[str, Any]], pool: str,
        candidate_ids: Sequence[int], capture_threshold: float,
        ) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    routes = ("q", "k") if pool == "qk" else (pool,)
    candidates = set(int(value) for value in candidate_ids)
    activation: Dict[int, Dict[int, float]] = defaultdict(dict)
    contribution: Dict[int, Dict[int, float]] = defaultdict(dict)
    observation = 0
    for record in _primary_records(records):
        trace = record["trace"]
        for route in routes:
            ids = np.asarray(trace[f"{route}_top_idx"])[:, 0, :]
            execution = np.asarray(trace[f"{route}_top_val"])[:, 0, :]
            coefficient = np.asarray(
                trace[f"{route}_top_coefficient"])[:, 0, :]
            captures = np.asarray(trace[f"{route}_captured_mass"])[:, 0]
            for layer in range(ids.shape[0]):
                if float(captures[layer]) >= float(capture_threshold):
                    for operator_id, gate, local in zip(
                            ids[layer], execution[layer], coefficient[layer]):
                        operator_id = int(operator_id)
                        if operator_id in candidates:
                            activation[operator_id][observation] = float(gate)
                            contribution[operator_id][observation] = abs(float(local))
                observation += 1
    for operator_id in candidates:
        activation.setdefault(operator_id, {})
        contribution.setdefault(operator_id, {})
    return dict(activation), dict(contribution)


def run_operator_functional_graph(
        ctx: AnalysisContext, records: Sequence[Dict[str, Any]],
        causal_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    neighbor_k = max(1, int(getattr(
        ctx.args, "functional_graph_neighbor_k", 16) or 16))
    capture_threshold = float(getattr(
        ctx.args, "transition_capture_threshold", PAIR_CAPTURE_THRESHOLD)
        or PAIR_CAPTURE_THRESHOLD)
    limits = {
        "qk": int(getattr(ctx.args, "functional_graph_max_operators_qk", 2048) or 2048),
        "v": int(getattr(ctx.args, "functional_graph_max_operators_v", 2048) or 2048),
        "rst": int(getattr(ctx.args, "functional_graph_max_operators_rst", 2048) or 2048),
    }
    pool_sizes = {
        "qk": int(ctx.model_cfg["n_qk"]),
        "v": int(ctx.model_cfg["n_v"]),
        "rst": int(ctx.model_cfg.get("n_rst", ctx.model_cfg.get("n_know"))),
    }
    seed = int(ctx.config.get("seed", 0))
    neighbor_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    pool_summaries: Dict[str, Any] = {}
    graph_state: Dict[str, Any] = {
        "neighbors": {}, "families": {}, "candidate_ids": {}}
    for pool_index, pool in enumerate(("qk", "v", "rst")):
        candidates, coverage = _functional_candidate_universe(
            records, causal_rows, pool, pool_sizes[pool],
            min(limits[pool], pool_sizes[pool]), seed + pool_index,
            capture_threshold)
        graph_state["candidate_ids"][pool] = candidates
        vectors = _candidate_pool_vectors(ctx, pool, candidates)
        parameter, pair_sample = _parameter_neighbors(
            vectors, candidates, neighbor_k, seed + 100 + pool_index)
        activation, contribution = _functional_signatures(
            records, pool, candidates, capture_threshold)
        activation_neighbors = _sparse_profile_neighbors(
            activation, candidates, neighbor_k)
        contribution_neighbors = _sparse_profile_neighbors(
            contribution, candidates, neighbor_k)
        sample_pairs = pair_sample.pop("pairs")
        for row in sample_pairs[:50000]:
            row["activation"] = _sparse_profile_cosine(
                activation.get(row["left"], {}), activation.get(row["right"], {}))
        classification = classify_function_address_pairs(
            [row["functional"] for row in sample_pairs],
            [row["address"] for row in sample_pairs])
        thresholds = classification.get("thresholds", {})
        no_pair_threshold = float(np.nextafter(1.0, 2.0))
        thresholds.setdefault("functional_high", no_pair_threshold)
        thresholds.setdefault("functional_low", no_pair_threshold)
        thresholds.setdefault("address_high", no_pair_threshold)
        thresholds.setdefault("address_low", no_pair_threshold)
        vector_index = {
            int(operator_id): index
            for index, operator_id in enumerate(candidates)}
        address_unit = _unit_rows(vectors["address"])
        read_unit = _unit_rows(vectors["read"])
        write_unit = _unit_rows(vectors["write"])
        for operator_id in candidates:
            left_index = vector_index[operator_id]
            for view in ("functional", "address"):
                for neighbor in parameter[operator_id][view]:
                    right_index = vector_index[int(neighbor["operator_id"])]
                    functional_value = float(
                        np.dot(read_unit[left_index], read_unit[right_index])
                        * np.dot(write_unit[left_index], write_unit[right_index]))
                    address_value = float(np.dot(
                        address_unit[left_index], address_unit[right_index]))
                    function_band = (
                        "high" if functional_value >= float(
                            thresholds["functional_high"]) else
                        "low" if functional_value <= float(
                            thresholds["functional_low"]) else None)
                    address_band = (
                        "high" if address_value >= float(
                            thresholds["address_high"]) else
                        "low" if address_value <= float(
                            thresholds["address_low"]) else None)
                    neighbor.update({
                        "read_cosine": float(np.dot(
                            read_unit[left_index], read_unit[right_index])),
                        "write_cosine": float(np.dot(
                            write_unit[left_index], write_unit[right_index])),
                        "rw_functional_similarity": functional_value,
                        "address_cosine": address_value,
                        "family_relation": (
                            f"function_{function_band}_address_{address_band}"
                            if function_band and address_band
                            else "mid_similarity_unclassified"),
                    })
            for profile_neighbors in (
                    activation_neighbors, contribution_neighbors):
                for neighbor in profile_neighbors[operator_id]:
                    right_index = vector_index[int(neighbor["operator_id"])]
                    read_value = float(np.dot(
                        read_unit[left_index], read_unit[right_index]))
                    write_value = float(np.dot(
                        write_unit[left_index], write_unit[right_index]))
                    functional_value = read_value * write_value
                    address_value = float(np.dot(
                        address_unit[left_index], address_unit[right_index]))
                    function_band = (
                        "high" if functional_value >= float(
                            thresholds["functional_high"]) else
                        "low" if functional_value <= float(
                            thresholds["functional_low"]) else None)
                    address_band = (
                        "high" if address_value >= float(
                            thresholds["address_high"]) else
                        "low" if address_value <= float(
                            thresholds["address_low"]) else None)
                    neighbor.update({
                        "read_cosine": read_value,
                        "write_cosine": write_value,
                        "rw_functional_similarity": functional_value,
                        "address_cosine": address_value,
                        "family_relation": (
                            f"function_{function_band}_address_{address_band}"
                            if function_band and address_band
                            else "mid_similarity_unclassified"),
                    })
        activation_values = [row.get("activation", 0.0) for row in sample_pairs[:50000]]
        activation_high = (
            float(np.quantile(activation_values, 0.90))
            if activation_values else None)
        if activation_high is not None and activation_high <= 0.0:
            activation_high = None
        functional_map = {
            operator_id: parameter[operator_id]["functional"]
            for operator_id in candidates}
        qualified_edges = set()
        for operator_id in candidates:
            possible = {
                int(row["operator_id"])
                for row in parameter[operator_id]["functional"]
            } | {
                int(row["operator_id"])
                for row in activation_neighbors[operator_id]
            }
            functional_scores = {
                int(row["operator_id"]): float(row["similarity"])
                for row in parameter[operator_id]["functional"]}
            activation_scores = {
                int(row["operator_id"]): float(row["similarity"])
                for row in activation_neighbors[operator_id]}
            for other in possible:
                if (functional_scores.get(other, -np.inf)
                        >= float(thresholds.get("functional_high", np.inf))
                        and activation_scores.get(other, -np.inf)
                        >= float(activation_high if activation_high is not None else np.inf)):
                    qualified_edges.add(tuple(sorted((operator_id, other))))
        families = mutual_neighbor_families(
            functional_map, sorted(qualified_edges))
        family_by_operator = {}
        for family_index, members in enumerate(families):
            family_id = f"{pool}_family_{family_index:05d}"
            family_rows.append({
                "pool": pool, "family_id": family_id,
                "members": members, "size": len(members),
                "construction": "mutual_functional_or_function_activation_quantile",
            })
            for operator_id in members:
                family_by_operator[operator_id] = family_id
        graph_state["families"][pool] = family_by_operator
        graph_state["neighbors"][pool] = {}
        for operator_id in candidates:
            row = {
                "pool": pool,
                "operator_id": operator_id,
                "family_id": family_by_operator[operator_id],
                "top_functional_neighbors": parameter[operator_id]["functional"],
                "top_address_neighbors": parameter[operator_id]["address"],
                "top_activation_profile_neighbors": activation_neighbors[operator_id],
                "top_contribution_profile_neighbors": contribution_neighbors[operator_id],
                "activation_observation_count": len(activation[operator_id]),
                "contribution_observation_count": len(contribution[operator_id]),
            }
            neighbor_rows.append(row)
            graph_state["neighbors"][pool][operator_id] = row
        sizes = [len(members) for members in families]
        counts = classification.get("counts", {})
        classified_n = max(sum(counts.values()), 1)
        empirical_sample = sample_pairs[:50000]
        pool_summaries[pool] = {
            "candidate_operator_count": len(candidates),
            "candidate_pool_coverage": len(candidates) / max(pool_sizes[pool], 1),
            **coverage,
            "family_count": len(families),
            "singleton_fraction": (
                float(np.mean(np.asarray(sizes) == 1)) if sizes else None),
            "family_size_mean": float(np.mean(sizes)) if sizes else None,
            "family_size_max": max(sizes) if sizes else 0,
            "function_high_address_high_fraction": (
                counts.get("function_high_address_high", 0) / classified_n),
            "function_high_address_low_fraction": (
                counts.get("function_high_address_low", 0) / classified_n),
            "function_low_address_high_fraction": (
                counts.get("function_low_address_high", 0) / classified_n),
            "address_function_spearman": spearman_correlation(
                [row["address"] for row in sample_pairs],
                [row["functional"] for row in sample_pairs]),
            "address_activation_spearman": spearman_correlation(
                [row["address"] for row in empirical_sample],
                [row.get("activation", 0.0) for row in empirical_sample]),
            "function_activation_spearman": spearman_correlation(
                [row["functional"] for row in empirical_sample],
                [row.get("activation", 0.0) for row in empirical_sample]),
            "similarity_thresholds": {
                **thresholds,
                "activation_high": activation_high,
            },
            "pair_sampling": pair_sample,
            "profile_neighbor_method": (
                "sparse_signature_countsketch_shortlist_then_exact_cosine"),
        }
    summary = {
        "status": "ready" if neighbor_rows else "partial",
        "neighbor_k": neighbor_k,
        "pools": pool_summaries,
        "artifacts": {
            "neighbors": ctx.store.path("operator_functional_neighbors.jsonl"),
            "families": ctx.store.path("operator_functional_families.jsonl"),
            "summary": ctx.store.path("operator_functional_graph_summary.json"),
        },
    }
    if ctx.is_primary:
        write_jsonl_atomic(
            ctx.store.path("operator_functional_neighbors.jsonl"), neighbor_rows)
        write_jsonl_atomic(
            ctx.store.path("operator_functional_families.jsonl"), family_rows)
        write_json_atomic(
            ctx.store.path("operator_functional_graph_summary.json"), summary)
    summary["_graph_state"] = graph_state
    return summary


def _load_functional_graph_state(ctx: AnalysisContext) -> Optional[Dict[str, Any]]:
    path = ctx.store.path("operator_functional_neighbors.jsonl")
    if not exists(path):
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    state: Dict[str, Any] = {
        "neighbors": defaultdict(dict),
        "families": defaultdict(dict),
        "candidate_ids": defaultdict(list),
    }
    for row in rows:
        pool = str(row["pool"])
        operator_id = int(row["operator_id"])
        state["neighbors"][pool][operator_id] = row
        state["families"][pool][operator_id] = row.get("family_id")
        state["candidate_ids"][pool].append(operator_id)
    return {
        key: {pool: value for pool, value in mapping.items()}
        for key, mapping in state.items()
    }


def _parse_group_sizes(value: Any, max_width: int) -> List[int]:
    if isinstance(value, str):
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif value is None:
        sizes = [1, 2, 4, 8]
    else:
        sizes = [int(item) for item in value]
    sizes = sorted(set(sizes))
    if not sizes or sizes[0] <= 0 or sizes[-1] > int(max_width):
        raise ValueError("group causal sizes must be positive and <= max width")
    return sizes


def _group_candidates(
        graph_state: Mapping[str, Any], record: Mapping[str, Any],
        route: str, layer: int, seed_operator: int, group_type: str,
        requested_size: int, random_seed: int) -> Tuple[List[int], Dict[str, Any]]:
    graph_pool = "qk" if route in ("q", "k") else route
    neighbors = graph_state["neighbors"][graph_pool].get(seed_operator, {})
    metadata: Dict[str, Any] = {}
    if group_type == "functional_family":
        family_id = graph_state["families"][graph_pool].get(seed_operator)
        same_family = {
            operator_id for operator_id, candidate_family
            in graph_state["families"][graph_pool].items()
            if candidate_family == family_id and operator_id != seed_operator
        }
        ordered = [
            int(row["operator_id"])
            for row in neighbors.get("top_functional_neighbors", [])
            if int(row["operator_id"]) in same_family]
        ordered.extend(sorted(same_family - set(ordered)))
        metadata["family_id"] = family_id
    elif group_type == "address_neighbors":
        ordered = [int(row["operator_id"])
                   for row in neighbors.get("top_address_neighbors", [])]
    elif group_type == "coactivation_neighbors":
        ordered = [int(row["operator_id"])
                   for row in neighbors.get(
                       "top_activation_profile_neighbors", [])]
    elif group_type == "random_active_size_matched":
        trace = record["trace"]
        ids = np.asarray(trace[f"{route}_top_idx"])[int(layer), 0, :]
        weights = np.asarray(trace[f"{route}_top_val"])[int(layer), 0, :]
        active = [(int(operator_id), float(weight))
                  for operator_id, weight in zip(ids, weights)
                  if int(operator_id) != int(seed_operator) and float(weight) > 0.0]
        seed_weight = next((
            float(weight) for operator_id, weight in zip(ids, weights)
            if int(operator_id) == int(seed_operator)), 0.0)
        magnitudes = np.asarray([weight for _, weight in active] + [seed_weight])
        edges = np.quantile(magnitudes, [0.25, 0.50, 0.75]) if magnitudes.size else []
        seed_bin = int(np.searchsorted(edges, seed_weight, side="right"))
        matched = [item for item in active
                   if int(np.searchsorted(edges, item[1], side="right")) == seed_bin]
        rng = np.random.default_rng(int(random_seed))
        if matched:
            order = rng.permutation(len(matched))
            ordered = [matched[index][0] for index in order]
        else:
            ordered = []
        metadata.update({
            "execution_magnitude_bin": seed_bin,
            "execution_magnitude_bin_edges": [float(value) for value in edges],
        })
    else:
        raise ValueError(f"unknown group type {group_type}")
    dedup = []
    for operator_id in ordered:
        if operator_id != seed_operator and operator_id not in dedup:
            dedup.append(operator_id)
    return [int(seed_operator)] + dedup[:max(0, int(requested_size) - 1)], metadata


def _group_summary(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ready"]
    effects = [abs(float(row["target_logprob_delta"])) for row in valid]
    synergies = [float(row["synergy"]) for row in valid
                 if _json_float(row.get("synergy")) is not None]
    effect_ci = _bootstrap_mean_ci(effects, seed)
    synergy_ci = _bootstrap_mean_ci(synergies, seed + 1)
    return {
        "n": len(valid),
        "mean_abs_target_logprob_delta": effect_ci["mean"],
        "abs_target_logprob_delta_ci95": effect_ci["ci95"],
        "mean_full_output_kl": (
            float(np.mean([float(row["full_output_kl"]) for row in valid]))
            if valid else None),
        "mean_recovery_ratio_final": (
            float(np.mean([float(row["recovery_ratio_final"]) for row in valid]))
            if valid else None),
        "mean_synergy": synergy_ci["mean"],
        "synergy_ci95": synergy_ci["ci95"],
        "superadditive_fraction": (
            float(np.mean(np.asarray(synergies) > 0.0)) if synergies else None),
    }


def run_group_causal_intervention(
        ctx: AnalysisContext, records: Sequence[Dict[str, Any]],
        causal_rows: Sequence[Mapping[str, Any]],
        graph_state: Mapping[str, Any]) -> Dict[str, Any]:
    max_width = int(getattr(ctx.args, "group_causal_max_width", 8) or 8)
    sizes = _parse_group_sizes(
        getattr(ctx.args, "group_causal_sizes", "1,2,4,8"), max_width)
    max_prompts = getattr(ctx.args, "group_causal_max_prompts", None)
    primary_records = _primary_records(records)
    if max_prompts is not None:
        primary_records = primary_records[:max(1, int(max_prompts))]
    record_by_prompt = {
        str(record["prompt"]["prompt_id"]): record for record in primary_records}
    ready_causal = [row for row in causal_rows if row.get("status") == "ready"]
    seeds: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for strategy in ("top_contribution", "top_gate"):
        for row in ready_causal:
            key = (str(row["prompt_id"]), str(row["pool"]))
            if str(row.get("strategy")) == strategy and key not in seeds:
                seeds[key] = row
    single_effects = {
        (str(row["prompt_id"]), str(row["pool"]), int(row["operator_id"])):
            abs(float(row["target_logprob_delta"]))
        for row in ready_causal
        if _json_float(row.get("target_logprob_delta")) is not None
    }
    forward = _canonical_group_causal_trace_forward(ctx, max_width)
    single_forward = _canonical_causal_trace_forward(ctx)
    pool_codes = {"q": 0, "k": 1, "v": 2, "rst": 3}
    group_types = (
        "functional_family", "address_neighbors",
        "coactivation_neighbors", "random_active_size_matched")
    result_rows: List[Dict[str, Any]] = []
    baseline_cache: Dict[Tuple[str, str, int], Any] = {}
    seed_value = int(ctx.config.get("seed", 0))
    for seed_index, ((prompt_id, route), seed_row) in enumerate(sorted(seeds.items())):
        record = record_by_prompt.get(prompt_id)
        if record is None:
            continue
        prompt = record["prompt"]
        layer = int(seed_row["layer"])
        target_position = int(record["target_token_index"])
        replicas = max(1, int(ctx.mesh.shape["data"]))
        input_ids = jax.device_put(jnp.asarray(np.repeat(
            np.asarray(prompt["input_array"], dtype=np.int32)[None, :],
            replicas, axis=0)), ctx.data_sharding)
        positions = jnp.full((replicas,), target_position, dtype=jnp.int32)
        cache_key = (prompt_id, route, layer)
        if cache_key not in baseline_cache:
            padded_empty = jnp.asarray(np.repeat(
                pad_group_operator_ids([], max_width)[None, :],
                replicas, axis=0))
            group_baseline = jax.device_get(forward(
                ctx.params, input_ids, positions, jnp.int32(layer),
                jnp.int32(pool_codes[route]), padded_empty))
            canonical_baseline = jax.device_get(single_forward(
                ctx.params, input_ids, positions, jnp.int32(layer),
                jnp.int32(pool_codes[route]),
                jnp.zeros((replicas,), dtype=jnp.int32), jnp.bool_(False)))
            if not all(np.array_equal(np.asarray(left), np.asarray(right))
                       for left, right in zip(group_baseline[:3], canonical_baseline[:3])):
                raise RuntimeError(
                    "fixed-width size-zero group baseline failed machine-exact parity")
            baseline_cache[cache_key] = group_baseline
        baseline = baseline_cache[cache_key]
        baseline_logits = np.asarray(baseline[0])
        baseline_residual = np.asarray(baseline[2])
        baseline_trace = baseline[3]
        length = int(prompt["length"])
        baseline_ce = _sequence_ce(
            baseline_logits, prompt["token_ids"], length)
        baseline_logp = _log_softmax_np(baseline_logits[0, :length])
        for type_index, group_type in enumerate(group_types):
            for size in sizes:
                group, group_meta = _group_candidates(
                    graph_state, record, route, layer,
                    int(seed_row["operator_id"]), group_type, size,
                    seed_value + seed_index * 100 + type_index * 10 + size)
                common = {
                    "prompt_id": prompt_id,
                    "phenomenon": prompt["phenomenon"],
                    "pool": route,
                    "layer": layer,
                    "seed_operator_id": int(seed_row["operator_id"]),
                    "seed_strategy": seed_row["strategy"],
                    "group_type": group_type,
                    "requested_group_size": int(size),
                    "group_operator_ids": group,
                    "fixed_width": max_width,
                    **group_meta,
                }
                if len(group) != int(size):
                    result_rows.append({
                        **common, "status": "partial",
                        "reason": "insufficient_qualified_group_members",
                        "actual_group_size": len(group),
                    })
                    continue
                selected = np.repeat(
                    pad_group_operator_ids(group, max_width)[None, :],
                    replicas, axis=0)
                after = jax.device_get(forward(
                    ctx.params, input_ids, positions, jnp.int32(layer),
                    jnp.int32(pool_codes[route]), jnp.asarray(selected)))
                logits = np.asarray(after[0])
                residual = np.asarray(after[2])
                after_ce = _sequence_ce(logits, prompt["token_ids"], length)
                after_logp = _log_softmax_np(logits[0, :length])
                base_prob = np.exp(baseline_logp)
                behavior_before = None if baseline_ce is None else -baseline_ce
                behavior_after = None if after_ce is None else -after_ce
                effect = (
                    None if behavior_before is None or behavior_after is None
                    else behavior_after - behavior_before)
                member_effects = [
                    single_effects.get((prompt_id, route, operator_id))
                    for operator_id in group]
                sum_single = (
                    sum(float(value) for value in member_effects)
                    if all(value is not None for value in member_effects) else None)
                max_single = (
                    max(float(value) for value in member_effects)
                    if all(value is not None for value in member_effects) else None)
                recovery = compute_causal_recovery_metrics(
                    baseline_trace, after[3], route=route, target_layer=layer,
                    baseline_final_residual=baseline_residual[0, target_position],
                    intervention_final_residual=residual[0, target_position],
                    baseline_logits=baseline_logits, intervention_logits=logits,
                    target_position=target_position)
                result_rows.append({
                    **common,
                    "status": "ready",
                    "actual_group_size": len(group),
                    "behavior_score_drop": (
                        None if behavior_before is None or behavior_after is None
                        else behavior_before - behavior_after),
                    "target_logprob_delta": effect,
                    "full_output_kl": float(np.mean(np.sum(
                        base_prob * (baseline_logp - after_logp), axis=-1))),
                    "next_token_kl": float(np.sum(
                        base_prob[-1] * (baseline_logp[-1] - after_logp[-1]))),
                    "top_prediction_changed": bool(
                        np.argmax(baseline_logp[-1]) != np.argmax(after_logp[-1])),
                    "final_residual_cosine": _cosine(
                        baseline_residual[0, target_position],
                        residual[0, target_position]),
                    "single_member_abs_effects": member_effects,
                    "sum_single_effect": sum_single,
                    "max_single_effect": max_single,
                    "synergy": (
                        abs(float(effect)) - sum_single
                        if effect is not None and sum_single is not None else None),
                    "group_effect_over_sum_single_effect": (
                        abs(float(effect)) / max(sum_single, 1.0e-12)
                        if effect is not None and sum_single is not None else None),
                    "group_effect_over_max_single_effect": (
                        abs(float(effect)) / max(max_single, 1.0e-12)
                        if effect is not None and max_single is not None else None),
                    **recovery,
                })
    valid = [row for row in result_rows if row.get("status") == "ready"]

    def grouped(key: str, offset: int) -> Dict[str, Any]:
        values = sorted({str(row[key]) for row in valid})
        return {
            value: _group_summary(
                [row for row in valid if str(row[key]) == value],
                seed_value + offset + index)
            for index, value in enumerate(values)
        }

    by_type = grouped("group_type", 100)
    def comparison(left: str, right: str) -> Dict[str, Any]:
        left_value = (by_type.get(left) or {}).get(
            "mean_abs_target_logprob_delta")
        right_value = (by_type.get(right) or {}).get(
            "mean_abs_target_logprob_delta")
        return {
            "left": left, "right": right,
            "mean_effect_difference": (
                float(left_value) - float(right_value)
                if left_value is not None and right_value is not None else None),
        }
    summary = {
        "status": "ready" if valid else "partial",
        "fixed_width": max_width,
        "group_sizes": sizes,
        "by_pool": grouped("pool", 200),
        "by_group_type": by_type,
        "by_group_size": grouped("requested_group_size", 300),
        "by_phenomenon": grouped("phenomenon", 400),
        "comparisons": {
            "functional_family_vs_random_active_size_matched": comparison(
                "functional_family", "random_active_size_matched"),
            "address_neighbors_vs_functional_family": comparison(
                "address_neighbors", "functional_family"),
            "coactivation_neighbors_vs_functional_family": comparison(
                "coactivation_neighbors", "functional_family"),
            "group_size_dose_response": grouped("requested_group_size", 500),
            "synergy_distribution": _group_summary(valid, seed_value + 600),
        },
        "artifacts": {
            "rows": ctx.store.path("group_interventions.jsonl"),
            "summary": ctx.store.path("group_intervention_summary.json"),
        },
    }
    if ctx.is_primary:
        write_jsonl_atomic(ctx.store.path("group_interventions.jsonl"), result_rows)
        write_json_atomic(ctx.store.path("group_intervention_summary.json"), summary)
    summary["_rows"] = result_rows
    return summary


RANKING_CORRELATION_PAIRS = (
    ("candidate_gate", "candidate_local_contribution"),
    ("candidate_gate", "immediate_causal_delta"),
    ("candidate_local_contribution", "immediate_causal_delta"),
    ("immediate_causal_delta", "final_residual_delta"),
    ("immediate_causal_delta", "abs_target_logprob_delta"),
    ("candidate_gate", "abs_target_logprob_delta"),
    ("candidate_local_contribution", "abs_target_logprob_delta"),
)


def _spearman_inference(
        rows: Sequence[Mapping[str, Any]], left_key: str, right_key: str,
        seed: int, draws: int = 500) -> Dict[str, Any]:
    pairs = [
        (float(row[left_key]), float(row[right_key]))
        for row in rows
        if _json_float(row.get(left_key)) is not None
        and _json_float(row.get(right_key)) is not None
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": None, "bootstrap_ci95": [None, None],
                "permutation_two_sided_p": None}
    left = np.asarray([pair[0] for pair in pairs])
    right = np.asarray([pair[1] for pair in pairs])
    observed = spearman_correlation(left, right)
    rng = np.random.default_rng(int(seed))
    bootstrap = []
    permutation = []
    for _ in range(int(draws)):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        value = spearman_correlation(left[indices], right[indices])
        if value is not None:
            bootstrap.append(value)
        null_value = spearman_correlation(left, rng.permutation(right))
        if null_value is not None:
            permutation.append(null_value)
    return {
        "n": len(pairs),
        "spearman": observed,
        "bootstrap_ci95": (
            [float(np.quantile(bootstrap, 0.025)),
             float(np.quantile(bootstrap, 0.975))]
            if bootstrap else [None, None]),
        "permutation_two_sided_p": (
            float((1 + np.sum(np.abs(permutation) >= abs(float(observed))))
                  / (len(permutation) + 1))
            if permutation and observed is not None else None),
        "inference": "bootstrap_ci95_and_permutation_null",
    }


def _ranking_correlation_block(
        rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    return {
        f"{left}_vs_{right}": _spearman_inference(
            rows, left, right, seed + index)
        for index, (left, right) in enumerate(RANKING_CORRELATION_PAIRS)
    }


def run_causal_ranking_calibration(
        ctx: AnalysisContext, causal_rows: Sequence[Mapping[str, Any]],
        group_summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    rows = []
    for row in causal_rows:
        if row.get("status") != "ready":
            continue
        rows.append({
            "prompt_id": row.get("prompt_id"),
            "phenomenon": row.get("phenomenon"),
            "pool": row.get("pool"),
            "strategy": row.get("strategy"),
            "layer": row.get("layer"),
            "operator_id": row.get("operator_id"),
            "candidate_gate": row.get(
                "candidate_admission", row.get("candidate_execution")),
            "candidate_execution": row.get("candidate_execution"),
            "candidate_local_contribution": row.get(
                "sidecar_estimated_abs_post_denominator_coefficient"),
            "immediate_causal_delta": row.get("immediate_delta_norm"),
            "final_residual_delta": row.get("final_delta_norm"),
            "abs_target_logprob_delta": (
                abs(float(row["target_logprob_delta"]))
                if _json_float(row.get("target_logprob_delta")) is not None else None),
            "full_output_kl": row.get("full_output_kl"),
        })
    seed = int(ctx.config.get("seed", 0))
    by_pool = {
        pool: _ranking_correlation_block(
            [row for row in rows if row["pool"] == pool], seed + 100 + index * 20)
        for index, pool in enumerate(sorted({str(row["pool"]) for row in rows}))
    }
    by_phenomenon = {
        phenomenon: _ranking_correlation_block(
            [row for row in rows if row["phenomenon"] == phenomenon],
            seed + 300 + index * 20)
        for index, phenomenon in enumerate(sorted({
            str(row["phenomenon"]) for row in rows}))
    }
    strategy_pairs = (
        ("top_gate", "active_random"),
        ("top_contribution", "active_random"),
        ("top_gate", "matched_active"),
        ("top_contribution", "matched_active"),
        ("matched_active", "active_random"),
    )
    metric_keys = (
        "abs_target_logprob_delta", "immediate_causal_delta", "full_output_kl")
    pairwise = {
        f"{left}_gt_{right}": {
            metric: pairwise_win_rate(rows, left, right, metric)
            for metric in metric_keys
        }
        for left, right in strategy_pairs
    }
    overall = _ranking_correlation_block(rows, seed)
    local = overall.get(
        "candidate_local_contribution_vs_immediate_causal_delta", {})
    immediate_final = overall.get(
        "immediate_causal_delta_vs_abs_target_logprob_delta", {})
    gate = overall.get("candidate_gate_vs_immediate_causal_delta", {})
    local_ci = local.get("bootstrap_ci95") or [None, None]
    final_ci = immediate_final.get("bootstrap_ci95") or [None, None]
    gate_ci = gate.get("bootstrap_ci95") or [None, None]
    group_summary = dict(group_summary or {})
    type_summary = group_summary.get("by_group_type", {})
    functional = type_summary.get("functional_family", {})
    random_group = type_summary.get("random_active_size_matched", {})
    functional_effect = functional.get("mean_abs_target_logprob_delta")
    random_effect = random_group.get("mean_abs_target_logprob_delta")
    synergy_ci = functional.get("synergy_ci95") or [None, None]
    judgments = {
        "local_ranking_valid": {
            "supported": bool(local_ci[0] is not None and local_ci[0] > 0.0),
            "evidence": local,
        },
        "downstream_compensation_dominant": {
            "supported": bool(
                final_ci[0] is not None and final_ci[1] is not None
                and final_ci[0] <= 0.0 <= final_ci[1]),
            "evidence": immediate_final,
        },
        "gate_ranking_weak": {
            "supported": bool(
                gate_ci[0] is not None and gate_ci[1] is not None
                and gate_ci[0] <= 0.0 <= gate_ci[1]
                and _json_float(local.get("spearman")) is not None
                and (_json_float(gate.get("spearman")) is None
                     or abs(float(local["spearman"])) > abs(float(gate["spearman"])))),
            "evidence": {"gate": gate, "local_contribution": local},
        },
        "functional_redundancy_candidate": {
            "supported": bool(
                functional_effect is not None and random_effect is not None
                and float(functional_effect) > float(random_effect)
                and synergy_ci[0] is not None and synergy_ci[0] > 0.0),
            "evidence": {
                "functional_family": functional,
                "random_active_size_matched": random_group,
            },
        },
    }
    summary = {
        "status": "ready" if rows else "partial",
        "row_count": len(rows),
        "correlations": {
            "overall": overall,
            "by_pool": by_pool,
            "by_phenomenon": by_phenomenon,
        },
        "pairwise_win_rates": pairwise,
        "judgments": judgments,
        "artifacts": {
            "summary": ctx.store.path("causal_ranking_calibration.json"),
            "rows": ctx.store.path("causal_ranking_rows.csv"),
        },
    }
    if ctx.is_primary:
        write_csv_atomic(ctx.store.path("causal_ranking_rows.csv"), rows)
        write_json_atomic(
            ctx.store.path("causal_ranking_calibration.json"), summary)
    return summary


def _failed_analysis_item(item: str, exc: Exception) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "dependencies": list(ITEM_DEPENDENCIES.get(item, ())),
    }


def _dependency_failed_item(item: str, dependencies: Sequence[str]) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error": "required dependency did not produce reusable ready/partial rows",
        "failed_dependencies": list(dependencies),
        "dependencies": list(ITEM_DEPENDENCIES.get(item, ())),
    }


def run_v4171_transition_items(
    ctx: AnalysisContext,
    items: Sequence[str],
) -> Dict[str, Any]:
    requested = set(items)
    execute = set(requested)
    if "causal_recovery_trace" in execute:
        execute.add("causal_intervention")
    if "operator_functional_graph" in execute:
        execute.add("trajectory_trace")
    if "group_causal_intervention" in execute:
        execute.update((
            "trajectory_trace", "causal_intervention",
            "operator_functional_graph"))
    if "causal_ranking_calibration" in execute:
        execute.update(("causal_intervention", "causal_recovery_trace"))
    selected = [item for item in CORE_TRANSITION_ITEMS if item in execute]
    if not selected:
        return {}
    if str(ctx.model_cfg.get("model_version")) != V4171_MODEL_VERSION:
        return {
            item: {
                "status": "failed",
                "reason": f"model_version={ctx.model_cfg.get('model_version')}",
            }
            for item in selected
        }
    result: Dict[str, Any] = {}
    if "global_router_audit" in selected:
        result["global_router_audit"] = run_global_router_audit(ctx)
    needs_trace = any(item != "global_router_audit" for item in selected)
    records: List[Dict[str, Any]] = []
    if needs_trace:
        cache, records = run_transition_trace_cache(ctx)
        result["transition_trace_cache"] = cache
        if "trajectory_trace" in selected:
            result["trajectory_trace"] = {
                **cache,
                "tensor_semantics": {
                    "qkv": "SRW features feeding attention; not residual updates",
                    "attention": "actual residual update after softmax attention and expand_O",
                    "rst": "actual residual update after denominator and fixed pool scale",
                },
            }
    if any(item in selected for item in ("context_divergence", "state_transition_decoupling")):
        prompt_hash_for_pairs = (result.get("transition_trace_cache") or {}).get(
            "prompt_set_hash")
        context = _load_resumable_summary(
            ctx, "context_divergence.json", prompt_hash_for_pairs,
            required_keys=("pairs",)) or {}
        decoupling = _load_resumable_summary(
            ctx, "state_transition_decoupling.json", prompt_hash_for_pairs,
            required_keys=("path_similarity",)) or {}
        needs_context = "context_divergence" in selected and not context
        needs_decoupling = (
            "state_transition_decoupling" in selected and not decoupling)
        if needs_context or needs_decoupling:
            computed_context, computed_decoupling, _ = build_pair_analyses(
                ctx, records)
            if not context:
                context = computed_context
            if not decoupling:
                decoupling = computed_decoupling
        if "context_divergence" in selected:
            result["context_divergence"] = context
        if "state_transition_decoupling" in selected:
            result["state_transition_decoupling"] = decoupling
    prompt_hash = (result.get("transition_trace_cache") or {}).get(
        "prompt_set_hash")
    causal_summary: Dict[str, Any] = {}
    causal_rows: List[Dict[str, Any]] = []
    if "causal_intervention" in selected:
        resumed_causal = _load_resumable_summary(
            ctx, "causal_intervention_summary.json", prompt_hash,
            required_keys=("zero_suppression_parity", "effects"))
        interventions_path = ctx.store.path("interventions.jsonl")
        if (resumed_causal is not None
                and bool((resumed_causal.get("zero_suppression_parity") or {}).get(
                    "machine_exact"))
                and exists(interventions_path)):
            causal_summary = resumed_causal
            causal_rows = read_jsonl(interventions_path)
        else:
            try:
                causal_summary = run_causal_intervention(ctx, records)
                causal_rows = list(causal_summary.pop("_result_rows", []))
            except RuntimeError as exc:
                if "parity" in str(exc).lower():
                    raise
                causal_summary = _failed_analysis_item(
                    "causal_intervention", exc)
            except Exception as exc:
                causal_summary = _failed_analysis_item(
                    "causal_intervention", exc)
        if "causal_intervention" in requested:
            result["causal_intervention"] = causal_summary
    recovery_summary: Dict[str, Any] = {}
    if "causal_recovery_trace" in selected:
        recovery_summary = _load_resumable_summary(
            ctx, "causal_recovery_summary.json", prompt_hash,
            required_keys=("overall", "classification_basis")) or {}
        if not recovery_summary:
            if not causal_rows:
                recovery_summary = _dependency_failed_item(
                    "causal_recovery_trace", ("causal_intervention",))
            else:
                try:
                    recovery_summary = run_causal_recovery_trace(
                        ctx, causal_rows)
                    recovery_summary.pop("_rows", None)
                except Exception as exc:
                    recovery_summary = _failed_analysis_item(
                        "causal_recovery_trace", exc)
        if "causal_recovery_trace" in requested:
            result["causal_recovery_trace"] = recovery_summary
    graph_summary: Dict[str, Any] = {}
    graph_state: Dict[str, Any] = {}
    if "operator_functional_graph" in selected:
        graph_summary = _load_resumable_summary(
            ctx, "operator_functional_graph_summary.json", prompt_hash,
            required_keys=("pools", "neighbor_k")) or {}
        if graph_summary:
            graph_state = _load_functional_graph_state(ctx) or {}
            if not graph_state:
                graph_summary = {}
        if not graph_summary:
            try:
                graph_summary = run_operator_functional_graph(
                    ctx, records, causal_rows)
                graph_state = dict(graph_summary.pop("_graph_state", {}))
            except Exception as exc:
                graph_summary = _failed_analysis_item(
                    "operator_functional_graph", exc)
        if "operator_functional_graph" in requested:
            result["operator_functional_graph"] = graph_summary
    group_summary: Dict[str, Any] = {}
    if "group_causal_intervention" in selected:
        group_summary = _load_resumable_summary(
            ctx, "group_intervention_summary.json", prompt_hash,
            required_keys=("by_group_type", "group_sizes")) or {}
        if not group_summary:
            if not causal_rows or not graph_state:
                failed = []
                if not causal_rows:
                    failed.append("causal_intervention")
                if not graph_state:
                    failed.append("operator_functional_graph")
                group_summary = _dependency_failed_item(
                    "group_causal_intervention", failed)
            else:
                try:
                    group_summary = run_group_causal_intervention(
                        ctx, records, causal_rows, graph_state)
                    group_summary.pop("_rows", None)
                except RuntimeError as exc:
                    if "parity" in str(exc).lower():
                        raise
                    group_summary = _failed_analysis_item(
                        "group_causal_intervention", exc)
                except Exception as exc:
                    group_summary = _failed_analysis_item(
                        "group_causal_intervention", exc)
        if "group_causal_intervention" in requested:
            result["group_causal_intervention"] = group_summary
    if "causal_ranking_calibration" in selected:
        ranking = _load_resumable_summary(
            ctx, "causal_ranking_calibration.json", prompt_hash,
            required_keys=("correlations", "pairwise_win_rates"))
        if ranking is None:
            if not causal_rows:
                ranking = _dependency_failed_item(
                    "causal_ranking_calibration", ("causal_intervention",))
            else:
                try:
                    ranking = run_causal_ranking_calibration(
                        ctx, causal_rows, group_summary)
                except Exception as exc:
                    ranking = _failed_analysis_item(
                        "causal_ranking_calibration", exc)
        if "causal_ranking_calibration" in requested:
            result["causal_ranking_calibration"] = ranking

    trace_summary = result.get("transition_trace_cache", {})
    parity = causal_summary.get("zero_suppression_parity") or None
    cross_graph = causal_summary.get(
        "normal_production_cross_graph_audit") or None
    provenance = _analysis_provenance(
        ctx, trace_summary.get("prompt_set_hash"), parity, cross_graph)
    for key, value in result.items():
        if isinstance(value, dict):
            if key != "transition_trace_cache":
                status = str(value.get("status") or "partial")
                if status not in ("ready", "partial", "not_requested", "failed"):
                    value["raw_status"] = status
                    value["status"] = (
                        "failed" if status.startswith("unsupported") else "partial")
            value.update(provenance)
            value.setdefault("dependencies", list(ITEM_DEPENDENCIES.get(key, ())))
    for value in (
            causal_summary, recovery_summary, graph_summary, group_summary):
        if value:
            value.update(provenance)
    for key, value in (
            ("causal_intervention", causal_summary),
            ("causal_recovery_trace", recovery_summary),
            ("operator_functional_graph", graph_summary),
            ("group_causal_intervention", group_summary)):
        if value:
            value.setdefault("dependencies", list(ITEM_DEPENDENCIES.get(key, ())))
    if causal_summary:
        causal_summary.update({
            "causal_baseline": "canonical_suppression_disabled",
            "effect_reference": "canonical_suppression_disabled",
            "canonical_parity_machine_exact": (
                bool(parity["machine_exact"]) if parity else None),
            "cross_graph_audit_blocking": False,
        })
    if ctx.is_primary:
        if parity:
            parity_artifact = dict(parity)
            parity_artifact.update(provenance)
            write_json_atomic(
                ctx.store.path("intervention_forward_parity.json"),
                parity_artifact)
        if cross_graph:
            cross_graph_artifact = dict(cross_graph)
            cross_graph_artifact.update(provenance)
            cross_graph_artifact["blocking"] = False
            write_json_atomic(
                ctx.store.path("normal_production_vs_canonical_neutral.json"),
                cross_graph_artifact)
        summary_paths = {
            "global_router_audit": "global_router_audit.json",
            "trajectory_trace": "transition_trace_summary.json",
            "context_divergence": "context_divergence.json",
            "state_transition_decoupling": "state_transition_decoupling.json",
            "causal_intervention": "causal_intervention_summary.json",
            "causal_recovery_trace": "causal_recovery_summary.json",
            "operator_functional_graph": "operator_functional_graph_summary.json",
            "group_causal_intervention": "group_intervention_summary.json",
            "causal_ranking_calibration": "causal_ranking_calibration.json",
        }
        dependency_summaries = {
            "global_router_audit": result.get("global_router_audit", {}),
            "trajectory_trace": (
                result.get("trajectory_trace")
                or result.get("transition_trace_cache", {})),
            "context_divergence": result.get("context_divergence", {}),
            "state_transition_decoupling": result.get(
                "state_transition_decoupling", {}),
            "causal_intervention": causal_summary,
            "causal_recovery_trace": recovery_summary,
            "operator_functional_graph": graph_summary,
            "group_causal_intervention": group_summary,
            "causal_ranking_calibration": result.get(
                "causal_ranking_calibration", {}),
        }
        for item, path in summary_paths.items():
            if dependency_summaries[item]:
                write_json_atomic(
                    ctx.store.path(path), dependency_summaries[item])
    if ctx.is_primary:
        metrics_rows = [
            {"item": item, **value}
            for item, value in result.items()
            if item != "transition_trace_cache" and isinstance(value, dict)
        ]
        write_jsonl_atomic(ctx.store.path("metrics.jsonl"), metrics_rows)
    return result
