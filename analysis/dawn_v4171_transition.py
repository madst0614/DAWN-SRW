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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import (
    AnalysisContext,
    analysis_model_module,
    maybe_load_tokenizer,
)
from analysis.dawn_analysis_storage import (
    open_path,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_npz_atomic,
)
from analysis.dawn_analysis_trace import TRACE_FIELDS, TRACE_POOLS, topk_trace_forward


V4171_MODEL_VERSION = "spatial-r1-v4.1.7.1"
DEFAULT_TRANSITION_PROMPT_SET = str(
    Path(__file__).resolve().parent / "prompts" / "v4171_transition_pairs.jsonl"
)
CORE_TRANSITION_ITEMS = (
    "global_router_audit",
    "trajectory_trace",
    "context_divergence",
    "state_transition_decoupling",
    "causal_intervention",
)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "row"


def _json_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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
        for field in TRACE_FIELDS
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
    topk = int(getattr(args, "trace_topk", None) or 64)
    prompts = [_tokenize_transition_row(tokenizer, row, seq_len) for row in rows]
    trace_fn = jax.jit(
        lambda p, x, t: topk_trace_forward(
            p, ctx.model_cfg, x, topk=topk, target_positions=t))
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
            input_ids = jnp.asarray(prompt["input_array"][None, :], dtype=jnp.int32)
            target = jnp.asarray([int(token_index)], dtype=jnp.int32)
            trace = jax.device_get(trace_fn(ctx.params, input_ids, target))
            record = _trace_internal_record(prompt, int(token_index), trace)
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
    captured = []
    for record in internal:
        for pool in TRACE_POOLS:
            captured.extend(
                np.asarray(record["trace"][f"{pool}_captured_mass"])[:, 0].tolist())
    summary = {
        "status": "ready",
        "prompt_set": prompt_set,
        "prompt_set_hash": prompt_hash,
        "num_prompts": len(prompts),
        "num_target_subtokens": len(internal),
        "trace_topk": topk,
        "span_aggregation": "per-subtoken JSONL plus span_mean/span_last vectors in NPZ",
        "captured_mass": {
            "mean": float(np.mean(captured)) if captured else None,
            "min": float(np.min(captured)) if captured else None,
            "p10": float(np.quantile(captured, 0.10)) if captured else None,
        },
        "sec": time.time() - started,
        "artifacts": {
            "trajectory_traces": ctx.store.path("trajectory_traces.jsonl"),
            "transition_trace_cache": ctx.store.path("transition_trace_cache.npz"),
        },
    }
    if summary["captured_mass"]["min"] is not None and summary["captured_mass"]["min"] < 0.90:
        summary["captured_mass_warning"] = (
            "Sparse metrics are approximate because at least one target trace captured <90% mass")
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
    keys = sorted(set(a) | set(b))
    if not keys:
        return {
            "gate_cosine": None,
            "weighted_jaccard": None,
            "intersection": 0,
            "union": 0,
        }
    va = np.asarray([a[key] for key in keys], dtype=np.float64)
    vb = np.asarray([b[key] for key in keys], dtype=np.float64)
    den = float(np.sum(np.maximum(va, vb)))
    return {
        "gate_cosine": _cosine(va, vb),
        "weighted_jaccard": (
            float(np.sum(np.minimum(va, vb)) / den) if den > 1.0e-12 else None),
        "intersection": len(set(a) & set(b)),
        "union": len(keys),
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
            components = [value for value in (query_cos, sparse["gate_cosine"], delta_cos)
                          if value is not None]
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
                "active_intersection": sparse["intersection"],
                "active_union": sparse["union"],
                "delta_similarity": delta_cos,
                "delta_relative_error": delta_rel,
                "trajectory_similarity": float(np.mean(components)) if components else None,
                "captured_mass_a": captured_a,
                "captured_mass_b": captured_b,
                "gate_similarity_exact": False,
                "update_kind": update_kind[pool],
            })
    return rows


def _context_divergence_summary(pair_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    contextual_relations = {
        "same_surface_different_function",
        "same_lexical_different_function",
    }
    for row in pair_rows:
        if row.get("is_random_null") or row.get("pair_type") not in contextual_relations:
            continue
        groups[(str(row["pair_id"]), str(row["pool"]))].append(row)
    summaries = []
    for (pair_id, pool), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: int(row["layer"]))
        sims = [row.get("weighted_jaccard") for row in rows]
        threshold = _quantile(sims, 0.25)
        diverged = [row for row in rows
                    if threshold is not None and row.get("weighted_jaccard") is not None
                    and float(row["weighted_jaccard"]) <= threshold]
        first_layer = int(diverged[0]["layer"]) if diverged else None
        max_row = min(
            rows,
            key=lambda row: float(row.get("weighted_jaccard")
                                  if row.get("weighted_jaccard") is not None else 1.0),
        ) if rows else None
        late = rows[max(0, len(rows) * 2 // 3):]
        reconverged = bool(
            diverged and threshold is not None and any(
                row.get("weighted_jaccard") is not None
                and float(row["weighted_jaccard"]) > threshold
                for row in late))
        summaries.append({
            "pair_id": pair_id,
            "pool": pool,
            "divergence_threshold_data_q25": threshold,
            "first_divergence_layer": first_layer,
            "maximum_divergence_layer": int(max_row["layer"]) if max_row else None,
            "late_reconvergence": reconverged,
            "mean_state_similarity": _json_float(np.mean([
                row["state_similarity"] for row in rows if row.get("state_similarity") is not None
            ])) if rows else None,
            "mean_query_similarity": _json_float(np.mean([
                row["query_similarity"] for row in rows if row.get("query_similarity") is not None
            ])) if rows else None,
            "mean_gate_similarity": _json_float(np.mean([
                row["gate_similarity"] for row in rows if row.get("gate_similarity") is not None
            ])) if rows else None,
            "mean_delta_similarity": _json_float(np.mean([
                row["delta_similarity"] for row in rows if row.get("delta_similarity") is not None
            ])) if rows else None,
            "min_captured_mass": min(
                min(float(row["captured_mass_a"]), float(row["captured_mass_b"]))
                for row in rows),
        })
    return {
        "status": "ready" if summaries else "insufficient_evidence",
        "pairs": summaries,
        "num_pairs": len({row["pair_id"] for row in summaries}),
        "gate_metric": "sparse_topk_captured_mass_weighted",
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
            def mean_key(key: str) -> Optional[float]:
                vals = [float(row[key]) for row in rows if row.get(key) is not None]
                return float(np.mean(vals)) if vals else None
            out.append({
                "pair_id": pair_id,
                "pair_type": rows[0].get("pair_type"),
                "phenomenon": rows[0].get("phenomenon"),
                "pool": pool,
                "is_random_null": is_null,
                "state_cos": mean_key("state_similarity"),
                "query_cos": mean_key("query_similarity"),
                "gate_sim": mean_key("gate_similarity"),
                "delta_cos": mean_key("delta_similarity"),
                "path_sim": mean_key("trajectory_similarity"),
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
    actual_paths = [float(row["path_sim"]) for row in rows if row.get("path_sim") is not None]
    null_paths = [float(row["path_sim"]) for row in null if row.get("path_sim") is not None]
    effect = (
        float(np.mean(actual_paths) - np.mean(null_paths))
        if actual_paths and null_paths else None)
    return {
        "status": "ready" if rows else "insufficient_evidence",
        "state_low_threshold_data_q25": state_threshold,
        "transition_high_threshold_random_null_q75": transition_threshold,
        "quadrants": dict(quadrants),
        "correlations": correlations,
        "path_similarity_effect_vs_random": effect,
        "path_similarity": _bootstrap_mean_ci(actual_paths, seed),
        "random_null_path_similarity": _bootstrap_mean_ci(null_paths, seed + 1),
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
        for idx, record_a in enumerate(primary):
            candidates = [
                row for row in primary
                if row["prompt"]["pair_id"] != record_a["prompt"]["pair_id"]
            ]
            if not candidates:
                continue
            length_a = int(record_a["prompt"]["length"])
            length_deltas = np.asarray([
                abs(int(row["prompt"]["length"]) - length_a)
                for row in candidates
            ], dtype=np.int32)
            closest = np.flatnonzero(length_deltas == int(length_deltas.min()))
            record_b = candidates[int(rng.choice(closest))]
            null_pair_id = f"random-null-{idx:04d}"
            rows = _pair_layer_rows(
                record_a,
                record_b,
                pair_type="random_length_matched_control",
                is_null=True,
            )
            for row in rows:
                row["pair_id"] = null_pair_id
            null_rows.extend(rows)
    context = _context_divergence_summary(actual_rows)
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


def _target_intervention_forward(
    params: Any,
    model_cfg: Dict[str, Any],
    input_ids: Any,
    target_position: Any,
    target_layer: Any,
    target_pool: Any,
    suppress_qk: Any,
    suppress_v: Any,
    suppress_rst: Any,
):
    """Analysis-only token/layer ablation with the canonical unpruned denominator."""
    model_module = analysis_model_module(model_cfg)
    params = model_module._squeeze_params(params)
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    target_position = jnp.asarray(target_position, dtype=jnp.int32)
    target_layer = jnp.asarray(target_layer, dtype=jnp.int32)
    target_pool = jnp.asarray(target_pool, dtype=jnp.int32)
    bsz, seq_len = input_ids.shape
    d_model = int(model_cfg["d_model"])
    n_layers = int(model_cfg["n_layers"])
    n_heads = int(model_cfg["n_heads"])
    d_head = d_model // n_heads
    execution_kwargs = model_module._angular_execution_kwargs_from_model_cfg(model_cfg)
    admission_den_power = float(execution_kwargs.pop("admission_den_power"))
    composition_mode = execution_kwargs.get(
        "srw_composition_mode", model_module.DEFAULT_SRW_COMPOSITION_MODE)
    temperature_qk = float(model_cfg.get(
        "soft_gate_T_qk", execution_kwargs["soft_gate_temperature"]))
    temperature_v = float(model_cfg.get(
        "soft_gate_T_v", execution_kwargs["soft_gate_temperature"]))
    temperature_rst = float(model_cfg.get(
        "soft_gate_T_rst", execution_kwargs["soft_gate_temperature"]))
    pool = model_module._pool_params_with_operator_keys(params["neuron_pool"])
    router = params["router"]
    qk_scale, v_scale, rst_scale = model_module._effective_pool_output_scales(
        pool, d_model, n_layers)
    positions = jnp.arange(seq_len)[None, :]
    x = (
        params["token_emb"]["embedding"][input_ids]
        + params["pos_emb"]["embedding"][positions]
    )
    token_selector = jax.nn.one_hot(
        target_position, seq_len, dtype=jnp.float32)[None, :, None]
    target_attention_updates = []
    target_rst_updates = []

    def srw(
        x_in: Any,
        query: Any,
        op_key: Any,
        raw_tau: Any,
        read: Any,
        write: Any,
        suppress_mask: Any,
        layer_index: int,
        pool_code: int,
        gate_temperature: float,
    ):
        local_execution_kwargs = dict(execution_kwargs)
        local_execution_kwargs["soft_gate_temperature"] = float(gate_temperature)
        _, admission, _, execution_weight, _ = model_module._angular_execution(
            query, op_key, raw_tau, None, **local_execution_kwargs)
        apply_here = (target_layer == int(layer_index)) & (target_pool == int(pool_code))
        keep = 1.0 - token_selector * jnp.asarray(suppress_mask, dtype=jnp.float32)[None, None, :]
        execution_weight = jnp.where(
            apply_here,
            execution_weight * keep,
            execution_weight,
        )
        read_n = model_module._forward_unit_direction(read.astype(jnp.float32))
        write_n = model_module._forward_unit_direction(write.astype(jnp.float32))
        response = x_in.astype(jnp.float32) @ read_n.T
        numerator = (execution_weight * response) @ write_n
        # Important: admission is deliberately not masked.  This is the
        # canonical selection/gate ablation denominator requested for v4171.
        den = model_module._composition_den(
            admission.sum(axis=-1, keepdims=True),
            admission_den_power,
            composition_mode,
        )
        return numerator.astype(jnp.float32) / den

    for layer_index in range(n_layers):
        bp = params[f"block_{layer_index}"]
        normed = model_module._layer_norm(
            x, bp["norm1"]["scale"], bp["norm1"]["bias"])
        queries = normed @ router["proj_attn"]["kernel"] + router["proj_attn"]["bias"]
        query_q, query_k, query_v = jnp.split(queries, 3, axis=-1)
        tau_all = normed @ router["raw_tau_attn"]["kernel"] + router["raw_tau_attn"]["bias"]
        q = srw(
            normed, query_q, pool["attn_qk_op_key"], tau_all[:, :, 0:1],
            pool["attn_qk_read"], pool["attn_qk_write"], suppress_qk,
            layer_index, 0, temperature_qk) * qk_scale
        k = srw(
            normed, query_k, pool["attn_qk_op_key"], tau_all[:, :, 1:2],
            pool["attn_qk_read"], pool["attn_qk_write"], suppress_qk,
            layer_index, 1, temperature_qk) * qk_scale
        v = srw(
            normed, query_v, pool["attn_v_op_key"], tau_all[:, :, 2:3],
            pool["attn_v_read"], pool["attn_v_write"], suppress_v,
            layer_index, 2, temperature_v) * v_scale
        q = q.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhsd,bhtd->bhst", q, k) / jnp.sqrt(jnp.float32(d_head))
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attention = jax.nn.softmax(scores, axis=-1)
        delta_attention = jnp.einsum("bhst,bhtd->bhsd", attention, v)
        delta_attention = delta_attention.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, d_model)
        delta_attention = delta_attention @ bp["attn"]["expand_O"]["kernel"]
        target_attention_updates.append(
            delta_attention[jnp.arange(bsz), target_position])
        x = x + delta_attention

        normed = model_module._layer_norm(
            x, bp["norm2"]["scale"], bp["norm2"]["bias"])
        query_rst = normed @ router["proj_rst"]["kernel"] + router["proj_rst"]["bias"]
        tau_rst = normed @ router["raw_tau_rst"]["kernel"] + router["raw_tau_rst"]["bias"]
        delta_rst = srw(
            normed, query_rst, pool["rst_op_key"], tau_rst,
            pool["rst_read"], pool["rst_write"], suppress_rst,
            layer_index, 3, temperature_rst) * rst_scale
        target_rst_updates.append(delta_rst[jnp.arange(bsz), target_position])
        x = x + delta_rst

    residual = x
    norm = params["norm"]
    x = model_module._layer_norm(x, norm["scale"], norm["bias"])
    logits = x @ params["token_emb"]["embedding"].T
    logits = model_module._slice_logits_to_logical_vocab(logits, model_cfg)
    return (
        logits,
        residual,
        jnp.stack(target_attention_updates, axis=0),
        jnp.stack(target_rst_updates, axis=0),
    )


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
    pool_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    trace = record["trace"]
    coefficients = np.abs(np.asarray(trace[f"{pool}_top_coefficient"])[:, 0, :])
    ids = np.asarray(trace[f"{pool}_top_idx"])[:, 0, :]
    weights = np.asarray(trace[f"{pool}_top_val"])[:, 0, :]
    flat = int(np.argmax(coefficients))
    layer, rank = np.unravel_index(flat, coefficients.shape)
    selected_id = int(ids[layer, rank])
    gate_flat = int(np.argmax(weights))
    gate_layer, gate_rank = np.unravel_index(gate_flat, weights.shape)
    top_gate_id = int(ids[gate_layer, gate_rank])
    active_ids = [int(value) for value in ids[layer].tolist()]
    seed_material = (
        f"{int(seed)}|{record['prompt']['prompt_id']}|{pool}".encode("utf-8"))
    local_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    active_options = [value for value in active_ids if value != selected_id]
    active_random = (
        int(rng.choice(active_options)) if active_options else selected_id)
    active_set = set(active_ids)
    inactive_random = selected_id
    for _ in range(64):
        candidate = int(rng.integers(0, max(1, int(pool_size))))
        if candidate not in active_set:
            inactive_random = candidate
            break
    if inactive_random == selected_id:
        inactive_random = next(
            (value for value in range(int(pool_size)) if value not in active_set),
            selected_id,
        )
    target_magnitude = float(coefficients[layer, rank])
    matched_options = [
        (abs(float(coefficients[layer, idx]) - target_magnitude), int(ids[layer, idx]))
        for idx in range(ids.shape[1]) if int(ids[layer, idx]) != selected_id
    ]
    matched_id = min(matched_options)[1] if matched_options else active_random
    candidates = [
        {"strategy": "top_contribution", "layer": int(layer), "operator_id": selected_id},
        {"strategy": "top_gate", "layer": int(gate_layer), "operator_id": top_gate_id},
        {"strategy": "inactive_random", "layer": int(layer), "operator_id": inactive_random},
        {"strategy": "active_random", "layer": int(layer), "operator_id": active_random},
        {"strategy": "matched_control", "layer": int(layer), "operator_id": matched_id},
    ]
    dedup = []
    seen = set()
    for row in candidates:
        key = (row["strategy"], row["layer"], row["operator_id"])
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    return dedup


def run_causal_intervention(
    ctx: AnalysisContext,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    primary = _primary_records(records)
    max_prompts = max(1, int(getattr(ctx.args, "causal_max_prompts", 2) or 2))
    primary = primary[:max_prompts]
    mcfg = ctx.config.get("model", {})
    pool_sizes = {
        "q": int(mcfg.get("n_qk", 0)),
        "k": int(mcfg.get("n_qk", 0)),
        "v": int(mcfg.get("n_v", 0)),
        "rst": int(mcfg.get("n_rst", mcfg.get("n_know", 0))),
    }
    pool_codes = {"q": 0, "k": 1, "v": 2, "rst": 3}
    forward = jax.jit(
        lambda p, x, pos, layer, pool_code, mq, mv, mr: _target_intervention_forward(
            p, ctx.model_cfg, x, pos, layer, pool_code, mq, mv, mr))
    result_rows: List[Dict[str, Any]] = []
    intervention_seed = int(ctx.config.get("seed", 0))
    started = time.time()
    for prompt_idx, record in enumerate(primary):
        prompt = record["prompt"]
        input_ids_np = np.asarray(prompt["input_array"], dtype=np.int32)[None, :]
        target_position = int(record["target_token_index"])
        zeros_qk = np.zeros((pool_sizes["q"],), dtype=np.float32)
        zeros_v = np.zeros((pool_sizes["v"],), dtype=np.float32)
        zeros_rst = np.zeros((pool_sizes["rst"],), dtype=np.float32)
        (baseline_logits, baseline_residual,
         baseline_attention_updates, baseline_rst_updates) = jax.device_get(forward(
            ctx.params,
            jnp.asarray(input_ids_np),
            jnp.int32(target_position),
            jnp.int32(-1),
            jnp.int32(-1),
            jnp.asarray(zeros_qk),
            jnp.asarray(zeros_v),
            jnp.asarray(zeros_rst),
        ))
        baseline_logits = np.asarray(baseline_logits)
        baseline_residual = np.asarray(baseline_residual)
        baseline_attention_updates = np.asarray(baseline_attention_updates)
        baseline_rst_updates = np.asarray(baseline_rst_updates)
        length = int(prompt["length"])
        pred_pos = max(0, length - 1)
        continuation_ids = prompt.get("continuation_token_ids") or []
        target_token_id = int(continuation_ids[0]) if continuation_ids else None
        base_logp = _log_softmax_np(baseline_logits[0, pred_pos])
        base_logp_all = _log_softmax_np(baseline_logits[0, :length])
        baseline_ce = _sequence_ce(baseline_logits, prompt["token_ids"], length)
        for pool in TRACE_POOLS:
            for candidate in _intervention_candidates(
                record, pool, pool_sizes[pool], intervention_seed):
                masks = {
                    "qk": zeros_qk.copy(),
                    "v": zeros_v.copy(),
                    "rst": zeros_rst.copy(),
                }
                mask_key = "qk" if pool in ("q", "k") else pool
                masks[mask_key][int(candidate["operator_id"])] = 1.0
                (logits, residual,
                 attention_updates, rst_updates) = jax.device_get(forward(
                    ctx.params,
                    jnp.asarray(input_ids_np),
                    jnp.int32(target_position),
                    jnp.int32(candidate["layer"]),
                    jnp.int32(pool_codes[pool]),
                    jnp.asarray(masks["qk"]),
                    jnp.asarray(masks["v"]),
                    jnp.asarray(masks["rst"]),
                ))
                logits = np.asarray(logits)
                residual = np.asarray(residual)
                attention_updates = np.asarray(attention_updates)
                rst_updates = np.asarray(rst_updates)
                logp = _log_softmax_np(logits[0, pred_pos])
                logp_all = _log_softmax_np(logits[0, :length])
                next_prob = np.exp(base_logp)
                next_token_kl = float(np.sum(next_prob * (base_logp - logp)))
                base_prob_all = np.exp(base_logp_all)
                full_output_kl = float(np.mean(np.sum(
                    base_prob_all * (base_logp_all - logp_all), axis=-1)))
                target_delta = (
                    float(logp[target_token_id] - base_logp[target_token_id])
                    if target_token_id is not None and target_token_id < logp.shape[-1]
                    else None)
                target_residual_base = baseline_residual[0, target_position]
                target_residual_new = residual[0, target_position]
                unrelated_pos = 0 if target_position != 0 else min(1, length - 1)
                layer = int(candidate["layer"])
                if pool == "rst":
                    local_base = baseline_rst_updates[layer, 0]
                    local_new = rst_updates[layer, 0]
                    local_kind = "rst_residual_update"
                else:
                    local_base = baseline_attention_updates[layer, 0]
                    local_new = attention_updates[layer, 0]
                    local_kind = "attention_residual_update"
                row = {
                    "prompt_id": prompt["prompt_id"],
                    "phenomenon": prompt["phenomenon"],
                    "pool": pool,
                    **candidate,
                    "intervention_type": "selection_gate_ablation_canonical_denominator",
                    "canonical_unpruned_admission_denominator": True,
                    "removed_operator_count": 1,
                    "baseline_ce": baseline_ce,
                    "validation_ce_delta": (
                        None if baseline_ce is None else
                        float(_sequence_ce(logits, prompt["token_ids"], length) - baseline_ce)),
                    "target_continuation_token_id": target_token_id,
                    "target_logprob_delta": target_delta,
                    "next_token_kl": next_token_kl,
                    "full_output_kl": full_output_kl,
                    "top_prediction_changed": bool(
                        int(np.argmax(base_logp)) != int(np.argmax(logp))),
                    "final_residual_cosine": _cosine(
                        target_residual_base, target_residual_new),
                    "final_residual_relative_error": float(
                        np.linalg.norm(target_residual_new - target_residual_base)
                        / max(float(np.linalg.norm(target_residual_base)), 1.0e-12)),
                    "layer_local_update_kind": local_kind,
                    "layer_local_update_cosine": _cosine(local_base, local_new),
                    "layer_local_update_relative_error": float(
                        np.linalg.norm(local_new - local_base)
                        / max(float(np.linalg.norm(local_base)), 1.0e-12)),
                    "unrelated_control_residual_relative_error": float(
                        np.linalg.norm(
                            residual[0, unrelated_pos] - baseline_residual[0, unrelated_pos])
                        / max(float(np.linalg.norm(baseline_residual[0, unrelated_pos])), 1.0e-12)),
                }
                result_rows.append(row)
        if ctx.is_primary:
            print(
                "CAUSAL_INTERVENTION "
                f"prompt={prompt_idx + 1:02d}/{len(primary):02d} "
                f"id={prompt['prompt_id']}",
                flush=True,
            )
    selected = [
        abs(float(row["target_logprob_delta"]))
        for row in result_rows
        if row["strategy"] in ("top_contribution", "top_gate")
        and row.get("target_logprob_delta") is not None
    ]
    controls = [
        abs(float(row["target_logprob_delta"]))
        for row in result_rows
        if row["strategy"] in ("inactive_random", "active_random", "matched_control")
        and row.get("target_logprob_delta") is not None
    ]
    summary = {
        "status": "ready" if result_rows else "insufficient_evidence",
        "intervention_type": "selection_gate_ablation_canonical_denominator",
        "canonical_unpruned_admission_denominator": True,
        "num_prompts": len(primary),
        "num_interventions": len(result_rows),
        "selected_abs_target_logprob_delta": _bootstrap_mean_ci(
            selected, int(ctx.config.get("seed", 0))),
        "control_abs_target_logprob_delta": _bootstrap_mean_ci(
            controls, int(ctx.config.get("seed", 0)) + 1),
        "selected_minus_control_effect": (
            float(np.mean(selected) - np.mean(controls)) if selected and controls else None),
        "sec": time.time() - started,
        "artifact": ctx.store.path("interventions.jsonl"),
        "limitations": [
            "generation change is not executed in the core intervention item",
            "address-RW permutation is not part of the core preset",
        ],
    }
    if ctx.is_primary:
        write_jsonl_atomic(ctx.store.path("interventions.jsonl"), result_rows)
        write_json_atomic(ctx.store.path("causal_intervention_summary.json"), summary)
    return summary


def run_v4171_transition_items(
    ctx: AnalysisContext,
    items: Sequence[str],
) -> Dict[str, Any]:
    selected = [item for item in CORE_TRANSITION_ITEMS if item in set(items)]
    if not selected:
        return {}
    if str(ctx.model_cfg.get("model_version")) != V4171_MODEL_VERSION:
        return {
            item: {
                "status": "unsupported_for_v4171",
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
        context, decoupling, _ = build_pair_analyses(ctx, records)
        if "context_divergence" in selected:
            result["context_divergence"] = context
        if "state_transition_decoupling" in selected:
            result["state_transition_decoupling"] = decoupling
    if "causal_intervention" in selected:
        result["causal_intervention"] = run_causal_intervention(ctx, records)
    if ctx.is_primary:
        metrics_rows = [
            {"item": item, **value}
            for item, value in result.items()
            if item != "transition_trace_cache" and isinstance(value, dict)
        ]
        write_jsonl_atomic(ctx.store.path("metrics.jsonl"), metrics_rows)
    return result
