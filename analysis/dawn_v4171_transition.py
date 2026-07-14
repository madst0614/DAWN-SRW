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
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import (
    AnalysisContext,
    maybe_load_tokenizer,
)
from analysis.dawn_analysis_storage import (
    open_path,
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
PAIR_CAPTURE_THRESHOLD = 0.95


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
    prompts = [_tokenize_transition_row(tokenizer, row, seq_len) for row in rows]
    _validate_tokenized_pairs(prompts)
    candidate_seed = int(ctx.config.get("seed", 0))
    data_replicas = max(1, int(ctx.mesh.shape["data"]))

    def trace_step(p, x, t):
        trace = topk_trace_forward(
            p,
            ctx.model_cfg,
            x,
            topk_qk=topk_qk,
            topk_v=topk_v,
            topk_rst=topk_rst,
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

    trace_fn = jax.jit(trace_step)
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
    captured_by_pool: Dict[str, List[float]] = {pool: [] for pool in TRACE_POOLS}
    for record in internal:
        for pool in TRACE_POOLS:
            captured_by_pool[pool].extend(
                np.asarray(record["trace"][f"{pool}_captured_mass"])[:, 0].tolist())
    captured = [value for values in captured_by_pool.values() for value in values]
    captured_summary = {
        pool: {
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "p10": float(np.quantile(values, 0.10)) if values else None,
        }
        for pool, values in captured_by_pool.items()
    }
    summary = {
        "status": "ready",
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
        "captured_mass_valid_threshold": 0.95,
        "sec": time.time() - started,
        "artifacts": {
            "trajectory_traces": ctx.store.path("trajectory_traces.jsonl"),
            "transition_trace_cache": ctx.store.path("transition_trace_cache.npz"),
        },
    }
    if summary["captured_mass"]["min"] is not None and summary["captured_mass"]["min"] < 0.95:
        summary["captured_mass_warning"] = (
            "Rows with either side below 95% captured mass are excluded from pair metrics")
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
            capture_valid = (
                captured_a >= PAIR_CAPTURE_THRESHOLD
                and captured_b >= PAIR_CAPTURE_THRESHOLD)
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
    rows, _ = load_transition_prompt_rows(prompt_set)
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
    _normal_production_cross_graph_audit(ctx, records, canonical_forward)
    parity = _intervention_forward_parity(
        ctx, records, canonical_forward)
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
        baseline_logits, _, baseline_residual = jax.device_get(
            canonical_forward(
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
                logits, _, residual = jax.device_get(canonical_forward(
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
        "status": "ready" if valid_rows else "insufficient_evidence",
        "intervention_type": "production_core_execution_suppression",
        "canonical_unpruned_admission_denominator": True,
        "zero_suppression_parity": parity,
        "normal_production_cross_graph_audit": cross_graph_audit,
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
