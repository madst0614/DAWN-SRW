"""Streaming operator usage atlas and operator-card generation."""

from __future__ import annotations

import heapq
import re
import random
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import (
    AnalysisContext,
    format_duration,
    host_aligned_batch_size,
    load_eval_data,
    maybe_load_tokenizer,
    sync_hosts,
    token_window_text,
)
from analysis.dawn_analysis_storage import (
    list_paths,
    read_json,
    should_skip_job,
    write_json_atomic,
    write_jsonl_atomic,
    write_npz_atomic,
)
from analysis.dawn_analysis_trace import topk_trace_forward


ATLAS_POOLS = ("qk", "v", "rst")
PART_RE = re.compile(r"part-host(?P<host>\d+)-(?P<part>\d+)\.json$")


def _pool_size(ctx: AnalysisContext, pool: str) -> int:
    if pool == "qk":
        return int(ctx.model_cfg["n_qk"])
    if pool == "v":
        return int(ctx.model_cfg["n_v"])
    return int(ctx.model_cfg["n_rst"])


def _merge_route_arrays(trace: Dict[str, np.ndarray], pool: str):
    if pool == "qk":
        ids = np.concatenate([trace["q_top_idx"], trace["k_top_idx"]], axis=2)
        vals = np.concatenate([trace["q_top_val"], trace["k_top_val"]], axis=2)
        route = np.concatenate([
            np.full(trace["q_top_idx"].shape[:-1], "q", dtype=object),
            np.full(trace["k_top_idx"].shape[:-1], "k", dtype=object),
        ], axis=2)
        return ids, vals, route
    key = "v" if pool == "v" else "rst"
    ids = trace[f"{key}_top_idx"]
    vals = trace[f"{key}_top_val"]
    route = np.full(ids.shape[:-1], key, dtype=object)
    return ids, vals, route


def _sparse_pairs(ids: np.ndarray, vals: np.ndarray) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
    flat_ids = ids.reshape(-1).astype(np.int64)
    flat_vals = vals.reshape(-1).astype(np.float64)
    mask = flat_vals > 0
    flat_ids = flat_ids[mask]
    flat_vals = flat_vals[mask]
    if flat_ids.size == 0:
        return [], [], []
    unique, inv = np.unique(flat_ids, return_inverse=True)
    counts = np.bincount(inv).astype(np.int64)
    mass_sum = np.bincount(inv, weights=flat_vals).astype(np.float64)
    mass_max = np.zeros(unique.shape[0], dtype=np.float64)
    np.maximum.at(mass_max, inv, flat_vals)
    return (
        [[int(op), int(cnt)] for op, cnt in zip(unique, counts)],
        [[int(op), float(val)] for op, val in zip(unique, mass_sum)],
        [[int(op), float(val)] for op, val in zip(unique, mass_max)],
    )


def _top_contexts_for_pool(pool: str, ids: np.ndarray, vals: np.ndarray,
                           route: np.ndarray, batch_tokens: np.ndarray,
                           tokenizer, top_contexts_per_op: int) -> Dict[str, List[Dict[str, Any]]]:
    # ids/vals: [layers, batch, positions_or_routes, topk]
    contexts: Dict[int, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    layers, batch, positions, topk = ids.shape
    flat_order = np.argwhere(vals > 0)
    if flat_order.shape[0] > 200_000:
        weights = vals[vals > 0].reshape(-1)
        kth = max(0, weights.size - 200_000)
        cutoff = np.partition(weights, kth)[kth]
        flat_order = np.argwhere(vals >= cutoff)
    for layer, row, pos, k_idx in flat_order:
        weight = float(vals[layer, row, pos, k_idx])
        op = int(ids[layer, row, pos, k_idx])
        token_pos = int(pos)
        if pool == "qk":
            token_pos = int(pos % batch_tokens.shape[1])
        route_name = str(route[layer, row, pos])
        token_ids = [int(x) for x in batch_tokens[row].tolist()]
        next_pos = min(token_pos + 1, len(token_ids) - 1)
        rec = {
            "layer": int(layer),
            "position": int(token_pos),
            "route": route_name,
            "weight": weight,
            "token_id": int(token_ids[token_pos]),
            "next_token_id": int(token_ids[next_pos]),
            "text_window": token_window_text(tokenizer, token_ids, token_pos),
        }
        heap = contexts[op]
        item = (weight, rec)
        if len(heap) < top_contexts_per_op:
            heapq.heappush(heap, item)
        elif weight > heap[0][0]:
            heapq.heapreplace(heap, item)
    out = {}
    for op, heap in contexts.items():
        out[str(op)] = [rec for _, rec in sorted(heap, key=lambda x: -x[0])]
    return out


def _part_from_trace(trace: Dict[str, np.ndarray], batch_tokens: np.ndarray,
                     tokenizer, top_contexts_per_op: int) -> Dict[str, Any]:
    part = {
        "counts": {},
        "mass_sum": {},
        "mass_max": {},
        "top_contexts": {},
        "tokens_observed": int(batch_tokens.size),
        "sequences_observed": int(batch_tokens.shape[0]),
    }
    for pool in ATLAS_POOLS:
        ids, vals, route = _merge_route_arrays(trace, pool)
        counts, mass_sum, mass_max = _sparse_pairs(ids, vals)
        part["counts"][pool] = counts
        part["mass_sum"][pool] = mass_sum
        part["mass_max"][pool] = mass_max
        part["top_contexts"][pool] = _top_contexts_for_pool(
            pool,
            ids,
            vals,
            route,
            batch_tokens,
            tokenizer,
            top_contexts_per_op,
        )
    return part


def _apply_pairs(arr: np.ndarray, pairs: List[List[Any]], op=np.add) -> None:
    for op_id, value in pairs:
        idx = int(op_id)
        if 0 <= idx < arr.shape[0]:
            if op is np.maximum:
                arr[idx] = max(arr[idx], value)
            else:
                arr[idx] += value


def _merge_contexts(dst: Dict[str, List[Dict[str, Any]]], src: Dict[str, List[Dict[str, Any]]],
                    limit: int) -> None:
    for op, rows in src.items():
        cur = dst.setdefault(str(op), [])
        cur.extend(rows)
        cur.sort(key=lambda r: -float(r.get("weight", 0.0)))
        del cur[limit:]


def _reduce_parts(ctx: AnalysisContext, part_paths: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    arrays = {}
    contexts: Dict[str, Dict[str, List[Dict[str, Any]]]] = {p: {} for p in ATLAS_POOLS}
    total_tokens = 0
    total_sequences = 0
    for pool in ATLAS_POOLS:
        n = _pool_size(ctx, pool)
        arrays[f"{pool}_usage_count"] = np.zeros((n,), dtype=np.int64)
        arrays[f"{pool}_mass_sum"] = np.zeros((n,), dtype=np.float64)
        arrays[f"{pool}_mass_max"] = np.zeros((n,), dtype=np.float64)

    for path in part_paths:
        part = read_json(path)
        total_tokens += int(part.get("tokens_observed", 0))
        total_sequences += int(part.get("sequences_observed", 0))
        for pool in ATLAS_POOLS:
            _apply_pairs(arrays[f"{pool}_usage_count"], part.get("counts", {}).get(pool, []))
            _apply_pairs(arrays[f"{pool}_mass_sum"], part.get("mass_sum", {}).get(pool, []))
            _apply_pairs(arrays[f"{pool}_mass_max"], part.get("mass_max", {}).get(pool, []), op=np.maximum)
            _merge_contexts(
                contexts[pool],
                part.get("top_contexts", {}).get(pool, {}),
                int(ctx.args.usage_top_contexts_per_op),
            )

    summary = {
        "tokens_observed": total_tokens,
        "sequences_observed": total_sequences,
        "pools": {},
    }
    for pool in ATLAS_POOLS:
        count = arrays[f"{pool}_usage_count"]
        mass = arrays[f"{pool}_mass_sum"]
        top_mass = np.argsort(-mass)[:100]
        top_freq = np.argsort(-count)[:100]
        active_ops = int(np.sum(count > 0))
        summary["pools"][pool] = {
            "N": int(count.shape[0]),
            "active_ops_observed": active_ops,
            "active_op_frac_observed": active_ops / max(1, int(count.shape[0])),
            "total_topk_mass": float(np.sum(mass)),
            "top_by_mass": [
                {"operator_id": int(i), "mass_sum": float(mass[i]), "usage_count": int(count[i])}
                for i in top_mass if mass[i] > 0
            ],
            "top_by_frequency": [
                {"operator_id": int(i), "usage_count": int(count[i]), "mass_sum": float(mass[i])}
                for i in top_freq if count[i] > 0
            ],
        }
    return arrays, summary, contexts


def _part_identity(path: str) -> Tuple[int, int] | None:
    name = path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    match = PART_RE.match(name)
    if not match:
        return None
    return int(match.group("host")), int(match.group("part"))


def _current_run_part_paths(ctx: AnalysisContext, max_parts: int) -> List[str]:
    paths = list_paths(ctx.store.path("usage", "usage_parts"), "part-host*-*.json")
    selected = []
    for path in paths:
        identity = _part_identity(path)
        if identity is None:
            continue
        host_id, part_idx = identity
        if 0 <= host_id < int(ctx.n_hosts) and 0 <= part_idx < int(max_parts):
            selected.append(path)
    return sorted(selected)


def _usage_metadata_matches(meta: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        try:
            if int(meta.get(key)) != int(expected_value):
                return False
        except Exception:
            return False
    return True


def _usage_part_matches(path: str, *, expected_params: Dict[str, Any],
                        host_id: int, part_idx: int) -> bool:
    if not should_skip_job(path, ["counts", "mass_sum", "analysis_params"]):
        return False
    try:
        part = read_json(path)
    except Exception:
        return False
    if not _usage_metadata_matches(part.get("analysis_params", {}), expected_params):
        return False
    try:
        return int(part.get("host_id")) == int(host_id) and int(part.get("batch_idx")) == int(part_idx)
    except Exception:
        return False


def _write_top_context_files(ctx: AnalysisContext, contexts: Dict[str, Dict[str, List[Dict[str, Any]]]],
                             summary: Dict[str, Any]) -> None:
    for pool in ATLAS_POOLS:
        selected = {
            int(row["operator_id"])
            for row in summary["pools"][pool].get("top_by_mass", [])[:50]
        }
        selected |= {
            int(row["operator_id"])
            for row in summary["pools"][pool].get("top_by_frequency", [])[:50]
        }
        for op in sorted(selected):
            rows = contexts.get(pool, {}).get(str(op), [])
            if not rows:
                continue
            write_jsonl_atomic(
                ctx.store.path("usage", "top_contexts", f"{pool}_operator_{op:06d}.jsonl"),
                rows,
            )


def _operator_cards(ctx: AnalysisContext, arrays: Dict[str, np.ndarray],
                    contexts: Dict[str, Dict[str, List[Dict[str, Any]]]],
                    summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards = []
    rng = random.Random(0)
    for pool in ATLAS_POOLS:
        count = arrays[f"{pool}_usage_count"]
        mass = arrays[f"{pool}_mass_sum"]
        selected = []
        selected.extend(int(r["operator_id"]) for r in summary["pools"][pool].get("top_by_mass", [])[:20])
        selected.extend(int(r["operator_id"]) for r in summary["pools"][pool].get("top_by_frequency", [])[:20])
        used = np.flatnonzero(count > 0)
        low = np.flatnonzero(count == 0)
        if low.size:
            selected.extend(int(x) for x in low[:10])
        if used.size:
            selected.extend(int(x) for x in rng.sample(list(map(int, used)), min(10, used.size)))
        seen = set()
        for op in selected:
            if op in seen:
                continue
            seen.add(op)
            c = int(count[op]) if 0 <= op < count.shape[0] else 0
            m = float(mass[op]) if 0 <= op < mass.shape[0] else 0.0
            cards.append({
                "pool": pool,
                "operator_id": int(op),
                "usage_count": c,
                "total_mass": m,
                "activation_frequency": c / max(1, int(summary.get("tokens_observed", 0))),
                "top_contexts": contexts.get(pool, {}).get(str(op), []),
                "read_write_geometry": None,
                "ablation_effect": None,
                "human_label_suggestion": None,
            })
    return cards


def _write_cards(ctx: AnalysisContext, cards: List[Dict[str, Any]]) -> None:
    write_jsonl_atomic(ctx.store.path("usage", "operator_cards.jsonl"), cards)
    lines = ["# Operator Card Examples", ""]
    for card in cards[:40]:
        lines.append(
            f"## {card['pool']} operator {card['operator_id']} "
            f"(mass={card['total_mass']:.4g}, count={card['usage_count']})"
        )
        for row in card.get("top_contexts", [])[:5]:
            lines.append(
                f"- L{row.get('layer')} pos {row.get('position')} "
                f"w={float(row.get('weight', 0.0)):.4g}: {row.get('text_window')}"
            )
        lines.append("")
    from analysis.dawn_analysis_storage import write_text_atomic

    write_text_atomic(ctx.store.path("usage", "operator_cards_top.md"), "\n".join(lines))


def run_usage_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "usage"
    summary_path = store.path("usage", "operator_usage_summary.json")
    npz_path = store.path("usage", "operator_usage_by_pool.npz")
    seq_len = int(args.usage_seq_len)
    requested_batch_size = int(args.usage_batch_size)
    batch_size = host_aligned_batch_size(requested_batch_size, ctx.n_hosts)
    topk = int(args.usage_topk)
    max_sequences = int(args.usage_max_sequences)
    top_contexts_per_op = int(args.usage_top_contexts_per_op)
    expected_params = {
        "seq_len": seq_len,
        "topk": topk,
        "global_batch_size": batch_size,
        "usage_max_sequences": max_sequences,
        "top_contexts_per_op": top_contexts_per_op,
    }
    if ctx.checkpoint_step is not None:
        expected_params["checkpoint_step"] = int(ctx.checkpoint_step)

    if args.resume and should_skip_job(summary_path, ["pools", "analysis_params"]) and should_skip_job(npz_path):
        summary = read_json(summary_path)
        if _usage_metadata_matches(summary.get("analysis_params", {}), expected_params):
            store.log_event(
                stage,
                "skip",
                message=(
                    "USAGE SKIP complete "
                    f"tokens={int(summary.get('tokens_observed', 0)):,} "
                    + " ".join(
                        f"{p}_active={summary.get('pools', {}).get(p, {}).get('active_ops_observed', 0):,}"
                        for p in ATLAS_POOLS
                    )
                ),
                **summary,
            )
            return summary

    store.set_stage_status(stage, "running")
    max_tokens = max_sequences * seq_len
    tokenizer = maybe_load_tokenizer(local_only=True)
    loader = load_eval_data(ctx.config, seq_len, batch_size, ctx.host_id, ctx.n_hosts, max_tokens=max_tokens)
    max_parts = len(loader)
    if args.max_jobs_per_stage is not None:
        max_parts = min(max_parts, int(args.max_jobs_per_stage))
    trace_fn = jax.jit(lambda p, x: topk_trace_forward(p, ctx.model_cfg, x, topk=topk))

    store.log_event(
        stage,
        "start",
        message=(
            f"USAGE START parts={max_parts} seq_len={seq_len} "
            f"batch_size={batch_size} requested_batch_size={requested_batch_size} "
            f"topk={topk} host={ctx.host_id}/{ctx.n_hosts}"
        ),
        parts=max_parts,
        seq_len=seq_len,
        batch_size=batch_size,
        requested_batch_size=requested_batch_size,
        topk=topk,
    )

    part_paths = []
    stage_t0 = time.time()
    for part_idx, (input_ids, _attention_mask) in enumerate(loader):
        if part_idx >= max_parts:
            break
        part_path = store.path("usage", "usage_parts", f"part-host{ctx.host_id:03d}-{part_idx:06d}.json")
        part_paths.append(part_path)
        if args.resume and _usage_part_matches(
            part_path,
            expected_params=expected_params,
            host_id=ctx.host_id,
            part_idx=part_idx,
        ):
            done = part_idx + 1
            elapsed = time.time() - stage_t0
            eta = (elapsed / done) * max(0, max_parts - done) if done else None
            store.log_event(
                stage,
                "part_skip",
                message=(
                    f"USAGE part {done:05d}/{max_parts:05d} SKIP "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                    f"host={ctx.host_id}/{ctx.n_hosts}"
                ),
            )
            continue
        job_id = f"part-host{ctx.host_id:03d}-{part_idx:06d}"
        store.mark_job_started(stage, job_id)
        part_t0 = time.time()
        batch_tokens = np.asarray(input_ids, dtype=np.int32)
        trace_np = {
            k: np.asarray(v)
            for k, v in jax.device_get(trace_fn(ctx.params, jnp.asarray(batch_tokens))).items()
        }
        part = _part_from_trace(
            trace_np,
            batch_tokens,
            tokenizer,
            top_contexts_per_op,
        )
        part["batch_idx"] = int(part_idx)
        part["host_id"] = int(ctx.host_id)
        part["analysis_params"] = {
            **expected_params,
            "requested_batch_size": requested_batch_size,
        }
        write_json_atomic(part_path, part)
        part_sec = time.time() - part_t0
        done = part_idx + 1
        elapsed = time.time() - stage_t0
        eta = (elapsed / done) * max(0, max_parts - done) if done else None
        tok_per_sec = part["tokens_observed"] / part_sec if part_sec > 0 else 0.0
        if ctx.is_primary:
            store.mark_job_complete(stage, job_id, part_path, {
                "tokens": part["tokens_observed"],
                "qk_unique": len(part["counts"]["qk"]),
                "v_unique": len(part["counts"]["v"]),
                "rst_unique": len(part["counts"]["rst"]),
            })
        store.log_event(
            stage,
            "part",
            message=(
                f"USAGE part {part_idx + 1:05d}/{max_parts:05d} "
                f"tokens={part['tokens_observed']:,} "
                f"part_sec={part_sec:.1f} tok/s={tok_per_sec:.0f} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                f"host={ctx.host_id}/{ctx.n_hosts} "
                f"unique(qk/v/rst)="
                f"{len(part['counts']['qk'])}/{len(part['counts']['v'])}/{len(part['counts']['rst'])}"
            ),
            tokens=part["tokens_observed"],
            part_sec=part_sec,
            tok_per_sec=tok_per_sec,
            elapsed_sec=elapsed,
            eta_sec=eta,
            qk_unique=len(part["counts"]["qk"]),
            v_unique=len(part["counts"]["v"]),
            rst_unique=len(part["counts"]["rst"]),
        )

    sync_hosts("dawn-analysis-usage-parts-done")

    if not ctx.is_primary:
        return {}

    reduce_t0 = time.time()
    all_part_paths = []
    for path in _current_run_part_paths(ctx, max_parts):
        identity = _part_identity(path)
        if identity is None:
            continue
        host_id, part_idx = identity
        if _usage_part_matches(
            path,
            expected_params=expected_params,
            host_id=host_id,
            part_idx=part_idx,
        ):
            all_part_paths.append(path)
    expected_parts = int(ctx.n_hosts) * int(max_parts)
    if len(all_part_paths) != expected_parts:
        raise RuntimeError(
            f"USAGE reduce expected {expected_parts} current-run parts "
            f"but found {len(all_part_paths)} matching parts."
        )
    store.log_event(
        stage,
        "reduce_start",
        message=f"USAGE REDUCE START parts={len(all_part_paths)}",
        parts=len(all_part_paths),
    )
    arrays, summary, contexts = _reduce_parts(ctx, all_part_paths)
    summary["analysis_params"] = {
        **expected_params,
        "requested_batch_size": requested_batch_size,
    }
    write_npz_atomic(npz_path, **arrays)
    write_json_atomic(summary_path, summary)
    _write_top_context_files(ctx, contexts, summary)
    cards = _operator_cards(ctx, arrays, contexts, summary)
    _write_cards(ctx, cards)
    store.mark_job_complete(stage, "reduce", summary_path, summary)
    store.set_stage_status(stage, "complete")
    store.log_event(
        stage,
        "summary",
        message=(
            "USAGE SUMMARY "
            f"tokens={summary['tokens_observed']:,} "
            f"reduce_sec={time.time() - reduce_t0:.1f} "
            + " ".join(
                f"{p}_active={summary['pools'][p]['active_ops_observed']:,}"
                for p in ATLAS_POOLS
            )
        ),
        **summary,
    )
    return summary
