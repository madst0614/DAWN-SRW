"""Small-budget causal operator suppression experiments."""

from __future__ import annotations

import hashlib
import json
import traceback
from typing import Any, Dict, Iterable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import (
    AnalysisContext,
    host_aligned_batch_size,
    load_eval_data,
)
from analysis.dawn_analysis_storage import (
    list_paths,
    read_json,
    read_npz,
    should_skip_job,
    write_csv_atomic,
    write_json_atomic,
)
from models import dawn_srw_v4166 as v4166


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> List[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_pools(value: str | None) -> List[str]:
    if not value:
        return ["qk", "v", "rst"]
    return [x.strip() for x in value.split(",") if x.strip()]


def _pool_size(ctx: AnalysisContext, pool: str) -> int:
    if pool == "qk":
        return int(ctx.model_cfg["n_qk"])
    if pool == "v":
        return int(ctx.model_cfg["n_v"])
    return int(ctx.model_cfg["n_rst"])


def _loss_from_logits(logits, input_ids):
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    preds = jnp.argmax(logits, axis=-1)
    valid = jnp.ones_like(labels, dtype=jnp.bool_)
    return (
        token_loss.sum(),
        ((preds == labels) & valid).astype(jnp.int32).sum(),
        valid.astype(jnp.int32).sum(),
    )


def _eval_forward(forward_fn, batches: List[np.ndarray]) -> Dict[str, Any]:
    loss_sum = jnp.float32(0.0)
    correct = jnp.int32(0)
    valid = jnp.int32(0)
    for batch in batches:
        ids = jnp.asarray(batch, dtype=jnp.int32)
        lsum, corr, val = _loss_from_logits(forward_fn(ids), ids)
        loss_sum = loss_sum + lsum
        correct = correct + corr
        valid = valid + val
    loss_sum_h, correct_h, valid_h = jax.device_get((loss_sum, correct, valid))
    loss = float(loss_sum_h) / max(1, int(valid_h))
    return {
        "loss_sum": float(loss_sum_h),
        "loss": loss,
        "accuracy": int(correct_h) / max(1, int(valid_h)),
        "correct": int(correct_h),
        "valid_count": int(valid_h),
    }


def _operator_lists(ctx: AnalysisContext) -> Dict[str, Dict[str, List[int]]]:
    usage_npz = ctx.store.path("usage", "operator_usage_by_pool.npz")
    out = {}
    if should_skip_job(usage_npz):
        arrays = read_npz(usage_npz)
    else:
        arrays = {}
    rng = np.random.default_rng(0)
    for pool in _parse_pools(ctx.args.ablation_pools):
        n = _pool_size(ctx, pool)
        mass = np.asarray(arrays.get(f"{pool}_mass_sum", np.zeros((n,), dtype=np.float64)))
        count = np.asarray(arrays.get(f"{pool}_usage_count", np.zeros((n,), dtype=np.int64)))
        top = np.argsort(-mass).astype(np.int64).tolist()
        low = np.argsort(count + (mass > 0).astype(np.int64) * 1_000_000_000).astype(np.int64).tolist()
        random_ids = rng.permutation(n).astype(np.int64).tolist()
        out[pool] = {
            "top": top,
            "low": low,
            "random": random_ids,
        }
    return out


def _job_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _make_mask(ctx: AnalysisContext, pool: str, ids: Sequence[int]) -> Dict[str, Any]:
    n = _pool_size(ctx, pool)
    arr = np.zeros((n,), dtype=np.bool_)
    ids = [int(i) for i in ids if 0 <= int(i) < n]
    if ids:
        arr[np.asarray(ids, dtype=np.int64)] = True
    key = "rst" if pool == "rst" else pool
    return {key: jnp.asarray(arr)}


def _load_batches(ctx: AnalysisContext) -> List[np.ndarray]:
    args = ctx.args
    seq_len = int(args.ablation_seq_len)
    batch_size = host_aligned_batch_size(int(args.ablation_batch_size), ctx.n_hosts)
    max_sequences = int(args.ablation_max_sequences)
    max_tokens = seq_len * max_sequences
    loader = load_eval_data(ctx.config, seq_len, batch_size, ctx.host_id, ctx.n_hosts, max_tokens)
    max_batches = len(loader)
    if args.max_jobs_per_stage is not None:
        max_batches = min(max_batches, int(args.max_jobs_per_stage))
    batches = []
    for i, (input_ids, _) in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(np.asarray(input_ids, dtype=np.int32))
    return batches


def run_ablation_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "ablation"
    store.set_stage_status(stage, "running")
    k_list = _parse_csv_ints(args.ablation_k_list, [1, 4, 16, 64])
    pools = _parse_pools(args.ablation_pools)
    operator_lists = _operator_lists(ctx)
    batches = _load_batches(ctx)
    requested_batch_size = int(args.ablation_batch_size)
    batch_size = host_aligned_batch_size(requested_batch_size, ctx.n_hosts)

    store.log_event(
        stage,
        "start",
        message=(
            f"ABLATION START jobs={len(pools) * len(k_list) * 3} "
            f"batches={len(batches)} batch_size={batch_size} "
            f"requested_batch_size={requested_batch_size} "
            f"k={k_list} pools={','.join(pools)} host={ctx.host_id}/{ctx.n_hosts}"
        ),
        pools=pools,
        k_list=k_list,
        batches=len(batches),
        batch_size=batch_size,
        requested_batch_size=requested_batch_size,
    )

    base_forward = jax.jit(v4166.build_suppressed_forward(ctx.params, ctx.model_cfg, {}))
    base = _eval_forward(base_forward, batches)
    if ctx.is_primary:
        store.log_event(
            stage,
            "base",
            message=f"ABLATION BASE loss={base['loss']:.6f} acc={base['accuracy']:.4f} tokens={base['valid_count']:,}",
            **base,
        )

    records = []
    for pool in pools:
        for strategy in ("top", "random", "low"):
            candidates = operator_lists[pool][strategy]
            for k in k_list:
                op_ids = [int(x) for x in candidates[: min(int(k), len(candidates))]]
                payload = {
                    "checkpoint_step": ctx.checkpoint_step,
                    "pool": pool,
                    "strategy": strategy,
                    "k": int(k),
                    "operator_ids": op_ids,
                }
                jid = _job_id(payload)
                path = store.path("ablation", "jobs", f"job-{jid}.json")
                if args.resume and should_skip_job(path, ["delta_loss"]):
                    rec = read_json(path)
                    records.append(rec)
                    store.log_event(
                        stage,
                        "job_skip",
                        message=(
                            f"ABLATION {pool}/{strategy}/k={k} SKIP "
                            f"delta_loss={float(rec.get('delta_loss', 0.0)):.6f}"
                        ),
                        **rec,
                    )
                    continue
                store.mark_job_started(stage, jid)
                try:
                    masks = _make_mask(ctx, pool, op_ids)
                    forward = jax.jit(v4166.build_suppressed_forward(ctx.params, ctx.model_cfg, masks))
                    ablated = _eval_forward(forward, batches)
                    rec = {
                        **payload,
                        "job_id": jid,
                        "base_loss": base["loss"],
                        "ablated_loss": ablated["loss"],
                        "delta_loss": ablated["loss"] - base["loss"],
                        "base_acc": base["accuracy"],
                        "ablated_acc": ablated["accuracy"],
                        "delta_acc": ablated["accuracy"] - base["accuracy"],
                        "valid_tokens": ablated["valid_count"],
                    }
                    if ctx.is_primary:
                        write_json_atomic(path, rec)
                        store.mark_job_complete(stage, jid, path, rec)
                        store.log_event(
                            stage,
                            "job",
                            message=(
                                f"ABLATION {pool}/{strategy}/k={k} "
                                f"delta_loss={rec['delta_loss']:.6f} "
                                f"delta_acc={rec['delta_acc']:.4f} "
                                f"ablated_loss={rec['ablated_loss']:.6f}"
                            ),
                            **rec,
                        )
                    records.append(rec)
                except Exception as exc:
                    err = traceback.format_exc()
                    store.mark_job_failed(stage, jid, str(exc), err)
                    store.log_event(
                        stage,
                        "job_failed",
                        message=f"ABLATION {pool}/{strategy}/k={k} FAILED {type(exc).__name__}: {exc}",
                        error=str(exc),
                        job_id=jid,
                    )
                    if args.fail_fast:
                        raise

    if ctx.is_primary:
        if args.resume:
            job_paths = list_paths(store.path("ablation", "jobs"), "job-*.json")
            all_records = [read_json(p) for p in job_paths if should_skip_job(p, ["delta_loss"])]
        else:
            all_records = list(records)
        csv_rows = [
            {
                "pool": r["pool"],
                "strategy": r["strategy"],
                "k": r["k"],
                "base_loss": r["base_loss"],
                "ablated_loss": r["ablated_loss"],
                "delta_loss": r["delta_loss"],
                "base_acc": r["base_acc"],
                "ablated_acc": r["ablated_acc"],
                "delta_acc": r["delta_acc"],
                "valid_tokens": r["valid_tokens"],
            }
            for r in all_records
        ]
        summary = {
            "base": base,
            "jobs": all_records,
            "num_jobs": len(all_records),
        }
        write_json_atomic(store.path("ablation", "summary.json"), summary)
        write_csv_atomic(store.path("ablation", "ablation_curve.csv"), csv_rows)
        store.mark_job_complete(stage, "summary", store.path("ablation", "summary.json"), summary)
        store.set_stage_status(stage, "complete")
        top_records = sorted(all_records, key=lambda r: -float(r.get("delta_loss", 0.0)))[:5]
        store.log_event(
            stage,
            "summary",
            message=(
                f"ABLATION SUMMARY jobs={len(all_records)} "
                + " ".join(
                    f"{r['pool']}/{r['strategy']}/k={r['k']} dL={float(r['delta_loss']):.4f}"
                    for r in top_records
                )
            ),
            **summary,
        )
        return summary
    return {}
