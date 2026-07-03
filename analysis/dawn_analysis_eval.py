"""Final validation evaluation stage."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp

from analysis.dawn_analysis_common import (
    AnalysisContext,
    create_ce_eval_step,
    load_eval_data,
    shard_batch_to_mesh,
)
from analysis.dawn_analysis_storage import (
    read_jsonl,
    should_skip_job,
    write_json_atomic,
    write_jsonl_atomic,
)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    loss_sum = sum(float(r.get("loss_sum", 0.0)) for r in rows)
    correct = sum(int(r.get("correct", 0)) for r in rows)
    valid = sum(int(r.get("valid_count", 0)) for r in rows)
    sec = sum(float(r.get("sec", 0.0)) for r in rows)
    loss = loss_sum / valid if valid else float("nan")
    return {
        "val_loss": loss,
        "ppl": math.exp(min(loss, 80.0)) if valid else float("nan"),
        "accuracy": correct / valid if valid else float("nan"),
        "valid_tokens": int(valid),
        "correct": int(correct),
        "total_sequences": int(sum(int(r.get("sequences", 0)) for r in rows)),
        "sec": sec,
        "tok_per_sec": valid / sec if sec > 0 else 0.0,
        "num_batches": len(rows),
    }


def run_eval_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "eval"
    final_path = store.path("eval", "final_eval.json")
    by_batch_path = store.path("eval", "final_eval_by_batch.jsonl")
    if args.resume and should_skip_job(final_path, ["val_loss", "valid_tokens"]):
        summary = store.load_manifest().get("stages", {}).get(stage, {}).get("summaries", {}).get("final")
        if not summary:
            summary = _aggregate(read_jsonl(by_batch_path))
        store.log_event(
            stage,
            "skip",
            message=(
                "EVAL SKIP complete "
                f"loss={float(summary.get('val_loss', 0.0)):.6f} "
                f"ppl={float(summary.get('ppl', 0.0)):.3f} "
                f"acc={float(summary.get('accuracy', 0.0)):.4f} "
                f"tokens={int(summary.get('valid_tokens', 0)):,}"
            ),
            **summary,
        )
        return summary

    store.set_stage_status(stage, "running")
    batch_size = int(args.eval_batch_size or ctx.config["training"].get("batch_size", 1))
    max_len = int(args.eval_seq_len or ctx.config["model"].get("max_seq_len", 512))
    max_tokens = int(args.eval_max_tokens or ctx.config.get("data", {}).get("max_val_tokens", 10_000_000))
    loader = load_eval_data(
        ctx.config,
        max_length=max_len,
        batch_size=batch_size,
        host_id=ctx.host_id,
        n_hosts=ctx.n_hosts,
        max_tokens=max_tokens,
    )
    max_batches = len(loader)
    if args.max_jobs_per_stage is not None:
        max_batches = min(max_batches, int(args.max_jobs_per_stage))
    step_fn = create_ce_eval_step(
        ctx.model,
        ctx.sharded_fns,
        minimal_train=True,
        return_prune_stats=False,
        cfg=ctx.config,
        total_training_steps=ctx.total_training_steps,
    )

    existing_rows = read_jsonl(by_batch_path) if args.resume and ctx.is_primary else []
    completed = {int(r["batch_idx"]) for r in existing_rows if "batch_idx" in r}
    rows_by_idx = {int(r["batch_idx"]): r for r in existing_rows if "batch_idx" in r}

    store.log_event(
        stage,
        "start",
        message=(
            "EVAL START "
            f"batches={max_batches} batch_size={batch_size} seq_len={max_len} "
            f"max_tokens={max_tokens:,} checkpoint_step={ctx.checkpoint_step}"
        ),
        batches=max_batches,
        batch_size=batch_size,
        seq_len=max_len,
        max_tokens=max_tokens,
    )

    for batch_idx, (input_ids, attention_mask) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if args.resume and batch_idx in completed:
            row = rows_by_idx[batch_idx]
            if ctx.is_primary:
                store.log_event(
                    stage,
                    "batch_skip",
                    message=(
                        "EVAL batch "
                        f"{batch_idx + 1:05d}/{max_batches:05d} SKIP "
                        f"loss={float(row.get('loss', 0.0)):.6f} "
                        f"acc={float(row.get('accuracy', 0.0)):.4f}"
                    ),
                    print_stdout=(batch_idx % max(1, int(args.log_every_batches)) == 0),
                    **row,
                )
            continue
        job_id = f"batch-{batch_idx:06d}"
        store.mark_job_started(stage, job_id)
        t0 = time.time()
        mesh_ids, mesh_mask = shard_batch_to_mesh(input_ids, attention_mask, ctx.data_sharding)
        loss, correct, valid_count = step_fn(
            ctx.params, mesh_ids, mesh_mask, jnp.int32(ctx.checkpoint_step or 0)
        )
        loss_f, correct_i, valid_i = jax.device_get((loss, correct, valid_count))
        sec = time.time() - t0
        loss_f = float(loss_f)
        correct_i = int(correct_i)
        valid_i = int(valid_i)
        row = {
            "batch_idx": int(batch_idx),
            "checkpoint_step": ctx.checkpoint_step,
            "loss": loss_f,
            "loss_sum": loss_f * valid_i,
            "correct": correct_i,
            "valid_count": valid_i,
            "accuracy": correct_i / valid_i if valid_i else 0.0,
            "sec": sec,
            "tok_per_sec": valid_i / sec if sec > 0 else 0.0,
            "sequences": int(input_ids.shape[0]) * int(ctx.n_hosts),
            "host_count": int(ctx.n_hosts),
        }
        rows_by_idx[batch_idx] = row
        if ctx.is_primary:
            rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
            write_jsonl_atomic(by_batch_path, rows)
            store.mark_job_complete(stage, job_id, by_batch_path, row)
            store.log_event(
                stage,
                "batch",
                message=(
                    "EVAL batch "
                    f"{batch_idx + 1:05d}/{max_batches:05d} "
                    f"loss={loss_f:.6f} acc={row['accuracy']:.4f} "
                    f"tokens={valid_i:,} tok/s={row['tok_per_sec']:.0f} "
                    f"sec={sec:.2f}"
                ),
                **row,
            )

    if ctx.is_primary:
        rows = [rows_by_idx[i] for i in sorted(rows_by_idx) if i < max_batches]
        summary = _aggregate(rows)
        summary.update({
            "checkpoint_step": ctx.checkpoint_step,
            "batch_size": batch_size,
            "max_val_tokens": max_tokens,
        })
        write_json_atomic(final_path, summary)
        store.mark_job_complete(stage, "final", final_path, summary)
        store.set_stage_status(stage, "complete")
        store.log_event(
            stage,
            "summary",
            message=(
                "EVAL SUMMARY "
                f"loss={summary['val_loss']:.6f} "
                f"ppl={summary['ppl']:.3f} "
                f"acc={summary['accuracy']:.4f} "
                f"tokens={summary['valid_tokens']:,} "
                f"tok/s={summary['tok_per_sec']:.0f}"
            ),
            **summary,
        )
        return summary
    return {}

