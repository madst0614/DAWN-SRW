"""Execution-pruning sweep stage."""

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
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_PRUNE_EPS = [0.0, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def parse_eps_list(value: str | None) -> List[float]:
    if not value:
        return list(DEFAULT_PRUNE_EPS)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def eps_tag(eps: float) -> str:
    if float(eps) == 0.0:
        return "0"
    return f"{float(eps):.0e}".replace("-", "m")


def _aggregate(rows: List[Dict[str, Any]], eps: float, base_loss: float | None = None) -> Dict[str, Any]:
    loss_sum = sum(float(r.get("loss_sum", 0.0)) for r in rows)
    correct = sum(int(r.get("correct", 0)) for r in rows)
    valid = sum(int(r.get("valid_count", 0)) for r in rows)
    sec = sum(float(r.get("sec", 0.0)) for r in rows)
    denom = max(1, len(rows))
    loss = loss_sum / valid if valid else float("nan")
    return {
        "eps": float(eps),
        "val_loss": loss,
        "delta_loss_vs_eps0": (loss - base_loss) if base_loss is not None and valid else None,
        "ppl": math.exp(min(loss, 80.0)) if valid else float("nan"),
        "accuracy": correct / valid if valid else float("nan"),
        "valid_tokens": int(valid),
        "sec": sec,
        "tok_per_sec": valid / sec if sec > 0 else 0.0,
        "estimated_compute_frac": sum(float(r.get("estimated_compute_frac", 0.0)) for r in rows) / denom,
        "execution_gate_mass_retained": sum(float(r.get("execution_gate_mass_retained", 1.0)) for r in rows) / denom,
        "execution_prune_gate_den_mean": sum(float(r.get("execution_prune_gate_den_mean", 0.0)) for r in rows) / denom,
        "execution_prune_unpruned_gate_den_mean": sum(float(r.get("execution_prune_unpruned_gate_den_mean", 0.0)) for r in rows) / denom,
        "execution_prune_no_active_frac": sum(float(r.get("execution_prune_no_active_frac", 0.0)) for r in rows) / denom,
        "num_batches": len(rows),
    }


def _run_one_eps(ctx: AnalysisContext, eps: float, max_batches: int,
                 batch_size: int, max_len: int, max_tokens: int,
                 base_loss: float | None) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    tag = eps_tag(eps)
    stage = "prune"
    eps_dir = store.path("prune", f"eps_{tag}")
    by_batch_path = store.path("prune", f"eps_{tag}", "by_batch.jsonl")
    summary_path = store.path("prune", f"eps_{tag}", "prune_summary.json")
    if args.resume and should_skip_job(summary_path, ["val_loss", "valid_tokens"]):
        summary = _aggregate(read_jsonl(by_batch_path), eps, base_loss)
        store.log_event(
            stage,
            "eps_skip",
            message=(
                f"PRUNE eps={eps:g} SKIP "
                f"loss={summary['val_loss']:.6f} "
                f"delta={summary.get('delta_loss_vs_eps0')} "
                f"compute={summary['estimated_compute_frac']:.4f}"
            ),
            **summary,
        )
        return summary

    loader = load_eval_data(
        ctx.config,
        max_length=max_len,
        batch_size=batch_size,
        host_id=ctx.host_id,
        n_hosts=ctx.n_hosts,
        max_tokens=max_tokens,
    )
    step_fn = create_ce_eval_step(
        ctx.model,
        ctx.sharded_fns,
        minimal_train=False,
        return_prune_stats=True,
        execution_prune_eps=eps,
        cfg=ctx.config,
        total_training_steps=ctx.total_training_steps,
    )
    existing_rows = read_jsonl(by_batch_path) if args.resume and ctx.is_primary else []
    rows_by_idx = {int(r["batch_idx"]): r for r in existing_rows if "batch_idx" in r}
    completed = set(rows_by_idx)

    store.log_event(
        stage,
        "eps_start",
        message=f"PRUNE eps={eps:g} START batches={max_batches} batch_size={batch_size} seq_len={max_len}",
        eps=eps,
        batches=max_batches,
    )

    for batch_idx, (input_ids, attention_mask) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if args.resume and batch_idx in completed:
            continue
        job_id = f"eps-{tag}-batch-{batch_idx:06d}"
        store.mark_job_started(stage, job_id)
        t0 = time.time()
        mesh_ids, mesh_mask = shard_batch_to_mesh(input_ids, attention_mask, ctx.data_sharding)
        ret = step_fn(ctx.params, mesh_ids, mesh_mask, jnp.int32(ctx.checkpoint_step or 0))
        (
            loss,
            correct,
            valid_count,
            compute_frac,
            mass_retained,
            gate_den,
            no_active,
            unpruned_den,
        ) = jax.device_get(ret)
        sec = time.time() - t0
        loss = float(loss)
        valid = int(valid_count)
        correct = int(correct)
        row = {
            "batch_idx": int(batch_idx),
            "eps": float(eps),
            "loss": loss,
            "loss_sum": loss * valid,
            "correct": correct,
            "valid_count": valid,
            "accuracy": correct / valid if valid else 0.0,
            "estimated_compute_frac": float(compute_frac),
            "execution_gate_mass_retained": float(mass_retained),
            "execution_prune_gate_den_mean": float(gate_den),
            "execution_prune_no_active_frac": float(no_active),
            "execution_prune_unpruned_gate_den_mean": float(unpruned_den),
            "sec": sec,
            "tok_per_sec": valid / sec if sec > 0 else 0.0,
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
                    f"PRUNE eps={eps:g} batch {batch_idx + 1:05d}/{max_batches:05d} "
                    f"loss={loss:.6f} acc={row['accuracy']:.4f} "
                    f"compute={row['estimated_compute_frac']:.4f} "
                    f"mass={row['execution_gate_mass_retained']:.4f} "
                    f"tok/s={row['tok_per_sec']:.0f}"
                ),
                **row,
            )

    if not ctx.is_primary:
        return {}
    rows = [rows_by_idx[i] for i in sorted(rows_by_idx) if i < max_batches]
    summary = _aggregate(rows, eps, base_loss)
    write_json_atomic(summary_path, summary)
    store.mark_job_complete(stage, f"eps-{tag}", summary_path, summary)
    store.log_event(
        stage,
        "eps_summary",
        message=(
            f"PRUNE eps={eps:g} SUMMARY "
            f"loss={summary['val_loss']:.6f} "
            f"delta={summary['delta_loss_vs_eps0']} "
            f"acc={summary['accuracy']:.4f} "
            f"compute={summary['estimated_compute_frac']:.4f} "
            f"mass={summary['execution_gate_mass_retained']:.4f}"
        ),
        **summary,
    )
    return summary


def run_prune_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "prune"
    store.set_stage_status(stage, "running")
    eps_values = parse_eps_list(args.prune_eps)
    batch_size = int(args.prune_batch_size or args.eval_batch_size or ctx.config["training"].get("batch_size", 1))
    max_len = int(args.eval_seq_len or ctx.config["model"].get("max_seq_len", 512))
    max_tokens = int(args.prune_max_tokens or args.eval_max_tokens or ctx.config.get("data", {}).get("max_val_tokens", 10_000_000))
    loader_probe = load_eval_data(ctx.config, max_len, batch_size, ctx.host_id, ctx.n_hosts, max_tokens)
    max_batches = len(loader_probe)
    if args.max_jobs_per_stage is not None:
        max_batches = min(max_batches, int(args.max_jobs_per_stage))

    summaries: List[Dict[str, Any]] = []
    base_loss = None
    for eps in eps_values:
        summary = _run_one_eps(ctx, eps, max_batches, batch_size, max_len, max_tokens, base_loss)
        if ctx.is_primary:
            if float(eps) == 0.0:
                base_loss = float(summary["val_loss"])
                summary["delta_loss_vs_eps0"] = 0.0
                write_json_atomic(store.path("prune", f"eps_{eps_tag(eps)}", "prune_summary.json"), summary)
            summaries.append(summary)

    if ctx.is_primary:
        if base_loss is not None:
            for s in summaries:
                s["delta_loss_vs_eps0"] = float(s["val_loss"]) - base_loss
        write_json_atomic(store.path("prune", "prune_sweep.json"), {"eps": summaries})
        csv_rows = [
            {
                "eps": s["eps"],
                "val_loss": s["val_loss"],
                "delta_loss": s.get("delta_loss_vs_eps0"),
                "accuracy": s["accuracy"],
                "valid_tokens": s["valid_tokens"],
                "estimated_compute_frac": s["estimated_compute_frac"],
                "gate_mass_retained": s["execution_gate_mass_retained"],
            }
            for s in summaries
        ]
        write_csv_atomic(store.path("prune", "prune_curve.csv"), csv_rows)
        final = {"eps": summaries, "num_eps": len(summaries)}
        store.mark_job_complete(stage, "sweep", store.path("prune", "prune_sweep.json"), final)
        store.set_stage_status(stage, "complete")
        best = min(summaries, key=lambda x: abs(float(x.get("delta_loss_vs_eps0") or 0.0))) if summaries else {}
        store.log_event(
            stage,
            "summary",
            message=(
                "PRUNE SUMMARY "
                f"eps_count={len(summaries)} "
                f"base_loss={base_loss} "
                f"nearest_delta_eps={best.get('eps')} "
                f"nearest_delta={best.get('delta_loss_vs_eps0')}"
            ),
            **final,
        )
        return final
    return {}

