#!/usr/bin/env python3
"""Resumable DAWN-SRW v4166 analysis pipeline.

Example:
  python scripts/analyze_dawn_srw_v4166.py \
    --checkpoint gs://.../run_xxx/checkpoints/latest \
    --output gs://dawn-tpu-data-c4/analysis/v4166_1p3B_final \
    --stages eval,prune,geometry,usage,trace,ablation,report \
    --resume
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis import ANALYSIS_VERSION
from analysis.dawn_analysis_common import (
    AnalysisContext,
    config_from_checkpoint_or_file,
    count_params,
    create_mesh_from_cfg,
    create_or_reuse_sharded_fns,
    host_info,
    is_primary_host,
    model_cfg_from_config,
    resolve_checkpoint,
    restore_params_and_cfg,
    verify_v4166_config,
    write_run_metadata,
)
from analysis.dawn_analysis_report import run_report_stage
from analysis.dawn_analysis_storage import AnalysisStore, set_default_store, write_json_atomic


FULL_STAGE_ORDER = ["eval", "prune", "geometry", "usage", "trace", "ablation", "report"]
MODEL_REQUIRED_STAGES = {"eval", "prune", "geometry", "usage", "trace", "ablation"}
DEFAULT_CHECKPOINT = (
    "gs://dawn-tpu-data-c4/checkpoints/"
    "dawn_srw_v4166_400M_c4_40B_v4_64/"
    "run_vspatial-r1-v4.1.6.6_20260622_212706_3201/"
    "checkpoints/000000076293"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze DAWN-SRW spatial-r1-v4.1.6.6 checkpoints.")
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Optional legacy fallback YAML. Normal analysis loads "
            "metadata.full_config directly from the Orbax checkpoint."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Orbax run/checkpoints/step/latest path. Optional for report-only. "
            f"Default for model stages: {DEFAULT_CHECKPOINT}"
        ),
    )
    p.add_argument("--output", required=True, help="Output directory, local or gs://.")
    p.add_argument("--stages", default="all", help="Comma-separated stages or all.")
    p.add_argument("--resume", action="store_true", default=True, help="Skip valid completed artifacts (default).")
    p.add_argument("--from-scratch", action="store_true", help="Recompute stages from the beginning instead of resuming artifacts.")
    p.add_argument("--retry-failed", action="store_true", help="Allow failed manifest jobs to be recomputed.")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first stage/job failure.")
    p.add_argument("--max-jobs-per-stage", type=int, default=None, help="Limit jobs/batches per stage for smoke tests.")
    p.add_argument("--init-distributed", action="store_true", help="Force jax.distributed.initialize() before device access.")
    p.add_argument("--no-init-distributed", action="store_true", help="Disable TPU environment auto distributed initialization.")
    p.add_argument("--mesh-data", type=int, default=None, help="Override config training.mesh_data.")
    p.add_argument("--mesh-model", type=int, default=None, help="Override config training.mesh_model.")
    p.add_argument("--log-every-batches", type=int, default=1, help="Stdout cadence for batch logs.")

    p.add_argument("--eval-max-tokens", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--eval-seq-len", type=int, default=None)

    p.add_argument("--prune-eps", default=None, help="Comma-separated eps list.")
    p.add_argument("--prune-max-tokens", type=int, default=None)
    p.add_argument("--prune-batch-size", type=int, default=None)

    p.add_argument("--geometry-max-sample", type=int, default=2048)

    p.add_argument("--usage-max-sequences", type=int, default=512)
    p.add_argument("--usage-batch-size", type=int, default=64)
    p.add_argument("--usage-seq-len", type=int, default=128)
    p.add_argument("--usage-topk", type=int, default=8)
    p.add_argument("--usage-top-contexts-per-op", type=int, default=16)

    p.add_argument("--trace-prompts", default=None)
    p.add_argument("--trace-max-prompts", type=int, default=6)
    p.add_argument("--trace-seq-len", type=int, default=128)
    p.add_argument("--trace-topk", type=int, default=None)
    p.add_argument("--trace-save-full-gates", action="store_true")

    p.add_argument("--ablation-max-sequences", type=int, default=64)
    p.add_argument("--ablation-batch-size", type=int, default=64)
    p.add_argument("--ablation-seq-len", type=int, default=128)
    p.add_argument("--ablation-k-list", default="1,16,64")
    p.add_argument("--ablation-pools", default="qk,v,rst")
    p.add_argument("--ablation-strategies", default="top")
    p.add_argument("--ablation-random-seeds", type=int, default=3)

    p.add_argument("--report-format", default="html,md", help="Comma-separated report outputs: html,md.")
    p.add_argument("--report-max-operator-cards", type=int, default=100)
    p.add_argument("--report-max-trace-figures", type=int, default=12)
    p.add_argument("--report-include-appendix", action="store_true", default=True)
    p.add_argument("--baseline-analysis-output", default=None, help="Optional baseline analysis output directory.")
    p.add_argument("--baseline-eval-json", default=None, help="Optional baseline eval/final_eval.json path.")

    p.add_argument("--enable-patching", action="store_true")
    p.add_argument("--enable-steering", action="store_true")
    return p.parse_args()


def parse_stages(stages_arg: str) -> List[str]:
    raw = [s.strip() for s in str(stages_arg).split(",") if s.strip()]
    if not raw or raw == ["all"]:
        return list(FULL_STAGE_ORDER)
    stages = []
    for stage in raw:
        if stage == "all":
            stages.extend(FULL_STAGE_ORDER)
        elif stage in ("patching", "steering"):
            stages.append(stage)
        elif stage in FULL_STAGE_ORDER:
            stages.append(stage)
        else:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {FULL_STAGE_ORDER + ['patching', 'steering']}")
    dedup = []
    for stage in stages:
        if stage not in dedup:
            dedup.append(stage)
    return dedup


def normalize_checkpoint_arg(path: str | None) -> str | None:
    if not path:
        return path
    path = str(path).strip()
    if path.startswith("gs://"):
        return path
    if path.startswith("dawn-tpu-data-c4/"):
        return "gs://" + path
    return path


def _looks_like_tpu_multihost_env() -> bool:
    keys = (
        "TPU_WORKER_ID",
        "CLOUD_TPU_TASK_ID",
        "TPU_PROCESS_BOUNDS",
        "JAX_COORDINATOR_ADDRESS",
        "MEGASCALE_COORDINATOR_ADDRESS",
        "COORDINATOR_ADDRESS",
    )
    return any(os.environ.get(key) for key in keys)


def maybe_init_distributed(args: argparse.Namespace, needs_model: bool) -> None:
    should_init = bool(args.init_distributed)
    should_init = should_init or os.environ.get("DAWN_ANALYSIS_INIT_DISTRIBUTED", "").lower() in ("1", "true", "yes")
    should_init = should_init or (
        bool(needs_model)
        and not bool(args.no_init_distributed)
        and _looks_like_tpu_multihost_env()
    )
    if not should_init:
        return
    try:
        print("ANALYSIS distributed initialize: starting", flush=True)
        jax.distributed.initialize()
        print(
            "ANALYSIS distributed initialize: "
            f"process_index={jax.process_index()} process_count={jax.process_count()}",
            flush=True,
        )
    except RuntimeError as exc:
        if "already initialized" in str(exc).lower():
            print("ANALYSIS distributed initialize: already initialized", flush=True)
            return
        raise


def build_context(args: argparse.Namespace, stages: List[str], store: AnalysisStore) -> AnalysisContext:
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT
    args.checkpoint = normalize_checkpoint_arg(args.checkpoint)
    checkpoint_dir, checkpoint_step, checkpoint_metadata = resolve_checkpoint(args.checkpoint)
    cfg = config_from_checkpoint_or_file(args.config, checkpoint_metadata)
    verify_v4166_config(cfg)
    mesh = create_mesh_from_cfg(cfg, args)
    data_sharding = NamedSharding(mesh, P("data", None))
    store.log_event(
        "load",
        "mesh",
        message=(
            "LOAD MESH "
            f"mesh_data={mesh.shape['data']} mesh_model={mesh.shape['model']} "
            f"devices={jax.device_count()} checkpoint_step={checkpoint_step}"
        ),
        **host_info(),
    )
    params, restored_metadata, model = restore_params_and_cfg(cfg, checkpoint_dir, int(checkpoint_step), mesh)
    checkpoint_metadata.update(restored_metadata or {})
    sharded_fns = create_or_reuse_sharded_fns(cfg, mesh, analysis=False)
    sharded_fns_analysis = create_or_reuse_sharded_fns(cfg, mesh, analysis=True)
    model_cfg = model_cfg_from_config(cfg)
    n_params = count_params(params)
    info_lines = model.get_model_info() if hasattr(model, "get_model_info") else []
    model_info = {
        "model_info_lines": info_lines,
        "param_count": n_params,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path_resolved": checkpoint_dir,
        "mesh": {"data": int(mesh.shape["data"]), "model": int(mesh.shape["model"])},
    }
    write_run_metadata(store, cfg, args, checkpoint_metadata, model_info)
    if store.is_primary:
        print(
            "LOAD MODEL "
            f"version={cfg['model'].get('model_version')} params={n_params:,} "
            f"checkpoint_step={checkpoint_step}",
            flush=True,
        )
        for line in info_lines:
            print("MODEL " + str(line), flush=True)
    steps_per_epoch = int(checkpoint_metadata.get("steps_per_epoch") or 0)
    num_epochs = int(cfg.get("training", {}).get("num_epochs") or 0)
    grad_accum = max(1, int(cfg.get("training", {}).get("gradient_accumulation_steps", 1)))
    planned_steps = (steps_per_epoch // grad_accum) * num_epochs if steps_per_epoch and num_epochs else 0
    total_training_steps = int(planned_steps or checkpoint_metadata.get("global_step") or checkpoint_step or 1)
    return AnalysisContext(
        args=args,
        store=store,
        config=cfg,
        config_path=args.config,
        checkpoint_path=checkpoint_dir,
        checkpoint_step=int(checkpoint_step),
        checkpoint_metadata=checkpoint_metadata,
        model=model,
        params=params,
        mesh=mesh,
        data_sharding=data_sharding,
        sharded_fns=sharded_fns,
        sharded_fns_analysis=sharded_fns_analysis,
        model_cfg=model_cfg,
        model_info=model_info,
        host_id=int(jax.process_index()),
        n_hosts=int(jax.process_count()),
        is_primary=store.is_primary,
        total_training_steps=total_training_steps,
    )


def run_stage(ctx: AnalysisContext, stage: str):
    if stage == "eval":
        from analysis.dawn_analysis_eval import run_eval_stage

        return run_eval_stage(ctx)
    if stage == "prune":
        from analysis.dawn_analysis_prune import run_prune_stage

        return run_prune_stage(ctx)
    if stage == "geometry":
        from analysis.dawn_analysis_geometry import run_geometry_stage

        return run_geometry_stage(ctx)
    if stage == "usage":
        from analysis.dawn_analysis_usage import run_usage_stage

        return run_usage_stage(ctx)
    if stage == "trace":
        from analysis.dawn_analysis_trace import run_trace_stage

        return run_trace_stage(ctx)
    if stage == "ablation":
        from analysis.dawn_analysis_ablation import run_ablation_stage

        return run_ablation_stage(ctx)
    if stage == "report":
        return run_report_stage(ctx)
    if stage == "patching":
        ctx.store.set_stage_status(stage, "complete")
        ctx.store.log_event(stage, "disabled", message="PATCHING disabled by default; scaffold directory is ready")
        return {}
    if stage == "steering":
        ctx.store.set_stage_status(stage, "complete")
        ctx.store.log_event(stage, "disabled", message="STEERING disabled by default; scaffold directory is ready")
        return {}
    raise ValueError(stage)


def main() -> int:
    args = parse_args()
    if args.from_scratch:
        args.resume = False
    stages = parse_stages(args.stages)
    needs_model = any(stage in MODEL_REQUIRED_STAGES for stage in stages)
    maybe_init_distributed(args, needs_model)
    primary = is_primary_host()
    store = AnalysisStore(args.output, is_primary=primary, analysis_version=ANALYSIS_VERSION)
    set_default_store(store)
    store.ensure_layout()
    if primary:
        print(
            "ANALYSIS START "
            f"version={ANALYSIS_VERSION} output={args.output} stages={','.join(stages)} "
            f"resume={bool(args.resume)} retry_failed={bool(args.retry_failed)}",
            flush=True,
        )
        store.update_manifest(
            analysis_version=ANALYSIS_VERSION,
            selected_stages=stages,
            cli_args=vars(args),
        )

    ctx = None
    if needs_model:
        ctx = build_context(args, stages, store)
    elif primary:
        print("ANALYSIS REPORT-ONLY mode: model/checkpoint load skipped", flush=True)

    failures = []
    for stage in stages:
        try:
            if stage == "report" and ctx is None:
                run_report_stage(None, store=store, args=args)
            elif ctx is None:
                raise ValueError(f"Stage {stage} requires a loaded model context.")
            else:
                run_stage(ctx, stage)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append((stage, exc))
            store.mark_job_failed(stage, "stage", str(exc), tb)
            store.log_event(
                stage,
                "failed",
                message=f"{stage.upper()} FAILED {type(exc).__name__}: {exc}",
                error=str(exc),
            )
            if args.fail_fast or stage in ("load",):
                raise
    if primary:
        if failures:
            print(
                "ANALYSIS COMPLETE_WITH_FAILURES "
                + " ".join(f"{stage}:{type(exc).__name__}" for stage, exc in failures),
                flush=True,
            )
        else:
            print(f"ANALYSIS COMPLETE output={args.output}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
