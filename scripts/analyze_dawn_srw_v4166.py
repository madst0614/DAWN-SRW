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
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis import ANALYSIS_VERSION
from analysis.dawn_analysis_common import (
    AnalysisContext,
    config_from_checkpoint_or_file,
    count_params,
    create_active_analysis_step,
    create_ce_eval_step,
    create_mesh_from_cfg,
    create_or_reuse_sharded_fns,
    get_train,
    host_info,
    is_primary_host,
    load_eval_data,
    model_cfg_from_config,
    resolve_checkpoint,
    restore_params_and_cfg,
    shard_batch_to_mesh,
    sync_hosts,
    verify_v4166_config,
    write_run_metadata,
)
from analysis.dawn_analysis_report import run_report_stage
from analysis.dawn_analysis_storage import (
    AnalysisStore,
    append_jsonl,
    exists,
    is_gcs_path,
    join_path,
    list_paths,
    read_jsonl,
    set_default_store,
    write_json_atomic,
    write_text_atomic,
)


FULL_STAGE_ORDER = ["eval", "prune", "geometry", "usage", "trace", "ablation", "report"]
MODEL_REQUIRED_STAGES = {"eval", "prune", "geometry", "usage", "trace", "ablation"}
DEFAULT_CHECKPOINT = (
    "gs://dawn-tpu-data-c4/checkpoints/"
    "dawn_srw_v4166_400M_c4_40B_v4_64/"
    "run_vspatial-r1-v4.1.6.6_20260622_212706_3201/"
    "checkpoints/000000076293"
)
DEFAULT_TRAIN_ANALYSIS_CONFIG = None
DEFAULT_TRAIN_ANALYSIS_CHECKPOINT_DIR = (
    "gs://dawn-tpu-data-c4/checkpoints/"
    "dawn_srw_v4166_1p3B_c4_20B_v4_64_new"
)
DEFAULT_TRAIN_ANALYSIS_BATCHES = 8
DEFAULT_TRAIN_ANALYSIS_PRUNE_EPS = "1e-6,1e-5,1e-4,1e-3"
TRAIN_ANALYSIS_POOLS = {
    "qk": "attn_qk",
    "v": "attn_v",
    "rst": "rst",
}
REFERENCE_400M = {
    ("qk", "active_tau"): (0.045, 0.065),
    ("v", "active_tau"): (0.090, 0.120),
    ("rst", "active_tau"): (0.085, 0.110),
    ("qk", "effective"): (0.025, 0.040),
    ("v", "effective"): (0.045, 0.070),
    ("rst", "effective"): (0.045, 0.070),
}


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
    p.add_argument("--output", default=None, help="Output directory, local or gs://.")
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

    p.add_argument("--train-analysis", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint-dir", dest="train_analysis_checkpoint_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--train-analysis-max-batches",
        type=int,
        default=int(os.environ.get("DAWN_TRAIN_ANALYSIS_MAX_BATCHES", DEFAULT_TRAIN_ANALYSIS_BATCHES)),
        help=argparse.SUPPRESS,
    )

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


def _path_name(path: str) -> str:
    return str(path).rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def _path_parent(path: str) -> str:
    path = str(path).rstrip("/\\").replace("\\", "/")
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _parse_float_list(value: str | None) -> List[float]:
    if not value:
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = jax.device_get(value)
    except Exception:
        pass
    try:
        if hasattr(value, "shape") and getattr(value, "shape", ()) not in ((), None):
            if getattr(value, "size", 1) != 1:
                return None
            value = value.reshape(()).item()
        elif hasattr(value, "item"):
            value = value.item()
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt_num(value: Any, digits: int = 3) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.{digits}f}"


def _fmt_delta(value: Any, digits: int = 4) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:+.{digits}f}"


def _fmt_tokens(value: Any) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    abs_v = abs(value_f)
    if abs_v >= 1e12:
        return f"{value_f / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{value_f / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{value_f / 1e6:.2f}M"
    return f"{value_f:,.0f}"


def _fmt_duration(seconds: Any) -> str:
    seconds_f = _safe_float(seconds)
    if seconds_f is None or seconds_f < 0:
        return "n/a"
    total = int(round(seconds_f))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _status_against_range(value: Any, low: float, high: float) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    if value_f < low:
        return "LOW"
    if value_f > high:
        return "HIGH"
    return "OK"


def _list_run_folders(base: str) -> List[str]:
    try:
        entries = list_paths(base, "*")
    except FileNotFoundError:
        return []
    runs = [
        str(path).rstrip("/")
        for path in entries
        if _path_name(str(path)).startswith("run_")
    ]
    return sorted(set(runs))


def _parse_datetime_to_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _epoch_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), timezone.utc).replace(microsecond=0).isoformat()


def _checkpoint_mtime(step_dir: str) -> Tuple[Optional[str], Optional[float]]:
    marker = join_path(step_dir, "commit_success.txt")
    if not exists(marker):
        return None, None
    if is_gcs_path(marker):
        try:
            import gcsfs

            fs = gcsfs.GCSFileSystem()
            try:
                info = fs.info(marker)
            except Exception:
                info = fs.info(marker.replace("gs://", "", 1))
            raw = (
                info.get("updated")
                or info.get("mtime")
                or info.get("created")
                or info.get("timeCreated")
            )
            ts = _parse_datetime_to_epoch(raw)
            return _epoch_to_iso(ts), ts
        except Exception:
            return None, None
    try:
        ts = Path(marker).stat().st_mtime
        return _epoch_to_iso(ts), ts
    except Exception:
        return None, None


def _checkpoint_infos(checkpoints_dir: str) -> List[Dict[str, Any]]:
    out = []
    try:
        entries = list_paths(checkpoints_dir, "*")
    except Exception:
        return out
    for path in entries:
        name = _path_name(str(path))
        if not name.isdigit():
            continue
        marker = join_path(path, "commit_success.txt")
        if not exists(marker):
            continue
        modified, ts = _checkpoint_mtime(str(path))
        out.append({
            "step": int(name),
            "path": str(path),
            "modified": modified,
            "modified_ts": ts,
        })
    return sorted(out, key=lambda row: int(row["step"]))


def _discover_latest_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    train = get_train()
    checkpoint_dir = str(checkpoint_dir).rstrip("/\\")
    if _path_name(checkpoint_dir) == "checkpoints":
        candidates = [_path_parent(checkpoint_dir)]
        configured_base = _path_parent(checkpoint_dir)
    else:
        runs = _list_run_folders(checkpoint_dir)
        candidates = runs if runs else [checkpoint_dir]
        configured_base = checkpoint_dir
    for run_folder in reversed(sorted(candidates)):
        step = train._latest_orbax_step_for_run(run_folder)
        if step is None:
            continue
        checkpoints_dir = join_path(run_folder, "checkpoints")
        infos = _checkpoint_infos(checkpoints_dir)
        latest = next((row for row in infos if int(row["step"]) == int(step)), None)
        latest_path = latest.get("path") if latest else join_path(checkpoints_dir, str(step))
        return {
            "configured_checkpoint_dir": configured_base,
            "run_folder": run_folder,
            "checkpoints_dir": checkpoints_dir,
            "latest_step": int(step),
            "latest_path": latest_path,
            "latest_modified": latest.get("modified") if latest else None,
            "checkpoint_infos": infos,
        }
    raise FileNotFoundError(f"No committed Orbax checkpoints found under {checkpoint_dir}")


def _prepare_train_analysis_args(args: argparse.Namespace, warnings: List[str]) -> Dict[str, Any]:
    args.config = (
        args.config
        or os.environ.get("DAWN_TRAIN_ANALYSIS_CONFIG")
        or DEFAULT_TRAIN_ANALYSIS_CONFIG
    )
    if args.config is not None:
        args.config = normalize_checkpoint_arg(args.config)
    if args.output is not None:
        args.output = normalize_checkpoint_arg(args.output)
    checkpoint_dir = (
        args.train_analysis_checkpoint_dir
        or os.environ.get("DAWN_TRAIN_ANALYSIS_CHECKPOINT_DIR")
        or None
    )
    if checkpoint_dir is None and args.checkpoint is None:
        checkpoint_dir = DEFAULT_TRAIN_ANALYSIS_CHECKPOINT_DIR
    if checkpoint_dir is not None:
        checkpoint_dir = normalize_checkpoint_arg(checkpoint_dir)
        info = _discover_latest_checkpoint(checkpoint_dir)
        args.checkpoint = join_path(info["checkpoints_dir"], str(info["latest_step"]))
        if args.output is None:
            args.output = join_path(info["configured_checkpoint_dir"], "side_analysis")
        return info
    args.checkpoint = normalize_checkpoint_arg(args.checkpoint)
    if args.output is None:
        parent = _path_parent(_path_parent(str(args.checkpoint)))
        args.output = join_path(parent, "side_analysis")
    try:
        checkpoints_dir, step, _ = resolve_checkpoint(args.checkpoint)
        infos = _checkpoint_infos(checkpoints_dir) if checkpoints_dir else []
        latest = next((row for row in infos if int(row["step"]) == int(step)), None)
        return {
            "configured_checkpoint_dir": _path_parent(checkpoints_dir) if checkpoints_dir else None,
            "run_folder": _path_parent(checkpoints_dir) if checkpoints_dir else None,
            "checkpoints_dir": checkpoints_dir,
            "latest_step": int(step) if step is not None else None,
            "latest_path": latest.get("path") if latest else args.checkpoint,
            "latest_modified": latest.get("modified") if latest else None,
            "checkpoint_infos": infos,
        }
    except Exception as exc:
        warnings.append(f"checkpoint metadata preflight skipped: {type(exc).__name__}: {exc}")
        return {
            "configured_checkpoint_dir": None,
            "run_folder": None,
            "checkpoints_dir": None,
            "latest_step": None,
            "latest_path": args.checkpoint,
            "latest_modified": None,
            "checkpoint_infos": [],
        }


def _init_train_analysis_store(output: str, primary: bool, warnings: List[str]) -> AnalysisStore:
    try:
        store = AnalysisStore(output, is_primary=primary, analysis_version=ANALYSIS_VERSION)
        if primary:
            store.ensure_layout()
        return store
    except Exception as exc:
        fallback = os.environ.get("DAWN_TRAIN_ANALYSIS_FALLBACK_OUTPUT", "runs/side_analysis")
        warnings.append(
            f"primary output unavailable ({type(exc).__name__}: {exc}); using fallback {fallback}"
        )
        store = AnalysisStore(fallback, is_primary=primary, analysis_version=ANALYSIS_VERSION)
        if primary:
            store.ensure_layout()
        return store


def _total_steps_from_config(cfg: Dict[str, Any]) -> int:
    tcfg = cfg.get("training", {})
    dcfg = cfg.get("data", {})
    batch_size = int(tcfg.get("batch_size", 1))
    seq_len = int(cfg.get("model", {}).get("max_seq_len", 512))
    max_train_tokens = int(dcfg.get("max_train_tokens", 0) or 0)
    if max_train_tokens > 0:
        return max(1, int(math.ceil(max_train_tokens / max(1, batch_size * seq_len))))
    return max(1, int(tcfg.get("total_steps_for_restore", 1)))


def _selection_status(cfg: Dict[str, Any], step: int, tokens: int) -> Dict[str, Any]:
    tcfg = cfg.get("training", {})
    scfg = tcfg.get("selection_calibration", {})
    if not isinstance(scfg, dict):
        scfg = {}
    anneal = scfg.get("annealing", {}) if isinstance(scfg.get("annealing", {}), dict) else {}
    shrink_start = int(anneal.get("shrink_start_tokens", 0) or 0)
    shrink_end = int(anneal.get("shrink_end_tokens", 0) or 0)
    if shrink_end > shrink_start:
        shrink_progress = min(1.0, max(0.0, (float(tokens) - shrink_start) / (shrink_end - shrink_start)))
    else:
        shrink_progress = 0.0

    def targets(name: str) -> Dict[str, Optional[float]]:
        raw = scfg.get(name, {})
        raw = raw if isinstance(raw, dict) else {}
        return {pool: _safe_float(raw.get(pool)) for pool in ("qk", "v", "rst")}

    start = targets("candidate_target_start")
    final = targets("candidate_target_final")
    now = {}
    for pool in ("qk", "v", "rst"):
        s = start.get(pool)
        f = final.get(pool)
        now[pool] = None if s is None or f is None else s + (f - s) * shrink_progress

    total_steps = _total_steps_from_config(cfg)
    frac = min(1.0, max(0.0, float(step) / max(1.0, float(total_steps))))
    start_power = float(tcfg.get("soft_gate_boundary_power_start", 3.0))
    mid_power = float(tcfg.get("soft_gate_boundary_power_mid", start_power))
    final_power = float(tcfg.get("soft_gate_boundary_power_final", mid_power))
    start_frac = float(tcfg.get("soft_gate_boundary_power_start_frac", 0.0))
    mid_frac = float(tcfg.get("soft_gate_boundary_power_mid_frac", 0.8))
    final_frac = float(tcfg.get("soft_gate_boundary_power_final_frac", 0.95))
    eps = 1e-6
    if frac < start_frac:
        boundary_power = start_power
    elif frac < mid_frac:
        u = min(1.0, max(0.0, (frac - start_frac) / max(mid_frac - start_frac, eps)))
        boundary_power = start_power + (mid_power - start_power) * (u * u)
    elif frac < final_frac:
        u = min(1.0, max(0.0, (frac - mid_frac) / max(final_frac - mid_frac, eps)))
        boundary_power = mid_power + (final_power - mid_power) * u
    else:
        boundary_power = final_power

    return {
        "enabled": bool(scfg.get("enabled", False)),
        "active_target": targets("active_target"),
        "candidate_start": start,
        "candidate_final": final,
        "candidate_now": now,
        "shrink_start_tokens": shrink_start,
        "shrink_end_tokens": shrink_end,
        "shrink_progress": shrink_progress,
        "soft_gate_boundary_power_now": boundary_power,
    }


def _run_active_dynamics(ctx: AnalysisContext, max_batches: int) -> Dict[str, Any]:
    args = ctx.args
    batch_size = int(ctx.config["training"].get("batch_size", 1))
    seq_len = int(ctx.config["model"].get("max_seq_len", 512))
    max_tokens = batch_size * seq_len * int(max_batches)
    loader = load_eval_data(
        ctx.config,
        max_length=seq_len,
        batch_size=batch_size,
        host_id=ctx.host_id,
        n_hosts=ctx.n_hosts,
        max_tokens=max_tokens,
    )
    max_batches = min(int(max_batches), len(loader))
    step_fn = create_active_analysis_step(
        ctx.model,
        ctx.sharded_fns_analysis,
        cfg=ctx.config,
        total_training_steps=ctx.total_training_steps,
    )
    wanted = {"loss", "correct", "valid_count", "aux_loss"}
    for prefix in TRAIN_ANALYSIS_POOLS.values():
        wanted.update({
            f"{prefix}_active_tau_frac",
            f"{prefix}_admission_active_eps_1e_2_frac",
            f"{prefix}_active_eps_1e_2_frac",
            f"{prefix}_mass_eps_1e_2",
            f"{prefix}_execution_mass_sum",
            f"{prefix}_execution_top1_frac",
            f"{prefix}_top1_gate_frac",
            f"{prefix}_tau_mean",
            f"{prefix}_tau_min",
            f"{prefix}_tau_max",
            f"{prefix}_rho_mean",
            f"{prefix}_score_std",
            f"{prefix}_rho_std",
            f"{prefix}_rho_max",
        })
    rows: List[Dict[str, Any]] = []
    if ctx.is_primary:
        print(
            "TRAIN_ANALYSIS ACTIVE START "
            f"batches={max_batches} batch_size={batch_size} seq_len={seq_len}",
            flush=True,
        )
    for batch_idx, (input_ids, attention_mask) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        t0 = time.time()
        mesh_ids, mesh_mask = shard_batch_to_mesh(input_ids, attention_mask, ctx.data_sharding)
        result = step_fn(ctx.params, mesh_ids, mesh_mask, jnp.int32(ctx.checkpoint_step or 0))
        selected = {key: value for key, value in result.items() if key in wanted}
        jax.block_until_ready(jax.tree.leaves(selected))
        selected = jax.device_get(selected)
        row = {key: _safe_float(value) for key, value in selected.items()}
        row["sec"] = time.time() - t0
        rows.append(row)
        if ctx.is_primary:
            loss = _fmt_num(row.get("loss"), 6)
            print(
                "TRAIN_ANALYSIS ACTIVE "
                f"batch={batch_idx + 1:05d}/{max_batches:05d} "
                f"loss={loss} sec={row['sec']:.2f}",
                flush=True,
            )
    if not ctx.is_primary:
        return {}

    valid_total = sum(int(row.get("valid_count") or 0) for row in rows)
    correct_total = sum(int(row.get("correct") or 0) for row in rows)
    loss_sum = sum(float(row.get("loss") or 0.0) * int(row.get("valid_count") or 0) for row in rows)

    def mean_key(key: str) -> Optional[float]:
        vals = [_safe_float(row.get(key)) for row in rows]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    pools = {}
    for pool, prefix in TRAIN_ANALYSIS_POOLS.items():
        active_tau = mean_key(f"{prefix}_active_tau_frac")
        admission = mean_key(f"{prefix}_admission_active_eps_1e_2_frac")
        effective = mean_key(f"{prefix}_active_eps_1e_2_frac")
        top1 = mean_key(f"{prefix}_execution_top1_frac")
        if top1 is None:
            top1 = mean_key(f"{prefix}_top1_gate_frac")
        gate_mass = mean_key(f"{prefix}_mass_eps_1e_2")
        if gate_mass is None:
            gate_mass = mean_key(f"{prefix}_execution_mass_sum")
        score_std = mean_key(f"{prefix}_score_std")
        if score_std is None:
            score_std = mean_key(f"{prefix}_rho_std")
        pools[pool] = {
            "active_tau": active_tau,
            "admission": admission,
            "effective": effective,
            "top1": top1,
            "tau_mean": mean_key(f"{prefix}_tau_mean"),
            "tau_min": mean_key(f"{prefix}_tau_min"),
            "tau_max": mean_key(f"{prefix}_tau_max"),
            "gate_mass": gate_mass,
            "score_mean": mean_key(f"{prefix}_rho_mean"),
            "score_std": score_std,
            "score_p95": None,
        }
    return {
        "num_batches": len(rows),
        "valid_tokens": valid_total,
        "val_loss": loss_sum / valid_total if valid_total else None,
        "accuracy": correct_total / valid_total if valid_total else None,
        "sec": sum(float(row.get("sec") or 0.0) for row in rows),
        "pools": pools,
    }


def _run_prune_light(ctx: AnalysisContext, max_batches: int, eps_values: List[float]) -> Dict[str, Any]:
    batch_size = int(ctx.config["training"].get("batch_size", 1))
    seq_len = int(ctx.config["model"].get("max_seq_len", 512))
    max_tokens = batch_size * seq_len * int(max_batches)
    eps_with_base = [0.0] + [eps for eps in eps_values if float(eps) != 0.0]
    summaries = []
    base_loss = None
    for eps in eps_with_base:
        loader = load_eval_data(
            ctx.config,
            max_length=seq_len,
            batch_size=batch_size,
            host_id=ctx.host_id,
            n_hosts=ctx.n_hosts,
            max_tokens=max_tokens,
        )
        local_max_batches = min(int(max_batches), len(loader))
        step_fn = create_ce_eval_step(
            ctx.model,
            ctx.sharded_fns,
            minimal_train=False,
            return_prune_stats=True,
            execution_prune_eps=float(eps),
            cfg=ctx.config,
            total_training_steps=ctx.total_training_steps,
        )
        if ctx.is_primary:
            print(
                "TRAIN_ANALYSIS PRUNE START "
                f"eps={eps:g} batches={local_max_batches}",
                flush=True,
            )
        loss_sum = 0.0
        correct = 0
        valid = 0
        compute_sum = 0.0
        mass_sum = 0.0
        no_active_sum = 0.0
        row_count = 0
        sec_sum = 0.0
        for batch_idx, (input_ids, attention_mask) in enumerate(loader):
            if batch_idx >= local_max_batches:
                break
            t0 = time.time()
            mesh_ids, mesh_mask = shard_batch_to_mesh(input_ids, attention_mask, ctx.data_sharding)
            ret = step_fn(ctx.params, mesh_ids, mesh_mask, jnp.int32(ctx.checkpoint_step or 0))
            ret = jax.device_get(ret)
            (
                loss,
                correct_i,
                valid_i,
                compute_frac,
                mass_retained,
                _gate_den,
                no_active,
                _unpruned_den,
            ) = ret
            loss_f = float(loss)
            valid_i = int(valid_i)
            correct_i = int(correct_i)
            loss_sum += loss_f * valid_i
            correct += correct_i
            valid += valid_i
            compute_sum += float(compute_frac)
            mass_sum += float(mass_retained)
            no_active_sum += float(no_active)
            row_count += 1
            sec_sum += time.time() - t0
        summary = {
            "eps": float(eps),
            "num_batches": row_count,
            "valid_tokens": valid,
            "val_loss": loss_sum / valid if valid else None,
            "accuracy": correct / valid if valid else None,
            "compute_frac": compute_sum / row_count if row_count else None,
            "gate_mass_retained": mass_sum / row_count if row_count else None,
            "no_active_frac": no_active_sum / row_count if row_count else None,
            "sec": sec_sum,
            "qk_eff": None,
            "v_eff": None,
            "rst_eff": None,
        }
        if float(eps) == 0.0:
            base_loss = summary["val_loss"]
        if base_loss is not None and summary["val_loss"] is not None:
            summary["loss_delta"] = float(summary["val_loss"]) - float(base_loss)
        else:
            summary["loss_delta"] = None
        if ctx.is_primary:
            print(
                "TRAIN_ANALYSIS PRUNE SUMMARY "
                f"eps={eps:g} loss={_fmt_num(summary['val_loss'], 6)} "
                f"delta={_fmt_delta(summary['loss_delta'])} "
                f"compute={_fmt_num(summary['compute_frac'], 4)}",
                flush=True,
            )
        if float(eps) != 0.0:
            summaries.append(summary)
    if not ctx.is_primary:
        return {}
    return {
        "base_loss": base_loss,
        "eps": summaries,
    }


def _progress_status(ctx: AnalysisContext, info: Dict[str, Any]) -> Dict[str, Any]:
    step = int(ctx.checkpoint_step or 0)
    tcfg = ctx.config.get("training", {})
    dcfg = ctx.config.get("data", {})
    batch_size = int(tcfg.get("batch_size", 1))
    seq_len = int(ctx.config.get("model", {}).get("max_seq_len", 512))
    tokens_per_step = batch_size * seq_len
    tokens = step * tokens_per_step
    max_tokens = int(dcfg.get("max_train_tokens", 0) or 0)
    progress = (tokens / max_tokens) if max_tokens > 0 else None
    total_steps = _total_steps_from_config(ctx.config)
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.0))
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps <= 0:
        warmup = "n/a"
    elif step >= warmup_steps:
        warmup = "done"
    else:
        warmup = f"{100.0 * step / max(1, warmup_steps):.1f}%"
    infos = info.get("checkpoint_infos") or []
    previous = None
    latest = None
    for row in infos:
        if int(row.get("step", -1)) <= step and row.get("modified_ts") is not None:
            previous = latest
            latest = row
    eta = None
    speed = None
    if previous and latest and int(latest["step"]) > int(previous["step"]):
        dt = float(latest["modified_ts"]) - float(previous["modified_ts"])
        dstep = int(latest["step"]) - int(previous["step"])
        if dt > 0 and dstep > 0:
            speed = dstep * tokens_per_step / dt
            if max_tokens > tokens and speed > 0:
                eta = (max_tokens - tokens) / speed
    return {
        "step": step,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "tokens": tokens,
        "max_train_tokens": max_tokens,
        "progress": progress,
        "warmup": warmup,
        "eta_seconds": eta,
        "tokens_per_second": speed,
    }


def _active_status(pool: str, active_tau: Any, tokens: int, selection: Dict[str, Any]) -> str:
    value = _safe_float(active_tau)
    if value is None:
        return "n/a"
    shrink_end = int(selection.get("shrink_end_tokens") or 0)
    if tokens >= shrink_end and shrink_end > 0:
        if pool == "rst" and value < 0.06:
            return "RST under-use"
        if pool in ("v", "rst") and value > 0.12:
            return "possible over-use"
        if pool == "qk" and value < 0.03:
            return "QK under-use"
        return "OK"
    if pool == "qk":
        if value < 0.03:
            return "QK too closed"
        if value > 0.07:
            return "QK too open"
    if pool == "v":
        if value < 0.03:
            return "V too closed"
        if value > 0.08:
            return "V early too open"
    if pool == "rst":
        if value < 0.03:
            return "RST too closed"
        if value > 0.07:
            return "RST early too open"
    return "OK"


def _compare_400m(active: Dict[str, Any]) -> Dict[str, str]:
    pools = active.get("pools", {})
    out = {}
    for (pool, metric), (low, high) in REFERENCE_400M.items():
        value = pools.get(pool, {}).get(metric)
        out[f"{pool}_{metric}"] = _status_against_range(value, low, high)
    return out


def _decision_lines(active: Dict[str, Any], selection: Dict[str, Any], progress: Dict[str, Any]) -> List[str]:
    pools = active.get("pools", {})
    tokens = int(progress.get("tokens") or 0)
    shrink_start = int(selection.get("shrink_start_tokens") or 0)
    rst = _safe_float(pools.get("rst", {}).get("active_tau"))
    v = _safe_float(pools.get("v", {}).get("active_tau"))
    qk = _safe_float(pools.get("qk", {}).get("active_tau"))
    if rst is None and v is None and qk is None:
        return ["Active/tau metrics are unavailable; inspect warnings before deciding."]
    lines = []
    if rst is not None and rst < 0.03 and (shrink_start <= 0 or tokens < shrink_start):
        lines.append("RST is currently too closed for early phase, but this may be acceptable before shrink_start.")
        lines.append("Keep current run if rst active_tau starts rising after 3B tokens.")
        lines.append("Reconsider only if rst active_tau stays below 0.03 through several checkpoints.")
        return lines
    statuses = [
        _active_status(pool, pools.get(pool, {}).get("active_tau"), tokens, selection)
        for pool in ("qk", "v", "rst")
    ]
    bad = [status for status in statuses if status not in ("OK", "n/a")]
    if bad:
        lines.append("Current run needs attention: " + "; ".join(bad) + ".")
        lines.append("Compare against the next checkpoint before changing config unless the same status repeats.")
    else:
        lines.append("Keep current run: active dynamics are within current phase guardrails.")
    return lines


def _trend_rows(store: AnalysisStore, current: Dict[str, Any], warnings: List[str]) -> List[Dict[str, Any]]:
    path = store.path("train_analysis.jsonl")
    try:
        rows = read_jsonl(path) if exists(path) else []
    except Exception as exc:
        warnings.append(f"trend read failed: {type(exc).__name__}: {exc}")
        rows = []
    rows.append(current)
    return rows[-5:]


def _scalar_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    active = summary.get("active_dynamics", {})
    pools = active.get("pools", {})
    prune = summary.get("effective_prune", {}).get("eps", [])
    compute_1e4 = next(
        (row.get("compute_frac") for row in prune if math.isclose(float(row.get("eps", -1.0)), 1e-4)),
        None,
    )
    progress = summary.get("progress", {})
    row = {
        "time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "step": progress.get("step"),
        "tokens": progress.get("tokens"),
        "val_loss": active.get("val_loss"),
        "compute_frac_1e_4": compute_1e4,
    }
    for pool in ("qk", "v", "rst"):
        pdata = pools.get(pool, {})
        row[f"{pool}_active_tau"] = pdata.get("active_tau")
        row[f"{pool}_effective"] = pdata.get("effective")
    return row


def _pool_targets_line(label: str, values: Dict[str, Any]) -> str:
    return (
        f"  {label:<20} "
        f"qk={_fmt_num(values.get('qk'), 4)}  "
        f"v={_fmt_num(values.get('v'), 4)}  "
        f"rst={_fmt_num(values.get('rst'), 4)}"
    )


def _format_train_analysis(summary: Dict[str, Any]) -> str:
    line = "=" * 60
    run = summary["run"]
    progress = summary["progress"]
    selection = summary["selection_calibration"]
    active = summary["active_dynamics"]
    prune = summary["effective_prune"]
    compare = summary["reference_400m"]
    trend = summary.get("recent_trend", [])
    warnings = summary.get("warnings", [])
    decision = summary.get("decision", [])
    out = [
        line,
        "DAWN-SRW v4166 TRAIN ANALYSIS",
        line,
        "Run:",
        f"  config          : {run.get('config')}",
        f"  checkpoint_dir  : {run.get('checkpoint_dir')}",
        f"  latest_ckpt     : step {run.get('latest_step')}",
        f"  latest_path     : {run.get('latest_path')}",
        f"  latest_modified : {run.get('latest_modified') or 'n/a'}",
        f"  analyzed_split  : val",
        f"  analysis_batches: {active.get('num_batches', run.get('analysis_batches'))}",
        "",
        "Progress:",
        f"  step            : {progress.get('step')}",
        (
            "  tokens          : "
            f"{_fmt_tokens(progress.get('tokens'))} / {_fmt_tokens(progress.get('max_train_tokens'))}  "
            f"({_fmt_num((progress.get('progress') or 0.0) * 100.0, 1)}%)"
        ),
        f"  warmup          : {progress.get('warmup')}",
        f"  shrink          : {_fmt_num(selection.get('shrink_progress'), 3)} / 1.000",
        (
            "  shrink_window   : "
            f"{_fmt_tokens(selection.get('shrink_start_tokens'))} -> "
            f"{_fmt_tokens(selection.get('shrink_end_tokens'))}"
        ),
        f"  boundary_power  : {_fmt_num(selection.get('soft_gate_boundary_power_now'), 3)}",
        f"  ETA             : {_fmt_duration(progress.get('eta_seconds'))}",
        "",
        "Selection calibration:",
        f"  enabled             : {selection.get('enabled')}",
        _pool_targets_line("active_target", selection.get("active_target", {})),
        _pool_targets_line("candidate_start", selection.get("candidate_start", {})),
        _pool_targets_line("candidate_final", selection.get("candidate_final", {})),
        _pool_targets_line("candidate_now", selection.get("candidate_now", {})),
        "",
        "Active dynamics:",
        "  pool    active_tau   admission   effective   top1       tau_mean   gate_mass   status",
    ]
    pools = active.get("pools", {})
    for pool in ("qk", "v", "rst"):
        pdata = pools.get(pool, {})
        status = pdata.get("status", "n/a")
        out.append(
            "  "
            f"{pool:<6} "
            f"{_fmt_num(pdata.get('active_tau'), 3):<12} "
            f"{_fmt_num(pdata.get('admission'), 3):<11} "
            f"{_fmt_num(pdata.get('effective'), 3):<11} "
            f"{_fmt_num(pdata.get('top1'), 3):<10} "
            f"{_fmt_num(pdata.get('tau_mean'), 3):<10} "
            f"{_fmt_num(pdata.get('gate_mass'), 3):<11} "
            f"{status}"
        )
    out.extend([
        "",
        "Effective prune:",
        "  eps       compute_frac   loss/ce_delta   qk_eff   v_eff    rst_eff",
    ])
    for row in prune.get("eps", []):
        out.append(
            "  "
            f"{row.get('eps'):<9.0e} "
            f"{_fmt_num(row.get('compute_frac'), 4):<14} "
            f"{_fmt_delta(row.get('loss_delta'), 4):<15} "
            f"{_fmt_num(row.get('qk_eff'), 3):<8} "
            f"{_fmt_num(row.get('v_eff'), 3):<8} "
            f"{_fmt_num(row.get('rst_eff'), 3)}"
        )
    out.extend([
        "",
        "400M reference comparison:",
        f"  qk active_tau: {compare.get('qk_active_tau', 'n/a')}",
        f"  v  active_tau: {compare.get('v_active_tau', 'n/a')}",
        f"  rst active_tau: {compare.get('rst_active_tau', 'n/a')}",
        f"  qk effective : {compare.get('qk_effective', 'n/a')}",
        f"  v  effective : {compare.get('v_effective', 'n/a')}",
        f"  rst effective: {compare.get('rst_effective', 'n/a')}",
        "",
        "Recent trend:",
        "  step      tokens    qk_act  v_act   rst_act qk_eff  v_eff   rst_eff compute@1e-4",
    ])
    if trend:
        for row in trend:
            out.append(
                "  "
                f"{str(row.get('step', 'n/a')):<9} "
                f"{_fmt_tokens(row.get('tokens')):<9} "
                f"{_fmt_num(row.get('qk_active_tau'), 3):<7} "
                f"{_fmt_num(row.get('v_active_tau'), 3):<7} "
                f"{_fmt_num(row.get('rst_active_tau'), 3):<7} "
                f"{_fmt_num(row.get('qk_effective'), 3):<7} "
                f"{_fmt_num(row.get('v_effective'), 3):<7} "
                f"{_fmt_num(row.get('rst_effective'), 3):<7} "
                f"{_fmt_num(row.get('compute_frac_1e_4'), 4)}"
            )
    else:
        out.append("  n/a")
    out.extend(["", "Decision:"])
    out.extend([f"  - {line_i}" for line_i in (decision or ["n/a"])])
    out.extend(["", "Warnings:"])
    if warnings:
        out.extend([f"  - {warning}" for warning in warnings])
    else:
        out.append("  - none")
    out.append(line)
    return "\n".join(out)


def _save_train_analysis(store: AnalysisStore, summary: Dict[str, Any],
                         text: str, scalar: Dict[str, Any],
                         warnings: List[str]) -> Tuple[AnalysisStore, str]:
    try:
        write_text_atomic(store.path("train_analysis_latest.txt"), text + "\n")
        append_jsonl(store.path("train_analysis.jsonl"), scalar)
        return store, store.output_dir
    except Exception as exc:
        fallback = os.environ.get("DAWN_TRAIN_ANALYSIS_FALLBACK_OUTPUT", "runs/side_analysis")
        warnings.append(f"save failed at {store.output_dir}: {type(exc).__name__}: {exc}; fallback={fallback}")
        store = AnalysisStore(fallback, is_primary=store.is_primary, analysis_version=store.analysis_version)
        store.ensure_layout()
        summary["warnings"] = warnings
        text = _format_train_analysis(summary)
        write_text_atomic(store.path("train_analysis_latest.txt"), text + "\n")
        append_jsonl(store.path("train_analysis.jsonl"), scalar)
        return store, store.output_dir


def run_train_analysis(args: argparse.Namespace, primary: bool) -> int:
    warnings: List[str] = []
    info = _prepare_train_analysis_args(args, warnings)
    store = _init_train_analysis_store(str(args.output), primary, warnings)
    set_default_store(store)
    if primary:
        print(
            "TRAIN_ANALYSIS START "
            f"config={args.config or 'checkpoint full_config'} "
            f"checkpoint_dir={info.get('configured_checkpoint_dir')} "
            f"output={store.output_dir}",
            flush=True,
        )
    ctx = build_context(args, ["train_analysis"], store)
    max_batches = max(1, int(args.train_analysis_max_batches or DEFAULT_TRAIN_ANALYSIS_BATCHES))
    eps_values = _parse_float_list(args.prune_eps or DEFAULT_TRAIN_ANALYSIS_PRUNE_EPS)
    active = _run_active_dynamics(ctx, max_batches)
    prune = _run_prune_light(ctx, max_batches, eps_values)
    sync_hosts("dawn-v4166-train-analysis-done")
    if not ctx.is_primary:
        return 0

    progress = _progress_status(ctx, info)
    selection = _selection_status(ctx.config, progress["step"], progress["tokens"])
    for pool, pdata in active.get("pools", {}).items():
        pdata["status"] = _active_status(pool, pdata.get("active_tau"), progress["tokens"], selection)
    if selection.get("enabled"):
        active_values = [
            active.get("pools", {}).get(pool, {}).get("active_tau")
            for pool in ("qk", "v", "rst")
        ]
        if all(_safe_float(value) is None for value in active_values):
            warnings.append("selection_calibration is enabled but no active_tau metrics were observed.")
    compare = _compare_400m(active)
    summary = {
        "run": {
            "config": args.config or "checkpoint full_config",
            "checkpoint_dir": info.get("configured_checkpoint_dir") or info.get("run_folder"),
            "latest_step": progress["step"],
            "latest_path": info.get("latest_path") or args.checkpoint,
            "latest_modified": info.get("latest_modified"),
            "analysis_batches": max_batches,
            "output": store.output_dir,
        },
        "progress": progress,
        "selection_calibration": selection,
        "active_dynamics": active,
        "effective_prune": prune,
        "reference_400m": compare,
        "warnings": warnings,
    }
    scalar = _scalar_row(summary)
    summary["recent_trend"] = _trend_rows(store, scalar, warnings)
    summary["decision"] = _decision_lines(active, selection, progress)
    summary["warnings"] = warnings
    text = _format_train_analysis(summary)
    store, saved_dir = _save_train_analysis(store, summary, text, scalar, warnings)
    summary["warnings"] = warnings
    text = _format_train_analysis(summary)
    print(text, flush=True)
    print("", flush=True)
    print(f"Saved: {join_path(saved_dir, 'train_analysis_latest.txt')}", flush=True)
    print(f"Saved: {join_path(saved_dir, 'train_analysis.jsonl')}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.train_analysis:
        maybe_init_distributed(args, True)
        return run_train_analysis(args, is_primary_host())
    if args.from_scratch:
        args.resume = False
    if args.output is None:
        raise ValueError("--output is required unless --train-analysis is used.")
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
