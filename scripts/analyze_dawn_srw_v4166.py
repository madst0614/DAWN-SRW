#!/usr/bin/env python3
"""Resumable DAWN-SRW v4166/v4171 analysis pipeline.

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
    V4166_MODEL_VERSION,
    V417X_MODEL_VERSIONS,
    config_from_checkpoint_or_file,
    count_params,
    create_active_analysis_step,
    create_ce_eval_step,
    create_composition_analysis_step,
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
    verify_analysis_config,
    write_run_metadata,
)
from analysis.dawn_analysis_report import run_report_stage
from analysis.dawn_analysis_storage import (
    AnalysisStore,
    append_text,
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
from analysis.dawn_operator_datasets import (
    DEFAULT_OPERATOR_CACHE_DIR,
    DEFAULT_OPERATOR_DATASET_ROOT,
)
from analysis.dawn_operator_analysis import run_operator_analysis
from analysis.dawn_train_analysis_items import (
    DEFAULT_TRAIN_ANALYSIS_PRESET,
    TRAIN_ANALYSIS_PRESETS,
    TrainAnalysisFormatters,
    emit_train_analysis_item_progress,
    format_train_analysis_items,
    parse_train_analysis_items,
    selected_item_catalog,
    train_analysis_catalog_text,
    train_analysis_required_sections,
)
from analysis.dawn_train_analysis_prompt import (
    build_train_prompt_decision,
    run_train_generation_samples,
    run_train_prompt_trace,
)
from analysis.dawn_v4171_transition import (
    DEFAULT_TRANSITION_PROMPT_SET,
    run_v4171_parity_only_smoke,
    run_v4171_transition_items,
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
DEFAULT_TRAIN_ANALYSIS_PRUNE_EPS = "1e-2,1e-1"
TRAIN_ANALYSIS_POOLS = {
    "qk": "attn_qk",
    "v": "attn_v",
    "rst": "rst",
}
TRAIN_ANALYSIS_QK_SPLIT = {
    "q": "attn_q",
    "k": "attn_k",
}
V4171_COMPOSITION_METRICS = (
    "admission_mass_mean",
    "admission_mass_max",
    "composition_den_mean",
    "composition_den_min",
    "composition_den_max",
    "composition_den_floor_frac",
)
REFERENCE_400M = {
    ("qk", "active_tau"): (0.045, 0.065),
    ("v", "active_tau"): (0.090, 0.120),
    ("rst", "active_tau"): (0.085, 0.110),
    ("qk", "effective"): (0.025, 0.040),
    ("v", "effective"): (0.045, 0.070),
    ("rst", "effective"): (0.045, 0.070),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze DAWN-SRW spatial-r1-v4.1.6.6/v4.1.7.1 checkpoints.")
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
    p.add_argument(
        "--train-analysis-preset",
        default=os.environ.get("DAWN_TRAIN_ANALYSIS_PRESET", DEFAULT_TRAIN_ANALYSIS_PRESET),
        help=f"Train-analysis item preset. Known: {','.join(TRAIN_ANALYSIS_PRESETS)}",
    )
    p.add_argument(
        "--train-analysis-items",
        default=os.environ.get("DAWN_TRAIN_ANALYSIS_ITEMS"),
        help="Comma-separated train-analysis item ids. Overrides --train-analysis-preset. Use all for every item.",
    )
    p.add_argument(
        "--list-train-analysis-items",
        action="store_true",
        help="Print train-analysis item/preset catalog and exit.",
    )
    p.add_argument(
        "--operator-dataset-root",
        default=os.environ.get("DAWN_OPERATOR_DATASET_ROOT", DEFAULT_OPERATOR_DATASET_ROOT),
        help="Operator-analysis dataset root prepared by the DAWN operator dataset workflow.",
    )
    p.add_argument(
        "--operator-analysis-profile", choices=("smoke", "monitor", "full"),
        default=os.environ.get("DAWN_OPERATOR_ANALYSIS_PROFILE", "monitor"))
    p.add_argument(
        "--operator-datasets", default=os.environ.get("DAWN_OPERATOR_DATASETS", "all"),
        help="Comma-separated prepared dataset ids, or all.")
    p.add_argument(
        "--operator-cache-dir",
        default=os.environ.get("DAWN_OPERATOR_CACHE_DIR", DEFAULT_OPERATOR_CACHE_DIR))
    p.add_argument("--operator-behavior-max-examples", type=int, default=None)
    p.add_argument("--operator-trace-max-examples", type=int, default=None)
    p.add_argument("--operator-causal-max-examples", type=int, default=None)
    p.add_argument("--operator-trace-per-group", type=int, default=None)
    p.add_argument("--operator-causal-per-group", type=int, default=None)
    p.add_argument(
        "--operator-analysis-resume", action=argparse.BooleanOptionalAction,
        default=True)
    p.add_argument(
        "--operator-analysis-seed", type=int,
        default=int(os.environ.get("DAWN_OPERATOR_ANALYSIS_SEED", 4171)))
    p.add_argument(
        "--transition-prompt-set",
        default=os.environ.get(
            "DAWN_V4171_TRANSITION_PROMPT_SET", DEFAULT_TRANSITION_PROMPT_SET),
        help="Controlled JSONL prompt pairs for v4171 transition items.",
    )
    p.add_argument(
        "--v4171-parity-only",
        action="store_true",
        help=(
            "Run blocking production-vs-suppression-disabled machine-exact "
            "parity and exit before transition tracing."),
    )
    p.add_argument(
        "--transition-max-prompts",
        type=int,
        default=None,
        help="Optional deterministic cap for v4171 transition prompts.",
    )
    p.add_argument(
        "--transition-topk-qk",
        type=int,
        default=int(os.environ.get("DAWN_V4171_TRANSITION_TOPK_QK", 512)),
        help="Target-only sparse top-k for both Q and K transition traces.",
    )
    p.add_argument(
        "--transition-topk-v",
        type=int,
        default=int(os.environ.get("DAWN_V4171_TRANSITION_TOPK_V", 2048)),
        help="Target-only sparse top-k for V transition traces.",
    )
    p.add_argument(
        "--transition-topk-rst",
        type=int,
        default=int(os.environ.get("DAWN_V4171_TRANSITION_TOPK_RST", 4096)),
        help="Target-only sparse top-k for RST transition traces.",
    )
    p.add_argument(
        "--transition-capture-threshold", type=float, default=0.95,
        help="Minimum captured mass for qualified transition observations.")
    p.add_argument(
        "--transition-adaptive-capture",
        action=argparse.BooleanOptionalAction, default=True,
        help="Retry only low-capture transition rows at cached larger top-k tiers.")
    p.add_argument(
        "--transition-adaptive-final-topk-v", type=int, default=8192,
        help="Maximum adaptive target-only V capture width.")
    p.add_argument(
        "--transition-adaptive-final-topk-rst", type=int, default=8192,
        help="Maximum adaptive target-only RST capture width.")
    p.add_argument(
        "--causal-max-prompts",
        type=int,
        default=int(os.environ.get("DAWN_V4171_CAUSAL_MAX_PROMPTS", 6)),
        help="Maximum controlled prompts used by the causal_intervention item.",
    )
    p.add_argument(
        "--rerouting-max-prompts", type=int, default=None,
        help="Optional deterministic prompt cap for causal_rerouting_trace.")
    p.add_argument(
        "--causal-recovery-neutral-log-band", type=float, default=0.05,
        help="Absolute log-relative-ratio band classified as approximately preserved.")
    p.add_argument(
        "--functional-graph-max-operators-qk", type=int, default=2048,
        help="Maximum deterministic QK candidates in the functional graph.")
    p.add_argument(
        "--functional-graph-max-operators-v", type=int, default=2048,
        help="Maximum deterministic V candidates in the functional graph.")
    p.add_argument(
        "--functional-graph-max-operators-rst", type=int, default=2048,
        help="Maximum deterministic RST candidates in the functional graph.")
    p.add_argument(
        "--functional-graph-neighbor-k", type=int, default=16,
        help="Top neighbors retained per operator and similarity view.")
    p.add_argument(
        "--group-causal-sizes", default="1,2,4,8",
        help="Comma-separated group sizes evaluated in one fixed-width graph.")
    p.add_argument(
        "--group-causal-max-width", type=int, default=8,
        help="Static padded operator-id width for every group intervention.")
    p.add_argument(
        "--group-causal-max-prompts", type=int, default=None,
        help="Optional deterministic prompt cap for group interventions.")
    p.add_argument(
        "--group-random-match-draws", type=int, default=64,
        help="Deterministic random-active draws used for contribution matching.")
    p.add_argument(
        "--group-contribution-match-max-relative-error", type=float, default=0.25,
        help="Maximum accepted local-contribution mass mismatch for matched controls.")
    p.add_argument(
        "--train-analysis-generation-max-prompts",
        type=int,
        default=int(os.environ.get("DAWN_TRAIN_ANALYSIS_GENERATION_MAX_PROMPTS", 3)),
        help="Maximum prompts for the generation_samples train-analysis item.",
    )
    p.add_argument(
        "--train-analysis-generation-max-tokens",
        type=int,
        default=int(os.environ.get("DAWN_TRAIN_ANALYSIS_GENERATION_MAX_TOKENS", 64)),
        help="Maximum new tokens per prompt for the generation_samples train-analysis item.",
    )
    p.add_argument(
        "--train-analysis-generation-temperature",
        type=float,
        default=float(os.environ.get("DAWN_TRAIN_ANALYSIS_GENERATION_TEMPERATURE", 0.8)),
        help="Generation temperature for the generation_samples item. 0 means greedy.",
    )
    p.add_argument(
        "--train-analysis-generation-top-k",
        type=int,
        default=int(os.environ.get("DAWN_TRAIN_ANALYSIS_GENERATION_TOP_K", 50)),
        help="Top-k sampling cutoff for generation_samples when temperature > 0.",
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
    verify_analysis_config(cfg)
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
    steps_per_epoch = int(checkpoint_metadata.get("steps_per_epoch") or 0)
    num_epochs = int(cfg.get("training", {}).get("num_epochs") or 0)
    grad_accum = max(1, int(cfg.get("training", {}).get("gradient_accumulation_steps", 1)))
    planned_steps = (steps_per_epoch // grad_accum) * num_epochs if steps_per_epoch and num_epochs else 0
    total_training_steps = int(planned_steps or checkpoint_metadata.get("global_step") or checkpoint_step or 1)
    model_cfg = model_cfg_from_config(
        cfg,
        checkpoint_step=int(checkpoint_step),
        total_training_steps=total_training_steps,
    )
    n_params = count_params(params)
    info_lines = model.get_model_info() if hasattr(model, "get_model_info") else []
    model_info = {
        "model_info_lines": info_lines,
        "param_count": n_params,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path_resolved": checkpoint_dir,
        "mesh": {"data": int(mesh.shape["data"]), "model": int(mesh.shape["model"])},
        "runtime_gate_state": model_cfg.get("runtime_gate_state"),
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


def _safe_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        value = jax.device_get(value)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        vals = [_safe_float(v) for v in value]
        vals = [v for v in vals if v is not None]
        return vals if vals else None
    try:
        if hasattr(value, "shape") and getattr(value, "shape", ()) not in ((), None):
            flat = value.reshape((-1,)).tolist()
            vals = [_safe_float(v) for v in flat]
            vals = [v for v in vals if v is not None]
            return vals if vals else None
    except Exception:
        return None
    value_f = _safe_float(value)
    return [value_f] if value_f is not None else None


def _safe_metric_value(value: Any) -> Any:
    vals = _safe_float_list(value)
    if vals is not None and len(vals) > 1:
        return vals
    return _safe_float(value)


def _finite(values: Any) -> List[float]:
    vals = _safe_float_list(values) or []
    return [v for v in vals if math.isfinite(v)]


def _quantile(values: Any, q: float) -> Optional[float]:
    vals = sorted(_finite(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _stat_summary(values: Any) -> Dict[str, Any]:
    vals = _finite(values)
    if not vals:
        return {"min": None, "p10": None, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "min": min(vals),
        "p10": _quantile(vals, 0.10),
        "mean": sum(vals) / len(vals),
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "max": max(vals),
    }


def _mean_lists(rows: List[Dict[str, Any]], key: str) -> Optional[List[float]]:
    arrays = [_safe_float_list(row.get(key)) for row in rows]
    arrays = [arr for arr in arrays if arr]
    if not arrays:
        return None
    n = max(len(arr) for arr in arrays)
    out = []
    for idx in range(n):
        vals = [arr[idx] for arr in arrays if idx < len(arr) and math.isfinite(arr[idx])]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _mean_scalar_key(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [_safe_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _quantile_from_summary(stats: Dict[str, Any], q: float) -> Optional[float]:
    points = []
    for q_i, key in ((0.0, "min"), (0.10, "p10"), (0.50, "p50"), (0.90, "p90"), (1.0, "max")):
        value = _safe_float(stats.get(key))
        if value is not None:
            points.append((q_i, value))
    if not points:
        return None
    q = min(1.0, max(0.0, float(q)))
    points = sorted(points)
    if q <= points[0][0]:
        return points[0][1]
    if q >= points[-1][0]:
        return points[-1][1]
    for (q0, v0), (q1, v1) in zip(points, points[1:]):
        if q0 <= q <= q1:
            span = max(q1 - q0, 1e-12)
            frac = (q - q0) / span
            return v0 * (1.0 - frac) + v1 * frac
    return points[-1][1]


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


def _fmt_pct(value: Any, digits: int = 1) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f * 100.0:.{digits}f}%"


def _fmt_eps(value: Any) -> str:
    value_f = _safe_float(value)
    if value_f is None:
        return "n/a"
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1.0:
        return f"{value_f:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value_f:g}"


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
    if _path_name(checkpoint_dir).isdigit():
        checkpoints_dir, step, _ = resolve_checkpoint(checkpoint_dir)
        if checkpoints_dir is None or step is None:
            raise FileNotFoundError(
                f"Could not resolve exact checkpoint {checkpoint_dir}")
        run_folder = _path_parent(checkpoints_dir)
        infos = _checkpoint_infos(checkpoints_dir)
        exact = next(
            (row for row in infos if int(row["step"]) == int(step)), None)
        return {
            "configured_checkpoint_dir": run_folder,
            "run_folder": run_folder,
            "checkpoints_dir": checkpoints_dir,
            "latest_step": int(step),
            "latest_path": (
                exact.get("path") if exact else
                join_path(checkpoints_dir, str(step))),
            "latest_modified": exact.get("modified") if exact else None,
            "checkpoint_infos": infos,
            "selection": "exact_numeric_checkpoint",
        }
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


def _selection_status(cfg: Dict[str, Any], step: int, tokens: int,
                      model_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    boundary_power = _safe_float((model_cfg or {}).get("soft_gate_boundary_power"))

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


def _pool_sizes(cfg: Dict[str, Any]) -> Dict[str, Optional[int]]:
    mcfg = cfg.get("model", {})
    n_rst = mcfg.get("n_rst", mcfg.get("n_know"))
    return {
        "qk": int(mcfg["n_qk"]) if mcfg.get("n_qk") is not None else None,
        "v": int(mcfg["n_v"]) if mcfg.get("n_v") is not None else None,
        "rst": int(n_rst) if n_rst is not None else None,
    }


def _build_per_layer_active(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"layers": [], "summary": {}}
    pool_sizes = _pool_sizes(cfg)
    by_pool: Dict[str, Dict[str, Optional[List[float]]]] = {}
    n_layers = int(cfg.get("model", {}).get("n_layers", 0) or 0)
    for pool, prefix in TRAIN_ANALYSIS_POOLS.items():
        pdata = {
            "active_tau": _mean_lists(rows, f"per_layer_{prefix}_active_tau_frac"),
            "admission": _mean_lists(rows, f"per_layer_{prefix}_admission_active_eps_1e_2_frac"),
            "effective": _mean_lists(rows, f"per_layer_{prefix}_active_eps_1e_2_frac"),
            "top1": _mean_lists(rows, f"per_layer_{prefix}_execution_top1_frac"),
            "top1_max": _mean_lists(rows, f"per_layer_{prefix}_execution_top1_frac_max"),
            "active_ops": _mean_lists(rows, f"per_layer_{prefix}_active_n_mean"),
            "effective_ops": _mean_lists(rows, f"per_layer_{prefix}_gate_eff_n"),
            "gate_eff_ratio": _mean_lists(rows, f"per_layer_{prefix}_gate_eff_ratio"),
        }
        by_pool[pool] = pdata
        for values in pdata.values():
            if values:
                n_layers = max(n_layers, len(values))
        effective = pdata.get("effective") or []
        pool_size = pool_sizes.get(pool)
        stats = _stat_summary(effective)
        dead_layers = sum(
            1 for value in effective
            if _safe_float(value) is not None and float(value) <= 1e-4
        )
        # Closed means materially below its own target. The target is filled later
        # in _add_target_ratio after selection calibration is known.
        out["summary"][pool] = {
            **stats,
            "dead_layers": dead_layers,
            "closed_layers": None,
            "pool_size": pool_size,
        }
    for layer_idx in range(n_layers):
        layer = {"layer": layer_idx}
        for pool, pdata in by_pool.items():
            for metric, values in pdata.items():
                layer[f"{pool}_{metric}"] = values[layer_idx] if values and layer_idx < len(values) else None
        out["layers"].append(layer)
    return out


def _build_select_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pool, prefix in TRAIN_ANALYSIS_POOLS.items():
        score = _mean_lists(rows, f"per_layer_{prefix}_rho_mean")
        tau = _mean_lists(rows, f"per_layer_{prefix}_tau_mean")
        margin = _mean_lists(rows, f"per_layer_{prefix}_selection_margin_mean")
        selected = _mean_lists(rows, f"per_layer_{prefix}_selected_frac")
        out[pool] = {
            "score_layer": _stat_summary(score),
            "tau_layer": _stat_summary(tau),
            "margin_layer": _stat_summary(margin),
            "pos_margin_frac": _quantile(selected, 0.50),
            "score_std_mean": _quantile(_mean_lists(rows, f"per_layer_{prefix}_rho_std"), 0.50),
            "score_max": _quantile(_mean_lists(rows, f"per_layer_{prefix}_rho_max"), 0.90),
        }
    return out


def _build_qk_split(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    pool_size = _pool_sizes(cfg).get("qk")
    by_side: Dict[str, Dict[str, Any]] = {}
    n_layers = int(cfg.get("model", {}).get("n_layers", 0) or 0)
    for side, prefix in TRAIN_ANALYSIS_QK_SPLIT.items():
        active_layer = _mean_lists(rows, f"per_layer_{prefix}_active_tau_frac")
        admission_layer = _mean_lists(rows, f"per_layer_{prefix}_admission_active_eps_1e_2_frac")
        effective_layer = _mean_lists(rows, f"per_layer_{prefix}_active_eps_1e_2_frac")
        active_ops_layer = _mean_lists(rows, f"per_layer_{prefix}_active_n_mean")
        if active_layer:
            n_layers = max(n_layers, len(active_layer))
        if admission_layer:
            n_layers = max(n_layers, len(admission_layer))
        if effective_layer:
            n_layers = max(n_layers, len(effective_layer))
        if active_ops_layer:
            n_layers = max(n_layers, len(active_ops_layer))

        effective_stats = _stat_summary(effective_layer)
        effective_scalar = _mean_scalar_key(rows, f"{prefix}_active_eps_1e_2_frac")
        if effective_scalar is None:
            effective_scalar = effective_stats.get("mean")
        active_ops = _mean_scalar_key(rows, f"{prefix}_active_n_mean")
        effective_ops = effective_scalar * float(pool_size) if effective_scalar is not None and pool_size else None
        by_side[side] = {
            "active_layer": active_layer,
            "admission_layer": admission_layer,
            "effective_layer": effective_layer,
            "active_ops_layer": active_ops_layer,
            "summary": {
                "active_tau": _mean_scalar_key(rows, f"{prefix}_active_tau_frac")
                or _mean_scalar_key(rows, f"{prefix}_active"),
                "admission": _mean_scalar_key(rows, f"{prefix}_admission_active_eps_1e_2_frac"),
                "effective": effective_scalar,
                "active_ops_mean": active_ops,
                "effective_ops_mean": effective_ops,
                "eff_min": effective_stats.get("min"),
                "eff_p10": effective_stats.get("p10"),
                "eff_mean": effective_stats.get("mean"),
                "eff_p90": effective_stats.get("p90"),
                "eff_max": effective_stats.get("max"),
            },
        }

    q_eff = _safe_float(by_side.get("q", {}).get("summary", {}).get("effective"))
    k_eff = _safe_float(by_side.get("k", {}).get("summary", {}).get("effective"))
    balance = q_eff / k_eff if q_eff is not None and k_eff is not None and k_eff > 0.0 else None
    qk_eff_layer = _mean_lists(rows, "per_layer_attn_qk_active_eps_1e_2_frac")
    has_layer_data = any(
        by_side.get(side, {}).get(key)
        for side in ("q", "k")
        for key in ("active_layer", "admission_layer", "effective_layer", "active_ops_layer")
    )
    layers = []
    if has_layer_data:
        for layer_idx in range(n_layers):
            row = {"layer": layer_idx}
            for side in ("q", "k"):
                side_data = by_side.get(side, {})
                for metric, values in (
                    ("active_tau", side_data.get("active_layer")),
                    ("admission", side_data.get("admission_layer")),
                    ("effective", side_data.get("effective_layer")),
                    ("active_ops", side_data.get("active_ops_layer")),
                ):
                    row[f"{side}_{metric}"] = values[layer_idx] if values and layer_idx < len(values) else None
                effective = _safe_float(row.get(f"{side}_effective"))
                row[f"{side}_effective_ops"] = (
                    effective * float(pool_size) if effective is not None and pool_size else None
                )
            row["qk_effective"] = (
                qk_eff_layer[layer_idx] if qk_eff_layer and layer_idx < len(qk_eff_layer) else None
            )
            layers.append(row)
    return {
        "summary": {
            "q": by_side.get("q", {}).get("summary", {}),
            "k": by_side.get("k", {}).get("summary", {}),
            "qk_effective_balance": balance,
            "pool_size": pool_size,
        },
        "layers": layers,
    }


def _build_concentration_max(active: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = active.get("per_layer_active", {}).get("layers", [])
    pools = active.get("pools", {})
    out = []
    for pool in ("qk", "v", "rst"):
        best = None
        best_top1_max = None
        for row in layers:
            top1_max = _safe_float(row.get(f"{pool}_top1_max"))
            if top1_max is None:
                top1_max = _safe_float(row.get(f"{pool}_top1"))
            if top1_max is None:
                continue
            if best_top1_max is None or top1_max > best_top1_max:
                best = row
                best_top1_max = top1_max
        if best is None:
            out.append({
                "pool": pool,
                "layer": None,
                "top1_mean": None,
                "layer_top1_max": None,
                "global_top1_max": pools.get(pool, {}).get("top1_max"),
                "active": None,
                "effective": None,
                "active_ops": None,
                "effective_ops": None,
                "operator_id": None,
            })
            continue
        out.append({
            "pool": pool,
            "layer": best.get("layer"),
            "top1_mean": best.get(f"{pool}_top1"),
            "layer_top1_max": best_top1_max,
            "global_top1_max": pools.get(pool, {}).get("top1_max"),
            "active": best.get(f"{pool}_active_tau"),
            "effective": best.get(f"{pool}_effective"),
            "active_ops": best.get(f"{pool}_active_ops"),
            "effective_ops": best.get(f"{pool}_effective_ops"),
            "operator_id": None,
        })
    return out


def _build_num_health(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = (
        "residual_norm",
        "residual_norm_max",
        "q_norm",
        "k_norm",
        "v_norm",
        "attn_logit_mean",
        "attn_logit_std",
        "attn_logit_max",
        "attn_softmax_top1_mean",
        "attn_softmax_entropy_mean",
        "attn_o_output_norm_mean",
        "attn_o_output_norm_max",
    )
    health = {}
    for key in scalar_keys:
        vals = [row.get(key) for row in rows]
        health[key] = _stat_summary(vals)
    no_nan = True
    for row in rows:
        for value in row.values():
            if isinstance(value, (list, tuple)):
                vals = _safe_float_list(value) or []
                no_nan = no_nan and all(math.isfinite(v) for v in vals)
            else:
                value_f = _safe_float(value)
                if value_f is not None:
                    no_nan = no_nan and math.isfinite(value_f)
    health["no_nan"] = no_nan
    return health


def _run_active_dynamics(
    ctx: AnalysisContext,
    max_batches: int,
    *,
    include_composition: bool = False,
) -> Dict[str, Any]:
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
    composition_step_fn = None
    if (
        include_composition
        and ctx.config.get("model", {}).get("model_version")
        in V417X_MODEL_VERSIONS
    ):
        composition_step_fn = create_composition_analysis_step(
            ctx.model, ctx.sharded_fns, cfg=ctx.config)
    wanted = {"loss", "correct", "valid_count", "aux_loss"}
    for prefix in TRAIN_ANALYSIS_POOLS.values():
        wanted.update({
            f"{prefix}_active_tau_frac",
            f"{prefix}_admission_active_eps_1e_2_frac",
            f"{prefix}_active_eps_1e_2_frac",
            f"{prefix}_mass_eps_1e_2",
            f"{prefix}_execution_mass_sum",
            f"{prefix}_execution_top1_frac",
            f"{prefix}_execution_top1_frac_max",
            f"{prefix}_top1_gate_frac",
            f"{prefix}_top1_gate_frac_max",
            f"{prefix}_tau_mean",
            f"{prefix}_tau_min",
            f"{prefix}_tau_max",
            f"{prefix}_rho_mean",
            f"{prefix}_score_std",
            f"{prefix}_rho_std",
            f"{prefix}_rho_max",
            f"{prefix}_active_n_mean",
            f"{prefix}_gate_eff_n",
            f"{prefix}_gate_eff_ratio",
        })
        for metric in (
            "active_tau_frac",
            "admission_active_eps_1e_2_frac",
            "active_eps_1e_2_frac",
            "mass_eps_1e_2",
            "margin_band_pos",
            "active_n_mean",
            "gate_eff_n",
            "gate_eff_ratio",
            "execution_top1_frac",
            "execution_top1_frac_max",
            "rho_mean",
            "rho_std",
            "rho_max",
            "tau_mean",
            "tau_min",
            "tau_max",
            "selection_margin_mean",
            "positive_margin_mean",
            "selected_frac",
        ):
            wanted.add(f"per_layer_{prefix}_{metric}")
        for metric in V4171_COMPOSITION_METRICS:
            wanted.add(f"{prefix}_{metric}")
    wanted.add("admission_den_power")
    for prefix in TRAIN_ANALYSIS_QK_SPLIT.values():
        wanted.update({
            f"{prefix}_active",
            f"{prefix}_active_tau_frac",
            f"{prefix}_admission_active_eps_1e_2_frac",
            f"{prefix}_active_eps_1e_2_frac",
            f"{prefix}_active_n_mean",
        })
        for metric in (
            "active_tau_frac",
            "admission_active_eps_1e_2_frac",
            "active_eps_1e_2_frac",
            "active_n_mean",
        ):
            wanted.add(f"per_layer_{prefix}_{metric}")
    wanted.update({
        "residual_norm",
        "residual_norm_max",
        "q_norm",
        "k_norm",
        "v_norm",
        "attn_logit_mean",
        "attn_logit_std",
        "attn_logit_max",
        "attn_softmax_top1_mean",
        "attn_softmax_entropy_mean",
        "attn_o_output_norm_mean",
        "attn_o_output_norm_max",
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
        if composition_step_fn is not None:
            result = dict(result)
            result.update(composition_step_fn(ctx.params, mesh_ids, mesh_mask))
        selected = {key: value for key, value in result.items() if key in wanted}
        jax.block_until_ready(jax.tree.leaves(selected))
        selected = jax.device_get(selected)
        row = {key: _safe_metric_value(value) for key, value in selected.items()}
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
    pool_sizes = _pool_sizes(ctx.config)
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
            "top1_max": mean_key(f"{prefix}_execution_top1_frac_max"),
            "tau_mean": mean_key(f"{prefix}_tau_mean"),
            "tau_min": mean_key(f"{prefix}_tau_min"),
            "tau_max": mean_key(f"{prefix}_tau_max"),
            "gate_mass": gate_mass,
            "score_mean": mean_key(f"{prefix}_rho_mean"),
            "score_std": score_std,
            "score_p95": None,
            "pool_size": pool_sizes.get(pool),
            "active_ops_mean": mean_key(f"{prefix}_active_n_mean"),
            "effective_ops_mean": mean_key(f"{prefix}_gate_eff_n"),
            "effective_ops_ratio": mean_key(f"{prefix}_gate_eff_ratio"),
        }
    composition = {
        "admission_den_power": mean_key("admission_den_power"),
        "pools": {
            pool: {
                metric: mean_key(f"{prefix}_{metric}")
                for metric in V4171_COMPOSITION_METRICS
            }
            for pool, prefix in TRAIN_ANALYSIS_POOLS.items()
        },
    }
    composition["available"] = any(
        value is not None
        for pool in composition["pools"].values()
        for value in pool.values()
    )
    per_layer_active = _build_per_layer_active(rows, ctx.config)
    active = {
        "num_batches": len(rows),
        "valid_tokens": valid_total,
        "val_loss": loss_sum / valid_total if valid_total else None,
        "accuracy": correct_total / valid_total if valid_total else None,
        "sec": sum(float(row.get("sec") or 0.0) for row in rows),
        "pools": pools,
        "per_layer_active": per_layer_active,
        "select_distribution": _build_select_distribution(rows),
        "qk_split": _build_qk_split(rows, ctx.config),
        "num_health": _build_num_health(rows),
        "composition_health": composition,
    }
    active["concentration_max"] = _build_concentration_max(active)
    return active


def _run_prune_light(ctx: AnalysisContext, max_batches: int, eps_values: List[float]) -> Dict[str, Any]:
    batch_size = int(ctx.config["training"].get("batch_size", 1))
    seq_len = int(ctx.config["model"].get("max_seq_len", 512))
    max_tokens = batch_size * seq_len * int(max_batches)
    eps_with_base = [0.0] + [eps for eps in eps_values if float(eps) != 0.0]
    summaries = []
    base_loss = None
    baseline = None
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
            return_pool_prune_stats=True,
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
        pool_sizes = _pool_sizes(ctx.config)
        pool_compute_sum = {pool: 0.0 for pool in TRAIN_ANALYSIS_POOLS}
        pool_eff_sum = {pool: 0.0 for pool in TRAIN_ANALYSIS_POOLS}
        pool_eff_ops_sum = {pool: 0.0 for pool in TRAIN_ANALYSIS_POOLS}
        pool_stats_count = 0
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
                *pool_ret,
            ) = ret
            if len(pool_ret) >= 9:
                qk_active_n, v_active_n, rst_active_n = pool_ret[0:3]
                qk_eff_frac, v_eff_frac, rst_eff_frac = pool_ret[3:6]
                qk_eff_n, v_eff_n, rst_eff_n = pool_ret[6:9]
                for pool, active_n, eff_frac, eff_n in (
                    ("qk", qk_active_n, qk_eff_frac, qk_eff_n),
                    ("v", v_active_n, v_eff_frac, v_eff_n),
                    ("rst", rst_active_n, rst_eff_frac, rst_eff_n),
                ):
                    size = pool_sizes.get(pool)
                    active_n_f = _safe_float(active_n)
                    if active_n_f is not None and size:
                        pool_compute_sum[pool] += active_n_f / float(size)
                    eff_frac_f = _safe_float(eff_frac)
                    eff_n_f = _safe_float(eff_n)
                    if eff_n_f is not None and size:
                        derived_eff_frac = eff_n_f / float(size)
                        if eff_frac_f is None or (abs(eff_frac_f) < 1e-12 and derived_eff_frac > 0.0):
                            eff_frac_f = derived_eff_frac
                    if eff_frac_f is not None:
                        pool_eff_sum[pool] += eff_frac_f
                    if eff_n_f is not None:
                        pool_eff_ops_sum[pool] += eff_n_f
                pool_stats_count += 1
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
            sec = time.time() - t0
            sec_sum += sec
            if ctx.is_primary:
                log_every = max(1, int(getattr(ctx.args, "log_every_batches", 1) or 1))
                if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == local_max_batches:
                    print(
                        "TRAIN_ANALYSIS PRUNE "
                        f"eps={eps:g} batch={batch_idx + 1:05d}/{local_max_batches:05d} "
                        f"loss={_fmt_num(loss_f, 6)} "
                        f"compute={_fmt_num(float(compute_frac), 4)} "
                        f"sec={sec:.2f}",
                        flush=True,
                    )
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
        if pool_stats_count:
            for pool in TRAIN_ANALYSIS_POOLS:
                summary[f"{pool}_compute"] = pool_compute_sum[pool] / pool_stats_count
                summary[f"{pool}_eff"] = pool_eff_sum[pool] / pool_stats_count
                summary[f"{pool}_eff_ops"] = pool_eff_ops_sum[pool] / pool_stats_count
                summary[f"{pool}_pool_size"] = pool_sizes.get(pool)
        if float(eps) == 0.0:
            base_loss = summary["val_loss"]
            baseline = summary
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
        "baseline": baseline,
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


def _add_target_ratio(active: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]:
    pools = active.get("pools", {})
    targets = selection.get("active_target", {}) or {}
    rows = []
    per_layer_summary = active.get("per_layer_active", {}).get("summary", {})
    for pool in ("qk", "v", "rst"):
        pdata = pools.get(pool, {})
        target = _safe_float(targets.get(pool))
        effective = _safe_float(pdata.get("effective"))
        ratio = None
        if target is not None and target > 0.0 and effective is not None:
            ratio = effective / target
        status = "n/a"
        if ratio is not None:
            if ratio < 0.50:
                status = "TOO_CLOSED"
            elif ratio < 0.80:
                status = "OK_LOW"
            elif ratio > 2.50:
                status = "OK_HIGH"
            else:
                status = "OK"
        pdata["target"] = target
        pdata["eff_target_ratio"] = ratio
        pdata["target_ratio_status"] = status
        layer_info = per_layer_summary.get(pool)
        if layer_info is not None:
            layer_effective = [
                row.get(f"{pool}_effective")
                for row in active.get("per_layer_active", {}).get("layers", [])
            ]
            if target is not None:
                layer_info["closed_layers"] = sum(
                    1 for value in layer_effective
                    if _safe_float(value) is not None and float(value) < target * 0.50
                )
        rows.append({
            "pool": pool,
            "target": target,
            "active_tau": pdata.get("active_tau"),
            "admission": pdata.get("admission"),
            "effective": effective,
            "eff_target_ratio": ratio,
            "status": status,
        })
    active["target_ratio"] = rows
    return active


def _build_target_quantile_gap(active: Dict[str, Any], selection: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = selection.get("active_target", {}) or {}
    candidates = selection.get("candidate_now", {}) or {}
    select_distribution = active.get("select_distribution", {}) or {}
    rows = []
    for pool in ("qk", "v", "rst"):
        dist = select_distribution.get(pool, {})
        score_summary = dist.get("score_layer", {}) or {}
        tau_summary = dist.get("tau_layer", {}) or {}
        target = _safe_float(targets.get(pool))
        candidate = _safe_float(candidates.get(pool))
        target_q = 1.0 - target if target is not None else None
        candidate_q = 1.0 - candidate if candidate is not None else None
        score_q_target = _quantile_from_summary(score_summary, target_q) if target_q is not None else None
        score_q_candidate = (
            _quantile_from_summary(score_summary, candidate_q) if candidate_q is not None else None
        )
        tau = _safe_float(tau_summary.get("p50"))
        if tau is None:
            tau = _safe_float(active.get("pools", {}).get(pool, {}).get("tau_mean"))
        rows.append({
            "pool": pool,
            "target": target,
            "candidate": candidate,
            "target_quantile": target_q,
            "candidate_quantile": candidate_q,
            "score_q_target": score_q_target,
            "score_q_candidate": score_q_candidate,
            "tau": tau,
            "gap_target": tau - score_q_target if tau is not None and score_q_target is not None else None,
            "gap_candidate": tau - score_q_candidate if tau is not None and score_q_candidate is not None else None,
            "source": "layer_score_quantile_approx",
        })
    return rows


def _build_calibration_state(active: Dict[str, Any], selection: Dict[str, Any],
                             cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    scfg = tcfg.get("selection_calibration", {})
    scfg = scfg if isinstance(scfg, dict) else {}
    enabled = bool(selection.get("enabled", scfg.get("enabled", False)))
    tau_lr_mult = _safe_float(tcfg.get("tau_lr_mult"))
    direct_tau_lr_mult = _safe_float(tcfg.get("direct_tau_lr_mult"))
    if not enabled:
        mode = "disabled"
    elif (tau_lr_mult is not None and tau_lr_mult == 0.0) and (
        direct_tau_lr_mult is None or direct_tau_lr_mult == 0.0
    ):
        mode = "enabled/static_tau_lr0"
    else:
        mode = "enabled/tau_trainable"
    rows = []
    targets = selection.get("active_target", {}) or {}
    candidates = selection.get("candidate_now", {}) or {}
    for pool in ("qk", "v", "rst"):
        pdata = active.get("pools", {}).get(pool, {})
        observed = _safe_float(pdata.get("admission"))
        candidate = _safe_float(candidates.get(pool))
        error = observed - candidate if observed is not None and candidate is not None else None
        rows.append({
            "pool": pool,
            "target": _safe_float(targets.get(pool)),
            "candidate": candidate,
            "observed_admission": observed,
            "error": error,
            "tau_before": None,
            "tau_after": pdata.get("tau_mean"),
            "tau_delta": None,
            "clamp_hit": "n/a",
            "stopgrad": "n/a",
            "tau_lr_mult": tau_lr_mult,
            "direct_tau_lr_mult": direct_tau_lr_mult,
            "mode": mode,
        })
    return rows


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
        reasons = []
        for pool in ("qk", "v", "rst"):
            pdata = pools.get(pool, {})
            ratio = _safe_float(pdata.get("eff_target_ratio"))
            admission = _safe_float(pdata.get("admission"))
            status = pdata.get("status")
            if status not in ("OK", "n/a", None):
                reasons.append(
                    f"{pool} {status}: eff/target={_fmt_num(ratio, 2)}, "
                    f"admission={_fmt_num(admission, 3)}"
                )
        lines.append("Current run needs attention: " + "; ".join(bad) + ".")
        if reasons:
            lines.append("Reason: " + "; ".join(reasons) + ".")
        lines.append("Compare against the next checkpoint before changing config unless the same status repeats.")
    else:
        lines.append("Keep current run: active dynamics are within current phase guardrails.")
    top1_max = max(
        (_safe_float(pools.get(pool, {}).get("top1_max")) or 0.0)
        for pool in ("qk", "v", "rst")
    )
    if top1_max < 0.10:
        lines.append(f"No obvious collapse: top1_max={_fmt_num(top1_max, 3)} < 0.100.")
    return lines


def _operator_family_decision(
        analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    causal = analysis.get("causal_intervention") or {}
    parity = causal.get("zero_suppression_parity") or {}
    canonical_zero_valid = bool(parity.get("machine_exact"))

    ranking = analysis.get("causal_ranking_calibration") or {}
    ranking_judgments = ranking.get("judgments") or {}
    local_ranking = ranking_judgments.get("local_ranking_valid") or {}
    local_evidence = local_ranking.get("evidence") or {}

    recovery = analysis.get("causal_recovery_trace") or {}
    compensation = recovery.get("downstream_compensation_dominant") or {}
    compensation_evidence = compensation.get("evidence") or {}

    rerouting = analysis.get("causal_rerouting_trace") or {}
    path_dependence = rerouting.get("path_dependence_supported") or {}
    path_evidence = path_dependence.get("evidence") or {}

    groups = analysis.get("group_causal_intervention") or {}
    redundancy = groups.get("functional_redundancy_supported") or {}
    redundancy_evidence = redundancy.get("evidence") or {}
    group_zero_parity = groups.get("zero_size_group_parity") or {}
    canonical_valid = bool(
        canonical_zero_valid and group_zero_parity.get("machine_exact"))

    graph = analysis.get("operator_functional_graph") or {}
    graph_pools = graph.get("pools") or {}
    address_function = {
        pool: _safe_float(row.get("address_function_spearman"))
        for pool, row in graph_pools.items()
    }
    address_function = {
        pool: value for pool, value in address_function.items()
        if value is not None
    }
    mean_abs_alignment = (
        sum(abs(value) for value in address_function.values())
        / len(address_function) if address_function else None)
    if mean_abs_alignment is None:
        alignment_strength = "unavailable"
    elif mean_abs_alignment < 0.20:
        alignment_strength = "weak"
    elif mean_abs_alignment < 0.50:
        alignment_strength = "moderate"
    else:
        alignment_strength = "strong"
    percolated_pools = {
        pool: bool(row.get("percolated")) for pool, row in graph_pools.items()
    }

    capture_by_item: Dict[str, Any] = {}
    for item in (
            "trajectory_trace", "operator_functional_graph",
            "causal_rerouting_trace"):
        capture = (analysis.get(item) or {}).get("capture_reliability") or {}
        if capture:
            capture_by_item[item] = {
                "total_observations": capture.get("total_observations", 0),
                "qualified_observations": capture.get(
                    "qualified_observations", 0),
                "excluded_observations": capture.get(
                    "excluded_observations", 0),
                "remaining_low_capture_count": capture.get(
                    "remaining_low_capture_count", 0),
                "pools": capture.get("pools", {}),
            }
    remaining_low_capture_by_item = {
        item: int(row.get("remaining_low_capture_count", 0) or 0)
        for item, row in capture_by_item.items()
    }
    capture_supported = bool(capture_by_item) and not any(
        remaining_low_capture_by_item.values())

    judgments = {
        "canonical_causal_path_valid": {
            "supported": canonical_valid,
            "evidence": {
                "canonical_zero_suppression": parity,
                "zero_size_group_suppression": group_zero_parity,
            },
        },
        "local_operator_ranking_valid": {
            "supported": bool(local_ranking.get("supported")),
            "evidence": local_evidence,
        },
        "downstream_compensation_dominant": compensation,
        "path_dependence_supported": path_dependence,
        "functional_redundancy_supported": redundancy,
        "address_function_alignment": {
            "strength": alignment_strength,
            "mean_abs_spearman": mean_abs_alignment,
            "spearman_by_pool": address_function,
            "strength_thresholds": {"weak_lt": 0.20, "moderate_lt": 0.50},
        },
        "functional_graph_percolated": {
            "percolated": any(percolated_pools.values()),
            "pools": percolated_pools,
            "largest_component_fraction_by_pool": {
                pool: row.get("largest_component_fraction")
                for pool, row in graph_pools.items()
            },
        },
        "capture_reliability": {
            "supported": capture_supported,
            "remaining_low_capture_count_by_item":
                remaining_low_capture_by_item,
            "items": capture_by_item,
        },
    }

    local_ci = local_evidence.get("bootstrap_ci95") or [None, None]
    paired = redundancy_evidence
    lines = [
        (
            "Canonical suppression path "
            f"{'passed' if canonical_valid else 'failed'} machine-exact zero parity "
            f"(CE abs diff={_fmt_num(parity.get('ce_abs_diff'), 8)}, "
            f"max logit abs diff={_fmt_num(parity.get('max_logit_abs_diff'), 8)}, "
            f"final residual max abs diff={_fmt_num(parity.get('final_residual_max_abs_diff'), 8)}, "
            f"zero-size group exact={group_zero_parity.get('machine_exact')}, "
            f"group comparisons={group_zero_parity.get('num_comparisons', 0)})."
        ),
        (
            "Local contribution ranking "
            f"{'predicts' if local_ranking.get('supported') else 'does not establish'} "
            "immediate causal importance "
            f"(Spearman={_fmt_num(local_evidence.get('spearman'), 3)}, "
            f"CI95={local_ci}, n={local_evidence.get('n', 0)})."
        ),
        (
            "Relative recovery metrics "
            f"{'support' if compensation.get('supported') else 'do not support'} "
            "dominant downstream compensation "
            f"(median ratio={_fmt_num(compensation_evidence.get('median_relative_delta_ratio'), 4)}, "
            f"recovery={_fmt_pct(compensation_evidence.get('relative_recovery_fraction'), 2)}, "
            f"amplification={_fmt_pct(compensation_evidence.get('relative_amplification_fraction'), 2)})."
        ),
        (
            "Rerouting traces "
            f"{'support' if path_dependence.get('supported') else 'do not support'} "
            "path-dependent downstream computation "
            f"(routing predicts final={path_evidence.get('routing_divergence_predicts_final_relative_effect')}, "
            f"important reconvergence fraction={_fmt_pct(path_evidence.get('important_reconvergence_fraction'), 2)})."
        ),
        (
            "Reciprocal functional neighborhoods "
            f"{'outperform' if redundancy.get('supported') else 'do not establish an advantage over'} "
            "contribution-matched random controls "
            f"(paired n={paired.get('paired_n', 0)}, "
            f"mean difference={_fmt_delta(paired.get('paired_mean_difference'), 5)}, "
            f"CI95={paired.get('bootstrap_ci95')})."
        ),
        (
            "Learned address and RW functional geometry are "
            f"{alignment_strength}ly aligned "
            f"(mean |Spearman|={_fmt_num(mean_abs_alignment, 3)}, by pool={address_function})."
            if alignment_strength in ("weak", "moderate", "strong") else
            "Learned address and RW functional geometry alignment is unavailable."
        ),
        (
            "Connected-component family interpretation is "
            f"{'disabled because the graph percolated' if any(percolated_pools.values()) else 'kept diagnostic-only'} "
            f"(percolated by pool={percolated_pools})."
        ),
        (
            "Sparse capture reliability "
            f"{'passed' if capture_supported else 'is partial'} "
            f"(remaining low-capture observations by item="
            f"{remaining_low_capture_by_item})."
        ),
    ]
    return judgments, lines


def _append_v4171_analysis_warnings(
        warnings: List[str], analysis: Dict[str, Any],
        selected_items: List[str]) -> None:
    for item in selected_items:
        data = analysis.get(item) or {}
        status = str(data.get("status", "not_requested"))
        capture_warnings = data.get("capture_warnings_by_pool") or {}
        if capture_warnings:
            counts = ", ".join(
                f"{pool}={count}" for pool, count in sorted(
                    capture_warnings.items()))
            warnings.append(
                f"{item} low-capture observations by pool: {counts}; "
                "affected path/profile conclusions remain partial.")
        if status in ("partial", "failed"):
            warnings.append(
                f"{item} status={status}: "
                f"{data.get('reason') or data.get('warning') or data.get('error') or 'inspect item limitations and artifacts'}")
    graph = analysis.get("operator_functional_graph") or {}
    for pool, row in (graph.get("pools") or {}).items():
        if row.get("percolated"):
            warnings.append(
                f"operator_functional_graph {pool} connected components are "
                "percolated and are not interpreted as functional families "
                f"(largest fraction={_fmt_num(row.get('largest_component_fraction'), 4)}).")
    group_warning = (analysis.get("group_causal_intervention") or {}).get(
        "warning")
    if group_warning:
        warnings.append(f"group_causal_intervention: {group_warning}")
    warnings[:] = list(dict.fromkeys(warnings))


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
    prune_summary = summary.get("effective_prune", {})
    prune = prune_summary.get("eps", [])
    baseline = prune_summary.get("baseline", {})
    compute_1e4 = next(
        (row.get("compute_frac") for row in prune if math.isclose(float(row.get("eps", -1.0)), 1e-4)),
        None,
    )
    best_prune = None
    for prune_row in prune:
        if _safe_float(prune_row.get("compute_frac")) is None:
            continue
        if best_prune is None or float(prune_row.get("compute_frac")) < float(best_prune.get("compute_frac")):
            best_prune = prune_row
    progress = summary.get("progress", {})
    row = {
        "time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "step": progress.get("step"),
        "tokens": progress.get("tokens"),
        "val_loss": active.get("val_loss"),
        "compute_frac_1e_4": compute_1e4,
        "pruned_eval_base_loss": baseline.get("val_loss", prune_summary.get("base_loss")),
        "pruned_eval_base_compute_frac": baseline.get("compute_frac"),
        "pruned_eval_best_eps": best_prune.get("eps") if best_prune else None,
        "pruned_eval_best_compute_frac": best_prune.get("compute_frac") if best_prune else None,
        "pruned_eval_best_loss_delta": best_prune.get("loss_delta") if best_prune else None,
    }
    for pool in ("qk", "v", "rst"):
        pdata = pools.get(pool, {})
        row[f"{pool}_active_tau"] = pdata.get("active_tau")
        row[f"{pool}_effective"] = pdata.get("effective")
        row[f"{pool}_eff_target_ratio"] = pdata.get("eff_target_ratio")
        row[f"{pool}_closed_layers"] = (
            active.get("per_layer_active", {})
            .get("summary", {})
            .get(pool, {})
            .get("closed_layers")
        )
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
    compare = summary["reference_400m"]
    trend = summary.get("recent_trend", [])
    warnings = summary.get("warnings", [])
    decision = summary.get("decision", [])
    items = list(summary.get("analysis_items") or TRAIN_ANALYSIS_PRESETS["full"])
    item_formatters = TrainAnalysisFormatters(
        num=_fmt_num,
        delta=_fmt_delta,
        pct=_fmt_pct,
        eps=_fmt_eps,
        safe_float=_safe_float,
    )
    out = [
        line,
        f"DAWN-SRW {run.get('model_version', 'unknown')} TRAIN ANALYSIS",
        line,
        "Run:",
        f"  model_version   : {run.get('model_version')}",
        f"  config          : {run.get('config')}",
        f"  checkpoint_dir  : {run.get('checkpoint_dir')}",
        f"  latest_ckpt     : step {run.get('latest_step')}",
        f"  latest_path     : {run.get('latest_path')}",
        f"  latest_modified : {run.get('latest_modified') or 'n/a'}",
        f"  analyzed_split  : val",
        f"  analysis_batches: {run.get('analysis_batches') if active.get('status') == 'not_requested' else active.get('num_batches', run.get('analysis_batches'))}",
        f"  analysis_preset : {summary.get('analysis_preset')}",
        f"  analysis_items  : {','.join(summary.get('analysis_items') or [])}",
        f"  required_sections: {','.join(summary.get('analysis_required_sections') or [])}",
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
    ]
    if active.get("status") == "not_requested":
        out.extend([
            "  status          : not_requested",
            f"  reason          : {active.get('reason', 'not requested by preset')}",
        ])
    else:
        out.append(
            "  pool    active_tau   admission   effective   top1       tau_mean   gate_mass   status")
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
    out.extend(format_train_analysis_items(summary, items, item_formatters))
    out.extend([
        "",
        f"{compare.get('label', '400M')} reference comparison:",
        f"  status        : {compare.get('status', 'ready')}",
        f"  reason        : {compare.get('reason', 'n/a')}",
        f"  qk active_tau: {compare.get('qk_active_tau', 'n/a')}",
        f"  v  active_tau: {compare.get('v_active_tau', 'n/a')}",
        f"  rst active_tau: {compare.get('rst_active_tau', 'n/a')}",
        f"  qk effective : {compare.get('qk_effective', 'n/a')}",
        f"  v  effective : {compare.get('v_effective', 'n/a')}",
        f"  rst effective: {compare.get('rst_effective', 'n/a')}",
        "",
        "Recent trend:",
        "  step      tokens    qk_act  v_act   rst_act qk_eff  v_eff   rst_eff best_eps best_compute best_dce",
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
                f"{_fmt_eps(row.get('pruned_eval_best_eps')):<8} "
                f"{_fmt_num(row.get('pruned_eval_best_compute_frac'), 4):<12} "
                f"{_fmt_delta(row.get('pruned_eval_best_loss_delta'), 4)}"
            )
    else:
        out.append("  n/a")
    if "decision_reason" not in items:
        out.extend(["", "Decision:"])
        out.extend([f"  - {line_i}" for line_i in (decision or ["n/a"])])
    out.extend(["", "Warnings:"])
    if warnings:
        out.extend([f"  - {warning}" for warning in warnings])
    else:
        out.append("  - none")
    out.append(line)
    return "\n".join(out)


def _train_analysis_history_entry(summary: Dict[str, Any], text: str) -> str:
    run = summary.get("run", {})
    progress = summary.get("progress", {})
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    header = [
        "",
        "#" * 60,
        (
            "TRAIN_ANALYSIS_HISTORY "
            f"appended_at={timestamp} "
            f"step={progress.get('step')} "
            f"preset={summary.get('analysis_preset')} "
            f"latest_path={run.get('latest_path')}"
        ),
        "#" * 60,
    ]
    return "\n".join(header) + "\n" + text.rstrip() + "\n"


def _append_train_analysis_history(store: AnalysisStore, summary: Dict[str, Any],
                                   text: str, warnings: List[str]) -> None:
    try:
        append_text(
            store.path("train_analysis_history.txt"),
            _train_analysis_history_entry(summary, text),
        )
    except Exception as exc:
        warnings.append(f"history append failed: {type(exc).__name__}: {exc}")


def _save_train_analysis(store: AnalysisStore, summary: Dict[str, Any],
                         text: str, scalar: Dict[str, Any],
                         warnings: List[str]) -> Tuple[AnalysisStore, str]:
    try:
        write_text_atomic(store.path("train_analysis_latest.txt"), text + "\n")
        write_json_atomic(store.path("train_analysis_latest.json"), summary)
        append_jsonl(store.path("train_analysis.jsonl"), scalar)
        _append_train_analysis_history(store, summary, text, warnings)
        return store, store.output_dir
    except Exception as exc:
        fallback = os.environ.get("DAWN_TRAIN_ANALYSIS_FALLBACK_OUTPUT", "runs/side_analysis")
        warnings.append(f"save failed at {store.output_dir}: {type(exc).__name__}: {exc}; fallback={fallback}")
        store = AnalysisStore(fallback, is_primary=store.is_primary, analysis_version=store.analysis_version)
        store.ensure_layout()
        summary["warnings"] = warnings
        text = _format_train_analysis(summary)
        write_text_atomic(store.path("train_analysis_latest.txt"), text + "\n")
        write_json_atomic(store.path("train_analysis_latest.json"), summary)
        append_jsonl(store.path("train_analysis.jsonl"), scalar)
        _append_train_analysis_history(store, summary, text, warnings)
        return store, store.output_dir


def run_train_analysis(args: argparse.Namespace, primary: bool) -> int:
    warnings: List[str] = []
    analysis_items = parse_train_analysis_items(args.train_analysis_preset, args.train_analysis_items)
    required_sections = train_analysis_required_sections(analysis_items)
    info = _prepare_train_analysis_args(args, warnings)
    store = _init_train_analysis_store(str(args.output), primary, warnings)
    set_default_store(store)
    if primary:
        print(
            "TRAIN_ANALYSIS START "
            f"config={args.config or 'checkpoint full_config'} "
            f"checkpoint_dir={info.get('configured_checkpoint_dir')} "
            f"output={store.output_dir} "
            f"preset={args.train_analysis_preset} "
            f"items={','.join(analysis_items)} "
            f"sections={','.join(required_sections)}",
            flush=True,
    )
    ctx = build_context(args, ["train_analysis"], store)
    if ctx.is_primary:
        print(
            "TRAIN_ANALYSIS MODEL "
            f"version={ctx.config.get('model', {}).get('model_version')} ",
            flush=True,
        )
    if args.v4171_parity_only:
        run_v4171_parity_only_smoke(ctx)
        sync_hosts("dawn-srw-v4171-parity-only-done")
        return 0
    max_batches = max(1, int(args.train_analysis_max_batches or DEFAULT_TRAIN_ANALYSIS_BATCHES))
    eps_values = _parse_float_list(args.prune_eps or DEFAULT_TRAIN_ANALYSIS_PRUNE_EPS)
    progress = _progress_status(ctx, info)
    selection = _selection_status(
        ctx.config, progress["step"], progress["tokens"], ctx.model_cfg)
    active = (
        _run_active_dynamics(
            ctx,
            max_batches,
            include_composition="composition_health" in analysis_items,
        )
        if "active" in required_sections else {
            "status": "not_requested",
            "reason": "active/tau metrics were not requested by this preset",
            "pools": {},
            "num_batches": 0,
        }
    )
    prune = _run_prune_light(ctx, max_batches, eps_values) if "prune" in required_sections else {}
    prompt_trace = run_train_prompt_trace(ctx) if "prompt_trace" in required_sections else {}
    prompt_decision = build_train_prompt_decision(prompt_trace) if "prompt_trace" in required_sections and ctx.is_primary else {}
    generation_samples = run_train_generation_samples(ctx) if "generation" in required_sections else {}
    v4171_transition_analysis = (
        run_v4171_transition_items(ctx, analysis_items)
        if "v4171_transition" in required_sections else {}
    )
    operator_analysis = (
        run_operator_analysis(ctx, analysis_items)
        if "operator_datasets" in required_sections else {}
    )
    sync_hosts("dawn-srw-train-analysis-done")
    if not ctx.is_primary:
        return 0

    if active.get("status") != "not_requested":
        for pool, pdata in active.get("pools", {}).items():
            pdata["status"] = _active_status(
                pool, pdata.get("active_tau"), progress["tokens"], selection)
        active = _add_target_ratio(active, selection)
        active["target_quantile_gap"] = _build_target_quantile_gap(active, selection)
        active["calibration_state"] = _build_calibration_state(
            active, selection, ctx.config)
    if selection.get("enabled") and active.get("status") != "not_requested":
        active_values = [
            active.get("pools", {}).get(pool, {}).get("active_tau")
            for pool in ("qk", "v", "rst")
        ]
        if all(_safe_float(value) is None for value in active_values):
            warnings.append("selection_calibration is enabled but no active_tau metrics were observed.")
    if v4171_transition_analysis:
        _append_v4171_analysis_warnings(
            warnings, v4171_transition_analysis, list(analysis_items))
    model_version = ctx.config.get("model", {}).get("model_version")
    if model_version == V4166_MODEL_VERSION:
        compare = _compare_400m(active)
        compare.update({"label": "v4166 400M", "status": "ready"})
    else:
        compare = {
            "label": "v4166 400M",
            "status": "not_applicable",
            "reason": "reference ranges are calibrated for v4166, not v4171",
        }
    operator_family_decision: Dict[str, Any] = {}
    operator_family_decision_lines: List[str] = []
    if args.train_analysis_preset == "v4171_operator_family":
        (operator_family_decision,
         operator_family_decision_lines) = _operator_family_decision(
            v4171_transition_analysis)
    summary = {
        "run": {
            "config": args.config or "checkpoint full_config",
            "model_version": model_version,
            "checkpoint_dir": info.get("configured_checkpoint_dir") or info.get("run_folder"),
            "latest_step": progress["step"],
            "latest_path": info.get("latest_path") or args.checkpoint,
            "latest_modified": info.get("latest_modified"),
            "analysis_batches": max_batches,
            "output": store.output_dir,
        },
        "analysis_preset": args.train_analysis_preset,
        "analysis_items": analysis_items,
        "analysis_item_catalog": selected_item_catalog(analysis_items),
        "analysis_required_sections": required_sections,
        "progress": progress,
        "selection_calibration": selection,
        "active_dynamics": active,
        "effective_prune": prune,
        "prompt_trace": prompt_trace,
        "prompt_decision": prompt_decision,
        "generation_samples": generation_samples,
        "v4171_transition_analysis": v4171_transition_analysis,
        "operator_family_decision": operator_family_decision,
        "operator_analysis": operator_analysis,
        "operator_analysis_datasets": (
            operator_analysis.get("dataset_manifest")
            if operator_analysis else {
                "status": "not_requested", "root": args.operator_dataset_root,
            }),
        "reference_400m": compare,
        "causal_baseline": (
            (v4171_transition_analysis.get("causal_intervention") or {}).get(
                "causal_baseline", "canonical_suppression_disabled")
            if model_version in V417X_MODEL_VERSIONS else None),
        "effect_reference": (
            (v4171_transition_analysis.get("causal_intervention") or {}).get(
                "effect_reference", "canonical_suppression_disabled")
            if model_version in V417X_MODEL_VERSIONS else None),
        "canonical_parity_machine_exact": (
            (v4171_transition_analysis.get("causal_intervention") or {}).get(
                "canonical_parity_machine_exact")
            if model_version in V417X_MODEL_VERSIONS else None),
        "cross_graph_audit_blocking": (
            False if model_version in V417X_MODEL_VERSIONS else None),
        "warnings": warnings,
    }
    scalar = _scalar_row(summary)
    summary["recent_trend"] = _trend_rows(store, scalar, warnings)
    summary["decision"] = (
        operator_family_decision_lines
        if args.train_analysis_preset == "v4171_operator_family"
        else _decision_lines(active, selection, progress))
    summary["warnings"] = warnings
    emit_train_analysis_item_progress(summary, analysis_items)
    text = _format_train_analysis(summary)
    store, saved_dir = _save_train_analysis(store, summary, text, scalar, warnings)
    summary["warnings"] = warnings
    text = _format_train_analysis(summary)
    print(text, flush=True)
    print("", flush=True)
    print(f"Saved: {join_path(saved_dir, 'train_analysis_latest.txt')}", flush=True)
    print(f"Saved: {join_path(saved_dir, 'train_analysis_latest.json')}", flush=True)
    print(f"Saved: {join_path(saved_dir, 'train_analysis_history.txt')}", flush=True)
    print(f"Saved: {join_path(saved_dir, 'train_analysis.jsonl')}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.from_scratch:
        args.resume = False
    if args.list_train_analysis_items:
        print(train_analysis_catalog_text())
        return 0
    if args.train_analysis:
        maybe_init_distributed(args, True)
        return run_train_analysis(args, is_primary_host())
    if args.v4171_parity_only:
        if args.output is None:
            raise ValueError("--output is required for --v4171-parity-only")
        maybe_init_distributed(args, True)
        primary = is_primary_host()
        store = AnalysisStore(
            args.output, is_primary=primary,
            analysis_version=ANALYSIS_VERSION)
        set_default_store(store)
        store.ensure_layout()
        ctx = build_context(args, ["train_analysis"], store)
        run_v4171_parity_only_smoke(ctx)
        sync_hosts("dawn-srw-v4171-parity-only-done")
        return 0
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
