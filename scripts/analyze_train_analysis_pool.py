#!/usr/bin/env python3
"""Run the canonical DAWN operator interpretability analysis pool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import (
    AnalysisContext,
    config_from_checkpoint_or_file,
    count_params,
    create_mesh_from_cfg,
    create_or_reuse_sharded_fns,
    is_primary_host,
    model_cfg_from_config,
    resolve_checkpoint,
    restore_params_and_cfg,
    sync_hosts,
)
from analysis.dawn_analysis_storage import AnalysisStore
from analysis.operator_interpretability.artifacts import DEFAULT_BENCHMARK_ROOT
from analysis.operator_interpretability.benchmark_registry import (
    BENCHMARK_SPECS,
    PRIMARY_BENCHMARK_IDS,
)
from analysis.operator_interpretability.protocol import (
    PROTOCOL_ID,
    PROTOCOL_SCHEMA_VERSION,
    ProtocolConfig,
    validate_model_version,
)
from analysis.operator_interpretability.runner import OperatorInterpretabilityRunner
from analysis.train_analysis_pool_items import (
    DEFAULT_TRAIN_ANALYSIS_POOL_PRESET,
    TRAIN_ANALYSIS_POOL_PRESETS,
    parse_train_analysis_pool_items,
    train_analysis_pool_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol-bound MIB-adapted and RAVEL-style analysis of the "
            "DAWN shared operator space."))
    parser.add_argument("--checkpoint", help="Orbax step, checkpoints directory, run, or latest path")
    parser.add_argument("--config", default=None, help="Fallback only when checkpoint metadata lacks full_config")
    parser.add_argument("--output", default=None)
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--benchmarks", default="primary",
        help="primary, all, or comma-separated canonical benchmark ids")
    parser.add_argument(
        "--preset", default=DEFAULT_TRAIN_ANALYSIS_POOL_PRESET,
        choices=tuple(TRAIN_ANALYSIS_POOL_PRESETS))
    parser.add_argument("--items", default=None, help="Comma-separated canonical items, or all")
    parser.add_argument("--list-items", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume only artifacts with an exactly matching protocol/checkpoint hash")
    parser.add_argument("--mesh-data", type=int, default=None)
    parser.add_argument("--mesh-model", type=int, default=None)
    distributed = parser.add_mutually_exclusive_group()
    distributed.add_argument("--init-distributed", action="store_true")
    distributed.add_argument("--no-init-distributed", action="store_true")

    defaults = ProtocolConfig()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--max-examples-per-phase", type=int,
        default=defaults.max_examples_per_phase)
    parser.add_argument("--capture-threshold", type=float, default=defaults.capture_threshold)
    parser.add_argument("--capture-topk-qk", type=int, default=defaults.capture_topk_qk)
    parser.add_argument("--capture-topk-v", type=int, default=defaults.capture_topk_v)
    parser.add_argument("--capture-topk-rst", type=int, default=defaults.capture_topk_rst)
    parser.add_argument("--capture-max-topk-qk", type=int, default=defaults.capture_max_topk_qk)
    parser.add_argument("--capture-max-topk-v", type=int, default=defaults.capture_max_topk_v)
    parser.add_argument("--capture-max-topk-rst", type=int, default=defaults.capture_max_topk_rst)
    parser.add_argument("--space-max-operators", type=int, default=defaults.space_max_operators)
    parser.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    parser.add_argument("--permutation-samples", type=int, default=defaults.permutation_samples)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--minimum-known-correct", type=int, default=defaults.minimum_known_correct)
    parser.add_argument(
        "--minimum-pairs-per-causal-variable", type=int,
        default=defaults.minimum_pairs_per_causal_variable)
    parser.add_argument("--family-neighbor-k", type=int, default=defaults.family_neighbor_k)
    parser.add_argument(
        "--family-similarity-quantile", type=float,
        default=defaults.family_similarity_quantile)
    parser.add_argument("--rank-stability-min", type=float, default=defaults.rank_stability_min)
    parser.add_argument(
        "--circuit-faithfulness-min", type=float,
        default=defaults.circuit_faithfulness_min)
    parser.add_argument(
        "--interchange-success-min", type=float,
        default=defaults.interchange_success_min)
    parser.add_argument(
        "--isolation-max-absolute-effect", type=float,
        default=defaults.isolation_max_absolute_effect)
    return parser.parse_args()


def _multihost_environment() -> bool:
    return any(os.environ.get(key) for key in (
        "TPU_WORKER_ID", "CLOUD_TPU_TASK_ID", "TPU_PROCESS_BOUNDS",
        "JAX_COORDINATOR_ADDRESS", "MEGASCALE_COORDINATOR_ADDRESS",
        "COORDINATOR_ADDRESS",
    ))


def _initialize_distributed(args: argparse.Namespace) -> None:
    enabled = bool(args.init_distributed) or (
        _multihost_environment() and not args.no_init_distributed)
    if not enabled:
        return
    try:
        jax.distributed.initialize()
    except RuntimeError as exc:
        if "already initialized" not in str(exc).lower():
            raise


def _benchmark_ids(value: str) -> tuple[str, ...]:
    key = str(value).strip().lower()
    if key == "primary":
        return PRIMARY_BENCHMARK_IDS
    if key == "all":
        return tuple(BENCHMARK_SPECS)
    result = tuple(part.strip() for part in key.split(",") if part.strip())
    unknown = [item for item in result if item not in BENCHMARK_SPECS]
    if unknown:
        raise ValueError(f"unknown benchmarks: {','.join(unknown)}")
    return result


def _default_output(checkpoints_dir: str, step: int) -> str:
    normalized = str(checkpoints_dir).rstrip("/\\").replace("\\", "/")
    run_root = normalized.rsplit("/", 1)[0]
    return f"{run_root}/side_analysis/train_analysis_pool/{int(step):012d}"


def _protocol_config(args: argparse.Namespace) -> ProtocolConfig:
    return ProtocolConfig(
        seed=args.seed,
        max_examples_per_phase=args.max_examples_per_phase,
        capture_threshold=args.capture_threshold,
        capture_topk_qk=args.capture_topk_qk,
        capture_topk_v=args.capture_topk_v,
        capture_topk_rst=args.capture_topk_rst,
        capture_max_topk_qk=args.capture_max_topk_qk,
        capture_max_topk_v=args.capture_max_topk_v,
        capture_max_topk_rst=args.capture_max_topk_rst,
        space_max_operators=args.space_max_operators,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        alpha=args.alpha,
        minimum_known_correct=args.minimum_known_correct,
        minimum_pairs_per_causal_variable=(
            args.minimum_pairs_per_causal_variable),
        family_neighbor_k=args.family_neighbor_k,
        family_similarity_quantile=args.family_similarity_quantile,
        rank_stability_min=args.rank_stability_min,
        circuit_faithfulness_min=args.circuit_faithfulness_min,
        interchange_success_min=args.interchange_success_min,
        isolation_max_absolute_effect=args.isolation_max_absolute_effect,
    ).validate()


def _build_context(args: argparse.Namespace) -> AnalysisContext:
    checkpoint_dir, checkpoint_step, checkpoint_metadata = resolve_checkpoint(
        args.checkpoint)
    if checkpoint_dir is None or checkpoint_step is None:
        raise ValueError("--checkpoint is required")
    config = config_from_checkpoint_or_file(args.config, checkpoint_metadata)
    validate_model_version(str(config.get("model", {}).get("model_version")))
    mesh = create_mesh_from_cfg(config, args)
    output = args.output or _default_output(checkpoint_dir, checkpoint_step)
    store = AnalysisStore(
        output, is_primary=is_primary_host(),
        analysis_version=f"{PROTOCOL_ID}-{PROTOCOL_SCHEMA_VERSION}")
    params, restored_metadata, model = restore_params_and_cfg(
        config, checkpoint_dir, int(checkpoint_step), mesh)
    checkpoint_metadata.update(restored_metadata or {})
    sharded_fns = create_or_reuse_sharded_fns(config, mesh, analysis=False)
    steps_per_epoch = int(checkpoint_metadata.get("steps_per_epoch") or 0)
    num_epochs = int(config.get("training", {}).get("num_epochs") or 0)
    accumulation = max(1, int(config.get("training", {}).get(
        "gradient_accumulation_steps", 1)))
    planned = (
        (steps_per_epoch // accumulation) * num_epochs
        if steps_per_epoch and num_epochs else 0)
    total_steps = int(
        planned or checkpoint_metadata.get("global_step")
        or checkpoint_step or 1)
    model_cfg = model_cfg_from_config(
        config, checkpoint_step=int(checkpoint_step),
        total_training_steps=total_steps)
    model_info = {
        "param_count": count_params(params),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_path_resolved": checkpoint_dir,
        "mesh": {"data": int(mesh.shape["data"]),
                 "model": int(mesh.shape["model"])},
    }
    if store.is_primary:
        print(
            "TRAIN_ANALYSIS_POOL load "
            f"version={model_cfg['model_version']} step={checkpoint_step} "
            f"params={model_info['param_count']} output={output}",
            flush=True)
    return AnalysisContext(
        args=args,
        store=store,
        config=config,
        config_path=args.config,
        checkpoint_path=checkpoint_dir,
        checkpoint_step=int(checkpoint_step),
        checkpoint_metadata=checkpoint_metadata,
        model=model,
        params=params,
        mesh=mesh,
        data_sharding=NamedSharding(mesh, P("data", None)),
        sharded_fns=sharded_fns,
        sharded_fns_analysis=None,
        model_cfg=model_cfg,
        model_info=model_info,
        host_id=int(jax.process_index()),
        n_hosts=int(jax.process_count()),
        is_primary=store.is_primary,
        total_training_steps=total_steps,
    )


def main() -> int:
    args = parse_args()
    if args.list_items:
        print(json.dumps(
            train_analysis_pool_catalog(), indent=2,
            sort_keys=True, ensure_ascii=False))
        return 0
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    items = parse_train_analysis_pool_items(args.preset, args.items)
    _initialize_distributed(args)
    context = _build_context(args)
    sync_hosts("train_analysis_pool_loaded")
    runner = OperatorInterpretabilityRunner(
        context,
        benchmark_root=args.benchmark_root,
        benchmark_ids=_benchmark_ids(args.benchmarks),
        protocol_config=_protocol_config(args),
        resume=args.resume,
    )
    summary = runner.run(items)
    sync_hosts("train_analysis_pool_complete")
    if context.is_primary:
        print(
            "TRAIN_ANALYSIS_POOL COMPLETE "
            f"status={summary['status']} "
            f"claim={summary.get('strongest_supported_claim')} "
            f"output={context.store.output_dir}",
            flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
