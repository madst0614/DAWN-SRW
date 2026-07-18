#!/usr/bin/env python3
"""Run the canonical DAWN operator interpretability analysis pool."""

from __future__ import annotations

import argparse
import gc
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
    get_train,
    git_info,
    is_primary_host,
    model_cfg_from_config,
    resolve_checkpoint,
    restore_params_and_cfg,
    sync_hosts,
)
from analysis.dawn_analysis_storage import (
    AnalysisStore,
    join_path,
    read_json,
    write_json_atomic,
)
from analysis.operator_interpretability.artifacts import (
    DEFAULT_BENCHMARK_ROOT,
    resolve_benchmark_build,
    write_protocol_bound_artifact,
)
from analysis.operator_interpretability.benchmark_schema import canonical_hash
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
    benchmark_ids_for_items,
    dependency_closure,
    item_definition,
    items_for_backend,
    parse_train_analysis_pool_items,
    train_analysis_pool_catalog,
)
from analysis.train_analysis_pool_targets import (
    DEFAULT_REGISTRY_PATH,
    ExecutionSelection,
    apply_execution_mesh,
    resolve_execution_selection,
    target_runtime_catalog,
    target_spec,
    validate_target_checkpoint_config,
)
from dawn.eval.zero_shot_protocol import (
    LM_EVAL_VERSION,
    NUM_FEWSHOT,
    PROTOCOL_NAME as ZERO_SHOT_PROTOCOL_NAME,
    PROTOCOL_VERSION as ZERO_SHOT_PROTOCOL_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Target/runtime-bound stock zero-shot and mechanistic analysis "
            "items for the DAWN shared operator space."))
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--target", help="Registered model/checkpoint target, for example v4171_400m")
    source.add_argument(
        "--checkpoint", help="Ad-hoc Orbax step, checkpoints directory, run, or latest path")
    parser.add_argument("--config", default=None, help="Fallback only when checkpoint metadata lacks full_config")
    parser.add_argument("--output", default=None)
    parser.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY_PATH),
        help="Target/runtime registry YAML")
    parser.add_argument(
        "--runtime", default="v4-64",
        help=("Physical runtime profile; registry default is v4-64. "
              "The target model axis and runtime device count resolve the mesh."))
    parser.add_argument(
        "--preset", default=DEFAULT_TRAIN_ANALYSIS_POOL_PRESET,
        choices=tuple(TRAIN_ANALYSIS_POOL_PRESETS))
    parser.add_argument("--items", default=None, help="Comma-separated canonical items, or all")
    parser.add_argument("--list-items", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--zero-shot-batch-size", type=int, default=32)
    parser.add_argument("--zero-shot-limit", type=int, default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume only artifacts with an exactly matching protocol/checkpoint hash")
    parser.add_argument(
        "--mesh-data", type=int, default=None,
        help="Ad-hoc checkpoint assertion; registered target mesh cannot be overridden")
    parser.add_argument(
        "--mesh-model", type=int, default=None,
        help="Ad-hoc checkpoint model-axis assertion; target owns this value")
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


def _zero_shot_args(
        args: argparse.Namespace, *, init_from: str, output_dir: str,
        tasks: list[str], selection: ExecutionSelection) -> argparse.Namespace:
    return argparse.Namespace(
        init_from=init_from,
        output_dir=output_dir,
        tasks=",".join(tasks),
        tokenizer=None,
        pad_token_id=None,
        eot_token_id=None,
        batch_size=args.zero_shot_batch_size,
        length_buckets="64,128,256,512",
        token_chunk_size=32768,
        max_gen_toks=256,
        limit=args.zero_shot_limit,
        bootstrap_iters=100000,
        validation_ce_tolerance=5e-5,
        model_name=selection.target_id,
        known_validation_loss=None,
        seed=args.seed,
        mesh_data=selection.mesh.mesh_data,
        mesh_model=selection.mesh.mesh_model,
    )


def _run_zero_shot_backend(
        args: argparse.Namespace, *, selection: ExecutionSelection,
        checkpoint_dir: str, checkpoint_step: int,
        checkpoint_config_hash: str, item_ids: list[str],
        store: AnalysisStore) -> dict:
    if args.zero_shot_batch_size <= 0:
        raise ValueError("--zero-shot-batch-size must be positive")
    if args.zero_shot_limit is not None and args.zero_shot_limit <= 0:
        raise ValueError("--zero-shot-limit must be positive when specified")
    tasks = [str(item_definition(item_id)["task_id"]) for item_id in item_ids]
    concrete_checkpoint = join_path(
        checkpoint_dir, f"{int(checkpoint_step):012d}")
    backend_output = store.path("backends", "stock_zero_shot")
    code = git_info()
    loaded = {}
    for item_id, task in zip(item_ids, tasks):
        path = store.path("items", "zero_shot", f"{task}.json")
        record = read_json(path, None) if args.resume else None
        if record is None:
            loaded = {}
            break
        if not isinstance(record, dict):
            raise ValueError(f"invalid zero-shot item artifact: {path}")
        protocol = record.get("protocol")
        payload = record.get("payload")
        if not isinstance(protocol, dict) or not isinstance(payload, dict):
            raise ValueError(f"invalid zero-shot protocol artifact: {path}")
        if record.get("protocol_hash") != canonical_hash(protocol):
            raise ValueError(f"zero-shot protocol hash mismatch: {path}")
        required_protocol_fields = (
            "task_version", "task_config_hash", "dataset", "tokenizer",
            "checkpoint_params_hash",
        )
        missing_protocol = [
            key for key in required_protocol_fields
            if key not in protocol
        ]
        if missing_protocol:
            raise ValueError(
                f"zero-shot artifact lacks protocol fields for {item_id}: "
                + ",".join(missing_protocol))
        expected = {
            "protocol_name": ZERO_SHOT_PROTOCOL_NAME,
            "protocol_version": ZERO_SHOT_PROTOCOL_VERSION,
            "lm_eval_version": LM_EVAL_VERSION,
            "num_fewshot": NUM_FEWSHOT,
            "task_id": task,
            "checkpoint_path": concrete_checkpoint,
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_config_hash": checkpoint_config_hash,
            "target_id": selection.target_id,
            "runtime_id": selection.mesh.runtime_id,
            "effective_mesh": {
                "data": selection.mesh.mesh_data,
                "model": selection.mesh.mesh_model,
            },
            "analysis_git_commit": code.get("git_commit"),
            "analysis_git_branch": code.get("git_branch"),
            "analysis_working_tree_clean": not bool(code.get("git_dirty")),
        }
        differences = [
            key for key, value in expected.items()
            if protocol.get(key) != value
        ]
        if differences:
            raise ValueError(
                f"zero-shot artifact protocol mismatch for {item_id}: "
                + ",".join(differences))
        if payload.get("item_id") != item_id:
            raise ValueError(f"zero-shot item identity mismatch: {path}")
        loaded[item_id] = payload
    if len(loaded) == len(item_ids):
        for item_id in item_ids:
            if store.is_primary:
                print(
                    f"TRAIN_ANALYSIS_POOL item={item_id} status=resume",
                    flush=True)
        return {
            "status": "complete",
            "requested_items": list(item_ids),
            "executed_items": list(item_ids),
            "item_status": {
                item_id: loaded[item_id].get("status")
                for item_id in item_ids
            },
            "resumed_items": list(item_ids),
            "output": backend_output,
        }
    from scripts.zero_shot_eval_jax import run_evaluation

    result = run_evaluation(_zero_shot_args(
        args,
        init_from=concrete_checkpoint,
        output_dir=backend_output,
        tasks=tasks,
        selection=selection,
    ))
    summary = dict(result["summary"])
    manifest = dict(result["manifest"])
    raw_results = dict(result["raw_results"])
    item_status = {}
    for item_id, task in zip(item_ids, tasks):
        metrics = (raw_results.get("results") or {}).get(task)
        if not isinstance(metrics, dict):
            raise RuntimeError(
                f"zero-shot backend omitted selected task result: {task}")
        protocol = {
            "protocol_name": manifest["protocol_name"],
            "protocol_version": manifest["protocol_version"],
            "lm_eval_version": manifest["lm_eval_version"],
            "num_fewshot": manifest["num_fewshot"],
            "task_id": task,
            "task_version": manifest["lm_eval_task_versions"][task],
            "task_config_hash": manifest["task_config_hashes"][task],
            "dataset": manifest["dataset_fingerprints_revisions"][task],
            "checkpoint_path": manifest["source_checkpoint_concrete_path"],
            "checkpoint_step": manifest["source_checkpoint_step"],
            "checkpoint_config_hash": manifest["source_checkpoint_config_hash"],
            "checkpoint_params_hash": manifest["params_hash"],
            "target_id": selection.target_id,
            "runtime_id": selection.mesh.runtime_id,
            "effective_mesh": manifest["mesh_shape"],
            "tokenizer": manifest["tokenizer"],
            "analysis_git_commit": manifest["dawn_git_commit"],
            "analysis_git_branch": manifest["dawn_git_branch"],
            "analysis_working_tree_clean": manifest["working_tree_clean"],
        }
        payload = {
            "item_id": item_id,
            "backend": "stock_zero_shot",
            "analysis_kind": "zero_shot_task",
            "task_id": task,
            "claim_role": "auxiliary",
            "status": "ready",
            "result": {
                "status": "ready",
                "metrics": metrics,
                "runtime": summary.get("task_runtime", {}).get(task, {}),
                "comparable": manifest.get("limit") is None,
                "backend_output": result["output_dir"],
            },
        }
        if store.is_primary:
            write_protocol_bound_artifact(
                store, f"items/zero_shot/{task}.json", payload,
                protocol=protocol)
        item_status[item_id] = "ready"
        if store.is_primary:
            print(
                f"TRAIN_ANALYSIS_POOL item={item_id} status=ready",
                flush=True)
    return {
        "status": "complete",
        "requested_items": list(item_ids),
        "executed_items": list(item_ids),
        "item_status": item_status,
        "output": result["output_dir"],
        "comparable": summary.get("comparable", False),
        "protocol_hashes": {
            item_id: canonical_hash({
                "task_config_hash": manifest["task_config_hashes"][task],
                "checkpoint_params_hash": manifest["params_hash"],
                "runtime_id": selection.mesh.runtime_id,
            })
            for item_id, task in zip(item_ids, tasks)
        },
    }


def _resolve_source(
        args: argparse.Namespace) -> tuple[
            ExecutionSelection, str, int, dict, dict, dict, str]:
    requested_checkpoint = args.checkpoint
    requested_config = args.config
    if args.target:
        registered = target_spec(args.target, registry_path=args.registry)
        requested_checkpoint = registered.checkpoint
        requested_config = registered.config
    checkpoint_dir, checkpoint_step, checkpoint_metadata = resolve_checkpoint(
        requested_checkpoint)
    if checkpoint_dir is None or checkpoint_step is None:
        raise ValueError("exactly one of --target or --checkpoint is required")
    checkpoint_config = config_from_checkpoint_or_file(
        requested_config, checkpoint_metadata)
    if args.target:
        validate_target_checkpoint_config(checkpoint_config, registered)
    checkpoint_config_hash = get_train()._config_sha256(checkpoint_config)
    checkpoint_training = checkpoint_config.get("training", {})
    checkpoint_mesh = {
        "data": int(checkpoint_training.get("mesh_data", 0) or 0),
        "model": int(checkpoint_training.get("mesh_model", 0) or 0),
    }
    ad_hoc_mesh_model = args.mesh_model
    if not args.target and ad_hoc_mesh_model is None:
        ad_hoc_mesh_model = checkpoint_mesh["model"]
    selection = resolve_execution_selection(
        target_id=args.target,
        checkpoint=args.checkpoint,
        config=args.config,
        runtime_id=args.runtime,
        registry_path=args.registry,
        mesh_model_override=ad_hoc_mesh_model,
        mesh_data_override=args.mesh_data,
    )
    config = apply_execution_mesh(
        checkpoint_config, selection,
        visible_device_count=int(jax.device_count()),
        visible_process_count=int(jax.process_count()))
    validate_model_version(str(config.get("model", {}).get("model_version")))
    return (
        selection, checkpoint_dir, int(checkpoint_step),
        checkpoint_metadata, config, checkpoint_mesh,
        checkpoint_config_hash)


def _build_context(
        args: argparse.Namespace, selection: ExecutionSelection,
        checkpoint_dir: str, checkpoint_step: int,
        checkpoint_metadata: dict, config: dict,
        checkpoint_mesh: dict) -> AnalysisContext:
    mesh = create_mesh_from_cfg(config)
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
        "target_id": selection.target_id,
        "runtime_id": selection.mesh.runtime_id,
        "accelerator_type": selection.mesh.accelerator_type,
        "checkpoint_mesh": checkpoint_mesh,
        "mesh": {"data": int(mesh.shape["data"]),
                 "model": int(mesh.shape["model"])},
    }
    if store.is_primary:
        print(
            "TRAIN_ANALYSIS_POOL load "
            f"target={selection.target_id or 'ad-hoc'} "
            f"runtime={selection.mesh.runtime_id} "
            f"mesh={int(mesh.shape['data'])}x{int(mesh.shape['model'])} "
            f"version={model_cfg['model_version']} step={checkpoint_step} "
            f"params={model_info['param_count']} output={output}",
            flush=True)
    return AnalysisContext(
        args=args,
        store=store,
        config=config,
        config_path=selection.config,
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
    if args.list_items or args.list_targets:
        catalog = {}
        if args.list_items:
            catalog.update(train_analysis_pool_catalog())
        if args.list_targets:
            catalog.update(target_runtime_catalog(registry_path=args.registry))
        print(json.dumps(
            catalog, indent=2,
            sort_keys=True, ensure_ascii=False))
        return 0
    if not args.target and not args.checkpoint:
        raise ValueError("exactly one of --target or --checkpoint is required")
    items = parse_train_analysis_pool_items(args.preset, args.items)
    executed = dependency_closure(items)
    mechanistic_items = items_for_backend(
        items, "operator_interpretability")
    zero_shot_items = items_for_backend(items, "stock_zero_shot")
    mechanistic_executed = dependency_closure(mechanistic_items)
    mechanistic_benchmark_ids = benchmark_ids_for_items(
        mechanistic_executed)
    _initialize_distributed(args)
    if mechanistic_items:
        benchmark_build = resolve_benchmark_build(args.benchmark_root)
        missing_benchmarks = sorted(
            set(mechanistic_benchmark_ids)
            - set(benchmark_build.manifest["benchmarks"]))
        if missing_benchmarks:
            raise FileNotFoundError(
                "published benchmark build lacks required ids: "
                f"build_id={benchmark_build.build_id} "
                f"missing={','.join(missing_benchmarks)}")
        if is_primary_host():
            print(
                "TRAIN_ANALYSIS_POOL benchmark "
                f"build_id={benchmark_build.build_id} "
                f"manifest={benchmark_build.manifest_path}",
                flush=True,
            )
    (selection, checkpoint_dir, checkpoint_step, checkpoint_metadata,
     config, checkpoint_mesh, checkpoint_config_hash) = _resolve_source(args)
    output = args.output or _default_output(checkpoint_dir, checkpoint_step)
    store = AnalysisStore(
        output, is_primary=is_primary_host(),
        analysis_version=f"train_analysis_pool-{PROTOCOL_SCHEMA_VERSION}")
    backend_summaries = {}
    item_status = {}
    strongest_claim = None

    if zero_shot_items:
        backend_summaries["stock_zero_shot"] = _run_zero_shot_backend(
            args,
            selection=selection,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
            checkpoint_config_hash=checkpoint_config_hash,
            item_ids=zero_shot_items,
            store=store,
        )
        item_status.update(
            backend_summaries["stock_zero_shot"]["item_status"])
        gc.collect()
        jax.clear_caches()
        sync_hosts("train_analysis_pool_zero_shot_complete")

    if mechanistic_items:
        context = _build_context(
            args, selection, checkpoint_dir, checkpoint_step,
            checkpoint_metadata, config, checkpoint_mesh)
        sync_hosts("train_analysis_pool_loaded")
        runner = OperatorInterpretabilityRunner(
            context,
            benchmark_root=args.benchmark_root,
            benchmark_ids=mechanistic_benchmark_ids,
            protocol_config=_protocol_config(args),
            resume=args.resume,
        )
        backend_summaries["operator_interpretability"] = runner.run(
            mechanistic_items)
        item_status.update(
            backend_summaries["operator_interpretability"]["item_status"])
        strongest_claim = backend_summaries[
            "operator_interpretability"].get("strongest_supported_claim")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "requested_items": list(items),
        "executed_items": list(executed),
        "item_status": item_status,
        "target": selection.to_dict(),
        "checkpoint_path_resolved": checkpoint_dir,
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_mesh": checkpoint_mesh,
        "effective_mesh": {
            "data": selection.mesh.mesh_data,
            "model": selection.mesh.mesh_model,
        },
        "model_version": config.get("model", {}).get("model_version"),
        "backends": backend_summaries,
        "strongest_supported_claim": strongest_claim,
    }
    sync_hosts("train_analysis_pool_complete")
    if store.is_primary:
        write_json_atomic(store.path("summary.json"), summary)
        print(
            "TRAIN_ANALYSIS_POOL COMPLETE "
            f"status={summary['status']} "
            f"claim={summary.get('strongest_supported_claim')} "
            f"output={store.output_dir}",
            flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
