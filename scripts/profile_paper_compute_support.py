#!/usr/bin/env python3
"""Profile v4172 exact/epsilon operator support on packed C4 validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_common import (
    git_info,
    materialize_global_tree,
    model_cfg_from_config,
    sync_hosts,
)
from analysis.dawn_analysis_storage import (
    exists,
    join_path,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.paper_compute import (
    DEFAULT_EPSILON_THRESHOLDS,
    summarize_support_histograms,
    support_profile_forward,
)
from scripts import c4_validation_eval_jax as c4_protocol
from scripts import train_jax as canonical
from scripts import zero_shot_eval_jax as shared
from scripts.paper_flop_accounting import run_accounting
from utils import data_jax


PROTOCOL_NAME = "v4172_packed_c4_operator_support_v1"
PROTOCOL_VERSION = 1


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _parse_epsilons(value: str) -> tuple[float, ...]:
    result = tuple(
        float(part.strip()) for part in str(value).split(",")
        if part.strip())
    if not result:
        raise ValueError("--epsilon-thresholds cannot be empty")
    if any(
            not np.isfinite(item) or item <= 0.0
            for item in result):
        raise ValueError("epsilon thresholds must be finite and positive")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(
            "epsilon thresholds must be unique and sorted ascending")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--expected-step", type=int, default=76293)
    parser.add_argument("--expected-checkpoint-config-hash", default=None)
    parser.add_argument("--expected-checkpoint-identity", default=None)
    parser.add_argument("--expected-model-config-hash", default=None)
    parser.add_argument("--output-parent", default=None)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--max-val-tokens", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--mesh-data", type=int, default=None)
    parser.add_argument("--mesh-model", type=int, default=None)
    parser.add_argument(
        "--epsilon-thresholds",
        default=",".join(f"{value:.0e}" for value in
                         DEFAULT_EPSILON_THRESHOLDS),
    )
    parser.add_argument(
        "--dawn-config",
        default=(
            "configs/"
            "train_config_v4172_400M_c4_40B_v4_64_ver1_"
            "den_qk0p5_v1p0_rst1p2.yaml"),
    )
    parser.add_argument(
        "--baseline-config",
        default=(
            "configs/"
            "train_config_baseline_tpu_400M_c4_40B_v4_64_tp2.yaml"),
    )
    parser.add_argument(
        "--baseline-parameter-count", type=int, default=393896960)
    parser.add_argument(
        "--accounting-batch-size", type=int, default=1)
    parser.add_argument(
        "--accounting-sequence-length", type=int, default=512)
    return parser.parse_args()


def _shared_run_id() -> str:
    width = 25
    local = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:8]
        if int(jax.process_index()) == 0 else "0" * width)
    encoded = np.frombuffer(local.encode("ascii"), dtype=np.uint8)
    if int(jax.process_count()) > 1:
        from jax.experimental import multihost_utils

        encoded = np.asarray(multihost_utils.broadcast_one_to_all(
            encoded, is_source=int(jax.process_index()) == 0))
    value = bytes(encoded.tolist()).decode("ascii")
    if len(value) != width or value == "0" * width:
        raise RuntimeError("failed to establish a shared support-profile run id")
    return value


def _parameter_schema_record(params: Any) -> list[dict[str, Any]]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        rows.append({
            "path": jax.tree_util.keystr(path),
            "shape": [int(value) for value in leaf.shape],
            "dtype": str(leaf.dtype),
        })
    return rows


def _planned_training_steps(
        config: Mapping[str, Any], checkpoint_metadata: Mapping[str, Any],
        checkpoint_step: int) -> int:
    training = config.get("training", {})
    steps_per_epoch = int(checkpoint_metadata.get("steps_per_epoch") or 0)
    num_epochs = int(training.get("num_epochs") or 0)
    accumulation = max(
        1, int(training.get("gradient_accumulation_steps", 1)))
    planned = (
        (steps_per_epoch // accumulation) * num_epochs
        if steps_per_epoch and num_epochs else 0)
    return int(
        planned or checkpoint_metadata.get("global_step")
        or checkpoint_step or 1)


def _software_record() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
        "flax": metadata.version("flax"),
        "optax": metadata.version("optax"),
        "numpy": np.__version__,
    }


def _write_summary_log(
        output_dir: str, summary: Mapping[str, Any],
        accounting: Mapping[str, Any]) -> None:
    lines = [
        "PAPER COMPUTE SUPPORT PROFILE",
        f"status={summary['status']}",
        f"protocol={summary['protocol']['name']}",
        f"checkpoint_step={summary['checkpoint']['step']}",
        (
            "checkpoint_identity="
            f"{summary['checkpoint']['checkpoint_identity']}"
        ),
        f"summary_hash={summary['summary_hash']}",
        (
            "positions="
            f"{summary['input_contract']['evaluated_positions']}"
        ),
        (
            "thresholds="
            + ",".join(
                row["threshold_id"]
                for row in summary["aggregate_rows"]
                if row["route"] == "q")
        ),
    ]
    exact_rows = [
        row for row in summary["aggregate_rows"]
        if row["threshold_id"] == "exact_margin_gt_0"]
    for row in exact_rows:
        lines.append(
            f"exact_{row['route']}_mean={row['mean']:.9f} "
            f"median={row['median']} p90={row['p90']} p99={row['p99']} "
            f"min={row['minimum']} max={row['maximum']} "
            f"pool_fraction={row['mean_pool_fraction']:.9f} "
            f"no_active_fraction={row['no_active_fraction']:.9f}")
    lines.extend([
        (
            "dawn_current_backbone_TFLOPs="
            f"{accounting['columns']['dawn_current_dense_execution']['total_flops'] / 1e12:.9f}"
        ),
        (
            "dawn_exact_support_backbone_TFLOPs="
            f"{accounting['columns']['dawn_exact_support_estimate']['total_flops'] / 1e12:.9f}"
        ),
        (
            "dense_transformer_backbone_TFLOPs="
            f"{accounting['columns']['dense_transformer']['total_flops'] / 1e12:.9f}"
        ),
        (
            "current_dawn_vs_dense="
            f"{accounting['comparisons']['current_dawn_vs_dense_transformer']:.9f}"
        ),
        (
            "exact_support_vs_current="
            f"{accounting['comparisons']['exact_support_vs_current_dawn']:.9f}"
        ),
        (
            "exact_support_vs_dense="
            f"{accounting['comparisons']['exact_support_vs_dense_transformer']:.9f}"
        ),
        "current_execution_sparse=false",
        "indexed_addressing_included=false",
        "measured_latency_claimed=false",
    ])
    write_text_atomic(
        join_path(output_dir, "summary.log"),
        "\n".join(lines) + "\n",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    epsilon_thresholds = _parse_epsilons(args.epsilon_thresholds)
    canonical._maybe_initialize_jax_distributed()
    canonical._require_orbax_checkpoint_compat()
    if jax.__version__ != "0.6.2":
        raise RuntimeError(
            f"paper support profile requires JAX 0.6.2, got {jax.__version__}")

    resolved, run_folder, step = shared._resolve_checkpoint_once(
        canonical, args.init_from)
    if int(step) != int(args.expected_step):
        raise RuntimeError(
            f"resolved checkpoint step={step}, expected={args.expected_step}")
    checkpoint_metadata = shared._restore_metadata(
        canonical, run_folder, step)
    config = deepcopy(checkpoint_metadata["full_config"])
    canonical._require_resume_full_config(config)
    checkpoint_config_hash = canonical._config_sha256(config)
    if (
            args.expected_checkpoint_config_hash
            and checkpoint_config_hash
            != str(args.expected_checkpoint_config_hash)):
        raise RuntimeError(
            "checkpoint config hash differs from the frozen contract: "
            f"actual={checkpoint_config_hash} "
            f"expected={args.expected_checkpoint_config_hash}")
    for key in ("full_config_sha256", "selected_full_config_sha256"):
        recorded = checkpoint_metadata.get(key)
        if recorded is not None and str(recorded) != checkpoint_config_hash:
            raise RuntimeError(
                f"checkpoint metadata {key} mismatch: "
                f"recorded={recorded} computed={checkpoint_config_hash}")
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(checkpoint_config_hash),
        name="checkpoint config hash")

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    if str(model_config.get("model_version")) != "spatial-r1-v4.1.7.2":
        raise RuntimeError(
            "paper support profile is frozen to spatial-r1-v4.1.7.2")
    if str(model_config.get("operator_key_mode")) != (
            "generalized_bilinear_rw"):
        raise RuntimeError(
            "paper support profile requires generalized_bilinear_rw")
    sequence_length = _positive(
        "model.max_seq_len", model_config.get("max_seq_len", 0))
    batch_size = _positive("--batch-size", args.batch_size)
    if batch_size % int(jax.process_count()) != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"host_count={jax.process_count()}")

    checkpoint_mesh_model = _positive(
        "training.mesh_model", training_config.get("mesh_model", 0))
    checkpoint_mesh_data = _positive(
        "training.mesh_data", training_config.get("mesh_data", 0))
    mesh_model = int(args.mesh_model or checkpoint_mesh_model)
    mesh_data = int(args.mesh_data or checkpoint_mesh_data)
    if mesh_model != checkpoint_mesh_model:
        raise RuntimeError(
            "support profiling must preserve checkpoint model-axis sharding")
    if mesh_data * mesh_model != int(jax.device_count()):
        raise RuntimeError(
            f"mesh={mesh_data}x{mesh_model} does not match "
            f"devices={jax.device_count()}")
    if batch_size % mesh_data != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by mesh_data={mesh_data}")
    training_config["mesh_data"] = mesh_data
    training_config["mesh_model"] = mesh_model

    canonical._maybe_materialize_vocab_parallel_config(config)
    mesh = canonical.create_mesh(mesh_data, mesh_model)
    model = canonical.build_model_from_config(config)
    _, target_params = shared._abstract_params(
        canonical, model, config, mesh)
    params = shared._restore_params_only(
        canonical,
        run_folder,
        step,
        target_params,
        checkpoint_metadata,
        mesh,
    )
    del target_params

    parameter_schema = _parameter_schema_record(params)
    parameter_schema_hash = canonical_hash(parameter_schema)
    checkpoint_dir = canonical._join_path(run_folder, "checkpoints")
    checkpoint_identity_record = {
        "path": str(checkpoint_dir),
        "step": int(step),
        "run_id": (
            str(checkpoint_metadata["run_id"])
            if checkpoint_metadata.get("run_id") is not None else None),
        "training_git_commit": (
            str(checkpoint_metadata.get("git_commit")
                or checkpoint_metadata.get("train_script_git_commit"))
            if (checkpoint_metadata.get("git_commit")
                or checkpoint_metadata.get("train_script_git_commit"))
            else None),
        "parameter_schema_hash": parameter_schema_hash,
        "identity_algorithm": (
            "resolved_path_step_run_metadata_and_parameter_schema"),
        "parameter_content_hash_included": False,
    }
    checkpoint_identity = canonical_hash(checkpoint_identity_record)
    if (
            args.expected_checkpoint_identity
            and checkpoint_identity != str(args.expected_checkpoint_identity)):
        raise RuntimeError(
            "checkpoint identity differs from the frozen contract: "
            f"actual={checkpoint_identity} "
            f"expected={args.expected_checkpoint_identity}")
    parameter_count = shared._parameter_count(params)
    c4_parameter_hash = shared.sha256_json({
        "algorithm": "checkpoint_identity_plus_sharded_param_schema_sha256",
        "checkpoint": resolved,
        "step": step,
        "schema": shared._param_schema(params),
    })
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(c4_parameter_hash),
        name="C4 parameter identity")

    runtime_state = canonical._checkpoint_final_runtime(config, resolved)
    sharded_fns = canonical.build_canonical_sharded_fns(
        config,
        mesh,
        for_eval=True,
        analysis=False,
        kernel_profile="production",
    )
    total_training_steps = _planned_training_steps(
        config, checkpoint_metadata, step)
    analysis_model_cfg = model_cfg_from_config(
        config,
        checkpoint_step=int(step),
        total_training_steps=total_training_steps,
    )
    model_config_hash = canonical_hash(analysis_model_cfg)
    if (
            args.expected_model_config_hash
            and model_config_hash != str(args.expected_model_config_hash)):
        raise RuntimeError(
            "analysis model config differs from the frozen contract: "
            f"actual={model_config_hash} "
            f"expected={args.expected_model_config_hash}")

    configured_max_tokens = _positive(
        "data.max_val_tokens", data_config.get("max_val_tokens", 0))
    max_val_tokens = int(args.max_val_tokens or configured_max_tokens)
    _positive("--max-val-tokens", max_val_tokens)
    max_sequences = max_val_tokens // sequence_length
    val_path = str(data_config.get("bin_val") or "")
    if not val_path:
        raise RuntimeError("checkpoint full_config.data.bin_val is missing")
    val_meta_path, val_metadata = c4_protocol._load_validation_metadata(
        data_jax, val_path)
    val_metadata_hash = shared.sha256_json(val_metadata)
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(val_metadata_hash),
        name="validation metadata")
    dataset = data_jax._build_dataset(
        val_path,
        sequence_length,
        max_sequences=max_sequences,
        local_cache_dir=data_config.get("local_cache_dir"),
        evict_previous_cache=False,
    )
    loader = data_jax.BinDataLoader(
        dataset,
        batch_size=batch_size,
        n_devices=1,
        start_step=0,
        n_hosts=int(jax.process_count()),
        host_id=int(jax.process_index()),
    )
    available_batches = int(len(loader))
    if available_batches <= 0:
        raise RuntimeError("validation dataset has no complete global batch")
    if int(args.max_batches) < 0:
        raise ValueError("--max-batches must be zero or positive")
    evaluated_batches = (
        min(int(args.max_batches), available_batches)
        if int(args.max_batches) > 0 else available_batches)
    evaluated_sequences = evaluated_batches * batch_size
    evaluated_positions = evaluated_sequences * sequence_length
    comparable_full_validation = (
        int(args.max_batches) == 0
        and max_val_tokens == configured_max_tokens)

    run_id = _shared_run_id()
    output_parent = (
        str(args.output_parent).rstrip("/")
        if args.output_parent
        else join_path(run_folder, "side_analysis"))
    output_dir = join_path(
        output_parent,
        f"run_analysis_{int(step):012d}_paper_compute_support_{run_id}",
    )
    manifest_path = join_path(output_dir, "run_manifest.json")
    if int(jax.process_index()) == 0 and exists(manifest_path):
        raise FileExistsError(
            f"independent support output already exists: {output_dir}")
    sync_hosts("paper_compute_output_reserved")

    analysis_code = git_info()
    if analysis_code.get("git_dirty"):
        raise RuntimeError(
            "paper support profiling requires a clean analysis worktree")
    protocol_record = {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "epsilon_thresholds": list(epsilon_thresholds),
        "support_definitions": {
            "exact": "selection_margin > 0",
            "approximate": "admission_weight > epsilon",
            "qk_union": "Q active OR K active for the same operator",
        },
        "forward_path": (
            "canonical production minimal kernels; address observation only"),
        "per_token_vectors_persisted": False,
        "histograms_persisted": False,
    }
    input_contract = {
        "validation_path": val_path,
        "validation_metadata_path": val_meta_path,
        "validation_metadata_hash": val_metadata_hash,
        "configured_max_val_tokens": configured_max_tokens,
        "requested_max_val_tokens": max_val_tokens,
        "available_sequences": int(len(dataset)),
        "available_batches": available_batches,
        "evaluated_batches": evaluated_batches,
        "evaluated_sequences": evaluated_sequences,
        "evaluated_positions": evaluated_positions,
        "dropped_incomplete_batch_sequences": (
            int(len(dataset)) - evaluated_sequences),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "comparable_full_validation": comparable_full_validation,
    }
    running_manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "output": output_dir,
        "protocol": protocol_record,
        "checkpoint": {
            "resolved": resolved,
            "run_folder": run_folder,
            "checkpoint_dir": checkpoint_dir,
            "step": int(step),
            "checkpoint_config_hash": checkpoint_config_hash,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_identity_record": checkpoint_identity_record,
            "model_config_hash": model_config_hash,
            "parameter_count": int(parameter_count),
            "c4_parameter_hash": c4_parameter_hash,
        },
        "execution": {
            "mesh": {"data": mesh_data, "model": mesh_model},
            "hosts": int(jax.process_count()),
            "devices": int(jax.device_count()),
            "checkpoint_final_runtime": runtime_state,
            "analysis_code": analysis_code,
            "software": _software_record(),
        },
        "input_contract": input_contract,
        "started_at": started_at.isoformat(),
    }
    if int(jax.process_index()) == 0:
        write_json_atomic(manifest_path, running_manifest)
        print(
            "PAPER_COMPUTE_SUPPORT_START "
            f"step={step} batches={evaluated_batches}/{available_batches} "
            f"batch={batch_size} seq={sequence_length} "
            f"positions={evaluated_positions} output={output_dir}",
            flush=True,
        )
        print(
            f"checkpoint_identity={checkpoint_identity} "
            f"model_config_hash={model_config_hash}",
            flush=True,
        )
    sync_hosts("paper_compute_manifest_written")

    @jax.jit
    def profile_step(model_params, input_ids):
        return support_profile_forward(
            model_params,
            analysis_model_cfg,
            input_ids,
            epsilon_thresholds=epsilon_thresholds,
            production_srw_fns=sharded_fns,
        )

    data_sharding = NamedSharding(mesh, P("data", None))
    accumulated = None
    batch_durations: list[float] = []
    profile_started = time.time()
    try:
        for batch_index, (input_ids, _attention_mask) in enumerate(loader):
            if batch_index >= evaluated_batches:
                break
            global_input_ids = canonical.shard_to_mesh(
                input_ids,
                data_sharding,
                (batch_size, sequence_length),
            )
            batch_started = time.time()
            current = profile_step(params, global_input_ids)
            accumulated = (
                current if accumulated is None
                else jax.tree.map(jnp.add, accumulated, current))
            jax.block_until_ready(accumulated["q"][0, 0, 0])
            batch_durations.append(time.time() - batch_started)
            if int(jax.process_index()) == 0 and (
                    batch_index == 0
                    or batch_index + 1 == evaluated_batches
                    or (batch_index + 1) % max(
                        1, evaluated_batches // 10) == 0):
                print(
                    "PAPER_COMPUTE_SUPPORT_PROGRESS "
                    f"batch={batch_index + 1}/{evaluated_batches} "
                    f"last_sec={batch_durations[-1]:.3f}",
                    flush=True,
                )
        if accumulated is None:
            raise RuntimeError("support profiling produced no batches")
        histograms = materialize_global_tree(accumulated)
        pool_sizes = {
            "q": int(model_config["n_qk"]),
            "k": int(model_config["n_qk"]),
            "qk_union": int(model_config["n_qk"]),
            "v": int(model_config["n_v"]),
            "rst": int(model_config["n_rst"]),
        }
        for route, values in histograms.items():
            totals = np.asarray(values, dtype=np.int64).sum(axis=-1)
            if not np.all(totals == evaluated_positions):
                raise RuntimeError(
                    f"{route} histogram observation mismatch: "
                    f"expected={evaluated_positions} "
                    f"observed_min={int(totals.min())} "
                    f"observed_max={int(totals.max())}")
        support_summary = summarize_support_histograms(
            histograms,
            epsilon_thresholds=epsilon_thresholds,
            pool_sizes=pool_sizes,
        )
        support_summary.pop("summary_hash", None)
        support_summary.update({
            "status": "complete",
            "protocol": protocol_record,
            "checkpoint": running_manifest["checkpoint"],
            "execution": running_manifest["execution"],
            "input_contract": input_contract,
            "runtime": {
                "total_profile_seconds": time.time() - profile_started,
                "first_batch_compile_and_execute_seconds": (
                    batch_durations[0]),
                "post_first_batch_mean_seconds": (
                    float(np.mean(batch_durations[1:]))
                    if len(batch_durations) > 1 else None),
                "post_first_batch_p90_seconds": (
                    float(np.quantile(batch_durations[1:], 0.90))
                    if len(batch_durations) > 1 else None),
                "latency_comparison_eligible": False,
                "note": (
                    "Profiler runtime includes histogram instrumentation and "
                    "is not a production kernel latency result."),
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        support_summary["summary_hash"] = canonical_hash(support_summary)
        support_json = join_path(output_dir, "support_summary.json")
        if int(jax.process_index()) == 0:
            write_json_atomic(support_json, support_summary)
            write_csv_atomic(
                join_path(output_dir, "support_by_layer.csv"),
                support_summary["layer_rows"])
            write_csv_atomic(
                join_path(output_dir, "support_aggregate.csv"),
                support_summary["aggregate_rows"])
        sync_hosts("paper_compute_support_written")

        accounting = None
        if int(jax.process_index()) == 0:
            accounting = run_accounting(
                dawn_config_path=args.dawn_config,
                baseline_config_path=args.baseline_config,
                support_json_path=support_json,
                output_dir=output_dir,
                batch_size=args.accounting_batch_size,
                sequence_length=args.accounting_sequence_length,
                dawn_parameter_count=int(parameter_count),
                baseline_parameter_count=args.baseline_parameter_count,
                write_figure=True,
            )
            _write_summary_log(output_dir, support_summary, accounting)
            completed_manifest = dict(running_manifest)
            completed_manifest.update({
                "status": "complete",
                "support_summary": support_json,
                "support_summary_hash": support_summary["summary_hash"],
                "flop_accounting": join_path(
                    output_dir, "flop_accounting.json"),
                "flop_accounting_hash": accounting["result_hash"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            write_json_atomic(manifest_path, completed_manifest)
            print(
                "PAPER_COMPUTE_SUPPORT_COMPLETE "
                f"summary_hash={support_summary['summary_hash']} "
                f"flop_hash={accounting['result_hash']} "
                f"output={output_dir}",
                flush=True,
            )
        sync_hosts("paper_compute_complete")
        return {
            "output": output_dir,
            "support_summary_hash": support_summary["summary_hash"],
            "accounting": accounting,
        }
    except Exception as exc:
        if int(jax.process_index()) == 0:
            failed_manifest = dict(running_manifest)
            failed_manifest.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            })
            write_json_atomic(manifest_path, failed_manifest)
        raise


def main() -> int:
    result = run(parse_args())
    if int(jax.process_index()) == 0:
        print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
