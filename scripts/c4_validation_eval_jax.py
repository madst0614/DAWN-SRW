#!/usr/bin/env python3
"""Evaluate one committed Orbax checkpoint on the packed C4 validation set."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import zero_shot_eval_jax as shared  # noqa: E402


PROTOCOL_NAME = "packed_c4_validation_v1"
PROTOCOL_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-from",
        required=True,
        help="Orbax run, checkpoints directory, or concrete committed step.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-val-tokens",
        type=int,
        default=0,
        help="0 uses data.max_val_tokens from the checkpoint config.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 evaluates every complete global batch under max-val-tokens.",
    )
    parser.add_argument("--token-chunk-size", type=int, default=32768)
    parser.add_argument("--mesh-data", type=int, default=None)
    parser.add_argument("--mesh-model", type=int, default=None)
    return parser.parse_args()


def _require_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _load_validation_metadata(
    data_jax: Any, val_path: str
) -> tuple[str, Mapping[str, Any]]:
    candidates = []
    for candidate in (val_path, *data_jax._fallback_paths_for(val_path)):
        meta_path = data_jax._meta_path_for(candidate)
        if meta_path not in candidates:
            candidates.append(meta_path)
    for meta_path in candidates:
        try:
            metadata = data_jax._read_json(meta_path)
        except Exception:
            continue
        if isinstance(metadata, dict):
            return str(meta_path), metadata
    raise RuntimeError(
        "cannot resolve validation metadata; tried: " + ", ".join(candidates)
    )


def _evaluate(
    *,
    canonical: Any,
    params: Any,
    model: Any,
    config: Mapping[str, Any],
    mesh: Any,
    sharded_fns: Any,
    runtime_state: Mapping[str, Any],
    val_loader: Any,
    max_batches: int,
    token_chunk_size: int,
) -> tuple[float, float, float]:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    eval_step = canonical.create_eval_step(
        model,
        sharded_fns=sharded_fns,
        runtime_state=runtime_state,
        ce_token_chunk_size=token_chunk_size,
    )
    data_sharding = NamedSharding(mesh, P("data", None))
    started = time.time()
    loss, accuracy = canonical.evaluate(
        eval_step,
        params,
        val_loader,
        jax.local_device_count(),
        max_batches=max_batches,
        verbose=jax.process_index() == 0,
        data_sharding_spec=data_sharding,
        current_step=0,
    )
    jax.block_until_ready((loss, accuracy))
    return float(loss), float(accuracy), time.time() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)

    import jax
    from scripts import train_jax as canonical
    from utils import data_jax

    canonical._maybe_initialize_jax_distributed()
    canonical._require_orbax_checkpoint_compat()

    resolved, run_folder, step = shared._resolve_checkpoint_once(
        canonical, args.init_from
    )
    checkpoint_metadata = shared._restore_metadata(canonical, run_folder, step)
    config = deepcopy(checkpoint_metadata["full_config"])
    canonical._require_resume_full_config(config)
    config_hash = canonical._config_sha256(config)
    if int(checkpoint_metadata.get("global_step", -1)) != int(step):
        raise RuntimeError(
            "checkpoint metadata step differs from the committed directory: "
            f"directory={step} metadata={checkpoint_metadata.get('global_step')}"
        )
    for key in ("full_config_sha256", "selected_full_config_sha256"):
        recorded = checkpoint_metadata.get(key)
        if recorded is not None and str(recorded) != config_hash:
            raise RuntimeError(
                f"checkpoint metadata {key} differs from restored full_config: "
                f"recorded={recorded} computed={config_hash}"
            )
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(config_hash), name="checkpoint config hash"
    )

    model_cfg = config["model"]
    training_cfg = config["training"]
    data_cfg = config["data"]
    max_length = _require_positive(
        "model.max_seq_len", int(model_cfg.get("max_seq_len", 0))
    )
    batch_size = _require_positive("--batch-size", args.batch_size)
    token_chunk_size = _require_positive(
        "--token-chunk-size", args.token_chunk_size
    )
    if batch_size % int(jax.process_count()) != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"host_count={jax.process_count()}"
        )

    checkpoint_mesh_model = _require_positive(
        "training.mesh_model", int(training_cfg.get("mesh_model", 0))
    )
    checkpoint_mesh_data = _require_positive(
        "training.mesh_data", int(training_cfg.get("mesh_data", 0))
    )
    mesh_model = int(args.mesh_model or checkpoint_mesh_model)
    mesh_data = int(args.mesh_data or checkpoint_mesh_data)
    if mesh_model != checkpoint_mesh_model:
        raise RuntimeError(
            "evaluation must preserve checkpoint model-axis sharding: "
            f"checkpoint={checkpoint_mesh_model} requested={mesh_model}"
        )
    if mesh_data * mesh_model != int(jax.device_count()):
        raise RuntimeError(
            "evaluation mesh differs from the current topology: "
            f"mesh={mesh_data}x{mesh_model} devices={jax.device_count()}"
        )
    training_cfg["mesh_data"] = mesh_data
    training_cfg["mesh_model"] = mesh_model

    canonical._maybe_materialize_vocab_parallel_config(config)
    mesh = canonical.create_mesh(mesh_data, mesh_model)
    model = canonical.build_model_from_config(config)
    _, target_params = shared._abstract_params(canonical, model, config, mesh)
    params = shared._restore_params_only(
        canonical,
        run_folder,
        step,
        target_params,
        checkpoint_metadata,
        mesh,
    )
    del target_params

    parameter_schema = shared._param_schema(params)
    parameter_count = shared._parameter_count(params)
    params_hash = shared.sha256_json(
        {
            "algorithm": (
                "checkpoint_identity_plus_sharded_param_schema_sha256"
            ),
            "checkpoint": resolved,
            "step": step,
            "schema": parameter_schema,
        }
    )
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(params_hash), name="parameter identity"
    )

    runtime_state = canonical._checkpoint_final_runtime(config, resolved)
    kernel_profile = (
        "production_diagnostics"
        if canonical._is_v417x_version(model_cfg["model_version"])
        else "production"
    )
    sharded_fns = canonical.build_canonical_sharded_fns(
        config,
        mesh,
        for_eval=True,
        analysis=False,
        kernel_profile=kernel_profile,
    )

    configured_max_tokens = _require_positive(
        "data.max_val_tokens", int(data_cfg.get("max_val_tokens", 0))
    )
    max_val_tokens = int(args.max_val_tokens or configured_max_tokens)
    _require_positive("--max-val-tokens", max_val_tokens)
    max_sequences = max_val_tokens // max_length
    if max_sequences <= 0:
        raise ValueError(
            f"max_val_tokens={max_val_tokens} is smaller than "
            f"sequence_length={max_length}"
        )
    val_path = str(data_cfg.get("bin_val") or "")
    if not val_path:
        raise RuntimeError("checkpoint full_config.data.bin_val is missing")
    val_meta_path, val_metadata = _load_validation_metadata(
        data_jax, val_path
    )
    val_metadata_hash = shared.sha256_json(val_metadata)
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(val_metadata_hash), name="validation metadata"
    )

    dataset = data_jax._build_dataset(
        val_path,
        max_length,
        max_sequences=max_sequences,
        local_cache_dir=data_cfg.get("local_cache_dir"),
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
    if args.max_batches < 0:
        raise ValueError("--max-batches must be zero or positive")
    evaluated_batches = (
        min(int(args.max_batches), available_batches)
        if int(args.max_batches) > 0
        else available_batches
    )
    comparable = (
        int(args.max_batches) == 0
        and max_val_tokens == configured_max_tokens
    )

    if jax.process_index() == 0:
        print(
            "C4_VALIDATION_START "
            f"step={step} batches={evaluated_batches}/{available_batches} "
            f"batch_size={batch_size} seq_len={max_length} "
            f"max_val_tokens={max_val_tokens} comparable={str(comparable).lower()}",
            flush=True,
        )
        print(f"source_checkpoint_params_hash={params_hash}", flush=True)

    val_loss, token_accuracy, wall_time = _evaluate(
        canonical=canonical,
        params=params,
        model=model,
        config=config,
        mesh=mesh,
        sharded_fns=sharded_fns,
        runtime_state=runtime_state,
        val_loader=loader,
        max_batches=evaluated_batches,
        token_chunk_size=token_chunk_size,
    )
    if not 0.0 < token_accuracy <= 1.0:
        raise RuntimeError(
            "validation token accuracy was not computed by the selected "
            f"evaluation kernel: {token_accuracy}"
        )
    valid_target_tokens = (
        evaluated_batches * batch_size * (max_length - 1)
    )
    evaluated_sequences = evaluated_batches * batch_size
    available_sequences = int(len(dataset))
    result_payload = {
        "cross_entropy": val_loss,
        "perplexity": math.exp(val_loss),
        "token_accuracy": token_accuracy,
        "token_accuracy_pct": 100.0 * token_accuracy,
        "valid_target_tokens": valid_target_tokens,
        "evaluated_sequences": evaluated_sequences,
        "evaluated_batches": evaluated_batches,
        "available_sequences": available_sequences,
        "dropped_incomplete_batch_sequences": (
            available_sequences - available_batches * batch_size
        ),
        "batch_size": batch_size,
        "sequence_length": max_length,
        "max_val_tokens": max_val_tokens,
        "comparable": comparable,
    }
    result_hash = shared.sha256_json(result_payload)
    shared._allgather_equal_bytes(
        jax, bytes.fromhex(result_hash), name="validation result"
    )

    first_param = next(iter(jax.tree.leaves(params)))
    git_info = shared._git_info()
    finished_at = datetime.now(timezone.utc)
    manifest = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_mode": "c4_validation_no_update",
        "kernel_profile": kernel_profile,
        "dawn_git_commit": git_info["commit"],
        "dawn_git_branch": git_info["branch"],
        "working_tree_clean": git_info["working_tree_clean"],
        "source_checkpoint_requested": args.init_from,
        "source_checkpoint_concrete_path": resolved,
        "source_checkpoint_step": int(step),
        "source_checkpoint_resolved_once": True,
        "source_checkpoint_config_hash": config_hash,
        "params_hash": params_hash,
        "params_hash_algorithm": (
            "checkpoint_identity_plus_sharded_param_schema_sha256"
        ),
        "parameter_count": parameter_count,
        "model_version": model_cfg.get("model_version"),
        "pretraining_token_count": (
            checkpoint_metadata.get("consumed_tokens")
            or data_cfg.get("max_train_tokens")
        ),
        "validation_data": val_path,
        "validation_metadata_path": val_meta_path,
        "validation_metadata_hash": val_metadata_hash,
        "validation_metadata": val_metadata,
        "mesh_shape": {"data": mesh_data, "model": mesh_model},
        "checkpoint_mesh_shape": {
            "data": checkpoint_mesh_data,
            "model": checkpoint_mesh_model,
        },
        "global_device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "host_count": int(jax.process_count()),
        "dtype": str(first_param.dtype),
        "token_chunk_size": token_chunk_size,
        "software_versions": {
            name: shared._distribution_version(name)
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "orbax-checkpoint",
                "numpy",
                "gcsfs",
                "fsspec",
            )
        },
        "start_timestamp": started_at.isoformat(),
        "end_timestamp": finished_at.isoformat(),
        "wall_time_sec": wall_time,
        "result_hash": result_hash,
        **result_payload,
    }

    if jax.process_index() == 0:
        canonical._makedirs(args.output_dir)
        shared._write_json(
            canonical,
            shared._output_path(
                canonical, args.output_dir, "validation_summary.json"
            ),
            result_payload,
        )
        shared._write_json(
            canonical,
            shared._output_path(
                canonical, args.output_dir, "run_manifest.json"
            ),
            manifest,
        )
        summary_log = "\n".join(
            (
                f"protocol_name={PROTOCOL_NAME}",
                f"source_checkpoint={resolved}",
                f"source_checkpoint_step={step}",
                f"source_checkpoint_config_hash={config_hash}",
                f"source_checkpoint_params_hash={params_hash}",
                "validation_result="
                + shared.stable_json_dumps(result_payload),
            )
        )
        shared._write_text(
            canonical,
            shared._output_path(
                canonical, args.output_dir, "summary.log"
            ),
            summary_log + "\n",
        )
        print("C4_VALIDATION_COMPLETE " + summary_log.splitlines()[-1], flush=True)

    return {"result": result_payload, "manifest": manifest}


def main() -> None:
    args = parse_args()
    if os.environ.get("DAWN_C4_VALIDATION_MAX_BATCHES"):
        raise RuntimeError(
            "implicit DAWN_C4_VALIDATION_MAX_BATCHES is forbidden; "
            "use --max-batches explicitly"
        )
    run(args)


if __name__ == "__main__":
    main()
