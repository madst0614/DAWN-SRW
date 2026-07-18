#!/usr/bin/env python3
"""Evaluate a frozen DAWN decoder with stock lm-eval 0.4.2 tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dawn.eval.zero_shot_protocol import (  # noqa: E402
    DEFAULT_LENGTH_BUCKETS,
    LM_EVAL_VERSION,
    NUM_FEWSHOT,
    PRIMARY_TASKS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    PreparedRequest,
    build_results_summary,
    csv_header_and_row,
    json_safe,
    normalize_tasks,
    normalize_buckets,
    prepare_causal_request,
    sha256_json,
    stable_json_dumps,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen DAWN JAX/TPU evaluation on the Mamba Table 3 zero-shot "
            "task set using stock lm-eval==0.4.2 definitions."))
    parser.add_argument("--init-from", required=True,
                        help="Orbax run/checkpoints directory or concrete step")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--tasks", default=",".join(PRIMARY_TASKS),
        help="Comma-separated stock task ids; default is the complete primary suite")
    parser.add_argument("--tokenizer", default=None,
                        help="Must match the tokenizer recorded by pretokenization")
    parser.add_argument(
        "--tokenizer-revision", default=None,
        help=("Immutable Hugging Face revision used for this evaluation. "
              "Required when source metadata did not record a revision."))
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument(
        "--eot-token-id", type=int, default=None,
        help=("Only set when verified from source preprocessing. No CLS/SEP "
              "fallback is inferred for bert-base-uncased."))
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Fixed global batch size; no auto/OOM retry")
    parser.add_argument("--length-buckets", default="64,128,256,512")
    parser.add_argument("--token-chunk-size", type=int, default=32768)
    parser.add_argument("--max-gen-toks", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke-test document limit per task")
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--validation-ce-tolerance", type=float, default=5e-5)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--known-validation-loss", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mesh-data", type=int, default=None)
    parser.add_argument("--mesh-model", type=int, default=None)
    return parser.parse_args()


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "<not installed>"


def _require_lm_eval_version() -> str:
    versions = {
        _distribution_version("lm_eval"),
        _distribution_version("lm-eval"),
    }
    versions.discard("<not installed>")
    if versions != {LM_EVAL_VERSION}:
        raise RuntimeError(
            f"lm-eval version mismatch: required={LM_EVAL_VERSION} "
            f"detected={sorted(versions) or '<not installed>'}. Install "
            "requirements_zero_shot_eval.txt; latest/floating harnesses are "
            "not accepted.")
    return LM_EVAL_VERSION


def _allgather_equal_bytes(jax: Any, payload: bytes, *, name: str) -> None:
    from jax.experimental.multihost_utils import process_allgather
    local = np.frombuffer(payload, dtype=np.uint8)
    gathered = np.asarray(process_allgather(local)).reshape(
        int(jax.process_count()), -1)
    values = [bytes(row.tolist()) for row in gathered]
    if any(value != payload for value in values):
        raise RuntimeError(
            f"host {name} mismatch: "
            + repr([value.hex() for value in values]))


def _broadcast_string(canonical: Any, value: Optional[str]) -> str:
    result = canonical._broadcast_str_from_host0(value, max_len=8192)
    if result is None:
        raise RuntimeError("host0 failed to broadcast a required string")
    return str(result)


def _resolve_checkpoint_once(canonical: Any, requested: str) -> Tuple[str, str, int]:
    import jax

    host_value = None
    host_step = None
    if jax.process_index() == 0:
        run_folder, step, found = canonical._resolve_orbax_resume_from(requested)
        if not found or run_folder is None or step is None:
            raise FileNotFoundError(
                "checkpoint could not be pinned to a committed concrete Orbax "
                f"step: requested={requested!r}")
        host_step = int(step)
        host_value = canonical._join_path(
            run_folder, "checkpoints", f"{host_step:012d}")
    resolved = _broadcast_string(canonical, host_value)
    step_text = _broadcast_string(
        canonical, None if host_step is None else str(host_step))
    step = int(step_text)
    name = canonical._path_name(resolved)
    if not name.isdigit() or int(name) != step:
        raise RuntimeError(
            f"resolved checkpoint is not a concrete numeric step: {resolved}")
    if not canonical._orbax_step_is_committed(resolved):
        raise RuntimeError(
            f"resolved checkpoint is incomplete: {resolved} missing "
            "commit_success.txt")
    run_folder = canonical._path_parent(canonical._path_parent(resolved))
    return resolved, run_folder, step


def _restore_metadata(canonical: Any, run_folder: str, step: int) -> Dict[str, Any]:
    checkpoint_root = canonical._join_path(run_folder, "checkpoints")
    metadata = canonical._restore_orbax_metadata(checkpoint_root, step)
    if not metadata:
        raise RuntimeError("checkpoint metadata is empty")
    if not isinstance(metadata.get("full_config"), dict):
        raise RuntimeError(
            "checkpoint metadata is missing full_config; evaluator has no "
            "config fallback")
    return metadata


def _tokenizer_vocab_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    payload = stable_json_dumps(sorted(
        ((str(token), int(index)) for token, index in vocab.items()),
        key=lambda item: (item[1], item[0])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_source_data_metadata(config: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    from utils import data_jax

    train_path = config.get("data", {}).get("bin_train")
    if not train_path:
        raise RuntimeError("checkpoint full_config.data.bin_train is missing")
    meta_path = data_jax._meta_path_for(train_path)
    try:
        source_meta = data_jax._read_json(meta_path)
    except Exception as exc:
        raise RuntimeError(
            "cannot establish source tokenizer provenance because the "
            f"pretokenized data metadata is unreadable: {meta_path}") from exc
    if not isinstance(source_meta, dict) or not source_meta.get("tokenizer"):
        raise RuntimeError(
            f"pretokenized data metadata lacks tokenizer id: {meta_path}")
    return meta_path, source_meta


def _load_tokenizer(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source_meta_path: str,
    source_meta: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    from transformers import AutoTokenizer, __version__ as transformers_version

    source_id = str(source_meta["tokenizer"])
    requested_id = str(args.tokenizer or source_id)
    if requested_id != source_id:
        raise RuntimeError(
            "tokenizer id differs from source preprocessing: "
            f"source={source_id!r} requested={requested_id!r}")
    source_revision = source_meta.get("tokenizer_revision")
    revision = args.tokenizer_revision or source_revision
    if not revision and not Path(requested_id).exists():
        raise RuntimeError(
            "source preprocessing metadata did not record tokenizer_revision. "
            "Pass --tokenizer-revision with an immutable revision; using a "
            "floating latest/main tokenizer is forbidden. "
            f"source_metadata={source_meta_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        requested_id,
        revision=revision,
        use_fast=True,
    )
    local_tokenizer = Path(requested_id).exists()
    resolved_revision = getattr(tokenizer, "init_kwargs", {}).get(
        "_commit_hash")
    if not local_tokenizer:
        resolved_revision = str(resolved_revision or revision or "")
        if re.fullmatch(r"[0-9a-fA-F]{40}", resolved_revision) is None:
            raise RuntimeError(
                "tokenizer revision did not resolve to an immutable 40-hex "
                "Hugging Face commit. Pass a commit SHA, not a branch/tag: "
                f"requested_revision={revision!r} "
                f"resolved_revision={resolved_revision!r}")
    logical_vocab = int(config["model"].get(
        "logical_vocab_size", config["model"]["vocab_size"]))
    source_vocab = int(source_meta.get("vocab_size", logical_vocab))
    if int(tokenizer.vocab_size) != source_vocab or source_vocab != logical_vocab:
        raise RuntimeError(
            "tokenizer/checkpoint vocabulary mismatch: "
            f"tokenizer={tokenizer.vocab_size} source_data={source_vocab} "
            f"checkpoint={logical_vocab}")
    effective_vocab = tokenizer.get_vocab()
    effective_ids = {int(value) for value in effective_vocab.values()}
    if (
        len(effective_vocab) != logical_vocab
        or effective_ids != set(range(logical_vocab))
    ):
        raise RuntimeError(
            "tokenizer contains added, missing, duplicated, or out-of-range "
            "token ids relative to the frozen checkpoint vocabulary: "
            f"effective_vocab={len(effective_vocab)} "
            f"checkpoint_vocab={logical_vocab}")
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    pad_token_id = args.pad_token_id
    if pad_token_id is None:
        if tokenizer_pad is None:
            raise RuntimeError(
                "source tokenizer has no padding token. Pass an explicitly "
                "verified --pad-token-id; evaluator will not add a token.")
        pad_token_id = int(tokenizer_pad)
    elif tokenizer_pad is not None and int(pad_token_id) != int(tokenizer_pad):
        raise RuntimeError(
            "explicit pad token differs from tokenizer policy: "
            f"explicit={pad_token_id} tokenizer={tokenizer_pad}")
    if not 0 <= int(pad_token_id) < logical_vocab:
        raise ValueError("pad_token_id is outside the logical vocabulary")

    eot_token_id = args.eot_token_id
    if eot_token_id is None:
        # EOS/BOS may only be used when the tokenizer itself declares it.  SEP
        # and CLS are intentionally never inferred as causal EOT/BOS.
        declared_eos = getattr(tokenizer, "eos_token_id", None)
        declared_bos = getattr(tokenizer, "bos_token_id", None)
        eot_token_id = (
            int(declared_eos) if declared_eos is not None
            else (int(declared_bos) if declared_bos is not None else None))
    if eot_token_id is not None and not 0 <= int(eot_token_id) < logical_vocab:
        raise ValueError("eot_token_id is outside the logical vocabulary")

    metadata = {
        "id": requested_id,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "local_path": local_tokenizer,
        "transformers_version": transformers_version,
        "source_metadata_path": source_meta_path,
        "source_metadata_hash": sha256_json(source_meta),
        "vocab_size": int(tokenizer.vocab_size),
        "effective_vocab_size": len(effective_vocab),
        "vocab_hash": _tokenizer_vocab_hash(tokenizer),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "eot_token_id": eot_token_id,
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": int(pad_token_id),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "cls_token": getattr(tokenizer, "cls_token", None),
        "cls_token_id": getattr(tokenizer, "cls_token_id", None),
        "sep_token": getattr(tokenizer, "sep_token", None),
        "sep_token_id": getattr(tokenizer, "sep_token_id", None),
        "do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "add_special_tokens": False,
        "automatic_special_token_insertion": False,
    }
    return tokenizer, metadata


def _abstract_params(canonical: Any, model: Any, config: Mapping[str, Any], mesh: Any):
    import jax
    import jax.numpy as jnp

    max_length = int(config["model"]["max_seq_len"])
    dummy = jax.ShapeDtypeStruct((1, max_length), jnp.int32)

    def initialize(input_ids):
        return model.init(
            {"params": jax.random.PRNGKey(0),
             "dropout": jax.random.PRNGKey(1)},
            input_ids,
            deterministic=True,
        )["params"]

    abstract = jax.eval_shape(initialize, dummy)
    model_version = str(config["model"].get("model_version", ""))
    operation_space_enabled = bool(config.get("training", {}).get(
        "operation_space", {}).get("enabled", False))
    shardings = canonical.get_param_shardings(
        abstract,
        mesh,
        model_version=model_version,
        operation_space_enabled=operation_space_enabled,
        vocab_size_padded=config["model"].get("vocab_size_padded"),
    )
    target = jax.tree.map(
        lambda shape, sharding: jax.ShapeDtypeStruct(
            shape=shape.shape, dtype=shape.dtype, sharding=sharding),
        abstract,
        shardings,
    )
    return abstract, target


def _restore_params_only(
    canonical: Any,
    run_folder: str,
    step: int,
    target_params: Any,
    metadata: Mapping[str, Any],
    mesh: Any,
):
    import jax

    checkpoint_root = canonical._join_path(run_folder, "checkpoints")
    manager = canonical._create_orbax_checkpoint_manager(
        checkpoint_root, create=False, read_only=True)
    try:
        restored = manager.restore(
            int(step),
            args=canonical.ocp.args.Composite(
                state=canonical.ocp.args.PyTreeRestore(
                    item={"params": target_params},
                    partial_restore=True,
                ),
                metadata=canonical.ocp.args.JsonRestore(),
            ),
        )
    finally:
        manager.close()
    state = canonical._composite_item(restored, "state")
    restored_metadata = canonical._composite_item(restored, "metadata")
    if not isinstance(state, dict) or set(state) != {"params"}:
        raise RuntimeError(
            "params-only Orbax restore returned unexpected state keys; "
            f"optimizer loading is forbidden: {sorted(state or {})}")
    if json_safe(restored_metadata) != json_safe(metadata):
        raise RuntimeError("metadata changed between pinning and params restore")
    params = state["params"]
    if canonical._is_v417x_version(
            metadata["full_config"]["model"].get("model_version")):
        canonical._validate_v4171_checkpoint_param_schema(
            params, target_params)
    jax.block_until_ready(params)
    for leaf in jax.tree.leaves(params):
        if not canonical._is_full_mesh_sharding(leaf, mesh):
            raise RuntimeError(
                "restored parameter is not resident on the full checkpoint mesh")
    return params


def _param_schema(params: Any) -> Dict[str, Any]:
    import jax

    def sharding_schema(value: Any) -> Dict[str, Any]:
        sharding = getattr(value, "sharding", None)
        return {
            # Device ids and addressable-device reprs are process-local.  A
            # PartitionSpec plus mesh shape/axis names is the stable identity
            # that every host can compare without gathering parameter data.
            "spec": str(getattr(sharding, "spec", None)),
            "mesh_axes": list(getattr(
                getattr(sharding, "mesh", None), "axis_names", ())),
            "mesh_shape": dict(getattr(
                getattr(sharding, "mesh", None), "shape", {})),
            "memory_kind": getattr(sharding, "memory_kind", None),
        }

    return {
        jax.tree_util.keystr(path): {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sharding": sharding_schema(value),
        }
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]
    }


def _parameter_count(params: Any) -> int:
    import jax
    return int(sum(np.prod(value.shape, dtype=np.int64)
                   for value in jax.tree.leaves(params)))


def _load_fixed_validation_rows(config: Mapping[str, Any], count: int) -> np.ndarray:
    from utils import data_jax

    model_length = int(config["model"]["max_seq_len"])
    val_path = config.get("data", {}).get("bin_val")
    if not val_path:
        raise RuntimeError(
            "validation CE cross-check requires full_config.data.bin_val")
    dataset = data_jax._build_dataset(
        val_path,
        model_length,
        max_sequences=int(count),
        local_cache_dir=config.get("data", {}).get("local_cache_dir"),
        evict_previous_cache=False,
    )
    rows = dataset.get_batch(0, int(count))
    if rows is None or rows.shape != (int(count), model_length):
        raise RuntimeError(
            "fixed validation CE batch has unexpected shape: "
            f"actual={None if rows is None else rows.shape} "
            f"expected={(int(count), model_length)}")
    return np.asarray(rows, dtype=np.int32)


def _validation_ce_cross_check(
    *,
    canonical: Any,
    scorer: Any,
    model: Any,
    params: Any,
    mesh: Any,
    base_sharded_fns: Any,
    runtime_state: Mapping[str, Any],
    config: Mapping[str, Any],
    global_batch_size: int,
    tolerance: float,
) -> Dict[str, float]:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    rows = _load_fixed_validation_rows(config, global_batch_size)
    local_batch = global_batch_size // int(jax.process_count())
    start = int(jax.process_index()) * local_batch
    stop = start + local_batch
    sharding = NamedSharding(mesh, P("data", None))
    shape = tuple(rows.shape)
    inputs = canonical.shard_to_mesh(rows[start:stop], sharding, shape)
    masks_np = np.ones_like(rows, dtype=np.int32)
    masks = canonical.shard_to_mesh(masks_np[start:stop], sharding, shape)
    labels = inputs
    eval_step = canonical.create_eval_step(
        model,
        sharded_fns=base_sharded_fns,
        runtime_state=runtime_state,
        ce_token_chunk_size=int(config.get("training", {}).get(
            "ce_token_chunk_size", 32768)),
    )
    existing_loss, _, valid_count = eval_step(
        params, inputs, labels, masks, jnp.asarray(0, jnp.int32))
    existing_loss, valid_count = jax.device_get((existing_loss, valid_count))
    existing_loss = float(np.asarray(existing_loss))
    valid_count = int(np.asarray(valid_count))
    expected_count = global_batch_size * (rows.shape[1] - 1)
    if valid_count != expected_count:
        raise RuntimeError(
            "existing validation path returned an unexpected token count: "
            f"actual={valid_count} expected={expected_count}")

    buckets = normalize_buckets(DEFAULT_LENGTH_BUCKETS, rows.shape[1])
    prepared = [
        prepare_causal_request(
            ordinal=index,
            task="validation_ce_cross_check",
            doc_id=index,
            choice_index=0,
            context="<fixed-source-token>",
            continuation="<fixed-source-token-sequence>",
            context_ids=[int(row[0])],
            continuation_ids=[int(x) for x in row[1:]],
            max_length=rows.shape[1],
            buckets=buckets,
        )
        for index, row in enumerate(rows)
    ]
    scored = scorer.score_requests(
        prepared,
        pad_token_id=int(config.get("_eval_pad_token_id", 0)))
    adapter_tokens = sum(item.scored_tokens for item in scored)
    if adapter_tokens != expected_count:
        raise RuntimeError(
            "adapter validation token count mismatch: "
            f"actual={adapter_tokens} expected={expected_count}")
    adapter_ce = -sum(item.loglikelihood for item in scored) / adapter_tokens
    difference = abs(float(adapter_ce) - existing_loss)
    if difference > float(tolerance):
        raise RuntimeError(
            "validation CE cross-check failed: "
            f"adapter={adapter_ce:.9f} existing={existing_loss:.9f} "
            f"abs_diff={difference:.9g} tolerance={tolerance:.9g}")
    return {
        "adapter_cross_entropy": float(adapter_ce),
        "existing_validation_cross_entropy": existing_loss,
        "valid_target_token_count": adapter_tokens,
        "absolute_difference": difference,
        "tolerance": float(tolerance),
    }


def _task_object(value: Any) -> Any:
    return value[1] if isinstance(value, tuple) else value


def _dataset_fingerprints(task: Any) -> Dict[str, Any]:
    dataset = getattr(task, "dataset", None)
    if dataset is None:
        return {}
    output = {}
    if hasattr(dataset, "items"):
        for split, value in dataset.items():
            info = getattr(value, "info", None)
            dataset_version = getattr(info, "version", None)
            output[str(split)] = {
                "fingerprint": getattr(value, "_fingerprint", None),
                "rows": len(value),
                "builder_name": getattr(info, "builder_name", None),
                "config_name": getattr(info, "config_name", None),
                "dataset_version": (
                    None if dataset_version is None else str(dataset_version)),
                "download_size": getattr(info, "download_size", None),
                "dataset_size": getattr(info, "dataset_size", None),
            }
    else:
        info = getattr(dataset, "info", None)
        dataset_version = getattr(info, "version", None)
        output["dataset"] = {
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "rows": len(dataset),
            "builder_name": getattr(info, "builder_name", None),
            "config_name": getattr(info, "config_name", None),
            "dataset_version": (
                None if dataset_version is None else str(dataset_version)),
            "download_size": getattr(info, "download_size", None),
            "dataset_size": getattr(info, "dataset_size", None),
        }
    return output


def _load_stock_tasks(
    tasks: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from lm_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(verbosity="INFO")
    task_dict = get_task_dict(list(tasks), manager)
    if set(task_dict) != set(tasks):
        raise RuntimeError(
            "stock task lookup mismatch: "
            f"requested={list(tasks)} found={sorted(task_dict)}")
    provenance = {}
    for name in tasks:
        task = _task_object(task_dict[name])
        if task is None:
            raise RuntimeError(f"stock task {name!r} resolved to None")
        task.set_config(key="num_fewshot", value=NUM_FEWSHOT)
        config = task.dump_config()
        if config.get("task") != name:
            raise RuntimeError(
                f"stock task identity mismatch: key={name} config={config.get('task')}")
        provenance[name] = {
            "version": json_safe(getattr(task, "VERSION", None)),
            "config_hash": sha256_json(config),
            "config": json_safe(config),
            "dataset_path": getattr(task, "DATASET_PATH", None),
            "dataset_name": getattr(task, "DATASET_NAME", None),
            "dataset_kwargs": json_safe(config.get("dataset_kwargs", {})),
            "dataset_fingerprints": _dataset_fingerprints(task),
            "evaluation_split": (
                task.config.test_split or task.config.validation_split),
            "expected_evaluation_rows": len(task.eval_docs),
        }
    return task_dict, provenance


def _validate_evaluated_row_counts(
    raw: Mapping[str, Any], task_provenance: Mapping[str, Any],
    limit: Optional[int], tasks: Sequence[str],
) -> None:
    results = raw.get("results", {})
    for task in tasks:
        metrics = results.get(task)
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"final task result missing: {task}")
        candidates = [key for key in metrics if key == "samples"]
        if len(candidates) != 1:
            raise RuntimeError(
                f"task {task} result lacks exact sample count: {sorted(metrics)}")
        actual = int(metrics["samples"])
        expected_full = int(task_provenance[task]["expected_evaluation_rows"])
        expected = min(int(limit), expected_full) if limit is not None else expected_full
        if actual != expected:
            raise RuntimeError(
                "sample count differs from evaluation split: "
                f"task={task} actual={actual} expected={expected}")


def _flatten_sample_response(value: Any) -> Optional[Tuple[float, bool]]:
    current = value
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    if isinstance(current, (list, tuple)) and len(current) == 2:
        try:
            return float(current[0]), bool(current[1])
        except (TypeError, ValueError):
            return None
    return None


def _build_sample_records(
    raw_results: Mapping[str, Any], traces: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
) -> Sequence[Dict[str, Any]]:
    trace_groups: Dict[Tuple[str, str], list] = {}
    for trace in traces:
        task = str(trace["task"])
        if task not in tasks:
            continue
        key = (task, stable_json_dumps(trace["document_id"]))
        trace_groups.setdefault(key, []).append(dict(trace))
    raw_samples = raw_results.get("samples", {})
    records = []
    for task in tasks:
        for sample in raw_samples.get(task, []):
            doc_id = sample.get("doc_id")
            key = (task, stable_json_dumps(json_safe(doc_id)))
            candidates = sorted(
                trace_groups.get(key, []),
                key=lambda value: int(value["choice_index"]))
            arguments = sample.get("arguments", [])
            if not candidates and arguments:
                # A hard failure is preferable to sample logs that cannot
                # independently reproduce the harness input.
                raise RuntimeError(
                    f"sample trace missing: task={task} doc_id={doc_id}")
            continuations = [item["continuation"] for item in candidates]
            scores = [float(item["summed_loglikelihood"]) for item in candidates]
            task_config = raw_results.get("configs", {}).get(task, {})
            delimiter = str(task_config.get("target_delimiter", " "))
            choice_texts = []
            for text in continuations:
                if delimiter and not text.startswith(delimiter):
                    raise RuntimeError(
                        "sample continuation does not begin with the stock "
                        f"target delimiter: task={task} delimiter={delimiter!r} "
                        f"continuation={text!r}")
                choice_texts.append(
                    text[len(delimiter):] if delimiter else text)
            if any(len(text) == 0 for text in choice_texts):
                raise RuntimeError(
                    f"stock choice has zero character length: task={task} "
                    f"doc_id={doc_id}")
            normalizers = [len(text) for text in choice_texts]
            normalized = [score / length for score, length in zip(
                scores, normalizers)]
            use_normalized = task in {
                "hellaswag", "piqa", "arc_easy", "arc_challenge"}
            selected = (
                int(np.argmax(normalized if use_normalized else scores))
                if candidates else None)
            target = json_safe(sample.get("target"))
            gold = target if isinstance(target, int) else (0 if task == "lambada_openai" else target)
            metric_name = (
                "acc_norm" if use_normalized else "acc")
            correct = sample.get(metric_name)
            if correct is None:
                matching = [
                    value for key_name, value in sample.items()
                    if key_name == metric_name or key_name.startswith(metric_name + ",")]
                correct = matching[0] if len(matching) == 1 else None
            recomputed_correct = None
            if task == "lambada_openai" and len(candidates) == 1:
                recomputed_correct = bool(candidates[0]["is_greedy"])
            elif isinstance(gold, int) and selected is not None:
                recomputed_correct = selected == gold
            if (
                correct is not None
                and recomputed_correct is not None
                and bool(correct) != bool(recomputed_correct)
            ):
                raise RuntimeError(
                    "sample score recomputation differs from stock harness: "
                    f"task={task} doc_id={doc_id} selected={selected} "
                    f"gold={gold} harness_correct={correct}")
            records.append({
                "task": task,
                "document_id": json_safe(doc_id),
                "formatted_context": (
                    candidates[0]["formatted_context"] if candidates else None),
                "candidate_continuations": continuations,
                "encoded_lengths": [item["encoded_length"] for item in candidates],
                "continuation_token_lengths": [
                    item["continuation_tokens"] for item in candidates],
                "candidate_summed_loglikelihoods": scores,
                "candidate_normalized_scores": normalized,
                "normalization": (
                    "stock_lm_eval_choice_character_length" if use_normalized
                    else "none"),
                "selected_candidate": selected,
                "gold_candidate": gold,
                "correct": None if correct is None else bool(correct),
                "truncated": any(bool(item["truncated"]) for item in candidates),
            })
    expected = sum(len(raw_samples.get(task, [])) for task in tasks)
    if len(records) != expected:
        raise RuntimeError(
            f"sample output count mismatch: records={len(records)} expected={expected}")
    return records


def _git_info() -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        proc = subprocess.run(
            ["git", *args], cwd=str(PROJECT_ROOT), capture_output=True,
            text=True, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else None
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_clean": status == "",
    }


def _known_validation_loss(args: argparse.Namespace, metadata: Mapping[str, Any]) -> Optional[float]:
    if args.known_validation_loss is not None:
        return float(args.known_validation_loss)
    for key in ("val_loss", "best_val_loss"):
        value = metadata.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _write_text(canonical: Any, path: str, text: str) -> None:
    with canonical._open_file(path, "w") as handle:
        handle.write(text)


def _write_json(canonical: Any, path: str, value: Any) -> None:
    _write_text(canonical, path, json.dumps(
        json_safe(value), indent=2, ensure_ascii=False, sort_keys=True))


def _output_path(canonical: Any, output_dir: str, name: str) -> str:
    return canonical._join_path(output_dir, name)


def _format_output_dir(
    canonical: Any, args: argparse.Namespace, model_name: str, step: int
) -> str:
    import jax
    host_value = None
    if jax.process_index() == 0:
        host_value = args.output_dir
        if host_value is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_model = "".join(
                ch if ch.isalnum() or ch in "._-" else "_"
                for ch in model_name)
            host_value = str(
                PROJECT_ROOT / "outputs" / "zero_shot" /
                f"{safe_model}_step{step:012d}_{stamp}")
    output_dir = _broadcast_string(canonical, host_value)
    canonical._makedirs(output_dir)
    return output_dir


def _log_required_startup(
    requested: str, resolved: str, step: int, config_hash: str
) -> str:
    lines = [
        f"source_checkpoint_requested={requested}",
        f"source_checkpoint_resolved={resolved}",
        f"source_checkpoint_step={step}",
        "source_checkpoint_resolved_once=true",
        "evaluation_mode=zero_shot_no_update",
        f"protocol_name={PROTOCOL_NAME}",
        f"source_checkpoint_config_hash={config_hash}",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    start_timestamp = datetime.now(timezone.utc)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tasks = normalize_tasks(
        [value.strip() for value in str(args.tasks).split(",") if value.strip()])

    # Importing JAX is safe before initialization; device/backend access is not.
    import jax
    from scripts import train_jax as canonical

    canonical._maybe_initialize_jax_distributed()
    canonical._require_orbax_checkpoint_compat()
    lm_eval_version = _require_lm_eval_version()
    resolved, run_folder, step = _resolve_checkpoint_once(
        canonical, args.init_from)
    metadata = _restore_metadata(canonical, run_folder, step)
    config = deepcopy(metadata["full_config"])
    canonical._require_resume_full_config(config)
    config_hash = canonical._config_sha256(config)
    if int(metadata.get("global_step", -1)) != step:
        raise RuntimeError(
            "checkpoint metadata step differs from pinned directory: "
            f"directory_step={step} metadata_step={metadata.get('global_step')}")
    for key in ("full_config_sha256", "selected_full_config_sha256"):
        recorded_hash = metadata.get(key)
        if recorded_hash is not None and str(recorded_hash) != config_hash:
            raise RuntimeError(
                f"checkpoint metadata {key} differs from restored full_config: "
                f"recorded={recorded_hash} computed={config_hash}")
    _allgather_equal_bytes(
        jax, bytes.fromhex(config_hash), name="checkpoint config hash")
    startup_log = _log_required_startup(
        args.init_from, resolved, step, config_hash)

    model_cfg = config["model"]
    training_cfg = config["training"]
    max_length = int(model_cfg.get("max_seq_len", 0))
    if max_length < 2:
        raise RuntimeError("checkpoint model.max_seq_len is missing/invalid")
    requested_buckets = tuple(
        int(value.strip()) for value in args.length_buckets.split(",")
        if value.strip())
    length_buckets = normalize_buckets(requested_buckets, max_length)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when specified")
    if args.limit is None and os.environ.get("DAWN_ZERO_SHOT_LIMIT"):
        raise RuntimeError(
            "full run detected DAWN_ZERO_SHOT_LIMIT in the environment; "
            "implicit limits are forbidden")

    source_meta_path, source_meta = _load_source_data_metadata(config)
    tokenizer, tokenizer_meta = _load_tokenizer(
        args, config, source_meta_path, source_meta)
    config["_eval_pad_token_id"] = tokenizer_meta["pad_token_id"]
    tokenizer_hash = sha256_json(tokenizer_meta)
    _allgather_equal_bytes(
        jax, bytes.fromhex(tokenizer_hash), name="tokenizer hash")
    if jax.process_index() == 0:
        print("tokenizer_policy=" + stable_json_dumps(tokenizer_meta), flush=True)

    checkpoint_mesh_model = int(training_cfg.get("mesh_model", 0))
    checkpoint_mesh_data = int(training_cfg.get("mesh_data", 0))
    if checkpoint_mesh_model <= 0 or checkpoint_mesh_data <= 0:
        raise RuntimeError(
            "checkpoint full_config must contain materialized positive "
            "training.mesh_model and training.mesh_data")
    mesh_model = int(args.mesh_model or checkpoint_mesh_model)
    if mesh_model != checkpoint_mesh_model:
        raise RuntimeError(
            "zero-shot mesh_model must preserve checkpoint model-axis "
            f"sharding: checkpoint={checkpoint_mesh_model} requested={mesh_model}")
    mesh_data = int(args.mesh_data or checkpoint_mesh_data)
    if mesh_data <= 0:
        raise RuntimeError("effective zero-shot mesh_data must be positive")
    if mesh_model * mesh_data != jax.device_count():
        raise RuntimeError(
            "effective zero-shot mesh differs from current topology: "
            f"effective={mesh_data}x{mesh_model} "
            f"devices={jax.device_count()}")
    training_cfg["mesh_model"] = mesh_model
    training_cfg["mesh_data"] = mesh_data
    canonical._maybe_materialize_vocab_parallel_config(config)
    mesh = canonical.create_mesh(mesh_data, mesh_model)
    model = canonical.build_model_from_config(config)
    abstract_params, target_params = _abstract_params(
        canonical, model, config, mesh)
    params = _restore_params_only(
        canonical, run_folder, step, target_params, metadata, mesh)
    del target_params
    param_schema = _param_schema(params)
    parameter_count = _parameter_count(params)
    params_hash = sha256_json({
        "algorithm": "checkpoint_identity_plus_sharded_param_schema_sha256",
        "checkpoint": resolved,
        "step": step,
        "schema": param_schema,
    })
    _allgather_equal_bytes(jax, bytes.fromhex(params_hash), name="params hash")
    print(f"source_checkpoint_params_hash={params_hash}", flush=True)

    runtime_state = canonical._checkpoint_final_runtime(config, resolved)
    base_sharded_fns = canonical.build_canonical_sharded_fns(
        config, mesh, for_eval=True, analysis=False)
    from dawn.eval.jax_runtime import JaxDawnScorer
    from dawn.eval.lm_eval_dawn_adapter import DawnJaxLM

    scorer = JaxDawnScorer(
        model=model,
        params=params,
        mesh=mesh,
        config=config,
        runtime_state=runtime_state,
        base_sharded_fns=base_sharded_fns,
        global_batch_size=args.batch_size,
        token_chunk_size=args.token_chunk_size,
    )
    validation_cross_check = _validation_ce_cross_check(
        canonical=canonical,
        scorer=scorer,
        model=model,
        params=params,
        mesh=mesh,
        base_sharded_fns=base_sharded_fns,
        runtime_state=runtime_state,
        config=config,
        global_batch_size=args.batch_size,
        tolerance=args.validation_ce_tolerance,
    )
    print(
        "validation_ce_cross_check="
        + stable_json_dumps(validation_cross_check), flush=True)
    scorer.consume_runtime_delta()

    adapter = DawnJaxLM(
        tokenizer=tokenizer,
        scorer=scorer,
        max_length=max_length,
        length_buckets=length_buckets,
        pad_token_id=int(tokenizer_meta["pad_token_id"]),
        eot_token_id=tokenizer_meta["eot_token_id"],
        max_gen_toks=args.max_gen_toks,
    )
    task_dict, task_provenance = _load_stock_tasks(tasks)
    task_hash = sha256_json(task_provenance)
    _allgather_equal_bytes(
        jax, bytes.fromhex(task_hash), name="task config/dataset hash")

    from lm_eval import evaluator
    raw_results = evaluator.evaluate(
        lm=adapter,
        task_dict=task_dict,
        limit=args.limit,
        cache_requests=False,
        rewrite_requests_cache=False,
        bootstrap_iters=args.bootstrap_iters,
        write_out=False,
        log_samples=True,
        verbosity="INFO",
    )
    if raw_results is None:
        raise RuntimeError("lm-eval returned no results")
    _validate_evaluated_row_counts(
        raw_results, task_provenance, args.limit, tasks)
    raw_results = dict(raw_results)
    raw_results["config"] = {
        "model": type(adapter).__name__,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "bootstrap_iters": args.bootstrap_iters,
        "num_fewshot": NUM_FEWSHOT,
    }
    raw_results["lm_eval_version"] = lm_eval_version
    raw_results["protocol_name"] = PROTOCOL_NAME
    raw_results["protocol_version"] = PROTOCOL_VERSION
    raw_results["source_checkpoint"] = resolved

    host_result_hash = sha256_json(raw_results)
    _allgather_equal_bytes(
        jax, bytes.fromhex(host_result_hash), name="final harness result")
    comparable = args.limit is None and tasks == PRIMARY_TASKS
    model_name = args.model_name or str(model_cfg.get("model_version", "DAWN"))
    validation_loss = _known_validation_loss(args, metadata)
    task_runtime = adapter.runtime_stats()
    summary = build_results_summary(
        raw_results,
        model=model_name,
        step=step,
        validation_loss=validation_loss,
        comparable=comparable,
        task_runtime=task_runtime,
        tasks=tasks,
    )
    summary["validation_ce_cross_check"] = validation_cross_check
    samples = _build_sample_records(
        raw_results, adapter.sample_traces(), tasks)
    output_dir = _format_output_dir(canonical, args, model_name, step)

    end_timestamp = datetime.now(timezone.utc)
    git_info = _git_info()
    first_param = next(iter(jax.tree.leaves(params)))
    task_versions = {
        task: task_provenance[task]["version"] for task in tasks}
    task_config_hashes = {
        task: task_provenance[task]["config_hash"] for task in tasks}
    dataset_info = {
        task: {
            "dataset_path": task_provenance[task]["dataset_path"],
            "dataset_name": task_provenance[task]["dataset_name"],
            "dataset_kwargs": task_provenance[task]["dataset_kwargs"],
            "fingerprints": task_provenance[task]["dataset_fingerprints"],
            "evaluation_split": task_provenance[task]["evaluation_split"],
        }
        for task in tasks
    }
    manifest = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_mode": "zero_shot_no_update",
        "num_fewshot": NUM_FEWSHOT,
        "tasks": list(tasks),
        "lm_eval_version": lm_eval_version,
        "software_versions": {
            name: _distribution_version(name)
            for name in (
                "jax", "jaxlib", "flax", "orbax-checkpoint", "lm-eval",
                "transformers", "tokenizers", "huggingface-hub", "datasets",
                "pyarrow", "fsspec", "gcsfs")
        },
        "lm_eval_task_versions": task_versions,
        "task_config_hashes": task_config_hashes,
        "dataset_fingerprints_revisions": dataset_info,
        "dawn_git_commit": git_info["commit"],
        "dawn_git_branch": git_info["branch"],
        "working_tree_clean": git_info["working_tree_clean"],
        "source_checkpoint_requested": args.init_from,
        "source_checkpoint_concrete_path": resolved,
        "source_checkpoint_step": step,
        "source_checkpoint_resolved_once": True,
        "source_checkpoint_config_hash": config_hash,
        "params_hash": params_hash,
        "params_hash_algorithm": (
            "checkpoint_identity_plus_sharded_param_schema_sha256"),
        "parameter_count": parameter_count,
        "model_version": model_cfg.get("model_version"),
        "pretraining_token_count": (
            metadata.get("consumed_tokens")
            or config.get("data", {}).get("max_train_tokens")),
        "known_validation_loss": validation_loss,
        "validation_ce_cross_check": validation_cross_check,
        "tokenizer": tokenizer_meta,
        "vocab_size": tokenizer_meta["vocab_size"],
        "max_sequence_length": max_length,
        "dtype": str(first_param.dtype),
        "mesh_shape": {"data": mesh_data, "model": mesh_model},
        "checkpoint_mesh_shape": {
            "data": checkpoint_mesh_data, "model": checkpoint_mesh_model},
        "global_device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "host_count": int(jax.process_count()),
        "batch_size": int(args.batch_size),
        "length_buckets": list(length_buckets),
        "token_chunk_size": int(args.token_chunk_size),
        "start_timestamp": start_timestamp.isoformat(),
        "end_timestamp": end_timestamp.isoformat(),
        "limit": args.limit,
        "comparable": comparable,
        "smoke_test_only": not comparable,
        "host_result_hash": host_result_hash,
    }

    if jax.process_index() == 0:
        _write_json(canonical, _output_path(
            canonical, output_dir, "results_harness_raw.json"), raw_results)
        _write_json(canonical, _output_path(
            canonical, output_dir, "results_summary.json"), summary)
        header, row = csv_header_and_row(summary)
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, lineterminator="\n")
        writer.writerow(header)
        writer.writerow(row)
        _write_text(canonical, _output_path(
            canonical, output_dir, "results_summary.csv"), csv_buffer.getvalue())
        sample_text = "".join(
            json.dumps(json_safe(sample), ensure_ascii=False, sort_keys=True)
            + "\n" for sample in samples)
        _write_text(canonical, _output_path(
            canonical, output_dir, "samples.jsonl"), sample_text)
        _write_json(canonical, _output_path(
            canonical, output_dir, "run_manifest.json"), manifest)
        log_lines = [
            startup_log,
            "tokenizer_policy=" + stable_json_dumps(tokenizer_meta),
            f"source_checkpoint_params_hash={params_hash}",
            "validation_ce_cross_check="
            + stable_json_dumps(validation_cross_check),
        ]
        for task in tasks:
            log_lines.append(
                f"task_runtime[{task}]="
                + stable_json_dumps(task_runtime.get(task, {})))
        log_lines.extend((
            f"host_result_hash={host_result_hash}",
            f"comparable={str(comparable).lower()}",
            f"smoke_test_only={str(not comparable).lower()}",
            f"output_dir={output_dir}",
        ))
        _write_text(canonical, _output_path(
            canonical, output_dir, "eval.log"), "\n".join(log_lines) + "\n")
        print(json.dumps(summary["table"], indent=2), flush=True)
        print(f"output_dir={output_dir}", flush=True)
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices("zero_shot_eval_outputs_written")
    return {
        "summary": summary,
        "manifest": manifest,
        "raw_results": raw_results,
        "output_dir": output_dir,
    }


def main() -> None:
    run_evaluation(_parse_args())


if __name__ == "__main__":
    main()
