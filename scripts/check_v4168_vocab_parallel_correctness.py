"""Correctness checker for V4168 vocab-parallel embedding and CE.

This script compares the V4168 row-sharded tied embedding path against a
dense tied-embedding reference, and optionally checks that train/eval
sharded_fns agree on the same real batch.
"""

import argparse
from copy import deepcopy
import inspect
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import train_jax as train_lib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from models.vocab_parallel import (  # noqa: E402
    make_vocab_parallel_ce,
    make_vocab_parallel_embedding,
)


class CheckFailure(AssertionError):
    """Raised for an expected correctness-check mismatch."""


def _is_host0() -> bool:
    return jax.process_index() == 0


def _log(message: str) -> None:
    if _is_host0():
        print(message, flush=True)


def _fail(section: str, message: str) -> None:
    _log(f"[FAIL] {section}: {message}")
    raise CheckFailure(f"{section}: {message}")


def _batch_seq_len(args, cfg):
    return (
        int(args.seq_len)
        if args.seq_len is not None
        else int(cfg["model"]["max_seq_len"])
    )


def _synthetic_seq_len(args, cfg):
    return (
        int(args.seq_len)
        if args.seq_len is not None
        else min(int(cfg["model"]["max_seq_len"]), 128)
    )


def _as_float_np(value: Any) -> np.ndarray:
    arr = np.asarray(jax.device_get(value))
    if arr.dtype == np.dtype("O"):
        arr = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        return arr
    return arr.astype(np.float64)


def _scalar_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


def _scalar_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def _array_stats(ref: Any, got: Any) -> Dict[str, float]:
    ref_np = _as_float_np(ref)
    got_np = _as_float_np(got)
    diff = got_np.astype(np.float64) - ref_np.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    ref_norm = float(np.linalg.norm(ref_np.astype(np.float64).ravel()))
    diff_norm = float(np.linalg.norm(diff.ravel()))
    rel_l2 = diff_norm / max(ref_norm, 1.0e-12)
    ref_max = float(np.max(np.abs(ref_np))) if ref_np.size else 0.0
    return {
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "ref_max": ref_max,
    }


def _check_array(
    section: str,
    ref: Any,
    got: Any,
    atol: float,
    rtol: float,
) -> Dict[str, float]:
    stats = _array_stats(ref, got)
    ref_np = _as_float_np(ref)
    got_np = _as_float_np(got)
    if not np.allclose(got_np, ref_np, atol=atol, rtol=rtol):
        _fail(
            section,
            "max_abs={max_abs:.6e} rel_l2={rel_l2:.6e} "
            "ref_max={ref_max:.6e} atol={atol:.6e} rtol={rtol:.6e}".format(
                **stats, atol=atol, rtol=rtol
            ),
        )
    return stats


def _check_scalar(
    section: str,
    ref: Any,
    got: Any,
    atol: float,
    rtol: float,
) -> Dict[str, float]:
    ref_f = _scalar_float(ref)
    got_f = _scalar_float(got)
    abs_diff = abs(got_f - ref_f)
    rel = abs_diff / max(abs(ref_f), 1.0e-12)
    if not np.isclose(got_f, ref_f, atol=atol, rtol=rtol):
        _fail(
            section,
            f"ref={ref_f:.9g} got={got_f:.9g} abs={abs_diff:.6e} "
            f"rel={rel:.6e} atol={atol:.6e} rtol={rtol:.6e}",
        )
    return {
        "ref": ref_f,
        "got": got_f,
        "abs": abs_diff,
        "rel": rel,
    }


def dense_reference_embedding(input_ids, full_embedding):
    return full_embedding[input_ids]


def dense_ce_reference(
    shift_x,
    full_embedding,
    labels,
    valid_mask,
    logical_vocab_size,
    compute_accuracy=True,
):
    emb = full_embedding[:logical_vocab_size]
    logits = shift_x @ emb.T
    safe_labels = jnp.where(valid_mask, labels, 0)
    target = jnp.take_along_axis(
        logits, safe_labels[..., None], axis=-1).squeeze(-1)
    token_ce = jax.nn.logsumexp(logits, axis=-1) - target
    token_ce = jnp.where(valid_mask, token_ce, 0.0)
    valid_count = valid_mask.astype(jnp.int32).sum()
    loss = token_ce.sum() / (valid_count.astype(jnp.float32) + 1e-8)
    if compute_accuracy:
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        correct = ((pred == labels) & valid_mask).astype(jnp.int32).sum()
    else:
        correct = jnp.array(0, dtype=jnp.int32)
    return loss, token_ce, correct, valid_count


def _resolve_config_path(path_arg: str):
    if path_arg.startswith("gs://"):
        return path_arg
    candidate = PROJECT_ROOT / path_arg
    if candidate.exists():
        return candidate
    raw = Path(path_arg)
    if raw.exists():
        return raw
    if train_lib._file_exists(path_arg):
        return path_arg
    raise FileNotFoundError(f"Config file not found: {candidate}")


def _load_config(args) -> Tuple[Dict[str, Any], Any]:
    config_path = _resolve_config_path(args.config)
    cfg = train_lib.load_config(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config must load to a mapping.")
    cfg = deepcopy(cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.logical_vocab_size is not None:
        cfg["model"]["logical_vocab_size"] = int(args.logical_vocab_size)
    token_chunk_size = (
        int(args.token_chunk_size)
        if args.token_chunk_size is not None
        else int(cfg["training"].get("ce_token_chunk_size", 32768))
    )
    if token_chunk_size <= 0:
        raise ValueError(
            f"training.ce_token_chunk_size must be > 0, got {token_chunk_size}")
    cfg["training"]["ce_token_chunk_size"] = token_chunk_size
    return cfg, config_path


def _maybe_set_vocab_from_data(cfg, args, seq_len: int) -> Tuple[Any, Any, int]:
    from utils.data_jax import load_data

    batch_size = int(cfg["training"]["batch_size"])
    train_loader, val_loader, vocab_size = load_data(
        cfg["data"],
        max_length=seq_len,
        batch_size=batch_size,
        n_devices=1,
        n_hosts=jax.process_count(),
        host_id=jax.process_index(),
    )
    if args.logical_vocab_size is None:
        cfg["model"]["vocab_size"] = int(vocab_size)
    return train_loader, val_loader, int(vocab_size)


def _materialize_and_validate(
    cfg,
    config_path,
    token_chunk_size,
    synthetic_seq_len,
    batch_seq_len,
):
    train_lib._maybe_materialize_vocab_parallel_config(cfg)
    model_version = str(cfg["model"].get("model_version", ""))
    mesh_model = int(cfg.get("training", {}).get("mesh_model", 1))
    mesh_data = int(cfg.get("training", {}).get("mesh_data", 0))
    if mesh_data == 0:
        mesh_data = jax.device_count() // mesh_model

    if model_version != train_lib.V4168_MODEL_VERSION:
        _fail(
            "config",
            f"model_version={model_version!r}, expected "
            f"{train_lib.V4168_MODEL_VERSION!r}",
        )
    if mesh_model <= 1:
        _fail("config", f"mesh_model must be > 1, got {mesh_model}")
    logical = cfg["model"].get("logical_vocab_size")
    padded = cfg["model"].get("vocab_size_padded")
    if logical is None:
        _fail("config", "model.logical_vocab_size missing after materialization")
    if padded is None:
        _fail("config", "model.vocab_size_padded missing after materialization")
    logical = int(logical)
    padded = int(padded)
    if padded % mesh_model != 0:
        _fail(
            "config",
            f"vocab_size_padded={padded} not divisible by mesh_model={mesh_model}",
        )
    if padded < logical:
        _fail(
            "config",
            f"vocab_size_padded={padded} < logical_vocab_size={logical}",
        )

    vocab_per_shard = padded // mesh_model
    _log("=== V4168 vocab-parallel correctness check ===")
    _log(f"config={config_path}")
    _log(f"[check] model_version={model_version}")
    _log(f"[check] mesh_data={mesh_data} mesh_model={mesh_model}")
    _log(f"[check] logical_vocab_size={logical}")
    _log(f"[check] vocab_size_padded={padded}")
    _log(f"[check] padded_rows={padded - logical}")
    _log(f"[check] vocab_per_shard={vocab_per_shard}")
    _log(f"[check] ce_token_chunk_size={token_chunk_size}")
    _log(f"[check] synthetic_seq_len={synthetic_seq_len}")
    _log(f"[check] batch_seq_len={batch_seq_len}")
    _log("[check] checker_forces_compute_accuracy=true")
    _log(f"mesh_data={mesh_data}")
    _log(f"mesh_model={mesh_model}")
    _log(f"logical_vocab_size={logical}")
    _log(f"vocab_size_padded={padded}")
    _log(f"vocab_per_shard={vocab_per_shard}")
    return {
        "model_version": model_version,
        "mesh_data": mesh_data,
        "mesh_model": mesh_model,
        "logical_vocab_size": logical,
        "vocab_size_padded": padded,
        "vocab_per_shard": vocab_per_shard,
    }


def _synthetic_arrays(seed, batch_size, seq_len, d_model, logical, padded):
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    rng = np.random.default_rng(int(seed))
    input_ids = rng.integers(
        0, logical, size=(batch_size, seq_len), dtype=np.int32)
    shift_x = rng.normal(
        loc=0.0, scale=0.5,
        size=(batch_size, seq_len - 1, d_model)).astype(np.float32)
    labels = rng.integers(
        0, logical, size=(batch_size, seq_len - 1), dtype=np.int32)
    valid_mask = (rng.random(size=(batch_size, seq_len - 1)) < 0.85)
    if not np.any(valid_mask):
        valid_mask[0, 0] = True
    full_embedding = rng.normal(
        loc=0.0, scale=0.02, size=(padded, d_model)).astype(np.float32)
    if padded > logical:
        full_embedding[logical:, :] = 1000.0
    return input_ids, shift_x, labels, valid_mask, full_embedding


def _run_synthetic(
    args,
    cfg,
    mesh,
    info,
    token_chunk_size,
):
    logical = int(info["logical_vocab_size"])
    padded = int(info["vocab_size_padded"])
    batch_size = int(cfg["training"]["batch_size"])
    seq_len = _synthetic_seq_len(args, cfg)
    d_model = int(
        args.d_model
        if args.d_model is not None
        else min(int(cfg["model"]["d_model"]), 256)
    )
    if batch_size % int(info["mesh_data"]) != 0:
        _fail(
            "synthetic",
            f"batch_size={batch_size} must be divisible by "
            f"mesh_data={info['mesh_data']}",
        )

    vp_embed = make_vocab_parallel_embedding(mesh, logical, padded)
    vp_ce_acc = make_vocab_parallel_ce(
        mesh,
        logical_vocab_size=logical,
        vocab_size_padded=padded,
        token_chunk_size=token_chunk_size,
        compute_accuracy=True,
        compute_logit_stats=True,
    )
    vp_ce_noacc = make_vocab_parallel_ce(
        mesh,
        logical_vocab_size=logical,
        vocab_size_padded=padded,
        token_chunk_size=token_chunk_size,
        compute_accuracy=False,
        compute_logit_stats=False,
    )

    input_ids, shift_x, labels, valid_mask, full_embedding = _synthetic_arrays(
        args.seed, batch_size, seq_len, d_model, logical, padded)

    data_2d = NamedSharding(mesh, P("data", None))
    data_3d = NamedSharding(mesh, P("data", None, None))
    model_2d = NamedSharding(mesh, P("model", None))
    replicated = NamedSharding(mesh, P())
    _log(f"[synthetic] logical_vocab_size={logical}")
    _log(f"[synthetic] vocab_size_padded={padded}")
    _log(f"[synthetic] padded_rows={padded - logical}")
    _log(f"[synthetic] max_token_id={int(np.max(input_ids))}")
    _log(f"[synthetic] embedding_sharding={model_2d.spec}")
    _log(f"[synthetic] input_sharding={data_2d.spec}")

    input_ids_s = jax.device_put(input_ids, data_2d)
    shift_x_s = jax.device_put(shift_x, data_3d)
    labels_s = jax.device_put(labels, data_2d)
    valid_mask_s = jax.device_put(valid_mask, data_2d)
    embedding_s = jax.device_put(full_embedding, model_2d)
    embedding_dense = jax.device_put(full_embedding, replicated)

    vp_emb_out = vp_embed(input_ids_s, embedding_s)
    dense_emb_out = dense_reference_embedding(input_ids_s, embedding_dense)
    emb_stats = _check_array(
        "embedding", dense_emb_out, vp_emb_out, args.atol, args.rtol)
    _log(
        "[embedding] max_abs={max_abs:.6e} rel_l2={rel_l2:.6e} PASS".format(
            **emb_stats))

    (vp_loss, vp_ptce, vp_correct, vp_valid, _logit_abs_max,
     _logit_norm_mean, _logit_mean, _logit_std) = vp_ce_acc(
        shift_x_s, embedding_s, labels_s, valid_mask_s)
    dense_loss, dense_ptce, dense_correct, dense_valid = dense_ce_reference(
        shift_x_s,
        embedding_dense,
        labels_s,
        valid_mask_s,
        logical,
        compute_accuracy=True,
    )

    loss_stats = _check_scalar(
        "ce-forward", dense_loss, vp_loss, args.atol, args.rtol)
    _log(
        "[ce-forward] dense_loss={ref:.9g} vp_loss={got:.9g} "
        "abs={abs:.6e} rel={rel:.6e} PASS".format(**loss_stats))

    pt_stats = _check_array(
        "per-token-ce", dense_ptce, vp_ptce, args.atol, args.rtol)
    _log(
        "[per-token-ce] max_abs={max_abs:.6e} rel_l2={rel_l2:.6e} "
        "PASS".format(**pt_stats))

    dense_correct_i = _scalar_int(dense_correct)
    vp_correct_i = _scalar_int(vp_correct)
    dense_valid_i = _scalar_int(dense_valid)
    vp_valid_i = _scalar_int(vp_valid)
    if dense_correct_i != vp_correct_i:
        _fail(
            "accuracy",
            f"dense_correct={dense_correct_i} vp_correct={vp_correct_i}",
        )
    if dense_valid_i != vp_valid_i:
        _fail(
            "valid_count",
            f"dense_valid={dense_valid_i} vp_valid={vp_valid_i}",
        )
    _log(
        f"[accuracy] dense_correct={dense_correct_i} "
        f"vp_correct={vp_correct_i} valid={dense_valid_i} PASS")

    if args.skip_grad:
        _log("[grad-x] SKIP (--skip-grad)")
        _log("[grad-embedding] SKIP (--skip-grad)")
        return

    def dense_loss_fn(x, emb):
        return dense_ce_reference(
            x,
            emb,
            labels_s,
            valid_mask_s,
            logical,
            compute_accuracy=False,
        )[0]

    def vp_loss_fn(x, emb):
        return vp_ce_noacc(x, emb, labels_s, valid_mask_s)[0]

    dense_grad_loss, dense_grads = jax.value_and_grad(
        dense_loss_fn, argnums=(0, 1))(shift_x_s, embedding_dense)
    vp_grad_loss, vp_grads = jax.value_and_grad(
        vp_loss_fn, argnums=(0, 1))(shift_x_s, embedding_s)
    dense_grad_x, dense_grad_emb = dense_grads
    vp_grad_x, vp_grad_emb = vp_grads
    dense_grad_emb_s = jax.device_put(dense_grad_emb, model_2d)

    _check_scalar(
        "grad-loss", dense_grad_loss, vp_grad_loss, args.grad_atol,
        args.grad_rtol)
    grad_x_stats = _check_array(
        "grad-x", dense_grad_x, vp_grad_x, args.grad_atol, args.grad_rtol)
    grad_emb_stats = _check_array(
        "grad-embedding",
        dense_grad_emb_s[:logical],
        vp_grad_emb[:logical],
        args.grad_atol,
        args.grad_rtol,
    )
    if padded > logical:
        padded_grad_max = float(np.max(np.abs(_as_float_np(vp_grad_emb[logical:]))))
    else:
        padded_grad_max = 0.0
    if padded_grad_max > float(args.grad_atol):
        _fail(
            "padded-grad",
            f"padded_grad_max={padded_grad_max:.6e} "
            f"grad_atol={args.grad_atol:.6e}",
        )
    _log(
        f"[grad] loss_dense={_scalar_float(dense_grad_loss):.9g} "
        f"loss_vp={_scalar_float(vp_grad_loss):.9g}")
    _log(
        "[grad-x] max_abs={max_abs:.6e} rel_l2={rel_l2:.6e} PASS".format(
            **grad_x_stats))
    _log(
        "[grad-embedding] max_abs={max_abs:.6e} rel_l2={rel_l2:.6e} "
        f"padded_grad_max={padded_grad_max:.6e} PASS".format(
            **grad_emb_stats))


def _auto_n_chunks(n_local, per_device_batch, seq_len, target_gb):
    full_gb = per_device_batch * seq_len * n_local * 2 / 1e9
    n_chunks = max(1, int(np.ceil(full_gb / target_gb)))
    while n_local % n_chunks != 0 and n_chunks < n_local:
        n_chunks += 1
    return min(n_chunks, n_local)


def _chunk_size_from_count(name, n_local, n_chunks):
    n_chunks = int(n_chunks)
    if n_chunks < 1:
        raise ValueError(f"{name} chunks must be >= 1, got {n_chunks}")
    if n_chunks > n_local:
        raise ValueError(
            f"{name} chunks={n_chunks} exceeds local pool size {n_local}")
    return max(1, int(np.ceil(n_local / n_chunks)))


def _factory_kwargs(factory, kwargs):
    sig = inspect.signature(factory)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _v4168_chunk_sizes(cfg, mesh_model, mesh_data, seq_len):
    batch_size = int(cfg["training"]["batch_size"])
    per_device_batch = batch_size // mesh_data
    target_chunk_gb = cfg["training"].get("target_chunk_gb", 2.0)
    n_rst = int(cfg["model"].get("n_rst", cfg["model"].get("n_know", 25200)))
    n_qk = int(cfg["model"].get("n_qk", cfg["model"].get("n_q", 1580)))
    n_v = int(cfg["model"].get("n_v", 2600))
    for name, value in (("n_rst", n_rst), ("n_qk", n_qk), ("n_v", n_v)):
        if value % mesh_model != 0:
            raise ValueError(
                f"model.{name}={value} must be divisible by "
                f"mesh_model={mesh_model} for model-axis sharding.")

    nrst_local = n_rst // mesh_model
    nqk_local = n_qk // mesh_model
    nv_local = n_v // mesh_model
    n_chunks_rst = cfg["training"].get(
        "n_chunks_rst",
        _auto_n_chunks(nrst_local, per_device_batch, seq_len, target_chunk_gb))
    n_chunks_qk = cfg["training"].get(
        "n_chunks_qk",
        _auto_n_chunks(nqk_local, per_device_batch, seq_len, target_chunk_gb))
    n_chunks_v = cfg["training"].get(
        "n_chunks_v",
        _auto_n_chunks(nv_local, per_device_batch, seq_len, target_chunk_gb))
    attn_qk_max_chunk = _chunk_size_from_count(
        "attn_qk", nqk_local, n_chunks_qk)
    attn_v_max_chunk = _chunk_size_from_count(
        "attn_v", nv_local, n_chunks_v)
    rst_max_chunk = _chunk_size_from_count("rst", nrst_local, n_chunks_rst)
    max_chunk = cfg["training"].get("max_chunk_size", None)
    if max_chunk is not None:
        attn_qk_max_chunk = attn_v_max_chunk = rst_max_chunk = int(max_chunk)
    return attn_qk_max_chunk, attn_v_max_chunk, rst_max_chunk


def _build_v4168_sharded_fns(cfg, mesh, token_chunk_size, seq_len):
    model_version = str(cfg["model"].get("model_version", ""))
    if model_version != train_lib.V4168_MODEL_VERSION:
        _fail("sharded_fns", f"expected V4168, got {model_version}")
    mesh_model = int(mesh.shape["model"])
    mesh_data = int(mesh.shape["data"])

    hardware_repack_config = train_lib._v4168_hardware_repack_config(
        cfg["training"], model_version)
    hardware_repack_enabled = bool(
        hardware_repack_config["hardware_repack_enabled"])
    hardware_sector_execution_enabled = bool(
        hardware_repack_config["hardware_sector_execution_enabled"])
    operation_space_repack_config = (
        train_lib._v4168_operation_space_repack_config(
            cfg["training"], cfg["model"], model_version))
    operation_space_enabled = bool(
        operation_space_repack_config.get("operation_space_enabled", False))
    opspace_layouts = (
        train_lib._v4168_operation_space_pool_layouts(
            cfg["training"], cfg["model"])
        if operation_space_enabled else {})

    (attn_qk_max_chunk, attn_v_max_chunk,
     rst_max_chunk) = _v4168_chunk_sizes(
        cfg, mesh_model, mesh_data, seq_len)

    module_name = train_lib._model_registry_entry(model_version)["module"]
    srw_module = __import__(module_name, fromlist=["make_sharded_srw"])
    make_sharded_srw = srw_module.make_sharded_srw
    make_sharded_srw_minimal = getattr(
        srw_module, "make_sharded_srw_minimal", None)
    make_sharded_srw_paired_minimal = getattr(
        srw_module, "make_sharded_srw_paired_minimal", None)
    make_sharded_srw_paired_dense_minimal = getattr(
        srw_module, "make_sharded_srw_paired_dense_minimal", None)

    srw_base_kwargs = {"mesh": mesh}
    srw_base_kwargs.update(train_lib._v4164_sharded_kwargs(cfg))

    def opspace_pool_kwargs(pool):
        if not operation_space_enabled:
            return {}
        layout = opspace_layouts.get(pool, {})
        if not isinstance(layout, dict):
            layout = {}
        return {
            "operation_space_execution_backend": str(layout.get(
                "execution_backend",
                "sparse_region_block" if pool == "rst" else "dense",
            )).lower(),
            "opspace_bucket_capacity_factor": float(layout.get(
                "bucket_capacity_factor",
                layout.get("block_capacity_factor", 1.25))),
            "opspace_high_regret_threshold": float(layout.get(
                "high_regret_threshold", 0.05)),
            "opspace_region_capacity_factor": float(layout.get(
                "region_capacity_factor", 1.25)),
            "opspace_block_capacity_factor": float(layout.get(
                "block_capacity_factor", 1.25)),
            "opspace_num_regions": int(layout.get(
                "num_regions", 32 if pool == "rst" else 8)),
            "opspace_blocks_per_region": int(layout.get(
                "blocks_per_region", 1)),
            "opspace_operators_per_block": int(layout.get(
                "operators_per_block", 128)),
            "opspace_visible_regions": int(layout.get(
                "visible_regions", 4 if pool == "rst" else 2)),
            "opspace_visible_blocks_per_region": int(layout.get(
                "visible_blocks_per_region", 1)),
            "opspace_region_score_pooling": str(layout.get(
                "region_score_pooling", "smoothmax")).lower(),
            "opspace_region_score_temperature": float(layout.get(
                "region_score_temperature", 0.25)),
            "opspace_load_smoothing_enabled": bool(layout.get(
                "load_smoothing_enabled", pool == "rst")),
            "opspace_load_smoothing_rst_region_weight": float(
                layout.get("load_smoothing_rst_region_weight", 3.0e-4)),
            "opspace_load_smoothing_rst_block_weight": float(
                layout.get("load_smoothing_rst_block_weight", 0.0)),
            "opspace_load_smoothing_load_temperature": float(
                layout.get("load_smoothing_load_temperature", 0.7)),
            "opspace_load_smoothing_space_temperature": float(
                layout.get("load_smoothing_space_temperature", 0.12)),
            "opspace_load_smoothing_alpha": float(
                layout.get("load_smoothing_alpha", 1.25)),
            "opspace_load_smoothing_warmup_tokens": float(
                layout.get("load_smoothing_warmup_tokens", 2.0e8)),
            "opspace_load_smoothing_peak_tokens": float(
                layout.get("load_smoothing_peak_tokens", 1.0e9)),
            "opspace_load_smoothing_final_weight_frac": float(
                layout.get("load_smoothing_final_weight_frac", 1.0)),
            "opspace_completion_enabled": bool(layout.get(
                "completion_enabled", pool == "rst")),
            "opspace_completion_spill_capacity_factor": float(
                layout.get("completion_spill_capacity_factor", 0.25)),
            "opspace_completion_fallback_on_spill_overflow": bool(layout.get(
                "completion_fallback_on_spill_overflow", True)),
            "opspace_completion_assert_all_processed": bool(layout.get(
                "completion_assert_all_processed", True)),
        }

    def srw_pool_kwargs(pool):
        kwargs = dict(srw_base_kwargs)
        model_cfg = cfg["model"]
        kwargs.update({
            "block_size": int(model_cfg.get(f"{pool}_block_size", 256)),
            "top_blocks": int(model_cfg.get(f"{pool}_top_blocks", 2)),
            "block_margin": float(model_cfg.get("block_margin", 0.0)),
            "hardware_sector_execution_enabled":
                hardware_sector_execution_enabled,
            "hardware_sector_debug_token_gather_fallback": bool(
                cfg["training"].get(
                    "hardware_sector_debug_token_gather_fallback", False)),
        })
        kwargs.update(opspace_pool_kwargs(pool))
        return kwargs

    sharded_single_v = make_sharded_srw(
        max_chunk_size=attn_v_max_chunk,
        **_factory_kwargs(make_sharded_srw, srw_pool_kwargs("v")))
    sharded_single_rst = make_sharded_srw(
        max_chunk_size=rst_max_chunk,
        **_factory_kwargs(make_sharded_srw, srw_pool_kwargs("rst")))

    sharded_single_qk_minimal = None
    sharded_single_v_minimal = None
    sharded_single_rst_minimal = None
    if make_sharded_srw_minimal is not None:
        sharded_single_v_minimal = make_sharded_srw_minimal(
            max_chunk_size=attn_v_max_chunk,
            **_factory_kwargs(
                make_sharded_srw_minimal, srw_pool_kwargs("v")))
        sharded_single_rst_minimal = make_sharded_srw_minimal(
            max_chunk_size=rst_max_chunk,
            **_factory_kwargs(
                make_sharded_srw_minimal, srw_pool_kwargs("rst")))

    if not hasattr(srw_module, "make_sharded_srw_paired"):
        _fail("sharded_fns", "V4168 requires paired sharded SRW factory")
    paired_factory = srw_module.make_sharded_srw_paired
    sharded_paired_attn_qk = paired_factory(
        max_chunk_size=attn_qk_max_chunk,
        **_factory_kwargs(paired_factory, srw_pool_kwargs("qk")))
    sharded_paired_attn_qk_minimal = None
    if make_sharded_srw_paired_minimal is not None:
        if operation_space_enabled:
            if make_sharded_srw_paired_dense_minimal is None:
                _fail(
                    "sharded_fns",
                    "operation_space enabled but dense minimal QK executor "
                    "is missing",
                )
            sharded_paired_attn_qk_minimal = (
                make_sharded_srw_paired_dense_minimal(
                    max_chunk_size=attn_qk_max_chunk,
                    **_factory_kwargs(
                        make_sharded_srw_paired_dense_minimal,
                        srw_pool_kwargs("qk"))))
        elif make_sharded_srw_paired_dense_minimal is not None:
            sharded_paired_attn_qk_minimal = (
                make_sharded_srw_paired_dense_minimal(
                    max_chunk_size=attn_qk_max_chunk,
                    **_factory_kwargs(
                        make_sharded_srw_paired_dense_minimal,
                        srw_pool_kwargs("qk"))))
        else:
            sharded_paired_attn_qk_minimal = (
                make_sharded_srw_paired_minimal(
                    max_chunk_size=attn_qk_max_chunk,
                    **_factory_kwargs(
                        make_sharded_srw_paired_minimal,
                        srw_pool_kwargs("qk"))))

    sharded_fns = {
        "single": sharded_single_v,
        "attn_v_single": sharded_single_v,
        "rst_single": sharded_single_rst,
        "paired": sharded_paired_attn_qk,
        "attn_qk_paired": sharded_paired_attn_qk,
    }
    if sharded_single_v_minimal is not None:
        sharded_fns.update({
            "attn_qk_single_minimal": sharded_single_qk_minimal,
            "attn_v_single_minimal": sharded_single_v_minimal,
            "rst_single_minimal": sharded_single_rst_minimal,
        })
    if sharded_paired_attn_qk_minimal is not None:
        sharded_fns["attn_qk_paired_minimal"] = (
            sharded_paired_attn_qk_minimal)

    if operation_space_enabled:
        if hardware_repack_enabled or hardware_sector_execution_enabled:
            _fail(
                "sharded_fns",
                "operation_space requires hardware repack and sector "
                "execution disabled",
            )
        for pool, required in (
            ("qk", "attn_qk_paired_minimal"),
            ("v", "attn_v_single_minimal"),
            ("rst", "rst_single_minimal"),
        ):
            if sharded_fns.get(required, None) is None:
                _fail(
                    "sharded_fns",
                    f"operation_space missing tau-free executor for {pool}: "
                    f"{required}",
                )
        sharded_fns.update({
            "operation_space_tau_free": True,
            "qk_backend": str(opspace_layouts["qk"].get(
                "execution_backend", "")).lower(),
            "v_backend": str(opspace_layouts["v"].get(
                "execution_backend", "")).lower(),
            "rst_backend": str(opspace_layouts["rst"].get(
                "execution_backend", "")).lower(),
        })

    logical = int(cfg["model"]["logical_vocab_size"])
    padded = int(cfg["model"]["vocab_size_padded"])
    if padded % mesh_model != 0:
        _fail(
            "sharded_fns",
            f"vocab_size_padded={padded} not divisible by "
            f"mesh_model={mesh_model}",
        )
    vocab_embed = make_vocab_parallel_embedding(mesh, logical, padded)
    vocab_ce_train = make_vocab_parallel_ce(
        mesh,
        logical_vocab_size=logical,
        vocab_size_padded=padded,
        token_chunk_size=token_chunk_size,
        compute_accuracy=True,
    )
    vocab_ce_eval = make_vocab_parallel_ce(
        mesh,
        logical_vocab_size=logical,
        vocab_size_padded=padded,
        token_chunk_size=token_chunk_size,
        compute_accuracy=True,
    )
    sharded_fns["vocab_parallel_embedding"] = vocab_embed
    sharded_fns["vocab_ce"] = vocab_ce_train
    sharded_fns_eval = dict(sharded_fns)
    sharded_fns_eval["vocab_ce"] = vocab_ce_eval

    if sharded_fns.get("vocab_parallel_embedding") is None:
        _fail("sharded_fns", "vocab_parallel_embedding not installed")
    if sharded_fns.get("vocab_ce") is None:
        _fail("sharded_fns", "vocab_ce not installed")
    if sharded_fns_eval.get("vocab_ce") is None:
        _fail("sharded_fns", "eval vocab_ce not installed")

    return sharded_fns, sharded_fns_eval, operation_space_enabled


def _resolve_checkpoint_target(args):
    if not args.resume_from:
        if args.checkpoint_step is not None:
            _fail("checkpoint", "--checkpoint-step requires --resume-from")
        return None, None
    run_folder, resolved_step, exists = train_lib._resolve_orbax_resume_from(
        args.resume_from)
    if run_folder is None:
        _fail("checkpoint", f"could not resolve resume target: {args.resume_from}")
    if args.checkpoint_step is not None:
        requested_step = int(args.checkpoint_step)
        steps = train_lib._list_orbax_steps_for_run(run_folder)
        if requested_step not in {int(step) for step in steps}:
            _fail(
                "checkpoint",
                f"checkpoint step {requested_step} not found under "
                f"{run_folder}",
            )
        resolved_step = requested_step
        exists = True
    if resolved_step is None or not exists:
        _fail(
            "checkpoint",
            f"no committed Orbax checkpoint found for {args.resume_from}",
        )
    return run_folder, int(resolved_step)


def _restore_params_from_checkpoint(args, cfg, mesh, template_params, rng):
    run_folder, resolved_step = _resolve_checkpoint_target(args)
    if run_folder is None:
        _log("[checkpoint] params_restored=false")
        return template_params

    train_lib._require_orbax_checkpoint_compat()
    checkpoint_dir = train_lib._join_path(run_folder, "checkpoints")
    manager = train_lib._create_orbax_checkpoint_manager(
        checkpoint_dir,
        checkpoint_interval=1,
        keep_last=None,
        create=False,
        read_only=True,
        best_tracking=False,
    )
    target_state = train_lib._build_orbax_state(
        template_params,
        {},
        rng,
        epoch=0,
        global_step=0,
        step_in_epoch=0,
        steps_per_epoch=0,
        best_val_loss=float("inf"),
        training_config=cfg.get("training", {}),
        full_config=cfg,
        model_config=cfg.get("model", {}),
    )
    restored_state, _restored_metadata = train_lib._restore_orbax_state(
        manager, resolved_step, target_state)
    if "params" not in restored_state:
        _fail("checkpoint", "Orbax checkpoint state is missing params")
    _assert_param_shapes_match(restored_state["params"], template_params)
    restored_params = train_lib._match_tree_to_template_on_mesh(
        restored_state["params"], template_params, mesh, name="params")
    restored_global_step = train_lib._state_scalar(
        restored_state,
        "global_step",
        train_lib._state_scalar(restored_state, "step", resolved_step, int),
        int,
    )
    _log(f"[checkpoint] resume_from={args.resume_from}")
    _log(f"[checkpoint] resolved_run_folder={run_folder}")
    _log(f"[checkpoint] resolved_step={resolved_step}")
    _log(f"[checkpoint] restored_global_step={restored_global_step}")
    _log("[checkpoint] params_restored=true")
    return restored_params


def _assert_param_shapes_match(restored_params, template_params):
    def _check(restored_val, template_val):
        restored_shape = tuple(np.shape(restored_val))
        template_shape = tuple(getattr(template_val, "shape", ()))
        if restored_shape != template_shape:
            raise ValueError(
                f"restored shape {restored_shape} != template shape "
                f"{template_shape}")
        return None

    try:
        jax.tree.map(_check, restored_params, template_params)
    except Exception as exc:
        _fail(
            "checkpoint",
            "restored params do not exactly match template shapes: "
            f"{type(exc).__name__}: {exc}",
        )


def _build_model_and_params(args, cfg, mesh, seed, operation_space_enabled):
    model = train_lib.build_model_from_config(cfg)
    rng = jax.random.PRNGKey(int(seed))
    _rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    max_seq_len = int(cfg["model"]["max_seq_len"])
    dummy_input = jnp.ones((1, max_seq_len), dtype=jnp.int32)
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy_input,
        deterministic=True,
    )
    params = variables["params"]
    param_shardings = train_lib.get_param_shardings(
        params,
        mesh,
        cfg["model"].get("model_version"),
        operation_space_enabled=operation_space_enabled,
        vocab_size_padded=cfg["model"].get("vocab_size_padded", None),
    )
    params = train_lib.shard_params_to_mesh(params, param_shardings)
    params = _restore_params_from_checkpoint(args, cfg, mesh, params, rng)
    return model, params


def _first_batch(loader, label, max_batches):
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        return batch
    raise RuntimeError(f"No {label} batch available.")


def _assert_vocab_parallel_paths(sharded_fns_train, sharded_fns_eval):
    for label, fns in (
        ("train", sharded_fns_train),
        ("eval", sharded_fns_eval),
    ):
        if not isinstance(fns, dict):
            _fail("path", f"{label} sharded_fns is not a dict")
        for key in ("vocab_parallel_embedding", "vocab_ce"):
            if key not in fns or fns[key] is None:
                _fail("path", f"{label} sharded_fns missing {key}")
    _log("[path] vocab_parallel_embedding_active=true")
    _log("[path] vocab_ce_train_active=true")
    _log("[path] vocab_ce_eval_active=true")
    _log("[path] minimal_train=true")


def _batch_sanity(
    args,
    cfg,
    source,
    input_ids_host,
    attention_mask_host,
    input_ids,
    attention_mask,
    global_shape,
    logical_vocab_size,
):
    config_seq_len = int(cfg["model"]["max_seq_len"])
    if args.seq_len is None and int(global_shape[1]) != config_seq_len:
        _fail(
            "batch",
            f"batch seq_len={int(global_shape[1])} must equal config "
            f"max_seq_len={config_seq_len} unless --seq-len is explicit",
        )
    input_min = _scalar_int(jnp.min(input_ids))
    input_max = _scalar_int(jnp.max(input_ids))
    zero_frac = _scalar_float(jnp.mean((input_ids == 0).astype(jnp.float32)))
    attention_mask_sum = _scalar_int(jnp.sum(attention_mask))
    valid_next_token_count = _scalar_int(jnp.sum(attention_mask[:, 1:] == 1))
    _log(f"[batch] source={source}")
    _log(f"[batch] host_shape={tuple(input_ids_host.shape)}")
    _log(f"[batch] global_shape={tuple(global_shape)}")
    _log(f"[batch] input_min={input_min}")
    _log(f"[batch] input_max={input_max}")
    _log(f"[batch] zero_frac={zero_frac:.6e}")
    _log(f"[batch] attention_mask_sum={attention_mask_sum}")
    _log(f"[batch] valid_next_token_count={valid_next_token_count}")
    if input_max >= int(logical_vocab_size):
        _fail(
            "batch",
            f"input_max={input_max} >= logical_vocab_size={logical_vocab_size}",
        )
    if attention_mask_sum == 0:
        _fail("batch", "attention_mask_sum == 0")
    if valid_next_token_count == 0:
        _fail("batch", "valid_next_token_count == 0")


def _require_model_output(result, label):
    for key in ("per_token_ce", "valid_mask", "valid_count"):
        if key not in result:
            _fail(label, f"model output missing {key}")
    valid_count = _scalar_int(result["valid_count"])
    if valid_count <= 0:
        _fail(label, f"valid_count must be > 0, got {valid_count}")
    return valid_count


def _same_batch_dense_ref(args, params, result, logical_vocab_size):
    if not args.same_batch_dense_ref:
        return
    required = ("shift_x", "shift_labels", "vocab_check_valid_mask")
    if any(key not in result for key in required):
        _log("[same-batch dense-ref] SKIP: hidden debug return unavailable")
        return
    embedding = params["token_emb"]["embedding"]
    dense_loss, dense_ptce, dense_correct, dense_valid = dense_ce_reference(
        result["shift_x"],
        embedding,
        result["shift_labels"],
        result["vocab_check_valid_mask"],
        int(logical_vocab_size),
        compute_accuracy=True,
    )
    loss_stats = _check_scalar(
        "same-batch dense-ref",
        dense_loss,
        result["loss"],
        args.atol,
        args.rtol,
    )
    pt_stats = _check_array(
        "same-batch dense-ref per-token-ce",
        dense_ptce,
        result["per_token_ce"],
        args.atol,
        args.rtol,
    )
    dense_correct_i = _scalar_int(dense_correct)
    vp_correct_i = _scalar_int(result["correct"])
    dense_valid_i = _scalar_int(dense_valid)
    vp_valid_i = _scalar_int(result["valid_count"])
    if dense_correct_i != vp_correct_i:
        _fail(
            "same-batch dense-ref accuracy",
            f"dense_correct={dense_correct_i} vp_correct={vp_correct_i}",
        )
    if dense_valid_i != vp_valid_i:
        _fail(
            "same-batch dense-ref valid_count",
            f"dense_valid={dense_valid_i} vp_valid={vp_valid_i}",
        )
    _log(
        "[same-batch dense-ref] dense_loss={ref:.9g} "
        "vp_loss={got:.9g} abs={abs:.6e} PASS".format(**loss_stats))
    _log(
        "[same-batch dense-ref per-token-ce] max_abs={max_abs:.6e} "
        "rel_l2={rel_l2:.6e} PASS".format(**pt_stats))


def _run_same_batch(
    args,
    cfg,
    mesh,
    info,
    token_chunk_size,
    train_loader,
    val_loader,
):
    seq_len = _batch_seq_len(args, cfg)
    sharded_fns_train, sharded_fns_eval, operation_space_enabled = (
        _build_v4168_sharded_fns(cfg, mesh, token_chunk_size, seq_len))
    _assert_vocab_parallel_paths(sharded_fns_train, sharded_fns_eval)
    model, params = _build_model_and_params(
        args, cfg, mesh, args.seed, operation_space_enabled)

    loader = train_loader if args.use_train_batch else val_loader
    loader_name = "train" if args.use_train_batch else "val"
    max_batches = 1 if args.use_train_batch else int(args.max_val_batches)
    input_ids_host, attention_mask_host = _first_batch(
        loader, loader_name, max_batches)

    batch_size = int(cfg["training"]["batch_size"])
    if int(input_ids_host.shape[1]) != seq_len:
        _fail(
            "same-batch",
            f"batch seq_len={int(input_ids_host.shape[1])} expected {seq_len}",
        )
    data_sharding = NamedSharding(mesh, P("data", None))
    global_shape = (batch_size, seq_len)
    input_ids = train_lib.shard_to_mesh(
        input_ids_host, data_sharding, global_shape)
    attention_mask = train_lib.shard_to_mesh(
        attention_mask_host, data_sharding, global_shape)
    logical = int(info["logical_vocab_size"])
    _batch_sanity(
        args,
        cfg,
        loader_name,
        input_ids_host,
        attention_mask_host,
        input_ids,
        attention_mask,
        global_shape,
        logical,
    )
    labels = jnp.where(attention_mask == 1, input_ids, -100)
    dropout_rng = jax.random.PRNGKey(0)

    apply_kwargs = {
        "labels": labels,
        "attention_mask": attention_mask,
        "deterministic": True,
        "analysis": False,
        "minimal_train": True,
        "compute_accuracy": True,
        "ce_token_chunk_size": token_chunk_size,
        "return_hidden_for_vocab_check": bool(args.same_batch_dense_ref),
        "rngs": {"dropout": dropout_rng},
    }
    train_like = model.apply(
        {"params": params},
        input_ids,
        sharded_fns=sharded_fns_train,
        **apply_kwargs,
    )
    eval_like = model.apply(
        {"params": params},
        input_ids,
        sharded_fns=sharded_fns_eval,
        **apply_kwargs,
    )
    _require_model_output(train_like, "same-batch train output")
    _require_model_output(eval_like, "same-batch eval output")
    _same_batch_dense_ref(args, params, train_like, logical)

    loss_stats = _check_scalar(
        "train/eval sharded_fns mismatch on same batch",
        train_like["loss"],
        eval_like["loss"],
        args.atol,
        args.rtol,
    )
    train_correct = _scalar_int(train_like["correct"])
    eval_correct = _scalar_int(eval_like["correct"])
    train_valid = _scalar_int(train_like["valid_count"])
    eval_valid = _scalar_int(eval_like["valid_count"])
    if train_correct != eval_correct:
        _fail(
            "train/eval sharded_fns mismatch on same batch",
            f"train_correct={train_correct} eval_correct={eval_correct}",
        )
    if train_valid != eval_valid:
        _fail(
            "train/eval sharded_fns mismatch on same batch",
            f"train_valid={train_valid} eval_valid={eval_valid}",
        )
    if "per_token_ce" not in train_like or "per_token_ce" not in eval_like:
        _fail(
            "train/eval sharded_fns mismatch on same batch",
            "per_token_ce missing from model output")
    pt_stats = _check_array(
        "train/eval sharded_fns mismatch on same batch",
        train_like["per_token_ce"],
        eval_like["per_token_ce"],
        args.atol,
        args.rtol,
    )
    _log(
        "[same-batch train/eval] train_loss={ref:.9g} "
        "eval_loss={got:.9g} abs={abs:.6e} PASS".format(**loss_stats))
    _log(
        "[same-batch per-token-ce] max_abs={max_abs:.6e} "
        "rel_l2={rel_l2:.6e} PASS".format(**pt_stats))
    _log(
        f"[same-batch accuracy] train_correct={train_correct} "
        f"eval_correct={eval_correct} valid={train_valid} PASS")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Check V4168 vocab-parallel embedding and CE correctness.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode", choices=("synthetic", "batch", "both"), default="both")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--logical-vocab-size", type=int, default=None)
    parser.add_argument("--token-chunk-size", type=int, default=None)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--grad-atol", type=float, default=5e-4)
    parser.add_argument("--grad-rtol", type=float, default=5e-3)
    parser.add_argument("--skip-grad", action="store_true")
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--same-batch-dense-ref", action="store_true")
    parser.add_argument(
        "--use-train-batch", dest="use_train_batch",
        action="store_true", default=True)
    parser.add_argument(
        "--no-use-train-batch", dest="use_train_batch",
        action="store_false")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        cfg, config_path = _load_config(args)
        token_chunk_size = int(cfg["training"]["ce_token_chunk_size"])

        train_lib._maybe_initialize_jax_distributed()

        synthetic_seq_len = _synthetic_seq_len(args, cfg)
        batch_seq_len = _batch_seq_len(args, cfg)
        train_loader = None
        val_loader = None
        if args.mode in ("batch", "both"):
            train_loader, val_loader, _vocab_size = _maybe_set_vocab_from_data(
                cfg, args, batch_seq_len)

        info = _materialize_and_validate(
            cfg,
            config_path,
            token_chunk_size,
            synthetic_seq_len,
            batch_seq_len,
        )
        mesh = train_lib.create_mesh(info["mesh_data"], info["mesh_model"])

        if args.mode in ("synthetic", "both"):
            _run_synthetic(args, cfg, mesh, info, token_chunk_size)
        if args.mode in ("batch", "both"):
            _run_same_batch(
                args,
                cfg,
                mesh,
                info,
                token_chunk_size,
                train_loader,
                val_loader,
            )
    except CheckFailure:
        _log("RESULT: FAIL")
        return 1
    except Exception as exc:
        _log(f"[FAIL] unexpected: {type(exc).__name__}: {exc}")
        _log("RESULT: FAIL")
        return 1

    _log("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
