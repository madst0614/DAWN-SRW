"""Common model, checkpoint, mesh, and data helpers for DAWN-SRW analysis."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import NamedSharding, PartitionSpec as P

from analysis.dawn_analysis_storage import (
    AnalysisStore,
    exists,
    is_gcs_path,
    join_path,
    json_safe,
    open_path,
    read_json,
    write_json_atomic,
)

V4166_MODEL_VERSION = "spatial-r1-v4.1.6.6"
V4171_MODEL_VERSION = "spatial-r1-v4.1.7.1"
V4172_MODEL_VERSION = "spatial-r1-v4.1.7.2"
V417X_MODEL_VERSIONS = (V4171_MODEL_VERSION, V4172_MODEL_VERSION)
SUPPORTED_ANALYSIS_MODEL_VERSIONS = (
    V4166_MODEL_VERSION,
    *V417X_MODEL_VERSIONS,
)
_TRAIN = None


def get_train():
    """Import the trainer lazily so report-only mode has minimal deps."""
    global _TRAIN
    if _TRAIN is None:
        from scripts import train_jax as train_jax

        _TRAIN = train_jax
    return _TRAIN


@dataclass
class AnalysisContext:
    args: Any
    store: AnalysisStore
    config: Dict[str, Any]
    config_path: Optional[str]
    checkpoint_path: Optional[str]
    checkpoint_step: Optional[int]
    checkpoint_metadata: Dict[str, Any]
    model: Any
    params: Any
    mesh: Any
    data_sharding: Any
    sharded_fns: Any
    sharded_fns_analysis: Any
    model_cfg: Dict[str, Any]
    model_info: Dict[str, Any]
    host_id: int
    n_hosts: int
    is_primary: bool
    total_training_steps: int


def is_primary_host() -> bool:
    try:
        return jax.process_index() == 0
    except Exception:
        return True


def host_info() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "host_id": int(jax.process_index()),
        "n_hosts": int(jax.process_count()),
        "local_devices": int(jax.local_device_count()),
        "global_devices": int(jax.device_count()),
    }


def load_analysis_config(config_path: str) -> Dict[str, Any]:
    with open_path(config_path, "r") as f:
        return yaml.safe_load(f)


def git_info() -> Dict[str, Any]:
    def run(args: Sequence[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return (proc.stdout or "").strip() or None

    return {
        "git_commit": run(["rev-parse", "HEAD"]),
        "git_branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run(["status", "--porcelain"])),
    }


def _path_name(path: str) -> str:
    return str(path).rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def _path_parent(path: str) -> str:
    path = str(path).rstrip("/\\").replace("\\", "/")
    return path.rsplit("/", 1)[0] if "/" in path else "."


def resolve_checkpoint(checkpoint_arg: Optional[str]) -> Tuple[Optional[str], Optional[int], Dict[str, Any]]:
    """Resolve a checkpoint argument to an Orbax checkpoints dir and step."""
    if checkpoint_arg is None:
        return None, None, {}
    target = str(checkpoint_arg).strip().rstrip("/\\")
    if target.startswith("dawn-tpu-data-c4/"):
        target = "gs://" + target
    name = _path_name(target)
    if name.lower() == "latest":
        checkpoints_dir = _path_parent(target)
        run_folder = _path_parent(checkpoints_dir)
        train = get_train()
        step = train._latest_orbax_step_for_run(run_folder)
        if step is None:
            raise FileNotFoundError(f"No committed Orbax checkpoints found under {checkpoints_dir}")
        metadata = train._restore_orbax_metadata(checkpoints_dir, step)
        return checkpoints_dir, int(step), metadata

    train = get_train()
    run_folder, step, found = train._resolve_orbax_resume_from(target)
    if run_folder is None:
        raise ValueError(f"Could not parse checkpoint path: {checkpoint_arg}")
    if step is None:
        step = train._latest_orbax_step_for_run(run_folder)
    if step is None:
        raise FileNotFoundError(f"No committed Orbax checkpoints found in run folder: {run_folder}")
    checkpoints_dir = train._join_path(run_folder, "checkpoints")
    if not found:
        # A requested numeric step can be valid even when the helper could not
        # confirm it before manager construction; metadata restore will fail
        # loudly if the checkpoint is actually missing.
        pass
    metadata = train._restore_orbax_metadata(checkpoints_dir, int(step))
    return checkpoints_dir, int(step), metadata


def config_from_checkpoint_or_file(config_path: Optional[str],
                                   checkpoint_metadata: Dict[str, Any]) -> Dict[str, Any]:
    full_config = checkpoint_metadata.get("full_config")
    if isinstance(full_config, dict) and full_config:
        return full_config
    if config_path is None:
        raise ValueError("A config path is required when checkpoint metadata has no full_config.")
    return load_analysis_config(config_path)


def verify_analysis_config(cfg: Dict[str, Any]) -> None:
    version = cfg.get("model", {}).get("model_version")
    if version not in SUPPORTED_ANALYSIS_MODEL_VERSIONS:
        raise ValueError(
            "Unsupported analysis model.model_version="
            f"{version!r}; supported={','.join(SUPPORTED_ANALYSIS_MODEL_VERSIONS)}"
        )


def verify_v4166_config(cfg: Dict[str, Any]) -> None:
    """Backward-compatible name for the supported-version validator."""
    verify_analysis_config(cfg)


def analysis_model_module(model_cfg: Dict[str, Any]):
    """Return the exact model module declared by an analysis config."""
    version = model_cfg.get("model_version")
    if version not in SUPPORTED_ANALYSIS_MODEL_VERSIONS:
        raise ValueError(
            "Unsupported analysis model.model_version="
            f"{version!r}; supported={','.join(SUPPORTED_ANALYSIS_MODEL_VERSIONS)}"
        )
    train = get_train()
    module_name = train._model_registry_entry(version)["module"]
    return importlib.import_module(module_name)


def build_analysis_model(cfg: Dict[str, Any]):
    verify_analysis_config(cfg)
    train = get_train()
    return train.build_model_from_config(cfg)


def build_v4166_model(cfg: Dict[str, Any]):
    """Backward-compatible wrapper; dispatches supported versions exactly."""
    return build_analysis_model(cfg)


def init_target_params(model: Any, cfg: Dict[str, Any]):
    max_seq_len = int(cfg["model"].get("max_seq_len", 512))
    rng = jax.random.PRNGKey(int(cfg.get("seed", 0)))
    dummy = jnp.ones((1, max_seq_len), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, deterministic=True)
    return variables["params"]


def create_mesh_from_cfg(cfg: Dict[str, Any], args: Any = None):
    tcfg = cfg.get("training", {})
    mesh_model = int(getattr(args, "mesh_model", None) or tcfg.get("mesh_model", 1))
    mesh_data = getattr(args, "mesh_data", None)
    if mesh_data is None:
        mesh_data = tcfg.get("mesh_data", None)
    if mesh_data is None:
        total = int(jax.device_count())
        if total % mesh_model != 0:
            raise ValueError(
                f"Cannot infer mesh_data: device_count={total} not divisible by mesh_model={mesh_model}"
            )
        mesh_data = total // mesh_model
    mesh_data = int(mesh_data)
    expected = mesh_data * mesh_model
    actual = int(jax.device_count())
    if expected != actual:
        raise ValueError(
            "Configured analysis mesh does not match visible JAX devices: "
            f"mesh_data={mesh_data} * mesh_model={mesh_model} = {expected}, "
            f"but jax.device_count()={actual}. "
            "Launch on the same device count as the checkpoint config, or pass "
            "--mesh-data/--mesh-model overrides. For the 400M v4_64 checkpoint, "
            "the checkpoint config expects mesh_model=2 and mesh_data=16 "
            "(32 devices total)."
        )
    train = get_train()
    return train.create_mesh(mesh_data, mesh_model)


def _chunk_sizes_for_cfg(cfg: Dict[str, Any], mesh) -> Dict[str, int]:
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})
    mesh_model = int(mesh.shape["model"])
    mesh_data = int(mesh.shape["data"])
    batch_size = int(tcfg.get("batch_size", 1))
    per_device_batch = max(1, batch_size // max(1, mesh_data))
    max_seq_len = int(mcfg.get("max_seq_len", 512))

    def auto_n_chunks(n_local: int, target_gb: float = 2.0) -> int:
        full_gb = per_device_batch * max_seq_len * n_local * 2 / 1e9
        nc = max(1, int(math.ceil(full_gb / target_gb)))
        while n_local % nc != 0 and nc < n_local:
            nc += 1
        return min(nc, n_local)

    def chunk_size(name: str, n_local: int, n_chunks: int) -> int:
        if n_chunks < 1:
            raise ValueError(f"{name} chunks must be >= 1, got {n_chunks}")
        if n_chunks > n_local:
            raise ValueError(f"{name} chunks={n_chunks} exceeds local pool size {n_local}")
        return max(1, int(math.ceil(n_local / n_chunks)))

    target_chunk_gb = float(tcfg.get("target_chunk_gb", 2.0))
    n_qk = int(mcfg.get("n_qk", 1580))
    n_v = int(mcfg.get("n_v", 2600))
    n_rst = int(mcfg.get("n_rst", mcfg.get("n_know", 25200)))
    for name, value in (("n_qk", n_qk), ("n_v", n_v), ("n_rst", n_rst)):
        if value % mesh_model != 0:
            raise ValueError(
                f"model.{name}={value} must be divisible by mesh_model={mesh_model}"
            )
    nqk_local = n_qk // mesh_model
    nv_local = n_v // mesh_model
    nrst_local = n_rst // mesh_model
    n_chunks_qk = int(tcfg.get("n_chunks_qk", auto_n_chunks(nqk_local, target_chunk_gb)))
    n_chunks_v = int(tcfg.get("n_chunks_v", auto_n_chunks(nv_local, target_chunk_gb)))
    n_chunks_rst = int(tcfg.get("n_chunks_rst", auto_n_chunks(nrst_local, target_chunk_gb)))
    return {
        "attn_qk": chunk_size("attn_qk", nqk_local, n_chunks_qk),
        "attn_v": chunk_size("attn_v", nv_local, n_chunks_v),
        "rst": chunk_size("rst", nrst_local, n_chunks_rst),
        "n_chunks_qk": n_chunks_qk,
        "n_chunks_v": n_chunks_v,
        "n_chunks_rst": n_chunks_rst,
        "nqk_local": nqk_local,
        "nv_local": nv_local,
        "nrst_local": nrst_local,
    }


def create_or_reuse_sharded_fns(cfg: Dict[str, Any], mesh, *, analysis: bool = False):
    version = cfg.get("model", {}).get("model_version")
    train = get_train()
    module_name = train._model_registry_entry(version)["module"]
    mod = __import__(module_name, fromlist=["make_sharded_srw"])
    chunk = _chunk_sizes_for_cfg(cfg, mesh)
    base_kwargs = {"mesh": mesh}
    base_kwargs.update(train._v4164_sharded_kwargs(cfg))

    def factory_kwargs(factory, kwargs):
        sig = inspect.signature(factory)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    v4168_version = getattr(
        train, "V4168_MODEL_VERSION", "spatial-r1-v4.1.6.8")
    opspace_cfg = cfg.get("training", {}).get("operation_space", {})
    if not isinstance(opspace_cfg, dict):
        opspace_cfg = {}
    opspace_enabled = (
        version == v4168_version and bool(opspace_cfg.get("enabled", False)))
    opspace_pools = (
        opspace_cfg.get("pools", {})
        if isinstance(opspace_cfg.get("pools", {}), dict) else {})
    opspace_layouts = (
        train._v4168_operation_space_pool_layouts(
            cfg.get("training", {}), cfg.get("model", {}))
        if opspace_enabled else {})

    def opspace_pool_kwargs(pool):
        if not opspace_enabled:
            return {}
        pool_cfg = opspace_pools.get(pool, {})
        if not isinstance(pool_cfg, dict):
            pool_cfg = {}
        layout = opspace_layouts.get(pool, {})
        k_exec_default = {"qk": 4, "v": 5, "rst": 32}[pool]
        lanes_default = 32 if pool == "rst" else 8
        k_exec_cfg = layout.get(
            "k_exec", pool_cfg.get("k_exec", k_exec_default))
        return {
            "operation_space_mode": str(
                layout.get(
                    "routing_mode",
                    pool_cfg.get(
                        "routing_mode",
                        pool_cfg.get("mode", "factorized_lane_mean")))
            ).lower(),
            "operation_space_routing_mode": str(
                layout.get("routing_mode", "factorized_lane_mean")).lower(),
            "operation_space_execution_mode": str(
                layout.get("execution_mode", "dense_masked")).lower(),
            "opspace_lanes": int(layout.get(
                "lanes", pool_cfg.get("lanes", lanes_default))),
            "opspace_tiles_per_lane": int(layout.get("tiles_per_lane", 1)),
            "opspace_exec_tiles_per_block": int(layout.get(
                "exec_tiles_per_block", 1)),
            "opspace_blocks_per_lane": int(layout.get(
                "blocks_per_lane", 1)),
            "opspace_block_size": int(layout.get(
                "block_size", layout.get(
                    "tile_size", opspace_cfg.get("tile_size", 128)))),
            "opspace_padded_ops": int(layout.get("padded_ops", 0)),
            "opspace_tile_size": int(layout.get(
                "tile_size", opspace_cfg.get("tile_size", 128))),
            "opspace_k_exec": int(k_exec_cfg),
            "opspace_bucket_capacity_factor": float(layout.get(
                "bucket_capacity_factor",
                1.5 if str(layout.get(
                    "execution_mode", "dense_masked")).lower()
                == "block_bucketed_pallas_final" else 1.25)),
            "opspace_bucket_chunk_size": int(layout.get(
                "bucket_chunk_size",
                128 if str(layout.get(
                    "execution_mode", "dense_masked")).lower()
                == "block_bucketed_pallas_final" else 1024)),
            "opspace_assignment_policy": str(layout.get(
                "assignment_policy", "low_regret")).lower(),
            "opspace_high_regret_threshold": float(layout.get(
                "high_regret_threshold", 0.05)),
            "opspace_lane_output_mode": str(layout.get(
                "lane_output_mode", "lane_local")).lower(),
        }

    def srw_pool_kwargs(pool):
        kwargs = dict(base_kwargs_for_analysis)
        if train._is_v417x_version(version):
            kwargs["admission_den_power"] = float(
                cfg["model"][f"admission_den_power_{pool}"])
        if version == v4168_version:
            m_cfg = cfg.get("model", {})
            kwargs.update({
                "block_size": int(m_cfg.get(f"{pool}_block_size", 256)),
                "top_blocks": int(m_cfg.get(f"{pool}_top_blocks", 2)),
                "block_margin": float(m_cfg.get("block_margin", 0.0)),
            })
            kwargs.update(opspace_pool_kwargs(pool))
        return kwargs

    make_single = mod.make_sharded_srw
    make_paired = getattr(mod, "make_sharded_srw_paired", None)
    make_single_min = getattr(mod, "make_sharded_srw_minimal", None)
    make_paired_min = getattr(mod, "make_sharded_srw_paired_minimal", None)
    make_single_suppression_min = getattr(
        mod, "make_sharded_srw_suppression_minimal", None)
    make_paired_suppression_min = getattr(
        mod, "make_sharded_srw_paired_suppression_minimal", None)
    make_paired_dense_min = getattr(
        mod, "make_sharded_srw_paired_dense_minimal", None)

    base_kwargs_for_analysis = dict(base_kwargs)
    if analysis and "analysis" in inspect.signature(make_single).parameters:
        base_kwargs_for_analysis["analysis"] = True

    single_v = make_single(
        max_chunk_size=chunk["attn_v"],
        **factory_kwargs(make_single, srw_pool_kwargs("v")),
    )
    single_rst = make_single(
        max_chunk_size=chunk["rst"],
        **factory_kwargs(make_single, srw_pool_kwargs("rst")),
    )
    if make_paired is None:
        return single_rst
    paired = make_paired(
        max_chunk_size=chunk["attn_qk"],
        **factory_kwargs(make_paired, srw_pool_kwargs("qk")),
    )
    fns = {
        "single": single_v,
        "attn_v_single": single_v,
        "rst_single": single_rst,
        "paired": paired,
        "attn_qk_paired": paired,
    }
    if not analysis and make_single_min is not None:
        fns["attn_qk_single_minimal"] = make_single_min(
            max_chunk_size=chunk["attn_qk"],
            **factory_kwargs(make_single_min, srw_pool_kwargs("qk")),
        )
        fns["attn_v_single_minimal"] = make_single_min(
            max_chunk_size=chunk["attn_v"],
            **factory_kwargs(make_single_min, srw_pool_kwargs("v")),
        )
        fns["rst_single_minimal"] = make_single_min(
            max_chunk_size=chunk["rst"],
            **factory_kwargs(make_single_min, srw_pool_kwargs("rst")),
        )
        if make_single_suppression_min is not None:
            fns["attn_v_single_suppression_minimal"] = (
                make_single_suppression_min(
                    max_chunk_size=chunk["attn_v"],
                    **factory_kwargs(
                        make_single_suppression_min, srw_pool_kwargs("v")),
                ))
            fns["rst_single_suppression_minimal"] = (
                make_single_suppression_min(
                    max_chunk_size=chunk["rst"],
                    **factory_kwargs(
                        make_single_suppression_min, srw_pool_kwargs("rst")),
                ))
    paired_min_factory = make_paired_min
    if opspace_enabled and make_paired_dense_min is not None:
        paired_min_factory = make_paired_dense_min
    if not analysis and paired_min_factory is not None:
        fns["attn_qk_paired_minimal"] = paired_min_factory(
            max_chunk_size=chunk["attn_qk"],
            **factory_kwargs(paired_min_factory, srw_pool_kwargs("qk")),
        )
        if make_paired_suppression_min is not None:
            fns["attn_qk_paired_suppression_minimal"] = (
                make_paired_suppression_min(
                    max_chunk_size=chunk["attn_qk"],
                    **factory_kwargs(
                        make_paired_suppression_min,
                        srw_pool_kwargs("qk")),
                ))
    return fns


def shard_params_for_mesh(params: Any, cfg: Dict[str, Any], mesh):
    model_version = cfg.get("model", {}).get("model_version")
    train = get_train()
    param_shardings = train.get_param_shardings(params, mesh, model_version)
    return train.shard_params_to_mesh(params, param_shardings)


def restore_params_and_cfg(config: Dict[str, Any], checkpoint_path: str,
                           checkpoint_step: int, mesh) -> Tuple[Any, Dict[str, Any], Any]:
    model = build_analysis_model(config)
    target_params = init_target_params(model, config)
    target_params = shard_params_for_mesh(target_params, config, mesh)
    train = get_train()
    optimizer = _create_restore_optimizer(config, target_params)
    target_opt_state = optimizer.init(target_params)
    target_opt_state = train._replicate_optimizer_state_scalars_to_mesh(
        target_opt_state, mesh)
    target_state = train._build_orbax_state(
        target_params,
        target_opt_state,
        jax.random.PRNGKey(int(config.get("seed", 0))),
        epoch=0,
        global_step=0,
        step_in_epoch=0,
        steps_per_epoch=0,
        best_val_loss=float("inf"),
        training_config=config.get("training", {}),
        full_config=config,
        model_config=config.get("model", {}),
    )
    manager = train._create_orbax_checkpoint_manager(
        checkpoint_path,
        create=False,
        read_only=True,
    )
    try:
        restored = manager.restore(
            int(checkpoint_step),
            args=train.ocp.args.Composite(
                state=train.ocp.args.StandardRestore(target_state),
                metadata=train.ocp.args.JsonRestore(),
            ),
        )
        state = train._composite_item(restored, "state")
        metadata = train._composite_item(restored, "metadata")
        if not isinstance(state, dict) or "params" not in state:
            raise RuntimeError("Orbax restore did not return state.params.")
        params = train._match_tree_to_template_on_mesh(
            state["params"], target_params, mesh, name="params"
        )
        return params, (metadata if isinstance(metadata, dict) else {}), model
    finally:
        manager.close()


def _scalar_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(()))


def scheduled_gate_state_from_config(
        cfg: Dict[str, Any],
        checkpoint_step: int,
        total_training_steps: int) -> Dict[str, Any]:
    """Return trainer-matched runtime gate values for an analysis checkpoint."""
    t = cfg.get("training", {})
    train = get_train()
    soft_gate_t_start = float(t.get("soft_gate_t_start", 1.5))
    soft_gate_t_final = float(t.get("soft_gate_t_final", 0.07))
    soft_gate_t_hold_frac = float(t.get("soft_gate_t_hold_frac", 0.10))
    soft_gate_t_anneal_end_frac = float(t.get("soft_gate_t_anneal_end_frac", 0.80))
    soft_gate_schedule = str(t.get(
        "soft_gate_t_schedule",
        t.get("soft_gate_schedule", "cosine")))
    soft_gate_t_power = float(t.get("soft_gate_t_power", 4.0))
    soft_gate_t_gompertz_center = float(t.get("soft_gate_t_gompertz_center", 0.25))
    soft_gate_t_gompertz_steepness = float(
        t.get("soft_gate_t_gompertz_steepness", 8.0))
    pool_schedules = train._training_soft_gate_pool_schedules(
        t,
        soft_gate_t_start,
        soft_gate_t_final,
        soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac,
        soft_gate_schedule,
        soft_gate_t_power,
        soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness,
    )
    step = jnp.asarray(int(checkpoint_step), dtype=jnp.float32)
    total = jnp.asarray(max(1, int(total_training_steps or 1)), dtype=jnp.float32)
    qk_t = train._scheduled_from_config(step, total, pool_schedules["qk"])
    v_t = train._scheduled_from_config(step, total, pool_schedules["v"])
    rst_t = train._scheduled_from_config(step, total, pool_schedules["rst"])
    boundary_power = train.scheduled_boundary_power_by_frac(
        step,
        total,
        True,
        float(t.get("soft_gate_boundary_power_start", 3.0)),
        float(t.get("soft_gate_boundary_power_mid", 3.15)),
        float(t.get("soft_gate_boundary_power_final", 4.0)),
        float(t.get("soft_gate_boundary_power_mid_frac", 0.800)),
        float(t.get("soft_gate_boundary_power_final_frac", 0.950)),
        float(t.get("soft_gate_boundary_power_start_frac", 0.0)),
    )
    qk_t_f = _scalar_float(qk_t)
    return {
        "source": "train_jax_schedule",
        "checkpoint_step": int(checkpoint_step),
        "total_training_steps": int(total_training_steps or 1),
        "soft_gate_temperature": qk_t_f,
        "soft_gate_T": qk_t_f,
        "soft_gate_T_qk": qk_t_f,
        "soft_gate_T_v": _scalar_float(v_t),
        "soft_gate_T_rst": _scalar_float(rst_t),
        "soft_gate_boundary_power": _scalar_float(boundary_power),
        "soft_gate_boundary_power_final": float(
            t.get("soft_gate_boundary_power_final", 4.0)),
    }


def model_cfg_from_config(
        cfg: Dict[str, Any],
        checkpoint_step: Optional[int] = None,
        total_training_steps: Optional[int] = None) -> Dict[str, Any]:
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    legacy_den_power = float(m.get(
        "admission_den_power", t.get("admission_den_power", 1.0)))
    model_cfg = {
        "model_version": m.get("model_version", V4166_MODEL_VERSION),
        "vocab_size": int(m.get("vocab_size", 30522)),
        "d_model": int(m.get("d_model", 384)),
        "n_layers": int(m.get("n_layers", 12)),
        "n_heads": int(m.get("n_heads", 6)),
        "max_seq_len": int(m.get("max_seq_len", 512)),
        "d_route": int(m.get("d_route", m.get("d_bottleneck", 128))),
        "n_qk": int(m.get("n_qk", 1580)),
        "n_v": int(m.get("n_v", 2600)),
        "n_rst": int(m.get("n_rst", m.get("n_know", 25200))),
        "n_know": int(m.get("n_know", m.get("n_rst", 25200))),
        "soft_gate_temperature": float(t.get("soft_gate_t_final", 0.07)),
        "soft_gate_boundary_power": float(t.get("soft_gate_boundary_power_final", 4.0)),
        "admission_den_power": legacy_den_power,
        "admission_den_power_qk": float(m.get(
            "admission_den_power_qk", legacy_den_power)),
        "admission_den_power_v": float(m.get(
            "admission_den_power_v", legacy_den_power)),
        "admission_den_power_rst": float(m.get(
            "admission_den_power_rst", legacy_den_power)),
        "srw_composition_mode": str(m.get(
            "srw_composition_mode", "linear_angular")),
        "heat_kernel_beta": float(m.get("heat_kernel_beta", 2.0)),
        "soft_gate_effective_active_eps": float(t.get("soft_gate_effective_active_eps", 1e-6)),
        "execution_prune_eps": float(m.get("execution_prune_eps", 0.0)),
    }
    for key in (
        "logical_vocab_size",
        "vocab_size_padded",
        "operator_key_mode",
        "operator_query_mode",
    ):
        if m.get(key) is not None:
            model_cfg[key] = m[key]
    if checkpoint_step is not None and total_training_steps is not None:
        gate_state = scheduled_gate_state_from_config(
            cfg, int(checkpoint_step), int(total_training_steps))
        model_cfg.update(gate_state)
        model_cfg["runtime_gate_state"] = dict(gate_state)
    return model_cfg


def admission_den_power_from_config(cfg: Dict[str, Any]) -> float:
    """Resolve the model-owned v417x value with legacy training fallback."""
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    return float(model_cfg.get(
        "admission_den_power", training_cfg.get("admission_den_power", 1.0)))


def admission_den_powers_from_config(
        cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return legacy, QK, V, and RST powers with legacy fallback."""
    model_cfg = cfg.get("model", {})
    legacy = admission_den_power_from_config(cfg)
    return (
        legacy,
        float(model_cfg.get("admission_den_power_qk", legacy)),
        float(model_cfg.get("admission_den_power_v", legacy)),
        float(model_cfg.get("admission_den_power_rst", legacy)),
    )


def count_params(params: Any) -> int:
    leaves = jax.tree.leaves(params)
    return int(sum(getattr(x, "size", 0) for x in leaves))


def _create_restore_optimizer(cfg: Dict[str, Any], params: Any):
    """Recreate the training optimizer state structure for Orbax restore.

    Orbax StandardRestore requires the saved state tree to match the target
    tree.  We do not use optimizer updates during analysis, but constructing
    this opt_state lets Orbax restore params directly into the target mesh
    without loading an unsharded checkpoint blob first.
    """
    import optax

    tcfg = cfg.get("training", {})
    lr = float(tcfg.get("lr", tcfg.get("learning_rate", 6.5e-4)))
    weight_decay = float(tcfg.get("weight_decay", 0.1))
    pool_weight_decay = float(tcfg.get("pool_weight_decay", 0.0))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.06))
    global_grad_clip = float(tcfg.get("global_grad_clip", 0.0))
    grad_accum_steps = int(tcfg.get("gradient_accumulation_steps", 1))
    total_steps = max(1, int(tcfg.get("total_steps_for_restore", 1)))
    warmup_steps = int(total_steps * warmup_ratio)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )

    pool_param_names = (
        "attn_qk_emb", "attn_v_emb", "rst_emb",
        "attn_qk_op_key", "attn_v_op_key", "rst_op_key",
        "rw_key_read_probe", "rw_key_write_probe",
        "attn_qk_op_read_proj", "attn_qk_op_write_proj",
        "attn_v_op_read_proj", "attn_v_op_write_proj",
        "rst_op_read_proj", "rst_op_write_proj",
        "qk_emb", "v_emb", "q_read", "k_read",
        "attn_qk_read", "attn_v_read", "rst_read",
        "qk_read", "v_read", "q_write", "k_write",
        "attn_qk_write", "attn_v_write", "rst_write",
        "qk_write", "v_write",
    )

    def path_str(path):
        return "/".join(str(p.key if hasattr(p, "key") else p) for p in path)

    def is_pool_param(ps: str) -> bool:
        return any(name in ps for name in pool_param_names)

    def is_excluded(ps: str) -> bool:
        leaf = ps.rsplit("/", 1)[-1]
        if leaf == "bias":
            return True
        if "scale" in ps and "norm" in ps.lower():
            return True
        if (
            ps.endswith("_scale")
            or ps.endswith("/qk_scale")
            or ps.endswith("/v_scale")
            or ps.endswith("/rst_scale")
            or ps.endswith("/attn_qk_scale")
            or ps.endswith("/attn_v_scale")
        ):
            return True
        return False

    def wd_mask_base(p):
        return jax.tree.map_with_path(
            lambda path, _: (not is_excluded(path_str(path))) and (not is_pool_param(path_str(path))),
            p,
        )

    def wd_mask_pool(p):
        return jax.tree.map_with_path(
            lambda path, _: (not is_excluded(path_str(path))) and is_pool_param(path_str(path)),
            p,
        )

    def no_param_mask(p):
        return jax.tree.map(lambda _: False, p)

    parts = [optax.masked(optax.set_to_zero(), mask=no_param_mask)]
    if global_grad_clip > 0.0:
        parts.append(optax.clip_by_global_norm(global_grad_clip))
    else:
        parts.append(optax.scale(1.0))
    parts.extend([
        optax.scale_by_adam(b2=0.95),
        optax.add_decayed_weights(weight_decay, mask=wd_mask_base),
        optax.add_decayed_weights(pool_weight_decay, mask=wd_mask_pool),
        optax.scale_by_learning_rate(schedule),
        optax.masked(optax.set_to_zero(), mask=no_param_mask),
    ])
    base = optax.chain(*parts)
    if grad_accum_steps > 1:
        return optax.MultiSteps(base, every_k_schedule=grad_accum_steps)
    return base


def create_ce_eval_step(model, sharded_fns=None, *, minimal_train: bool = True,
                        return_prune_stats: bool = False,
                        return_pool_prune_stats: bool = False,
                        execution_prune_eps: float = 0.0,
                        total_training_steps: int = 1,
                        cfg: Optional[Dict[str, Any]] = None):
    """Trainer-matched CE eval step used by eval/prune stages."""
    cfg = cfg or {}
    t = cfg.get("training", {})
    train = get_train()
    soft_gate_t_start = float(t.get("soft_gate_t_start", 1.5))
    soft_gate_t_final = float(t.get("soft_gate_t_final", 0.07))
    soft_gate_t_hold_frac = float(t.get("soft_gate_t_hold_frac", 0.10))
    soft_gate_t_anneal_end_frac = float(t.get("soft_gate_t_anneal_end_frac", 0.80))
    soft_gate_schedule = str(t.get("soft_gate_schedule", "cosine"))
    soft_gate_t_power = float(t.get("soft_gate_t_power", 4.0))
    soft_gate_t_gompertz_center = float(t.get("soft_gate_t_gompertz_center", 0.25))
    soft_gate_t_gompertz_steepness = float(t.get("soft_gate_t_gompertz_steepness", 8.0))
    pool_schedules = train._training_soft_gate_pool_schedules(
        t,
        soft_gate_t_start,
        soft_gate_t_final,
        soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac,
        soft_gate_schedule,
        soft_gate_t_power,
        soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness,
    )
    eval_step = train.create_eval_step(
        model,
        sharded_fns=sharded_fns,
        return_dead_stats=bool(return_prune_stats),
        return_prune_stats=bool(return_prune_stats),
        execution_prune_eps=float(execution_prune_eps),
        total_training_steps=int(total_training_steps or 1),
        soft_gate_schedule_active=True,
        soft_gate_t_start=soft_gate_t_start,
        soft_gate_t_final=soft_gate_t_final,
        soft_gate_t_hold_frac=soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac=soft_gate_t_anneal_end_frac,
        soft_gate_schedule=soft_gate_schedule,
        soft_gate_t_power=soft_gate_t_power,
        soft_gate_t_gompertz_center=soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness=soft_gate_t_gompertz_steepness,
        pool_specific_gate_t=True,
        soft_gate_pool_schedules=pool_schedules,
        boundary_power_schedule_active=True,
        soft_gate_boundary_power_start=float(t.get("soft_gate_boundary_power_start", 3.0)),
        soft_gate_boundary_power_mid=float(t.get("soft_gate_boundary_power_mid", 3.15)),
        soft_gate_boundary_power_final=float(t.get("soft_gate_boundary_power_final", 4.0)),
        soft_gate_boundary_power_start_frac=float(t.get("soft_gate_boundary_power_start_frac", 0.0)),
        soft_gate_boundary_power_mid_frac=float(t.get("soft_gate_boundary_power_mid_frac", 0.800)),
        soft_gate_boundary_power_final_frac=float(t.get("soft_gate_boundary_power_final_frac", 0.950)),
        admission_den_power=admission_den_power_from_config(cfg),
    )

    def step(params, input_ids, attention_mask, current_step):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        ret = eval_step(
            params, input_ids, labels, attention_mask, current_step)
        if not return_prune_stats:
            return ret
        (
            loss,
            correct,
            valid_count,
            _attn_dead_count,
            _rst_dead_count,
            compute_frac,
            mass_retained,
            gate_den,
            _gate_den_min,
            no_active,
            unpruned_den,
            *pool_stats,
        ) = ret
        base = (
            loss,
            correct,
            valid_count,
            compute_frac,
            mass_retained,
            gate_den,
            no_active,
            unpruned_den,
        )
        if return_pool_prune_stats:
            return base + tuple(pool_stats)
        return base

    return step


def create_active_analysis_step(model, sharded_fns=None, *,
                                total_training_steps: int = 1,
                                cfg: Optional[Dict[str, Any]] = None):
    """Trainer-matched full-stat analysis step for light side analysis."""
    cfg = cfg or {}
    t = cfg.get("training", {})
    train = get_train()
    soft_gate_t_start = float(t.get("soft_gate_t_start", 1.5))
    soft_gate_t_final = float(t.get("soft_gate_t_final", 0.07))
    soft_gate_t_hold_frac = float(t.get("soft_gate_t_hold_frac", 0.10))
    soft_gate_t_anneal_end_frac = float(t.get("soft_gate_t_anneal_end_frac", 0.80))
    soft_gate_schedule = str(t.get("soft_gate_schedule", "cosine"))
    soft_gate_t_power = float(t.get("soft_gate_t_power", 4.0))
    soft_gate_t_gompertz_center = float(t.get("soft_gate_t_gompertz_center", 0.25))
    soft_gate_t_gompertz_steepness = float(t.get("soft_gate_t_gompertz_steepness", 8.0))
    pool_schedules = train._training_soft_gate_pool_schedules(
        t,
        soft_gate_t_start,
        soft_gate_t_final,
        soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac,
        soft_gate_schedule,
        soft_gate_t_power,
        soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness,
    )
    return train.create_analysis_step(
        model,
        sharded_fns=sharded_fns,
        total_training_steps=int(total_training_steps or 1),
        soft_gate_schedule_active=True,
        soft_gate_t_start=soft_gate_t_start,
        soft_gate_t_final=soft_gate_t_final,
        soft_gate_t_hold_frac=soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac=soft_gate_t_anneal_end_frac,
        soft_gate_schedule=soft_gate_schedule,
        soft_gate_t_power=soft_gate_t_power,
        soft_gate_t_gompertz_center=soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness=soft_gate_t_gompertz_steepness,
        pool_specific_gate_t=True,
        soft_gate_pool_schedules=pool_schedules,
        boundary_power_schedule_active=True,
        soft_gate_boundary_power_start=float(t.get("soft_gate_boundary_power_start", 3.0)),
        soft_gate_boundary_power_mid=float(t.get("soft_gate_boundary_power_mid", 3.15)),
        soft_gate_boundary_power_final=float(t.get("soft_gate_boundary_power_final", 4.0)),
        soft_gate_boundary_power_start_frac=float(t.get("soft_gate_boundary_power_start_frac", 0.0)),
        soft_gate_boundary_power_mid_frac=float(t.get("soft_gate_boundary_power_mid_frac", 0.800)),
        soft_gate_boundary_power_final_frac=float(t.get("soft_gate_boundary_power_final_frac", 0.950)),
        admission_den_power=admission_den_power_from_config(cfg),
        ce_token_chunk_size=int(t.get("ce_token_chunk_size", 32768)),
    )


def create_composition_analysis_step(
        model, sharded_fns=None, *, cfg: Optional[Dict[str, Any]] = None):
    """Return the v417x minimal-path composition diagnostics step.

    The trainer's full ``analysis=True`` path owns per-layer diagnostics, while
    v417x's exact admission-mass/composition-denominator scalars are emitted by
    its production minimal path. Run this only when the composition item is
    selected so other analysis presets do not pay for a second forward.
    """
    cfg = cfg or {}
    version = cfg.get("model", {}).get("model_version")
    if version not in V417X_MODEL_VERSIONS:
        raise ValueError(
            "Composition analysis is only defined for "
            f"{V417X_MODEL_VERSIONS}, got {version!r}")
    t = cfg.get("training", {})
    admission_den_power = admission_den_power_from_config(cfg)
    ce_token_chunk_size = int(t.get("ce_token_chunk_size", 32768))
    soft_gate_temperature = float(t.get("soft_gate_t_final", 0.07))
    soft_gate_boundary_power = float(
        t.get("soft_gate_boundary_power_final", 4.0))

    @jax.jit
    def step(params, input_ids, attention_mask):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        result = model.apply(
            {"params": params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(0)},
            sharded_fns=sharded_fns,
            analysis=False,
            minimal_train=True,
            admission_den_power=admission_den_power,
            soft_gate_temperature=soft_gate_temperature,
            soft_gate_boundary_power=soft_gate_boundary_power,
            soft_gate_boundary_power_final=soft_gate_boundary_power,
            execution_prune_eps=jnp.float32(0.0),
            ce_token_chunk_size=ce_token_chunk_size,
            compute_accuracy=False,
        )
        keys = (
            "admission_den_power",
            *(
                f"{pool}_{name}"
                for pool in ("attn_qk", "attn_v", "rst")
                for name in (
                    "admission_mass_mean", "admission_mass_max",
                    "composition_den_mean", "composition_den_min",
                    "composition_den_max", "composition_den_floor_frac",
                )
            ),
        )
        return {key: result[key] for key in keys}

    return step


def load_eval_data(cfg: Dict[str, Any], max_length: int, batch_size: int,
                   host_id: int, n_hosts: int, max_tokens: Optional[int] = None):
    """Load validation data only, preserving train_jax's flat per-host batches."""
    from utils.data_jax import BinDataLoader, _build_dataset

    data_cfg = dict(cfg.get("data", {}))
    val_path = data_cfg.get("bin_val")
    if not val_path:
        raise ValueError("config.data.bin_val is required for analysis.")
    local_cache_dir = data_cfg.get("local_cache_dir", None)
    max_sequences = None
    if max_tokens is not None:
        max_sequences = max(1, int(max_tokens) // int(max_length))
    elif data_cfg.get("max_val_tokens"):
        max_sequences = max(1, int(data_cfg["max_val_tokens"]) // int(max_length))
    dataset = _build_dataset(
        val_path,
        int(max_length),
        max_sequences,
        local_cache_dir,
        evict_previous_cache=False,
    )
    loader = BinDataLoader(
        dataset,
        batch_size=int(batch_size),
        n_devices=1,
        n_hosts=int(n_hosts),
        host_id=int(host_id),
    )
    return loader


def host_aligned_batch_size(batch_size: int, n_hosts: int) -> int:
    """Round a global batch size up so BinDataLoader can split it by host."""
    batch_size = max(1, int(batch_size))
    n_hosts = max(1, int(n_hosts))
    if batch_size % n_hosts == 0:
        return batch_size
    return ((batch_size + n_hosts - 1) // n_hosts) * n_hosts


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    try:
        seconds_f = float(seconds)
    except Exception:
        return "unknown"
    if not math.isfinite(seconds_f) or seconds_f < 0:
        return "unknown"
    total = int(round(seconds_f))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def sync_hosts(label: str) -> None:
    if int(jax.process_count()) <= 1:
        return
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(str(label))


def shard_batch_to_mesh(input_ids, attention_mask, data_sharding):
    train = get_train()
    gb = int(input_ids.shape[0]) * int(jax.process_count())
    gs = (gb, int(input_ids.shape[1]))
    return (
        train.shard_to_mesh(input_ids, data_sharding, gs),
        train.shard_to_mesh(attention_mask, data_sharding, gs),
    )


def assign_jobs_to_host(jobs: Sequence[Any], host_id: int, n_hosts: int) -> List[Any]:
    return [job for i, job in enumerate(jobs) if i % int(n_hosts) == int(host_id)]


def maybe_load_tokenizer(local_only: bool = True):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=local_only)
    except Exception:
        return None


def token_window_text(tokenizer, ids: Sequence[int], center: int, radius: int = 8) -> str:
    start = max(0, int(center) - int(radius))
    end = min(len(ids), int(center) + int(radius) + 1)
    window = [int(x) for x in ids[start:end]]
    if tokenizer is not None:
        try:
            return tokenizer.decode(window, clean_up_tokenization_spaces=False)
        except Exception:
            pass
    return "ids:" + " ".join(str(x) for x in window)


def write_run_metadata(store: AnalysisStore, cfg: Dict[str, Any],
                       args: Any, checkpoint_metadata: Dict[str, Any],
                       model_info: Optional[Dict[str, Any]] = None) -> None:
    if not store.is_primary:
        return
    store.ensure_layout()
    selected_stages = getattr(args, "stages", None)
    manifest_fields = {
        "source_config_path": getattr(args, "config", None),
        "checkpoint_path": getattr(args, "checkpoint", None),
        "checkpoint_step": checkpoint_metadata.get("global_step", getattr(args, "checkpoint_step", None)),
        "checkpoint_metadata": json_safe(checkpoint_metadata),
        "model_version": cfg.get("model", {}).get("model_version"),
        "data_source": cfg.get("data", {}).get("bin_val"),
        "tokenizer": "bert-base-uncased",
        "vocab_size": cfg.get("model", {}).get("vocab_size"),
        "selected_stages": selected_stages,
        "per_stage_config": {
            "eval_max_tokens": getattr(args, "eval_max_tokens", None),
            "prune_eps": getattr(args, "prune_eps", None),
            "usage_max_sequences": getattr(args, "usage_max_sequences", None),
            "trace_max_prompts": getattr(args, "trace_max_prompts", None),
            "ablation_max_sequences": getattr(args, "ablation_max_sequences", None),
        },
        **git_info(),
        **host_info(),
    }
    if model_info:
        manifest_fields.update(model_info)
    store.update_manifest(**manifest_fields)
    write_json_atomic(store.path("config_snapshot.json"), cfg)
    if model_info:
        write_json_atomic(store.path("model_info.json"), model_info)
