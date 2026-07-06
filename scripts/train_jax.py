"""
DAWN-SRW v4.1.6.4 JAX trainer (TPU multi-device).

Active DAWN-SRW training path:
- model versions are registered in MODEL_REGISTRY
- JAX/Flax SPMD training with sharded SRW pools
- GCS checkpoint support and auto-resume
- optax optimizer with warmup + cosine decay

Usage:
    python scripts/train_jax.py --config configs/train_config_v4164_40M_c4_5B_ggauss_boundary_panneal.yaml
    python scripts/train_jax.py --config configs/train_config_v4166_40M_c4_5B_ggauss_boundary_panneal.yaml
    python scripts/train_jax.py --config configs/train_config_v4164_40M_c4_5B_ggauss_boundary_panneal.yaml --resume-from gs://.../run_v...
    python scripts/train_jax.py --config configs/train_config_v4164_40M_c4_5B_ggauss_boundary_panneal.yaml --from-scratch
"""

import sys
import os
import signal
import json
import re
import math
import subprocess
import inspect
import hashlib
import socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather
try:
    from jax.experimental.multihost_utils import broadcast_one_to_all as _bcast_one_to_all
    _HAVE_BROADCAST = True
except ImportError:
    _bcast_one_to_all = None
    _HAVE_BROADCAST = False
import optax
try:
    import orbax.checkpoint as ocp
    _ORBAX_IMPORT_ERROR = None
except Exception as exc:
    ocp = None
    _ORBAX_IMPORT_ERROR = exc
import numpy as np
import time
import random
import argparse
from importlib import metadata as importlib_metadata
import yaml
import numpy as np
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Any, Callable, Optional

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map

# Official v4.1.6.4 model path.
from models.dawn_srw_v4164 import (
    DAWN as DAWN_SRW_V4164,
    _raw_tau_init_from_cosine_tau as _v4164_raw_tau_init_from_cosine_tau,
    _tau_init_calibration_scores as _v4164_tau_init_calibration_scores,
)
from models.dawn_srw_v4166 import (
    DAWN_SRW_V4166,
    _pool_operator_keys as _v4166_pool_operator_keys,
    _raw_tau_init_from_cosine_tau as _v4166_raw_tau_init_from_cosine_tau,
    _tau_init_calibration_scores as _v4166_tau_init_calibration_scores,
)
from models.dawn_srw_v4167 import (
    DAWN_SRW_V4167,
    _pool_operator_keys as _v4167_pool_operator_keys,
    _raw_tau_init_from_cosine_tau as _v4167_raw_tau_init_from_cosine_tau,
    _tau_init_calibration_scores as _v4167_tau_init_calibration_scores,
)
from models.dawn_srw_v4168 import (
    DAWN_SRW_V4168,
    OPSPACE_FINAL_RUNTIME_DIAG_NAMES as _V4168_OPSPACE_FINAL_RUNTIME_DIAG_NAMES,
    OPSPACE_RUNTIME_DIAG_NAMES as _V4168_OPSPACE_RUNTIME_DIAG_NAMES,
    hardware_sector_static_metrics as _v4168_hardware_sector_static_metrics,
    maybe_hardware_repack as _v4168_maybe_hardware_repack,
    operation_space_static_metrics as _v4168_operation_space_static_metrics,
    maybe_operation_space_repack as _v4168_maybe_operation_space_repack,
    _pool_operator_keys as _v4168_pool_operator_keys,
    _raw_tau_init_from_cosine_tau as _v4168_raw_tau_init_from_cosine_tau,
    _tau_init_calibration_scores as _v4168_tau_init_calibration_scores,
)
from models.baseline_transformer_jax import (
    VanillaTransformer,
    create_baseline_sharded_fns,
)


def _jax_distributed_is_initialized():
    try:
        # Safe after initialization. Before initialization this may still
        # return values in single-process cases, so do not rely on it alone
        # for policy.
        return bool(getattr(jax.distributed, "is_initialized", lambda: False)())
    except Exception:
        return False


def _maybe_initialize_jax_distributed():
    """Initialize JAX distributed before any backend/device access.

    This trainer uses TPU multi-host execution and Orbax CheckpointManager.
    Orbax CheckpointManager requires a JAX distributed runtime for multihost
    save/restore progress synchronization.

    On Cloud TPU, JAX supports no-argument auto-detection, so this trainer
    always calls `jax.distributed.initialize()` as a runtime invariant.
    """
    if _jax_distributed_is_initialized():
        print("JAX distributed runtime already initialized.", flush=True)
        return True

    try:
        print(
            "Initializing jax.distributed with Cloud TPU auto-detection.",
            flush=True,
        )
        jax.distributed.initialize()
        print(
            "Initialized jax.distributed: "
            f"process_index={jax.process_index()} "
            f"process_count={jax.process_count()}",
            flush=True,
        )
        return True
    except RuntimeError as exc:
        if "already initialized" in str(exc).lower():
            print("jax.distributed already initialized.", flush=True)
            return True
        raise RuntimeError(
            "Failed to initialize jax.distributed before device access. "
            "This trainer requires JAX distributed for TPU multi-host and "
            "Orbax CheckpointManager checkpointing."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize jax.distributed before device access. "
            "This trainer requires JAX distributed for TPU multi-host and "
            "Orbax CheckpointManager checkpointing."
        ) from exc


_MULTIHOST_STARTUP_CONTEXT = {}


def _stable_short_hash(value):
    return hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:8]


def _assert_multihost_same_startup_context(context, max_len=65536):
    """Fail if TPU hosts launched with different startup-critical context."""
    local_context = dict(context or {})
    local_context.setdefault('process_index', jax.process_index())
    local_context.setdefault('host_id', jax.process_index())
    local_context.setdefault('process_count', jax.process_count())

    payload = json.dumps(
        local_context,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    encoded = payload.encode('utf-8')
    if len(encoded) > max_len:
        raise ValueError(
            f"startup context JSON too large: {len(encoded)} > {max_len}"
        )

    buf = np.zeros(max_len, dtype=np.uint8)
    buf[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    gathered = np.asarray(process_allgather(buf))
    gathered = gathered.reshape((jax.process_count(), max_len))

    contexts = []
    for row in gathered:
        raw = bytes(row).rstrip(b'\x00')
        contexts.append(json.loads(raw.decode('utf-8')) if raw else {})
    contexts = sorted(
        contexts,
        key=lambda item: int(item.get('process_index', -1)),
    )

    ignored = {'host_id', 'hostname', 'process_index'}
    comparable = [
        {k: v for k, v in item.items() if k not in ignored}
        for item in contexts
    ]
    expected = comparable[0] if comparable else {}
    if any(item != expected for item in comparable):
        print(
            "Distributed startup context mismatch across hosts:\n"
            + json.dumps(contexts, indent=2, sort_keys=True, default=str),
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            "Distributed startup context mismatch across hosts; all hosts "
            "must launch the same script, config, checkpoint path, and mesh "
            "settings before Orbax CheckpointManager construction."
        )

    global _MULTIHOST_STARTUP_CONTEXT
    _MULTIHOST_STARTUP_CONTEXT = dict(local_context)
    if jax.process_index() == 0:
        print(
            "Distributed startup context check passed: "
            f"script={local_context.get('trainer_script')} "
            f"config={local_context.get('config_path')} "
            f"checkpoint_dir={local_context.get('checkpoint_dir')}",
            flush=True,
        )
    return contexts


def _print_jax_distributed_identity(trainer_script, config_path,
                                    checkpoint_dir, from_scratch,
                                    resume_step):
    print("=== JAX distributed identity ===", flush=True)
    print(f"hostname={socket.gethostname()}", flush=True)
    print(f"process_index={jax.process_index()}", flush=True)
    print(f"process_count={jax.process_count()}", flush=True)
    print(f"local_device_count={jax.local_device_count()}", flush=True)
    print(f"global_device_count={jax.device_count()}", flush=True)
    print(f"trainer_script={trainer_script}", flush=True)
    print(f"config_path={config_path}", flush=True)
    print(f"checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"from_scratch={bool(from_scratch)}", flush=True)
    print(f"resume_step={resume_step}", flush=True)


def _strict_multihost_barrier(name: str, context=None):
    from jax.experimental import multihost_utils
    try:
        print(
            f"[process {jax.process_index()}] entering barrier {name} "
            f"context={context}",
            flush=True,
        )
        multihost_utils.sync_global_devices(name)
        print(
            f"[process {jax.process_index()}] passed barrier {name}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[process {jax.process_index()}] FAILED barrier {name} "
            f"context={context}: {exc}",
            flush=True,
        )
        raise


# ============================================================
# Constants
# ============================================================

# Log cadence is config-driven: see log_interval / log_analysis_multiplier
# in `training:`.

V4164_MODEL_VERSION = 'spatial-r1-v4.1.6.4'
V4166_MODEL_VERSION = 'spatial-r1-v4.1.6.6'
V4167_MODEL_VERSION = 'spatial-r1-v4.1.6.7'
V4168_MODEL_VERSION = 'spatial-r1-v4.1.6.8'
BASELINE_MODEL_VERSION = 'baseline-JAX'
LEGACY_BASELINE_MODEL_VERSION = 'baseline'
OFFICIAL_MODEL_VERSION = V4164_MODEL_VERSION
ACTIVE_SRW_MODEL_VERSIONS = (
    V4164_MODEL_VERSION, V4166_MODEL_VERSION, V4167_MODEL_VERSION,
    V4168_MODEL_VERSION)
RW_KEY_SRW_MODEL_VERSIONS = (
    V4166_MODEL_VERSION, V4167_MODEL_VERSION, V4168_MODEL_VERSION)
FIXED_TAU_SRW_MODEL_VERSIONS = (V4167_MODEL_VERSION,)
CHECKPOINT_SCHEMA_VERSION = 3
DEFAULT_SELECTION_CALIBRATION_SCORE_CHUNK_TOKENS = 2048
MODEL_REGISTRY = {
    BASELINE_MODEL_VERSION: {
        'class': VanillaTransformer,
        'module': 'models.baseline_transformer_jax',
        'raw_tau_init_from_cosine_tau': None,
        'tau_init_calibration_scores': None,
        'is_baseline': True,
    },
    V4164_MODEL_VERSION: {
        'class': DAWN_SRW_V4164,
        'module': 'models.dawn_srw_v4164',
        'raw_tau_init_from_cosine_tau': _v4164_raw_tau_init_from_cosine_tau,
        'tau_init_calibration_scores': _v4164_tau_init_calibration_scores,
    },
    V4166_MODEL_VERSION: {
        'class': DAWN_SRW_V4166,
        'module': 'models.dawn_srw_v4166',
        'raw_tau_init_from_cosine_tau': _v4166_raw_tau_init_from_cosine_tau,
        'tau_init_calibration_scores': _v4166_tau_init_calibration_scores,
    },
    V4167_MODEL_VERSION: {
        'class': DAWN_SRW_V4167,
        'module': 'models.dawn_srw_v4167',
        'raw_tau_init_from_cosine_tau': _v4167_raw_tau_init_from_cosine_tau,
        'tau_init_calibration_scores': _v4167_tau_init_calibration_scores,
    },
    V4168_MODEL_VERSION: {
        'class': DAWN_SRW_V4168,
        'module': 'models.dawn_srw_v4168',
        'raw_tau_init_from_cosine_tau': _v4168_raw_tau_init_from_cosine_tau,
        'tau_init_calibration_scores': _v4168_tau_init_calibration_scores,
    },
}


def _orbax_checkpoint_version():
    try:
        return importlib_metadata.version("orbax-checkpoint")
    except importlib_metadata.PackageNotFoundError:
        return "<not installed>"
    except Exception as exc:
        return f"<unknown: {type(exc).__name__}: {exc}>"


def _get_dotted_attr(root, dotted):
    current = root
    for part in dotted.split('.'):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _require_orbax_checkpoint_compat():
    version = _orbax_checkpoint_version()
    if _ORBAX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import orbax.checkpoint. Install the tested "
            "checkpoint dependency with `pip install orbax-checkpoint==0.11.24`. "
            f"Detected orbax-checkpoint version: {version}. "
            f"Import error: {_ORBAX_IMPORT_ERROR}") from _ORBAX_IMPORT_ERROR

    required = (
        'CheckpointManager',
        'CheckpointManagerOptions',
        'args.Composite',
        'args.StandardSave',
        'args.StandardRestore',
        'args.JsonSave',
        'args.JsonRestore',
    )
    missing = [
        f"ocp.{name}"
        for name in required
        if _get_dotted_attr(ocp, name) is None
    ]
    if missing:
        raise RuntimeError(
            "Installed orbax-checkpoint is missing required APIs for this "
            f"Orbax-only trainer. Version: {version}. Missing: "
            + ", ".join(missing))
    print(f"Detected orbax-checkpoint version: {version}", flush=True)
    return version


def _is_active_srw_version(version):
    return str(version) in ACTIVE_SRW_MODEL_VERSIONS


def _is_baseline_version(version):
    return str(version) in (
        BASELINE_MODEL_VERSION, LEGACY_BASELINE_MODEL_VERSION)


def _is_rw_key_srw_version(version):
    return str(version) in RW_KEY_SRW_MODEL_VERSIONS


def _is_fixed_tau_srw_version(version):
    return str(version) in FIXED_TAU_SRW_MODEL_VERSIONS


def _pool_operator_keys_for_version(version):
    version = str(version)
    if version == V4166_MODEL_VERSION:
        return _v4166_pool_operator_keys
    if version == V4167_MODEL_VERSION:
        return _v4167_pool_operator_keys
    if version == V4168_MODEL_VERSION:
        return _v4168_pool_operator_keys
    raise ValueError(f"{version} does not expose RW-derived operator keys.")


def _model_registry_entry(version):
    if str(version) == LEGACY_BASELINE_MODEL_VERSION:
        version = BASELINE_MODEL_VERSION
    try:
        return MODEL_REGISTRY[str(version)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model_version={version!r}; "
            f"supported={list(MODEL_REGISTRY)}") from exc

DIRECT_TAU_SELECT_METRIC_NAMES = (
    'rho_mean', 'rho_std', 'rho_max',
    'tau_mean', 'tau_min', 'tau_max',
    'raw_tau_mean', 'raw_tau_min', 'raw_tau_max',
    'selection_margin_mean',
    'positive_margin_mean', 'positive_margin_max',
    'selected_frac', 'no_active_frac',
)
DIRECT_TAU_EXPOSURE_METRIC_NAMES = (
    'angular_exposure_mean', 'angular_exposure_min',
    'angular_exposure_max', 'dead_exposure_frac',
    'weak_exposure_frac', 'dead_exposure_target',
)
DIRECT_TAU_SPARSITY_METRIC_NAMES = (
    'active_tau_frac', 'active_tau_count',
    'admission_active_eps_1e_6_frac',
    'admission_active_eps_1e_6_count',
    'admission_active_eps_1e_5_frac',
    'admission_active_eps_1e_5_count',
    'admission_active_eps_1e_4_frac',
    'admission_active_eps_1e_4_count',
    'admission_active_eps_1e_3_frac',
    'admission_active_eps_1e_3_count',
    'admission_active_eps_1e_2_frac',
    'admission_active_eps_1e_2_count',
    'admission_active_eps_1e_1_frac',
    'admission_active_eps_1e_1_count',
    'active_eps_1e_6_frac', 'active_eps_1e_6_count',
    'active_eps_1e_5_frac', 'active_eps_1e_5_count',
    'active_eps_1e_4_frac', 'active_eps_1e_4_count',
    'active_eps_1e_3_frac', 'active_eps_1e_3_count',
    'active_eps_1e_2_frac', 'active_eps_1e_2_count',
    'active_eps_1e_1_frac', 'active_eps_1e_1_count',
    'mass_eps_1e_6', 'mass_eps_1e_5', 'mass_eps_1e_4',
    'mass_eps_1e_3', 'mass_eps_1e_2', 'mass_eps_1e_1',
    'projected_Tfinal_active_eps_1e_6_frac',
    'projected_Tfinal_active_eps_1e_6_count',
    'projected_Tfinal_active_eps_1e_4_frac',
    'projected_Tfinal_active_eps_1e_4_count',
    'projected_Tfinal_active_eps_1e_3_frac',
    'projected_Tfinal_active_eps_1e_3_count',
    'projected_Tfinal_mass_eps_1e_6',
    'projected_Tfinal_mass_eps_1e_4',
    'projected_Tfinal_mass_eps_1e_3',
    'margin_band_gt_0',
    'margin_band_m0_01_0',
    'margin_band_m0_03_m0_01',
    'margin_band_m0_10_m0_03',
    'margin_band_lt_m0_10',
    'margin_band_pos',
    'margin_band_near_m0_03_0',
    'margin_band_far_lt_m0_10',
)
DIRECT_TAU_ATTN_SPLIT_METRIC_NAMES = (
    'raw_gate_max', 'gate_sum', 'active_n_mean',
    'tau_abs_mean', 'dead_penalty', 'dead_count',
    'int_max', 'drive_mean', 'gate_den_sum_mean', 'gate_eff_n',
    'gate_eff_ratio', 'top1_gate_frac', 'top1_gate_frac_max',
    'score_std',
)

PAGE_METRIC_NAMES = (
    'pages_enabled',
    'page_size',
    'page_capacity',
    'page_count_total',
    'page_count_effective',
    'page_top1_frac',
    'page_entropy',
    'page_score_max',
    'page_score_mean',
    'page_score_std',
    'candidate_ops',
    'candidate_valid_ops',
    'candidate_frac',
    'candidate_valid_frac',
    'candidate_den_mean',
    'candidate_execution_mass',
    'estimated_compute_frac_page',
    'page_fallback_used_frac',
    'page_random_used_frac',
    'page_no_route_frac',
)

V4164_SCALAR_METRIC_NAMES = (
    'admission_den_sum',
    'attn_admission_den_sum',
    'attn_qk_admission_den_sum',
    'attn_v_admission_den_sum',
    'rst_admission_den_sum',
    'execution_mass_sum',
    'attn_execution_mass_sum',
    'attn_qk_execution_mass_sum',
    'attn_v_execution_mass_sum',
    'rst_execution_mass_sum',
    'drive_mean',
    'attn_drive_mean',
    'attn_qk_drive_mean',
    'attn_v_drive_mean',
    'rst_drive_mean',
    'drive_max',
    'attn_drive_max',
    'attn_qk_drive_max',
    'attn_v_drive_max',
    'rst_drive_max',
    'execution_eff_n',
    'attn_execution_eff_n',
    'attn_qk_execution_eff_n',
    'attn_v_execution_eff_n',
    'rst_execution_eff_n',
    'execution_top1_frac',
    'execution_top1_frac_max',
    'attn_execution_top1_frac',
    'attn_execution_top1_frac_max',
    'attn_qk_execution_top1_frac',
    'attn_qk_execution_top1_frac_max',
    'attn_v_execution_top1_frac',
    'attn_v_execution_top1_frac_max',
    'rst_execution_top1_frac',
    'rst_execution_top1_frac_max',
)

UPDATE_CAP_GROUP_SPECS = (
    ('proj_attn', 'pA', 'update_cap_proj_attn_hit',
     'update_cap_proj_attn_ratio_pre', 'update_cap_proj_attn_scale',
     'ratio_pre'),
    ('proj_rst', 'pR', 'update_cap_proj_rst_hit',
     'update_cap_proj_rst_ratio_pre', 'update_cap_proj_rst_scale',
     'ratio_pre'),
    ('op_key_qk', 'oQ', 'update_cap_op_key_qk_hit',
     'update_cap_op_key_qk_ratio_pre', 'update_cap_op_key_qk_scale',
     'ratio_pre'),
    ('op_key_v', 'oV', 'update_cap_op_key_v_hit',
     'update_cap_op_key_v_ratio_pre', 'update_cap_op_key_v_scale',
     'ratio_pre'),
    ('op_key_rst', 'oR', 'update_cap_op_key_rst_hit',
     'update_cap_op_key_rst_ratio_pre', 'update_cap_op_key_rst_scale',
     'ratio_pre'),
    ('tau_attn', 'tA', 'update_cap_tau_attn_hit',
     'update_cap_tau_attn_abs_pre', 'update_cap_tau_attn_scale',
     'abs_pre'),
    ('tau_rst', 'tR', 'update_cap_tau_rst_hit',
     'update_cap_tau_rst_abs_pre', 'update_cap_tau_rst_scale',
     'abs_pre'),
    ('raw_tau_qk', 'rQ', 'update_cap_raw_tau_qk_hit',
     'update_cap_raw_tau_qk_abs_pre', 'update_cap_raw_tau_qk_scale',
     'abs_pre'),
    ('raw_tau_v', 'rV', 'update_cap_raw_tau_v_hit',
     'update_cap_raw_tau_v_abs_pre', 'update_cap_raw_tau_v_scale',
     'abs_pre'),
    ('raw_tau_rst', 'rR', 'update_cap_raw_tau_rst_hit',
     'update_cap_raw_tau_rst_abs_pre', 'update_cap_raw_tau_rst_scale',
     'abs_pre'),
    ('scan_attn', 'sA', 'update_cap_scan_attn_hit',
     'update_cap_scan_attn_abs_pre', 'update_cap_scan_attn_scale',
     'abs_pre'),
    ('scan_rst', 'sR', 'update_cap_scan_rst_hit',
     'update_cap_scan_rst_abs_pre', 'update_cap_scan_rst_scale',
     'abs_pre'),
)


def _cap_window_hit_key(group_name):
    return f'update_cap_window_{group_name}_hit_count'


def _cap_window_pre_max_key(group_name, pre_kind):
    return f'update_cap_window_{group_name}_{pre_kind}_max'


def _cap_window_scale_min_key(group_name):
    return f'update_cap_window_{group_name}_scale_min'


def _init_update_cap_window_stats():
    stats = {'update_cap_window_steps': jnp.int32(0)}
    for group_name, _, _, _, _, pre_kind in UPDATE_CAP_GROUP_SPECS:
        stats[_cap_window_hit_key(group_name)] = jnp.float32(0.0)
        stats[_cap_window_pre_max_key(group_name, pre_kind)] = jnp.float32(0.0)
        stats[_cap_window_scale_min_key(group_name)] = jnp.float32(1.0)
    return stats


def _accumulate_update_cap_window_stats(window_stats, metrics):
    out = {'update_cap_window_steps':
           window_stats['update_cap_window_steps'] + jnp.int32(1)}
    for (group_name, _, hit_key, pre_key, scale_key,
         pre_kind) in UPDATE_CAP_GROUP_SPECS:
        win_hit_key = _cap_window_hit_key(group_name)
        win_pre_key = _cap_window_pre_max_key(group_name, pre_kind)
        win_scale_key = _cap_window_scale_min_key(group_name)
        out[win_hit_key] = (
            window_stats[win_hit_key]
            + metrics.get(hit_key, jnp.float32(0.0)))
        out[win_pre_key] = jnp.maximum(
            window_stats[win_pre_key],
            metrics.get(pre_key, jnp.float32(0.0)))
        out[win_scale_key] = jnp.minimum(
            window_stats[win_scale_key],
            metrics.get(scale_key, jnp.float32(1.0)))
    return out


def _attach_update_cap_window_stats(rec, window_stats):
    if not window_stats:
        return rec
    rec['update_cap_window_steps'] = int(
        window_stats.get('update_cap_window_steps', 0))
    for group_name, _, _, _, _, pre_kind in UPDATE_CAP_GROUP_SPECS:
        hit_key = _cap_window_hit_key(group_name)
        pre_key = _cap_window_pre_max_key(group_name, pre_kind)
        scale_key = _cap_window_scale_min_key(group_name)
        rec[hit_key] = float(window_stats.get(hit_key, 0.0))
        rec[pre_key] = float(window_stats.get(pre_key, 0.0))
        rec[scale_key] = float(window_stats.get(scale_key, 1.0))
    return rec


def _format_update_cap_window_line(rec, indent="  ", is_v4164=False):
    def _g(key, default=0.0):
        val = rec.get(key, default)
        if val is None:
            return float(default)
        return float(val)

    steps = int(_g('update_cap_window_steps', 0.0))
    if steps <= 0:
        return None
    hit_parts = []
    pre_parts = []
    scale_parts = []
    hit_total = 0.0
    for group_name, label, _, _, _, pre_kind in UPDATE_CAP_GROUP_SPECS:
        if is_v4164 and group_name in ('tau_attn', 'tau_rst'):
            continue
        if (not is_v4164) and group_name.startswith('raw_tau_'):
            continue
        if (is_v4164 and group_name.startswith('raw_tau_')
                and _g('update_cap_raw_tau_enabled') <= 0.0):
            continue
        hit_parts.append(
            f"{label}={_g(_cap_window_hit_key(group_name)):.0f}")
        hit_total += _g(_cap_window_hit_key(group_name))
        pre_parts.append(
            f"{label}={_g(_cap_window_pre_max_key(group_name, pre_kind)):.1e}")
        scale_parts.append(
            (label, _g(_cap_window_scale_min_key(group_name), 1.0)))
    if hit_total <= 0.0:
        return None
    min_scale_label, min_scale = min(scale_parts, key=lambda item: item[1])
    return (
        f"{indent}update_cap_window: steps={steps} "
        f"hit[{' '.join(hit_parts)}] "
        f"min_scale={min_scale:.3f}@{min_scale_label} "
        f"max_pre[{' '.join(pre_parts)}]"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def print_xla_oom_diagnostics():
    """Print the newest XLA dump files likely to contain HBM allocation info."""
    dump_dir = Path(os.environ.get("XLA_DUMP_DIR", "/tmp/xla_dump_train"))
    if not dump_dir.exists():
        print(f"  XLA dump dir not found: {dump_dir}", flush=True)
        print("  Set XLA_DUMP_DIR and XLA_FLAGS=--xla_dump_to=$XLA_DUMP_DIR "
              "--xla_dump_hlo_as_text before launching Python.", flush=True)
        return

    patterns = (
        "*memory*",
        "*buffer*",
        "*after_optimizations.txt",
        "*.txt",
    )
    files = []
    seen = set()
    for pat in patterns:
        for path in dump_dir.rglob(pat):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[:12]

    if not files:
        print(f"  No XLA text dumps found under {dump_dir}", flush=True)
        return

    print(f"\n  === XLA OOM diagnostics ===", flush=True)
    print(f"  Dump dir: {dump_dir}", flush=True)
    print("  Newest relevant dump files:", flush=True)
    for path in files[:5]:
        try:
            size_mb = path.stat().st_size / 1e6
            print(f"    {path} ({size_mb:.1f} MB)", flush=True)
        except OSError:
            print(f"    {path}", flush=True)

    needles = (
        "Total hbm usage",
        "Program hbm requirement",
        "Largest program allocations",
        "Allocation type: HLO temp",
        "Size:",
        "source_file=",
        "Shape:",
    )
    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if not any(n in text for n in needles[:3]):
            continue
        print(f"\n  --- XLA memory excerpt: {path} ---", flush=True)
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines)
                if any(n in line for n in needles[:3])]
        start = max(0, hits[0] - 2) if hits else 0
        end = min(len(lines), start + 90)
        printed = 0
        for line in lines[start:end]:
            if any(n in line for n in needles) or "Operator:" in line:
                print(f"  {line[:240]}", flush=True)
                printed += 1
                if printed >= 36:
                    print("  ... excerpt truncated; inspect dump file above for full report.",
                          flush=True)
                    break
        return

    print("  No memory report excerpt found yet. Inspect latest dumps above.",
          flush=True)


# ============================================================
# Config
# ============================================================

def load_config(config_path):
    """Load config from local or GCS path."""
    path_str = str(config_path)
    if path_str.startswith("gs://"):
        with _open_file(path_str, "r") as f:
            return yaml.safe_load(f)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _json_safe(obj):
    """Convert config snapshots to JSON/msgpack-friendly Python values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return str(obj)
    if hasattr(obj, 'tolist'):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, 'item'):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        try:
            return _json_safe(np.asarray(obj).tolist())
        except Exception:
            pass
    return str(obj)


def _stable_json_dumps(obj):
    return json.dumps(
        _json_safe(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _config_sha256(obj):
    payload = _stable_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_config_snapshot(obj):
    return _json_safe(deepcopy(obj))


def _materialized_config_snapshot(cfg, training_config):
    full_cfg = deepcopy(cfg)
    merged_training = deepcopy(cfg.get('training', {}))
    if training_config:
        merged_training.update(deepcopy(training_config))
    full_cfg['training'] = merged_training
    return _json_safe(full_cfg)


ACTIVE_SRW_RESUME_REQUIRED_FIELDS = (
    ('model', 'model_version'),
    ('model', 'd_model'),
    ('model', 'n_layers'),
    ('model', 'n_heads'),
    ('model', 'max_seq_len'),
    ('model', 'd_route'),
    ('model', 'n_qk'),
    ('model', 'n_v'),
    ('model', 'n_rst'),
    ('data', 'max_train_tokens'),
    ('training', 'batch_size'),
    ('training', 'num_epochs'),
    ('training', 'mesh_model'),
    ('training', 'mesh_data'),
    ('training', 'gradient_accumulation_steps'),
    ('training', 'lr'),
    ('training', 'warmup_ratio'),
    ('training', 'weight_decay'),
    ('training', 'checkpoint_interval'),
    ('training', 'val_interval'),
    ('training', 'log_interval'),
    ('training', 'log_analysis_multiplier'),
    ('training', 'heavy_geometry_multiplier'),
    ('training', 'n_chunks_qk'),
    ('training', 'n_chunks_v'),
    ('training', 'n_chunks_rst'),
    ('training', 'admission_den_power'),
    ('training', 'admission_den_grad_scale'),
    ('training', 'soft_gate_effective_active_eps'),
)

V4168_OPSPACE_RESUME_REQUIRED_FIELDS = (
    ('model', 'model_version'),
    ('model', 'd_model'),
    ('model', 'n_layers'),
    ('model', 'n_heads'),
    ('model', 'max_seq_len'),
    ('model', 'd_route'),
    ('model', 'n_qk'),
    ('model', 'n_v'),
    ('model', 'n_rst'),
    ('data', 'max_train_tokens'),
    ('training', 'batch_size'),
    ('training', 'num_epochs'),
    ('training', 'mesh_model'),
    ('training', 'mesh_data'),
    ('training', 'gradient_accumulation_steps'),
    ('training', 'lr'),
    ('training', 'warmup_ratio'),
    ('training', 'weight_decay'),
    ('training', 'checkpoint_interval'),
    ('training', 'val_interval'),
    ('training', 'log_interval'),
    ('training', 'log_analysis_multiplier'),
    ('training', 'heavy_geometry_multiplier'),
    ('training', 'n_chunks_qk'),
    ('training', 'n_chunks_v'),
    ('training', 'n_chunks_rst'),
    ('training', 'admission_den_power'),
    ('training', 'operation_space', 'enabled'),
    ('training', 'operation_space', 'pools', 'qk', 'execution_backend'),
    ('training', 'operation_space', 'pools', 'qk', 'num_regions'),
    ('training', 'operation_space', 'pools', 'qk', 'blocks_per_region'),
    ('training', 'operation_space', 'pools', 'qk', 'operators_per_block'),
    ('training', 'operation_space', 'pools', 'qk', 'visible_regions'),
    ('training', 'operation_space', 'pools', 'qk',
     'visible_blocks_per_region'),
    ('training', 'operation_space', 'pools', 'v', 'execution_backend'),
    ('training', 'operation_space', 'pools', 'v', 'num_regions'),
    ('training', 'operation_space', 'pools', 'v', 'blocks_per_region'),
    ('training', 'operation_space', 'pools', 'v', 'operators_per_block'),
    ('training', 'operation_space', 'pools', 'v', 'visible_regions'),
    ('training', 'operation_space', 'pools', 'v',
     'visible_blocks_per_region'),
    ('training', 'operation_space', 'pools', 'rst', 'execution_backend'),
)

V4168_OPSPACE_REPACK_RESUME_REQUIRED_FIELDS = (
    ('training', 'operation_space', 'repack', 'enabled'),
    ('training', 'operation_space', 'repack', 'start_step'),
    ('training', 'operation_space', 'repack', 'interval_steps'),
    ('training', 'operation_space', 'repack', 'max_swaps', 'qk'),
    ('training', 'operation_space', 'repack', 'max_swaps', 'v'),
    ('training', 'operation_space', 'repack', 'max_swaps', 'rst'),
)

V4168_OPSPACE_FINAL_RESUME_REQUIRED_FIELDS = (
    ('training', 'operation_space', 'pools', 'rst',
     'high_regret_threshold'),
)

V4168_OPSPACE_REGION_BLOCK_RESUME_REQUIRED_FIELDS = (
    ('training', 'operation_space', 'pools', 'rst', 'num_regions'),
    ('training', 'operation_space', 'pools', 'rst', 'blocks_per_region'),
    ('training', 'operation_space', 'pools', 'rst', 'operators_per_block'),
    ('training', 'operation_space', 'pools', 'rst', 'visible_regions'),
    ('training', 'operation_space', 'pools', 'rst',
     'visible_blocks_per_region'),
    ('training', 'operation_space', 'pools', 'rst',
     'region_score_pooling'),
    ('training', 'operation_space', 'pools', 'rst',
     'region_score_temperature'),
    ('training', 'operation_space', 'pools', 'rst',
     'region_capacity_factor'),
    ('training', 'operation_space', 'pools', 'rst',
     'block_capacity_factor'),
)


def _missing_config_paths(cfg, paths):
    missing = []
    for path in paths:
        cur = cfg
        for key in path:
            if (not isinstance(cur, dict)
                    or key not in cur
                    or cur[key] is None):
                missing.append('.'.join(path))
                break
            cur = cur[key]
    return missing


def _require_resume_full_config(full_config):
    if full_config is None or full_config == {}:
        raise RuntimeError(
            "Resume checkpoint is missing full_config. Automatic config "
            "fallback is disabled. Restart from scratch or use a separate "
            "explicit legacy/eval-only loader."
        )
    if not isinstance(full_config, dict):
        raise RuntimeError(
            "Resume checkpoint full_config is invalid. Expected a config "
            "dict; restart from scratch."
        )

    missing_top = [
        key for key in ('model', 'training') if key not in full_config
    ]
    if missing_top:
        raise RuntimeError(
            "Resume checkpoint full_config is incomplete: missing "
            + "/".join(missing_top)
            + ". This checkpoint cannot be resumed deterministically."
        )
    if not isinstance(full_config['model'], dict):
        raise RuntimeError(
            "Resume checkpoint full_config.model is invalid. Cannot resume "
            "deterministically."
        )
    if not isinstance(full_config['training'], dict):
        raise RuntimeError(
            "Resume checkpoint full_config.training is invalid. Cannot resume "
            "deterministically."
        )


def _require_resume_materialized_fields(full_config):
    model_version_missing = _missing_config_paths(
        full_config, (('model', 'model_version'),))
    if model_version_missing:
        raise RuntimeError(
            "Resume checkpoint full_config is missing required materialized "
            "fields. Automatic config fallback is disabled. Missing keys: "
            + ", ".join(model_version_missing)
        )
    model_version = full_config['model']['model_version']
    if not _is_active_srw_version(model_version):
        return
    training_cfg = (
        full_config.get('training', {})
        if isinstance(full_config.get('training', {}), dict) else {})
    opspace_cfg = (
        training_cfg.get('operation_space', {})
        if isinstance(training_cfg.get('operation_space', {}), dict) else {})
    if (str(model_version) == V4168_MODEL_VERSION
            and bool(opspace_cfg.get('enabled', False))):
        required_fields = list(V4168_OPSPACE_RESUME_REQUIRED_FIELDS)
        pools_cfg = (
            opspace_cfg.get('pools', {})
            if isinstance(opspace_cfg.get('pools', {}), dict) else {})
        rst_cfg = (
            pools_cfg.get('rst', {})
            if isinstance(pools_cfg.get('rst', {}), dict) else {})
        rst_execution_backend = str(rst_cfg.get(
            'execution_backend', '')).lower()
        if rst_execution_backend == 'sparse_region_block':
            required_fields.extend(V4168_OPSPACE_FINAL_RESUME_REQUIRED_FIELDS)
            required_fields.extend(
                V4168_OPSPACE_REGION_BLOCK_RESUME_REQUIRED_FIELDS)
        repack_cfg = (
            opspace_cfg.get('repack', {})
            if isinstance(opspace_cfg.get('repack', {}), dict) else {})
        if bool(repack_cfg.get('enabled', False)):
            required_fields.extend(V4168_OPSPACE_REPACK_RESUME_REQUIRED_FIELDS)
        missing = _missing_config_paths(full_config, tuple(required_fields))
        if missing:
            raise RuntimeError(
                "Resume checkpoint full_config is missing required "
                "materialized v4168 operation-space fields. Automatic "
                "config fallback is disabled. Missing keys: "
                + ", ".join(missing)
            )
        return

    missing = _missing_config_paths(
        full_config, ACTIVE_SRW_RESUME_REQUIRED_FIELDS)
    if missing:
        raise RuntimeError(
            "Resume checkpoint full_config is missing required materialized "
            "active SRW fields. Automatic config fallback is disabled. "
            "Missing keys: " + ", ".join(missing)
        )


def _write_json_file(path, obj):
    with _open_file(path, 'w') as f:
        f.write(json.dumps(_json_safe(obj), indent=2, default=str))


def _read_json_file(path):
    with _open_file(path, 'r') as f:
        return json.loads(f.read())


# ============================================================
# Model Registry
# ============================================================

def _v4164_base_kwargs(cfg):
    """Init kwargs shared by active DAWN variants."""
    m = cfg['model']
    t = cfg['training']
    return dict(
        vocab_size=m.get('vocab_size', 30522),
        d_model=m.get('d_model', 384),
        n_layers=m.get('n_layers', 12),
        n_heads=m.get('n_heads', 6),
        max_seq_len=m.get('max_seq_len', 512),
        d_route=m.get('d_route', m.get('d_bottleneck', 128)),
        n_qk=m.get('n_qk', 1580),
        n_v=m.get('n_v', 2600),
        n_know=m.get('n_know', 25200),
        dropout_rate=m.get('dropout', 0.1),
        router_dropout=m.get('router_dropout', 0.1),
        gradient_checkpointing=m.get('gradient_checkpointing', False),
        n_chunks_know=t.get('n_chunks_know', 1),
        n_chunks_qk=t.get('n_chunks_qk', 1),
        n_chunks_v=t.get('n_chunks_v', 1),
    )


def _baseline_kwargs(cfg):
    m = cfg["model"]
    return dict(
        vocab_size=m.get("vocab_size", 30522),
        logical_vocab_size=m.get("logical_vocab_size", None),
        vocab_size_padded=m.get("vocab_size_padded", None),
        d_model=m.get("d_model", 384),
        d_ff=m.get("d_ff", 1536),
        n_layers=m.get("n_layers", 12),
        n_heads=m.get("n_heads", 6),
        max_seq_len=m.get("max_seq_len", 512),
        dropout_rate=m.get("dropout", 0.1),
        gradient_checkpointing=m.get("gradient_checkpointing", False),
    )


def _maybe_materialize_vocab_parallel_config(cfg):
    model_version = str(cfg["model"].get("model_version", ""))
    mesh_model = int(cfg.get("training", {}).get("mesh_model", 1))
    if mesh_model <= 1:
        return
    if model_version not in (
            V4167_MODEL_VERSION,
            BASELINE_MODEL_VERSION,
            LEGACY_BASELINE_MODEL_VERSION):
        return
    logical_vocab_size = int(cfg["model"]["vocab_size"])
    from models.vocab_parallel import padded_vocab_size
    padded = padded_vocab_size(logical_vocab_size, mesh_model)
    cfg["model"]["logical_vocab_size"] = logical_vocab_size
    cfg["model"]["vocab_size_padded"] = padded


def _v4164_model_base_kwargs(cfg):
    """v4164 d_route-only routing kwargs."""
    kw = _v4164_base_kwargs(cfg)
    m = cfg['model']
    d_route = m.get('d_route')
    if d_route is None:
        old_total = (
            m.get('tag' + '_dim', 0)
            + m.get('read_' + 's' + 'ig_dim', 0)
            + m.get('write_' + 's' + 'ig_dim', 0)
        )
        d_route = old_total or m.get('d_bottleneck', 128)
    kw['d_route'] = d_route
    return kw


def _v4164_tau_init_config(cfg):
    """Parse and validate v4164 explicit or quantile tau init."""
    model_cfg = cfg['model']
    training_cfg = cfg['training']

    def _cfg_get(name, default=None):
        if name in model_cfg:
            return model_cfg[name]
        if name in training_cfg:
            return training_cfg[name]
        return default

    mode = str(_cfg_get('tau_init_mode', 'explicit')).strip().lower()
    if mode not in ('explicit', 'quantile_frac'):
        raise ValueError(
            "v4164 tau_init_mode must be 'explicit' or 'quantile_frac', "
            f"got {mode!r}.")

    parsed = {'mode': mode}
    if mode == 'explicit':
        explicit = {
            'qk': _cfg_get('tau_init_attn_qk', None),
            'v': _cfg_get('tau_init_attn_v', None),
            'rst': _cfg_get('tau_init_rst', None),
        }
        if any(value is None for value in explicit.values()):
            raise ValueError(
                "v4164 requires explicit cosine-space "
                "tau_init_attn_qk/v/rst.")
        parsed['explicit'] = explicit
        return parsed

    targets = {}
    for pool in ('qk', 'v', 'rst'):
        name = f'tau_init_target_{pool}_frac'
        value = _cfg_get(name, None)
        if value is None:
            raise ValueError(
                f"v4164 tau_init_mode=quantile_frac "
                f"requires {name}.")
        value = float(value)
        if not np.isfinite(value) or not (0.0 < value < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {value}.")
        targets[pool] = value

    tau_min = float(_cfg_get('tau_init_min', -1.0))
    tau_max = float(_cfg_get('tau_init_max', 1.0))
    if (not np.isfinite(tau_min) or not np.isfinite(tau_max)
            or tau_min < -1.0 or tau_max > 1.0 or tau_min > tau_max):
        raise ValueError(
            "tau_init_min/max must be finite cosine values satisfying "
            f"-1 <= min <= max <= 1, got {tau_min}/{tau_max}.")
    calibration_tokens = int(_cfg_get('tau_init_calibration_tokens', 128))
    if calibration_tokens <= 0:
        raise ValueError(
            "tau_init_calibration_tokens must be > 0, got "
            f"{calibration_tokens}.")
    parsed.update({
        'targets': targets,
        'tau_min': tau_min,
        'tau_max': tau_max,
        'calibration_tokens': calibration_tokens,
    })
    return parsed


def _cfg_bool(value, *, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ('1', 'true', 'yes', 'on'):
            return True
        if norm in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return bool(value)


def _v4168_hardware_repack_config(training_cfg, model_version):
    """Parse hardware-sector repack/execution config with disabled defaults."""
    is_v4168 = str(model_version) == V4168_MODEL_VERSION
    enabled = (
        _cfg_bool(training_cfg.get('hardware_repack_enabled', False),
                  name='training.hardware_repack_enabled')
        if is_v4168 else False)
    sector_execution_enabled = (
        _cfg_bool(
            training_cfg.get(
                'hardware_sector_execution_enabled', enabled),
            name='training.hardware_sector_execution_enabled')
        if is_v4168 else False)
    interval_steps = int(training_cfg.get(
        'hardware_repack_interval_steps', 100))
    strategy = str(training_cfg.get(
        'hardware_repack_strategy', 'balanced_vq')).lower()
    farthest_per_sector = int(training_cfg.get(
        'hardware_repack_farthest_per_sector', 10))
    gain_eps = float(training_cfg.get('hardware_repack_gain_eps', 1.0e-3))
    max_move_frac = float(training_cfg.get(
        'hardware_repack_max_move_frac', 0.08))
    vq_iterations = int(training_cfg.get(
        'hardware_repack_vq_iterations', 4))
    warmup_steps = int(training_cfg.get('hardware_repack_warmup_steps', 0))
    freeze_after_step = training_cfg.get(
        'hardware_repack_freeze_after_step', None)
    if freeze_after_step is not None:
        freeze_after_step = int(freeze_after_step)

    if enabled and not is_v4168:
        raise ValueError(
            "training.hardware_repack_enabled is only supported for "
            f"{V4168_MODEL_VERSION}.")
    if sector_execution_enabled and not is_v4168:
        raise ValueError(
            "training.hardware_sector_execution_enabled is only supported for "
            f"{V4168_MODEL_VERSION}.")
    if interval_steps <= 0:
        raise ValueError(
            "training.hardware_repack_interval_steps must be > 0, got "
            f"{interval_steps}.")
    if farthest_per_sector < 0:
        raise ValueError(
            "training.hardware_repack_farthest_per_sector must be >= 0, got "
            f"{farthest_per_sector}.")
    if strategy not in ('balanced_vq', 'sector_swap', 'legacy_swap', 'legacy'):
        raise ValueError(
            "training.hardware_repack_strategy must be 'balanced_vq' or "
            f"'sector_swap', got {strategy!r}.")
    if gain_eps < 0.0:
        raise ValueError(
            "training.hardware_repack_gain_eps must be >= 0, got "
            f"{gain_eps}.")
    if not (0.0 <= max_move_frac <= 1.0):
        raise ValueError(
            "training.hardware_repack_max_move_frac must be in [0, 1], got "
            f"{max_move_frac}.")
    if vq_iterations <= 0:
        raise ValueError(
            "training.hardware_repack_vq_iterations must be > 0, got "
            f"{vq_iterations}.")
    if warmup_steps < 0:
        raise ValueError(
            "training.hardware_repack_warmup_steps must be >= 0, got "
            f"{warmup_steps}.")
    if freeze_after_step is not None and freeze_after_step < 0:
        raise ValueError(
            "training.hardware_repack_freeze_after_step must be null or >= 0, "
            f"got {freeze_after_step}.")

    return {
        'hardware_repack_enabled': bool(enabled),
        'hardware_sector_execution_enabled': bool(sector_execution_enabled),
        'hardware_repack_interval_steps': interval_steps,
        'hardware_repack_strategy': strategy,
        'hardware_repack_farthest_per_sector': farthest_per_sector,
        'hardware_repack_gain_eps': gain_eps,
        'hardware_repack_max_move_frac': max_move_frac,
        'hardware_repack_vq_iterations': vq_iterations,
        'hardware_repack_warmup_steps': warmup_steps,
        'hardware_repack_freeze_after_step': freeze_after_step,
    }


def _v4168_should_hardware_repack(step, repack_cfg):
    if not repack_cfg.get('hardware_repack_enabled', False):
        return False
    step = int(step)
    if step < int(repack_cfg.get('hardware_repack_warmup_steps', 0)):
        return False
    freeze_after = repack_cfg.get('hardware_repack_freeze_after_step', None)
    if freeze_after is not None and step > int(freeze_after):
        return False
    interval = int(repack_cfg.get('hardware_repack_interval_steps', 100))
    return step > 0 and (step % interval == 0)


def _ceil_to_multiple(value, multiple):
    value = int(value)
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def _v4168_operation_space_device_count(training_cfg):
    mesh_model = max(1, int(training_cfg.get('mesh_model', 1)))
    return mesh_model


def _v4168_operation_space_cfg(training_cfg):
    opspace = training_cfg.get('operation_space', {})
    if opspace is None:
        return {}
    if not isinstance(opspace, dict):
        raise ValueError("training.operation_space must be a mapping.")
    return opspace


def _migrate_v4168_operation_space_full_config_for_clean_schema(full_config):
    """Resume-only cleanup for removed v4168 operation-space config fields."""
    if not isinstance(full_config, dict):
        return full_config
    normalized = deepcopy(full_config)
    model_cfg = normalized.get('model', {})
    if not isinstance(model_cfg, dict):
        return normalized
    if str(model_cfg.get('model_version', '')) != V4168_MODEL_VERSION:
        return normalized
    training_cfg = normalized.get('training', {})
    if not isinstance(training_cfg, dict):
        return normalized
    opspace = training_cfg.get('operation_space', {})
    if not isinstance(opspace, dict):
        return normalized
    pools = opspace.get('pools', {})
    if not isinstance(pools, dict):
        return normalized

    removed_pool_keys = (
        'routing_mode',
        'execution_mode',
        'execution_backend_mode',
        'lanes',
        'local_lanes',
        'tiles_per_lane',
        'tile_size',
        'k_exec',
        'exec_tiles_per_block',
        'blocks_per_lane',
        'block_size',
        'local_padded_ops',
        'local_operator_slots',
        'exec_slots',
        'region_block_pallas',
        'block_bucketed_dense',
        'block_bucketed_pallas_final',
        'output_tiled',
        'bucket_chunk_size',
        'assignment_policy',
    )
    for label in ('qk', 'v', 'rst'):
        pool = pools.get(label, {})
        if not isinstance(pool, dict):
            continue
        for key in removed_pool_keys:
            pool.pop(key, None)

    opspace.pop('tile_size', None)
    return normalized


def _v4168_validate_operation_space_shape(opspace):
    if not opspace:
        return
    removed_top_keys = {'version', 'tile_size'}
    present_removed_top = sorted(set(opspace) & removed_top_keys)
    if present_removed_top:
        raise ValueError(
            "training.operation_space."
            + ", training.operation_space.".join(present_removed_top)
            + " was removed from the clean Region-Block atlas config.")
    allowed = {'enabled', 'pools', 'repack'}
    extra = sorted(set(opspace) - allowed)
    if extra:
        raise ValueError(
            "training.operation_space only supports enabled, pools, and "
            f"repack; remove: {', '.join(extra)}")
    pools = opspace.get('pools', {})
    if pools is None:
        pools = {}
    if not isinstance(pools, dict):
        raise ValueError("training.operation_space.pools must be a mapping.")
    extra_pools = sorted(set(pools) - {'qk', 'v', 'rst'})
    if extra_pools:
        raise ValueError(
            "training.operation_space.pools only supports qk, v, rst; "
            f"remove: {', '.join(extra_pools)}")
    removed_pool_keys = {
        'mode',
        'routing_mode',
        'execution_mode',
        'execution_backend_mode',
        'lanes',
        'local_lanes',
        'k_exec',
        'tiles_per_lane',
        'tile_size',
        'exec_tiles_per_block',
        'blocks_per_lane',
        'block_size',
        'local_padded_ops',
        'local_operator_slots',
        'exec_slots',
        'region_block_pallas',
        'block_bucketed_dense',
        'block_bucketed_pallas_final',
        'output_tiled',
        'bucket_chunk_size',
        'assignment_policy',
        'factorized_lane_mean',
        'lane_output_' + 'mode',
        'visible_regions_' + 'start',
        'visible_regions_' + 'mid',
        'visible_regions_' + 'final',
        'candidate_region_' + 'count',
        'block_mixing_' + 'enabled',
    }
    common_pool_keys = {
        'execution_backend',
        'num_regions',
        'blocks_per_region',
        'operators_per_block',
        'visible_regions',
        'visible_blocks_per_region',
    }
    allowed_rst_keys = {
        *common_pool_keys,
        'region_score_pooling',
        'region_score_temperature',
        'region_capacity_factor',
        'block_capacity_factor',
        'bucket_capacity_factor',
        'high_regret_threshold',
    }
    for label in ('qk', 'v', 'rst'):
        pool = pools.get(label, {})
        if pool is None:
            pool = {}
        if not isinstance(pool, dict):
            raise ValueError(
                f"training.operation_space.pools.{label} must be a mapping.")
        present_removed = sorted(set(pool) & removed_pool_keys)
        if present_removed:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}."
                + (f", training.operation_space.pools.{label}."
                   ).join(present_removed)
                + " was removed from the clean Region-Block atlas config.")
        allowed_pool_keys = (
            allowed_rst_keys if label == 'rst' else common_pool_keys)
        extra = sorted(set(pool) - allowed_pool_keys)
        if extra:
            raise ValueError(
                f"training.operation_space.pools.{label} only supports "
                "the clean operation-space fields for that pool; remove: "
                f"{', '.join(extra)}")
    repack = opspace.get('repack', {})
    if repack is None:
        repack = {}
    if not isinstance(repack, dict):
        raise ValueError("training.operation_space.repack must be a mapping.")
    allowed_repack = {'enabled', 'start_step', 'interval_steps', 'max_swaps'}
    extra = sorted(set(repack) - allowed_repack)
    if extra:
        raise ValueError(
            "training.operation_space.repack only supports enabled, "
            f"start_step, interval_steps, and max_swaps; remove: "
            f"{', '.join(extra)}")
    max_swaps = repack.get('max_swaps', {})
    if max_swaps is None:
        max_swaps = {}
    if not isinstance(max_swaps, dict):
        raise ValueError(
            "training.operation_space.repack.max_swaps must be a mapping.")
    extra = sorted(set(max_swaps) - {'qk', 'v', 'rst'})
    if extra:
        raise ValueError(
            "training.operation_space.repack.max_swaps only supports qk, v, "
            "and rst; "
            f"remove: {', '.join(extra)}")


def _v4168_operation_space_pool_layouts(training_cfg, model_cfg):
    opspace = _v4168_operation_space_cfg(training_cfg)
    _v4168_validate_operation_space_shape(opspace)
    pools = opspace.get('pools', {})
    if not isinstance(pools, dict):
        pools = {}
    device_count = _v4168_operation_space_device_count(training_cfg)
    defaults = {
        'qk': {'n_key': 'n_qk', 'execution_backend': 'dense'},
        'v': {'n_key': 'n_v', 'execution_backend': 'dense'},
        'rst': {'n_key': 'n_rst', 'execution_backend': 'sparse_region_block'},
    }
    layouts = {}
    for label, defaults_i in defaults.items():
        pool = pools.get(label, {})
        if not isinstance(pool, dict) or not pool:
            raise ValueError(
                f"training.operation_space.pools.{label} is required when "
                "operation_space.enabled=true.")
        execution_backend = str(pool.get(
            'execution_backend',
            defaults_i['execution_backend'])).strip().lower()
        expected_backend = defaults_i['execution_backend']
        if execution_backend != expected_backend:
            raise ValueError(
                f"training.operation_space.pools.{label}.execution_backend "
                f"must be {expected_backend!r}, got "
                f"{execution_backend!r}.")
        n_ops = int(model_cfg.get(
            defaults_i['n_key'],
            model_cfg.get('n_know', 0) if label == 'rst' else 0))
        if n_ops <= 0:
            raise ValueError(
                f"model.{defaults_i['n_key']} must be > 0 for operation_space.")
        num_regions = int(pool.get('num_regions'))
        blocks_per_region = int(pool.get('blocks_per_region'))
        operators_per_block = int(pool.get('operators_per_block'))
        visible_regions = int(pool.get('visible_regions'))
        visible_blocks_per_region = int(pool.get(
            'visible_blocks_per_region'))
        for field_name, value in (
                ('num_regions', num_regions),
                ('blocks_per_region', blocks_per_region),
                ('operators_per_block', operators_per_block),
                ('visible_regions', visible_regions),
                ('visible_blocks_per_region', visible_blocks_per_region)):
            if value < 1:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{label}.{field_name} must be >= 1, got {value}.")
        if visible_regions > num_regions:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.visible_regions={visible_regions} must be <= "
                f"num_regions={num_regions}.")
        if visible_blocks_per_region > blocks_per_region:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.visible_blocks_per_region="
                f"{visible_blocks_per_region} must be <= "
                f"blocks_per_region={blocks_per_region}.")
        if num_regions % device_count != 0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.num_regions={num_regions} must be divisible "
                f"by mesh_model={device_count}.")
        regions_per_owner = num_regions // device_count
        total_capacity = (
            num_regions * blocks_per_region * operators_per_block)
        if n_ops > total_capacity:
            raise ValueError(
                "training.operation_space.pools."
                f"{label} capacity {total_capacity} is smaller than "
                f"model.{defaults_i['n_key']}={n_ops}.")
        region_score_pooling = str(pool.get(
            'region_score_pooling', 'smoothmax')).strip().lower()
        if label == 'rst' and region_score_pooling != 'smoothmax':
            raise ValueError(
                "training.operation_space.pools.rst."
                "region_score_pooling must be 'smoothmax'.")
        region_score_temperature = float(pool.get(
            'region_score_temperature', 0.25))
        if region_score_temperature <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.region_score_temperature must be > 0, got "
                f"{region_score_temperature}.")
        region_capacity_factor = float(pool.get(
            'region_capacity_factor', 1.25))
        block_capacity_factor = float(pool.get(
            'block_capacity_factor', 1.25))
        if region_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.region_capacity_factor must be > 0, got "
                f"{region_capacity_factor}.")
        if block_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.block_capacity_factor must be > 0, got "
                f"{block_capacity_factor}.")
        bucket_capacity_factor = float(pool.get(
            'bucket_capacity_factor', block_capacity_factor))
        if bucket_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.bucket_capacity_factor must be > 0, got "
                f"{bucket_capacity_factor}.")
        high_regret_threshold = float(pool.get(
            'high_regret_threshold', 0.05))
        if high_regret_threshold < 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.high_regret_threshold must be >= 0, got "
                f"{high_regret_threshold}.")
        physical_visible_ops_per_token = (
            visible_regions * visible_blocks_per_region
            * operators_per_block)
        layouts[label] = {
            'execution_backend': execution_backend,
            'num_regions': num_regions,
            'regions_per_owner': regions_per_owner,
            'blocks_per_region': blocks_per_region,
            'operators_per_block': operators_per_block,
            'operators_per_region': (
                blocks_per_region * operators_per_block),
            'visible_regions': visible_regions,
            'visible_blocks_per_region': visible_blocks_per_region,
            'visible_ops_per_token': physical_visible_ops_per_token,
            'physical_visible_ops_per_token': (
                physical_visible_ops_per_token),
            'region_score_pooling': region_score_pooling,
            'region_score_temperature': region_score_temperature,
            'region_capacity_factor': region_capacity_factor,
            'block_capacity_factor': block_capacity_factor,
            'bucket_capacity_factor': bucket_capacity_factor,
            'high_regret_threshold': high_regret_threshold,
            'global_operator_capacity': total_capacity,
            'local_operator_capacity': total_capacity // device_count,
            'operator_capacity': total_capacity,
            'invalid_operator_capacity': total_capacity - n_ops,
            'num_devices': device_count,
        }
    return layouts


def _v4168_operation_space_repack_config(training_cfg, model_cfg,
                                         model_version):
    """Parse v4168 tau-free ReLU operation-space repack config."""
    is_v4168 = str(model_version) == V4168_MODEL_VERSION
    opspace = _v4168_operation_space_cfg(training_cfg)
    _v4168_validate_operation_space_shape(opspace)
    operation_space_enabled = bool(opspace.get('enabled', False))
    if operation_space_enabled and not is_v4168:
        raise ValueError(
            "training.operation_space.enabled is only supported for "
            f"{V4168_MODEL_VERSION}.")
    if operation_space_enabled and (
            _cfg_bool(training_cfg.get('hardware_repack_enabled', False),
                      name='training.hardware_repack_enabled')
            or _cfg_bool(
                training_cfg.get('hardware_sector_execution_enabled', False),
                name='training.hardware_sector_execution_enabled')):
        raise ValueError(
            "v4168 operation_space requires training.hardware_repack_enabled "
            "and training.hardware_sector_execution_enabled to remain false.")
    repack = opspace.get('repack', {})
    if not isinstance(repack, dict):
        repack = {}
    max_swaps = repack.get('max_swaps', {})
    if not isinstance(max_swaps, dict):
        max_swaps = {}
    enabled = (
        bool(opspace.get('enabled', False))
        and _cfg_bool(repack.get('enabled', False),
                      name='training.operation_space.repack.enabled')
        and is_v4168)
    start_step = int(repack.get('start_step', 1000))
    interval_steps = int(repack.get('interval_steps', 100))
    pool_layouts = (
        _v4168_operation_space_pool_layouts(training_cfg, model_cfg)
        if is_v4168 else {})

    def _repack_pool(label, max_swaps_default):
        return {
            'enabled': label == 'rst',
            'max_swaps_per_repack': int(max_swaps.get(
                label, max_swaps_default)),
        }

    parsed = {
        'operation_space_enabled': bool(operation_space_enabled and is_v4168),
        'operation_space_repack_enabled': bool(enabled),
        'operation_space_repack_start_step': start_step,
        'operation_space_repack_interval_steps': interval_steps,
        'operation_space_pool_layouts': pool_layouts,
        'operation_space_repack_pools': {
            'rst': _repack_pool('rst', 256),
            'v': _repack_pool('v', 0),
            'qk': _repack_pool('qk', 0),
        },
    }
    if start_step < 0:
        raise ValueError(
            "training.operation_space.repack.start_step must be >= 0, got "
            f"{start_step}.")
    if interval_steps <= 0:
        raise ValueError(
            "training.operation_space.repack.interval_steps must be > 0, got "
            f"{interval_steps}.")
    for label in ('qk', 'v', 'rst'):
        if int(max_swaps.get(label, 256 if label == 'rst' else 0)) < 0:
            raise ValueError(
                f"training.operation_space.repack.max_swaps.{label} must "
                "be >= 0.")
    return parsed


def _v4168_should_operation_space_repack(step, repack_cfg):
    if not repack_cfg.get('operation_space_repack_enabled', False):
        return False
    step = int(step)
    if step < int(repack_cfg.get('operation_space_repack_start_step', 1000)):
        return False
    interval = int(repack_cfg.get(
        'operation_space_repack_interval_steps', 100))
    return step > 0 and (step % interval == 0)


def _materialize_v4167_model_config(cfg):
    m = cfg['model']
    n_layers = int(m.get('n_layers', 12))

    deprecated_keys = (
        'qk_num_stages', 'v_num_stages', 'rst_num_stages',
        'router_num_stages', 'n_qk_shared', 'n_v_shared', 'n_rst_shared')
    present_deprecated = [key for key in deprecated_keys if key in m]
    if present_deprecated:
        raise ValueError(
            "v4167 GSL config no longer accepts shared/stage pool keys: "
            + ", ".join(present_deprecated))

    if 'stage_count' in m:
        raise ValueError(
            "model.stage_count is derived from n_layers // layers_per_stage; "
            "do not set it in config.")
    if 'router_count' in m:
        raise ValueError(
            "model.router_count is derived from router_scope=local and "
            "n_layers; do not set it in config.")

    def _int_field(name, default=None, *, required=False):
        if required and name not in m:
            raise ValueError(f"v4167 GSL requires model.{name}.")
        value = int(m.get(name, default))
        if value < 0:
            raise ValueError(f"model.{name} must be >= 0, got {value}")
        m[name] = value
        return value

    m['fixed_tau'] = _cfg_bool(m.get('fixed_tau', True), name='model.fixed_tau')
    if not m['fixed_tau']:
        raise ValueError("v4167 supports fixed_tau: true only.")

    if 'layers_per_stage' not in m:
        raise ValueError("v4167 GSL requires model.layers_per_stage.")
    layers_per_stage = int(m['layers_per_stage'])
    if layers_per_stage <= 0:
        raise ValueError(
            f"model.layers_per_stage must be > 0, got {layers_per_stage}")
    if n_layers % layers_per_stage != 0:
        raise ValueError(
            f"model.n_layers={n_layers} must be divisible by "
            f"model.layers_per_stage={layers_per_stage}")
    m['layers_per_stage'] = layers_per_stage
    stage_count = n_layers // layers_per_stage

    router_scope = str(m.get('router_scope', 'local')).strip().lower()
    if router_scope != 'local':
        raise ValueError(
            f"v4167 GSL currently requires model.router_scope: local, "
            f"got {m.get('router_scope')!r}.")
    m['router_scope'] = 'local'

    n_qk_global = _int_field('n_qk_global', required=True)
    n_qk_stage = _int_field('n_qk_stage', required=True)
    n_qk_local = _int_field('n_qk_local', required=True)
    n_v_global = _int_field('n_v_global', required=True)
    n_v_stage = _int_field('n_v_stage', required=True)
    n_v_local = _int_field('n_v_local', required=True)
    n_rst_global = _int_field('n_rst_global', required=True)
    n_rst_stage = _int_field('n_rst_stage', required=True)
    n_rst_local = _int_field('n_rst_local', required=True)

    totals = {
        'n_qk': n_qk_global + stage_count * n_qk_stage + n_layers * n_qk_local,
        'n_v': n_v_global + stage_count * n_v_stage + n_layers * n_v_local,
        'n_rst': n_rst_global + stage_count * n_rst_stage + n_layers * n_rst_local,
    }
    for key, total in totals.items():
        if key in m and int(m[key]) != total:
            raise ValueError(
                f"model.{key}={m[key]} must equal global + "
                f"stage_count*stage + n_layers*local for v4167 GSL "
                f"({total}).")
        m[key] = total
    m['n_know'] = m.get('n_know', m['n_rst'])
    if int(m['n_know']) != int(m['n_rst']):
        raise ValueError(
            f"model.n_know={m['n_know']} must match model.n_rst={m['n_rst']} "
            "for v4167.")

    m['qk_visible_n'] = n_qk_global + n_qk_stage + n_qk_local
    m['v_visible_n'] = n_v_global + n_v_stage + n_v_local
    m['rst_visible_n'] = n_rst_global + n_rst_stage + n_rst_local
    return m


def _dawn_srw_kwargs(cfg):
    """Official v4.1.6.4 DAWN-SRW constructor kwargs."""
    version = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)
    if str(version) == V4167_MODEL_VERSION:
        _materialize_v4167_model_config(cfg)
    kw = _v4164_model_base_kwargs(cfg)
    m = cfg['model']
    t = cfg['training']
    if 'n_rst' not in m and 'n_know' not in m:
        raise ValueError("v4164 requires model.n_rst or model.n_know checkpoint alias.")
    kw['n_rst'] = m.get('n_rst', m.get('n_know'))
    kw['n_know'] = m.get('n_know', None)
    kw['n_chunks_rst'] = t.get('n_chunks_rst', t.get('n_chunks_know', 1))
    opspace_cfg = t.get('operation_space', {})
    operation_space_tau_free_enabled = (
        str(version) == V4168_MODEL_VERSION
        and isinstance(opspace_cfg, dict)
        and bool(opspace_cfg.get('enabled', False))
    )
    if operation_space_tau_free_enabled:
        kw['tau_init_attn_qk'] = 0.0
        kw['tau_init_attn_v'] = 0.0
        kw['tau_init_rst'] = 0.0
    else:
        tau_init_cfg = _v4164_tau_init_config(cfg)
        if tau_init_cfg['mode'] == 'explicit':
            kw['tau_init_attn_qk'] = tau_init_cfg['explicit']['qk']
            kw['tau_init_attn_v'] = tau_init_cfg['explicit']['v']
            kw['tau_init_rst'] = tau_init_cfg['explicit']['rst']
        else:
            fixed_tau_values = {
                'tau_init_attn_qk': t.get(
                    'tau_init_attn_qk',
                    m.get('tau_init_attn_qk',
                          t.get('selection_calibration_tau_qk', None))),
                'tau_init_attn_v': t.get(
                    'tau_init_attn_v',
                    m.get('tau_init_attn_v',
                          t.get('selection_calibration_tau_v', None))),
                'tau_init_rst': t.get(
                    'tau_init_rst',
                    m.get('tau_init_rst',
                          t.get('selection_calibration_tau_rst', None))),
            }
            if (_is_fixed_tau_srw_version(version)
                    and all(value is not None
                            for value in fixed_tau_values.values())):
                kw.update(fixed_tau_values)
            else:
                # Fresh quantile starts overwrite these before optimizer init.
                kw['tau_init_attn_qk'] = 0.0
                kw['tau_init_attn_v'] = 0.0
                kw['tau_init_rst'] = 0.0
    if str(version) == V4167_MODEL_VERSION:
        kw.update({
            'fixed_tau': m.get('fixed_tau', True),
            'logical_vocab_size': m.get('logical_vocab_size', None),
            'vocab_size_padded': m.get('vocab_size_padded', None),
            'layers_per_stage': m['layers_per_stage'],
            'router_scope': m['router_scope'],
            'n_qk_global': m['n_qk_global'],
            'n_qk_stage': m['n_qk_stage'],
            'n_qk_local': m['n_qk_local'],
            'n_v_global': m['n_v_global'],
            'n_v_stage': m['n_v_stage'],
            'n_v_local': m['n_v_local'],
            'n_rst_global': m['n_rst_global'],
            'n_rst_stage': m['n_rst_stage'],
            'n_rst_local': m['n_rst_local'],
        })
    if str(version) == V4168_MODEL_VERSION:
        kw.update({
            'qk_block_size': int(m.get('qk_block_size', 256)),
            'v_block_size': int(m.get('v_block_size', 256)),
            'rst_block_size': int(m.get('rst_block_size', 256)),
            'qk_top_blocks': int(m.get('qk_top_blocks', 2)),
            'v_top_blocks': int(m.get('v_top_blocks', 2)),
            'rst_top_blocks': int(m.get('rst_top_blocks', 2)),
            'block_margin': float(m.get('block_margin', 0.0)),
        })
    return kw


def _v4164_sharded_kwargs(cfg):
    """Fixed v4164 sharded SRW execution kwargs."""
    t = cfg['training']
    admission_den_power_cfg = t.get(
        'admission_den_power',
        t.get('v4164_den_power', 1.0))
    admission_den_grad_scale_cfg = t.get(
        'admission_den_grad_scale',
        t.get('v4164_den_grad_scale', 1.0))
    return dict(
        dead_exposure_target=0.0,
        soft_gate_effective_active_eps=float(
            t.get('soft_gate_effective_active_eps', 1.0e-6)),
        admission_den_power=float(admission_den_power_cfg),
        admission_den_grad_scale=float(admission_den_grad_scale_cfg),
    )




def build_model_from_config(cfg):
    """Build an active DAWN-SRW model from config."""
    version = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)
    if _is_baseline_version(version):
        return VanillaTransformer(**_baseline_kwargs(cfg))
    entry = _model_registry_entry(version)
    kwargs = _dawn_srw_kwargs(cfg)
    print(f"route dims: d_route={kwargs['d_route']}")
    return entry['class'](**kwargs)


def _srw_selection_score_setup(params, cfg, max_tokens):
    """Build the minimal score params/kwargs used by SRW score calibration."""
    version = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)
    entry = _model_registry_entry(version)
    score_impl = entry['tau_init_calibration_scores']
    if score_impl is None:
        raise ValueError(
            f"{version} score calibration requires "
            f"{entry['module']} to import cleanly.")
    # Keep the one-time JIT signature small: calibration needs only selection
    # geometry, not tau params or LM output weights.
    if str(version) == V4167_MODEL_VERSION:
        pool_operator_keys = _pool_operator_keys_for_version(version)
        pool_params = {k: v for k, v in params['neuron_pool'].items()}
        pool_params.update(jax.jit(pool_operator_keys)(params['neuron_pool']))
    elif _is_rw_key_srw_version(version):
        pool_operator_keys = _pool_operator_keys_for_version(version)
        pool_params = jax.jit(pool_operator_keys)(params['neuron_pool'])
    else:
        pool_params = {
            'attn_qk_emb': params['neuron_pool']['attn_qk_emb'],
            'attn_v_emb': params['neuron_pool']['attn_v_emb'],
            'rst_emb': params['neuron_pool']['rst_emb'],
        }
    score_params = {
        'token_emb': params['token_emb'],
        'pos_emb': params['pos_emb'],
        'block_0': {
            'norm1': params['block_0']['norm1'],
            'norm2': params['block_0']['norm2'],
        },
        'router': {
            'proj_attn': params['router']['proj_attn'],
            'proj_rst': params['router']['proj_rst'],
        },
        'neuron_pool': pool_params,
    }
    if _is_rw_key_srw_version(version):
        score_params['router'].update({
            'q_op_write_query_proj':
                params['router']['q_op_write_query_proj'],
            'k_op_write_query_proj':
                params['router']['k_op_write_query_proj'],
            'v_op_write_query_proj':
                params['router']['v_op_write_query_proj'],
            'rst_op_write_query_proj':
                params['router']['rst_op_write_query_proj'],
        })
    score_kwargs = {
        'max_tokens': int(max_tokens),
    }
    return version, score_impl, score_params, score_kwargs


def _score_route_values(sampled, route):
    values = np.asarray(sampled[route], dtype=np.float32).reshape(-1)
    mask_key = f'{route}_valid_mask'
    if mask_key in sampled:
        mask = np.asarray(sampled[mask_key], dtype=bool).reshape(-1)
        values = values[mask]
    return values


def _score_route_meta(sampled, route, values):
    def _scalar(name, default):
        key = f'{route}_{name}'
        if key not in sampled:
            return float(default)
        try:
            value = np.asarray(sampled[key]).reshape(-1)[0]
            return float(value)
        except Exception:
            return float(default)

    full_default = 1.0
    if route in sampled and np.asarray(sampled[route]).ndim > 0:
        full_default = float(np.asarray(sampled[route]).shape[-1])
    full_pool_size = _scalar('full_pool_size', full_default)
    candidate_valid_count = _scalar(
        'candidate_valid_count', full_pool_size)
    candidate_count = _scalar('candidate_count', full_pool_size)
    pages_enabled = _scalar('pages_enabled', 0.0) > 0.5
    candidate_frac = (
        candidate_valid_count / max(full_pool_size, 1.0)
        if pages_enabled else 1.0)
    return {
        'pages_enabled': bool(pages_enabled),
        'candidate_valid_count': float(candidate_valid_count),
        'candidate_count': float(candidate_count),
        'full_pool_size': float(full_pool_size),
        'candidate_frac': float(candidate_frac),
        'sample_count': int(values.size),
    }


def _pool_score_page_stats(route_stats):
    q_meta = route_stats['q']
    k_meta = route_stats['k']
    qk_group = {
        'pages_enabled': bool(
            q_meta['pages_enabled'] or k_meta['pages_enabled']),
        'candidate_valid_count': 0.5 * (
            q_meta['candidate_valid_count']
            + k_meta['candidate_valid_count']),
        'candidate_count': 0.5 * (
            q_meta['candidate_count'] + k_meta['candidate_count']),
        'full_pool_size': max(
            q_meta['full_pool_size'], k_meta['full_pool_size']),
    }
    qk_group['candidate_frac'] = (
        qk_group['candidate_valid_count']
        / max(qk_group['full_pool_size'], 1.0)
        if qk_group['pages_enabled'] else 1.0)
    out = {
        'q': q_meta,
        'k': k_meta,
        'qk': qk_group,
        'v': route_stats['v'],
        'rst': route_stats['rst'],
    }
    return out


def _sample_srw_selection_scores(params, input_ids, cfg, max_tokens):
    """Return host-side fresh-init selection score samples by SRW pool."""
    _version, score_impl, score_params, score_kwargs = (
        _srw_selection_score_setup(params, cfg, max_tokens))
    score_fn = jax.jit(partial(score_impl, **score_kwargs))
    score_out = score_fn(score_params, input_ids)
    sampled = jax.device_get(score_out)
    scores = {
        name: _score_route_values(sampled, name)
        for name in ('q', 'k', 'v', 'rst')
    }
    scores['qk'] = np.concatenate((scores['q'], scores['k']))
    route_stats = {
        name: _score_route_meta(sampled, name, scores[name])
        for name in ('q', 'k', 'v', 'rst')
    }
    return scores, sampled, _pool_score_page_stats(route_stats)


def _local_target_from_pool_target(target_pool, candidate_frac,
                                   pages_enabled):
    target_pool = float(target_pool)
    candidate_frac = float(candidate_frac)
    if not pages_enabled:
        return float(np.clip(target_pool, 1.0e-4, 0.95))
    return float(np.clip(
        target_pool / max(candidate_frac, 1.0e-8), 1.0e-4, 0.95))


def _score_distribution_stats(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        raise ValueError("tau calibration score sample is empty.")
    return {
        'rho_cand_mean': float(np.mean(values)),
        'rho_cand_std': float(np.std(values)),
        'rho_cand_p50': float(np.quantile(values, 0.50)),
        'rho_cand_p90': float(np.quantile(values, 0.90)),
        'rho_cand_p99': float(np.quantile(values, 0.99)),
    }


def _tau_calibration_diag(pool, scores, pool_stats, target_pool,
                          target_local, tau):
    stats = _score_distribution_stats(scores)
    meta = dict(pool_stats)
    return {
        'pool': pool,
        'target_pool': float(target_pool),
        'candidate_frac': float(meta.get('candidate_frac', 1.0)),
        'target_local': float(target_local),
        'tau': float(tau),
        'candidate_count': float(meta.get('candidate_count', 0.0)),
        'valid_candidate_count': float(
            meta.get('candidate_valid_count', 0.0)),
        'full_pool_size': float(meta.get('full_pool_size', 0.0)),
        'pages_enabled': bool(meta.get('pages_enabled', False)),
        **stats,
    }


def _compute_srw_quantile_tau_init(params, input_ids, cfg,
                                   tau_init_cfg):
    """Compute host-side quantiles from a small deterministic score sample."""
    version = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)
    del version
    scores, sampled, page_stats = _sample_srw_selection_scores(
        params, input_ids, cfg, tau_init_cfg['calibration_tokens'])

    tau = {}
    estimated_active = {}
    estimated_active_local = {}
    estimated_active_pool = {}
    target_local = {}
    tau_calibration = {}
    for pool in ('qk', 'v', 'rst'):
        target = tau_init_cfg['targets'][pool]
        meta = page_stats[pool]
        local_target = _local_target_from_pool_target(
            target, meta.get('candidate_frac', 1.0),
            bool(meta.get('pages_enabled', False)))
        quantile_tau = _array_quantile(scores[pool], 1.0 - local_target)
        quantile_tau = float(np.clip(
            quantile_tau, tau_init_cfg['tau_min'], tau_init_cfg['tau_max']))
        tau[pool] = quantile_tau
        target_local[pool] = local_target
        active_local = float(np.mean(scores[pool] > quantile_tau))
        active_pool = active_local * float(meta.get('candidate_frac', 1.0))
        estimated_active_local[pool] = active_local
        estimated_active_pool[pool] = active_pool
        estimated_active[pool] = active_pool
        tau_calibration[pool] = _tau_calibration_diag(
            pool, scores[pool], meta, target, local_target, quantile_tau)

    return {
        'type': 'tau_init',
        'tau_init_mode': 'quantile_frac',
        'tau_init_target_frac': dict(tau_init_cfg['targets']),
        'tau_init_target_local_frac': target_local,
        'tau_init_quantile_tau': tau,
        'tau_init_est_active': estimated_active,
        'tau_init_est_active_local': estimated_active_local,
        'tau_init_est_active_pool': estimated_active_pool,
        'tau_init_target_qk_frac': tau_init_cfg['targets']['qk'],
        'tau_init_target_v_frac': tau_init_cfg['targets']['v'],
        'tau_init_target_rst_frac': tau_init_cfg['targets']['rst'],
        'tau_init_target_local_qk_frac': target_local['qk'],
        'tau_init_target_local_v_frac': target_local['v'],
        'tau_init_target_local_rst_frac': target_local['rst'],
        'tau_init_quantile_tau_qk': tau['qk'],
        'tau_init_quantile_tau_v': tau['v'],
        'tau_init_quantile_tau_rst': tau['rst'],
        'tau_init_est_active_qk': estimated_active_pool['qk'],
        'tau_init_est_active_v': estimated_active_pool['v'],
        'tau_init_est_active_rst': estimated_active_pool['rst'],
        'tau_init_est_active_local_qk': estimated_active_local['qk'],
        'tau_init_est_active_local_v': estimated_active_local['v'],
        'tau_init_est_active_local_rst': estimated_active_local['rst'],
        'tau_init_est_active_q': (
            float(np.mean(scores['q'] > tau['qk']))
            * float(page_stats['q'].get('candidate_frac', 1.0))),
        'tau_init_est_active_k': (
            float(np.mean(scores['k'] > tau['qk']))
            * float(page_stats['k'].get('candidate_frac', 1.0))),
        'tau_init_est_active_local_q': float(
            np.mean(scores['q'] > tau['qk'])),
        'tau_init_est_active_local_k': float(
            np.mean(scores['k'] > tau['qk'])),
        'tau_calibration': tau_calibration,
        'tau_init_calibration': {
            'batch': 'first_train_batch_host0',
            'token_sampling': 'evenly_spaced_flat',
            'tokens': int(np.asarray(sampled.get(
                'tokens', np.asarray(sampled['q']).shape[0]))),
            'neurons_qk': int(page_stats['qk']['full_pool_size']),
            'neurons_v': int(page_stats['v']['full_pool_size']),
            'neurons_rst': int(page_stats['rst']['full_pool_size']),
        },
    }


def _dict_without_private_keys(src):
    return {k: v for k, v in dict(src).items() if not str(k).startswith('_')}


def _operation_space_disabled_selection_calibration_config(cfg):
    raw = cfg.get('training', {}).get('selection_calibration', None)
    out = {'enabled': False, 'present': raw is not None}
    if isinstance(raw, dict):
        out['raw'] = _dict_without_private_keys(raw)
    return out


def _array_quantile(values, q):
    return float(np.quantile(np.asarray(values), q))


def _selection_calibration_config(cfg, tau_init_cfg=None):
    """Parse optional training.selection_calibration configuration."""
    tcfg = cfg.get('training', {})
    if 'selection_policy' in tcfg:
        raise ValueError(
            "Use training.selection_calibration; selection_policy is not "
            "a supported config key.")

    raw = tcfg.get('selection_calibration', None)
    if raw is None:
        return {'enabled': False, 'present': False}
    if not isinstance(raw, dict):
        raise ValueError("training.selection_calibration must be a mapping.")
    if 'run_on' in raw:
        raise ValueError(
            "training.selection_calibration.run_on was removed. Selection "
            "calibration policy is fixed: fresh init computes, resume "
            "restores.")

    enabled = _cfg_bool(
        raw.get('enabled', False),
        name='selection_calibration.enabled')
    if not enabled:
        return {
            'enabled': False,
            'present': True,
            'raw': _dict_without_private_keys(raw),
        }

    forbidden = (
        'max_tokens',
        'calibration_batches',
        'tau',
        'soft_gate',
        'candidate_multiplier',
        'min_candidate_delta',
        'max_candidate_frac',
        'min_band',
        'max_band',
    )
    present_forbidden = [key for key in forbidden if key in raw]
    if present_forbidden:
        raise ValueError(
            "training.selection_calibration enabled schema does not support "
            f"these legacy keys: {present_forbidden}.")

    calibration_tokens = int(raw.get('calibration_tokens', 0))
    calibration_max_batches = int(raw.get('calibration_max_batches', 0))
    histogram_bins = int(raw.get('histogram_bins', 0))
    score_chunk_tokens = int(raw.get(
        'score_chunk_tokens',
        DEFAULT_SELECTION_CALIBRATION_SCORE_CHUNK_TOKENS))
    score_min = float(raw.get('score_min', -1.0))
    score_max = float(raw.get('score_max', 1.0))
    candidate_eps = float(raw.get('candidate_eps', 1.0e-3))
    if calibration_tokens <= 0:
        raise ValueError(
            "selection_calibration.calibration_tokens must be > 0, "
            f"got {calibration_tokens}.")
    if calibration_max_batches <= 0:
        raise ValueError(
            "selection_calibration.calibration_max_batches must be > 0, "
            f"got {calibration_max_batches}.")
    if histogram_bins <= 0:
        raise ValueError(
            "selection_calibration.histogram_bins must be > 0, "
            f"got {histogram_bins}.")
    if score_chunk_tokens <= 0:
        raise ValueError(
            "selection_calibration.score_chunk_tokens must be > 0, "
            f"got {score_chunk_tokens}.")
    if (not np.isfinite(score_min) or not np.isfinite(score_max)
            or score_min >= score_max):
        raise ValueError(
            "selection_calibration score range must satisfy "
            f"score_min < score_max, got {score_min}/{score_max}.")
    if not np.isfinite(candidate_eps) or not (0.0 < candidate_eps < 1.0):
        raise ValueError(
            "selection_calibration.candidate_eps must be in (0, 1), "
            f"got {candidate_eps}.")

    def _pool_targets(name):
        src = raw.get(name)
        if not isinstance(src, dict):
            raise ValueError(f"selection_calibration.{name} must be a mapping.")
        out = {}
        for pool in POOL_SCHEDULE_NAMES:
            if pool not in src:
                raise ValueError(
                    f"selection_calibration.{name}.{pool} is required.")
            value = float(src[pool])
            if not np.isfinite(value) or not (0.0 < value < 1.0):
                raise ValueError(
                    f"selection_calibration.{name}.{pool} must be in "
                    f"(0, 1), got {value}.")
            out[pool] = value
        return out

    def _pool_positive_values(name):
        src = raw.get(name)
        if not isinstance(src, dict):
            raise ValueError(f"selection_calibration.{name} must be a mapping.")
        out = {}
        for pool in POOL_SCHEDULE_NAMES:
            if pool not in src:
                raise ValueError(
                    f"selection_calibration.{name}.{pool} is required.")
            value = float(src[pool])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"selection_calibration.{name}.{pool} must be > 0, "
                    f"got {value}.")
            out[pool] = value
        return out

    active_target = _pool_targets('active_target')
    candidate_target_start = _pool_targets('candidate_target_start')
    candidate_target_final = _pool_targets('candidate_target_final')
    b_floor = _pool_positive_values('B_floor')
    for pool in POOL_SCHEDULE_NAMES:
        if candidate_target_start[pool] <= active_target[pool]:
            raise ValueError(
                "selection_calibration.candidate_target_start must be "
                f"larger than active_target for {pool}, got "
                f"{candidate_target_start[pool]} <= {active_target[pool]}.")
        if candidate_target_final[pool] < active_target[pool]:
            raise ValueError(
                "selection_calibration.candidate_target_final must be "
                f">= active_target for {pool}, got "
                f"{candidate_target_final[pool]} < {active_target[pool]}.")
    annealing = raw.get('annealing')
    if not isinstance(annealing, dict):
        raise ValueError(
            "selection_calibration.annealing must be a mapping.")
    unit = str(annealing.get('unit', '')).strip().lower()
    if unit != 'tokens':
        raise ValueError(
            "selection_calibration.annealing.unit must be 'tokens', "
            f"got {annealing.get('unit')!r}.")
    shrink_start_tokens = int(annealing.get('shrink_start_tokens', 0))
    shrink_end_tokens = int(annealing.get('shrink_end_tokens', 0))
    if not (0 <= shrink_start_tokens < shrink_end_tokens):
        raise ValueError(
            "selection_calibration annealing tokens must satisfy "
            "0 <= shrink_start_tokens < shrink_end_tokens, got "
            f"{shrink_start_tokens}, {shrink_end_tokens}.")

    max_train_tokens = int(cfg.get('data', {}).get('max_train_tokens', 0))
    if max_train_tokens <= 0:
        raise ValueError(
            "data.max_train_tokens must be > 0 for selection_calibration.")
    formation_end_frac = float(np.clip(
        shrink_start_tokens / max_train_tokens, 0.0, 1.0))
    sharpen_end_frac = float(np.clip(
        shrink_end_tokens / max_train_tokens, 0.0, 1.0))
    if not (0.0 < formation_end_frac < sharpen_end_frac <= 1.0):
        raise ValueError(
            "selection_calibration annealing fractions must satisfy "
            "0 < formation_end_frac < sharpen_end_frac <= 1 after "
            f"conversion/clamp, got {formation_end_frac}, "
            f"{sharpen_end_frac}.")

    boundary_power_start = float(tcfg.get(
        'soft_gate_boundary_power_start', 3.0))
    boundary_power_mid = float(tcfg.get(
        'soft_gate_boundary_power_mid', 3.0))
    boundary_power_final = float(tcfg.get(
        'soft_gate_boundary_power_final', 4.0))
    raw_out = _dict_without_private_keys(raw)
    raw_out['annealing'] = {
        'unit': unit,
        'shrink_start_tokens': shrink_start_tokens,
        'shrink_end_tokens': shrink_end_tokens,
    }

    return {
        'enabled': True,
        'present': True,
        'raw': raw_out,
        'calibration_tokens': calibration_tokens,
        'calibration_max_batches': calibration_max_batches,
        'histogram_bins': histogram_bins,
        'score_chunk_tokens': score_chunk_tokens,
        'score_min': score_min,
        'score_max': score_max,
        'candidate_eps': candidate_eps,
        'active_target': active_target,
        'candidate_target_start': candidate_target_start,
        'candidate_target_final': candidate_target_final,
        'B_floor': b_floor,
        'annealing': {
            'unit': unit,
            'shrink_start_tokens': shrink_start_tokens,
            'shrink_end_tokens': shrink_end_tokens,
        },
        'formation_end_frac': formation_end_frac,
        'sharpen_end_frac': sharpen_end_frac,
        'boundary_power_start': boundary_power_start,
        'boundary_power_mid': boundary_power_mid,
        'boundary_power_final': boundary_power_final,
    }


def _histogram_quantile(counts, score_min, score_max, q):
    """Approximate a quantile from fixed-width histogram counts."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size <= 0:
        raise ValueError("histogram quantile counts must be a 1-D array.")
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("histogram quantile requires at least one count.")
    q = float(np.clip(q, 0.0, 1.0))
    cumulative = np.cumsum(counts)
    threshold = q * total
    idx = int(np.searchsorted(cumulative, threshold, side='left'))
    idx = max(0, min(idx, counts.size - 1))
    bin_width = (float(score_max) - float(score_min)) / float(counts.size)
    return float(score_min + (idx + 0.5) * bin_width)


SELECTION_CALIBRATION_SCORE_ROUTES = ('q', 'k', 'v', 'rst')
SELECTION_CALIBRATION_PAGE_STAT_FIELDS = (
    'candidate_valid_count_weighted_sum',
    'candidate_count_weighted_sum',
    'candidate_group_count',
    'full_pool_size',
    'pages_enabled',
)
SELECTION_CALIBRATION_PAGE_STAT_KEYS = tuple(
    f'{route}_{field}'
    for route in SELECTION_CALIBRATION_SCORE_ROUTES
    for field in SELECTION_CALIBRATION_PAGE_STAT_FIELDS
)


def _make_selection_calibration_histogram_fn(
        score_impl, score_kwargs, histogram_bins, score_min, score_max):
    """Create a jitted per-batch histogram function returning counts only."""
    histogram_bins = int(histogram_bins)
    score_min = float(score_min)
    score_max = float(score_max)

    def _hist_counts(x, valid_mask=None):
        x = jnp.ravel(jnp.asarray(x, dtype=jnp.float32))
        if valid_mask is None:
            valid = jnp.ones_like(x, dtype=jnp.bool_)
        else:
            valid = jnp.ravel(jnp.asarray(valid_mask, dtype=jnp.bool_))
        x = jnp.clip(x, score_min, score_max)
        x = jnp.where(valid, x, jnp.float32(score_min))

        scale = float(histogram_bins) / float(score_max - score_min)
        idx = jnp.floor((x - float(score_min)) * scale).astype(jnp.int32)
        idx = jnp.clip(idx, 0, int(histogram_bins) - 1)

        return jnp.bincount(
            idx,
            weights=valid.astype(jnp.int32),
            length=int(histogram_bins)).astype(jnp.int32)

    def _hist_fn(score_params, input_ids):
        score_tensors = score_impl(score_params, input_ids, **score_kwargs)
        q_counts = _hist_counts(
            score_tensors['q'], score_tensors.get('q_valid_mask', None))
        k_counts = _hist_counts(
            score_tensors['k'], score_tensors.get('k_valid_mask', None))
        page_stats = {}
        for route in SELECTION_CALIBRATION_SCORE_ROUTES:
            values = score_tensors[route]
            full_pool_size = jnp.asarray(
                score_tensors.get(
                    f'{route}_full_pool_size',
                    jnp.float32(values.shape[-1])),
                dtype=jnp.float32)
            candidate_valid_count = jnp.asarray(
                score_tensors.get(
                    f'{route}_candidate_valid_count', full_pool_size),
                dtype=jnp.float32)
            candidate_count = jnp.asarray(
                score_tensors.get(
                    f'{route}_candidate_count', full_pool_size),
                dtype=jnp.float32)
            candidate_group_count = jnp.asarray(
                score_tensors.get(
                    f'{route}_candidate_group_count', jnp.float32(1.0)),
                dtype=jnp.float32)
            pages_enabled = jnp.asarray(
                score_tensors.get(
                    f'{route}_pages_enabled', jnp.float32(0.0)),
                dtype=jnp.float32)
            page_stats[
                f'{route}_candidate_valid_count_weighted_sum'] = (
                    candidate_valid_count * candidate_group_count)
            page_stats[f'{route}_candidate_count_weighted_sum'] = (
                candidate_count * candidate_group_count)
            page_stats[f'{route}_candidate_group_count'] = (
                candidate_group_count)
            page_stats[f'{route}_full_pool_size'] = full_pool_size
            page_stats[f'{route}_pages_enabled'] = pages_enabled
        return {
            'qk': q_counts + k_counts,
            'v': _hist_counts(
                score_tensors['v'], score_tensors.get('v_valid_mask', None)),
            'rst': _hist_counts(
                score_tensors['rst'],
                score_tensors.get('rst_valid_mask', None)),
        }, jnp.asarray(
            score_tensors.get('tokens', score_tensors['q'].shape[0]),
            dtype=jnp.int32), page_stats

    return jax.jit(_hist_fn)


def _selection_calibration_row_chunk_size(
        batch_rows, seq_len, score_chunk_tokens):
    """Pick a stable row chunk whose token count fits the score budget."""
    batch_rows = int(batch_rows)
    seq_len = int(seq_len)
    score_chunk_tokens = int(score_chunk_tokens)
    if batch_rows <= 0:
        raise ValueError(
            f"selection calibration batch must have rows, got {batch_rows}.")
    if seq_len <= 0:
        raise ValueError(
            f"selection calibration sequence length must be > 0, got {seq_len}.")
    if score_chunk_tokens <= 0:
        raise ValueError(
            "selection_calibration.score_chunk_tokens must be > 0, "
            f"got {score_chunk_tokens}.")

    row_chunk = max(1, min(batch_rows, score_chunk_tokens // seq_len))
    while row_chunk > 1 and batch_rows % row_chunk != 0:
        row_chunk -= 1
    return row_chunk


def _collect_selection_calibration_histograms(
        params, train_loader, cfg, selection_calibration_cfg):
    """Stream real training batches and keep only host-side hist counts."""
    global_calibration_tokens = int(
        selection_calibration_cfg['calibration_tokens'])
    process_count = max(1, int(jax.process_count()))
    local_calibration_tokens = int(
        math.ceil(global_calibration_tokens / process_count))
    score_chunk_tokens = int(
        selection_calibration_cfg['score_chunk_tokens'])
    _version, score_impl, score_params, score_kwargs = (
        _srw_selection_score_setup(params, cfg, score_chunk_tokens))
    hist_fn = _make_selection_calibration_histogram_fn(
        score_impl,
        score_kwargs,
        selection_calibration_cfg['histogram_bins'],
        selection_calibration_cfg['score_min'],
        selection_calibration_cfg['score_max'])
    counts = {
        pool: np.zeros(
            selection_calibration_cfg['histogram_bins'], dtype=np.int64)
        for pool in POOL_SCHEDULE_NAMES
    }
    page_stats = {
        key: 0.0 for key in SELECTION_CALIBRATION_PAGE_STAT_KEYS
    }
    seen_tokens = 0
    actual_batches = 0
    if jax.process_index() == 0:
        print(
            "\r[selection_calibration] "
            "compiling/running histogram calibration "
            f"(score_chunk_tokens={score_chunk_tokens})...",
            end="",
            flush=True)
    for input_ids, _attention_mask in train_loader:
        if (seen_tokens >= local_calibration_tokens
                or actual_batches >=
                selection_calibration_cfg['calibration_max_batches']):
            break
        input_ids = np.asarray(input_ids, dtype=np.int32)
        if input_ids.ndim != 2:
            raise ValueError(
                "selection calibration input_ids must be rank 2 "
                f"[batch, seq], got shape={input_ids.shape}.")
        batch_rows, seq_len = input_ids.shape
        row_chunk = _selection_calibration_row_chunk_size(
            batch_rows, seq_len, score_chunk_tokens)
        batch_processed = False
        for row_start in range(0, batch_rows, row_chunk):
            if seen_tokens >= local_calibration_tokens:
                break
            row_end = min(row_start + row_chunk, batch_rows)
            batch_counts, token_count, batch_page_stats = hist_fn(
                score_params, input_ids[row_start:row_end])
            host_counts = jax.device_get(batch_counts)
            for pool in POOL_SCHEDULE_NAMES:
                counts[pool] += np.asarray(
                    host_counts[pool], dtype=np.int64)
            host_page_stats = jax.device_get(batch_page_stats)
            for key in SELECTION_CALIBRATION_PAGE_STAT_KEYS:
                value = float(host_page_stats.get(key, 0.0))
                if key.endswith('_full_pool_size') or key.endswith('_pages_enabled'):
                    page_stats[key] = max(page_stats[key], value)
                else:
                    page_stats[key] += value
            seen_tokens += int(jax.device_get(token_count))
            batch_processed = True
        if not batch_processed:
            break
        actual_batches += 1
        if jax.process_index() == 0:
            pct = min(
                100.0,
                100.0 * seen_tokens / max(1, local_calibration_tokens))
            print(
                "\r[selection_calibration] "
                f"batch={actual_batches} "
                f"seen_tokens_local={seen_tokens}/{local_calibration_tokens} "
                f"({pct:.1f}%)",
                end="",
                flush=True)
    if jax.process_index() == 0:
        print("", flush=True)
    if actual_batches <= 0:
        raise ValueError(
            "selection_calibration requires at least one training batch.")
    return (
        counts,
        page_stats,
        seen_tokens,
        actual_batches,
        local_calibration_tokens,
        process_count,
    )


def _aggregate_selection_calibration_histograms(
        local_counts, local_page_stats, local_seen_tokens,
        local_actual_batches):
    """Aggregate per-host histogram counts for multi-host calibration."""
    if jax.process_count() <= 1:
        return (
            local_counts,
            dict(local_page_stats),
            int(local_seen_tokens),
            int(local_actual_batches),
        )

    local_stack = np.stack(
        [local_counts[pool] for pool in POOL_SCHEDULE_NAMES],
        axis=0).astype(np.int64)
    gathered_stack = np.asarray(process_allgather(local_stack))
    gathered_stack = gathered_stack.reshape(
        (jax.process_count(),) + local_stack.shape)
    global_stack = np.sum(gathered_stack, axis=0, dtype=np.int64)

    local_meta = np.asarray(
        [local_seen_tokens, local_actual_batches], dtype=np.int64)
    gathered_meta = np.asarray(process_allgather(local_meta))
    gathered_meta = gathered_meta.reshape((jax.process_count(), 2))
    global_counts = {
        pool: global_stack[idx]
        for idx, pool in enumerate(POOL_SCHEDULE_NAMES)
    }
    local_page_vec = np.asarray(
        [float(local_page_stats.get(key, 0.0))
         for key in SELECTION_CALIBRATION_PAGE_STAT_KEYS],
        dtype=np.float64)
    gathered_page_vec = np.asarray(process_allgather(local_page_vec))
    gathered_page_vec = gathered_page_vec.reshape(
        (jax.process_count(), local_page_vec.shape[0]))
    global_page_stats = {}
    for idx, key in enumerate(SELECTION_CALIBRATION_PAGE_STAT_KEYS):
        if key.endswith('_full_pool_size') or key.endswith('_pages_enabled'):
            global_page_stats[key] = float(np.max(gathered_page_vec[:, idx]))
        else:
            global_page_stats[key] = float(np.sum(gathered_page_vec[:, idx]))
    return (
        global_counts,
        global_page_stats,
        int(np.sum(gathered_meta[:, 0], dtype=np.int64)),
        int(np.max(gathered_meta[:, 1])),
    )


def _selection_calibration_pool_page_stats(page_stats):
    page_stats = dict(page_stats or {})

    def _route(route):
        groups = float(page_stats.get(
            f'{route}_candidate_group_count', 0.0))
        full_pool_size = float(page_stats.get(
            f'{route}_full_pool_size', 0.0))
        pages_enabled = (
            float(page_stats.get(f'{route}_pages_enabled', 0.0)) > 0.5)
        if groups > 0.0:
            valid_count = float(page_stats.get(
                f'{route}_candidate_valid_count_weighted_sum', 0.0)) / groups
            candidate_count = float(page_stats.get(
                f'{route}_candidate_count_weighted_sum', 0.0)) / groups
        else:
            valid_count = full_pool_size
            candidate_count = full_pool_size
        candidate_frac = (
            valid_count / max(full_pool_size, 1.0)
            if pages_enabled else 1.0)
        return {
            'pages_enabled': bool(pages_enabled),
            'candidate_valid_count': float(valid_count),
            'candidate_count': float(candidate_count),
            'full_pool_size': float(full_pool_size),
            'candidate_frac': float(candidate_frac),
        }

    route = {name: _route(name) for name in SELECTION_CALIBRATION_SCORE_ROUTES}
    q = route['q']
    k = route['k']
    qk = {
        'pages_enabled': bool(q['pages_enabled'] or k['pages_enabled']),
        'candidate_valid_count': 0.5 * (
            q['candidate_valid_count'] + k['candidate_valid_count']),
        'candidate_count': 0.5 * (
            q['candidate_count'] + k['candidate_count']),
        'full_pool_size': max(q['full_pool_size'], k['full_pool_size']),
    }
    qk['candidate_frac'] = (
        qk['candidate_valid_count'] / max(qk['full_pool_size'], 1.0)
        if qk['pages_enabled'] else 1.0)
    return {
        'qk': qk,
        'v': route['v'],
        'rst': route['rst'],
    }


def _histogram_distribution_stats(counts, score_min, score_max):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("histogram stats require at least one count.")
    bin_width = (float(score_max) - float(score_min)) / float(counts.size)
    centers = (
        float(score_min)
        + (np.arange(counts.size, dtype=np.float64) + 0.5) * bin_width)
    mean = float(np.sum(counts * centers) / total)
    var = float(np.sum(counts * np.square(centers - mean)) / total)
    return {
        'rho_cand_mean': mean,
        'rho_cand_std': float(np.sqrt(max(var, 0.0))),
        'rho_cand_p50': _histogram_quantile(
            counts, score_min, score_max, 0.50),
        'rho_cand_p90': _histogram_quantile(
            counts, score_min, score_max, 0.90),
        'rho_cand_p99': _histogram_quantile(
            counts, score_min, score_max, 0.99),
    }


def _compute_srw_selection_calibration(
        histogram_counts, cfg, selection_calibration_cfg, page_stats,
        seen_tokens, actual_batches, local_calibration_tokens,
        process_count):
    """Calibrate tau init and soft-gate candidate bands from histograms."""
    model_cfg = cfg['model']
    tau_min = float(model_cfg.get('tau_init_min', -1.0))
    tau_max = float(model_cfg.get('tau_init_max', 1.0))
    if (not np.isfinite(tau_min) or not np.isfinite(tau_max)
            or tau_min > tau_max):
        raise ValueError(
            "model.tau_init_min/max must satisfy tau_init_min <= "
            f"tau_init_max, got {tau_min}/{tau_max}.")

    tau = {}
    q_start = {}
    q_final = {}
    b_start = {}
    b_final = {}
    final_hard_close = {}
    b_final_set_by_floor = {}
    active_target = selection_calibration_cfg['active_target']
    candidate_start = selection_calibration_cfg['candidate_target_start']
    candidate_final = selection_calibration_cfg['candidate_target_final']
    b_floor = selection_calibration_cfg['B_floor']
    score_min = selection_calibration_cfg['score_min']
    score_max = selection_calibration_cfg['score_max']
    eps_scale = -np.log(selection_calibration_cfg['candidate_eps'])
    power_start = selection_calibration_cfg['boundary_power_start']
    power_final = selection_calibration_cfg['boundary_power_final']
    scale_start = eps_scale ** (1.0 / power_start)
    scale_final = eps_scale ** (1.0 / power_final)
    pool_page_stats = _selection_calibration_pool_page_stats(page_stats)
    active_target_local = {}
    candidate_start_local = {}
    candidate_final_local = {}
    tau_calibration = {}
    for pool in POOL_SCHEDULE_NAMES:
        counts = histogram_counts[pool]
        meta = pool_page_stats[pool]
        active_target_local[pool] = _local_target_from_pool_target(
            active_target[pool], meta.get('candidate_frac', 1.0),
            bool(meta.get('pages_enabled', False)))
        candidate_start_local[pool] = _local_target_from_pool_target(
            candidate_start[pool], meta.get('candidate_frac', 1.0),
            bool(meta.get('pages_enabled', False)))
        candidate_final_local[pool] = _local_target_from_pool_target(
            candidate_final[pool], meta.get('candidate_frac', 1.0),
            bool(meta.get('pages_enabled', False)))
        tau_raw = _histogram_quantile(
            counts, score_min, score_max, 1.0 - active_target_local[pool])
        tau_pool = float(np.clip(tau_raw, tau_min, tau_max))
        q_start_pool = _histogram_quantile(
            counts, score_min, score_max, 1.0 - candidate_start_local[pool])
        q_final_pool = _histogram_quantile(
            counts, score_min, score_max, 1.0 - candidate_final_local[pool])
        tau[pool] = tau_pool
        q_start[pool] = q_start_pool
        q_final[pool] = q_final_pool
        final_hard_close[pool] = (
            candidate_final[pool] == active_target[pool])

        delta_start = max(tau_pool - q_start_pool, 0.0)
        b_start_raw = delta_start / scale_start
        b_start[pool] = float(max(b_floor[pool], b_start_raw))

        delta_final = max(tau_pool - q_final_pool, 0.0)
        b_final_raw = delta_final / scale_final
        if final_hard_close[pool]:
            b_final[pool] = float(b_floor[pool])
        else:
            b_final[pool] = float(max(b_floor[pool], b_final_raw))
        b_final_set_by_floor[pool] = (b_final[pool] == float(b_floor[pool]))
        tau_calibration[pool] = {
            'pool': pool,
            'target_pool': float(active_target[pool]),
            'candidate_frac': float(meta.get('candidate_frac', 1.0)),
            'target_local': float(active_target_local[pool]),
            'tau': float(tau_pool),
            'candidate_count': float(meta.get('candidate_count', 0.0)),
            'valid_candidate_count': float(
                meta.get('candidate_valid_count', 0.0)),
            'full_pool_size': float(meta.get('full_pool_size', 0.0)),
            'pages_enabled': bool(meta.get('pages_enabled', False)),
            **_histogram_distribution_stats(counts, score_min, score_max),
        }

    return {
        'type': 'selection_calibration',
        'selection_calibration_enabled': True,
        'selection_calibration_applied': True,
        'selection_calibration_seen_tokens_target':
            int(selection_calibration_cfg['calibration_tokens']),
        'selection_calibration_tokens_target_global':
            int(selection_calibration_cfg['calibration_tokens']),
        'selection_calibration_tokens_target_local':
            int(local_calibration_tokens),
        'selection_calibration_process_count': int(process_count),
        'selection_calibration_seen_tokens': int(seen_tokens),
        'selection_calibration_seen_tokens_global': int(seen_tokens),
        'selection_calibration_actual_batches': int(actual_batches),
        'selection_calibration_actual_batches_per_host_or_max':
            int(actual_batches),
        'selection_calibration_histogram_bins':
            int(selection_calibration_cfg['histogram_bins']),
        'selection_calibration_score_chunk_tokens':
            int(selection_calibration_cfg['score_chunk_tokens']),
        'selection_calibration_score_min': float(score_min),
        'selection_calibration_score_max': float(score_max),
        'selection_calibration_candidate_eps':
            float(selection_calibration_cfg['candidate_eps']),
        'selection_calibration_active_target': dict(active_target),
        'selection_calibration_candidate_target_start': dict(candidate_start),
        'selection_calibration_candidate_target_final': dict(candidate_final),
        'selection_calibration_active_target_local':
            dict(active_target_local),
        'selection_calibration_candidate_target_start_local':
            dict(candidate_start_local),
        'selection_calibration_candidate_target_final_local':
            dict(candidate_final_local),
        'selection_calibration_page_candidate_stats': pool_page_stats,
        'selection_calibration_B_floor': dict(b_floor),
        'selection_calibration_tau': tau,
        'selection_calibration_q_start': q_start,
        'selection_calibration_q_final': q_final,
        'selection_calibration_B_start': b_start,
        'selection_calibration_B_final': b_final,
        'selection_calibration_final_hard_close': final_hard_close,
        'selection_calibration_B_final_set_by_floor': b_final_set_by_floor,
        'selection_calibration_shrink_start_tokens':
            int(selection_calibration_cfg['annealing']['shrink_start_tokens']),
        'selection_calibration_shrink_end_tokens':
            int(selection_calibration_cfg['annealing']['shrink_end_tokens']),
        'selection_calibration_formation_end_frac':
            float(selection_calibration_cfg['formation_end_frac']),
        'selection_calibration_sharpen_end_frac':
            float(selection_calibration_cfg['sharpen_end_frac']),
        'tau_init_mode': 'selection_calibration',
        'tau_init_quantile_tau': tau,
        'tau_calibration': tau_calibration,
    }


def _selection_calibration_materialized_training_updates(
        selection_summary, selection_calibration_cfg):
    updates = {}
    b_start = selection_summary['selection_calibration_B_start']
    b_final = selection_summary['selection_calibration_B_final']
    formation_end_frac = float(
        selection_summary['selection_calibration_formation_end_frac'])
    sharpen_end_frac = float(
        selection_summary['selection_calibration_sharpen_end_frac'])
    for pool in POOL_SCHEDULE_NAMES:
        prefix = f'soft_gate_t_{pool}'
        updates.update({
            f'{prefix}_schedule': 'developmental_band',
            f'{prefix}_sort': float(b_start[pool]),
            f'{prefix}_band': float(b_start[pool]),
            f'{prefix}_mid': float(b_start[pool]),
            f'{prefix}_late': float(b_final[pool]),
            f'{prefix}_final': float(b_final[pool]),
            f'{prefix}_sort_end_frac': 0.0,
            f'{prefix}_band_reach_frac': formation_end_frac,
            f'{prefix}_formation_end_frac': formation_end_frac,
            f'{prefix}_sharpen_end_frac': sharpen_end_frac,
            f'{prefix}_formation_power': 1.0,
            f'{prefix}_sharpen_power': 1.0,
        })
    updates.update({
        'soft_gate_boundary_power_start':
            float(selection_calibration_cfg['boundary_power_start']),
        'soft_gate_boundary_power_mid':
            float(selection_calibration_cfg['boundary_power_mid']),
        'soft_gate_boundary_power_final':
            float(selection_calibration_cfg['boundary_power_final']),
        'soft_gate_boundary_power_start_frac': 0.0,
        'soft_gate_boundary_power_mid_frac': formation_end_frac,
        'soft_gate_boundary_power_final_frac': sharpen_end_frac,
    })
    return updates


def _apply_selection_calibrated_soft_gate_schedules(
        pool_schedules, materialized_updates):
    """Return soft-gate schedules with calibrated developmental bands."""
    updated = {pool: dict(pool_schedules[pool]) for pool in POOL_SCHEDULE_NAMES}
    for pool in POOL_SCHEDULE_NAMES:
        prefix = f'soft_gate_t_{pool}'
        for key in (
                'schedule',
                'sort',
                'band',
                'mid',
                'late',
                'final',
                'sort_end_frac',
                'band_reach_frac',
                'formation_end_frac',
                'sharpen_end_frac',
                'formation_power',
                'sharpen_power'):
            updated[pool][key] = materialized_updates[f'{prefix}_{key}']
    return updated


def _selection_calibration_checkpoint_updates(selection_summary):
    tau = selection_summary['selection_calibration_tau']
    b_start = selection_summary['selection_calibration_B_start']
    b_final = selection_summary['selection_calibration_B_final']
    return {
        'selection_calibration_applied': True,
        'selection_calibration_seen_tokens':
            selection_summary['selection_calibration_seen_tokens'],
        'selection_calibration_actual_batches':
            selection_summary['selection_calibration_actual_batches'],
        'selection_calibration_histogram_bins':
            selection_summary['selection_calibration_histogram_bins'],
        'selection_calibration_tau_qk': tau['qk'],
        'selection_calibration_tau_v': tau['v'],
        'selection_calibration_tau_rst': tau['rst'],
        'selection_calibration_B_start_qk': b_start['qk'],
        'selection_calibration_B_start_v': b_start['v'],
        'selection_calibration_B_start_rst': b_start['rst'],
        'selection_calibration_B_final_qk': b_final['qk'],
        'selection_calibration_B_final_v': b_final['v'],
        'selection_calibration_B_final_rst': b_final['rst'],
        'selection_calibration_formation_end_frac':
            selection_summary['selection_calibration_formation_end_frac'],
        'selection_calibration_sharpen_end_frac':
            selection_summary['selection_calibration_sharpen_end_frac'],
    }


SELECTION_CALIBRATION_RESUME_REQUIRED_FIELDS = (
    'selection_calibration_applied',
    'selection_calibration_seen_tokens',
    'selection_calibration_actual_batches',
    'selection_calibration_histogram_bins',
    'selection_calibration_tau_qk',
    'selection_calibration_tau_v',
    'selection_calibration_tau_rst',
    'selection_calibration_B_start_qk',
    'selection_calibration_B_start_v',
    'selection_calibration_B_start_rst',
    'selection_calibration_B_final_qk',
    'selection_calibration_B_final_v',
    'selection_calibration_B_final_rst',
    'selection_calibration_formation_end_frac',
    'selection_calibration_sharpen_end_frac',
    'soft_gate_t_qk_schedule',
    'soft_gate_t_qk_sort',
    'soft_gate_t_qk_band',
    'soft_gate_t_qk_mid',
    'soft_gate_t_qk_late',
    'soft_gate_t_qk_final',
    'soft_gate_t_qk_sort_end_frac',
    'soft_gate_t_qk_band_reach_frac',
    'soft_gate_t_qk_formation_end_frac',
    'soft_gate_t_qk_sharpen_end_frac',
    'soft_gate_t_qk_formation_power',
    'soft_gate_t_qk_sharpen_power',
    'soft_gate_t_v_schedule',
    'soft_gate_t_v_sort',
    'soft_gate_t_v_band',
    'soft_gate_t_v_mid',
    'soft_gate_t_v_late',
    'soft_gate_t_v_final',
    'soft_gate_t_v_sort_end_frac',
    'soft_gate_t_v_band_reach_frac',
    'soft_gate_t_v_formation_end_frac',
    'soft_gate_t_v_sharpen_end_frac',
    'soft_gate_t_v_formation_power',
    'soft_gate_t_v_sharpen_power',
    'soft_gate_t_rst_schedule',
    'soft_gate_t_rst_sort',
    'soft_gate_t_rst_band',
    'soft_gate_t_rst_mid',
    'soft_gate_t_rst_late',
    'soft_gate_t_rst_final',
    'soft_gate_t_rst_sort_end_frac',
    'soft_gate_t_rst_band_reach_frac',
    'soft_gate_t_rst_formation_end_frac',
    'soft_gate_t_rst_sharpen_end_frac',
    'soft_gate_t_rst_formation_power',
    'soft_gate_t_rst_sharpen_power',
    'soft_gate_boundary_power_start',
    'soft_gate_boundary_power_mid',
    'soft_gate_boundary_power_final',
    'soft_gate_boundary_power_start_frac',
    'soft_gate_boundary_power_mid_frac',
    'soft_gate_boundary_power_final_frac',
    'admission_den_power',
    'admission_den_grad_scale',
    'soft_gate_effective_active_eps',
)


def _require_selection_calibration_resume_fields(training_cfg):
    """Require checkpoint-materialized selection calibration resume fields."""
    if not isinstance(training_cfg, dict):
        missing = list(SELECTION_CALIBRATION_RESUME_REQUIRED_FIELDS)
    else:
        missing = [
            key for key in SELECTION_CALIBRATION_RESUME_REQUIRED_FIELDS
            if key not in training_cfg or training_cfg[key] is None
        ]
    if missing:
        raise RuntimeError(
            'Resume checkpoint is missing required materialized selection '
            'calibration fields. This checkpoint cannot be resumed '
            'deterministically; restart from scratch. Missing keys: '
            + ', '.join(missing)
        )
    if not bool(training_cfg.get('selection_calibration_applied', False)):
        raise RuntimeError(
            'Resume checkpoint selection_calibration_applied is not true. '
            'This checkpoint cannot be resumed deterministically with '
            'selection calibration enabled; restart from scratch.'
        )


def _selection_calibration_summary_lines(summary):
    active = summary['selection_calibration_active_target']
    active_local = summary.get(
        'selection_calibration_active_target_local', active)
    candidate_start = summary['selection_calibration_candidate_target_start']
    candidate_start_local = summary.get(
        'selection_calibration_candidate_target_start_local',
        candidate_start)
    candidate_final = summary['selection_calibration_candidate_target_final']
    candidate_final_local = summary.get(
        'selection_calibration_candidate_target_final_local',
        candidate_final)
    tau = summary['selection_calibration_tau']
    b_start = summary['selection_calibration_B_start']
    b_final = summary['selection_calibration_B_final']
    b_floor = summary['selection_calibration_B_floor']
    final_hard_close = summary.get(
        'selection_calibration_final_hard_close', {})
    b_final_set_by_floor = summary.get(
        'selection_calibration_B_final_set_by_floor', {})
    final_hard_close_any = any(
        bool(final_hard_close.get(pool, False))
        for pool in POOL_SCHEDULE_NAMES)
    lines = [
        "enabled=true",
        "policy=fresh_init_computes_resume_restores",
        "calibration_tokens_target_global="
        f"{summary['selection_calibration_tokens_target_global']}",
        "calibration_tokens_target_local="
        f"{summary['selection_calibration_tokens_target_local']}",
        f"process_count={summary['selection_calibration_process_count']}",
        "seen_tokens_global="
        f"{summary['selection_calibration_seen_tokens_global']}",
        "actual_batches_per_host_or_max="
        f"{summary['selection_calibration_actual_batches_per_host_or_max']}",
        f"histogram_bins={summary['selection_calibration_histogram_bins']}",
        "score_chunk_tokens="
        f"{summary.get('selection_calibration_score_chunk_tokens', 'unknown')}",
        "score_range=["
        f"{summary['selection_calibration_score_min']}, "
        f"{summary['selection_calibration_score_max']}]",
        f"candidate_eps={summary['selection_calibration_candidate_eps']}",
        "",
        "active_target["
        f"qk={active['qk']} v={active['v']} rst={active['rst']}]",
        "active_target_local["
        f"qk={active_local['qk']} v={active_local['v']} "
        f"rst={active_local['rst']}]",
        "candidate_target_start["
        f"qk={candidate_start['qk']} v={candidate_start['v']} "
        f"rst={candidate_start['rst']}]",
        "candidate_target_start_local["
        f"qk={candidate_start_local['qk']} "
        f"v={candidate_start_local['v']} "
        f"rst={candidate_start_local['rst']}]",
        "candidate_target_final["
        f"qk={candidate_final['qk']} v={candidate_final['v']} "
        f"rst={candidate_final['rst']}]",
        "candidate_target_final_local["
        f"qk={candidate_final_local['qk']} "
        f"v={candidate_final_local['v']} "
        f"rst={candidate_final_local['rst']}]",
        "",
        "tau["
        f"qk={tau['qk']:.6f} v={tau['v']:.6f} "
        f"rst={tau['rst']:.6f}]",
        "B_start["
        f"qk={b_start['qk']:.6f} v={b_start['v']:.6f} "
        f"rst={b_start['rst']:.6f}]",
        "B_final["
        f"qk={b_final['qk']:.6f} v={b_final['v']:.6f} "
        f"rst={b_final['rst']:.6f}]",
        "B_floor["
        f"qk={b_floor['qk']:.6f} v={b_floor['v']:.6f} "
        f"rst={b_floor['rst']:.6f}]",
        "final_hard_close="
        f"{str(final_hard_close_any).lower()} pools["
        f"qk={str(bool(final_hard_close.get('qk', False))).lower()} "
        f"v={str(bool(final_hard_close.get('v', False))).lower()} "
        f"rst={str(bool(final_hard_close.get('rst', False))).lower()}]",
        "B_final set by B_floor["
        f"qk={str(bool(b_final_set_by_floor.get('qk', False))).lower()} "
        f"v={str(bool(b_final_set_by_floor.get('v', False))).lower()} "
        f"rst={str(bool(b_final_set_by_floor.get('rst', False))).lower()}]",
        "",
        "annealing_tokens["
        "shrink_start="
        f"{summary['selection_calibration_shrink_start_tokens']} "
        "shrink_end="
        f"{summary['selection_calibration_shrink_end_tokens']}]",
        "fractions["
        "formation_end="
        f"{summary['selection_calibration_formation_end_frac']:.6f} "
        "sharpen_end="
        f"{summary['selection_calibration_sharpen_end_frac']:.6f}]",
    ]
    for pool in ('qk', 'v', 'rst'):
        diag = summary.get('tau_calibration', {}).get(pool)
        if diag:
            lines.append(
                "tau_calib: "
                f"pool={pool} "
                f"target_pool={diag['target_pool']:.6f} "
                f"candidate_frac={diag['candidate_frac']:.6f} "
                f"target_local={diag['target_local']:.6f} "
                f"tau={diag['tau']:.6f} "
                f"rho_cand_mean={diag['rho_cand_mean']:+.6f} "
                f"rho_cand_std={diag['rho_cand_std']:.6f} "
                f"rho_cand_p50={diag['rho_cand_p50']:+.6f} "
                f"rho_cand_p90={diag['rho_cand_p90']:+.6f} "
                f"rho_cand_p99={diag['rho_cand_p99']:+.6f} "
                f"cand={diag['candidate_count']:.0f} "
                f"valid={diag['valid_candidate_count']:.1f} "
                f"full={diag['full_pool_size']:.0f}")
    return lines


def _set_srw_quantile_tau_biases(params, tau_summary, model_version=OFFICIAL_MODEL_VERSION):
    """Overwrite SRW raw tau biases, preserving the pytree structure."""
    entry = _model_registry_entry(model_version)
    raw_tau_init_from_cosine_tau = entry['raw_tau_init_from_cosine_tau']
    tau = tau_summary['tau_init_quantile_tau']
    raw_qk = raw_tau_init_from_cosine_tau(tau['qk'])
    raw_v = raw_tau_init_from_cosine_tau(tau['v'])
    raw_rst = raw_tau_init_from_cosine_tau(tau['rst'])

    def _replace(path, value):
        keys = tuple(
            str(p.key if hasattr(p, 'key') else p)
            for p in path)
        if keys[-3:] == ('router', 'raw_tau_attn', 'bias'):
            return jnp.stack([raw_qk, raw_qk, raw_v]).astype(value.dtype)
        if keys[-3:] == ('router', 'raw_tau_rst', 'bias'):
            return jnp.full_like(value, raw_rst)
        return value

    return jax.tree.map_with_path(_replace, params)


def _materialize_fixed_tau_config(cfg, training_config, tau_summary,
                                  model_version=OFFICIAL_MODEL_VERSION):
    """Store calibrated fixed tau in config/metadata for param-free tau models."""
    if not _is_fixed_tau_srw_version(model_version):
        return False
    tau = tau_summary['tau_init_quantile_tau']
    updates = {
        'tau_init_attn_qk': float(tau['qk']),
        'tau_init_attn_v': float(tau['v']),
        'tau_init_rst': float(tau['rst']),
    }
    cfg.setdefault('model', {}).update(updates)
    cfg.setdefault('training', {}).update(updates)
    training_config.update(updates)
    return True


def _v4164_tau_init_summary_lines(summary):
    targets = summary['tau_init_target_frac']
    local_targets = summary.get('tau_init_target_local_frac', targets)
    tau = summary['tau_init_quantile_tau']
    active = summary['tau_init_est_active']
    active_local = summary.get('tau_init_est_active_local', active)
    sample = summary['tau_init_calibration']
    lines = [
        "tau_init_mode=quantile_frac",
        "tau_init_target_frac["
        f"qk={targets['qk']:.3f} v={targets['v']:.3f} "
        f"rst={targets['rst']:.3f}]",
        "tau_init_target_local_frac["
        f"qk={local_targets['qk']:.6f} "
        f"v={local_targets['v']:.6f} "
        f"rst={local_targets['rst']:.6f}]",
        "tau_init_quantile_tau["
        f"qk={tau['qk']:.6f} v={tau['v']:.6f} rst={tau['rst']:.6f}]",
        "tau_init_est_active_pool["
        f"qk={active['qk']:.6f} v={active['v']:.6f} "
        f"rst={active['rst']:.6f}]",
        "tau_init_est_active_local["
        f"qk={active_local['qk']:.6f} "
        f"v={active_local['v']:.6f} "
        f"rst={active_local['rst']:.6f}]",
        "tau_init_est_active_qk_split["
        f"q={summary['tau_init_est_active_q']:.6f} "
        f"k={summary['tau_init_est_active_k']:.6f}]",
        "tau_init_est_active_local_qk_split["
        f"q={summary.get('tau_init_est_active_local_q', summary['tau_init_est_active_q']):.6f} "
        f"k={summary.get('tau_init_est_active_local_k', summary['tau_init_est_active_k']):.6f}]",
        "tau_init_calibration_sample["
        f"batch={sample['batch']} token_sampling={sample['token_sampling']} "
        f"tokens={sample['tokens']} "
        f"neurons_qk={sample['neurons_qk']} "
        f"neurons_v={sample['neurons_v']} "
        f"neurons_rst={sample['neurons_rst']}]",
    ]
    for pool in ('qk', 'v', 'rst'):
        diag = summary.get('tau_calibration', {}).get(pool)
        if diag:
            lines.append(
                "tau_calib: "
                f"pool={pool} "
                f"target_pool={diag['target_pool']:.6f} "
                f"candidate_frac={diag['candidate_frac']:.6f} "
                f"target_local={diag['target_local']:.6f} "
                f"tau={diag['tau']:.6f} "
                f"rho_cand_mean={diag['rho_cand_mean']:+.6f} "
                f"rho_cand_std={diag['rho_cand_std']:.6f} "
                f"rho_cand_p50={diag['rho_cand_p50']:+.6f} "
                f"rho_cand_p90={diag['rho_cand_p90']:+.6f} "
                f"rho_cand_p99={diag['rho_cand_p99']:+.6f} "
                f"cand={diag['candidate_count']:.0f} "
                f"valid={diag['valid_candidate_count']:.1f} "
                f"full={diag['full_pool_size']:.0f}")
    return lines


# ============================================================
# GCS / file I/O helpers
# ============================================================

_GCS_FS_CACHE = None


def _get_gcs_fs():
    """Cached gcsfs.GCSFileSystem singleton.

    Avoids per-call auth + init overhead. Returns None if gcsfs is not
    installed; callers are expected to fall back to tensorflow or raise.
    """
    global _GCS_FS_CACHE
    if _GCS_FS_CACHE is None:
        try:
            import gcsfs
            _GCS_FS_CACHE = gcsfs.GCSFileSystem()
        except ImportError:
            return None
    return _GCS_FS_CACHE


def _is_gcs(path):
    return str(path).startswith("gs://")


def _open_file(path, mode="rb"):
    """Open a file for read/write, supporting GCS paths."""
    path_str = str(path)
    if _is_gcs(path_str):
        fs = _get_gcs_fs()
        if fs is not None:
            return fs.open(path_str, mode)
        try:
            import tensorflow as tf
            return tf.io.gfile.GFile(path_str, mode)
        except ImportError:
            raise ImportError(
                "GCS support requires 'gcsfs' or 'tensorflow'. "
                "Install with: pip install gcsfs"
            )
    else:
        p = Path(path_str)
        if "w" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, mode)


def _file_exists(path):
    """Check if a file exists (local or GCS)."""
    path_str = str(path)
    if _is_gcs(path_str):
        fs = _get_gcs_fs()
        if fs is not None:
            return fs.exists(path_str)
        try:
            import tensorflow as tf
            return tf.io.gfile.exists(path_str)
        except ImportError:
            raise ImportError(
                f"Cannot check GCS path {path_str}: "
                f"neither gcsfs nor tensorflow available.")
    return Path(path_str).exists()


def _list_files(directory, pattern="*"):
    """List files in a directory (local or GCS), sorted by name."""
    dir_str = str(directory)

    def _sort_key(path):
        name = path.rsplit('/', 1)[-1] if '/' in path else path
        return name

    if _is_gcs(dir_str):
        fs = _get_gcs_fs()
        if fs is not None:
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = fs.glob(dir_str + pattern)
            return sorted(["gs://" + f for f in files], key=_sort_key)
        try:
            import tensorflow as tf
            if not dir_str.endswith("/"):
                dir_str += "/"
            files = tf.io.gfile.glob(dir_str + pattern)
            return sorted(files, key=_sort_key)
        except ImportError:
            raise ImportError(
                f"Cannot list GCS path {dir_str}: "
                f"neither gcsfs nor tensorflow available.")
    return sorted((str(f) for f in Path(dir_str).glob(pattern)), key=_sort_key)


def _makedirs(path):
    """Create directory (local only; GCS doesn't need explicit mkdir)."""
    if not _is_gcs(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _orbax_root_is_listable(path_str):
    """Return True only when Orbax/epath can list the checkpoint root."""
    try:
        from etils import epath
        list(epath.Path(str(path_str).rstrip('/')).iterdir())
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _gcsfs_object_path(path_str):
    """Convert gs://bucket/name to the bucket/name form accepted by gcsfs."""
    path_str = str(path_str)
    if path_str.startswith('gs://'):
        return path_str[len('gs://'):]
    return path_str


def _materialize_gcs_directory_object(path_str):
    """Create the GCS directory placeholder object path/, not path/.marker.

    Orbax scans direct children of the checkpoint root as step entries, so this
    must not create arbitrary files such as .orbax_root, .keep, or _marker
    under the checkpoint root.
    """
    path_str = str(path_str).rstrip('/')
    dir_object_gs = path_str + '/'

    # Prefer gcsfs because the Orbax/etils failure path commonly goes through
    # gcsfs on TPU runs.  Use the directory object itself, not a child marker.
    fs = _get_gcs_fs()
    if fs is not None:
        object_path = _gcsfs_object_path(dir_object_gs)
        with fs.open(object_path, 'wb') as f:
            f.write(b'')
        return

    import tensorflow as tf
    with tf.io.gfile.GFile(dir_object_gs, 'wb') as f:
        f.write(b'')


def _ensure_orbax_checkpoint_root(path):
    """Ensure an Orbax checkpoint root is visible/listable before init.

    Orbax may call CheckpointManager._load_checkpoint_infos() during manager
    construction, which lists the checkpoint root before the first save.  On
    GCS, mkdir can succeed without making a fresh empty prefix listable through
    epath/gcsfs.  For GCS roots, verify listability after mkdir and, if needed,
    materialize the directory object itself (path/) without creating any child
    marker file under the checkpoint root.
    """
    path_str = str(path).rstrip('/')
    last_exc = None

    if _is_gcs(path_str):
        try:
            from etils import epath
            epath.Path(path_str).mkdir(parents=True, exist_ok=True)
        except Exception as epath_exc:
            last_exc = epath_exc

        try:
            import tensorflow as tf
            tf.io.gfile.makedirs(path_str)
        except Exception as tf_exc:
            last_exc = tf_exc

        if _orbax_root_is_listable(path_str):
            return

        try:
            _materialize_gcs_directory_object(path_str)
        except Exception as mat_exc:
            last_exc = mat_exc

        if _orbax_root_is_listable(path_str):
            return

        raise RuntimeError(
            f"Failed to create/list Orbax checkpoint root on GCS: {path_str}"
        ) from last_exc

    Path(path_str).mkdir(parents=True, exist_ok=True)

def _ensure_orbax_dir(path):
    """Backward-compatible alias for Orbax checkpoint root creation."""
    return _ensure_orbax_checkpoint_root(path)


def _join_path(base, *parts):
    base_str = str(base)
    if _is_gcs(base_str):
        out = base_str.rstrip('/')
        for part in parts:
            out += '/' + str(part).strip('/\\')
        return out
    return str(Path(base_str, *map(str, parts)))


def _path_name(path):
    return str(path).rstrip('/\\').replace('\\', '/').rsplit('/', 1)[-1]


def _path_parent(path):
    path_str = str(path).rstrip('/\\').replace('\\', '/')
    if '/' not in path_str:
        return '.'
    return path_str.rsplit('/', 1)[0]


# ============================================================
# Parameter count
# ============================================================

def count_parameters(params):
    """Count total parameters in a pytree."""
    return sum(x.size for x in jax.tree.leaves(params))


# ============================================================
# Orthogonality + diversity loss (inline for jit)
# ============================================================

def compute_orthogonality_loss(params, rank, knowledge_rank, n_feature_qk, n_restore_qk):
    """Compute orthogonality loss from shared neuron params.

    Matches the model's 6-group computation:
      f_neurons = [feature_qk ; feature_v]  -> split at n_feature_qk
      r_neurons = [restore_qk ; restore_v]  -> split at n_restore_qk
      feature_know, restore_know             -> separate params
    """
    sn = params['shared_neurons']
    I_rank = jnp.eye(rank)[jnp.newaxis]
    I_know = jnp.eye(knowledge_rank)[jnp.newaxis]

    f_neurons = sn['f_neurons']
    r_neurons = sn['r_neurons']
    feature_know = sn['feature_know']
    restore_know = sn['restore_know']

    # Split f_neurons into feature_qk [N_fqk, D, R] and feature_v [N_fv, D, R]
    W_fqk = f_neurons[:n_feature_qk]
    W_fv = f_neurons[n_feature_qk:]
    WtW_fqk = jnp.matmul(W_fqk.transpose(0, 2, 1), W_fqk)
    loss_fqk = ((WtW_fqk - I_rank) ** 2).mean()
    WtW_fv = jnp.matmul(W_fv.transpose(0, 2, 1), W_fv)
    loss_fv = ((WtW_fv - I_rank) ** 2).mean()

    # Split r_neurons into restore_qk [N_rqk, R, D] and restore_v [N_rv, R, D]
    W_rqk = r_neurons[:n_restore_qk]
    W_rv = r_neurons[n_restore_qk:]
    WWt_rqk = jnp.matmul(W_rqk, W_rqk.transpose(0, 2, 1))
    loss_rqk = ((WWt_rqk - I_rank) ** 2).mean()
    WWt_rv = jnp.matmul(W_rv, W_rv.transpose(0, 2, 1))
    loss_rv = ((WWt_rv - I_rank) ** 2).mean()

    WtW_fk = jnp.matmul(feature_know.transpose(0, 2, 1), feature_know)
    loss_fk = ((WtW_fk - I_know) ** 2).mean()

    WWt_rk = jnp.matmul(restore_know, restore_know.transpose(0, 2, 1))
    loss_rk = ((WWt_rk - I_know) ** 2).mean()

    return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fk + loss_rk) / 6


def compute_knowledge_diversity_loss(params):
    """Compute knowledge diversity loss from shared neuron params."""
    sn = params['shared_neurons']

    feat_know = sn['feature_know']
    feat_flat = feat_know.reshape(feat_know.shape[0], -1)
    feat_norm = feat_flat / (jnp.linalg.norm(feat_flat, axis=-1, keepdims=True) + 1e-8)
    feat_sim = jnp.matmul(feat_norm, feat_norm.T)
    mask_f = ~jnp.eye(feat_sim.shape[0], dtype=jnp.bool_)
    feat_loss = jnp.abs(feat_sim * mask_f).sum() / mask_f.sum()

    rest_know = sn['restore_know']
    rest_flat = rest_know.reshape(rest_know.shape[0], -1)
    rest_norm = rest_flat / (jnp.linalg.norm(rest_flat, axis=-1, keepdims=True) + 1e-8)
    rest_sim = jnp.matmul(rest_norm, rest_norm.T)
    mask_r = ~jnp.eye(rest_sim.shape[0], dtype=jnp.bool_)
    rest_loss = jnp.abs(rest_sim * mask_r).sum() / mask_r.sum()

    return (feat_loss + rest_loss) / 2


def compute_spatial_diversity_loss(params):
    """Compute neuron diversity loss for rank-1 spatial/SRW neurons.

    Penalizes high cosine similarity between neurons in each pool.
    Replaces orthogonality + knowledge diversity for spatial-r1.
    For large pools (>4096), uses deterministic strided sampling to avoid O(N^2).
    Supports current DAWN-SRW pool param names.
    """
    pool = params['neuron_pool']

    def _pool_div(neurons, max_sample=4096):
        N = neurons.shape[0]
        if N > max_sample:
            stride = max(1, N // max_sample)
            neurons = neurons[::stride][:max_sample]
        n = neurons / (jnp.linalg.norm(neurons, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.matmul(n, n.T)
        mask = ~jnp.eye(sim.shape[0], dtype=jnp.bool_)
        denom = mask.sum()
        return jnp.where(
            denom > 0,
            jnp.abs(sim * mask).sum() / denom,
            jnp.float32(0.0),
        )

    def _get_pool_arrays(pool):
        """Return list of neuron arrays from current v4164 pool params."""
        arrays = []

        # Current v4164 DAWN-SRW pool names.
        for prefix in ('attn_qk', 'attn_v', 'rst'):
            for suffix in ('emb', 'read', 'write'):
                key = f'{prefix}_{suffix}'
                if key in pool:
                    arrays.append(pool[key])
        if arrays:
            return arrays
        return arrays

    pool_arrays = _get_pool_arrays(pool)
    if not pool_arrays:
        # Some experimental variants may not expose a recognized neuron pool.
        # Keep compile/OOM checks from misreporting a naming mismatch as OOM.
        return jnp.float32(0.0)
    return sum(_pool_div(a) for a in pool_arrays) / len(pool_arrays)


# ============================================================
# Train / eval steps (pmap for multi-device)
# ============================================================

def _model_accepts_analysis(model):
    """Return True if model.__call__ accepts an `analysis` kwarg.

    v4164 accepts it and routes the full-stats forward. there raises, so we must gate it.
    """
    import inspect as _inspect
    try:
        return 'analysis' in _inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _model_accepts_soft_gate_schedule(model):
    """Return True if model.__call__ accepts soft-gate schedule kwargs."""
    import inspect as _inspect
    try:
        params = _inspect.signature(model.__call__).parameters
        return 'soft_gate_temperature' in params
    except (TypeError, ValueError):
        return False


def _model_accepts_soft_gate_t_final(model):
    """Return True if model.__call__ accepts projected final-T diagnostics."""
    import inspect as _inspect
    try:
        return 'soft_gate_t_final' in _inspect.signature(
            model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _model_accepts_execution_prune_eps(model):
    """Return True if model.__call__ accepts eval-time execution pruning."""
    import inspect as _inspect
    try:
        return 'execution_prune_eps' in _inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _model_accepts_soft_gate_boundary_power(model):
    """Return True if model.__call__ accepts boundary-power kwargs."""
    import inspect as _inspect
    try:
        return 'soft_gate_boundary_power' in _inspect.signature(
            model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _model_accepts_admission_den_power(model):
    """Return True if model.__call__ accepts runtime admission_den_power."""
    import inspect as _inspect
    try:
        return 'admission_den_power' in _inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _model_accepts_minimal_train(model):
    """Return True if model.__call__ accepts the minimal train path switch."""
    import inspect as _inspect
    try:
        return 'minimal_train' in _inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _model_accepts_training_tokens(model):
    """Return True if model.__call__ accepts consumed-token scheduling."""
    import inspect as _inspect
    try:
        return 'training_tokens' in _inspect.signature(
            model.__call__).parameters
    except (TypeError, ValueError):
        return False


def _scalar0(x):
    return jnp.asarray(x, dtype=jnp.float32).reshape(())


POOL_SCHEDULE_NAMES = ('qk', 'v', 'rst')
SOFT_GATE_T_SCHEDULE_NAMES = (
    'constant', 'linear', 'cosine',
    'log', 'log_linear', 'log_power', 'log_gompertz',
    'developmental_band')
DEVELOPMENTAL_BAND_TEMP_KEYS = ('sort', 'band', 'mid', 'late', 'final')
DEVELOPMENTAL_BAND_FRAC_KEYS = (
    'sort_end_frac', 'band_reach_frac',
    'formation_end_frac', 'sharpen_end_frac')
DEVELOPMENTAL_BAND_REQUIRED_KEYS = (
    DEVELOPMENTAL_BAND_TEMP_KEYS + DEVELOPMENTAL_BAND_FRAC_KEYS)


def _training_soft_gate_pool_schedules(
        tcfg, soft_gate_t_start, soft_gate_t_final,
        soft_gate_t_hold_frac, soft_gate_t_anneal_end_frac,
        soft_gate_schedule, soft_gate_t_power,
        soft_gate_t_gompertz_center, soft_gate_t_gompertz_steepness):
    out = {}
    for pool in POOL_SCHEDULE_NAMES:
        prefix = f'soft_gate_t_{pool}'
        cfg = {
            'start': float(tcfg.get(f'{prefix}_start', soft_gate_t_start)),
            'final': float(tcfg.get(f'{prefix}_final', soft_gate_t_final)),
            'hold_frac': float(tcfg.get(
                f'{prefix}_hold_frac', soft_gate_t_hold_frac)),
            'anneal_end_frac': float(tcfg.get(
                f'{prefix}_anneal_end_frac', soft_gate_t_anneal_end_frac)),
            'schedule': str(tcfg.get(f'{prefix}_schedule', soft_gate_schedule)),
            'power': float(tcfg.get(f'{prefix}_power', soft_gate_t_power)),
            'gompertz_center': float(tcfg.get(
                f'{prefix}_gompertz_center',
                soft_gate_t_gompertz_center)),
            'gompertz_steepness': float(tcfg.get(
                f'{prefix}_gompertz_steepness',
                soft_gate_t_gompertz_steepness)),
        }
        cfg['_final_pool_present'] = f'{prefix}_final' in tcfg
        cfg['_final_present'] = (
            cfg['_final_pool_present'] or 'soft_gate_t_final' in tcfg)

        dev_defaults = {
            'sort': soft_gate_t_start,
            'band': soft_gate_t_start,
            'mid': soft_gate_t_final,
            'late': soft_gate_t_final,
            'sort_end_frac': soft_gate_t_hold_frac,
            'band_reach_frac': soft_gate_t_hold_frac,
            'formation_end_frac': soft_gate_t_anneal_end_frac,
            'sharpen_end_frac': soft_gate_t_anneal_end_frac,
            'formation_power': 1.5,
            'sharpen_power': 1.0,
        }
        for key, default in dev_defaults.items():
            pool_key = f'{prefix}_{key}'
            global_key = f'soft_gate_t_{key}'
            if pool_key in tcfg:
                raw = tcfg[pool_key]
            else:
                raw = tcfg.get(global_key, default)
            cfg[key] = float(raw)
            cfg[f'_{key}_pool_present'] = pool_key in tcfg
            cfg[f'_{key}_present'] = (
                cfg[f'_{key}_pool_present'] or global_key in tcfg)
        out[pool] = cfg
    return out


def _coerce_pool_schedule_configs(pool_schedules, defaults):
    pool_schedules = pool_schedules or {}
    out = {}
    for pool in POOL_SCHEDULE_NAMES:
        src = pool_schedules.get(pool, {})
        out[pool] = {
            'start': jnp.float32(src.get('start', defaults['start'])),
            'final': jnp.float32(src.get('final', defaults['final'])),
            'hold_frac': jnp.float32(src.get(
                'hold_frac', defaults['hold_frac'])),
            'anneal_end_frac': jnp.float32(src.get(
                'anneal_end_frac', defaults['anneal_end_frac'])),
            'schedule': str(src.get('schedule', defaults['schedule'])).lower(),
            'power': jnp.float32(src.get('power', defaults.get('power', 1.0))),
            'gompertz_center': jnp.float32(src.get(
                'gompertz_center', defaults.get('gompertz_center', 0.25))),
            'gompertz_steepness': jnp.float32(src.get(
                'gompertz_steepness', defaults.get(
                    'gompertz_steepness', 8.0))),
            'sort': jnp.float32(src.get(
                'sort', defaults.get('sort', defaults['start']))),
            'band': jnp.float32(src.get(
                'band', defaults.get('band', defaults['start']))),
            'mid': jnp.float32(src.get(
                'mid', defaults.get('mid', defaults['final']))),
            'late': jnp.float32(src.get(
                'late', defaults.get('late', defaults['final']))),
            'sort_end_frac': jnp.float32(src.get(
                'sort_end_frac', defaults.get(
                    'sort_end_frac', defaults['hold_frac']))),
            'band_reach_frac': jnp.float32(src.get(
                'band_reach_frac', defaults.get(
                    'band_reach_frac', defaults['hold_frac']))),
            'formation_end_frac': jnp.float32(src.get(
                'formation_end_frac', defaults.get(
                    'formation_end_frac', defaults['anneal_end_frac']))),
            'sharpen_end_frac': jnp.float32(src.get(
                'sharpen_end_frac', defaults.get(
                    'sharpen_end_frac', defaults['anneal_end_frac']))),
            'formation_power': jnp.float32(src.get(
                'formation_power', defaults.get('formation_power', 1.5))),
            'sharpen_power': jnp.float32(src.get(
                'sharpen_power', defaults.get('sharpen_power', 1.0))),
        }
    return out


def _scheduled_from_config(step, total_steps, cfg):
    return scheduled_value_by_frac(
        step, total_steps,
        cfg['start'], cfg['final'],
        cfg['hold_frac'], cfg['anneal_end_frac'],
        cfg['schedule'], cfg['power'],
        cfg['gompertz_center'], cfg['gompertz_steepness'],
        cfg['sort'], cfg['band'], cfg['mid'], cfg['late'],
        cfg['sort_end_frac'], cfg['band_reach_frac'],
        cfg['formation_end_frac'], cfg['sharpen_end_frac'],
        cfg['formation_power'], cfg['sharpen_power'])


def _flatten_soft_gate_pool_schedules(pool_schedules):
    out = {}
    for pool, cfg in pool_schedules.items():
        prefix = f'soft_gate_t_{pool}'
        out.update({
            f'{prefix}_start': cfg['start'],
            f'{prefix}_final': cfg['final'],
            f'{prefix}_hold_frac': cfg['hold_frac'],
            f'{prefix}_anneal_end_frac': cfg['anneal_end_frac'],
            f'{prefix}_schedule': cfg['schedule'],
            f'{prefix}_power': cfg['power'],
            f'{prefix}_gompertz_center': cfg['gompertz_center'],
            f'{prefix}_gompertz_steepness': cfg['gompertz_steepness'],
            f'{prefix}_sort': cfg['sort'],
            f'{prefix}_band': cfg['band'],
            f'{prefix}_mid': cfg['mid'],
            f'{prefix}_late': cfg['late'],
            f'{prefix}_sort_end_frac': cfg['sort_end_frac'],
            f'{prefix}_band_reach_frac': cfg['band_reach_frac'],
            f'{prefix}_formation_end_frac': cfg['formation_end_frac'],
            f'{prefix}_sharpen_end_frac': cfg['sharpen_end_frac'],
            f'{prefix}_formation_power': cfg['formation_power'],
            f'{prefix}_sharpen_power': cfg['sharpen_power'],
        })
    return out


def scheduled_value_by_frac(step, total_steps, start, final, hold_frac,
                            end_frac, schedule='cosine', power=1.0,
                            gompertz_center=0.25,
                            gompertz_steepness=8.0,
                            sort=None, band=None, mid=None, late=None,
                            sort_end_frac=0.0, band_reach_frac=0.0,
                            formation_end_frac=1.0,
                            sharpen_end_frac=1.0,
                            formation_power=1.5,
                            sharpen_power=1.0):
    """Piecewise hold/anneal/hold scalar schedule by training fraction."""
    step_f = jnp.asarray(step, dtype=jnp.float32)
    total_f = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    frac = jnp.clip(step_f / total_f, 0.0, 1.0)
    hold = jnp.asarray(hold_frac, dtype=jnp.float32)
    end = jnp.maximum(jnp.asarray(end_frac, dtype=jnp.float32),
                      hold + 1.0e-6)
    progress = jnp.clip((frac - hold) / (end - hold), 0.0, 1.0)
    schedule_name = str(schedule).lower()
    start_f = jnp.asarray(start, dtype=jnp.float32)
    final_f = jnp.asarray(final, dtype=jnp.float32)
    if schedule_name == 'constant':
        mix = jnp.float32(0.0)
    elif schedule_name == 'linear':
        mix = progress
    elif schedule_name == 'cosine':
        mix = 0.5 - 0.5 * jnp.cos(jnp.pi * progress)
    elif schedule_name in ('log', 'log_linear'):
        mix = progress
        log_start = jnp.maximum(start_f, jnp.float32(1.0e-12))
        log_final = jnp.maximum(final_f, jnp.float32(1.0e-12))
        log_val = (1.0 - mix) * jnp.log(log_start) + mix * jnp.log(log_final)
        val = jnp.exp(log_val)
        return jnp.where(frac < hold, start_f,
                         jnp.where(frac >= end, final_f, val))
    elif schedule_name == 'log_power':
        power_f = jnp.asarray(power, dtype=jnp.float32)
        mix = 1.0 - jnp.power(1.0 - progress, power_f)
        log_start = jnp.maximum(start_f, jnp.float32(1.0e-12))
        log_final = jnp.maximum(final_f, jnp.float32(1.0e-12))
        log_val = (1.0 - mix) * jnp.log(log_start) + mix * jnp.log(log_final)
        val = jnp.exp(log_val)
        return jnp.where(frac < hold, start_f,
                         jnp.where(frac >= end, final_f, val))
    elif schedule_name == 'log_gompertz':
        center_f = jnp.asarray(gompertz_center, dtype=jnp.float32)
        steep_f = jnp.asarray(gompertz_steepness, dtype=jnp.float32)

        def _gom(x):
            return jnp.exp(-jnp.exp(-steep_f * (x - center_f)))

        g0 = _gom(jnp.float32(0.0))
        g1 = _gom(jnp.float32(1.0))
        gp = _gom(progress)
        mix = (gp - g0) / jnp.maximum(g1 - g0, 1.0e-8)
        mix = jnp.clip(mix, 0.0, 1.0)

        log_start = jnp.maximum(start_f, jnp.float32(1.0e-12))
        log_final = jnp.maximum(final_f, jnp.float32(1.0e-12))
        log_val = (1.0 - mix) * jnp.log(log_start) + mix * jnp.log(log_final)
        val = jnp.exp(log_val)
        return jnp.where(frac < hold, start_f,
                         jnp.where(frac >= end, final_f, val))
    elif schedule_name == 'developmental_band':
        eps = jnp.float32(1.0e-6)

        def _as_f32(x, fallback):
            return jnp.asarray(fallback if x is None else x,
                               dtype=jnp.float32)

        def _smoothstep(u):
            u = jnp.clip(u, 0.0, 1.0)
            return u * u * (3.0 - 2.0 * u)

        def _geom_interp(a, b, u):
            log_a = jnp.log(jnp.maximum(a, jnp.float32(1.0e-12)))
            log_b = jnp.log(jnp.maximum(b, jnp.float32(1.0e-12)))
            return jnp.exp((1.0 - u) * log_a + u * log_b)

        sort_f = _as_f32(sort, start_f)
        band_f = _as_f32(band, start_f)
        mid_f = _as_f32(mid, final_f)
        late_f = _as_f32(late, final_f)
        sort_end = jnp.asarray(sort_end_frac, dtype=jnp.float32)
        band_reach = jnp.asarray(band_reach_frac, dtype=jnp.float32)
        formation_end = jnp.asarray(formation_end_frac, dtype=jnp.float32)
        sharpen_end = jnp.asarray(sharpen_end_frac, dtype=jnp.float32)
        formation_power_f = jnp.asarray(formation_power, dtype=jnp.float32)
        sharpen_power_f = jnp.asarray(sharpen_power, dtype=jnp.float32)

        u_contract = jnp.clip(
            (frac - sort_end) / jnp.maximum(band_reach - sort_end, eps),
            0.0, 1.0)
        contract_val = _geom_interp(
            sort_f, band_f, _smoothstep(u_contract))

        u_form = jnp.clip(
            ((frac - band_reach)
             / jnp.maximum(formation_end - band_reach, eps)),
            0.0, 1.0)
        form_val = _geom_interp(
            band_f, mid_f, jnp.power(u_form, formation_power_f))

        u_sharp = jnp.clip(
            ((frac - formation_end)
             / jnp.maximum(sharpen_end - formation_end, eps)),
            0.0, 1.0)
        sharp_val = _geom_interp(
            mid_f, late_f,
            _smoothstep(jnp.power(u_sharp, sharpen_power_f)))

        u_final = jnp.clip(
            (frac - sharpen_end) / jnp.maximum(1.0 - sharpen_end, eps),
            0.0, 1.0)
        final_val = _geom_interp(late_f, final_f, _smoothstep(u_final))

        return jnp.where(
            frac < sort_end, sort_f,
            jnp.where(
                frac < band_reach, contract_val,
                jnp.where(
                    frac < formation_end, form_val,
                    jnp.where(frac < sharpen_end, sharp_val, final_val))))
    else:
        raise ValueError(
            f"Unsupported schedule={schedule!r}; expected cosine, linear, "
            "constant, log, log_linear, log_power, log_gompertz, or "
            "developmental_band.")
    val = start_f + (final_f - start_f) * mix
    return jnp.where(frac < hold, start_f,
                     jnp.where(frac >= end, final_f, val))


_scheduled_scalar = scheduled_value_by_frac


def scheduled_boundary_power_by_frac(
        step, total_steps, enabled, start, mid, final,
        mid_frac, final_frac, start_frac=0.0):
    step_f = jnp.asarray(step, dtype=jnp.float32)
    total_f = jnp.maximum(jnp.asarray(total_steps, dtype=jnp.float32), 1.0)
    frac = jnp.clip(step_f / total_f, 0.0, 1.0)
    start_f = jnp.asarray(start, dtype=jnp.float32)
    mid_f = jnp.asarray(mid, dtype=jnp.float32)
    final_f = jnp.asarray(final, dtype=jnp.float32)
    start_frac_f = jnp.asarray(start_frac, dtype=jnp.float32)
    mid_frac_f = jnp.asarray(mid_frac, dtype=jnp.float32)
    final_frac_f = jnp.asarray(final_frac, dtype=jnp.float32)
    eps = jnp.float32(1.0e-6)
    u_mid = jnp.clip(
        (frac - start_frac_f) / jnp.maximum(mid_frac_f - start_frac_f, eps),
        0.0, 1.0)
    u_final = jnp.clip(
        (frac - mid_frac_f) / jnp.maximum(final_frac_f - mid_frac_f, eps),
        0.0, 1.0)
    mid_val = start_f + (mid_f - start_f) * jnp.square(u_mid)
    final_val = mid_f + (final_f - mid_f) * u_final
    scheduled = jnp.where(
        frac < start_frac_f, start_f,
        jnp.where(frac < mid_frac_f, mid_val,
            jnp.where(frac < final_frac_f, final_val, final_f)))
    return jnp.where(jnp.asarray(enabled), scheduled, jnp.float32(2.0))


def _soft_gate_schedule_expected_msg():
    return (
        "expected constant, linear, cosine, log, log_linear, log_power, "
        "log_gompertz, or developmental_band.")


def _validate_soft_gate_schedule_config(
        label, cfg, require_pool_specific_devband_fields=False):
    schedule_name = str(cfg['schedule']).lower()
    if schedule_name not in SOFT_GATE_T_SCHEDULE_NAMES:
        raise ValueError(
            f"Unsupported {label}_schedule={cfg['schedule']!r}; "
            f"{_soft_gate_schedule_expected_msg()}")

    if schedule_name == 'developmental_band':
        missing = []
        for key in DEVELOPMENTAL_BAND_REQUIRED_KEYS:
            present_key = (
                f'_{key}_pool_present'
                if require_pool_specific_devband_fields
                else f'_{key}_present')
            if not cfg.get(present_key, False):
                missing.append(f'{label}_{key}')
        if missing:
            raise ValueError(
                f"{label}_schedule=developmental_band requires: "
                f"{', '.join(missing)}")

        for key in DEVELOPMENTAL_BAND_TEMP_KEYS:
            if cfg[key] <= 0.0:
                raise ValueError(
                    f"{label}_{key} must be > 0 for developmental_band, "
                    f"got {cfg[key]}")
        sort_end = cfg['sort_end_frac']
        band_reach = cfg['band_reach_frac']
        formation_end = cfg['formation_end_frac']
        sharpen_end = cfg['sharpen_end_frac']
        if not (0.0 <= sort_end < band_reach
                <= formation_end < sharpen_end <= 1.0):
            raise ValueError(
                f"{label} developmental_band fractions must satisfy "
                "0 <= sort_end_frac < band_reach_frac <= "
                "formation_end_frac < sharpen_end_frac <= 1, got "
                f"{sort_end}, {band_reach}, {formation_end}, {sharpen_end}")
        if cfg['formation_power'] <= 0.0:
            raise ValueError(
                f"{label}_formation_power must be > 0, "
                f"got {cfg['formation_power']}")
        if cfg['sharpen_power'] <= 0.0:
            raise ValueError(
                f"{label}_sharpen_power must be > 0, "
                f"got {cfg['sharpen_power']}")
        return

    if cfg['start'] <= 0.0:
        raise ValueError(
            f"{label}_start must be > 0, got {cfg['start']}")
    if cfg['final'] <= 0.0:
        raise ValueError(
            f"{label}_final must be > 0, got {cfg['final']}")
    if cfg['anneal_end_frac'] <= cfg['hold_frac']:
        raise ValueError(
            f"{label}_anneal_end_frac must be > {label}_hold_frac, got "
            f"{cfg['anneal_end_frac']} <= {cfg['hold_frac']}")
    if schedule_name == 'log_power' and cfg['power'] <= 0.0:
        raise ValueError(
            f"{label}_power must be > 0, got {cfg['power']}")
    if schedule_name == 'log_gompertz':
        if not (0.0 < cfg['gompertz_center'] < 1.0):
            raise ValueError(
                f"{label}_gompertz_center must be > 0 and < 1, "
                f"got {cfg['gompertz_center']}")
        if cfg['gompertz_steepness'] <= 0.0:
            raise ValueError(
                f"{label}_gompertz_steepness must be > 0, "
                f"got {cfg['gompertz_steepness']}")


def _global_norm_array(x):
    x = jax.lax.stop_gradient(jnp.asarray(x, dtype=jnp.float32))
    return jnp.sqrt(jnp.sum(jnp.square(x)) + 1e-12)


def _row_norm_stats(x, prefix, full=False):
    n = jax.lax.stop_gradient(
        jnp.linalg.norm(jnp.asarray(x, dtype=jnp.float32), axis=-1))
    out = {
        f'{prefix}_mean': n.mean(),
        f'{prefix}_std': n.std(),
        f'{prefix}_max': n.max(),
    }
    if full:
        out.update({
            f'{prefix}_min': n.min(),
            f'{prefix}_p50': jnp.quantile(n, 0.50),
            f'{prefix}_p95': jnp.quantile(n, 0.95),
            f'{prefix}_p99': jnp.quantile(n, 0.99),
        })
    return out


def _op_gain_stats(read, write, prefix, full=False):
    r = jax.lax.stop_gradient(
        jnp.linalg.norm(jnp.asarray(read, dtype=jnp.float32), axis=-1))
    w = jax.lax.stop_gradient(
        jnp.linalg.norm(jnp.asarray(write, dtype=jnp.float32), axis=-1))
    g = r * w
    out = {
        f'{prefix}_mean': g.mean(),
        f'{prefix}_std': g.std(),
        f'{prefix}_max': g.max(),
    }
    if full:
        out.update({
            f'{prefix}_min': g.min(),
            f'{prefix}_p50': jnp.quantile(g, 0.50),
            f'{prefix}_p95': jnp.quantile(g, 0.95),
            f'{prefix}_p99': jnp.quantile(g, 0.99),
        })
    return out


def _depth_scale_enabled(model=None, model_cfg=None):
    """PureCore always reports fixed depth-scaled pool outputs."""
    return True


def _depth_pool_scales_for(model=None, model_cfg=None):
    if model is not None:
        d_model = getattr(model, 'd_model')
        n_layers = getattr(model, 'n_layers')
    else:
        d_model = model_cfg['d_model']
        n_layers = model_cfg['n_layers']
    dm = jnp.asarray(d_model, dtype=jnp.float32)
    nl = jnp.asarray(n_layers, dtype=jnp.float32)
    return (
        jax.lax.stop_gradient(jnp.sqrt(dm / nl)),
        jax.lax.stop_gradient(jnp.sqrt(dm / nl)),
        jax.lax.stop_gradient(jnp.sqrt(dm / nl)),
    )


def _pool_param_diagnostics(params, full=False, model=None, model_cfg=None):
    """Observational pool norm/gain diagnostics; never feeds loss."""
    pool = params.get('neuron_pool', {})
    out = {}
    fixed_scales = _depth_pool_scales_for(model, model_cfg)
    model_version = getattr(model, '__version__', None)
    if model_version is None and model_cfg is not None:
        model_version = model_cfg.get('model_version')

    def _diag_op_key(read, write, read_proj, write_proj, eps=1.0e-6):
        def _unit(x):
            x = jnp.asarray(x, dtype=jnp.float32)
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        r_key = _unit(read) @ read_proj
        w_key = _unit(write) @ write_proj
        return _unit(_unit(r_key) * _unit(w_key))

    specs = (
        ('attn_qk', 'attn_qk_emb', 'attn_qk_read', 'attn_qk_write', 'attn_qk_scale'),
        ('attn_v', 'attn_v_emb', 'attn_v_read', 'attn_v_write', 'attn_v_scale'),
        ('rst', 'rst_emb', 'rst_read', 'rst_write', 'rst_scale'),
    )

    def _flat_pool_tensor(prefix, kind):
        key = f'{prefix}_{kind}'
        if key in pool:
            return pool[key]
        parts = []
        for scope in ('global', 'stage', 'local'):
            part_key = f'{prefix}_{kind}_{scope}'
            if part_key in pool:
                part = pool[part_key]
                parts.append(part.reshape((-1, part.shape[-1])))
        if not parts:
            return None
        return jnp.concatenate(parts, axis=0)

    for i, (name, emb_key, read_key, write_key, scale_key) in enumerate(specs):
        read = _flat_pool_tensor(name, 'read')
        write = _flat_pool_tensor(name, 'write')
        if (_is_active_srw_version(model_version)
                and _is_rw_key_srw_version(model_version)):
            op_read_key = f'{name}_op_read_proj'
            op_write_key = f'{name}_op_write_proj'
            if name == 'attn_qk':
                op_read_key = 'attn_qk_op_read_proj'
                op_write_key = 'attn_qk_op_write_proj'
            if name == 'attn_v':
                op_read_key = 'attn_v_op_read_proj'
                op_write_key = 'attn_v_op_write_proj'
            if (read is not None and write is not None
                    and op_read_key in pool and op_write_key in pool):
                op_key = _diag_op_key(
                    read, write,
                    pool[op_read_key], pool[op_write_key])
                out.update(_row_norm_stats(
                    op_key, f'{name}_op_key_norm', full))
        elif emb_key in pool:
            out.update(_row_norm_stats(pool[emb_key], f'{name}_emb_norm', full))
        if read is not None:
            out.update(_row_norm_stats(read, f'{name}_read_norm', full))
        if write is not None:
            out.update(_row_norm_stats(write, f'{name}_write_norm', full))
        if read is not None and write is not None:
            out.update(_op_gain_stats(read, write, f'{name}_op_gain', full))
        if scale_key in pool:
            out[f'{name}_pool_scale'] = fixed_scales[i]
            if full:
                out[f'{name}_learned_pool_scale_unused'] = _scalar0(
                    pool[scale_key])
    return out


def _pool_update_diagnostics(params, grads):
    """Approximate per-group update observability: grad_norm / param_norm."""
    pool_p = params.get('neuron_pool', {})
    pool_g = grads.get('neuron_pool', {})
    out = {}
    specs = (
        ('attn_qk', 'attn_qk_emb', 'attn_qk_read', 'attn_qk_write'),
        ('attn_v', 'attn_v_emb', 'attn_v_read', 'attn_v_write'),
        ('rst', 'rst_emb', 'rst_read', 'rst_write'),
    )

    def _partition_norm(tree, prefix, kind):
        key = f'{prefix}_{kind}'
        if key in tree:
            return _global_norm_array(tree[key])
        norms_sq = jnp.float32(0.0)
        found = False
        for scope in ('global', 'stage', 'local'):
            part_key = f'{prefix}_{kind}_{scope}'
            if part_key in tree:
                n = _global_norm_array(tree[part_key])
                norms_sq = norms_sq + jnp.square(n)
                found = True
        if not found:
            return jnp.float32(0.0)
        return jnp.sqrt(norms_sq + 1e-12)

    for name, emb_key, read_key, write_key in specs:
        for short, key in (('emb', emb_key), ('read', read_key), ('write', write_key)):
            if short in ('read', 'write'):
                p_norm = _partition_norm(pool_p, name, short)
                g_norm = _partition_norm(pool_g, name, short)
            else:
                p_norm = (_global_norm_array(pool_p[key])
                          if key in pool_p else jnp.float32(0.0))
                g_norm = (_global_norm_array(pool_g[key])
                          if key in pool_g else jnp.float32(0.0))
            out[f'{name}_{short}_param_norm'] = p_norm
            out[f'{name}_{short}_grad_norm'] = g_norm
            out[f'{name}_{short}_grad_ratio'] = g_norm / (p_norm + 1e-8)
    op_specs = (
        ('attn_qk', 'op_read_proj', 'attn_qk_op_read_proj'),
        ('attn_qk', 'op_write_proj', 'attn_qk_op_write_proj'),
        ('attn_v', 'op_read_proj', 'attn_v_op_read_proj'),
        ('attn_v', 'op_write_proj', 'attn_v_op_write_proj'),
        ('rst', 'op_read_proj', 'rst_op_read_proj'),
        ('rst', 'op_write_proj', 'rst_op_write_proj'),
    )
    for name, short, key in op_specs:
        p_norm = (_global_norm_array(pool_p[key])
                  if key in pool_p else jnp.float32(0.0))
        g_norm = (_global_norm_array(pool_g[key])
                  if key in pool_g else jnp.float32(0.0))
        out[f'{name}_{short}_param_norm'] = p_norm
        out[f'{name}_{short}_grad_norm'] = g_norm
        out[f'{name}_{short}_grad_ratio'] = g_norm / (p_norm + 1e-8)
    return out



def create_train_step(model, optimizer, orth_weight, div_weight, lb_weight,
                      tau_reg_weight, dead_penalty_weight,
                      inactive_aux_weight, inactive_aux_asymmetry,
                      inactive_aux_weight_q,
                      inactive_aux_weight_k,
                      inactive_aux_weight_v,
                      inactive_aux_weight_rst,
                      rank, knowledge_rank, n_feature_qk, n_restore_qk,
                      weight_decay=0.0, pool_weight_decay=0.0,
                      inactive_aux_warmup_steps=5000,
                      inactive_aux_lower_bound=-0.5,
                      inactive_aux_upper_bound=2.0,
                      inactive_aux_bound_eps=1.0e-3,
                      inactive_aux_dev_mode='raw',
                      inactive_aux_ce_clip_std=2.0,
                      inactive_aux_z_clip=2.0,
                      inactive_aux_z_tanh=True,
                      inactive_aux_weighted_clip=0.0,
                      inactive_aux_normalize_by_layers=False,
                      inactive_aux_asymmetry_q=None,
                      inactive_aux_asymmetry_k=None,
                      inactive_aux_asymmetry_v=None,
                      inactive_aux_asymmetry_rst=None,
                      inactive_aux_enabled=True,
                      dead_penalty_qk_weight=0.0,
                      dead_penalty_v_weight=0.0,
                      dead_penalty_rst_weight=0.0,
                      cb1a_enabled=False,
                      cb1a_weight=0.0,
                      cb1a_challenge_weight=0.0,
                      cb1a_prune_weight=0.0,
                      cb1a_qk_weight=0.0,
                      cb1a_v_weight=0.0,
                      cb1a_rst_weight=0.0,
                      cb1a_qk_challenge_weight=None,
                      cb1a_qk_prune_weight=None,
                      cb1a_v_challenge_weight=None,
                      cb1a_v_prune_weight=None,
                      cb1a_rst_challenge_weight=None,
                      cb1a_rst_prune_weight=None,
                      cb1a_ce_mode='sigmoid_z',
                      cb1a_eps=1.0e-8,
                      dead_penalty_weighted_clip=0.0,
                      global_grad_clip=0.0,
                      tau_lr_mult=1.0,
                      tau_grad_clip=0.0,
                      router_proj_lr_mult=1.0,
                      router_proj_grad_clip=0.0,
                      router_scan_lr_mult=1.0,
                      router_scan_grad_clip=0.0,
                      route_emb_lr_mult=1.0,
                      route_emb_grad_clip=0.0,
                      op_key_lr_mult=1.0,
                      op_key_grad_clip=0.0,
                      enable_control_update_caps=False,
                      router_proj_update_ratio_cap=0.0,
                      route_emb_update_ratio_cap=0.0,
                      tau_update_abs_cap=0.0,
                      scan_update_abs_cap=0.0,
                      total_training_steps=1,
                      soft_gate_schedule_active=False,
                      soft_gate_t_start=1.5,
                      soft_gate_t_final=0.07,
                      soft_gate_t_hold_frac=0.10,
                      soft_gate_t_anneal_end_frac=0.80,
                      soft_gate_schedule='cosine',
                      soft_gate_t_power=4.0,
                      soft_gate_t_gompertz_center=0.25,
                      soft_gate_t_gompertz_steepness=8.0,
                      pool_specific_gate_t=False,
                      soft_gate_pool_schedules=None,
                      boundary_power_schedule_active=False,
                      soft_gate_boundary_power_start=3.0,
                      soft_gate_boundary_power_mid=3.15,
                      soft_gate_boundary_power_final=4.0,
                      soft_gate_boundary_power_start_frac=0.0,
                      soft_gate_boundary_power_mid_frac=0.800,
                      soft_gate_boundary_power_final_frac=0.950,
                      admission_den_power=1.0,
                      inactive_aux_start_frac=0.0,
                      inactive_aux_full_frac=0.0,
                      inactive_aux_schedule='linear',
                      sharded_fns=None, mesh=None,
                      is_baseline=False,
                      compact_train_metrics=False,
                      keep_train_layer_metrics=False,
                      tokens_per_step=0):
    """Create a jit-compiled training step. Mesh SPMD handles parallelism.

    Creates the official v4164 train step and regular metric payload.
    """
    # Shard_map'd valid-weighted global statistics reducer. Inputs are
    # sharded on 'data' (batch-parallel); psum aggregates across shards + hosts.
    # of raw CE deviations, which prevents a few easy/hard outlier tokens from
    # applying a very large auxiliary control signal to router tau.
    _global_mean_std_reducer = None
    if mesh is not None:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P('data', None),       # values [B, S-1]
                           P('data', None)),      # valid_mask [B, S-1]
                 out_specs=(P(), P()),             # mean, std scalars replicated
                 check_rep=False)
        def _mean_std_reducer_fn(values, vmask):
            vm_f = vmask.astype(jnp.float32)
            vals_f = values.astype(jnp.float32)
            local_cnt = vm_f.sum()
            local_sum = (vals_f * vm_f).sum()
            local_sq = (jnp.square(vals_f) * vm_f).sum()
            g_cnt = jax.lax.psum(local_cnt, 'data')
            g_sum = jax.lax.psum(local_sum, 'data')
            g_sq = jax.lax.psum(local_sq, 'data')
            mean = g_sum / (g_cnt + 1e-8)
            var = jnp.maximum(g_sq / (g_cnt + 1e-8) - jnp.square(mean), 0.0)
            std = jnp.sqrt(var + 1e-8)
            return mean, std
        _global_mean_std_reducer = _mean_std_reducer_fn

    _inactive_aux_weight_q = jnp.float32(
        inactive_aux_weight if inactive_aux_weight_q is None else inactive_aux_weight_q)
    _inactive_aux_weight_k = jnp.float32(
        inactive_aux_weight if inactive_aux_weight_k is None else inactive_aux_weight_k)
    _inactive_aux_weight_v = jnp.float32(
        inactive_aux_weight if inactive_aux_weight_v is None else inactive_aux_weight_v)
    _inactive_aux_weight_rst = jnp.float32(
        inactive_aux_weight if inactive_aux_weight_rst is None else inactive_aux_weight_rst)
    _asym_q = jnp.float32(
        inactive_aux_asymmetry
        if inactive_aux_asymmetry_q is None else inactive_aux_asymmetry_q)
    _asym_k = jnp.float32(
        inactive_aux_asymmetry
        if inactive_aux_asymmetry_k is None else inactive_aux_asymmetry_k)
    _asym_v = jnp.float32(
        inactive_aux_asymmetry
        if inactive_aux_asymmetry_v is None else inactive_aux_asymmetry_v)
    _asym_rst = jnp.float32(
        inactive_aux_asymmetry
        if inactive_aux_asymmetry_rst is None else inactive_aux_asymmetry_rst)
    _asym = (_asym_q + _asym_k + _asym_v + _asym_rst) / jnp.float32(4.0)
    _inactive_aux_lower = jnp.float32(inactive_aux_lower_bound)
    _inactive_aux_upper = jnp.float32(inactive_aux_upper_bound)
    _inactive_aux_eps = jnp.float32(inactive_aux_bound_eps)
    _warmup_steps = jnp.int32(inactive_aux_warmup_steps)
    _inactive_aux_dev_mode = str(inactive_aux_dev_mode).lower()
    _inactive_aux_use_bounded_z = _inactive_aux_dev_mode in ('bounded_z', 'robust_z', 'z')
    _inactive_aux_ce_clip_std = jnp.float32(inactive_aux_ce_clip_std)
    _inactive_aux_z_clip = jnp.float32(inactive_aux_z_clip)
    _inactive_aux_z_tanh = bool(inactive_aux_z_tanh)
    _inactive_aux_weighted_clip = jnp.float32(inactive_aux_weighted_clip)
    _inactive_aux_norm_by_layers = bool(inactive_aux_normalize_by_layers)
    _inactive_aux_norm_by_layers_f = jnp.float32(
        1.0 if _inactive_aux_norm_by_layers else 0.0)
    _inactive_aux_enabled = bool(inactive_aux_enabled)
    _dead_penalty_qk_weight = jnp.float32(dead_penalty_qk_weight)
    _dead_penalty_v_weight = jnp.float32(dead_penalty_v_weight)
    _dead_penalty_rst_weight = jnp.float32(dead_penalty_rst_weight)
    _cb1a_enabled = bool(cb1a_enabled)
    _cb1a_weight = jnp.float32(cb1a_weight)
    _cb1a_challenge_weight = jnp.float32(cb1a_challenge_weight)
    _cb1a_prune_weight = jnp.float32(cb1a_prune_weight)
    _cb1a_qk_weight = jnp.float32(cb1a_qk_weight)
    _cb1a_v_weight = jnp.float32(cb1a_v_weight)
    _cb1a_rst_weight = jnp.float32(cb1a_rst_weight)
    _cb1a_qk_challenge_weight = jnp.float32(
        cb1a_challenge_weight if cb1a_qk_challenge_weight is None
        else cb1a_qk_challenge_weight)
    _cb1a_qk_prune_weight = jnp.float32(
        cb1a_prune_weight if cb1a_qk_prune_weight is None
        else cb1a_qk_prune_weight)
    _cb1a_v_challenge_weight = jnp.float32(
        cb1a_challenge_weight if cb1a_v_challenge_weight is None
        else cb1a_v_challenge_weight)
    _cb1a_v_prune_weight = jnp.float32(
        cb1a_prune_weight if cb1a_v_prune_weight is None
        else cb1a_v_prune_weight)
    _cb1a_rst_challenge_weight = jnp.float32(
        cb1a_challenge_weight if cb1a_rst_challenge_weight is None
        else cb1a_rst_challenge_weight)
    _cb1a_rst_prune_weight = jnp.float32(
        cb1a_prune_weight if cb1a_rst_prune_weight is None
        else cb1a_rst_prune_weight)
    _cb1a_ce_mode = str(cb1a_ce_mode).lower()
    if _cb1a_ce_mode != 'sigmoid_z':
        raise ValueError(
            f"Unsupported cb1a_ce_mode={cb1a_ce_mode!r}; "
            "expected 'sigmoid_z'.")
    _cb1a_eps = jnp.float32(cb1a_eps)
    _dead_weighted_clip = jnp.float32(dead_penalty_weighted_clip)
    _global_grad_clip = jnp.float32(global_grad_clip)
    _tau_lr_mult = jnp.float32(tau_lr_mult)
    _tau_grad_clip = jnp.float32(tau_grad_clip)
    _router_proj_lr_mult = jnp.float32(router_proj_lr_mult)
    _router_proj_grad_clip = jnp.float32(router_proj_grad_clip)
    _router_scan_lr_mult = jnp.float32(router_scan_lr_mult)
    _router_scan_grad_clip = jnp.float32(router_scan_grad_clip)
    _route_emb_lr_mult = jnp.float32(route_emb_lr_mult)
    _route_emb_grad_clip = jnp.float32(route_emb_grad_clip)
    _op_key_lr_mult = jnp.float32(op_key_lr_mult)
    _op_key_grad_clip = jnp.float32(op_key_grad_clip)
    _enable_control_update_caps = bool(enable_control_update_caps)
    _router_proj_update_ratio_cap = jnp.float32(router_proj_update_ratio_cap)
    _route_emb_update_ratio_cap = jnp.float32(route_emb_update_ratio_cap)
    _tau_update_abs_cap = jnp.float32(tau_update_abs_cap)
    _scan_update_abs_cap = jnp.float32(scan_update_abs_cap)
    _pass_analysis_kw = _model_accepts_analysis(model)
    _pass_soft_gate_schedule_kw = _model_accepts_soft_gate_schedule(model)
    _pass_soft_gate_t_final_kw = _model_accepts_soft_gate_t_final(model)
    _pass_execution_prune_kw = _model_accepts_execution_prune_eps(model)
    _pass_boundary_power_kw = _model_accepts_soft_gate_boundary_power(model)
    _pass_den_power_kw = _model_accepts_admission_den_power(model)
    _pass_minimal_train_kw = _model_accepts_minimal_train(model)
    _pass_training_tokens_kw = _model_accepts_training_tokens(model)
    _model_version = getattr(
        model, '__version__', getattr(type(model), '__version__', ''))
    _use_minimal_train_path = (
        str(_model_version) == V4168_MODEL_VERSION
        and _pass_minimal_train_kw)
    _is_baseline_model = bool(is_baseline) or _is_baseline_version(_model_version)
    _is_soft_direct_tau = _is_active_srw_version(_model_version)
    _is_v4166_model = _is_rw_key_srw_version(_model_version)
    _is_boundary_power_model = _is_active_srw_version(_model_version)
    if _is_baseline_model:
        _cb1a_enabled = False
        _dead_penalty_qk_weight = jnp.float32(0.0)
        _dead_penalty_v_weight = jnp.float32(0.0)
        _dead_penalty_rst_weight = jnp.float32(0.0)
        _dead_weighted_clip = jnp.float32(0.0)
        _inactive_aux_enabled = False
    if _is_soft_direct_tau:
        # Official v4164 train path keeps the soft DirectTau loss surface.
        _cb1a_enabled = False
        _dead_penalty_qk_weight = jnp.float32(0.0)
        _dead_penalty_v_weight = jnp.float32(0.0)
        _dead_penalty_rst_weight = jnp.float32(0.0)
        _dead_weighted_clip = jnp.float32(0.0)
        _inactive_aux_norm_by_layers = True
        _inactive_aux_norm_by_layers_f = jnp.float32(1.0)
    _inactive_aux_requires_no_active_direct = False

    _soft_gate_runtime_enabled = bool(
        soft_gate_schedule_active and _is_soft_direct_tau)
    _total_training_steps = jnp.float32(max(1, int(total_training_steps or 1)))
    _soft_gate_t_start = jnp.float32(soft_gate_t_start)
    _soft_gate_t_final = jnp.float32(soft_gate_t_final)
    _soft_gate_t_hold_frac = jnp.float32(soft_gate_t_hold_frac)
    _soft_gate_t_anneal_end_frac = jnp.float32(soft_gate_t_anneal_end_frac)
    _soft_gate_schedule = str(soft_gate_schedule).lower()
    _soft_gate_t_power = jnp.float32(soft_gate_t_power)
    _soft_gate_t_gompertz_center = jnp.float32(soft_gate_t_gompertz_center)
    _soft_gate_t_gompertz_steepness = jnp.float32(
        soft_gate_t_gompertz_steepness)
    _soft_gate_pool_defaults = {
        'start': soft_gate_t_start,
        'final': soft_gate_t_final,
        'hold_frac': soft_gate_t_hold_frac,
        'anneal_end_frac': soft_gate_t_anneal_end_frac,
        'schedule': soft_gate_schedule,
        'power': soft_gate_t_power,
        'gompertz_center': soft_gate_t_gompertz_center,
        'gompertz_steepness': soft_gate_t_gompertz_steepness,
    }
    _soft_gate_pool_cfg = _coerce_pool_schedule_configs(
        (soft_gate_pool_schedules
         if (pool_specific_gate_t
             or _soft_gate_schedule == 'developmental_band')
         else None),
        _soft_gate_pool_defaults)
    _boundary_power_schedule_active = bool(
        boundary_power_schedule_active)
    _soft_gate_boundary_power_start = jnp.float32(
        soft_gate_boundary_power_start)
    _soft_gate_boundary_power_mid = jnp.float32(
        soft_gate_boundary_power_mid)
    _soft_gate_boundary_power_final = jnp.float32(
        soft_gate_boundary_power_final)
    _soft_gate_boundary_power_start_frac = jnp.float32(
        soft_gate_boundary_power_start_frac)
    _soft_gate_boundary_power_mid_frac = jnp.float32(
        soft_gate_boundary_power_mid_frac)
    _soft_gate_boundary_power_final_frac = jnp.float32(
        soft_gate_boundary_power_final_frac)
    _admission_den_power = jnp.float32(admission_den_power)
    _tokens_per_step = jnp.float32(max(0, int(tokens_per_step or 0)))
    _inactive_aux_start_frac = jnp.float32(inactive_aux_start_frac)
    _inactive_aux_full_frac = jnp.float32(inactive_aux_full_frac)
    _inactive_aux_schedule = str(inactive_aux_schedule).lower()
    _compact_train_metrics = bool(compact_train_metrics)
    _train_drift_diagnostics = not _compact_train_metrics
    _keep_train_layer_metrics = bool(keep_train_layer_metrics)
    _heavy_keys = {
        'per_token_ce',
        'valid_mask',
        'attn_tau_direct',
        'rst_tau_direct',
        'attn_tau_offset',
        'rst_tau_offset',
        'attn_no_active_direct',
        'rst_no_active_direct',
    }
    if not _keep_train_layer_metrics:
        _heavy_keys.update({
            'per_layer_attn_out_norm',
            'per_layer_rst_out_norm',
        })
    _train_result_heavy_keys = frozenset(_heavy_keys)

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, dropout_key,
                   prev_op_key_snap, step):
        labels = jnp.where(attention_mask == 1, input_ids, -100)

        def loss_fn(params):
            extra_kw = {}
            if sharded_fns is not None:
                extra_kw['sharded_fns'] = sharded_fns
            if _pass_analysis_kw:
                extra_kw['analysis'] = False
            if _use_minimal_train_path:
                extra_kw['minimal_train'] = True
            if _pass_training_tokens_kw:
                extra_kw['training_tokens'] = (
                    step.astype(jnp.float32) * _tokens_per_step)
            if _soft_gate_runtime_enabled:
                soft_gate_T_qk = _scheduled_from_config(
                    step, _total_training_steps, _soft_gate_pool_cfg['qk'])
                soft_gate_T_v = _scheduled_from_config(
                    step, _total_training_steps, _soft_gate_pool_cfg['v'])
                soft_gate_T_rst = _scheduled_from_config(
                    step, _total_training_steps, _soft_gate_pool_cfg['rst'])
                inactive_aux_schedule_scale = _scheduled_scalar(
                    step, _total_training_steps,
                    0.0, 1.0, _inactive_aux_start_frac, _inactive_aux_full_frac,
                    _inactive_aux_schedule)
            else:
                soft_gate_T_qk = jnp.float32(0.07)
                soft_gate_T_v = jnp.float32(0.07)
                soft_gate_T_rst = jnp.float32(0.07)
                inactive_aux_schedule_scale = (step >= _warmup_steps).astype(jnp.float32)
            boundary_power_p = scheduled_boundary_power_by_frac(
                step, _total_training_steps,
                _boundary_power_schedule_active and _is_boundary_power_model,
                _soft_gate_boundary_power_start,
                _soft_gate_boundary_power_mid,
                _soft_gate_boundary_power_final,
                _soft_gate_boundary_power_mid_frac,
                _soft_gate_boundary_power_final_frac,
                _soft_gate_boundary_power_start_frac)
            soft_gate_T = soft_gate_T_qk
            if _soft_gate_runtime_enabled and _pass_soft_gate_schedule_kw:
                extra_kw['soft_gate_temperature'] = soft_gate_T_qk
                extra_kw['soft_gate_T_qk'] = soft_gate_T_qk
                extra_kw['soft_gate_T_v'] = soft_gate_T_v
                extra_kw['soft_gate_T_rst'] = soft_gate_T_rst
            if _pass_boundary_power_kw:
                extra_kw['soft_gate_boundary_power'] = boundary_power_p
                extra_kw['soft_gate_boundary_power_final'] = (
                    _soft_gate_boundary_power_final)
            if _pass_soft_gate_t_final_kw:
                extra_kw['soft_gate_t_final'] = _soft_gate_t_final
            if _pass_den_power_kw:
                extra_kw['admission_den_power'] = _admission_den_power
            if _pass_execution_prune_kw:
                # Training never execution-prunes; pruning is eval-sweep only.
                extra_kw['execution_prune_eps'] = jnp.float32(0.0)
            result = model.apply(
                {'params': params},
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                deterministic=False,
                rngs={'dropout': dropout_key},
                **extra_kw,
            )
            ce_loss = result['loss']
            aux_loss = result.get('aux_loss', jnp.float32(0.0))
            tau_reg = result.get('tau_reg', jnp.float32(0.0))
            dead_penalty_unweighted = result.get(
                'dead_penalty', jnp.float32(0.0))
            dead_qk_raw = result.get('attn_qk_dead_penalty', None)
            dead_v_raw = result.get('attn_v_dead_penalty', None)
            dead_rst_raw = result.get('rst_dead_penalty', None)
            if (dead_qk_raw is not None
                    and dead_v_raw is not None
                    and dead_rst_raw is not None):
                dead_penalty = dead_qk_raw + dead_v_raw + dead_rst_raw
                dead_penalty_weighted_unclipped = (
                    _dead_penalty_qk_weight * dead_qk_raw
                    + _dead_penalty_v_weight * dead_v_raw
                    + _dead_penalty_rst_weight * dead_rst_raw)
            else:
                dead_penalty = dead_penalty_unweighted
                dead_penalty_weighted_unclipped = (
                    jnp.float32(dead_penalty_weight) * dead_penalty)
            per_token_ce = result.get('per_token_ce', None)
            attn_tau_off = result.get('attn_tau_direct', result.get('attn_tau_offset', None))
            rst_tau_off = result.get('rst_tau_direct', result.get('rst_tau_offset', None))
            attn_no_active = result.get('attn_no_active_direct', None)
            rst_no_active = result.get('rst_no_active_direct', None)
            valid_mask = result.get('valid_mask', None)
            if (_inactive_aux_enabled and _inactive_aux_requires_no_active_direct
                    and (attn_no_active is None or rst_no_active is None)):
                raise ValueError(
                    "The disabled auxiliary path requires attn_no_active_direct "
                    "and rst_no_active_direct from the model.")
            have_inactive_aux = (_inactive_aux_enabled
                            and per_token_ce is not None
                            and attn_tau_off is not None
                            and rst_tau_off is not None
                            and (not _inactive_aux_requires_no_active_direct
                                 or (attn_no_active is not None
                                     and rst_no_active is not None))
                            and valid_mask is not None
                            and _global_mean_std_reducer is not None)
            if have_inactive_aux:
                vmask_f = valid_mask.astype(jnp.float32)
                ce_sg = jax.lax.stop_gradient(per_token_ce.astype(jnp.float32))
                mean_ce0, std_ce0 = _global_mean_std_reducer(ce_sg, valid_mask)
                if _inactive_aux_use_bounded_z:
                    # Robust center/scale: first clip extreme CE tokens using
                    # global mean/std, then recompute a clipped mean/std.  This
                    # prevents a few hard/easy tokens from shifting the batch
                    # the main loss and creating a huge one-step tau control signal.
                    clip_lo = mean_ce0 - _inactive_aux_ce_clip_std * std_ce0
                    clip_hi = mean_ce0 + _inactive_aux_ce_clip_std * std_ce0
                    ce_clip = jnp.clip(ce_sg, clip_lo, clip_hi)
                    global_mean_ce, global_std_ce = _global_mean_std_reducer(
                        ce_clip, valid_mask)
                    raw_deviation = ce_sg - jax.lax.stop_gradient(global_mean_ce)
                    z = raw_deviation / (jax.lax.stop_gradient(global_std_ce) + 1e-8)
                    if _inactive_aux_z_tanh:
                        deviation = _inactive_aux_z_clip * jnp.tanh(z / _inactive_aux_z_clip)
                    else:
                        deviation = jnp.clip(z, -_inactive_aux_z_clip, _inactive_aux_z_clip)
                    deviation = jax.lax.stop_gradient(deviation)
                else:
                    global_mean_ce = jax.lax.stop_gradient(mean_ce0)
                    raw_deviation = ce_sg - global_mean_ce
                    deviation = jax.lax.stop_gradient(raw_deviation)
                # Asymmetric: full push on hard tokens, per-route asymmetry
                # on easy tokens. The CE measurement is shared, but Q/K/V/RST
                # can apply different negative-side feedback strength.
                signal_q = jax.lax.stop_gradient(
                    jnp.where(deviation > 0, deviation, _asym_q * deviation))
                signal_k = jax.lax.stop_gradient(
                    jnp.where(deviation > 0, deviation, _asym_k * deviation))
                signal_v = jax.lax.stop_gradient(
                    jnp.where(deviation > 0, deviation, _asym_v * deviation))
                signal_rst = jax.lax.stop_gradient(
                    jnp.where(deviation > 0, deviation, _asym_rst * deviation))

                # contribution is turned off per-element when further push in
                # that direction would breach [lower, upper].
                # attn_tau_off shape [L, B, S, 3]; rst_tau_off [L, B, S, 1].
                a_tau_t = attn_tau_off[:, :, :-1, :]     # [L, B, S-1, 3]
                k_tau_t = rst_tau_off[:, :, :-1, :]     # [L, B, S-1, 1]
                if attn_no_active is None:
                    a_no_active_t = jnp.zeros_like(a_tau_t, dtype=jnp.bool_)
                else:
                    a_no_active_t = attn_no_active[:, :, :-1, :]
                if rst_no_active is None:
                    k_no_active_t = jnp.zeros_like(k_tau_t, dtype=jnp.bool_)
                else:
                    k_no_active_t = rst_no_active[:, :, :-1, :]
                inactive_aux_layer_count = jnp.float32(a_tau_t.shape[0])
                a_dev_b = jnp.stack(
                    (signal_q, signal_k, signal_v), axis=-1)[None, :, :, :]
                k_dev_b = signal_rst[None, :, :, None]
                vmask_b = vmask_f[None, :, :, None]       # [1, B, S-1, 1]
                a_easy = a_dev_b < 0.0
                k_easy = k_dev_b < 0.0
                a_already_off = jax.lax.stop_gradient(
                    a_no_active_t.astype(jnp.bool_))
                k_already_off = jax.lax.stop_gradient(
                    k_no_active_t.astype(jnp.bool_))
                a_easy_shutoff = a_easy & a_already_off
                k_easy_shutoff = k_easy & k_already_off
                a_keep = jax.lax.stop_gradient(
                    (~a_easy_shutoff).astype(jnp.float32))
                k_keep = jax.lax.stop_gradient(
                    (~k_easy_shutoff).astype(jnp.float32))
                a_dev_b = a_dev_b * a_keep
                k_dev_b = k_dev_b * k_keep
                a_tau_ref = jax.lax.stop_gradient(a_tau_t)
                k_tau_ref = jax.lax.stop_gradient(k_tau_t)

                # Per-element bound-hit masks. Hard off, not soft decay.
                # The block decision is measurement feedback; the loss below
                # still differentiates through the original tau_offset tensor.
                a_down_off = (a_dev_b > 0) & (a_tau_ref <= _inactive_aux_lower + _inactive_aux_eps)
                a_up_off   = (a_dev_b < 0) & (a_tau_ref >= _inactive_aux_upper - _inactive_aux_eps)
                a_off_mask = jax.lax.stop_gradient(a_down_off | a_up_off)
                k_down_off = (k_dev_b > 0) & (k_tau_ref <= _inactive_aux_lower + _inactive_aux_eps)
                k_up_off   = (k_dev_b < 0) & (k_tau_ref >= _inactive_aux_upper - _inactive_aux_eps)
                k_off_mask = jax.lax.stop_gradient(k_down_off | k_up_off)

                a_active = jax.lax.stop_gradient(jnp.where(a_off_mask, 0.0, 1.0))
                k_active = jax.lax.stop_gradient(jnp.where(k_off_mask, 0.0, 1.0))

                # tau_offset distribution diagnostics (stop_gradient, obs-only).
                _a_tau_flat = jax.lax.stop_gradient(attn_tau_off)
                _k_tau_flat = jax.lax.stop_gradient(rst_tau_off)
                attn_tau_off_min = _a_tau_flat.min()
                attn_tau_off_max = _a_tau_flat.max()
                attn_tau_off_p99 = jnp.quantile(_a_tau_flat, 0.99)
                attn_tau_off_p01 = jnp.quantile(_a_tau_flat, 0.01)
                attn_tau_off_neg_frac = (_a_tau_flat < 0).astype(jnp.float32).mean()
                rst_tau_off_min = _k_tau_flat.min()
                rst_tau_off_max = _k_tau_flat.max()
                rst_tau_off_p99 = jnp.quantile(_k_tau_flat, 0.99)
                rst_tau_off_p01 = jnp.quantile(_k_tau_flat, 0.01)
                rst_tau_off_neg_frac = (_k_tau_flat < 0).astype(jnp.float32).mean()

                # Per-element contribution -reduce. Gradient flows through
                # the tau_offset tensor only (signal is stop_gradient'd).
                vsum_eps = vmask_f.sum() + 1e-8
                inactive_aux_norm = vsum_eps
                if _inactive_aux_norm_by_layers:
                    inactive_aux_norm = inactive_aux_norm * inactive_aux_layer_count
                a_contrib_pre = a_dev_b * a_tau_t * vmask_b
                k_contrib_pre = k_dev_b * k_tau_t * vmask_b
                a_contrib = a_contrib_pre * a_active
                k_contrib = k_contrib_pre * k_active
                a_q_contrib_pre = a_contrib_pre[..., 0:1]
                a_k_contrib_pre = a_contrib_pre[..., 1:2]
                a_v_contrib_pre = a_contrib_pre[..., 2:3]
                a_q_contrib = a_contrib[..., 0:1]
                a_k_contrib = a_contrib[..., 1:2]
                a_v_contrib = a_contrib[..., 2:3]
                inactive_aux_q_pre_bound = a_q_contrib_pre.sum() / inactive_aux_norm
                inactive_aux_k_pre_bound = a_k_contrib_pre.sum() / inactive_aux_norm
                inactive_aux_qk_pre_bound = (
                    inactive_aux_q_pre_bound + inactive_aux_k_pre_bound)
                inactive_aux_v_pre_bound = a_v_contrib_pre.sum() / inactive_aux_norm
                inactive_aux_attn_pre_bound = a_contrib_pre.sum() / inactive_aux_norm
                inactive_aux_rst_pre_bound = k_contrib_pre.sum() / inactive_aux_norm
                inactive_aux_loss_pre_bound = (
                    inactive_aux_attn_pre_bound + inactive_aux_rst_pre_bound)
                inactive_aux_q_raw = a_q_contrib.sum() / inactive_aux_norm
                inactive_aux_k_raw = a_k_contrib.sum() / inactive_aux_norm
                inactive_aux_qk_raw = inactive_aux_q_raw + inactive_aux_k_raw
                inactive_aux_v_raw = a_v_contrib.sum() / inactive_aux_norm
                inactive_aux_attn_raw = a_contrib.sum() / inactive_aux_norm
                inactive_aux_rst_raw = k_contrib.sum() / inactive_aux_norm
                inactive_aux_loss_raw = inactive_aux_attn_raw + inactive_aux_rst_raw

                # Observational stats (same interface as before).
                pos_mask = (deviation > 0).astype(jnp.float32) * vmask_f
                neg_mask = (deviation < 0).astype(jnp.float32) * vmask_f
                pos_frac = pos_mask.sum() / vsum_eps
                pos_mean = (jnp.maximum(deviation, 0.0) * vmask_f).sum() / (
                    pos_mask.sum() + 1e-8)
                neg_mean = (jnp.maximum(-deviation, 0.0) * vmask_f).sum() / (
                    neg_mask.sum() + 1e-8)

                # Off fractions replace pool-mean block fractions. Denominator
                # is total (layer x batch x valid-time x route) slots.
                _a_tot = vmask_b.sum() * a_tau_t.shape[0] * a_tau_t.shape[-1]
                _qk_tot = vmask_b.sum() * a_tau_t.shape[0] * 2
                _q_tot = vmask_b.sum() * a_tau_t.shape[0]
                _key_tot = vmask_b.sum() * a_tau_t.shape[0]
                _v_tot = vmask_b.sum() * a_tau_t.shape[0]
                _k_tot = vmask_b.sum() * k_tau_t.shape[0] * k_tau_t.shape[-1]
                inactive_aux_no_active_easy_shutoff_q = jax.lax.stop_gradient(
                    (a_easy_shutoff[..., 0:1].astype(jnp.float32)
                     * vmask_b).sum() / (_q_tot + 1e-8))
                inactive_aux_no_active_easy_shutoff_k = jax.lax.stop_gradient(
                    (a_easy_shutoff[..., 1:2].astype(jnp.float32)
                     * vmask_b).sum() / (_key_tot + 1e-8))
                inactive_aux_no_active_easy_shutoff_v = jax.lax.stop_gradient(
                    (a_easy_shutoff[..., 2:3].astype(jnp.float32)
                     * vmask_b).sum() / (_v_tot + 1e-8))
                inactive_aux_no_active_easy_shutoff_rst = jax.lax.stop_gradient(
                    (k_easy_shutoff.astype(jnp.float32) * vmask_b).sum()
                    / (_k_tot + 1e-8))
                block_frac_a = jax.lax.stop_gradient(
                    (a_off_mask.astype(jnp.float32) * vmask_b).sum() / (_a_tot + 1e-8))
                block_frac_qk = jax.lax.stop_gradient(
                    (a_off_mask[..., :2].astype(jnp.float32) * vmask_b).sum()
                    / (_qk_tot + 1e-8))
                block_frac_q = jax.lax.stop_gradient(
                    (a_off_mask[..., 0:1].astype(jnp.float32) * vmask_b).sum()
                    / (_q_tot + 1e-8))
                block_frac_key = jax.lax.stop_gradient(
                    (a_off_mask[..., 1:2].astype(jnp.float32) * vmask_b).sum()
                    / (_key_tot + 1e-8))
                block_frac_v = jax.lax.stop_gradient(
                    (a_off_mask[..., 2:3].astype(jnp.float32) * vmask_b).sum()
                    / (_v_tot + 1e-8))
                block_frac_rst = jax.lax.stop_gradient(
                    (k_off_mask.astype(jnp.float32) * vmask_b).sum() / (_k_tot + 1e-8))

                _dev_sg = jax.lax.stop_gradient(
                    jnp.stack((signal_q, signal_k, signal_v, signal_rst), axis=0)
                    * vmask_f[None, :, :])
                dev_pos_max = _dev_sg.max()
                dev_neg_max = (-_dev_sg).max()
            else:
                global_mean_ce = jnp.float32(0.0)
                inactive_aux_loss_raw = jnp.float32(0.0)
                inactive_aux_q_raw = jnp.float32(0.0)
                inactive_aux_k_raw = jnp.float32(0.0)
                inactive_aux_qk_raw = jnp.float32(0.0)
                inactive_aux_v_raw = jnp.float32(0.0)
                inactive_aux_attn_raw = jnp.float32(0.0)
                inactive_aux_rst_raw = jnp.float32(0.0)
                inactive_aux_loss_pre_bound = jnp.float32(0.0)
                inactive_aux_q_pre_bound = jnp.float32(0.0)
                inactive_aux_k_pre_bound = jnp.float32(0.0)
                inactive_aux_qk_pre_bound = jnp.float32(0.0)
                inactive_aux_v_pre_bound = jnp.float32(0.0)
                inactive_aux_attn_pre_bound = jnp.float32(0.0)
                inactive_aux_rst_pre_bound = jnp.float32(0.0)
                inactive_aux_layer_count = jnp.float32(0.0)
                inactive_aux_norm = jnp.float32(0.0)
                pos_frac = jnp.float32(0.0)
                pos_mean = jnp.float32(0.0)
                neg_mean = jnp.float32(0.0)
                block_frac_a = jnp.float32(0.0)
                block_frac_q = jnp.float32(0.0)
                block_frac_key = jnp.float32(0.0)
                block_frac_qk = jnp.float32(0.0)
                block_frac_v = jnp.float32(0.0)
                block_frac_rst = jnp.float32(0.0)
                inactive_aux_no_active_easy_shutoff_q = jnp.float32(0.0)
                inactive_aux_no_active_easy_shutoff_k = jnp.float32(0.0)
                inactive_aux_no_active_easy_shutoff_v = jnp.float32(0.0)
                inactive_aux_no_active_easy_shutoff_rst = jnp.float32(0.0)
                dev_pos_max = jnp.float32(0.0)
                dev_neg_max = jnp.float32(0.0)
                attn_tau_off_min = jnp.float32(0.0)
                attn_tau_off_max = jnp.float32(0.0)
                attn_tau_off_p99 = jnp.float32(0.0)
                attn_tau_off_p01 = jnp.float32(0.0)
                attn_tau_off_neg_frac = jnp.float32(0.0)
                rst_tau_off_min = jnp.float32(0.0)
                rst_tau_off_max = jnp.float32(0.0)
                rst_tau_off_p99 = jnp.float32(0.0)
                rst_tau_off_p01 = jnp.float32(0.0)
                rst_tau_off_neg_frac = jnp.float32(0.0)

            attn_cb1a_challenge = result.get('attn_cb1a_challenge_gap', None)
            attn_cb1a_prune = result.get('attn_cb1a_prune_gap', None)
            rst_cb1a_challenge = result.get('rst_cb1a_challenge_gap', None)
            rst_cb1a_prune = result.get('rst_cb1a_prune_gap', None)
            have_cb1a = (_cb1a_enabled
                         and per_token_ce is not None
                         and valid_mask is not None
                         and attn_cb1a_challenge is not None
                         and attn_cb1a_prune is not None
                         and rst_cb1a_challenge is not None
                         and rst_cb1a_prune is not None)
            if have_cb1a:
                vmask_f = valid_mask.astype(jnp.float32)
                ce_sg = jax.lax.stop_gradient(per_token_ce.astype(jnp.float32))
                if _global_mean_std_reducer is not None:
                    cb1a_ce_mean, cb1a_ce_std = _global_mean_std_reducer(
                        ce_sg, valid_mask)
                else:
                    cb1a_cnt = vmask_f.sum()
                    cb1a_ce_mean = (ce_sg * vmask_f).sum() / (
                        cb1a_cnt + _cb1a_eps)
                    cb1a_var = (
                        jnp.square(ce_sg - cb1a_ce_mean) * vmask_f
                    ).sum() / (cb1a_cnt + _cb1a_eps)
                    cb1a_ce_std = jnp.sqrt(
                        jnp.maximum(cb1a_var, 0.0) + _cb1a_eps)
                cb1a_ce_z = jax.lax.stop_gradient(
                    (ce_sg - jax.lax.stop_gradient(cb1a_ce_mean))
                    / (jax.lax.stop_gradient(cb1a_ce_std) + _cb1a_eps))
                cb1a_hard_weight = jax.lax.stop_gradient(
                    jax.nn.sigmoid(cb1a_ce_z))
                cb1a_easy_weight = jax.lax.stop_gradient(
                    jax.nn.sigmoid(-cb1a_ce_z))

                # Gaps are [L,B,S,route]. CE is next-token loss for positions
                # [B,S-1], so use routing decisions from positions 0..S-2.
                a_chal = attn_cb1a_challenge[:, :, :-1, :]
                a_prune = attn_cb1a_prune[:, :, :-1, :]
                k_chal = rst_cb1a_challenge[:, :, :-1, :]
                k_prune = rst_cb1a_prune[:, :, :-1, :]
                hard_b = cb1a_hard_weight[None, :, :, None]
                easy_b = cb1a_easy_weight[None, :, :, None]
                vmask_b = vmask_f[None, :, :, None]
                vsum = vmask_f.sum()
                a_layers = jnp.float32(a_chal.shape[0])
                k_layers = jnp.float32(k_chal.shape[0])

                qk_challenge_raw = (
                    (a_chal[..., :2] * hard_b * vmask_b).sum()
                    / (vsum * a_layers * jnp.float32(2.0) + _cb1a_eps))
                qk_prune_raw = (
                    (a_prune[..., :2] * easy_b * vmask_b).sum()
                    / (vsum * a_layers * jnp.float32(2.0) + _cb1a_eps))
                v_challenge_raw = (
                    (a_chal[..., 2:3] * hard_b * vmask_b).sum()
                    / (vsum * a_layers + _cb1a_eps))
                v_prune_raw = (
                    (a_prune[..., 2:3] * easy_b * vmask_b).sum()
                    / (vsum * a_layers + _cb1a_eps))
                rst_challenge_raw = (
                    (k_chal * hard_b * vmask_b).sum()
                    / (vsum * k_layers + _cb1a_eps))
                rst_prune_raw = (
                    (k_prune * easy_b * vmask_b).sum()
                    / (vsum * k_layers + _cb1a_eps))

                cb1a_challenge_raw = (
                    qk_challenge_raw + v_challenge_raw + rst_challenge_raw)
                cb1a_prune_raw = (
                    qk_prune_raw + v_prune_raw + rst_prune_raw)
                cb1a_qk_raw = (
                    _cb1a_qk_challenge_weight * qk_challenge_raw
                    + _cb1a_qk_prune_weight * qk_prune_raw)
                cb1a_v_raw = (
                    _cb1a_v_challenge_weight * v_challenge_raw
                    + _cb1a_v_prune_weight * v_prune_raw)
                cb1a_rst_raw = (
                    _cb1a_rst_challenge_weight * rst_challenge_raw
                    + _cb1a_rst_prune_weight * rst_prune_raw)
                cb1a_raw = (
                    cb1a_qk_raw + cb1a_v_raw + cb1a_rst_raw)
                cb1a_loss_weighted = _cb1a_weight * (
                    _cb1a_qk_weight * cb1a_qk_raw
                    + _cb1a_v_weight * cb1a_v_raw
                    + _cb1a_rst_weight * cb1a_rst_raw)
            else:
                cb1a_ce_mean = jnp.float32(0.0)
                cb1a_ce_std = jnp.float32(0.0)
                qk_challenge_raw = jnp.float32(0.0)
                qk_prune_raw = jnp.float32(0.0)
                v_challenge_raw = jnp.float32(0.0)
                v_prune_raw = jnp.float32(0.0)
                rst_challenge_raw = jnp.float32(0.0)
                rst_prune_raw = jnp.float32(0.0)
                cb1a_challenge_raw = jnp.float32(0.0)
                cb1a_prune_raw = jnp.float32(0.0)
                cb1a_qk_raw = jnp.float32(0.0)
                cb1a_v_raw = jnp.float32(0.0)
                cb1a_rst_raw = jnp.float32(0.0)
                cb1a_raw = jnp.float32(0.0)
                cb1a_loss_weighted = jnp.float32(0.0)

            # meaningful; early CE-dominated learning keeps tau gradient clean.
            inactive_aux_active = inactive_aux_schedule_scale
            inactive_aux_loss_weighted_unclipped = (
                (_inactive_aux_weight_q * inactive_aux_q_raw
                 + _inactive_aux_weight_k * inactive_aux_k_raw
                 + _inactive_aux_weight_v * inactive_aux_v_raw
                 + _inactive_aux_weight_rst * inactive_aux_rst_raw)
                * inactive_aux_active)
            inactive_aux_loss_weighted = jnp.where(
                _inactive_aux_weighted_clip > 0.0,
                jnp.clip(inactive_aux_loss_weighted_unclipped,
                         -_inactive_aux_weighted_clip, _inactive_aux_weighted_clip),
                inactive_aux_loss_weighted_unclipped)
            dead_penalty_weighted = jnp.where(
                _dead_weighted_clip > 0.0,
                jnp.minimum(dead_penalty_weighted_unclipped, _dead_weighted_clip),
                dead_penalty_weighted_unclipped)
            if _is_baseline_model:
                tau_reg = jnp.float32(0.0)
                dead_penalty = jnp.float32(0.0)
                dead_penalty_weighted_unclipped = jnp.float32(0.0)
                dead_penalty_weighted = jnp.float32(0.0)
                inactive_aux_loss_weighted_unclipped = jnp.float32(0.0)
                inactive_aux_loss_weighted = jnp.float32(0.0)
                cb1a_loss_weighted = jnp.float32(0.0)
                orth_loss = jnp.float32(0.0)
                div_loss = jnp.float32(0.0)
                total_loss = ce_loss
            elif _is_soft_direct_tau:
                # v4164 reports admission exposure diagnostics but keeps the
                # active loss surface clean. No dead repair loss is active here.
                dead_penalty_weighted_unclipped = jnp.float32(0.0)
                dead_penalty_weighted = jnp.float32(0.0)
                orth_loss = jnp.float32(0.0)
                div_loss = jnp.float32(0.0)
                total_loss = ce_loss + inactive_aux_loss_weighted
            else:
                orth_loss = jnp.float32(0.0)
                div_loss = compute_spatial_diversity_loss(params)
                total_loss = (ce_loss
                              + lb_weight * aux_loss
                              + tau_reg_weight * tau_reg
                              + div_weight * div_loss
                              + dead_penalty_weighted
                              + inactive_aux_loss_weighted
                              + cb1a_loss_weighted)

            inactive_aux_stats = dict(
                global_mean_ce=global_mean_ce,
                inactive_aux_loss_raw=inactive_aux_loss_raw,
                inactive_aux_q_raw=inactive_aux_q_raw,
                inactive_aux_k_raw=inactive_aux_k_raw,
                inactive_aux_qk_raw=inactive_aux_qk_raw,
                inactive_aux_v_raw=inactive_aux_v_raw,
                inactive_aux_attn_raw=inactive_aux_attn_raw,
                inactive_aux_rst_raw=inactive_aux_rst_raw,
                inactive_aux_loss_pre_bound=inactive_aux_loss_pre_bound,
                inactive_aux_q_pre_bound=inactive_aux_q_pre_bound,
                inactive_aux_k_pre_bound=inactive_aux_k_pre_bound,
                inactive_aux_qk_pre_bound=inactive_aux_qk_pre_bound,
                inactive_aux_v_pre_bound=inactive_aux_v_pre_bound,
                inactive_aux_attn_pre_bound=inactive_aux_attn_pre_bound,
                inactive_aux_rst_pre_bound=inactive_aux_rst_pre_bound,
                pos_frac=pos_frac, pos_mean=pos_mean, neg_mean=neg_mean,
                block_frac_a=block_frac_a, block_frac_q=block_frac_q,
                block_frac_key=block_frac_key, block_frac_qk=block_frac_qk,
                block_frac_v=block_frac_v, block_frac_rst=block_frac_rst,
                inactive_aux_no_active_easy_shutoff_q=inactive_aux_no_active_easy_shutoff_q,
                inactive_aux_no_active_easy_shutoff_k=inactive_aux_no_active_easy_shutoff_k,
                inactive_aux_no_active_easy_shutoff_v=inactive_aux_no_active_easy_shutoff_v,
                inactive_aux_no_active_easy_shutoff_rst=inactive_aux_no_active_easy_shutoff_rst,
                dev_pos_max=dev_pos_max, dev_neg_max=dev_neg_max,
                attn_tau_off_min=attn_tau_off_min, attn_tau_off_max=attn_tau_off_max,
                attn_tau_off_p99=attn_tau_off_p99, attn_tau_off_p01=attn_tau_off_p01,
                attn_tau_off_neg_frac=attn_tau_off_neg_frac,
                rst_tau_off_min=rst_tau_off_min, rst_tau_off_max=rst_tau_off_max,
                rst_tau_off_p99=rst_tau_off_p99, rst_tau_off_p01=rst_tau_off_p01,
                rst_tau_off_neg_frac=rst_tau_off_neg_frac,
                inactive_aux_active=inactive_aux_active,
                inactive_aux_loss_weighted_unclipped=inactive_aux_loss_weighted_unclipped,
                inactive_aux_loss_weighted_clipped=inactive_aux_loss_weighted,
                cb1a_ce_mean=cb1a_ce_mean,
                cb1a_ce_std=cb1a_ce_std,
                cb1a_raw=cb1a_raw,
                cb1a_challenge_raw=cb1a_challenge_raw,
                cb1a_prune_raw=cb1a_prune_raw,
                cb1a_weighted=cb1a_loss_weighted,
                cb1a_qk_challenge_raw=qk_challenge_raw,
                cb1a_qk_prune_raw=qk_prune_raw,
                cb1a_v_challenge_raw=v_challenge_raw,
                cb1a_v_prune_raw=v_prune_raw,
                cb1a_rst_challenge_raw=rst_challenge_raw,
                cb1a_rst_prune_raw=rst_prune_raw,
                cb1a_qk_raw=cb1a_qk_raw,
                cb1a_v_raw=cb1a_v_raw,
                cb1a_rst_raw=cb1a_rst_raw,
                cb1a_qk_challenge_weight=_cb1a_qk_challenge_weight,
                cb1a_qk_prune_weight=_cb1a_qk_prune_weight,
                cb1a_v_challenge_weight=_cb1a_v_challenge_weight,
                cb1a_v_prune_weight=_cb1a_v_prune_weight,
                cb1a_rst_challenge_weight=_cb1a_rst_challenge_weight,
                cb1a_rst_prune_weight=_cb1a_rst_prune_weight,
                dead_penalty_raw_unweighted=dead_penalty_unweighted,
                dead_penalty_raw_weighted_pools=dead_penalty,
                dead_penalty_qk_weight=_dead_penalty_qk_weight,
                dead_penalty_v_weight=_dead_penalty_v_weight,
                dead_penalty_rst_weight=_dead_penalty_rst_weight,
                dead_penalty_weighted_unclipped=dead_penalty_weighted_unclipped,
                dead_penalty_weighted_clipped=dead_penalty_weighted,
                inactive_aux_layer_norm_enabled=_inactive_aux_norm_by_layers_f,
                inactive_aux_layer_count=inactive_aux_layer_count,
                inactive_aux_norm=inactive_aux_norm,
                step_in_train=step,
                soft_gate_T=soft_gate_T,
                soft_gate_T_qk=soft_gate_T_qk,
                soft_gate_T_v=soft_gate_T_v,
                soft_gate_T_rst=soft_gate_T_rst,
                boundary_power_p=boundary_power_p,
                admission_den_power=_admission_den_power,
                inactive_aux_schedule_scale=inactive_aux_schedule_scale,
            )
            result_payload = result
            if _compact_train_metrics:
                result_payload = {
                    k: v for k, v in result.items()
                    if k not in _train_result_heavy_keys}
            return total_loss, (ce_loss, aux_loss, tau_reg, orth_loss, div_loss,
                                dead_penalty, inactive_aux_stats, result_payload)

        (total_loss, (ce_loss, aux_loss, tau_reg, orth_loss, div_loss,
                      dead_penalty, inactive_aux_stats, result)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params)

        # XLA SPMD handles gradient all-reduce automatically
        # (loss computed on sharded data -gradients consistent across shards)
        raw_grads_before_tau_stabilize = grads

        def _pre_tree_sq(tree):
            leaves = jax.tree.leaves(tree)
            if not leaves:
                return jnp.float32(0.0)
            return sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                       for x in leaves)

        def _pre_path_tree(tree, *keys):
            cur = tree
            for key in keys:
                if not hasattr(cur, '__contains__') or key not in cur:
                    return {}
                cur = cur[key]
            return cur

        def _path_to_str(path):
            return '/'.join(str(p.key if hasattr(p, 'key') else p) for p in path)

        def _path_parts(ps):
            return tuple(part for part in str(ps).split('/') if part)

        def _has_path_part(ps, *names):
            parts = _path_parts(ps)
            return any(name in parts for name in names)

        def _is_raw_tau_attn_combined_path(ps):
            return _has_path_part(ps, 'raw_tau_attn')

        def _is_raw_tau_qk_path(ps):
            return _has_path_part(ps, 'raw_tau_qk', 'raw_tau_attn_qk')

        def _is_raw_tau_v_path(ps):
            return _has_path_part(ps, 'raw_tau_v', 'raw_tau_attn_v')

        def _is_raw_tau_rst_path(ps):
            return _has_path_part(ps, 'raw_tau_rst')

        def _is_generic_raw_tau_path(ps):
            return _has_path_part(ps, 'raw_tau')

        def _is_tau_attn_bias_path(ps):
            return _has_path_part(ps, 'tau_attn')

        def _is_tau_rst_bias_path(ps):
            return _has_path_part(ps, 'tau_rst')

        def _is_raw_tau_path(ps):
            return (_is_raw_tau_attn_combined_path(ps)
                    or _is_raw_tau_qk_path(ps)
                    or _is_raw_tau_v_path(ps)
                    or _is_raw_tau_rst_path(ps)
                    or _is_generic_raw_tau_path(ps))

        # ------------------------------------------------------------------
        # Control-side stabilisation.
        #
        # Important: LR multipliers are applied to Adam *updates*, not raw
        # gradients. Scaling gradients before Adam is mostly cancelled by
        # Adam's m/sqrt(v) normalisation and is therefore not a true LR
        # multiplier. Raw-gradient clipping remains pre-Adam as a safety valve.
        # ------------------------------------------------------------------
        def _group_sq(tree, path_pred):
            def _visit(path, leaf):
                if path_pred(_path_to_str(path)):
                    x = leaf.astype(jnp.float32)
                    return jnp.sum(jnp.square(x))
                return jnp.float32(0.0)
            leaves = jax.tree.leaves(jax.tree.map_with_path(_visit, tree))
            if not leaves:
                return jnp.float32(0.0)
            return sum(leaves)

        def _is_tau_path(ps):
            return (_is_tau_attn_bias_path(ps)
                    or _is_tau_rst_bias_path(ps)
                    or _is_raw_tau_path(ps))

        def _is_router_proj_path(ps):
            return (('router/proj_attn' in ps)
                    or ('router/proj_rst' in ps)
                    or ('router/q_op_write_query_proj' in ps)
                    or ('router/k_op_write_query_proj' in ps)
                    or ('router/v_op_write_query_proj' in ps)
                    or ('router/rst_op_write_query_proj' in ps))

        def _is_router_scan_path(ps):
            return (('router/raw_scan_offset_attn' in ps)
                    or ('router/raw_scan_offset_rst' in ps)
                    or ('router/scan_attn' in ps)
                    or ('router/scan_rst' in ps))

        def _is_legacy_operator_key_path(ps):
            return (('neuron_pool/attn_qk_emb' in ps)
                    or ('neuron_pool/attn_v_emb' in ps)
                    or ('neuron_pool/rst_emb' in ps)
                    or ('neuron_pool/qk_emb' in ps)
                    or ('neuron_pool/v_emb' in ps)
                    or ('neuron_pool/know_emb' in ps))

        def _is_op_key_proj_path(ps):
            return (('neuron_pool/attn_qk_op_read_proj' in ps)
                    or ('neuron_pool/attn_qk_op_write_proj' in ps)
                    or ('neuron_pool/attn_v_op_read_proj' in ps)
                    or ('neuron_pool/attn_v_op_write_proj' in ps)
                    or ('neuron_pool/rst_op_read_proj' in ps)
                    or ('neuron_pool/rst_op_write_proj' in ps))

        def _is_router_proj_attn_path(ps):
            return (('router/proj_attn' in ps)
                    or ('router/q_op_write_query_proj' in ps)
                    or ('router/k_op_write_query_proj' in ps)
                    or ('router/v_op_write_query_proj' in ps))

        def _is_router_proj_rst_path(ps):
            return (('router/proj_rst' in ps)
                    or ('router/rst_op_write_query_proj' in ps))

        def _is_op_key_qk_path(ps):
            return (('neuron_pool/attn_qk_emb' in ps)
                    or ('neuron_pool/qk_emb' in ps)
                    or ('neuron_pool/attn_qk_op_read_proj' in ps)
                    or ('neuron_pool/attn_qk_op_write_proj' in ps))

        def _is_op_key_v_path(ps):
            return (('neuron_pool/attn_v_emb' in ps)
                    or ('neuron_pool/v_emb' in ps)
                    or ('neuron_pool/attn_v_op_read_proj' in ps)
                    or ('neuron_pool/attn_v_op_write_proj' in ps))

        def _is_op_key_rst_path(ps):
            return (('neuron_pool/rst_emb' in ps)
                    or ('neuron_pool/know_emb' in ps)
                    or ('neuron_pool/rst_op_read_proj' in ps)
                    or ('neuron_pool/rst_op_write_proj' in ps))

        def _is_tau_attn_path(ps):
            return (_is_tau_attn_bias_path(ps)
                    or _is_raw_tau_attn_combined_path(ps)
                    or _is_raw_tau_qk_path(ps)
                    or _is_raw_tau_v_path(ps)
                    or _is_generic_raw_tau_path(ps))

        def _is_tau_rst_path(ps):
            return _is_tau_rst_bias_path(ps) or _is_raw_tau_rst_path(ps)

        def _is_scan_attn_path(ps):
            return (('router/raw_scan_offset_attn' in ps)
                    or ('router/scan_attn' in ps))

        def _is_scan_rst_path(ps):
            return (('router/raw_scan_offset_rst' in ps)
                    or ('router/scan_rst' in ps))

        def _group_max_abs(tree, path_pred):
            def _visit(path, leaf):
                if path_pred(_path_to_str(path)):
                    return jnp.max(jnp.abs(leaf.astype(jnp.float32)))
                return jnp.float32(0.0)
            leaves = jax.tree.leaves(jax.tree.map_with_path(_visit, tree))
            if not leaves:
                return jnp.float32(0.0)
            out = jnp.float32(0.0)
            for leaf in leaves:
                out = jnp.maximum(out, leaf)
            return out

        def _raw_tau_component_leaf(ps, leaf, component):
            x = leaf.astype(jnp.float32)
            if _is_raw_tau_attn_combined_path(ps):
                if x.ndim > 0 and x.shape[-1] >= 3:
                    if component == 'qk':
                        return x[..., :2]
                    if component == 'v':
                        return x[..., 2:3]
                return None
            if _is_generic_raw_tau_path(ps):
                parts = _path_parts(ps)
                if ('rst' in parts) or ('know' in parts):
                    return x if component == 'rst' else None
                if ('v' in parts) or ('attn_v' in parts):
                    return x if component == 'v' else None
                if ('qk' in parts) or ('attn_qk' in parts):
                    return x if component == 'qk' else None
                if x.ndim > 0 and x.shape[-1] >= 3:
                    if component == 'qk':
                        return x[..., :2]
                    if component == 'v':
                        return x[..., 2:3]
                return x if component == 'qk' else None
            if _is_raw_tau_qk_path(ps):
                return x if component == 'qk' else None
            if _is_raw_tau_v_path(ps):
                return x if component == 'v' else None
            if _is_raw_tau_rst_path(ps):
                return x if component == 'rst' else None
            return None

        def _raw_tau_component_sq(tree, component):
            def _visit(path, leaf):
                sub = _raw_tau_component_leaf(
                    _path_to_str(path), leaf, component)
                if sub is None:
                    return jnp.float32(0.0)
                return jnp.sum(jnp.square(sub))
            leaves = jax.tree.leaves(jax.tree.map_with_path(_visit, tree))
            if not leaves:
                return jnp.float32(0.0)
            return sum(leaves)

        def _raw_tau_component_max_abs(tree, component):
            def _visit(path, leaf):
                sub = _raw_tau_component_leaf(
                    _path_to_str(path), leaf, component)
                if sub is None:
                    return jnp.float32(0.0)
                return jnp.max(jnp.abs(sub))
            leaves = jax.tree.leaves(jax.tree.map_with_path(_visit, tree))
            if not leaves:
                return jnp.float32(0.0)
            out = jnp.float32(0.0)
            for leaf in leaves:
                out = jnp.maximum(out, leaf)
            return out

        def _clip_scale(group_norm, clip_value):
            return jnp.where(
                clip_value > 0.0,
                jnp.minimum(1.0, clip_value / (group_norm + 1e-8)),
                jnp.float32(1.0))

        tau_grad_norm_raw = jnp.sqrt(_group_sq(grads, _is_tau_path) + 1e-12)
        router_proj_grad_norm_raw = jnp.sqrt(
            _group_sq(grads, _is_router_proj_path) + 1e-12)
        router_scan_grad_norm_raw = jnp.sqrt(
            _group_sq(grads, _is_router_scan_path) + 1e-12)
        route_emb_grad_norm_raw = jnp.sqrt(
            _group_sq(grads, _is_legacy_operator_key_path) + 1e-12)
        op_key_grad_norm_raw = jnp.sqrt(
            _group_sq(grads, _is_op_key_proj_path) + 1e-12)

        tau_clip_scale = _clip_scale(tau_grad_norm_raw, _tau_grad_clip)
        router_proj_clip_scale = _clip_scale(
            router_proj_grad_norm_raw, _router_proj_grad_clip)
        router_scan_clip_scale = _clip_scale(
            router_scan_grad_norm_raw, _router_scan_grad_clip)
        route_emb_clip_scale = _clip_scale(
            route_emb_grad_norm_raw, _route_emb_grad_clip)
        op_key_clip_scale = _clip_scale(
            op_key_grad_norm_raw, _op_key_grad_clip)

        def _clip_control_grad(path, g):
            ps = _path_to_str(path)
            scale = jnp.float32(1.0)
            scale = jnp.where(_is_tau_path(ps), tau_clip_scale, scale)
            scale = jnp.where(_is_router_proj_path(ps), router_proj_clip_scale, scale)
            scale = jnp.where(_is_router_scan_path(ps), router_scan_clip_scale, scale)
            scale = jnp.where(
                _is_legacy_operator_key_path(ps), route_emb_clip_scale, scale)
            scale = jnp.where(_is_op_key_proj_path(ps), op_key_clip_scale, scale)
            return g * scale.astype(g.dtype)

        if (float(tau_grad_clip) > 0.0
                or float(router_proj_grad_clip) > 0.0
                or float(router_scan_grad_clip) > 0.0
                or float(route_emb_grad_clip) > 0.0
                or float(op_key_grad_clip) > 0.0):
            grads = jax.tree.map_with_path(_clip_control_grad, grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)

        def _scale_control_update(path, u):
            ps = _path_to_str(path)
            mult = jnp.float32(1.0)
            mult = jnp.where(_is_tau_path(ps), _tau_lr_mult, mult)
            mult = jnp.where(_is_router_proj_path(ps), _router_proj_lr_mult, mult)
            mult = jnp.where(_is_router_scan_path(ps), _router_scan_lr_mult, mult)
            mult = jnp.where(
                _is_legacy_operator_key_path(ps), _route_emb_lr_mult, mult)
            mult = jnp.where(_is_op_key_proj_path(ps), _op_key_lr_mult, mult)
            return u * mult.astype(u.dtype)

        if (float(tau_lr_mult) != 1.0
                or float(router_proj_lr_mult) != 1.0
                or float(router_scan_lr_mult) != 1.0
                or float(route_emb_lr_mult) != 1.0
                or float(op_key_lr_mult) != 1.0):
            updates = jax.tree.map_with_path(_scale_control_update, updates)

        # ------------------------------------------------------------------
        # Actual-update trust-region caps.
        #
        # These caps are applied *after* Adam and after the existing LR
        # multipliers. They do not change the forward pass or the loss. They
        # only prevent rare control-geometry jumps where the router projection,
        # route embeddings, tau, or scan parameters move too far in one step.
        # ------------------------------------------------------------------
        def _ratio_cap_stats(params_tree, updates_tree, path_pred, cap):
            p_sq = _group_sq(params_tree, path_pred)
            u_sq = _group_sq(updates_tree, path_pred)
            p_norm = jnp.sqrt(p_sq)
            u_norm = jnp.sqrt(u_sq)
            has_group = p_sq > 0.0
            ratio_pre = jnp.where(
                has_group,
                u_norm / (p_norm + 1e-8),
                jnp.float32(0.0))
            active = cap > 0.0
            scale = jnp.where(
                active & has_group,
                jnp.minimum(jnp.float32(1.0), cap / (ratio_pre + 1e-12)),
                jnp.float32(1.0))
            ratio_post = ratio_pre * scale
            hit = jnp.where(active & has_group & (ratio_pre > cap),
                            jnp.float32(1.0), jnp.float32(0.0))
            return ratio_pre, ratio_post, scale, hit

        def _abs_cap_stats(updates_tree, path_pred, cap):
            abs_pre = _group_max_abs(updates_tree, path_pred)
            active = cap > 0.0
            scale = jnp.where(
                active,
                jnp.minimum(jnp.float32(1.0), cap / (abs_pre + 1e-12)),
                jnp.float32(1.0))
            abs_post = abs_pre * scale
            hit = jnp.where(active & (abs_pre > cap), jnp.float32(1.0), jnp.float32(0.0))
            return abs_pre, abs_post, scale, hit

        def _raw_tau_abs_cap_stats(updates_tree, component, cap):
            abs_pre = _raw_tau_component_max_abs(updates_tree, component)
            active = cap > 0.0
            scale = jnp.where(
                active,
                jnp.minimum(jnp.float32(1.0), cap / (abs_pre + 1e-12)),
                jnp.float32(1.0))
            abs_post = abs_pre * scale
            hit = jnp.where(active & (abs_pre > cap), jnp.float32(1.0), jnp.float32(0.0))
            return abs_pre, abs_post, scale, hit

        if _enable_control_update_caps:
            (upd_proj_attn_ratio_pre, upd_proj_attn_ratio_post,
             upd_proj_attn_scale, upd_proj_attn_hit) = _ratio_cap_stats(
                params, updates, _is_router_proj_attn_path,
                _router_proj_update_ratio_cap)
            (upd_proj_rst_ratio_pre, upd_proj_rst_ratio_post,
             upd_proj_rst_scale, upd_proj_rst_hit) = _ratio_cap_stats(
                params, updates, _is_router_proj_rst_path,
                _router_proj_update_ratio_cap)
            (upd_op_key_qk_ratio_pre, upd_op_key_qk_ratio_post,
             upd_op_key_qk_scale, upd_op_key_qk_hit) = _ratio_cap_stats(
                params, updates, _is_op_key_qk_path,
                _route_emb_update_ratio_cap)
            (upd_op_key_v_ratio_pre, upd_op_key_v_ratio_post,
             upd_op_key_v_scale, upd_op_key_v_hit) = _ratio_cap_stats(
                params, updates, _is_op_key_v_path,
                _route_emb_update_ratio_cap)
            (upd_op_key_rst_ratio_pre, upd_op_key_rst_ratio_post,
             upd_op_key_rst_scale, upd_op_key_rst_hit) = _ratio_cap_stats(
                params, updates, _is_op_key_rst_path,
                _route_emb_update_ratio_cap)
            (upd_tau_attn_abs_pre, upd_tau_attn_abs_post,
             upd_tau_attn_scale, upd_tau_attn_hit) = _abs_cap_stats(
                updates, _is_tau_attn_bias_path, _tau_update_abs_cap)
            (upd_tau_rst_abs_pre, upd_tau_rst_abs_post,
             upd_tau_rst_scale, upd_tau_rst_hit) = _abs_cap_stats(
                updates, _is_tau_rst_bias_path, _tau_update_abs_cap)
            (upd_raw_tau_qk_abs_pre, upd_raw_tau_qk_abs_post,
             upd_raw_tau_qk_scale, upd_raw_tau_qk_hit) = _raw_tau_abs_cap_stats(
                updates, 'qk', _tau_update_abs_cap)
            (upd_raw_tau_v_abs_pre, upd_raw_tau_v_abs_post,
             upd_raw_tau_v_scale, upd_raw_tau_v_hit) = _raw_tau_abs_cap_stats(
                updates, 'v', _tau_update_abs_cap)
            (upd_raw_tau_rst_abs_pre, upd_raw_tau_rst_abs_post,
             upd_raw_tau_rst_scale, upd_raw_tau_rst_hit) = _raw_tau_abs_cap_stats(
                updates, 'rst', _tau_update_abs_cap)
            (upd_scan_attn_abs_pre, upd_scan_attn_abs_post,
             upd_scan_attn_scale, upd_scan_attn_hit) = _abs_cap_stats(
                updates, _is_scan_attn_path, _scan_update_abs_cap)
            (upd_scan_rst_abs_pre, upd_scan_rst_abs_post,
             upd_scan_rst_scale, upd_scan_rst_hit) = _abs_cap_stats(
                updates, _is_scan_rst_path, _scan_update_abs_cap)

            def _cap_control_update(path, u):
                ps = _path_to_str(path)
                if _is_raw_tau_attn_combined_path(ps):
                    if u.ndim > 0 and u.shape[-1] >= 3:
                        out = u
                        out = out.at[..., :2].multiply(
                            upd_raw_tau_qk_scale.astype(u.dtype))
                        out = out.at[..., 2:3].multiply(
                            upd_raw_tau_v_scale.astype(u.dtype))
                        return out
                    return u
                if _is_generic_raw_tau_path(ps):
                    parts = _path_parts(ps)
                    if ('rst' in parts) or ('know' in parts):
                        return u * upd_raw_tau_rst_scale.astype(u.dtype)
                    if ('v' in parts) or ('attn_v' in parts):
                        return u * upd_raw_tau_v_scale.astype(u.dtype)
                    if ('qk' in parts) or ('attn_qk' in parts):
                        return u * upd_raw_tau_qk_scale.astype(u.dtype)
                    if u.ndim > 0 and u.shape[-1] >= 3:
                        out = u
                        out = out.at[..., :2].multiply(
                            upd_raw_tau_qk_scale.astype(u.dtype))
                        out = out.at[..., 2:3].multiply(
                            upd_raw_tau_v_scale.astype(u.dtype))
                        return out
                    return u * upd_raw_tau_qk_scale.astype(u.dtype)
                if _is_raw_tau_qk_path(ps):
                    return u * upd_raw_tau_qk_scale.astype(u.dtype)
                if _is_raw_tau_v_path(ps):
                    return u * upd_raw_tau_v_scale.astype(u.dtype)
                if _is_raw_tau_rst_path(ps):
                    return u * upd_raw_tau_rst_scale.astype(u.dtype)
                scale = jnp.float32(1.0)
                scale = jnp.where(_is_router_proj_attn_path(ps), upd_proj_attn_scale, scale)
                scale = jnp.where(_is_router_proj_rst_path(ps), upd_proj_rst_scale, scale)
                scale = jnp.where(_is_op_key_qk_path(ps), upd_op_key_qk_scale, scale)
                scale = jnp.where(_is_op_key_v_path(ps), upd_op_key_v_scale, scale)
                scale = jnp.where(_is_op_key_rst_path(ps), upd_op_key_rst_scale, scale)
                scale = jnp.where(_is_tau_attn_bias_path(ps), upd_tau_attn_scale, scale)
                scale = jnp.where(_is_tau_rst_bias_path(ps), upd_tau_rst_scale, scale)
                scale = jnp.where(_is_scan_attn_path(ps), upd_scan_attn_scale, scale)
                scale = jnp.where(_is_scan_rst_path(ps), upd_scan_rst_scale, scale)
                return u * scale.astype(u.dtype)

            updates = jax.tree.map_with_path(_cap_control_update, updates)
        else:
            upd_proj_attn_ratio_pre = upd_proj_attn_ratio_post = jnp.float32(0.0)
            upd_proj_attn_scale = jnp.float32(1.0)
            upd_proj_attn_hit = jnp.float32(0.0)
            upd_proj_rst_ratio_pre = upd_proj_rst_ratio_post = jnp.float32(0.0)
            upd_proj_rst_scale = jnp.float32(1.0)
            upd_proj_rst_hit = jnp.float32(0.0)
            upd_op_key_qk_ratio_pre = upd_op_key_qk_ratio_post = jnp.float32(0.0)
            upd_op_key_qk_scale = jnp.float32(1.0)
            upd_op_key_qk_hit = jnp.float32(0.0)
            upd_op_key_v_ratio_pre = upd_op_key_v_ratio_post = jnp.float32(0.0)
            upd_op_key_v_scale = jnp.float32(1.0)
            upd_op_key_v_hit = jnp.float32(0.0)
            upd_op_key_rst_ratio_pre = upd_op_key_rst_ratio_post = jnp.float32(0.0)
            upd_op_key_rst_scale = jnp.float32(1.0)
            upd_op_key_rst_hit = jnp.float32(0.0)
            upd_tau_attn_abs_pre = upd_tau_attn_abs_post = jnp.float32(0.0)
            upd_tau_attn_scale = jnp.float32(1.0)
            upd_tau_attn_hit = jnp.float32(0.0)
            upd_tau_rst_abs_pre = upd_tau_rst_abs_post = jnp.float32(0.0)
            upd_tau_rst_scale = jnp.float32(1.0)
            upd_tau_rst_hit = jnp.float32(0.0)
            upd_raw_tau_qk_abs_pre = upd_raw_tau_qk_abs_post = jnp.float32(0.0)
            upd_raw_tau_qk_scale = jnp.float32(1.0)
            upd_raw_tau_qk_hit = jnp.float32(0.0)
            upd_raw_tau_v_abs_pre = upd_raw_tau_v_abs_post = jnp.float32(0.0)
            upd_raw_tau_v_scale = jnp.float32(1.0)
            upd_raw_tau_v_hit = jnp.float32(0.0)
            upd_raw_tau_rst_abs_pre = upd_raw_tau_rst_abs_post = jnp.float32(0.0)
            upd_raw_tau_rst_scale = jnp.float32(1.0)
            upd_raw_tau_rst_hit = jnp.float32(0.0)
            upd_scan_attn_abs_pre = upd_scan_attn_abs_post = jnp.float32(0.0)
            upd_scan_attn_scale = jnp.float32(1.0)
            upd_scan_attn_hit = jnp.float32(0.0)
            upd_scan_rst_abs_pre = upd_scan_rst_abs_post = jnp.float32(0.0)
            upd_scan_rst_scale = jnp.float32(1.0)
            upd_scan_rst_hit = jnp.float32(0.0)

        new_params = optax.apply_updates(params, updates)

        def _tree_sq(tree):
            leaves = jax.tree.leaves(tree)
            if not leaves:
                return jnp.float32(0.0)
            return sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                       for x in leaves)

        def _tree_norm(tree):
            return jnp.sqrt(_tree_sq(tree) + 1e-12)

        def _child_norm(tree, key):
            return _tree_norm(tree[key]) if key in tree else jnp.float32(0.0)

        def _pool_partition_norm(tree, prefix, kind):
            return (
                _child_norm(tree, f'{prefix}_{kind}')
                + _child_norm(tree, f'{prefix}_{kind}_global')
                + _child_norm(tree, f'{prefix}_{kind}_stage')
                + _child_norm(tree, f'{prefix}_{kind}_local'))

        def _path_tree(tree, *keys):
            cur = tree
            for key in keys:
                if not hasattr(cur, '__contains__') or key not in cur:
                    return {}
                cur = cur[key]
            return cur

        grad_norm = _tree_norm(grads)
        if float(global_grad_clip) > 0.0:
            grad_global_postclip = jnp.minimum(grad_norm, _global_grad_clip)
        else:
            grad_global_postclip = grad_norm

        _grouter = grads.get('router', {})
        _gpool = grads.get('neuron_pool', {})
        grad_router_proj_attn = (
            _child_norm(_grouter, 'proj_attn')
            + _child_norm(_grouter, 'q_op_write_query_proj')
            + _child_norm(_grouter, 'k_op_write_query_proj')
            + _child_norm(_grouter, 'v_op_write_query_proj'))
        grad_router_proj_rst = (
            _child_norm(_grouter, 'proj_rst')
            + _child_norm(_grouter, 'rst_op_write_query_proj'))
        grad_router_raw_tau_qk = jnp.sqrt(
            _raw_tau_component_sq(grads, 'qk') + 1e-12)
        grad_router_raw_tau_v = jnp.sqrt(
            _raw_tau_component_sq(grads, 'v') + 1e-12)
        grad_router_raw_tau_rst = jnp.sqrt(
            _raw_tau_component_sq(grads, 'rst') + 1e-12)
        grad_router_tau_attn = (
            _child_norm(_grouter, 'tau_attn')
            + grad_router_raw_tau_qk + grad_router_raw_tau_v)
        grad_router_tau_rst = (
            _child_norm(_grouter, 'tau_rst')
            + grad_router_raw_tau_rst)
        grad_router_scan_attn = _child_norm(_grouter, 'raw_scan_offset_attn')
        grad_router_scan_rst = _child_norm(_grouter, 'raw_scan_offset_rst')
        grad_pool_attn_qk_emb = _child_norm(_gpool, 'attn_qk_emb')
        grad_pool_attn_qk_op_key = (
            _child_norm(_gpool, 'attn_qk_op_read_proj')
            + _child_norm(_gpool, 'attn_qk_op_write_proj'))
        grad_pool_attn_qk_read = _pool_partition_norm(
            _gpool, 'attn_qk', 'read')
        grad_pool_attn_qk_write = _pool_partition_norm(
            _gpool, 'attn_qk', 'write')
        grad_pool_attn_v_emb = _child_norm(_gpool, 'attn_v_emb')
        grad_pool_attn_v_op_key = (
            _child_norm(_gpool, 'attn_v_op_read_proj')
            + _child_norm(_gpool, 'attn_v_op_write_proj'))
        grad_pool_attn_v_read = _pool_partition_norm(
            _gpool, 'attn_v', 'read')
        grad_pool_attn_v_write = _pool_partition_norm(
            _gpool, 'attn_v', 'write')
        grad_pool_rst_emb = _child_norm(_gpool, 'rst_emb')
        grad_pool_rst_op_key = (
            _child_norm(_gpool, 'rst_op_read_proj')
            + _child_norm(_gpool, 'rst_op_write_proj'))
        grad_pool_rst_read = _pool_partition_norm(_gpool, 'rst', 'read')
        grad_pool_rst_write = _pool_partition_norm(_gpool, 'rst', 'write')
        if False:
            grad_token_emb = _tree_norm(_path_tree(grads, 'token_emb'))
            grad_pos_emb = _tree_norm(_path_tree(grads, 'pos_emb'))
            grad_pool_scales = jnp.sqrt(
                _tree_sq(_path_tree(_gpool, 'attn_qk_scale'))
                + _tree_sq(_path_tree(_gpool, 'attn_v_scale'))
                + _tree_sq(_path_tree(_gpool, 'rst_scale'))
                + 1e-12)
            grad_expand_O_sq = jnp.float32(0.0)
            grad_layernorms_sq = _tree_sq(_path_tree(grads, 'norm'))
            _grad_expand_layers = []
            _grad_ln_layers = []
            for _i in range(getattr(model, 'n_layers', 0)):
                _gb = _path_tree(grads, f'block_{_i}')
                _expand_sq_i = _tree_sq(_path_tree(_gb, 'attn', 'expand_O'))
                _ln_sq_i = (
                    _tree_sq(_path_tree(_gb, 'norm1'))
                    + _tree_sq(_path_tree(_gb, 'norm2')))
                grad_expand_O_sq = (
                    grad_expand_O_sq
                    + _expand_sq_i)
                grad_layernorms_sq = (
                    grad_layernorms_sq
                    + _ln_sq_i)
                _grad_expand_layers.append(jnp.sqrt(_expand_sq_i + 1e-12))
                _grad_ln_layers.append(jnp.sqrt(_ln_sq_i + 1e-12))
            grad_expand_O = jnp.sqrt(grad_expand_O_sq + 1e-12)
            grad_layernorms = jnp.sqrt(grad_layernorms_sq + 1e-12)
            grad_lm_head_or_token_tied = grad_token_emb
            grad_expand_O_per_layer = (
                jnp.stack(_grad_expand_layers)
                if _grad_expand_layers else jnp.zeros((1,), dtype=jnp.float32))
            grad_layernorms_per_layer = (
                jnp.stack(_grad_ln_layers)
                if _grad_ln_layers else jnp.zeros((1,), dtype=jnp.float32))
            grad_router_proj_attn_per_layer = jnp.asarray(
                [grad_router_proj_attn], dtype=jnp.float32)
            grad_router_proj_rst_per_layer = jnp.asarray(
                [grad_router_proj_rst], dtype=jnp.float32)
            grad_router_tau_attn_per_layer = jnp.asarray(
                [grad_router_tau_attn], dtype=jnp.float32)
            grad_router_tau_rst_per_layer = jnp.asarray(
                [grad_router_tau_rst], dtype=jnp.float32)
            grad_router_raw_tau_qk_per_layer = jnp.asarray(
                [grad_router_raw_tau_qk], dtype=jnp.float32)
            grad_router_raw_tau_v_per_layer = jnp.asarray(
                [grad_router_raw_tau_v], dtype=jnp.float32)
            grad_router_raw_tau_rst_per_layer = jnp.asarray(
                [grad_router_raw_tau_rst], dtype=jnp.float32)
            grad_pool_attn_qk_rw = jnp.asarray(
                [grad_pool_attn_qk_read, grad_pool_attn_qk_write],
                dtype=jnp.float32)
            grad_pool_attn_v_rw = jnp.asarray(
                [grad_pool_attn_v_read, grad_pool_attn_v_write],
                dtype=jnp.float32)
            grad_pool_rst_rw = jnp.asarray(
                [grad_pool_rst_read, grad_pool_rst_write],
                dtype=jnp.float32)
        else:
            grad_token_emb = jnp.float32(0.0)
            grad_pos_emb = jnp.float32(0.0)
            grad_pool_scales = jnp.float32(0.0)
            grad_expand_O = jnp.float32(0.0)
            grad_layernorms = jnp.float32(0.0)
            grad_lm_head_or_token_tied = jnp.float32(0.0)
            grad_expand_O_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_layernorms_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_proj_attn_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_proj_rst_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_tau_attn_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_tau_rst_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_raw_tau_qk_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_raw_tau_v_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_router_raw_tau_rst_per_layer = jnp.zeros((1,), dtype=jnp.float32)
            grad_pool_attn_qk_rw = jnp.zeros((2,), dtype=jnp.float32)
            grad_pool_attn_v_rw = jnp.zeros((2,), dtype=jnp.float32)
            grad_pool_rst_rw = jnp.zeros((2,), dtype=jnp.float32)
        grad_router_proj = (
            grad_router_proj_attn + grad_router_proj_rst)
        grad_router_tau = (
            grad_router_tau_attn + grad_router_tau_rst)
        grad_router_scan = (
            grad_router_scan_attn + grad_router_scan_rst)
        grad_pool_emb = (
            grad_pool_attn_qk_emb + grad_pool_attn_v_emb
            + grad_pool_rst_emb)
        grad_pool_op_key = (
            grad_pool_attn_qk_op_key + grad_pool_attn_v_op_key
            + grad_pool_rst_op_key)
        grad_pool_read = (
            grad_pool_attn_qk_read + grad_pool_attn_v_read
            + grad_pool_rst_read)
        grad_pool_write = (
            grad_pool_attn_qk_write + grad_pool_attn_v_write
            + grad_pool_rst_write)
        _ppool = params.get('neuron_pool', {})
        if False:
            pool_weight_decay_loss = jnp.float32(0.5 * pool_weight_decay) * (
                _tree_sq(_path_tree(_ppool, 'attn_qk_emb'))
                + _tree_sq(_path_tree(_ppool, 'attn_qk_read'))
                + _tree_sq(_path_tree(_ppool, 'attn_qk_write'))
                + _tree_sq(_path_tree(_ppool, 'attn_v_emb'))
                + _tree_sq(_path_tree(_ppool, 'attn_v_read'))
                + _tree_sq(_path_tree(_ppool, 'attn_v_write'))
                + _tree_sq(_path_tree(_ppool, 'rst_emb'))
                + _tree_sq(_path_tree(_ppool, 'rst_read'))
                + _tree_sq(_path_tree(_ppool, 'rst_write')))
            normal_weight_decay_loss = (
                jnp.float32(0.5 * weight_decay)
                * jnp.maximum(_tree_sq(params) - _tree_sq(_ppool),
                              jnp.float32(0.0)))
        else:
            pool_weight_decay_loss = jnp.float32(0.0)
            normal_weight_decay_loss = jnp.float32(0.0)
        if False:
            pool_diag = _pool_param_diagnostics(params, full=False, model=model)
            pool_update_diag = _pool_update_diagnostics(params, grads)
        else:
            pool_diag = {}
            pool_update_diag = {}

        # Operator-key drift is diagnostic-only. It is computed inside jit so
        # every host participates in the same reductions on multi-host meshes.
        if (not _train_drift_diagnostics) or _is_baseline_model:
            drift_attn_qk_op_key = jnp.float32(0.0)
            drift_attn_v_op_key = jnp.float32(0.0)
            drift_rst_op_key = jnp.float32(0.0)
        elif 'neuron_pool' not in new_params:
            drift_attn_qk_op_key = jnp.float32(0.0)
            drift_attn_v_op_key = jnp.float32(0.0)
            drift_rst_op_key = jnp.float32(0.0)
        else:
            _pool = new_params['neuron_pool']
            def _drift_unit(x):
                x = jnp.asarray(x, dtype=jnp.float32)
                return x / (
                    jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

            def _drift_op_key(read, write, read_proj, write_proj):
                r_key = _drift_unit(read) @ read_proj
                w_key = _drift_unit(write) @ write_proj
                return _drift_unit(_drift_unit(r_key) * _drift_unit(w_key))

            if ('attn_qk_read_global' in _pool
                    or 'attn_qk_read_shared' in _pool):
                def _flat_partitioned_op_key(prefix):
                    if f'{prefix}_read_global' in _pool:
                        parts = [_drift_op_key(
                            _pool[f'{prefix}_read_global'],
                            _pool[f'{prefix}_write_global'],
                            _pool[f'{prefix}_op_read_proj'],
                            _pool[f'{prefix}_op_write_proj'])]
                        for _scope in ('stage', 'local'):
                            read_key = f'{prefix}_read_{_scope}'
                            write_key = f'{prefix}_write_{_scope}'
                            if read_key in _pool:
                                op_key = _drift_op_key(
                                    _pool[read_key], _pool[write_key],
                                    _pool[f'{prefix}_op_read_proj'],
                                    _pool[f'{prefix}_op_write_proj'])
                                parts.append(op_key.reshape(
                                    (-1, parts[0].shape[-1])))
                        return jnp.concatenate(parts, axis=0)

                    op_shared = _drift_op_key(
                        _pool[f'{prefix}_read_shared'],
                        _pool[f'{prefix}_write_shared'],
                        _pool[f'{prefix}_op_read_proj'],
                        _pool[f'{prefix}_op_write_proj'])
                    read_stage_key = f'{prefix}_read_stage'
                    write_stage_key = f'{prefix}_write_stage'
                    if read_stage_key not in _pool:
                        return op_shared
                    op_stage = _drift_op_key(
                        _pool[read_stage_key],
                        _pool[write_stage_key],
                        _pool[f'{prefix}_op_read_proj'],
                        _pool[f'{prefix}_op_write_proj'])
                    return jnp.concatenate(
                        [op_shared,
                         op_stage.reshape((-1, op_shared.shape[-1]))],
                        axis=0)

                _cur_qk = _flat_partitioned_op_key('attn_qk')
                _cur_v = _flat_partitioned_op_key('attn_v')
                _cur_rst = _flat_partitioned_op_key('rst')
            elif 'attn_qk_op_read_proj' in _pool:
                _cur_qk = _drift_op_key(
                    _pool['attn_qk_read'], _pool['attn_qk_write'],
                    _pool['attn_qk_op_read_proj'],
                    _pool['attn_qk_op_write_proj'])
                _cur_v = _drift_op_key(
                    _pool['attn_v_read'], _pool['attn_v_write'],
                    _pool['attn_v_op_read_proj'],
                    _pool['attn_v_op_write_proj'])
                _cur_rst = _drift_op_key(
                    _pool['rst_read'], _pool['rst_write'],
                    _pool['rst_op_read_proj'], _pool['rst_op_write_proj'])
            elif 'attn_qk_emb' in _pool:
                _cur_qk = _pool['attn_qk_emb']
                _cur_v = _pool['attn_v_emb']
                _cur_rst = _pool['rst_emb']
            elif 'qk_emb' in _pool:
                _cur_qk = _pool['qk_emb']
                _cur_v = _pool['v_emb']
                _cur_rst = _pool['rst_emb']
            else:
                # Some archived pool variants expose read tensors instead of emb
                # tensors; keep the diagnostic slots comparable when resuming them.
                _cur_qk = _pool['q_read']
                _cur_v = _pool['v_read']
                _cur_rst = _pool['rst_read']
            _prev_qk = prev_op_key_snap['attn_qk_op_key']
            _prev_v = prev_op_key_snap['attn_v_op_key']
            _prev_rst = prev_op_key_snap['rst_op_key']
            drift_attn_qk_op_key = (jnp.linalg.norm(_cur_qk - _prev_qk)
                            / (jnp.linalg.norm(_prev_qk) + 1e-8))
            drift_attn_v_op_key = (jnp.linalg.norm(_cur_v - _prev_v)
                           / (jnp.linalg.norm(_prev_v) + 1e-8))
            drift_rst_op_key = (jnp.linalg.norm(_cur_rst - _prev_rst)
                                / (jnp.linalg.norm(_prev_rst) + 1e-8))
        drift_attn_qk_emb = (
            jnp.float32(0.0) if _is_v4166_model else drift_attn_qk_op_key)
        drift_attn_v_emb = (
            jnp.float32(0.0) if _is_v4166_model else drift_attn_v_op_key)
        drift_rst_emb = (
            jnp.float32(0.0) if _is_v4166_model else drift_rst_op_key)

        # Tau / scan-offset parameters (read inside jit -safe, no cross-device issue)
        tau_rst_b = params.get('router', {}).get(
            'tau_rst', params.get('router', {}).get('tau_rst', {})).get(
                'bias', jnp.zeros(1))
        tau_attn_b = params.get('router', {}).get('tau_attn', {}).get(
            'bias', jnp.zeros(3))
        tau_q_b = params.get('router', {}).get('tau_q', {}).get(
            'bias', jnp.zeros(1))
        tau_k_b = params.get('router', {}).get('tau_k', {}).get(
            'bias', jnp.zeros(1))
        tau_v_b = params.get('router', {}).get('tau_v', {}).get(
            'bias', jnp.zeros(1))
        raw_scan_offset_rst_b = params.get('router', {}).get(
            'raw_scan_offset_rst',
            params.get('router', {}).get('raw_scan_offset_rst', {})).get(
                'bias', jnp.zeros(1))
        raw_scan_offset_attn_b = params.get('router', {}).get(
            'raw_scan_offset_attn',
            params.get('router', {}).get('raw_scan_offset_attn', {})).get(
                'bias', jnp.zeros(3))
        # loss_fn-local weighted aux variables are returned through
        inactive_aux_loss_weighted_metric = inactive_aux_stats['inactive_aux_loss_weighted_clipped']
        inactive_aux_loss_weighted_unclipped_metric = inactive_aux_stats['inactive_aux_loss_weighted_unclipped']
        cb1a_weighted_metric = inactive_aux_stats['cb1a_weighted']
        dead_penalty_weighted_metric = inactive_aux_stats['dead_penalty_weighted_clipped']
        dead_penalty_weighted_unclipped_metric = inactive_aux_stats['dead_penalty_weighted_unclipped']

        if _is_baseline_model:
            aux_loss_weighted_metric = jnp.float32(0.0)
            load_balance_loss_weighted_metric = jnp.float32(0.0)
            tau_reg_weighted_metric = jnp.float32(0.0)
            orth_loss_weighted_metric = jnp.float32(0.0)
            diversity_loss_weighted_metric = jnp.float32(0.0)
            inactive_aux_loss_weighted_metric = jnp.float32(0.0)
            inactive_aux_loss_weighted_unclipped_metric = jnp.float32(0.0)
            cb1a_weighted_metric = jnp.float32(0.0)
            dead_penalty_weighted_metric = jnp.float32(0.0)
            dead_penalty_weighted_unclipped_metric = jnp.float32(0.0)
        elif _is_soft_direct_tau:
            aux_loss_weighted_metric = jnp.float32(0.0)
            load_balance_loss_weighted_metric = jnp.float32(0.0)
            tau_reg_weighted_metric = jnp.float32(0.0)
            orth_loss_weighted_metric = jnp.float32(0.0)
            diversity_loss_weighted_metric = jnp.float32(0.0)
            cb1a_weighted_metric = jnp.float32(0.0)
            dead_penalty_weighted_metric = jnp.float32(0.0)
            dead_penalty_weighted_unclipped_metric = jnp.float32(0.0)
        else:
            aux_loss_weighted_metric = lb_weight * aux_loss
            load_balance_loss_weighted_metric = lb_weight * aux_loss
            tau_reg_weighted_metric = tau_reg_weight * tau_reg
            orth_loss_weighted_metric = orth_weight * orth_loss
            diversity_loss_weighted_metric = div_weight * div_loss

        reconstructed_total_loss_metric = (
            ce_loss
            + aux_loss_weighted_metric
            + tau_reg_weighted_metric
            + orth_loss_weighted_metric
            + diversity_loss_weighted_metric
            + dead_penalty_weighted_metric
            + inactive_aux_loss_weighted_metric
            + cb1a_weighted_metric)
        metrics = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'aux_loss': aux_loss,
            'aux_loss_raw': aux_loss,
            'aux_loss_weighted': aux_loss_weighted_metric,
            'load_balance_loss_raw': aux_loss,
            'load_balance_loss_weighted': load_balance_loss_weighted_metric,
            'tau_reg': tau_reg,
            'tau_reg_weighted': tau_reg_weighted_metric,
            'orth_loss': orth_loss,
            'orth_loss_weighted': orth_loss_weighted_metric,
            'div_loss': div_loss,
            'diversity_loss_raw': div_loss,
            'diversity_loss_weighted': diversity_loss_weighted_metric,
            'dead_penalty_weight': jnp.float32(dead_penalty_weight),
            'dead_penalty_weighted_total': dead_penalty_weighted_metric,
            'dead_penalty_raw_unweighted': inactive_aux_stats['dead_penalty_raw_unweighted'],
            'dead_penalty_raw_weighted_pools': inactive_aux_stats['dead_penalty_raw_weighted_pools'],
            'dead_penalty_qk_weight': inactive_aux_stats['dead_penalty_qk_weight'],
            'dead_penalty_v_weight': inactive_aux_stats['dead_penalty_v_weight'],
            'dead_penalty_rst_weight': inactive_aux_stats['dead_penalty_rst_weight'],
            'attn_qk_dead_penalty': result.get(
                'attn_qk_dead_penalty', jnp.float32(0.0)),
            'attn_v_dead_penalty': result.get(
                'attn_v_dead_penalty', jnp.float32(0.0)),
            'attn_qk_dead_count': result.get(
                'attn_qk_dead_count', jnp.float32(0.0)),
            'attn_v_dead_count': result.get(
                'attn_v_dead_count', jnp.float32(0.0)),
            'inactive_aux_warmup_factor': inactive_aux_stats['inactive_aux_active'],
            'inactive_aux_weight_effective': (
                (_inactive_aux_weight_q + _inactive_aux_weight_k
                 + _inactive_aux_weight_v + _inactive_aux_weight_rst)
                / jnp.float32(4.0) * inactive_aux_stats['inactive_aux_active']),
            'soft_gate_T': inactive_aux_stats['soft_gate_T'],
            'soft_gate_T_qk': inactive_aux_stats['soft_gate_T_qk'],
            'soft_gate_T_v': inactive_aux_stats['soft_gate_T_v'],
            'soft_gate_T_rst': inactive_aux_stats['soft_gate_T_rst'],
            'boundary_power_p': inactive_aux_stats['boundary_power_p'],
            'admission_den_power': inactive_aux_stats['admission_den_power'],
            'inactive_aux_effective_weight': (
                (_inactive_aux_weight_q + _inactive_aux_weight_k
                 + _inactive_aux_weight_v + _inactive_aux_weight_rst)
                / jnp.float32(4.0) * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_schedule_scale': inactive_aux_stats['inactive_aux_schedule_scale'],
            'inactive_aux_asymmetry': _asym,
            'inactive_aux_asymmetry_q': _asym_q,
            'inactive_aux_asymmetry_k': _asym_k,
            'inactive_aux_asymmetry_qk': (
                (_asym_q + _asym_k) / jnp.float32(2.0)),
            'inactive_aux_asymmetry_v': _asym_v,
            'inactive_aux_asymmetry_rst': _asym_rst,
            'inactive_aux_layer_norm_enabled': inactive_aux_stats[
                'inactive_aux_layer_norm_enabled'],
            'inactive_aux_layer_count': inactive_aux_stats['inactive_aux_layer_count'],
            'inactive_aux_norm': inactive_aux_stats['inactive_aux_norm'],
            'inactive_aux_no_active_easy_shutoff_q': inactive_aux_stats[
                'inactive_aux_no_active_easy_shutoff_q'],
            'inactive_aux_no_active_easy_shutoff_k': inactive_aux_stats[
                'inactive_aux_no_active_easy_shutoff_k'],
            'inactive_aux_no_active_easy_shutoff_v': inactive_aux_stats[
                'inactive_aux_no_active_easy_shutoff_v'],
            'inactive_aux_no_active_easy_shutoff_rst': inactive_aux_stats[
                'inactive_aux_no_active_easy_shutoff_rst'],
            'inactive_aux_loss_raw_total': inactive_aux_stats['inactive_aux_loss_raw'],
            'inactive_aux_loss_raw_q': inactive_aux_stats['inactive_aux_q_raw'],
            'inactive_aux_loss_raw_k': inactive_aux_stats['inactive_aux_k_raw'],
            'inactive_aux_loss_raw_qk': inactive_aux_stats['inactive_aux_qk_raw'],
            'inactive_aux_loss_raw_v': inactive_aux_stats['inactive_aux_v_raw'],
            'inactive_aux_loss_raw_attn': inactive_aux_stats['inactive_aux_attn_raw'],
            'inactive_aux_loss_raw_rst': inactive_aux_stats['inactive_aux_rst_raw'],
            'inactive_aux_loss_weighted_q': (
                _inactive_aux_weight_q * inactive_aux_stats['inactive_aux_q_raw']
                * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_loss_weighted_k': (
                _inactive_aux_weight_k * inactive_aux_stats['inactive_aux_k_raw']
                * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_loss_weighted_qk': (
                (_inactive_aux_weight_q * inactive_aux_stats['inactive_aux_q_raw']
                 + _inactive_aux_weight_k * inactive_aux_stats['inactive_aux_k_raw'])
                * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_loss_weighted_v': (
                _inactive_aux_weight_v * inactive_aux_stats['inactive_aux_v_raw']
                * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_loss_weighted_attn': (
                (_inactive_aux_weight_q * inactive_aux_stats['inactive_aux_q_raw']
                 + _inactive_aux_weight_k * inactive_aux_stats['inactive_aux_k_raw']
                 + _inactive_aux_weight_v * inactive_aux_stats['inactive_aux_v_raw'])
                * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_loss_weighted_rst': (
                _inactive_aux_weight_rst * inactive_aux_stats['inactive_aux_rst_raw']
                * inactive_aux_stats['inactive_aux_active']),
            'inactive_aux_loss_weighted_total': inactive_aux_loss_weighted_metric,
            'inactive_aux_loss_weighted_unclipped': inactive_aux_loss_weighted_unclipped_metric,
            'inactive_aux_loss_weighted_clipped': inactive_aux_loss_weighted_metric,
            'cb1a_raw': inactive_aux_stats['cb1a_raw'],
            'cb1a_w': cb1a_weighted_metric,
            'cb1a_weighted': cb1a_weighted_metric,
            'cb1a_challenge_raw': inactive_aux_stats['cb1a_challenge_raw'],
            'cb1a_prune_raw': inactive_aux_stats['cb1a_prune_raw'],
            'cb1a_challenge': result.get(
                'cb1a_challenge', inactive_aux_stats['cb1a_challenge_raw']),
            'cb1a_prune': result.get(
                'cb1a_prune', inactive_aux_stats['cb1a_prune_raw']),
            'cb1a_valid': result.get('cb1a_valid', jnp.float32(0.0)),
            'cb1a_has_above': result.get('cb1a_has_above', jnp.float32(0.0)),
            'cb1a_has_below': result.get('cb1a_has_below', jnp.float32(0.0)),
            'cb1a_qk_challenge': inactive_aux_stats['cb1a_qk_challenge_raw'],
            'cb1a_qk_prune': inactive_aux_stats['cb1a_qk_prune_raw'],
            'cb1a_v_challenge': inactive_aux_stats['cb1a_v_challenge_raw'],
            'cb1a_v_prune': inactive_aux_stats['cb1a_v_prune_raw'],
            'cb1a_rst_challenge': inactive_aux_stats['cb1a_rst_challenge_raw'],
            'cb1a_rst_prune': inactive_aux_stats['cb1a_rst_prune_raw'],
            'cb1a_qk_raw': inactive_aux_stats['cb1a_qk_raw'],
            'cb1a_v_raw': inactive_aux_stats['cb1a_v_raw'],
            'cb1a_rst_raw': inactive_aux_stats['cb1a_rst_raw'],
            'cb1a_weight': _cb1a_weight,
            'cb1a_challenge_weight': _cb1a_challenge_weight,
            'cb1a_prune_weight': _cb1a_prune_weight,
            'cb1a_qk_weight': _cb1a_qk_weight,
            'cb1a_v_weight': _cb1a_v_weight,
            'cb1a_rst_weight': _cb1a_rst_weight,
            'cb1a_qk_challenge_weight': inactive_aux_stats[
                'cb1a_qk_challenge_weight'],
            'cb1a_qk_prune_weight': inactive_aux_stats['cb1a_qk_prune_weight'],
            'cb1a_v_challenge_weight': inactive_aux_stats[
                'cb1a_v_challenge_weight'],
            'cb1a_v_prune_weight': inactive_aux_stats['cb1a_v_prune_weight'],
            'cb1a_rst_challenge_weight': inactive_aux_stats[
                'cb1a_rst_challenge_weight'],
            'cb1a_rst_prune_weight': inactive_aux_stats['cb1a_rst_prune_weight'],
            'cb1a_ce_mean': inactive_aux_stats['cb1a_ce_mean'],
            'cb1a_ce_std': inactive_aux_stats['cb1a_ce_std'],
            'dead_penalty_weighted_unclipped': dead_penalty_weighted_unclipped_metric,
            'dead_penalty_weighted_clipped': dead_penalty_weighted_metric,
            'inactive_aux_raw_pre_bound': inactive_aux_stats['inactive_aux_loss_pre_bound'],
            'inactive_aux_q_pre_bound': inactive_aux_stats['inactive_aux_q_pre_bound'],
            'inactive_aux_k_pre_bound': inactive_aux_stats['inactive_aux_k_pre_bound'],
            'inactive_aux_qk_pre_bound': inactive_aux_stats['inactive_aux_qk_pre_bound'],
            'inactive_aux_v_pre_bound': inactive_aux_stats['inactive_aux_v_pre_bound'],
            'inactive_aux_raw_post_bound': inactive_aux_stats['inactive_aux_loss_raw'],
            'inactive_aux_attn_pre_bound': inactive_aux_stats['inactive_aux_attn_pre_bound'],
            'inactive_aux_rst_pre_bound': inactive_aux_stats['inactive_aux_rst_pre_bound'],
            'pool_weight_decay_loss': pool_weight_decay_loss,
            'normal_weight_decay_loss': normal_weight_decay_loss,
            'total_loss_minus_ce': total_loss - ce_loss,
            'reconstructed_total_loss': reconstructed_total_loss_metric,
            'reconstructed_loss_error': jnp.abs(
                total_loss - reconstructed_total_loss_metric),
            'dead_loss_raw': dead_penalty,
            'dead_loss_weighted': dead_penalty_weighted_metric,
            'inactive_aux_loss_raw': inactive_aux_stats['inactive_aux_loss_raw'],
            'inactive_aux_loss_weighted': inactive_aux_loss_weighted_metric,
            'weight_decay_pool': pool_weight_decay_loss,
            'weight_decay_normal': normal_weight_decay_loss,
            'correct': result['correct'],
            'valid_count': result['valid_count'],
            'grad_norm': grad_norm,
            'grad_global_preclip': grad_norm,
            'grad_global_postclip': grad_global_postclip,
            'grad_token_emb': grad_token_emb,
            'grad_pos_emb': grad_pos_emb,
            'grad_router_proj_attn': grad_router_proj_attn,
            'grad_router_proj_rst': grad_router_proj_rst,
            'grad_router_raw_tau_qk': grad_router_raw_tau_qk,
            'grad_router_raw_tau_v': grad_router_raw_tau_v,
            'grad_router_raw_tau_rst': grad_router_raw_tau_rst,
            'grad_router_tau_attn': grad_router_tau_attn,
            'grad_router_tau_rst': grad_router_tau_rst,
            'grad_router_scan_attn': grad_router_scan_attn,
            'grad_router_scan_rst': grad_router_scan_rst,
            'grad_pool_attn_qk_emb': grad_pool_attn_qk_emb,
            'grad_pool_attn_qk_op_key': grad_pool_attn_qk_op_key,
            'grad_pool_attn_qk_read': grad_pool_attn_qk_read,
            'grad_pool_attn_qk_write': grad_pool_attn_qk_write,
            'grad_pool_attn_v_emb': grad_pool_attn_v_emb,
            'grad_pool_attn_v_op_key': grad_pool_attn_v_op_key,
            'grad_pool_attn_v_read': grad_pool_attn_v_read,
            'grad_pool_attn_v_write': grad_pool_attn_v_write,
            'grad_pool_rst_emb': grad_pool_rst_emb,
            'grad_pool_rst_op_key': grad_pool_rst_op_key,
            'grad_pool_rst_read': grad_pool_rst_read,
            'grad_pool_rst_write': grad_pool_rst_write,
            'grad_pool_scales': grad_pool_scales,
            'grad_expand_O': grad_expand_O,
            'grad_layernorms': grad_layernorms,
            'grad_lm_head_or_token_tied': grad_lm_head_or_token_tied,
            'grad_router_proj_attn_per_layer': grad_router_proj_attn_per_layer,
            'grad_router_proj_rst_per_layer': grad_router_proj_rst_per_layer,
            'grad_router_raw_tau_qk_per_layer': grad_router_raw_tau_qk_per_layer,
            'grad_router_raw_tau_v_per_layer': grad_router_raw_tau_v_per_layer,
            'grad_router_raw_tau_rst_per_layer': grad_router_raw_tau_rst_per_layer,
            'grad_router_tau_attn_per_layer': grad_router_tau_attn_per_layer,
            'grad_router_tau_rst_per_layer': grad_router_tau_rst_per_layer,
            'grad_pool_attn_qk_rw': grad_pool_attn_qk_rw,
            'grad_pool_attn_v_rw': grad_pool_attn_v_rw,
            'grad_pool_rst_rw': grad_pool_rst_rw,
            'grad_expand_O_per_layer': grad_expand_O_per_layer,
            'grad_layernorms_per_layer': grad_layernorms_per_layer,
            'grad_router_proj': grad_router_proj,
            'grad_router_tau': grad_router_tau,
            'grad_router_scan': grad_router_scan,
            'grad_pool_emb': grad_pool_emb,
            'grad_pool_op_key': grad_pool_op_key,
            'grad_pool_read': grad_pool_read,
            'grad_pool_write': grad_pool_write,
            'attn_aux': result.get('attn_aux', jnp.float32(0.0)),
            'rst_aux': result.get('rst_aux', jnp.float32(0.0)),
            'drive_mean': result.get('drive_mean', jnp.float32(0.0)),
            'drive_max': result.get('drive_max', jnp.float32(0.0)),
            'attn_drive_mean': result.get('attn_drive_mean', jnp.float32(0.0)),
            'attn_drive_max': result.get('attn_drive_max', jnp.float32(0.0)),
            'attn_qk_drive_mean': result.get('attn_qk_drive_mean', jnp.float32(0.0)),
            'attn_qk_drive_max': result.get('attn_qk_drive_max', jnp.float32(0.0)),
            'attn_v_drive_mean': result.get('attn_v_drive_mean', jnp.float32(0.0)),
            'attn_v_drive_max': result.get('attn_v_drive_max', jnp.float32(0.0)),
            'rst_drive_mean': result.get('rst_drive_mean', jnp.float32(0.0)),
            'rst_drive_max': result.get('rst_drive_max', jnp.float32(0.0)),
            'weight_mean': result.get('weight_mean', jnp.float32(0.0)),
            'weight_max': result.get('weight_max', jnp.float32(0.0)),
            'attn_qk_weight_mean': result.get('attn_qk_weight_mean', jnp.float32(0.0)),
            'attn_qk_weight_max': result.get('attn_qk_weight_max', jnp.float32(0.0)),
            'attn_v_weight_mean': result.get('attn_v_weight_mean', jnp.float32(0.0)),
            'attn_v_weight_max': result.get('attn_v_weight_max', jnp.float32(0.0)),
            'rst_weight_mean': result.get('rst_weight_mean', jnp.float32(0.0)),
            'rst_weight_max': result.get('rst_weight_max', jnp.float32(0.0)),
            'admission_mean': result.get('admission_mean', jnp.float32(0.0)),
            'admission_max': result.get('admission_max', jnp.float32(0.0)),
            'attn_qk_admission_mean': result.get('attn_qk_admission_mean', jnp.float32(0.0)),
            'attn_qk_admission_max': result.get('attn_qk_admission_max', jnp.float32(0.0)),
            'attn_v_admission_mean': result.get('attn_v_admission_mean', jnp.float32(0.0)),
            'attn_v_admission_max': result.get('attn_v_admission_max', jnp.float32(0.0)),
            'rst_admission_mean': result.get('rst_admission_mean', jnp.float32(0.0)),
            'rst_admission_max': result.get('rst_admission_max', jnp.float32(0.0)),
            'load_mean': result.get('load_mean', jnp.float32(0.0)),
            'normalization_load_mean': result.get('normalization_load_mean', jnp.float32(0.0)),
            'den_mean': result.get('den_mean', jnp.float32(0.0)),
            # Core v4164 activity.
            'rst_active': result.get('rst_active', jnp.float32(0.0)),
            'rst_strong': result.get('rst_strong', jnp.float32(0.0)),
            'rst_score_std': result.get(
                'rst_score_std', result.get('rst_rho_std', jnp.float32(0.0))),
            'rst_raw_gate_max': result.get('rst_raw_gate_max', jnp.float32(0.0)),
            'rst_gate_sum': result.get('rst_gate_sum', jnp.float32(0.0)),
            'rst_active_n_mean': result.get('rst_active_n_mean', jnp.float32(0.0)),
            'rst_gate_eff_n': result.get('rst_gate_eff_n', jnp.float32(0.0)),
            'rst_gate_eff_ratio': result.get('rst_gate_eff_ratio', jnp.float32(0.0)),
            'rst_top1_gate_frac': result.get('rst_top1_gate_frac', jnp.float32(0.0)),
            'rst_top1_gate_frac_max': result.get('rst_top1_gate_frac_max', jnp.float32(0.0)),
            'attn_qk_active': result.get('attn_qk_active', jnp.float32(0.0)),
            'attn_q_active_n_mean': result.get(
                'attn_q_active_n_mean',
                result.get('attn_qk_active_n_mean',
                           result.get('attn_active_n_mean', jnp.float32(0.0)))),
            'attn_k_active_n_mean': result.get(
                'attn_k_active_n_mean',
                result.get('attn_qk_active_n_mean',
                           result.get('attn_active_n_mean', jnp.float32(0.0)))),
            'attn_q_active': result.get(
                'attn_q_active',
                result.get('attn_qk_active', jnp.float32(0.0))),
            'attn_k_active': result.get(
                'attn_k_active',
                result.get('attn_qk_active', jnp.float32(0.0))),
            'attn_v_active': result.get('attn_v_active', jnp.float32(0.0)),
            'attn_strong': result.get('attn_strong', jnp.float32(0.0)),
            'attn_qk_strong': result.get(
                'attn_qk_strong',
                result.get('attn_strong', jnp.float32(0.0))),
            'attn_q_strong': result.get(
                'attn_q_strong',
                result.get('attn_qk_strong',
                           result.get('attn_strong', jnp.float32(0.0)))),
            'attn_k_strong': result.get(
                'attn_k_strong',
                result.get('attn_qk_strong',
                           result.get('attn_strong', jnp.float32(0.0)))),
            'attn_v_strong': result.get(
                'attn_v_strong',
                result.get('attn_strong', jnp.float32(0.0))),
            'attn_score_std': result.get(
                'attn_score_std', result.get('attn_rho_std', jnp.float32(0.0))),
            'attn_score_mean': result.get('attn_score_mean', jnp.float32(0.0)),
            'attn_raw_gate_max': result.get('attn_raw_gate_max', jnp.float32(0.0)),
            'attn_gate_sum': result.get('attn_gate_sum', jnp.float32(0.0)),
            'attn_active_n_mean': result.get('attn_active_n_mean', jnp.float32(0.0)),
            'attn_gate_eff_n': result.get('attn_gate_eff_n', jnp.float32(0.0)),
            'attn_gate_eff_ratio': result.get('attn_gate_eff_ratio', jnp.float32(0.0)),
            'attn_top1_gate_frac': result.get('attn_top1_gate_frac', jnp.float32(0.0)),
            'attn_top1_gate_frac_max': result.get('attn_top1_gate_frac_max', jnp.float32(0.0)),
            'attn_out_norm': result.get('attn_out_norm', jnp.float32(0.0)),
            # tau structure.
            'attn_tau_mean': result.get('attn_tau_mean', jnp.float32(0.0)),
            'rst_tau_mean': result.get('rst_tau_mean', jnp.float32(0.0)),
            'rst_score_mean': result.get('rst_score_mean', jnp.float32(0.0)),
            'attn_tau_abs_mean': result.get('attn_tau_abs_mean', jnp.float32(0.0)),
            'rst_tau_abs_mean': result.get('rst_tau_abs_mean', jnp.float32(0.0)),
            'attn_rho_mean': result.get('attn_rho_mean', jnp.float32(0.0)),
            'attn_rho_std': result.get('attn_rho_std', jnp.float32(0.0)),
            'attn_rho_max': result.get('attn_rho_max', jnp.float32(0.0)),
            'attn_raw_tau_mean': result.get('attn_raw_tau_mean', result.get('attn_tau_raw_mean', jnp.float32(0.0))),
            'attn_raw_tau_min': result.get('attn_raw_tau_min', result.get('attn_raw_tau_mean', result.get('attn_tau_raw_mean', jnp.float32(0.0)))),
            'attn_raw_tau_max': result.get('attn_raw_tau_max', result.get('attn_raw_tau_mean', result.get('attn_tau_raw_mean', jnp.float32(0.0)))),
            'attn_tau_min': result.get('attn_tau_min', result.get('attn_tau_floor_mean', jnp.float32(0.0))),
            'attn_tau_max': result.get('attn_tau_max', jnp.float32(0.0)),

            'attn_selection_margin_mean': result.get('attn_selection_margin_mean', jnp.float32(0.0)),
            'attn_positive_margin_mean': result.get('attn_positive_margin_mean', jnp.float32(0.0)),
            'attn_positive_margin_max': result.get('attn_positive_margin_max', jnp.float32(0.0)),
            'attn_selected_frac': result.get('attn_selected_frac', jnp.float32(0.0)),
            'attn_no_active_frac': result.get('attn_no_active_frac', jnp.float32(0.0)),
            'rst_rho_mean': result.get('rst_rho_mean', jnp.float32(0.0)),
            'rst_rho_std': result.get('rst_rho_std', jnp.float32(0.0)),
            'rst_rho_max': result.get('rst_rho_max', jnp.float32(0.0)),
            'rst_raw_tau_mean': result.get('rst_raw_tau_mean', result.get('rst_tau_raw_mean', jnp.float32(0.0))),
            'rst_raw_tau_min': result.get('rst_raw_tau_min', result.get('rst_raw_tau_mean', result.get('rst_tau_raw_mean', jnp.float32(0.0)))),
            'rst_raw_tau_max': result.get('rst_raw_tau_max', result.get('rst_raw_tau_mean', result.get('rst_tau_raw_mean', jnp.float32(0.0)))),
            'rst_tau_min': result.get('rst_tau_min', result.get('rst_tau_floor_mean', jnp.float32(0.0))),
            'rst_tau_max': result.get('rst_tau_max', jnp.float32(0.0)),

            'rst_selection_margin_mean': result.get('rst_selection_margin_mean', jnp.float32(0.0)),
            'rst_positive_margin_mean': result.get('rst_positive_margin_mean', jnp.float32(0.0)),
            'rst_positive_margin_max': result.get('rst_positive_margin_max', jnp.float32(0.0)),
            'rst_selected_frac': result.get('rst_selected_frac', jnp.float32(0.0)),
            'rst_no_active_frac': result.get('rst_no_active_frac', jnp.float32(0.0)),
            # Operator-key norm stats (REGULAR subset; *_max moved to analysis_step).
            'rst_op_key_norm': result.get(
                'rst_op_key_norm', result.get('rst_emb_norm', jnp.float32(0.0))),
            'rst_op_key_norm_min': result.get(
                'rst_op_key_norm_min',
                result.get('rst_emb_norm_min', jnp.float32(0.0))),
            'rst_op_key_norm_std': result.get(
                'rst_op_key_norm_std',
                result.get('rst_emb_norm_std', jnp.float32(0.0))),
            'attn_qk_op_key_norm_mean': result.get(
                'attn_qk_op_key_norm_mean',
                result.get('attn_qk_emb_norm_mean', jnp.float32(0.0))),
            'attn_qk_op_key_norm_min': result.get(
                'attn_qk_op_key_norm_min',
                result.get('attn_qk_emb_norm_min', jnp.float32(0.0))),
            'attn_qk_op_key_norm_std': result.get(
                'attn_qk_op_key_norm_std',
                result.get('attn_qk_emb_norm_std', jnp.float32(0.0))),
            'attn_v_op_key_norm_mean': result.get(
                'attn_v_op_key_norm_mean',
                result.get('attn_v_emb_norm_mean', jnp.float32(0.0))),
            'attn_v_op_key_norm_min': result.get(
                'attn_v_op_key_norm_min',
                result.get('attn_v_emb_norm_min', jnp.float32(0.0))),
            'attn_v_op_key_norm_std': result.get(
                'attn_v_op_key_norm_std',
                result.get('attn_v_emb_norm_std', jnp.float32(0.0))),
            'rst_read_norm': result.get('rst_read_norm', jnp.float32(0.0)),
            'rst_write_norm': result.get('rst_write_norm', jnp.float32(0.0)),
            # tau bias (scalar learned params).
            'tau_rst_bias': tau_rst_b[0],
            'tau_attn_bias_0': tau_attn_b[0],
            'tau_attn_bias_1': tau_attn_b[1],
            'tau_attn_bias_2': tau_attn_b[2],
            'tau_q_bias': tau_q_b[0],
            'tau_k_bias': tau_k_b[0],
            'tau_v_bias': tau_v_b[0],
            'raw_scan_offset_rst_bias': raw_scan_offset_rst_b[0],
            'raw_scan_offset_attn_bias_0': raw_scan_offset_attn_b[0],
            'raw_scan_offset_attn_bias_1': raw_scan_offset_attn_b[1],
            'raw_scan_offset_attn_bias_2': raw_scan_offset_attn_b[2],
            # Output norms (REGULAR subset).
            'rst_out_norm': result.get('rst_out_norm', jnp.float32(0.0)),
            'attn_qk_raw_norm': result.get('attn_qk_raw_norm', jnp.float32(0.0)),
            'attn_v_raw_norm': result.get('attn_v_raw_norm', jnp.float32(0.0)),
            'rst_raw_out_norm': result.get('rst_raw_out_norm', jnp.float32(0.0)),
            # z_mean_active (kept: cheap scalar).
            'rst_z_mean_active': result.get('rst_z_mean_active', jnp.float32(0.0)),
            'attn_qk_z_mean_active': result.get('attn_qk_z_mean_active', jnp.float32(0.0)),
            'attn_v_z_mean_active': result.get('attn_v_z_mean_active', jnp.float32(0.0)),
            # Dead-only penalty.
            'dead_penalty': dead_penalty,
            'attn_dead_penalty': result.get('attn_dead_penalty', jnp.float32(0.0)),
            'rst_dead_penalty': result.get('rst_dead_penalty', jnp.float32(0.0)),
            'attn_dead_count': result.get('attn_dead_count', jnp.float32(0.0)),
            'rst_dead_count': result.get('rst_dead_count', jnp.float32(0.0)),
            'attn_angular_exposure_mean': result.get(
                'attn_angular_exposure_mean', jnp.float32(0.0)),
            'attn_angular_exposure_min': result.get(
                'attn_angular_exposure_min', jnp.float32(0.0)),
            'attn_angular_exposure_max': result.get(
                'attn_angular_exposure_max', jnp.float32(0.0)),
            'attn_dead_exposure_frac': result.get(
                'attn_dead_exposure_frac', jnp.float32(0.0)),
            'attn_weak_exposure_frac': result.get(
                'attn_weak_exposure_frac', jnp.float32(0.0)),
            'attn_dead_exposure_target': result.get(
                'attn_dead_exposure_target', jnp.float32(0.0)),
            'rst_angular_exposure_mean': result.get(
                'rst_angular_exposure_mean', jnp.float32(0.0)),
            'rst_angular_exposure_min': result.get(
                'rst_angular_exposure_min', jnp.float32(0.0)),
            'rst_angular_exposure_max': result.get(
                'rst_angular_exposure_max', jnp.float32(0.0)),
            'rst_dead_exposure_frac': result.get(
                'rst_dead_exposure_frac', jnp.float32(0.0)),
            'rst_weak_exposure_frac': result.get(
                'rst_weak_exposure_frac', jnp.float32(0.0)),
            'rst_dead_exposure_target': result.get(
                'rst_dead_exposure_target', jnp.float32(0.0)),
            # v4164 admission exposure diagnostics. Older hard-boundary
            # exposure aliases above are kept only for JSONL continuity.
            'attn_soft_exposure_mean': result.get('attn_soft_exposure_mean', jnp.float32(0.0)),
            'attn_soft_exposure_min': result.get('attn_soft_exposure_min', jnp.float32(0.0)),
            'attn_soft_exposure_max': result.get('attn_soft_exposure_max', jnp.float32(0.0)),
            'attn_soft_dead_frac_eps_1e_6': result.get('attn_soft_dead_frac_eps_1e_6', jnp.float32(0.0)),
            'attn_soft_dead_frac_eps_1e_5': result.get('attn_soft_dead_frac_eps_1e_5', jnp.float32(0.0)),
            'attn_soft_dead_frac_eps_1e_4': result.get('attn_soft_dead_frac_eps_1e_4', jnp.float32(0.0)),
            'attn_qk_soft_exposure_mean': result.get('attn_qk_soft_exposure_mean', jnp.float32(0.0)),
            'attn_qk_soft_exposure_min': result.get('attn_qk_soft_exposure_min', jnp.float32(0.0)),
            'attn_qk_soft_exposure_max': result.get('attn_qk_soft_exposure_max', jnp.float32(0.0)),
            'attn_qk_soft_dead_frac_eps_1e_6': result.get('attn_qk_soft_dead_frac_eps_1e_6', jnp.float32(0.0)),
            'attn_qk_soft_dead_frac_eps_1e_5': result.get('attn_qk_soft_dead_frac_eps_1e_5', jnp.float32(0.0)),
            'attn_qk_soft_dead_frac_eps_1e_4': result.get('attn_qk_soft_dead_frac_eps_1e_4', jnp.float32(0.0)),
            'attn_v_soft_exposure_mean': result.get('attn_v_soft_exposure_mean', jnp.float32(0.0)),
            'attn_v_soft_exposure_min': result.get('attn_v_soft_exposure_min', jnp.float32(0.0)),
            'attn_v_soft_exposure_max': result.get('attn_v_soft_exposure_max', jnp.float32(0.0)),
            'attn_v_soft_dead_frac_eps_1e_6': result.get('attn_v_soft_dead_frac_eps_1e_6', jnp.float32(0.0)),
            'attn_v_soft_dead_frac_eps_1e_5': result.get('attn_v_soft_dead_frac_eps_1e_5', jnp.float32(0.0)),
            'attn_v_soft_dead_frac_eps_1e_4': result.get('attn_v_soft_dead_frac_eps_1e_4', jnp.float32(0.0)),
            'rst_soft_exposure_mean': result.get('rst_soft_exposure_mean', jnp.float32(0.0)),
            'rst_soft_exposure_min': result.get('rst_soft_exposure_min', jnp.float32(0.0)),
            'rst_soft_exposure_max': result.get('rst_soft_exposure_max', jnp.float32(0.0)),
            'rst_soft_dead_frac_eps_1e_6': result.get('rst_soft_dead_frac_eps_1e_6', jnp.float32(0.0)),
            'rst_soft_dead_frac_eps_1e_5': result.get('rst_soft_dead_frac_eps_1e_5', jnp.float32(0.0)),
            'rst_soft_dead_frac_eps_1e_4': result.get('rst_soft_dead_frac_eps_1e_4', jnp.float32(0.0)),
            'dead_count_total': (
                result.get('attn_dead_count', jnp.float32(0.0))
                + result.get('rst_dead_count', jnp.float32(0.0))),
            'dead_penalty_raw_total': dead_penalty,
            'attn_dead_penalty_raw': result.get(
                'attn_dead_penalty', jnp.float32(0.0)),
            'rst_dead_penalty_raw': result.get(
                'rst_dead_penalty', jnp.float32(0.0)),
            'dead_penalty_per_dead': (
                dead_penalty / jnp.maximum(
                    result.get('attn_dead_count', jnp.float32(0.0))
                    + result.get('rst_dead_count', jnp.float32(0.0)),
                    jnp.float32(1.0))),
            'attn_dead_penalty_per_dead': (
                result.get('attn_dead_penalty', jnp.float32(0.0))
                / jnp.maximum(result.get('attn_dead_count', jnp.float32(0.0)),
                              jnp.float32(1.0))),
            'rst_dead_penalty_per_dead': (
                result.get('rst_dead_penalty', jnp.float32(0.0))
                / jnp.maximum(result.get('rst_dead_count', jnp.float32(0.0)),
                              jnp.float32(1.0))),
            'global_mean_ce': inactive_aux_stats['global_mean_ce'],
            'pos_frac': inactive_aux_stats['pos_frac'],
            'pos_mean': inactive_aux_stats['pos_mean'],
            'neg_mean': inactive_aux_stats['neg_mean'],
            'inactive_aux_pos_frac': inactive_aux_stats['pos_frac'],
            'inactive_aux_pos_avg': inactive_aux_stats['pos_mean'],
            'inactive_aux_neg_avg': inactive_aux_stats['neg_mean'],
            'inactive_aux_dev_pos': inactive_aux_stats['dev_pos_max'],
            'inactive_aux_dev_neg': inactive_aux_stats['dev_neg_max'],
            'inactive_aux_loss_raw': inactive_aux_stats['inactive_aux_loss_raw'],
            'inactive_aux_q_raw': inactive_aux_stats['inactive_aux_q_raw'],
            'inactive_aux_k_raw': inactive_aux_stats['inactive_aux_k_raw'],
            'inactive_aux_qk_raw': inactive_aux_stats['inactive_aux_qk_raw'],
            'inactive_aux_v_raw': inactive_aux_stats['inactive_aux_v_raw'],
            'inactive_aux_attn_raw': inactive_aux_stats['inactive_aux_attn_raw'],
            'inactive_aux_rst_raw': inactive_aux_stats['inactive_aux_rst_raw'],
            'inactive_aux_loss_weighted': inactive_aux_loss_weighted_metric,
            'inactive_aux_active': inactive_aux_stats['inactive_aux_active'],
            'inactive_aux_block_frac_q': inactive_aux_stats['block_frac_q'],
            'inactive_aux_block_frac_key': inactive_aux_stats['block_frac_key'],
            'inactive_aux_block_frac_qk': inactive_aux_stats['block_frac_qk'],
            'inactive_aux_block_frac_v': inactive_aux_stats['block_frac_v'],
            'inactive_aux_block_frac_a': inactive_aux_stats['block_frac_a'],
            'inactive_aux_block_frac_rst': inactive_aux_stats['block_frac_rst'],
            'inactive_aux_block_frac_k': inactive_aux_stats['block_frac_rst'],
            'inactive_aux_block_q': inactive_aux_stats['block_frac_q'],
            'inactive_aux_block_key': inactive_aux_stats['block_frac_key'],
            'inactive_aux_block_qk': inactive_aux_stats['block_frac_qk'],
            'inactive_aux_block_v': inactive_aux_stats['block_frac_v'],
            'inactive_aux_block_attn': inactive_aux_stats['block_frac_a'],
            'inactive_aux_block_rst': inactive_aux_stats['block_frac_rst'],
            'dev_pos_max': inactive_aux_stats['dev_pos_max'],
            'dev_neg_max': inactive_aux_stats['dev_neg_max'],
            'attn_tau_off_min': inactive_aux_stats['attn_tau_off_min'],
            'attn_tau_off_max': inactive_aux_stats['attn_tau_off_max'],
            'attn_tau_off_p99': inactive_aux_stats['attn_tau_off_p99'],
            'attn_tau_off_p01': inactive_aux_stats['attn_tau_off_p01'],
            'attn_tau_off_neg_frac': inactive_aux_stats['attn_tau_off_neg_frac'],
            'rst_tau_off_min': inactive_aux_stats['rst_tau_off_min'],
            'rst_tau_off_max': inactive_aux_stats['rst_tau_off_max'],
            'rst_tau_off_p99': inactive_aux_stats['rst_tau_off_p99'],
            'rst_tau_off_p01': inactive_aux_stats['rst_tau_off_p01'],
            'rst_tau_off_neg_frac': inactive_aux_stats['rst_tau_off_neg_frac'],
            # v4164 intensity and gate-denominator diagnostics.
            'attn_int_max': result.get('attn_int_max', jnp.float32(0.0)),
            'rst_int_max': result.get('rst_int_max', jnp.float32(0.0)),
            'attn_int_cap_frac': result.get('attn_int_cap_frac', jnp.float32(0.0)),
            'rst_int_cap_frac': result.get('rst_int_cap_frac', jnp.float32(0.0)),
            'attn_intensity_sum_mean': result.get('attn_intensity_sum_mean', jnp.float32(0.0)),
            'rst_intensity_sum_mean': result.get('rst_intensity_sum_mean', jnp.float32(0.0)),
            'attn_gate_den_sum_mean': result.get('attn_gate_den_sum_mean', jnp.float32(0.0)),
            'rst_gate_den_sum_mean': result.get('rst_gate_den_sum_mean', jnp.float32(0.0)),
            # Output/logit diagnostics are always-on cheap scalar reductions from the model.
            'residual_norm': result.get('residual_norm', jnp.float32(0.0)),
            'residual_norm_max': result.get('residual_norm_max', jnp.float32(0.0)),
            'token_emb_norm': result.get('token_emb_norm', jnp.float32(0.0)),
            'token_emb_norm_max': result.get('token_emb_norm_max', jnp.float32(0.0)),
            'logit_max': result.get('logit_max', jnp.float32(0.0)),
            'logit_norm_mean': result.get('logit_norm_mean', jnp.float32(0.0)),
            'logit_mean': result.get('logit_mean', jnp.float32(0.0)),
            'logit_std': result.get('logit_std', jnp.float32(0.0)),
            'attn_logit_max_mean': result.get('attn_logit_max_mean', jnp.float32(0.0)),
            'attn_contrib_den_sum': result.get('attn_contrib_den_sum', jnp.float32(0.0)),
            'rst_contrib_den_sum': result.get('rst_contrib_den_sum', jnp.float32(0.0)),
            'attn_contrib_den_mean': result.get(
                'attn_contrib_den_mean',
                result.get('attn_contrib_den_sum', jnp.float32(0.0))),
            'rst_contrib_den_mean': result.get(
                'rst_contrib_den_mean',
                result.get('rst_contrib_den_sum', jnp.float32(0.0))),
            'attn_contrib_den': result.get('attn_contrib_den', jnp.float32(0.0)),
            'rst_contrib_den': result.get('rst_contrib_den', jnp.float32(0.0)),
            'attn_contrib_den_max': result.get('attn_contrib_den_max', jnp.float32(0.0)),
            'rst_contrib_den_max': result.get('rst_contrib_den_max', jnp.float32(0.0)),
            'attn_contrib_den_min': result.get('attn_contrib_den_min', jnp.float32(0.0)),
            'rst_contrib_den_min': result.get('rst_contrib_den_min', jnp.float32(0.0)),
            'attn_contrib_den_floor_frac': result.get('attn_contrib_den_floor_frac', jnp.float32(0.0)),
            'rst_contrib_den_floor_frac': result.get('rst_contrib_den_floor_frac', jnp.float32(0.0)),
            'attn_compose_norm': result.get('attn_compose_norm', jnp.float32(0.0)),
            'rst_compose_norm': result.get('rst_compose_norm', jnp.float32(0.0)),
            'attn_compose_norm_mean': result.get(
                'attn_compose_norm_mean',
                result.get('attn_compose_norm', jnp.float32(0.0))),
            'rst_compose_norm_mean': result.get(
                'rst_compose_norm_mean',
                result.get('rst_compose_norm', jnp.float32(0.0))),
            'attn_compose_norm_max': result.get('attn_compose_norm_max', jnp.float32(0.0)),
            'rst_compose_norm_max': result.get('rst_compose_norm_max', jnp.float32(0.0)),
            'attn_coherence': result.get('attn_coherence', jnp.float32(0.0)),
            'rst_coherence': result.get('rst_coherence', jnp.float32(0.0)),
            'attn_coherence_max': result.get('attn_coherence_max', jnp.float32(0.0)),
            'rst_coherence_max': result.get('rst_coherence_max', jnp.float32(0.0)),
            'attn_den_ratio': result.get('attn_den_ratio', jnp.float32(0.0)),
            'rst_den_ratio': result.get('rst_den_ratio', jnp.float32(0.0)),
            'attn_den_ratio_mean': result.get(
                'attn_den_ratio_mean',
                result.get('attn_den_ratio', jnp.float32(0.0))),
            'rst_den_ratio_mean': result.get(
                'rst_den_ratio_mean',
                result.get('rst_den_ratio', jnp.float32(0.0))),
            'attn_den_ratio_max': result.get('attn_den_ratio_max', jnp.float32(0.0)),
            'rst_den_ratio_max': result.get('rst_den_ratio_max', jnp.float32(0.0)),
            'attn_raw_out_norm_mean': result.get('attn_raw_out_norm_mean', jnp.float32(0.0)),
            'rst_raw_out_norm_mean': result.get('rst_raw_out_norm_mean', jnp.float32(0.0)),
            'attn_raw_out_norm_max': result.get('attn_raw_out_norm_max', jnp.float32(0.0)),
            'rst_raw_out_norm_max': result.get('rst_raw_out_norm_max', jnp.float32(0.0)),
            'attn_normalized_out_norm_mean': result.get('attn_normalized_out_norm_mean', jnp.float32(0.0)),
            'rst_normalized_out_norm_mean': result.get('rst_normalized_out_norm_mean', jnp.float32(0.0)),
            'attn_normalized_out_norm_max': result.get('attn_normalized_out_norm_max', jnp.float32(0.0)),
            'rst_normalized_out_norm_max': result.get('rst_normalized_out_norm_max', jnp.float32(0.0)),
            'attn_scaled_out_norm_mean': result.get('attn_scaled_out_norm_mean', jnp.float32(0.0)),
            'rst_scaled_out_norm_mean': result.get('rst_scaled_out_norm_mean', jnp.float32(0.0)),
            'attn_scaled_out_norm_max': result.get('attn_scaled_out_norm_max', jnp.float32(0.0)),
            'rst_scaled_out_norm_max': result.get('rst_scaled_out_norm_max', jnp.float32(0.0)),
            'attn_den_cost_mean': result.get('attn_den_cost_mean', jnp.float32(0.0)),
            'rst_den_cost_mean': result.get('rst_den_cost_mean', jnp.float32(0.0)),
            'attn_act_cost_mean': result.get('attn_act_cost_mean', jnp.float32(0.0)),
            'rst_act_cost_mean': result.get('rst_act_cost_mean', jnp.float32(0.0)),
            'attn_current_cost_mean': result.get('attn_current_cost_mean', jnp.float32(0.0)),
            'rst_current_cost_mean': result.get('rst_current_cost_mean', jnp.float32(0.0)),
            # Operator-key drift (relative L2) since prev snapshot.
            'drift_attn_qk_emb': drift_attn_qk_emb,
            'drift_attn_v_emb': drift_attn_v_emb,
            'drift_rst_emb': drift_rst_emb,
            'drift_attn_qk_op_key': drift_attn_qk_op_key,
            'drift_attn_v_op_key': drift_attn_v_op_key,
            'drift_rst_op_key': drift_rst_op_key,
            # Actual post-Adam control update caps. *_pre are measured after
            # LR multipliers and before capping; *_post are after cap scaling.
            'update_cap_proj_attn_ratio_pre': upd_proj_attn_ratio_pre,
            'update_cap_proj_attn_ratio_post': upd_proj_attn_ratio_post,
            'update_cap_proj_attn_scale': upd_proj_attn_scale,
            'update_cap_proj_attn_hit': upd_proj_attn_hit,
            'update_cap_proj_rst_ratio_pre': upd_proj_rst_ratio_pre,
            'update_cap_proj_rst_ratio_post': upd_proj_rst_ratio_post,
            'update_cap_proj_rst_scale': upd_proj_rst_scale,
            'update_cap_proj_rst_hit': upd_proj_rst_hit,
            'update_cap_op_key_qk_ratio_pre': upd_op_key_qk_ratio_pre,
            'update_cap_op_key_qk_ratio_post': upd_op_key_qk_ratio_post,
            'update_cap_op_key_qk_scale': upd_op_key_qk_scale,
            'update_cap_op_key_qk_hit': upd_op_key_qk_hit,
            'update_cap_op_key_v_ratio_pre': upd_op_key_v_ratio_pre,
            'update_cap_op_key_v_ratio_post': upd_op_key_v_ratio_post,
            'update_cap_op_key_v_scale': upd_op_key_v_scale,
            'update_cap_op_key_v_hit': upd_op_key_v_hit,
            'update_cap_op_key_rst_ratio_pre': upd_op_key_rst_ratio_pre,
            'update_cap_op_key_rst_ratio_post': upd_op_key_rst_ratio_post,
            'update_cap_op_key_rst_scale': upd_op_key_rst_scale,
            'update_cap_op_key_rst_hit': upd_op_key_rst_hit,
            'update_cap_tau_attn_abs_pre': upd_tau_attn_abs_pre,
            'update_cap_tau_attn_abs_post': upd_tau_attn_abs_post,
            'update_cap_tau_attn_scale': upd_tau_attn_scale,
            'update_cap_tau_attn_hit': upd_tau_attn_hit,
            'update_cap_tau_rst_abs_pre': upd_tau_rst_abs_pre,
            'update_cap_tau_rst_abs_post': upd_tau_rst_abs_post,
            'update_cap_tau_rst_scale': upd_tau_rst_scale,
            'update_cap_tau_rst_hit': upd_tau_rst_hit,
            'update_cap_raw_tau_enabled': jnp.float32(
                1.0 if (_enable_control_update_caps
                        and float(tau_update_abs_cap) > 0.0) else 0.0),
            'update_cap_raw_tau_qk_abs_pre': upd_raw_tau_qk_abs_pre,
            'update_cap_raw_tau_qk_abs_post': upd_raw_tau_qk_abs_post,
            'update_cap_raw_tau_qk_scale': upd_raw_tau_qk_scale,
            'update_cap_raw_tau_qk_hit': upd_raw_tau_qk_hit,
            'update_cap_raw_tau_v_abs_pre': upd_raw_tau_v_abs_pre,
            'update_cap_raw_tau_v_abs_post': upd_raw_tau_v_abs_post,
            'update_cap_raw_tau_v_scale': upd_raw_tau_v_scale,
            'update_cap_raw_tau_v_hit': upd_raw_tau_v_hit,
            'update_cap_raw_tau_rst_abs_pre': upd_raw_tau_rst_abs_pre,
            'update_cap_raw_tau_rst_abs_post': upd_raw_tau_rst_abs_post,
            'update_cap_raw_tau_rst_scale': upd_raw_tau_rst_scale,
            'update_cap_raw_tau_rst_hit': upd_raw_tau_rst_hit,
            'update_cap_scan_attn_abs_pre': upd_scan_attn_abs_pre,
            'update_cap_scan_attn_abs_post': upd_scan_attn_abs_post,
            'update_cap_scan_attn_scale': upd_scan_attn_scale,
            'update_cap_scan_attn_hit': upd_scan_attn_hit,
            'update_cap_scan_rst_abs_pre': upd_scan_rst_abs_pre,
            'update_cap_scan_rst_abs_post': upd_scan_rst_abs_post,
            'update_cap_scan_rst_scale': upd_scan_rst_scale,
            'update_cap_scan_rst_hit': upd_scan_rst_hit,
        }
        if (not _compact_train_metrics) or _keep_train_layer_metrics:
            metrics.update({
                'per_layer_attn_out_norm': result.get(
                    'per_layer_attn_out_norm', jnp.zeros(1)),
                'per_layer_rst_out_norm': result.get(
                    'per_layer_rst_out_norm', jnp.zeros(1)),
            })
        for _name in V4164_SCALAR_METRIC_NAMES:
            metrics[_name] = result.get(_name, jnp.float32(0.0))
        for _pool in ('attn_qk', 'attn_v', 'rst'):
            for _name in PAGE_METRIC_NAMES:
                metrics[f'{_pool}_{_name}'] = result.get(
                    f'{_pool}_{_name}', jnp.float32(0.0))
        for _name in (
                'estimated_compute_frac_page',
                'selected_page_count'):
            metrics[_name] = result.get(_name, jnp.float32(0.0))
        for _pool in ('attn_v', 'rst'):
            for _name in (
                    'sector_fill_mean',
                    'sector_fill_max',
                    'sector_overflow_count',
                    'selected_sector_frac',
                    'effective_operator_frac'):
                _key = f'sector/{_pool}/{_name}'
                metrics[_key] = result.get(_key, jnp.float32(0.0))
        for _pool in ('attn_v', 'rst'):
            for _name in _V4168_OPSPACE_RUNTIME_DIAG_NAMES:
                _key = f'opspace/{_pool}/{_name}'
                metrics[_key] = result.get(_key, jnp.float32(0.0))
        for _pool in ('attn_v', 'rst'):
            for _name in _V4168_OPSPACE_FINAL_RUNTIME_DIAG_NAMES:
                _key = f'opspace/{_pool}/final/{_name}'
                metrics[_key] = result.get(_key, jnp.float32(0.0))
        for _pool in ('attn_v', 'rst'):
            for _name in _V4168_OPSPACE_FINAL_RUNTIME_DIAG_NAMES:
                _alias_key = f'opspace/{_pool}/{_name}'
                _final_key = f'opspace/{_pool}/final/{_name}'
                if _alias_key not in metrics:
                    metrics[_alias_key] = result.get(
                        _alias_key,
                        result.get(_final_key, jnp.float32(0.0)))
        for _pool in ('attn_qk', 'attn_v'):
            for _name in DIRECT_TAU_ATTN_SPLIT_METRIC_NAMES:
                _fallback = result.get(
                    f'attn_{_name}', jnp.float32(0.0))
                if _name == 'score_std':
                    _fallback = result.get(
                        'attn_score_std',
                        result.get('attn_rho_std', jnp.float32(0.0)))
                metrics[f'{_pool}_{_name}'] = result.get(
                    f'{_pool}_{_name}', _fallback)
            for _name in DIRECT_TAU_SELECT_METRIC_NAMES:
                metrics[f'{_pool}_{_name}'] = result.get(
                    f'{_pool}_{_name}',
                    result.get(f'attn_{_name}', jnp.float32(0.0)))
            for _name in DIRECT_TAU_EXPOSURE_METRIC_NAMES:
                metrics[f'{_pool}_{_name}'] = result.get(
                    f'{_pool}_{_name}',
                    result.get(f'attn_{_name}', jnp.float32(0.0)))
        if not _inactive_aux_enabled:
            for _key in (
                    'inactive_aux_warmup_factor',
                    'inactive_aux_weight_effective',
                    'inactive_aux_asymmetry',
                    'inactive_aux_asymmetry_q',
                    'inactive_aux_asymmetry_k',
                    'inactive_aux_asymmetry_qk',
                    'inactive_aux_asymmetry_v',
                    'inactive_aux_asymmetry_rst',
                    'inactive_aux_loss_raw_total',
                    'inactive_aux_loss_raw_q',
                    'inactive_aux_loss_raw_k',
                    'inactive_aux_loss_raw_qk',
                    'inactive_aux_loss_raw_v',
                    'inactive_aux_loss_raw_attn',
                    'inactive_aux_loss_raw_rst',
                    'inactive_aux_loss_weighted_q',
                    'inactive_aux_loss_weighted_k',
                    'inactive_aux_loss_weighted_qk',
                    'inactive_aux_loss_weighted_v',
                    'inactive_aux_loss_weighted_attn',
                    'inactive_aux_loss_weighted_rst',
                    'inactive_aux_loss_weighted_total',
                    'inactive_aux_loss_weighted_unclipped',
                    'inactive_aux_loss_weighted_clipped',
                    'inactive_aux_layer_norm_enabled',
                    'inactive_aux_layer_count',
                    'inactive_aux_norm',
                    'inactive_aux_no_active_easy_shutoff_q',
                    'inactive_aux_no_active_easy_shutoff_k',
                    'inactive_aux_no_active_easy_shutoff_v',
                    'inactive_aux_no_active_easy_shutoff_rst',
                    'inactive_aux_raw_pre_bound',
                    'inactive_aux_q_pre_bound',
                    'inactive_aux_k_pre_bound',
                    'inactive_aux_qk_pre_bound',
                    'inactive_aux_v_pre_bound',
                    'inactive_aux_raw_post_bound',
                    'inactive_aux_attn_pre_bound',
                    'inactive_aux_rst_pre_bound',
                    'global_mean_ce',
                    'pos_frac',
                    'pos_mean',
                    'neg_mean',
                    'inactive_aux_pos_frac',
                    'inactive_aux_pos_avg',
                    'inactive_aux_neg_avg',
                    'inactive_aux_dev_pos',
                    'inactive_aux_dev_neg',
                    'inactive_aux_block_q',
                    'inactive_aux_block_key',
                    'inactive_aux_block_qk',
                    'inactive_aux_block_v',
                    'inactive_aux_block_attn',
                    'inactive_aux_block_rst',
                    'inactive_aux_loss_raw',
                    'inactive_aux_q_raw',
                    'inactive_aux_k_raw',
                    'inactive_aux_qk_raw',
                    'inactive_aux_v_raw',
                    'inactive_aux_attn_raw',
                    'inactive_aux_rst_raw',
                    'inactive_aux_loss_weighted',
                    'inactive_aux_active',
                    'inactive_aux_block_frac_q',
                    'inactive_aux_block_frac_key',
                    'inactive_aux_block_frac_qk',
                    'inactive_aux_block_frac_v',
                    'inactive_aux_block_frac_a',
                    'inactive_aux_block_frac_rst',
                    'inactive_aux_block_frac_k',
                    'dev_pos_max',
                    'dev_neg_max'):
                metrics.pop(_key, None)
        metrics.update(pool_diag)
        metrics.update(pool_update_diag)

        return new_params, new_opt_state, metrics

    return train_step


def create_eval_step(model, sharded_fns=None, return_dead_stats=False,
                     return_prune_stats=False, execution_prune_eps=0.0,
                     total_training_steps=1,
                     soft_gate_schedule_active=False,
                     soft_gate_t_start=1.5,
                     soft_gate_t_final=0.07,
                     soft_gate_t_hold_frac=0.10,
                     soft_gate_t_anneal_end_frac=0.80,
                     soft_gate_schedule='cosine',
                     soft_gate_t_power=4.0,
                     soft_gate_t_gompertz_center=0.25,
                     soft_gate_t_gompertz_steepness=8.0,
                     pool_specific_gate_t=False,
                     soft_gate_pool_schedules=None,
                     boundary_power_schedule_active=False,
                     soft_gate_boundary_power_start=3.0,
                     soft_gate_boundary_power_mid=3.15,
                     soft_gate_boundary_power_final=4.0,
                     soft_gate_boundary_power_start_frac=0.0,
                     soft_gate_boundary_power_mid_frac=0.800,
                     soft_gate_boundary_power_final_frac=0.950,
                     admission_den_power=1.0):
    """Create a jit-compiled evaluation step.

    Uses the SLIM forward (analysis=False). Eval normally needs only loss /
    correct / valid_count, with optional scalar dead stats for validation
    logging.
    """
    _pass_analysis_kw = _model_accepts_analysis(model)
    _pass_soft_gate_schedule_kw = _model_accepts_soft_gate_schedule(model)
    _pass_soft_gate_t_final_kw = _model_accepts_soft_gate_t_final(model)
    _pass_execution_prune_kw = _model_accepts_execution_prune_eps(model)
    _pass_boundary_power_kw = _model_accepts_soft_gate_boundary_power(model)
    _pass_den_power_kw = _model_accepts_admission_den_power(model)
    _pass_minimal_train_kw = _model_accepts_minimal_train(model)
    _execution_prune_eps = jnp.float32(execution_prune_eps)
    _return_prune_stats = bool(return_prune_stats)
    _model_version = getattr(
        model, '__version__', getattr(type(model), '__version__', ''))
    _use_minimal_train_path = (
        str(_model_version) == V4168_MODEL_VERSION
        and _pass_minimal_train_kw)
    _soft_gate_runtime_enabled = bool(
        soft_gate_schedule_active
        and _is_active_srw_version(_model_version))
    _is_boundary_power_model = _is_active_srw_version(_model_version)
    _total_training_steps = jnp.float32(max(1, int(total_training_steps or 1)))
    _soft_gate_t_start = jnp.float32(soft_gate_t_start)
    _soft_gate_t_final = jnp.float32(soft_gate_t_final)
    _soft_gate_t_hold_frac = jnp.float32(soft_gate_t_hold_frac)
    _soft_gate_t_anneal_end_frac = jnp.float32(soft_gate_t_anneal_end_frac)
    _soft_gate_schedule = str(soft_gate_schedule).lower()
    _soft_gate_t_power = jnp.float32(soft_gate_t_power)
    _soft_gate_t_gompertz_center = jnp.float32(soft_gate_t_gompertz_center)
    _soft_gate_t_gompertz_steepness = jnp.float32(
        soft_gate_t_gompertz_steepness)
    _soft_gate_pool_defaults = {
        'start': soft_gate_t_start,
        'final': soft_gate_t_final,
        'hold_frac': soft_gate_t_hold_frac,
        'anneal_end_frac': soft_gate_t_anneal_end_frac,
        'schedule': soft_gate_schedule,
        'power': soft_gate_t_power,
        'gompertz_center': soft_gate_t_gompertz_center,
        'gompertz_steepness': soft_gate_t_gompertz_steepness,
    }
    _soft_gate_pool_cfg = _coerce_pool_schedule_configs(
        (soft_gate_pool_schedules
         if (pool_specific_gate_t
             or _soft_gate_schedule == 'developmental_band')
         else None),
        _soft_gate_pool_defaults)
    _boundary_power_schedule_active = bool(
        boundary_power_schedule_active)
    _soft_gate_boundary_power_start = jnp.float32(
        soft_gate_boundary_power_start)
    _soft_gate_boundary_power_mid = jnp.float32(
        soft_gate_boundary_power_mid)
    _soft_gate_boundary_power_final = jnp.float32(
        soft_gate_boundary_power_final)
    _soft_gate_boundary_power_start_frac = jnp.float32(
        soft_gate_boundary_power_start_frac)
    _soft_gate_boundary_power_mid_frac = jnp.float32(
        soft_gate_boundary_power_mid_frac)
    _soft_gate_boundary_power_final_frac = jnp.float32(
        soft_gate_boundary_power_final_frac)
    _admission_den_power = jnp.float32(admission_den_power)

    @jax.jit
    def eval_step(params, input_ids, attention_mask, step):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        eval_rng = jax.random.PRNGKey(0)
        extra_kw = {}
        if sharded_fns is not None:
            extra_kw['sharded_fns'] = sharded_fns
        if _pass_analysis_kw:
            extra_kw['analysis'] = False
        if _use_minimal_train_path:
            extra_kw['minimal_train'] = True
        if _soft_gate_runtime_enabled and _pass_soft_gate_schedule_kw:
            soft_gate_T_qk = _scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['qk'])
            extra_kw['soft_gate_temperature'] = soft_gate_T_qk
            extra_kw['soft_gate_T_qk'] = soft_gate_T_qk
            extra_kw['soft_gate_T_v'] = _scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['v'])
            extra_kw['soft_gate_T_rst'] = _scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['rst'])
        if _pass_boundary_power_kw:
            boundary_power_p = scheduled_boundary_power_by_frac(
                step, _total_training_steps,
                _boundary_power_schedule_active and _is_boundary_power_model,
                _soft_gate_boundary_power_start,
                _soft_gate_boundary_power_mid,
                _soft_gate_boundary_power_final,
                _soft_gate_boundary_power_mid_frac,
                _soft_gate_boundary_power_final_frac,
                _soft_gate_boundary_power_start_frac)
            extra_kw['soft_gate_boundary_power'] = boundary_power_p
            extra_kw['soft_gate_boundary_power_final'] = (
                _soft_gate_boundary_power_final)
        if _pass_soft_gate_t_final_kw:
            extra_kw['soft_gate_t_final'] = _soft_gate_t_final
        if _pass_den_power_kw:
            extra_kw['admission_den_power'] = _admission_den_power
        if _pass_execution_prune_kw:
            extra_kw['execution_prune_eps'] = _execution_prune_eps
        result = model.apply(
            {'params': params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={'dropout': eval_rng},
            **extra_kw,
        )
        if return_dead_stats:
            base_ret = (
                result['loss'],
                result['correct'],
                result['valid_count'],
                result.get('attn_dead_count', jnp.float32(0.0)),
                result.get(
                    'rst_dead_count',
                    result.get('know_dead_count', jnp.float32(0.0))),
            )
            if _return_prune_stats:
                base_ret = base_ret + (
                    result.get('execution_estimated_compute_frac', jnp.float32(0.0)),
                    result.get('execution_gate_mass_retained', jnp.float32(1.0)),
                    result.get('execution_prune_gate_den_mean', jnp.float32(0.0)),
                    result.get('execution_prune_gate_den_min', jnp.float32(0.0)),
                    result.get('execution_prune_no_active_frac', jnp.float32(0.0)),
                    result.get('execution_prune_unpruned_gate_den_mean', jnp.float32(0.0)),
                )
            return base_ret
        return result['loss'], result['correct'], result['valid_count']

    return eval_step


def create_analysis_step(model, sharded_fns=None,
                         total_training_steps=1,
                         soft_gate_schedule_active=False,
                         soft_gate_t_start=1.5,
                         soft_gate_t_final=0.07,
                         soft_gate_t_hold_frac=0.10,
                         soft_gate_t_anneal_end_frac=0.80,
                         soft_gate_schedule='cosine',
                         soft_gate_t_power=4.0,
                         soft_gate_t_gompertz_center=0.25,
                         soft_gate_t_gompertz_steepness=8.0,
                         pool_specific_gate_t=False,
                         soft_gate_pool_schedules=None,
                         boundary_power_schedule_active=False,
                         soft_gate_boundary_power_start=3.0,
                         soft_gate_boundary_power_mid=3.15,
                         soft_gate_boundary_power_final=4.0,
                         soft_gate_boundary_power_start_frac=0.0,
                         soft_gate_boundary_power_mid_frac=0.800,
                         soft_gate_boundary_power_final_frac=0.950,
                         admission_den_power=1.0):
    """Create a jit-compiled analysis step (FULL forward, observational).

    Runs the model with `analysis=True` and the ANALYSIS variant of
    sharded_fns. Returns a dict of distribution / boundary
    stats that the 2-tier logger's ANALYSIS block consumes. Called
    once per val tick (val_interval), so the compile cost amortises.
    """
    _pass_analysis_kw = _model_accepts_analysis(model)
    _pass_soft_gate_schedule_kw = _model_accepts_soft_gate_schedule(model)
    _pass_soft_gate_t_final_kw = _model_accepts_soft_gate_t_final(model)
    _pass_execution_prune_kw = _model_accepts_execution_prune_eps(model)
    _pass_boundary_power_kw = _model_accepts_soft_gate_boundary_power(model)
    _pass_den_power_kw = _model_accepts_admission_den_power(model)
    _soft_gate_runtime_enabled = bool(
        soft_gate_schedule_active
        and _is_active_srw_version(
            getattr(model, '__version__', getattr(type(model), '__version__', ''))))
    _is_boundary_power_model = _is_active_srw_version(
        getattr(model, '__version__', getattr(type(model), '__version__', '')))
    _total_training_steps = jnp.float32(max(1, int(total_training_steps or 1)))
    _soft_gate_t_start = jnp.float32(soft_gate_t_start)
    _soft_gate_t_final = jnp.float32(soft_gate_t_final)
    _soft_gate_t_hold_frac = jnp.float32(soft_gate_t_hold_frac)
    _soft_gate_t_anneal_end_frac = jnp.float32(soft_gate_t_anneal_end_frac)
    _soft_gate_schedule = str(soft_gate_schedule).lower()
    _soft_gate_t_power = jnp.float32(soft_gate_t_power)
    _soft_gate_t_gompertz_center = jnp.float32(soft_gate_t_gompertz_center)
    _soft_gate_t_gompertz_steepness = jnp.float32(
        soft_gate_t_gompertz_steepness)
    _soft_gate_pool_defaults = {
        'start': soft_gate_t_start,
        'final': soft_gate_t_final,
        'hold_frac': soft_gate_t_hold_frac,
        'anneal_end_frac': soft_gate_t_anneal_end_frac,
        'schedule': soft_gate_schedule,
        'power': soft_gate_t_power,
        'gompertz_center': soft_gate_t_gompertz_center,
        'gompertz_steepness': soft_gate_t_gompertz_steepness,
    }
    _soft_gate_pool_cfg = _coerce_pool_schedule_configs(
        (soft_gate_pool_schedules
         if (pool_specific_gate_t
             or _soft_gate_schedule == 'developmental_band')
         else None),
        _soft_gate_pool_defaults)
    _boundary_power_schedule_active = bool(
        boundary_power_schedule_active)
    _soft_gate_boundary_power_start = jnp.float32(
        soft_gate_boundary_power_start)
    _soft_gate_boundary_power_mid = jnp.float32(
        soft_gate_boundary_power_mid)
    _soft_gate_boundary_power_final = jnp.float32(
        soft_gate_boundary_power_final)
    _soft_gate_boundary_power_start_frac = jnp.float32(
        soft_gate_boundary_power_start_frac)
    _soft_gate_boundary_power_mid_frac = jnp.float32(
        soft_gate_boundary_power_mid_frac)
    _soft_gate_boundary_power_final_frac = jnp.float32(
        soft_gate_boundary_power_final_frac)
    _admission_den_power = jnp.float32(admission_den_power)

    @jax.jit
    def analysis_step(params, input_ids, attention_mask, step):
        labels = jnp.where(attention_mask == 1, input_ids, -100)
        eval_rng = jax.random.PRNGKey(0)
        extra_kw = {}
        if sharded_fns is not None:
            extra_kw['sharded_fns'] = sharded_fns
        if _pass_analysis_kw:
            extra_kw['analysis'] = True
        if _soft_gate_runtime_enabled and _pass_soft_gate_schedule_kw:
            soft_gate_T_qk = _scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['qk'])
            extra_kw['soft_gate_temperature'] = soft_gate_T_qk
            extra_kw['soft_gate_T_qk'] = soft_gate_T_qk
            extra_kw['soft_gate_T_v'] = _scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['v'])
            extra_kw['soft_gate_T_rst'] = _scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['rst'])
        if _pass_boundary_power_kw:
            boundary_power_p = scheduled_boundary_power_by_frac(
                step, _total_training_steps,
                _boundary_power_schedule_active and _is_boundary_power_model,
                _soft_gate_boundary_power_start,
                _soft_gate_boundary_power_mid,
                _soft_gate_boundary_power_final,
                _soft_gate_boundary_power_mid_frac,
                _soft_gate_boundary_power_final_frac,
                _soft_gate_boundary_power_start_frac)
            extra_kw['soft_gate_boundary_power'] = boundary_power_p
            extra_kw['soft_gate_boundary_power_final'] = (
                _soft_gate_boundary_power_final)
        if _pass_soft_gate_t_final_kw:
            extra_kw['soft_gate_t_final'] = _soft_gate_t_final
        if _pass_den_power_kw:
            extra_kw['admission_den_power'] = _admission_den_power
        if _pass_execution_prune_kw:
            extra_kw['execution_prune_eps'] = jnp.float32(0.0)
        result = model.apply(
            {'params': params},
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            deterministic=True,
            rngs={'dropout': eval_rng},
            **extra_kw,
        )
        result = dict(result)
        result.update(_pool_param_diagnostics(params, full=True, model=model))
        return result

    return analysis_step


def create_geometry_step(max_sample=512):
    """Rare, observational geometry diagnostics on a deterministic row sample."""
    max_sample = int(max_sample)

    def _geom_one(x, prefix):
        x = jax.lax.stop_gradient(jnp.asarray(x, dtype=jnp.float32))
        n = x.shape[0]
        stride = max(1, n // max_sample)
        xs = x[::stride][:max_sample]
        xs = xs - xs.mean(axis=0, keepdims=True)
        s = jnp.linalg.svd(xs, full_matrices=False, compute_uv=False)
        energy = jnp.sum(jnp.square(s))
        eff_rank = energy / (jnp.max(jnp.square(s)) + 1e-8)
        xn = xs / (jnp.linalg.norm(xs, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.abs(xn @ xn.T)
        mask = 1.0 - jnp.eye(sim.shape[0], dtype=jnp.float32)
        denom = mask.sum() + 1e-8
        sim_off = sim * mask
        s5 = jnp.pad(s, (0, max(0, 5 - s.shape[0])))[:5]
        return {
            f'{prefix}_geom_rank': eff_rank,
            f'{prefix}_geom_cos_mean': sim_off.sum() / denom,
            f'{prefix}_geom_cos_max': sim_off.max(),
            f'{prefix}_geom_sv0': s5[0],
            f'{prefix}_geom_sv1': s5[1],
            f'{prefix}_geom_sv2': s5[2],
            f'{prefix}_geom_sv3': s5[3],
            f'{prefix}_geom_sv4': s5[4],
        }

    def _geom_op_key(read, write, read_proj, write_proj, eps=1.0e-6):
        def _unit(x):
            x = jnp.asarray(x, dtype=jnp.float32)
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
        r_key = _unit(read) @ read_proj
        w_key = _unit(write) @ write_proj
        return _unit(_unit(r_key) * _unit(w_key))

    @jax.jit
    def geometry_step(params):
        pool = params.get('neuron_pool', {})
        out = {}
        for name, emb_key, read_key, write_key, op_read_key, op_write_key in (
                ('attn_qk', 'attn_qk_emb', 'attn_qk_read',
                 'attn_qk_write', 'attn_qk_op_read_proj',
                 'attn_qk_op_write_proj'),
                ('attn_v', 'attn_v_emb', 'attn_v_read',
                 'attn_v_write', 'attn_v_op_read_proj',
                 'attn_v_op_write_proj'),
                ('rst', 'rst_emb', 'rst_read', 'rst_write',
                 'rst_op_read_proj', 'rst_op_write_proj')):
            if (read_key in pool and write_key in pool
                    and op_read_key in pool and op_write_key in pool):
                op_key = _geom_op_key(
                    pool[read_key], pool[write_key],
                    pool[op_read_key], pool[op_write_key])
                out.update(_geom_one(op_key, f'{name}_op_key'))
            if emb_key in pool:
                out.update(_geom_one(pool[emb_key], f'{name}_emb'))
            if read_key in pool:
                out.update(_geom_one(pool[read_key], f'{name}_read'))
            if write_key in pool:
                out.update(_geom_one(pool[write_key], f'{name}_write'))
        return out

    return geometry_step


# ============================================================
# Mesh-based sharding (model parallel + data parallel)
# ============================================================

def create_mesh(mesh_data, mesh_model):
    """Create 2D Mesh for data + model parallelism."""
    devices = jax.devices()
    n_devices = len(devices)
    assert n_devices == mesh_data * mesh_model, (
        f"mesh_data({mesh_data}) * mesh_model({mesh_model}) = "
        f"{mesh_data * mesh_model} != {n_devices} devices")
    # Mesh interpretation:
    # The mesh is data x model. The data axis shards token rows. The model
    # axis shards operation regions inside each data group. Each data group
    # has its own region capacities; capacities are not shared across data
    # replicas. num_regions must be divisible by mesh_model.
    device_array = np.array(devices).reshape(mesh_data, mesh_model)
    return Mesh(device_array, ('data', 'model'))


def get_param_shardings(params, mesh, model_version=None,
                        operation_space_enabled=False):
    """Create model-version-aware parameter shardings."""
    del operation_space_enabled
    replicated = NamedSharding(mesh, P())
    vector_sharded = NamedSharding(mesh, P('model'))
    col_sharded = NamedSharding(mesh, P(None, 'model'))
    row_sharded = NamedSharding(mesh, P('model', None))
    n_sharded = NamedSharding(mesh, P('model', None))
    n_sharded_3d = NamedSharding(mesh, P('model', None, None))
    stage_n_sharded_3d = NamedSharding(mesh, P(None, 'model', None))
    router_input_sharded_3d = NamedSharding(mesh, P(None, 'model', None))
    version = str(model_version) if model_version is not None else None
    pool_root = params.get('neuron_pool', {}) if hasattr(params, 'get') else {}
    is_stage_partitioned_pool = (
        'attn_qk_read_shared' in pool_root
        or 'attn_qk_read_global' in pool_root)

    def _get_sharding(path, value):
        key_path = tuple(
            p.key if hasattr(p, 'key') else str(p) for p in path)
        path_str = '/'.join(str(p) for p in key_path)
        leaf = str(path[-1].key if hasattr(path[-1], 'key') else path[-1])
        if (path_str == 'token_emb/embedding'
                and (version == V4167_MODEL_VERSION
                     or _is_baseline_version(version))):
            return row_sharded
        if _is_baseline_version(version):
            if (len(key_path) >= 4
                    and str(key_path[0]).startswith('layer_')):
                module = key_path[1]
                submodule = key_path[2]
                name = key_path[3]
                if module == 'attn':
                    if submodule in ('q_proj', 'k_proj', 'v_proj'):
                        if name == 'kernel':
                            return col_sharded
                        if name == 'bias':
                            return vector_sharded
                    if submodule == 'o_proj' and name == 'kernel':
                        return row_sharded
                if module == 'ffn':
                    if submodule == 'Dense_0':
                        if name == 'kernel':
                            return col_sharded
                        if name == 'bias':
                            return vector_sharded
                    if submodule == 'Dense_1':
                        if name == 'kernel':
                            return row_sharded
                        if name == 'bias':
                            return replicated
            return replicated
        if version == V4167_MODEL_VERSION:
            if (path_str.startswith('router/proj_attn/kernel')
                    or path_str.startswith('router/proj_rst/kernel')
                    or path_str.startswith('router/q_op_write_query_proj')
                    or path_str.startswith('router/k_op_write_query_proj')
                    or path_str.startswith('router/v_op_write_query_proj')
                    or path_str.startswith('router/rst_op_write_query_proj')):
                return router_input_sharded_3d
            if (len(key_path) >= 4
                    and str(key_path[0]).startswith('block_')
                    and key_path[1:] == ('attn', 'expand_O', 'kernel')):
                return row_sharded
        # NeuronPool params: shard N axis (first dim) on 'model'
        if 'neuron_pool' in path_str:
            if (is_stage_partitioned_pool
                    and (leaf.endswith('_stage')
                         or leaf.endswith('_local'))):
                return stage_n_sharded_3d
            if (is_stage_partitioned_pool
                    and (leaf.endswith('_op_read_proj')
                         or leaf.endswith('_op_write_proj'))):
                return replicated
            if value.ndim == 2:
                return n_sharded       # [N, d_bn] or [N, D]
            elif value.ndim == 3:
                return n_sharded_3d    # [N, D, R] tensor pools
            else:
                return replicated
        return replicated

    # Build matching pytree of shardings
    return jax.tree.map_with_path(
        lambda path, x: _get_sharding(path, x), params)


def _print_param_sharding_summary(param_shardings, model_version):
    version = str(model_version)

    def _path_str(path):
        return '/'.join(str(p.key if hasattr(p, 'key') else p) for p in path)

    interesting = []
    for path, sharding in jax.tree.leaves_with_path(param_shardings):
        ps = _path_str(path)
        if _is_baseline_version(version):
            if ps == 'token_emb/embedding':
                interesting.append((ps, sharding))
            elif (ps.startswith('layer_0/attn/')
                    and ps.endswith('/kernel')):
                interesting.append((ps, sharding))
            elif (ps.startswith('layer_0/ffn/')
                  and ps.endswith('/kernel')):
                interesting.append((ps, sharding))
        elif version == V4167_MODEL_VERSION:
            if ps in (
                    'token_emb/embedding',
                    'router/proj_attn/kernel',
                    'router/proj_rst/kernel',
                    'router/q_op_write_query_proj',
                    'router/k_op_write_query_proj',
                    'router/v_op_write_query_proj',
                    'router/rst_op_write_query_proj',
                    'block_0/attn/expand_O/kernel'):
                interesting.append((ps, sharding))
            elif ps.startswith('neuron_pool/') and len(interesting) < 16:
                interesting.append((ps, sharding))
        elif version == V4168_MODEL_VERSION:
            if ps in (
                    'neuron_pool/attn_v_read',
                    'neuron_pool/attn_v_write',
                    'neuron_pool/rst_read',
                    'neuron_pool/rst_write',
                    'neuron_pool/attn_qk_read',
                    'neuron_pool/attn_qk_write'):
                interesting.append((ps, sharding))
    if not interesting:
        return
    print("\n=== Parameter sharding summary ===", flush=True)
    for ps, sharding in interesting:
        spec = getattr(sharding, 'spec', '<unknown>')
        print(f"  {ps}: {spec}", flush=True)


def shard_params_to_mesh(params, param_shardings):
    """Place params on mesh according to shardings."""
    return jax.tree.map(
        lambda p, s: jax.device_put(p, s),
        params, param_shardings)


def shard_to_mesh(data, sharding, global_shape):
    """Multi-host: create global array from host-local data.

    Uses make_array_from_callback which correctly maps mesh indices
    to data slices, regardless of how devices map to hosts.

    data: [per_host_batch, ...] -this host's data portion
    sharding: NamedSharding
    global_shape: (global_batch, ...)
    """
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    per_host = data.shape[0]

    def data_callback(index):
        # index is a tuple of slices for each dimension.
        # The batch slice tells us which global rows this device needs.
        batch_slice = index[0]
        start = batch_slice.start or 0
        stop = batch_slice.stop or global_shape[0]
        local_start = start - host_id * per_host
        local_stop = stop - host_id * per_host
        if 0 <= local_start < per_host:
            return np.array(data[local_start:local_stop])
        # Previously returned silent zeros -that corrupts training with
        # a zero-batch whenever the mesh's host locality doesn't match
        # the data partition. Fail loud instead so the misconfiguration
        # is caught at setup rather than showing up as mysterious loss.
        raise RuntimeError(
            f"shard_to_mesh: device requests global index [{start}, {stop}) "
            f"but host {host_id} has local range [0, {per_host}) "
            f"(local_start={local_start}). Mesh layout likely doesn't match "
            f"host locality. Check create_mesh() device order.")

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


# ============================================================
# Helpers
# ============================================================

def shard_batch(batch, n_devices):
    """Reshape a batch for pmap: (B, ...) -> (n_devices, B//n_devices, ...).

    If the batch is already sharded (leading dim == n_devices), return as-is.
    """
    if isinstance(batch, (tuple, list)):
        return type(batch)(shard_batch(x, n_devices) for x in batch)
    if batch.shape[0] == n_devices:
        return batch  # already sharded by data loader
    return batch.reshape(n_devices, batch.shape[0] // n_devices, *batch.shape[1:])


# ============================================================
# Evaluation loop
# ============================================================

def evaluate(eval_step_fn, params, val_loader, n_devices, max_batches=200,
             verbose=True, data_sharding_spec=None,
             return_dead_stats=False, return_prune_stats=False, current_step=0):
    """Run evaluation and return avg loss and accuracy.

    All hosts must call this (pmap requires it), but only verbose=True host prints.
    Accumulates on device -one TPU-to-CPU sync at the end instead of three
    per batch -so eval stays fast on 1B-scale runs.
    """
    total_loss_jax = jnp.float32(0.0)
    total_correct_jax = jnp.int32(0)
    total_valid_jax = jnp.int32(0)
    dead_attn_sum_jax = jnp.float32(0.0)
    dead_attn_max_jax = jnp.float32(0.0)
    dead_rst_sum_jax = jnp.float32(0.0)
    dead_rst_max_jax = jnp.float32(0.0)
    prune_compute_sum_jax = jnp.float32(0.0)
    prune_mass_sum_jax = jnp.float32(0.0)
    prune_den_sum_jax = jnp.float32(0.0)
    prune_den_min_jax = jnp.float32(1.0e30)
    prune_no_active_sum_jax = jnp.float32(0.0)
    prune_unpruned_den_sum_jax = jnp.float32(0.0)
    dead_batches = 0

    eval_total = min(max_batches, len(val_loader))
    eval_start = time.time()
    batch_idx = -1

    for batch_idx, (input_ids, attention_mask) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        if data_sharding_spec is not None:
            gb = input_ids.shape[0] * jax.process_count()
            gs = (gb, input_ids.shape[1])
            input_ids = shard_to_mesh(input_ids, data_sharding_spec, gs)
            attention_mask = shard_to_mesh(attention_mask, data_sharding_spec, gs)

        if return_dead_stats:
            eval_ret = eval_step_fn(
                params, input_ids, attention_mask, jnp.int32(current_step))
            if return_prune_stats:
                (ce_loss, correct, valid_count,
                 attn_dead_count, rst_dead_count,
                 prune_compute_frac, prune_gate_mass_retained,
                 prune_gate_den_mean, prune_gate_den_min,
                 prune_no_active_frac,
                 prune_unpruned_gate_den_mean) = eval_ret
                prune_compute_sum_jax = prune_compute_sum_jax + jnp.asarray(
                    prune_compute_frac, dtype=jnp.float32)
                prune_mass_sum_jax = prune_mass_sum_jax + jnp.asarray(
                    prune_gate_mass_retained, dtype=jnp.float32)
                prune_den_sum_jax = prune_den_sum_jax + jnp.asarray(
                    prune_gate_den_mean, dtype=jnp.float32)
                prune_den_min_jax = jnp.minimum(
                    prune_den_min_jax,
                    jnp.asarray(prune_gate_den_min, dtype=jnp.float32))
                prune_no_active_sum_jax = (
                    prune_no_active_sum_jax
                    + jnp.asarray(prune_no_active_frac, dtype=jnp.float32))
                prune_unpruned_den_sum_jax = prune_unpruned_den_sum_jax + jnp.asarray(
                    prune_unpruned_gate_den_mean, dtype=jnp.float32)
            else:
                (ce_loss, correct, valid_count,
                 attn_dead_count, rst_dead_count) = eval_ret
            attn_dead_count = jnp.asarray(attn_dead_count, dtype=jnp.float32)
            rst_dead_count = jnp.asarray(rst_dead_count, dtype=jnp.float32)
            # These are per-validation-batch dead statistics. They are not
            # persistent lifetime dead-neuron counts.
            dead_attn_sum_jax = dead_attn_sum_jax + attn_dead_count
            dead_attn_max_jax = jnp.maximum(dead_attn_max_jax, attn_dead_count)
            dead_rst_sum_jax = dead_rst_sum_jax + rst_dead_count
            dead_rst_max_jax = jnp.maximum(dead_rst_max_jax, rst_dead_count)
            dead_batches += 1
        else:
            ce_loss, correct, valid_count = eval_step_fn(
                params, input_ids, attention_mask, jnp.int32(current_step))

        total_loss_jax = total_loss_jax + ce_loss * valid_count.astype(jnp.float32)
        total_correct_jax = total_correct_jax + correct
        total_valid_jax = total_valid_jax + valid_count

    totals_payload = {
        'loss': total_loss_jax,
        'correct': total_correct_jax,
        'valid': total_valid_jax,
    }
    if return_dead_stats:
        totals_payload.update({
            'dead_attn_sum': dead_attn_sum_jax,
            'dead_attn_max': dead_attn_max_jax,
            'dead_rst_sum': dead_rst_sum_jax,
            'dead_rst_max': dead_rst_max_jax,
        })
        if return_prune_stats:
            totals_payload.update({
                'prune_compute_sum': prune_compute_sum_jax,
                'prune_mass_sum': prune_mass_sum_jax,
                'prune_den_sum': prune_den_sum_jax,
                'prune_den_min': prune_den_min_jax,
                'prune_no_active_sum': prune_no_active_sum_jax,
                'prune_unpruned_den_sum': prune_unpruned_den_sum_jax,
            })
    totals = jax.device_get(totals_payload)
    total_loss = float(totals['loss'])
    total_correct = int(totals['correct'])
    total_valid = int(totals['valid'])

    eval_elapsed = time.time() - eval_start
    done = min(batch_idx + 1, eval_total) if batch_idx >= 0 else 0
    if verbose:
        print(f"  Eval: {done}/{eval_total} batches, {eval_elapsed:.1f}s", flush=True)
    avg_loss = total_loss / total_valid if total_valid > 0 else 0.0
    avg_acc = total_correct / total_valid if total_valid > 0 else 0.0
    if return_dead_stats:
        denom = float(dead_batches) if dead_batches > 0 else 1.0
        dead_stats = {
            'val_attn_dead_mean': float(totals['dead_attn_sum']) / denom,
            'val_attn_dead_max': float(totals['dead_attn_max']),
            'val_rst_dead_mean': float(totals['dead_rst_sum']) / denom,
            'val_rst_dead_max': float(totals['dead_rst_max']),
            'val_dead_batches': int(dead_batches),
        }
        if return_prune_stats:
            dead_stats.update({
                'estimated_compute_frac': float(totals['prune_compute_sum']) / denom,
                'gate_mass_retained': float(totals['prune_mass_sum']) / denom,
                'prune_gate_den_mean': float(totals['prune_den_sum']) / denom,
                'prune_gate_den_min': float(totals['prune_den_min']),
                'prune_no_active_frac': float(totals['prune_no_active_sum']) / denom,
                'prune_unpruned_gate_den_mean': float(totals['prune_unpruned_den_sum']) / denom,
            })
        return avg_loss, avg_acc, dead_stats
    return avg_loss, avg_acc


def _format_prune_eps(eps):
    return f"{float(eps):.0e}".replace('-', 'm')


def run_eval_prune_sweep(eval_prune_step_fns, params, val_loader, n_devices,
                         data_sharding_spec, current_step, base_loss, base_acc,
                         verbose=False):
    records = {}
    for eps, step_fn in eval_prune_step_fns.items():
        val_loader.reset()
        loss, acc, stats = evaluate(
            step_fn, params, val_loader, n_devices, verbose=verbose,
            data_sharding_spec=data_sharding_spec, return_dead_stats=True,
            return_prune_stats=True, current_step=current_step)
        tag = _format_prune_eps(eps)
        records[f'val_loss_prune_eps_{tag}'] = loss
        records[f'val_acc_prune_eps_{tag}'] = acc
        records[f'val_loss_delta_prune_eps_{tag}'] = loss - base_loss
        records[f'val_acc_delta_prune_eps_{tag}'] = acc - base_acc
        records[f'estimated_compute_frac_prune_eps_{tag}'] = float(
            stats.get('estimated_compute_frac', 0.0))
        records[f'gate_mass_retained_prune_eps_{tag}'] = float(
            stats.get('gate_mass_retained', 1.0))
        records[f'prune_gate_den_mean_eps_{tag}'] = float(
            stats.get('prune_gate_den_mean', 0.0))
        records[f'prune_gate_den_min_eps_{tag}'] = float(
            stats.get('prune_gate_den_min', 0.0))
        records[f'prune_no_active_frac_eps_{tag}'] = float(
            stats.get('prune_no_active_frac', 0.0))
        records[f'prune_unpruned_gate_den_mean_eps_{tag}'] = float(
            stats.get('prune_unpruned_gate_den_mean', 0.0))
    return records


# ============================================================
# Orbax checkpoint save / load
# ============================================================


def _checkpoint_consumed_counts(step, training_config=None, full_config=None,
                                model_config=None):
    """Best-effort example/token counters for checkpoint metadata."""
    training_config = training_config or {}
    full_config = full_config or {}
    model_config = model_config or {}
    full_training = (
        full_config.get('training', {})
        if isinstance(full_config, dict) else {})
    full_model = (
        full_config.get('model', {})
        if isinstance(full_config, dict) else {})

    batch_size = training_config.get(
        'batch_size', full_training.get('batch_size'))
    max_seq_len = model_config.get(
        'max_seq_len', full_model.get('max_seq_len'))
    try:
        consumed_examples = int(step) * int(batch_size)
    except (TypeError, ValueError):
        consumed_examples = None
    try:
        consumed_tokens = int(consumed_examples) * int(max_seq_len)
    except (TypeError, ValueError):
        consumed_tokens = None
    return consumed_examples, consumed_tokens


LEGACY_FLAX_CHECKPOINT_ERROR = (
    "Legacy .flax checkpoints are not supported in this Orbax-only branch. "
    "Use the old branch or write a separate converter."
)


def _orbax_best_metric(metrics):
    if not isinstance(metrics, dict):
        return float('inf')
    val = metrics.get('val_loss')
    try:
        val = float(val)
    except (TypeError, ValueError):
        return float('inf')
    return val if np.isfinite(val) else float('inf')


def _checkpoint_manager_options_accepts(name):
    """Return whether the installed Orbax options object accepts a kwarg."""
    try:
        sig = inspect.signature(ocp.CheckpointManagerOptions)
    except (TypeError, ValueError, AttributeError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )


def _orbax_multiprocessing_options(primary_host=0):
    """Create Orbax MultiprocessingOptions across compatible API paths."""
    candidates = (
        ('options', 'MultiprocessingOptions'),
        ('checkpoint_manager', 'MultiprocessingOptions'),
        ('', 'MultiprocessingOptions'),
    )
    for module_name, attr_name in candidates:
        module = ocp if not module_name else getattr(ocp, module_name, None)
        cls = getattr(module, attr_name, None) if module is not None else None
        if cls is not None:
            return cls(primary_host=primary_host)
    raise RuntimeError(
        "Could not find Orbax MultiprocessingOptions API. "
        "Check orbax-checkpoint version compatibility."
    )


def _sync_orbax_checkpoint_root_created(name='orbax_checkpoint_root_created',
                                        context=None):
    """Strict host barrier before Orbax manager construction."""
    _strict_multihost_barrier(name, context=context)


def _orbax_manager_debug_context(name, checkpoint_dir, create, read_only,
                                 best_tracking, primary_host):
    startup = _MULTIHOST_STARTUP_CONTEXT or {}
    return {
        'barrier': name,
        'process_index': jax.process_index(),
        'process_count': jax.process_count(),
        'checkpoint_dir': str(checkpoint_dir),
        'create': bool(create),
        'read_only': bool(read_only),
        'best_tracking': bool(best_tracking),
        'primary_host': int(primary_host),
        'trainer_script': startup.get('trainer_script'),
        'config_path': startup.get('config_path'),
        'hostname': socket.gethostname(),
    }


def _strict_orbax_manager_barrier(name, checkpoint_dir, create, read_only,
                                  best_tracking, primary_host):
    context = _orbax_manager_debug_context(
        name, checkpoint_dir, create, read_only, best_tracking, primary_host)
    try:
        _strict_multihost_barrier(name, context=context)
    except Exception as exc:
        print(
            "Orbax CheckpointManager synchronization failure:\n"
            + json.dumps(
                context,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise


def _orbax_async_checkpointing_enabled():
    """Return whether this trainer may use Orbax async checkpointing."""
    # Keep checkpointing synchronous for correctness-first TPU runs even
    # though the trainer initializes jax.distributed before device access.
    return False


def _create_orbax_checkpoint_manager(checkpoint_dir, checkpoint_interval=1,
                                     keep_last=3, create=True,
                                     read_only=False,
                                     best_tracking=False):
    """Create the Orbax manager for Composite(state, metadata) checkpoints."""
    create = bool(create)
    read_only = bool(read_only)
    best_tracking = bool(best_tracking)
    primary_host = 0
    enable_async_checkpointing = _orbax_async_checkpointing_enabled()
    if enable_async_checkpointing:
        raise AssertionError(
            "Orbax async checkpointing must remain disabled for this "
            "correctness-first TPU trainer."
        )

    if create and not read_only:
        if jax.process_index() == primary_host:
            _ensure_orbax_checkpoint_root(checkpoint_dir)
        _strict_orbax_manager_barrier(
            "orbax_root_created:" + _stable_short_hash(checkpoint_dir),
            checkpoint_dir,
            create,
            read_only,
            best_tracking,
            primary_host,
        )

    manager_barrier_suffix = (
        _stable_short_hash(checkpoint_dir)
        + f":best={int(best_tracking)}"
        + f":create={int(create)}"
        + f":read={int(read_only)}"
    )

    options_kwargs = {
        'save_interval_steps': max(1, int(checkpoint_interval or 1)),
        'keep_checkpoints_without_metrics': True,
        'step_format_fixed_length': 12,
        'create': create,
        'read_only': read_only,
        'enable_async_checkpointing': enable_async_checkpointing,
    }
    if best_tracking:
        options_kwargs['best_fn'] = _orbax_best_metric
        options_kwargs['best_mode'] = 'min'
    if _checkpoint_manager_options_accepts('multiprocessing_options'):
        options_kwargs['multiprocessing_options'] = (
            _orbax_multiprocessing_options(primary_host=primary_host))
    if _checkpoint_manager_options_accepts('single_host_load_and_broadcast'):
        options_kwargs['single_host_load_and_broadcast'] = True
    if keep_last is not None and int(keep_last) > 0:
        # max_to_keep is the stable retention option across supported Orbax
        # versions.
        options_kwargs['max_to_keep'] = int(keep_last)

    if jax.process_index() == primary_host:
        print(
            f"Creating Orbax CheckpointManager: dir={checkpoint_dir} "
            f"create={create} read_only={read_only} "
            f"best_tracking={best_tracking} "
            f"primary_host={primary_host}",
            flush=True,
        )
        print(
            "Orbax checkpointing mode: synchronous "
            "(async disabled; jax.distributed initialized)",
            flush=True,
        )

    options = ocp.CheckpointManagerOptions(**options_kwargs)

    _strict_orbax_manager_barrier(
        "before_orbax_manager:" + manager_barrier_suffix,
        checkpoint_dir,
        create,
        read_only,
        best_tracking,
        primary_host,
    )
    try:
        manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
            item_names=('state', 'metadata'),
        )
    except Exception as exc:
        print(
            "Orbax CheckpointManager construction failure:\n"
            + json.dumps(
                _orbax_manager_debug_context(
                    "manager_construct:" + manager_barrier_suffix,
                    checkpoint_dir,
                    create,
                    read_only,
                    best_tracking,
                    primary_host),
                indent=2,
                sort_keys=True,
                default=str,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    _strict_orbax_manager_barrier(
        "after_orbax_manager:" + manager_barrier_suffix,
        checkpoint_dir,
        create,
        read_only,
        best_tracking,
        primary_host,
    )
    return manager


def _is_orbax_missing_dir_error(exc):
    """Return True if Orbax/GCS reports a missing checkpoint directory."""
    seen = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)

        if isinstance(cur, FileNotFoundError):
            return True

        msg = str(cur).lower()
        if isinstance(cur, ValueError) and 'does not exist' in msg:
            return True
        if 'no such file or directory' in msg:
            return True
        if 'filenotfounderror' in msg and 'checkpoints' in msg:
            return True

        stack.append(getattr(cur, '__cause__', None))
        stack.append(getattr(cur, '__context__', None))

    return False


def _parse_orbax_resume_target(resume_from):
    target = str(resume_from).strip().rstrip('/\\')
    if not target:
        return None, None
    if target.endswith('.flax'):
        raise ValueError(LEGACY_FLAX_CHECKPOINT_ERROR)
    name = _path_name(target)
    if name.isdigit():
        parent = _path_parent(target)
        if _path_name(parent) != 'checkpoints':
            raise ValueError(
                "--resume-from STEP is ambiguous; pass a run folder or "
                "run_folder/checkpoints/STEP.")
        return _path_parent(parent), int(name)
    if name == 'checkpoints':
        return _path_parent(target), None
    return target, None


def _orbax_step_from_path_name(name):
    """Return int step if name is a valid Orbax numeric step dir, else None."""
    name = str(name).rstrip('/\\')
    name = name.replace('\\', '/').rsplit('/', 1)[-1]
    if not name:
        return None
    if name.isdigit():
        return int(name)
    return None


def _orbax_commit_success_path(step_dir):
    return _join_path(step_dir, "commit_success.txt")


def _orbax_step_is_committed(step_dir):
    return _file_exists(_orbax_commit_success_path(step_dir))


def _list_orbax_steps_for_run(run_folder):
    """List Orbax checkpoint steps without constructing an Orbax manager.

    This is safe for host0-only discovery because it performs only
    filesystem/GCS listing and does not enter Orbax/JAX multihost barriers.
    """
    ckpt_dir = _join_path(run_folder, 'checkpoints')
    try:
        entries = _list_files(ckpt_dir, '*')
    except Exception as exc:
        msg = str(exc).lower()
        missing_dir_error = (
            'not found' in msg
            or 'no such' in msg
            or 'does not exist' in msg
            or '404' in msg
        )
        if missing_dir_error or _is_orbax_missing_dir_error(exc):
            return []
        raise

    steps = []
    skipped = []
    for path in entries:
        step = _orbax_step_from_path_name(_path_name(path))
        if step is None:
            continue
        if not _orbax_step_is_committed(path):
            skipped.append((step, path))
            continue
        steps.append(step)

    steps = sorted(set(int(step) for step in steps))

    if skipped and jax.process_index() == 0:
        for step, path in skipped[:8]:
            print(
                "Auto-resume discovery: skipping incomplete Orbax checkpoint "
                f"step={step} path={path} missing commit_success.txt",
                flush=True,
            )
        if len(skipped) > 8:
            print(
                "Auto-resume discovery: skipped "
                f"{len(skipped)} incomplete checkpoints total.",
                flush=True,
            )

    return steps


def _latest_orbax_step_for_run(run_folder):
    """Return latest checkpoint step using filesystem listing only."""
    steps = _list_orbax_steps_for_run(run_folder)
    return steps[-1] if steps else None


def _resolve_orbax_resume_from(resume_from):
    run_folder, requested_step = _parse_orbax_resume_target(resume_from)
    if run_folder is None:
        return None, None, False
    steps = _list_orbax_steps_for_run(run_folder)
    step_set = set(steps)

    if requested_step is not None:
        requested_step = int(requested_step)
        ckpt_dir = _join_path(run_folder, 'checkpoints')
        requested_step_dir = None
        try:
            entries = _list_files(ckpt_dir, '*')
        except Exception:
            entries = []
        for path in entries:
            step = _orbax_step_from_path_name(_path_name(path))
            if step == requested_step:
                requested_step_dir = path
                break
        if (
            requested_step_dir is not None
            and not _orbax_step_is_committed(requested_step_dir)
        ):
            raise RuntimeError(
                "Requested Orbax checkpoint step exists but is "
                "incomplete/uncommitted:\n"
                f"{requested_step_dir} is missing commit_success.txt")
        return run_folder, requested_step, requested_step in step_set
    if not steps:
        return run_folder, None, False
    return run_folder, int(steps[-1]), True


def _composite_item(restored, name):
    if isinstance(restored, dict):
        return restored.get(name)
    try:
        return restored[name]
    except Exception:
        pass
    if hasattr(restored, 'items'):
        try:
            return dict(restored.items()).get(name)
        except Exception:
            pass
    return None


def _restore_orbax_metadata(checkpoint_dir, step):
    manager = _create_orbax_checkpoint_manager(
        checkpoint_dir,
        create=False,
        read_only=True,
    )
    try:
        restored = manager.restore(
            int(step),
            args=ocp.args.Composite(metadata=ocp.args.JsonRestore()),
        )
        metadata = _composite_item(restored, 'metadata')
        return metadata if isinstance(metadata, dict) else {}
    finally:
        manager.close()


def _safe_git_info():
    def _run_git(args):
        try:
            proc = subprocess.run(
                ['git', *args],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        value = (proc.stdout or '').strip()
        return value or None

    return {
        'git_branch': _run_git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'git_commit': _run_git(['rev-parse', 'HEAD']),
    }


def _is_single_device_jax_array(x):
    jax_array_type = getattr(jax, "Array", None)
    if jax_array_type is None or not isinstance(x, jax_array_type):
        return False
    sharding = getattr(x, "sharding", None)
    return type(sharding).__name__ == "SingleDeviceSharding"


def _host_local_jax_array_to_numpy(x):
    return np.asarray(jax.device_get(x))


def _sanitize_host_local_checkpoint_leaf(x):
    """Convert host-local SingleDevice JAX arrays to NumPy for Orbax save.

    Orbax StandardSave in multi-host mode rejects host-local SingleDevice
    jax.Array leaves. Sharded global arrays, such as model params and large
    optimizer tensors, must be preserved as JAX arrays.
    """
    if _is_single_device_jax_array(x):
        return _host_local_jax_array_to_numpy(x)
    return x


def _sanitize_opt_state_for_orbax_save(opt_state):
    return jax.tree_util.tree_map(
        _sanitize_host_local_checkpoint_leaf,
        opt_state,
    )


def _rng_to_orbax_state_array(rng):
    arr = np.asarray(jax.device_get(rng), dtype=np.uint32)
    if arr.size != 2:
        raise ValueError(
            f"Expected PRNGKey with 2 uint32 values, got shape={arr.shape}."
        )
    return np.reshape(arr, (2,))


def _build_orbax_state(params, opt_state, rng, epoch, global_step,
                       step_in_epoch, steps_per_epoch, best_val_loss,
                       consumed_examples=None, consumed_tokens=None,
                       training_config=None, full_config=None,
                       model_config=None):
    if consumed_examples is None or consumed_tokens is None:
        inferred_examples, inferred_tokens = _checkpoint_consumed_counts(
            global_step, training_config, full_config, model_config)
        if consumed_examples is None:
            consumed_examples = inferred_examples
        if consumed_tokens is None:
            consumed_tokens = inferred_tokens
    return {
        'params': params,
        'opt_state': _sanitize_opt_state_for_orbax_save(opt_state),
        'rng': _rng_to_orbax_state_array(rng),
        'epoch': np.asarray(int(epoch), dtype=np.int64),
        'global_step': np.asarray(int(global_step), dtype=np.int64),
        'step': np.asarray(int(global_step), dtype=np.int64),
        'step_in_epoch': np.asarray(int(step_in_epoch), dtype=np.int64),
        'steps_per_epoch': np.asarray(int(steps_per_epoch), dtype=np.int64),
        'best_val_loss': np.asarray(float(best_val_loss), dtype=np.float64),
        'consumed_examples': np.asarray(
            -1 if consumed_examples is None else int(consumed_examples),
            dtype=np.int64),
        'consumed_tokens': np.asarray(
            -1 if consumed_tokens is None else int(consumed_tokens),
            dtype=np.int64),
    }


def _json_loss(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _build_orbax_metadata(run_id, global_step, epoch, step_in_epoch,
                          steps_per_epoch, best_val_loss, val_loss,
                          checkpoint_kind, model_config, training_config,
                          full_config, raw_config, config_path,
                          consumed_examples=None, consumed_tokens=None,
                          git_info=None):
    model_snapshot = _safe_config_snapshot(model_config or {})
    training_snapshot = _safe_config_snapshot(training_config or {})
    full_snapshot = _safe_config_snapshot(full_config or {})
    raw_snapshot = _safe_config_snapshot(raw_config or {})
    full_config_sha256 = _config_sha256(full_snapshot)
    if consumed_examples is None or consumed_tokens is None:
        inferred_examples, inferred_tokens = _checkpoint_consumed_counts(
            global_step, training_snapshot, full_snapshot, model_snapshot)
        if consumed_examples is None:
            consumed_examples = inferred_examples
        if consumed_tokens is None:
            consumed_tokens = inferred_tokens
    metadata = {
        'type': 'dawn_srw_orbax_checkpoint',
        'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
        'created_at': datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'run_id': str(run_id),
        'global_step': int(global_step),
        'epoch': int(epoch),
        'step_in_epoch': int(step_in_epoch),
        'steps_per_epoch': int(steps_per_epoch),
        'best_val_loss': _json_loss(best_val_loss),
        'val_loss': _json_loss(val_loss),
        'checkpoint_kind': str(checkpoint_kind),
        'consumed_examples': (
            None if consumed_examples is None else int(consumed_examples)),
        'consumed_tokens': (
            None if consumed_tokens is None else int(consumed_tokens)),
        'model_config': model_snapshot,
        'training_config': training_snapshot,
        'full_config': full_snapshot,
        'full_config_sha256': full_config_sha256,
        'selected_full_config_sha256': full_config_sha256,
        'raw_config': raw_snapshot,
        'config_path': str(config_path),
    }
    for key, value in (git_info or {}).items():
        if value:
            metadata[key] = value
    return _json_safe(metadata)


def _checkpoint_metrics(val_loss=None, best_val_loss=None, train_loss=None):
    metrics = {}
    if val_loss is not None:
        val = _json_loss(val_loss)
        if val is not None:
            metrics['val_loss'] = val
    if best_val_loss is not None:
        best = _json_loss(best_val_loss)
        if best is not None:
            metrics['best_val_loss'] = best
    if train_loss is not None:
        train = _json_loss(train_loss)
        if train is not None:
            metrics['train_loss'] = train
    return metrics


def _orbax_manager_has_step(manager, step):
    """Return True if a manager already knows about a checkpoint step."""
    try:
        steps = manager.all_steps()
    except Exception:
        return False
    try:
        return int(step) in {int(existing_step) for existing_step in steps}
    except Exception:
        return step in steps


def _is_orbax_step_already_exists_error(exc):
    """Detect Orbax duplicate-step save errors across supported versions."""
    if exc.__class__.__name__ == 'StepAlreadyExistsError':
        return True
    msg = str(exc).lower()
    return (
        'checkpoint for step' in msg
        and 'already exists' in msg
    )


def save_orbax_checkpoint(manager, params, opt_state, rng, epoch,
                          global_step, step_in_epoch, steps_per_epoch,
                          best_val_loss, model_config, training_config,
                          full_config, raw_config, config_path, run_id,
                          checkpoint_kind, val_loss=None, train_loss=None,
                          git_info=None, wait=False):
    checkpoint_step = int(global_step)
    if _orbax_manager_has_step(manager, checkpoint_step):
        if jax.process_index() == 0:
            print(
                "Orbax checkpoint save skipped: "
                f"step {checkpoint_step} already exists.",
                flush=True,
            )
        return False

    consumed_examples, consumed_tokens = _checkpoint_consumed_counts(
        checkpoint_step, training_config, full_config, model_config)
    state = _build_orbax_state(
        params, opt_state, rng, epoch, checkpoint_step, step_in_epoch,
        steps_per_epoch, best_val_loss,
        consumed_examples=consumed_examples,
        consumed_tokens=consumed_tokens,
        training_config=training_config,
        full_config=full_config,
        model_config=model_config,
    )
    metadata = _build_orbax_metadata(
        run_id, checkpoint_step, epoch, step_in_epoch, steps_per_epoch,
        best_val_loss, val_loss, checkpoint_kind, model_config,
        training_config, full_config, raw_config, config_path,
        consumed_examples=consumed_examples,
        consumed_tokens=consumed_tokens,
        git_info=git_info,
    )
    metrics = _checkpoint_metrics(
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        train_loss=train_loss,
    )
    if (jax.process_index() == 0
            and not getattr(save_orbax_checkpoint,
                            "_printed_serialization_check", False)):
        rng_leaf = state.get('rng')
        print(
            "Orbax state serialization check: "
            f"rng_type={type(rng_leaf).__name__} "
            f"rng_shape={getattr(rng_leaf, 'shape', None)} "
            f"rng_dtype={getattr(rng_leaf, 'dtype', None)}",
            flush=True,
        )
        save_orbax_checkpoint._printed_serialization_check = True
    try:
        saved = manager.save(
            checkpoint_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                metadata=ocp.args.JsonSave(metadata),
            ),
            metrics=metrics,
            force=True,
        )
    except Exception as exc:
        if _is_orbax_step_already_exists_error(exc):
            if jax.process_index() == 0:
                print(
                    "Orbax checkpoint save skipped after duplicate-step "
                    f"response: step {checkpoint_step} already exists.",
                    flush=True,
                )
            return False
        raise
    manager.check_for_errors()
    if wait:
        manager.wait_until_finished()
        manager.check_for_errors()
    return saved


def _restore_orbax_state(manager, step, target_state):
    restored = manager.restore(
        int(step),
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(target_state),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    state = _composite_item(restored, 'state')
    metadata = _composite_item(restored, 'metadata')
    return (state if isinstance(state, dict) else {},
            metadata if isinstance(metadata, dict) else {})


def _sharding_device_set(sharding):
    try:
        return set(sharding.device_set)
    except Exception:
        return None


def _mesh_device_set(mesh):
    try:
        return set(np.asarray(mesh.devices).reshape(-1).tolist())
    except Exception:
        return set(jax.devices())


def _is_full_mesh_sharding(value, mesh):
    sharding = getattr(value, 'sharding', None)
    if sharding is None:
        return False
    devs = _sharding_device_set(sharding)
    if devs is None:
        return False
    return devs == _mesh_device_set(mesh)


def _maybe_get_opt_state_count(opt_state):
    candidates = []
    try:
        candidates.append(opt_state.count)
    except Exception:
        pass
    try:
        candidates.append(opt_state[2].count)
    except Exception:
        pass
    try:
        candidates.append(opt_state[0].count)
    except Exception:
        pass

    for value in candidates:
        try:
            return int(jax.device_get(value))
        except Exception:
            continue
    return None


def _debug_leaf_device_ids(x):
    try:
        return sorted(int(d.id) for d in x.sharding.device_set)
    except Exception:
        try:
            return sorted(int(d.id) for d in x.devices())
        except Exception:
            return "unknown"


def _put_leaf_on_template_or_replicated_mesh(restored_val, template_val, mesh):
    """Match dtype/shape, then place on template sharding or replicated mesh.

    For scalar optimizer leaves such as opt_state count, optimizer.init can
    create a SingleDeviceSharding leaf on device 0. Such leaves are incompatible
    with jitted computations whose params live on the full mesh, so they must
    be replicated to NamedSharding(mesh, P()).
    """
    if template_val is None:
        return restored_val

    dtype = getattr(template_val, 'dtype', None)
    restored_arr = jnp.asarray(restored_val, dtype=dtype)

    if hasattr(template_val, 'shape'):
        restored_arr = jnp.reshape(restored_arr, template_val.shape)

    replicated = NamedSharding(mesh, P())

    # Scalars and any incomplete/single-device leaves should be replicated
    # across the full mesh.
    if getattr(template_val, 'shape', ()) == ():
        return jax.device_put(restored_arr, replicated)

    if not _is_full_mesh_sharding(template_val, mesh):
        return jax.device_put(restored_arr, replicated)

    sharding = getattr(template_val, 'sharding', None)
    if sharding is not None:
        return jax.device_put(restored_arr, sharding)

    return jax.device_put(restored_arr, replicated)


def _match_tree_to_template_on_mesh(restored_tree, template_tree, mesh, *, name):
    try:
        return jax.tree.map(
            lambda restored_val, template_val:
                _put_leaf_on_template_or_replicated_mesh(
                    restored_val, template_val, mesh),
            restored_tree,
            template_tree,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to match restored {name} to target mesh/template."
        ) from exc


def _replicate_optimizer_state_scalars_to_mesh(opt_state, mesh):
    """Ensure fresh optimizer state has mesh-compatible scalar leaves."""
    replicated = NamedSharding(mesh, P())

    def _fix_leaf(x):
        if not hasattr(x, 'shape'):
            return x
        if x.shape == ():
            return jax.device_put(jnp.asarray(x), replicated)

        # If optimizer.init produced a non-mesh leaf, replicate it.
        if not _is_full_mesh_sharding(x, mesh):
            return jax.device_put(jnp.asarray(x), replicated)

        return x

    return jax.tree.map(_fix_leaf, opt_state)


def _state_scalar(state, key, default=None, cast=None):
    value = state.get(key, default)
    try:
        value = jax.device_get(value)
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        if value.size == 0:
            value = default
        else:
            value = value.reshape(-1)[0].item()
    elif hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    if cast is not None and value is not None:
        return cast(value)
    return value


# ============================================================
# Logging
# ============================================================

class GCSLogger:
    """Logger that writes to a local file and syncs to GCS on sync().

    GCS doesn't support true append -each open('a')/write/close overwrites.
    So we always append to a local file and upload the full file to GCS
    on every sync() call. Callers decide the sync cadence (training
    loop syncs once per FAST log boundary); the logger itself doesn't
    throttle. Uploading the whole file every FAST log is cheap in
    GCS API cost ($5 per 1M write ops) and in host-0 wall time
    (percent-of-a-percent over a multi-hour run), and the
    near-real-time visibility is worth it.
    """

    def __init__(self, gcs_path, local_path, resume=False):
        self.gcs_path = gcs_path
        self.local_path = local_path
        self._dirty = False
        if local_path:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if resume and gcs_path and local_path:
            # Seed the local file from existing GCS contents so the
            # subsequent open('a')/write path appends in-place; without
            # the seed the first sync() would overwrite GCS with only
            # this session's tail. If the GCS file doesn't exist yet we
            # silently continue (fresh-looking logger).
            try:
                with _open_file(gcs_path, 'rb') as f:
                    data = f.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
            except FileNotFoundError:
                pass
            except Exception as e:
                if jax.process_index() == 0:
                    print(f"  [warn] could not seed log from {gcs_path}: {e}", flush=True)

    def write(self, text):
        with open(self.local_path, 'a') as f:
            f.write(text)
        self._dirty = True

    def sync(self):
        """Upload local file to GCS if there are unflushed writes."""
        if not self._dirty or not self.gcs_path:
            return
        try:
            with open(self.local_path, 'rb') as f:
                data = f.read()
            with _open_file(self.gcs_path, 'wb') as f:
                f.write(data)
            self._dirty = False
        except Exception as e:
            if jax.process_index() == 0:
                print(f"  [warn] GCS sync failed: {e}", flush=True)


# Module-level loggers -set up in main()
_train_logger = None
_jsonl_logger = None


def _setup_loggers(training_log_file, jsonl_log_file, resume=False):
    """Create training text + JSONL loggers.

    resume=True downloads existing GCS content to the local scratch file
    first so new lines append rather than overwrite.
    """
    global _train_logger, _jsonl_logger
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "dawn_logs"
    tmpdir.mkdir(parents=True, exist_ok=True)

    if _is_gcs(training_log_file):
        local_txt = str(tmpdir / Path(training_log_file).name)
        _train_logger = GCSLogger(training_log_file, local_txt, resume=resume)
    else:
        _train_logger = GCSLogger(None, training_log_file, resume=resume)

    if _is_gcs(jsonl_log_file):
        local_jsonl = str(tmpdir / Path(jsonl_log_file).name)
        _jsonl_logger = GCSLogger(jsonl_log_file, local_jsonl, resume=resume)
    else:
        _jsonl_logger = GCSLogger(None, jsonl_log_file, resume=resume)


def sync_logs():
    """Flush local logs to GCS for live visibility."""
    if _train_logger:
        _train_logger.sync()
    if _jsonl_logger:
        _jsonl_logger.sync()


def log_message(msg, log_file=None):
    """Print and write to training log file. Host 0 only."""
    if jax.process_index() != 0:
        return
    print(msg, flush=True)
    if _train_logger:
        try:
            _train_logger.write(msg + '\n')
        except Exception as e:
            print(f"  [warn] log_message write failed: {e}", flush=True)


def format_time(seconds):
    """Format seconds to H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def log_jsonl(record):
    """Append a JSON-lines record to the JSONL log file. Host 0 only."""
    if jax.process_index() != 0:
        return
    if not _jsonl_logger:
        return
    try:
        line = json.dumps(record, default=str)
        _jsonl_logger.write(line + '\n')
    except Exception as e:
        print(f"  [warn] log_jsonl write failed: {e}", flush=True)


def _dead_pool_totals(ctx):
    attn_total = float(ctx.get('n_qk_cfg', 0) + ctx.get('n_v_cfg', 0))
    rst_total = float(ctx.get(
        'n_rst_cfg',
        ctx.get('n_know_cfg', 0)))
    return attn_total, rst_total


def _attach_validation_dead_fractions(rec, ctx):
    attn_total, rst_total = _dead_pool_totals(ctx)
    a_mean = float(rec.get(
        'val_attn_dead_mean',
        rec.get('attn_dead_count', 0.0)))
    a_max = float(rec.get(
        'val_attn_dead_max',
        rec.get('attn_dead_count', 0.0)))
    rst_mean = float(rec.get(
        'val_rst_dead_mean',
        rec.get('rst_dead_count', 0.0)))
    rst_max = float(rec.get(
        'val_rst_dead_max',
        rec.get('rst_dead_count', 0.0)))
    rec['val_attn_dead_mean'] = a_mean
    rec['val_attn_dead_max'] = a_max
    rec['val_rst_dead_mean'] = rst_mean
    rec['val_rst_dead_max'] = rst_max
    rec['val_attn_dead_frac_mean'] = (
        a_mean / attn_total if attn_total > 0.0 else 0.0)
    rec['val_attn_dead_frac_max'] = (
        a_max / attn_total if attn_total > 0.0 else 0.0)
    rec['val_rst_dead_frac_mean'] = (
        rst_mean / rst_total if rst_total > 0.0 else 0.0)
    rec['val_rst_dead_frac_max'] = (
        rst_max / rst_total if rst_total > 0.0 else 0.0)
    return rec


def _print_validation_dead_stats(rec, ctx):
    rec = _attach_validation_dead_fractions(dict(rec), ctx)
    log_message(
        f"  dead_val[a_mean={rec['val_attn_dead_mean']:.1f}"
        f" a_max={rec['val_attn_dead_max']:.1f}"
        f" rst_mean={rec['val_rst_dead_mean']:.1f}"
        f" rst_max={rec['val_rst_dead_max']:.1f}]"
        f" | dead_frac[a_mean={rec['val_attn_dead_frac_mean'] * 100:.3f}%"
        f" a_max={rec['val_attn_dead_frac_max'] * 100:.3f}%"
        f" rst_mean={rec['val_rst_dead_frac_mean'] * 100:.3f}%"
        f" rst_max={rec['val_rst_dead_frac_max'] * 100:.3f}%]"
    )


def check_nan_inf(metrics_dict, global_step, epoch):
    """Check for NaN/INF in loss metrics. Returns True if NaN/INF detected."""
    total = metrics_dict.get('total_loss', 0.0)
    if np.isnan(total) or np.isinf(total):
        if jax.process_index() == 0:
            print(f"\n[WARNING] NaN/INF detected at step {global_step}!")
            print(f"  total_loss: {total}")
            print(f"  ce_loss:    {metrics_dict.get('ce_loss', 'N/A')}")
            print(f"  aux_loss:   {metrics_dict.get('aux_loss', 'N/A')}")
            print(f"  tau_reg:    {metrics_dict.get('tau_reg', 'N/A')}")
            print(f"  orth_loss:  {metrics_dict.get('orth_loss', 'N/A')}")
            print(f"  div_loss:   {metrics_dict.get('div_loss', 'N/A')}")
        return True
    return False


# ============================================================
# 2-tier periodic logging (REGULAR / ANALYSIS)
# ============================================================
#
# REGULAR  every log_interval steps                        (default 100)
# ANALYSIS every log_interval * log_analysis_multiplier steps
#
# v4164: ANALYSIS is not emitted on every REGULAR tick. The distribution /
# boundary / saturation stats require a separate forward with the full-stats
# shard_map kernels (analysis_step), so the multiplier controls that cost.
#
# REGULAR carries the training-dynamics block (loss, activity, tau
# adds distribution-shape / boundary /
# saturation diagnostics.
#
# _build_analysis_record accepts `base={}` on the new path -the
# `base`/`metrics` split is kept so ANALYSIS
# record is now standalone.




def _fmt_act_count(frac, total):
    """Format 'XX.X%(N)' -active fraction with the implied count."""
    return f"{frac * 100:.1f}%({int(round(frac * total))})"


SPARSITY_LOG_POOLS = (
    ('attn_qk', 'qk'),
    ('attn_q', 'q'),
    ('attn_k', 'k'),
    ('attn_v', 'v'),
    ('rst', 'rst'),
)
SPARSITY_LOG_POOLS_COMPACT = (
    ('attn_q', 'q'),
    ('attn_k', 'k'),
    ('attn_v', 'v'),
    ('rst', 'rst'),
)
SPARSITY_CURRENT_EPS_LOG = (
    ('1e-6', '1e_6'),
    ('1e-5', '1e_5'),
    ('1e-4', '1e_4'),
    ('1e-3', '1e_3'),
    ('1e-2', '1e_2'),
    ('1e-1', '1e_1'),
)
SPARSITY_CURRENT_EPS_LOG_COMPACT = (
    ('1e-1', '1e_1'),
    ('1e-2', '1e_2'),
    ('1e-3', '1e_3'),
)
SPARSITY_PROJECTED_EPS_LOG = (
    ('1e-6', '1e_6'),
    ('1e-4', '1e_4'),
    ('1e-3', '1e_3'),
)
SPARSITY_PROJECTED_EPS_LOG_COMPACT = (
    ('1e-6', '1e_6'),
)
SPARSITY_MARGIN_BAND_LOG = (
    ('>0', 'margin_band_gt_0'),
    ('[-0.01,0]', 'margin_band_m0_01_0'),
    ('[-0.03,-0.01]', 'margin_band_m0_03_m0_01'),
    ('[-0.10,-0.03]', 'margin_band_m0_10_m0_03'),
    ('<-0.10', 'margin_band_lt_m0_10'),
)
SPARSITY_MARGIN_BAND_LOG_COMPACT = (
    ('pos', 'margin_band_pos'),
    ('near', 'margin_band_near_m0_03_0'),
    ('far', 'margin_band_far_lt_m0_10'),
)


def _fmt_sparsity_frac_count(rec, pool, prefix):
    frac = float(rec.get(f'{pool}_{prefix}_frac', 0.0) or 0.0)
    count = float(rec.get(f'{pool}_{prefix}_count', 0.0) or 0.0)
    return f"{frac * 100:.2f}%({count:.2f})"


def _fmt_optional_pct_count(value, count):
    pct = _fmt_optional_pct(value)
    if count is None:
        return pct
    return f"{pct}({count:.2f})"


def _fmt_sparsity_mass(rec, pool, key):
    mass = float(rec.get(f'{pool}_{key}', 0.0) or 0.0)
    return f"{mass * 100:.2f}%"


def _fmt_sparsity_pool_values(rec, formatter, pools=SPARSITY_LOG_POOLS):
    return " ".join(
        f"{label}={formatter(pool)}"
        for pool, label in pools)


def _rec_float(rec, key, default=0.0):
    return float(rec.get(key, default) or 0.0)


def _optional_rec_float(rec, key):
    if key not in rec:
        return None
    try:
        return float(rec[key])
    except Exception:
        return None


def _fmt_optional_pct(value):
    if value is None:
        return "nan%"
    return f"{100.0 * value:.1f}%"


def _first_optional_rec_float(rec, keys):
    for key in keys:
        value = _optional_rec_float(rec, key)
        if value is not None:
            return value
    return None


def _active_tau_count_for_label(rec, label, frac_key):
    count_keys = {
        'q': ('attn_q_active_tau_count', 'q_active_tau_count'),
        'k': ('attn_k_active_tau_count', 'k_active_tau_count'),
        'qk': ('attn_qk_active_tau_count', 'qk_active_tau_count'),
        'v': ('attn_v_active_tau_count', 'v_active_tau_count'),
        'rst': ('rst_active_tau_count',),
    }.get(label, ())
    count = _first_optional_rec_float(rec, count_keys)
    if count is not None:
        return count
    frac = _optional_rec_float(rec, frac_key)
    if frac is None:
        return None
    page_prefix = {
        'q': 'attn_qk',
        'k': 'attn_qk',
        'qk': 'attn_qk',
        'v': 'attn_v',
        'rst': 'rst',
    }.get(label)
    if page_prefix is None:
        return None
    if frac_key.endswith('_local'):
        denom_keys = (
            f'{label}_valid_candidate_count',
            f'{page_prefix}_candidate_valid_ops',
            f'{page_prefix}_candidate_valid_count',
        )
    else:
        denom_keys = (
            f'{label}_full_pool_size',
            f'{page_prefix}_full_pool_size',
        )
    denom = _first_optional_rec_float(rec, denom_keys)
    if denom is None:
        return None
    return frac * denom


def _fmt_active_label_pct_count(rec, label, scope):
    key = f'{label}_active_{scope}'
    return _fmt_optional_pct_count(
        _optional_rec_float(rec, key),
        _active_tau_count_for_label(rec, label, key))


def _attach_page_aware_metrics(rec, ctx=None):
    ctx = ctx or {}

    def _g(key, default=0.0):
        try:
            return float(rec.get(key, default) or 0.0)
        except Exception:
            return float(default)

    def _full_size(prefix, cfg_key):
        valid = _g(f'{prefix}_candidate_valid_ops', 0.0)
        frac = _g(f'{prefix}_candidate_valid_frac', 0.0)
        if valid > 0.0 and frac > 0.0:
            return valid / frac
        return float(ctx.get(cfg_key, valid if valid > 0.0 else 1.0) or 1.0)

    page_prefixes = {
        'qk': ('attn_qk', 'n_qk_cfg'),
        'v': ('attn_v', 'n_v_cfg'),
        'rst': ('rst', 'n_rst_cfg'),
    }
    for label, (prefix, cfg_key) in page_prefixes.items():
        full = max(_full_size(prefix, cfg_key), 1.0)
        valid = _g(f'{prefix}_candidate_valid_ops', full)
        if valid <= 0.0:
            valid = full
        cand = _g(f'{prefix}_candidate_ops', valid)
        valid_frac = valid / full
        rec[f'{label}_candidate_count'] = cand
        rec[f'{label}_valid_candidate_count'] = valid
        rec[f'{label}_full_pool_size'] = full
        rec[f'{label}_candidate_valid_frac'] = valid_frac
        rec[f'{prefix}_candidate_ops'] = cand
        rec[f'{prefix}_candidate_valid_ops'] = valid
        rec[f'{prefix}_candidate_valid_frac'] = valid_frac
        rec[f'{prefix}_full_pool_size'] = full

        admission = _g(f'{prefix}_admission_den_sum', _g(
            f'{prefix}_gate_den_sum_mean', 0.0))
        execution = _g(f'{prefix}_execution_mass_sum', _g(
            f'{prefix}_gate_sum', 0.0))
        eff = _g(f'{prefix}_execution_eff_n', _g(
            f'{prefix}_gate_eff_n', 0.0))
        for metric, value in (
                ('admission', admission),
                ('execution', execution),
                ('eff', eff)):
            local = value / max(valid, 1.0e-8)
            pool = value / max(full, 1.0e-8)
            rec[f'{label}_{metric}_local'] = local
            rec[f'{label}_{metric}_pool'] = pool
            rec[f'{prefix}_{metric}_local'] = local
            rec[f'{prefix}_{metric}_pool'] = pool

    active_specs = (
        ('q', 'attn_q', 'attn_qk'),
        ('k', 'attn_k', 'attn_qk'),
        ('qk', 'attn_qk', 'attn_qk'),
        ('v', 'attn_v', 'attn_v'),
        ('rst', 'rst', 'rst'),
    )
    for label, metric_prefix, page_prefix in active_specs:
        local = _optional_rec_float(rec, f'{metric_prefix}_active_tau_frac')
        if local is None:
            local = _optional_rec_float(rec, f'{label}_active_tau_frac')
        if local is None:
            continue
        frac = _g(f'{page_prefix}_candidate_valid_frac', 1.0)
        if frac <= 0.0:
            frac = 1.0
        pool = float(local) * max(frac, 0.0)
        rec[f'{label}_active_local'] = float(local)
        rec[f'{label}_active_pool'] = pool
        rec[f'{metric_prefix}_active_local'] = float(local)
        rec[f'{metric_prefix}_active_pool'] = pool

    return rec


def _active_tau_display_value(rec, active_tau_keys, active_key):
    value = _first_optional_rec_float(rec, active_tau_keys)
    fallback = _optional_rec_float(rec, active_key)
    if (
        fallback is not None
        and (value is None or (abs(value) <= 1.0e-12 and abs(fallback) > 1.0e-12))
    ):
        return fallback
    return value


def _print_active_tau_regular_line(rec):
    if any(f'{name}_active_local' in rec for name in ('q', 'k', 'qk', 'v', 'rst')):
        log_message(
            f"  active_local: q={_fmt_active_label_pct_count(rec, 'q', 'local')}"
            f" k={_fmt_active_label_pct_count(rec, 'k', 'local')}"
            f" qk={_fmt_active_label_pct_count(rec, 'qk', 'local')}"
            f" v={_fmt_active_label_pct_count(rec, 'v', 'local')}"
            f" rst={_fmt_active_label_pct_count(rec, 'rst', 'local')}"
        )
        log_message(
            f"  active_pool: q={_fmt_active_label_pct_count(rec, 'q', 'pool')}"
            f" k={_fmt_active_label_pct_count(rec, 'k', 'pool')}"
            f" qk={_fmt_active_label_pct_count(rec, 'qk', 'pool')}"
            f" v={_fmt_active_label_pct_count(rec, 'v', 'pool')}"
            f" rst={_fmt_active_label_pct_count(rec, 'rst', 'pool')}"
        )
        return
    explicit_keys = (
        'q_active_tau_frac', 'k_active_tau_frac', 'qk_active_tau_frac',
        'v_active_tau_frac', 'rst_active_tau_frac',
        'attn_q_active_tau_frac', 'attn_k_active_tau_frac',
        'attn_qk_active_tau_frac', 'attn_v_active_tau_frac',
    )
    if (
        rec.get('_active_tau_regular_available') is False
        and not any(key in rec for key in explicit_keys)
    ):
        return
    q = _active_tau_display_value(
        rec, ('q_active_tau_frac', 'attn_q_active_tau_frac'), 'attn_q_active')
    k = _active_tau_display_value(
        rec, ('k_active_tau_frac', 'attn_k_active_tau_frac'), 'attn_k_active')
    qk = _first_optional_rec_float(
        rec, ('qk_active_tau_frac', 'attn_qk_active_tau_frac'))
    v = _active_tau_display_value(
        rec, ('v_active_tau_frac', 'attn_v_active_tau_frac'), 'attn_v_active')
    rst = _active_tau_display_value(
        rec, ('rst_active_tau_frac',), 'rst_active')
    if (
        q is not None
        and k is not None
        and (qk is None or (abs(qk) <= 1.0e-12 and abs(q + k) > 1.0e-12))
    ):
        qk = 0.5 * (q + k)
    if all(value is None for value in (q, k, qk, v, rst)):
        return
    log_message(
        f"  active_tau: q={_fmt_optional_pct(q)}"
        f" k={_fmt_optional_pct(k)}"
        f" qk={_fmt_optional_pct(qk)}"
        f" v={_fmt_optional_pct(v)}"
        f" rst={_fmt_optional_pct(rst)}"
    )


def _print_regular_host_timing(raw_step_time_window, logging_time, ctx):
    mode = str(ctx.get('regular_console_host_timing', 'always')).lower()
    if mode in ('off', 'false', 'none'):
        return
    if mode == 'warn_only':
        threshold = float(ctx.get(
            'regular_console_logging_overhead_warn', 0.05))
        ratio = logging_time / max(raw_step_time_window, 1.0e-8)
        if threshold > 0.0 and ratio > threshold:
            log_message(
                f"warn: logging_overhead={ratio * 100:.1f}% "
                f"raw_step_time_window={raw_step_time_window:.3f}s "
                f"logging_time={logging_time:.3f}s")
        return
    log_message(
        f"  host_timing: raw_step_time_window={raw_step_time_window:.3f}s "
        f"logging_time={logging_time:.3f}s")


def _print_v4164_soft_sparsity_block(rec, level='compact'):
    compact = str(level or 'compact').lower() == 'compact'
    current_eps = (
        SPARSITY_CURRENT_EPS_LOG_COMPACT if compact
        else SPARSITY_CURRENT_EPS_LOG)
    projected_eps = (
        SPARSITY_PROJECTED_EPS_LOG_COMPACT if compact
        else SPARSITY_PROJECTED_EPS_LOG)
    margin_bands = (
        SPARSITY_MARGIN_BAND_LOG_COMPACT if compact
        else SPARSITY_MARGIN_BAND_LOG)
    pools = SPARSITY_LOG_POOLS_COMPACT if compact else SPARSITY_LOG_POOLS
    log_message(
        "  active_tau: "
        + _fmt_sparsity_pool_values(
            rec,
            lambda pool: _fmt_sparsity_frac_count(
                rec, pool, 'active_tau'),
            pools=pools))
    for eps_label, suffix in current_eps:
        line = (
            f"  gate@{eps_label}: "
            + _fmt_sparsity_pool_values(
                rec,
                lambda pool, suffix=suffix: _fmt_sparsity_frac_count(
                    rec, pool, f'active_eps_{suffix}'),
                pools=pools))
        if not compact:
            line += (
                " | mass "
                + _fmt_sparsity_pool_values(
                    rec,
                    lambda pool, suffix=suffix: _fmt_sparsity_mass(
                        rec, pool, f'mass_eps_{suffix}'),
                    pools=pools))
        log_message(line)
    for eps_label, suffix in projected_eps:
        line = (
            f"  projected_Tfinal@{eps_label}: "
            + _fmt_sparsity_pool_values(
                rec,
                lambda pool, suffix=suffix: _fmt_sparsity_frac_count(
                    rec, pool, f'projected_Tfinal_active_eps_{suffix}'),
                pools=pools))
        if not compact:
            line += (
                " | mass "
                + _fmt_sparsity_pool_values(
                    rec,
                    lambda pool, suffix=suffix: _fmt_sparsity_mass(
                        rec, pool, f'projected_Tfinal_mass_eps_{suffix}'),
                    pools=pools))
        log_message(line)
    log_message(
        "  margin_bands: "
        + " ".join(
            f"{band}["
            + _fmt_sparsity_pool_values(
                rec,
                lambda pool, key=key: _fmt_sparsity_mass(rec, pool, key),
                pools=pools)
            + "]"
            for band, key in margin_bands))


def _print_v4164_sparsity_block(rec):
    if any(f'{name}_active_local' in rec for name in ('q', 'k', 'qk', 'v', 'rst')):
        _print_active_tau_regular_line(rec)

        def _fmt_local_pool(pool, prefix):
            local = _rec_float(rec, f'{pool}_{prefix}_frac', 0.0)
            frac = _rec_float(rec, f'{pool}_candidate_valid_frac', 1.0)
            count = _optional_rec_float(rec, f'{pool}_{prefix}_count')
            count_part = "" if count is None else f"({count:.2f})"
            return f"{local * 100:.2f}%/{local * frac * 100:.2f}%{count_part}"

        for eps_label, suffix in (('1e-2', '1e_2'), ('1e-1', '1e_1')):
            log_message(
                f"  admission_local/pool@{eps_label}: "
                f"qk={_fmt_local_pool('attn_qk', f'admission_active_eps_{suffix}')} "
                f"v={_fmt_local_pool('attn_v', f'admission_active_eps_{suffix}')} "
                f"rst={_fmt_local_pool('rst', f'admission_active_eps_{suffix}')}")
        for eps_label, suffix in (
                ('1e-4', '1e_4'),
                ('1e-3', '1e_3'),
                ('1e-2', '1e_2')):
            log_message(
                f"  weight_local/pool@{eps_label}: "
                f"qk={_fmt_local_pool('attn_qk', f'active_eps_{suffix}')} "
                f"v={_fmt_local_pool('attn_v', f'active_eps_{suffix}')} "
                f"rst={_fmt_local_pool('rst', f'active_eps_{suffix}')}")
        return
    qkv_rst_pools = (
        ('attn_qk', 'qk'),
        ('attn_v', 'v'),
        ('rst', 'rst'),
    )
    log_message(
        "  active_tau: "
        + _fmt_sparsity_pool_values(
            rec,
            lambda pool: _fmt_sparsity_frac_count(rec, pool, 'active_tau'),
            pools=SPARSITY_LOG_POOLS))
    for eps_label, suffix in (('1e-2', '1e_2'), ('1e-1', '1e_1')):
        log_message(
            f"  admission@{eps_label}: "
            + _fmt_sparsity_pool_values(
                rec,
                lambda pool, suffix=suffix: _fmt_sparsity_frac_count(
                    rec, pool, f'admission_active_eps_{suffix}'),
                pools=qkv_rst_pools))
    for eps_label, suffix in (
            ('1e-4', '1e_4'),
            ('1e-3', '1e_3'),
            ('1e-2', '1e_2')):
        log_message(
            f"  weight@{eps_label}: "
            + _fmt_sparsity_pool_values(
                rec,
                lambda pool, suffix=suffix: _fmt_sparsity_frac_count(
                    rec, pool, f'active_eps_{suffix}'),
                pools=qkv_rst_pools))


def _soft_gate_schedule_shape(ctx):
    schedule = str(ctx.get('soft_gate_schedule', 'cosine')).lower()
    if schedule == 'log_gompertz':
        center = float(ctx.get('soft_gate_t_gompertz_center', 0.25))
        steepness = float(ctx.get('soft_gate_t_gompertz_steepness', 8.0))
        return schedule, f" center={center} steepness={steepness}"
    if schedule == 'log_power':
        power = float(ctx.get('soft_gate_t_power', 4.0))
        return schedule, f" power={power:.4g}"
    return schedule, ""


def _fixed_depth_pool_scale_from_ctx(ctx):
    model_cfg = ctx.get('model_config') or ctx.get('model_cfg') or {}
    d_model = ctx.get('d_model_cfg', ctx.get('d_model', model_cfg.get('d_model')))
    n_layers = ctx.get(
        'n_layers_cfg', ctx.get('n_layers', model_cfg.get('n_layers')))
    try:
        d_model = float(d_model)
        n_layers = float(n_layers)
    except (TypeError, ValueError):
        return 0.0
    if d_model <= 0.0 or n_layers <= 0.0:
        return 0.0
    return math.sqrt(d_model / n_layers)


def _build_regular_record(metrics, win_avgs, ctx, global_step, epoch):
    """REGULAR tier: all training-dynamics fields needed for live monitoring.

    Equivalent to the former FAST + DEEP record merged; consumers of the
    old train_fast / train_deep JSONL types should switch to type='train'.
    """
    m = metrics
    fixed_pool_scale = _fixed_depth_pool_scale_from_ctx(ctx)
    is_rw_key_model = _is_rw_key_srw_version(ctx.get('model_version'))

    def _metric_present(*keys):
        return any(key in m for key in keys)

    def _arr(key):
        try:
            return [float(v) for v in np.asarray(
                jax.device_get(m.get(key, jnp.zeros((0,), dtype=jnp.float32)))
            ).reshape(-1)]
        except Exception:
            return []

    rec = {
        'step': global_step,
        'epoch': epoch,
        # Loss components (window-averaged).
        'total_loss': win_avgs['loss'],
        'ce_loss': win_avgs['ce'],
        'aux_loss': win_avgs['aux'],
        'tau_reg': win_avgs['tau_reg'],
        'orth_loss': win_avgs['orth'],
        'div_loss': win_avgs['div'],
        'aux_loss_raw': float(m.get('aux_loss_raw', win_avgs['aux'])),
        'aux_loss_weighted': float(m.get(
            'aux_loss_weighted', ctx['lb_weight'] * win_avgs['aux'])),
        'load_balance_loss_raw': float(m.get(
            'load_balance_loss_raw', win_avgs['aux'])),
        'load_balance_loss_weighted': float(m.get(
            'load_balance_loss_weighted', ctx['lb_weight'] * win_avgs['aux'])),
        'diversity_loss_raw': float(m.get('diversity_loss_raw', win_avgs['div'])),
        'diversity_loss_weighted': float(m.get(
            'diversity_loss_weighted', ctx['div_weight'] * win_avgs['div'])),
        'aux_weighted': ctx['lb_weight'] * win_avgs['aux'],
        'tau_reg_weighted': ctx['tau_reg_weight'] * win_avgs['tau_reg'],
        'orth_weighted': ctx['orth_weight'] * win_avgs['orth'],
        'div_weighted': ctx['div_weight'] * win_avgs['div'],
        'pool_weight_decay_loss': float(m.get('pool_weight_decay_loss', 0.0)),
        'normal_weight_decay_loss': float(m.get('normal_weight_decay_loss', 0.0)),
        'reconstructed_total_loss': float(m.get(
            'reconstructed_total_loss', m.get('total_loss', 0.0))),
        'reconstructed_loss_error': float(m.get('reconstructed_loss_error', 0.0)),
        'dead_loss_raw': float(m.get('dead_loss_raw', m.get('dead_penalty', 0.0))),
        'dead_loss_weighted': float(m.get(
            'dead_loss_weighted', m.get('dead_penalty_weighted_total', 0.0))),
        'inactive_aux_loss_raw': float(m.get(
            'inactive_aux_loss_raw', m.get('inactive_aux_loss_raw_total', 0.0))),
        'inactive_aux_loss_weighted': float(m.get(
            'inactive_aux_loss_weighted', m.get('inactive_aux_loss_weighted_total', 0.0))),
        'weight_decay_pool': float(m.get(
            'weight_decay_pool', m.get('pool_weight_decay_loss', 0.0))),
        'weight_decay_normal': float(m.get(
            'weight_decay_normal', m.get('normal_weight_decay_loss', 0.0))),
        'total_loss_minus_ce': float(m.get(
            'total_loss_minus_ce', win_avgs['loss'] - win_avgs['ce'])),
        'reconstructed_total_loss': float(m.get(
            'reconstructed_total_loss', m.get('total_loss', win_avgs['loss']))),
        'reconstructed_loss_error': float(m.get('reconstructed_loss_error', 0.0)),
        'dead_loss_raw': float(m.get('dead_loss_raw', m.get('dead_penalty', 0.0))),
        'dead_loss_weighted': float(m.get(
            'dead_loss_weighted', m.get('dead_penalty_weighted_total', 0.0))),
        'inactive_aux_loss_raw': float(m.get(
            'inactive_aux_loss_raw', m.get('inactive_aux_loss_raw_total', 0.0))),
        'inactive_aux_loss_weighted': float(m.get(
            'inactive_aux_loss_weighted', m.get('inactive_aux_loss_weighted_total', 0.0))),
        'weight_decay_pool': float(m.get(
            'weight_decay_pool', m.get('pool_weight_decay_loss', 0.0))),
        'weight_decay_normal': float(m.get(
            'weight_decay_normal', m.get('normal_weight_decay_loss', 0.0))),
        # Dead-only penalty.
        'dead_penalty': float(m.get('dead_penalty', 0.0)),
        'attn_dead_penalty': float(m.get('attn_dead_penalty', 0.0)),
        'rst_dead_penalty': float(m.get('rst_dead_penalty', 0.0)),
        'dead_penalty_raw_total': float(m.get('dead_penalty_raw_total', 0.0)),
        'dead_penalty_raw_unweighted': float(m.get(
            'dead_penalty_raw_unweighted', m.get('dead_penalty_raw_total', 0.0))),
        'dead_penalty_raw_weighted_pools': float(m.get(
            'dead_penalty_raw_weighted_pools', m.get('dead_penalty', 0.0))),
        'attn_dead_penalty_raw': float(m.get('attn_dead_penalty_raw', 0.0)),
        'rst_dead_penalty_raw': float(m.get('rst_dead_penalty_raw', 0.0)),
        'dead_penalty_qk_weight': float(m.get('dead_penalty_qk_weight', 1.0)),
        'dead_penalty_v_weight': float(m.get('dead_penalty_v_weight', 1.0)),
        'dead_penalty_rst_weight': float(m.get('dead_penalty_rst_weight', 1.0)),
        'attn_qk_dead_penalty': float(m.get('attn_qk_dead_penalty', 0.0)),
        'attn_v_dead_penalty': float(m.get('attn_v_dead_penalty', 0.0)),
        'attn_qk_dead_count': float(m.get('attn_qk_dead_count', 0.0)),
        'attn_v_dead_count': float(m.get('attn_v_dead_count', 0.0)),
        'dead_penalty_weight': float(m.get(
            'dead_penalty_weight', ctx['dead_penalty_weight'])),
        'dead_penalty_weighted': ctx['dead_penalty_weight'] * float(m.get('dead_penalty', 0.0)),
        'dead_penalty_weighted_total': float(m.get(
            'dead_penalty_weighted_total',
            ctx['dead_penalty_weight'] * float(m.get('dead_penalty', 0.0)))),
        'dead_count_total': float(m.get('dead_count_total', 0.0)),
        'dead_penalty_per_dead': float(m.get('dead_penalty_per_dead', 0.0)),
        'attn_dead_penalty_per_dead': float(m.get(
            'attn_dead_penalty_per_dead', 0.0)),
        'rst_dead_penalty_per_dead': float(m.get(
            'rst_dead_penalty_per_dead', 0.0)),
        'attn_angular_exposure_mean': float(m.get(
            'attn_angular_exposure_mean', 0.0)),
        'attn_angular_exposure_min': float(m.get(
            'attn_angular_exposure_min', 0.0)),
        'attn_angular_exposure_max': float(m.get(
            'attn_angular_exposure_max', 0.0)),
        'attn_dead_exposure_frac': float(m.get(
            'attn_dead_exposure_frac', 0.0)),
        'attn_weak_exposure_frac': float(m.get(
            'attn_weak_exposure_frac', 0.0)),
        'attn_dead_exposure_target': float(m.get(
            'attn_dead_exposure_target', 0.0)),
        'rst_angular_exposure_mean': float(m.get(
            'rst_angular_exposure_mean', 0.0)),
        'rst_angular_exposure_min': float(m.get(
            'rst_angular_exposure_min', 0.0)),
        'rst_angular_exposure_max': float(m.get(
            'rst_angular_exposure_max', 0.0)),
        'rst_dead_exposure_frac': float(m.get(
            'rst_dead_exposure_frac', 0.0)),
        'rst_weak_exposure_frac': float(m.get(
            'rst_weak_exposure_frac', 0.0)),
        'rst_dead_exposure_target': float(m.get(
            'rst_dead_exposure_target', 0.0)),
        'attn_soft_exposure_mean': float(m.get('attn_soft_exposure_mean', 0.0)),
        'attn_soft_exposure_min': float(m.get('attn_soft_exposure_min', 0.0)),
        'attn_soft_exposure_max': float(m.get('attn_soft_exposure_max', 0.0)),
        'attn_soft_dead_frac_eps_1e_6': float(m.get('attn_soft_dead_frac_eps_1e_6', 0.0)),
        'attn_soft_dead_frac_eps_1e_5': float(m.get('attn_soft_dead_frac_eps_1e_5', 0.0)),
        'attn_soft_dead_frac_eps_1e_4': float(m.get('attn_soft_dead_frac_eps_1e_4', 0.0)),
        'attn_qk_soft_exposure_mean': float(m.get('attn_qk_soft_exposure_mean', 0.0)),
        'attn_qk_soft_exposure_min': float(m.get('attn_qk_soft_exposure_min', 0.0)),
        'attn_qk_soft_exposure_max': float(m.get('attn_qk_soft_exposure_max', 0.0)),
        'attn_qk_soft_dead_frac_eps_1e_6': float(m.get('attn_qk_soft_dead_frac_eps_1e_6', 0.0)),
        'attn_qk_soft_dead_frac_eps_1e_5': float(m.get('attn_qk_soft_dead_frac_eps_1e_5', 0.0)),
        'attn_qk_soft_dead_frac_eps_1e_4': float(m.get('attn_qk_soft_dead_frac_eps_1e_4', 0.0)),
        'attn_v_soft_exposure_mean': float(m.get('attn_v_soft_exposure_mean', 0.0)),
        'attn_v_soft_exposure_min': float(m.get('attn_v_soft_exposure_min', 0.0)),
        'attn_v_soft_exposure_max': float(m.get('attn_v_soft_exposure_max', 0.0)),
        'attn_v_soft_dead_frac_eps_1e_6': float(m.get('attn_v_soft_dead_frac_eps_1e_6', 0.0)),
        'attn_v_soft_dead_frac_eps_1e_5': float(m.get('attn_v_soft_dead_frac_eps_1e_5', 0.0)),
        'attn_v_soft_dead_frac_eps_1e_4': float(m.get('attn_v_soft_dead_frac_eps_1e_4', 0.0)),
        'rst_soft_exposure_mean': float(m.get('rst_soft_exposure_mean', 0.0)),
        'rst_soft_exposure_min': float(m.get('rst_soft_exposure_min', 0.0)),
        'rst_soft_exposure_max': float(m.get('rst_soft_exposure_max', 0.0)),
        'rst_soft_dead_frac_eps_1e_6': float(m.get('rst_soft_dead_frac_eps_1e_6', 0.0)),
        'rst_soft_dead_frac_eps_1e_5': float(m.get('rst_soft_dead_frac_eps_1e_5', 0.0)),
        'rst_soft_dead_frac_eps_1e_4': float(m.get('rst_soft_dead_frac_eps_1e_4', 0.0)),
        'inactive_aux_loss_raw': float(m.get('inactive_aux_loss_raw', 0.0)),
        'inactive_aux_loss_weighted': float(m.get('inactive_aux_loss_weighted', 0.0)),
        'inactive_aux_loss_raw_total': float(m.get(
            'inactive_aux_loss_raw_total', m.get('inactive_aux_loss_raw', 0.0))),
        'inactive_aux_loss_raw_q': float(m.get(
            'inactive_aux_loss_raw_q', m.get('inactive_aux_q_raw', 0.0))),
        'inactive_aux_loss_raw_k': float(m.get(
            'inactive_aux_loss_raw_k', m.get('inactive_aux_k_raw', 0.0))),
        'inactive_aux_loss_raw_qk': float(m.get(
            'inactive_aux_loss_raw_qk', m.get('inactive_aux_qk_raw', 0.0))),
        'inactive_aux_loss_raw_v': float(m.get(
            'inactive_aux_loss_raw_v', m.get('inactive_aux_v_raw', 0.0))),
        'inactive_aux_loss_raw_attn': float(m.get(
            'inactive_aux_loss_raw_attn', m.get('inactive_aux_attn_raw', 0.0))),
        'inactive_aux_loss_raw_rst': float(m.get(
            'inactive_aux_loss_raw_rst', m.get('inactive_aux_rst_raw', 0.0))),
        'inactive_aux_warmup_factor': float(m.get('inactive_aux_warmup_factor', 0.0)),
        'inactive_aux_weight_effective': float(m.get(
            'inactive_aux_weight_effective', 0.0)),
        'soft_gate_T': float(m.get('soft_gate_T', 0.0)),
        'soft_gate_T_qk': float(m.get(
            'soft_gate_T_qk', m.get('soft_gate_T', 0.0))),
        'soft_gate_T_v': float(m.get(
            'soft_gate_T_v', m.get('soft_gate_T', 0.0))),
        'soft_gate_T_rst': float(m.get(
            'soft_gate_T_rst', m.get('soft_gate_T', 0.0))),
        'boundary_power_p': float(m.get(
            'boundary_power_p',
            m.get('soft_gate_boundary_power', 2.0))),
        'admission_den_power': float(m.get(
            'admission_den_power', m.get('den_power', 1.0))),
        'den_power': float(m.get(
            'den_power', m.get('admission_den_power', 1.0))),
        'soft_gate_schedule': ctx.get('soft_gate_schedule', 'cosine'),
        'soft_gate_t_gompertz_center': float(ctx.get(
            'soft_gate_t_gompertz_center', 0.25)),
        'soft_gate_t_gompertz_steepness': float(ctx.get(
            'soft_gate_t_gompertz_steepness', 8.0)),
        'inactive_aux_effective_weight': float(m.get(
            'inactive_aux_effective_weight', m.get('inactive_aux_weight_effective', 0.0))),
        'inactive_aux_schedule_scale': float(m.get('inactive_aux_schedule_scale', 0.0)),
        'inactive_aux_asymmetry': float(m.get('inactive_aux_asymmetry', 0.0)),
        'inactive_aux_asymmetry_q': float(m.get(
            'inactive_aux_asymmetry_q', m.get('inactive_aux_asymmetry', 0.0))),
        'inactive_aux_asymmetry_k': float(m.get(
            'inactive_aux_asymmetry_k', m.get('inactive_aux_asymmetry', 0.0))),
        'inactive_aux_asymmetry_qk': float(m.get(
            'inactive_aux_asymmetry_qk', m.get('inactive_aux_asymmetry', 0.0))),
        'inactive_aux_asymmetry_v': float(m.get(
            'inactive_aux_asymmetry_v', m.get('inactive_aux_asymmetry', 0.0))),
        'inactive_aux_asymmetry_rst': float(m.get(
            'inactive_aux_asymmetry_rst', m.get('inactive_aux_asymmetry', 0.0))),
        'inactive_aux_layer_norm_enabled': float(m.get(
            'inactive_aux_layer_norm_enabled', 0.0)),
        'inactive_aux_layer_count': float(m.get('inactive_aux_layer_count', 0.0)),
        'inactive_aux_norm': float(m.get('inactive_aux_norm', 0.0)),
        'inactive_aux_no_active_easy_shutoff_q': float(m.get(
            'inactive_aux_no_active_easy_shutoff_q', 0.0)),
        'inactive_aux_no_active_easy_shutoff_k': float(m.get(
            'inactive_aux_no_active_easy_shutoff_k', 0.0)),
        'inactive_aux_no_active_easy_shutoff_v': float(m.get(
            'inactive_aux_no_active_easy_shutoff_v', 0.0)),
        'inactive_aux_no_active_easy_shutoff_rst': float(m.get(
            'inactive_aux_no_active_easy_shutoff_rst', 0.0)),
        'inactive_aux_loss_weighted_q': float(m.get(
            'inactive_aux_loss_weighted_q', 0.0)),
        'inactive_aux_loss_weighted_k': float(m.get(
            'inactive_aux_loss_weighted_k', 0.0)),
        'inactive_aux_loss_weighted_qk': float(m.get(
            'inactive_aux_loss_weighted_qk', 0.0)),
        'inactive_aux_loss_weighted_v': float(m.get(
            'inactive_aux_loss_weighted_v', 0.0)),
        'inactive_aux_loss_weighted_attn': float(m.get(
            'inactive_aux_loss_weighted_attn', 0.0)),
        'inactive_aux_loss_weighted_rst': float(m.get(
            'inactive_aux_loss_weighted_rst', 0.0)),
        'inactive_aux_loss_weighted_total': float(m.get(
            'inactive_aux_loss_weighted_total', m.get('inactive_aux_loss_weighted', 0.0))),
        'inactive_aux_raw_pre_bound': float(m.get('inactive_aux_raw_pre_bound', 0.0)),
        'inactive_aux_q_pre_bound': float(m.get('inactive_aux_q_pre_bound', 0.0)),
        'inactive_aux_k_pre_bound': float(m.get('inactive_aux_k_pre_bound', 0.0)),
        'inactive_aux_qk_pre_bound': float(m.get('inactive_aux_qk_pre_bound', 0.0)),
        'inactive_aux_v_pre_bound': float(m.get('inactive_aux_v_pre_bound', 0.0)),
        'inactive_aux_raw_post_bound': float(m.get('inactive_aux_raw_post_bound', 0.0)),
        # CB1A boundary audition loss.
        'cb1a_enabled': bool(ctx.get('cb1a_enabled', False)),
        'cb1a_ce_mean': float(m.get('cb1a_ce_mean', 0.0)),
        'cb1a_ce_std': float(m.get('cb1a_ce_std', 0.0)),
        'cb1a_raw': float(m.get('cb1a_raw', 0.0)),
        'cb1a_w': float(m.get('cb1a_w', m.get('cb1a_weighted', 0.0))),
        'cb1a_weighted': float(m.get('cb1a_weighted', m.get('cb1a_w', 0.0))),
        'cb1a_challenge_raw': float(m.get('cb1a_challenge_raw', 0.0)),
        'cb1a_prune_raw': float(m.get('cb1a_prune_raw', 0.0)),
        'cb1a_challenge': float(m.get('cb1a_challenge', 0.0)),
        'cb1a_prune': float(m.get('cb1a_prune', 0.0)),
        'cb1a_valid': float(m.get('cb1a_valid', 0.0)),
        'cb1a_has_above': float(m.get('cb1a_has_above', 0.0)),
        'cb1a_has_below': float(m.get('cb1a_has_below', 0.0)),
        'cb1a_qk_challenge_raw': float(m.get(
            'cb1a_qk_challenge_raw', m.get('cb1a_qk_challenge', 0.0))),
        'cb1a_qk_prune_raw': float(m.get(
            'cb1a_qk_prune_raw', m.get('cb1a_qk_prune', 0.0))),
        'cb1a_v_challenge_raw': float(m.get(
            'cb1a_v_challenge_raw', m.get('cb1a_v_challenge', 0.0))),
        'cb1a_v_prune_raw': float(m.get(
            'cb1a_v_prune_raw', m.get('cb1a_v_prune', 0.0))),
        'cb1a_rst_challenge_raw': float(m.get(
            'cb1a_rst_challenge_raw', m.get('cb1a_rst_challenge', 0.0))),
        'cb1a_rst_prune_raw': float(m.get(
            'cb1a_rst_prune_raw', m.get('cb1a_rst_prune', 0.0))),
        'cb1a_qk_challenge': float(m.get(
            'cb1a_qk_challenge', m.get('cb1a_qk_challenge_raw', 0.0))),
        'cb1a_qk_prune': float(m.get(
            'cb1a_qk_prune', m.get('cb1a_qk_prune_raw', 0.0))),
        'cb1a_v_challenge': float(m.get(
            'cb1a_v_challenge', m.get('cb1a_v_challenge_raw', 0.0))),
        'cb1a_v_prune': float(m.get(
            'cb1a_v_prune', m.get('cb1a_v_prune_raw', 0.0))),
        'cb1a_rst_challenge': float(m.get(
            'cb1a_rst_challenge', m.get('cb1a_rst_challenge_raw', 0.0))),
        'cb1a_rst_prune': float(m.get(
            'cb1a_rst_prune', m.get('cb1a_rst_prune_raw', 0.0))),
        'cb1a_qk_raw': float(m.get('cb1a_qk_raw', 0.0)),
        'cb1a_v_raw': float(m.get('cb1a_v_raw', 0.0)),
        'cb1a_rst_raw': float(m.get('cb1a_rst_raw', 0.0)),
        'cb1a_weight': float(m.get('cb1a_weight', 0.0)),
        'cb1a_challenge_weight': float(m.get('cb1a_challenge_weight', 1.0)),
        'cb1a_prune_weight': float(m.get('cb1a_prune_weight', 1.0)),
        'cb1a_qk_weight': float(m.get('cb1a_qk_weight', 1.0)),
        'cb1a_v_weight': float(m.get('cb1a_v_weight', 1.0)),
        'cb1a_rst_weight': float(m.get('cb1a_rst_weight', 1.0)),
        'cb1a_qk_challenge_weight': float(m.get(
            'cb1a_qk_challenge_weight',
            m.get('cb1a_challenge_weight', 1.0))),
        'cb1a_qk_prune_weight': float(m.get(
            'cb1a_qk_prune_weight',
            m.get('cb1a_prune_weight', 1.0))),
        'cb1a_v_challenge_weight': float(m.get(
            'cb1a_v_challenge_weight',
            m.get('cb1a_challenge_weight', 1.0))),
        'cb1a_v_prune_weight': float(m.get(
            'cb1a_v_prune_weight',
            m.get('cb1a_prune_weight', 1.0))),
        'cb1a_rst_challenge_weight': float(m.get(
            'cb1a_rst_challenge_weight',
            m.get('cb1a_challenge_weight', 1.0))),
        'cb1a_rst_prune_weight': float(m.get(
            'cb1a_rst_prune_weight',
            m.get('cb1a_prune_weight', 1.0))),
        # Accuracy / training status.
        'accuracy': win_avgs['acc'],
        'grad_norm': float(m['grad_norm']),
        'grad_global_preclip': float(m.get('grad_global_preclip', m['grad_norm'])),
        'grad_global_postclip': float(m.get('grad_global_postclip', 0.0)),
        'grad_token_emb': float(m.get('grad_token_emb', 0.0)),
        'grad_pos_emb': float(m.get('grad_pos_emb', 0.0)),
        'grad_router_proj_attn': float(m.get('grad_router_proj_attn', 0.0)),
        'grad_router_proj_rst': float(m.get('grad_router_proj_rst', 0.0)),
        'grad_router_raw_tau_qk': float(m.get('grad_router_raw_tau_qk', 0.0)),
        'grad_router_raw_tau_v': float(m.get('grad_router_raw_tau_v', 0.0)),
        'grad_router_raw_tau_rst': float(m.get('grad_router_raw_tau_rst', 0.0)),
        'grad_router_tau_attn': float(m.get('grad_router_tau_attn', 0.0)),
        'grad_router_tau_rst': float(m.get('grad_router_tau_rst', 0.0)),
        'grad_router_scan_attn': float(m.get('grad_router_scan_attn', 0.0)),
        'grad_router_scan_rst': float(m.get('grad_router_scan_rst', 0.0)),
        'grad_pool_attn_qk_emb': float(m.get('grad_pool_attn_qk_emb', 0.0)),
        'grad_pool_attn_qk_read': float(m.get('grad_pool_attn_qk_read', 0.0)),
        'grad_pool_attn_qk_write': float(m.get('grad_pool_attn_qk_write', 0.0)),
        'grad_pool_attn_v_emb': float(m.get('grad_pool_attn_v_emb', 0.0)),
        'grad_pool_attn_v_read': float(m.get('grad_pool_attn_v_read', 0.0)),
        'grad_pool_attn_v_write': float(m.get('grad_pool_attn_v_write', 0.0)),
        'grad_pool_rst_emb': float(m.get('grad_pool_rst_emb', 0.0)),
        'grad_pool_rst_read': float(m.get('grad_pool_rst_read', 0.0)),
        'grad_pool_rst_write': float(m.get('grad_pool_rst_write', 0.0)),
        'grad_pool_scales': float(m.get('grad_pool_scales', 0.0)),
        'grad_expand_O': float(m.get('grad_expand_O', 0.0)),
        'grad_layernorms': float(m.get('grad_layernorms', 0.0)),
        'grad_lm_head_or_token_tied': float(m.get(
            'grad_lm_head_or_token_tied', 0.0)),
        'grad_router_proj_attn_per_layer': _arr('grad_router_proj_attn_per_layer'),
        'grad_router_proj_rst_per_layer': _arr('grad_router_proj_rst_per_layer'),
        'grad_router_raw_tau_qk_per_layer': _arr('grad_router_raw_tau_qk_per_layer'),
        'grad_router_raw_tau_v_per_layer': _arr('grad_router_raw_tau_v_per_layer'),
        'grad_router_raw_tau_rst_per_layer': _arr('grad_router_raw_tau_rst_per_layer'),
        'grad_router_tau_attn_per_layer': _arr('grad_router_tau_attn_per_layer'),
        'grad_router_tau_rst_per_layer': _arr('grad_router_tau_rst_per_layer'),
        'grad_pool_attn_qk_rw': _arr('grad_pool_attn_qk_rw'),
        'grad_pool_attn_v_rw': _arr('grad_pool_attn_v_rw'),
        'grad_pool_rst_rw': _arr('grad_pool_rst_rw'),
        'grad_expand_O_per_layer': _arr('grad_expand_O_per_layer'),
        'grad_layernorms_per_layer': _arr('grad_layernorms_per_layer'),
        'grad_router_proj': float(m.get('grad_router_proj', 0.0)),
        'grad_router_tau': float(m.get('grad_router_tau', 0.0)),
        'grad_router_scan': float(m.get('grad_router_scan', 0.0)),
        'grad_pool_emb': float(m.get('grad_pool_emb', 0.0)),
        'grad_pool_op_key': float(m.get('grad_pool_op_key', 0.0)),
        'grad_pool_read': float(m.get('grad_pool_read', 0.0)),
        'grad_pool_write': float(m.get('grad_pool_write', 0.0)),
        'update_cap_proj_attn_ratio_pre': float(m.get('update_cap_proj_attn_ratio_pre', 0.0)),
        'update_cap_proj_attn_ratio_post': float(m.get('update_cap_proj_attn_ratio_post', 0.0)),
        'update_cap_proj_attn_scale': float(m.get('update_cap_proj_attn_scale', 1.0)),
        'update_cap_proj_attn_hit': float(m.get('update_cap_proj_attn_hit', 0.0)),
        'update_cap_proj_rst_ratio_pre': float(m.get('update_cap_proj_rst_ratio_pre', 0.0)),
        'update_cap_proj_rst_ratio_post': float(m.get('update_cap_proj_rst_ratio_post', 0.0)),
        'update_cap_proj_rst_scale': float(m.get('update_cap_proj_rst_scale', 1.0)),
        'update_cap_proj_rst_hit': float(m.get('update_cap_proj_rst_hit', 0.0)),
        'update_cap_op_key_qk_ratio_pre': float(m.get('update_cap_op_key_qk_ratio_pre', 0.0)),
        'update_cap_op_key_qk_ratio_post': float(m.get('update_cap_op_key_qk_ratio_post', 0.0)),
        'update_cap_op_key_qk_scale': float(m.get('update_cap_op_key_qk_scale', 1.0)),
        'update_cap_op_key_qk_hit': float(m.get('update_cap_op_key_qk_hit', 0.0)),
        'update_cap_op_key_v_ratio_pre': float(m.get('update_cap_op_key_v_ratio_pre', 0.0)),
        'update_cap_op_key_v_ratio_post': float(m.get('update_cap_op_key_v_ratio_post', 0.0)),
        'update_cap_op_key_v_scale': float(m.get('update_cap_op_key_v_scale', 1.0)),
        'update_cap_op_key_v_hit': float(m.get('update_cap_op_key_v_hit', 0.0)),
        'update_cap_op_key_rst_ratio_pre': float(m.get('update_cap_op_key_rst_ratio_pre', 0.0)),
        'update_cap_op_key_rst_ratio_post': float(m.get('update_cap_op_key_rst_ratio_post', 0.0)),
        'update_cap_op_key_rst_scale': float(m.get('update_cap_op_key_rst_scale', 1.0)),
        'update_cap_op_key_rst_hit': float(m.get('update_cap_op_key_rst_hit', 0.0)),
        'update_cap_tau_attn_abs_pre': float(m.get('update_cap_tau_attn_abs_pre', 0.0)),
        'update_cap_tau_attn_abs_post': float(m.get('update_cap_tau_attn_abs_post', 0.0)),
        'update_cap_tau_attn_scale': float(m.get('update_cap_tau_attn_scale', 1.0)),
        'update_cap_tau_attn_hit': float(m.get('update_cap_tau_attn_hit', 0.0)),
        'update_cap_tau_rst_abs_pre': float(m.get('update_cap_tau_rst_abs_pre', 0.0)),
        'update_cap_tau_rst_abs_post': float(m.get('update_cap_tau_rst_abs_post', 0.0)),
        'update_cap_tau_rst_scale': float(m.get('update_cap_tau_rst_scale', 1.0)),
        'update_cap_tau_rst_hit': float(m.get('update_cap_tau_rst_hit', 0.0)),
        'update_cap_raw_tau_enabled': float(m.get('update_cap_raw_tau_enabled', 0.0)),
        'update_cap_raw_tau_qk_abs_pre': float(m.get('update_cap_raw_tau_qk_abs_pre', 0.0)),
        'update_cap_raw_tau_qk_abs_post': float(m.get('update_cap_raw_tau_qk_abs_post', 0.0)),
        'update_cap_raw_tau_qk_scale': float(m.get('update_cap_raw_tau_qk_scale', 1.0)),
        'update_cap_raw_tau_qk_hit': float(m.get('update_cap_raw_tau_qk_hit', 0.0)),
        'update_cap_raw_tau_v_abs_pre': float(m.get('update_cap_raw_tau_v_abs_pre', 0.0)),
        'update_cap_raw_tau_v_abs_post': float(m.get('update_cap_raw_tau_v_abs_post', 0.0)),
        'update_cap_raw_tau_v_scale': float(m.get('update_cap_raw_tau_v_scale', 1.0)),
        'update_cap_raw_tau_v_hit': float(m.get('update_cap_raw_tau_v_hit', 0.0)),
        'update_cap_raw_tau_rst_abs_pre': float(m.get('update_cap_raw_tau_rst_abs_pre', 0.0)),
        'update_cap_raw_tau_rst_abs_post': float(m.get('update_cap_raw_tau_rst_abs_post', 0.0)),
        'update_cap_raw_tau_rst_scale': float(m.get('update_cap_raw_tau_rst_scale', 1.0)),
        'update_cap_raw_tau_rst_hit': float(m.get('update_cap_raw_tau_rst_hit', 0.0)),
        'update_cap_scan_attn_abs_pre': float(m.get('update_cap_scan_attn_abs_pre', 0.0)),
        'update_cap_scan_attn_abs_post': float(m.get('update_cap_scan_attn_abs_post', 0.0)),
        'update_cap_scan_attn_scale': float(m.get('update_cap_scan_attn_scale', 1.0)),
        'update_cap_scan_attn_hit': float(m.get('update_cap_scan_attn_hit', 0.0)),
        'update_cap_scan_rst_abs_pre': float(m.get('update_cap_scan_rst_abs_pre', 0.0)),
        'update_cap_scan_rst_abs_post': float(m.get('update_cap_scan_rst_abs_post', 0.0)),
        'update_cap_scan_rst_scale': float(m.get('update_cap_scan_rst_scale', 1.0)),
        'update_cap_scan_rst_hit': float(m.get('update_cap_scan_rst_hit', 0.0)),
        'lr': ctx['current_lr'],
        'steps_per_sec': ctx['steps_per_sec'],
        'elapsed': ctx['total_elapsed'],
        # Drift (reduced inside train_step).
        'drift_attn_qk_emb': float(m.get('drift_attn_qk_emb', 0.0)),
        'drift_attn_v_emb': float(m.get('drift_attn_v_emb', 0.0)),
        'drift_rst_emb': float(m.get('drift_rst_emb', 0.0)),
        'drift_attn_qk_op_key': float(m.get(
            'drift_attn_qk_op_key', m.get('drift_attn_qk_emb', 0.0))),
        'drift_attn_v_op_key': float(m.get(
            'drift_attn_v_op_key', m.get('drift_attn_v_emb', 0.0))),
        'drift_rst_op_key': float(m.get(
            'drift_rst_op_key', m.get('drift_rst_emb', 0.0))),
        # Core activity.
        'drive_mean': float(m.get('drive_mean', 0.0)),
        'drive_max': float(m.get('drive_max', 0.0)),
        'attn_qk_drive_mean': float(m.get('attn_qk_drive_mean', 0.0)),
        'attn_qk_drive_max': float(m.get('attn_qk_drive_max', 0.0)),
        'attn_v_drive_mean': float(m.get('attn_v_drive_mean', 0.0)),
        'attn_v_drive_max': float(m.get('attn_v_drive_max', 0.0)),
        'rst_drive_mean': float(m.get('rst_drive_mean', 0.0)),
        'rst_drive_max': float(m.get('rst_drive_max', 0.0)),
        'weight_mean': float(m.get('weight_mean', 0.0)),
        'weight_max': float(m.get('weight_max', 0.0)),
        'attn_qk_weight_mean': float(m.get('attn_qk_weight_mean', 0.0)),
        'attn_qk_weight_max': float(m.get('attn_qk_weight_max', 0.0)),
        'attn_v_weight_mean': float(m.get('attn_v_weight_mean', 0.0)),
        'attn_v_weight_max': float(m.get('attn_v_weight_max', 0.0)),
        'rst_weight_mean': float(m.get('rst_weight_mean', 0.0)),
        'rst_weight_max': float(m.get('rst_weight_max', 0.0)),
        'admission_mean': float(m.get('admission_mean', 0.0)),
        'admission_max': float(m.get('admission_max', 0.0)),
        'attn_qk_admission_mean': float(m.get('attn_qk_admission_mean', 0.0)),
        'attn_qk_admission_max': float(m.get('attn_qk_admission_max', 0.0)),
        'attn_v_admission_mean': float(m.get('attn_v_admission_mean', 0.0)),
        'attn_v_admission_max': float(m.get('attn_v_admission_max', 0.0)),
        'rst_admission_mean': float(m.get('rst_admission_mean', 0.0)),
        'rst_admission_max': float(m.get('rst_admission_max', 0.0)),
        'load_mean': float(m.get('load_mean', 0.0)),
        'normalization_load_mean': float(m.get(
            'normalization_load_mean', m.get('load_mean', 0.0))),
        'den_mean': float(m.get('den_mean', 0.0)),
        'attn_qk_active': float(m.get('attn_qk_active', 0.0)),
        'attn_q_active': float(m.get(
            'attn_q_active', m.get('attn_qk_active', 0.0))),
        'attn_k_active': float(m.get(
            'attn_k_active', m.get('attn_qk_active', 0.0))),
        'attn_v_active': float(m.get('attn_v_active', 0.0)),
        'rst_active': float(m.get('rst_active', 0.0)),
        'attn_strong': float(m.get('attn_strong', 0.0)),
        'attn_qk_strong': float(m.get('attn_qk_strong', m.get('attn_strong', 0.0))),
        'attn_q_strong': float(m.get(
            'attn_q_strong', m.get('attn_qk_strong', m.get('attn_strong', 0.0)))),
        'attn_k_strong': float(m.get(
            'attn_k_strong', m.get('attn_qk_strong', m.get('attn_strong', 0.0)))),
        'attn_v_strong': float(m.get('attn_v_strong', m.get('attn_strong', 0.0))),
        'rst_strong': float(m.get('rst_strong', 0.0)),
        'attn_raw_gate_max': float(m.get('attn_raw_gate_max', 0.0)),
        'rst_raw_gate_max': float(m.get('rst_raw_gate_max', 0.0)),
        'attn_int_max': float(m.get('attn_int_max', 0.0)),
        'rst_int_max': float(m.get('rst_int_max', 0.0)),
        'attn_int_cap_frac': float(m.get('attn_int_cap_frac', 0.0)),
        'rst_int_cap_frac': float(m.get('rst_int_cap_frac', 0.0)),
        'attn_den_cost_mean': float(m.get('attn_den_cost_mean', 0.0)),
        'rst_den_cost_mean': float(m.get('rst_den_cost_mean', 0.0)),
        'attn_contrib_den_sum': float(m.get('attn_contrib_den_sum', 0.0)),
        'rst_contrib_den_sum': float(m.get('rst_contrib_den_sum', 0.0)),
        'attn_contrib_den_mean': float(m.get(
            'attn_contrib_den_mean', m.get('attn_contrib_den_sum', 0.0))),
        'rst_contrib_den_mean': float(m.get(
            'rst_contrib_den_mean', m.get('rst_contrib_den_sum', 0.0))),
        'attn_contrib_den': float(m.get('attn_contrib_den', 0.0)),
        'rst_contrib_den': float(m.get('rst_contrib_den', 0.0)),
        'attn_contrib_den_max': float(m.get('attn_contrib_den_max', 0.0)),
        'rst_contrib_den_max': float(m.get('rst_contrib_den_max', 0.0)),
        'attn_contrib_den_min': float(m.get('attn_contrib_den_min', 0.0)),
        'rst_contrib_den_min': float(m.get('rst_contrib_den_min', 0.0)),
        'attn_contrib_den_floor_frac': float(m.get('attn_contrib_den_floor_frac', 0.0)),
        'rst_contrib_den_floor_frac': float(m.get('rst_contrib_den_floor_frac', 0.0)),
        'attn_compose_norm': float(m.get('attn_compose_norm', 0.0)),
        'rst_compose_norm': float(m.get('rst_compose_norm', 0.0)),
        'attn_compose_norm_mean': float(m.get(
            'attn_compose_norm_mean', m.get('attn_compose_norm', 0.0))),
        'rst_compose_norm_mean': float(m.get(
            'rst_compose_norm_mean', m.get('rst_compose_norm', 0.0))),
        'attn_compose_norm_max': float(m.get('attn_compose_norm_max', 0.0)),
        'rst_compose_norm_max': float(m.get('rst_compose_norm_max', 0.0)),
        'attn_coherence': float(m.get('attn_coherence', 0.0)),
        'rst_coherence': float(m.get('rst_coherence', 0.0)),
        'attn_coherence_max': float(m.get('attn_coherence_max', 0.0)),
        'rst_coherence_max': float(m.get('rst_coherence_max', 0.0)),
        'attn_den_ratio': float(m.get('attn_den_ratio', 0.0)),
        'rst_den_ratio': float(m.get('rst_den_ratio', 0.0)),
        'attn_den_ratio_mean': float(m.get(
            'attn_den_ratio_mean', m.get('attn_den_ratio', 0.0))),
        'rst_den_ratio_mean': float(m.get(
            'rst_den_ratio_mean', m.get('rst_den_ratio', 0.0))),
        'attn_den_ratio_max': float(m.get('attn_den_ratio_max', 0.0)),
        'rst_den_ratio_max': float(m.get('rst_den_ratio_max', 0.0)),
        'attn_raw_out_norm_mean': float(m.get('attn_raw_out_norm_mean', 0.0)),
        'rst_raw_out_norm_mean': float(m.get('rst_raw_out_norm_mean', 0.0)),
        'attn_raw_out_norm_max': float(m.get('attn_raw_out_norm_max', 0.0)),
        'rst_raw_out_norm_max': float(m.get('rst_raw_out_norm_max', 0.0)),
        'attn_normalized_out_norm_mean': float(m.get('attn_normalized_out_norm_mean', 0.0)),
        'rst_normalized_out_norm_mean': float(m.get('rst_normalized_out_norm_mean', 0.0)),
        'attn_normalized_out_norm_max': float(m.get('attn_normalized_out_norm_max', 0.0)),
        'rst_normalized_out_norm_max': float(m.get('rst_normalized_out_norm_max', 0.0)),
        'attn_scaled_out_norm_mean': float(m.get('attn_scaled_out_norm_mean', 0.0)),
        'rst_scaled_out_norm_mean': float(m.get('rst_scaled_out_norm_mean', 0.0)),
        'attn_scaled_out_norm_max': float(m.get('attn_scaled_out_norm_max', 0.0)),
        'rst_scaled_out_norm_max': float(m.get('rst_scaled_out_norm_max', 0.0)),
        'attn_act_cost_mean': float(m.get('attn_act_cost_mean', 0.0)),
        'rst_act_cost_mean': float(m.get('rst_act_cost_mean', 0.0)),
        'attn_current_cost_mean': float(m.get('attn_current_cost_mean', 0.0)),
        'rst_current_cost_mean': float(m.get('rst_current_cost_mean', 0.0)),
        'admission_den_sum': float(m.get('admission_den_sum', 0.0)),
        'attn_admission_den_sum': float(m.get('attn_admission_den_sum', 0.0)),
        'attn_qk_admission_den_sum': float(m.get('attn_qk_admission_den_sum', 0.0)),
        'attn_v_admission_den_sum': float(m.get('attn_v_admission_den_sum', 0.0)),
        'rst_admission_den_sum': float(m.get('rst_admission_den_sum', 0.0)),
        'execution_mass_sum': float(m.get('execution_mass_sum', 0.0)),
        'attn_execution_mass_sum': float(m.get('attn_execution_mass_sum', 0.0)),
        'attn_qk_execution_mass_sum': float(m.get('attn_qk_execution_mass_sum', 0.0)),
        'attn_v_execution_mass_sum': float(m.get('attn_v_execution_mass_sum', 0.0)),
        'rst_execution_mass_sum': float(m.get('rst_execution_mass_sum', 0.0)),
        'attn_qk_load': float(m.get('attn_qk_admission_den_sum', 0.0)),
        'attn_v_load': float(m.get('attn_v_admission_den_sum', 0.0)),
        'rst_load': float(m.get('rst_admission_den_sum', 0.0)),
        'attn_qk_mass': float(m.get('attn_qk_execution_mass_sum', 0.0)),
        'attn_v_mass': float(m.get('attn_v_execution_mass_sum', 0.0)),
        'rst_mass': float(m.get('rst_execution_mass_sum', 0.0)),
        'drive_mean': float(m.get('drive_mean', 0.0)),
        'attn_drive_mean': float(m.get('attn_drive_mean', 0.0)),
        'attn_qk_drive_mean': float(m.get('attn_qk_drive_mean', 0.0)),
        'attn_v_drive_mean': float(m.get('attn_v_drive_mean', 0.0)),
        'rst_drive_mean': float(m.get('rst_drive_mean', 0.0)),
        'drive_max': float(m.get('drive_max', 0.0)),
        'attn_drive_max': float(m.get('attn_drive_max', 0.0)),
        'attn_qk_drive_max': float(m.get('attn_qk_drive_max', 0.0)),
        'attn_v_drive_max': float(m.get('attn_v_drive_max', 0.0)),
        'rst_drive_max': float(m.get('rst_drive_max', 0.0)),
        'execution_eff_n': float(m.get('execution_eff_n', 0.0)),
        'attn_execution_eff_n': float(m.get('attn_execution_eff_n', 0.0)),
        'attn_qk_execution_eff_n': float(m.get('attn_qk_execution_eff_n', 0.0)),
        'attn_v_execution_eff_n': float(m.get('attn_v_execution_eff_n', 0.0)),
        'rst_execution_eff_n': float(m.get('rst_execution_eff_n', 0.0)),
        'execution_top1_frac': float(m.get('execution_top1_frac', 0.0)),
        'execution_top1_frac_max': float(m.get('execution_top1_frac_max', 0.0)),
        'attn_execution_top1_frac': float(m.get('attn_execution_top1_frac', 0.0)),
        'attn_execution_top1_frac_max': float(m.get('attn_execution_top1_frac_max', 0.0)),
        'attn_qk_execution_top1_frac': float(m.get('attn_qk_execution_top1_frac', 0.0)),
        'attn_qk_execution_top1_frac_max': float(m.get('attn_qk_execution_top1_frac_max', 0.0)),
        'attn_v_execution_top1_frac': float(m.get('attn_v_execution_top1_frac', 0.0)),
        'attn_v_execution_top1_frac_max': float(m.get('attn_v_execution_top1_frac_max', 0.0)),
        'rst_execution_top1_frac': float(m.get('rst_execution_top1_frac', 0.0)),
        'rst_execution_top1_frac_max': float(m.get('rst_execution_top1_frac_max', 0.0)),
        'attn_gate_sum': float(m.get('attn_gate_sum', 0.0)),
        'rst_gate_sum': float(m.get('rst_gate_sum', 0.0)),
        'attn_active_n_mean': float(m.get('attn_active_n_mean', 0.0)),
        'attn_q_active_n_mean': float(m.get(
            'attn_q_active_n_mean',
            m.get('attn_qk_active_n_mean', m.get('attn_active_n_mean', 0.0)))),
        'attn_k_active_n_mean': float(m.get(
            'attn_k_active_n_mean',
            m.get('attn_qk_active_n_mean', m.get('attn_active_n_mean', 0.0)))),
        'rst_active_n_mean': float(m.get('rst_active_n_mean', 0.0)),
        'attn_gate_eff_n': float(m.get('attn_gate_eff_n', 0.0)),
        'attn_gate_eff_ratio': float(m.get('attn_gate_eff_ratio', 0.0)),
        'attn_top1_gate_frac': float(m.get('attn_top1_gate_frac', 0.0)),
        'attn_top1_gate_frac_max': float(m.get('attn_top1_gate_frac_max', 0.0)),
        'rst_gate_eff_n': float(m.get('rst_gate_eff_n', 0.0)),
        'rst_gate_eff_ratio': float(m.get('rst_gate_eff_ratio', 0.0)),
        'rst_top1_gate_frac': float(m.get('rst_top1_gate_frac', 0.0)),
        'rst_top1_gate_frac_max': float(m.get('rst_top1_gate_frac_max', 0.0)),
        'attn_dead_count': float(m.get('attn_dead_count', 0.0)),
        'rst_dead_count': float(m.get('rst_dead_count', 0.0)),
        'attn_angular_exposure_mean': float(m.get(
            'attn_angular_exposure_mean', 0.0)),
        'attn_angular_exposure_min': float(m.get(
            'attn_angular_exposure_min', 0.0)),
        'attn_angular_exposure_max': float(m.get(
            'attn_angular_exposure_max', 0.0)),
        'attn_dead_exposure_frac': float(m.get(
            'attn_dead_exposure_frac', 0.0)),
        'attn_weak_exposure_frac': float(m.get(
            'attn_weak_exposure_frac', 0.0)),
        'attn_dead_exposure_target': float(m.get(
            'attn_dead_exposure_target', 0.0)),
        'rst_angular_exposure_mean': float(m.get(
            'rst_angular_exposure_mean', 0.0)),
        'rst_angular_exposure_min': float(m.get(
            'rst_angular_exposure_min', 0.0)),
        'rst_angular_exposure_max': float(m.get(
            'rst_angular_exposure_max', 0.0)),
        'rst_dead_exposure_frac': float(m.get(
            'rst_dead_exposure_frac', 0.0)),
        'rst_weak_exposure_frac': float(m.get(
            'rst_weak_exposure_frac', 0.0)),
        'rst_dead_exposure_target': float(m.get(
            'rst_dead_exposure_target', 0.0)),
        'attn_soft_exposure_mean': float(m.get('attn_soft_exposure_mean', 0.0)),
        'attn_soft_exposure_min': float(m.get('attn_soft_exposure_min', 0.0)),
        'attn_soft_exposure_max': float(m.get('attn_soft_exposure_max', 0.0)),
        'attn_soft_dead_frac_eps_1e_6': float(m.get('attn_soft_dead_frac_eps_1e_6', 0.0)),
        'attn_soft_dead_frac_eps_1e_5': float(m.get('attn_soft_dead_frac_eps_1e_5', 0.0)),
        'attn_soft_dead_frac_eps_1e_4': float(m.get('attn_soft_dead_frac_eps_1e_4', 0.0)),
        'attn_qk_soft_exposure_mean': float(m.get('attn_qk_soft_exposure_mean', 0.0)),
        'attn_qk_soft_exposure_min': float(m.get('attn_qk_soft_exposure_min', 0.0)),
        'attn_qk_soft_exposure_max': float(m.get('attn_qk_soft_exposure_max', 0.0)),
        'attn_qk_soft_dead_frac_eps_1e_6': float(m.get('attn_qk_soft_dead_frac_eps_1e_6', 0.0)),
        'attn_qk_soft_dead_frac_eps_1e_5': float(m.get('attn_qk_soft_dead_frac_eps_1e_5', 0.0)),
        'attn_qk_soft_dead_frac_eps_1e_4': float(m.get('attn_qk_soft_dead_frac_eps_1e_4', 0.0)),
        'attn_v_soft_exposure_mean': float(m.get('attn_v_soft_exposure_mean', 0.0)),
        'attn_v_soft_exposure_min': float(m.get('attn_v_soft_exposure_min', 0.0)),
        'attn_v_soft_exposure_max': float(m.get('attn_v_soft_exposure_max', 0.0)),
        'attn_v_soft_dead_frac_eps_1e_6': float(m.get('attn_v_soft_dead_frac_eps_1e_6', 0.0)),
        'attn_v_soft_dead_frac_eps_1e_5': float(m.get('attn_v_soft_dead_frac_eps_1e_5', 0.0)),
        'attn_v_soft_dead_frac_eps_1e_4': float(m.get('attn_v_soft_dead_frac_eps_1e_4', 0.0)),
        'rst_soft_exposure_mean': float(m.get('rst_soft_exposure_mean', 0.0)),
        'rst_soft_exposure_min': float(m.get('rst_soft_exposure_min', 0.0)),
        'rst_soft_exposure_max': float(m.get('rst_soft_exposure_max', 0.0)),
        'rst_soft_dead_frac_eps_1e_6': float(m.get('rst_soft_dead_frac_eps_1e_6', 0.0)),
        'rst_soft_dead_frac_eps_1e_5': float(m.get('rst_soft_dead_frac_eps_1e_5', 0.0)),
        'rst_soft_dead_frac_eps_1e_4': float(m.get('rst_soft_dead_frac_eps_1e_4', 0.0)),
        'attn_tau_mean': float(m.get('attn_tau_mean', 0.0)),
        'rst_tau_mean': float(m.get('rst_tau_mean', 0.0)),
        'attn_score_mean': float(m.get('attn_score_mean', 0.0)),
        'rst_score_mean': float(m.get('rst_score_mean', 0.0)),
        'attn_out_norm': float(m.get('attn_out_norm', 0.0)),
        'rst_out_norm': float(m.get('rst_out_norm', 0.0)),
        # tau structure (bias + offset distribution).
        'tau_rst_bias': float(m.get('tau_rst_bias', 0.0)),
        'tau_attn_bias_0': float(m.get('tau_attn_bias_0', 0.0)),
        'tau_attn_bias_1': float(m.get('tau_attn_bias_1', 0.0)),
        'tau_attn_bias_2': float(m.get('tau_attn_bias_2', 0.0)),
        'tau_q_bias': float(m.get('tau_q_bias', 0.0)),
        'tau_k_bias': float(m.get('tau_k_bias', 0.0)),
        'tau_v_bias': float(m.get('tau_v_bias', 0.0)),
        'raw_scan_offset_rst_bias': float(m.get('raw_scan_offset_rst_bias', 0.0)),
        'raw_scan_offset_attn_bias_0': float(m.get('raw_scan_offset_attn_bias_0', 0.0)),
        'raw_scan_offset_attn_bias_1': float(m.get('raw_scan_offset_attn_bias_1', 0.0)),
        'raw_scan_offset_attn_bias_2': float(m.get('raw_scan_offset_attn_bias_2', 0.0)),
        'attn_tau_abs_mean': float(m.get('attn_tau_abs_mean', 0.0)),
        'rst_tau_abs_mean': float(m.get('rst_tau_abs_mean', 0.0)),
        'attn_rho_mean': float(m.get('attn_rho_mean', 0.0)),
        'attn_rho_std': float(m.get('attn_rho_std', 0.0)),
        'attn_rho_max': float(m.get('attn_rho_max', 0.0)),
        'attn_raw_tau_mean': float(m.get('attn_raw_tau_mean', m.get('attn_tau_raw_mean', 0.0))),
        'attn_raw_tau_min': float(m.get(
            'attn_raw_tau_min',
            m.get('attn_raw_tau_mean', m.get('attn_tau_raw_mean', 0.0)))),
        'attn_raw_tau_max': float(m.get(
            'attn_raw_tau_max',
            m.get('attn_raw_tau_mean', m.get('attn_tau_raw_mean', 0.0)))),
        'attn_tau_min': float(m.get('attn_tau_min', m.get('attn_tau_floor_mean', 0.0))),
        'attn_tau_max': float(m.get('attn_tau_max', 0.0)),

        'attn_selection_margin_mean': float(m.get('attn_selection_margin_mean', 0.0)),
        'attn_positive_margin_mean': float(m.get('attn_positive_margin_mean', 0.0)),
        'attn_positive_margin_max': float(m.get('attn_positive_margin_max', 0.0)),
        'attn_selected_frac': float(m.get('attn_selected_frac', 0.0)),
        'attn_no_active_frac': float(m.get('attn_no_active_frac', 0.0)),
        'rst_rho_mean': float(m.get('rst_rho_mean', 0.0)),
        'rst_rho_std': float(m.get('rst_rho_std', 0.0)),
        'rst_rho_max': float(m.get('rst_rho_max', 0.0)),
        'rst_raw_tau_mean': float(m.get('rst_raw_tau_mean', m.get('rst_tau_raw_mean', 0.0))),
        'rst_raw_tau_min': float(m.get(
            'rst_raw_tau_min',
            m.get('rst_raw_tau_mean', m.get('rst_tau_raw_mean', 0.0)))),
        'rst_raw_tau_max': float(m.get(
            'rst_raw_tau_max',
            m.get('rst_raw_tau_mean', m.get('rst_tau_raw_mean', 0.0)))),
        'rst_tau_min': float(m.get('rst_tau_min', m.get('rst_tau_floor_mean', 0.0))),
        'rst_tau_max': float(m.get('rst_tau_max', 0.0)),

        'rst_selection_margin_mean': float(m.get('rst_selection_margin_mean', 0.0)),
        'rst_positive_margin_mean': float(m.get('rst_positive_margin_mean', 0.0)),
        'rst_positive_margin_max': float(m.get('rst_positive_margin_max', 0.0)),
        'rst_selected_frac': float(m.get('rst_selected_frac', 0.0)),
        'rst_no_active_frac': float(m.get('rst_no_active_frac', 0.0)),
        'attn_tau_off_min': float(m.get('attn_tau_off_min', 0.0)),
        'attn_tau_off_max': float(m.get('attn_tau_off_max', 0.0)),
        'attn_tau_off_p99': float(m.get('attn_tau_off_p99', 0.0)),
        'attn_tau_off_p01': float(m.get('attn_tau_off_p01', 0.0)),
        'attn_tau_off_neg_frac': float(m.get('attn_tau_off_neg_frac', 0.0)),
        'rst_tau_off_min': float(m.get('rst_tau_off_min', 0.0)),
        'rst_tau_off_max': float(m.get('rst_tau_off_max', 0.0)),
        'rst_tau_off_p99': float(m.get('rst_tau_off_p99', 0.0)),
        'rst_tau_off_p01': float(m.get('rst_tau_off_p01', 0.0)),
        'rst_tau_off_neg_frac': float(m.get('rst_tau_off_neg_frac', 0.0)),
        'attn_score_std': float(m.get(
            'attn_score_std', m.get('attn_rho_std', 0.0))),
        'rst_score_std': float(m.get(
            'rst_score_std', m.get('rst_rho_std', 0.0))),
        # Operator-key norm stats.
        'rst_op_key_norm': float(m.get(
            'rst_op_key_norm', m.get('rst_emb_norm', 0.0))),
        'rst_op_key_norm_min': float(m.get(
            'rst_op_key_norm_min', m.get('rst_emb_norm_min', 0.0))),
        'rst_op_key_norm_std': float(m.get(
            'rst_op_key_norm_std', m.get('rst_emb_norm_std', 0.0))),
        'attn_qk_op_key_norm_mean': float(m.get(
            'attn_qk_op_key_norm_mean',
            m.get('attn_qk_emb_norm_mean', 0.0))),
        'attn_qk_op_key_norm_min': float(m.get(
            'attn_qk_op_key_norm_min',
            m.get('attn_qk_emb_norm_min', 0.0))),
        'attn_qk_op_key_norm_std': float(m.get(
            'attn_qk_op_key_norm_std',
            m.get('attn_qk_emb_norm_std', 0.0))),
        'attn_qk_op_key_norm_max': float(m.get(
            'attn_qk_op_key_norm_max',
            m.get('attn_qk_emb_norm_max', 0.0))),
        'attn_v_op_key_norm_mean': float(m.get(
            'attn_v_op_key_norm_mean',
            m.get('attn_v_emb_norm_mean', 0.0))),
        'attn_v_op_key_norm_min': float(m.get(
            'attn_v_op_key_norm_min',
            m.get('attn_v_emb_norm_min', 0.0))),
        'attn_v_op_key_norm_std': float(m.get(
            'attn_v_op_key_norm_std',
            m.get('attn_v_emb_norm_std', 0.0))),
        'attn_v_op_key_norm_max': float(m.get(
            'attn_v_op_key_norm_max',
            m.get('attn_v_emb_norm_max', 0.0))),
        'rst_read_norm': float(m.get('rst_read_norm', 0.0)),
        'rst_write_norm': float(m.get('rst_write_norm', 0.0)),
        'rst_op_key_norm_max': float(m.get(
            'rst_op_key_norm_max', m.get('rst_emb_norm_max', 0.0))),
        'attn_qk_read_norm_mean': float(m.get('attn_qk_read_norm_mean', 0.0)),
        'attn_qk_read_norm_std': float(m.get('attn_qk_read_norm_std', 0.0)),
        'attn_qk_read_norm_max': float(m.get('attn_qk_read_norm_max', 0.0)),
        'attn_qk_write_norm_mean': float(m.get('attn_qk_write_norm_mean', 0.0)),
        'attn_qk_write_norm_std': float(m.get('attn_qk_write_norm_std', 0.0)),
        'attn_qk_write_norm_max': float(m.get('attn_qk_write_norm_max', 0.0)),
        'attn_v_read_norm_mean': float(m.get('attn_v_read_norm_mean', 0.0)),
        'attn_v_read_norm_std': float(m.get('attn_v_read_norm_std', 0.0)),
        'attn_v_read_norm_max': float(m.get('attn_v_read_norm_max', 0.0)),
        'attn_v_write_norm_mean': float(m.get('attn_v_write_norm_mean', 0.0)),
        'attn_v_write_norm_std': float(m.get('attn_v_write_norm_std', 0.0)),
        'attn_v_write_norm_max': float(m.get('attn_v_write_norm_max', 0.0)),
        'rst_read_norm_mean': float(m.get('rst_read_norm_mean', m.get('rst_read_norm', 0.0))),
        'rst_read_norm_std': float(m.get('rst_read_norm_std', 0.0)),
        'rst_read_norm_max': float(m.get('rst_read_norm_max', 0.0)),
        'rst_write_norm_mean': float(m.get('rst_write_norm_mean', m.get('rst_write_norm', 0.0))),
        'rst_write_norm_std': float(m.get('rst_write_norm_std', 0.0)),
        'rst_write_norm_max': float(m.get('rst_write_norm_max', 0.0)),
        'attn_qk_op_gain_mean': float(m.get('attn_qk_op_gain_mean', 0.0)),
        'attn_qk_op_gain_std': float(m.get('attn_qk_op_gain_std', 0.0)),
        'attn_qk_op_gain_p99': float(m.get('attn_qk_op_gain_p99', 0.0)),
        'attn_qk_op_gain_max': float(m.get('attn_qk_op_gain_max', 0.0)),
        'attn_v_op_gain_mean': float(m.get('attn_v_op_gain_mean', 0.0)),
        'attn_v_op_gain_std': float(m.get('attn_v_op_gain_std', 0.0)),
        'attn_v_op_gain_p99': float(m.get('attn_v_op_gain_p99', 0.0)),
        'attn_v_op_gain_max': float(m.get('attn_v_op_gain_max', 0.0)),
        'rst_op_gain_mean': float(m.get('rst_op_gain_mean', 0.0)),
        'rst_op_gain_std': float(m.get('rst_op_gain_std', 0.0)),
        'rst_op_gain_p99': float(m.get('rst_op_gain_p99', 0.0)),
        'rst_op_gain_max': float(m.get('rst_op_gain_max', 0.0)),
        'attn_qk_pool_scale': float(
            m.get('attn_qk_pool_scale', fixed_pool_scale)),
        'attn_v_pool_scale': float(
            m.get('attn_v_pool_scale', fixed_pool_scale)),
        'rst_pool_scale': float(m.get('rst_pool_scale', fixed_pool_scale)),
        'attn_qk_raw_norm': float(m.get('attn_qk_raw_norm', 0.0)),
        'attn_v_raw_norm': float(m.get('attn_v_raw_norm', 0.0)),
        'rst_raw_out_norm': float(m.get('rst_raw_out_norm', 0.0)),
        'residual_norm': float(m.get('residual_norm', 0.0)),
        'residual_norm_max': float(m.get('residual_norm_max', 0.0)),
        'logit_max': float(m.get('logit_max', 0.0)),
        'logit_norm_mean': float(m.get('logit_norm_mean', 0.0)),
        'logit_mean': float(m.get('logit_mean', 0.0)),
        'logit_std': float(m.get('logit_std', 0.0)),
        'token_emb_norm': float(m.get('token_emb_norm', m.get('token_emb_norm', 0.0))),
        'token_emb_norm_max': float(m.get('token_emb_norm_max', 0.0)),
        'attn_logit_max_mean': float(m.get('attn_logit_max_mean', 0.0)),
        'rst_z_mean_active': float(m.get('rst_z_mean_active', 0.0)),
        'attn_qk_z_mean_active': float(m.get('attn_qk_z_mean_active', 0.0)),
        'attn_v_z_mean_active': float(m.get('attn_v_z_mean_active', 0.0)),
        'global_mean_ce': float(m.get('global_mean_ce', 0.0)),
        'pos_frac': float(m.get('pos_frac', 0.0)),
        'pos_mean': float(m.get('pos_mean', 0.0)),
        'neg_mean': float(m.get('neg_mean', 0.0)),
        'inactive_aux_pos_frac': float(m.get('inactive_aux_pos_frac', m.get('pos_frac', 0.0))),
        'inactive_aux_pos_avg': float(m.get('inactive_aux_pos_avg', m.get('pos_mean', 0.0))),
        'inactive_aux_neg_avg': float(m.get('inactive_aux_neg_avg', m.get('neg_mean', 0.0))),
        'inactive_aux_dev_pos': float(m.get('inactive_aux_dev_pos', m.get('dev_pos_max', 0.0))),
        'inactive_aux_dev_neg': float(m.get('inactive_aux_dev_neg', m.get('dev_neg_max', 0.0))),
        'inactive_aux_block_attn': float(m.get(
            'inactive_aux_block_attn', m.get('inactive_aux_block_frac_a', 0.0))),
        'inactive_aux_block_q': float(m.get(
            'inactive_aux_block_q', m.get('inactive_aux_block_frac_q', 0.0))),
        'inactive_aux_block_key': float(m.get(
            'inactive_aux_block_key', m.get('inactive_aux_block_frac_key', 0.0))),
        'inactive_aux_block_qk': float(m.get(
            'inactive_aux_block_qk', m.get('inactive_aux_block_frac_qk', 0.0))),
        'inactive_aux_block_v': float(m.get(
            'inactive_aux_block_v', m.get('inactive_aux_block_frac_v', 0.0))),
        'inactive_aux_block_rst': float(m.get(
            'inactive_aux_block_rst', m.get(
                'inactive_aux_block_frac_rst', m.get('inactive_aux_block_frac_k', 0.0)))),
        'inactive_aux_q_raw': float(m.get('inactive_aux_q_raw', 0.0)),
        'inactive_aux_k_raw': float(m.get('inactive_aux_k_raw', 0.0)),
        'inactive_aux_qk_raw': float(m.get('inactive_aux_qk_raw', 0.0)),
        'inactive_aux_v_raw': float(m.get('inactive_aux_v_raw', 0.0)),
        'inactive_aux_attn_raw': float(m.get('inactive_aux_attn_raw', 0.0)),
        'inactive_aux_rst_raw': float(m.get('inactive_aux_rst_raw', 0.0)),
        'inactive_aux_block_frac_q': float(m.get('inactive_aux_block_frac_q', 0.0)),
        'inactive_aux_block_frac_key': float(m.get('inactive_aux_block_frac_key', 0.0)),
        'inactive_aux_block_frac_qk': float(m.get('inactive_aux_block_frac_qk', 0.0)),
        'inactive_aux_block_frac_v': float(m.get('inactive_aux_block_frac_v', 0.0)),
        'inactive_aux_block_frac_a': float(m.get('inactive_aux_block_frac_a', 0.0)),
        'inactive_aux_block_frac_rst': float(m.get(
            'inactive_aux_block_frac_rst', m.get('inactive_aux_block_frac_k', 0.0))),
        'inactive_aux_block_frac_k': float(m.get('inactive_aux_block_frac_k', 0.0)),
        'dev_pos_max': float(m.get('dev_pos_max', 0.0)),
        'dev_neg_max': float(m.get('dev_neg_max', 0.0)),
        'timestamp': datetime.now().isoformat(),
    }
    rec['_active_tau_regular_available'] = any(
        key in m for key in (
            'q_active_tau_frac', 'k_active_tau_frac', 'qk_active_tau_frac',
            'v_active_tau_frac', 'rst_active_tau_frac',
            'attn_q_active_tau_frac', 'attn_k_active_tau_frac',
            'attn_qk_active_tau_frac', 'attn_v_active_tau_frac',
            'attn_q_active', 'attn_k_active', 'attn_qk_active',
            'attn_v_active', 'rst_active',
        ))
    rec['_rst_op_key_norm_max_available'] = _metric_present(
        'rst_op_key_norm_max',
        *(() if is_rw_key_model else ('rst_emb_norm_max',)))
    rec['_attn_qk_op_key_norm_max_available'] = _metric_present(
        'attn_qk_op_key_norm_max',
        *(() if is_rw_key_model else ('attn_qk_emb_norm_max',)))
    rec['_attn_v_op_key_norm_max_available'] = _metric_present(
        'attn_v_op_key_norm_max',
        *(() if is_rw_key_model else ('attn_v_emb_norm_max',)))
    for _prefix in ('rst', 'attn_qk', 'attn_v'):
        if not rec.get(f'_{_prefix}_op_key_norm_max_available', False):
            rec[f'{_prefix}_op_key_norm_max'] = None
    for _pool in ('attn_qk', 'attn_v'):
        for _name in DIRECT_TAU_ATTN_SPLIT_METRIC_NAMES:
            _fallback = rec.get(f"attn_{_name}", 0.0)
            if _name == 'score_std':
                _fallback = rec.get('attn_score_std', 0.0)
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}', _fallback))
        for _name in DIRECT_TAU_SELECT_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}',
                rec.get(f'attn_{_name}', 0.0)))
        for _name in DIRECT_TAU_EXPOSURE_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}',
                rec.get(f'attn_{_name}', 0.0)))
    for _pool in ('attn_qk', 'attn_v', 'attn_q', 'attn_k', 'rst'):
        for _name in DIRECT_TAU_SPARSITY_METRIC_NAMES:
            _key = f'{_pool}_{_name}'
            if _key in m:
                rec[_key] = float(m[_key])
    if rec['attn_top1_gate_frac'] == 0.0:
        rec['attn_top1_gate_frac'] = rec['attn_raw_gate_max'] / max(rec['attn_gate_sum'], 1e-8)
    if rec['rst_top1_gate_frac'] == 0.0:
        rec['rst_top1_gate_frac'] = rec['rst_raw_gate_max'] / max(rec['rst_gate_sum'], 1e-8)
    _lr = float(ctx.get('current_lr', 0.0))
    for _pool in ('qk', 'v', 'know'):
        for _part in ('emb', 'read', 'write'):
            rec[f'{_pool}_{_part}_grad_ratio'] = float(
                m.get(f'{_pool}_{_part}_grad_ratio', 0.0))
            rec[f'{_pool}_{_part}_update_ratio'] = (
                _lr * rec[f'{_pool}_{_part}_grad_ratio'])
    rec.pop('attn_den_cost_mean', None)
    rec.pop('rst_den_cost_mean', None)
    rec.update({
        'attn_gate_den_sum_mean': float(m.get(
            'attn_gate_den_sum_mean', m.get('attn_intensity_sum_mean', 0.0))),
        'rst_gate_den_sum_mean': float(m.get(
            'rst_gate_den_sum_mean', m.get('rst_intensity_sum_mean', 0.0))),
    })
    for _pool in ('attn_qk', 'attn_v', 'rst'):
        for _name in PAGE_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(
                m.get(f'{_pool}_{_name}', 0.0))
    for _name in (
            'estimated_compute_frac_page',
            'selected_page_count'):
        rec[_name] = float(m.get(_name, 0.0))
    _attach_page_aware_metrics(rec, ctx)
    # Per-layer norms (materialise lists).
    try:
        pl_a = jax.device_get(m['per_layer_attn_out_norm']).tolist()
        pl_k = jax.device_get(m['per_layer_rst_out_norm']).tolist()
    except Exception:
        pl_a, pl_k = [], []
    rec['per_layer_attn_out_norm'] = pl_a
    rec['per_layer_rst_out_norm'] = pl_k
    for _key, _value in m.items():
        if isinstance(_key, str) and (
                _key.startswith('repack/')
                or _key.startswith('sector/')
                or _key.startswith('opspace/')):
            rec[_key] = float(_value)
    return rec


def get_metric(metrics, *keys, default=None):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return default


def _metric_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def fmt_float(x, digits=3, default="n/a"):
    value = _metric_float(x)
    if value is None:
        return default
    return f"{value:.{digits}f}"


def fmt_pct(x, digits=2, default="n/a"):
    value = _metric_float(x)
    if value is None:
        return default
    return f"{value * 100.0:.{digits}f}%"


def fmt_intlike(x, default="n/a"):
    value = _metric_float(x)
    if value is None:
        return default
    return str(int(round(value)))


def _fmt_short_float(x, default="n/a"):
    value = _metric_float(x)
    if value is None:
        return default
    return f"{value:.3g}"


def _opspace_metric(rec, prefixes, name, default=None):
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    keys = []
    for prefix in prefixes:
        keys.append(f"{prefix}/{name}")
        keys.append(f"{prefix}/final/{name}")
    return get_metric(rec, *keys, default=default)


def _opspace_has_metric(rec, prefixes, name):
    return _opspace_metric(rec, prefixes, name, default=None) is not None


def _opspace_product(rec, prefixes, *names):
    product = 1.0
    for name in names:
        value = _metric_float(_opspace_metric(rec, prefixes, name))
        if value is None:
            return None
        product *= value
    return product


def _opspace_logical_ops(rec, prefixes):
    visible_ops = _metric_float(
        _opspace_metric(rec, prefixes, 'visible_ops_per_token'))
    if visible_ops is not None:
        return visible_ops
    return _opspace_product(
        rec, prefixes, 'visible_regions',
        'visible_blocks_per_region', 'operators_per_block')


def _opspace_physical_ops(rec, prefixes):
    physical_ops = _metric_float(
        _opspace_metric(rec, prefixes, 'physical_visible_ops_per_token'))
    if physical_ops is not None:
        return physical_ops
    return _opspace_product(
        rec, prefixes, 'visible_regions',
        'visible_blocks_per_region', 'operators_per_block')


def _opspace_dense_ops(rec, prefixes):
    dense_ops = _opspace_product(
        rec, prefixes, 'num_regions',
        'blocks_per_region', 'operators_per_block')
    if dense_ops is not None:
        return dense_ops
    return _opspace_product(
        rec, prefixes, 'region_count',
        'blocks_per_region', 'operators_per_block')


def _opspace_compute_frac(rec, prefixes, name, visible_ops):
    compute_frac = _metric_float(_opspace_metric(rec, prefixes, name))
    if compute_frac is not None:
        return compute_frac
    dense_ops = _opspace_dense_ops(rec, prefixes)
    if visible_ops is None or dense_ops is None or dense_ops <= 0.0:
        return None
    return visible_ops / dense_ops


def _opspace_backend_name(rec, prefixes):
    for key, label in (
            ('execution_backend_sparse_region_block',
             'sparse_region_block'),
            ('execution_backend_dense', 'dense')):
        value = _metric_float(_opspace_metric(rec, prefixes, key))
        if value is not None and value > 0.5:
            return label
    return "n/a"


def _opspace_backend_summary(rec):
    backends = {
        'qk': _opspace_backend_name(rec, 'opspace/qk'),
        'v': _opspace_backend_name(rec, ('opspace/v', 'opspace/attn_v')),
        'rst': _opspace_backend_name(rec, 'opspace/rst'),
    }
    present = [value for value in backends.values() if value != "n/a"]
    if present and len(set(present)) == 1:
        return present[0]
    return "qk={qk}/v={v}/rst={rst}".format(**backends)


def _opspace_warning_lines(rec):
    checks = (
        ('semantic_drop_frac', 'rst semantic_drop_frac > 0',
         lambda value: value > 0.0),
        ('assignment_collision_count', 'rst assignment_collision_count > 0',
         lambda value: value > 0.0),
        ('all_processed', 'rst all_processed != 1',
         lambda value: value < 0.5),
        ('no_nan', 'rst no_nan != 1',
         lambda value: value < 0.5),
    )
    lines = []
    for name, message, pred in checks:
        value = _metric_float(_opspace_metric(rec, 'opspace/rst', name))
        if value is not None and pred(value):
            lines.append(f"  [opspace/warn] {message}")
    return lines


def _print_train_progress_line(rec, ctx):
    log_message(
        f"  [train] loss={rec['total_loss']:.4f}"
        f" ce={rec['ce_loss']:.4f}"
        f" aux={rec['aux_loss']:.4f}"
        f" grad={rec['grad_norm']:.2f}"
        f" acc={rec['accuracy']:.4f}"
        f" lr={rec['lr']:.2e}"
        f" tok={ctx['progress']:.1f}%"
    )


def _print_v4168_opspace_regular_block(rec):
    qk_prefix = 'opspace/qk'
    v_prefixes = ('opspace/v', 'opspace/attn_v')
    rst_prefix = 'opspace/rst'
    den_power = _fmt_short_float(
        rec.get('admission_den_power', rec.get('den_power', None)))
    log_message(
        "  [opspace] mode=tau_free_relu2"
        f" backend={_opspace_backend_summary(rec)}"
        " direct_tau=false selection_calibration=false"
        f" gate=relu2 den=max(sum(gate),1)^{den_power}"
    )

    qk_diag_present = any(
        _opspace_has_metric(rec, qk_prefix, name)
        for name in ('gate_mass_mean', 'relu_gate_count_mean', 'no_nan'))
    qk_line = (
        "  [opspace/qk]"
        f" backend={_opspace_backend_name(rec, qk_prefix)}"
        f" visible_regions={fmt_intlike(_opspace_metric(rec, qk_prefix, 'visible_regions'))}"
        f" visible_ops={fmt_intlike(_opspace_logical_ops(rec, qk_prefix))}"
    )
    if qk_diag_present:
        qk_line += (
            f" gate_mass={fmt_float(_opspace_metric(rec, qk_prefix, 'gate_mass_mean'), 2)}"
            f" relu_active={fmt_float(_opspace_metric(rec, qk_prefix, 'relu_gate_count_mean'), 1)}")
    else:
        qk_line += " status=diag_missing"
    log_message(qk_line)

    v_parts = [
        "  [opspace/v]",
        f"backend={_opspace_backend_name(rec, v_prefixes)}",
        f"visible_regions={fmt_intlike(_opspace_metric(rec, v_prefixes, 'visible_regions'))}",
        f"visible_ops={fmt_intlike(_opspace_logical_ops(rec, v_prefixes))}",
    ]
    if _opspace_has_metric(rec, v_prefixes, 'gate_mass_mean'):
        v_parts.append(
            f"gate_mass={fmt_float(_opspace_metric(rec, v_prefixes, 'gate_mass_mean'), 2)}")
    if _opspace_has_metric(rec, v_prefixes, 'relu_gate_count_mean'):
        v_parts.append(
            f"relu_active={fmt_float(_opspace_metric(rec, v_prefixes, 'relu_gate_count_mean'), 1)}")
    if _opspace_has_metric(rec, v_prefixes, 'no_nan'):
        v_parts.append(
            f"no_nan={fmt_intlike(_opspace_metric(rec, v_prefixes, 'no_nan'))}")
    log_message(" ".join(v_parts))

    logical_ops = _opspace_logical_ops(rec, rst_prefix)
    physical_ops = _opspace_physical_ops(rec, rst_prefix)
    logical_compute = _opspace_compute_frac(
        rec, rst_prefix, 'logical_compute_frac_vs_dense', logical_ops)
    physical_compute = _opspace_compute_frac(
        rec, rst_prefix, 'physical_compute_frac_vs_dense', physical_ops)
    bucket_fill = _metric_float(
        _opspace_metric(rec, rst_prefix, 'bucket_fill_mean'))
    bucket_cap = _metric_float(
        _opspace_metric(rec, rst_prefix, 'bucket_capacity'))
    region_cap = _opspace_metric(
        rec, rst_prefix, 'region_capacity',
        default=_opspace_metric(rec, rst_prefix, 'bucket_capacity'))
    capacity_regions = _metric_float(
        _opspace_metric(rec, rst_prefix, 'regions_per_owner'))
    bucket_total = None
    if (bucket_fill is not None and bucket_cap is not None
            and capacity_regions is not None and capacity_regions > 0.0):
        bucket_total = bucket_cap * capacity_regions
    if bucket_total is not None and bucket_total > 0.0:
        bucket_part = (
            f"bucket={fmt_intlike(bucket_fill)}/{fmt_intlike(bucket_total)}"
            f" fill={fmt_pct(bucket_fill / bucket_total, 1)}")
    else:
        bucket_part = (
            f"bucket_fill={fmt_intlike(bucket_fill)}"
            f" fill=n/a")
    log_message(
        "  [opspace/rst]"
        f" backend={_opspace_backend_name(rec, rst_prefix)}"
        f" visible_regions={fmt_intlike(_opspace_metric(rec, rst_prefix, 'visible_regions'))}"
        f" visible_ops={fmt_intlike(logical_ops)}"
        f" physical_visible_ops={fmt_intlike(physical_ops)}"
        f" compute={fmt_pct(logical_compute, 2)}/{fmt_pct(physical_compute, 2)}"
        f" {bucket_part}"
        f" cap={fmt_intlike(region_cap)}"
        f" accept={fmt_float(_opspace_metric(rec, rst_prefix, 'primary_accept_frac'), 3)}"
        f" reroute={fmt_float(_opspace_metric(rec, rst_prefix, 'reroute_frac'), 3)}"
        f" drop={fmt_float(_opspace_metric(rec, rst_prefix, 'semantic_drop_frac'), 6)}"
        f" processed={fmt_intlike(_opspace_metric(rec, rst_prefix, 'processed_requests'))}/"
        f"{fmt_intlike(_opspace_metric(rec, rst_prefix, 'selected_requests'))}"
        f" all={fmt_intlike(_opspace_metric(rec, rst_prefix, 'all_processed'))}"
        f" nan={fmt_intlike(_opspace_metric(rec, rst_prefix, 'no_nan'))}"
        f" gate_mass={fmt_float(_opspace_metric(rec, rst_prefix, 'gate_mass_mean'), 2)}"
        f" relu_active={fmt_float(_opspace_metric(rec, rst_prefix, 'relu_gate_count_mean'), 1)}"
    )
    for line in _opspace_warning_lines(rec):
        log_message(line)
    log_message(
        "  [legacy/direct_tau_inactive] hidden for operation-space QK/V/RST")


def _format_output_stab_line(rec, indent="  "):
    def _g(key, default=0.0):
        return float(rec.get(key, default) or 0.0)

    return (
        f"{indent}output_stab: "
        f"contrib_den[a={_g('attn_contrib_den_mean', _g('attn_contrib_den_sum')):.2g}/"
        f"{_g('attn_contrib_den_max'):.2g}/"
        f"{_g('attn_contrib_den_min'):.2g}/"
        f"{_g('attn_contrib_den_floor_frac'):.1e} "
        f"rst={_g('rst_contrib_den_mean', _g('rst_contrib_den_sum')):.2g}/"
        f"{_g('rst_contrib_den_max'):.2g}/"
        f"{_g('rst_contrib_den_min'):.2g}/"
        f"{_g('rst_contrib_den_floor_frac'):.1e}] "
        f"execution[a={_g('attn_compose_norm_mean', _g('attn_compose_norm')):.3f}/"
        f"{_g('attn_compose_norm_max'):.3f} "
        f"rst={_g('rst_compose_norm_mean', _g('rst_compose_norm')):.3f}/"
        f"{_g('rst_compose_norm_max'):.3f}] "
        f"den_ratio[a={_g('attn_den_ratio_mean', _g('attn_den_ratio')):.3f}/"
        f"{_g('attn_den_ratio_max'):.3f} "
        f"rst={_g('rst_den_ratio_mean', _g('rst_den_ratio')):.3f}/"
        f"{_g('rst_den_ratio_max'):.3f}] "
        f"out_norm raw[a={_g('attn_raw_out_norm_mean'):.3f}/"
        f"{_g('attn_raw_out_norm_max'):.3f} "
        f"rst={_g('rst_raw_out_norm_mean'):.3f}/"
        f"{_g('rst_raw_out_norm_max'):.3f}] "
        f"normed[a={_g('attn_normalized_out_norm_mean'):.3f}/"
        f"{_g('attn_normalized_out_norm_max'):.3f} "
        f"rst={_g('rst_normalized_out_norm_mean'):.3f}/"
        f"{_g('rst_normalized_out_norm_max'):.3f}] "
        f"scaled[a={_g('attn_scaled_out_norm_mean'):.3f}/"
        f"{_g('attn_scaled_out_norm_max'):.3f} "
        f"rst={_g('rst_scaled_out_norm_mean'):.3f}/"
        f"{_g('rst_scaled_out_norm_max'):.3f}]"
    )


def _should_print_cb1a_line(rec, ctx):
    return (
        bool(ctx.get('cb1a_enabled') or rec.get('cb1a_enabled', False))
        or float(rec.get('cb1a_weight', 0.0) or 0.0) > 0.0
        or float(rec.get('cb1a_raw', 0.0) or 0.0) != 0.0
    )


def _print_cb1a_regular_block(rec):
    def _g(key, default=0.0):
        return float(rec.get(key, default) or 0.0)

    qk_challenge = _g(
        'cb1a_qk_challenge_raw', rec.get('cb1a_qk_challenge', 0.0))
    qk_prune = _g('cb1a_qk_prune_raw', rec.get('cb1a_qk_prune', 0.0))
    v_challenge = _g(
        'cb1a_v_challenge_raw', rec.get('cb1a_v_challenge', 0.0))
    v_prune = _g('cb1a_v_prune_raw', rec.get('cb1a_v_prune', 0.0))
    rst_challenge = _g(
        'cb1a_rst_challenge_raw', rec.get('cb1a_rst_challenge', 0.0))
    rst_prune = _g('cb1a_rst_prune_raw', rec.get('cb1a_rst_prune', 0.0))

    log_message(
        f"  cb1a: ce[m={_g('cb1a_ce_mean'):.3f} std={_g('cb1a_ce_std'):.3f}]"
        f" raw={_g('cb1a_raw'):.6f}"
        f" weighted={_g('cb1a_weighted', rec.get('cb1a_w', 0.0)):.6f}"
        f" challenge={_g('cb1a_challenge_raw'):.6f}"
        f" prune={_g('cb1a_prune_raw'):.6f}"
        f" pool[qk={_g('cb1a_qk_raw'):.6f}"
        f" v={_g('cb1a_v_raw'):.6f}"
        f" rst={_g('cb1a_rst_raw'):.6f}]"
        f" valid={_g('cb1a_valid'):.3f}"
    )
    log_message(
        f"  cb1a_split: qk[ch={qk_challenge:.6f} pr={qk_prune:.6f}]"
        f" v[ch={v_challenge:.6f} pr={v_prune:.6f}]"
        f" rst[ch={rst_challenge:.6f} pr={rst_prune:.6f}]"
        f" direct_w[qk=({_g('cb1a_qk_challenge_weight', 0.0):.4g},"
        f"{_g('cb1a_qk_prune_weight', 0.0):.4g})"
        f" v=({_g('cb1a_v_challenge_weight', 0.0):.4g},"
        f"{_g('cb1a_v_prune_weight', 0.0):.4g})"
        f" rst=({_g('cb1a_rst_challenge_weight', 0.0):.4g},"
        f"{_g('cb1a_rst_prune_weight', 0.0):.4g})]"
    )


def _print_regular_block(rec, ctx):
    """Print REGULAR tier -~8 lines covering the live training dynamics."""
    is_v4164 = _is_active_srw_version(ctx.get('model_version'))
    is_v4166 = _is_rw_key_srw_version(ctx.get('model_version'))
    is_v4168 = str(ctx.get('model_version')) == V4168_MODEL_VERSION
    is_official_soft_direct_tau = _is_active_srw_version(ctx.get('model_version'))
    official_soft_sparsity_compact = False

    def _fmt_optional_max(prefix):
        if rec.get(f'_{prefix}_op_key_norm_max_available', False):
            return f"{rec[f'{prefix}_op_key_norm_max']:.2f}"
        return "n/a"
    route_std_label = 'rho_std' if is_official_soft_direct_tau else 'score_std'
    def _g(key, default=0.0):
        return float(rec.get(key, default))
    opspace_active = (
        is_v4168 and (
            bool(ctx.get('operation_space_enabled', False))
            or _g('opspace/attn_v/enabled', 0.0) > 0.5))
    aux_note = (
        " aux_is_not_total_minus_ce"
        if is_official_soft_direct_tau
        else "")
    log_message(
        f"[Step {rec['step']}/{ctx['total_micro_steps']} ({ctx['progress']:.1f}%)] "
        f"loss={rec['total_loss']:.4f} ce={rec['ce_loss']:.4f} aux={rec['aux_loss']:.4f} "
        f"total_minus_ce={rec['total_loss_minus_ce']:.4f}{aux_note} | "
        f"grad={rec['grad_norm']:.2f} | "
        f"acc={rec['accuracy']:.4f} lr={rec['lr']:.2e}"
    )
    if opspace_active:
        _print_v4168_opspace_regular_block(rec)
        _print_train_progress_line(rec, ctx)
    if is_v4164:
        if is_official_soft_direct_tau:
            if not opspace_active:
                log_message(
                    f"  strong: q={rec['attn_q_strong']*100:.1f}%"
                    f" k={rec['attn_k_strong']*100:.1f}%"
                    f" v={rec['attn_v_strong']*100:.1f}%"
                    f" rst={rec['rst_strong']*100:.1f}%"
                )
                _print_active_tau_regular_line(rec)
            _soft_gate_label = (
                'soft_gate_B'
                if _is_active_srw_version(ctx.get('model_version'))
                else 'soft_gate_T')
            _power_part = (
                f" boundary_power_p={rec.get('boundary_power_p', 2.0):.3f}"
                f" admission_den_power={rec.get('admission_den_power', rec.get('den_power', 1.0)):.3f}"
                if _is_active_srw_version(ctx.get('model_version'))
                else "")
            if not opspace_active:
                log_message(
                    f"  {_soft_gate_label}: qk={rec['soft_gate_T_qk']:.6f}"
                    f" v={rec['soft_gate_T_v']:.6f}"
                    f" rst={rec['soft_gate_T_rst']:.6f}"
                    f"{_power_part}"
                )
        else:
            log_message(
                f"  act: q={_fmt_act_count(rec['attn_q_active'], ctx['n_qk_cfg'])}"
                f" k={_fmt_act_count(rec['attn_k_active'], ctx['n_qk_cfg'])}"
                f" v={_fmt_act_count(rec['attn_v_active'], ctx['n_v_cfg'])}"
                f" rst={_fmt_act_count(rec['rst_active'], ctx['n_rst_cfg'])}"
                f" | strong: q={rec['attn_q_strong']*100:.1f}%"
                f" k={rec['attn_k_strong']*100:.1f}%"
                f" v={rec['attn_v_strong']*100:.1f}%"
                f" rst={rec['rst_strong']*100:.1f}%"
            )
    else:
        log_message(
            f"  act: attn_qk={_fmt_act_count(rec['attn_qk_active'], ctx['n_qk_cfg'])}"
            f" attn_v={_fmt_act_count(rec['attn_v_active'], ctx['n_v_cfg'])}"
            f" rst={_fmt_act_count(rec['rst_active'], ctx['n_rst_cfg'])}"
            f" | strong: attn_qk={rec['attn_qk_strong']*100:.1f}%"
            f" attn_v={rec['attn_v_strong']*100:.1f}%"
            f" rst={rec['rst_strong']*100:.1f}%"
        )
    if is_v4164:
        _weight_label = 'admission'
        _select_status = ""
        if not opspace_active:
            log_message(
                f"  select: tau[qk={rec['attn_qk_tau_mean']:.4f}"
                f" v={rec['attn_v_tau_mean']:.4f}"
                f" rst={rec['rst_tau_mean']:.4f}]"
                f" margin[qk={rec['attn_qk_selection_margin_mean']:+.4f}"
                f" v={rec['attn_v_selection_margin_mean']:+.4f}"
                f" rst={rec['rst_selection_margin_mean']:+.4f}]"
                f" {_weight_label}[qk={rec['attn_qk_positive_margin_mean']:.4f}"
                f" v={rec['attn_v_positive_margin_mean']:.4f}"
                f" rst={rec['rst_positive_margin_mean']:.4f}]"
                f"{_select_status}"
            )
    if is_v4164 and not is_official_soft_direct_tau:
        log_message(
            f"  gate_max[qk={rec['attn_qk_raw_gate_max']:.1f}"
            f" v={rec['attn_v_raw_gate_max']:.1f}"
            f" rst={rec['rst_raw_gate_max']:.1f}]"
            f" int_max[qk={rec['attn_qk_int_max']:.1f}"
            f" v={rec['attn_v_int_max']:.1f} rst={rec['rst_int_max']:.1f}]"
            f" dead[qk={int(rec['attn_qk_dead_count'])}"
            f" v={int(rec['attn_v_dead_count'])}"
            f" rst={int(rec['rst_dead_count'])}]"
            f" drift[qk={rec['drift_attn_qk_emb']:.2e}"
            f" v={rec['drift_attn_v_emb']:.2e}"
            f" rst={rec['drift_rst_emb']:.2e}]"
        )
    elif not is_v4164:
        log_message(
            f"  gate_max[a={rec['attn_raw_gate_max']:.1f}"
            f" rst={rec['rst_raw_gate_max']:.1f}]"
            f" int_max[a={rec['attn_int_max']:.1f} rst={rec['rst_int_max']:.1f}]"
            f" dead[a={int(rec['attn_dead_count'])} rst={int(rec['rst_dead_count'])}]"
            f" drift[qk={rec['drift_attn_qk_emb']:.2e}"
            f" attn_v={rec['drift_attn_v_emb']:.2e}"
            f" rst={rec['drift_rst_emb']:.2e}]"
        )
    if _is_active_srw_version(ctx.get('model_version')):
        pass
    if is_v4164:
        _pool_scale_part = ""
        if not official_soft_sparsity_compact:
            _pool_scale_part = (
                f" | pool_scale qk={rec['attn_qk_pool_scale']:.3f}"
                f" v={rec['attn_v_pool_scale']:.3f}"
                f" rst={rec['rst_pool_scale']:.3f}")
        if (_is_active_srw_version(ctx.get('model_version'))
                and not opspace_active):
            log_message(
                f"  drive: "
                f"qk[m={rec['attn_qk_drive_mean']:.5f}"
                f" max={rec['attn_qk_drive_max']:.5f}] "
                f"v[m={rec['attn_v_drive_mean']:.5f}"
                f" max={rec['attn_v_drive_max']:.5f}] "
                f"rst[m={rec['rst_drive_mean']:.5f}"
                f" max={rec['rst_drive_max']:.5f}]"
            )
            log_message(
                f"  admission_den: qk={rec['attn_qk_admission_den_sum']:.1f}"
                f" v={rec['attn_v_admission_den_sum']:.1f}"
                f" rst={rec['rst_admission_den_sum']:.1f}"
            )
            log_message(
                f"  execution_mass: qk={rec['attn_qk_execution_mass_sum']:.1f}"
                f" v={rec['attn_v_execution_mass_sum']:.1f}"
                f" rst={rec['rst_execution_mass_sum']:.1f}"
            )
            def _lp(label, metric):
                return (
                    f"{rec.get(f'{label}_{metric}_local', 0.0) * 100:.2f}%/"
                    f"{rec.get(f'{label}_{metric}_pool', 0.0) * 100:.2f}%")
            log_message(
                "  admission local/pool: "
                f"qk={_lp('qk', 'admission')} "
                f"v={_lp('v', 'admission')} "
                f"rst={_lp('rst', 'admission')}")
            log_message(
                "  execution local/pool: "
                f"qk={_lp('qk', 'execution')} "
                f"v={_lp('v', 'execution')} "
                f"rst={_lp('rst', 'execution')}")
            log_message(
                "  eff local/pool: "
                f"qk={_lp('qk', 'eff')} "
                f"v={_lp('v', 'eff')} "
                f"rst={_lp('rst', 'eff')}")
            log_message(
                f"  execution_conc: qk[eff={rec['attn_qk_execution_eff_n']:.1f}"
                f" top1={rec['attn_qk_execution_top1_frac']:.3f}]"
                f" v[eff={rec['attn_v_execution_eff_n']:.1f}"
                f" top1={rec['attn_v_execution_top1_frac']:.3f}]"
                f" rst[eff={rec['rst_execution_eff_n']:.1f}"
                f" top1={rec['rst_execution_top1_frac']:.3f}]"
                f"{_pool_scale_part}"
            )
        elif not _is_active_srw_version(ctx.get('model_version')):
            log_message(
                f"  gate_conc: qk[eff={rec['attn_qk_gate_eff_n']:.1f}"
                f" ratio={rec['attn_qk_gate_eff_ratio']:.3f}"
                f" top1={rec['attn_qk_top1_gate_frac']:.3f}]"
                f" v[eff={rec['attn_v_gate_eff_n']:.1f}"
                f" ratio={rec['attn_v_gate_eff_ratio']:.3f}"
                f" top1={rec['attn_v_top1_gate_frac']:.3f}]"
                f" rst[eff={rec['rst_gate_eff_n']:.1f}"
                f" ratio={rec['rst_gate_eff_ratio']:.3f}"
                f" top1={rec['rst_top1_gate_frac']:.3f}]"
                f"{_pool_scale_part}"
            )
    else:
        log_message(
            f"  gate_conc: a[eff={rec['attn_gate_eff_n']:.1f}"
            f" ratio={rec['attn_gate_eff_ratio']:.3f}"
            f" top1={rec['attn_top1_gate_frac']:.3f}]"
            f" k[eff={rec['rst_gate_eff_n']:.1f}"
            f" ratio={rec['rst_gate_eff_ratio']:.3f}"
            f" top1={rec['rst_top1_gate_frac']:.3f}]"
            f" | pool_scale attn_qk={rec['attn_qk_pool_scale']:.3f}"
            f" attn_v={rec['attn_v_pool_scale']:.3f} rst={rec['rst_pool_scale']:.3f}"
        )
    if ctx.get('model_version') in (
            OFFICIAL_MODEL_VERSION):
        if is_v4164:
            if _is_active_srw_version(ctx.get('model_version')):
                pass
            else:
                log_message(
                    f"  admission_den mean[qk={rec['attn_qk_gate_den_sum_mean']:.1f}"
                    f" v={rec['attn_v_gate_den_sum_mean']:.1f}"
                    f" rst={rec['rst_gate_den_sum_mean']:.1f}]"
                )
        else:
            log_message(
                f"  admission_den mean[a={rec['attn_gate_den_sum_mean']:.1f}"
                f" rst={rec['rst_gate_den_sum_mean']:.1f}]"
            )
    _update_cap_hit_total = sum(
        float(rec.get(_key, 0.0) or 0.0)
        for _key in (
            'update_cap_proj_attn_hit',
            'update_cap_proj_rst_hit',
            'update_cap_op_key_qk_hit',
            'update_cap_op_key_v_hit',
            'update_cap_op_key_rst_hit',
            'update_cap_tau_attn_hit',
            'update_cap_tau_rst_hit',
            'update_cap_raw_tau_qk_hit',
            'update_cap_raw_tau_v_hit',
            'update_cap_raw_tau_rst_hit',
            'update_cap_scan_attn_hit',
            'update_cap_scan_rst_hit',
        ))
    _cap_scale_min = min(
        rec.get('update_cap_proj_attn_scale', 1.0),
        rec.get('update_cap_proj_rst_scale', 1.0),
        rec.get('update_cap_op_key_qk_scale', 1.0),
        rec.get('update_cap_op_key_v_scale', 1.0),
        rec.get('update_cap_op_key_rst_scale', 1.0),
        rec.get('update_cap_tau_attn_scale', 1.0),
        rec.get('update_cap_tau_rst_scale', 1.0),
        rec.get('update_cap_raw_tau_qk_scale', 1.0),
        rec.get('update_cap_raw_tau_v_scale', 1.0),
        rec.get('update_cap_raw_tau_rst_scale', 1.0),
        rec.get('update_cap_scan_attn_scale', 1.0),
        rec.get('update_cap_scan_rst_scale', 1.0),
    )
    _op_key_cap_pre_label = 'op_key_pre' if is_v4166 else 'emb_pre'
    if is_v4164 and _update_cap_hit_total > 0.0:
        raw_tau_part = ""
        raw_tau_hit_part = ""
        if rec.get('update_cap_raw_tau_enabled', 0.0) > 0.0:
            raw_tau_hit_part = (
                f" rQ={rec.get('update_cap_raw_tau_qk_hit', 0.0):.0f}"
                f" rV={rec.get('update_cap_raw_tau_v_hit', 0.0):.0f}"
                f" rR={rec.get('update_cap_raw_tau_rst_hit', 0.0):.0f}")
            raw_tau_part = (
                f" raw_tau[qk pre={rec.get('update_cap_raw_tau_qk_abs_pre', 0.0):.1e}"
                f" post={rec.get('update_cap_raw_tau_qk_abs_post', 0.0):.1e}"
                f" hit={rec.get('update_cap_raw_tau_qk_hit', 0.0):.0f};"
                f" v pre={rec.get('update_cap_raw_tau_v_abs_pre', 0.0):.1e}"
                f" post={rec.get('update_cap_raw_tau_v_abs_post', 0.0):.1e}"
                f" hit={rec.get('update_cap_raw_tau_v_hit', 0.0):.0f};"
                f" rst pre={rec.get('update_cap_raw_tau_rst_abs_pre', 0.0):.1e}"
                f" post={rec.get('update_cap_raw_tau_rst_abs_post', 0.0):.1e}"
                f" hit={rec.get('update_cap_raw_tau_rst_hit', 0.0):.0f}]")
        log_message(
            f"  update_cap: hit[pA={rec.get('update_cap_proj_attn_hit', 0.0):.0f}"
            f" pR={rec.get('update_cap_proj_rst_hit', 0.0):.0f}"
            f" oQ={rec.get('update_cap_op_key_qk_hit', 0.0):.0f}"
            f" oV={rec.get('update_cap_op_key_v_hit', 0.0):.0f}"
            f" oR={rec.get('update_cap_op_key_rst_hit', 0.0):.0f}"
            f"{raw_tau_hit_part}"
            f" sA={rec.get('update_cap_scan_attn_hit', 0.0):.0f}"
            f" sR={rec.get('update_cap_scan_rst_hit', 0.0):.0f}]"
            f" scale_min={_cap_scale_min:.3f}"
            f" proj_pre[a={rec.get('update_cap_proj_attn_ratio_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_proj_rst_ratio_pre', 0.0):.1e}]"
            f" {_op_key_cap_pre_label}[q={rec.get('update_cap_op_key_qk_ratio_pre', 0.0):.1e}"
            f" v={rec.get('update_cap_op_key_v_ratio_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_op_key_rst_ratio_pre', 0.0):.1e}]"
            f"{raw_tau_part}"
            f" scan_pre[a={rec.get('update_cap_scan_attn_abs_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_scan_rst_abs_pre', 0.0):.1e}]"
        )
    elif _update_cap_hit_total > 0.0:
        log_message(
            f"  update_cap: hit[pA={rec.get('update_cap_proj_attn_hit', 0.0):.0f}"
            f" pR={rec.get('update_cap_proj_rst_hit', 0.0):.0f}"
            f" oQ={rec.get('update_cap_op_key_qk_hit', 0.0):.0f}"
            f" oV={rec.get('update_cap_op_key_v_hit', 0.0):.0f}"
            f" oR={rec.get('update_cap_op_key_rst_hit', 0.0):.0f}"
            f" tA={rec.get('update_cap_tau_attn_hit', 0.0):.0f}"
            f" tR={rec.get('update_cap_tau_rst_hit', 0.0):.0f}"
            f" sA={rec.get('update_cap_scan_attn_hit', 0.0):.0f}"
            f" sR={rec.get('update_cap_scan_rst_hit', 0.0):.0f}]"
            f" scale_min={_cap_scale_min:.3f}"
            f" proj_pre[a={rec.get('update_cap_proj_attn_ratio_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_proj_rst_ratio_pre', 0.0):.1e}]"
            f" {_op_key_cap_pre_label}[q={rec.get('update_cap_op_key_qk_ratio_pre', 0.0):.1e}"
            f" v={rec.get('update_cap_op_key_v_ratio_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_op_key_rst_ratio_pre', 0.0):.1e}]"
            f" tau_pre[a={rec.get('update_cap_tau_attn_abs_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_tau_rst_abs_pre', 0.0):.1e}]"
            f" scan_pre[a={rec.get('update_cap_scan_attn_abs_pre', 0.0):.1e}"
            f" r={rec.get('update_cap_scan_rst_abs_pre', 0.0):.1e}]"
        )
    _cap_window_line = _format_update_cap_window_line(
        rec, indent="  ", is_v4164=is_v4164)
    if _cap_window_line:
        log_message(_cap_window_line)
    if is_v4164:
        if opspace_active:
            pass
        else:
            log_message(
                f"  tau: tau_mean[qk={rec['attn_qk_tau_mean']:+.3f}"
                f" v={rec['attn_v_tau_mean']:+.3f}"
                f" rst={rec['rst_tau_mean']:+.3f}]"
                f" abs[qk={rec['attn_qk_tau_abs_mean']:.3f}"
                f" v={rec['attn_v_tau_abs_mean']:.3f}"
                f" rst={rec['rst_tau_abs_mean']:.3f}]"
            )
    else:
        log_message(
            f"  tau: rst_b={rec['tau_rst_bias']:+.2f}"
            f" attn_b=[{rec['tau_attn_bias_0']:+.2f} {rec['tau_attn_bias_1']:+.2f} {rec['tau_attn_bias_2']:+.2f}]"
            f" | tau_mean[attn={rec['attn_tau_mean']:+.3f} rst={rec['rst_tau_mean']:+.3f}]"
            f" abs[attn={rec['attn_tau_abs_mean']:.3f} rst={rec['rst_tau_abs_mean']:.3f}]"
        )
        log_message(
            f"  tau_off rst[min={rec['rst_tau_off_min']:+.2f} p01={rec['rst_tau_off_p01']:+.2f}"
            f" p99={rec['rst_tau_off_p99']:+.2f} max={rec['rst_tau_off_max']:+.2f}"
            f" neg={rec['rst_tau_off_neg_frac']*100:.1f}%]"
            f" attn[min={rec['attn_tau_off_min']:+.2f} p01={rec['attn_tau_off_p01']:+.2f}"
            f" p99={rec['attn_tau_off_p99']:+.2f} max={rec['attn_tau_off_max']:+.2f}"
            f" neg={rec['attn_tau_off_neg_frac']*100:.1f}%]"
        )
    if is_v4164 and not is_official_soft_direct_tau:
        rst_raw_tau_min = rec.get('rst_raw_tau_min', rec.get('rst_raw_tau_mean', 0.0))
        rst_raw_tau_max = rec.get('rst_raw_tau_max', rec.get('rst_raw_tau_mean', 0.0))
        qk_raw_tau_min = rec.get(
            'attn_qk_raw_tau_min', rec.get('attn_qk_raw_tau_mean', 0.0))
        qk_raw_tau_max = rec.get(
            'attn_qk_raw_tau_max', rec.get('attn_qk_raw_tau_mean', 0.0))
        v_raw_tau_min = rec.get(
            'attn_v_raw_tau_min', rec.get('attn_v_raw_tau_mean', 0.0))
        v_raw_tau_max = rec.get(
            'attn_v_raw_tau_max', rec.get('attn_v_raw_tau_mean', 0.0))
        log_message(
            f"  raw_tau qk[min={qk_raw_tau_min:+.2f} max={qk_raw_tau_max:+.2f}]"
            f" v[min={v_raw_tau_min:+.2f} max={v_raw_tau_max:+.2f}]"
            f" rst[min={rst_raw_tau_min:+.2f} max={rst_raw_tau_max:+.2f}]"
        )
    if is_v4166 and not opspace_active:
        log_message(
            f"  {route_std_label}[attn={rec['attn_score_std']:.2f} rst={rec['rst_score_std']:.2f}]"
            f" | op_key_n rst[m={rec['rst_op_key_norm']:.2f} s={rec['rst_op_key_norm_std']:.2f}"
            f" min={rec['rst_op_key_norm_min']:.2f} max={_fmt_optional_max('rst')}]"
            f" attn_qk[m={rec['attn_qk_op_key_norm_mean']:.2f} s={rec['attn_qk_op_key_norm_std']:.2f}"
            f" min={rec['attn_qk_op_key_norm_min']:.2f} max={_fmt_optional_max('attn_qk')}]"
            f" attn_v[m={rec['attn_v_op_key_norm_mean']:.2f} s={rec['attn_v_op_key_norm_std']:.2f}"
            f" min={rec['attn_v_op_key_norm_min']:.2f} max={_fmt_optional_max('attn_v')}]"
        )
    elif is_v4164 and not is_official_soft_direct_tau:
        log_message(
            f"  emb_n rst[m={rec['rst_emb_norm']:.2f} s={rec['rst_emb_norm_std']:.2f}"
            f" min={rec['rst_emb_norm_min']:.2f} max={rec['rst_emb_norm_max']:.2f}]"
            f" qk[m={rec['attn_qk_emb_norm_mean']:.2f} s={rec['attn_qk_emb_norm_std']:.2f}"
            f" min={rec['attn_qk_emb_norm_min']:.2f} max={rec['attn_qk_emb_norm_max']:.2f}]"
            f" v[m={rec['attn_v_emb_norm_mean']:.2f} s={rec['attn_v_emb_norm_std']:.2f}"
            f" min={rec['attn_v_emb_norm_min']:.2f} max={rec['attn_v_emb_norm_max']:.2f}]"
        )
    elif not is_v4164:
        log_message(
            f"  {route_std_label}[attn={rec['attn_score_std']:.2f} rst={rec['rst_score_std']:.2f}]"
            f" | emb_n rst[m={rec['rst_emb_norm']:.2f} s={rec['rst_emb_norm_std']:.2f}"
            f" min={rec['rst_emb_norm_min']:.2f} max={rec['rst_emb_norm_max']:.2f}]"
            f" attn_qk[m={rec['attn_qk_emb_norm_mean']:.2f} s={rec['attn_qk_emb_norm_std']:.2f}"
            f" min={rec['attn_qk_emb_norm_min']:.2f} max={rec['attn_qk_emb_norm_max']:.2f}]"
            f" attn_v[m={rec['attn_v_emb_norm_mean']:.2f} s={rec['attn_v_emb_norm_std']:.2f}"
            f" min={rec['attn_v_emb_norm_min']:.2f} max={rec['attn_v_emb_norm_max']:.2f}]"
        )
    if not is_official_soft_direct_tau:
        log_message(
            f"  rw_n: attn_qk r[m={rec['attn_qk_read_norm_mean']:.2f} s={rec['attn_qk_read_norm_std']:.2f}"
            f" max={rec['attn_qk_read_norm_max']:.2f}]"
            f" w[m={rec['attn_qk_write_norm_mean']:.2f} s={rec['attn_qk_write_norm_std']:.2f}"
            f" max={rec['attn_qk_write_norm_max']:.2f}]"
            f" | attn_v r[m={rec['attn_v_read_norm_mean']:.2f} s={rec['attn_v_read_norm_std']:.2f}"
            f" max={rec['attn_v_read_norm_max']:.2f}]"
            f" w[m={rec['attn_v_write_norm_mean']:.2f} s={rec['attn_v_write_norm_std']:.2f}"
            f" max={rec['attn_v_write_norm_max']:.2f}]"
            f" | k r[m={rec['rst_read_norm_mean']:.2f} s={rec['rst_read_norm_std']:.2f}"
            f" max={rec['rst_read_norm_max']:.2f}]"
            f" w[m={rec['rst_write_norm_mean']:.2f} s={rec['rst_write_norm_std']:.2f}"
            f" max={rec['rst_write_norm_max']:.2f}]"
        )
        log_message(
            f"  op_gain: attn_qk[m={rec['attn_qk_op_gain_mean']:.2f} s={rec['attn_qk_op_gain_std']:.2f}"
            f" max={rec['attn_qk_op_gain_max']:.2f}]"
            f" attn_v[m={rec['attn_v_op_gain_mean']:.2f} s={rec['attn_v_op_gain_std']:.2f}"
            f" max={rec['attn_v_op_gain_max']:.2f}]"
            f" k[m={rec['rst_op_gain_mean']:.2f} s={rec['rst_op_gain_std']:.2f}"
            f" max={rec['rst_op_gain_max']:.2f}]"
        )
    if _should_print_cb1a_line(rec, ctx):
        _print_cb1a_regular_block(rec)
    _pl_a = rec.get('per_layer_attn_out_norm', []) or []
    _pl_k = rec.get('per_layer_rst_out_norm', []) or []
    if (_pl_a or _pl_k) and not is_official_soft_direct_tau:
        log_message(
            f"  per_layer out: attn=[{' '.join(f'{v:.2f}' for v in _pl_a)}]"
            f" know=[{' '.join(f'{v:.2f}' for v in _pl_k)}]"
        )
    log_message(
        f"  time: {format_time(ctx['epoch_elapsed'])}<{format_time(ctx['eta'])},"
        f" {ctx['s_per_it']:.2f}s/it"
    )


def _layer_out_max_from_rec(rec):
    layer_out_max = 0.0
    for key in ('per_layer_attn_out_norm', 'per_layer_rst_out_norm'):
        vals = rec.get(key, []) or []
        try:
            layer_out_max = max(layer_out_max, max(float(v) for v in vals))
        except (TypeError, ValueError):
            pass
    return layer_out_max




def _active_max_from_rec(rec):
    def _g(key, default=0.0):
        return float(rec.get(key, default) or 0.0)
    attn_qk = _g('attn_qk_active')
    return max(_g('attn_q_active', attn_qk),
               _g('attn_k_active', attn_qk),
               _g('attn_v_active'),
               _g('rst_active'))


def _top1_max_from_rec(rec):
    def _g(key, default=0.0):
        return float(rec.get(key, default) or 0.0)
    return max(_g('attn_top1_gate_frac_max', _g('attn_top1_gate_frac')),
               _g('rst_top1_gate_frac_max', _g('rst_top1_gate_frac')))


def _write_json_file(path, obj):
    with _open_file(path, 'w') as f:
        f.write(json.dumps(_json_safe(obj), indent=2, default=str))


def _write_npy_file(path, arr):
    import io
    bio = io.BytesIO()
    np.save(bio, np.asarray(arr))
    with _open_file(path, 'wb') as f:
        f.write(bio.getvalue())


def _jsonable_diag_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _finite_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _fmt_grad_array(rec, key):
    vals = rec.get(key, [])
    arr = np.asarray(vals, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "[]"
    return "[" + ", ".join(f"{float(v):.6g}" for v in arr) + "]"


def _build_analysis_record(base, metrics, ctx):
    """ANALYSIS tier: distribution shape, boundary, and saturation.

    In v4164 this is fed by analysis_step (a separate full-stats forward
    run at val ticks), not by train_step. `base` is an empty dict on the
    official path. All ANALYSIS fields come from
    `metrics`, which is the dict returned by analysis_step. Needs
    `attn_out_norm` / `rst_out_norm` for the raw_n print line, so
    those are pulled from analysis_result too.
    """
    m = metrics
    rec = dict(base)
    fixed_pool_scale = _fixed_depth_pool_scale_from_ctx(ctx)
    # tau per-route std (attn [3]) -materialise once.
    try:
        a_tau_s = np.asarray(jax.device_get(m.get('attn_tau_std', jnp.zeros(3))))
        if a_tau_s.size < 3:
            a_tau_s = np.zeros(3, dtype=np.float32)
    except Exception:
        a_tau_s = np.zeros(3, dtype=np.float32)
    rec.update({
        'attn_out_norm': float(m.get('attn_out_norm', 0.0)),
        'rst_out_norm': float(m.get('rst_out_norm', 0.0)),
        'attn_score_skew': float(m.get(
            'attn_score_skew', m.get('attn_rho_skew', 0.0))),
        'rst_score_skew': float(m.get(
            'rst_score_skew', m.get('rst_rho_skew', 0.0))),
        'attn_score_kurt': float(m.get(
            'attn_score_kurt', m.get('attn_rho_kurt', 0.0))),
        'rst_score_kurt': float(m.get(
            'rst_score_kurt', m.get('rst_rho_kurt', 0.0))),
        'attn_rho_mean': float(m.get('attn_rho_mean', 0.0)),
        'attn_rho_std': float(m.get('attn_rho_std', 0.0)),
        'attn_rho_max': float(m.get('attn_rho_max', 0.0)),
        'rst_rho_mean': float(m.get('rst_rho_mean', 0.0)),
        'rst_rho_std': float(m.get('rst_rho_std', 0.0)),
        'rst_rho_max': float(m.get('rst_rho_max', 0.0)),
        'attn_active_per_token_std': float(m.get('attn_active_per_token_std', 0.0)),
        'rst_active_per_token_std': float(m.get('rst_active_per_token_std', 0.0)),
        'attn_gate_entropy': float(m.get('attn_gate_entropy', 0.0)),
        'rst_gate_entropy': float(m.get('rst_gate_entropy', 0.0)),
        'attn_qk_phi_binary': float(m.get('attn_qk_phi_binary', 0.0)),
        'attn_v_phi_binary': float(m.get('attn_v_phi_binary', 0.0)),
        'rst_phi_binary': float(m.get('rst_phi_binary', 0.0)),
        'attn_z_lt_075': float(m.get('attn_z_lt_075', 0.0)),
        'rst_z_lt_075': float(m.get('rst_z_lt_075', 0.0)),
        'attn_z_lt_030': float(m.get('attn_z_lt_030', 0.0)),
        'rst_z_lt_030': float(m.get('rst_z_lt_030', 0.0)),
        # These are per-validation-batch dead statistics. They are not
        # persistent lifetime dead-neuron counts.
        'attn_dead_count': float(m.get('attn_dead_count', 0.0)),
        'rst_dead_count': float(m.get('rst_dead_count', 0.0)),
        'val_attn_dead_mean': float(m.get(
            'val_attn_dead_mean',
            m.get('attn_dead_count', 0.0))),
        'val_attn_dead_max': float(m.get(
            'val_attn_dead_max',
            m.get('attn_dead_count', 0.0))),
        'val_rst_dead_mean': float(m.get(
            'val_rst_dead_mean',
            m.get('rst_dead_count', 0.0))),
        'val_rst_dead_max': float(m.get(
            'val_rst_dead_max',
            m.get('rst_dead_count', 0.0))),
        'val_dead_batches': int(m.get('val_dead_batches', 1)),
        'attn_int_cap_frac': float(m.get('attn_int_cap_frac', 0.0)),
        'rst_int_cap_frac': float(m.get('rst_int_cap_frac', 0.0)),
        'attn_int_max': float(m.get('attn_int_max', float('nan'))),
        'rst_int_max': float(m.get('rst_int_max', float('nan'))),
        'attn_qk_op_key_norm_max': float(m.get(
            'attn_qk_op_key_norm_max',
            m.get('attn_qk_emb_norm_max', 0.0))),
        'attn_v_op_key_norm_max': float(m.get(
            'attn_v_op_key_norm_max',
            m.get('attn_v_emb_norm_max', 0.0))),
        'rst_op_key_norm_max': float(m.get(
            'rst_op_key_norm_max', m.get('rst_emb_norm_max', 0.0))),
        'attn_tau_std_q': float(a_tau_s[0]),
        'attn_tau_std_k': float(a_tau_s[1]),
        'attn_tau_std_v': float(a_tau_s[2]),
        'rst_tau_std': float(m.get('rst_tau_std', 0.0)),
        'attn_tau_kernel_norm': float(m.get('attn_tau_kernel_norm', 0.0)),
        'rst_tau_kernel_norm': float(m.get('rst_tau_kernel_norm', 0.0)),
        'attn_qk_raw_norm': float(m.get('attn_qk_raw_norm', 0.0)),
        'attn_v_raw_norm': float(m.get('attn_v_raw_norm', 0.0)),
        'rst_raw_out_norm': float(m.get('rst_raw_out_norm', 0.0)),
        'attn_z_sum': float(m.get('attn_z_sum', 0.0)),
        'rst_z_sum': float(m.get('rst_z_sum', 0.0)),
        'attn_den_cost': float(m.get('attn_den_cost', 0.0)),
        'rst_den_cost': float(m.get('rst_den_cost', 0.0)),
        'attn_contrib_den_sum': float(m.get('attn_contrib_den_sum', 0.0)),
        'rst_contrib_den_sum': float(m.get('rst_contrib_den_sum', 0.0)),
        'attn_contrib_den_mean': float(m.get(
            'attn_contrib_den_mean', m.get('attn_contrib_den_sum', 0.0))),
        'rst_contrib_den_mean': float(m.get(
            'rst_contrib_den_mean', m.get('rst_contrib_den_sum', 0.0))),
        'attn_contrib_den': float(m.get('attn_contrib_den', 0.0)),
        'rst_contrib_den': float(m.get('rst_contrib_den', 0.0)),
        'attn_contrib_den_max': float(m.get('attn_contrib_den_max', 0.0)),
        'rst_contrib_den_max': float(m.get('rst_contrib_den_max', 0.0)),
        'attn_contrib_den_min': float(m.get('attn_contrib_den_min', 0.0)),
        'rst_contrib_den_min': float(m.get('rst_contrib_den_min', 0.0)),
        'attn_contrib_den_floor_frac': float(m.get('attn_contrib_den_floor_frac', 0.0)),
        'rst_contrib_den_floor_frac': float(m.get('rst_contrib_den_floor_frac', 0.0)),
        'attn_compose_norm': float(m.get('attn_compose_norm', 0.0)),
        'rst_compose_norm': float(m.get('rst_compose_norm', 0.0)),
        'attn_compose_norm_mean': float(m.get(
            'attn_compose_norm_mean', m.get('attn_compose_norm', 0.0))),
        'rst_compose_norm_mean': float(m.get(
            'rst_compose_norm_mean', m.get('rst_compose_norm', 0.0))),
        'attn_compose_norm_max': float(m.get('attn_compose_norm_max', 0.0)),
        'rst_compose_norm_max': float(m.get('rst_compose_norm_max', 0.0)),
        'attn_coherence': float(m.get('attn_coherence', 0.0)),
        'rst_coherence': float(m.get('rst_coherence', 0.0)),
        'attn_coherence_max': float(m.get('attn_coherence_max', 0.0)),
        'rst_coherence_max': float(m.get('rst_coherence_max', 0.0)),
        'attn_den_ratio': float(m.get('attn_den_ratio', 0.0)),
        'rst_den_ratio': float(m.get('rst_den_ratio', 0.0)),
        'attn_den_ratio_mean': float(m.get(
            'attn_den_ratio_mean', m.get('attn_den_ratio', 0.0))),
        'rst_den_ratio_mean': float(m.get(
            'rst_den_ratio_mean', m.get('rst_den_ratio', 0.0))),
        'attn_den_ratio_max': float(m.get('attn_den_ratio_max', 0.0)),
        'rst_den_ratio_max': float(m.get('rst_den_ratio_max', 0.0)),
        'attn_raw_out_norm_mean': float(m.get('attn_raw_out_norm_mean', 0.0)),
        'rst_raw_out_norm_mean': float(m.get('rst_raw_out_norm_mean', 0.0)),
        'attn_raw_out_norm_max': float(m.get('attn_raw_out_norm_max', 0.0)),
        'rst_raw_out_norm_max': float(m.get('rst_raw_out_norm_max', 0.0)),
        'attn_normalized_out_norm_mean': float(m.get('attn_normalized_out_norm_mean', 0.0)),
        'rst_normalized_out_norm_mean': float(m.get('rst_normalized_out_norm_mean', 0.0)),
        'attn_normalized_out_norm_max': float(m.get('attn_normalized_out_norm_max', 0.0)),
        'rst_normalized_out_norm_max': float(m.get('rst_normalized_out_norm_max', 0.0)),
        'attn_scaled_out_norm_mean': float(m.get('attn_scaled_out_norm_mean', 0.0)),
        'rst_scaled_out_norm_mean': float(m.get('rst_scaled_out_norm_mean', 0.0)),
        'attn_scaled_out_norm_max': float(m.get('attn_scaled_out_norm_max', 0.0)),
        'rst_scaled_out_norm_max': float(m.get('rst_scaled_out_norm_max', 0.0)),
        'attn_activation_cost': float(m.get('attn_activation_cost', 0.0)),
        'rst_activation_cost': float(m.get('rst_activation_cost', 0.0)),
        'attn_current_cost': float(m.get('attn_current_cost', 0.0)),
        'rst_current_cost': float(m.get('rst_current_cost', 0.0)),
        'admission_den_sum': float(m.get('admission_den_sum', 0.0)),
        'attn_admission_den_sum': float(m.get('attn_admission_den_sum', 0.0)),
        'attn_qk_admission_den_sum': float(m.get('attn_qk_admission_den_sum', 0.0)),
        'attn_v_admission_den_sum': float(m.get('attn_v_admission_den_sum', 0.0)),
        'rst_admission_den_sum': float(m.get('rst_admission_den_sum', 0.0)),
        'execution_mass_sum': float(m.get('execution_mass_sum', 0.0)),
        'attn_execution_mass_sum': float(m.get('attn_execution_mass_sum', 0.0)),
        'attn_qk_execution_mass_sum': float(m.get('attn_qk_execution_mass_sum', 0.0)),
        'attn_v_execution_mass_sum': float(m.get('attn_v_execution_mass_sum', 0.0)),
        'rst_execution_mass_sum': float(m.get('rst_execution_mass_sum', 0.0)),
        'attn_qk_load': float(m.get('attn_qk_admission_den_sum', 0.0)),
        'attn_v_load': float(m.get('attn_v_admission_den_sum', 0.0)),
        'rst_load': float(m.get('rst_admission_den_sum', 0.0)),
        'attn_qk_mass': float(m.get('attn_qk_execution_mass_sum', 0.0)),
        'attn_v_mass': float(m.get('attn_v_execution_mass_sum', 0.0)),
        'rst_mass': float(m.get('rst_execution_mass_sum', 0.0)),
        'drive_mean': float(m.get('drive_mean', 0.0)),
        'attn_drive_mean': float(m.get('attn_drive_mean', 0.0)),
        'attn_qk_drive_mean': float(m.get('attn_qk_drive_mean', 0.0)),
        'attn_v_drive_mean': float(m.get('attn_v_drive_mean', 0.0)),
        'rst_drive_mean': float(m.get('rst_drive_mean', 0.0)),
        'drive_max': float(m.get('drive_max', 0.0)),
        'attn_drive_max': float(m.get('attn_drive_max', 0.0)),
        'attn_qk_drive_max': float(m.get('attn_qk_drive_max', 0.0)),
        'attn_v_drive_max': float(m.get('attn_v_drive_max', 0.0)),
        'rst_drive_max': float(m.get('rst_drive_max', 0.0)),
        'execution_eff_n': float(m.get('execution_eff_n', 0.0)),
        'attn_execution_eff_n': float(m.get('attn_execution_eff_n', 0.0)),
        'attn_qk_execution_eff_n': float(m.get('attn_qk_execution_eff_n', 0.0)),
        'attn_v_execution_eff_n': float(m.get('attn_v_execution_eff_n', 0.0)),
        'rst_execution_eff_n': float(m.get('rst_execution_eff_n', 0.0)),
        'execution_top1_frac': float(m.get('execution_top1_frac', 0.0)),
        'execution_top1_frac_max': float(m.get('execution_top1_frac_max', 0.0)),
        'attn_execution_top1_frac': float(m.get('attn_execution_top1_frac', 0.0)),
        'attn_execution_top1_frac_max': float(m.get('attn_execution_top1_frac_max', 0.0)),
        'attn_qk_execution_top1_frac': float(m.get('attn_qk_execution_top1_frac', 0.0)),
        'attn_qk_execution_top1_frac_max': float(m.get('attn_qk_execution_top1_frac_max', 0.0)),
        'attn_v_execution_top1_frac': float(m.get('attn_v_execution_top1_frac', 0.0)),
        'attn_v_execution_top1_frac_max': float(m.get('attn_v_execution_top1_frac_max', 0.0)),
        'rst_execution_top1_frac': float(m.get('rst_execution_top1_frac', 0.0)),
        'rst_execution_top1_frac_max': float(m.get('rst_execution_top1_frac_max', 0.0)),
        'residual_norm': float(m.get('residual_norm', 0.0)),
        'residual_norm_max': float(m.get('residual_norm_max', 0.0)),
        'token_emb_norm': float(m.get('token_emb_norm', m.get('token_emb_norm', 0.0))),
        'token_emb_norm_max': float(m.get('token_emb_norm_max', 0.0)),
        'o_proj_norm': float(m.get('o_proj_norm', 0.0)),
        'q_norm': float(m.get('q_norm', 0.0)),
        'k_norm': float(m.get('k_norm', 0.0)),
        'v_norm': float(m.get('v_norm', 0.0)),
        'logit_max': float(m.get('logit_max', 0.0)),
        'logit_norm_mean': float(m.get('logit_norm_mean', 0.0)),
        'logit_mean': float(m.get('logit_mean', 0.0)),
        'logit_std': float(m.get('logit_std', 0.0)),
        'attn_logit_max_mean': float(m.get('attn_logit_max_mean', 0.0)),
        'o_input_norm': float(m.get('o_input_norm', 0.0)),
        'attn_q_norm_mean': float(m.get('attn_q_norm_mean', 0.0)),
        'attn_q_norm_std': float(m.get('attn_q_norm_std', 0.0)),
        'attn_q_norm_max': float(m.get('attn_q_norm_max', 0.0)),
        'attn_k_norm_mean': float(m.get('attn_k_norm_mean', 0.0)),
        'attn_k_norm_std': float(m.get('attn_k_norm_std', 0.0)),
        'attn_k_norm_max': float(m.get('attn_k_norm_max', 0.0)),
        'attn_logit_mean': float(m.get('attn_logit_mean', 0.0)),
        'attn_logit_std': float(m.get('attn_logit_std', 0.0)),
        'attn_logit_max': float(m.get('attn_logit_max', 0.0)),
        'attn_softmax_top1_mean': float(m.get('attn_softmax_top1_mean', 0.0)),
        'attn_softmax_top1_max': float(m.get('attn_softmax_top1_max', 0.0)),
        'attn_logit_gap_top1_top2_mean': float(m.get(
            'attn_logit_gap_top1_top2_mean', 0.0)),
        'attn_logit_gap_top1_top2_max': float(m.get(
            'attn_logit_gap_top1_top2_max', 0.0)),
        'attn_softmax_entropy_mean': float(m.get(
            'attn_softmax_entropy_mean', 0.0)),
        'attn_softmax_entropy_min': float(m.get(
            'attn_softmax_entropy_min', 0.0)),
        'attn_o_input_norm_mean': float(m.get('attn_o_input_norm_mean', 0.0)),
        'attn_o_input_norm_max': float(m.get('attn_o_input_norm_max', 0.0)),
        'attn_o_output_norm_mean': float(m.get('attn_o_output_norm_mean', 0.0)),
        'attn_o_output_norm_max': float(m.get('attn_o_output_norm_max', 0.0)),
    })
    for _pool in ('attn_qk', 'attn_v'):
        for _name in DIRECT_TAU_ATTN_SPLIT_METRIC_NAMES:
            _fallback = rec.get(f"attn_{_name}", 0.0)
            if _name == 'score_std':
                _fallback = rec.get('attn_score_std', rec.get('attn_rho_std', 0.0))
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}', _fallback))
        for _name in DIRECT_TAU_SELECT_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}',
                rec.get(f'attn_{_name}', 0.0)))
        for _name in DIRECT_TAU_EXPOSURE_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}',
                rec.get(f'attn_{_name}', 0.0)))
    for _pool in ('attn_qk', 'attn_v', 'attn_q', 'attn_k', 'rst'):
        for _name in DIRECT_TAU_SPARSITY_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(m.get(
                f'{_pool}_{_name}', 0.0))
    try:
        rec['attn_logit_max_layer'] = int(jax.device_get(
            m.get('attn_logit_max_layer', -1)))
    except Exception:
        rec['attn_logit_max_layer'] = -1
    # Full pool diagnostics emitted by _pool_param_diagnostics() use the
    # current SRW names: attn_qk_*, attn_v_*, rst_*. Copy qk/v/know
    # alias metrics too, then provide a conservative fallback to the
    # non-full scalar names.
    def _copy_full_pool_stats(dst_prefix, src_prefix=None):
        src_prefix = src_prefix or dst_prefix
        for _kind in (
                'op_key_norm', 'emb_norm', 'read_norm',
                'write_norm', 'op_gain'):
            base_key = f'{src_prefix}_{_kind}'
            for _stat in ('mean', 'std', 'min', 'p50', 'p95', 'p99', 'max'):
                key = f'{base_key}_{_stat}'
                # Some regular-path metrics use e.g. rst_emb_norm as the mean
                # instead of rst_emb_norm_mean.  Use that as a mean fallback.
                fallback = m.get(base_key, 0.0) if _stat == 'mean' else 0.0
                rec[f'{dst_prefix}_{_kind}_{_stat}'] = float(m.get(key, fallback))
        rec[f'{dst_prefix}_pool_scale'] = float(
            m.get(f'{src_prefix}_pool_scale', fixed_pool_scale))

    for _pool in ('attn_qk', 'attn_v', 'rst'):
        _copy_full_pool_stats(_pool)

    # Keep qk/v/know aliases populated for JSONL continuity.
    for _dst, _src in (('qk', 'attn_qk'), ('v', 'attn_v'), ('know', 'rst')):
        _copy_full_pool_stats(_dst, _src)
    for _pool in ('attn_qk', 'attn_v', 'rst'):
        for _name in PAGE_METRIC_NAMES:
            rec[f'{_pool}_{_name}'] = float(
                m.get(f'{_pool}_{_name}', 0.0))
    for _name in (
            'estimated_compute_frac_page',
            'selected_page_count'):
        rec[_name] = float(m.get(_name, 0.0))
    _attach_page_aware_metrics(rec, ctx)
    rec['attn_gate_eff_n'] = float(m.get('attn_gate_eff_n', 0.0))
    rec['attn_gate_eff_ratio'] = float(m.get('attn_gate_eff_ratio', 0.0))
    rec['attn_top1_gate_frac'] = float(m.get('attn_top1_gate_frac', 0.0))
    rec['attn_top1_gate_frac_max'] = float(m.get('attn_top1_gate_frac_max', 0.0))
    rec['rst_gate_eff_n'] = float(m.get('rst_gate_eff_n', 0.0))
    rec['rst_gate_eff_ratio'] = float(m.get('rst_gate_eff_ratio', 0.0))
    rec['rst_top1_gate_frac'] = float(m.get('rst_top1_gate_frac', 0.0))
    rec['rst_top1_gate_frac_max'] = float(m.get('rst_top1_gate_frac_max', 0.0))
    if rec['attn_top1_gate_frac'] == 0.0:
        rec['attn_top1_gate_frac'] = (
            float(m.get('attn_raw_gate_max', 0.0))
            / max(float(m.get('attn_gate_sum', 0.0)), 1e-8))
    if rec['rst_top1_gate_frac'] == 0.0:
        rec['rst_top1_gate_frac'] = (
            float(m.get('rst_raw_gate_max', 0.0))
            / max(float(m.get('rst_gate_sum', 0.0)), 1e-8))
    if rec['attn_top1_gate_frac_max'] == 0.0:
        rec['attn_top1_gate_frac_max'] = rec['attn_top1_gate_frac']
    if rec['rst_top1_gate_frac_max'] == 0.0:
        rec['rst_top1_gate_frac_max'] = rec['rst_top1_gate_frac']
    _lr = float(ctx.get('current_lr', 0.0))
    # Gradient ratios come from train_step metrics, not analysis_step.  The
    # current SRW names are attn_qk/attn_v/rst; keep qk/v aliases for display.
    for _dst, _src in (('qk', 'attn_qk'), ('v', 'attn_v'), ('rst', 'rst'),
                       ('know', 'rst')):
        for _part in ('emb', 'read', 'write'):
            _val = float(m.get(f'{_dst}_{_part}_grad_ratio',
                               m.get(f'{_src}_{_part}_grad_ratio', 0.0)))
            rec[f'{_dst}_{_part}_grad_ratio'] = _val
            rec[f'{_dst}_{_part}_update_ratio'] = _lr * _val
    rec.pop('attn_den_cost', None)
    rec.pop('rst_den_cost', None)
    rec.update({
        'attn_gate_den_sum': float(m.get(
            'attn_gate_den_sum',
            m.get('attn_gate_den_sum_mean',
                  m.get('attn_den_cost', 0.0)))),
        'rst_gate_den_sum': float(m.get(
            'rst_gate_den_sum',
            m.get('rst_gate_den_sum_mean',
                  m.get('rst_den_cost', 0.0)))),
    })
    _attach_validation_dead_fractions(rec, ctx)
    # HBM (host-0 local device 0 snapshot).
    try:
        mem = jax.local_devices()[0].memory_stats()
        if mem:
            used = mem.get('bytes_in_use', 0) / 1e9
            peak = mem.get('peak_bytes_in_use', 0) / 1e9
            limit = mem.get('bytes_limit', 0) / 1e9
            rec['hbm_used_gb'] = float(used)
            rec['hbm_peak_gb'] = float(peak)
            rec['hbm_limit_gb'] = float(limit)
    except Exception:
        pass
    return rec


def _print_analysis_block(rec, ctx):
    # Analysis logging must never kill a run.  Missing optional fields are
    # printed as 0.0 instead of raising KeyError.
    is_v4164 = _is_active_srw_version(ctx.get('model_version'))
    is_v4166 = _is_rw_key_srw_version(ctx.get('model_version'))

    def _g(key, default=0.0):
        return float(rec.get(key, default))

    def _full(prefix):
        return (f"m={_g(f'{prefix}_mean'):.2f} s={_g(f'{prefix}_std'):.2f}"
                f" min={_g(f'{prefix}_min'):.2f} p50={_g(f'{prefix}_p50'):.2f}"
                f" p95={_g(f'{prefix}_p95'):.2f} p99={_g(f'{prefix}_p99'):.2f}"
                f" max={_g(f'{prefix}_max'):.2f}")

    def _emb_full(prefix):
        return (f"m={_g(f'{prefix}_mean'):.2f} s={_g(f'{prefix}_std'):.2f}"
                f" min={_g(f'{prefix}_min'):.2f}"
                f" p95={_g(f'{prefix}_p95'):.2f} p99={_g(f'{prefix}_p99'):.2f}"
                f" max={_g(f'{prefix}_max'):.2f}")

    log_message(
        f"  dist k[skew={rec['rst_score_skew']:+.2f} kurt={rec['rst_score_kurt']:.2f}"
        f" apt_std={rec['rst_active_per_token_std']:.1f} ent={rec['rst_gate_entropy']:.2f}]"
        f" a[skew={rec['attn_score_skew']:+.2f} kurt={rec['attn_score_kurt']:.2f}"
        f" apt_std={rec['attn_active_per_token_std']:.1f} ent={rec['attn_gate_entropy']:.2f}]"
    )
    if is_v4164:
        log_message(
            f"  rho_stats qk[mean={_g('attn_qk_rho_mean'):+.5f}"
            f" std={_g('attn_qk_rho_std'):.5f}"
            f" max={_g('attn_qk_rho_max'):.5f}]"
            f" v[mean={_g('attn_v_rho_mean'):+.5f}"
            f" std={_g('attn_v_rho_std'):.5f}"
            f" max={_g('attn_v_rho_max'):.5f}]"
            f" rst[mean={_g('rst_rho_mean'):+.5f}"
            f" std={_g('rst_rho_std'):.5f}"
            f" max={_g('rst_rho_max'):.5f}]"
        )
    if _is_active_srw_version(ctx.get('model_version')):
        if _is_active_srw_version(ctx.get('model_version')):
            _print_v4164_sparsity_block(rec)
        else:
            _print_v4164_soft_sparsity_block(rec, 'full')
    log_message(
        f"  boundary k[phi={rec['rst_phi_binary']*100:.1f}%"
        f" z<075={rec['rst_z_lt_075']*100:.1f}%"
        f" z<030={rec['rst_z_lt_030']*100:.1f}%]"
        f" a_qk[phi={rec['attn_qk_phi_binary']*100:.1f}%]"
        f" a_v[phi={rec['attn_v_phi_binary']*100:.1f}%]"
        f" attn[z<075={rec['attn_z_lt_075']*100:.1f}%"
        f" z<030={rec['attn_z_lt_030']*100:.1f}%]"
    )
    if is_v4166:
        log_message(
            f"  saturation cap_frac[attn={_g('attn_int_cap_frac')*100:.4f}%"
            f" rst={_g('rst_int_cap_frac')*100:.4f}%]"
            f" | int_max[attn={_g('attn_int_max', float('nan')):.3f}"
            f" rst={_g('rst_int_max', float('nan')):.3f}]"
            f" | op_key_max rst={rec['rst_op_key_norm_max']:.2f}"
            f" attn_qk={rec['attn_qk_op_key_norm_max']:.2f}"
            f" attn_v={rec['attn_v_op_key_norm_max']:.2f}"
        )
    else:
        log_message(
            f"  saturation cap_frac[attn={_g('attn_int_cap_frac')*100:.4f}%"
            f" rst={_g('rst_int_cap_frac')*100:.4f}%]"
            f" | int_max[attn={_g('attn_int_max', float('nan')):.3f}"
            f" rst={_g('rst_int_max', float('nan')):.3f}]"
            f" | emb_max rst={rec['rst_emb_norm_max']:.2f}"
            f" attn_qk={rec['attn_qk_emb_norm_max']:.2f}"
            f" attn_v={rec['attn_v_emb_norm_max']:.2f}"
        )
    _print_validation_dead_stats(rec, ctx)
    if is_v4166:
        log_message(
            f"  op_key_full qk[{_emb_full('attn_qk_op_key_norm')}]"
            f" v[{_emb_full('attn_v_op_key_norm')}]"
            f" k[{_emb_full('rst_op_key_norm')}]"
        )
    else:
        log_message(
            f"  emb_full qk[{_emb_full('attn_qk_emb_norm')}]"
            f" v[{_emb_full('attn_v_emb_norm')}]"
            f" k[{_emb_full('rst_emb_norm')}]"
        )
    log_message(
        f"  rw_full qk_r[{_full('attn_qk_read_norm')}]"
        f" qk_w[{_full('attn_qk_write_norm')}]"
        f" v_r[{_full('attn_v_read_norm')}]"
        f" v_w[{_full('attn_v_write_norm')}]"
        f" k_r[{_full('rst_read_norm')}]"
        f" k_w[{_full('rst_write_norm')}]"
    )
    log_message(
        f"  op_gain_full qk[{_full('attn_qk_op_gain')}]"
        f" v[{_full('attn_v_op_gain')}]"
        f" k[{_full('rst_op_gain')}]"
    )
    if is_v4164:
        log_message(
            f"  gate_conc qk[eff={rec['attn_qk_gate_eff_n']:.1f}"
            f" ratio={rec['attn_qk_gate_eff_ratio']:.3f}"
            f" top1_m={rec['attn_qk_top1_gate_frac']:.3f}"
            f" top1_max={rec['attn_qk_top1_gate_frac_max']:.3f}]"
            f" v[eff={rec['attn_v_gate_eff_n']:.1f}"
            f" ratio={rec['attn_v_gate_eff_ratio']:.3f}"
            f" top1_m={rec['attn_v_top1_gate_frac']:.3f}"
            f" top1_max={rec['attn_v_top1_gate_frac_max']:.3f}]"
            f" rst[eff={rec['rst_gate_eff_n']:.1f}"
            f" ratio={rec['rst_gate_eff_ratio']:.3f}"
            f" top1_m={rec['rst_top1_gate_frac']:.3f}"
            f" top1_max={rec['rst_top1_gate_frac_max']:.3f}]"
            f" | pool_scale qk={rec['attn_qk_pool_scale']:.3f}"
            f" v={rec['attn_v_pool_scale']:.3f} rst={rec['rst_pool_scale']:.3f}"
        )
        if any(f'{label}_candidate_valid_frac' in rec
               for label in ('qk', 'v', 'rst')):
            def _lp(label, metric):
                return (
                    f"{rec.get(f'{label}_{metric}_local', 0.0) * 100:.2f}%/"
                    f"{rec.get(f'{label}_{metric}_pool', 0.0) * 100:.2f}%")

            log_message(
                "  candidate: "
                f"qk[cand={rec.get('qk_candidate_count', 0.0):.0f}"
                f" valid={rec.get('qk_valid_candidate_count', 0.0):.1f}"
                f" full={rec.get('qk_full_pool_size', 0.0):.0f}"
                f" frac_valid={rec.get('qk_candidate_valid_frac', 0.0):.4f}] "
                f"v[cand={rec.get('v_candidate_count', 0.0):.0f}"
                f" valid={rec.get('v_valid_candidate_count', 0.0):.1f}"
                f" full={rec.get('v_full_pool_size', 0.0):.0f}"
                f" frac_valid={rec.get('v_candidate_valid_frac', 0.0):.4f}] "
                f"rst[cand={rec.get('rst_candidate_count', 0.0):.0f}"
                f" valid={rec.get('rst_valid_candidate_count', 0.0):.1f}"
                f" full={rec.get('rst_full_pool_size', 0.0):.0f}"
                f" frac_valid={rec.get('rst_candidate_valid_frac', 0.0):.4f}]")
            log_message(
                "  admission local/pool: "
                f"qk={_lp('qk', 'admission')} "
                f"v={_lp('v', 'admission')} "
                f"rst={_lp('rst', 'admission')}")
            log_message(
                "  execution local/pool: "
                f"qk={_lp('qk', 'execution')} "
                f"v={_lp('v', 'execution')} "
                f"rst={_lp('rst', 'execution')}")
            log_message(
                "  eff local/pool: "
                f"qk={_lp('qk', 'eff')} "
                f"v={_lp('v', 'eff')} "
                f"rst={_lp('rst', 'eff')}")
    else:
        log_message(
            f"  gate_conc a[eff={rec['attn_gate_eff_n']:.1f}"
            f" ratio={rec['attn_gate_eff_ratio']:.3f}"
            f" top1_m={rec['attn_top1_gate_frac']:.3f}"
            f" top1_max={rec['attn_top1_gate_frac_max']:.3f}]"
            f" k[eff={rec['rst_gate_eff_n']:.1f}"
            f" ratio={rec['rst_gate_eff_ratio']:.3f}"
            f" top1_m={rec['rst_top1_gate_frac']:.3f}"
            f" top1_max={rec['rst_top1_gate_frac_max']:.3f}]"
            f" | pool_scale attn_qk={rec['attn_qk_pool_scale']:.3f}"
            f" attn_v={rec['attn_v_pool_scale']:.3f} rst={rec['rst_pool_scale']:.3f}"
        )
    if ctx.get('model_version') in (
            OFFICIAL_MODEL_VERSION):
        if is_v4164:
            log_message(
                f"  admission_den: qk={rec['attn_qk_gate_den_sum_mean']:.1f}"
                f" v={rec['attn_v_gate_den_sum_mean']:.1f}"
                f" rst={rec['rst_gate_den_sum']:.1f}"
            )
        else:
            log_message(
                f"  admission_den: a={rec['attn_gate_den_sum']:.1f}"
                f" rst={rec['rst_gate_den_sum']:.1f}"
            )
    log_message(
        f"  tau_struct k_std={rec['rst_tau_std']:.2f}"
        f" a_std=[{rec['attn_tau_std_q']:.2f} {rec['attn_tau_std_k']:.2f} {rec['attn_tau_std_v']:.2f}]"
        f" k_kern={rec['rst_tau_kernel_norm']:.1f}"
        f" a_kern={rec['attn_tau_kernel_norm']:.1f}"
    )
    log_message(
        f"  raw_n attn_qk={rec['attn_qk_raw_norm']:.2f}"
        f" attn_v={rec['attn_v_raw_norm']:.2f}"
        f" rst={rec['rst_raw_out_norm']:.2f}"
        f" | out_n a={rec['attn_out_norm']:.2f}"
        f" rst={rec['rst_out_norm']:.2f}"
    )
    log_message(
        f"  output resid={rec['residual_norm']:.2f}"
        f" tok_emb={rec['token_emb_norm']:.2f}"
        f" o_proj={rec['o_proj_norm']:.2f}"
        f" q={rec['q_norm']:.2f}"
        f" rst={rec['k_norm']:.2f}"
        f" attn_v={rec['v_norm']:.2f}"
        f" logit_max={rec['logit_max']:.1f}"
        f" o_in={rec['o_input_norm']:.2f}"
    )
    if not is_v4166:
        log_message(
            f"  grad_ratio qk[emb={rec['qk_emb_grad_ratio']:.2e}"
            f" r={rec['qk_read_grad_ratio']:.2e} w={rec['qk_write_grad_ratio']:.2e}]"
            f" v[emb={rec['v_emb_grad_ratio']:.2e}"
            f" r={rec['v_read_grad_ratio']:.2e} w={rec['v_write_grad_ratio']:.2e}]"
            f" k[emb={rec['rst_emb_grad_ratio']:.2e}"
            f" r={rec['rst_read_grad_ratio']:.2e} w={rec['rst_write_grad_ratio']:.2e}]"
        )
    if 'hbm_used_gb' in rec:
        log_message(
            f"  HBM: {rec['hbm_used_gb']:.2f}G / {rec['hbm_limit_gb']:.2f}G"
            f" (peak={rec['hbm_peak_gb']:.2f}G,"
            f" free={rec['hbm_limit_gb'] - rec['hbm_used_gb']:.2f}G)"
        )


def _print_geometry_block(geom):
    def _line(name, label):
        sv = [float(geom.get(f'{name}_geom_sv{i}', 0.0)) for i in range(5)]
        log_message(
            f"  geom {label}[rank={float(geom.get(f'{name}_geom_rank', 0.0)):.1f}"
            f" cos_m={float(geom.get(f'{name}_geom_cos_mean', 0.0)):.3f}"
            f" cos_max={float(geom.get(f'{name}_geom_cos_max', 0.0)):.3f}"
            f" sv5=[{' '.join(f'{v:.2f}' for v in sv)}]]"
        )
    def _page_line(prefix, label):
        if f'{prefix}_page_centroid_rank' not in geom:
            return
        log_message(
            f"  page_geom {label}:"
            f" cent_rank={float(geom.get(f'{prefix}_page_centroid_rank', 0.0)):.1f}"
            f" cent_cos_m={float(geom.get(f'{prefix}_page_centroid_cos_mean', 0.0)):.3f}"
            f" cent_cos_max={float(geom.get(f'{prefix}_page_centroid_cos_max', 0.0)):.3f}"
            f" compact={float(geom.get(f'{prefix}_page_compact_cos_mean', 0.0)):.3f}/"
            f"{float(geom.get(f'{prefix}_page_compact_cos_p05', 0.0)):.3f}/"
            f"{float(geom.get(f'{prefix}_page_compact_cos_min', 0.0)):.3f}"
            f" radius={float(geom.get(f'{prefix}_page_radius_mean', 0.0)):.3f}/"
            f"{float(geom.get(f'{prefix}_page_radius_p95', 0.0)):.3f}/"
            f"{float(geom.get(f'{prefix}_page_radius_max', 0.0)):.3f}"
            f" pad={float(geom.get(f'{prefix}_page_padding_frac', 0.0)):.3f}"
        )
    for _name, _label in (
            ('attn_qk_op_key', 'attn_qk_op_key'),
            ('attn_qk_emb', 'attn_qk_emb'),
            ('attn_qk_read', 'attn_qk_r'),
            ('attn_qk_write', 'attn_qk_w'),
            ('attn_v_op_key', 'attn_v_op_key'),
            ('attn_v_emb', 'attn_v_emb'),
            ('v_emb', 'v_emb'),
            ('attn_v_read', 'attn_v_r'),
            ('attn_v_write', 'attn_v_w'),
            ('rst_op_key', 'rst_op_key'),
            ('rst_emb', 'k_emb'),
            ('rst_read', 'k_r'),
            ('rst_write', 'k_w')):
        if f'{_name}_geom_rank' in geom:
            _line(_name, _label)
    for _name, _label in (
            ('attn_qk', 'attn_qk'),
            ('attn_v', 'attn_v'),
            ('rst', 'rst')):
        _page_line(_name, _label)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train DAWN-SRW v4164 (JAX/Flax, Multi-Device)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start training from scratch (ignore existing checkpoints)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config (global)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--oom-check', action='store_true',
                        help=('Run the startup OOM/JIT train_step probe '
                              '(disabled by default; can also set '
                              'training.oom_check: true).'))
    parser.add_argument('--speed-check', action='store_true',
                        help=('Run the startup step-time/profiling check '
                              '(disabled by default; implies --oom-check; '
                              'can also set training.speed_check: true).'))
    parser.add_argument('--resume-from', '--resume', dest='resume_from',
                        type=str, default=None,
                        help=('Resume from an Orbax run folder or specific '
                              'Orbax step directory. '
                              'Examples: gs://.../run_v... or '
                              'gs://.../run_v.../checkpoints/000000001000'))
    cli_args = parser.parse_args()

    # ----------------------------------------------------------
    # Load config
    # ----------------------------------------------------------
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        # Try as absolute path (or GCS)
        if _file_exists(cli_args.config):
            config_path = cli_args.config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_config(config_path)
    _maybe_initialize_jax_distributed()
    _require_orbax_checkpoint_compat()
    raw_cfg_snapshot = deepcopy(cfg)
    current_yaml_config_snapshot = deepcopy(cfg)
    seed = cfg.get('seed', 42)
    set_seed(seed)

    # Training params (from YAML first, may be overridden by checkpoint config below).
    tcfg = cfg['training']
    model_version_cfg = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)
    is_v4164_cfg = _is_active_srw_version(model_version_cfg)
    is_baseline = _is_baseline_version(model_version_cfg)
    hardware_repack_config = _v4168_hardware_repack_config(
        tcfg, model_version_cfg)
    hardware_repack_enabled = bool(
        hardware_repack_config['hardware_repack_enabled'])
    hardware_sector_execution_enabled = bool(
        hardware_repack_config['hardware_sector_execution_enabled'])
    operation_space_repack_config = _v4168_operation_space_repack_config(
        tcfg, cfg['model'], model_version_cfg)
    operation_space_repack_enabled = bool(
        operation_space_repack_config['operation_space_repack_enabled'])
    operation_space_tau_free_enabled = (
        str(model_version_cfg) == V4168_MODEL_VERSION
        and bool(operation_space_repack_config.get(
            'operation_space_enabled', False))
    )
    tau_init_cfg = (
        None
        if operation_space_tau_free_enabled
        else (
            _v4164_tau_init_config(cfg)
            if _is_active_srw_version(model_version_cfg)
            else None))
    selection_calibration_cfg = (
        _operation_space_disabled_selection_calibration_config(cfg)
        if operation_space_tau_free_enabled
        else _selection_calibration_config(cfg, tau_init_cfg))
    if (selection_calibration_cfg.get('enabled', False)
            and not _is_active_srw_version(model_version_cfg)):
        raise ValueError(
            "training.selection_calibration is only supported for active "
            "DAWN-SRW model versions.")
    # Optional config-driven resume. CLI --resume remains an override for
    # ad-hoc launches, but diagnostic configs can be one-shot.
    configured_resume_from = (
        cli_args.resume_from
        or tcfg.get('resume_from')
        or cfg.get('resume_from'))
    run_speed_check = bool(
        cli_args.speed_check
        or tcfg.get('speed_check', tcfg.get('run_speed_check', False)))
    run_oom_check = bool(
        cli_args.oom_check
        or run_speed_check
        or tcfg.get('oom_check', tcfg.get('run_oom_check', False)))
    # Resume log append policy. Defaults preserve the previous behavior.
    training_log_append_on_resume = bool(
        tcfg.get('training_log_append_on_resume', True))
    batch_size = cli_args.batch_size or tcfg['batch_size']  # global batch size
    num_epochs = cli_args.epochs or tcfg['num_epochs']
    lr = cli_args.lr or tcfg.get('lr', tcfg.get('learning_rate', 6.5e-4))
    weight_decay = tcfg.get('weight_decay', 0.1)
    # losses are off. Keep optimizer weight_decay separate below.
    pool_weight_decay = 0.0
    warmup_ratio = tcfg.get('warmup_ratio', 0.06)
    orth_weight = tcfg.get('orthogonality_weight', 0.01)
    div_weight = 0.0
    lb_weight = 0.0
    tau_reg_weight = 0.0
    dead_penalty_weight = 0.0
    dead_penalty_qk_weight = 0.0
    dead_penalty_v_weight = 0.0
    dead_penalty_rst_weight = 0.0
    dead_exposure_target = 0.0
    inactive_aux_weight = 0.0
    inactive_aux_weight_qk = 0.0
    inactive_aux_weight_q = 0.0
    inactive_aux_weight_k = 0.0
    inactive_aux_weight_v = 0.0
    inactive_aux_weight_rst = 0.0
    inactive_aux_asymmetry = 0.0
    inactive_aux_asymmetry_qk = 0.0
    inactive_aux_asymmetry_q = 0.0
    inactive_aux_asymmetry_k = 0.0
    inactive_aux_asymmetry_v = 0.0
    inactive_aux_asymmetry_rst = 0.0
    inactive_aux_warmup_steps = 0
    inactive_aux_lower_bound = 0.0
    inactive_aux_upper_bound = 0.0
    inactive_aux_bound_eps = 0.0
    inactive_aux_dev_mode = 'raw'
    inactive_aux_ce_clip_std = 0.0
    inactive_aux_z_clip = 0.0
    inactive_aux_z_tanh = False
    inactive_aux_weighted_clip = 0.0
    inactive_aux_normalize_by_layers = True
    inactive_aux_enabled = False
    soft_gate_schedule_active = True
    soft_gate_t_start = float(tcfg.get('soft_gate_t_start', 1.5))
    soft_gate_t_final = float(tcfg.get('soft_gate_t_final', 0.07))
    soft_gate_t_hold_frac = float(tcfg.get('soft_gate_t_hold_frac', 0.10))
    soft_gate_t_anneal_end_frac = float(
        tcfg.get('soft_gate_t_anneal_end_frac', 0.80))
    soft_gate_schedule = str(tcfg.get(
        'soft_gate_t_schedule', tcfg.get('soft_gate_schedule', 'cosine')))
    soft_gate_t_power = float(tcfg.get('soft_gate_t_power', 4.0))
    soft_gate_t_gompertz_center = float(
        tcfg.get('soft_gate_t_gompertz_center', 0.25))
    soft_gate_t_gompertz_steepness = float(
        tcfg.get('soft_gate_t_gompertz_steepness', 8.0))
    pool_specific_gate_t = True
    soft_gate_pool_schedules = _training_soft_gate_pool_schedules(
        tcfg, soft_gate_t_start, soft_gate_t_final,
        soft_gate_t_hold_frac, soft_gate_t_anneal_end_frac,
        soft_gate_schedule, soft_gate_t_power,
        soft_gate_t_gompertz_center, soft_gate_t_gompertz_steepness)
    boundary_power_schedule_active = True
    soft_gate_boundary_power_start = float(tcfg.get(
        'soft_gate_boundary_power_start', 3.0))
    soft_gate_boundary_power_mid = float(tcfg.get(
        'soft_gate_boundary_power_mid', 3.15))
    soft_gate_boundary_power_final = float(tcfg.get(
        'soft_gate_boundary_power_final', 4.0))
    soft_gate_boundary_power_start_frac = float(tcfg.get(
        'soft_gate_boundary_power_start_frac', 0.0))
    soft_gate_boundary_power_mid_frac = float(tcfg.get(
        'soft_gate_boundary_power_mid_frac', 0.800))
    soft_gate_boundary_power_final_frac = float(tcfg.get(
        'soft_gate_boundary_power_final_frac', 0.950))
    soft_gate_effective_active_eps = float(
        tcfg.get('soft_gate_effective_active_eps', 1.0e-6))
    admission_den_power = float(tcfg.get(
        'admission_den_power',
        tcfg.get('v4164_den_power', 1.0)))
    admission_den_grad_scale = float(tcfg.get(
        'admission_den_grad_scale',
        tcfg.get('v4164_den_grad_scale', 1.0)))
    admission_den_config_keys = (
        'admission_den_power',
        'admission_den_grad_scale',
    )
    current_admission_den_config_override = any(
        key in tcfg for key in admission_den_config_keys)
    regular_console_level_default = 'full'
    regular_console_level = str(tcfg.get(
        'regular_console_level',
        regular_console_level_default)).lower()
    if regular_console_level not in ('compact', 'full'):
        raise ValueError(
            "training.regular_console_level must be 'compact' or 'full'.")
    regular_console_host_timing_default = 'always'
    regular_console_host_timing = str(tcfg.get(
        'regular_console_host_timing',
        regular_console_host_timing_default)).lower()
    if regular_console_host_timing not in (
            'always', 'warn_only', 'off', 'false', 'none'):
        raise ValueError(
            "training.regular_console_host_timing must be 'always', "
            "'warn_only', or 'off'.")
    regular_console_top1_warn = float(tcfg.get(
        'regular_console_top1_warn', 0.05))
    regular_console_drive_max_warn = float(tcfg.get(
        'regular_console_drive_max_warn', 1.20))
    regular_console_logging_overhead_warn = float(tcfg.get(
        'regular_console_logging_overhead_warn', 0.05))
    eval_effective_prune_enabled = bool(tcfg.get(
        'eval_effective_prune_enabled',
        _is_active_srw_version(model_version_cfg)))
    eval_effective_prune_eps_list = list(tcfg.get(
        'eval_effective_prune_eps_list', [1.0e-6, 1.0e-5, 1.0e-4]))
    if not _is_active_srw_version(model_version_cfg):
        eval_effective_prune_enabled = False
        eval_effective_prune_eps_list = []
    if operation_space_tau_free_enabled:
        eval_effective_prune_enabled = False
        eval_effective_prune_eps_list = []
    ignored_tau_ce_grad_scale_keys = (
        sorted(k for k in tcfg if k.startswith('tau_ce_grad_scale'))
        if _is_active_srw_version(model_version_cfg)
        else [])
    inactive_aux_start_frac = 0.0
    inactive_aux_full_frac = 0.0
    inactive_aux_schedule = 'linear'
    if not inactive_aux_enabled:
        inactive_aux_weight = 0.0
        inactive_aux_weight_q = 0.0
        inactive_aux_weight_k = 0.0
        inactive_aux_weight_qk = 0.0
        inactive_aux_weight_v = 0.0
        inactive_aux_weight_rst = 0.0
        inactive_aux_asymmetry_q = 0.0
        inactive_aux_asymmetry_k = 0.0
        inactive_aux_asymmetry_qk = 0.0
        inactive_aux_asymmetry_v = 0.0
        inactive_aux_asymmetry_rst = 0.0
        inactive_aux_weighted_clip = 0.0
    cb1a_enabled = False
    cb1a_weight = 0.0
    cb1a_challenge_weight = 0.0
    cb1a_prune_weight = 0.0
    cb1a_qk_weight = 0.0
    cb1a_v_weight = 0.0
    cb1a_rst_weight = 0.0
    cb1a_qk_challenge_weight = 0.0
    cb1a_qk_prune_weight = 0.0
    cb1a_v_challenge_weight = 0.0
    cb1a_v_prune_weight = 0.0
    cb1a_rst_challenge_weight = 0.0
    cb1a_rst_prune_weight = 0.0
    cb1a_ce_mode = 'sigmoid_z'
    cb1a_eps = 1.0e-8
    dead_penalty_weighted_clip = 0.0
    global_grad_clip = tcfg.get('global_grad_clip', 0.0)
    tau_lr_mult = tcfg.get('tau_lr_mult', 1.0)
    tau_grad_clip = tcfg.get('tau_grad_clip', 0.0)
    router_proj_lr_mult = tcfg.get('router_proj_lr_mult', 1.0)
    router_proj_grad_clip = tcfg.get('router_proj_grad_clip', 0.0)
    router_scan_lr_mult = tcfg.get('router_scan_lr_mult', 1.0)
    router_scan_grad_clip = tcfg.get('router_scan_grad_clip', 0.0)
    route_emb_lr_mult = tcfg.get('route_emb_lr_mult', 1.0)
    route_emb_grad_clip = tcfg.get('route_emb_grad_clip', 0.0)
    op_key_lr_mult = tcfg.get('op_key_lr_mult', route_emb_lr_mult)
    op_key_grad_clip = tcfg.get('op_key_grad_clip', route_emb_grad_clip)
    enable_control_update_caps = tcfg.get('enable_control_update_caps', False)
    router_proj_update_ratio_cap = tcfg.get('router_proj_update_ratio_cap', 0.0)
    route_emb_update_ratio_cap = tcfg.get('route_emb_update_ratio_cap', 0.0)
    tau_update_abs_cap = tcfg.get('tau_update_abs_cap', 0.0)
    scan_update_abs_cap = tcfg.get('scan_update_abs_cap', 0.0)
    ckpt_interval = int(tcfg.get('checkpoint_interval', 5000))
    checkpoint_keep_last = int(tcfg.get(
        'checkpoint_keep_last',
        tcfg.get('max_checkpoints_to_keep', 3)))
    best_checkpoint_keep_last = int(tcfg.get('best_checkpoint_keep_last', 3))
    # 2-tier logging cadence.
    log_interval = int(tcfg.get('log_interval', 100))
    log_analysis_multiplier = int(tcfg.get('log_analysis_multiplier', 20))
    heavy_geometry_multiplier = int(tcfg.get('heavy_geometry_multiplier', 5))

    max_seq_len = cfg['model'].get('max_seq_len', 512)

    base_checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints_jax')
    _makedirs(base_checkpoint_dir)

    # ----------------------------------------------------------
    # Run folder: base_checkpoint_dir/run_v{version}_{timestamp}_{rand}/
    # All checkpoints + logs go in the same run folder (like train.py).
    # ----------------------------------------------------------
    resume_path = None
    resume_step = None
    checkpoint_dir = None  # will be set to a run folder
    latest_checkpoint_manager = None
    best_checkpoint_manager = None

    def _join(base, name):
        return _join_path(base, name)

    def _list_run_folders(base):
        """List run_* subdirectories under base (local or GCS).

        FileNotFoundError on GCS is treated as "no prior runs yet" to
        match the local-path behavior (Path.exists() check below) -
        first training on a fresh checkpoint_dir shouldn't fail. Every
        other exception still propagates so credential / permission
        failures can't masquerade as "nothing to resume".
        """
        if _is_gcs(base):
            fs = _get_gcs_fs()
            if fs is None:
                raise ImportError(
                    f"Cannot list GCS path {base}: gcsfs not available.")
            bucket_path = base.replace('gs://', '').rstrip('/')
            try:
                entries = fs.ls(bucket_path)
            except FileNotFoundError:
                return []
            runs = sorted([
                'gs://' + e for e in entries
                if '/run_' in e
            ])
            return runs
        else:
            p = Path(base)
            if not p.exists():
                return []
            return sorted([
                str(d) for d in p.iterdir()
                if d.is_dir() and d.name.startswith('run_')
            ])

    def _broadcast_str_from_host0(s, max_len=512):
        """Broadcast a string (or None) from host 0 to all hosts.

        Must be called collectively on every host. Each host passes its
        local value; only host 0's value is adopted everywhere. Empty
        string and None both encode as all-zero padding and decode back
        to None. max_len caps the payload (GCS URLs usually fit well
        under 512 bytes).
        """
        if s is None:
            s = ''
        encoded = s.encode('utf-8')
        if len(encoded) > max_len:
            raise ValueError(
                f"Path too long for broadcast: {len(encoded)} > {max_len}")
        buf = np.zeros(max_len, dtype=np.uint8)
        if jax.process_index() == 0:
            buf[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

        if _HAVE_BROADCAST:
            broadcast_buf = np.asarray(_bcast_one_to_all(buf))
        else:
            gathered = np.asarray(process_allgather(buf))
            # Shape can be (n_hosts, max_len) or flat (n_hosts * max_len,)
            # depending on JAX version -pick host 0's slice either way.
            if gathered.ndim == 1:
                broadcast_buf = gathered[:max_len]
            else:
                broadcast_buf = gathered[0]
        result = bytes(broadcast_buf).rstrip(b'\x00').decode('utf-8')
        return result if result else None

    # Auto-resume: find latest run folder with Orbax checkpoints
    # (unless --from-scratch). --resume-from takes priority.
    #
    # Only host 0 lists GCS; the resulting (run folder, step) is broadcast
    # to all hosts. Independent per-host listing can diverge
    # under gcsfs caching, concurrent cleanup, or preemption-timing
    # races -a split resume mis-syncs global_step across the mesh and
    # later halts collectives inside train_step.
    if not cli_args.from_scratch:
        if configured_resume_from:
            _parse_orbax_resume_target(configured_resume_from)
        _host0_resume_path = None
        _host0_checkpoint_dir = None
        _host0_resume_step = None
        _host0_explicit_missing = False
        _host0_explicit_error = None

        if jax.process_index() == 0:
            if configured_resume_from:
                try:
                    folder, selected_step, found = _resolve_orbax_resume_from(
                        configured_resume_from)
                except RuntimeError as exc:
                    _host0_explicit_error = str(exc)
                else:
                    if found:
                        _host0_resume_path = folder
                        _host0_checkpoint_dir = folder
                        _host0_resume_step = int(selected_step)
                        print(f"  Resume from specified folder: {_host0_checkpoint_dir}")
                        print(f"  Resuming from Orbax step: {_host0_resume_step}")
                    else:
                        _host0_explicit_missing = True
                        print(
                            f"  No Orbax checkpoint found in "
                            f"{configured_resume_from}")
            else:
                run_folders = _list_run_folders(base_checkpoint_dir)
                print(
                    "Auto-resume discovery: scanning checkpoint step "
                    "directories by filesystem listing.",
                    flush=True,
                )
                for folder in reversed(run_folders):
                    selected_step = _latest_orbax_step_for_run(folder)
                    print(
                        "Auto-resume discovery: "
                        f"folder={folder} latest_step={selected_step}",
                        flush=True,
                    )
                    if selected_step is not None:
                        _host0_resume_path = folder
                        _host0_checkpoint_dir = folder
                        _host0_resume_step = int(selected_step)
                        print(f"  Auto-resume: found checkpoint in {_host0_checkpoint_dir}")
                        print(f"  Resuming from Orbax step: {_host0_resume_step}")
                        break

        # Collective broadcast -all hosts must call.
        resume_path = _broadcast_str_from_host0(_host0_resume_path)
        checkpoint_dir = _broadcast_str_from_host0(_host0_checkpoint_dir)
        _resume_step_str = _broadcast_str_from_host0(
            '' if _host0_resume_step is None else str(_host0_resume_step))
        resume_step = int(_resume_step_str) if _resume_step_str else None
        _explicit_error = _broadcast_str_from_host0(
            _host0_explicit_error, max_len=4096)
        if _explicit_error:
            raise RuntimeError(_explicit_error)
        # Broadcast the explicit-missing signal as a single-byte string
        # so every host raises together.
        _missing_signal = _broadcast_str_from_host0(
            'MISSING' if _host0_explicit_missing else '')
        if _missing_signal == 'MISSING':
            raise FileNotFoundError(
                f"No Orbax checkpoint found in {configured_resume_from}")

    # Create new run folder if not resuming
    if checkpoint_dir is None:
        host0_checkpoint_dir = None
        if jax.process_index() == 0:
            import random as _random
            from datetime import timezone, timedelta
            kst = timezone(timedelta(hours=9))
            ts = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
            rand_suffix = _random.randint(1000, 9999)
            version = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)
            run_name = f"run_v{version}_{ts}_{rand_suffix}"
            host0_checkpoint_dir = _join(base_checkpoint_dir, run_name)
        checkpoint_dir = _broadcast_str_from_host0(
            host0_checkpoint_dir, max_len=2048)
        if checkpoint_dir is None:
            raise RuntimeError("Failed to broadcast new run folder from host 0.")
        _makedirs(checkpoint_dir)
        if jax.process_index() == 0:
            if cli_args.from_scratch:
                print(f"  Starting from scratch (--from-scratch)")
            print(f"  Created new run folder: {checkpoint_dir}")

    log_dir = checkpoint_dir  # logs go in same run folder
    run_id = _path_name(checkpoint_dir)
    checkpoint_git_info = _safe_git_info()
    train_script_git_commit = checkpoint_git_info.get('git_commit')
    _print_jax_distributed_identity(
        'scripts/train_jax.py',
        config_path,
        checkpoint_dir,
        cli_args.from_scratch,
        resume_step,
    )

    # ----------------------------------------------------------
    # Resume config policy: checkpoint full_config is the only source of truth.
    # ----------------------------------------------------------
    saved_training_config = None
    saved_raw_config = None
    saved_full_config = None
    saved_model_config = None
    resume_config_source = 'current_yaml'
    resume_config_restore_read_only = False
    resume_readonly_log_printed = False
    selection_calibration_resume_training_updates = {}
    selection_calibration_resume_restored = False
    selection_calibration_restore_required = False
    if resume_step is not None:
        startup_mesh_model = int(tcfg.get('mesh_model', 1))
        startup_mesh_data = int(tcfg.get('mesh_data', 0))
        if startup_mesh_data == 0:
            startup_mesh_data = jax.device_count() // startup_mesh_model
        _assert_multihost_same_startup_context({
            'trainer_script': 'scripts/train_jax.py',
            'config_path': str(config_path),
            'model_version': model_version_cfg,
            'checkpoint_dir': checkpoint_dir,
            'resume_step': resume_step,
            'from_scratch': bool(cli_args.from_scratch),
            'process_count': jax.process_count(),
            'mesh_data': startup_mesh_data,
            'mesh_model': startup_mesh_model,
            'batch_size': batch_size,
            'train_script_git_commit': train_script_git_commit,
            'hostname': socket.gethostname(),
            'host_id': jax.process_index(),
            'process_index': jax.process_index(),
        })
        resume_config_restore_read_only = True
        checkpoint_metadata = {}
        try:
            checkpoint_metadata = _restore_orbax_metadata(
                _join(checkpoint_dir, 'checkpoints'), resume_step)
        except Exception as e:
            raise RuntimeError(
                "Failed to read resume checkpoint metadata. Automatic config "
                "fallback is disabled; cannot resume deterministically."
            ) from e

        checkpoint_full_config = checkpoint_metadata.get('full_config')
        checkpoint_raw_config = checkpoint_metadata.get('raw_config')
        _require_resume_full_config(checkpoint_full_config)
        saved_full_config = (
            _migrate_v4168_operation_space_full_config_for_clean_schema(
                checkpoint_full_config))
        saved_raw_config = (
            deepcopy(checkpoint_raw_config)
            if isinstance(checkpoint_raw_config, dict)
            and checkpoint_raw_config else None)
        _require_resume_materialized_fields(saved_full_config)

        cfg = deepcopy(saved_full_config)
        saved_training_config = deepcopy(cfg['training'])
        saved_model_config = deepcopy(cfg['model'])
        resume_config_source = 'checkpoint full_config'
        raw_cfg_snapshot = deepcopy(current_yaml_config_snapshot)
        seed = cfg.get('seed', seed)
        set_seed(seed)
        tcfg = cfg['training']
        model_version_cfg = cfg['model']['model_version']
        is_v4164_cfg = _is_active_srw_version(model_version_cfg)
        is_baseline = _is_baseline_version(model_version_cfg)
        hardware_repack_config = _v4168_hardware_repack_config(
            tcfg, model_version_cfg)
        hardware_repack_enabled = bool(
            hardware_repack_config['hardware_repack_enabled'])
        hardware_sector_execution_enabled = bool(
            hardware_repack_config['hardware_sector_execution_enabled'])
        operation_space_repack_config = _v4168_operation_space_repack_config(
            tcfg, cfg['model'], model_version_cfg)
        operation_space_repack_enabled = bool(
            operation_space_repack_config['operation_space_repack_enabled'])
        operation_space_tau_free_enabled = (
            str(model_version_cfg) == V4168_MODEL_VERSION
            and bool(operation_space_repack_config.get(
                'operation_space_enabled', False))
        )
        tau_init_cfg = (
            None
            if operation_space_tau_free_enabled
            else (
                _v4164_tau_init_config(cfg)
                if _is_active_srw_version(model_version_cfg)
                else None))
        selection_calibration_cfg = (
            _operation_space_disabled_selection_calibration_config(cfg)
            if operation_space_tau_free_enabled
            else _selection_calibration_config(cfg, tau_init_cfg))
        max_seq_len = cfg['model']['max_seq_len']
        training_log_append_on_resume = bool(tcfg.get(
            'training_log_append_on_resume',
            training_log_append_on_resume))
        ckpt_interval = int(tcfg['checkpoint_interval'])
        checkpoint_keep_last = int(tcfg.get(
            'checkpoint_keep_last',
            tcfg.get('max_checkpoints_to_keep', checkpoint_keep_last)))
        best_checkpoint_keep_last = int(tcfg.get(
            'best_checkpoint_keep_last', best_checkpoint_keep_last))
        current_admission_den_config_override = False
        if jax.process_index() == 0:
            print("Resume config source: checkpoint full_config")
            print("Resume config fallback: disabled")
            print("  Preserving existing run-folder config snapshots.")

        checkpoint_full_training_config = (
            saved_full_config.get('training')
            if isinstance(saved_full_config, dict)
            and isinstance(saved_full_config.get('training'), dict)
            else None)
        checkpoint_selection_calibration_applied = (
            isinstance(checkpoint_full_training_config, dict)
            and bool(checkpoint_full_training_config.get(
                'selection_calibration_applied', False)))
        selection_calibration_restore_required = bool(
            (not operation_space_tau_free_enabled)
            and (selection_calibration_cfg.get('enabled', False)
                 or checkpoint_selection_calibration_applied))
        if selection_calibration_restore_required:
            _require_selection_calibration_resume_fields(
                checkpoint_full_training_config)
            if resume_config_source != 'checkpoint full_config':
                raise RuntimeError(
                    'Selection calibration resume requires checkpoint '
                    'full_config as the source of truth; restart from '
                    'scratch.')
            saved_training_config = deepcopy(checkpoint_full_training_config)
            cfg = deepcopy(saved_full_config)
            cfg['training'] = deepcopy(saved_training_config)
            if isinstance(cfg.get('model'), dict):
                saved_model_config = deepcopy(cfg['model'])
            tcfg = cfg.setdefault('training', {})

        if saved_training_config:
            saved_admission_den_signature = {
                key: saved_training_config.get(key, '<missing>')
                for key in admission_den_config_keys
            }
            current_admission_den_signature = {
                'admission_den_power': admission_den_power,
                'admission_den_grad_scale': admission_den_grad_scale,
            }
            if current_admission_den_config_override:
                def _norm_sig_value(v):
                    try:
                        return round(float(v), 10)
                    except (TypeError, ValueError):
                        return str(v).lower()

                admission_den_mismatch = any(
                    _norm_sig_value(saved_admission_den_signature[key])
                    != _norm_sig_value(current_admission_den_signature[key])
                    for key in admission_den_config_keys)
                if admission_den_mismatch and jax.process_index() == 0:
                    print(
                        "  Warning: checkpoint training config has stale or "
                        "different admission_den settings. Keeping "
                        "the current config override; use a fresh run for the "
                        "new admission_den settings or remove the override to "
                        "reproduce the checkpoint config.")
            # Apply restored training config. Resume ignores launch-time
            # training/model overrides and keeps checkpoint full_config first.
            batch_size = saved_training_config['batch_size']
            num_epochs = saved_training_config['num_epochs']
            lr = saved_training_config['lr']
            weight_decay = saved_training_config['weight_decay']
            warmup_ratio = saved_training_config['warmup_ratio']
            orth_weight = saved_training_config.get('orthogonality_weight', orth_weight)
            boundary_power_schedule_active = True
            soft_gate_t_start = float(saved_training_config.get(
                'soft_gate_t_start', soft_gate_t_start))
            soft_gate_t_final = float(saved_training_config.get(
                'soft_gate_t_final', soft_gate_t_final))
            soft_gate_t_hold_frac = float(saved_training_config.get(
                'soft_gate_t_hold_frac', soft_gate_t_hold_frac))
            soft_gate_t_anneal_end_frac = float(saved_training_config.get(
                'soft_gate_t_anneal_end_frac',
                soft_gate_t_anneal_end_frac))
            soft_gate_schedule = str(saved_training_config.get(
                'soft_gate_t_schedule',
                saved_training_config.get(
                    'soft_gate_schedule', soft_gate_schedule)))
            soft_gate_t_power = float(saved_training_config.get(
                'soft_gate_t_power', soft_gate_t_power))
            soft_gate_t_gompertz_center = float(saved_training_config.get(
                'soft_gate_t_gompertz_center',
                soft_gate_t_gompertz_center))
            soft_gate_t_gompertz_steepness = float(
                saved_training_config.get(
                    'soft_gate_t_gompertz_steepness',
                    soft_gate_t_gompertz_steepness))
            soft_gate_pool_schedules = _training_soft_gate_pool_schedules(
                saved_training_config, soft_gate_t_start,
                soft_gate_t_final, soft_gate_t_hold_frac,
                soft_gate_t_anneal_end_frac, soft_gate_schedule,
                soft_gate_t_power, soft_gate_t_gompertz_center,
                soft_gate_t_gompertz_steepness)
            soft_gate_effective_active_eps = float(
                saved_training_config.get(
                    'soft_gate_effective_active_eps', 1.0e-6))
            regular_console_level = str(saved_training_config.get(
                'regular_console_level', regular_console_level)).lower()
            if regular_console_level not in ('compact', 'full'):
                raise ValueError(
                    "training.regular_console_level must be 'compact' or 'full'.")
            regular_console_host_timing = str(saved_training_config.get(
                'regular_console_host_timing',
                regular_console_host_timing)).lower()
            if regular_console_host_timing not in (
                    'always', 'warn_only', 'off', 'false', 'none'):
                raise ValueError(
                    "training.regular_console_host_timing must be 'always', "
                    "'warn_only', or 'off'.")
            regular_console_top1_warn = float(saved_training_config.get(
                'regular_console_top1_warn', regular_console_top1_warn))
            regular_console_drive_max_warn = float(saved_training_config.get(
                'regular_console_drive_max_warn',
                regular_console_drive_max_warn))
            regular_console_logging_overhead_warn = float(
                saved_training_config.get(
                    'regular_console_logging_overhead_warn',
                    regular_console_logging_overhead_warn))
            eval_effective_prune_enabled = bool(
                saved_training_config.get(
                    'eval_effective_prune_enabled',
                    eval_effective_prune_enabled))
            eval_effective_prune_eps_list = list(
                saved_training_config.get(
                    'eval_effective_prune_eps_list',
                    eval_effective_prune_eps_list))
            if not _is_active_srw_version(model_version_cfg):
                eval_effective_prune_enabled = False
                eval_effective_prune_eps_list = []
            if operation_space_tau_free_enabled:
                eval_effective_prune_enabled = False
                eval_effective_prune_eps_list = []
            soft_gate_boundary_power_start = float(
                saved_training_config.get(
                    'soft_gate_boundary_power_start',
                    soft_gate_boundary_power_start))
            soft_gate_boundary_power_mid = float(
                saved_training_config.get(
                    'soft_gate_boundary_power_mid',
                    soft_gate_boundary_power_mid))
            soft_gate_boundary_power_final = float(
                saved_training_config.get(
                    'soft_gate_boundary_power_final',
                    soft_gate_boundary_power_final))
            soft_gate_boundary_power_start_frac = float(
                saved_training_config.get(
                    'soft_gate_boundary_power_start_frac',
                    soft_gate_boundary_power_start_frac))
            soft_gate_boundary_power_mid_frac = float(
                saved_training_config.get(
                    'soft_gate_boundary_power_mid_frac',
                    soft_gate_boundary_power_mid_frac))
            soft_gate_boundary_power_final_frac = float(
                saved_training_config.get(
                    'soft_gate_boundary_power_final_frac',
                    soft_gate_boundary_power_final_frac))
            if not current_admission_den_config_override:
                admission_den_power = float(
                    saved_training_config['admission_den_power'])
                admission_den_grad_scale = float(
                    saved_training_config.get(
                        'admission_den_grad_scale', 1.0))
            pool_weight_decay = 0.0
            div_weight = 0.0
            lb_weight = 0.0
            tau_reg_weight = 0.0
            dead_penalty_weight = 0.0
            dead_penalty_qk_weight = 0.0
            dead_penalty_v_weight = 0.0
            dead_penalty_rst_weight = 0.0
            dead_exposure_target = 0.0
            inactive_aux_enabled = False
            inactive_aux_weight = 0.0
            inactive_aux_weight_q = 0.0
            inactive_aux_weight_k = 0.0
            inactive_aux_weight_qk = 0.0
            inactive_aux_weight_v = 0.0
            inactive_aux_weight_rst = 0.0
            inactive_aux_asymmetry = 0.0
            inactive_aux_asymmetry_q = 0.0
            inactive_aux_asymmetry_k = 0.0
            inactive_aux_asymmetry_qk = 0.0
            inactive_aux_asymmetry_v = 0.0
            inactive_aux_asymmetry_rst = 0.0
            inactive_aux_weighted_clip = 0.0
            dead_penalty_weighted_clip = 0.0
            cb1a_enabled = False
            cb1a_weight = 0.0
            cb1a_challenge_weight = 0.0
            cb1a_prune_weight = 0.0
            cb1a_qk_weight = 0.0
            cb1a_v_weight = 0.0
            cb1a_rst_weight = 0.0
            cb1a_qk_challenge_weight = 0.0
            cb1a_qk_prune_weight = 0.0
            cb1a_v_challenge_weight = 0.0
            cb1a_v_prune_weight = 0.0
            cb1a_rst_challenge_weight = 0.0
            cb1a_rst_prune_weight = 0.0
            global_grad_clip = saved_training_config.get(
                'global_grad_clip', global_grad_clip)
            tau_lr_mult = saved_training_config.get(
                'tau_lr_mult', tau_lr_mult)
            tau_grad_clip = saved_training_config.get(
                'tau_grad_clip', tau_grad_clip)
            router_proj_lr_mult = saved_training_config.get(
                'router_proj_lr_mult', router_proj_lr_mult)
            router_proj_grad_clip = saved_training_config.get(
                'router_proj_grad_clip', router_proj_grad_clip)
            router_scan_lr_mult = saved_training_config.get(
                'router_scan_lr_mult', router_scan_lr_mult)
            router_scan_grad_clip = saved_training_config.get(
                'router_scan_grad_clip', router_scan_grad_clip)
            route_emb_lr_mult = saved_training_config.get(
                'route_emb_lr_mult', route_emb_lr_mult)
            route_emb_grad_clip = saved_training_config.get(
                'route_emb_grad_clip', route_emb_grad_clip)
            op_key_lr_mult = saved_training_config.get(
                'op_key_lr_mult', op_key_lr_mult)
            op_key_grad_clip = saved_training_config.get(
                'op_key_grad_clip', op_key_grad_clip)
            enable_control_update_caps = saved_training_config.get(
                'enable_control_update_caps', enable_control_update_caps)
            router_proj_update_ratio_cap = saved_training_config.get(
                'router_proj_update_ratio_cap', router_proj_update_ratio_cap)
            route_emb_update_ratio_cap = saved_training_config.get(
                'route_emb_update_ratio_cap', route_emb_update_ratio_cap)
            tau_update_abs_cap = saved_training_config.get(
                'tau_update_abs_cap', tau_update_abs_cap)
            scan_update_abs_cap = saved_training_config.get(
                'scan_update_abs_cap', scan_update_abs_cap)
            if not cli_args.oom_check:
                run_oom_check = bool(saved_training_config.get(
                    'oom_check',
                    saved_training_config.get('run_oom_check', run_oom_check)))
            if not cli_args.speed_check:
                run_speed_check = bool(saved_training_config.get(
                    'speed_check',
                    saved_training_config.get('run_speed_check', run_speed_check)))
            run_oom_check = bool(run_oom_check or run_speed_check)
            training_log_append_on_resume = bool(
                saved_training_config.get(
                    'training_log_append_on_resume',
                    training_log_append_on_resume))
            ckpt_interval = int(saved_training_config['checkpoint_interval'])
            checkpoint_keep_last = int(saved_training_config.get(
                'checkpoint_keep_last',
                saved_training_config.get(
                    'max_checkpoints_to_keep', checkpoint_keep_last)))
            best_checkpoint_keep_last = int(saved_training_config.get(
                'best_checkpoint_keep_last', best_checkpoint_keep_last))
            log_interval = int(saved_training_config['log_interval'])
            log_analysis_multiplier = int(
                saved_training_config['log_analysis_multiplier'])
            heavy_geometry_multiplier = int(
                saved_training_config['heavy_geometry_multiplier'])

            pool_weight_decay = 0.0
            div_weight = 0.0
            lb_weight = 0.0
            tau_reg_weight = 0.0
            dead_penalty_weight = 0.0
            dead_penalty_qk_weight = 0.0
            dead_penalty_v_weight = 0.0
            dead_penalty_rst_weight = 0.0
            dead_exposure_target = 0.0
            inactive_aux_enabled = False
            inactive_aux_weight = 0.0
            inactive_aux_weight_q = 0.0
            inactive_aux_weight_k = 0.0
            inactive_aux_weight_qk = 0.0
            inactive_aux_weight_v = 0.0
            inactive_aux_weight_rst = 0.0
            inactive_aux_asymmetry = 0.0
            inactive_aux_asymmetry_q = 0.0
            inactive_aux_asymmetry_k = 0.0
            inactive_aux_asymmetry_qk = 0.0
            inactive_aux_asymmetry_v = 0.0
            inactive_aux_asymmetry_rst = 0.0
            inactive_aux_weighted_clip = 0.0
            dead_penalty_weighted_clip = 0.0
            cb1a_enabled = False
            cb1a_weight = 0.0
            cb1a_challenge_weight = 0.0
            cb1a_prune_weight = 0.0
            cb1a_qk_weight = 0.0
            cb1a_v_weight = 0.0
            cb1a_rst_weight = 0.0
            cb1a_qk_challenge_weight = 0.0
            cb1a_qk_prune_weight = 0.0
            cb1a_v_challenge_weight = 0.0
            cb1a_v_prune_weight = 0.0
            cb1a_rst_challenge_weight = 0.0
            cb1a_rst_prune_weight = 0.0
            if jax.process_index() == 0:
                print(
                    f"  Training config restored from {resume_config_source} "
                    "(restored config takes precedence)")

        restore_saved_selection_calibration = (
            selection_calibration_restore_required)
        if restore_saved_selection_calibration:
            for _pool in POOL_SCHEDULE_NAMES:
                _prefix = f'soft_gate_t_{_pool}'
                for _key in (
                        'schedule',
                        'sort',
                        'band',
                        'mid',
                        'late',
                        'final',
                        'sort_end_frac',
                        'band_reach_frac',
                        'formation_end_frac',
                        'sharpen_end_frac',
                        'formation_power',
                        'sharpen_power'):
                    _field = f'{_prefix}_{_key}'
                    _raw_value = saved_training_config[_field]
                    _value = (
                        str(_raw_value)
                        if _key == 'schedule'
                        else float(_raw_value))
                    soft_gate_pool_schedules[_pool][_key] = _value
                    selection_calibration_resume_training_updates[_field] = (
                        _value)
            soft_gate_boundary_power_start = float(
                saved_training_config['soft_gate_boundary_power_start'])
            soft_gate_boundary_power_mid = float(
                saved_training_config['soft_gate_boundary_power_mid'])
            soft_gate_boundary_power_final = float(
                saved_training_config['soft_gate_boundary_power_final'])
            soft_gate_boundary_power_start_frac = float(
                saved_training_config['soft_gate_boundary_power_start_frac'])
            soft_gate_boundary_power_mid_frac = float(
                saved_training_config['soft_gate_boundary_power_mid_frac'])
            soft_gate_boundary_power_final_frac = float(
                saved_training_config['soft_gate_boundary_power_final_frac'])
            admission_den_power = float(
                saved_training_config['admission_den_power'])
            admission_den_grad_scale = float(
                saved_training_config['admission_den_grad_scale'])
            soft_gate_effective_active_eps = float(
                saved_training_config['soft_gate_effective_active_eps'])
            for _key in SELECTION_CALIBRATION_RESUME_REQUIRED_FIELDS:
                if _key not in selection_calibration_resume_training_updates:
                    selection_calibration_resume_training_updates[_key] = (
                        saved_training_config[_key])
            cfg.setdefault('training', {}).update(
                selection_calibration_resume_training_updates)
            selection_calibration_resume_restored = True
            if jax.process_index() == 0:
                print(
                    "  Selection calibration policy: resume restores from "
                    "checkpoint; recomputation disabled.")
                print(
                    "  Selection calibration schedule restored from "
                    "checkpoint config.")

    if is_v4164_cfg:
        pool_weight_decay = 0.0
        div_weight = 0.0
        lb_weight = 0.0
        dead_penalty_weight = 0.0
        dead_penalty_qk_weight = 0.0
        dead_penalty_v_weight = 0.0
        dead_penalty_rst_weight = 0.0
        dead_exposure_target = 0.0
        inactive_aux_enabled = False
        soft_gate_schedule_active = True
        pool_specific_gate_t = True
        boundary_power_schedule_active = True

    is_baseline = _is_baseline_version(model_version_cfg)
    if is_baseline:
        pool_weight_decay = 0.0
        div_weight = 0.0
        lb_weight = 0.0
        tau_reg_weight = 0.0
        dead_penalty_weight = 0.0
        dead_penalty_qk_weight = 0.0
        dead_penalty_v_weight = 0.0
        dead_penalty_rst_weight = 0.0
        dead_exposure_target = 0.0
        inactive_aux_enabled = False
        inactive_aux_weight = 0.0
        inactive_aux_weight_q = 0.0
        inactive_aux_weight_k = 0.0
        inactive_aux_weight_qk = 0.0
        inactive_aux_weight_v = 0.0
        inactive_aux_weight_rst = 0.0
        inactive_aux_asymmetry = 0.0
        inactive_aux_asymmetry_q = 0.0
        inactive_aux_asymmetry_k = 0.0
        inactive_aux_asymmetry_qk = 0.0
        inactive_aux_asymmetry_v = 0.0
        inactive_aux_asymmetry_rst = 0.0
        inactive_aux_weighted_clip = 0.0
        dead_penalty_weighted_clip = 0.0
        cb1a_enabled = False
        cb1a_weight = 0.0
        cb1a_challenge_weight = 0.0
        cb1a_prune_weight = 0.0
        cb1a_qk_weight = 0.0
        cb1a_v_weight = 0.0
        cb1a_rst_weight = 0.0
        soft_gate_schedule_active = False
        pool_specific_gate_t = False
        boundary_power_schedule_active = False

    compact_train_metrics = _is_active_srw_version(model_version_cfg)
    drift_diagnostics_enabled = (
        (not bool(compact_train_metrics)) and not is_baseline)

    if not inactive_aux_enabled:
        inactive_aux_weight = 0.0
        inactive_aux_weight_q = 0.0
        inactive_aux_weight_k = 0.0
        inactive_aux_weight_qk = 0.0
        inactive_aux_weight_v = 0.0
        inactive_aux_weight_rst = 0.0
        inactive_aux_asymmetry_q = 0.0
        inactive_aux_asymmetry_k = 0.0
        inactive_aux_asymmetry_qk = 0.0
        inactive_aux_asymmetry_v = 0.0
        inactive_aux_asymmetry_rst = 0.0
        inactive_aux_weighted_clip = 0.0

    if soft_gate_schedule_active:
        _soft_gate_schedule_name = soft_gate_schedule.lower()
        if _soft_gate_schedule_name not in SOFT_GATE_T_SCHEDULE_NAMES:
            raise ValueError(
                f"Unsupported soft_gate_t_schedule={soft_gate_schedule!r}; "
                f"{_soft_gate_schedule_expected_msg()}")
        if not (_soft_gate_schedule_name == 'developmental_band'
                and pool_specific_gate_t):
            if _soft_gate_schedule_name == 'developmental_band':
                _validate_soft_gate_schedule_config(
                    'soft_gate_t', soft_gate_pool_schedules['qk'])
            else:
                _validate_soft_gate_schedule_config('soft_gate_t', {
                    'schedule': soft_gate_schedule,
                    'start': soft_gate_t_start,
                    'final': soft_gate_t_final,
                    'hold_frac': soft_gate_t_hold_frac,
                    'anneal_end_frac': soft_gate_t_anneal_end_frac,
                    'power': soft_gate_t_power,
                    'gompertz_center': soft_gate_t_gompertz_center,
                    'gompertz_steepness': soft_gate_t_gompertz_steepness,
                })
        if pool_specific_gate_t:
            for _pool, _cfg in soft_gate_pool_schedules.items():
                _validate_soft_gate_schedule_config(
                    f"soft_gate_t_{_pool}", _cfg,
                    require_pool_specific_devband_fields=True)
    if boundary_power_schedule_active:
        if soft_gate_boundary_power_start <= 0.0:
            raise ValueError(
                "soft_gate_boundary_power_start must be > 0, got "
                f"{soft_gate_boundary_power_start}")
        if soft_gate_boundary_power_mid <= 0.0:
            raise ValueError(
                "soft_gate_boundary_power_mid must be > 0, got "
                f"{soft_gate_boundary_power_mid}")
        if soft_gate_boundary_power_final <= 0.0:
            raise ValueError(
                "soft_gate_boundary_power_final must be > 0, got "
                f"{soft_gate_boundary_power_final}")
        if not (0.0 <= soft_gate_boundary_power_start_frac
                < soft_gate_boundary_power_mid_frac
                < soft_gate_boundary_power_final_frac <= 1.0):
            raise ValueError(
                "soft_gate_boundary_power fractions must satisfy "
                "0 <= start_frac < mid_frac < final_frac <= 1, got "
                f"{soft_gate_boundary_power_start_frac}, "
                f"{soft_gate_boundary_power_mid_frac}, "
                f"{soft_gate_boundary_power_final_frac}")
    if admission_den_power < 0.0:
        raise ValueError(
            f"admission_den_power must be >= 0, got {admission_den_power}")
    if not (0.0 <= admission_den_grad_scale <= 1.0):
        raise ValueError(
            "admission_den_grad_scale must be in [0, 1], got "
            f"{admission_den_grad_scale}")

    # Build training_config dict for saving in checkpoints
    training_config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'pool_weight_decay': pool_weight_decay,
        'warmup_ratio': warmup_ratio,
        'orthogonality_weight': orth_weight,
        'diversity_weight': div_weight,
        'load_balance_weight': lb_weight,
        'tau_reg_weight': tau_reg_weight,
        'dead_penalty_weight': dead_penalty_weight,
        'dead_penalty_qk_weight': dead_penalty_qk_weight,
        'dead_penalty_v_weight': dead_penalty_v_weight,
        'dead_penalty_rst_weight': dead_penalty_rst_weight,
        'dead_exposure_target': dead_exposure_target,
        'inactive_aux_weight': inactive_aux_weight,
        'inactive_aux_weight_qk': inactive_aux_weight_qk,
        'inactive_aux_weight_v': inactive_aux_weight_v,
        'inactive_aux_weight_rst': inactive_aux_weight_rst,
        'inactive_aux_asymmetry': inactive_aux_asymmetry,
        'inactive_aux_asymmetry_qk': inactive_aux_asymmetry_qk,
        'inactive_aux_asymmetry_v': inactive_aux_asymmetry_v,
        'inactive_aux_asymmetry_rst': inactive_aux_asymmetry_rst,
        'inactive_aux_warmup_steps': inactive_aux_warmup_steps,
        'inactive_aux_lower_bound': inactive_aux_lower_bound,
        'inactive_aux_upper_bound': inactive_aux_upper_bound,
        'inactive_aux_bound_eps': inactive_aux_bound_eps,
        'inactive_aux_dev_mode': inactive_aux_dev_mode,
        'inactive_aux_ce_clip_std': inactive_aux_ce_clip_std,
        'inactive_aux_z_clip': inactive_aux_z_clip,
        'inactive_aux_z_tanh': inactive_aux_z_tanh,
        'inactive_aux_weighted_clip': inactive_aux_weighted_clip,
        'inactive_aux_normalize_by_layers': inactive_aux_normalize_by_layers,
        'inactive_aux_enabled': inactive_aux_enabled,
        'soft_gate_t_start': soft_gate_t_start,
        'soft_gate_t_final': soft_gate_t_final,
        'soft_gate_t_hold_frac': soft_gate_t_hold_frac,
        'soft_gate_t_anneal_end_frac': soft_gate_t_anneal_end_frac,
        'soft_gate_t_schedule': soft_gate_schedule,
        'soft_gate_t_power': soft_gate_t_power,
        'soft_gate_t_gompertz_center': soft_gate_t_gompertz_center,
        'soft_gate_t_gompertz_steepness': soft_gate_t_gompertz_steepness,
        **_flatten_soft_gate_pool_schedules(soft_gate_pool_schedules),
        'soft_gate_boundary_power_start': soft_gate_boundary_power_start,
        'soft_gate_boundary_power_mid': soft_gate_boundary_power_mid,
        'soft_gate_boundary_power_final': soft_gate_boundary_power_final,
        'soft_gate_boundary_power_start_frac': soft_gate_boundary_power_start_frac,
        'soft_gate_boundary_power_mid_frac': soft_gate_boundary_power_mid_frac,
        'soft_gate_boundary_power_final_frac': soft_gate_boundary_power_final_frac,
        'admission_den_power': admission_den_power,
        'admission_den_grad_scale': admission_den_grad_scale,
        'hardware_repack_enabled': hardware_repack_enabled,
        'hardware_sector_execution_enabled':
            hardware_sector_execution_enabled,
        'hardware_repack_interval_steps':
            hardware_repack_config['hardware_repack_interval_steps'],
        'hardware_repack_strategy':
            hardware_repack_config['hardware_repack_strategy'],
        'hardware_repack_farthest_per_sector':
            hardware_repack_config['hardware_repack_farthest_per_sector'],
        'hardware_repack_gain_eps':
            hardware_repack_config['hardware_repack_gain_eps'],
        'hardware_repack_max_move_frac':
            hardware_repack_config['hardware_repack_max_move_frac'],
        'hardware_repack_vq_iterations':
            hardware_repack_config['hardware_repack_vq_iterations'],
        'hardware_repack_warmup_steps':
            hardware_repack_config['hardware_repack_warmup_steps'],
        'hardware_repack_freeze_after_step':
            hardware_repack_config['hardware_repack_freeze_after_step'],
        'soft_gate_effective_active_eps': soft_gate_effective_active_eps,
        'regular_console_level': regular_console_level,
        'regular_console_host_timing': regular_console_host_timing,
        'regular_console_top1_warn': regular_console_top1_warn,
        'regular_console_drive_max_warn': regular_console_drive_max_warn,
        'regular_console_logging_overhead_warn':
            regular_console_logging_overhead_warn,
        'eval_effective_prune_enabled': eval_effective_prune_enabled,
        'eval_effective_prune_eps_list': eval_effective_prune_eps_list,
        'inactive_aux_start_frac': inactive_aux_start_frac,
        'inactive_aux_full_frac': inactive_aux_full_frac,
        'inactive_aux_schedule': inactive_aux_schedule,
        'cb1a_enabled': cb1a_enabled,
        'cb1a_qk_challenge_weight': cb1a_qk_challenge_weight,
        'cb1a_qk_prune_weight': cb1a_qk_prune_weight,
        'cb1a_v_challenge_weight': cb1a_v_challenge_weight,
        'cb1a_v_prune_weight': cb1a_v_prune_weight,
        'cb1a_rst_challenge_weight': cb1a_rst_challenge_weight,
        'cb1a_rst_prune_weight': cb1a_rst_prune_weight,
        'cb1a_tau_stopgrad': True,
        'cb1a_anchor_stopgrad': True,
        'cb1a_forward_influence': False,
        'cb1a_ce_mode': cb1a_ce_mode,
        'cb1a_eps': cb1a_eps,
        'dead_penalty_weighted_clip': dead_penalty_weighted_clip,
        'global_grad_clip': global_grad_clip,
        'tau_lr_mult': tau_lr_mult,
        'tau_grad_clip': tau_grad_clip,
        'router_proj_lr_mult': router_proj_lr_mult,
        'router_proj_grad_clip': router_proj_grad_clip,
        'router_scan_lr_mult': router_scan_lr_mult,
        'router_scan_grad_clip': router_scan_grad_clip,
        'route_emb_lr_mult': route_emb_lr_mult,
        'route_emb_grad_clip': route_emb_grad_clip,
        'op_key_lr_mult': op_key_lr_mult,
        'op_key_grad_clip': op_key_grad_clip,
        'enable_control_update_caps': enable_control_update_caps,
        'router_proj_update_ratio_cap': router_proj_update_ratio_cap,
        'route_emb_update_ratio_cap': route_emb_update_ratio_cap,
        'tau_update_abs_cap': tau_update_abs_cap,
        'scan_update_abs_cap': scan_update_abs_cap,
        'tau_init_attn_qk': tcfg.get(
            'tau_init_attn_qk',
            cfg['model'].get('tau_init_attn_qk', None)),
        'tau_init_attn_v': tcfg.get(
            'tau_init_attn_v',
            cfg['model'].get('tau_init_attn_v', None)),
        'tau_init_rst': tcfg.get(
            'tau_init_rst',
            cfg['model'].get('tau_init_rst', None)),
        'oom_check': run_oom_check,
        'speed_check': run_speed_check,
        'checkpoint_interval': ckpt_interval,
        'checkpoint_keep_last': checkpoint_keep_last,
        'best_checkpoint_keep_last': best_checkpoint_keep_last,
        'training_log_append_on_resume': training_log_append_on_resume,
        'log_interval': log_interval,
        'log_analysis_multiplier': log_analysis_multiplier,
        'heavy_geometry_multiplier': heavy_geometry_multiplier,
    }
    if _is_rw_key_srw_version(model_version_cfg):
        for _key in (
                'route_emb_lr_mult',
                'route_emb_grad_clip',
                'route_emb_update_ratio_cap'):
            training_config.pop(_key, None)
    if selection_calibration_cfg.get('present', False):
        training_config['selection_calibration'] = (
            selection_calibration_cfg.get('raw', {'enabled': False}))
    if isinstance(tcfg.get('operation_space'), dict):
        training_config['operation_space'] = deepcopy(tcfg['operation_space'])
    if operation_space_tau_free_enabled:
        _opspace_training_layouts = operation_space_repack_config.get(
            'operation_space_pool_layouts', {})
        if not isinstance(_opspace_training_layouts, dict):
            _opspace_training_layouts = {}
        _raw_opspace_training = (
            deepcopy(tcfg.get('operation_space', {}))
            if isinstance(tcfg.get('operation_space', {}), dict) else {})
        _raw_repack_training = (
            deepcopy(_raw_opspace_training.get('repack', {}))
            if isinstance(_raw_opspace_training.get('repack', {}), dict)
            else {})
        _repack_pool_cfg = operation_space_repack_config.get(
            'operation_space_repack_pools', {})
        if not isinstance(_repack_pool_cfg, dict):
            _repack_pool_cfg = {}

        def _opspace_materialized_pool(_label):
            _layout = _opspace_training_layouts.get(_label, {})
            if not isinstance(_layout, dict):
                _layout = {}
            _pool = {
                'execution_backend': str(_layout.get(
                    'execution_backend',
                    'sparse_region_block' if _label == 'rst'
                    else 'dense')).lower(),
                'num_regions': int(_layout.get('num_regions', 1)),
                'blocks_per_region': int(_layout.get(
                    'blocks_per_region', 1)),
                'operators_per_block': int(_layout.get(
                    'operators_per_block', 1)),
                'visible_regions': int(_layout.get('visible_regions', 1)),
                'visible_blocks_per_region': int(_layout.get(
                    'visible_blocks_per_region', 1)),
            }
            if _label == 'rst':
                _pool.update({
                    'region_score_pooling': str(_layout.get(
                        'region_score_pooling', 'smoothmax')).lower(),
                    'region_score_temperature': float(_layout.get(
                        'region_score_temperature', 0.25)),
                    'region_capacity_factor': float(_layout.get(
                        'region_capacity_factor', 1.25)),
                    'block_capacity_factor': float(_layout.get(
                        'block_capacity_factor', 1.25)),
                    'bucket_capacity_factor': float(_layout.get(
                        'bucket_capacity_factor',
                        _layout.get('block_capacity_factor', 1.25))),
                    'high_regret_threshold': float(_layout.get(
                        'high_regret_threshold', 0.05)),
                })
            return _pool

        def _opspace_materialized_max_swaps(_label):
            _pool = _repack_pool_cfg.get(_label, {})
            if not isinstance(_pool, dict):
                _pool = {}
            _raw_max_swaps = (
                _raw_repack_training.get('max_swaps', {})
                if isinstance(_raw_repack_training.get('max_swaps', {}), dict)
                else {})
            return int(_pool.get(
                'max_swaps_per_repack',
                _raw_max_swaps.get(_label, 256 if _label == 'rst' else 0)))

        _opspace_materialized_training = {
            'enabled': True,
            'pools': {
                'qk': _opspace_materialized_pool('qk'),
                'v': _opspace_materialized_pool('v'),
                'rst': _opspace_materialized_pool('rst'),
            },
        }
        if _raw_repack_training or operation_space_repack_enabled:
            _opspace_materialized_training['repack'] = {
                'enabled': bool(operation_space_repack_enabled),
                'start_step': int(operation_space_repack_config.get(
                    'operation_space_repack_start_step',
                    _raw_repack_training.get('start_step', 1000))),
                'interval_steps': int(operation_space_repack_config.get(
                    'operation_space_repack_interval_steps',
                    _raw_repack_training.get('interval_steps', 100))),
                'max_swaps': {
                    'qk': _opspace_materialized_max_swaps('qk'),
                    'v': _opspace_materialized_max_swaps('v'),
                    'rst': _opspace_materialized_max_swaps('rst'),
                },
            }
        training_config['operation_space'] = deepcopy(
            _opspace_materialized_training)
        cfg.setdefault('training', {})['operation_space'] = deepcopy(
            _opspace_materialized_training)

        def _opspace_backend_name(_label, _default):
            _layout = _opspace_training_layouts.get(_label, {})
            if not isinstance(_layout, dict):
                _layout = {}
            return str(_layout.get('execution_backend', _default)).lower()

        _inactive_tau_selection_keys = (
            'selection_calibration',
            'selection_calibration_applied',
            'selection_calibration_seen_tokens',
            'selection_calibration_actual_batches',
            'selection_calibration_histogram_bins',
            'selection_calibration_tau_qk',
            'selection_calibration_tau_v',
            'selection_calibration_tau_rst',
            'selection_calibration_B_start_qk',
            'selection_calibration_B_start_v',
            'selection_calibration_B_start_rst',
            'selection_calibration_B_final_qk',
            'selection_calibration_B_final_v',
            'selection_calibration_B_final_rst',
            'selection_calibration_formation_end_frac',
            'selection_calibration_sharpen_end_frac',
            'tau_init_mode',
            'tau_init_attn_qk',
            'tau_init_attn_v',
            'tau_init_rst',
            'tau_init_min',
            'tau_init_max',
            'tau_init_target_qk_frac',
            'tau_init_target_v_frac',
            'tau_init_target_rst_frac',
            'tau_init_calibration_tokens',
            'soft_gate_effective_active_eps',
            'admission_den_grad_scale',
            'soft_gate_boundary_power_start',
            'soft_gate_boundary_power_mid',
            'soft_gate_boundary_power_final',
            'soft_gate_boundary_power_start_frac',
            'soft_gate_boundary_power_mid_frac',
            'soft_gate_boundary_power_final_frac',
            'soft_gate_t_start',
            'soft_gate_t_final',
            'soft_gate_t_hold_frac',
            'soft_gate_t_anneal_end_frac',
            'soft_gate_t_schedule',
            'soft_gate_schedule',
            'soft_gate_t_power',
            'soft_gate_t_gompertz_center',
            'soft_gate_t_gompertz_steepness',
        )
        for _key in _inactive_tau_selection_keys:
            training_config.pop(_key, None)
            cfg.setdefault('training', {}).pop(_key, None)
        for _key in (
                'tau_init_attn_qk',
                'tau_init_attn_v',
                'tau_init_rst',
                'tau_init_min',
                'tau_init_max',
                'tau_init_mode',
                'tau_init_target_qk_frac',
                'tau_init_target_v_frac',
                'tau_init_target_rst_frac'):
            cfg.setdefault('model', {}).pop(_key, None)
        _operation_space_training_marks = {
            'operation_space_tau_free_active': True,
            'direct_tau_active_for_qk_v_rst': False,
            'selection_calibration_active_for_qk_v_rst': False,
            'tau_init_required_for_qk_v_rst': False,
            'qk_v_rst_gate_rule': 'relu2_operation_space',
            'qk_backend': _opspace_backend_name('qk', 'dense'),
            'v_backend': _opspace_backend_name('v', 'dense'),
            'rst_backend': _opspace_backend_name(
                'rst', 'sparse_region_block'),
            'tau_fields_compatibility_only': False,
        }
        training_config.update(_operation_space_training_marks)
        cfg.setdefault('training', {}).update(_operation_space_training_marks)
    if selection_calibration_resume_training_updates:
        training_config.update(selection_calibration_resume_training_updates)
        cfg.setdefault('training', {}).update(
            selection_calibration_resume_training_updates)
    if (_is_active_srw_version(model_version_cfg)
            and not operation_space_tau_free_enabled
            and tau_init_cfg is not None):
        training_config['tau_init_mode'] = tau_init_cfg['mode']
        if tau_init_cfg['mode'] == 'quantile_frac':
            training_config.update({
                'tau_init_target_qk_frac': tau_init_cfg['targets']['qk'],
                'tau_init_target_v_frac': tau_init_cfg['targets']['v'],
                'tau_init_target_rst_frac': tau_init_cfg['targets']['rst'],
                'tau_init_min': tau_init_cfg['tau_min'],
                'tau_init_max': tau_init_cfg['tau_max'],
                'tau_init_calibration_tokens':
                    tau_init_cfg['calibration_tokens'],
            })
        for _clean_key in list(training_config.keys()):
            if (_clean_key.startswith('cb1a_')
                    or _clean_key == 'inactive_aux_normalize_by_layers'):
                training_config.pop(_clean_key, None)
                cfg.setdefault('training', {}).pop(_clean_key, None)

    for _key in (
            'inactive_aux_weight_q',
            'inactive_aux_weight_k',
            'inactive_aux_asymmetry_q',
            'inactive_aux_asymmetry_k'):
        cfg.setdefault('training', {}).pop(_key, None)
    if not inactive_aux_enabled:
        for _key in (
                'inactive_aux_weight',
                'inactive_aux_weight_q',
                'inactive_aux_weight_k',
                'inactive_aux_weight_qk',
                'inactive_aux_weight_v',
                'inactive_aux_weight_rst',
                'inactive_aux_asymmetry',
                'inactive_aux_asymmetry_q',
                'inactive_aux_asymmetry_k',
                'inactive_aux_asymmetry_qk',
                'inactive_aux_asymmetry_v',
                'inactive_aux_asymmetry_rst',
                'inactive_aux_warmup_steps',
                'inactive_aux_lower_bound',
                'inactive_aux_upper_bound',
                'inactive_aux_bound_eps',
                'inactive_aux_dev_mode',
                'inactive_aux_ce_clip_std',
                'inactive_aux_z_clip',
                'inactive_aux_z_tanh',
                'inactive_aux_weighted_clip'):
            training_config.pop(_key, None)
            cfg.setdefault('training', {}).pop(_key, None)
    if is_v4164_cfg:
        for _key in (
                'pool_weight_decay',
                'diversity_weight',
                'load_balance_weight',
                'dead_penalty_weight',
                'dead_penalty_qk_weight',
                'dead_penalty_v_weight',
                'dead_penalty_rst_weight',
                'dead_exposure_target',
                'inactive_aux_enabled',
                'inactive_aux_start_frac',
                'inactive_aux_full_frac',
                'inactive_aux_schedule',
                                                'dead_penalty_weighted_clip'):
            training_config.pop(_key, None)
            cfg.setdefault('training', {}).pop(_key, None)
    cfg.setdefault('training', {}).update(training_config)

    # ----------------------------------------------------------
    # Detect devices (multi-host aware)
    # ----------------------------------------------------------
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    is_host0 = (host_id == 0)
    n_local_devices = jax.local_device_count()
    local_devices = jax.local_devices()

    # ALL hosts print device info (for multi-host diagnostics)
    print(f"[Host {host_id}/{n_hosts}] "
          f"local_devices={n_local_devices} total_devices={jax.device_count()} "
          f"backend={jax.default_backend()} "
          f"devices={[str(d) for d in local_devices]}", flush=True)

    startup_mesh_model = int(tcfg.get('mesh_model', 1))
    startup_mesh_data = int(tcfg.get('mesh_data', 0))
    if startup_mesh_data == 0:
        startup_mesh_data = jax.device_count() // startup_mesh_model
    _assert_multihost_same_startup_context({
        'trainer_script': 'scripts/train_jax.py',
        'config_path': str(config_path),
        'model_version': model_version_cfg,
        'checkpoint_dir': checkpoint_dir,
        'resume_step': resume_step,
        'from_scratch': bool(cli_args.from_scratch),
        'process_count': n_hosts,
        'mesh_data': startup_mesh_data,
        'mesh_model': startup_mesh_model,
        'batch_size': batch_size,
        'train_script_git_commit': train_script_git_commit,
        'hostname': socket.gethostname(),
        'host_id': host_id,
        'process_index': host_id,
    })

    per_host_batch = batch_size // n_hosts
    per_device_batch = per_host_batch // n_local_devices

    assert batch_size % n_hosts == 0, (
        f"Global batch_size ({batch_size}) must be divisible by n_hosts ({n_hosts})"
    )
    assert per_host_batch % n_local_devices == 0, (
        f"per_host_batch ({per_host_batch}) must be divisible by "
        f"n_local_devices ({n_local_devices})"
    )

    if is_host0:
        print(f"\n{'='*60}")
        print(f"DAWN Training (Multi-Host Multi-Device) -- {cfg['model'].get('model_version', 'unknown')}")
        print(f"{'='*60}")
        print(f"JAX version: {jax.__version__}")
        print(f"Hosts: {n_hosts}, Host ID: {host_id}")
        print(f"Local devices: {local_devices}")
        print(f"Local device count: {n_local_devices}")
        print(f"Total device count: {jax.device_count()}")
        print(f"Backend: {jax.default_backend()}")
        print(f"Config: {config_path}")
        if resume_step is None:
            print("Config source: current YAML")
            if _is_active_srw_version(model_version_cfg):
                print(
                    "Selection calibration policy: fresh init computes; "
                    "resume restores from checkpoint.")
            elif is_baseline:
                print("Selection calibration/tau init: disabled for baseline.")
        print(f"Run folder: {checkpoint_dir}")
        print(f"Latest checkpoint dir: {_join(checkpoint_dir, 'checkpoints')}")
        print(
            f"Best checkpoint dir: "
            f"{_join(checkpoint_dir, 'best_checkpoints')}")
        print(
            f"Latest checkpoint keep_last={checkpoint_keep_last}, "
            f"interval={ckpt_interval}")
        print(f"Best checkpoint keep_last={best_checkpoint_keep_last}")
        print(f"Seed: {seed}")
        print(f"Global batch size: {batch_size}")
        print(f"Per-host batch size: {per_host_batch}")
        print(f"Per-device batch size: {per_device_batch}")

    # ----------------------------------------------------------
    # Load data (multi-host: each host loads its own data slice)
    # ----------------------------------------------------------
    if is_host0:
        print(f"\n{'='*60}")
        print("Loading data...")
        print(f"{'='*60}")

    from utils.data_jax import load_data
    train_loader, val_loader, vocab_size = load_data(
        cfg['data'],
        max_length=max_seq_len,
        batch_size=batch_size,
        n_devices=1,  # flat (per_host_batch, seq_len) -shard_to_mesh handles splitting
        n_hosts=n_hosts,
        host_id=host_id,
    )
    if is_host0:
        print(f"Vocab size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------
    cfg['model']['vocab_size'] = vocab_size
    _maybe_materialize_vocab_parallel_config(cfg)
    model = build_model_from_config(cfg)

    # Initialize
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones((1, max_seq_len), dtype=jnp.int32)

    if is_host0:
        print("=== Starting model.init ===", flush=True)
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        dummy_input,
        deterministic=True,
    )
    params = variables['params']

    if is_host0:
        print("=== model.init done ===", flush=True)
        n_params = count_parameters(params)
        print(f"\nModel parameters: {n_params:,}")
        for line in model.get_model_info():
            print(line)
        if str(model_version_cfg) == V4167_MODEL_VERSION:
            m = cfg['model']
            stage_count = int(m['n_layers']) // int(m['layers_per_stage'])
            router_count = int(m['n_layers'])
            d_model_cfg = float(m['d_model'])
            print("v4167 GSL pool sanity:")
            print(
                f"  total pool counts: qk={m['n_qk']} "
                f"v={m['n_v']} rst={m['n_rst']}")
            print(
                "  global/stage/local counts: "
                f"qk={m['n_qk_global']}/{m['n_qk_stage']}/"
                f"{m['n_qk_local']} "
                f"v={m['n_v_global']}/{m['n_v_stage']}/"
                f"{m['n_v_local']} "
                f"rst={m['n_rst_global']}/{m['n_rst_stage']}/"
                f"{m['n_rst_local']}")
            print(
                f"  visible counts per layer: qk={m['qk_visible_n']} "
                f"v={m['v_visible_n']} rst={m['rst_visible_n']}")
            print(
                f"  layers_per_stage={m['layers_per_stage']} "
                f"stage_count={stage_count} "
                f"router_scope={m['router_scope']} "
                f"router_count={router_count}")
            print(f"  estimated parameter count: {n_params:,}")
            print(
                "  ratio QK/V/RST (total/d_model): "
                f"qk={m['n_qk'] / d_model_cfg:.3f} "
                f"v={m['n_v'] / d_model_cfg:.3f} "
                f"rst={m['n_rst'] / d_model_cfg:.3f}")
            print(f"  fixed_tau={m['fixed_tau']}")

    _has_resume_checkpoint = resume_step is not None

    rank = cfg['model'].get('rank', 64)
    knowledge_rank = cfg['model'].get('knowledge_rank', 128)

    # ----------------------------------------------------------
    # Optimizer (warmup + cosine decay + optional gradient accumulation)
    # ----------------------------------------------------------
    grad_accum_steps = tcfg.get('gradient_accumulation_steps', 1)

    steps_per_epoch = len(train_loader)
    # Schedule counts optimizer steps (after accumulation), not micro-steps
    effective_steps_per_epoch = steps_per_epoch // grad_accum_steps
    total_steps = num_epochs * effective_steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )

    # v4164 per-group WD: pool tensors (attn-qk/attn-v/RST emb/read/write) get
    # pool_weight_decay; dense kernels get weight_decay. Bias / LayerNorm /
    # learnable *_scale excluded from both groups.
    #
    # optax.adamw is chain(scale_by_adam, add_decayed_weights, scale_by_lr).
    # To apply two different WDs we decompose it: one scale_by_adam, then
    # two masked add_decayed_weights (base + pool -masks are disjoint so
    # each param is touched at most once), then a single scale_by_lr.

    _MODEL_VERSION = OFFICIAL_MODEL_VERSION
    _FORWARD_UNIT_RW_VERSIONS = set()

    _POOL_PARAM_NAMES = (
        'attn_qk_emb', 'attn_v_emb', 'rst_emb',
        'attn_qk_op_read_proj', 'attn_qk_op_write_proj',
        'attn_v_op_read_proj', 'attn_v_op_write_proj',
        'rst_op_read_proj', 'rst_op_write_proj',
        'qk_emb', 'v_emb', 'rst_emb',
        'q_read', 'k_read',
        'attn_qk_read', 'attn_v_read', 'rst_read',
        'qk_read', 'v_read', 'rst_read',
        'q_write', 'k_write',
        'attn_qk_write', 'attn_v_write', 'rst_write',
        'qk_write', 'v_write', 'rst_write',
    )
    _RW_PARAM_NAMES = (
        'q_read', 'k_read',
        'attn_qk_read', 'attn_v_read', 'rst_read',
        'qk_read', 'v_read', 'rst_read',
        'q_write', 'k_write',
        'attn_qk_write', 'attn_v_write', 'rst_write',
        'qk_write', 'v_write', 'rst_write',
    )

    def _path_str(path):
        return '/'.join(str(p.key if hasattr(p, 'key') else p) for p in path)

    def _is_pool_param(path_str):
        return any(name in path_str for name in _POOL_PARAM_NAMES)

    def _is_rw_param(path_str):
        return any(name in path_str for name in _RW_PARAM_NAMES)

    def _is_excluded(path_str):
        leaf = path_str.rsplit('/', 1)[-1]
        if leaf == 'bias':
            return True
        if 'scale' in path_str and 'norm' in path_str.lower():
            return True  # LayerNorm scale
        if path_str.endswith('_scale') or path_str.endswith('/qk_scale') \
           or path_str.endswith('/v_scale') or path_str.endswith('/rst_scale') \
           or path_str.endswith('/attn_qk_scale') \
           or path_str.endswith('/attn_v_scale') \
           or path_str.endswith('/rst_scale'):
            return True  # learnable output_scale
        if _MODEL_VERSION in _FORWARD_UNIT_RW_VERSIONS and _is_rw_param(path_str):
            return True  # forward-normalized read/write directions
        return False

    def _wd_mask_base(params):
        def _f(path, _):
            ps = _path_str(path)
            if _is_excluded(ps):
                return False
            return not _is_pool_param(ps)
        return jax.tree.map_with_path(_f, params)

    def _wd_mask_pool(params):
        def _f(path, _):
            ps = _path_str(path)
            if _is_excluded(ps):
                return False
            return _is_pool_param(ps)
        return jax.tree.map_with_path(_f, params)

    def _no_param_mask(params):
        return jax.tree.map(lambda _: False, params)

    optimizer_parts = [
        optax.masked(optax.set_to_zero(), mask=_no_param_mask),
    ]
    if float(global_grad_clip) > 0.0:
        optimizer_parts.append(optax.clip_by_global_norm(global_grad_clip))
    else:
        # Keep the global-clip opt_state slot for checkpoint resume stability,
        # but make it an exact no-op unless config explicitly enables clipping.
        optimizer_parts.append(optax.scale(1.0))
    optimizer_parts.extend([
        optax.scale_by_adam(b2=0.95),
        optax.add_decayed_weights(weight_decay, mask=_wd_mask_base),
        optax.add_decayed_weights(pool_weight_decay, mask=_wd_mask_pool),
        optax.scale_by_learning_rate(schedule),
        optax.masked(optax.set_to_zero(), mask=_no_param_mask),
    ])
    base_optimizer = optax.chain(*optimizer_parts)

    if is_host0:
        def _count_true(mask):
            n = [0]
            def _f(v):
                if v:
                    n[0] += 1
                return v
            jax.tree.map(_f, mask)
            return n[0]
        def _collect_pool_paths(mask):
            out = []
            def _f(path, v):
                if v:
                    out.append(_path_str(path))
                return v
            jax.tree.map_with_path(_f, mask)
            return out
        _base_mask = _wd_mask_base(params)
        _pool_mask = _wd_mask_pool(params)
        print(f"  WD groups: base ({weight_decay}) = {_count_true(_base_mask)} tensors, "
              f"pool ({pool_weight_decay}) = {_count_true(_pool_mask)} tensors")
        _pool_paths = _collect_pool_paths(_pool_mask)
        if _pool_paths:
            print(f"    pool params: {_pool_paths[:9]}")
    if grad_accum_steps > 1:
        optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=grad_accum_steps)
    else:
        optimizer = base_optimizer

    if is_host0:
        print(f"\nTraining config:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Global batch size: {batch_size}")
        print(f"  Per-host batch size: {per_host_batch}")
        print(f"  Per-device batch size: {per_device_batch}")
        print(f"  Hosts: {n_hosts}")
        print(f"  Local devices: {n_local_devices}")
        print(f"  Total devices: {jax.device_count()}")
        print(f"  Grad accum steps: {grad_accum_steps}")
        print(f"  Effective batch size: {batch_size * grad_accum_steps}")
        print("  Startup checks: "
              f"oom={'on' if run_oom_check else 'off'}, "
              f"speed={'on' if run_speed_check else 'off'}")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total optimizer steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  LR: {lr}")
        _inactive_aux_stabilizer_part = (
            f"expl_clip={inactive_aux_weighted_clip}, "
            f"inactive_aux_dev_mode={inactive_aux_dev_mode}, "
            if inactive_aux_enabled else "")
        _route_or_op_key_part = (
            f"op_key_lr_mult={op_key_lr_mult}, "
            if _is_rw_key_srw_version(model_version_cfg)
            else (
                f"route_emb_lr_mult={route_emb_lr_mult}, "
                f"op_key_lr_mult={op_key_lr_mult}, "))
        print("  Stabilizers: "
              f"global_grad_clip={global_grad_clip}, "
              f"tau_lr_mult={tau_lr_mult}, tau_grad_clip={tau_grad_clip}, "
              f"router_proj_lr_mult={router_proj_lr_mult}, "
              f"router_scan_lr_mult={router_scan_lr_mult}, "
              f"{_route_or_op_key_part}"
              f"dead_clip={dead_penalty_weighted_clip}, "
              f"{_inactive_aux_stabilizer_part}"
              "checkpoint_backend=orbax")
        _route_or_op_key_cap_part = (
            f"op_key_ratio={route_emb_update_ratio_cap}, "
            if _is_rw_key_srw_version(model_version_cfg)
            else f"emb_ratio={route_emb_update_ratio_cap}, ")
        print("  Control update caps: "
              f"enabled={enable_control_update_caps}, "
              f"proj_ratio={router_proj_update_ratio_cap}, "
              f"{_route_or_op_key_cap_part}"
              f"tau_abs={tau_update_abs_cap}, scan_abs={scan_update_abs_cap}")
        print(f"  Weight decay: {weight_decay} (pool: {pool_weight_decay})")
        print(f"  Orth weight: {orth_weight}")
        print(f"  Div weight: {div_weight}")
        print(f"  LB weight: {lb_weight}")
        print(f"  Tau reg weight: {tau_reg_weight}")
        print(f"  Dead penalty: global={dead_penalty_weight} "
              f"exposure_target={dead_exposure_target} "
              f"direct_w[qk={dead_penalty_qk_weight}, v={dead_penalty_v_weight}, "
              f"rst={dead_penalty_rst_weight}]")
        if inactive_aux_enabled:
            print(f"  Inactive auxiliary weight: {inactive_aux_weight}")
            print("    weight split: "
                  f"qk={inactive_aux_weight_qk} "
                  f"v={inactive_aux_weight_v} rst={inactive_aux_weight_rst}")
            print("    asymmetry split: "
                  f"qk={inactive_aux_asymmetry_qk} "
                  f"v={inactive_aux_asymmetry_v} rst={inactive_aux_asymmetry_rst}")
            print("    normalization: "
                  f"by_layers={inactive_aux_normalize_by_layers}")
            print(f"    warmup_steps={inactive_aux_warmup_steps} "
                  f"bounds=[{inactive_aux_lower_bound}, {inactive_aux_upper_bound}] "
                  f"eps={inactive_aux_bound_eps}")
        print(f"  Dropout: residual={cfg['model'].get('dropout', 0.0)} "
              f"router={cfg['model'].get('router_dropout', 0.0)}")
        print(f"  Module path: {_model_registry_entry(model_version_cfg)['module']}")
        if _is_rw_key_srw_version(model_version_cfg):
            print("  RW-key operator path: live-gradient RW keys, "
                  "RW-matched operator queries")
        if operation_space_tau_free_enabled:
            _opspace_layouts = operation_space_repack_config.get(
                'operation_space_pool_layouts', {})
            if not isinstance(_opspace_layouts, dict):
                _opspace_layouts = {}

            def _opspace_startup_layout_line(_label):
                _layout = _opspace_layouts.get(_label, {})
                if not isinstance(_layout, dict):
                    _layout = {}
                _backend = str(_layout.get(
                    'execution_backend', 'unknown')).lower()
                _visible_ops = (
                    int(_layout.get('visible_regions', 0))
                    * int(_layout.get('visible_blocks_per_region', 0))
                    * int(_layout.get('operators_per_block', 0)))
                _line = (
                    f"[opspace/{_label}] "
                    f"backend={_backend} "
                    f"num_regions={int(_layout.get('num_regions', 0))} "
                    f"blocks_per_region="
                    f"{int(_layout.get('blocks_per_region', 0))} "
                    f"operators_per_block="
                    f"{int(_layout.get('operators_per_block', 0))} "
                    f"visible_regions="
                    f"{int(_layout.get('visible_regions', 0))} "
                    f"visible_blocks_per_region="
                    f"{int(_layout.get('visible_blocks_per_region', 0))} "
                    f"visible_ops={_visible_ops}")
                if _label == 'rst':
                    _line += (
                        f" region_capacity_factor="
                        f"{float(_layout.get('region_capacity_factor', 1.25))} "
                        f"block_capacity_factor="
                        f"{float(_layout.get('block_capacity_factor', 1.25))} "
                        f"high_regret_threshold="
                        f"{float(_layout.get('high_regret_threshold', 0.05))} "
                        f"region_score_pooling="
                        f"{_layout.get('region_score_pooling', 'smoothmax')}")
                return _line

            print("[opspace] v4168 unified region/block/operator atlas active")
            print("[opspace] DirectTau/admission/drive/selection_calibration disabled for QK/V/RST outputs")
            print("[opspace] tau_init config not required in tau-free operation-space mode")
            print("[opspace] denominator = max(sum(relu2_gate), 1.0) ** admission_den_power")
            print(_opspace_startup_layout_line('qk'))
            print(_opspace_startup_layout_line('v'))
            print(_opspace_startup_layout_line('rst'))
            print(
                f"  Gate ({cfg['model'].get('model_version')} operation-space tau-free): "
                "direct_tau_active_for_qk_v_rst=false "
                "selection_calibration_active_for_qk_v_rst=false "
                "tau_init_required_for_qk_v_rst=false "
                f"dropout={cfg['model'].get('dropout', None)} "
                f"router_dropout={cfg['model'].get('router_dropout', None)}")
            print("  Effective pruning: disabled for operation-space QK/V/RST")
        else:
            print("  Tau parameterization: bounded sigmoid min/max")
            print("  tau = -1 + 2 * sigmoid(raw_tau)")
            print("  Boundary admission: one-sided generalized Gaussian")
            print("  drive = softplus((rho-tau)/B) / softplus((1-tau)/B)")
            print("  execution_weight = admission * drive")
            print("  admission_den = max(sum(admission), 1.0) ** admission_den_power")
            print(
                "  admission_den_grad = admission_den_grad_scale * live_admission_den_grad "
                "+ detached remainder")
            print("  Boundary power:")
            print(
                f"    start={soft_gate_boundary_power_start} "
                f"mid={soft_gate_boundary_power_mid} "
                f"final={soft_gate_boundary_power_final} "
                f"start_frac={soft_gate_boundary_power_start_frac} "
                f"mid_frac={soft_gate_boundary_power_mid_frac} "
                f"final_frac={soft_gate_boundary_power_final_frac}")
            print("  Admission denominator:")
            print(
                f"    admission_den_power={admission_den_power} "
                f"admission_den_grad_scale={admission_den_grad_scale}")
            print(f"    pool_specific={pool_specific_gate_t} "
                  f"effective_active_eps={soft_gate_effective_active_eps}")
            _scale_label = 'B'
            def _devband_summary(_cfg):
                return (
                    f"sort={_cfg['sort']} band={_cfg['band']} "
                    f"mid={_cfg['mid']} late={_cfg['late']} "
                    f"final={_cfg['final']} "
                    f"sort_end_frac={_cfg['sort_end_frac']} "
                    f"band_reach_frac={_cfg['band_reach_frac']} "
                    f"formation_end_frac={_cfg['formation_end_frac']} "
                    f"sharpen_end_frac={_cfg['sharpen_end_frac']} "
                    f"formation_power={_cfg['formation_power']} "
                    f"sharpen_power={_cfg['sharpen_power']}")
            if pool_specific_gate_t:
                for _pool in POOL_SCHEDULE_NAMES:
                    _cfg = soft_gate_pool_schedules[_pool]
                    _schedule_name = str(_cfg['schedule']).lower()
                    if _schedule_name == 'developmental_band':
                        print(
                            f"    {_pool}: schedule={_cfg['schedule']} "
                            f"{_devband_summary(_cfg)}")
                    else:
                        _shape_msg = (
                            f"gompertz_center={_cfg['gompertz_center']} "
                            f"gompertz_steepness={_cfg['gompertz_steepness']} "
                            if _schedule_name == 'log_gompertz'
                            else f"power={_cfg['power']} ")
                        print(
                            f"    {_pool}: {_scale_label}_start={_cfg['start']} "
                            f"{_scale_label}_final={_cfg['final']} "
                            f"hold_frac={_cfg['hold_frac']} "
                            f"anneal_end_frac={_cfg['anneal_end_frac']} "
                            f"schedule={_cfg['schedule']} {_shape_msg}")
            else:
                if soft_gate_schedule.lower() == 'developmental_band':
                    _cfg = soft_gate_pool_schedules['qk']
                    print(f"    schedule={soft_gate_schedule} "
                          f"{_devband_summary(_cfg)}")
                else:
                    _soft_gate_shape_msg = (
                        f"gompertz_center={soft_gate_t_gompertz_center} "
                        f"gompertz_steepness={soft_gate_t_gompertz_steepness} "
                        if soft_gate_schedule.lower() == 'log_gompertz'
                        else f"power={soft_gate_t_power} ")
                    print(f"    {_scale_label}_start={soft_gate_t_start} "
                          f"{_scale_label}_final={soft_gate_t_final} "
                          f"hold_frac={soft_gate_t_hold_frac} "
                          f"anneal_end_frac={soft_gate_t_anneal_end_frac} "
                          f"schedule={soft_gate_schedule} "
                          f"{_soft_gate_shape_msg}")
            print(f"  tau control: tau_lr_mult={tau_lr_mult}")
            if ignored_tau_ce_grad_scale_keys:
                print("  tau_ce_grad_scale config fields are ignored in "
                      "v4164; tau movement is controlled by tau_lr_mult.")
            print("  Effective pruning:")
            print(
                f"    console={regular_console_level} "
                f"host_timing={regular_console_host_timing}")
            print(f"    eval enabled={eval_effective_prune_enabled} eps={eval_effective_prune_eps_list}")
            if _is_active_srw_version(model_version_cfg):
                _gate_intensity_part = ""
                gate_msg = (
                    f"  Gate ({cfg['model'].get('model_version')} soft-annealed-direct-tau): "
                    f"tau_init_mode={tau_init_cfg['mode']} "
                    f"tau_init_attn_qk={tcfg.get('tau_init_attn_qk', cfg['model'].get('tau_init_attn_qk', None))} "
                    f"tau_init_attn_v={tcfg.get('tau_init_attn_v', cfg['model'].get('tau_init_attn_v', None))} "
                    f"tau_init_rst={tcfg.get('tau_init_rst', cfg['model'].get('tau_init_rst', None))} "
                    f"{_gate_intensity_part}"
                    f"dropout={cfg['model'].get('dropout', None)} "
                    f"router_dropout={cfg['model'].get('router_dropout', None)}"
                )
                print(gate_msg)
            elif is_baseline:
                print("  Baseline loss: CE only; SRW tau/selection disabled.")

    # ----------------------------------------------------------
    # Resume defaults. Orbax state restore runs after mesh sharding so
    # params/opt_state restore directly into their target shardings.
    # ----------------------------------------------------------
    start_epoch = 0
    global_step = 0
    start_step_in_epoch = 0
    best_val_loss = float('inf')
    _has_resume_checkpoint = resume_step is not None
    orbax_root = _join_path(checkpoint_dir, "checkpoints")

    if _has_resume_checkpoint:
        if is_host0:
            print(f"\nResuming from run folder: {resume_path}")
            print(f"  Orbax step: {resume_step}")
    else:
        if is_host0:
            if not cli_args.from_scratch:
                print("\nNo checkpoint found. Starting from scratch.")
            else:
                print("\nStarting from scratch (--from-scratch).")

    tau_init_summary = None
    selection_calibration_summary = None
    selection_calibration_tau_applied = False
    if (selection_calibration_cfg.get('enabled', False)
            and operation_space_tau_free_enabled
            and not _has_resume_checkpoint):
        if is_host0:
            print(
                "[opspace] selection_calibration present in config but "
                "skipped: tau-free QK/V/RST do not use tau/admission.",
                flush=True)
    elif (selection_calibration_cfg.get('enabled', False)
            and not _has_resume_checkpoint):
        if is_host0:
            print(
                "Selection calibration policy: fresh init computes; "
                "resume restores from checkpoint.",
                flush=True)
            print(
                "Selection calibration: computing from fresh "
                "initialization...",
                flush=True)
        if len(train_loader) <= 0:
            raise ValueError(
                "selection_calibration requires at least one training batch.")
        (
            local_histograms,
            local_page_stats,
            local_seen_tokens,
            local_actual_batches,
            local_calibration_tokens,
            calibration_process_count,
        ) = _collect_selection_calibration_histograms(
            params, train_loader, cfg, selection_calibration_cfg)
        (
            calibration_histograms,
            calibration_page_stats,
            seen_tokens,
            actual_calibration_batches,
        ) = _aggregate_selection_calibration_histograms(
            local_histograms, local_page_stats,
            local_seen_tokens, local_actual_batches)
        _selection_summary_json = None
        if is_host0:
            selection_calibration_summary = (
                _compute_srw_selection_calibration(
                    calibration_histograms, cfg, selection_calibration_cfg,
                    calibration_page_stats,
                    seen_tokens, actual_calibration_batches,
                    local_calibration_tokens, calibration_process_count))
            _selection_summary_json = json.dumps(selection_calibration_summary)
        _selection_summary_json = _broadcast_str_from_host0(
            _selection_summary_json, max_len=32768)
        if not _selection_summary_json:
            raise RuntimeError(
                "Failed to broadcast selection calibration summary.")
        selection_calibration_summary = json.loads(_selection_summary_json)
        params = _set_srw_quantile_tau_biases(
            params, selection_calibration_summary, model_version_cfg)
        fixed_tau_materialized = _materialize_fixed_tau_config(
            cfg, training_config, selection_calibration_summary,
            model_version_cfg)
        if fixed_tau_materialized:
            model = build_model_from_config(cfg)
            if is_host0:
                print(
                    "v4167 fixed tau materialized from selection calibration: "
                    f"qk={cfg['model']['tau_init_attn_qk']:.6f} "
                    f"v={cfg['model']['tau_init_attn_v']:.6f} "
                    f"rst={cfg['model']['tau_init_rst']:.6f}",
                    flush=True)
        selection_calibration_tau_applied = True
        materialized_training_updates = (
            _selection_calibration_materialized_training_updates(
                selection_calibration_summary, selection_calibration_cfg))
        soft_gate_pool_schedules = (
            _apply_selection_calibrated_soft_gate_schedules(
                soft_gate_pool_schedules, materialized_training_updates))
        soft_gate_boundary_power_start = materialized_training_updates[
            'soft_gate_boundary_power_start']
        soft_gate_boundary_power_mid = materialized_training_updates[
            'soft_gate_boundary_power_mid']
        soft_gate_boundary_power_final = materialized_training_updates[
            'soft_gate_boundary_power_final']
        soft_gate_boundary_power_start_frac = materialized_training_updates[
            'soft_gate_boundary_power_start_frac']
        soft_gate_boundary_power_mid_frac = materialized_training_updates[
            'soft_gate_boundary_power_mid_frac']
        soft_gate_boundary_power_final_frac = materialized_training_updates[
            'soft_gate_boundary_power_final_frac']
        calibrated_flat_schedules = _flatten_soft_gate_pool_schedules(
            soft_gate_pool_schedules)
        training_config.update(calibrated_flat_schedules)
        training_config.update(materialized_training_updates)
        training_config.update(
            _selection_calibration_checkpoint_updates(
                selection_calibration_summary))
        cfg.setdefault('training', {}).update(calibrated_flat_schedules)
        cfg.setdefault('training', {}).update(materialized_training_updates)
        cfg.setdefault('training', {}).update(
            _selection_calibration_checkpoint_updates(
                selection_calibration_summary))
        if is_host0:
            print("\n=== Selection calibration ===", flush=True)
            for _line in _selection_calibration_summary_lines(
                    selection_calibration_summary):
                print(_line, flush=True)
    elif (selection_calibration_cfg.get('enabled', False)
          and _has_resume_checkpoint
          and is_host0):
        if not selection_calibration_resume_restored:
            print(
                "\nSelection calibration policy: resume restores from "
                "checkpoint; recomputation disabled.",
                flush=True)

    if (_is_active_srw_version(model_version_cfg)
            and not operation_space_tau_free_enabled
            and tau_init_cfg is not None
            and tau_init_cfg['mode'] == 'quantile_frac'
            and not _has_resume_checkpoint
            and not selection_calibration_tau_applied):
        if len(train_loader) <= 0:
            raise ValueError(
                "tau_init_mode=quantile_frac requires at least one "
                "training batch for calibration.")
        _tau_init_summary_json = None
        if is_host0:
            calibration_input_ids, _ = next(iter(train_loader))
            tau_init_summary = _compute_srw_quantile_tau_init(
                params, calibration_input_ids, cfg, tau_init_cfg)
            _tau_init_summary_json = json.dumps(tau_init_summary)
        _tau_init_summary_json = _broadcast_str_from_host0(
            _tau_init_summary_json, max_len=16384)
        if not _tau_init_summary_json:
            raise RuntimeError(
                "Failed to broadcast quantile tau initialization summary.")
        tau_init_summary = json.loads(_tau_init_summary_json)
        params = _set_srw_quantile_tau_biases(
            params, tau_init_summary, model_version_cfg)
        fixed_tau_materialized = _materialize_fixed_tau_config(
            cfg, training_config, tau_init_summary, model_version_cfg)
        if fixed_tau_materialized:
            model = build_model_from_config(cfg)
            if is_host0:
                print(
                    "v4167 fixed tau materialized from quantile init: "
                    f"qk={cfg['model']['tau_init_attn_qk']:.6f} "
                    f"v={cfg['model']['tau_init_attn_v']:.6f} "
                    f"rst={cfg['model']['tau_init_rst']:.6f}",
                    flush=True)
        if is_host0:
            print("\n=== Quantile tau initialization ===", flush=True)
            for _line in _v4164_tau_init_summary_lines(tau_init_summary):
                print(_line, flush=True)

    raw_config_snapshot = _safe_config_snapshot(raw_cfg_snapshot)
    if _has_resume_checkpoint:
        full_config_snapshot = _safe_config_snapshot(saved_full_config)
        current_materialized_config_snapshot = _materialized_config_snapshot(
            current_yaml_config_snapshot,
            (current_yaml_config_snapshot.get('training', {})
             if isinstance(current_yaml_config_snapshot, dict)
             else {}))
    else:
        full_config_snapshot = _materialized_config_snapshot(
            cfg, training_config)
        current_materialized_config_snapshot = full_config_snapshot
    selected_full_config_sha256 = _config_sha256(full_config_snapshot)
    current_materialized_config_sha256 = _config_sha256(
        current_materialized_config_snapshot)
    checkpoint_full_config_sha256 = (
        _config_sha256(saved_full_config)
        if _has_resume_checkpoint and saved_full_config is not None
        else None)

    # Save config snapshots for this run (host 0 only)
    if is_host0:
        if _has_resume_checkpoint:
            print(
                "Config hash: "
                f"selected_full_config_sha256={selected_full_config_sha256} "
                f"checkpoint_full_config_sha256={checkpoint_full_config_sha256} "
                "current_materialized_config_sha256="
                f"{current_materialized_config_sha256}",
                flush=True,
            )
        else:
            print(
                "Config hash: "
                f"selected_full_config_sha256={selected_full_config_sha256}",
                flush=True,
            )
        if _has_resume_checkpoint:
            try:
                session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                session_path = _join(
                    checkpoint_dir,
                    f"config_resume_session_{session_ts}.json")
                session_cfg = {
                    'type': 'resume_session_config',
                    'timestamp': datetime.now().isoformat(),
                    'config_path': str(config_path),
                    'resume_path': resume_path,
                    'resume_step': int(resume_step),
                    'config_json_read_only':
                        bool(resume_config_restore_read_only),
                    'resume_config_source': resume_config_source,
                    'resume_config_fallback': 'disabled',
                    'checkpoint_raw_config_present':
                        saved_raw_config is not None,
                    'current_materialized_config_sha256':
                        current_materialized_config_sha256,
                    'checkpoint_full_config_sha256':
                        checkpoint_full_config_sha256,
                    'selected_full_config_sha256':
                        selected_full_config_sha256,
                    'current_yaml_config':
                        _safe_config_snapshot(current_yaml_config_snapshot),
                    'checkpoint_full_config':
                        _safe_config_snapshot(saved_full_config),
                    'selected_full_config': full_config_snapshot,
                    'current_raw_config': raw_config_snapshot,
                    'current_materialized_config':
                        current_materialized_config_snapshot,
                }
                _write_json_file(session_path, session_cfg)
                print(f"  Saved resume session config snapshot: {session_path}")
            except Exception as e:
                print(
                    f"  Warning: Failed to save resume session config: {e}")

        if resume_config_restore_read_only:
            if not resume_readonly_log_printed:
                print(
                    "Resume detected; preserving existing config.json and "
                    "config_raw.json snapshots.")
            print("  Skipped run-folder config snapshot rewrite on resume.")
        else:
            try:
                cj_path = _join(checkpoint_dir, 'config.json')
                config_record_snapshot = deepcopy(full_config_snapshot)
                config_record_snapshot['_raw_config'] = raw_config_snapshot
                config_record_snapshot['_metadata'] = {
                    'type': 'fresh_run_config_snapshot',
                    'timestamp': datetime.now().isoformat(),
                    'config_path': str(config_path),
                    'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'current_materialized_config_sha256':
                        current_materialized_config_sha256,
                    'selected_full_config_sha256':
                        selected_full_config_sha256,
                }
                _write_json_file(cj_path, config_record_snapshot)
                print(f"  Saved config.json: {cj_path}")
                print(
                    "  Top-level config keys saved: "
                    + ", ".join(sorted(config_record_snapshot.keys())))
            except Exception as e:
                print(f"  Warning: Failed to save config snapshot: {e}")

    # ----------------------------------------------------------
    # Create Mesh + shard params
    # ----------------------------------------------------------
    n_feature_qk = cfg['model'].get('n_feature_qk', 56)
    n_restore_qk = cfg['model'].get('n_restore_qk', 56)
    model_version = cfg['model'].get('model_version', OFFICIAL_MODEL_VERSION)

    mesh_model = cfg['training'].get('mesh_model', 1)
    mesh_data = cfg['training'].get('mesh_data', 0)  # 0 = auto
    total_devices = jax.device_count()
    if mesh_data == 0:
        mesh_data = total_devices // mesh_model

    mesh = create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    per_device_batch = batch_size // mesh_data

    # Auto n_chunks: target ~2GB per chunk (bf16)
    def auto_n_chunks(N, target_gb=2.0):
        full_gb = per_device_batch * max_seq_len * N * 2 / 1e9  # bf16
        nc = max(1, int(np.ceil(full_gb / target_gb)))
        while N % nc != 0 and nc < N:
            nc += 1
        return min(nc, N)

    def _chunk_size_from_count(name, n_local, n_chunks):
        n_chunks = int(n_chunks)
        if n_chunks < 1:
            raise ValueError(f"{name} chunks must be >= 1, got {n_chunks}")
        if n_chunks > n_local:
            raise ValueError(
                f"{name} chunks={n_chunks} exceeds local pool size {n_local}")
        return max(1, int(np.ceil(n_local / n_chunks)))

    if _is_active_srw_version(model_version_cfg):
        target_chunk_gb = cfg['training'].get('target_chunk_gb', 2.0)
        n_rst = cfg['model'].get('n_rst', cfg['model'].get('n_know', 25200))
        n_qk = cfg['model'].get('n_qk', cfg['model'].get('n_q', 1580))
        n_v = cfg['model'].get('n_v', 2600)
        if str(model_version_cfg) == V4167_MODEL_VERSION:
            shard_checks = []
            for _name, _N in (
                    ('n_qk_global', cfg['model']['n_qk_global']),
                    ('n_qk_stage', cfg['model']['n_qk_stage']),
                    ('n_qk_local', cfg['model']['n_qk_local']),
                    ('n_v_global', cfg['model']['n_v_global']),
                    ('n_v_stage', cfg['model']['n_v_stage']),
                    ('n_v_local', cfg['model']['n_v_local']),
                    ('n_rst_global', cfg['model']['n_rst_global']),
                    ('n_rst_stage', cfg['model']['n_rst_stage']),
                    ('n_rst_local', cfg['model']['n_rst_local'])):
                if int(_N) > 0:
                    shard_checks.append((_name, _N))
            chunk_n_qk = cfg['model']['qk_visible_n']
            chunk_n_v = cfg['model']['v_visible_n']
            chunk_n_rst = cfg['model']['rst_visible_n']
        else:
            shard_checks = [('n_rst', n_rst), ('n_qk', n_qk), ('n_v', n_v)]
            chunk_n_qk = n_qk
            chunk_n_v = n_v
            chunk_n_rst = n_rst
        for _name, _N in shard_checks:
            if _N % mesh_model != 0:
                raise ValueError(
                    f"{_name}={_N} must be divisible by mesh_model={mesh_model} "
                    "for model-axis sharding.")
        # N_local = visible_N / mesh_model (each chip's share).
        nrst_local = chunk_n_rst // mesh_model
        nqk_local = chunk_n_qk // mesh_model
        nv_local = chunk_n_v // mesh_model

        n_chunks_rst = cfg['training'].get(
            'n_chunks_rst', auto_n_chunks(nrst_local, target_chunk_gb))
        n_chunks_qk = cfg['training'].get(
            'n_chunks_qk', auto_n_chunks(nqk_local, target_chunk_gb))
        n_chunks_v = cfg['training'].get(
            'n_chunks_v', auto_n_chunks(nv_local, target_chunk_gb))

        attn_qk_max_chunk = _chunk_size_from_count(
            'attn_qk', nqk_local, n_chunks_qk)
        attn_v_max_chunk = _chunk_size_from_count(
            'attn_v', nv_local, n_chunks_v)
        rst_max_chunk = _chunk_size_from_count(
            'rst', nrst_local, n_chunks_rst)
    else:
        n_rst = n_qk = n_v = 0
        nrst_local = nqk_local = nv_local = 0
        n_chunks_rst = n_chunks_qk = n_chunks_v = 1
        attn_qk_max_chunk = attn_v_max_chunk = rst_max_chunk = 1

    if is_host0:
        print(f"\n=== Mesh: ({mesh_data}, {mesh_model}) = "
              f"{total_devices} devices, per_device_batch={per_device_batch} ===")
        if _is_active_srw_version(model_version_cfg):
            print(f"  Chunks: rst={n_chunks_rst} (cs={nrst_local // max(n_chunks_rst,1)}), "
                  f"qk={n_chunks_qk}, attn_v={n_chunks_v}")
            chunk_mem = per_device_batch * max_seq_len * rst_max_chunk * 2 / 1e9
            print(f"  Est chunk mem (rst): {chunk_mem:.2f}GB bf16")
        elif is_baseline and mesh_model > 1:
            print("  Baseline params: model-axis tensor/vocab parallel shards.")
        elif is_baseline:
            print("  Baseline params: replicated; SRW shard_map disabled.")

    if is_baseline and mesh_model > 1:
        for _name, _value in (
                ('model.n_heads', cfg['model']['n_heads']),
                ('model.d_model', cfg['model']['d_model']),
                ('model.d_ff', cfg['model']['d_ff'])):
            if int(_value) % int(mesh_model) != 0:
                raise ValueError(
                    f"{_name}={_value} must be divisible by "
                    f"mesh_model={mesh_model} for baseline TP.")
    if str(model_version_cfg) == V4167_MODEL_VERSION:
        for _name, _value in (
                ('model.n_heads', cfg['model']['n_heads']),
                ('model.d_model', cfg['model']['d_model'])):
            if int(_value) % int(mesh_model) != 0:
                raise ValueError(
                    f"{_name}={_value} must be divisible by "
                    f"mesh_model={mesh_model} for v4167 TP attention/O.")

    # Shard params using the model-version-specific policy.
    param_shardings = get_param_shardings(
        params, mesh, model_version_cfg,
        operation_space_enabled=operation_space_tau_free_enabled)
    if is_host0:
        _print_param_sharding_summary(param_shardings, model_version_cfg)
    params = shard_params_to_mesh(params, param_shardings)

    opt_state = optimizer.init(params)
    opt_state = _replicate_optimizer_state_scalars_to_mesh(opt_state, mesh)
    target_params = params
    target_opt_state = opt_state
    latest_checkpoint_manager = _create_orbax_checkpoint_manager(
        _join(checkpoint_dir, 'checkpoints'),
        checkpoint_interval=ckpt_interval,
        keep_last=checkpoint_keep_last,
        create=True,
        best_tracking=False,
    )
    best_checkpoint_manager = _create_orbax_checkpoint_manager(
        _join(checkpoint_dir, 'best_checkpoints'),
        checkpoint_interval=1,
        keep_last=best_checkpoint_keep_last,
        create=True,
        best_tracking=True,
    )

    if _has_resume_checkpoint:
        target_state = _build_orbax_state(
            target_params, target_opt_state, rng,
            epoch=0,
            global_step=0,
            step_in_epoch=0,
            steps_per_epoch=steps_per_epoch,
            best_val_loss=float('inf'),
            training_config=training_config,
            full_config=full_config_snapshot,
            model_config=cfg['model'],
        )
        restored_state, restored_metadata = _restore_orbax_state(
            latest_checkpoint_manager, resume_step, target_state)
        params = _match_tree_to_template_on_mesh(
            restored_state['params'], target_params, mesh, name='params')
        opt_state = _match_tree_to_template_on_mesh(
            restored_state['opt_state'], target_opt_state, mesh,
            name='opt_state')
        if is_host0:
            print("  Restored params/optimizer state matched to full mesh sharding.")
        if 'rng' not in restored_state:
            raise KeyError("Orbax checkpoint state is missing required rng.")
        restored_rng = np.asarray(
            restored_state['rng'], dtype=np.uint32).reshape((2,))
        rng = jnp.asarray(restored_rng, dtype=jnp.uint32)
        start_epoch = _state_scalar(restored_state, 'epoch', 0, int)
        global_step = _state_scalar(
            restored_state, 'global_step',
            _state_scalar(restored_state, 'step', 0, int),
            int)
        best_val_loss = _state_scalar(
            restored_state, 'best_val_loss', float('inf'), float)
        saved_step_in_epoch = _state_scalar(
            restored_state, 'step_in_epoch', 0, int)
        saved_steps_per_epoch = _state_scalar(
            restored_state, 'steps_per_epoch', 0, int)
        if saved_step_in_epoch > 0 and saved_steps_per_epoch == steps_per_epoch:
            start_step_in_epoch = saved_step_in_epoch
        elif saved_step_in_epoch > 0:
            if is_host0:
                print(
                    f"  Warning: steps_per_epoch changed "
                    f"({saved_steps_per_epoch} -> {steps_per_epoch}), "
                    "cannot use step_in_epoch for resume. Starting epoch "
                    "from beginning.")
            start_step_in_epoch = 0
        if is_host0:
            opt_count = _maybe_get_opt_state_count(opt_state)
            print(f"  [resume check] global_step={global_step}", flush=True)
            print(f"  [resume check] opt_state_count={opt_count}", flush=True)
            print(f"  [resume check] best_val_loss={best_val_loss}", flush=True)
            print(
                f"  [resume check] step_in_epoch={start_step_in_epoch}",
                flush=True)
            if opt_count is not None and int(opt_count) != int(global_step):
                print(
                    "  WARNING: optimizer count does not match global_step: "
                    f"opt_count={opt_count} global_step={global_step}",
                    flush=True,
                )
            try:
                param_leaf = params['block_0']['attn']['expand_O']['kernel']
                print(
                    "  [resume sharding check] param expand_O devices="
                    f"{_debug_leaf_device_ids(param_leaf)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    "  [resume sharding check] param inspect failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            opt_count_leaf = None
            try:
                opt_count_leaf = opt_state[2].count
            except Exception:
                pass
            if opt_count_leaf is not None:
                print(
                    "  [resume sharding check] opt_state count devices="
                    f"{_debug_leaf_device_ids(opt_count_leaf)}",
                    flush=True,
                )
            print(
                f"  Restored Orbax checkpoint: epoch={start_epoch}, "
                f"global_step={global_step}, "
                f"step_in_epoch={start_step_in_epoch}, "
                f"best_val_loss={best_val_loss:.4f}")
            if restored_metadata:
                print(
                    "  Restored checkpoint metadata kind="
                    f"{restored_metadata.get('checkpoint_kind', '<unknown>')}")
            print("  Restored training RNG from Orbax checkpoint.")
    elif is_host0:
        print(f"  Orbax checkpoints: {_join(checkpoint_dir, 'checkpoints')}")

    # Fail-fast check: global_step must match across hosts after restore.
    if n_hosts > 1:
        _gs_local = np.array([global_step], dtype=np.int64)
        _gs_all = np.asarray(process_allgather(_gs_local)).flatten()
        if not np.all(_gs_all == global_step):
            raise RuntimeError(
                f"global_step inconsistent across hosts after resume: "
                f"host {host_id} sees {global_step}, all hosts: "
                f"{_gs_all.tolist()}. Resume broadcast likely failed or "
                "checkpoint files diverged.")
        if is_host0:
            print(
                f"  [verified] global_step={global_step} consistent "
                f"across {n_hosts} hosts")

    # Create shard_map functions if mesh_model > 1 or the model demands
    # the sharded path.
    #
    # v4164: `_sharded_fns` is the slim train path; `_sharded_fns_analysis`
    # is the full observational path used only by analysis_step.
    _sharded_fns = None
    _sharded_fns_analysis = None
    _force_sharded = _is_active_srw_version(model_version_cfg)
    if is_baseline and mesh_model > 1:
        _sharded_fns = create_baseline_sharded_fns(mesh, cfg)
        if is_host0:
            print(
                f"  baseline-JAX shard_map enabled "
                f"(mesh_model={mesh_model}; attention+ffn+vocab TP)",
                flush=True)
    elif _is_active_srw_version(model_version_cfg) and (
            mesh_model > 1 or _force_sharded):
        _srw_module_name = _model_registry_entry(model_version_cfg)['module']
        _v4164_module = __import__(_srw_module_name, fromlist=['make_sharded_srw'])
        make_sharded_srw = _v4164_module.make_sharded_srw
        make_sharded_srw_minimal = getattr(
            _v4164_module, 'make_sharded_srw_minimal', None)
        make_sharded_srw_paired_minimal = getattr(
            _v4164_module, 'make_sharded_srw_paired_minimal', None)
        make_sharded_srw_paired_dense_minimal = getattr(
            _v4164_module, 'make_sharded_srw_paired_dense_minimal', None)
        max_chunk = cfg['training'].get('max_chunk_size', None)
        if max_chunk is not None:
            attn_qk_max_chunk = attn_v_max_chunk = rst_max_chunk = int(max_chunk)

        _srw_base_kwargs = {'mesh': mesh}
        _srw_base_kwargs.update(_v4164_sharded_kwargs(cfg))
        import inspect as _inspect
        def _factory_kwargs(factory, kwargs):
            sig = _inspect.signature(factory)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                return dict(kwargs)
            return {k: v for k, v in kwargs.items() if k in sig.parameters}

        _opspace_cfg = cfg.get('training', {}).get(
            'operation_space',
            cfg.get('model', {}).get('operation_space', {}))
        if not isinstance(_opspace_cfg, dict):
            _opspace_cfg = {}
        _opspace_enabled = operation_space_tau_free_enabled
        _opspace_pools = (
            _opspace_cfg.get('pools', {})
            if isinstance(_opspace_cfg.get('pools', {}), dict) else {})
        _opspace_layouts = (
            _v4168_operation_space_pool_layouts(
                cfg.get('training', {}), cfg.get('model', {}))
            if _opspace_enabled else {})

        def _opspace_pool_kwargs(pool):
            if not _opspace_enabled:
                return {}
            layout = _opspace_layouts.get(pool, {})
            if not isinstance(layout, dict):
                layout = {}
            return {
                'operation_space_execution_backend': str(layout.get(
                    'execution_backend',
                    'sparse_region_block' if pool == 'rst'
                    else 'dense')).lower(),
                'opspace_bucket_capacity_factor': float(layout.get(
                    'bucket_capacity_factor',
                    layout.get('block_capacity_factor', 1.25))),
                'opspace_high_regret_threshold': float(layout.get(
                    'high_regret_threshold', 0.05)),
                'opspace_region_capacity_factor': float(layout.get(
                    'region_capacity_factor', 1.25)),
                'opspace_block_capacity_factor': float(layout.get(
                    'block_capacity_factor', 1.25)),
                'opspace_num_regions': int(layout.get(
                    'num_regions', 32 if pool == 'rst' else 8)),
                'opspace_blocks_per_region': int(layout.get(
                    'blocks_per_region', 1)),
                'opspace_operators_per_block': int(layout.get(
                    'operators_per_block', 128)),
                'opspace_visible_regions': int(layout.get(
                    'visible_regions', 4 if pool == 'rst' else 2)),
                'opspace_visible_blocks_per_region': int(layout.get(
                    'visible_blocks_per_region', 1)),
                'opspace_region_score_pooling': str(layout.get(
                    'region_score_pooling', 'smoothmax')).lower(),
                'opspace_region_score_temperature': float(layout.get(
                    'region_score_temperature', 0.25)),
            }

        def _srw_pool_kwargs(pool):
            kwargs = dict(_srw_base_kwargs)
            if str(model_version_cfg) == V4168_MODEL_VERSION:
                m_cfg = cfg['model']
                kwargs.update({
                    'block_size': int(m_cfg.get(f'{pool}_block_size', 256)),
                    'top_blocks': int(m_cfg.get(f'{pool}_top_blocks', 2)),
                    'block_margin': float(m_cfg.get('block_margin', 0.0)),
                    'hardware_sector_execution_enabled':
                        hardware_sector_execution_enabled,
                    'hardware_sector_debug_token_gather_fallback': bool(
                        cfg['training'].get(
                            'hardware_sector_debug_token_gather_fallback',
                            False)),
                })
                kwargs.update(_opspace_pool_kwargs(pool))
            return kwargs

        _supports_analysis = (
            'analysis' in _inspect.signature(make_sharded_srw).parameters
        )
        # Slim train kernel; analysis defaults to False.
        _sharded_single_v = make_sharded_srw(
            max_chunk_size=attn_v_max_chunk,
            **_factory_kwargs(make_sharded_srw, _srw_pool_kwargs('v')))
        _sharded_single_rst = make_sharded_srw(
            max_chunk_size=rst_max_chunk,
            **_factory_kwargs(make_sharded_srw, _srw_pool_kwargs('rst')))
        _sharded_single_qk_minimal = None
        _sharded_single_v_minimal = None
        _sharded_single_rst_minimal = None
        if make_sharded_srw_minimal is not None:
            if str(model_version_cfg) != V4168_MODEL_VERSION:
                _sharded_single_qk_minimal = make_sharded_srw_minimal(
                    max_chunk_size=attn_qk_max_chunk,
                    **_factory_kwargs(
                        make_sharded_srw_minimal, _srw_pool_kwargs('qk')))
            _sharded_single_v_minimal = make_sharded_srw_minimal(
                max_chunk_size=attn_v_max_chunk,
                **_factory_kwargs(
                    make_sharded_srw_minimal, _srw_pool_kwargs('v')))
            _sharded_single_rst_minimal = make_sharded_srw_minimal(
                max_chunk_size=rst_max_chunk,
                **_factory_kwargs(
                    make_sharded_srw_minimal, _srw_pool_kwargs('rst')))
        if hasattr(_v4164_module, 'make_sharded_srw_paired'):
            _paired_factory = _v4164_module.make_sharded_srw_paired
            _sharded_paired_attn_qk = _v4164_module.make_sharded_srw_paired(
                max_chunk_size=attn_qk_max_chunk,
                **_factory_kwargs(_paired_factory, _srw_pool_kwargs('qk')))
            _sharded_paired_attn_qk_minimal = None
            if make_sharded_srw_paired_minimal is not None:
                if operation_space_tau_free_enabled:
                    if make_sharded_srw_paired_dense_minimal is None:
                        raise RuntimeError(
                            "operation_space enabled but required tau-free "
                            "executor is missing: pool=qk "
                            f"execution_backend={_opspace_layouts.get('qk', {}).get('execution_backend', 'unknown')}")
                    _sharded_paired_attn_qk_minimal = (
                        make_sharded_srw_paired_dense_minimal(
                            max_chunk_size=attn_qk_max_chunk,
                            **_factory_kwargs(
                                make_sharded_srw_paired_dense_minimal,
                                _srw_pool_kwargs('qk'))))
                elif (str(model_version_cfg) == V4168_MODEL_VERSION
                      and make_sharded_srw_paired_dense_minimal is not None):
                    _sharded_paired_attn_qk_minimal = (
                        make_sharded_srw_paired_dense_minimal(
                            max_chunk_size=attn_qk_max_chunk,
                            **_factory_kwargs(
                                make_sharded_srw_paired_dense_minimal,
                                _srw_pool_kwargs('qk'))))
                else:
                    _sharded_paired_attn_qk_minimal = (
                        make_sharded_srw_paired_minimal(
                            max_chunk_size=attn_qk_max_chunk,
                            **_factory_kwargs(
                                make_sharded_srw_paired_minimal,
                                _srw_pool_kwargs('qk'))))
            _sharded_fns = {
                'single': _sharded_single_v,
                'attn_v_single': _sharded_single_v,
                'rst_single': _sharded_single_rst,
                'paired': _sharded_paired_attn_qk,
                'attn_qk_paired': _sharded_paired_attn_qk,
            }
            if _sharded_single_v_minimal is not None:
                _sharded_fns.update({
                    'attn_qk_single_minimal': _sharded_single_qk_minimal,
                    'attn_v_single_minimal': _sharded_single_v_minimal,
                    'rst_single_minimal': _sharded_single_rst_minimal,
                })
            if _sharded_paired_attn_qk_minimal is not None:
                _sharded_fns['attn_qk_paired_minimal'] = (
                    _sharded_paired_attn_qk_minimal)
        else:
            _sharded_fns = _sharded_single_rst
        if str(model_version_cfg) == V4167_MODEL_VERSION:
            _extra_factory = getattr(
                _v4164_module, 'create_v4167_tp_sharded_fns', None)
            if _extra_factory is None:
                raise RuntimeError(
                    "v4167 module is missing create_v4167_tp_sharded_fns.")
            if not isinstance(_sharded_fns, dict):
                raise RuntimeError(
                    "v4167 TP extras require dict-style sharded_fns.")
            _v4167_extra_fns = _extra_factory(mesh, cfg)
            _sharded_fns.update(_v4167_extra_fns)
        if operation_space_tau_free_enabled:
            if hardware_repack_enabled or hardware_sector_execution_enabled:
                raise RuntimeError(
                    "operation_space enabled requires hardware repack and "
                    "hardware sector execution to remain disabled.")
            if not isinstance(_sharded_fns, dict):
                raise RuntimeError(
                    "operation_space enabled but required tau-free QK/V/RST "
                    "executor is missing")
            _required_opspace_layouts = {'qk', 'v', 'rst'}
            if (not isinstance(_opspace_layouts, dict)
                    or set(_opspace_layouts) & _required_opspace_layouts
                    != _required_opspace_layouts):
                raise RuntimeError(
                    "operation_space enabled but required QK/V/RST pool "
                    "layout is missing")
            _opspace_backends = {
                _pool: str(_opspace_layouts[_pool].get(
                    'execution_backend', '')).lower()
                for _pool in ('qk', 'v', 'rst')
            }
            for _pool, _required_executor in (
                    ('qk', 'attn_qk_paired_minimal'),
                    ('v', 'attn_v_single_minimal'),
                    ('rst', 'rst_single_minimal')):
                if _sharded_fns.get(_required_executor, None) is None:
                    raise RuntimeError(
                        "operation_space enabled but required tau-free "
                        "executor is missing: "
                        f"pool={_pool} "
                        f"execution_backend={_opspace_backends[_pool]}")
            _sharded_fns.update({
                'operation_space_tau_free': True,
                'qk_backend': _opspace_backends['qk'],
                'v_backend': _opspace_backends['v'],
                'rst_backend': _opspace_backends['rst'],
            })
        # Analysis (observation only). Factory kwargs forward analysis=True
        # only when the v4164 factory advertises the kwarg.
        if _supports_analysis and not operation_space_tau_free_enabled:
            _sharded_single_v_a = make_sharded_srw(
                analysis=True, max_chunk_size=attn_v_max_chunk,
                **_factory_kwargs(make_sharded_srw, _srw_pool_kwargs('v')))
            _sharded_single_rst_a = make_sharded_srw(
                analysis=True, max_chunk_size=rst_max_chunk,
                **_factory_kwargs(make_sharded_srw, _srw_pool_kwargs('rst')))
            if hasattr(_v4164_module, 'make_sharded_srw_paired'):
                _paired_factory = _v4164_module.make_sharded_srw_paired
                _sharded_paired_a = _v4164_module.make_sharded_srw_paired(
                    analysis=True, max_chunk_size=attn_qk_max_chunk,
                    **_factory_kwargs(_paired_factory, _srw_pool_kwargs('qk')))
                _sharded_fns_analysis = {
                    'single': _sharded_single_v_a,
                    'attn_v_single': _sharded_single_v_a,
                    'rst_single': _sharded_single_rst_a,
                    'paired': _sharded_paired_a,
                    'attn_qk_paired': _sharded_paired_a,
                }
            else:
                _sharded_fns_analysis = _sharded_single_rst_a
            if (str(model_version_cfg) == V4167_MODEL_VERSION
                    and isinstance(_sharded_fns_analysis, dict)):
                _sharded_fns_analysis.update(_v4167_extra_fns)
        if is_host0:
            _extra_msg = (
                "; v4167 TP extras=router_dense,attention_o,vocab_parallel"
                if str(model_version_cfg) == V4167_MODEL_VERSION else "")
            if str(model_version_cfg) == V4168_MODEL_VERSION:
                _v4168_exec_mode = (
                    "operation_space_tau_free_relu"
                    if _opspace_enabled
                    else ("vq_ivf_sector_bucketed"
                          if hardware_sector_execution_enabled
                          else "block_sparse_fallback"))
                _extra_msg = (
                    f"; v4168 minimal {_v4168_exec_mode} "
                    f"opspace_backend qk/v/rst="
                    f"{_opspace_layouts.get('qk', {}).get('execution_backend', 'n/a')}/"
                    f"{_opspace_layouts.get('v', {}).get('execution_backend', 'n/a')}/"
                    f"{_opspace_layouts.get('rst', {}).get('execution_backend', 'n/a')}"
                    if _opspace_enabled
                    else (
                        f"; v4168 minimal {_v4168_exec_mode} "
                        f"block_size qk/v/rst={cfg['model'].get('qk_block_size', 256)}/"
                        f"{cfg['model'].get('v_block_size', 256)}/"
                        f"{cfg['model'].get('rst_block_size', 256)}, "
                        f"top_blocks qk/v/rst={cfg['model'].get('qk_top_blocks', 2)}/"
                        f"{cfg['model'].get('v_top_blocks', 2)}/"
                        f"{cfg['model'].get('rst_top_blocks', 2)}"))
            _qk_mode_msg = (
                ("QK operation-space"
                 if _opspace_enabled else "QK dense-distributed")
                if str(model_version_cfg) == V4168_MODEL_VERSION
                else "QK fused")
            _analysis_kernel_status = (
                "off"
                if operation_space_tau_free_enabled
                else ("on" if _supports_analysis else "off"))
            print(f"  shard_map enabled (mesh_model={mesh_model}, {_qk_mode_msg}"
                  f"; chunks attn_qk/attn_v/rst={n_chunks_qk}/{n_chunks_v}/{n_chunks_rst}"
                  f"; max_chunk attn_qk/attn_v/rst={attn_qk_max_chunk}/{attn_v_max_chunk}/{rst_max_chunk}"
                  f"; analysis kernels={_analysis_kernel_status}"
                  f"{_extra_msg})")
            if str(model_version_cfg) == V4168_MODEL_VERSION:
                if _opspace_enabled:
                    _qk_layout = _opspace_layouts.get('qk', {})
                    _v_layout = _opspace_layouts.get('v', {})
                    _rst_layout = _opspace_layouts.get('rst', {})
                    _qk_backend = str(_qk_layout.get(
                        'execution_backend', 'dense')).lower()
                    _v_backend = str(_v_layout.get(
                        'execution_backend', 'dense')).lower()
                    _rst_backend = str(_rst_layout.get(
                        'execution_backend', 'sparse_region_block')).lower()

                    def _opspace_pool_line(_name, _layout, _backend):
                        _visible_ops = (
                            int(_layout.get('visible_regions', 0))
                            * int(_layout.get(
                                'visible_blocks_per_region', 0))
                            * int(_layout.get('operators_per_block', 0)))
                        return (
                            f"[opspace/{_name}] backend={_backend} "
                            f"regions={int(_layout.get('num_regions', 0))} "
                            f"blocks/region="
                            f"{int(_layout.get('blocks_per_region', 0))} "
                            f"ops/block="
                            f"{int(_layout.get('operators_per_block', 0))} "
                            f"visible_regions="
                            f"{int(_layout.get('visible_regions', 0))} "
                            f"visible_blocks="
                            f"{int(_layout.get('visible_blocks_per_region', 0))} "
                            f"visible_ops={_visible_ops}")

                    _rst_compute = (
                        float(_rst_layout.get(
                            'physical_visible_ops_per_token', 0.0))
                        / float(max(
                            int(_rst_layout.get(
                                'global_operator_capacity', 1)), 1)))
                    print(
                        "[opspace] atlas=region_block "
                        f"qk_backend={_qk_backend} "
                        f"v_backend={_v_backend} "
                        f"rst_backend={_rst_backend} "
                        "direct_tau=false selection_calibration=false "
                        "gate=relu2\n"
                        f"{_opspace_pool_line('qk', _qk_layout, _qk_backend)}\n"
                        f"{_opspace_pool_line('v', _v_layout, _v_backend)}\n"
                        f"{_opspace_pool_line('rst', _rst_layout, _rst_backend)} "
                        f"compute={_rst_compute * 100.0:.2f}%\n"
                        f"  operation_space_repack_enabled={operation_space_repack_enabled}\n"
                        "  direct_tau_active_for_qk_v_rst=false\n"
                        "  selection_calibration_active_for_qk_v_rst=false\n"
                        "  tau_init_required_for_qk_v_rst=false",
                        flush=True)
                else:
                    print(
                        "v4168 hardware routing policy:\n"
                        "  qk: dense_distributed\n"
                        f"  v: vq_ivf_sector block_size={cfg['model'].get('v_block_size', 256)} "
                        f"top_blocks={cfg['model'].get('v_top_blocks', 2)}\n"
                        f"  rst: vq_ivf_sector block_size={cfg['model'].get('rst_block_size', 256)} "
                        f"top_blocks={cfg['model'].get('rst_top_blocks', 2)}\n"
                        f"  repack_strategy={hardware_repack_config['hardware_repack_strategy']}\n"
                        f"  repack_interval_steps={hardware_repack_config['hardware_repack_interval_steps']}\n"
                        f"  repack_warmup_steps={hardware_repack_config['hardware_repack_warmup_steps']}\n"
                        f"  max_move_frac={hardware_repack_config['hardware_repack_max_move_frac']}\n"
                        "  bucket_capacity_mult=1.5\n"
                        "  bucket_capacity_round=multiple_of_128\n"
                        "  fallback_in_fast_graph=false\n"
                        f"  main_val_path={'sector_bucketed' if hardware_sector_execution_enabled else 'block_sparse_fallback'}\n"
                        "  dense_ref_enabled=false",
                        flush=True)

    train_step_fn = create_train_step(
        model, optimizer, orth_weight, div_weight, lb_weight,
        tau_reg_weight, dead_penalty_weight,
        inactive_aux_weight, inactive_aux_asymmetry,
        inactive_aux_weight_q, inactive_aux_weight_k,
        inactive_aux_weight_v, inactive_aux_weight_rst,
        rank, knowledge_rank, n_feature_qk, n_restore_qk,
        weight_decay=weight_decay, pool_weight_decay=pool_weight_decay,
        inactive_aux_warmup_steps=inactive_aux_warmup_steps,
        inactive_aux_lower_bound=inactive_aux_lower_bound,
        inactive_aux_upper_bound=inactive_aux_upper_bound,
        inactive_aux_bound_eps=inactive_aux_bound_eps,
        inactive_aux_dev_mode=inactive_aux_dev_mode,
        inactive_aux_ce_clip_std=inactive_aux_ce_clip_std,
        inactive_aux_z_clip=inactive_aux_z_clip,
        inactive_aux_z_tanh=inactive_aux_z_tanh,
        inactive_aux_weighted_clip=inactive_aux_weighted_clip,
        inactive_aux_normalize_by_layers=inactive_aux_normalize_by_layers,
        inactive_aux_asymmetry_q=inactive_aux_asymmetry_q,
        inactive_aux_asymmetry_k=inactive_aux_asymmetry_k,
        inactive_aux_asymmetry_v=inactive_aux_asymmetry_v,
        inactive_aux_asymmetry_rst=inactive_aux_asymmetry_rst,
        inactive_aux_enabled=inactive_aux_enabled,
        dead_penalty_qk_weight=dead_penalty_qk_weight,
        dead_penalty_v_weight=dead_penalty_v_weight,
        dead_penalty_rst_weight=dead_penalty_rst_weight,
        cb1a_enabled=cb1a_enabled,
        cb1a_weight=cb1a_weight,
        cb1a_challenge_weight=cb1a_challenge_weight,
        cb1a_prune_weight=cb1a_prune_weight,
        cb1a_qk_weight=cb1a_qk_weight,
        cb1a_v_weight=cb1a_v_weight,
        cb1a_rst_weight=cb1a_rst_weight,
        cb1a_qk_challenge_weight=cb1a_qk_challenge_weight,
        cb1a_qk_prune_weight=cb1a_qk_prune_weight,
        cb1a_v_challenge_weight=cb1a_v_challenge_weight,
        cb1a_v_prune_weight=cb1a_v_prune_weight,
        cb1a_rst_challenge_weight=cb1a_rst_challenge_weight,
        cb1a_rst_prune_weight=cb1a_rst_prune_weight,
        cb1a_ce_mode=cb1a_ce_mode,
        cb1a_eps=cb1a_eps,
        dead_penalty_weighted_clip=dead_penalty_weighted_clip,
        global_grad_clip=global_grad_clip,
        tau_lr_mult=tau_lr_mult,
        tau_grad_clip=tau_grad_clip,
        router_proj_lr_mult=router_proj_lr_mult,
        router_proj_grad_clip=router_proj_grad_clip,
        router_scan_lr_mult=router_scan_lr_mult,
        router_scan_grad_clip=router_scan_grad_clip,
        route_emb_lr_mult=route_emb_lr_mult,
        route_emb_grad_clip=route_emb_grad_clip,
        op_key_lr_mult=op_key_lr_mult,
        op_key_grad_clip=op_key_grad_clip,
        enable_control_update_caps=enable_control_update_caps,
        router_proj_update_ratio_cap=router_proj_update_ratio_cap,
        route_emb_update_ratio_cap=route_emb_update_ratio_cap,
        tau_update_abs_cap=tau_update_abs_cap,
        scan_update_abs_cap=scan_update_abs_cap,
        total_training_steps=total_steps,
        soft_gate_schedule_active=soft_gate_schedule_active,
        soft_gate_t_start=soft_gate_t_start,
        soft_gate_t_final=soft_gate_t_final,
        soft_gate_t_hold_frac=soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac=soft_gate_t_anneal_end_frac,
        soft_gate_schedule=soft_gate_schedule,
        soft_gate_t_power=soft_gate_t_power,
        soft_gate_t_gompertz_center=soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness=soft_gate_t_gompertz_steepness,
        pool_specific_gate_t=pool_specific_gate_t,
        soft_gate_pool_schedules=soft_gate_pool_schedules,
        boundary_power_schedule_active=boundary_power_schedule_active,
        soft_gate_boundary_power_start=soft_gate_boundary_power_start,
        soft_gate_boundary_power_mid=soft_gate_boundary_power_mid,
        soft_gate_boundary_power_final=soft_gate_boundary_power_final,
        soft_gate_boundary_power_start_frac=soft_gate_boundary_power_start_frac,
        soft_gate_boundary_power_mid_frac=soft_gate_boundary_power_mid_frac,
        soft_gate_boundary_power_final_frac=soft_gate_boundary_power_final_frac,
        admission_den_power=admission_den_power,
        inactive_aux_start_frac=inactive_aux_start_frac,
        inactive_aux_full_frac=inactive_aux_full_frac,
        inactive_aux_schedule=inactive_aux_schedule,
        sharded_fns=_sharded_fns, mesh=mesh,
        is_baseline=is_baseline,
        compact_train_metrics=compact_train_metrics,
        keep_train_layer_metrics=False,
        tokens_per_step=int(batch_size) * int(max_seq_len))
    eval_step_fn = create_eval_step(
        model, sharded_fns=_sharded_fns, return_dead_stats=True,
        total_training_steps=total_steps,
        soft_gate_schedule_active=soft_gate_schedule_active,
        soft_gate_t_start=soft_gate_t_start,
        soft_gate_t_final=soft_gate_t_final,
        soft_gate_t_hold_frac=soft_gate_t_hold_frac,
        soft_gate_t_anneal_end_frac=soft_gate_t_anneal_end_frac,
        soft_gate_schedule=soft_gate_schedule,
        soft_gate_t_power=soft_gate_t_power,
        soft_gate_t_gompertz_center=soft_gate_t_gompertz_center,
        soft_gate_t_gompertz_steepness=soft_gate_t_gompertz_steepness,
        pool_specific_gate_t=pool_specific_gate_t,
        soft_gate_pool_schedules=soft_gate_pool_schedules,
        boundary_power_schedule_active=boundary_power_schedule_active,
        soft_gate_boundary_power_start=soft_gate_boundary_power_start,
        soft_gate_boundary_power_mid=soft_gate_boundary_power_mid,
        soft_gate_boundary_power_final=soft_gate_boundary_power_final,
        soft_gate_boundary_power_start_frac=soft_gate_boundary_power_start_frac,
        soft_gate_boundary_power_mid_frac=soft_gate_boundary_power_mid_frac,
        soft_gate_boundary_power_final_frac=soft_gate_boundary_power_final_frac,
        admission_den_power=admission_den_power)
    eval_prune_step_fns = {}
    if eval_effective_prune_enabled:
        for _eps in eval_effective_prune_eps_list:
            _eps_f = float(_eps)
            eval_prune_step_fns[_eps_f] = create_eval_step(
                model, sharded_fns=_sharded_fns, return_dead_stats=True,
                return_prune_stats=True, execution_prune_eps=_eps_f,
                total_training_steps=total_steps,
                soft_gate_schedule_active=soft_gate_schedule_active,
                soft_gate_t_start=soft_gate_t_start,
                soft_gate_t_final=soft_gate_t_final,
                soft_gate_t_hold_frac=soft_gate_t_hold_frac,
                soft_gate_t_anneal_end_frac=soft_gate_t_anneal_end_frac,
                soft_gate_schedule=soft_gate_schedule,
                soft_gate_t_power=soft_gate_t_power,
                soft_gate_t_gompertz_center=soft_gate_t_gompertz_center,
                soft_gate_t_gompertz_steepness=soft_gate_t_gompertz_steepness,
                pool_specific_gate_t=pool_specific_gate_t,
                soft_gate_pool_schedules=soft_gate_pool_schedules,
                boundary_power_schedule_active=boundary_power_schedule_active,
                soft_gate_boundary_power_start=soft_gate_boundary_power_start,
                soft_gate_boundary_power_mid=soft_gate_boundary_power_mid,
                soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                soft_gate_boundary_power_start_frac=soft_gate_boundary_power_start_frac,
                soft_gate_boundary_power_mid_frac=soft_gate_boundary_power_mid_frac,
                soft_gate_boundary_power_final_frac=soft_gate_boundary_power_final_frac,
                admission_den_power=admission_den_power)
    # v4164 analysis_step is active when the full analysis kernels exist.
    if operation_space_tau_free_enabled:
        analysis_step_fn = None
    elif _sharded_fns_analysis is not None:
        analysis_step_fn = create_analysis_step(
            model, sharded_fns=_sharded_fns_analysis,
            total_training_steps=total_steps,
            soft_gate_schedule_active=soft_gate_schedule_active,
            soft_gate_t_start=soft_gate_t_start,
            soft_gate_t_final=soft_gate_t_final,
            soft_gate_t_hold_frac=soft_gate_t_hold_frac,
            soft_gate_t_anneal_end_frac=soft_gate_t_anneal_end_frac,
            soft_gate_schedule=soft_gate_schedule,
            soft_gate_t_power=soft_gate_t_power,
            soft_gate_t_gompertz_center=soft_gate_t_gompertz_center,
            soft_gate_t_gompertz_steepness=soft_gate_t_gompertz_steepness,
            pool_specific_gate_t=pool_specific_gate_t,
            soft_gate_pool_schedules=soft_gate_pool_schedules,
            boundary_power_schedule_active=boundary_power_schedule_active,
            soft_gate_boundary_power_start=soft_gate_boundary_power_start,
            soft_gate_boundary_power_mid=soft_gate_boundary_power_mid,
            soft_gate_boundary_power_final=soft_gate_boundary_power_final,
            soft_gate_boundary_power_start_frac=soft_gate_boundary_power_start_frac,
            soft_gate_boundary_power_mid_frac=soft_gate_boundary_power_mid_frac,
            soft_gate_boundary_power_final_frac=soft_gate_boundary_power_final_frac,
            admission_den_power=admission_den_power)
    else:
        analysis_step_fn = None
    # No current-train-batch diagnostic forward.
    geometry_step_fn = None if is_baseline else create_geometry_step(
        max_sample=int(tcfg.get(
            'geometry_max_sample',
            tcfg.get('heavy_geometry_max_sample', 512))))

    # Initial operator-key drift snapshot. Identity here means drift=0 on the
    # first step; legacy pools use their route embeddings as the signature.
    def _drift_snap(p):
        def _unit(x):
            x = jnp.asarray(x, dtype=jnp.float32)
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

        def _op_key(read, write, read_proj, write_proj):
            r_key = _unit(read) @ read_proj
            w_key = _unit(write) @ write_proj
            return _unit(_unit(r_key) * _unit(w_key))

        if 'neuron_pool' not in p:
            z = jnp.float32(0.0)
            return {
                'attn_qk_op_key': z,
                'attn_v_op_key': z,
                'rst_op_key': z,
            }

        pool = p['neuron_pool']
        if ('attn_qk_read_global' in pool
                or 'attn_qk_read_shared' in pool):
            def _flat_op_key(prefix):
                if f'{prefix}_read_global' in pool:
                    parts = [_op_key(
                        pool[f'{prefix}_read_global'],
                        pool[f'{prefix}_write_global'],
                        pool[f'{prefix}_op_read_proj'],
                        pool[f'{prefix}_op_write_proj'])]
                    for _scope in ('stage', 'local'):
                        read_key = f'{prefix}_read_{_scope}'
                        write_key = f'{prefix}_write_{_scope}'
                        if read_key in pool:
                            op_key = _op_key(
                                pool[read_key], pool[write_key],
                                pool[f'{prefix}_op_read_proj'],
                                pool[f'{prefix}_op_write_proj'])
                            parts.append(op_key.reshape(
                                (-1, parts[0].shape[-1])))
                    return jnp.concatenate(parts, axis=0)

                op_shared = _op_key(
                    pool[f'{prefix}_read_shared'],
                    pool[f'{prefix}_write_shared'],
                    pool[f'{prefix}_op_read_proj'],
                    pool[f'{prefix}_op_write_proj'])
                stage_read_key = f'{prefix}_read_stage'
                stage_write_key = f'{prefix}_write_stage'
                if stage_read_key not in pool:
                    return op_shared
                op_stage = _op_key(
                    pool[stage_read_key], pool[stage_write_key],
                    pool[f'{prefix}_op_read_proj'],
                    pool[f'{prefix}_op_write_proj'])
                return jnp.concatenate(
                    [op_shared, op_stage.reshape((-1, op_shared.shape[-1]))],
                    axis=0)
            return {
                'attn_qk_op_key': _flat_op_key('attn_qk'),
                'attn_v_op_key': _flat_op_key('attn_v'),
                'rst_op_key': _flat_op_key('rst'),
            }
        if 'attn_qk_op_read_proj' in pool:
            return {
                'attn_qk_op_key': _op_key(
                    pool['attn_qk_read'], pool['attn_qk_write'],
                    pool['attn_qk_op_read_proj'],
                    pool['attn_qk_op_write_proj']),
                'attn_v_op_key': _op_key(
                    pool['attn_v_read'], pool['attn_v_write'],
                    pool['attn_v_op_read_proj'],
                    pool['attn_v_op_write_proj']),
                'rst_op_key': _op_key(
                    pool['rst_read'], pool['rst_write'],
                    pool['rst_op_read_proj'], pool['rst_op_write_proj']),
            }
        if 'attn_qk_emb' in pool:
            return {
                'attn_qk_op_key': pool['attn_qk_emb'],
                'attn_v_op_key': pool['attn_v_emb'],
                'rst_op_key': pool['rst_emb'],
            }
        if 'qk_emb' in pool:
            return {
                'attn_qk_op_key': pool['qk_emb'],
                'attn_v_op_key': pool['v_emb'],
                'rst_op_key': pool['rst_emb'],
            }
        return {
            'attn_qk_op_key': pool['q_read'],
            'attn_v_op_key': pool['v_read'],
            'rst_op_key': pool['rst_read'],
        }

    def _dummy_drift_snap():
        z = jnp.zeros((), dtype=jnp.float32)
        return {
            'attn_qk_op_key': z,
            'attn_v_op_key': z,
            'rst_op_key': z,
        }

    # ----------------------------------------------------------
    # OOM check + JIT pre-compile
    # ----------------------------------------------------------
    class _SkipStartupCheck(Exception):
        pass

    if is_host0 and run_oom_check:
        print(f"\n=== OOM check: real train_step (forward+backward) "
              f"per_device_batch={per_device_batch}, seq_len={max_seq_len} ===", flush=True)
    try:
        if not run_oom_check:
            if is_host0:
                print("\n=== Startup OOM/speed checks skipped (disabled by default) ===", flush=True)
                print("  Use --oom-check for the startup train_step probe, "
                      "or --speed-check for the post-JIT timing breakdown.",
                      flush=True)
            raise _SkipStartupCheck()

        global_shape = (batch_size, max_seq_len)
        dummy_ids = shard_to_mesh(
            jnp.zeros((per_host_batch, max_seq_len), dtype=jnp.int32),
            data_sharding, global_shape)
        dummy_mask = shard_to_mesh(
            jnp.ones((per_host_batch, max_seq_len), dtype=jnp.int32),
            data_sharding, global_shape)
        rng, dummy_step_rng = jax.random.split(rng)

        if drift_diagnostics_enabled:
            _dummy_op_key_snap = _drift_snap(params)
        else:
            _dummy_op_key_snap = _dummy_drift_snap()

        # First call: JIT compilation (slow)
        jit_start = time.time()
        _dp, _do, dummy_metrics = train_step_fn(
            params, opt_state, dummy_ids, dummy_mask, dummy_step_rng,
            _dummy_op_key_snap, jnp.asarray(0, jnp.int32))
        jax.block_until_ready(dummy_metrics['total_loss'])
        jit_time = time.time() - jit_start
        jit_loss = float(dummy_metrics['total_loss'])
        if is_host0:
            print(f"  JIT compile: {jit_time:.1f}s", flush=True)
            print(f"  train_step OK -- loss={jit_loss:.4f}", flush=True)

        if run_speed_check:
            # Free first step outputs before second call
            del _dp, _do, dummy_metrics

            # Second call: measure actual step time (post-JIT)
            rng, dummy_step_rng2 = jax.random.split(rng)
            step_start = time.time()
            _dp2, _do2, dummy_metrics2 = train_step_fn(
                params, opt_state, dummy_ids, dummy_mask, dummy_step_rng2,
                _dummy_op_key_snap, jnp.asarray(0, jnp.int32))
            jax.block_until_ready(dummy_metrics2['total_loss'])
            step_time = time.time() - step_start
        else:
            step_time = None

        if is_host0:
            if run_speed_check:
                print(f"  Step time: {step_time*1000:.1f}ms/batch", flush=True)
            else:
                print("  Speed check skipped (disabled by default)", flush=True)

            # Show memory usage after JIT compilation
            try:
                mem = jax.local_devices()[0].memory_stats()
                if mem:
                    used = mem.get('bytes_in_use', 0) / 1e9
                    peak = mem.get('peak_bytes_in_use', 0) / 1e9
                    limit = mem.get('bytes_limit', 0) / 1e9
                    print(f"  HBM: {used:.2f}G / {limit:.2f}G (peak={peak:.2f}G, free={limit - used:.2f}G)", flush=True)
            except Exception:
                pass

        # === Step-time breakdown (sharded, 1 layer) ===
        # NOTE: runs on ALL hosts -shard_map/psum require collective participation.
        # Only print statements are guarded by is_host0.
        class _SkipBreakdown(Exception):
            pass

        try:
            if not run_speed_check:
                raise _SkipBreakdown("speed check disabled")
            if is_baseline:
                raise _SkipBreakdown("baseline has no SRW layer breakdown")
            if operation_space_tau_free_enabled:
                raise _SkipBreakdown(
                    "operation-space tau-free path disables legacy DirectTau "
                    "component breakdown")

            _is_sharded = _sharded_fns is not None
            _uses_scan_offset = _is_active_srw_version(model_version)
            if is_host0:
                print(f"\n  === Step-time breakdown (1 layer, "
                      f"{'sharded' if _is_sharded else 'single-device'}) ===",
                      flush=True)

            _v4164_module = __import__(
                _model_registry_entry(model_version)['module'],
                fromlist=['_layer_norm', '_attn_forward', '_rst_forward', '_srw_chunked'])
            _layer_norm, _attn_forward, _rst_forward, _srw_chunked = _v4164_module._layer_norm, _v4164_module._attn_forward, _v4164_module._rst_forward, _v4164_module._srw_chunked

            # Use actual sharded params (no device_get)
            pool_p = params['neuron_pool']
            router_p = params['router']
            block_p = params['block_0']
            d_model = cfg['model']['d_model']
            n_heads = cfg['model']['n_heads']
            n_qk_cfg = cfg['model'].get('n_qk', 1580)
            n_v_cfg = cfg['model'].get('n_v', 2620)
            rd = cfg['model'].get('router_dropout', 0.1)
            dd = cfg['model'].get('dropout', 0.1)
            prof_rng = jax.random.PRNGKey(42)

            # Create properly sharded dummy_x [B, S, D]
            dummy_x_local = jnp.zeros(
                (per_host_batch, max_seq_len, d_model), dtype=jnp.float32)
            x_sharding = NamedSharding(mesh, P('data', None, None))
            global_x_shape = (batch_size, max_seq_len, d_model)
            dummy_x = shard_to_mesh(dummy_x_local, x_sharding, global_x_shape)

            N_RUNS = 5

            def _hbm_gb():
                """Current HBM usage in GB (device 0)."""
                try:
                    mem = jax.local_devices()[0].memory_stats()
                    if mem:
                        return mem.get('bytes_in_use', 0) / 1e9
                except Exception:
                    pass
                return 0.0

            def _peak_hbm_gb():
                """Peak HBM usage in GB (device 0)."""
                try:
                    mem = jax.local_devices()[0].memory_stats()
                    if mem:
                        return mem.get('peak_bytes_in_use', 0) / 1e9
                except Exception:
                    pass
                return 0.0

            def _t(fn, n=N_RUNS):
                """Time a function + measure HBM delta.
                Returns (ms, delta_gb, peak_gb)."""
                r = fn(); jax.block_until_ready(jax.tree.leaves(r))
                del r
                hbm_before = _hbm_gb()
                t0 = time.time()
                for _ in range(n):
                    r = fn(); jax.block_until_ready(jax.tree.leaves(r))
                elapsed = (time.time() - t0) / n * 1000
                hbm_after = _hbm_gb()
                peak = _peak_hbm_gb()
                return (elapsed, hbm_after - hbm_before, peak)

            # --- Jit-compiled component functions for profiling ---
            def _get_param(container, new_key, alias_key):
                return container[new_key] if new_key in container else container[alias_key]

            # 1) LayerNorm
            @jax.jit
            def prof_layernorm(x, scale, bias):
                return _layer_norm(x, scale, bias)

            # 2) Attn router: proj + split + tau
            @jax.jit
            def prof_attn_router(x, router_p):
                h_all = (x @ router_p['proj_attn']['kernel']
                         + router_p['proj_attn']['bias'])
                h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
                h_Q = h_Q / (jnp.linalg.norm(h_Q, axis=-1, keepdims=True) + 1e-8)
                h_K = h_K / (jnp.linalg.norm(h_K, axis=-1, keepdims=True) + 1e-8)
                h_V = h_V / (jnp.linalg.norm(h_V, axis=-1, keepdims=True) + 1e-8)
                if _is_rw_key_srw_version(model_version):
                    def _operator_query(read_query, key):
                        write_query = x @ router_p[key]
                        write_query = write_query / (
                            jnp.linalg.norm(
                                write_query, axis=-1, keepdims=True) + 1e-8)
                        operator_query = read_query * write_query
                        return operator_query / (
                            jnp.linalg.norm(
                                operator_query, axis=-1, keepdims=True) + 1e-8)
                    h_Q = _operator_query(h_Q, 'q_op_write_query_proj')
                    h_K = _operator_query(h_K, 'k_op_write_query_proj')
                    h_V = _operator_query(h_V, 'v_op_write_query_proj')
                tau_all = (x @ router_p.get('tau_attn', router_p.get('raw_tau_attn'))['kernel']
                           + router_p.get('tau_attn', router_p.get('raw_tau_attn'))['bias'])
                if _uses_scan_offset:
                    scan_p = _get_param(
                        router_p, 'raw_scan_offset_attn', 'raw_scan_offset_attn')
                    raw_scan_offset_all = (x @ scan_p['kernel'] + scan_p['bias'])
                else:
                    raw_scan_offset_all = jnp.zeros_like(tau_all)
                return h_Q, h_K, h_V, tau_all, raw_scan_offset_all

            # 3) QK fused shard_map (paired)
            @jax.jit
            def prof_qk_fused(x, h_Q, h_K, qk_norm, tau_all, raw_scan_offset_all, qk_read, qk_write):
                fused_paired = (_sharded_fns.get('attn_qk_paired', _sharded_fns['paired'])
                                if isinstance(_sharded_fns, dict)
                                else _sharded_fns[1])
                h_QK = jnp.stack([h_Q, h_K], axis=2)
                tau_QK = jnp.stack(
                    [tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)
                raw_scan_offset_QK = jnp.stack(
                    [raw_scan_offset_all[:, :, 0:1], raw_scan_offset_all[:, :, 1:2]], axis=2)
                if _uses_scan_offset:
                    results = fused_paired(
                        x, h_QK, qk_norm, tau_QK, raw_scan_offset_QK, qk_read, qk_write)
                else:
                    results = fused_paired(
                        x, h_QK, qk_norm, tau_QK, qk_read, qk_write)
                QK_out, act = results[0], results[1]
                return QK_out[:, :, 0, :], QK_out[:, :, 1, :], act

            # 3b) QK non-sharded fallback
            @jax.jit
            def prof_qk_chunked(x, h_Q, h_K, qk_norm, tau_all, qk_read, qk_write):
                Q, *_ = _srw_chunked(x, h_Q, qk_norm, tau_all[:, :, 0:1],
                                       qk_read, qk_write, n_chunks_qk)
                K, *_ = _srw_chunked(x, h_K, qk_norm, tau_all[:, :, 1:2],
                                       qk_read, qk_write, n_chunks_qk)
                return Q, K

            # 4) V shard_map (single)
            @jax.jit
            def prof_v_sharded(x, h_V, v_norm, tau_v, raw_scan_offset_v, v_read, v_write):
                fused_single = (_sharded_fns.get('attn_v_single', _sharded_fns['single'])
                                if isinstance(_sharded_fns, dict)
                                else _sharded_fns[0])
                if _uses_scan_offset:
                    return fused_single(
                        x, h_V, v_norm, tau_v, raw_scan_offset_v, v_read, v_write)
                return fused_single(
                    x, h_V, v_norm, tau_v, v_read, v_write)

            # 4b) V non-sharded fallback
            @jax.jit
            def prof_v_chunked(x, h_V, v_norm, tau_v, v_read, v_write):
                return _srw_chunked(x, h_V, v_norm, tau_v,
                                    v_read, v_write, n_chunks_v)

            # 5) Self-attention (QK scores + softmax + wV + O_proj)
            @jax.jit
            def prof_self_attn(Q, K, V, Ok):
                B, S, D = Q.shape
                dh = D // n_heads
                Qr = Q.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
                Kr = K.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
                Vr = V.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
                sc = jnp.sqrt(jnp.float32(dh))
                scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / sc
                causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
                scores = jnp.where(causal, scores,
                                   jnp.finfo(scores.dtype).min)
                attn_w = jax.nn.softmax(scores, axis=-1)
                out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
                out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
                return out @ Ok

            # 6) Know router
            @jax.jit
            def prof_rst_router(x, router_p):
                proj_p = _get_param(router_p, 'proj_rst', 'proj_know')
                h = (x @ proj_p['kernel'] + proj_p['bias'])
                h = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
                if _is_rw_key_srw_version(model_version):
                    write_query = x @ router_p['rst_op_write_query_proj']
                    write_query = write_query / (
                        jnp.linalg.norm(
                            write_query, axis=-1, keepdims=True) + 1e-8)
                    h = h * write_query
                    h = h / (
                        jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
                tau_p = _get_param(router_p, 'tau_rst', 'tau_rst')
                tau = (x @ tau_p['kernel'] + tau_p['bias'])
                if _uses_scan_offset:
                    scan_p = _get_param(
                        router_p, 'raw_scan_offset_rst', 'raw_scan_offset_rst')
                    raw_scan_offset = (x @ scan_p['kernel'] + scan_p['bias'])
                else:
                    raw_scan_offset = jnp.zeros_like(tau)
                return h, tau, raw_scan_offset

            # 7) Know shard_map (single)
            @jax.jit
            def prof_rst_sharded(x, h, rst_norm, tau, raw_scan_offset, rst_read, rst_write):
                fused_single = (_sharded_fns.get('rst_single', _sharded_fns['single'])
                                if isinstance(_sharded_fns, dict)
                                else _sharded_fns[0])
                if _uses_scan_offset:
                    return fused_single(
                        x, h, rst_norm, tau, raw_scan_offset, rst_read, rst_write)
                return fused_single(
                    x, h, rst_norm, tau, rst_read, rst_write)

            # 7b) Know non-sharded fallback
            @jax.jit
            def prof_rst_chunked(x, h, rst_norm, tau, rst_read, rst_write):
                return _srw_chunked(x, h, rst_norm, tau,
                                    rst_read, rst_write, n_chunks_rst)

            # --- Prepare intermediate values ---
            pool_select_p = (
                _v4164_module._pool_params_with_operator_keys(pool_p)
                if hasattr(_v4164_module, '_pool_params_with_operator_keys')
                else pool_p)
            if ('attn_qk_read_global' in pool_p
                    and hasattr(_v4164_module, '_visible_pool_params')):
                pool_select_p = _v4164_module._visible_pool_params(
                    pool_select_p,
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32))
            pool_read_p = (
                pool_select_p if 'attn_qk_read' in pool_select_p else pool_p)
            qk_op_key = _get_param(
                pool_select_p, 'attn_qk_op_key',
                'attn_qk_emb' if 'attn_qk_emb' in pool_select_p else 'qk_emb')
            v_op_key = _get_param(
                pool_select_p, 'attn_v_op_key',
                'attn_v_emb' if 'attn_v_emb' in pool_select_p else 'v_emb')
            rst_op_key = _get_param(
                pool_select_p, 'rst_op_key', 'rst_emb')
            qk_read = _get_param(pool_read_p, 'attn_qk_read', 'qk_read')
            qk_write = _get_param(pool_read_p, 'attn_qk_write', 'qk_write')
            v_read = _get_param(pool_read_p, 'attn_v_read', 'v_read')
            v_write = _get_param(pool_read_p, 'attn_v_write', 'v_write')
            rst_read = _get_param(pool_read_p, 'rst_read', 'rst_read')
            rst_write = _get_param(pool_read_p, 'rst_write', 'rst_write')
            qk_norm = qk_op_key / (jnp.linalg.norm(
                qk_op_key, axis=-1, keepdims=True) + 1e-8)
            v_norm = v_op_key / (jnp.linalg.norm(
                v_op_key, axis=-1, keepdims=True) + 1e-8)
            rst_norm = rst_op_key / (jnp.linalg.norm(
                rst_op_key, axis=-1, keepdims=True) + 1e-8)

            normed = prof_layernorm(
                dummy_x, block_p['norm1']['scale'],
                block_p['norm1']['bias'])
            jax.block_until_ready(normed)

            h_Q, h_K, h_V, tau_all, raw_scan_offset_all = prof_attn_router(normed, router_p)
            jax.block_until_ready(tau_all)

            if _is_sharded:
                Q, K, *_ = prof_qk_fused(
                    normed, h_Q, h_K, qk_norm, tau_all, raw_scan_offset_all,
                    qk_read, qk_write)
                V, *_ = prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                    v_read, v_write)
            else:
                Q, K = prof_qk_chunked(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    qk_read, qk_write)
                V, *_ = prof_v_chunked(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    v_read, v_write)
            jax.block_until_ready((Q, K, V))

            h_rst, tau_rst, raw_scan_offset_rst = prof_rst_router(normed, router_p)
            jax.block_until_ready(tau_rst)
            if _is_sharded:
                _kout = prof_rst_sharded(
                    normed, h_rst, rst_norm, tau_rst, raw_scan_offset_rst,
                    rst_read, rst_write)[0]
            else:
                _kout, _, _, _, _, _, _, _ = prof_rst_chunked(
                    normed, h_rst, rst_norm, tau_rst,
                    rst_read, rst_write)
            jax.block_until_ready(_kout)

            # --- Timed + memory measurements ---
            # Each _t() returns (ms, delta_hbm_gb, peak_hbm_gb)
            hbm_before_profile = _hbm_gb()
            items = []  # [(name, ms, delta_gb, peak_gb)]

            ms, dg, pk = _t(lambda: prof_layernorm(
                dummy_x, block_p['norm1']['scale'],
                block_p['norm1']['bias']))
            items.append(("LayerNorm", ms, dg, pk))

            ms, dg, pk = _t(lambda: prof_attn_router(normed, router_p))
            items.append(("A router(proj+tau)", ms, dg, pk))

            if _is_sharded:
                ms, dg, pk = _t(lambda: prof_qk_fused(
                    normed, h_Q, h_K, qk_norm, tau_all, raw_scan_offset_all,
                    qk_read, qk_write))
                items.append(("A QK fused shard", ms, dg, pk))
                ms, dg, pk = _t(lambda: prof_v_sharded(
                    normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                    v_read, v_write))
                items.append(("A V shard", ms, dg, pk))
            else:
                ms, dg, pk = _t(lambda: prof_qk_chunked(
                    normed, h_Q, h_K, qk_norm, tau_all,
                    qk_read, qk_write))
                items.append(("A QK chunked(x2)", ms, dg, pk))
                ms, dg, pk = _t(lambda: prof_v_chunked(
                    normed, h_V, v_norm, tau_all[:, :, 2:3],
                    v_read, v_write))
                items.append(("A V chunked", ms, dg, pk))

            Ok = block_p['attn']['expand_O']['kernel']
            ms, dg, pk = _t(lambda: prof_self_attn(Q, K, V, Ok))
            items.append(("A self-attn(QKV)", ms, dg, pk))

            ms, dg, pk = _t(lambda: prof_layernorm(
                dummy_x, block_p['norm2']['scale'],
                block_p['norm2']['bias']))
            items.append(("LayerNorm (know)", ms, dg, pk))

            ms, dg, pk = _t(lambda: prof_rst_router(normed, router_p))
            items.append(("K router(proj+tau)", ms, dg, pk))

            if _is_sharded:
                ms, dg, pk = _t(lambda: prof_rst_sharded(
                    normed, h_rst, rst_norm, tau_rst, raw_scan_offset_rst,
                    rst_read, rst_write))
                items.append(("K know shard", ms, dg, pk))
            else:
                ms, dg, pk = _t(lambda: prof_rst_chunked(
                    normed, h_rst, rst_norm, tau_rst,
                    rst_read, rst_write))
                items.append(("K know chunked", ms, dg, pk))

            # --- Print breakdown (time + memory) --- host0 only
            if is_host0:
                total_ms = sum(ms for _, ms, _, _ in items)
                max_peak = max(pk for _, _, _, pk in items)
                n_layers = cfg['model']['n_layers']

                try:
                    mem = jax.local_devices()[0].memory_stats()
                    hbm_limit = mem.get('bytes_limit', 0) / 1e9 if mem else 0
                except Exception:
                    hbm_limit = 0

                print(f"\n  === Op breakdown (1 layer fwd, {total_ms:.0f} ms, "
                      f"peak={max_peak:.2f}G) ===", flush=True)
                print(f"    {'Op':22s} {'Time':>8s} {'%':>5s}  "
                      f"{'HBM d':>7s}  {'Peak':>7s}  {''}",
                      flush=True)
                print(f"    {'-'*22} {'-'*8} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*20}",
                      flush=True)
                for name, ms_val, dg_val, pk_val in items:
                    pct = ms_val / total_ms * 100 if total_ms > 0 else 0
                    bar = '#' * int(pct / 2)
                    dg_str = f"{dg_val:+.3f}G" if abs(dg_val) > 0.001 else "     -"
                    print(f"    {name:22s} {ms_val:7.1f}ms {pct:4.0f}%  "
                          f"{dg_str:>7s}  {pk_val:5.2f}G  {bar}",
                          flush=True)

                # Group summaries
                attn_ms = sum(ms for n, ms, _, _ in items if n.startswith('A '))
                rst_ms = sum(ms for n, ms, _, _ in items if n.startswith('K '))
                norm_ms = sum(ms for n, ms, _, _ in items if n.startswith('LayerNorm'))
                print(f"    {'-'*22} {'-'*8}", flush=True)
                print(f"    {'Attention total':22s} {attn_ms:7.1f}ms "
                      f"{attn_ms/total_ms*100:.0f}%", flush=True)
                print(f"    {'Knowledge total':22s} {rst_ms:7.1f}ms "
                      f"{rst_ms/total_ms*100:.0f}%", flush=True)
                print(f"    {'LayerNorm total':22s} {norm_ms:7.1f}ms "
                      f"{norm_ms/total_ms*100:.0f}%", flush=True)
                print(f"    {'Layer total':22s} {total_ms:7.1f}ms", flush=True)
                print(f"    Est. {n_layers}-layer fwd: "
                      f"{total_ms * n_layers:.0f} ms "
                      f"(actual step incl. grad+opt)", flush=True)

                # Overall HBM summary
                hbm_now = _hbm_gb()
                print(f"\n  === HBM Summary (per device) ===", flush=True)
                print(f"    Before profile:         {hbm_before_profile:.2f}G",
                      flush=True)
                print(f"    After profile:          {hbm_now:.2f}G",
                      flush=True)
                print(f"    Peak during profile:    {max_peak:.2f}G",
                      flush=True)
                if hbm_limit > 0:
                    print(f"    Device limit:           {hbm_limit:.2f}G",
                          flush=True)
                    print(f"    Headroom:               "
                          f"{hbm_limit - max_peak:.2f}G "
                          f"({(hbm_limit - max_peak)/hbm_limit*100:.0f}%)",
                          flush=True)

            del normed, h_Q, h_K, h_V, tau_all, Q, K, V
            del h_rst, tau_rst, _kout, dummy_x
        except _SkipBreakdown as e:
            if is_host0:
                print(f"  Breakdown skipped: {e}", flush=True)
        except Exception as e:
            if is_host0:
                import traceback
                print(f"  Breakdown failed: {e}", flush=True)
                traceback.print_exc()

        # Clear XLA compilation cache and free profiling memory
        import gc
        gc.collect()
        jax.clear_caches()

        if is_host0 and run_speed_check:
            # Estimate total training time
            total_steps = len(train_loader) * num_epochs
            remaining_steps = total_steps - global_step
            est_seconds = remaining_steps * step_time
            est_hours = est_seconds / 3600
            print(f"  Estimated time: {est_hours:.1f}h ({remaining_steps:,} steps @ {step_time*1000:.1f}ms)", flush=True)

        del dummy_ids, dummy_mask
        if run_speed_check:
            del _dp2, _do2, dummy_metrics2
        else:
            del _dp, _do, dummy_metrics
        if is_host0:
            print("=== OOM check passed (JIT compiled) ===\n", flush=True)
    except _SkipStartupCheck:
        pass
    except Exception as e:
        if is_host0:
            msg = str(e)
            is_oom = (
                'RESOURCE_EXHAUSTED' in msg
                or 'out of memory' in msg.lower()
                or 'oom' in msg.lower()
            )
            if is_oom:
                print(f"\n  *** OOM check FAILED: {e}")
                print(f"  The model + gradients do not fit in device memory.")
                print(f"  Try: reduce batch_size, enable gradient_checkpointing, or use a smaller model.")
                print_xla_oom_diagnostics()
            else:
                print(f"\n  *** train_step check FAILED: {type(e).__name__}: {e}")
                print("  This is not necessarily OOM; it is a code/runtime error during the dummy train_step.")
        raise

    # ----------------------------------------------------------
    # Training log file (host 0 only)
    # ----------------------------------------------------------
    if is_host0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # On resume, reuse the existing log filenames from the run folder
        # so the session appends to the prior log instead of fragmenting
        # into training_log_<ts1>.txt + training_log_<ts2>.txt + ...
        _existing_logs = sorted(_list_files(log_dir, "training_log_*.txt"))
        _existing_jsonls = sorted(_list_files(log_dir, "metrics_*.jsonl"))
        _is_log_resume = (
            training_log_append_on_resume
            and (resume_path is not None)
            and bool(_existing_logs))
        if _is_log_resume:
            training_log_file = _existing_logs[-1]
            jsonl_log_file = (_existing_jsonls[-1] if _existing_jsonls
                              else _join(log_dir, f'metrics_{timestamp}.jsonl'))
        else:
            training_log_file = _join(log_dir, f'training_log_{timestamp}.txt')
            jsonl_log_file = _join(log_dir, f'metrics_{timestamp}.jsonl')
        # Set up loggers (local append + periodic GCS sync)
        _setup_loggers(
            training_log_file, jsonl_log_file, resume=_is_log_resume)

        n_params = count_parameters(params)
        log_message(f"DAWN {model_version} Training Log (Multi-Host) - {timestamp}")
        log_message(f"Config: {config_path}")
        log_message(f"Parameters: {n_params:,}")
        if _is_rw_key_srw_version(model_version_cfg):
            log_message("RW-key operator path: live-gradient RW keys, "
                        "RW-matched operator queries")
        log_message(f"Hosts: {n_hosts}, Local devices: {n_local_devices}, Total: {jax.device_count()}")
        log_message(f"Total steps: {total_steps}")
        if tau_init_summary is not None:
            log_jsonl(tau_init_summary)
        if selection_calibration_summary is not None:
            log_jsonl(selection_calibration_summary)
        log_message(
            "Resume log append policy: "
            f"training={training_log_append_on_resume}")
        log_message("")
        sync_logs()

    # ----------------------------------------------------------
    # Set data loader resume position
    # ----------------------------------------------------------
    if start_step_in_epoch > 0:
        if is_host0:
            print(f"  Resuming data loader at step_in_epoch={start_step_in_epoch}")
        train_loader.reset(start_step=start_step_in_epoch)

    # ----------------------------------------------------------
    # SIGTERM handler for spot TPU preemption
    # ----------------------------------------------------------
    preemption_requested = [False]  # mutable container for closure

    def handle_preemption(signum, frame):
        """Flag-only SIGTERM handler (spot preemption).

        Saving from a signal handler is unsafe on multi-host; hosts can
        receive SIGTERM at different Python points. We just flag here; the
        main loop cooperatively saves after the inner-loop break.
        """
        if preemption_requested[0]:
            return
        preemption_requested[0] = True
        print(f"\n!!! SIGTERM received (host {host_id}) at step={global_step} -- flagging preemption !!!", flush=True)

    signal.signal(signal.SIGTERM, handle_preemption)
    if is_host0:
        print("  SIGTERM handler registered (spot preemption safety)")

    _rst_final_backend_enabled = False
    if operation_space_tau_free_enabled:
        _opspace_runtime_layouts = operation_space_repack_config.get(
            'operation_space_pool_layouts', {})
        if not isinstance(_opspace_runtime_layouts, dict):
            _opspace_runtime_layouts = {}
        _rst_runtime_layout = _opspace_runtime_layouts.get('rst', {})
        if not isinstance(_rst_runtime_layout, dict):
            _rst_runtime_layout = {}
        _rst_final_backend_enabled = (
            str(_rst_runtime_layout.get('execution_backend', '')).lower()
            == 'sparse_region_block')

    # ----------------------------------------------------------
    if is_host0:
        print(f"\n{'='*60}")
        print("=== Starting training loop ===", flush=True)
        print(f"{'='*60}")

    train_start_time = time.time()
    total_micro_steps = num_epochs * steps_per_epoch
    val_interval = int(cfg['training'].get('val_interval', 5000))
    epoch_step_counter = start_step_in_epoch  # tracks position within current epoch

    # Logging cadence. REGULAR every log_interval steps. ANALYSIS every
    # log_interval * log_analysis_multiplier steps.
    LOG_REGULAR = log_interval
    LOG_ANALYSIS = max(1, log_interval * log_analysis_multiplier)
    LOG_GEOMETRY = max(1, LOG_REGULAR * heavy_geometry_multiplier)
    main_val_path = (
        'operation_space_tau_free_relu'
        if operation_space_tau_free_enabled else (
            'sector_bucketed'
            if hardware_sector_execution_enabled
            else ('block_sparse_fallback'
                  if str(model_version_cfg) == V4168_MODEL_VERSION
                  else 'standard')))
    if is_host0:
        print(f"  Log cadence: regular={LOG_REGULAR}"
              f" analysis={LOG_ANALYSIS}"
              f" geometry={LOG_GEOMETRY}"
              f" val={val_interval}",
              flush=True)

    # Operator-key drift snapshot. Non-compact runs refresh the real snap at
    # log events; compact/baseline runs carry dummy zero leaves.
    if drift_diagnostics_enabled:
        _prev_op_key_snap = _drift_snap(params)
    else:
        _prev_op_key_snap = _dummy_drift_snap()
    _latest_val_dead_stats = None
    _latest_val_dead_step = None

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        # Epoch accumulators on device -one device_get per epoch at the
        # end, rather than per-step float()/int() sync.
        _epoch_loss_jax = jnp.float32(0.0)
        # Keep epoch counts in fp32: JAX x64 is typically disabled on TPU, so
        # jnp.int64 silently truncates to int32 and overflows on multi-billion
        # token epochs.
        _epoch_correct_jax = jnp.float32(0.0)
        _epoch_valid_jax = jnp.float32(0.0)
        epoch_steps = 0

        # Window accumulators on device -one device_get per log boundary.
        _win_loss_jax = jnp.float32(0.0)
        _win_ce_jax = jnp.float32(0.0)
        _win_aux_jax = jnp.float32(0.0)
        _win_tau_reg_jax = jnp.float32(0.0)
        _win_orth_jax = jnp.float32(0.0)
        _win_div_jax = jnp.float32(0.0)
        _win_correct_jax = jnp.int32(0)
        _win_valid_jax = jnp.int32(0)
        _regular_cap_window_jax = _init_update_cap_window_stats()
        win_count = 0
        win_start_time = time.time()

        for local_step, (input_ids, attention_mask) in enumerate(train_loader):

            # Cross-host SIGTERM sync every 10 steps. Handles the case where
            # spot preemption fires on only some hosts first -without this,
            # a flagged host would break while unflagged hosts continue into
            # the next train_step collective and hang. Cost: one bool
            # all-gather per 10 steps (bytes).
            if local_step % 10 == 0:
                _preempt_any = bool(np.any(process_allgather(
                    np.array([preemption_requested[0]], dtype=np.bool_))))
                if _preempt_any and not preemption_requested[0]:
                    preemption_requested[0] = True
                    if is_host0:
                        print(
                            f"Preemption detected on another host"
                            f" (step={global_step}) -- cooperative break.",
                            flush=True,
                        )

            if preemption_requested[0]:
                if is_host0:
                    print("Preemption requested -- exiting training loop.", flush=True)
                break

            # Shard data and run the official v4164 train step.
            rng, step_rng = jax.random.split(rng)
            step_rng = jax.random.fold_in(step_rng, host_id)  # per-host dropout
            input_ids = shard_to_mesh(
                input_ids, data_sharding, (batch_size, max_seq_len))
            attention_mask = shard_to_mesh(
                attention_mask, data_sharding, (batch_size, max_seq_len))

            params, opt_state, metrics = train_step_fn(
                params, opt_state,
                input_ids, attention_mask, step_rng, _prev_op_key_snap,
                jnp.asarray(global_step, jnp.int32))

            step_after_update = global_step + 1
            if hardware_sector_execution_enabled:
                metrics.update({
                    k: jnp.asarray(v, dtype=jnp.float32)
                    for k, v in _v4168_hardware_sector_static_metrics(
                        cfg['model'],
                        model_axis_size=mesh_model,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        data_axis_size=mesh_data,
                        bucketed_execution_enabled=
                        hardware_sector_execution_enabled).items()
                })
            if operation_space_tau_free_enabled:
                metrics.update({
                    k: jnp.asarray(v, dtype=jnp.float32)
                    for k, v in _v4168_operation_space_static_metrics(
                        cfg['model'],
                        config=operation_space_repack_config,
                        model_axis_size=mesh_model).items()
                })
            if _v4168_should_operation_space_repack(
                    step_after_update, operation_space_repack_config):
                params, opt_state, repack_metrics = (
                    _v4168_maybe_operation_space_repack(
                        params, opt_state, cfg['model'], mesh,
                        step_after_update, operation_space_repack_config))
                if repack_metrics:
                    _repack_moved = (
                        float(repack_metrics.get(
                            'repack/total_moved_count', 0.0)) > 0.0)
                    if _repack_moved and drift_diagnostics_enabled:
                        _prev_op_key_snap = _drift_snap(params)
                        repack_metrics['repack/drift_snapshot_refreshed'] = 1.0
                    metrics.update({
                        k: jnp.asarray(v, dtype=jnp.float32)
                        for k, v in repack_metrics.items()
                    })
                    if is_host0:
                        for _pool in ('v', 'rst'):
                            _prefix = f'repack/{_pool}/'
                            if (_prefix + 'perm_checksum') in repack_metrics:
                                log_message(
                                    f"[repack/{_pool}] "
                                    f"step={int(step_after_update)} "
                                    f"candidates={float(repack_metrics.get(_prefix + 'candidates', 0.0)):.0f} "
                                    f"swaps={float(repack_metrics.get(_prefix + 'swaps', 0.0)):.0f} "
                                    f"gain_mean={float(repack_metrics.get(_prefix + 'mean_swap_gain', 0.0)):.6f} "
                                    f"gain_max={float(repack_metrics.get(_prefix + 'max_swap_gain', 0.0)):.6f} "
                                    f"drift_p95_before={float(repack_metrics.get(_prefix + 'drift_p95_before', 0.0)):.6f} "
                                    f"drift_p95_after={float(repack_metrics.get(_prefix + 'drift_p95_after', 0.0)):.6f} "
                                    f"checksum={float(repack_metrics.get(_prefix + 'perm_checksum', 0.0)):.0f}")
                        log_jsonl({
                            'type': 'operation_space_repack',
                            'step': int(step_after_update),
                            'epoch': int(epoch),
                            **{k: float(v)
                               for k, v in repack_metrics.items()},
                            'timestamp': datetime.now().isoformat(),
                        })
                        sync_logs()
            if _v4168_should_hardware_repack(
                    step_after_update, hardware_repack_config):
                params, opt_state, repack_metrics = (
                    _v4168_maybe_hardware_repack(
                        params, opt_state, cfg['model'], mesh,
                        step_after_update, hardware_repack_config))
                if repack_metrics:
                    _repack_moved = (
                        float(repack_metrics.get(
                            'repack/total_moved_count', 0.0)) > 0.0)
                    if _repack_moved and drift_diagnostics_enabled:
                        _prev_op_key_snap = _drift_snap(params)
                        repack_metrics['repack/drift_snapshot_refreshed'] = 1.0
                    metrics.update({
                        k: jnp.asarray(v, dtype=jnp.float32)
                        for k, v in repack_metrics.items()
                    })
                    if is_host0:
                        for _pool in ('attn_v', 'rst'):
                            _prefix = f'repack/{_pool}/'
                            if (_prefix + 'moved_frac') in repack_metrics:
                                log_message(
                                    f"[VQ repack] pool="
                                    f"{'v' if _pool == 'attn_v' else 'rst'} "
                                    f"moved_frac={float(repack_metrics.get(_prefix + 'moved_frac', 0.0)):.6f} "
                                    f"mean_compactness_cos={float(repack_metrics.get(_prefix + 'mean_compactness_cos', 0.0)):.6f} "
                                    f"min_sector_size={float(repack_metrics.get(_prefix + 'min_sector_size', 0.0)):.0f} "
                                    f"max_sector_size={float(repack_metrics.get(_prefix + 'max_sector_size', 0.0)):.0f} "
                                    f"mean_sector_radius={float(repack_metrics.get(_prefix + 'mean_sector_radius', 0.0)):.6f} "
                                    f"max_sector_radius={float(repack_metrics.get(_prefix + 'max_sector_radius', 0.0)):.6f}")
                        log_jsonl({
                            'type': 'hardware_repack',
                            'step': int(step_after_update),
                            'epoch': int(epoch),
                            **{k: float(v)
                               for k, v in repack_metrics.items()},
                            'timestamp': datetime.now().isoformat(),
                        })
                        sync_logs()

            # Scalar helper kept for log-block use (m_grad etc.).
            def _m(v):
                return float(v)

            # Device-side accumulation -no per-step TPU-to-CPU sync on the
            # regression/metric scalars. Window + epoch values are
            # materialized only at log boundary and end of epoch.
            # Token-weighted accumulation: every window/epoch loss is summed
            # as (loss * valid_count) so the final avg divides by total
            # valid tokens, matching evaluate()'s token-level mean. Makes
            # train/val loss directly comparable.
            _valid_f = metrics['valid_count'].astype(jnp.float32)
            _win_loss_jax = _win_loss_jax + metrics['total_loss'] * _valid_f
            _win_ce_jax = _win_ce_jax + metrics['ce_loss'] * _valid_f
            _win_aux_jax = _win_aux_jax + metrics['aux_loss'] * _valid_f
            _win_tau_reg_jax = _win_tau_reg_jax + metrics.get('tau_reg', jnp.float32(0.0)) * _valid_f
            _win_orth_jax = _win_orth_jax + metrics['orth_loss'] * _valid_f
            _win_div_jax = _win_div_jax + metrics['div_loss'] * _valid_f
            _win_correct_jax = _win_correct_jax + metrics['correct']
            _win_valid_jax = _win_valid_jax + metrics['valid_count']
            _regular_cap_window_jax = _accumulate_update_cap_window_stats(
                _regular_cap_window_jax, metrics)

            _epoch_loss_jax = _epoch_loss_jax + metrics['ce_loss'] * _valid_f
            _epoch_correct_jax = (
                _epoch_correct_jax + metrics['correct'].astype(jnp.float32))
            _epoch_valid_jax = (
                _epoch_valid_jax + metrics['valid_count'].astype(jnp.float32))

            win_count += 1
            epoch_steps += 1

            # Per-step NaN check on total_loss only. A single scalar sync
            # catches loss explosions immediately; the full 6-key check runs
            # at log boundary on already-materialized window averages.
            _m_total_for_nan = float(metrics['total_loss'])
            if not np.isfinite(_m_total_for_nan):
                raise ValueError(
                    f"NaN/INF total_loss at epoch {epoch}, step {global_step + 1}")
            if _rst_final_backend_enabled:
                _rst_final_guard = jax.device_get({
                    'semantic_drop_frac': metrics.get(
                        'opspace/rst/semantic_drop_frac',
                        metrics.get(
                            'opspace/rst/final/semantic_drop_frac',
                            jnp.float32(0.0))),
                    'all_processed': metrics.get(
                        'opspace/rst/all_processed', jnp.float32(1.0)),
                    'no_nan': metrics.get(
                        'opspace/rst/no_nan',
                        metrics.get(
                            'opspace/rst/final/no_nan',
                            jnp.float32(1.0))),
                    'unresolved_count': metrics.get(
                        'opspace/rst/final/unresolved_count',
                        jnp.float32(0.0)),
                    'assignment_collision_count': metrics.get(
                        'opspace/rst/assignment_collision_count',
                        metrics.get(
                            'opspace/rst/final/assignment_collision_count',
                            jnp.float32(0.0))),
                })
                _rst_semantic_drop = float(
                    _rst_final_guard['semantic_drop_frac'])
                _rst_all_processed = float(
                    _rst_final_guard['all_processed'])
                _rst_collision_count = float(
                    _rst_final_guard['assignment_collision_count'])
                _rst_no_nan = float(_rst_final_guard['no_nan'])
                if (_rst_semantic_drop > 0.0
                        or _rst_all_processed < 1.0
                        or _rst_collision_count > 0.0
                        or _rst_no_nan < 1.0):
                    raise RuntimeError(
                        "region/block RST fail-loud guard "
                        f"at epoch {epoch}, step {global_step + 1}: "
                        f"opspace/rst/semantic_drop_frac="
                        f"{_rst_semantic_drop:.9g}, "
                        f"opspace/rst/all_processed="
                        f"{_rst_all_processed:.9g}, "
                        f"opspace/rst/no_nan={_rst_no_nan:.9g}, "
                        "opspace/rst/final/"
                        "unresolved_count="
                        f"{float(_rst_final_guard['unresolved_count']):.9g}, "
                        "opspace/rst/final/assignment_collision_count="
                        f"{_rst_collision_count:.9g}")

            global_step += 1
            epoch_step_counter += 1

            # ---- REGULAR periodic logging ----
            # ANALYSIS is driven from the val path (below), not from here -
            # the ANALYSIS stats now require a separate forward with the
            # full-stats kernels and only run on val ticks.
            _is_early_log = global_step in (1, 5, 10, 20, 50)
            is_regular = (global_step % LOG_REGULAR == 0) or _is_early_log

            if is_regular:
                # Refresh the real op-key drift snapshot on every host when
                # diagnostics are enabled; compact/baseline keep the dummy snap.
                if drift_diagnostics_enabled:
                    _prev_op_key_snap = _drift_snap(params)
                _raw_step_time_window = time.time() - win_start_time
                _regular_logging_t0 = time.time()
                # One TPU-to-CPU sync for the whole window.
                _win_vals = jax.device_get({
                    'loss': _win_loss_jax, 'ce': _win_ce_jax,
                    'aux': _win_aux_jax, 'tau_reg': _win_tau_reg_jax,
                    'orth': _win_orth_jax, 'div': _win_div_jax,
                    'correct': _win_correct_jax, 'valid': _win_valid_jax,
                })
                _win_correct_py = int(_win_vals['correct'])
                _win_valid_py = int(_win_vals['valid'])
                _vdiv = _win_valid_py if _win_valid_py > 0 else 1
                win_avgs = {
                    'loss':    float(_win_vals['loss'])    / _vdiv,
                    'ce':      float(_win_vals['ce'])      / _vdiv,
                    'aux':     float(_win_vals['aux'])     / _vdiv,
                    'tau_reg': float(_win_vals['tau_reg']) / _vdiv,
                    'orth':    float(_win_vals['orth'])    / _vdiv,
                    'div':     float(_win_vals['div'])     / _vdiv,
                    'acc':     _win_correct_py             / _vdiv,
                }
                # Full NaN/INF check on the materialized window averages.
                if check_nan_inf({
                    'total_loss': win_avgs['loss'], 'ce_loss': win_avgs['ce'],
                    'aux_loss': win_avgs['aux'], 'tau_reg': win_avgs['tau_reg'],
                    'orth_loss': win_avgs['orth'], 'div_loss': win_avgs['div'],
                }, global_step, epoch):
                    raise ValueError(
                        f"NaN/INF window averages at epoch {epoch}, step {global_step}")

                if is_host0:
                    _elapsed = time.time() - win_start_time
                    _steps_per_sec = (win_count / _elapsed) if _elapsed > 0 else 0.0
                    _opt_step = global_step // grad_accum_steps
                    _current_lr = float(schedule(_opt_step))
                    _total_elapsed = time.time() - train_start_time
                    _epoch_elapsed = time.time() - epoch_start
                    _progress = (global_step / total_micro_steps * 100
                                 if total_micro_steps > 0 else 0.0)
                    _s_per_it = _epoch_elapsed / epoch_steps if epoch_steps > 0 else 0.0
                    # ETA based on absolute epoch position so resume mid-epoch
                    # doesn't over-estimate (epoch_steps counts only this
                    # run's steps; epoch_step_counter starts from
                    # start_step_in_epoch).
                    _remaining = max(steps_per_epoch - epoch_step_counter, 0)
                    _eta = _s_per_it * _remaining
                    ctx = {
                        'lb_weight': lb_weight,
                        'tau_reg_weight': tau_reg_weight,
                        'orth_weight': orth_weight,
                        'div_weight': div_weight,
                        'dead_penalty_weight': dead_penalty_weight,
                        'dead_exposure_target': dead_exposure_target,
                        'cb1a_enabled': bool(cb1a_enabled),
                        'n_qk_cfg': cfg['model'].get(
                            'n_qk', cfg['model'].get('n_q', 0)),
                        'n_v_cfg': cfg['model'].get('n_v', 0),
                        'n_rst_cfg': cfg['model'].get(
                            'n_rst', cfg['model'].get('n_know', 0)),
                        'd_model_cfg': cfg['model'].get('d_model', 0),
                        'n_layers_cfg': cfg['model'].get('n_layers', 0),
                        'current_lr': _current_lr,
                        'steps_per_sec': _steps_per_sec,
                        'total_elapsed': _total_elapsed,
                        'epoch_elapsed': _epoch_elapsed,
                        'eta': _eta,
                        's_per_it': _s_per_it,
                        'total_micro_steps': total_micro_steps,
                        'progress': _progress,
                        'model_version': model_version,
                        'operation_space_enabled':
                            operation_space_tau_free_enabled,
                        'regular_console_level': regular_console_level,
                        'regular_console_host_timing':
                            regular_console_host_timing,
                        'regular_console_top1_warn':
                            regular_console_top1_warn,
                        'regular_console_drive_max_warn':
                            regular_console_drive_max_warn,
                        'regular_console_logging_overhead_warn':
                            regular_console_logging_overhead_warn,
                                'soft_gate_schedule': soft_gate_schedule,
                        'soft_gate_t_power': soft_gate_t_power,
                        'soft_gate_t_gompertz_center':
                            soft_gate_t_gompertz_center,
                        'soft_gate_t_gompertz_steepness':
                            soft_gate_t_gompertz_steepness,
                        'soft_gate_boundary_power_start':
                            soft_gate_boundary_power_start,
                        'soft_gate_boundary_power_mid':
                            soft_gate_boundary_power_mid,
                        'soft_gate_boundary_power_final':
                            soft_gate_boundary_power_final,
                        'soft_gate_boundary_power_mid_frac':
                            soft_gate_boundary_power_mid_frac,
                        'soft_gate_boundary_power_final_frac':
                            soft_gate_boundary_power_final_frac,
                        'intensity_beta': (
                            0.0
                            if _is_active_srw_version(model_version_cfg)
                            else float(tcfg.get('intensity_beta', 0.0))),
                        'd_select': (
                            0
                            if _is_active_srw_version(model_version_cfg)
                            else int(cfg['model'].get('d_select', 0) or 0)),
                        'd_intensity': (
                            0
                            if _is_active_srw_version(model_version_cfg)
                            else int(
                                cfg['model'].get('d_route', 0)
                                - cfg['model'].get('d_select', 0))),
                        'd_route_unified': (
                            int(cfg['model'].get('d_route', 0))
                            if _is_active_srw_version(model_version_cfg)
                            else 0),
                    }
                    rec = _build_regular_record(metrics, win_avgs, ctx, global_step, epoch)
                    rec = _attach_update_cap_window_stats(
                        rec, jax.device_get(_regular_cap_window_jax))
                    rec['raw_step_time_window'] = float(_raw_step_time_window)
                    rec['logging_time'] = 0.0
                    _print_regular_block(rec, ctx)
                    rec.pop('_active_tau_regular_available', None)
                    log_jsonl({'type': 'train', **rec})
                    sync_logs()
                    _regular_logging_time = time.time() - _regular_logging_t0
                    rec['logging_time'] = float(_regular_logging_time)
                    _print_regular_host_timing(
                        _raw_step_time_window, _regular_logging_time, ctx)
                    log_jsonl({
                        'type': 'train_timing',
                        'step': int(global_step),
                        'epoch': int(epoch),
                        'raw_step_time_window': float(_raw_step_time_window),
                        'logging_time': float(_regular_logging_time),
                        'timestamp': datetime.now().isoformat(),
                    })
                    sync_logs()

                # Reset window accumulators (all hosts)
                _win_loss_jax = jnp.float32(0.0)
                _win_ce_jax = jnp.float32(0.0)
                _win_aux_jax = jnp.float32(0.0)
                _win_tau_reg_jax = jnp.float32(0.0)
                _win_orth_jax = jnp.float32(0.0)
                _win_div_jax = jnp.float32(0.0)
                _win_correct_jax = jnp.int32(0)
                _win_valid_jax = jnp.int32(0)
                _regular_cap_window_jax = _init_update_cap_window_stats()
                win_count = 0
                win_start_time = time.time()
            # ---- Mid-epoch validation (all hosts run eval, host 0 saves/logs) ----
            _do_val = (global_step % val_interval == 0 and global_step > 0)
            _do_analysis = (global_step % LOG_ANALYSIS == 0 and global_step > 0)
            _do_geometry = (global_step % LOG_GEOMETRY == 0 and global_step > 0)
            _do_ckpt = (global_step % ckpt_interval == 0 and global_step > 0)
            _new_best = False

            if _do_val:
                if is_host0:
                    log_message(f"\n  Mid-epoch validation at step {global_step}...")
                val_loader.reset()
                val_loss, val_acc, val_dead_stats = evaluate(
                    eval_step_fn, params, val_loader, n_local_devices,
                    verbose=is_host0, data_sharding_spec=data_sharding,
                    return_dead_stats=True, current_step=global_step)
                _latest_val_dead_stats = val_dead_stats
                _latest_val_dead_step = global_step
                prune_eval_log = {}
                if eval_prune_step_fns:
                    prune_eval_log = run_eval_prune_sweep(
                        eval_prune_step_fns, params, val_loader,
                        n_local_devices, data_sharding, global_step,
                        val_loss, val_acc, verbose=False)
                if is_host0:
                    _val_dead_ctx = {
                        'n_qk_cfg': cfg['model'].get(
                            'n_qk', cfg['model'].get('n_q', 0)),
                        'n_v_cfg': cfg['model'].get('n_v', 0),
                        'n_rst_cfg': cfg['model'].get(
                            'n_rst', cfg['model'].get('n_know', 0)),
                    }
                    val_dead_log = _attach_validation_dead_fractions(
                        dict(val_dead_stats), _val_dead_ctx)
                    log_message(
                        f"  Val path={main_val_path}, "
                        f"Val loss={val_loss:.4f}, Val acc={val_acc:.4f}")
                    if prune_eval_log:
                        for _eps in eval_effective_prune_eps_list:
                            _tag = _format_prune_eps(_eps)
                            log_message(
                                f"  Pruned eval eps={float(_eps):.0e}: "
                                f"loss={prune_eval_log.get('val_loss_prune_eps_' + _tag, 0.0):.4f} "
                                f"delta_loss={prune_eval_log.get('val_loss_delta_prune_eps_' + _tag, 0.0):+.4f} "
                                f"acc={prune_eval_log.get('val_acc_prune_eps_' + _tag, 0.0):.4f} "
                                f"compute={prune_eval_log.get('estimated_compute_frac_prune_eps_' + _tag, 0.0):.4f} "
                                f"mass={prune_eval_log.get('gate_mass_retained_prune_eps_' + _tag, 0.0):.4f}")
                    if not (_do_analysis and analysis_step_fn is not None):
                        _print_validation_dead_stats(val_dead_log, _val_dead_ctx)
                    log_jsonl({
                        'type': 'val',
                        'step': global_step,
                        'epoch': epoch,
                        'main_val_path': main_val_path,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        **val_dead_log,
                        **prune_eval_log,
                        'timestamp': datetime.now().isoformat(),
                    })
                if np.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _new_best = True


            # ---- ANALYSIS: run full-stats forward on one val batch ----
            # Single analysis forward at the configured analysis cadence. Compiles
            # once on first call (extra HBM + time logged). Result dict
            # is released after the JSONL write so HBM snaps back.
            if _do_analysis and analysis_step_fn is not None:
                val_loader.reset()
                _analysis_batch = None
                for _ab_ids, _ab_mask in val_loader:
                    _analysis_batch = (_ab_ids, _ab_mask)
                    break
                if _analysis_batch is not None:
                    _a_ids, _a_mask = _analysis_batch
                    _a_gb = _a_ids.shape[0] * jax.process_count()
                    _a_gs = (_a_gb, _a_ids.shape[1])
                    _a_ids = shard_to_mesh(_a_ids, data_sharding, _a_gs)
                    _a_mask = shard_to_mesh(_a_mask, data_sharding, _a_gs)
                    try:
                        _a_compile_start = time.time()
                        analysis_result = analysis_step_fn(
                            params, _a_ids, _a_mask, jnp.int32(global_step))
                        # Force the computation so HBM usage of the
                        # analysis kernels registers now, not on the
                        # next Python line.
                        jax.block_until_ready(
                            analysis_result.get('aux_loss',
                                                jnp.float32(0.0)))
                        _a_elapsed = time.time() - _a_compile_start
                        if is_host0:
                            _ctx_a = {
                                'n_qk_cfg': cfg['model'].get(
                                    'n_qk', cfg['model'].get('n_q', 0)),
                                'n_v_cfg': cfg['model'].get('n_v', 0),
                                'n_rst_cfg': cfg['model'].get(
                                    'n_rst', cfg['model'].get('n_know', 0)),
                                'd_model_cfg': cfg['model'].get('d_model', 0),
                                'n_layers_cfg': cfg['model'].get(
                                    'n_layers', 0),
                                'current_lr': float(schedule(global_step // grad_accum_steps)),
                                'model_version': model_version,
                            }
                            analysis_payload = dict(analysis_result)
                            if _latest_val_dead_step == global_step \
                                    and _latest_val_dead_stats is not None:
                                analysis_payload.update(_latest_val_dead_stats)
                            for _pool in ('qk', 'v', 'know'):
                                for _part in ('emb', 'read', 'write'):
                                    _key = f'{_pool}_{_part}_grad_ratio'
                                    analysis_payload[_key] = metrics.get(
                                        _key, jnp.float32(0.0))
                            a_rec = _build_analysis_record(
                                {}, analysis_payload, _ctx_a)
                            a_rec['step'] = global_step
                            a_rec['epoch'] = epoch
                            a_rec['analysis_step_sec'] = float(_a_elapsed)
                            if _do_analysis:
                                _print_analysis_block(a_rec, _ctx_a)
                                log_jsonl({'type': 'train_analysis', **a_rec})
                            sync_logs()
                    finally:
                        # Explicit release -jit-returned dict holds
                        # TPU buffers that outlive the val block
                        # otherwise.
                        try:
                            del analysis_result
                        except NameError:
                            pass
                        del _a_ids, _a_mask, _analysis_batch

            if _do_geometry and geometry_step_fn is not None:
                try:
                    geom = geometry_step_fn(params)
                    jax.block_until_ready(
                        geom.get(
                            'attn_qk_op_key_geom_rank',
                            geom.get('attn_qk_emb_geom_rank',
                                     jnp.float32(0.0))))
                    if is_host0:
                        geom_host = jax.device_get(geom)
                        log_message("  Rare geometry diagnostics:")
                        _print_geometry_block(geom_host)
                        log_jsonl({
                            'type': 'geometry',
                            'step': global_step,
                            'epoch': epoch,
                            **{k: float(v) for k, v in geom_host.items()},
                            'timestamp': datetime.now().isoformat(),
                        })
                        sync_logs()
                finally:
                    try:
                        del geom
                    except NameError:
                        pass

            # ---- Split Orbax save paths ----
            if _do_ckpt:
                saved = save_orbax_checkpoint(
                    latest_checkpoint_manager,
                    params, opt_state, rng,
                    epoch, global_step, epoch_step_counter,
                    steps_per_epoch, best_val_loss,
                    cfg['model'], training_config,
                    full_config_snapshot, raw_config_snapshot,
                    config_path, run_id,
                    'regular',
                    val_loss=None,
                    git_info=checkpoint_git_info,
                )
                if is_host0 and saved:
                    log_message(
                        f"  Orbax latest checkpoint saved at "
                        f"step {global_step}")
            if _new_best:
                save_orbax_checkpoint(
                    best_checkpoint_manager,
                    params, opt_state, rng,
                    epoch, global_step, epoch_step_counter,
                    steps_per_epoch, best_val_loss,
                    cfg['model'], training_config,
                    full_config_snapshot, raw_config_snapshot,
                    config_path, run_id,
                    'best',
                    val_loss=val_loss,
                    git_info=checkpoint_git_info,
                )
                if is_host0:
                    log_message(
                        f"  New best Orbax checkpoint saved at "
                        f"step {global_step}! "
                        f"val_loss={best_val_loss:.4f}")

        if preemption_requested[0]:
            # Cooperative emergency save. Previously this ran from the
            # SIGTERM signal handler, which was unsafe because hosts enter
            # the handler asynchronously.
            try:
                save_orbax_checkpoint(
                    latest_checkpoint_manager,
                    params, opt_state, rng,
                    epoch, global_step, epoch_step_counter,
                    steps_per_epoch, best_val_loss,
                    cfg['model'], training_config,
                    full_config_snapshot, raw_config_snapshot,
                    config_path, run_id,
                    'emergency',
                    git_info=checkpoint_git_info,
                    wait=True,
                )
                if is_host0:
                    print(
                        f"!!! Emergency Orbax latest checkpoint saved at "
                        f"step {global_step} !!!",
                        flush=True)
            except Exception as e:
                if is_host0:
                    print(f"!!! Emergency save FAILED: {e} !!!", flush=True)
            break

        # ---- End of epoch ----
        epoch_elapsed = time.time() - epoch_start
        # Single TPU-to-CPU sync for the whole epoch totals.
        _ep = jax.device_get({
            'loss': _epoch_loss_jax,
            'correct': _epoch_correct_jax,
            'valid': _epoch_valid_jax,
        })
        epoch_loss = float(_ep['loss'])
        epoch_correct = float(_ep['correct'])
        epoch_valid = float(_ep['valid'])
        epoch_avg_loss = epoch_loss / epoch_valid if epoch_valid > 0 else 0.0
        epoch_avg_acc = epoch_correct / epoch_valid if epoch_valid > 0 else 0.0

        if is_host0:
            log_message(
                f"\n{'='*60}\n"
                f"Epoch {epoch} complete in {format_time(epoch_elapsed)}\n"
                f"  Train loss={epoch_avg_loss:.4f}, Train acc={epoch_avg_acc:.4f}\n"
                f"{'='*60}"
            )

        # End-of-epoch validation (all hosts must participate in eval)
        if is_host0:
            log_message("  Running end-of-epoch validation...")
        val_loader.reset()
        val_loss, val_acc, val_dead_stats = evaluate(
            eval_step_fn, params, val_loader, n_local_devices,
            verbose=is_host0, data_sharding_spec=data_sharding,
            return_dead_stats=True, current_step=global_step)
        _latest_val_dead_stats = val_dead_stats
        _latest_val_dead_step = global_step
        prune_eval_log = {}
        if eval_prune_step_fns:
            prune_eval_log = run_eval_prune_sweep(
                eval_prune_step_fns, params, val_loader, n_local_devices,
                data_sharding, global_step, val_loss, val_acc, verbose=False)

        is_best = np.isfinite(val_loss) and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if is_host0:
            _val_dead_ctx = {
                'n_qk_cfg': cfg['model'].get(
                    'n_qk', cfg['model'].get('n_q', 0)),
                'n_v_cfg': cfg['model'].get('n_v', 0),
                'n_rst_cfg': cfg['model'].get(
                    'n_rst', cfg['model'].get('n_know', 0)),
            }
            val_dead_log = _attach_validation_dead_fractions(
                dict(val_dead_stats), _val_dead_ctx)
            log_message(
                f"  Val path={main_val_path}, "
                f"Val loss={val_loss:.4f}, Val acc={val_acc:.4f}")
            if prune_eval_log:
                for _eps in eval_effective_prune_eps_list:
                    _tag = _format_prune_eps(_eps)
                    log_message(
                        f"  Pruned eval eps={float(_eps):.0e}: "
                        f"loss={prune_eval_log.get('val_loss_prune_eps_' + _tag, 0.0):.4f} "
                        f"delta_loss={prune_eval_log.get('val_loss_delta_prune_eps_' + _tag, 0.0):+.4f} "
                        f"acc={prune_eval_log.get('val_acc_prune_eps_' + _tag, 0.0):.4f} "
                        f"compute={prune_eval_log.get('estimated_compute_frac_prune_eps_' + _tag, 0.0):.4f} "
                        f"mass={prune_eval_log.get('gate_mass_retained_prune_eps_' + _tag, 0.0):.4f}")
            _print_validation_dead_stats(val_dead_log, _val_dead_ctx)
            log_jsonl({
                'type': 'val_epoch',
                'step': global_step,
                'epoch': epoch,
                'main_val_path': main_val_path,
                'val_loss': val_loss,
                'val_acc': val_acc,
                **val_dead_log,
                **prune_eval_log,
                'train_loss': epoch_avg_loss,
                'train_acc': epoch_avg_acc,
                'epoch_time': epoch_elapsed,
                'timestamp': datetime.now().isoformat(),
            })

        save_orbax_checkpoint(
            latest_checkpoint_manager,
            params, opt_state, rng,
            epoch + 1, global_step, 0,
            steps_per_epoch, best_val_loss,
            cfg['model'], training_config,
            full_config_snapshot, raw_config_snapshot,
            config_path, run_id,
            'epoch',
            val_loss=val_loss,
            train_loss=epoch_avg_loss,
            git_info=checkpoint_git_info,
        )
        if is_best:
            save_orbax_checkpoint(
                best_checkpoint_manager,
                params, opt_state, rng,
                epoch + 1, global_step, 0,
                steps_per_epoch, best_val_loss,
                cfg['model'], training_config,
                full_config_snapshot, raw_config_snapshot,
                config_path, run_id,
                'best',
                val_loss=val_loss,
                train_loss=epoch_avg_loss,
                git_info=checkpoint_git_info,
            )
        if is_host0:
            log_message(
                f"  Epoch Orbax latest checkpoint saved at "
                f"step {global_step}")
            if is_best:
                log_message(
                    f"  New best Orbax checkpoint saved at "
                    f"step {global_step}! "
                    f"val_loss={best_val_loss:.4f}")
            log_message(f"  Best val loss so far: {best_val_loss:.4f}")
            sync_logs()

        # Reset data loader for next epoch (no re-read, just reset position)
        if epoch < num_epochs - 1:
            train_loader.reset(start_step=0)
            epoch_step_counter = 0

    # ----------------------------------------------------------
    # Done
    # ----------------------------------------------------------
    if not preemption_requested[0]:
        final_epoch = (
            int(epoch + 1) if 'epoch' in locals() else int(start_epoch))
        try:
            save_orbax_checkpoint(
                latest_checkpoint_manager,
                params, opt_state, rng,
                final_epoch, global_step, epoch_step_counter,
                steps_per_epoch, best_val_loss,
                cfg['model'], training_config,
                full_config_snapshot, raw_config_snapshot,
                config_path, run_id,
                'final',
                git_info=checkpoint_git_info,
                wait=True,
            )
            if is_host0:
                log_message(
                    f"  Final Orbax latest checkpoint saved at "
                    f"step {global_step}")
        except Exception as e:
            if is_host0:
                print(f"  Warning: final Orbax checkpoint failed: {e}",
                      flush=True)

    total_time = time.time() - train_start_time
    if is_host0:
        log_message(
            f"\n{'='*60}\n"
            f"Training complete!\n"
            f"  Total time: {format_time(total_time)}\n"
            f"  Best val loss: {best_val_loss:.4f}\n"
            f"  Final step: {global_step}\n"
            f"{'='*60}"
        )
        sync_logs()

    if latest_checkpoint_manager is not None:
        latest_checkpoint_manager.close()
    if best_checkpoint_manager is not None:
        best_checkpoint_manager.close()


if __name__ == '__main__':
    main()
