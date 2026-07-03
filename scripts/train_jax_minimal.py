"""
Minimal DAWN-SRW v4.1.6.6/v4.1.6.7/v4.1.6.8 JAX trainer.

This path is dedicated to large v4166/v4167/v4168 CE training.  It reuses the full
trainer's config, selection calibration, sharding, optimizer, data, logging,
and Orbax helpers, but avoids analysis/geometry/prune/drift diagnostics and
uses model.apply(..., minimal_train=True).
"""

import argparse
import inspect
import json
import math
import random
import signal
import socket
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding, PartitionSpec as P

import train_jax as full


OFFICIAL_MODEL_VERSION = full.OFFICIAL_MODEL_VERSION
POOL_SCHEDULE_NAMES = ('qk', 'v', 'rst')
SUPPORTED_MINIMAL_SRW_VERSIONS = (
    full.V4166_MODEL_VERSION,
    full.V4167_MODEL_VERSION,
    full.V4168_MODEL_VERSION,
)

debug_interval = 0
debug_enabled = False


def _stage_log(stage, extra=None, *, force=False):
    if not (debug_enabled or force):
        return
    msg = {
        "stage": stage,
        "hostname": socket.gethostname(),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "time": datetime.now().isoformat(),
    }
    if extra:
        msg.update(extra)
    print("[minimal-stage] " + json.dumps(msg, sort_keys=True), flush=True)


def _quiet_multihost_barrier(name):
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices(name)


def _stage_barrier(stage):
    name = f"minimal:{stage}"
    context = {
        "hostname": socket.gethostname(),
        "process_index": int(jax.process_index()),
        "trainer": "train_jax_minimal.py",
    }
    _stage_log(f"enter_barrier:{stage}")
    try:
        if debug_enabled:
            full._strict_multihost_barrier(name, context=context)
        else:
            _quiet_multihost_barrier(name)
    except Exception as exc:
        _stage_log(
            f"FAILED barrier:{stage}",
            {**context, "barrier": name, "error": str(exc)},
            force=True)
        raise
    _stage_log(f"passed_barrier:{stage}")


def _model_accepts_minimal_train(model):
    try:
        sig = inspect.signature(model.__call__)
    except (TypeError, ValueError):
        return False
    return 'minimal_train' in sig.parameters


def _broadcast_str_from_host0(s, max_len=65536):
    if s is None:
        s = ''
    encoded = s.encode('utf-8')
    if len(encoded) > max_len:
        raise ValueError(f"broadcast string too long: {len(encoded)} > {max_len}")
    buf = np.zeros(max_len, dtype=np.uint8)
    if jax.process_index() == 0:
        buf[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    if getattr(full, '_HAVE_BROADCAST', False):
        out = np.asarray(full._bcast_one_to_all(buf))
    else:
        gathered = np.asarray(process_allgather(buf))
        out = gathered[:max_len] if gathered.ndim == 1 else gathered[0]
    result = bytes(out).rstrip(b'\x00').decode('utf-8')
    return result if result else None


def _list_run_folders(base):
    if full._is_gcs(base):
        fs = full._get_gcs_fs()
        if fs is None:
            raise ImportError(f"Cannot list GCS path {base}: gcsfs not available.")
        bucket_path = base.replace('gs://', '').rstrip('/')
        try:
            entries = fs.ls(bucket_path)
        except FileNotFoundError:
            return []
        return sorted(['gs://' + e for e in entries if '/run_' in e])
    p = Path(base)
    if not p.exists():
        return []
    return sorted(str(d) for d in p.iterdir()
                  if d.is_dir() and d.name.startswith('run_'))


def _resolve_or_create_run(cfg, cli_args, is_host0):
    base_checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints_jax')
    full._makedirs(base_checkpoint_dir)
    configured_resume_from = (
        cli_args.resume_from
        or cfg.get('training', {}).get('resume_from')
        or cfg.get('resume_from'))

    resume_path = None
    checkpoint_dir = None
    resume_step = None

    if not cli_args.from_scratch:
        if configured_resume_from:
            full._parse_orbax_resume_target(configured_resume_from)
        host0_resume_path = None
        host0_checkpoint_dir = None
        host0_resume_step = None
        host0_explicit_missing = False
        host0_error = None

        if is_host0:
            if configured_resume_from:
                try:
                    folder, selected_step, found = full._resolve_orbax_resume_from(
                        configured_resume_from)
                except RuntimeError as exc:
                    host0_error = str(exc)
                else:
                    if found:
                        host0_resume_path = folder
                        host0_checkpoint_dir = folder
                        host0_resume_step = int(selected_step)
                        print(f"  Resume from specified folder: {folder}")
                        print(f"  Resuming from Orbax step: {host0_resume_step}")
                    else:
                        host0_explicit_missing = True
            else:
                for folder in reversed(_list_run_folders(base_checkpoint_dir)):
                    selected_step = full._latest_orbax_step_for_run(folder)
                    print(
                        "Auto-resume discovery: "
                        f"folder={folder} latest_step={selected_step}",
                        flush=True)
                    if selected_step is not None:
                        host0_resume_path = folder
                        host0_checkpoint_dir = folder
                        host0_resume_step = int(selected_step)
                        break

        resume_path = _broadcast_str_from_host0(host0_resume_path)
        checkpoint_dir = _broadcast_str_from_host0(host0_checkpoint_dir)
        resume_step_s = _broadcast_str_from_host0(
            '' if host0_resume_step is None else str(host0_resume_step))
        resume_step = int(resume_step_s) if resume_step_s else None
        error = _broadcast_str_from_host0(host0_error, max_len=4096)
        if error:
            raise RuntimeError(error)
        missing = _broadcast_str_from_host0(
            'MISSING' if host0_explicit_missing else '')
        if missing == 'MISSING':
            raise FileNotFoundError(
                f"No Orbax checkpoint found in {configured_resume_from}")

    if checkpoint_dir is None:
        host0_checkpoint_dir = None
        if is_host0:
            kst = timezone(timedelta(hours=9))
            ts = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
            rand_suffix = random.randint(1000, 9999)
            version = cfg['model'].get(
                'model_version', full.OFFICIAL_MODEL_VERSION)
            run_name = f"run_v{version}_{ts}_{rand_suffix}"
            host0_checkpoint_dir = full._join_path(
                base_checkpoint_dir, run_name)
        checkpoint_dir = _broadcast_str_from_host0(
            host0_checkpoint_dir, max_len=2048)
        if checkpoint_dir is None:
            raise RuntimeError("Failed to broadcast new run folder from host 0.")
        full._makedirs(checkpoint_dir)
        if is_host0:
            if cli_args.from_scratch:
                print("  Starting from scratch (--from-scratch).")
            print(f"  New run folder: {checkpoint_dir}")

    return resume_path, checkpoint_dir, resume_step


def _maybe_load_resume_config(cfg, checkpoint_dir, resume_step, is_host0):
    if resume_step is None:
        return cfg, None
    try:
        checkpoint_metadata = full._restore_orbax_metadata(
            full._join_path(checkpoint_dir, 'checkpoints'), resume_step)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read resume checkpoint metadata. Automatic config "
            "fallback is disabled; cannot resume deterministically."
        ) from exc

    checkpoint_full_config = checkpoint_metadata.get('full_config')
    checkpoint_raw_config = checkpoint_metadata.get('raw_config')
    full._require_resume_full_config(checkpoint_full_config)
    saved_cfg = deepcopy(checkpoint_full_config)
    full._require_resume_materialized_fields(saved_cfg)
    record = {
        'full_config': deepcopy(saved_cfg),
        'raw_config': (
            deepcopy(checkpoint_raw_config)
            if isinstance(checkpoint_raw_config, dict)
            and checkpoint_raw_config else None),
        'metadata': deepcopy(checkpoint_metadata),
        'source': 'checkpoint full_config',
    }
    if is_host0:
        print("Resume config source: checkpoint full_config")
        print("Resume config fallback: disabled")
        print("  Preserving existing run-folder config snapshots.")
    return saved_cfg, record


def _path_str(path):
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


def _is_tau_path(ps):
    return _is_tau_attn_bias_path(ps) or _is_tau_rst_bias_path(ps) or _is_raw_tau_path(ps)


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
    return ('router/proj_rst' in ps) or ('router/rst_op_write_query_proj' in ps)


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
    return ('router/raw_scan_offset_attn' in ps) or ('router/scan_attn' in ps)


def _is_scan_rst_path(ps):
    return ('router/raw_scan_offset_rst' in ps) or ('router/scan_rst' in ps)


def _tree_sq(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.float32(0.0)
    return sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves)


def _tree_norm(tree):
    return jnp.sqrt(_tree_sq(tree) + 1e-12)


def _group_sq(tree, path_pred):
    def _visit(path, leaf):
        if path_pred(_path_str(path)):
            x = leaf.astype(jnp.float32)
            return jnp.sum(jnp.square(x))
        return jnp.float32(0.0)
    leaves = jax.tree.leaves(jax.tree.map_with_path(_visit, tree))
    if not leaves:
        return jnp.float32(0.0)
    return sum(leaves)


def _group_max_abs(tree, path_pred):
    def _visit(path, leaf):
        if path_pred(_path_str(path)):
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


def _raw_tau_component_max_abs(tree, component):
    def _visit(path, leaf):
        sub = _raw_tau_component_leaf(_path_str(path), leaf, component)
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


def create_train_step_minimal(
        model, optimizer, sharded_fns=None,
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
        scan_update_abs_cap=0.0):
    _pass_analysis_kw = full._model_accepts_analysis(model)
    _pass_minimal_kw = _model_accepts_minimal_train(model)
    _pass_soft_gate_schedule_kw = full._model_accepts_soft_gate_schedule(model)
    _pass_soft_gate_t_final_kw = full._model_accepts_soft_gate_t_final(model)
    _pass_execution_prune_kw = full._model_accepts_execution_prune_eps(model)
    _pass_boundary_power_kw = full._model_accepts_soft_gate_boundary_power(model)
    _pass_den_power_kw = full._model_accepts_admission_den_power(model)
    _model_version = getattr(model, '__version__', getattr(type(model), '__version__', ''))
    _soft_gate_runtime_enabled = bool(
        soft_gate_schedule_active and full._is_active_srw_version(_model_version))
    _is_boundary_power_model = full._is_active_srw_version(_model_version)
    _total_training_steps = jnp.float32(max(1, int(total_training_steps or 1)))
    _soft_gate_t_final = jnp.float32(soft_gate_t_final)
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
    _soft_gate_schedule = str(soft_gate_schedule).lower()
    _soft_gate_pool_cfg = full._coerce_pool_schedule_configs(
        (soft_gate_pool_schedules
         if (pool_specific_gate_t
             or _soft_gate_schedule == 'developmental_band')
         else None),
        _soft_gate_pool_defaults)
    _boundary_power_schedule_active = bool(boundary_power_schedule_active)
    _soft_gate_boundary_power_start = jnp.float32(soft_gate_boundary_power_start)
    _soft_gate_boundary_power_mid = jnp.float32(soft_gate_boundary_power_mid)
    _soft_gate_boundary_power_final = jnp.float32(soft_gate_boundary_power_final)
    _soft_gate_boundary_power_start_frac = jnp.float32(
        soft_gate_boundary_power_start_frac)
    _soft_gate_boundary_power_mid_frac = jnp.float32(
        soft_gate_boundary_power_mid_frac)
    _soft_gate_boundary_power_final_frac = jnp.float32(
        soft_gate_boundary_power_final_frac)
    _admission_den_power = jnp.float32(admission_den_power)
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

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt_state, input_ids, attention_mask,
                   dropout_key, step):
        labels = jnp.where(attention_mask == 1, input_ids, -100)

        def loss_fn(params):
            extra_kw = {}
            if sharded_fns is not None:
                extra_kw['sharded_fns'] = sharded_fns
            if _pass_analysis_kw:
                extra_kw['analysis'] = False
            if _pass_minimal_kw:
                extra_kw['minimal_train'] = True
            if _soft_gate_runtime_enabled:
                soft_gate_T_qk = full._scheduled_from_config(
                    step, _total_training_steps, _soft_gate_pool_cfg['qk'])
                soft_gate_T_v = full._scheduled_from_config(
                    step, _total_training_steps, _soft_gate_pool_cfg['v'])
                soft_gate_T_rst = full._scheduled_from_config(
                    step, _total_training_steps, _soft_gate_pool_cfg['rst'])
            else:
                soft_gate_T_qk = jnp.float32(0.07)
                soft_gate_T_v = jnp.float32(0.07)
                soft_gate_T_rst = jnp.float32(0.07)
            boundary_power_p = full.scheduled_boundary_power_by_frac(
                step, _total_training_steps,
                _boundary_power_schedule_active and _is_boundary_power_model,
                _soft_gate_boundary_power_start,
                _soft_gate_boundary_power_mid,
                _soft_gate_boundary_power_final,
                _soft_gate_boundary_power_mid_frac,
                _soft_gate_boundary_power_final_frac,
                _soft_gate_boundary_power_start_frac)
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
                extra_kw['execution_prune_eps'] = jnp.float32(0.0)
            result = model.apply(
                {'params': params},
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                deterministic=False,
                rngs={'dropout': dropout_key},
                **extra_kw)
            ce_loss = result['loss']
            total_loss = ce_loss
            return total_loss, (
                ce_loss, result['correct'], result['valid_count'])

        (total_loss, (ce_loss, correct, valid_count)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True)(params))

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
            ps = _path_str(path)
            scale = jnp.float32(1.0)
            scale = jnp.where(_is_tau_path(ps), tau_clip_scale, scale)
            scale = jnp.where(_is_router_proj_path(ps), router_proj_clip_scale, scale)
            scale = jnp.where(_is_router_scan_path(ps), router_scan_clip_scale, scale)
            scale = jnp.where(_is_legacy_operator_key_path(ps), route_emb_clip_scale, scale)
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
            ps = _path_str(path)
            mult = jnp.float32(1.0)
            mult = jnp.where(_is_tau_path(ps), _tau_lr_mult, mult)
            mult = jnp.where(_is_router_proj_path(ps), _router_proj_lr_mult, mult)
            mult = jnp.where(_is_router_scan_path(ps), _router_scan_lr_mult, mult)
            mult = jnp.where(_is_legacy_operator_key_path(ps), _route_emb_lr_mult, mult)
            mult = jnp.where(_is_op_key_proj_path(ps), _op_key_lr_mult, mult)
            return u * mult.astype(u.dtype)

        if (float(tau_lr_mult) != 1.0
                or float(router_proj_lr_mult) != 1.0
                or float(router_scan_lr_mult) != 1.0
                or float(route_emb_lr_mult) != 1.0
                or float(op_key_lr_mult) != 1.0):
            updates = jax.tree.map_with_path(_scale_control_update, updates)

        if _enable_control_update_caps:
            def _ratio_cap_scale(params_tree, updates_tree, path_pred, cap):
                p_sq = _group_sq(params_tree, path_pred)
                u_sq = _group_sq(updates_tree, path_pred)
                p_norm = jnp.sqrt(p_sq)
                u_norm = jnp.sqrt(u_sq)
                ratio = jnp.where(p_sq > 0.0, u_norm / (p_norm + 1e-8), 0.0)
                return jnp.where(
                    cap > 0.0,
                    jnp.minimum(jnp.float32(1.0), cap / (ratio + 1e-12)),
                    jnp.float32(1.0))

            def _abs_cap_scale(updates_tree, path_pred, cap):
                abs_pre = _group_max_abs(updates_tree, path_pred)
                return jnp.where(
                    cap > 0.0,
                    jnp.minimum(jnp.float32(1.0), cap / (abs_pre + 1e-12)),
                    jnp.float32(1.0))

            def _raw_tau_abs_cap_scale(updates_tree, component, cap):
                abs_pre = _raw_tau_component_max_abs(updates_tree, component)
                return jnp.where(
                    cap > 0.0,
                    jnp.minimum(jnp.float32(1.0), cap / (abs_pre + 1e-12)),
                    jnp.float32(1.0))

            proj_attn_scale = _ratio_cap_scale(
                params, updates, _is_router_proj_attn_path,
                _router_proj_update_ratio_cap)
            proj_rst_scale = _ratio_cap_scale(
                params, updates, _is_router_proj_rst_path,
                _router_proj_update_ratio_cap)
            op_key_qk_scale = _ratio_cap_scale(
                params, updates, _is_op_key_qk_path,
                _route_emb_update_ratio_cap)
            op_key_v_scale = _ratio_cap_scale(
                params, updates, _is_op_key_v_path,
                _route_emb_update_ratio_cap)
            op_key_rst_scale = _ratio_cap_scale(
                params, updates, _is_op_key_rst_path,
                _route_emb_update_ratio_cap)
            tau_attn_scale = _abs_cap_scale(
                updates, _is_tau_attn_bias_path, _tau_update_abs_cap)
            tau_rst_scale = _abs_cap_scale(
                updates, _is_tau_rst_bias_path, _tau_update_abs_cap)
            raw_tau_qk_scale = _raw_tau_abs_cap_scale(
                updates, 'qk', _tau_update_abs_cap)
            raw_tau_v_scale = _raw_tau_abs_cap_scale(
                updates, 'v', _tau_update_abs_cap)
            raw_tau_rst_scale = _raw_tau_abs_cap_scale(
                updates, 'rst', _tau_update_abs_cap)
            scan_attn_scale = _abs_cap_scale(
                updates, _is_scan_attn_path, _scan_update_abs_cap)
            scan_rst_scale = _abs_cap_scale(
                updates, _is_scan_rst_path, _scan_update_abs_cap)

            def _cap_control_update(path, u):
                ps = _path_str(path)
                if _is_raw_tau_attn_combined_path(ps):
                    if u.ndim > 0 and u.shape[-1] >= 3:
                        out = u
                        out = out.at[..., :2].multiply(
                            raw_tau_qk_scale.astype(u.dtype))
                        out = out.at[..., 2:3].multiply(
                            raw_tau_v_scale.astype(u.dtype))
                        return out
                    return u
                if _is_generic_raw_tau_path(ps):
                    parts = _path_parts(ps)
                    if ('rst' in parts) or ('know' in parts):
                        return u * raw_tau_rst_scale.astype(u.dtype)
                    if ('v' in parts) or ('attn_v' in parts):
                        return u * raw_tau_v_scale.astype(u.dtype)
                    if ('qk' in parts) or ('attn_qk' in parts):
                        return u * raw_tau_qk_scale.astype(u.dtype)
                    if u.ndim > 0 and u.shape[-1] >= 3:
                        out = u
                        out = out.at[..., :2].multiply(
                            raw_tau_qk_scale.astype(u.dtype))
                        out = out.at[..., 2:3].multiply(
                            raw_tau_v_scale.astype(u.dtype))
                        return out
                    return u * raw_tau_qk_scale.astype(u.dtype)
                if _is_raw_tau_qk_path(ps):
                    return u * raw_tau_qk_scale.astype(u.dtype)
                if _is_raw_tau_v_path(ps):
                    return u * raw_tau_v_scale.astype(u.dtype)
                if _is_raw_tau_rst_path(ps):
                    return u * raw_tau_rst_scale.astype(u.dtype)
                scale = jnp.float32(1.0)
                scale = jnp.where(_is_router_proj_attn_path(ps), proj_attn_scale, scale)
                scale = jnp.where(_is_router_proj_rst_path(ps), proj_rst_scale, scale)
                scale = jnp.where(_is_op_key_qk_path(ps), op_key_qk_scale, scale)
                scale = jnp.where(_is_op_key_v_path(ps), op_key_v_scale, scale)
                scale = jnp.where(_is_op_key_rst_path(ps), op_key_rst_scale, scale)
                scale = jnp.where(_is_tau_attn_bias_path(ps), tau_attn_scale, scale)
                scale = jnp.where(_is_tau_rst_bias_path(ps), tau_rst_scale, scale)
                scale = jnp.where(_is_scan_attn_path(ps), scan_attn_scale, scale)
                scale = jnp.where(_is_scan_rst_path(ps), scan_rst_scale, scale)
                return u * scale.astype(u.dtype)

            updates = jax.tree.map_with_path(_cap_control_update, updates)

        new_params = optax.apply_updates(params, updates)
        metrics = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'correct': correct,
            'valid_count': valid_count,
            'grad_norm': _tree_norm(grads),
        }
        return new_params, new_opt_state, metrics

    return train_step


def create_eval_step_minimal(
        model, sharded_fns=None,
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
    _pass_analysis_kw = full._model_accepts_analysis(model)
    _pass_minimal_kw = _model_accepts_minimal_train(model)
    _pass_soft_gate_schedule_kw = full._model_accepts_soft_gate_schedule(model)
    _pass_soft_gate_t_final_kw = full._model_accepts_soft_gate_t_final(model)
    _pass_execution_prune_kw = full._model_accepts_execution_prune_eps(model)
    _pass_boundary_power_kw = full._model_accepts_soft_gate_boundary_power(model)
    _pass_den_power_kw = full._model_accepts_admission_den_power(model)
    _model_version = getattr(model, '__version__', getattr(type(model), '__version__', ''))
    _soft_gate_runtime_enabled = bool(
        soft_gate_schedule_active and full._is_active_srw_version(_model_version))
    _is_boundary_power_model = full._is_active_srw_version(_model_version)
    _total_training_steps = jnp.float32(max(1, int(total_training_steps or 1)))
    _soft_gate_t_final = jnp.float32(soft_gate_t_final)
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
    _soft_gate_schedule = str(soft_gate_schedule).lower()
    _soft_gate_pool_cfg = full._coerce_pool_schedule_configs(
        (soft_gate_pool_schedules
         if (pool_specific_gate_t
             or _soft_gate_schedule == 'developmental_band')
         else None),
        _soft_gate_pool_defaults)
    _boundary_power_schedule_active = bool(boundary_power_schedule_active)
    _soft_gate_boundary_power_start = jnp.float32(soft_gate_boundary_power_start)
    _soft_gate_boundary_power_mid = jnp.float32(soft_gate_boundary_power_mid)
    _soft_gate_boundary_power_final = jnp.float32(soft_gate_boundary_power_final)
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
        extra_kw = {}
        if sharded_fns is not None:
            extra_kw['sharded_fns'] = sharded_fns
        if _pass_analysis_kw:
            extra_kw['analysis'] = False
        if _pass_minimal_kw:
            extra_kw['minimal_train'] = True
        if _soft_gate_runtime_enabled and _pass_soft_gate_schedule_kw:
            soft_gate_T_qk = full._scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['qk'])
            extra_kw['soft_gate_temperature'] = soft_gate_T_qk
            extra_kw['soft_gate_T_qk'] = soft_gate_T_qk
            extra_kw['soft_gate_T_v'] = full._scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['v'])
            extra_kw['soft_gate_T_rst'] = full._scheduled_from_config(
                step, _total_training_steps, _soft_gate_pool_cfg['rst'])
        if _pass_boundary_power_kw:
            boundary_power_p = full.scheduled_boundary_power_by_frac(
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
            rngs={'dropout': jax.random.PRNGKey(0)},
            **extra_kw)
        return result['loss'], result['correct'], result['valid_count']

    return eval_step


def _chunk_size_from_count(name, n_local, n_chunks):
    n_chunks = int(n_chunks)
    if n_chunks < 1:
        raise ValueError(f"{name} chunks must be >= 1, got {n_chunks}")
    if n_chunks > n_local:
        raise ValueError(
            f"{name} chunks={n_chunks} exceeds local pool size {n_local}")
    return max(1, int(math.ceil(n_local / n_chunks)))


def _make_minimal_sharded_fns(cfg, mesh, mesh_model, batch_size, max_seq_len,
                              per_data_shard_batch, is_host0):
    model_version = str(cfg['model'].get(
        'model_version', full.OFFICIAL_MODEL_VERSION))
    if not full._is_active_srw_version(model_version):
        return None, (1, 1, 1), (1, 1, 1)
    if model_version == full.V4167_MODEL_VERSION:
        full._dawn_srw_kwargs(cfg)

    target_chunk_gb = cfg['training'].get('target_chunk_gb', 2.0)
    if model_version == full.V4167_MODEL_VERSION:
        for name in (
                'n_qk_global', 'n_qk_stage', 'n_qk_local',
                'n_v_global', 'n_v_stage', 'n_v_local',
                'n_rst_global', 'n_rst_stage', 'n_rst_local'):
            value = int(cfg['model'][name])
            if value % mesh_model != 0:
                raise ValueError(
                    f"{name}={value} must be divisible by "
                    f"mesh_model={mesh_model} for v4167 model-axis GSL "
                    "sharding.")
        n_qk_for_chunks = int(cfg['model']['qk_visible_n'])
        n_v_for_chunks = int(cfg['model']['v_visible_n'])
        n_rst_for_chunks = int(cfg['model']['rst_visible_n'])
    else:
        n_qk_for_chunks = int(cfg['model'].get(
            'n_qk', cfg['model'].get('n_q', 1580)))
        n_v_for_chunks = int(cfg['model'].get('n_v', 2600))
        n_rst_for_chunks = int(cfg['model'].get(
            'n_rst', cfg['model'].get('n_know', 25200)))
        for name, value in (
                ('n_qk', n_qk_for_chunks),
                ('n_v', n_v_for_chunks),
                ('n_rst', n_rst_for_chunks)):
            if value % mesh_model != 0:
                raise ValueError(
                    f"{name}={value} must be divisible by "
                    f"mesh_model={mesh_model}")

    nqk_local = n_qk_for_chunks // mesh_model
    nv_local = n_v_for_chunks // mesh_model
    nrst_local = n_rst_for_chunks // mesh_model

    def auto_n_chunks(n, target_gb=2.0):
        full_gb = per_data_shard_batch * max_seq_len * n * 2 / 1e9
        nc = max(1, int(np.ceil(full_gb / target_gb)))
        while n % nc != 0 and nc < n:
            nc += 1
        return min(nc, n)

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
    rst_max_chunk = _chunk_size_from_count('rst', nrst_local, n_chunks_rst)

    module_name = full._model_registry_entry(model_version)['module']
    srw_module = __import__(module_name, fromlist=[
        'make_sharded_srw_minimal',
        'make_sharded_srw_paired_minimal',
        'create_v4167_tp_sharded_fns',
    ])
    make_single = getattr(srw_module, 'make_sharded_srw_minimal')
    make_paired = getattr(srw_module, 'make_sharded_srw_paired_minimal')
    max_chunk = cfg['training'].get('max_chunk_size', None)
    if max_chunk is not None:
        attn_qk_max_chunk = attn_v_max_chunk = rst_max_chunk = int(max_chunk)

    base_kwargs = {'mesh': mesh}
    base_kwargs.update(full._v4164_sharded_kwargs(cfg))

    def _factory_kwargs(factory, kwargs):
        sig = inspect.signature(factory)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    def _srw_pool_kwargs(pool):
        kwargs = dict(base_kwargs)
        if model_version == full.V4168_MODEL_VERSION:
            m_cfg = cfg['model']
            kwargs.update({
                'block_size': int(m_cfg.get(f'{pool}_block_size', 256)),
                'top_blocks': int(m_cfg.get(f'{pool}_top_blocks', 2)),
                'block_margin': float(m_cfg.get('block_margin', 0.0)),
            })
        return kwargs

    sharded_single_v = make_single(
        max_chunk_size=attn_v_max_chunk,
        **_factory_kwargs(make_single, _srw_pool_kwargs('v')))
    sharded_single_rst = make_single(
        max_chunk_size=rst_max_chunk,
        **_factory_kwargs(make_single, _srw_pool_kwargs('rst')))
    sharded_single_qk = make_single(
        max_chunk_size=attn_qk_max_chunk,
        **_factory_kwargs(make_single, _srw_pool_kwargs('qk')))
    sharded_paired_attn_qk = make_paired(
        max_chunk_size=attn_qk_max_chunk,
        **_factory_kwargs(make_paired, _srw_pool_kwargs('qk')))

    sharded_fns = {
        'single': sharded_single_rst,
        'attn_qk_single_minimal': sharded_single_qk,
        'attn_v_single': sharded_single_v,
        'rst_single': sharded_single_rst,
        'paired': sharded_paired_attn_qk,
        'attn_qk_paired': sharded_paired_attn_qk,
        'attn_v_single_minimal': sharded_single_v,
        'rst_single_minimal': sharded_single_rst,
        'attn_qk_paired_minimal': sharded_paired_attn_qk,
    }
    if model_version == full.V4167_MODEL_VERSION:
        extra_factory = getattr(srw_module, 'create_v4167_tp_sharded_fns', None)
        if extra_factory is None:
            raise RuntimeError(
                "v4167 module is missing create_v4167_tp_sharded_fns.")
        v4167_extra_fns = extra_factory(mesh, cfg)
        sharded_fns.update(v4167_extra_fns)
        required = (
            'attn_qk_paired_minimal',
            'attn_v_single_minimal',
            'rst_single_minimal',
            'v4167_router_dense',
            'v4167_tp_attention_o',
            'vocab_parallel_embedding',
            'vocab_parallel_ce',
        )
        missing = [name for name in required if name not in sharded_fns]
        if missing:
            raise RuntimeError(
                "v4167 minimal sharded_fns missing required entries: "
                + ", ".join(missing))

    if is_host0:
        extra_msg = (
            "; v4167 TP extras=router_dense,attention_o,vocab_parallel"
            if model_version == full.V4167_MODEL_VERSION else "")
        if model_version == full.V4168_MODEL_VERSION:
            extra_msg = (
                "; v4168 block_sparse "
                f"block_size qk/v/rst={cfg['model'].get('qk_block_size', 256)}/"
                f"{cfg['model'].get('v_block_size', 256)}/"
                f"{cfg['model'].get('rst_block_size', 256)}, "
                f"top_blocks qk/v/rst={cfg['model'].get('qk_top_blocks', 2)}/"
                f"{cfg['model'].get('v_top_blocks', 2)}/"
                f"{cfg['model'].get('rst_top_blocks', 2)}")
        print(
            "  shard_map minimal enabled "
            f"(mesh_model={mesh_model}, "
            f"{'QK single-route' if model_version == full.V4168_MODEL_VERSION else 'QK fused'}; "
            f"per_data_shard_batch={per_data_shard_batch}; "
            f"chunks attn_qk/attn_v/rst={n_chunks_qk}/{n_chunks_v}/{n_chunks_rst}; "
            f"max_chunk attn_qk/attn_v/rst={attn_qk_max_chunk}/{attn_v_max_chunk}/{rst_max_chunk}; "
            f"analysis kernels=off{extra_msg})")

    return sharded_fns, (n_chunks_qk, n_chunks_v, n_chunks_rst), (
        attn_qk_max_chunk, attn_v_max_chunk, rst_max_chunk)


def _write_fresh_config_snapshot(checkpoint_dir, cfg, raw_cfg_snapshot,
                                 config_path, training_config):
    full_config_snapshot = full._materialized_config_snapshot(
        cfg, training_config)
    raw_config_snapshot = full._safe_config_snapshot(raw_cfg_snapshot)
    materialized_sha = full._config_sha256(full_config_snapshot)
    selected_sha = full._config_sha256(full_config_snapshot)
    record = deepcopy(full_config_snapshot)
    record['_raw_config'] = raw_config_snapshot
    record['_metadata'] = {
        'type': 'fresh_run_config_snapshot',
        'timestamp': datetime.now().isoformat(),
        'config_path': str(config_path),
        'checkpoint_schema_version': full.CHECKPOINT_SCHEMA_VERSION,
        'current_materialized_config_sha256': materialized_sha,
        'selected_full_config_sha256': selected_sha,
        'trainer': 'scripts/train_jax_minimal.py',
    }
    full._write_json_file(full._join_path(checkpoint_dir, 'config.json'), record)
    return full_config_snapshot, raw_config_snapshot, selected_sha, materialized_sha


def main():
    parser = argparse.ArgumentParser(
        description='Minimal DAWN-SRW v4166/v4167 JAX trainer')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume-from', '--resume', dest='resume_from',
                        type=str, default=None)
    parser.add_argument(
        '--debug',
        nargs='?',
        const=1,
        default=0,
        type=int,
        help='Enable verbose minimal trainer diagnostics. Optional interval.')
    cli_args = parser.parse_args()
    global debug_interval, debug_enabled
    debug_interval = int(cli_args.debug or 0)
    debug_enabled = debug_interval > 0

    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        if full._file_exists(cli_args.config):
            config_path = cli_args.config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = full.load_config(config_path)
    raw_cfg_snapshot = deepcopy(cfg)
    current_yaml_config_snapshot = deepcopy(cfg)
    full._maybe_initialize_jax_distributed()
    full._require_orbax_checkpoint_compat()

    n_hosts = jax.process_count()
    host_id = jax.process_index()
    is_host0 = host_id == 0
    n_local_devices = jax.local_device_count()

    resume_path, checkpoint_dir, resume_step = _resolve_or_create_run(
        cfg, cli_args, is_host0)
    cfg, resume_config_record = _maybe_load_resume_config(
        cfg, checkpoint_dir, resume_step, is_host0)
    deprecated_log_dir_present = (
        'log_dir' in current_yaml_config_snapshot or 'log_dir' in cfg)

    seed = cfg.get('seed', 42)
    full.set_seed(seed)
    tcfg = cfg['training']
    model_version_cfg = cfg['model'].get(
        'model_version', full.OFFICIAL_MODEL_VERSION)
    if model_version_cfg not in SUPPORTED_MINIMAL_SRW_VERSIONS:
        raise ValueError(
            "scripts/train_jax_minimal.py supports only minimal active SRW "
            f"versions {SUPPORTED_MINIMAL_SRW_VERSIONS}, "
            f"got {model_version_cfg!r}")

    tau_init_cfg = full._v4164_tau_init_config(cfg)
    selection_calibration_cfg = full._selection_calibration_config(
        cfg, tau_init_cfg)

    if resume_step is not None and is_host0 and (
            cli_args.batch_size is not None
            or cli_args.epochs is not None
            or cli_args.lr is not None):
        print(
            "Resume detected; ignoring launch-time --epochs/--batch-size/--lr "
            "because checkpoint full_config is the source of truth.",
            flush=True)
    batch_size = (
        int(tcfg['batch_size']) if resume_step is not None
        else (cli_args.batch_size or int(tcfg['batch_size'])))
    num_epochs = (
        int(tcfg['num_epochs']) if resume_step is not None
        else (cli_args.epochs or int(tcfg['num_epochs'])))
    lr = (
        float(tcfg.get('lr', tcfg.get('learning_rate', 6.5e-4)))
        if resume_step is not None
        else (cli_args.lr or float(
            tcfg.get('lr', tcfg.get('learning_rate', 6.5e-4)))))
    weight_decay = float(tcfg.get('weight_decay', 0.1))
    pool_weight_decay = 0.0
    warmup_ratio = float(tcfg.get('warmup_ratio', 0.06))
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
    soft_gate_pool_schedules = full._training_soft_gate_pool_schedules(
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
    admission_den_power = float(tcfg.get(
        'admission_den_power', tcfg.get('v4164_den_power', 1.0)))
    admission_den_grad_scale = float(tcfg.get(
        'admission_den_grad_scale', tcfg.get('v4164_den_grad_scale', 1.0)))

    if soft_gate_schedule_active:
        if soft_gate_schedule.lower() not in full.SOFT_GATE_T_SCHEDULE_NAMES:
            raise ValueError(
                f"Unsupported soft_gate_t_schedule={soft_gate_schedule!r}; "
                f"{full._soft_gate_schedule_expected_msg()}")
        for pool, pool_cfg in soft_gate_pool_schedules.items():
            full._validate_soft_gate_schedule_config(
                f"soft_gate_t_{pool}", pool_cfg,
                require_pool_specific_devband_fields=True)

    global_grad_clip = float(tcfg.get('global_grad_clip', 0.0))
    tau_lr_mult = float(tcfg.get('tau_lr_mult', 1.0))
    tau_grad_clip = float(tcfg.get('tau_grad_clip', 0.0))
    router_proj_lr_mult = float(tcfg.get('router_proj_lr_mult', 1.0))
    router_proj_grad_clip = float(tcfg.get('router_proj_grad_clip', 0.0))
    router_scan_lr_mult = float(tcfg.get('router_scan_lr_mult', 1.0))
    router_scan_grad_clip = float(tcfg.get('router_scan_grad_clip', 0.0))
    route_emb_lr_mult = float(tcfg.get('route_emb_lr_mult', 1.0))
    route_emb_grad_clip = float(tcfg.get('route_emb_grad_clip', 0.0))
    op_key_lr_mult = float(tcfg.get('op_key_lr_mult', route_emb_lr_mult))
    op_key_grad_clip = float(tcfg.get('op_key_grad_clip', route_emb_grad_clip))
    enable_control_update_caps = bool(tcfg.get('enable_control_update_caps', False))
    router_proj_update_ratio_cap = float(
        tcfg.get('router_proj_update_ratio_cap', 0.0))
    route_emb_update_ratio_cap = float(
        tcfg.get('route_emb_update_ratio_cap', 0.0))
    tau_update_abs_cap = float(tcfg.get('tau_update_abs_cap', 0.0))
    scan_update_abs_cap = float(tcfg.get('scan_update_abs_cap', 0.0))

    ckpt_interval = int(tcfg.get('checkpoint_interval', 5000))
    val_interval = int(tcfg.get('val_interval', 5000))
    log_interval = int(tcfg.get('log_interval', 100))
    checkpoint_keep_last = int(tcfg.get(
        'checkpoint_keep_last', tcfg.get('max_checkpoints_to_keep', 3)))
    checkpointing_enabled = ckpt_interval > 0
    checkpoint_manager_required = checkpointing_enabled or resume_step is not None
    max_seq_len = int(cfg['model'].get('max_seq_len', 512))

    training_config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'pool_weight_decay': pool_weight_decay,
        'warmup_ratio': warmup_ratio,
        'gradient_accumulation_steps': tcfg.get(
            'gradient_accumulation_steps', 1),
        'soft_gate_t_start': soft_gate_t_start,
        'soft_gate_t_final': soft_gate_t_final,
        'soft_gate_t_hold_frac': soft_gate_t_hold_frac,
        'soft_gate_t_anneal_end_frac': soft_gate_t_anneal_end_frac,
        'soft_gate_t_schedule': soft_gate_schedule,
        'soft_gate_t_power': soft_gate_t_power,
        'soft_gate_t_gompertz_center': soft_gate_t_gompertz_center,
        'soft_gate_t_gompertz_steepness': soft_gate_t_gompertz_steepness,
        **full._flatten_soft_gate_pool_schedules(soft_gate_pool_schedules),
        'soft_gate_boundary_power_start': soft_gate_boundary_power_start,
        'soft_gate_boundary_power_mid': soft_gate_boundary_power_mid,
        'soft_gate_boundary_power_final': soft_gate_boundary_power_final,
        'soft_gate_boundary_power_start_frac': soft_gate_boundary_power_start_frac,
        'soft_gate_boundary_power_mid_frac': soft_gate_boundary_power_mid_frac,
        'soft_gate_boundary_power_final_frac': soft_gate_boundary_power_final_frac,
        'admission_den_power': admission_den_power,
        'admission_den_grad_scale': admission_den_grad_scale,
        'soft_gate_effective_active_eps': float(
            tcfg.get('soft_gate_effective_active_eps', 1.0e-6)),
        'eval_effective_prune_enabled': False,
        'checkpoint_interval': ckpt_interval,
        'checkpoint_keep_last': checkpoint_keep_last,
        'training_log_append_on_resume': bool(
            tcfg.get('training_log_append_on_resume', True)),
        'val_interval': val_interval,
        'log_interval': log_interval,
        'log_analysis_multiplier': int(tcfg.get('log_analysis_multiplier', 1000000)),
        'heavy_geometry_multiplier': int(tcfg.get('heavy_geometry_multiplier', 1000000)),
        'global_grad_clip': global_grad_clip,
        'tau_lr_mult': tau_lr_mult,
        'tau_grad_clip': tau_grad_clip,
        'router_proj_lr_mult': router_proj_lr_mult,
        'router_proj_grad_clip': router_proj_grad_clip,
        'router_scan_lr_mult': router_scan_lr_mult,
        'router_scan_grad_clip': router_scan_grad_clip,
        'op_key_lr_mult': op_key_lr_mult,
        'op_key_grad_clip': op_key_grad_clip,
        'enable_control_update_caps': enable_control_update_caps,
        'router_proj_update_ratio_cap': router_proj_update_ratio_cap,
        'route_emb_update_ratio_cap': route_emb_update_ratio_cap,
        'tau_update_abs_cap': tau_update_abs_cap,
        'scan_update_abs_cap': scan_update_abs_cap,
        'tau_init_mode': tau_init_cfg['mode'],
    }
    if selection_calibration_cfg.get('present', False):
        training_config['selection_calibration'] = selection_calibration_cfg.get(
            'raw', {'enabled': False})

    cfg.setdefault('training', {}).update(training_config)

    if debug_enabled:
        print(
            f"[Host {host_id}/{n_hosts}] local_devices={n_local_devices} "
            f"total_devices={jax.device_count()} backend={jax.default_backend()} "
            f"devices={[str(d) for d in jax.local_devices()]}",
            flush=True)
    elif is_host0:
        print(
            f"[Host {host_id}/{n_hosts}] local_devices={n_local_devices} "
            f"total_devices={jax.device_count()} backend={jax.default_backend()}",
            flush=True)

    train_script_git_commit = full._safe_git_info().get('git_commit')
    if debug_enabled:
        full._print_jax_distributed_identity(
            'scripts/train_jax_minimal.py',
            config_path,
            checkpoint_dir,
            cli_args.from_scratch,
            resume_step,
        )

    startup_mesh_model = int(cfg['training'].get('mesh_model', 1))
    startup_mesh_data = int(cfg['training'].get('mesh_data', 0))
    if startup_mesh_data == 0:
        startup_mesh_data = jax.device_count() // startup_mesh_model
    startup_contexts = full._assert_multihost_same_startup_context({
        'trainer_script': 'scripts/train_jax_minimal.py',
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
    if debug_enabled and is_host0:
        print(
            "Distributed startup contexts:\n"
            + json.dumps(startup_contexts, indent=2, sort_keys=True,
                         default=str),
            flush=True)
    _stage_barrier("after_startup_context_check")

    per_host_batch = batch_size // n_hosts
    per_device_batch = per_host_batch // n_local_devices
    if batch_size % n_hosts != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by n_hosts={n_hosts}")
    if per_host_batch % n_local_devices != 0:
        raise ValueError(
            f"per_host_batch={per_host_batch} must be divisible by "
            f"n_local_devices={n_local_devices}")

    if is_host0:
        print(f"\n=== DAWN-SRW {model_version_cfg} Minimal Training ===")
        print(f"Config: {config_path}")
        print(f"Run folder: {checkpoint_dir}")
        print(f"Global batch size: {batch_size}")
        print(f"Per-host batch size: {per_host_batch}")
        print(f"Per-device batch size: {per_device_batch}")
        print("Analysis/geometry/prune/drift diagnostics: off")

    _stage_log("before_load_data", {
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "n_hosts": n_hosts,
        "host_id": host_id,
    })
    from utils.data_jax import load_data
    train_loader, val_loader, vocab_size = load_data(
        cfg['data'],
        max_length=max_seq_len,
        batch_size=batch_size,
        n_devices=1,
        n_hosts=n_hosts,
        host_id=host_id,
    )
    _stage_log("after_load_data", {
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "vocab_size": vocab_size,
    })
    _stage_barrier("after_load_data")
    if is_host0:
        print(f"Vocab size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    cfg['model']['vocab_size'] = vocab_size
    full._maybe_materialize_vocab_parallel_config(cfg)
    _stage_log("before_build_model")
    model = full.build_model_from_config(cfg)
    _stage_log("after_build_model")
    _stage_barrier("after_build_model")
    if not _model_accepts_minimal_train(model):
        raise RuntimeError(
            f"{model_version_cfg} model does not expose minimal_train; "
            "minimal trainer requires the model-level minimal path.")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones((1, max_seq_len), dtype=jnp.int32)
    if is_host0:
        print("=== Starting model.init ===", flush=True)
    _stage_log("before_model_init")
    variables = model.init(
        {'params': init_rng, 'dropout': dropout_rng},
        dummy_input,
        deterministic=True,
    )
    params = variables['params']
    _stage_log("after_model_init", {
        "param_count": full.count_parameters(params) if host_id == 0 else -1,
    })
    _stage_barrier("after_model_init")
    if is_host0:
        print("=== model.init done ===", flush=True)
        print(f"Model parameters: {full.count_parameters(params):,}")
        for line in model.get_model_info():
            print(line)

    selection_calibration_summary = None
    tau_init_summary = None
    _stage_log("before_selection_calibration_branch", {
        "enabled": bool(selection_calibration_cfg.get("enabled", False)),
        "resume_step": None if resume_step is None else int(resume_step),
    })
    _stage_barrier("before_selection_calibration")
    if (selection_calibration_cfg.get('enabled', False)
            and resume_step is None):
        if len(train_loader) <= 0:
            raise RuntimeError(
                "selection_calibration requires at least one training batch.")
        if is_host0:
            print("\nSelection calibration: computing from fresh init.")
        _stage_log("before_collect_selection_calibration_histograms")
        _stage_barrier("before_collect_selection_calibration_histograms")
        (
            local_histograms,
            local_page_stats,
            local_seen_tokens,
            local_actual_batches,
            local_calibration_tokens,
            calibration_process_count,
        ) = full._collect_selection_calibration_histograms(
            params, train_loader, cfg, selection_calibration_cfg)
        _stage_log("after_collect_selection_calibration_histograms", {
            "seen_tokens_local": int(local_seen_tokens),
            "actual_batches": int(local_actual_batches),
        })
        _stage_barrier("after_collect_selection_calibration_histograms")
        _stage_log("before_aggregate_selection_calibration_histograms")
        _stage_barrier("before_aggregate_selection_calibration_histograms")
        (
            calibration_histograms,
            calibration_page_stats,
            seen_tokens,
            actual_calibration_batches,
        ) = full._aggregate_selection_calibration_histograms(
            local_histograms, local_page_stats,
            local_seen_tokens, local_actual_batches)
        _stage_log("after_aggregate_selection_calibration_histograms")
        _stage_barrier("after_aggregate_selection_calibration_histograms")
        selection_json = None
        if is_host0:
            selection_calibration_summary = (
                full._compute_srw_selection_calibration(
                    calibration_histograms, cfg, selection_calibration_cfg,
                    calibration_page_stats, seen_tokens,
                    actual_calibration_batches,
                    local_calibration_tokens, calibration_process_count))
            selection_json = json.dumps(selection_calibration_summary)
        _stage_log("before_selection_json_broadcast")
        _stage_barrier("before_selection_json_broadcast")
        selection_json = _broadcast_str_from_host0(
            selection_json, max_len=32768)
        if not selection_json:
            raise RuntimeError(
                "Failed to broadcast selection calibration summary.")
        _stage_log("after_selection_json_broadcast")
        _stage_barrier("after_selection_json_broadcast")
        selection_calibration_summary = json.loads(selection_json)
        params = full._set_srw_quantile_tau_biases(
            params, selection_calibration_summary, model_version_cfg)
        fixed_tau_materialized = full._materialize_fixed_tau_config(
            cfg,
            training_config,
            selection_calibration_summary,
            model_version_cfg,
        )
        if fixed_tau_materialized:
            model = full.build_model_from_config(cfg)
            if is_host0:
                print(
                    "v4167 fixed tau materialized from selection calibration: "
                    f"qk={cfg['model']['tau_init_attn_qk']:.6f} "
                    f"v={cfg['model']['tau_init_attn_v']:.6f} "
                    f"rst={cfg['model']['tau_init_rst']:.6f}",
                    flush=True,
                )
            if not _model_accepts_minimal_train(model):
                raise RuntimeError(
                    f"{model_version_cfg} model does not expose "
                    "minimal_train after fixed-tau materialization; minimal "
                    "trainer requires the model-level minimal path.")
        _stage_log("after_selection_calibration_applied")
        _stage_barrier("after_selection_calibration_applied")
        materialized_updates = (
            full._selection_calibration_materialized_training_updates(
                selection_calibration_summary, selection_calibration_cfg))
        soft_gate_pool_schedules = (
            full._apply_selection_calibrated_soft_gate_schedules(
                soft_gate_pool_schedules, materialized_updates))
        soft_gate_boundary_power_start = materialized_updates[
            'soft_gate_boundary_power_start']
        soft_gate_boundary_power_mid = materialized_updates[
            'soft_gate_boundary_power_mid']
        soft_gate_boundary_power_final = materialized_updates[
            'soft_gate_boundary_power_final']
        soft_gate_boundary_power_start_frac = materialized_updates[
            'soft_gate_boundary_power_start_frac']
        soft_gate_boundary_power_mid_frac = materialized_updates[
            'soft_gate_boundary_power_mid_frac']
        soft_gate_boundary_power_final_frac = materialized_updates[
            'soft_gate_boundary_power_final_frac']
        calibrated_flat = full._flatten_soft_gate_pool_schedules(
            soft_gate_pool_schedules)
        training_config.update(materialized_updates)
        training_config.update(calibrated_flat)
        training_config.update(full._selection_calibration_checkpoint_updates(
            selection_calibration_summary))
        cfg.setdefault('training', {}).update(training_config)
        if is_host0:
            print("\n=== Selection calibration ===")
            for line in full._selection_calibration_summary_lines(
                    selection_calibration_summary):
                print(line)
    elif (selection_calibration_cfg.get('enabled', False)
          and resume_step is not None):
        full._require_selection_calibration_resume_fields(
            cfg.get('training', {}))
        if is_host0:
            print("Selection calibration: restored from checkpoint config.")
    elif tau_init_cfg and tau_init_cfg['mode'] == 'quantile_frac':
        if len(train_loader) <= 0:
            raise RuntimeError("tau init calibration requires training data.")
        calibration_input_ids, _ = next(iter(train_loader))
        tau_init_summary = full._compute_srw_quantile_tau_init(
            params, calibration_input_ids, cfg, tau_init_cfg)
        params = full._set_srw_quantile_tau_biases(
            params, tau_init_summary, model_version_cfg)
        fixed_tau_materialized = full._materialize_fixed_tau_config(
            cfg, training_config, tau_init_summary, model_version_cfg)
        if fixed_tau_materialized:
            model = full.build_model_from_config(cfg)
            if is_host0:
                print(
                    "v4167 fixed tau materialized from quantile init: "
                    f"qk={cfg['model']['tau_init_attn_qk']:.6f} "
                    f"v={cfg['model']['tau_init_attn_v']:.6f} "
                    f"rst={cfg['model']['tau_init_rst']:.6f}",
                    flush=True)
            if not _model_accepts_minimal_train(model):
                raise RuntimeError(
                    f"{model_version_cfg} model does not expose "
                    "minimal_train after fixed-tau materialization; minimal "
                    "trainer requires the model-level minimal path.")
        if is_host0:
            for line in full._v4164_tau_init_summary_lines(tau_init_summary):
                print(line)

    raw_config_snapshot = full._safe_config_snapshot(raw_cfg_snapshot)
    if resume_config_record is None:
        full_config_snapshot = full._materialized_config_snapshot(
            cfg, training_config)
        selected_full_config_sha256 = full._config_sha256(
            full_config_snapshot)
        current_materialized_config_sha256 = selected_full_config_sha256
        if is_host0:
            (
                full_config_snapshot,
                raw_config_snapshot,
                selected_full_config_sha256,
                current_materialized_config_sha256,
            ) = _write_fresh_config_snapshot(
                checkpoint_dir, cfg, raw_cfg_snapshot, config_path,
                training_config)
            print(
                "Config hash: "
                f"selected_full_config_sha256={selected_full_config_sha256}",
                flush=True)
            print(
                f"Saved config.json: "
                f"{full._join_path(checkpoint_dir, 'config.json')}")
    else:
        saved_full_config = resume_config_record['full_config']
        full_config_snapshot = full._safe_config_snapshot(saved_full_config)
        current_materialized_config_snapshot = full._materialized_config_snapshot(
            current_yaml_config_snapshot,
            (current_yaml_config_snapshot.get('training', {})
             if isinstance(current_yaml_config_snapshot, dict)
             else {}))
        selected_full_config_sha256 = full._config_sha256(
            full_config_snapshot)
        current_materialized_config_sha256 = full._config_sha256(
            current_materialized_config_snapshot)
        checkpoint_full_config_sha256 = full._config_sha256(saved_full_config)
        if is_host0:
            print(
                "Config hash: "
                f"selected_full_config_sha256={selected_full_config_sha256} "
                f"checkpoint_full_config_sha256={checkpoint_full_config_sha256} "
                "current_materialized_config_sha256="
                f"{current_materialized_config_sha256}",
                flush=True)
            try:
                session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                session_path = full._join_path(
                    checkpoint_dir,
                    f"config_resume_session_{session_ts}.json")
                session_cfg = {
                    'type': 'resume_session_config',
                    'timestamp': datetime.now().isoformat(),
                    'config_path': str(config_path),
                    'resume_path': resume_path,
                    'resume_step': int(resume_step),
                    'config_json_read_only': True,
                    'resume_config_source': resume_config_record.get(
                        'source', 'checkpoint full_config'),
                    'resume_config_fallback': 'disabled',
                    'checkpoint_raw_config_present':
                        resume_config_record.get('raw_config') is not None,
                    'current_materialized_config_sha256':
                        current_materialized_config_sha256,
                    'checkpoint_full_config_sha256':
                        checkpoint_full_config_sha256,
                    'selected_full_config_sha256':
                        selected_full_config_sha256,
                    'current_yaml_config':
                        full._safe_config_snapshot(current_yaml_config_snapshot),
                    'checkpoint_full_config':
                        full._safe_config_snapshot(saved_full_config),
                    'selected_full_config': full_config_snapshot,
                    'current_raw_config': raw_config_snapshot,
                    'current_materialized_config':
                        current_materialized_config_snapshot,
                }
                full._write_json_file(session_path, session_cfg)
                print(f"  Saved resume session config snapshot: {session_path}")
            except Exception as exc:
                print(
                    f"  Warning: Failed to save resume session config: {exc}")
            print(
                "Resume detected; preserving existing config.json and "
                "config_raw.json snapshots.")
            print("  Skipped run-folder config snapshot rewrite on resume.")
    _stage_log("after_config_snapshot")
    _stage_barrier("after_config_snapshot")

    mesh_model = int(cfg['training'].get('mesh_model', 1))
    mesh_data = int(cfg['training'].get('mesh_data', 0))
    total_devices = jax.device_count()
    if mesh_data == 0:
        mesh_data = total_devices // mesh_model
    if batch_size % mesh_data != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"mesh_data={mesh_data}")
    per_data_shard_batch = batch_size // mesh_data
    _stage_log("before_create_mesh")
    mesh = full.create_mesh(mesh_data, mesh_model)
    _stage_log("after_create_mesh", {
        "mesh_data": mesh_data,
        "mesh_model": mesh_model,
        "total_devices": total_devices,
    })
    _stage_barrier("after_create_mesh")
    data_sharding = NamedSharding(mesh, P('data', None))
    if is_host0:
        print(
            f"\n=== Mesh: ({mesh_data}, {mesh_model}) = "
            f"{total_devices} devices, per_device_batch={per_device_batch}, "
            f"per_data_shard_batch={per_data_shard_batch} ===")

    _stage_log("before_shard_params")
    if 'model_version' not in inspect.signature(
            full.get_param_shardings).parameters:
        raise RuntimeError(
            "full.get_param_shardings must accept model_version before "
            "scripts/train_jax_minimal.py can run v4167 minimal training. "
            "Update scripts/train_jax.py first.")
    param_shardings = full.get_param_shardings(
        params, mesh, model_version=model_version_cfg)
    if is_host0:
        full._print_param_sharding_summary(
            param_shardings, model_version_cfg)
    params = full.shard_params_to_mesh(params, param_shardings)
    _stage_log("after_shard_params")
    _stage_barrier("after_shard_params")

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

    _MODEL_VERSION = str(model_version_cfg)
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

    _stage_log("before_optimizer_init")
    opt_state = optimizer.init(params)
    opt_state = full._replicate_optimizer_state_scalars_to_mesh(opt_state, mesh)
    _stage_log("after_optimizer_init")
    _stage_barrier("after_optimizer_init")
    target_params = params
    target_opt_state = opt_state

    checkpoint_manager_path = full._join_path(checkpoint_dir, 'checkpoints')
    if is_host0:
        print(
            "Minimal trainer checkpoint manager: "
            f"{'enabled' if checkpoint_manager_required else 'disabled'}",
            flush=True)
        print(f"checkpoint_interval={ckpt_interval}", flush=True)
        print(f"manager path={checkpoint_manager_path}", flush=True)
        if ckpt_interval <= 0:
            print(
                "Checkpointing disabled because checkpoint_interval <= 0",
                flush=True)
        if resume_step is not None and ckpt_interval <= 0:
            print(
                "Orbax CheckpointManager still enabled for requested resume.",
                flush=True)

    latest_checkpoint_manager = None
    if checkpoint_manager_required:
        latest_checkpoint_manager = full._create_orbax_checkpoint_manager(
            checkpoint_manager_path,
            checkpoint_interval=ckpt_interval,
            keep_last=checkpoint_keep_last,
            create=True,
            best_tracking=False)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if resume_step is not None:
        target_state = full._build_orbax_state(
            target_params, target_opt_state, rng,
            epoch=0,
            global_step=0,
            step_in_epoch=0,
            steps_per_epoch=steps_per_epoch,
            best_val_loss=best_val_loss,
            training_config=training_config,
            full_config=full_config_snapshot,
            model_config=cfg['model'],
        )
        restored_state, restored_metadata = full._restore_orbax_state(
            latest_checkpoint_manager, resume_step, target_state)
        params = full._match_tree_to_template_on_mesh(
            restored_state['params'], target_params, mesh, name='params')
        opt_state = full._match_tree_to_template_on_mesh(
            restored_state['opt_state'], target_opt_state, mesh,
            name='opt_state')
        rng = jnp.asarray(
            np.asarray(restored_state['rng'], dtype=np.uint32).reshape((2,)),
            dtype=jnp.uint32)
        start_epoch = full._state_scalar(restored_state, 'epoch', 0, int)
        global_step = full._state_scalar(
            restored_state, 'global_step',
            full._state_scalar(restored_state, 'step', 0, int),
            int)
        best_val_loss = full._state_scalar(
            restored_state, 'best_val_loss', float('inf'), float)
        saved_step_in_epoch = full._state_scalar(
            restored_state, 'step_in_epoch', 0, int)
        saved_steps_per_epoch = full._state_scalar(
            restored_state, 'steps_per_epoch', 0, int)
        if saved_step_in_epoch > 0 and saved_steps_per_epoch == steps_per_epoch:
            start_step_in_epoch = saved_step_in_epoch
        if is_host0:
            print(
                "Restored Orbax checkpoint: "
                f"epoch={start_epoch}, global_step={global_step}, "
                f"step_in_epoch={start_step_in_epoch}, "
                f"best_val_loss={best_val_loss:.4f}")
            if restored_metadata:
                print(
                    "  Restored checkpoint metadata kind="
                    f"{restored_metadata.get('checkpoint_kind', '<unknown>')}")
    elif is_host0 and checkpointing_enabled:
        print(f"Orbax checkpoints: {checkpoint_manager_path}")

    if n_hosts > 1:
        gathered = np.asarray(process_allgather(
            np.array([global_step], dtype=np.int64))).flatten()
        if not np.all(gathered == global_step):
            raise RuntimeError(
                f"global_step inconsistent across hosts: {gathered.tolist()}")

    _stage_log("before_make_minimal_sharded_fns")
    sharded_fns, chunk_counts, max_chunks = _make_minimal_sharded_fns(
        cfg, mesh, mesh_model, batch_size, max_seq_len,
        per_data_shard_batch, is_host0)
    _stage_log("after_make_minimal_sharded_fns", {
        "chunk_counts": [int(x) for x in chunk_counts],
        "max_chunks": [int(x) for x in max_chunks],
    })
    _stage_barrier("after_make_minimal_sharded_fns")
    del chunk_counts, max_chunks

    step_kwargs = dict(
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
    _stage_log("before_create_train_eval_steps")
    train_step_fn = create_train_step_minimal(
        model, optimizer, sharded_fns=sharded_fns,
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
        **step_kwargs)
    eval_step_fn = create_eval_step_minimal(
        model, sharded_fns=sharded_fns, **step_kwargs)
    _stage_log("after_create_train_eval_steps")
    _stage_barrier("after_create_train_eval_steps")

    if is_host0:
        print("\nTraining config:")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  LR: {lr}")
        print(f"  Log interval: {log_interval}")
        print(f"  Val interval: {val_interval}")
        print(f"  Checkpoint interval: {ckpt_interval}")
        print("  analysis kernels=off")

    log_dir = checkpoint_dir
    full._makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    existing_logs = sorted(full._list_files(log_dir, "training_log_*.txt"))
    existing_jsonls = sorted(full._list_files(log_dir, "metrics_*.jsonl"))
    append_logs = bool(
        training_config.get('training_log_append_on_resume', True)
        and resume_path is not None and existing_logs)
    if append_logs:
        training_log_file = existing_logs[-1]
        jsonl_log_file = (
            existing_jsonls[-1] if existing_jsonls
            else full._join_path(log_dir, f'metrics_{timestamp}.jsonl'))
    else:
        training_log_file = full._join_path(
            log_dir, f'training_log_{timestamp}.txt')
        jsonl_log_file = full._join_path(
            log_dir, f'metrics_{timestamp}.jsonl')
    full._setup_loggers(training_log_file, jsonl_log_file, resume=append_logs)
    if is_host0:
        full.log_message(
            f"DAWN {model_version_cfg} Minimal Training Log - {timestamp}")
        full.log_message(f"Config: {config_path}")
        if deprecated_log_dir_present:
            full.log_message(
                "Deprecated top-level log_dir ignored; logs are stored in "
                "the run folder.")
        full.log_message(f"Parameters: {full.count_parameters(params):,}")
        full.log_message(f"Hosts: {n_hosts}, total_devices={jax.device_count()}")
        full.log_message(f"Total steps: {total_steps}")
        if tau_init_summary is not None:
            full.log_jsonl(tau_init_summary)
        if selection_calibration_summary is not None:
            full.log_jsonl(selection_calibration_summary)
        full.sync_logs()

    if start_step_in_epoch > 0:
        train_loader.reset(start_step=start_step_in_epoch)

    preemption_requested = [False]

    def handle_preemption(signum, frame):
        del signum, frame
        if preemption_requested[0]:
            return
        preemption_requested[0] = True
        print(
            f"\nSIGTERM received (host {host_id}) at step={global_step}; "
            "will save cooperatively.",
            flush=True)

    signal.signal(signal.SIGTERM, handle_preemption)

    checkpoint_git_info = full._safe_git_info()
    run_id = full._path_name(checkpoint_dir)
    train_start_time = time.time()
    total_micro_steps = num_epochs * steps_per_epoch
    epoch_step_counter = start_step_in_epoch

    if is_host0:
        print("\n=== Starting minimal training loop ===", flush=True)
    _stage_log("before_training_loop")
    _stage_barrier("before_training_loop")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_steps = 0
        win_start_time = time.time()
        win_count = 0
        win_loss = jnp.float32(0.0)
        win_ce = jnp.float32(0.0)
        win_correct = jnp.float32(0.0)
        win_valid = jnp.float32(0.0)
        epoch_loss = jnp.float32(0.0)
        epoch_correct = jnp.float32(0.0)
        epoch_valid = jnp.float32(0.0)
        last_grad = jnp.float32(0.0)

        for local_step, (input_ids, attention_mask) in enumerate(train_loader):
            if local_step % 10 == 0:
                preempt_any = bool(np.any(process_allgather(
                    np.array([preemption_requested[0]], dtype=np.bool_))))
                if preempt_any:
                    preemption_requested[0] = True
            if preemption_requested[0]:
                break

            rng, step_rng = jax.random.split(rng)
            step_rng = jax.random.fold_in(step_rng, host_id)
            input_ids = full.shard_to_mesh(
                input_ids, data_sharding, (batch_size, max_seq_len))
            attention_mask = full.shard_to_mesh(
                attention_mask, data_sharding, (batch_size, max_seq_len))

            params, opt_state, metrics = train_step_fn(
                params, opt_state, input_ids, attention_mask, step_rng,
                jnp.asarray(global_step, jnp.int32))

            valid_f = metrics['valid_count'].astype(jnp.float32)
            win_loss = win_loss + metrics['total_loss'] * valid_f
            win_ce = win_ce + metrics['ce_loss'] * valid_f
            win_correct = win_correct + metrics['correct'].astype(jnp.float32)
            win_valid = win_valid + valid_f
            epoch_loss = epoch_loss + metrics['ce_loss'] * valid_f
            epoch_correct = (
                epoch_correct + metrics['correct'].astype(jnp.float32))
            epoch_valid = epoch_valid + valid_f
            last_grad = metrics['grad_norm']
            win_count += 1
            epoch_steps += 1
            global_step += 1
            epoch_step_counter += 1

            is_early_log = global_step in (1, 5, 10, 20, 50)
            do_log = (global_step % log_interval == 0) or is_early_log
            do_val = (global_step % val_interval == 0 and global_step > 0)
            do_ckpt = (
                ckpt_interval > 0
                and global_step % ckpt_interval == 0
                and global_step > 0)

            if do_log:
                vals = jax.device_get({
                    'loss': win_loss,
                    'ce': win_ce,
                    'correct': win_correct,
                    'valid': win_valid,
                    'grad': last_grad,
                })
                valid_py = float(vals['valid'])
                denom = valid_py if valid_py > 0 else 1.0
                loss_py = float(vals['loss']) / denom
                ce_py = float(vals['ce']) / denom
                acc_py = float(vals['correct']) / denom
                grad_py = float(vals['grad'])
                if not np.isfinite(loss_py):
                    raise ValueError(
                        f"NaN/INF loss at epoch {epoch}, step {global_step}")
                if is_host0:
                    elapsed = time.time() - win_start_time
                    sec_per_it = elapsed / max(1, win_count)
                    current_lr = float(schedule(global_step // grad_accum_steps))
                    pct = 100.0 * global_step / max(1, total_micro_steps)
                    elapsed_total = time.time() - train_start_time
                    remaining_steps = max(total_micro_steps - global_step, 0)
                    eta_seconds = remaining_steps * sec_per_it
                    eta_str = full.format_time(eta_seconds)
                    elapsed_str = full.format_time(elapsed_total)
                    tokens_per_step = batch_size * max_seq_len
                    tok_per_sec = tokens_per_step / max(sec_per_it, 1.0e-12)
                    line = (
                        f"[Step {global_step}/{total_micro_steps} "
                        f"({pct:.1f}%)] "
                        f"loss={loss_py:.4f} ce={ce_py:.4f} aux=0.0000 | "
                        f"grad={grad_py:.4f} | acc={acc_py:.4f} "
                        f"lr={current_lr:.6g} "
                        f"time={sec_per_it:.2f}s/it "
                        f"eta={eta_str} elapsed={elapsed_str} "
                        f"tok/s={tok_per_sec:.0f}")
                    full.log_message(line)
                    full.log_jsonl({
                        'type': 'train',
                        'step': int(global_step),
                        'epoch': int(epoch),
                        'total_loss': ce_py,
                        'loss': ce_py,
                        'ce_loss': ce_py,
                        'aux_loss': 0.0,
                        'tau_reg': 0.0,
                        'orth_loss': 0.0,
                        'div_loss': 0.0,
                        'accuracy': acc_py,
                        'acc': acc_py,
                        'grad_norm': grad_py,
                        'lr': current_lr,
                        'sec_per_it': sec_per_it,
                        'eta_seconds': eta_seconds,
                        'elapsed_seconds': elapsed_total,
                        'tok_per_sec': tok_per_sec,
                        'timestamp': datetime.now().isoformat(),
                    })
                    full.sync_logs()
                win_start_time = time.time()
                win_count = 0
                win_loss = jnp.float32(0.0)
                win_ce = jnp.float32(0.0)
                win_correct = jnp.float32(0.0)
                win_valid = jnp.float32(0.0)

            if do_val:
                if is_host0:
                    full.log_message(f"Validation at step {global_step}...")
                val_loader.reset()
                val_loss, val_acc = full.evaluate(
                    eval_step_fn, params, val_loader, n_local_devices,
                    verbose=is_host0, data_sharding_spec=data_sharding,
                    return_dead_stats=False, current_step=global_step)
                if np.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                if is_host0:
                    full.log_message(
                        f"  Val loss={val_loss:.4f}, Val acc={val_acc:.4f}")
                    full.log_jsonl({
                        'type': 'val',
                        'step': int(global_step),
                        'epoch': int(epoch),
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc),
                        'timestamp': datetime.now().isoformat(),
                    })
                    full.sync_logs()

            if do_ckpt:
                saved = full.save_orbax_checkpoint(
                    latest_checkpoint_manager,
                    params, opt_state, rng,
                    epoch, global_step, epoch_step_counter,
                    steps_per_epoch, best_val_loss,
                    cfg['model'], training_config,
                    full_config_snapshot, raw_config_snapshot,
                    config_path, run_id,
                    'regular',
                    val_loss=None,
                    git_info=checkpoint_git_info)
                if is_host0 and saved:
                    full.log_message(
                        f"  Orbax checkpoint saved at step {global_step}")

        if preemption_requested[0]:
            try:
                if checkpointing_enabled:
                    full.save_orbax_checkpoint(
                        latest_checkpoint_manager,
                        params, opt_state, rng,
                        epoch, global_step, epoch_step_counter,
                        steps_per_epoch, best_val_loss,
                        cfg['model'], training_config,
                        full_config_snapshot, raw_config_snapshot,
                        config_path, run_id,
                        'emergency',
                        git_info=checkpoint_git_info,
                        wait=True)
                    if is_host0:
                        print(
                            f"Emergency checkpoint saved at step {global_step}",
                            flush=True)
                elif is_host0:
                    print(
                        "Emergency checkpoint skipped because "
                        "checkpoint_interval <= 0",
                        flush=True)
            except Exception as exc:
                if is_host0:
                    print(f"Emergency save failed: {exc}", flush=True)
            break

        ep = jax.device_get({
            'loss': epoch_loss,
            'correct': epoch_correct,
            'valid': epoch_valid,
        })
        ep_valid = float(ep['valid'])
        ep_loss = float(ep['loss']) / ep_valid if ep_valid > 0 else 0.0
        ep_acc = float(ep['correct']) / ep_valid if ep_valid > 0 else 0.0
        if is_host0:
            full.log_message(
                f"Epoch {epoch} complete in "
                f"{full.format_time(time.time() - epoch_start)} "
                f"train_loss={ep_loss:.4f} train_acc={ep_acc:.4f}")

        if epoch < num_epochs - 1:
            train_loader.reset(start_step=0)
            epoch_step_counter = 0

    if not preemption_requested[0] and checkpointing_enabled:
        final_epoch = int(epoch + 1) if 'epoch' in locals() else int(start_epoch)
        try:
            full.save_orbax_checkpoint(
                latest_checkpoint_manager,
                params, opt_state, rng,
                final_epoch, global_step, epoch_step_counter,
                steps_per_epoch, best_val_loss,
                cfg['model'], training_config,
                full_config_snapshot, raw_config_snapshot,
                config_path, run_id,
                'final',
                git_info=checkpoint_git_info,
                wait=True)
            if is_host0:
                full.log_message(
                    f"Final Orbax checkpoint saved at step {global_step}")
        except Exception as exc:
            if is_host0:
                print(f"Warning: final Orbax checkpoint failed: {exc}")
    elif (not preemption_requested[0]) and is_host0:
        print(
            "Final checkpoint skipped because checkpoint_interval <= 0",
            flush=True)

    if is_host0:
        elapsed = time.time() - train_start_time
        full.log_message(
            f"Minimal training complete: step={global_step}, "
            f"elapsed={full.format_time(elapsed)}")
        full.sync_logs()

    if latest_checkpoint_manager is not None:
        latest_checkpoint_manager.close()


if __name__ == '__main__':
    main()
