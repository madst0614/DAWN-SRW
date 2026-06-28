"""
Minimal DAWN-SRW v4.1.6.6 JAX trainer.

This path is dedicated to large v4166 CE training.  It reuses the full
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


POOL_SCHEDULE_NAMES = ('qk', 'v', 'rst')


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
    config_path = full._join_path(checkpoint_dir, 'config.json')
    config_json = None
    if is_host0 and full._file_exists(config_path):
        config_json = json.dumps(full._read_json_file(config_path))
    config_json = _broadcast_str_from_host0(config_json, max_len=2 * 1024 * 1024)
    if not config_json:
        if is_host0:
            print("  Warning: resume config.json not found; using current YAML.")
        return cfg, None
    record = json.loads(config_json)
    saved_cfg = {
        k: deepcopy(v) for k, v in record.items()
        if not str(k).startswith('_')
    }
    if is_host0:
        print(f"  Training config restored from {config_path}")
    return saved_cfg, record


def _path_str(path):
    return '/'.join(str(p.key if hasattr(p, 'key') else p) for p in path)


_POOL_PARAM_NAMES = (
    'attn_qk_emb', 'attn_v_emb', 'rst_emb',
    'attn_qk_op_read_proj', 'attn_qk_op_write_proj',
    'attn_v_op_read_proj', 'attn_v_op_write_proj',
    'rst_op_read_proj', 'rst_op_write_proj',
    'qk_emb', 'v_emb',
    'q_read', 'k_read',
    'attn_qk_read', 'attn_v_read', 'rst_read',
    'qk_read', 'v_read',
    'q_write', 'k_write',
    'attn_qk_write', 'attn_v_write', 'rst_write',
    'qk_write', 'v_write',
)


def _is_pool_param(path_str):
    return any(name in path_str for name in _POOL_PARAM_NAMES)


def _is_excluded_from_wd(path_str):
    leaf = path_str.rsplit('/', 1)[-1]
    if leaf == 'bias':
        return True
    if 'scale' in path_str and 'norm' in path_str.lower():
        return True
    return (
        path_str.endswith('_scale')
        or path_str.endswith('/qk_scale')
        or path_str.endswith('/v_scale')
        or path_str.endswith('/rst_scale')
        or path_str.endswith('/attn_qk_scale')
        or path_str.endswith('/attn_v_scale'))


def create_optimizer_minimal(params, schedule, weight_decay,
                             pool_weight_decay, global_grad_clip=0.0):
    def _wd_mask_base(params):
        def _f(path, _):
            ps = _path_str(path)
            return (not _is_excluded_from_wd(ps)) and (not _is_pool_param(ps))
        return jax.tree.map_with_path(_f, params)

    def _wd_mask_pool(params):
        def _f(path, _):
            ps = _path_str(path)
            return (not _is_excluded_from_wd(ps)) and _is_pool_param(ps)
        return jax.tree.map_with_path(_f, params)

    def _no_param_mask(params):
        return jax.tree.map(lambda _: False, params)

    parts = [optax.masked(optax.set_to_zero(), mask=_no_param_mask)]
    if float(global_grad_clip) > 0.0:
        parts.append(optax.clip_by_global_norm(global_grad_clip))
    else:
        parts.append(optax.scale(1.0))
    parts.extend([
        optax.scale_by_adam(b2=0.95),
        optax.add_decayed_weights(weight_decay, mask=_wd_mask_base),
        optax.add_decayed_weights(pool_weight_decay, mask=_wd_mask_pool),
        optax.scale_by_learning_rate(schedule),
        optax.masked(optax.set_to_zero(), mask=_no_param_mask),
    ])
    return optax.chain(*parts)


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

    @jax.jit
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
                              per_device_batch, is_host0):
    model_version = cfg['model'].get('model_version', full.OFFICIAL_MODEL_VERSION)
    if not full._is_active_srw_version(model_version):
        return None, (1, 1, 1), (1, 1, 1)

    target_chunk_gb = cfg['training'].get('target_chunk_gb', 2.0)
    n_rst = cfg['model'].get('n_rst', cfg['model'].get('n_know', 25200))
    n_qk = cfg['model'].get('n_qk', cfg['model'].get('n_q', 1580))
    n_v = cfg['model'].get('n_v', 2600)
    for name, n in (('n_rst', n_rst), ('n_qk', n_qk), ('n_v', n_v)):
        if n % mesh_model != 0:
            raise ValueError(
                f"{name}={n} must be divisible by mesh_model={mesh_model}")

    nrst_local = n_rst // mesh_model
    nqk_local = n_qk // mesh_model
    nv_local = n_v // mesh_model

    def auto_n_chunks(n, target_gb=2.0):
        full_gb = per_device_batch * max_seq_len * n * 2 / 1e9
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

    sharded_single_v = make_single(
        max_chunk_size=attn_v_max_chunk,
        **_factory_kwargs(make_single, base_kwargs))
    sharded_single_rst = make_single(
        max_chunk_size=rst_max_chunk,
        **_factory_kwargs(make_single, base_kwargs))
    sharded_paired_attn_qk = make_paired(
        max_chunk_size=attn_qk_max_chunk,
        **_factory_kwargs(make_paired, base_kwargs))

    if is_host0:
        print(
            "  shard_map minimal enabled "
            f"(mesh_model={mesh_model}, QK fused; "
            f"chunks attn_qk/attn_v/rst={n_chunks_qk}/{n_chunks_v}/{n_chunks_rst}; "
            f"max_chunk attn_qk/attn_v/rst={attn_qk_max_chunk}/{attn_v_max_chunk}/{rst_max_chunk}; "
            "analysis kernels=off)")

    return {
        'single': sharded_single_rst,
        'attn_v_single': sharded_single_v,
        'rst_single': sharded_single_rst,
        'paired': sharded_paired_attn_qk,
        'attn_qk_paired': sharded_paired_attn_qk,
        'attn_v_single_minimal': sharded_single_v,
        'rst_single_minimal': sharded_single_rst,
        'attn_qk_paired_minimal': sharded_paired_attn_qk,
    }, (n_chunks_qk, n_chunks_v, n_chunks_rst), (
        attn_qk_max_chunk, attn_v_max_chunk, rst_max_chunk)


def _write_fresh_config_snapshot(checkpoint_dir, cfg, raw_cfg_snapshot,
                                 config_path, training_config):
    full_config_snapshot = deepcopy(cfg)
    raw_config_snapshot = deepcopy(raw_cfg_snapshot)
    materialized_snapshot = full._materialized_config_snapshot(
        full_config_snapshot, training_config)
    materialized_sha = full._config_sha256(materialized_snapshot)
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
    return full_config_snapshot, raw_config_snapshot


def main():
    parser = argparse.ArgumentParser(
        description='Minimal DAWN-SRW v4166 JAX trainer')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume-from', '--resume', dest='resume_from',
                        type=str, default=None)
    cli_args = parser.parse_args()

    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        if full._file_exists(cli_args.config):
            config_path = cli_args.config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = full.load_config(config_path)
    raw_cfg_snapshot = deepcopy(cfg)
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

    seed = cfg.get('seed', 42)
    full.set_seed(seed)
    tcfg = cfg['training']
    model_version_cfg = cfg['model'].get(
        'model_version', full.OFFICIAL_MODEL_VERSION)
    if model_version_cfg != full.V4166_MODEL_VERSION:
        raise ValueError(
            "scripts/train_jax_minimal.py is dedicated to "
            f"{full.V4166_MODEL_VERSION}, got {model_version_cfg!r}")

    tau_init_cfg = full._v4164_tau_init_config(cfg)
    selection_calibration_cfg = full._selection_calibration_config(
        cfg, tau_init_cfg)

    batch_size = cli_args.batch_size or int(tcfg['batch_size'])
    num_epochs = cli_args.epochs or int(tcfg['num_epochs'])
    lr = cli_args.lr or float(tcfg.get('lr', tcfg.get('learning_rate', 6.5e-4)))
    weight_decay = float(tcfg.get('weight_decay', 0.1))
    pool_weight_decay = 0.0
    warmup_ratio = float(tcfg.get('warmup_ratio', 0.06))
    grad_accum_steps = int(tcfg.get('gradient_accumulation_steps', 1))
    if grad_accum_steps != 1:
        raise ValueError(
            "Minimal trainer does not use gradient accumulation; "
            f"got gradient_accumulation_steps={grad_accum_steps}")

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
        'gradient_accumulation_steps': grad_accum_steps,
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

    print(
        f"[Host {host_id}/{n_hosts}] local_devices={n_local_devices} "
        f"total_devices={jax.device_count()} backend={jax.default_backend()} "
        f"devices={[str(d) for d in jax.local_devices()]}",
        flush=True)

    train_script_git_commit = full._safe_git_info().get('git_commit')
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
    full._assert_multihost_same_startup_context({
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
        print("\n=== DAWN-SRW v4166 Minimal Training ===")
        print(f"Config: {config_path}")
        print(f"Run folder: {checkpoint_dir}")
        print(f"Global batch size: {batch_size}")
        print(f"Per-host batch size: {per_host_batch}")
        print(f"Per-device batch size: {per_device_batch}")
        print("Analysis/geometry/prune/drift diagnostics: off")

    from utils.data_jax import load_data
    train_loader, val_loader, vocab_size = load_data(
        cfg['data'],
        max_length=max_seq_len,
        batch_size=batch_size,
        n_devices=1,
        n_hosts=n_hosts,
        host_id=host_id,
    )
    if is_host0:
        print(f"Vocab size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    cfg['model']['vocab_size'] = vocab_size
    model = full.build_model_from_config(cfg)
    if not _model_accepts_minimal_train(model):
        raise RuntimeError(
            "v4166 model does not expose minimal_train; minimal trainer "
            "requires the model-level minimal path.")

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
        print(f"Model parameters: {full.count_parameters(params):,}")
        for line in model.get_model_info():
            print(line)

    selection_calibration_summary = None
    tau_init_summary = None
    if (selection_calibration_cfg.get('enabled', False)
            and resume_step is None):
        if len(train_loader) <= 0:
            raise RuntimeError(
                "selection_calibration requires at least one training batch.")
        if is_host0:
            print("\nSelection calibration: computing from fresh init.")
        (hist_counts, seen_tokens, actual_batches,
         local_calibration_tokens, calibration_process_count) = (
            full._collect_selection_calibration_histograms(
                params, train_loader, cfg, selection_calibration_cfg))
        hist_counts, seen_tokens, actual_batches = (
            full._aggregate_selection_calibration_histograms(
                hist_counts, seen_tokens, actual_batches))
        selection_json = None
        if is_host0:
            selection_calibration_summary = (
                full._compute_srw_selection_calibration(
                    hist_counts, cfg, selection_calibration_cfg,
                    seen_tokens, actual_batches,
                    local_calibration_tokens, calibration_process_count))
            selection_json = json.dumps(selection_calibration_summary)
        selection_json = _broadcast_str_from_host0(
            selection_json, max_len=32768)
        selection_calibration_summary = json.loads(selection_json)
        params = full._set_srw_quantile_tau_biases(
            params, selection_calibration_summary, model_version_cfg)
        materialized_updates = (
            full._selection_calibration_materialized_training_updates(
                selection_calibration_summary, selection_calibration_cfg))
        soft_gate_pool_schedules = (
            full._apply_selection_calibrated_soft_gate_schedules(
                soft_gate_pool_schedules, materialized_updates))
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
        if is_host0:
            for line in full._v4164_tau_init_summary_lines(tau_init_summary):
                print(line)

    if resume_config_record is None and is_host0:
        full_config_snapshot, raw_config_snapshot = _write_fresh_config_snapshot(
            checkpoint_dir, cfg, raw_cfg_snapshot, config_path, training_config)
        print(f"Saved config.json: {full._join_path(checkpoint_dir, 'config.json')}")
    else:
        full_config_snapshot = deepcopy(cfg)
        raw_config_snapshot = deepcopy(
            resume_config_record.get('_raw_config', cfg)
            if resume_config_record else raw_cfg_snapshot)

    mesh_model = int(cfg['training'].get('mesh_model', 1))
    mesh_data = int(cfg['training'].get('mesh_data', 0))
    total_devices = jax.device_count()
    if mesh_data == 0:
        mesh_data = total_devices // mesh_model
    mesh = full.create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    if is_host0:
        print(
            f"\n=== Mesh: ({mesh_data}, {mesh_model}) = "
            f"{total_devices} devices, per_device_batch={per_device_batch} ===")

    param_shardings = full.get_param_shardings(params, mesh)
    params = full.shard_params_to_mesh(params, param_shardings)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )
    optimizer = create_optimizer_minimal(
        params, schedule, weight_decay, pool_weight_decay,
        global_grad_clip=global_grad_clip)
    opt_state = optimizer.init(params)
    opt_state = full._replicate_optimizer_state_scalars_to_mesh(opt_state, mesh)
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

    sharded_fns, chunk_counts, max_chunks = _make_minimal_sharded_fns(
        cfg, mesh, mesh_model, batch_size, max_seq_len,
        per_device_batch, is_host0)
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

    log_dir = cfg.get('log_dir', full._join_path(checkpoint_dir, 'logs'))
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
    total_micro_steps = total_steps
    epoch_step_counter = start_step_in_epoch

    if is_host0:
        print("\n=== Starting minimal training loop ===", flush=True)

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

            do_log = (global_step % log_interval == 0) or global_step in (1, 2, 3)
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
                    current_lr = float(schedule(global_step))
                    line = (
                        f"[Step {global_step}/{total_micro_steps}] "
                        f"loss={loss_py:.4f} ce={ce_py:.4f} "
                        f"acc={acc_py:.4f} grad={grad_py:.4f} "
                        f"lr={current_lr:.6g} time={sec_per_it:.2f} sec/it")
                    full.log_message(line)
                    full.log_jsonl({
                        'type': 'train',
                        'step': int(global_step),
                        'epoch': int(epoch),
                        'loss': loss_py,
                        'ce_loss': ce_py,
                        'acc': acc_py,
                        'grad_norm': grad_py,
                        'lr': current_lr,
                        'sec_per_it': sec_per_it,
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
