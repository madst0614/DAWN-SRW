"""
DAWN-SRW v4.1.6.7 Global CEU Operator Pages

Page-sparse RW execution version of the v4166/v4167 experimental path.

Implemented concepts:
- cosine-space tau reference with bounded sigmoid min/max mapping
- one-sided generalized Gaussian boundary DirectTau admission
- boundary-relative drive RW composition
- RW-derived live-gradient operator-key selection
- RW-matched operator queries
- Global CEU operator pages with page-routed candidate RW execution
- scheduled soft-gate boundary-scale input
- scheduled boundary power input
- tau movement controlled by optimizer-side tau_lr_mult
- train-time effective gate statistics
- validation-time execution pruning through execution_prune_eps
"""


import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from flax.core import FrozenDict, freeze, unfreeze
from typing import Optional, Dict, Any
from functools import partial
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

SELECT_DIAG_NAMES = (
    'rho_mean',                 # analysis/validation only; zero on fast train path
    'rho_std',                  # analysis/validation only; zero on fast train path
    'rho_max',
    'tau_mean',
    'tau_min',
    'tau_max',
    'raw_tau_mean',
    'raw_tau_min',
    'raw_tau_max',
    'selection_margin_mean',
    'positive_margin_mean',
    'positive_margin_max',
    'selected_frac',
    'no_active_frac',
)
SELECT_DIAG_COUNT = len(SELECT_DIAG_NAMES)
(
    SELECT_RHO_MEAN,
    SELECT_RHO_STD,
    SELECT_RHO_MAX,
    SELECT_TAU_MEAN,
    SELECT_TAU_MIN,
    SELECT_TAU_MAX,
    SELECT_RAW_TAU_MEAN,
    SELECT_RAW_TAU_MIN,
    SELECT_RAW_TAU_MAX,
    SELECT_SELECTION_MARGIN_MEAN,
    SELECT_POSITIVE_MARGIN_MEAN,
    SELECT_POSITIVE_MARGIN_MAX,
    SELECT_SELECTED_FRAC,
    SELECT_NO_ACTIVE_FRAC,
) = range(SELECT_DIAG_COUNT)


# v4164 exposure diagnostics are admission based, not hard score>tau based.
# The historical DEAD_EXPOSURE_* constant names are kept as internal slot names
# so tuple layouts stay stable, but the values below mean:
#   mean/min/max of max_batch_token(admission_i), and dead fractions at
#   eps=1e-6, 1e-5, 1e-4.
DEAD_EXPOSURE_DIAG_NAMES = (
    'soft_exposure_mean',
    'soft_exposure_min',
    'soft_exposure_max',
    'soft_dead_frac_eps_1e_6',
    'soft_dead_frac_eps_1e_5',
    'soft_dead_frac_eps_1e_4',
)
DEAD_EXPOSURE_DIAG_COUNT = len(DEAD_EXPOSURE_DIAG_NAMES)
(
    DEAD_EXPOSURE_MEAN,
    DEAD_EXPOSURE_MIN,
    DEAD_EXPOSURE_MAX,
    DEAD_EXPOSURE_DEAD_FRAC,
    DEAD_EXPOSURE_WEAK_FRAC,
    DEAD_EXPOSURE_TARGET,
) = range(DEAD_EXPOSURE_DIAG_COUNT)


ATTN_SPLIT_CORE_NAMES = (
    'qk_raw_gate_max',
    'v_raw_gate_max',
    'qk_gate_sum',
    'v_gate_sum',
    'qk_active_n_mean',
    'v_active_n_mean',
    'qk_tau_abs_mean',
    'v_tau_abs_mean',
    'qk_dead_count',
    'v_dead_count',
    'qk_int_max',
    'v_int_max',
    'qk_gate_den_sum_mean',
    'v_gate_den_sum_mean',
    'qk_gate_eff_n',
    'v_gate_eff_n',
    'qk_gate_eff_ratio',
    'v_gate_eff_ratio',
    'qk_top1_gate_frac',
    'v_top1_gate_frac',
    'qk_top1_gate_frac_max',
    'v_top1_gate_frac_max',
    'qk_dead_penalty',
    'v_dead_penalty',
    'q_active_frac',
    'k_active_frac',
    'q_strong_frac',
    'k_strong_frac',
    'q_active_n_mean',
    'k_active_n_mean',
    'qk_drive_mean',
    'v_drive_mean',
)
ATTN_SPLIT_CORE_COUNT = len(ATTN_SPLIT_CORE_NAMES)
(
    ATTN_SPLIT_QK_RAW_GATE_MAX,
    ATTN_SPLIT_V_RAW_GATE_MAX,
    ATTN_SPLIT_QK_GATE_SUM,
    ATTN_SPLIT_V_GATE_SUM,
    ATTN_SPLIT_QK_ACTIVE_N_MEAN,
    ATTN_SPLIT_V_ACTIVE_N_MEAN,
    ATTN_SPLIT_QK_TAU_ABS_MEAN,
    ATTN_SPLIT_V_TAU_ABS_MEAN,
    ATTN_SPLIT_QK_DEAD_COUNT,
    ATTN_SPLIT_V_DEAD_COUNT,
    ATTN_SPLIT_QK_INT_MAX,
    ATTN_SPLIT_V_INT_MAX,
    ATTN_SPLIT_QK_GATE_DEN_SUM_MEAN,
    ATTN_SPLIT_V_GATE_DEN_SUM_MEAN,
    ATTN_SPLIT_QK_GATE_EFF_N,
    ATTN_SPLIT_V_GATE_EFF_N,
    ATTN_SPLIT_QK_GATE_EFF_RATIO,
    ATTN_SPLIT_V_GATE_EFF_RATIO,
    ATTN_SPLIT_QK_TOP1_GATE_FRAC,
    ATTN_SPLIT_V_TOP1_GATE_FRAC,
    ATTN_SPLIT_QK_TOP1_GATE_FRAC_MAX,
    ATTN_SPLIT_V_TOP1_GATE_FRAC_MAX,
    ATTN_SPLIT_QK_DEAD_PENALTY,
    ATTN_SPLIT_V_DEAD_PENALTY,
    ATTN_SPLIT_Q_ACTIVE_FRAC,
    ATTN_SPLIT_K_ACTIVE_FRAC,
    ATTN_SPLIT_Q_STRONG_FRAC,
    ATTN_SPLIT_K_STRONG_FRAC,
    ATTN_SPLIT_Q_ACTIVE_N_MEAN,
    ATTN_SPLIT_K_ACTIVE_N_MEAN,
    ATTN_SPLIT_QK_DRIVE_MEAN,
    ATTN_SPLIT_V_DRIVE_MEAN,
) = range(ATTN_SPLIT_CORE_COUNT)


GATE_CURRENT_EPS = (
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
)
GATE_PROJECTED_EPS = (
    1.0e-6,
    1.0e-4,
    1.0e-3,
)
MARGIN_BAND_NAMES = (
    'gt_0',
    'm0_01_0',
    'm0_03_m0_01',
    'm0_10_m0_03',
    'lt_m0_10',
)
GATE_EPS_NAME_SUFFIXES = (
    '1e_6',
    '1e_5',
    '1e_4',
    '1e_3',
    '1e_2',
    '1e_1',
)
GATE_PROJECTED_EPS_NAME_SUFFIXES = (
    '1e_6',
    '1e_4',
    '1e_3',
)
GATE_SPARSITY_DIAG_NAMES = (
    ('active_tau_frac', 'active_tau_count')
    + tuple(
        name
        for suffix in GATE_EPS_NAME_SUFFIXES
        for name in (
            f'admission_active_eps_{suffix}_frac',
            f'admission_active_eps_{suffix}_count',
        ))
    + tuple(
        name
        for suffix in GATE_EPS_NAME_SUFFIXES
        for name in (
            f'active_eps_{suffix}_frac',
            f'active_eps_{suffix}_count',
        ))
    + tuple(f'mass_eps_{suffix}' for suffix in GATE_EPS_NAME_SUFFIXES)
    + tuple(
        name
        for suffix in GATE_PROJECTED_EPS_NAME_SUFFIXES
        for name in (
            f'projected_Tfinal_active_eps_{suffix}_frac',
            f'projected_Tfinal_active_eps_{suffix}_count',
        ))
    + tuple(
        f'projected_Tfinal_mass_eps_{suffix}'
        for suffix in GATE_PROJECTED_EPS_NAME_SUFFIXES)
    + tuple(f'margin_band_{name}' for name in MARGIN_BAND_NAMES)
    + (
        'margin_band_pos',
        'margin_band_near_m0_03_0',
        'margin_band_far_lt_m0_10',
    )
)
GATE_SPARSITY_DIAG_COUNT = len(GATE_SPARSITY_DIAG_NAMES)
GATE_SPARSITY_DIAG_INDEX = {
    name: i for i, name in enumerate(GATE_SPARSITY_DIAG_NAMES)
}
GATE_EPS_SUFFIX_TO_VALUE = {
    '1e_6': 1.0e-6,
    '1e_5': 1.0e-5,
    '1e_4': 1.0e-4,
    '1e_3': 1.0e-3,
    '1e_2': 1.0e-2,
    '1e_1': 1.0e-1,
}

def _gate_eps_values_from_suffixes(suffixes):
    return tuple(float(GATE_EPS_SUFFIX_TO_VALUE[s]) for s in suffixes)


def _pool_output_scales(d_model, n_layers):
    dm = jnp.asarray(d_model, dtype=jnp.float32)
    nl = jnp.asarray(n_layers, dtype=jnp.float32)
    qk_scale = jnp.sqrt(dm / nl)
    v_scale = jnp.sqrt(dm / nl)
    rst_scale = jnp.sqrt(dm / nl)
    return (
        jax.lax.stop_gradient(qk_scale),
        jax.lax.stop_gradient(v_scale),
        jax.lax.stop_gradient(rst_scale),
    )


def _effective_pool_output_scales(pool_params, d_model, n_layers):
    """PureCore uses fixed depth-scaled pool outputs.

    The learned scale parameters stay in the checkpoint tree for resume stability,
    but the active forward path ignores them.
    """
    return _pool_output_scales(d_model, n_layers)


# ================================================================
# V4.1.6.4 annealed generalized Gaussian DirectTau in cosine space.
#
#   rho              = cosine(q, RW-derived operator key)
#   raw_tau          = learned cosine-space reference
#   tau              = -1 + 2 * sigmoid(raw_tau)
#   margin, r        = rho - tau, margin / boundary_scale
#   d_neg            = lambda_neg * softplus(-r / lambda_neg)
#   admission        = exp(-(d_neg ** boundary_power) - gamma(p) * exp(-r / A(p)))
#   drive            = softplus(margin / B) / softplus((1 - tau) / B)
#   execution_weight = admission * drive
#   admission_den    = max(sum(admission), 1.0) ** admission_den_power
# ================================================================

DEFAULT_D_ROUTE = 64
RW_FORWARD_NORM_EPS = 1e-6     # forward-only read/write direction floor
MODEL_VERSION = "spatial-r1-v4.1.6.7"


PAGE_DIAG_NAMES = (
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
    'page_fallback_used_frac',
    'page_random_used_frac',
    'page_no_route_frac',
    'selected_page_count',
    'estimated_compute_frac_page',
)
PAGE_DIAG_COUNT = len(PAGE_DIAG_NAMES)
PAGE_DIAG_INDEX = {name: i for i, name in enumerate(PAGE_DIAG_NAMES)}


# ================================================================
# 1. Helpers
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_rate, x.shape)
    dropped = jnp.where(mask, x / keep_rate, 0.0)
    # Eval path returns x unscaled; train path uses inverted dropout.
    return jnp.where(deterministic, x, dropped)


def _layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _forward_unit_direction(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True)
                + RW_FORWARD_NORM_EPS)


def _read_write_operator_query(read_query, x, write_query_proj, eps=1e-6):
    """RW-matched read-query x write-query signature for selection."""
    read_query = read_query.astype(jnp.float32)
    read_query = read_query / (
        jnp.linalg.norm(read_query, axis=-1, keepdims=True) + eps)

    write_query = x.astype(jnp.float32) @ write_query_proj
    write_query = write_query / (
        jnp.linalg.norm(write_query, axis=-1, keepdims=True) + eps)

    operator_query = read_query * write_query
    operator_query = operator_query / (
        jnp.linalg.norm(operator_query, axis=-1, keepdims=True) + eps)
    return operator_query.astype(jnp.float32)


def _rw_operator_key(read, write, read_proj, write_proj, *, eps=1e-6):
    """Compressed live-gradient RW operator identity for v4167 selection.

    Gradients flow through read/write and both projections. The projections
    are bias-free so (-read, -write) preserves the same key.
    """
    eps = jnp.asarray(eps, dtype=jnp.float32)

    def _unit(x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

    read_dir = _unit(read.astype(jnp.float32))
    write_dir = _unit(write.astype(jnp.float32))

    r_key = read_dir @ read_proj
    w_key = write_dir @ write_proj

    r_key = _unit(r_key.astype(jnp.float32))
    w_key = _unit(w_key.astype(jnp.float32))

    op_key = r_key * w_key
    op_key = _unit(op_key.astype(jnp.float32))

    return op_key.astype(jnp.float32)


def _pool_operator_keys(pool_params):
    return {
        'attn_qk_op_key': _rw_operator_key(
            pool_params['attn_qk_read'],
            pool_params['attn_qk_write'],
            pool_params['attn_qk_op_read_proj'],
            pool_params['attn_qk_op_write_proj']),
        'attn_v_op_key': _rw_operator_key(
            pool_params['attn_v_read'],
            pool_params['attn_v_write'],
            pool_params['attn_v_op_read_proj'],
            pool_params['attn_v_op_write_proj']),
        'rst_op_key': _rw_operator_key(
            pool_params['rst_read'],
            pool_params['rst_write'],
            pool_params['rst_op_read_proj'],
            pool_params['rst_op_write_proj']),
    }


def _pool_params_with_operator_keys(pool_params):
    """Attach per-forward shared op keys without recomputing in each layer."""
    out = unfreeze(pool_params) if isinstance(pool_params, FrozenDict) else dict(pool_params)
    out.update(_pool_operator_keys(pool_params))
    return out


def _ensure_pool_operator_keys(pool_params):
    if 'attn_qk_op_key' in pool_params:
        return pool_params
    return _pool_params_with_operator_keys(pool_params)


@jax.custom_vjp
def _block_tau_up_when_no_active(tau, no_active):
    return tau


def _block_tau_up_fwd(tau, no_active):
    return tau, no_active


def _block_tau_up_bwd(no_active, g):
    no_active = no_active.astype(jnp.bool_)
    block_up = no_active & (g < 0.0)
    g_tau = jnp.where(block_up, 0.0, g)
    return g_tau, None


_block_tau_up_when_no_active.defvjp(
    _block_tau_up_fwd,
    _block_tau_up_bwd,
)


def scaled_normal(scale=0.02):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


def unit_norm_init(scale=1.0):
    def init(key, shape, dtype=jnp.float32):
        x = jax.random.normal(key, shape, dtype)
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return x / norms * scale
    return init


TAU_MIN = -1.0
TAU_MAX = 1.0


def _tau_from_param(raw_tau):
    """v4164 cosine-space bounded tau using a sigmoid min/max map."""
    raw_tau = jnp.asarray(raw_tau, dtype=jnp.float32)
    return TAU_MIN + (TAU_MAX - TAU_MIN) * jax.nn.sigmoid(raw_tau)


def _raw_tau_init_from_cosine_tau(tau_init, eps=1.0e-4):
    """Map desired cosine-space tau in [-1, 1] to raw sigmoid parameter."""
    tau_init = jnp.asarray(tau_init, dtype=jnp.float32)
    p = (tau_init - TAU_MIN) / (TAU_MAX - TAU_MIN)
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.log(p) - jnp.log1p(-p)


def _boundary_soft_weight_from_margin(margin, boundary_scale, boundary_power):
    boundary_scale = jnp.maximum(
        jnp.asarray(boundary_scale, dtype=jnp.float32),
        jnp.float32(1.0e-4))
    boundary_power = jnp.maximum(
        jnp.asarray(boundary_power, dtype=jnp.float32),
        jnp.float32(1.0e-4))
    r = margin / boundary_scale

    lambda_neg = jnp.float32(0.10)
    d_neg = lambda_neg * jax.nn.softplus(-r / lambda_neg)

    alpha = jnp.clip(
        jnp.float32(4.0) - boundary_power,
        jnp.float32(0.0),
        jnp.float32(1.0))

    gamma_start = jnp.float32(0.025)
    gamma = gamma_start * jnp.power(alpha, jnp.float32(3.0))

    A_start = jnp.float32(3.0)
    A_final = jnp.float32(0.8)
    A = A_final + (A_start - A_final) * alpha

    active_exp_arg = jnp.minimum(-r / A, jnp.float32(30.0))
    active_gap = gamma * jnp.exp(active_exp_arg)
    return jnp.exp(-jnp.power(d_neg, boundary_power) - active_gap)


def _drive_from_margin(margin, tau, boundary_scale, eps=1.0e-6):
    boundary_scale = jnp.maximum(
        jnp.asarray(boundary_scale, dtype=jnp.float32),
        jnp.float32(1.0e-4))
    tau = jnp.asarray(tau, dtype=jnp.float32)
    max_margin = jnp.maximum(
        jnp.float32(1.0) - tau,
        jnp.float32(eps))

    numerator = jax.nn.softplus(margin / boundary_scale)
    denominator = jax.nn.softplus(max_margin / boundary_scale)
    return numerator / jnp.maximum(denominator, jnp.float32(eps))


def _boundary_gate_from_margin(margin, boundary_scale, boundary_power):
    """Compatibility name for projected admission diagnostics."""
    return _boundary_soft_weight_from_margin(
        margin, boundary_scale, boundary_power)


def _compute_admission_drive(score, tau, boundary_scale,
                             boundary_power=2.0,
                             effective_active_eps=1.0e-6,
                             execution_prune_eps=0.0):
    """v4164 admission plus boundary-relative drive composition."""
    execution_prune_eps = jnp.asarray(execution_prune_eps, dtype=jnp.float32)
    margin = score - tau
    admission_unpruned = _boundary_soft_weight_from_margin(
        margin, boundary_scale, boundary_power)
    admission = jnp.where(
        execution_prune_eps > 0.0,
        jnp.where(
            admission_unpruned >= execution_prune_eps,
            admission_unpruned, 0.0),
        admission_unpruned)
    drive = _drive_from_margin(
        margin, tau, boundary_scale)
    execution_weight = admission * drive
    active_mask = (
        admission >= jnp.asarray(effective_active_eps, dtype=jnp.float32))
    return margin, admission, drive, execution_weight, active_mask




def make_sharded_srw(mesh, max_chunk_size=2048,
                     analysis=False,
                     dead_exposure_target=0.1,
                     soft_gate_effective_active_eps=1.0e-6,
                     admission_den_power=1.0,
                     admission_den_grad_scale=1.0,
                     operator_pages_enabled=False,
                     operator_page_size=128,
                     operator_page_capacity=32,
                     operator_page_microgroup_sequences=2,
                     operator_page_score_mode='maxmean',
                     operator_page_fallback_pages=0,
                     operator_page_random_pages=0):
    """Create fused shard_map'd angular Select + SRW.

    Fast train path: one chunked pass computes rho, tau, gate, and SRW.
    Analysis path may compute rho distribution moments for diagnostics.

    One-sided generalized Gaussian boundary DirectTau admission:
        rho              = cosine(q, RW-derived operator key)
        margin           = rho - tau
        admission        = exp(-((max(tau - rho, 0) / B) ** p))
        drive            = softplus(margin / B) / softplus((1 - tau) / B)
        execution_weight = admission * drive
        admission_den    = max(sum(admission), 1.0) ** admission_den_power


    `analysis=False` (default, train path): returns the SLIM tuple plus
    four gate-concentration diagnostics, and skips distribution-shape stats
    (skew/kurt), selection-residency/entropy diagnostics and drive extrema.
    XLA DCE's the unused work.
    `analysis=True`: returns the SLIM/concentration tuple followed by
    observational scalars/arrays for route shape, gate concentration, and
    denominator diagnostics.
    Used by analysis_step at val time only.
    """
    if operator_pages_enabled:
        return make_sharded_srw_page(
            mesh, max_chunk_size=max_chunk_size, analysis=analysis,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            operator_page_size=operator_page_size,
            operator_page_capacity=operator_page_capacity,
            operator_page_microgroup_sequences=operator_page_microgroup_sequences,
            operator_page_score_mode=operator_page_score_mode,
            operator_page_fallback_pages=operator_page_fallback_pages,
            operator_page_random_pages=operator_page_random_pages)
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']
    _dead_exposure_target = jnp.float32(dead_exposure_target)
    _soft_gate_effective_active_eps = jnp.float32(soft_gate_effective_active_eps)
    _sparsity_diag_enabled = bool(analysis)
    _current_suffixes = GATE_EPS_NAME_SUFFIXES
    _projected_suffixes = GATE_PROJECTED_EPS_NAME_SUFFIXES
    _current_eps_values = _gate_eps_values_from_suffixes(_current_suffixes)
    _projected_eps_values = _gate_eps_values_from_suffixes(_projected_suffixes)
    _compute_sparsity_mass = True
    _compact_margin_bands = False
    _angular_strong_margin = jnp.float32(0.05)
    # v4164 divisive normalization:
    #   admission_den_forward = max(sum(admission), 1) ** admission_den_power
    #   backward gradient through admission_den is scaled by admission_den_grad_scale.
    # admission_den_power=0.5 is RMS/energy-like inhibition;
    # admission_den_grad_scale=0 detaches the inhibitory denominator gradient,
    # 1 restores the full live admission_den path.
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))

    # SLIM out_specs: train path.
    _slim_out_specs = (
        P('data', None, None),   # out [B,S,D]
        P('data', None, None),   # active [B,S,1]
        P('data', None, None),   # gate_max [B,S,1]
        P(),                     # lb_loss scalar
        P(),                     # rho_std scalar
        P(),                     # gate_sum scalar (sum gate, observational)
        P(),                     # active_n_mean scalar
        P('data', None, None),   # strong [B,S,1]
        P('data', None, None),   # soft_weight_mean_active [B,S,1]
        P(),                     # tau_abs_mean scalar
        P(),                     # dead_penalty scalar
        P(),                     # dead_count scalar
        P(),                     # int_max scalar
        P(),                     # den_cost_mean scalar
        P(),                     # selection_cost_mean scalar
        P(),                     # current_cost_mean scalar
        P(),                     # selection_residency_loss scalar (disabled selection-residency)
        P(),                     # edge_margin_stat scalar
        P('data', None, None),   # tau_direct [B,S,1]
        P('data', None, None),   # no_active_direct [B,S,1]
    )
    # ANALYSIS extras appended after slim.
    _analysis_extra_specs = (
        P('data', None, None),   # margin_band [B,S,1]
        P(),                     # margin_band_wide_frac scalar
        P(),                     # margin_band_mid_frac scalar
        P(),                     # rho_skew scalar
        P(),                     # active_per_token_std scalar
        P(),                     # gate_entropy scalar
        P(),                     # den_cost_out
        P(),                     # selection_cost_out
        P(),                     # current_cost_out
        P(),                     # rho_kurt scalar
        P(),                     # int_cap_frac scalar
    )
    _conc_out_specs = (
        P(),                     # gate_eff_n_mean scalar
        P(),                     # gate_eff_ratio_mean scalar
        P(),                     # top1_gate_frac_mean scalar
        P(),                     # top1_gate_frac_max scalar
    )
    _select_diag_specs = tuple(P() for _ in range(SELECT_DIAG_COUNT))
    _dead_exposure_diag_specs = tuple(
        P() for _ in range(DEAD_EXPOSURE_DIAG_COUNT))
    _sparsity_diag_specs = (
        P(),                     # gate sparsity diagnostics [metric]
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs + _conc_out_specs)
    _out_specs = _out_specs + _select_diag_specs
    _out_specs = _out_specs + _dead_exposure_diag_specs
    _out_specs = _out_specs + _sparsity_diag_specs
    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),    # x [B,S,D]
                       P('data', None, None),    # h [B,S,d_route]
                       P('model', None),          # op key [N_local,d_route]
                       P('data', None, None),    # raw_tau [B,S,1]
                       P('model', None),          # read [N_local, D]
                       P('model', None),          # write [N_local, D]
                       P(),                       # soft_gate_temperature scalar
                       P(),                       # soft_gate_t_final scalar
                       P(),                       # soft_gate_boundary_power scalar
                       P(),                       # soft_gate_boundary_power_final scalar
                       P()),                      # execution_prune_eps scalar
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw(x, h, op_key_local, raw_tau,
                       read_local, write_local,
                       soft_gate_temperature, soft_gate_t_final,
                       soft_gate_boundary_power,
                       soft_gate_boundary_power_final,
                       execution_prune_eps):
        N_local = op_key_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        op_key_bf = op_key_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1 = jnp.zeros((B, S, 1))
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_pos_inf = jnp.float32(1.0e30)

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_bf, start, cs, axis=0)

        def operator_relation(h_in, op_key):
            # Cosine between operator query and RW-derived operator key.
            q_unit = _forward_unit_direction(
                h_in.astype(jnp.float32)).astype(jnp.bfloat16)
            op_key_unit = _forward_unit_direction(
                op_key.astype(jnp.float32)).astype(jnp.bfloat16)
            rho = (q_unit @ op_key_unit.T).astype(jnp.float32)
            rho_exposure = (
                jax.lax.stop_gradient(q_unit) @ op_key_unit.T
            ).astype(jnp.float32)
            return rho, rho_exposure

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            # v4.1.6.4: stored read/write params stay raw, but SRW
            # execution uses their directions.
            rc_dir = _forward_unit_direction(rc_f)
            wc_dir = _forward_unit_direction(wc_f)
            return ec, rc_dir.astype(jnp.bfloat16), wc_dir.astype(jnp.bfloat16)

        # Boundary DirectTau. Fast train path does not scan rho
        # statistics for tau; rho distribution moments are analysis-only.
        N_total = N_local * _model_axis_size
        tau = _tau_from_param(raw_tau)
        tau_ref = jax.lax.stop_gradient(tau)
        if analysis:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, cube_sum, quad_sum = carry
                s = i * cs
                op_key = op_key_chunk(s)
                rho, _ = operator_relation(h_bf, op_key)
                s_sum = s_sum + rho.sum(axis=-1, keepdims=True)
                sq_sum = sq_sum + (rho ** 2).sum(axis=-1, keepdims=True)
                cube_sum = cube_sum + (rho ** 3).sum(axis=-1, keepdims=True)
                quad_sum = quad_sum + (rho ** 4).sum(axis=-1, keepdims=True)
                return (s_sum, sq_sum, cube_sum, quad_sum), None

            z_bs1 = jnp.zeros((B, S, 1))
            (local_sum, local_sq, local_cube, local_quad), _ = jax.lax.scan(
                stats_step, (z_bs1, z_bs1, z_bs1, z_bs1), jnp.arange(nc))
            global_sum = jax.lax.psum(local_sum, 'model')
            global_sq = jax.lax.psum(local_sq, 'model')
            rho_mean = global_sum / N_total
            rho_var = jnp.maximum(global_sq / N_total - rho_mean ** 2, 0.0)
            rho_std = jnp.sqrt(rho_var) + 1e-8
            global_cube = jax.lax.psum(local_cube, 'model')
            global_quad = jax.lax.psum(local_quad, 'model')
            cube_mean = global_cube / N_total
            central_third = cube_mean - 3.0 * rho_mean * (rho_std ** 2) - rho_mean ** 3
            rho_skew = jax.lax.stop_gradient((central_third / (rho_std ** 3 + 1e-8)).mean())
            quad_mean = global_quad / N_total
            central_fourth = (quad_mean - 4.0 * rho_mean * cube_mean
                              + 6.0 * (rho_mean ** 2) * (rho_std ** 2) + 3.0 * rho_mean ** 4)
            rho_kurt = jax.lax.stop_gradient((central_fourth / (rho_std ** 4 + 1e-8)).mean())
        else:
            rho_mean = jnp.zeros((B, S, 1), dtype=jnp.float32)
            rho_std = jnp.zeros((B, S, 1), dtype=jnp.float32)

        # Load-balance over rho distribution is disabled in the fast path for
        # v4.1.6.4 does not require a rho-statistics pass for regular train.
        rho_lb = jnp.float32(0.0)

        def edge_margin_stat_terms(rho):
            positive_selection_margin = jax.nn.relu(rho - tau_ref)
            return jnp.square(positive_selection_margin)

        def angular_compose_parts(rho):
            (selection_margin, admission, drive, execution_weight,
             active_mask) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            strong_mask = selection_margin > _angular_strong_margin
            return (
                selection_margin,
                admission,
                drive,
                execution_weight,
                active_mask,
                strong_mask,
            )

        def update_select_diag(carry, rho, selection_margin, positive_margin):
            (total_selected, total_selection_margin_sum,
             total_positive_margin_sum, total_positive_margin_max,
             total_rho_max, total_selection_margin_max) = carry
            selected = (selection_margin > 0.0).astype(jnp.float32)
            return (
                total_selected + selected.sum(axis=-1, keepdims=True),
                total_selection_margin_sum + selection_margin.sum(),
                total_positive_margin_sum + positive_margin.sum(),
                jnp.maximum(total_positive_margin_max,
                            positive_margin.max()),
                jnp.maximum(total_rho_max, rho.max()),
                jnp.maximum(total_selection_margin_max,
                            selection_margin.max(axis=-1, keepdims=True)),
            )

        select_diag_carry0 = (
            z1,
            jnp.float32(0.0),
            jnp.float32(0.0),
            diag_neg_inf,
            diag_neg_inf,
            jnp.full((B, S, 1), diag_neg_inf, dtype=jnp.float32),
        )
        current_eps = jnp.asarray(_current_eps_values, dtype=jnp.float32)
        projected_eps = jnp.asarray(_projected_eps_values, dtype=jnp.float32)
        margin_band_count = 3 if _compact_margin_bands else len(MARGIN_BAND_NAMES)
        sparsity_carry0 = (
            jnp.float32(0.0),
            jnp.zeros((len(_current_eps_values),), dtype=jnp.float32),
            jnp.zeros((len(_current_eps_values),), dtype=jnp.float32),
            jnp.zeros((len(_current_eps_values),), dtype=jnp.float32),
            jnp.float32(0.0),
            jnp.zeros((len(_projected_eps_values),), dtype=jnp.float32),
            jnp.zeros((len(_projected_eps_values),), dtype=jnp.float32),
            jnp.float32(0.0),
            jnp.zeros((margin_band_count,), dtype=jnp.float32),
        )

        def gate_sparsity_parts(selection_margin, admission, execution_weight):
            margin_sg = jax.lax.stop_gradient(selection_margin)
            admission_sg = jax.lax.stop_gradient(admission)
            execution_sg = jax.lax.stop_gradient(execution_weight)
            active_tau = margin_sg > 0.0

            admission_active = admission_sg[..., None] > current_eps
            admission_active_count = admission_active.astype(jnp.float32).sum(
                axis=(0, 1, 2))
            current_active = execution_sg[..., None] > current_eps
            current_active_count = current_active.astype(jnp.float32).sum(
                axis=(0, 1, 2))
            if _compute_sparsity_mass:
                current_mass = (
                    execution_sg[..., None] * current_active.astype(jnp.float32)
                ).sum(axis=(0, 1, 2))
                gate_mass = execution_sg.sum()
            else:
                current_mass = jnp.zeros_like(current_active_count)
                gate_mass = jnp.float32(0.0)

            projected_gate = _boundary_gate_from_margin(
                margin_sg, soft_gate_t_final, soft_gate_boundary_power_final)
            projected_active = projected_gate[..., None] > projected_eps
            projected_active_count = projected_active.astype(jnp.float32).sum(
                axis=(0, 1, 2))
            if _compute_sparsity_mass:
                projected_mass = (
                    projected_gate[..., None]
                    * projected_active.astype(jnp.float32)
                ).sum(axis=(0, 1, 2))
                projected_gate_mass = projected_gate.sum()
            else:
                projected_mass = jnp.zeros_like(projected_active_count)
                projected_gate_mass = jnp.float32(0.0)

            if _compact_margin_bands:
                margin_bands = jnp.stack((
                    active_tau.astype(jnp.float32).sum(),
                    ((margin_sg >= -0.03) & (margin_sg <= 0.0)).astype(jnp.float32).sum(),
                    (margin_sg < -0.10).astype(jnp.float32).sum(),
                )).astype(jnp.float32)
            else:
                margin_bands = jnp.stack((
                    active_tau.astype(jnp.float32).sum(),
                    ((margin_sg >= -0.01) & (margin_sg <= 0.0)).astype(jnp.float32).sum(),
                    ((margin_sg >= -0.03) & (margin_sg < -0.01)).astype(jnp.float32).sum(),
                    ((margin_sg >= -0.10) & (margin_sg < -0.03)).astype(jnp.float32).sum(),
                    (margin_sg < -0.10).astype(jnp.float32).sum(),
                )).astype(jnp.float32)
            return (
                active_tau.astype(jnp.float32).sum(),
                admission_active_count,
                current_active_count,
                current_mass,
                gate_mass,
                projected_active_count,
                projected_mass,
                projected_gate_mass,
                margin_bands,
            )

        def add_sparsity_carry(a, b):
            return tuple(x + y for x, y in zip(a, b))

        def finalize_sparsity_diag(carry):
            (active_tau_count, admission_active_count,
             current_active_count, current_mass, gate_mass,
             projected_active_count, projected_mass, projected_gate_mass,
             margin_bands) = carry
            active_tau_count = jax.lax.psum(active_tau_count, 'model')
            admission_active_count = jax.lax.psum(
                admission_active_count, 'model')
            current_active_count = jax.lax.psum(
                current_active_count, 'model')
            current_mass = jax.lax.psum(current_mass, 'model')
            gate_mass = jax.lax.psum(gate_mass, 'model')
            projected_active_count = jax.lax.psum(
                projected_active_count, 'model')
            projected_mass = jax.lax.psum(projected_mass, 'model')
            projected_gate_mass = jax.lax.psum(projected_gate_mass, 'model')
            margin_bands = jax.lax.psum(margin_bands, 'model')

            token_count = jnp.float32(B * S)
            element_count = token_count * jnp.float32(N_total)
            admission_frac = admission_active_count / element_count
            admission_count = admission_active_count / token_count
            current_frac = current_active_count / element_count
            current_count = current_active_count / token_count
            projected_frac = projected_active_count / element_count
            projected_count = projected_active_count / token_count
            out = jnp.zeros((GATE_SPARSITY_DIAG_COUNT,), dtype=jnp.float32)
            out = out.at[GATE_SPARSITY_DIAG_INDEX['active_tau_frac']].set(
                active_tau_count / element_count)
            out = out.at[GATE_SPARSITY_DIAG_INDEX['active_tau_count']].set(
                active_tau_count / token_count)
            for _i, _suffix in enumerate(_current_suffixes):
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX[
                        f'admission_active_eps_{_suffix}_frac']
                ].set(admission_frac[_i])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX[
                        f'admission_active_eps_{_suffix}_count']
                ].set(admission_count[_i])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX[f'active_eps_{_suffix}_frac']
                ].set(current_frac[_i])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX[f'active_eps_{_suffix}_count']
                ].set(current_count[_i])
                if _compute_sparsity_mass:
                    out = out.at[
                        GATE_SPARSITY_DIAG_INDEX[f'mass_eps_{_suffix}']
                    ].set(current_mass[_i] / jnp.maximum(gate_mass, 1.0e-8))
            for _i, _suffix in enumerate(_projected_suffixes):
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX[
                        f'projected_Tfinal_active_eps_{_suffix}_frac']
                ].set(projected_frac[_i])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX[
                        f'projected_Tfinal_active_eps_{_suffix}_count']
                ].set(projected_count[_i])
                if _compute_sparsity_mass:
                    out = out.at[
                        GATE_SPARSITY_DIAG_INDEX[
                            f'projected_Tfinal_mass_eps_{_suffix}']
                    ].set(projected_mass[_i] / jnp.maximum(
                        projected_gate_mass, 1.0e-8))
            margin_frac = margin_bands / element_count
            if _compact_margin_bands:
                out = out.at[GATE_SPARSITY_DIAG_INDEX['margin_band_gt_0']].set(
                    margin_frac[0])
                out = out.at[GATE_SPARSITY_DIAG_INDEX['margin_band_lt_m0_10']].set(
                    margin_frac[2])
                out = out.at[GATE_SPARSITY_DIAG_INDEX['margin_band_pos']].set(
                    margin_frac[0])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX['margin_band_near_m0_03_0']
                ].set(margin_frac[1])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX['margin_band_far_lt_m0_10']
                ].set(margin_frac[2])
            else:
                for _i, _name in enumerate(MARGIN_BAND_NAMES):
                    out = out.at[
                        GATE_SPARSITY_DIAG_INDEX[f'margin_band_{_name}']
                    ].set(margin_frac[_i])
                out = out.at[GATE_SPARSITY_DIAG_INDEX['margin_band_pos']].set(
                    margin_frac[0])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX['margin_band_near_m0_03_0']
                ].set(margin_frac[1] + margin_frac[2])
                out = out.at[
                    GATE_SPARSITY_DIAG_INDEX['margin_band_far_lt_m0_10']
                ].set(margin_frac[4])
            return jax.lax.stop_gradient(out.astype(jnp.float32))

        def soft_gate_exposure_parts(gate_unpruned):
            # v4164 admission exposure diagnostic only.  Do not use the hard
            # score > tau boundary as a dead definition: with high T,
            # score < tau can still produce meaningful boundary gate mass.
            # A unit is considered soft-dead only if its actual admission
            # mass is essentially zero across the batch/tokens.
            if not analysis:
                return (
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                )
            local_soft_exposure = jax.lax.stop_gradient(
                gate_unpruned).max(axis=(0, 1))  # [cs]
            soft_exposure = jax.lax.all_gather(
                local_soft_exposure, 'data', axis=0).max(axis=0)
            soft_dead_1e6 = soft_exposure <= jnp.float32(1.0e-6)
            soft_dead_1e5 = soft_exposure <= jnp.float32(1.0e-5)
            soft_dead_1e4 = soft_exposure <= jnp.float32(1.0e-4)
            return (
                jnp.float32(0.0),
                soft_dead_1e6.astype(jnp.float32).sum(),
                soft_exposure.sum(),
                soft_exposure.min(),
                soft_exposure.max(),
                soft_dead_1e5.astype(jnp.float32).sum(),
                soft_dead_1e4.astype(jnp.float32).sum(),
            )

        if analysis:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_margin_band, total_den_cost,
                 total_selection_cost, total_current_cost,
                 total_margin_band_wide, total_margin_band_mid, total_g_log_g,
                 total_dead_penalty, total_dead_count,
                 total_exposure_sum, total_exposure_min,
                 total_exposure_max, total_weak_exposure_count,
                 total_int_max, total_int_cap_count, total_selection_residency_sum,
                 total_selection_residency_count,
                 total_edge_margin_stat,
                 sparsity_carry, select_diag_carry) = carry
                s = i * cs
                op_key, rc, wc = op_key_rw_chunk(s)
                rho, rho_exposure = operator_relation(h_bf, op_key)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = drive.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = drive.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_margin_band = jnp.zeros((B, S, 1), dtype=jnp.float32)
                chunk_margin_band_wide = jnp.zeros((B, S, 1), dtype=jnp.float32)
                chunk_margin_band_mid = jnp.zeros((B, S, 1), dtype=jnp.float32)
                g_safe = execution_weight + 1e-8
                chunk_g_log_g = (execution_weight * jnp.log(g_safe)).sum(
                    axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission))
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, execution_weight.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_margin_band + chunk_margin_band,
                        total_den_cost + chunk_den_cost,
                        total_selection_cost,
                        total_current_cost + chunk_current_cost,
                        total_margin_band_wide + chunk_margin_band_wide,
                        total_margin_band_mid + chunk_margin_band_mid,
                        total_g_log_g + chunk_g_log_g,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        total_exposure_sum + chunk_exposure_sum,
                        jnp.minimum(total_exposure_min, chunk_exposure_min),
                        jnp.maximum(total_exposure_max, chunk_exposure_max),
                        total_weak_exposure_count + chunk_weak_exposure_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_int_cap_count + chunk_int_cap_count,
                        total_selection_residency_sum + chunk_selection_residency_sum,
                        total_selection_residency_count + chunk_soft_dead_1e4_count,
                        total_edge_margin_stat + chunk_edge_margin_stat,
                        add_sparsity_carry(sparsity_carry, chunk_sparsity),
                        select_diag_carry), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_margin_band, total_den_cost, total_selection_cost,
             total_current_cost, total_margin_band_wide, total_margin_band_mid,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_exposure_sum, total_exposure_min,
             total_exposure_max, total_weak_exposure_count,
             total_int_max, total_int_cap_count, total_selection_residency_sum,
             total_selection_residency_count,
             total_edge_margin_stat,
             sparsity_carry,
             select_diag_carry), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), diag_pos_inf, diag_neg_inf,
                 jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0),
                 sparsity_carry0,
                 select_diag_carry0),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_den_cost,
                 total_selection_cost, total_current_cost,
                 total_dead_penalty, total_dead_count,
                 total_exposure_sum, total_exposure_min,
                 total_exposure_max, total_weak_exposure_count,
                 total_int_max, total_int_cap_count, total_selection_residency_sum,
                 total_selection_residency_count,
                 total_edge_margin_stat,
                 sparsity_carry, select_diag_carry) = carry
                s = i * cs
                op_key, rc, wc = op_key_rw_chunk(s)
                rho, rho_exposure = operator_relation(h_bf, op_key)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = drive.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = drive.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission))
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, execution_weight.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_den_cost + chunk_den_cost,
                        total_selection_cost,
                        total_current_cost + chunk_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        total_exposure_sum + chunk_exposure_sum,
                        jnp.minimum(total_exposure_min, chunk_exposure_min),
                        jnp.maximum(total_exposure_max, chunk_exposure_max),
                        total_weak_exposure_count + chunk_weak_exposure_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_int_cap_count + chunk_int_cap_count,
                        total_selection_residency_sum + chunk_selection_residency_sum,
                        total_selection_residency_count + chunk_soft_dead_1e4_count,
                        total_edge_margin_stat + chunk_edge_margin_stat,
                        add_sparsity_carry(sparsity_carry, chunk_sparsity),
                        select_diag_carry), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_selection_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_exposure_sum, total_exposure_min,
             total_exposure_max, total_weak_exposure_count,
             total_int_max, total_int_cap_count, total_selection_residency_sum,
             total_selection_residency_count,
             total_edge_margin_stat,
             sparsity_carry,
             select_diag_carry), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                 diag_pos_inf, diag_neg_inf, jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0),
                 sparsity_carry0,
                 select_diag_carry0),
                jnp.arange(nc))


        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')  # sum(execution_weight)
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        # Denominator intentionally uses admission only, not execution_weight.
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        global_selection_cost = jax.lax.psum(total_selection_cost, 'model')
        global_current_cost = jax.lax.psum(total_current_cost, 'model')
        selection_residency_loss = jnp.float32(0.0)
        edge_margin_stat = (
            jax.lax.psum(total_edge_margin_stat, 'model')
            / jnp.float32(B * S * N_total))
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        # v4164: sublinear divisive inhibition over admission mass.
        # Forward:
        #   admission_den_forward = max(sum(admission), 1) ** admission_den_power
        # Backward:
        #   only den_grad_scale fraction of the denominator gradient is live.
        # This keeps the inhibitory/gain-control term biologically plausible
        # while avoiding the early full-denominator collapse observed with
        # admission_den_grad_scale=1 and admission_den_power=1.
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = admission_den_sg + _admission_den_grad_scale * (admission_den_forward - admission_den_sg)
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_active = jax.lax.psum(total_active, 'model')
        (total_selected, total_selection_margin_sum,
         total_positive_margin_sum, total_positive_margin_max,
         total_rho_max, total_selection_margin_max) = select_diag_carry
        global_selected = jax.lax.psum(total_selected, 'model')
        global_selection_margin_sum = jax.lax.psum(
            total_selection_margin_sum, 'model')
        global_positive_margin_sum = jax.lax.psum(
            total_positive_margin_sum, 'model')
        global_positive_margin_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_positive_margin_max), 'model')
        global_rho_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_rho_max), 'model')
        global_selection_margin_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_selection_margin_max), 'model')
        no_active_direct = jax.lax.stop_gradient(
            (global_selection_margin_max <= 0.0).astype(jnp.float32))
        tau_direct = _block_tau_up_when_no_active(
            tau, no_active_direct.astype(jnp.bool_))
        # Measurement path: detached copies for diagnostics / feedback refs.
        # The forward denominator is already detached above; numerator paths
        # through execution_weight remain live for SRW output gradients.
        global_weighted_cost_m = jax.lax.stop_gradient(global_weighted_cost)
        global_gate_sq_m = jax.lax.stop_gradient(global_gate_sq)
        global_den_cost_m = jax.lax.stop_gradient(global_den_cost)
        global_selection_cost_m = jax.lax.stop_gradient(global_selection_cost)
        global_current_cost_m = jax.lax.stop_gradient(global_current_cost)
        global_active_m = jax.lax.stop_gradient(global_active)
        global_strong_m = jax.lax.stop_gradient(
            jax.lax.psum(total_strong, 'model'))
        global_gate_max_m = jax.lax.stop_gradient(global_gate_max)
        global_selected_m = jax.lax.stop_gradient(global_selected)
        active_frac = global_active_m / N_total
        strong_frac = global_strong_m / N_total
        positive_margin_mean_active = (
            jax.lax.stop_gradient(global_positive_margin_sum)
            / (global_active_m + 1e-8))

        rho_std_out = jax.lax.stop_gradient(rho_std.mean())
        es_out = global_weighted_cost_m.mean()          # sum(gate), observational
        active_n_mean = global_active_m.mean()
        gate_eff_n = ((global_weighted_cost_m ** 2)
                      / (global_gate_sq_m + 1e-8))
        gate_eff_ratio = gate_eff_n / jnp.maximum(global_active_m, 1.0)
        top1_gate_frac = global_gate_max_m / jnp.maximum(
            global_weighted_cost_m, 1e-8)
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        global_exposure_sum = jax.lax.stop_gradient(
            jax.lax.psum(total_exposure_sum, 'model'))
        global_exposure_min = jax.lax.pmin(
            jax.lax.stop_gradient(total_exposure_min), 'model')
        global_exposure_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_exposure_max), 'model')
        global_weak_exposure_count = jax.lax.stop_gradient(
            jax.lax.psum(total_weak_exposure_count, 'model'))
        # pmax has no VJP; wrap the input in stop_gradient.
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost_m.mean()
        selection_cost_mean = global_selection_cost_m.mean()
        current_cost_mean = global_current_cost_m.mean()

        rho_count = jnp.float32(B * S * N_total)
        token_count = jnp.float32(B * S)
        raw_tau_sg = jax.lax.stop_gradient(raw_tau)
        tau_sg = jax.lax.stop_gradient(tau)
        select_diag_out = (
            jax.lax.stop_gradient(rho_mean.mean()),
            jax.lax.stop_gradient(rho_std.mean()),
            jax.lax.stop_gradient(global_rho_max),
            tau_sg.mean(),
            tau_sg.min(),
            tau_sg.max(),
            raw_tau_sg.mean(),
            raw_tau_sg.min(),
            raw_tau_sg.max(),
            jax.lax.stop_gradient(global_selection_margin_sum / rho_count),
            jax.lax.stop_gradient(global_positive_margin_sum / rho_count),
            jax.lax.stop_gradient(global_positive_margin_max),
            jax.lax.stop_gradient(global_selected_m.mean() / N_total),
            jax.lax.stop_gradient(no_active_direct.mean()),
        )
        dead_exposure_diag_out = (
            global_exposure_sum / jnp.float32(N_total),
            global_exposure_min,
            global_exposure_max,
            dead_count_out / jnp.float32(N_total),
            global_weak_exposure_count / jnp.float32(N_total),
            jax.lax.stop_gradient(
                jax.lax.psum(total_selection_residency_count, 'model')
                / jnp.float32(N_total)),
        )
        sparsity_diag_out = (
            finalize_sparsity_diag(sparsity_carry)
            if _sparsity_diag_enabled
            else jnp.zeros((GATE_SPARSITY_DIAG_COUNT,), dtype=jnp.float32))

        slim_out = (out.astype(jnp.float32), active_frac, global_gate_max, rho_lb,
                    rho_std_out, es_out, active_n_mean, strong_frac,
                    positive_margin_mean_active,
                    tau_abs_mean, dead_penalty_out, dead_count_out, int_max_out,
                    den_cost_mean, selection_cost_mean, current_cost_mean,
                    selection_residency_loss, edge_margin_stat,
                    tau_direct.astype(jnp.float32),
                    no_active_direct)
        conc_out = (gate_eff_n.mean(), gate_eff_ratio.mean(),
                    top1_gate_frac.mean(), top1_gate_frac.max())
        if not analysis:
            return (slim_out + conc_out + select_diag_out
                    + dead_exposure_diag_out + (sparsity_diag_out,)
                   )

        # --- Analysis-only extras ---
        margin_band_frac = jax.lax.psum(total_margin_band, 'model') / N_total
        margin_band_frac = jax.lax.stop_gradient(margin_band_frac)
        # Safety floor: active can collapse to 0 at init; clamp to 1.0.
        _active_denom = jnp.maximum(global_active_m, 1.0)
        margin_band_wide_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_margin_band_wide, 'model') / _active_denom).mean())
        margin_band_mid_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_margin_band_mid, 'model') / _active_denom).mean())
        active_per_token_std = global_active_m.std()
        global_g_log_g = jax.lax.stop_gradient(
            jax.lax.psum(total_g_log_g, 'model'))
        gate_sum_eps = jnp.maximum(global_weighted_cost_m, 1e-6)
        safe_glogg = jnp.where(
            global_weighted_cost_m > 1e-6, global_g_log_g, 0.0)
        entropy_per_token = -safe_glogg / gate_sum_eps + jnp.log(gate_sum_eps)
        entropy_per_token = jnp.where(
            jnp.isfinite(entropy_per_token), entropy_per_token, 0.0)
        gate_entropy = entropy_per_token.mean()
        den_cost_out = global_den_cost_m.mean()
        selection_cost_out = global_selection_cost_m.mean()
        current_cost_out = global_current_cost_m.mean()
        int_cap_frac_out = jax.lax.stop_gradient(
            jax.lax.psum(total_int_cap_count, 'model')
            / jnp.float32(B * S * N_total))
        return (slim_out + conc_out
                + (margin_band_frac, margin_band_wide_frac,
                   margin_band_mid_frac, rho_skew, active_per_token_std,
                   gate_entropy, den_cost_out, selection_cost_out,
                   current_cost_out, rho_kurt, int_cap_frac_out)
                + select_diag_out + dead_exposure_diag_out
                + (sparsity_diag_out,)
               )

    return fused_gate_srw


def make_sharded_srw_paired(mesh, max_chunk_size=2048,
                            analysis=False,
                            dead_exposure_target=0.1,
                            soft_gate_effective_active_eps=1.0e-6,
                            admission_den_power=1.0,
                            admission_den_grad_scale=1.0,
                            operator_pages_enabled=False,
                            operator_page_size=128,
                            operator_page_capacity=8,
                            operator_page_microgroup_sequences=2,
                            operator_page_score_mode='maxmean',
                            operator_page_fallback_pages=0,
                            operator_page_random_pages=0):
    """Fused Q+K shard_map: two routes sharing same pool in one shard_map call.

    h is [B,S,2,d_route] (h_Q, h_K stacked on axis=2).
    raw_tau is [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].

    Boundary DirectTau admission with drive composition:
        admission = exp(-((max(tau - rho, 0) / B) ** p))
        drive = softplus((rho - tau) / B) / softplus((1 - tau) / B)
        execution_weight = admission * drive
        admission_den = max(sum(admission), 1.0) ** admission_den_power
    analysis: see make_sharded_srw docstring.
    """
    if operator_pages_enabled:
        return make_sharded_srw_paired_page(
            mesh, max_chunk_size=max_chunk_size, analysis=analysis,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            operator_page_size=operator_page_size,
            operator_page_capacity=operator_page_capacity,
            operator_page_microgroup_sequences=operator_page_microgroup_sequences,
            operator_page_score_mode=operator_page_score_mode,
            operator_page_fallback_pages=operator_page_fallback_pages,
            operator_page_random_pages=operator_page_random_pages)
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']
    _dead_exposure_target = jnp.float32(dead_exposure_target)
    _soft_gate_effective_active_eps = jnp.float32(soft_gate_effective_active_eps)
    _sparsity_diag_enabled = bool(analysis)
    _current_suffixes = GATE_EPS_NAME_SUFFIXES
    _projected_suffixes = GATE_PROJECTED_EPS_NAME_SUFFIXES
    _current_eps_values = _gate_eps_values_from_suffixes(_current_suffixes)
    _projected_eps_values = _gate_eps_values_from_suffixes(_projected_suffixes)
    _compute_sparsity_mass = True
    _compact_margin_bands = False
    _angular_strong_margin = jnp.float32(0.05)
    # v4164 divisive normalization:
    #   admission_den_forward = max(sum(admission), 1) ** admission_den_power
    #   backward gradient through admission_den is scaled by admission_den_grad_scale.
    # admission_den_power=0.5 is RMS/energy-like inhibition;
    # admission_den_grad_scale=0 detaches the inhibitory denominator gradient,
    # 1 restores the full live admission_den path.
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))

    _slim_out_specs = (
        P('data', None, None, None),  # out [B,S,2,D]
        P('data', None, None),        # active [B,S,1]
        P('data', None, None),        # gate_max [B,S,1]
        P(),                          # lb_loss scalar
        P(),                          # rho_std scalar
        P(),                          # gate_sum scalar
        P(),                          # active_n_mean scalar
        P('data', None, None),        # strong [B,S,1]
        P('data', None, None),        # soft_weight_mean_active [B,S,1]
        P(),                          # tau_abs_mean scalar
        P(),                          # dead_penalty scalar
        P(),                          # dead_count scalar
        P(),                          # int_max scalar
        P(),                          # den_cost_mean scalar
        P(),                          # selection_cost_mean scalar
        P(),                          # current_cost_mean scalar
        P(),                          # selection_residency_loss scalar (disabled selection-residency)
        P(),                          # edge_margin_stat scalar
        P('data', None, None, None),  # tau_direct [B,S,2,1]
        P('data', None, None, None),  # no_active_direct [B,S,2,1]
    )
    _analysis_extra_specs = (
        P('data', None, None),        # margin_band [B,S,1]
        P(),                          # margin_band_wide_frac scalar
        P(),                          # margin_band_mid_frac scalar
        P(),                          # rho_skew scalar
        P(),                          # active_per_token_std scalar
        P(),                          # gate_entropy scalar
        P(),                          # den_cost_out scalar
        P(),                          # selection_cost_out scalar
        P(),                          # current_cost_out scalar
        P(),                          # rho_kurt scalar
        P(),                          # int_cap_frac scalar
    )
    _conc_out_specs = (
        P(),                          # gate_eff_n_mean scalar
        P(),                          # gate_eff_ratio_mean scalar
        P(),                          # top1_gate_frac_mean scalar
        P(),                          # top1_gate_frac_max scalar
    )
    _route_split_out_specs = (
        P(),                          # q_active_frac scalar
        P(),                          # k_active_frac scalar
        P(),                          # q_strong_frac scalar
        P(),                          # k_strong_frac scalar
        P(),                          # q_active_n_mean scalar
        P(),                          # k_active_n_mean scalar
    )
    _select_diag_specs = tuple(P() for _ in range(SELECT_DIAG_COUNT))
    _dead_exposure_diag_specs = tuple(
        P() for _ in range(DEAD_EXPOSURE_DIAG_COUNT))
    _sparsity_diag_specs = (
        P(),                          # gate sparsity diagnostics [2, metric]
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _route_split_out_specs
                  + _analysis_extra_specs
                  if analysis
                  else _slim_out_specs + _conc_out_specs + _route_split_out_specs)
    _out_specs = _out_specs + _select_diag_specs
    _out_specs = _out_specs + _dead_exposure_diag_specs
    _out_specs = _out_specs + _sparsity_diag_specs
    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),        # x [B,S,D]
                       P('data', None, None, None),  # h [B,S,2,d_route]
                       P('model', None),              # op key [N_local,d_route]
                       P('data', None, None, None),  # raw_tau [B,S,2,1]
                       P('model', None),              # read [N_local, D]
                       P('model', None),              # write [N_local, D]
                       P(),                           # soft_gate_temperature scalar
                       P(),                           # soft_gate_t_final scalar
                       P(),                           # soft_gate_boundary_power scalar
                       P(),                           # soft_gate_boundary_power_final scalar
                       P()),                          # execution_prune_eps scalar
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw_paired(x, h, op_key_local, raw_tau,
                              read_local, write_local,
                              soft_gate_temperature, soft_gate_t_final,
                              soft_gate_boundary_power,
                              soft_gate_boundary_power_final,
                              execution_prune_eps):
        N_local = op_key_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        # h: [B,S,2,d_route], raw_tau: [B,S,2,1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        op_key_bf = op_key_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_pos_inf = jnp.float32(1.0e30)

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_bf, start, cs, axis=0)

        def operator_relation(h_in, op_key):
            # Cosine between operator query and RW-derived operator key.
            q_unit = _forward_unit_direction(
                h_in.astype(jnp.float32)).astype(jnp.bfloat16)
            op_key_unit = _forward_unit_direction(
                op_key.astype(jnp.float32)).astype(jnp.bfloat16)
            rho = jnp.einsum(
                'bsrd,nd->bsrn', q_unit, op_key_unit).astype(jnp.float32)
            rho_exposure = jnp.einsum(
                'bsrd,nd->bsrn',
                jax.lax.stop_gradient(q_unit),
                op_key_unit).astype(jnp.float32)
            return rho, rho_exposure

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            # v4.1.6.4: stored read/write params stay raw, but SRW
            # execution uses their directions.
            rc_dir = _forward_unit_direction(rc_f)
            wc_dir = _forward_unit_direction(wc_f)
            return ec, rc_dir.astype(jnp.bfloat16), wc_dir.astype(jnp.bfloat16)

        # Boundary DirectTau. Fast train path does not scan rho
        # statistics for tau; rho distribution moments are analysis-only.
        N_total = N_local * _model_axis_size
        tau = _tau_from_param(raw_tau)
        tau_ref = jax.lax.stop_gradient(tau)
        if analysis:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, cube_sum, quad_sum = carry
                s = i * cs
                op_key = op_key_chunk(s)
                rho, _ = operator_relation(h_bf, op_key)
                s_sum = s_sum + rho.sum(axis=-1, keepdims=True)
                sq_sum = sq_sum + (rho ** 2).sum(axis=-1, keepdims=True)
                cube_sum = cube_sum + (rho ** 3).sum(axis=-1, keepdims=True)
                quad_sum = quad_sum + (rho ** 4).sum(axis=-1, keepdims=True)
                return (s_sum, sq_sum, cube_sum, quad_sum), None

            z_bsr1 = jnp.zeros((B, S, 2, 1))
            (local_sum, local_sq, local_cube, local_quad), _ = jax.lax.scan(
                stats_step, (z_bsr1, z_bsr1, z_bsr1, z_bsr1), jnp.arange(nc))
            global_sum = jax.lax.psum(local_sum, 'model')
            global_sq = jax.lax.psum(local_sq, 'model')
            rho_mean = global_sum / N_total
            rho_var = jnp.maximum(global_sq / N_total - rho_mean ** 2, 0.0)
            rho_std = jnp.sqrt(rho_var) + 1e-8
            global_cube = jax.lax.psum(local_cube, 'model')
            global_quad = jax.lax.psum(local_quad, 'model')
            cube_mean = global_cube / N_total
            central_third = cube_mean - 3.0 * rho_mean * (rho_std ** 2) - rho_mean ** 3
            rho_skew = jax.lax.stop_gradient((central_third / (rho_std ** 3 + 1e-8)).mean())
            quad_mean = global_quad / N_total
            central_fourth = (quad_mean - 4.0 * rho_mean * cube_mean
                              + 6.0 * (rho_mean ** 2) * (rho_std ** 2) + 3.0 * rho_mean ** 4)
            rho_kurt = jax.lax.stop_gradient((central_fourth / (rho_std ** 4 + 1e-8)).mean())
        else:
            rho_mean = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
            rho_std = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)

        # Load-balance over rho distribution is disabled in the fast path for
        # v4.1.6.4 does not require a rho-statistics pass for regular train.
        rho_lb = jnp.float32(0.0)

        def edge_margin_stat_terms(rho):
            positive_selection_margin = jax.nn.relu(rho - tau_ref)
            return jnp.square(positive_selection_margin)

        def angular_compose_parts(rho):
            (selection_margin, admission, drive, execution_weight,
             active_mask) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            strong_mask = selection_margin > _angular_strong_margin
            return (
                selection_margin,
                admission,
                drive,
                execution_weight,
                active_mask,
                strong_mask,
            )

        def update_select_diag(carry, rho, selection_margin, positive_margin):
            (total_selected, total_selection_margin_sum,
             total_positive_margin_sum, total_positive_margin_max,
             total_rho_max, total_selection_margin_max) = carry
            selected = (selection_margin > 0.0).astype(jnp.float32)
            return (
                total_selected + selected.sum(axis=-1, keepdims=True),
                total_selection_margin_sum + selection_margin.sum(),
                total_positive_margin_sum + positive_margin.sum(),
                jnp.maximum(total_positive_margin_max,
                            positive_margin.max()),
                jnp.maximum(total_rho_max, rho.max()),
                jnp.maximum(total_selection_margin_max,
                            selection_margin.max(axis=-1, keepdims=True)),
            )

        select_diag_carry0 = (
            z1_r,
            jnp.float32(0.0),
            jnp.float32(0.0),
            diag_neg_inf,
            diag_neg_inf,
            jnp.full((B, S, 2, 1), diag_neg_inf, dtype=jnp.float32),
        )
        current_eps = jnp.asarray(_current_eps_values, dtype=jnp.float32)
        projected_eps = jnp.asarray(_projected_eps_values, dtype=jnp.float32)
        margin_band_count = 3 if _compact_margin_bands else len(MARGIN_BAND_NAMES)
        sparsity_carry0 = (
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2, len(_current_eps_values)), dtype=jnp.float32),
            jnp.zeros((2, len(_current_eps_values)), dtype=jnp.float32),
            jnp.zeros((2, len(_current_eps_values)), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2, len(_projected_eps_values)), dtype=jnp.float32),
            jnp.zeros((2, len(_projected_eps_values)), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((2, margin_band_count), dtype=jnp.float32),
        )

        def gate_sparsity_parts(selection_margin, admission, execution_weight):
            margin_sg = jax.lax.stop_gradient(selection_margin)
            admission_sg = jax.lax.stop_gradient(admission)
            execution_sg = jax.lax.stop_gradient(execution_weight)
            active_tau = margin_sg > 0.0

            admission_active = admission_sg[..., None] > current_eps
            admission_active_count = admission_active.astype(jnp.float32).sum(
                axis=(0, 1, 3))
            current_active = execution_sg[..., None] > current_eps
            current_active_count = current_active.astype(jnp.float32).sum(
                axis=(0, 1, 3))
            if _compute_sparsity_mass:
                current_mass = (
                    execution_sg[..., None] * current_active.astype(jnp.float32)
                ).sum(axis=(0, 1, 3))
                gate_mass = execution_sg.sum(axis=(0, 1, 3))
            else:
                current_mass = jnp.zeros_like(current_active_count)
                gate_mass = jnp.zeros((2,), dtype=jnp.float32)

            projected_gate = _boundary_gate_from_margin(
                margin_sg, soft_gate_t_final, soft_gate_boundary_power_final)
            projected_active = projected_gate[..., None] > projected_eps
            projected_active_count = projected_active.astype(jnp.float32).sum(
                axis=(0, 1, 3))
            if _compute_sparsity_mass:
                projected_mass = (
                    projected_gate[..., None]
                    * projected_active.astype(jnp.float32)
                ).sum(axis=(0, 1, 3))
                projected_gate_mass = projected_gate.sum(axis=(0, 1, 3))
            else:
                projected_mass = jnp.zeros_like(projected_active_count)
                projected_gate_mass = jnp.zeros((2,), dtype=jnp.float32)

            if _compact_margin_bands:
                margin_bands = jnp.stack((
                    active_tau.astype(jnp.float32).sum(axis=(0, 1, 3)),
                    ((margin_sg >= -0.03) & (margin_sg <= 0.0)).astype(jnp.float32).sum(axis=(0, 1, 3)),
                    (margin_sg < -0.10).astype(jnp.float32).sum(axis=(0, 1, 3)),
                ), axis=1).astype(jnp.float32)
            else:
                margin_bands = jnp.stack((
                    active_tau.astype(jnp.float32).sum(axis=(0, 1, 3)),
                    ((margin_sg >= -0.01) & (margin_sg <= 0.0)).astype(jnp.float32).sum(axis=(0, 1, 3)),
                    ((margin_sg >= -0.03) & (margin_sg < -0.01)).astype(jnp.float32).sum(axis=(0, 1, 3)),
                    ((margin_sg >= -0.10) & (margin_sg < -0.03)).astype(jnp.float32).sum(axis=(0, 1, 3)),
                    (margin_sg < -0.10).astype(jnp.float32).sum(axis=(0, 1, 3)),
                ), axis=1).astype(jnp.float32)
            return (
                active_tau.astype(jnp.float32).sum(axis=(0, 1, 3)),
                admission_active_count,
                current_active_count,
                current_mass,
                gate_mass,
                projected_active_count,
                projected_mass,
                projected_gate_mass,
                margin_bands,
            )

        def add_sparsity_carry(a, b):
            return tuple(x + y for x, y in zip(a, b))

        def finalize_sparsity_diag(carry):
            (active_tau_count, admission_active_count,
             current_active_count, current_mass, gate_mass,
             projected_active_count, projected_mass, projected_gate_mass,
             margin_bands) = carry
            active_tau_count = jax.lax.psum(active_tau_count, 'model')
            admission_active_count = jax.lax.psum(
                admission_active_count, 'model')
            current_active_count = jax.lax.psum(
                current_active_count, 'model')
            current_mass = jax.lax.psum(current_mass, 'model')
            gate_mass = jax.lax.psum(gate_mass, 'model')
            projected_active_count = jax.lax.psum(
                projected_active_count, 'model')
            projected_mass = jax.lax.psum(projected_mass, 'model')
            projected_gate_mass = jax.lax.psum(projected_gate_mass, 'model')
            margin_bands = jax.lax.psum(margin_bands, 'model')

            token_count = jnp.float32(B * S)
            element_count = token_count * jnp.float32(N_total)
            admission_frac = admission_active_count / element_count
            admission_count = admission_active_count / token_count
            current_frac = current_active_count / element_count
            current_count = current_active_count / token_count
            projected_frac = projected_active_count / element_count
            projected_count = projected_active_count / token_count
            out = jnp.zeros((2, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32)
            out = out.at[:, GATE_SPARSITY_DIAG_INDEX['active_tau_frac']].set(
                active_tau_count / element_count)
            out = out.at[:, GATE_SPARSITY_DIAG_INDEX['active_tau_count']].set(
                active_tau_count / token_count)
            for _i, _suffix in enumerate(_current_suffixes):
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX[
                        f'admission_active_eps_{_suffix}_frac']
                ].set(admission_frac[:, _i])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX[
                        f'admission_active_eps_{_suffix}_count']
                ].set(admission_count[:, _i])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX[f'active_eps_{_suffix}_frac']
                ].set(current_frac[:, _i])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX[f'active_eps_{_suffix}_count']
                ].set(current_count[:, _i])
                if _compute_sparsity_mass:
                    out = out.at[
                        :, GATE_SPARSITY_DIAG_INDEX[f'mass_eps_{_suffix}']
                    ].set(current_mass[:, _i] / jnp.maximum(gate_mass, 1.0e-8))
            for _i, _suffix in enumerate(_projected_suffixes):
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX[
                        f'projected_Tfinal_active_eps_{_suffix}_frac']
                ].set(projected_frac[:, _i])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX[
                        f'projected_Tfinal_active_eps_{_suffix}_count']
                ].set(projected_count[:, _i])
                if _compute_sparsity_mass:
                    out = out.at[
                        :, GATE_SPARSITY_DIAG_INDEX[
                            f'projected_Tfinal_mass_eps_{_suffix}']
                    ].set(projected_mass[:, _i] / jnp.maximum(
                        projected_gate_mass, 1.0e-8))
            margin_frac = margin_bands / element_count
            if _compact_margin_bands:
                out = out.at[:, GATE_SPARSITY_DIAG_INDEX['margin_band_gt_0']].set(
                    margin_frac[:, 0])
                out = out.at[:, GATE_SPARSITY_DIAG_INDEX['margin_band_lt_m0_10']].set(
                    margin_frac[:, 2])
                out = out.at[:, GATE_SPARSITY_DIAG_INDEX['margin_band_pos']].set(
                    margin_frac[:, 0])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX['margin_band_near_m0_03_0']
                ].set(margin_frac[:, 1])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX['margin_band_far_lt_m0_10']
                ].set(margin_frac[:, 2])
            else:
                for _i, _name in enumerate(MARGIN_BAND_NAMES):
                    out = out.at[
                        :, GATE_SPARSITY_DIAG_INDEX[f'margin_band_{_name}']
                    ].set(margin_frac[:, _i])
                out = out.at[:, GATE_SPARSITY_DIAG_INDEX['margin_band_pos']].set(
                    margin_frac[:, 0])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX['margin_band_near_m0_03_0']
                ].set(margin_frac[:, 1] + margin_frac[:, 2])
                out = out.at[
                    :, GATE_SPARSITY_DIAG_INDEX['margin_band_far_lt_m0_10']
                ].set(margin_frac[:, 4])
            return jax.lax.stop_gradient(out.astype(jnp.float32))

        def soft_gate_exposure_parts(gate_unpruned):
            # v4164 admission exposure diagnostic only.  Do not use the hard
            # score > tau boundary as a dead definition: with high T,
            # score < tau can still produce meaningful boundary gate mass.
            # A unit is considered soft-dead only if its actual admission
            # mass is essentially zero across the batch/tokens/routes.
            if not analysis:
                return (
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                )
            local_soft_exposure = jax.lax.stop_gradient(
                gate_unpruned).max(axis=(0, 1, 2))  # [cs]
            soft_exposure = jax.lax.all_gather(
                local_soft_exposure, 'data', axis=0).max(axis=0)
            soft_dead_1e6 = soft_exposure <= jnp.float32(1.0e-6)
            soft_dead_1e5 = soft_exposure <= jnp.float32(1.0e-5)
            soft_dead_1e4 = soft_exposure <= jnp.float32(1.0e-4)
            return (
                jnp.float32(0.0),
                soft_dead_1e6.astype(jnp.float32).sum(),
                soft_exposure.sum(),
                soft_exposure.min(),
                soft_exposure.max(),
                soft_dead_1e5.astype(jnp.float32).sum(),
                soft_dead_1e4.astype(jnp.float32).sum(),
            )

        if analysis:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_margin_band, total_den_cost,
                 total_selection_cost, total_current_cost,
                 total_margin_band_wide, total_margin_band_mid, total_g_log_g,
                 total_dead_penalty, total_dead_count,
                 total_exposure_sum, total_exposure_min,
                 total_exposure_max, total_weak_exposure_count,
                 total_int_max, total_int_cap_count, total_selection_residency_sum,
                 total_selection_residency_count,
                 total_edge_margin_stat,
                 sparsity_carry, select_diag_carry) = carry
                s = i * cs
                op_key, rc, wc = op_key_rw_chunk(s)
                rho, rho_exposure = operator_relation(h_bf, op_key)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = drive.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = drive.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T  # [B,S,N]
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_margin_band = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
                chunk_margin_band_wide = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
                chunk_margin_band_mid = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
                g_safe = execution_weight + 1e-8
                chunk_g_log_g = (execution_weight * jnp.log(g_safe)).sum(
                    axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission))
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, execution_weight.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_margin_band + chunk_margin_band,
                        total_den_cost + chunk_den_cost,
                        total_selection_cost,
                        total_current_cost + chunk_current_cost,
                        total_margin_band_wide + chunk_margin_band_wide,
                        total_margin_band_mid + chunk_margin_band_mid,
                        total_g_log_g + chunk_g_log_g,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        total_exposure_sum + chunk_exposure_sum,
                        jnp.minimum(total_exposure_min, chunk_exposure_min),
                        jnp.maximum(total_exposure_max, chunk_exposure_max),
                        total_weak_exposure_count + chunk_weak_exposure_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_int_cap_count + chunk_int_cap_count,
                        total_selection_residency_sum + chunk_selection_residency_sum,
                        total_selection_residency_count + chunk_soft_dead_1e4_count,
                        total_edge_margin_stat + chunk_edge_margin_stat,
                        add_sparsity_carry(sparsity_carry, chunk_sparsity),
                        select_diag_carry), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_margin_band, total_den_cost, total_selection_cost,
             total_current_cost, total_margin_band_wide, total_margin_band_mid,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_exposure_sum, total_exposure_min,
             total_exposure_max, total_weak_exposure_count,
             total_int_max, total_int_cap_count, total_selection_residency_sum,
             total_selection_residency_count,
             total_edge_margin_stat,
             sparsity_carry,
             select_diag_carry), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), diag_pos_inf, diag_neg_inf,
                 jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0),
                 sparsity_carry0,
                 select_diag_carry0),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_den_cost,
                 total_selection_cost, total_current_cost,
                 total_dead_penalty, total_dead_count,
                 total_exposure_sum, total_exposure_min,
                 total_exposure_max, total_weak_exposure_count,
                 total_int_max, total_int_cap_count, total_selection_residency_sum,
                 total_selection_residency_count,
                 total_edge_margin_stat,
                 sparsity_carry, select_diag_carry) = carry
                s = i * cs
                op_key, rc, wc = op_key_rw_chunk(s)
                rho, rho_exposure = operator_relation(h_bf, op_key)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = drive.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = drive.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission))
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, execution_weight.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_den_cost + chunk_den_cost,
                        total_selection_cost,
                        total_current_cost + chunk_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        total_exposure_sum + chunk_exposure_sum,
                        jnp.minimum(total_exposure_min, chunk_exposure_min),
                        jnp.maximum(total_exposure_max, chunk_exposure_max),
                        total_weak_exposure_count + chunk_weak_exposure_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_int_cap_count + chunk_int_cap_count,
                        total_selection_residency_sum + chunk_selection_residency_sum,
                        total_selection_residency_count + chunk_soft_dead_1e4_count,
                        total_edge_margin_stat + chunk_edge_margin_stat,
                        add_sparsity_carry(sparsity_carry, chunk_sparsity),
                        select_diag_carry), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_selection_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_exposure_sum, total_exposure_min,
             total_exposure_max, total_weak_exposure_count,
             total_int_max, total_int_cap_count, total_selection_residency_sum,
             total_selection_residency_count,
             total_edge_margin_stat,
             sparsity_carry,
             select_diag_carry), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                 diag_pos_inf, diag_neg_inf, jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0),
                 jnp.float32(0.0),
                 sparsity_carry0,
                 select_diag_carry0),
                jnp.arange(nc))


        # Normalize per route independently
        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')   # sum(execution_weight)
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        global_selection_cost = jax.lax.psum(total_selection_cost, 'model')
        global_current_cost = jax.lax.psum(total_current_cost, 'model')
        selection_residency_loss = jnp.float32(0.0)
        edge_margin_stat = (
            jax.lax.psum(total_edge_margin_stat, 'model')
            / jnp.float32(B * S * 2 * N_total))
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        # v4164: sublinear divisive inhibition over admission mass.
        # Forward:
        #   admission_den_forward = max(sum(admission), 1) ** admission_den_power
        # Backward:
        #   only den_grad_scale fraction of the denominator gradient is live.
        # This keeps the inhibitory/gain-control term biologically plausible
        # while avoiding the early full-denominator collapse observed with
        # admission_den_grad_scale=1 and admission_den_power=1.
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = admission_den_sg + _admission_den_grad_scale * (admission_den_forward - admission_den_sg)
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_active = jax.lax.psum(total_active, 'model')
        (total_selected, total_selection_margin_sum,
         total_positive_margin_sum, total_positive_margin_max,
         total_rho_max, total_selection_margin_max) = select_diag_carry
        global_selected = jax.lax.psum(total_selected, 'model')
        global_selection_margin_sum = jax.lax.psum(
            total_selection_margin_sum, 'model')
        global_positive_margin_sum = jax.lax.psum(
            total_positive_margin_sum, 'model')
        global_positive_margin_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_positive_margin_max), 'model')
        global_rho_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_rho_max), 'model')
        global_selection_margin_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_selection_margin_max), 'model')
        no_active_direct = jax.lax.stop_gradient(
            (global_selection_margin_max <= 0.0).astype(jnp.float32))
        tau_direct = _block_tau_up_when_no_active(
            tau, no_active_direct.astype(jnp.bool_))
        # Measurement path: detached copies for diagnostics / feedback refs.
        # The forward denominator is already detached above; numerator paths
        # through execution_weight remain live for SRW output gradients.
        global_weighted_cost_m = jax.lax.stop_gradient(global_weighted_cost)
        global_gate_sq_m = jax.lax.stop_gradient(global_gate_sq)
        global_den_cost_m = jax.lax.stop_gradient(global_den_cost)
        global_selection_cost_m = jax.lax.stop_gradient(global_selection_cost)
        global_current_cost_m = jax.lax.stop_gradient(global_current_cost)
        global_active_m = jax.lax.stop_gradient(global_active)
        global_strong_m = jax.lax.stop_gradient(
            jax.lax.psum(total_strong, 'model'))
        global_gate_max_m = jax.lax.stop_gradient(global_gate_max)
        global_selected_m = jax.lax.stop_gradient(global_selected)
        active_frac = global_active_m / N_total
        active_frac_mean = active_frac.mean(axis=2)
        strong_frac = global_strong_m / N_total
        strong_frac_mean = strong_frac.mean(axis=2)
        positive_margin_mean_active = (
            jax.lax.stop_gradient(global_positive_margin_sum)
            / (global_active_m + 1e-8))
        positive_margin_mean_active_mean = (
            positive_margin_mean_active.mean(axis=2))
        raw_gate_max_mean = global_gate_max_m.mean(axis=2)

        rho_std_out = jax.lax.stop_gradient(rho_std.mean())
        es_out = global_weighted_cost_m.mean()
        active_n_mean = global_active_m.mean()
        gate_eff_n = ((global_weighted_cost_m ** 2)
                      / (global_gate_sq_m + 1e-8))
        gate_eff_ratio = gate_eff_n / jnp.maximum(global_active_m, 1.0)
        top1_gate_frac = global_gate_max_m / jnp.maximum(
            global_weighted_cost_m, 1e-8)
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        global_exposure_sum = jax.lax.stop_gradient(
            jax.lax.psum(total_exposure_sum, 'model'))
        global_exposure_min = jax.lax.pmin(
            jax.lax.stop_gradient(total_exposure_min), 'model')
        global_exposure_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_exposure_max), 'model')
        global_weak_exposure_count = jax.lax.stop_gradient(
            jax.lax.psum(total_weak_exposure_count, 'model'))
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost_m.mean()
        selection_cost_mean = global_selection_cost_m.mean()
        current_cost_mean = global_current_cost_m.mean()

        rho_count = jnp.float32(B * S * 2 * N_total)
        token_count = jnp.float32(B * S * 2)
        raw_tau_sg = jax.lax.stop_gradient(raw_tau)
        tau_sg = jax.lax.stop_gradient(tau)
        select_diag_out = (
            jax.lax.stop_gradient(rho_mean.mean()),
            jax.lax.stop_gradient(rho_std.mean()),
            jax.lax.stop_gradient(global_rho_max),
            tau_sg.mean(),
            tau_sg.min(),
            tau_sg.max(),
            raw_tau_sg.mean(),
            raw_tau_sg.min(),
            raw_tau_sg.max(),
            jax.lax.stop_gradient(global_selection_margin_sum / rho_count),
            jax.lax.stop_gradient(global_positive_margin_sum / rho_count),
            jax.lax.stop_gradient(global_positive_margin_max),
            jax.lax.stop_gradient(global_selected_m.mean() / N_total),
            jax.lax.stop_gradient(no_active_direct.mean()),
        )
        dead_exposure_diag_out = (
            global_exposure_sum / jnp.float32(N_total),
            global_exposure_min,
            global_exposure_max,
            dead_count_out / jnp.float32(N_total),
            global_weak_exposure_count / jnp.float32(N_total),
            jax.lax.stop_gradient(
                jax.lax.psum(total_selection_residency_count, 'model')
                / jnp.float32(N_total)),
        )

        slim_out = (out.astype(jnp.float32), active_frac_mean, raw_gate_max_mean, rho_lb,
                    rho_std_out, es_out, active_n_mean, strong_frac_mean,
                    positive_margin_mean_active_mean, tau_abs_mean,
                    dead_penalty_out, dead_count_out,
                    int_max_out, den_cost_mean, selection_cost_mean,
                    current_cost_mean, selection_residency_loss,
                    edge_margin_stat,
                    tau_direct.astype(jnp.float32),
                    no_active_direct)
        conc_out = (gate_eff_n.mean(), gate_eff_ratio.mean(),
                    top1_gate_frac.mean(), top1_gate_frac.max())
        route_split_out = (
            active_frac[:, :, 0, :].mean(),
            active_frac[:, :, 1, :].mean(),
            strong_frac[:, :, 0, :].mean(),
            strong_frac[:, :, 1, :].mean(),
            global_active_m[:, :, 0, :].mean(),
            global_active_m[:, :, 1, :].mean(),
        )
        sparsity_diag_out = (
            finalize_sparsity_diag(sparsity_carry)
            if _sparsity_diag_enabled
            else jnp.zeros((2, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32))
        if not analysis:
            return (slim_out + conc_out + route_split_out + select_diag_out
                    + dead_exposure_diag_out + (sparsity_diag_out,)
                   )

        # --- Analysis-only extras ---
        margin_band_frac = jax.lax.psum(total_margin_band, 'model') / N_total
        margin_band_frac_mean = jax.lax.stop_gradient(margin_band_frac).mean(axis=2)
        _active_denom = jnp.maximum(global_active_m, 1.0)
        margin_band_wide_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_margin_band_wide, 'model') / _active_denom).mean())
        margin_band_mid_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_margin_band_mid, 'model') / _active_denom).mean())
        active_per_token_std = global_active_m.std()
        global_g_log_g = jax.lax.stop_gradient(
            jax.lax.psum(total_g_log_g, 'model'))
        gate_sum_eps = jnp.maximum(global_weighted_cost_m, 1e-6)
        safe_glogg = jnp.where(
            global_weighted_cost_m > 1e-6, global_g_log_g, 0.0)
        entropy_per_token = -safe_glogg / gate_sum_eps + jnp.log(gate_sum_eps)
        entropy_per_token = jnp.where(
            jnp.isfinite(entropy_per_token), entropy_per_token, 0.0)
        gate_entropy = entropy_per_token.mean()
        den_cost_out = global_den_cost_m.mean()
        selection_cost_out = global_selection_cost_m.mean()
        current_cost_out = global_current_cost_m.mean()
        int_cap_frac_out = jax.lax.stop_gradient(
            jax.lax.psum(total_int_cap_count, 'model')
            / jnp.float32(B * S * 2 * N_total))
        return (slim_out + conc_out + route_split_out
                + (margin_band_frac_mean, margin_band_wide_frac,
                   margin_band_mid_frac, rho_skew, active_per_token_std,
                   gate_entropy, den_cost_out, selection_cost_out,
                   current_cost_out, rho_kurt, int_cap_frac_out)
                + select_diag_out + dead_exposure_diag_out
                + (sparsity_diag_out,)
               )

    return fused_gate_srw_paired


def make_sharded_srw_minimal(mesh, max_chunk_size=2048,
                             dead_exposure_target=0.1,
                             soft_gate_effective_active_eps=1.0e-6,
                             admission_den_power=1.0,
                             admission_den_grad_scale=1.0,
                             operator_pages_enabled=False,
                             operator_page_size=128,
                             operator_page_capacity=32,
                             operator_page_microgroup_sequences=2,
                             operator_page_score_mode='maxmean',
                             operator_page_fallback_pages=0,
                             operator_page_random_pages=0):
    """Create a shard_map'd single-route SRW kernel that returns only output."""
    if operator_pages_enabled:
        return make_sharded_srw_page_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            operator_page_size=operator_page_size,
            operator_page_capacity=operator_page_capacity,
            operator_page_microgroup_sequences=operator_page_microgroup_sequences,
            operator_page_score_mode=operator_page_score_mode,
            operator_page_fallback_pages=operator_page_fallback_pages,
            operator_page_random_pages=operator_page_random_pages)
    del dead_exposure_target
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=P('data', None, None),
             check_rep=False)
    def fused_gate_srw_minimal(x, h, op_key_local, raw_tau,
                               read_local, write_local,
                               soft_gate_temperature, soft_gate_t_final,
                               soft_gate_boundary_power,
                               soft_gate_boundary_power_final,
                               execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        N_local = op_key_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        op_key_bf = op_key_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        tau = _tau_from_param(raw_tau)

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_bf, start, cs, axis=0)

        def operator_relation(h_in, op_key):
            q_unit = _forward_unit_direction(
                h_in.astype(jnp.float32)).astype(jnp.bfloat16)
            op_key_unit = _forward_unit_direction(
                op_key.astype(jnp.float32)).astype(jnp.bfloat16)
            return (q_unit @ op_key_unit.T).astype(jnp.float32)

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_dir = _forward_unit_direction(rc.astype(jnp.float32))
            wc_dir = _forward_unit_direction(wc.astype(jnp.float32))
            return ec, rc_dir.astype(jnp.bfloat16), wc_dir.astype(jnp.bfloat16)

        def angular_compose_parts(rho):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            return admission, execution_weight

        @jax.checkpoint
        def gate_srw_step(carry, i):
            raw_out, total_den_cost = carry
            s = i * cs
            op_key, rc, wc = op_key_rw_chunk(s)
            rho = operator_relation(h_bf, op_key)
            admission, execution_weight = angular_compose_parts(rho)
            xr = x_bf @ rc.T
            a = execution_weight * xr.astype(jnp.float32)
            c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
            chunk_den_cost = admission.sum(axis=-1, keepdims=True)
            return (raw_out + c_out,
                    total_den_cost + chunk_den_cost), None

        (raw_out, total_den_cost), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, D), dtype=jnp.float32),
             jnp.zeros((B, S, 1), dtype=jnp.float32)),
            jnp.arange(nc))

        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')
        return out.astype(jnp.float32)

    return fused_gate_srw_minimal


def make_sharded_srw_paired_minimal(mesh, max_chunk_size=2048,
                                    dead_exposure_target=0.1,
                                    soft_gate_effective_active_eps=1.0e-6,
                                    admission_den_power=1.0,
                                    admission_den_grad_scale=1.0,
                                    operator_pages_enabled=False,
                                    operator_page_size=128,
                                    operator_page_capacity=8,
                                    operator_page_microgroup_sequences=2,
                                    operator_page_score_mode='maxmean',
                                    operator_page_fallback_pages=0,
                                    operator_page_random_pages=0):
    """Create a shard_map'd Q/K SRW kernel that returns only paired output."""
    if operator_pages_enabled:
        return make_sharded_srw_paired_page_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            operator_page_size=operator_page_size,
            operator_page_capacity=operator_page_capacity,
            operator_page_microgroup_sequences=operator_page_microgroup_sequences,
            operator_page_score_mode=operator_page_score_mode,
            operator_page_fallback_pages=operator_page_fallback_pages,
            operator_page_random_pages=operator_page_random_pages)
    del dead_exposure_target
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None, None),
                       P('model', None),
                       P('data', None, None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=P('data', None, None, None),
             check_rep=False)
    def fused_gate_srw_paired_minimal(x, h, op_key_local, raw_tau,
                                      read_local, write_local,
                                      soft_gate_temperature,
                                      soft_gate_t_final,
                                      soft_gate_boundary_power,
                                      soft_gate_boundary_power_final,
                                      execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        N_local = op_key_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, _, _ = h.shape
        D = x.shape[-1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        op_key_bf = op_key_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        tau = _tau_from_param(raw_tau)

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_bf, start, cs, axis=0)

        def operator_relation(h_in, op_key):
            q_unit = _forward_unit_direction(
                h_in.astype(jnp.float32)).astype(jnp.bfloat16)
            op_key_unit = _forward_unit_direction(
                op_key.astype(jnp.float32)).astype(jnp.bfloat16)
            return jnp.einsum(
                'bsrd,nd->bsrn', q_unit, op_key_unit).astype(jnp.float32)

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_dir = _forward_unit_direction(rc.astype(jnp.float32))
            wc_dir = _forward_unit_direction(wc.astype(jnp.float32))
            return ec, rc_dir.astype(jnp.bfloat16), wc_dir.astype(jnp.bfloat16)

        def angular_compose_parts(rho):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            return admission, execution_weight

        @jax.checkpoint
        def gate_srw_step(carry, i):
            raw_out, total_den_cost = carry
            s = i * cs
            op_key, rc, wc = op_key_rw_chunk(s)
            rho = operator_relation(h_bf, op_key)
            admission, execution_weight = angular_compose_parts(rho)
            xr = x_bf @ rc.T
            a = execution_weight * xr.astype(jnp.float32)[:, :, None, :]
            c_out = jnp.einsum(
                'bsrn,nd->bsrd',
                a.astype(jnp.bfloat16),
                wc).astype(jnp.float32)
            chunk_den_cost = admission.sum(axis=-1, keepdims=True)
            return (raw_out + c_out,
                    total_den_cost + chunk_den_cost), None

        (raw_out, total_den_cost), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
             jnp.zeros((B, S, 2, 1), dtype=jnp.float32)),
            jnp.arange(nc))

        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')
        return out.astype(jnp.float32)

    return fused_gate_srw_paired_minimal


def _pad_operator_pool_for_pages(op_key, read, write, page_size):
    """Pad one model-axis local operator shard to a static page multiple."""
    page_size = int(page_size)
    n_valid = int(op_key.shape[0])
    n_pad = ((n_valid + page_size - 1) // page_size) * page_size
    pad_n = n_pad - n_valid
    page_count = n_pad // page_size
    op_key_pad = jnp.pad(op_key, ((0, pad_n), (0, 0)))
    read_pad = jnp.pad(read, ((0, pad_n), (0, 0)))
    write_pad = jnp.pad(write, ((0, pad_n), (0, 0)))
    valid_mask = jnp.arange(n_pad, dtype=jnp.int32) < jnp.int32(n_valid)
    return op_key_pad, read_pad, write_pad, valid_mask, n_valid, n_pad, page_count


def _validate_operator_page_args(pool_name, n_valid, page_size,
                                 page_capacity, fallback_pages,
                                 random_pages):
    page_size = int(page_size)
    page_capacity = int(page_capacity)
    fallback_pages = int(fallback_pages)
    random_pages = int(random_pages)
    if page_size <= 0:
        raise ValueError(f"{pool_name} operator_page_size must be > 0.")
    n_valid = int(n_valid)
    if n_valid <= 0:
        raise ValueError(f"{pool_name} local operator count must be > 0.")
    page_count = (n_valid + page_size - 1) // page_size
    if page_capacity <= 0:
        raise ValueError(f"{pool_name} operator_page_capacity must be > 0.")
    if fallback_pages < 0 or random_pages < 0:
        raise ValueError(
            f"{pool_name} fallback/random pages must be non-negative.")
    effective_pages = page_capacity + fallback_pages + random_pages
    if effective_pages > page_count:
        raise ValueError(
            f"{pool_name} requests {effective_pages} candidate pages "
            f"(capacity={page_capacity}, fallback={fallback_pages}, "
            f"random={random_pages}) but only {page_count} local pages exist.")
    return page_count, effective_pages


def _operator_page_tables(op_key_local, read_local, write_local, page_size):
    op_key_local = _forward_unit_direction(op_key_local.astype(jnp.float32))
    read_local = _forward_unit_direction(read_local.astype(jnp.float32))
    write_local = _forward_unit_direction(write_local.astype(jnp.float32))
    (op_key_pad, read_pad, write_pad, valid_mask, n_valid, n_pad,
     page_count) = _pad_operator_pool_for_pages(
        op_key_local, read_local, write_local, page_size)
    op_key_pages = op_key_pad.reshape(
        page_count, int(page_size), op_key_local.shape[-1]).astype(jnp.float32)
    read_pages = read_pad.reshape(
        page_count, int(page_size), read_local.shape[-1]).astype(jnp.float32)
    write_pages = write_pad.reshape(
        page_count, int(page_size), write_local.shape[-1]).astype(jnp.float32)
    valid_pages = valid_mask.reshape(page_count, int(page_size))
    valid_pages_f = valid_pages.astype(jnp.float32)
    valid_count = valid_pages_f.sum(axis=1, keepdims=True)
    page_keys = _forward_unit_direction(
        (op_key_pages * valid_pages_f[..., None]).sum(axis=1)
        / jnp.maximum(valid_count, jnp.float32(1.0)))
    valid_page_mask = valid_count[:, 0] > 0.0
    return (
        op_key_pages, read_pages, write_pages, valid_pages,
        valid_page_mask, page_keys, n_valid, n_pad, page_count)


def _operator_page_table_arg(op_key_local, read_local, write_local, page_size):
    """Shard-local normalized page tables for reuse across layers.

    The returned pytree contains only arrays that are safe to pass through
    shard_map.  Full-pool size diagnostics are recovered from valid_pages with
    a model-axis psum inside the page kernels, so no host-side/global table is
    needed.
    """
    (op_key_pages, read_pages, write_pages, valid_pages, valid_page_mask,
     page_keys, _, _, _) = _operator_page_tables(
        op_key_local, read_local, write_local, page_size)
    return (
        op_key_pages, read_pages, write_pages, valid_pages,
        valid_page_mask, page_keys)


def _unpack_operator_page_tables(page_tables):
    return page_tables


OPERATOR_PAGE_TABLE_SPECS = (
    P('model', None, None),  # op_key_pages [P_local, page_size, d_route]
    P('model', None, None),  # read_pages   [P_local, page_size, d_model]
    P('model', None, None),  # write_pages  [P_local, page_size, d_model]
    P('model', None),        # valid_pages  [P_local, page_size]
    P('model'),              # valid_page_mask [P_local]
    P('model', None),        # page_keys [P_local, d_route]
)


def make_sharded_operator_page_tables(mesh, operator_page_size=128):
    """Build reusable shard-local page tables for one operator pool.

    This is intentionally inside the JAX graph and inside shard_map: each model
    shard converts only its local operator slice into page tables, gradients stay
    live through op_key/read/write, and no global all_gather or host precompute
    is introduced.
    """
    _page_size = int(operator_page_size)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('model', None),
                       P('model', None),
                       P('model', None)),
             out_specs=OPERATOR_PAGE_TABLE_SPECS,
             check_rep=False)
    def build_page_tables(op_key_local, read_local, write_local):
        return _operator_page_table_arg(
            op_key_local, read_local, write_local, _page_size)

    return build_page_tables


def _pool_params_with_operator_page_tables(pool_params, sharded_fns):
    """Attach per-forward reusable page tables when builders are available.

    The builders are supplied by the trainer because they are mesh-dependent.
    Missing builders mean the caller is using the old full-pool path or a
    non-page pool; in that case the original pool params are returned.
    """
    if not isinstance(sharded_fns, dict):
        return pool_params
    out = dict(pool_params)

    def _maybe_add(prefix, builder_key, op_key_key, read_key, write_key):
        builder = sharded_fns.get(builder_key)
        if builder is None or f'{prefix}_page_tables' in out:
            return
        out[f'{prefix}_page_tables'] = builder(
            out[op_key_key], out[read_key], out[write_key])

    _maybe_add(
        'attn_qk', 'attn_qk_page_tables',
        'attn_qk_op_key', 'attn_qk_read', 'attn_qk_write')
    _maybe_add(
        'attn_v', 'attn_v_page_tables',
        'attn_v_op_key', 'attn_v_read', 'attn_v_write')
    _maybe_add(
        'rst', 'rst_page_tables',
        'rst_op_key', 'rst_read', 'rst_write')
    return out


def _validate_operator_page_table_args(pool_name, page_count, page_size,
                                       page_capacity, fallback_pages,
                                       random_pages):
    page_size = int(page_size)
    page_capacity = int(page_capacity)
    fallback_pages = int(fallback_pages)
    random_pages = int(random_pages)
    page_count = int(page_count)
    if page_size <= 0:
        raise ValueError(f"{pool_name} operator_page_size must be > 0.")
    if page_count <= 0:
        raise ValueError(f"{pool_name} local page table must be non-empty.")
    if page_capacity <= 0:
        raise ValueError(f"{pool_name} operator_page_capacity must be > 0.")
    if fallback_pages < 0 or random_pages < 0:
        raise ValueError(
            f"{pool_name} fallback/random pages must be non-negative.")
    effective_pages = page_capacity + fallback_pages + random_pages
    if effective_pages > page_count:
        raise ValueError(
            f"{pool_name} requests {effective_pages} candidate pages "
            f"(capacity={page_capacity}, fallback={fallback_pages}, "
            f"random={random_pages}) but only {page_count} local pages exist.")
    return effective_pages


def _sorted_selected_page_ids(fallback_ids, top_ids, random_ids):
    selected_page_ids = jnp.concatenate(
        (fallback_ids, top_ids.astype(jnp.int32), random_ids), axis=1)
    return jnp.sort(selected_page_ids, axis=1)


def _normalize_operator_page_score_mode(score_mode):
    score_mode = str(score_mode).strip().lower()
    if score_mode not in ('mean', 'maxmean'):
        raise ValueError(
            "operator_page_score_mode must be 'mean' or 'maxmean', "
            f"got {score_mode!r}.")
    return score_mode


def _static_page_diag(n_global, model_axis_size, page_size, page_capacity,
                      fallback_pages, random_pages):
    del model_axis_size
    n_global = int(n_global)
    page_size = int(page_size)
    page_count_total = (
        (n_global + page_size - 1) // page_size if page_size > 0 else 0)
    candidate_ops = jnp.float32(
        (int(page_capacity) + int(fallback_pages) + int(random_pages))
        * page_size)
    candidate_valid_ops = jnp.minimum(candidate_ops, jnp.float32(n_global))
    n_global_f = jnp.maximum(jnp.float32(n_global), jnp.float32(1.0))
    effective_pages = jnp.float32(
        int(page_capacity) + int(fallback_pages) + int(random_pages))
    page_diag = jnp.zeros((PAGE_DIAG_COUNT,), dtype=jnp.float32)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['page_size']].set(
        jnp.float32(page_size))
    page_diag = page_diag.at[PAGE_DIAG_INDEX['page_capacity']].set(
        jnp.float32(page_capacity))
    page_diag = page_diag.at[PAGE_DIAG_INDEX['page_count_total']].set(
        jnp.float32(page_count_total))
    page_diag = page_diag.at[PAGE_DIAG_INDEX['page_count_effective']].set(
        effective_pages)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_ops']].set(
        candidate_ops)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_valid_ops']].set(
        candidate_valid_ops)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_frac']].set(
        candidate_ops / n_global_f)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_valid_frac']].set(
        candidate_valid_ops / n_global_f)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['page_fallback_used_frac']].set(
        jnp.float32(fallback_pages) / jnp.maximum(effective_pages, 1.0))
    page_diag = page_diag.at[PAGE_DIAG_INDEX['page_random_used_frac']].set(
        jnp.float32(random_pages) / jnp.maximum(effective_pages, 1.0))
    page_diag = page_diag.at[PAGE_DIAG_INDEX['selected_page_count']].set(
        effective_pages)
    page_diag = page_diag.at[PAGE_DIAG_INDEX['estimated_compute_frac_page']].set(
        candidate_ops / n_global_f)
    return page_diag


def _operator_pages_pool_enabled_static(enabled, pools, pool):
    if not bool(enabled):
        return False
    if pools is None:
        return True
    if isinstance(pools, bool):
        return pools
    if isinstance(pools, dict):
        return bool(pools.get(pool, pools.get(f'operator_pages_{pool}', False)))
    if isinstance(pools, str):
        values = {
            item.strip().lower()
            for item in pools.replace(',', ' ').split()
            if item.strip()
        }
    else:
        values = {str(item).strip().lower() for item in pools}
    return 'all' in values or pool in values


def _operator_page_static_result(prefix, enabled, n_ops, page_size,
                                 page_capacity, fallback_pages,
                                 random_pages):
    enabled_f = jnp.float32(1.0 if enabled else 0.0)
    page_size = int(page_size)
    page_capacity = int(page_capacity)
    fallback_pages = int(fallback_pages)
    random_pages = int(random_pages)
    candidate_pages = (
        page_capacity + fallback_pages + random_pages if enabled else 0)
    candidate_ops = candidate_pages * page_size if enabled else int(n_ops)
    candidate_valid_ops = (
        min(candidate_ops, int(n_ops)) if enabled else int(n_ops))
    n_ops_f = jnp.maximum(jnp.float32(n_ops), jnp.float32(1.0))
    candidate_frac = (
        jnp.float32(candidate_ops) / n_ops_f if enabled else jnp.float32(1.0))
    candidate_valid_frac = (
        jnp.float32(candidate_valid_ops) / n_ops_f
        if enabled else jnp.float32(1.0))
    page_count = (
        (int(n_ops) + page_size - 1) // page_size
        if enabled and page_size > 0 else 0)
    return {
        f'{prefix}_pages_enabled': enabled_f,
        f'{prefix}_page_size': jnp.float32(page_size),
        f'{prefix}_page_capacity': jnp.float32(page_capacity if enabled else 0),
        f'{prefix}_page_count_total': jnp.float32(page_count),
        f'{prefix}_page_count_effective': jnp.float32(candidate_pages),
        f'{prefix}_page_top1_frac': jnp.float32(0.0),
        f'{prefix}_page_entropy': jnp.float32(0.0),
        f'{prefix}_page_score_max': jnp.float32(0.0),
        f'{prefix}_page_score_mean': jnp.float32(0.0),
        f'{prefix}_page_score_std': jnp.float32(0.0),
        f'{prefix}_candidate_ops': jnp.float32(candidate_ops),
        f'{prefix}_candidate_valid_ops': jnp.float32(candidate_valid_ops),
        f'{prefix}_candidate_frac': candidate_frac,
        f'{prefix}_candidate_valid_frac': candidate_valid_frac,
        f'{prefix}_candidate_frac_of_pool': candidate_frac,
        f'{prefix}_candidate_den_mean': jnp.float32(0.0),
        f'{prefix}_candidate_execution_mass': jnp.float32(0.0),
        f'{prefix}_estimated_compute_frac_page': candidate_frac,
        f'{prefix}_page_fallback_used_frac': (
            jnp.float32(fallback_pages) / jnp.maximum(
                jnp.float32(candidate_pages), 1.0)
            if enabled else jnp.float32(0.0)),
        f'{prefix}_page_random_used_frac': (
            jnp.float32(random_pages) / jnp.maximum(
                jnp.float32(candidate_pages), 1.0)
            if enabled else jnp.float32(0.0)),
        f'{prefix}_page_no_route_frac': jnp.float32(0.0),
    }


def _select_operator_pages_for_calibration(
        h, op_key, pool_name, page_size, page_capacity,
        microgroup_sequences, score_mode, fallback_pages, random_pages,
        paired=False):
    """Run the forward page top-k path and return candidate rho + masks."""
    n_ops = int(op_key.shape[0])
    page_count, effective_pages = _validate_operator_page_args(
        pool_name, n_ops, page_size, page_capacity,
        fallback_pages, random_pages)
    k_candidates = effective_pages * int(page_size)
    score_mode = _normalize_operator_page_score_mode(score_mode)

    batch_size = int(h.shape[0])
    seq_len = int(h.shape[1])
    route_dim = int(h.shape[-1])
    mg = min(max(1, int(microgroup_sequences)), batch_size)
    group_count = (batch_size + mg - 1) // mg
    batch_pad = group_count * mg
    pad_b = batch_pad - batch_size
    if paired:
        route_count = int(h.shape[2])
        h_pad = jnp.pad(h, ((0, pad_b), (0, 0), (0, 0), (0, 0)))
        h_groups = h_pad.reshape(
            group_count, mg, seq_len, route_count, route_dim)
    else:
        h_pad = jnp.pad(h, ((0, pad_b), (0, 0), (0, 0)))
        h_groups = h_pad.reshape(group_count, mg, seq_len, route_dim)

    # Only op_key participates in page scoring/candidate rho.  The read/write
    # arguments are unused placeholders so this follows _operator_page_tables.
    (op_key_pages_f, _, _, valid_pages, valid_page_mask, page_keys,
     _, _, _) = _operator_page_tables(op_key, op_key, op_key, page_size)
    page_key_bf = page_keys.astype(jnp.bfloat16)
    h_unit = _forward_unit_direction(
        h_groups.astype(jnp.float32)).astype(jnp.bfloat16)

    if paired:
        if score_mode == 'maxmean':
            evidence = jnp.einsum(
                'gmsrd,pd->gmsrp', h_unit, page_key_bf).astype(jnp.float32)
            evidence = evidence.max(axis=3)
            page_scores = (
                evidence.mean(axis=(1, 2)) + evidence.max(axis=(1, 2))
            ) * jnp.float32(0.5)
        else:
            route_summary = _forward_unit_direction(
                h_groups.astype(jnp.float32).mean(axis=(1, 2, 3)))
            page_scores = (
                route_summary.astype(jnp.bfloat16) @ page_key_bf.T
            ).astype(jnp.float32)
    else:
        if score_mode == 'maxmean':
            evidence = jnp.einsum(
                'gmsd,pd->gmsp', h_unit, page_key_bf).astype(jnp.float32)
            page_scores = (
                evidence.mean(axis=(1, 2)) + evidence.max(axis=(1, 2))
            ) * jnp.float32(0.5)
        else:
            route_summary = _forward_unit_direction(
                h_groups.astype(jnp.float32).mean(axis=(1, 2)))
            page_scores = (
                route_summary.astype(jnp.bfloat16) @ page_key_bf.T
            ).astype(jnp.float32)
    page_scores = jnp.where(
        valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30))

    protected = jnp.zeros((group_count, page_count), dtype=jnp.bool_)
    if int(fallback_pages):
        fallback_ids = jnp.arange(int(fallback_pages), dtype=jnp.int32)
        fallback_ids = jnp.broadcast_to(
            fallback_ids[None, :], (group_count, int(fallback_pages)))
        protected = protected.at[:, :int(fallback_pages)].set(True)
    else:
        fallback_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)
    if int(random_pages):
        random_span = page_count - int(fallback_pages)
        random_offsets = (
            jnp.arange(group_count, dtype=jnp.int32)[:, None]
            * jnp.int32(1103515245)
            + jnp.arange(int(random_pages), dtype=jnp.int32)[None, :]
            * jnp.int32(12345)
            + jnp.int32(17))
        random_ids = int(fallback_pages) + jnp.mod(
            random_offsets, jnp.int32(random_span))
        protected = protected.at[
            jnp.arange(group_count, dtype=jnp.int32)[:, None],
            random_ids].set(True)
    else:
        random_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)

    top_scores = jnp.where(protected, jnp.float32(-1.0e30), page_scores)
    _, top_ids = jax.lax.top_k(top_scores, int(page_capacity))
    selected_page_ids = _sorted_selected_page_ids(
        fallback_ids, top_ids, random_ids)
    cand_valid = valid_pages[selected_page_ids].reshape(
        group_count, k_candidates)
    cand_key = op_key_pages_f[selected_page_ids].reshape(
        group_count, k_candidates, op_key.shape[-1])
    cand_key = cand_key.astype(jnp.bfloat16)

    if paired:
        rho_raw = jnp.einsum(
            'gmsrd,gkd->gmsrk', h_unit, cand_key).astype(jnp.float32)
        valid = jnp.broadcast_to(
            cand_valid[:, None, None, None, :],
            (group_count, mg, seq_len, route_count, k_candidates))
        rho = rho_raw.reshape(
            batch_pad, seq_len, route_count, k_candidates)[:batch_size]
        valid = valid.reshape(
            batch_pad, seq_len, route_count, k_candidates)[:batch_size]
    else:
        rho_raw = jnp.einsum(
            'gmsd,gkd->gmsk', h_unit, cand_key).astype(jnp.float32)
        valid = jnp.broadcast_to(
            cand_valid[:, None, None, :],
            (group_count, mg, seq_len, k_candidates))
        rho = rho_raw.reshape(
            batch_pad, seq_len, k_candidates)[:batch_size]
        valid = valid.reshape(
            batch_pad, seq_len, k_candidates)[:batch_size]

    candidate_valid_count = cand_valid.astype(jnp.float32).sum(axis=-1).mean()
    return {
        'rho': jnp.where(valid, rho, jnp.float32(0.0)),
        'valid_mask': valid,
        'candidate_valid_count': candidate_valid_count,
        'candidate_count': jnp.asarray(k_candidates, dtype=jnp.float32),
        'full_pool_size': jnp.asarray(n_ops, dtype=jnp.float32),
        'candidate_group_count': jnp.asarray(group_count, dtype=jnp.float32),
    }


def _page_gate_sparsity_diag(selection_margin, admission, execution_weight,
                             valid_mask, token_count,
                             soft_gate_t_final,
                             soft_gate_boundary_power_final):
    """Candidate-local sparsity diagnostics for page-routed SRW."""
    current_eps = jnp.asarray(GATE_CURRENT_EPS, dtype=jnp.float32)
    projected_eps = jnp.asarray(GATE_PROJECTED_EPS, dtype=jnp.float32)
    margin_sg = jax.lax.stop_gradient(selection_margin)
    admission_sg = jax.lax.stop_gradient(admission)
    execution_sg = jax.lax.stop_gradient(execution_weight)
    valid_b = jax.lax.stop_gradient(valid_mask.astype(jnp.bool_))
    valid_f = valid_b.astype(jnp.float32)

    element_count = jax.lax.psum(valid_f.sum(), 'model')
    element_count = jnp.maximum(element_count, jnp.float32(1.0))
    token_count = jnp.maximum(jnp.asarray(token_count, dtype=jnp.float32), 1.0)

    active_tau = (margin_sg > 0.0) & valid_b
    active_tau_count = jax.lax.psum(
        active_tau.astype(jnp.float32).sum(), 'model')

    admission_active = (
        (admission_sg[..., None] > current_eps)
        & valid_b[..., None])
    admission_active_count = jax.lax.psum(
        admission_active.astype(jnp.float32).sum(axis=tuple(
            range(admission_active.ndim - 1))), 'model')
    current_active = (
        (execution_sg[..., None] > current_eps)
        & valid_b[..., None])
    current_active_count = jax.lax.psum(
        current_active.astype(jnp.float32).sum(axis=tuple(
            range(current_active.ndim - 1))), 'model')
    current_mass = jax.lax.psum(
        (execution_sg[..., None] * current_active.astype(jnp.float32)).sum(
            axis=tuple(range(current_active.ndim - 1))), 'model')
    gate_mass = jax.lax.psum((execution_sg * valid_f).sum(), 'model')

    projected_gate = _boundary_gate_from_margin(
        margin_sg, soft_gate_t_final, soft_gate_boundary_power_final) * valid_f
    projected_active = (
        (projected_gate[..., None] > projected_eps)
        & valid_b[..., None])
    projected_active_count = jax.lax.psum(
        projected_active.astype(jnp.float32).sum(axis=tuple(
            range(projected_active.ndim - 1))), 'model')
    projected_mass = jax.lax.psum(
        (projected_gate[..., None] * projected_active.astype(jnp.float32)).sum(
            axis=tuple(range(projected_active.ndim - 1))), 'model')
    projected_gate_mass = jax.lax.psum(projected_gate.sum(), 'model')

    margin_bands = jnp.stack((
        active_tau.astype(jnp.float32).sum(),
        (((margin_sg >= -0.01) & (margin_sg <= 0.0)) & valid_b
         ).astype(jnp.float32).sum(),
        (((margin_sg >= -0.03) & (margin_sg < -0.01)) & valid_b
         ).astype(jnp.float32).sum(),
        (((margin_sg >= -0.10) & (margin_sg < -0.03)) & valid_b
         ).astype(jnp.float32).sum(),
        ((margin_sg < -0.10) & valid_b).astype(jnp.float32).sum(),
    )).astype(jnp.float32)
    margin_bands = jax.lax.psum(margin_bands, 'model')

    out = jnp.zeros((GATE_SPARSITY_DIAG_COUNT,), dtype=jnp.float32)
    out = out.at[GATE_SPARSITY_DIAG_INDEX['active_tau_frac']].set(
        active_tau_count / element_count)
    out = out.at[GATE_SPARSITY_DIAG_INDEX['active_tau_count']].set(
        active_tau_count / token_count)
    for _i, _suffix in enumerate(GATE_EPS_NAME_SUFFIXES):
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[
                f'admission_active_eps_{_suffix}_frac']
        ].set(admission_active_count[_i] / element_count)
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[
                f'admission_active_eps_{_suffix}_count']
        ].set(admission_active_count[_i] / token_count)
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[f'active_eps_{_suffix}_frac']
        ].set(current_active_count[_i] / element_count)
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[f'active_eps_{_suffix}_count']
        ].set(current_active_count[_i] / token_count)
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[f'mass_eps_{_suffix}']
        ].set(current_mass[_i] / jnp.maximum(gate_mass, 1.0e-8))
    for _i, _suffix in enumerate(GATE_PROJECTED_EPS_NAME_SUFFIXES):
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[
                f'projected_Tfinal_active_eps_{_suffix}_frac']
        ].set(projected_active_count[_i] / element_count)
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[
                f'projected_Tfinal_active_eps_{_suffix}_count']
        ].set(projected_active_count[_i] / token_count)
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[
                f'projected_Tfinal_mass_eps_{_suffix}']
        ].set(projected_mass[_i] / jnp.maximum(projected_gate_mass, 1.0e-8))

    margin_frac = margin_bands / element_count
    for _i, _name in enumerate(MARGIN_BAND_NAMES):
        out = out.at[
            GATE_SPARSITY_DIAG_INDEX[f'margin_band_{_name}']
        ].set(margin_frac[_i])
    out = out.at[GATE_SPARSITY_DIAG_INDEX['margin_band_pos']].set(
        margin_frac[0])
    out = out.at[
        GATE_SPARSITY_DIAG_INDEX['margin_band_near_m0_03_0']
    ].set(margin_frac[1] + margin_frac[2])
    out = out.at[
        GATE_SPARSITY_DIAG_INDEX['margin_band_far_lt_m0_10']
    ].set(margin_frac[4])
    return jax.lax.stop_gradient(out.astype(jnp.float32))


def make_sharded_srw_page_minimal(mesh, max_chunk_size=2048,
                                  dead_exposure_target=0.1,
                                  soft_gate_effective_active_eps=1.0e-6,
                                  admission_den_power=1.0,
                                  admission_den_grad_scale=1.0,
                                  operator_page_size=128,
                                  operator_page_capacity=32,
                                  operator_page_microgroup_sequences=2,
                                  operator_page_score_mode='maxmean',
                                  operator_page_fallback_pages=0,
                                  operator_page_random_pages=0):
    """Minimal page-sparse single-route SRW.

    Page selection is per micro-local batch group.  Operators remain one global
    learned atlas; selected pages only define the dense candidate matrix used by
    the v4166 DirectTau RW execution.
    """
    del max_chunk_size, dead_exposure_target
    _model_axis_size = mesh.shape['model']
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    _page_size = int(operator_page_size)
    _page_capacity = int(operator_page_capacity)
    _microgroup_sequences = max(1, int(operator_page_microgroup_sequences))
    _score_mode = _normalize_operator_page_score_mode(
        operator_page_score_mode)
    _fallback_pages = int(operator_page_fallback_pages)
    _random_pages = int(operator_page_random_pages)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       OPERATOR_PAGE_TABLE_SPECS,
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=P('data', None, None),
             check_rep=False)
    def fused_gate_srw_page_minimal(x, h, page_tables, raw_tau,
                                    read_local, write_local,
                                    soft_gate_temperature,
                                    soft_gate_t_final,
                                    soft_gate_boundary_power,
                                    soft_gate_boundary_power_final,
                                    execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        del read_local, write_local
        (op_key_pages_f, read_pages_f, write_pages_f, valid_pages,
         valid_page_mask, page_keys) = _unpack_operator_page_tables(
            page_tables)
        page_count = op_key_pages_f.shape[0]
        effective_pages = _validate_operator_page_table_args(
            'single-route', page_count, _page_size, _page_capacity,
            _fallback_pages, _random_pages)
        K = effective_pages * _page_size

        B, S, D = x.shape
        mg = min(_microgroup_sequences, B)
        group_count = (B + mg - 1) // mg
        B_pad = group_count * mg
        pad_b = B_pad - B
        x_pad = jnp.pad(x, ((0, pad_b), (0, 0), (0, 0)))
        h_pad = jnp.pad(h, ((0, pad_b), (0, 0), (0, 0)))
        raw_tau_pad = jnp.pad(raw_tau, ((0, pad_b), (0, 0), (0, 0)))

        x_groups = x_pad.reshape(group_count, mg, S, D)
        h_groups = h_pad.reshape(group_count, mg, S, h.shape[-1])
        raw_tau_groups = raw_tau_pad.reshape(group_count, mg, S, 1)

        page_key_bf = page_keys.astype(jnp.bfloat16)
        h_unit = _forward_unit_direction(
            h_groups.astype(jnp.float32)).astype(jnp.bfloat16)
        if _score_mode == 'maxmean':
            evidence = jnp.einsum(
                'gmsd,pd->gmsp', h_unit, page_key_bf).astype(jnp.float32)
            page_scores = (
                evidence.mean(axis=(1, 2)) + evidence.max(axis=(1, 2))
            ) * jnp.float32(0.5)
        else:
            route_summary = _forward_unit_direction(
                h_groups.astype(jnp.float32).mean(axis=(1, 2)))
            page_scores = (
                route_summary.astype(jnp.bfloat16) @ page_key_bf.T
            ).astype(jnp.float32)
        page_scores = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30))

        protected = jnp.zeros((group_count, page_count), dtype=jnp.bool_)
        if _fallback_pages:
            fallback_ids = jnp.arange(_fallback_pages, dtype=jnp.int32)
            fallback_ids = jnp.broadcast_to(
                fallback_ids[None, :], (group_count, _fallback_pages))
            protected = protected.at[:, :_fallback_pages].set(True)
        else:
            fallback_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)
        if _random_pages:
            random_span = page_count - _fallback_pages
            random_offsets = (
                jnp.arange(group_count, dtype=jnp.int32)[:, None]
                * jnp.int32(1103515245)
                + jnp.arange(_random_pages, dtype=jnp.int32)[None, :]
                * jnp.int32(12345)
                + jnp.int32(17))
            random_ids = _fallback_pages + jnp.mod(
                random_offsets, jnp.int32(random_span))
            protected = protected.at[
                jnp.arange(group_count, dtype=jnp.int32)[:, None],
                random_ids].set(True)
        else:
            random_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)

        top_scores = jnp.where(protected, jnp.float32(-1.0e30), page_scores)
        _, top_ids = jax.lax.top_k(top_scores, _page_capacity)
        selected_page_ids = _sorted_selected_page_ids(
            fallback_ids, top_ids, random_ids)
        cand_valid = valid_pages[selected_page_ids].reshape(group_count, K)
        cand_valid_f = cand_valid.astype(jnp.float32)

        cand_key = op_key_pages_f[selected_page_ids].reshape(
            group_count, K, op_key_pages_f.shape[-1])
        cand_read = read_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_write = write_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_key = cand_key.astype(jnp.bfloat16)

        rho_raw = jnp.einsum(
            'gmsd,gkd->gmsk', h_unit, cand_key).astype(jnp.float32)
        cand_valid_b = cand_valid[:, None, None, :]
        rho_for_gate = jnp.where(cand_valid_b, rho_raw, jnp.float32(-1.0))
        tau = _tau_from_param(raw_tau_groups)
        _, admission, _, execution_weight, _ = _compute_admission_drive(
            rho_for_gate, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=_soft_gate_effective_active_eps,
            execution_prune_eps=execution_prune_eps)
        admission = admission * cand_valid_f[:, None, None, :]
        execution_weight = execution_weight * cand_valid_f[:, None, None, :]
        xr = jnp.einsum(
            'gmsd,gkd->gmsk', x_groups.astype(jnp.bfloat16), cand_read)
        a = execution_weight * xr.astype(jnp.float32)
        raw_out = jnp.einsum(
            'gmsk,gkd->gmsd', a.astype(jnp.bfloat16), cand_write
        ).astype(jnp.float32)
        den_cost = admission.sum(axis=-1, keepdims=True)

        raw_out = raw_out.reshape(B_pad, S, D)[:B]
        den_cost = den_cost.reshape(B_pad, S, 1)[:B]
        global_den_cost = jax.lax.psum(den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')
        return out.astype(jnp.float32)

    return fused_gate_srw_page_minimal


def make_sharded_srw_paired_page_minimal(mesh, max_chunk_size=2048,
                                         dead_exposure_target=0.1,
                                         soft_gate_effective_active_eps=1.0e-6,
                                         admission_den_power=1.0,
                                         admission_den_grad_scale=1.0,
                                         operator_page_size=128,
                                         operator_page_capacity=8,
                                         operator_page_microgroup_sequences=2,
                                         operator_page_score_mode='maxmean',
                                         operator_page_fallback_pages=0,
                                         operator_page_random_pages=0):
    """Minimal page-sparse paired Q/K SRW over a shared qk page set."""
    del max_chunk_size, dead_exposure_target
    _model_axis_size = mesh.shape['model']
    del _model_axis_size
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    _page_size = int(operator_page_size)
    _page_capacity = int(operator_page_capacity)
    _microgroup_sequences = max(1, int(operator_page_microgroup_sequences))
    _score_mode = _normalize_operator_page_score_mode(
        operator_page_score_mode)
    _fallback_pages = int(operator_page_fallback_pages)
    _random_pages = int(operator_page_random_pages)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None, None),
                       OPERATOR_PAGE_TABLE_SPECS,
                       P('data', None, None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=P('data', None, None, None),
             check_rep=False)
    def fused_gate_srw_paired_page_minimal(x, h, page_tables, raw_tau,
                                           read_local, write_local,
                                           soft_gate_temperature,
                                           soft_gate_t_final,
                                           soft_gate_boundary_power,
                                           soft_gate_boundary_power_final,
                                           execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        del read_local, write_local
        (op_key_pages_f, read_pages_f, write_pages_f, valid_pages,
         valid_page_mask, page_keys) = _unpack_operator_page_tables(
            page_tables)
        page_count = op_key_pages_f.shape[0]
        effective_pages = _validate_operator_page_table_args(
            'qk-paired', page_count, _page_size, _page_capacity,
            _fallback_pages, _random_pages)
        K = effective_pages * _page_size

        B, S, R, _ = h.shape
        D = x.shape[-1]
        mg = min(_microgroup_sequences, B)
        group_count = (B + mg - 1) // mg
        B_pad = group_count * mg
        pad_b = B_pad - B
        x_pad = jnp.pad(x, ((0, pad_b), (0, 0), (0, 0)))
        h_pad = jnp.pad(h, ((0, pad_b), (0, 0), (0, 0), (0, 0)))
        raw_tau_pad = jnp.pad(
            raw_tau, ((0, pad_b), (0, 0), (0, 0), (0, 0)))

        x_groups = x_pad.reshape(group_count, mg, S, D)
        h_groups = h_pad.reshape(group_count, mg, S, R, h.shape[-1])
        raw_tau_groups = raw_tau_pad.reshape(group_count, mg, S, R, 1)

        page_key_bf = page_keys.astype(jnp.bfloat16)
        h_unit = _forward_unit_direction(
            h_groups.astype(jnp.float32)).astype(jnp.bfloat16)

        if _score_mode == 'maxmean':
            evidence = jnp.einsum(
                'gmsrd,pd->gmsrp', h_unit, page_key_bf).astype(jnp.float32)
            evidence = evidence.max(axis=3)
            page_scores = (
                evidence.mean(axis=(1, 2)) + evidence.max(axis=(1, 2))
            ) * jnp.float32(0.5)
        else:
            route_summary = _forward_unit_direction(
                h_groups.astype(jnp.float32).mean(axis=(1, 2, 3)))
            page_scores = (
                route_summary.astype(jnp.bfloat16) @ page_key_bf.T
            ).astype(jnp.float32)
        page_scores = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30))

        protected = jnp.zeros((group_count, page_count), dtype=jnp.bool_)
        if _fallback_pages:
            fallback_ids = jnp.arange(_fallback_pages, dtype=jnp.int32)
            fallback_ids = jnp.broadcast_to(
                fallback_ids[None, :], (group_count, _fallback_pages))
            protected = protected.at[:, :_fallback_pages].set(True)
        else:
            fallback_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)
        if _random_pages:
            random_span = page_count - _fallback_pages
            random_offsets = (
                jnp.arange(group_count, dtype=jnp.int32)[:, None]
                * jnp.int32(1103515245)
                + jnp.arange(_random_pages, dtype=jnp.int32)[None, :]
                * jnp.int32(12345)
                + jnp.int32(17))
            random_ids = _fallback_pages + jnp.mod(
                random_offsets, jnp.int32(random_span))
            protected = protected.at[
                jnp.arange(group_count, dtype=jnp.int32)[:, None],
                random_ids].set(True)
        else:
            random_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)

        top_scores = jnp.where(protected, jnp.float32(-1.0e30), page_scores)
        _, top_ids = jax.lax.top_k(top_scores, _page_capacity)
        selected_page_ids = _sorted_selected_page_ids(
            fallback_ids, top_ids, random_ids)
        cand_valid = valid_pages[selected_page_ids].reshape(group_count, K)
        cand_valid_f = cand_valid.astype(jnp.float32)

        cand_key = op_key_pages_f[selected_page_ids].reshape(
            group_count, K, op_key_pages_f.shape[-1])
        cand_read = read_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_write = write_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_key = cand_key.astype(jnp.bfloat16)

        rho_raw = jnp.einsum(
            'gmsrd,gkd->gmsrk', h_unit, cand_key).astype(jnp.float32)
        cand_valid_b = cand_valid[:, None, None, None, :]
        rho_for_gate = jnp.where(cand_valid_b, rho_raw, jnp.float32(-1.0))
        tau = _tau_from_param(raw_tau_groups)
        _, admission, _, execution_weight, _ = _compute_admission_drive(
            rho_for_gate, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=_soft_gate_effective_active_eps,
            execution_prune_eps=execution_prune_eps)
        admission = admission * cand_valid_f[:, None, None, None, :]
        execution_weight = (
            execution_weight * cand_valid_f[:, None, None, None, :])
        xr = jnp.einsum(
            'gmsd,gkd->gmsk', x_groups.astype(jnp.bfloat16), cand_read)
        a = execution_weight * xr.astype(jnp.float32)[:, :, :, None, :]
        raw_out = jnp.einsum(
            'gmsrk,gkd->gmsrd', a.astype(jnp.bfloat16), cand_write
        ).astype(jnp.float32)
        den_cost = admission.sum(axis=-1, keepdims=True)

        raw_out = raw_out.reshape(B_pad, S, R, D)[:B]
        den_cost = den_cost.reshape(B_pad, S, R, 1)[:B]
        global_den_cost = jax.lax.psum(den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')
        return out.astype(jnp.float32)

    return fused_gate_srw_paired_page_minimal


def make_sharded_srw_page(mesh, max_chunk_size=2048, analysis=False,
                          dead_exposure_target=0.1,
                          soft_gate_effective_active_eps=1.0e-6,
                          admission_den_power=1.0,
                          admission_den_grad_scale=1.0,
                          operator_page_size=128,
                          operator_page_capacity=32,
                          operator_page_microgroup_sequences=2,
                          operator_page_score_mode='maxmean',
                          operator_page_fallback_pages=0,
                          operator_page_random_pages=0):
    """Page-sparse single-route SRW with v4166-compatible diagnostics."""
    del max_chunk_size, dead_exposure_target
    _model_axis_size = mesh.shape['model']
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    _page_size = int(operator_page_size)
    _page_capacity = int(operator_page_capacity)
    _microgroup_sequences = max(1, int(operator_page_microgroup_sequences))
    _score_mode = _normalize_operator_page_score_mode(
        operator_page_score_mode)
    _fallback_pages = int(operator_page_fallback_pages)
    _random_pages = int(operator_page_random_pages)
    _angular_strong_margin = jnp.float32(0.05)

    _slim_out_specs = (
        P('data', None, None), P('data', None, None),
        P('data', None, None), P(), P(), P(), P(),
        P('data', None, None), P('data', None, None),
        P(), P(), P(), P(), P(), P(), P(), P(), P(),
        P('data', None, None), P('data', None, None),
    )
    _conc_out_specs = (P(), P(), P(), P())
    _analysis_extra_specs = (
        P('data', None, None), P(), P(), P(), P(), P(),
        P(), P(), P(), P(), P(),
    )
    _select_diag_specs = tuple(P() for _ in range(SELECT_DIAG_COUNT))
    _dead_exposure_diag_specs = tuple(
        P() for _ in range(DEAD_EXPOSURE_DIAG_COUNT))
    _out_specs = (_slim_out_specs + _conc_out_specs
                  + (_analysis_extra_specs if analysis else ())
                  + _select_diag_specs + _dead_exposure_diag_specs
                  + (P(), P()))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       OPERATOR_PAGE_TABLE_SPECS,
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw_page(x, h, page_tables, raw_tau,
                            read_local, write_local,
                            soft_gate_temperature, soft_gate_t_final,
                            soft_gate_boundary_power,
                            soft_gate_boundary_power_final,
                            execution_prune_eps):
        del read_local, write_local
        (op_key_pages_f, read_pages_f, write_pages_f, valid_pages,
         valid_page_mask, page_keys) = _unpack_operator_page_tables(
            page_tables)
        page_count = op_key_pages_f.shape[0]
        effective_pages = _validate_operator_page_table_args(
            'single-route', page_count, _page_size, _page_capacity,
            _fallback_pages, _random_pages)
        K = effective_pages * _page_size
        N_total = jax.lax.psum(
            valid_pages.astype(jnp.float32).sum(), 'model')
        N_total = jnp.maximum(N_total, jnp.float32(1.0))
        K_total = K * _model_axis_size

        B, S, D = x.shape
        mg = min(_microgroup_sequences, B)
        group_count = (B + mg - 1) // mg
        B_pad = group_count * mg
        pad_b = B_pad - B
        x_pad = jnp.pad(x, ((0, pad_b), (0, 0), (0, 0)))
        h_pad = jnp.pad(h, ((0, pad_b), (0, 0), (0, 0)))
        raw_tau_pad = jnp.pad(raw_tau, ((0, pad_b), (0, 0), (0, 0)))

        x_groups = x_pad.reshape(group_count, mg, S, D)
        h_groups = h_pad.reshape(group_count, mg, S, h.shape[-1])
        raw_tau_groups = raw_tau_pad.reshape(group_count, mg, S, 1)

        page_key_bf = page_keys.astype(jnp.bfloat16)
        h_unit = _forward_unit_direction(
            h_groups.astype(jnp.float32)).astype(jnp.bfloat16)

        if _score_mode == 'maxmean':
            evidence = jnp.einsum(
                'gmsd,pd->gmsp', h_unit, page_key_bf).astype(jnp.float32)
            page_scores = (
                evidence.mean(axis=(1, 2)) + evidence.max(axis=(1, 2))
            ) * jnp.float32(0.5)
        else:
            route_summary = _forward_unit_direction(
                h_groups.astype(jnp.float32).mean(axis=(1, 2)))
            page_scores = (
                route_summary.astype(jnp.bfloat16) @ page_key_bf.T
            ).astype(jnp.float32)
        page_scores = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30))

        protected = jnp.zeros((group_count, page_count), dtype=jnp.bool_)
        if _fallback_pages:
            fallback_ids = jnp.arange(_fallback_pages, dtype=jnp.int32)
            fallback_ids = jnp.broadcast_to(
                fallback_ids[None, :], (group_count, _fallback_pages))
            protected = protected.at[:, :_fallback_pages].set(True)
        else:
            fallback_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)
        if _random_pages:
            random_span = page_count - _fallback_pages
            random_offsets = (
                jnp.arange(group_count, dtype=jnp.int32)[:, None]
                * jnp.int32(1103515245)
                + jnp.arange(_random_pages, dtype=jnp.int32)[None, :]
                * jnp.int32(12345)
                + jnp.int32(17))
            random_ids = _fallback_pages + jnp.mod(
                random_offsets, jnp.int32(random_span))
            protected = protected.at[
                jnp.arange(group_count, dtype=jnp.int32)[:, None],
                random_ids].set(True)
        else:
            random_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)
        top_scores = jnp.where(protected, jnp.float32(-1.0e30), page_scores)
        _, top_ids = jax.lax.top_k(top_scores, _page_capacity)
        selected_page_ids = _sorted_selected_page_ids(
            fallback_ids, top_ids, random_ids)
        cand_valid = valid_pages[selected_page_ids].reshape(group_count, K)
        cand_valid_f = cand_valid.astype(jnp.float32)
        candidate_valid_ops = jax.lax.stop_gradient(
            jax.lax.psum(cand_valid_f.sum(axis=-1), 'model').mean())

        cand_key = op_key_pages_f[selected_page_ids].reshape(
            group_count, K, op_key_pages_f.shape[-1])
        cand_read = read_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_write = write_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_key = cand_key.astype(jnp.bfloat16)

        rho_raw = jnp.einsum(
            'gmsd,gkd->gmsk', h_unit, cand_key).astype(jnp.float32)
        cand_valid_b = cand_valid[:, None, None, :]
        rho_for_gate = jnp.where(cand_valid_b, rho_raw, jnp.float32(-1.0))
        rho = jnp.where(cand_valid_b, rho_raw, jnp.float32(0.0))
        tau = _tau_from_param(raw_tau_groups)
        (selection_margin, admission, drive, execution_weight,
         active_mask) = _compute_admission_drive(
            rho_for_gate, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=_soft_gate_effective_active_eps,
            execution_prune_eps=execution_prune_eps)
        selection_margin = jnp.where(
            cand_valid_b, selection_margin, jnp.float32(-1.0e9))
        drive = drive * cand_valid_f[:, None, None, :]
        admission = admission * cand_valid_f[:, None, None, :]
        execution_weight = execution_weight * cand_valid_f[:, None, None, :]
        active_mask = active_mask & cand_valid_b
        strong_mask = (selection_margin > _angular_strong_margin) & cand_valid_b
        xr = jnp.einsum(
            'gmsd,gkd->gmsk', x_groups.astype(jnp.bfloat16), cand_read)
        raw_out = jnp.einsum(
            'gmsk,gkd->gmsd',
            (execution_weight * xr.astype(jnp.float32)).astype(jnp.bfloat16),
            cand_write).astype(jnp.float32)

        den_cost = admission.sum(axis=-1, keepdims=True)
        weighted_cost = execution_weight.sum(axis=-1, keepdims=True)
        gate_sq = jnp.square(execution_weight).sum(axis=-1, keepdims=True)
        gate_max = execution_weight.max(axis=-1, keepdims=True)
        active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
        strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
        positive_margin = jax.nn.relu(selection_margin)
        positive_margin_sum = positive_margin.sum(axis=-1, keepdims=True)
        selection_margin_sum = jnp.where(
            cand_valid_b, selection_margin, jnp.float32(0.0)).sum()
        selected = (
            (selection_margin > 0.0) & cand_valid_b).astype(jnp.float32)
        selected_count = selected.sum(axis=-1, keepdims=True)
        rho_sum = rho.sum(axis=-1, keepdims=True)
        rho_sq_sum = jnp.square(rho).sum(axis=-1, keepdims=True)
        total_rho_max = jnp.where(
            cand_valid_b, rho_raw, jnp.float32(-1.0)).max()
        total_positive_margin_max = positive_margin.max()
        selection_margin_max = selection_margin.max(axis=-1, keepdims=True)
        edge_margin_stat = jnp.square(positive_margin).sum()
        int_max = drive.max()
        current_cost = drive.sum(axis=-1, keepdims=True) / jnp.maximum(
            cand_valid_f.sum(axis=-1)[:, None, None, None],
            jnp.float32(1.0))

        def _unpad(t, *shape_tail):
            return t.reshape((B_pad, S) + shape_tail)[:B]

        selection_margin_unpad = _unpad(selection_margin, K)
        admission_unpad = _unpad(admission, K)
        execution_weight_unpad = _unpad(execution_weight, K)
        valid_unpad = jnp.broadcast_to(
            cand_valid[:, None, None, :],
            (group_count, mg, S, K)).reshape(B_pad, S, K)[:B]

        raw_out = raw_out.reshape(B_pad, S, D)[:B]
        den_cost = _unpad(den_cost, 1)
        weighted_cost = _unpad(weighted_cost, 1)
        gate_sq = _unpad(gate_sq, 1)
        gate_max = _unpad(gate_max, 1)
        active = _unpad(active, 1)
        strong = _unpad(strong, 1)
        positive_margin_sum = _unpad(positive_margin_sum, 1)
        selected_count = _unpad(selected_count, 1)
        rho_sum = _unpad(rho_sum, 1)
        rho_sq_sum = _unpad(rho_sq_sum, 1)
        selection_margin_max = _unpad(selection_margin_max, 1)

        global_den_cost = jax.lax.psum(den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_weighted = jax.lax.psum(weighted_cost, 'model')
        global_gate_sq = jax.lax.psum(gate_sq, 'model')
        global_gate_max = jax.lax.pmax(
            jax.lax.stop_gradient(gate_max), 'model')
        global_active = jax.lax.psum(active, 'model')
        global_strong = jax.lax.psum(strong, 'model')
        global_positive_margin_sum = jax.lax.psum(
            positive_margin_sum, 'model')
        global_selected = jax.lax.psum(selected_count, 'model')
        global_rho_sum = jax.lax.psum(rho_sum, 'model')
        global_rho_sq_sum = jax.lax.psum(rho_sq_sum, 'model')
        global_selection_margin_max = jax.lax.pmax(
            jax.lax.stop_gradient(selection_margin_max), 'model')
        no_active_direct = jax.lax.stop_gradient(
            (global_selection_margin_max <= 0.0).astype(jnp.float32))
        tau_direct = _block_tau_up_when_no_active(
            _tau_from_param(raw_tau), no_active_direct.astype(jnp.bool_))

        candidate_valid_ops_safe = jnp.maximum(
            candidate_valid_ops, jnp.float32(1.0))
        rho_count = jnp.float32(B * S) * candidate_valid_ops_safe
        token_count = jnp.float32(B * S)
        rho_mean_token = global_rho_sum / candidate_valid_ops_safe
        rho_var_token = jnp.maximum(
            global_rho_sq_sum / candidate_valid_ops_safe
            - jnp.square(rho_mean_token),
            0.0)
        rho_std_token = jnp.sqrt(rho_var_token + 1e-8)
        active_frac = jax.lax.stop_gradient(
            global_active / candidate_valid_ops_safe)
        strong_frac = jax.lax.stop_gradient(
            global_strong / candidate_valid_ops_safe)
        positive_margin_mean_active = (
            jax.lax.stop_gradient(global_positive_margin_sum)
            / (jax.lax.stop_gradient(global_active) + 1e-8))
        gate_eff_n = (
            jnp.square(jax.lax.stop_gradient(global_weighted))
            / (jax.lax.stop_gradient(global_gate_sq) + 1e-8))
        gate_eff_ratio = gate_eff_n / jnp.maximum(
            jax.lax.stop_gradient(global_active), 1.0)
        top1_gate_frac = global_gate_max / jnp.maximum(
            jax.lax.stop_gradient(global_weighted), 1e-8)
        tau_sg = jax.lax.stop_gradient(_tau_from_param(raw_tau))
        raw_tau_sg = jax.lax.stop_gradient(raw_tau)
        selection_margin_mean = (
            jax.lax.psum(selection_margin_sum, 'model') / rho_count)
        positive_margin_mean = (
            jax.lax.psum(positive_margin.sum(), 'model') / rho_count)
        edge_margin_out = (
            jax.lax.psum(edge_margin_stat, 'model') / rho_count)
        int_max_out = jax.lax.pmax(jax.lax.stop_gradient(int_max), 'model')
        current_cost_mean = jax.lax.stop_gradient(
            jax.lax.psum(current_cost.reshape(B_pad, S, 1)[:B], 'model')).mean()

        probs = jax.nn.softmax(page_scores, axis=-1)
        page_entropy = -jnp.sum(
            probs * jnp.log(jnp.maximum(probs, 1e-30)), axis=-1).mean()
        page_top1 = probs.max(axis=-1).mean()
        valid_page_f = valid_page_mask.astype(jnp.float32)
        valid_page_den = jnp.maximum(valid_page_f.sum(), jnp.float32(1.0))
        score_sum = (page_scores * valid_page_f[None, :]).sum(axis=-1)
        page_score_mean_per_group = score_sum / valid_page_den
        page_score_delta = jnp.where(
            valid_page_mask[None, :],
            page_scores - page_score_mean_per_group[:, None],
            jnp.float32(0.0))
        page_score_var_per_group = (
            jnp.square(page_score_delta).sum(axis=-1) / valid_page_den)
        page_score_std = jnp.sqrt(page_score_var_per_group + 1.0e-8).mean()
        page_score_max = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30)
        ).max()
        page_score_mean = page_score_mean_per_group.mean()
        candidate_ops = jnp.float32(K_total)
        candidate_valid_frac = candidate_valid_ops / N_total
        candidate_valid_group = jax.lax.psum(
            cand_valid_f.sum(axis=-1), 'model')
        page_no_route_frac = (
            candidate_valid_group <= 0.0).astype(jnp.float32).mean()
        page_diag = jnp.zeros((PAGE_DIAG_COUNT,), dtype=jnp.float32)
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_size']].set(
            jnp.float32(_page_size))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_capacity']].set(
            jnp.float32(_page_capacity))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_count_total']].set(
            jnp.float32(page_count * _model_axis_size))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_count_effective']].set(
            jnp.float32(effective_pages))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_top1_frac']].set(
            jax.lax.stop_gradient(page_top1))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_entropy']].set(
            jax.lax.stop_gradient(page_entropy))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_score_max']].set(
            jax.lax.stop_gradient(page_score_max))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_score_mean']].set(
            jax.lax.stop_gradient(page_score_mean))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_score_std']].set(
            jax.lax.stop_gradient(page_score_std))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_ops']].set(
            candidate_ops)
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_valid_ops']].set(
            jax.lax.stop_gradient(candidate_valid_ops))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_frac']].set(
            candidate_ops / N_total)
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_valid_frac']].set(
            jax.lax.stop_gradient(candidate_valid_frac))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_den_mean']].set(
            jax.lax.stop_gradient(global_den_cost.mean()))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['candidate_execution_mass']].set(
                jax.lax.stop_gradient(global_weighted.mean()))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['page_fallback_used_frac']].set(
                jnp.float32(_fallback_pages) / jnp.float32(effective_pages))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['page_random_used_frac']].set(
                jnp.float32(_random_pages) / jnp.float32(effective_pages))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_no_route_frac']].set(
            jax.lax.stop_gradient(page_no_route_frac))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['selected_page_count']].set(
            jnp.float32(effective_pages))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['estimated_compute_frac_page']].set(
                candidate_ops / N_total)

        slim_out = (
            out.astype(jnp.float32),
            active_frac,
            global_gate_max,
            jnp.float32(0.0),
            jax.lax.stop_gradient(rho_std_token.mean()),
            jax.lax.stop_gradient(global_weighted).mean(),
            jax.lax.stop_gradient(global_active).mean(),
            strong_frac,
            positive_margin_mean_active,
            tau_sg.mean(),
            jnp.float32(0.0),
            jnp.float32(0.0),
            int_max_out,
            jax.lax.stop_gradient(global_den_cost).mean(),
            jnp.float32(0.0),
            current_cost_mean,
            jnp.float32(0.0),
            edge_margin_out,
            tau_direct.astype(jnp.float32),
            no_active_direct,
        )
        conc_out = (
            gate_eff_n.mean(),
            gate_eff_ratio.mean(),
            top1_gate_frac.mean(),
            top1_gate_frac.max(),
        )
        select_diag = (
            jax.lax.stop_gradient(rho_mean_token.mean()),
            jax.lax.stop_gradient(rho_std_token.mean()),
            jax.lax.pmax(jax.lax.stop_gradient(total_rho_max), 'model'),
            tau_sg.mean(), tau_sg.min(), tau_sg.max(),
            raw_tau_sg.mean(), raw_tau_sg.min(), raw_tau_sg.max(),
            jax.lax.stop_gradient(selection_margin_mean),
            jax.lax.stop_gradient(positive_margin_mean),
            jax.lax.pmax(
            jax.lax.stop_gradient(total_positive_margin_max), 'model'),
            jax.lax.stop_gradient(
                global_selected.mean() / candidate_valid_ops_safe),
            jax.lax.stop_gradient(no_active_direct.mean()),
        )
        exposure_diag = tuple(
            jnp.float32(0.0) for _ in range(DEAD_EXPOSURE_DIAG_COUNT))
        sparsity_diag = _page_gate_sparsity_diag(
            selection_margin_unpad,
            admission_unpad,
            execution_weight_unpad,
            valid_unpad,
            jnp.float32(B * S),
            soft_gate_t_final,
            soft_gate_boundary_power_final)

        if not analysis:
            return (slim_out + conc_out + select_diag
                    + exposure_diag + (sparsity_diag, page_diag))

        margin_band = active_frac
        entropy_per_token = -(
            jax.lax.stop_gradient(global_weighted)
            * jnp.log(jnp.maximum(jax.lax.stop_gradient(global_weighted), 1e-30))
        ) / jnp.maximum(jax.lax.stop_gradient(global_weighted), 1e-8)
        analysis_out = (
            margin_band,
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jax.lax.stop_gradient(global_active).std(),
            jax.lax.stop_gradient(entropy_per_token.mean()),
            jax.lax.stop_gradient(global_den_cost).mean(),
            jnp.float32(0.0),
            current_cost_mean,
            jnp.float32(0.0),
            jnp.float32(0.0),
        )
        return (slim_out + conc_out + analysis_out + select_diag
                + exposure_diag + (sparsity_diag, page_diag))

    return fused_gate_srw_page


def make_sharded_srw_paired_page(mesh, max_chunk_size=2048, analysis=False,
                                 dead_exposure_target=0.1,
                                 soft_gate_effective_active_eps=1.0e-6,
                                 admission_den_power=1.0,
                                 admission_den_grad_scale=1.0,
                                 operator_page_size=128,
                                 operator_page_capacity=8,
                                 operator_page_microgroup_sequences=2,
                                 operator_page_score_mode='maxmean',
                                 operator_page_fallback_pages=0,
                                 operator_page_random_pages=0):
    """Page-sparse paired Q/K SRW with v4166-compatible diagnostics."""
    del max_chunk_size, dead_exposure_target
    _model_axis_size = mesh.shape['model']
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    _page_size = int(operator_page_size)
    _page_capacity = int(operator_page_capacity)
    _microgroup_sequences = max(1, int(operator_page_microgroup_sequences))
    _score_mode = _normalize_operator_page_score_mode(
        operator_page_score_mode)
    _fallback_pages = int(operator_page_fallback_pages)
    _random_pages = int(operator_page_random_pages)
    _angular_strong_margin = jnp.float32(0.05)

    _slim_out_specs = (
        P('data', None, None, None),  # out [B,S,2,D]
        P('data', None, None),        # active [B,S,1]
        P('data', None, None),        # gate_max [B,S,1]
        P(), P(), P(), P(),
        P('data', None, None),        # strong [B,S,1]
        P('data', None, None),        # positive-margin mean [B,S,1]
        P(), P(), P(), P(), P(), P(), P(), P(), P(),
        P('data', None, None, None),  # tau_direct [B,S,2,1]
        P('data', None, None, None),  # no_active_direct [B,S,2,1]
    )
    _conc_out_specs = (P(), P(), P(), P())
    _route_split_out_specs = (P(), P(), P(), P(), P(), P())
    _analysis_extra_specs = (
        P('data', None, None), P(), P(), P(), P(), P(),
        P(), P(), P(), P(), P(),
    )
    _select_diag_specs = tuple(P() for _ in range(SELECT_DIAG_COUNT))
    _dead_exposure_diag_specs = tuple(
        P() for _ in range(DEAD_EXPOSURE_DIAG_COUNT))
    _out_specs = (
        _slim_out_specs + _conc_out_specs + _route_split_out_specs
        + (_analysis_extra_specs if analysis else ())
        + _select_diag_specs + _dead_exposure_diag_specs + (P(), P()))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None, None),
                       OPERATOR_PAGE_TABLE_SPECS,
                       P('data', None, None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw_paired_page(x, h, page_tables, raw_tau,
                                   read_local, write_local,
                                   soft_gate_temperature,
                                   soft_gate_t_final,
                                   soft_gate_boundary_power,
                                   soft_gate_boundary_power_final,
                                   execution_prune_eps):
        del read_local, write_local
        (op_key_pages_f, read_pages_f, write_pages_f, valid_pages,
         valid_page_mask, page_keys) = _unpack_operator_page_tables(
            page_tables)
        page_count = op_key_pages_f.shape[0]
        effective_pages = _validate_operator_page_table_args(
            'qk-paired', page_count, _page_size, _page_capacity,
            _fallback_pages, _random_pages)
        K = effective_pages * _page_size
        N_total = jax.lax.psum(
            valid_pages.astype(jnp.float32).sum(), 'model')
        N_total = jnp.maximum(N_total, jnp.float32(1.0))
        K_total = K * _model_axis_size

        B, S, R, _ = h.shape
        D = x.shape[-1]
        mg = min(_microgroup_sequences, B)
        group_count = (B + mg - 1) // mg
        B_pad = group_count * mg
        pad_b = B_pad - B
        x_pad = jnp.pad(x, ((0, pad_b), (0, 0), (0, 0)))
        h_pad = jnp.pad(h, ((0, pad_b), (0, 0), (0, 0), (0, 0)))
        raw_tau_pad = jnp.pad(
            raw_tau, ((0, pad_b), (0, 0), (0, 0), (0, 0)))

        x_groups = x_pad.reshape(group_count, mg, S, D)
        h_groups = h_pad.reshape(group_count, mg, S, R, h.shape[-1])
        raw_tau_groups = raw_tau_pad.reshape(group_count, mg, S, R, 1)

        page_key_bf = page_keys.astype(jnp.bfloat16)
        h_unit = _forward_unit_direction(
            h_groups.astype(jnp.float32)).astype(jnp.bfloat16)

        if _score_mode == 'maxmean':
            evidence = jnp.einsum(
                'gmsrd,pd->gmsrp', h_unit, page_key_bf).astype(jnp.float32)
            evidence = evidence.max(axis=3)
            page_scores = (
                evidence.mean(axis=(1, 2)) + evidence.max(axis=(1, 2))
            ) * jnp.float32(0.5)
        else:
            route_summary = _forward_unit_direction(
                h_groups.astype(jnp.float32).mean(axis=(1, 2, 3)))
            page_scores = (
                route_summary.astype(jnp.bfloat16) @ page_key_bf.T
            ).astype(jnp.float32)
        page_scores = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30))

        protected = jnp.zeros((group_count, page_count), dtype=jnp.bool_)
        if _fallback_pages:
            fallback_ids = jnp.arange(_fallback_pages, dtype=jnp.int32)
            fallback_ids = jnp.broadcast_to(
                fallback_ids[None, :], (group_count, _fallback_pages))
            protected = protected.at[:, :_fallback_pages].set(True)
        else:
            fallback_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)
        if _random_pages:
            random_span = page_count - _fallback_pages
            random_offsets = (
                jnp.arange(group_count, dtype=jnp.int32)[:, None]
                * jnp.int32(1103515245)
                + jnp.arange(_random_pages, dtype=jnp.int32)[None, :]
                * jnp.int32(12345)
                + jnp.int32(17))
            random_ids = _fallback_pages + jnp.mod(
                random_offsets, jnp.int32(random_span))
            protected = protected.at[
                jnp.arange(group_count, dtype=jnp.int32)[:, None],
                random_ids].set(True)
        else:
            random_ids = jnp.zeros((group_count, 0), dtype=jnp.int32)

        top_scores = jnp.where(protected, jnp.float32(-1.0e30), page_scores)
        _, top_ids = jax.lax.top_k(top_scores, _page_capacity)
        selected_page_ids = _sorted_selected_page_ids(
            fallback_ids, top_ids, random_ids)
        cand_valid = valid_pages[selected_page_ids].reshape(group_count, K)
        cand_valid_f = cand_valid.astype(jnp.float32)
        candidate_valid_ops = jax.lax.stop_gradient(
            jax.lax.psum(cand_valid_f.sum(axis=-1), 'model').mean())

        cand_key = op_key_pages_f[selected_page_ids].reshape(
            group_count, K, op_key_pages_f.shape[-1])
        cand_read = read_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_write = write_pages_f[selected_page_ids].reshape(
            group_count, K, D).astype(jnp.bfloat16)
        cand_key = cand_key.astype(jnp.bfloat16)

        rho_raw = jnp.einsum(
            'gmsrd,gkd->gmsrk', h_unit, cand_key).astype(jnp.float32)
        cand_valid_b = cand_valid[:, None, None, None, :]
        rho_for_gate = jnp.where(cand_valid_b, rho_raw, jnp.float32(-1.0))
        rho = jnp.where(cand_valid_b, rho_raw, jnp.float32(0.0))
        tau = _tau_from_param(raw_tau_groups)
        (selection_margin, admission, drive, execution_weight,
         active_mask) = _compute_admission_drive(
            rho_for_gate, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=_soft_gate_effective_active_eps,
            execution_prune_eps=execution_prune_eps)
        selection_margin = jnp.where(
            cand_valid_b, selection_margin, jnp.float32(-1.0e9))
        drive = drive * cand_valid_f[:, None, None, None, :]
        admission = admission * cand_valid_f[:, None, None, None, :]
        execution_weight = (
            execution_weight * cand_valid_f[:, None, None, None, :])
        active_mask = active_mask & cand_valid_b
        strong_mask = (selection_margin > _angular_strong_margin) & cand_valid_b

        xr = jnp.einsum(
            'gmsd,gkd->gmsk', x_groups.astype(jnp.bfloat16), cand_read)
        raw_out = jnp.einsum(
            'gmsrk,gkd->gmsrd',
            (execution_weight * xr.astype(jnp.float32)[:, :, :, None, :]
             ).astype(jnp.bfloat16),
            cand_write).astype(jnp.float32)

        den_cost = admission.sum(axis=-1, keepdims=True)
        weighted_cost = execution_weight.sum(axis=-1, keepdims=True)
        gate_sq = jnp.square(execution_weight).sum(axis=-1, keepdims=True)
        gate_max = execution_weight.max(axis=-1, keepdims=True)
        active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
        strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
        positive_margin = jax.nn.relu(selection_margin)
        positive_margin_sum = positive_margin.sum(axis=-1, keepdims=True)
        selection_margin_sum = jnp.where(
            cand_valid_b, selection_margin, jnp.float32(0.0)).sum()
        selected = (
            (selection_margin > 0.0) & cand_valid_b).astype(jnp.float32)
        selected_count = selected.sum(axis=-1, keepdims=True)
        rho_sum = rho.sum(axis=-1, keepdims=True)
        rho_sq_sum = jnp.square(rho).sum(axis=-1, keepdims=True)
        total_rho_max = jnp.where(
            cand_valid_b, rho_raw, jnp.float32(-1.0)).max()
        total_positive_margin_max = positive_margin.max()
        selection_margin_max = selection_margin.max(axis=-1, keepdims=True)
        edge_margin_stat = jnp.square(positive_margin).sum()
        int_max = drive.max()
        current_cost = drive.sum(axis=-1, keepdims=True) / jnp.maximum(
            cand_valid_f.sum(axis=-1)[:, None, None, None, None],
            jnp.float32(1.0))

        def _unpad(t, *shape_tail):
            return t.reshape((B_pad, S, R) + shape_tail)[:B]

        selection_margin_unpad = _unpad(selection_margin, K)
        admission_unpad = _unpad(admission, K)
        execution_weight_unpad = _unpad(execution_weight, K)
        valid_unpad = jnp.broadcast_to(
            cand_valid[:, None, None, None, :],
            (group_count, mg, S, R, K)).reshape(B_pad, S, R, K)[:B]

        raw_out = raw_out.reshape(B_pad, S, R, D)[:B]
        den_cost = _unpad(den_cost, 1)
        weighted_cost = _unpad(weighted_cost, 1)
        gate_sq = _unpad(gate_sq, 1)
        gate_max = _unpad(gate_max, 1)
        active = _unpad(active, 1)
        strong = _unpad(strong, 1)
        positive_margin_sum = _unpad(positive_margin_sum, 1)
        selected_count = _unpad(selected_count, 1)
        rho_sum = _unpad(rho_sum, 1)
        rho_sq_sum = _unpad(rho_sq_sum, 1)
        selection_margin_max = _unpad(selection_margin_max, 1)
        current_cost = _unpad(current_cost, 1)

        global_den_cost = jax.lax.psum(den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = raw_out / admission_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_weighted = jax.lax.psum(weighted_cost, 'model')
        global_gate_sq = jax.lax.psum(gate_sq, 'model')
        global_gate_max = jax.lax.pmax(
            jax.lax.stop_gradient(gate_max), 'model')
        global_active = jax.lax.psum(active, 'model')
        global_strong = jax.lax.psum(strong, 'model')
        global_positive_margin_sum = jax.lax.psum(
            positive_margin_sum, 'model')
        global_selected = jax.lax.psum(selected_count, 'model')
        global_rho_sum = jax.lax.psum(rho_sum, 'model')
        global_rho_sq_sum = jax.lax.psum(rho_sq_sum, 'model')
        global_selection_margin_max = jax.lax.pmax(
            jax.lax.stop_gradient(selection_margin_max), 'model')
        no_active_direct = jax.lax.stop_gradient(
            (global_selection_margin_max <= 0.0).astype(jnp.float32))
        tau_direct = _block_tau_up_when_no_active(
            _tau_from_param(raw_tau), no_active_direct.astype(jnp.bool_))

        candidate_valid_ops_safe = jnp.maximum(
            candidate_valid_ops, jnp.float32(1.0))
        rho_count = jnp.float32(B * S * R) * candidate_valid_ops_safe
        rho_mean_token = global_rho_sum / candidate_valid_ops_safe
        rho_var_token = jnp.maximum(
            global_rho_sq_sum / candidate_valid_ops_safe
            - jnp.square(rho_mean_token),
            0.0)
        rho_std_token = jnp.sqrt(rho_var_token + 1e-8)
        active_frac = jax.lax.stop_gradient(
            global_active / candidate_valid_ops_safe)
        strong_frac = jax.lax.stop_gradient(
            global_strong / candidate_valid_ops_safe)
        positive_margin_mean_active = (
            jax.lax.stop_gradient(global_positive_margin_sum)
            / (jax.lax.stop_gradient(global_active) + 1e-8))
        gate_eff_n = (
            jnp.square(jax.lax.stop_gradient(global_weighted))
            / (jax.lax.stop_gradient(global_gate_sq) + 1e-8))
        gate_eff_ratio = gate_eff_n / jnp.maximum(
            jax.lax.stop_gradient(global_active), 1.0)
        top1_gate_frac = global_gate_max / jnp.maximum(
            jax.lax.stop_gradient(global_weighted), 1e-8)
        tau_sg = jax.lax.stop_gradient(_tau_from_param(raw_tau))
        raw_tau_sg = jax.lax.stop_gradient(raw_tau)
        selection_margin_mean = (
            jax.lax.psum(selection_margin_sum, 'model') / rho_count)
        positive_margin_mean = (
            jax.lax.psum(positive_margin.sum(), 'model') / rho_count)
        edge_margin_out = (
            jax.lax.psum(edge_margin_stat, 'model') / rho_count)
        int_max_out = jax.lax.pmax(jax.lax.stop_gradient(int_max), 'model')
        current_cost_mean = jax.lax.stop_gradient(
            jax.lax.psum(current_cost, 'model')).mean()

        probs = jax.nn.softmax(page_scores, axis=-1)
        page_entropy = -jnp.sum(
            probs * jnp.log(jnp.maximum(probs, 1e-30)), axis=-1).mean()
        page_top1 = probs.max(axis=-1).mean()
        valid_page_f = valid_page_mask.astype(jnp.float32)
        valid_page_den = jnp.maximum(valid_page_f.sum(), jnp.float32(1.0))
        score_sum = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(0.0)
        ).sum(axis=-1)
        page_score_mean_per_group = score_sum / valid_page_den
        page_score_delta = jnp.where(
            valid_page_mask[None, :],
            page_scores - page_score_mean_per_group[:, None],
            jnp.float32(0.0))
        page_score_std = jnp.sqrt(
            jnp.square(page_score_delta).sum(axis=-1)
            / valid_page_den + 1.0e-8).mean()
        page_score_max = jnp.where(
            valid_page_mask[None, :], page_scores, jnp.float32(-1.0e30)
        ).max()
        page_score_mean = page_score_mean_per_group.mean()
        candidate_ops = jnp.float32(K_total)
        candidate_valid_frac = candidate_valid_ops / N_total
        candidate_valid_group = jax.lax.psum(
            cand_valid_f.sum(axis=-1), 'model')
        page_no_route_frac = (
            candidate_valid_group <= 0.0).astype(jnp.float32).mean()
        page_diag = jnp.zeros((PAGE_DIAG_COUNT,), dtype=jnp.float32)
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_size']].set(
            jnp.float32(_page_size))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_capacity']].set(
            jnp.float32(_page_capacity))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_count_total']].set(
            jnp.float32(page_count * _model_axis_size))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_count_effective']].set(
            jnp.float32(effective_pages))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_top1_frac']].set(
            jax.lax.stop_gradient(page_top1))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_entropy']].set(
            jax.lax.stop_gradient(page_entropy))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_score_max']].set(
            jax.lax.stop_gradient(page_score_max))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_score_mean']].set(
            jax.lax.stop_gradient(page_score_mean))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_score_std']].set(
            jax.lax.stop_gradient(page_score_std))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_ops']].set(
            candidate_ops)
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_valid_ops']].set(
            jax.lax.stop_gradient(candidate_valid_ops))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_frac']].set(
            candidate_ops / N_total)
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_valid_frac']].set(
            jax.lax.stop_gradient(candidate_valid_frac))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['candidate_den_mean']].set(
            jax.lax.stop_gradient(global_den_cost.mean()))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['candidate_execution_mass']].set(
                jax.lax.stop_gradient(global_weighted.mean()))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['page_fallback_used_frac']].set(
                jnp.float32(_fallback_pages) / jnp.float32(effective_pages))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['page_random_used_frac']].set(
                jnp.float32(_random_pages) / jnp.float32(effective_pages))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['page_no_route_frac']].set(
            jax.lax.stop_gradient(page_no_route_frac))
        page_diag = page_diag.at[PAGE_DIAG_INDEX['selected_page_count']].set(
            jnp.float32(effective_pages))
        page_diag = page_diag.at[
            PAGE_DIAG_INDEX['estimated_compute_frac_page']].set(
                candidate_ops / N_total)

        z = jnp.float32(0.0)
        active_mean = active_frac.mean(axis=2)
        gate_max_mean = global_gate_max.max(axis=2)
        strong_mean = strong_frac.mean(axis=2)
        positive_active_mean = positive_margin_mean_active.mean(axis=2)
        qk_den_cost_mean = jax.lax.stop_gradient(global_den_cost.mean())
        qk_execution_mass = jax.lax.stop_gradient(global_weighted.mean())
        q_active_frac = active_frac[:, :, 0, :].mean()
        k_active_frac = active_frac[:, :, 1, :].mean()
        q_strong_frac = strong_frac[:, :, 0, :].mean()
        k_strong_frac = strong_frac[:, :, 1, :].mean()
        q_active_n_mean = jax.lax.stop_gradient(
            global_active[:, :, 0, :]).mean()
        k_active_n_mean = jax.lax.stop_gradient(
            global_active[:, :, 1, :]).mean()

        slim_out = (
            out.astype(jnp.float32),
            active_mean,
            gate_max_mean,
            z,
            jax.lax.stop_gradient(rho_std_token.mean()),
            qk_execution_mass,
            jax.lax.stop_gradient(global_active).mean(),
            strong_mean,
            positive_active_mean,
            jnp.abs(tau_sg).mean(),
            z,
            z,
            int_max_out,
            qk_den_cost_mean,
            z,
            current_cost_mean,
            z,
            edge_margin_out,
            tau_direct.astype(jnp.float32),
            no_active_direct.astype(jnp.float32))
        conc_out = (
            gate_eff_n.mean(),
            gate_eff_ratio.mean(),
            top1_gate_frac.mean(),
            top1_gate_frac.max(),
        )
        route_split = (
            q_active_frac, k_active_frac,
            q_strong_frac, k_strong_frac,
            q_active_n_mean, k_active_n_mean)
        if analysis:
            entropy_per_token = -(
                jax.lax.stop_gradient(global_weighted)
                * jnp.log(jnp.maximum(
                    jax.lax.stop_gradient(global_weighted), 1e-30))
            ) / jnp.maximum(jax.lax.stop_gradient(global_weighted), 1e-8)
            analysis_out = (
                active_mean,
                z,
                z,
                z,
                jax.lax.stop_gradient(global_active).std(),
                jax.lax.stop_gradient(entropy_per_token.mean()),
                qk_den_cost_mean,
                z,
                current_cost_mean,
                z,
                z,
            )
        else:
            analysis_out = ()
        select_diag = (
            jax.lax.stop_gradient(rho_mean_token.mean()),
            jax.lax.stop_gradient(rho_std_token.mean()),
            jax.lax.pmax(jax.lax.stop_gradient(total_rho_max), 'model'),
            tau_sg.mean(), tau_sg.min(), tau_sg.max(),
            raw_tau_sg.mean(), raw_tau_sg.min(), raw_tau_sg.max(),
            jax.lax.stop_gradient(selection_margin_mean),
            jax.lax.stop_gradient(positive_margin_mean),
            jax.lax.pmax(
                jax.lax.stop_gradient(total_positive_margin_max), 'model'),
            jax.lax.stop_gradient(
                global_selected.mean() / candidate_valid_ops_safe),
            jax.lax.stop_gradient(no_active_direct.mean()),
        )
        exposure_diag = (z, z, z, z, z, z)
        sparsity_diag = jnp.stack((
            _page_gate_sparsity_diag(
                selection_margin_unpad[:, :, 0, :],
                admission_unpad[:, :, 0, :],
                execution_weight_unpad[:, :, 0, :],
                valid_unpad[:, :, 0, :],
                jnp.float32(B * S),
                soft_gate_t_final,
                soft_gate_boundary_power_final),
            _page_gate_sparsity_diag(
                selection_margin_unpad[:, :, 1, :],
                admission_unpad[:, :, 1, :],
                execution_weight_unpad[:, :, 1, :],
                valid_unpad[:, :, 1, :],
                jnp.float32(B * S),
                soft_gate_t_final,
                soft_gate_boundary_power_final),
        )).astype(jnp.float32)
        return (slim_out + conc_out + route_split + analysis_out
                + select_diag + exposure_diag + (sparsity_diag, page_diag))

    return fused_gate_srw_paired_page


# ================================================================
# 4. NeuronPool -- read/write directions + RW-derived operator keys
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    d_model: int
    d_route: int
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Checkpoint/config alias for rst pool size.

    def setup(self):
        db = self.d_route
        dm = self.d_model
        n_rst_eff = self.n_rst if self.n_rst is not None else self.n_know
        if n_rst_eff is None:
            raise ValueError("NeuronPool requires n_rst or n_know checkpoint alias.")

        # Read vectors define what each neuron extracts from x.
        # Stored vectors remain raw; SRW forward uses their directions.
        self.attn_qk_read = self.param('attn_qk_read', unit_norm_init(), (self.n_qk, dm))
        self.attn_v_read = self.param('attn_v_read', unit_norm_init(), (self.n_v, dm))
        self.rst_read = self.param('rst_read', unit_norm_init(), (n_rst_eff, dm))

        # Write vectors define the output direction for each neuron.
        # Raw parameter norms are still observable diagnostics.
        self.attn_qk_write = self.param('attn_qk_write', unit_norm_init(), (self.n_qk, dm))
        self.attn_v_write = self.param('attn_v_write', unit_norm_init(), (self.n_v, dm))
        self.rst_write = self.param('rst_write', unit_norm_init(), (n_rst_eff, dm))

        op_proj_scale = math.sqrt(float(dm) / float(db))
        op_proj_init = nn.initializers.orthogonal(scale=op_proj_scale)
        self.attn_qk_op_read_proj = self.param(
            'attn_qk_op_read_proj', op_proj_init, (dm, db))
        self.attn_qk_op_write_proj = self.param(
            'attn_qk_op_write_proj', op_proj_init, (dm, db))
        self.attn_v_op_read_proj = self.param(
            'attn_v_op_read_proj', op_proj_init, (dm, db))
        self.attn_v_op_write_proj = self.param(
            'attn_v_op_write_proj', op_proj_init, (dm, db))
        self.rst_op_read_proj = self.param(
            'rst_op_read_proj', op_proj_init, (dm, db))
        self.rst_op_write_proj = self.param(
            'rst_op_write_proj', op_proj_init, (dm, db))

        # Per-pool learnable output scale, initialized to sqrt(d_model).
        self.attn_qk_scale = self.param('attn_qk_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.attn_v_scale = self.param('attn_v_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.rst_scale = self.param('rst_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)

        # No dynamic-tau alpha params; tau comes from Router offsets + scan.


# ================================================================
# 5. Router -- route queries and cosine-space tau references
# ================================================================

class Router(nn.Module):
    d_model: int
    d_route: int
    n_qk: int
    n_v: int
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Checkpoint/config alias for rst pool size.
    router_dropout: float = 0.1
    # Constructor receives cosine-space tau values. The train driver may use
    # safe placeholders before one-time quantile calibration.
    tau_init_attn_qk: Optional[float] = None
    tau_init_attn_v: Optional[float] = None
    tau_init_rst: Optional[float] = None

    def setup(self):
        db = self.d_route
        dm = self.d_model
        n_rst_eff = self.n_rst if self.n_rst is not None else self.n_know
        if n_rst_eff is None:
            raise ValueError("Router requires n_rst or n_know checkpoint alias.")

        missing_tau = [
            name for name, value in (
                ('tau_init_attn_qk', self.tau_init_attn_qk),
                ('tau_init_attn_v', self.tau_init_attn_v),
                ('tau_init_rst', self.tau_init_rst),
            ) if value is None
        ]
        if missing_tau:
            raise ValueError(
                "v4167 requires explicit cosine-space tau_init_attn_qk/v/rst; "
                f"missing {', '.join(missing_tau)}.")
        qk_tau_init = float(self.tau_init_attn_qk)
        v_tau_init = float(self.tau_init_attn_v)
        rst_tau_init = float(self.tau_init_rst)
        raw_tau_attn_bias_init = jnp.asarray(
            [
                _raw_tau_init_from_cosine_tau(qk_tau_init),
                _raw_tau_init_from_cosine_tau(qk_tau_init),
                _raw_tau_init_from_cosine_tau(v_tau_init),
            ],
            dtype=jnp.float32)
        raw_tau_rst_bias_init = jnp.asarray(
            _raw_tau_init_from_cosine_tau(rst_tau_init), dtype=jnp.float32)
        query_proj_init = nn.initializers.orthogonal(scale=1.0)
        self.proj_attn = nn.Dense(
            db * 3,
            name='proj_attn',
            kernel_init=query_proj_init,
            bias_init=nn.initializers.zeros)
        self.proj_rst = nn.Dense(
            db,
            name='proj_rst',
            kernel_init=query_proj_init,
            bias_init=nn.initializers.zeros)
        self.q_op_write_query_proj = self.param(
            'q_op_write_query_proj', query_proj_init, (dm, db))
        self.k_op_write_query_proj = self.param(
            'k_op_write_query_proj', query_proj_init, (dm, db))
        self.v_op_write_query_proj = self.param(
            'v_op_write_query_proj', query_proj_init, (dm, db))
        self.rst_op_write_query_proj = self.param(
            'rst_op_write_query_proj', query_proj_init, (dm, db))
        self.raw_tau_attn = nn.Dense(3, name='raw_tau_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: raw_tau_attn_bias_init.astype(d))
        self.raw_tau_rst = nn.Dense(1, name='raw_tau_rst',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, raw_tau_rst_bias_init, d))


# ================================================================
# 6. Pure functions for scan body
# ================================================================

def _attn_forward_minimal(x, pool_params, router_params, expand_O_kernel, rng,
                          n_qk, n_v,
                          n_heads, d_model, n_layers,
                          router_dropout, dropout_rate, deterministic,
                          sharded_fns,
                          soft_gate_temperature=0.07,
                          soft_gate_t_final=0.07,
                          soft_gate_T_qk=None,
                          soft_gate_T_v=None,
                          soft_gate_boundary_power=2.0,
                          soft_gate_boundary_power_final=4.0,
                          admission_den_power=1.0,
                          execution_prune_eps=0.0):
    """Minimal v4167 attention path: SRW output, causal attention, O-proj."""
    del n_qk, n_v, admission_den_power
    B, S, D = x.shape
    soft_gate_T_qk = (
        soft_gate_temperature if soft_gate_T_qk is None else soft_gate_T_qk)
    soft_gate_T_v = (
        soft_gate_temperature if soft_gate_T_v is None else soft_gate_T_v)
    pool_params = _ensure_pool_operator_keys(pool_params)

    qk_op_key = pool_params['attn_qk_op_key']
    qk_read = pool_params['attn_qk_read']
    qk_write = pool_params['attn_qk_write']
    qk_page_tables = pool_params.get('attn_qk_page_tables', None)
    qk_op_arg = qk_page_tables if qk_page_tables is not None else qk_op_key
    v_op_key = pool_params['attn_v_op_key']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']
    v_page_tables = pool_params.get('attn_v_page_tables', None)
    v_op_arg = v_page_tables if v_page_tables is not None else v_op_key

    rng, rng_drop = jax.random.split(rng)
    attn_read_query = (
        x @ router_params['proj_attn']['kernel']
        + router_params['proj_attn']['bias'])
    attn_read_query = safe_dropout(
        attn_read_query, router_dropout, deterministic, rng_drop)
    q_read_query, k_read_query, v_read_query = jnp.split(
        attn_read_query, 3, axis=-1)
    h_Q = _read_write_operator_query(
        q_read_query, x, router_params['q_op_write_query_proj'])
    h_K = _read_write_operator_query(
        k_read_query, x, router_params['k_op_write_query_proj'])
    h_V = _read_write_operator_query(
        v_read_query, x, router_params['v_op_write_query_proj'])

    raw_tau_all = (
        x @ router_params['raw_tau_attn']['kernel']
        + router_params['raw_tau_attn']['bias'])
    qk_scale, v_scale, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    if isinstance(sharded_fns, dict):
        fused_paired = sharded_fns.get(
            'attn_qk_paired_minimal',
            sharded_fns.get('attn_qk_paired', sharded_fns['paired']))
        fused_single_v = sharded_fns.get(
            'attn_v_single_minimal',
            sharded_fns.get('attn_v_single', sharded_fns['single']))
    else:
        fused_single_v, fused_paired = sharded_fns

    h_QK = jnp.stack([h_Q, h_K], axis=2)
    raw_tau_QK = jnp.stack(
        [raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2]], axis=2)
    QK_out = fused_paired(
        x, h_QK, qk_op_arg, raw_tau_QK, qk_read, qk_write,
        soft_gate_T_qk, soft_gate_t_final, soft_gate_boundary_power,
        soft_gate_boundary_power_final, execution_prune_eps)
    Q = QK_out[:, :, 0, :] * qk_scale
    K = QK_out[:, :, 1, :] * qk_scale
    V = fused_single_v(
        x, h_V, v_op_arg, raw_tau_all[:, :, 2:3], v_read, v_write,
        soft_gate_T_v, soft_gate_t_final, soft_gate_boundary_power,
        soft_gate_boundary_power_final, execution_prune_eps)
    V = V * v_scale

    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    rng, rng_attn_drop = jax.random.split(rng)

    @jax.checkpoint
    def _attn_scores(Q, K, V, rng_drop):
        attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        attn_scores = jnp.where(
            causal, attn_scores, jnp.finfo(attn_scores.dtype).min)
        attn_w = jax.nn.softmax(attn_scores, axis=-1)
        attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_drop)
        return jnp.einsum('bhst,bhtd->bhsd', attn_w, V)

    out = _attn_scores(Q, K, V, rng_attn_drop)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    rng, rng_out = jax.random.split(rng)
    return safe_dropout(out, dropout_rate, deterministic, rng_out)


def _rst_forward_minimal(x, pool_params, router_params, rng,
                         router_dropout, dropout_rate, deterministic,
                         sharded_fns,
                         d_model=None, n_layers=None,
                         soft_gate_temperature=0.07,
                         soft_gate_t_final=0.07,
                         soft_gate_T_rst=None,
                         soft_gate_boundary_power=2.0,
                         soft_gate_boundary_power_final=4.0,
                         admission_den_power=1.0,
                         execution_prune_eps=0.0):
    """Minimal v4167 RST path: one SRW output and residual dropout."""
    del admission_den_power
    if d_model is None or n_layers is None:
        raise ValueError(
            "depth-scaled pool outputs require d_model and n_layers.")
    soft_gate_T_rst = (
        soft_gate_temperature if soft_gate_T_rst is None else soft_gate_T_rst)
    pool_params = _ensure_pool_operator_keys(pool_params)
    rst_op_key = pool_params['rst_op_key']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']
    rst_page_tables = pool_params.get('rst_page_tables', None)
    rst_op_arg = rst_page_tables if rst_page_tables is not None else rst_op_key

    rng, rng_drop = jax.random.split(rng)
    rst_read_query = (
        x @ router_params['proj_rst']['kernel']
        + router_params['proj_rst']['bias'])
    rst_read_query = safe_dropout(
        rst_read_query, router_dropout, deterministic, rng_drop)
    h = _read_write_operator_query(
        rst_read_query, x, router_params['rst_op_write_query_proj'])
    raw_tau = (
        x @ router_params['raw_tau_rst']['kernel']
        + router_params['raw_tau_rst']['bias'])
    _, _, rst_scale = _pool_output_scales(d_model, n_layers)
    if isinstance(sharded_fns, dict):
        fused_single = sharded_fns.get(
            'rst_single_minimal',
            sharded_fns.get('rst_single', sharded_fns['single']))
    else:
        fused_single, _ = sharded_fns
    out = fused_single(
        x, h, rst_op_arg, raw_tau, rst_read, rst_write,
        soft_gate_T_rst, soft_gate_t_final, soft_gate_boundary_power,
        soft_gate_boundary_power_final, execution_prune_eps)
    out = out * rst_scale
    rng, rng_out = jax.random.split(rng)
    return safe_dropout(out, dropout_rate, deterministic, rng_out)


def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v,
                  n_heads, d_model, n_layers,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False,
                  soft_gate_temperature=0.07,
                  soft_gate_t_final=0.07,
                  soft_gate_T_qk=None,
                  soft_gate_T_v=None,
                  soft_gate_boundary_power=2.0,
                  soft_gate_boundary_power_final=4.0,
                  admission_den_power=1.0,
                  execution_prune_eps=0.0):
    """v4167 sharded-only path. sharded_fns=(fused_single, fused_paired) required.

    `analysis=False` (train path): returns the SLIM tuple. `analysis=True`:
    returns the SLIM tuple extended with observational ANALYSIS stats
    (see scan_body below for the full unpack shape).
    """
    B, S, D = x.shape
    soft_gate_T_qk = (
        soft_gate_temperature if soft_gate_T_qk is None else soft_gate_T_qk)
    soft_gate_T_v = (
        soft_gate_temperature if soft_gate_T_v is None else soft_gate_T_v)
    pool_params = _ensure_pool_operator_keys(pool_params)
    qk_op_key = pool_params['attn_qk_op_key']
    qk_read = pool_params['attn_qk_read']
    qk_write = pool_params['attn_qk_write']
    qk_page_tables = pool_params.get('attn_qk_page_tables', None)
    qk_op_arg = qk_page_tables if qk_page_tables is not None else qk_op_key
    v_op_key = pool_params['attn_v_op_key']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']
    v_page_tables = pool_params.get('attn_v_page_tables', None)
    v_op_arg = v_page_tables if v_page_tables is not None else v_op_key

    # RW-derived operator keys are passed into the sharded SRW closure.
    # The closure forward-normalizes them for selection stability.
    qk_op_key_unit = qk_op_arg
    v_op_key_unit = v_op_arg

    _qk_op_key_norms = jax.lax.stop_gradient(
        jnp.linalg.norm(qk_op_key, axis=-1))
    attn_qk_op_key_norm_mean = _qk_op_key_norms.mean()
    attn_qk_op_key_norm_min = _qk_op_key_norms.min()
    attn_qk_op_key_norm_std = _qk_op_key_norms.std()
    _v_op_key_norms = jax.lax.stop_gradient(
        jnp.linalg.norm(v_op_key, axis=-1))
    attn_v_op_key_norm_mean = _v_op_key_norms.mean()
    attn_v_op_key_norm_min = _v_op_key_norms.min()
    attn_v_op_key_norm_std = _v_op_key_norms.std()
    if analysis:
        attn_qk_op_key_norm_max = _qk_op_key_norms.max()
        attn_v_op_key_norm_max = _v_op_key_norms.max()

    rng, rng_drop = jax.random.split(rng)
    # read-query side: state-conditioned condition/read demand.
    attn_read_query = (
        x @ router_params['proj_attn']['kernel']
        + router_params['proj_attn']['bias'])
    attn_read_query = safe_dropout(
        attn_read_query, router_dropout, deterministic, rng_drop)
    q_read_query, k_read_query, v_read_query = jnp.split(
        attn_read_query, 3, axis=-1)
    # operator query: RW-matched read-query x write-query signature.
    h_Q = _read_write_operator_query(
        q_read_query, x, router_params['q_op_write_query_proj'])
    h_K = _read_write_operator_query(
        k_read_query, x, router_params['k_op_write_query_proj'])
    h_V = _read_write_operator_query(
        v_read_query, x, router_params['v_op_write_query_proj'])

    raw_tau_all = (
        x @ router_params['raw_tau_attn']['kernel']
        + router_params['raw_tau_attn']['bias'])
    tau_all = _tau_from_param(raw_tau_all)
    if analysis:
        _tau_all_sg = jax.lax.stop_gradient(tau_all)
        attn_tau_std = _tau_all_sg.std(axis=(0, 1))  # [3] Q/K/V
        attn_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['raw_tau_attn']['kernel']) ** 2) + 1e-12)

    qk_scale, v_scale, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    if isinstance(sharded_fns, dict):
        fused_paired = sharded_fns.get('attn_qk_paired', sharded_fns.get('qk_paired', sharded_fns['paired']))
        fused_single_v = sharded_fns.get('attn_v_single', sharded_fns.get('v_single', sharded_fns['single']))
    else:
        fused_single_v, fused_paired = sharded_fns
    h_QK = jnp.stack([h_Q, h_K], axis=2)
    raw_tau_QK = jnp.stack([raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2]], axis=2)
    qk_ret = fused_paired(x, h_QK, qk_op_key_unit, raw_tau_QK,
                           qk_read, qk_write,
                           soft_gate_T_qk, soft_gate_t_final,
                           soft_gate_boundary_power,
                           soft_gate_boundary_power_final,
                           execution_prune_eps)
    (QK_out, qk_active, qk_raw_gmax, qk_lb, qk_sstd, qk_es, qk_anm,
     qk_strong, qk_positive_margin_active, qk_tau_abs,
     qk_dead_pen, qk_dead_cnt, qk_int_max,
     qk_den_cost_mean, qk_selection_cost_mean, qk_current_cost_mean,
     qk_selection_residency, qk_edge_margin_stat,
     qk_tau_direct, qk_no_active_direct) = qk_ret[:20]
    (qk_gate_eff_n, qk_gate_eff_ratio,
     qk_top1_gate_frac, qk_top1_gate_frac_max) = qk_ret[20:24]
    (q_active, k_active, q_strong, k_strong,
     q_active_n_mean, k_active_n_mean) = qk_ret[24:30]
    qk_offset = 30
    if analysis:
        (qk_margin_band, qk_margin_band_wide, qk_margin_band_mid, qk_skew, qk_apt_std, qk_entropy,
         qk_den_cost, qk_selection_cost, qk_current_cost,
         qk_kurt, qk_int_cap) = qk_ret[qk_offset:qk_offset + 11]
        qk_offset += 11
        qk_raw_norm = jnp.linalg.norm(QK_out, axis=-1).mean()
    qk_select_start = qk_offset
    qk_select_diag = qk_ret[qk_select_start:qk_select_start + SELECT_DIAG_COUNT]
    qk_exposure_start = qk_select_start + SELECT_DIAG_COUNT
    qk_exposure_diag = qk_ret[
        qk_exposure_start:qk_exposure_start + DEAD_EXPOSURE_DIAG_COUNT]
    Q = QK_out[:, :, 0, :] * qk_scale
    K = QK_out[:, :, 1, :] * qk_scale
    v_ret = fused_single_v(x, h_V, v_op_key_unit, raw_tau_all[:, :, 2:3],
                           v_read, v_write,
                           soft_gate_T_v, soft_gate_t_final,
                           soft_gate_boundary_power,
                           soft_gate_boundary_power_final,
                           execution_prune_eps)
    (V, v_active, v_raw_gmax, v_lb, v_sstd, v_es, v_anm,
     v_strong, v_positive_margin_active, v_tau_abs,
     v_dead_pen, v_dead_cnt, v_int_max,
     v_den_cost_mean, v_selection_cost_mean, v_current_cost_mean,
     v_selection_residency, v_edge_margin_stat,
     v_tau_direct, v_no_active_direct) = v_ret[:20]
    (v_gate_eff_n, v_gate_eff_ratio,
     v_top1_gate_frac, v_top1_gate_frac_max) = v_ret[20:24]
    v_offset = 24
    if analysis:
        (v_margin_band, v_margin_band_wide, v_margin_band_mid, v_skew, v_apt_std, v_entropy,
         v_den_cost, v_selection_cost, v_current_cost,
         v_kurt, v_int_cap) = v_ret[v_offset:v_offset + 11]
        v_offset += 11
        v_raw_norm = jnp.linalg.norm(V, axis=-1).mean()
    v_select_start = v_offset
    v_select_diag = v_ret[v_select_start:v_select_start + SELECT_DIAG_COUNT]
    v_exposure_start = v_select_start + SELECT_DIAG_COUNT
    v_exposure_diag = v_ret[
        v_exposure_start:v_exposure_start + DEAD_EXPOSURE_DIAG_COUNT]
    qk_sparsity_start = qk_exposure_start + DEAD_EXPOSURE_DIAG_COUNT
    qk_route_sparsity_diag = qk_ret[qk_sparsity_start]
    q_sparsity_diag = qk_route_sparsity_diag[0]
    k_sparsity_diag = qk_route_sparsity_diag[1]
    qk_sparsity_diag = qk_route_sparsity_diag.mean(axis=0)
    qk_page_diag = jnp.zeros((PAGE_DIAG_COUNT,), dtype=jnp.float32)
    qk_page_offset = qk_sparsity_start + 1
    if len(qk_ret) > qk_page_offset:
        qk_page_diag = qk_ret[qk_page_offset]
    v_sparsity_start = v_exposure_start + DEAD_EXPOSURE_DIAG_COUNT
    v_sparsity_diag = v_ret[v_sparsity_start]
    v_page_diag = jnp.zeros((PAGE_DIAG_COUNT,), dtype=jnp.float32)
    v_page_offset = v_sparsity_start + 1
    if len(v_ret) > v_page_offset:
        v_page_diag = v_ret[v_page_offset]
    V = V * v_scale

    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    rng, rng_attn_drop = jax.random.split(rng)
    @jax.checkpoint
    def _attn_scores(Q, K, V, rng_drop):
        attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        attn_scores = jnp.where(causal, attn_scores,
                                jnp.finfo(attn_scores.dtype).min)
        attn_w = jax.nn.softmax(attn_scores, axis=-1)
        if analysis:
            scores_sg = jax.lax.stop_gradient(attn_scores.astype(jnp.float32))
            score_floor = jnp.finfo(scores_sg.dtype).min
            causal_4d = causal[None, None, :, :]
            causal_f = causal_4d.astype(jnp.float32)
            valid_count = causal_f.sum() * jnp.float32(B * n_heads)
            valid_scores = jnp.where(causal_4d, scores_sg, 0.0)
            attn_logit_mean = valid_scores.sum() / valid_count
            attn_logit_var = (
                jnp.where(causal_4d, scores_sg - attn_logit_mean, 0.0) ** 2
            ).sum() / valid_count
            attn_logit_std = jnp.sqrt(attn_logit_var + 1e-12)

            masked_scores = jnp.where(causal_4d, scores_sg, score_floor)
            attn_logit_max_dbg = jnp.max(masked_scores)

            attn_w_sg = jax.lax.stop_gradient(attn_w.astype(jnp.float32))
            softmax_top1 = jnp.max(attn_w_sg, axis=-1)
            softmax_top1_mean = softmax_top1.mean()
            softmax_top1_max = softmax_top1.max()

            top1_logits = jnp.max(masked_scores, axis=-1)
            top1_idx = jnp.argmax(masked_scores, axis=-1)
            attn_idx = jnp.arange(S)
            second_scores = jnp.where(
                attn_idx[None, None, None, :] == top1_idx[..., None],
                score_floor,
                masked_scores)
            top2_logits = jnp.max(second_scores, axis=-1)
            has_top2 = (jnp.arange(S) + 1) > 1
            top2_logits = jnp.where(
                has_top2[None, None, :], top2_logits, top1_logits)
            logit_gap = top1_logits - top2_logits
            logit_gap_mean = logit_gap.mean()
            logit_gap_max = logit_gap.max()

            entropy_terms = jnp.where(
                attn_w_sg > 0.0,
                attn_w_sg * jnp.log(jnp.maximum(attn_w_sg, 1e-30)),
                0.0)
            softmax_entropy = -jnp.sum(entropy_terms, axis=-1)
            softmax_entropy_mean = softmax_entropy.mean()
            softmax_entropy_min = softmax_entropy.min()
        attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_drop)
        out_dbg = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
        if analysis:
            return (
                out_dbg,
                attn_logit_mean, attn_logit_std,
                attn_logit_max_dbg,
                softmax_top1_mean, softmax_top1_max,
                logit_gap_mean, logit_gap_max,
                softmax_entropy_mean, softmax_entropy_min,
            )
        return out_dbg

    if analysis:
        q_norms_dbg = jnp.linalg.norm(Q, axis=-1)
        k_norms_dbg = jnp.linalg.norm(K, axis=-1)
        v_norms_dbg = jnp.linalg.norm(V, axis=-1)
    if analysis:
        q_norm = q_norms_dbg.mean()
        q_norm_std = q_norms_dbg.std()
        q_norm_max = q_norms_dbg.max()
        k_norm = k_norms_dbg.mean()
        k_norm_std = k_norms_dbg.std()
        k_norm_max = k_norms_dbg.max()
        v_norm_dbg = v_norms_dbg.mean()

    if analysis:
        (out,
         attn_logit_mean, attn_logit_std, attn_logit_max_actual,
         softmax_top1_mean, attn_softmax_top1_max,
         logit_gap_mean, logit_gap_max,
         softmax_entropy_mean, softmax_entropy_min) = _attn_scores(
            Q, K, V, rng_attn_drop)
    else:
        out = _attn_scores(Q, K, V, rng_attn_drop)
    if analysis:
        o_input_norm = jnp.linalg.norm(out, axis=-1).mean()
        v_norm_max = v_norms_dbg.max()
        o_input_norm_max = jnp.linalg.norm(out, axis=-1).max()
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    attn_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    if analysis:
        o_out_norm_max = jnp.linalg.norm(out, axis=-1).max()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load-balance loss from gate distributions + tau regularization.
    tau_reg = jnp.maximum(tau_all, 0.0).mean() * 0.01
    # Q/K share the qk pool, while V has its own pool.  Keep the historical
    # /3 scaling so the aux magnitude stays comparable to older runs.
    aux = (qk_lb + v_lb) / 3.0 + tau_reg
    attn_raw_gmax = jnp.maximum(qk_raw_gmax.mean(), v_raw_gmax.mean())
    attn_rho_std_slim = (qk_sstd + v_sstd) / 2
    attn_gate_sum = (qk_es + v_es) / 2
    attn_active_n_mean = (qk_anm + v_anm) / 2
    attn_tau_mean = tau_all.mean()
    attn_strong = (qk_strong.mean() + v_strong.mean()) / 2
    attn_qk_positive_margin_mean_active = qk_positive_margin_active.mean()
    attn_v_positive_margin_mean_active = v_positive_margin_active.mean()
    attn_tau_abs_mean = (qk_tau_abs + v_tau_abs) / 2
    attn_dead_penalty = qk_dead_pen + v_dead_pen
    attn_dead_count = jax.lax.stop_gradient(qk_dead_cnt + v_dead_cnt)
    attn_int_max = jnp.maximum(qk_int_max, v_int_max)
    attn_den_cost_mean = (qk_den_cost_mean + v_den_cost_mean) / 2
    attn_selection_cost_mean = (qk_selection_cost_mean + v_selection_cost_mean) / 2
    attn_current_cost_mean = (qk_current_cost_mean + v_current_cost_mean) / 2
    attn_gate_eff_n = (qk_gate_eff_n + v_gate_eff_n) / 2
    attn_gate_eff_ratio = (qk_gate_eff_ratio + v_gate_eff_ratio) / 2
    attn_top1_gate_frac = (qk_top1_gate_frac + v_top1_gate_frac) / 2
    attn_top1_gate_frac_max = jnp.maximum(qk_top1_gate_frac_max,
                                          v_top1_gate_frac_max)
    attn_select_diag = tuple(
        (qk_select_diag[i] + v_select_diag[i]) / 2.0
        for i in range(SELECT_DIAG_COUNT))
    qk_n = jnp.float32(n_qk)
    v_n = jnp.float32(n_v)
    attn_n = jnp.maximum(qk_n + v_n, 1.0)
    attn_exposure_diag = (
        (qk_exposure_diag[DEAD_EXPOSURE_MEAN] * qk_n
         + v_exposure_diag[DEAD_EXPOSURE_MEAN] * v_n) / attn_n,
        jnp.minimum(qk_exposure_diag[DEAD_EXPOSURE_MIN],
                    v_exposure_diag[DEAD_EXPOSURE_MIN]),
        jnp.maximum(qk_exposure_diag[DEAD_EXPOSURE_MAX],
                    v_exposure_diag[DEAD_EXPOSURE_MAX]),
        (qk_exposure_diag[DEAD_EXPOSURE_DEAD_FRAC] * qk_n
         + v_exposure_diag[DEAD_EXPOSURE_DEAD_FRAC] * v_n) / attn_n,
        (qk_exposure_diag[DEAD_EXPOSURE_WEAK_FRAC] * qk_n
         + v_exposure_diag[DEAD_EXPOSURE_WEAK_FRAC] * v_n) / attn_n,
        (qk_exposure_diag[DEAD_EXPOSURE_TARGET]
         + v_exposure_diag[DEAD_EXPOSURE_TARGET]) / 2.0,
    )
    attn_split_core = jnp.stack((
        qk_raw_gmax.mean(),
        v_raw_gmax.mean(),
        qk_es,
        v_es,
        qk_anm,
        v_anm,
        qk_tau_abs,
        v_tau_abs,
        qk_dead_cnt,
        v_dead_cnt,
        qk_int_max,
        v_int_max,
        qk_den_cost_mean,
        v_den_cost_mean,
        qk_gate_eff_n,
        v_gate_eff_n,
        qk_gate_eff_ratio,
        v_gate_eff_ratio,
        qk_top1_gate_frac,
        v_top1_gate_frac,
        qk_top1_gate_frac_max,
        v_top1_gate_frac_max,
        qk_dead_pen,
        v_dead_pen,
        q_active,
        k_active,
        q_strong,
        k_strong,
        q_active_n_mean,
        k_active_n_mean,
        qk_current_cost_mean,
        v_current_cost_mean,
    )).astype(jnp.float32)
    attn_qk_select_diag = jnp.stack(qk_select_diag).astype(jnp.float32)
    attn_v_select_diag = jnp.stack(v_select_diag).astype(jnp.float32)
    attn_qk_exposure_diag = jnp.stack(qk_exposure_diag).astype(jnp.float32)
    attn_v_exposure_diag = jnp.stack(v_exposure_diag).astype(jnp.float32)
    # Per-layer direct tau/no-active stacks are diagnostics only.
    attn_tau_direct = jnp.concatenate(
        (qk_tau_direct[:, :, 0, :],
         qk_tau_direct[:, :, 1, :],
         v_tau_direct),
        axis=-1)
    attn_no_active_direct = jax.lax.stop_gradient(jnp.concatenate(
        (qk_no_active_direct[:, :, 0, :],
         qk_no_active_direct[:, :, 1, :],
         v_no_active_direct),
        axis=-1).astype(jnp.float32))
    slim_ret = (out, aux, qk_active.mean(), v_active.mean(), attn_raw_gmax,
                attn_rho_std_slim, attn_gate_sum, attn_active_n_mean,
                attn_out_norm, attn_tau_mean,
                attn_strong,
                qk_strong.mean(), v_strong.mean(),
                attn_qk_positive_margin_mean_active,
                attn_v_positive_margin_mean_active,
                attn_tau_abs_mean,
                attn_qk_op_key_norm_mean, attn_v_op_key_norm_mean,
                attn_qk_op_key_norm_min, attn_qk_op_key_norm_std,
                attn_v_op_key_norm_min, attn_v_op_key_norm_std,
                attn_dead_penalty, attn_dead_count,
                attn_tau_direct, attn_no_active_direct,
                attn_int_max,
                attn_den_cost_mean, attn_selection_cost_mean,
                attn_current_cost_mean,
                attn_gate_eff_n, attn_gate_eff_ratio,
                attn_top1_gate_frac, attn_top1_gate_frac_max,
                qk_selection_residency, v_selection_residency,
                qk_edge_margin_stat, v_edge_margin_stat,
                *attn_select_diag,
                *attn_exposure_diag,
                attn_split_core,
                attn_qk_select_diag,
                attn_v_select_diag,
                attn_qk_exposure_diag,
                attn_v_exposure_diag,
                qk_sparsity_diag,
                v_sparsity_diag,
                q_sparsity_diag,
                k_sparsity_diag,
                qk_page_diag,
                v_page_diag)
    if not analysis:
        ret = slim_ret
        return ret

    attn_qk_margin_band = qk_margin_band.mean()
    attn_v_margin_band = v_margin_band.mean()
    attn_margin_band_wide_frac = (qk_margin_band_wide + v_margin_band_wide) / 2
    attn_margin_band_mid_frac = (qk_margin_band_mid + v_margin_band_mid) / 2
    attn_rho_skew = (qk_skew + v_skew) / 2
    attn_active_per_token_std = (qk_apt_std + v_apt_std) / 2
    attn_gate_entropy = (qk_entropy + v_entropy) / 2
    attn_den_cost = (qk_den_cost + v_den_cost) / 2
    attn_selection_cost = (qk_selection_cost + v_selection_cost) / 2
    attn_current_cost = (qk_current_cost + v_current_cost) / 2
    attn_rho_kurt = (qk_kurt + v_kurt) / 2
    attn_int_cap_frac = (qk_int_cap + v_int_cap) / 2.0
    analysis_ret = slim_ret + (
        qk_raw_norm, v_raw_norm,
        q_norm, k_norm, v_norm_dbg, attn_logit_max_actual, o_input_norm,
        attn_qk_margin_band, attn_v_margin_band,
        attn_tau_std, attn_tau_kernel_norm,
        attn_margin_band_wide_frac, attn_margin_band_mid_frac,
        attn_rho_skew, attn_active_per_token_std, attn_gate_entropy,
        attn_den_cost,
        attn_selection_cost, attn_current_cost,
        attn_qk_op_key_norm_max, attn_v_op_key_norm_max,
        attn_rho_kurt,
        attn_int_cap_frac,
        q_norm_std, q_norm_max,
        k_norm_std, k_norm_max,
        attn_logit_mean, attn_logit_std,
        softmax_top1_mean, attn_softmax_top1_max,
        logit_gap_mean, logit_gap_max,
        softmax_entropy_mean, softmax_entropy_min,
        o_input_norm_max, o_out_norm_max,
    )
    return analysis_ret


def _rst_forward(x, pool_params, router_params, rng,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False,
                  d_model=None, n_layers=None,
                  soft_gate_temperature=0.07,
                  soft_gate_t_final=0.07,
                  soft_gate_T_rst=None,
                  soft_gate_boundary_power=2.0,
                  soft_gate_boundary_power_final=4.0,
                  admission_den_power=1.0,
                  execution_prune_eps=0.0):
    """v4167 sharded-only path. sharded_fns=(fused_single, fused_paired) required.

    `analysis` see _attn_forward docstring.
    """
    B, S, D = x.shape
    soft_gate_T_rst = (
        soft_gate_temperature
        if soft_gate_T_rst is None else soft_gate_T_rst)
    pool_params = _ensure_pool_operator_keys(pool_params)
    rst_op_key = pool_params['rst_op_key']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']
    rst_page_tables = pool_params.get('rst_page_tables', None)
    rst_op_arg = rst_page_tables if rst_page_tables is not None else rst_op_key

    rng, rng_drop = jax.random.split(rng)
    rst_read_query = (
        x @ router_params['proj_rst']['kernel']
        + router_params['proj_rst']['bias'])
    rst_read_query = safe_dropout(
        rst_read_query, router_dropout, deterministic, rng_drop)
    h = _read_write_operator_query(
        rst_read_query, x, router_params['rst_op_write_query_proj'])

    # RW-derived operator keys are passed into the sharded SRW closure.
    # The closure forward-normalizes them for selection stability.
    rst_op_key_unit = rst_op_arg
    raw_tau = (
        x @ router_params['raw_tau_rst']['kernel']
        + router_params['raw_tau_rst']['bias'])
    tau = _tau_from_param(raw_tau)
    if analysis:
        rst_tau_std = jax.lax.stop_gradient(tau).std()
        rst_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['raw_tau_rst']['kernel']) ** 2) + 1e-12)
    if d_model is None or n_layers is None:
        raise ValueError(
            "depth-scaled pool outputs require d_model and n_layers.")
    _, _, rst_scale = _pool_output_scales(d_model, n_layers)
    if isinstance(sharded_fns, dict):
        fused_single = sharded_fns.get('rst_single', sharded_fns['single'])
    else:
        fused_single, _ = sharded_fns
    rst_ret = fused_single(x, h, rst_op_key_unit, raw_tau,
                            rst_read, rst_write,
                            soft_gate_T_rst, soft_gate_t_final,
                            soft_gate_boundary_power,
                            soft_gate_boundary_power_final,
                            execution_prune_eps)
    (out, active_frac, raw_gate_max, lb_loss, rho_std_slim, gate_sum, active_n_mean,
     strong_frac, positive_margin_mean_active, rst_tau_abs_mean,
     rst_dead_penalty, rst_dead_count, rst_int_max,
     rst_den_cost_mean, rst_selection_cost_mean, rst_current_cost_mean,
     rst_selection_residency, rst_edge_margin_stat,
     rst_tau_direct, rst_no_active_direct) = rst_ret[:20]
    (rst_gate_eff_n, rst_gate_eff_ratio,
     rst_top1_gate_frac, rst_top1_gate_frac_max) = rst_ret[20:24]
    rst_offset = 24
    if analysis:
        (margin_band_frac, rst_margin_band_wide_frac, rst_margin_band_mid_frac,
         rst_rho_skew, rst_active_per_token_std, rst_gate_entropy,
         rst_den_cost, rst_selection_cost, rst_current_cost,
         rst_rho_kurt, rst_int_cap_frac) = rst_ret[rst_offset:rst_offset + 11]
        rst_offset += 11
        rst_raw_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rst_select_start = rst_offset
    rst_select_diag = rst_ret[rst_select_start:rst_select_start + SELECT_DIAG_COUNT]
    rst_exposure_start = rst_select_start + SELECT_DIAG_COUNT
    rst_exposure_diag = rst_ret[
        rst_exposure_start:rst_exposure_start + DEAD_EXPOSURE_DIAG_COUNT]
    rst_sparsity_start = rst_exposure_start + DEAD_EXPOSURE_DIAG_COUNT
    rst_sparsity_diag = rst_ret[rst_sparsity_start]
    rst_page_diag = jnp.zeros((PAGE_DIAG_COUNT,), dtype=jnp.float32)
    rst_page_offset = rst_sparsity_start + 1
    if len(rst_ret) > rst_page_offset:
        rst_page_diag = rst_ret[rst_page_offset]
    out = out * rst_scale
    rst_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rst_out_norm_max = jnp.linalg.norm(out, axis=-1).max()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    _rst_op_key_norms = jax.lax.stop_gradient(
        jnp.linalg.norm(rst_op_key, axis=-1))
    rst_op_key_norm = _rst_op_key_norms.mean()
    rst_op_key_norm_min = _rst_op_key_norms.min()
    rst_op_key_norm_std = _rst_op_key_norms.std()
    if analysis:
        rst_op_key_norm_max = _rst_op_key_norms.max()
    read_norm_val = jnp.linalg.norm(rst_read, axis=-1).mean()
    write_norm_val = jnp.linalg.norm(rst_write, axis=-1).mean()
    rst_tau_mean = tau.mean()
    rst_strong = strong_frac.mean()
    rst_positive_margin_mean_active = positive_margin_mean_active.mean()
    slim_ret = (out, aux, active_frac, raw_gate_max, rho_std_slim, gate_sum, active_n_mean,
                rst_op_key_norm, read_norm_val, write_norm_val, rst_out_norm,
                rst_tau_mean, rst_strong, rst_positive_margin_mean_active,
                rst_tau_abs_mean,
                rst_op_key_norm_min, rst_op_key_norm_std,
                rst_dead_penalty, rst_dead_count,
                rst_tau_direct, rst_no_active_direct,
                rst_int_max,
                rst_den_cost_mean, rst_selection_cost_mean,
                rst_current_cost_mean,
                rst_gate_eff_n, rst_gate_eff_ratio,
                rst_top1_gate_frac, rst_top1_gate_frac_max,
                rst_selection_residency,
                rst_edge_margin_stat,
                *rst_select_diag,
                *rst_exposure_diag,
                rst_sparsity_diag,
                rst_page_diag)
    if not analysis:
        ret = slim_ret
        return ret

    rst_margin_band = margin_band_frac.mean()
    analysis_ret = slim_ret + (
        rst_raw_out_norm,
        rst_tau_std, rst_tau_kernel_norm,
        rst_margin_band_wide_frac, rst_margin_band_mid_frac,
        rst_rho_skew, rst_active_per_token_std, rst_gate_entropy,
        rst_den_cost,
        rst_selection_cost, rst_current_cost,
        rst_op_key_norm_max,
        rst_rho_kurt,
        rst_margin_band,
        rst_int_cap_frac,
    )
    return analysis_ret


# ================================================================
# 7. Flax modules (init path only)
# ================================================================

class AttentionLayer(nn.Module):
    """Attention Layer container.

    The Attention Layer performs model decisions over the attention-qk and
    attention-v pools to construct Q/K/V, then applies causal self-attention
    for relational state interaction. The real forward path is _attn_forward().
    """
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))


class DAWNBlock(nn.Module):
    """DAWN-SRW Block = Attention Layer + RST Layer.

    The RST Layer is the concrete post-attention layer that selects and
    composes RW operators to refine the residual state and transition it to the
    next representation state.

    Container for per-layer norms + attn (expand_O) submodules.
    The real forward path is scan_body in DAWN.__call__."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn = AttentionLayer(
            d_model=self.d_model, n_heads=self.n_heads,
            dropout_rate=self.dropout_rate)




# ================================================================
# 8. DAWN Model
# ================================================================

class DAWN_SRW_V4167(nn.Module):
    """DAWN-SRW v4.1.6.7 with global CEU operator pages."""
    __version__ = MODEL_VERSION

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_route: int = DEFAULT_D_ROUTE
    n_qk: int = 1580
    n_v: int = 2600
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Checkpoint/config alias; n_rst is canonical.
    router_dropout: float = 0.1
    n_chunks_rst: Optional[int] = None
    n_chunks_know: int = 1    # Config alias; n_chunks_rst is canonical.
    n_chunks_qk: int = 1     # N-axis chunking for qk pool
    n_chunks_v: int = 1      # N-axis chunking for v pool
    operator_pages_enabled: bool = True
    operator_pages_pools: Optional[Any] = ('qk', 'v', 'rst')
    operator_page_size_qk: int = 128
    operator_page_size_v: int = 128
    operator_page_size_rst: int = 128
    operator_page_capacity_qk: int = 8
    operator_page_capacity_v: int = 8
    operator_page_capacity_rst: int = 32
    operator_page_microgroup_sequences: int = 2
    operator_page_score_mode: str = 'maxmean'
    operator_page_fallback_pages: int = 0
    operator_page_random_pages: int = 0
    operator_page_cost_weight: float = 0.0
    operator_pages_analysis_full_scan: bool = False
    # Future page-centroid options are intentionally not implemented in this
    # first training patch: learned anchors, compactness/separation losses, and
    # periodic repacking all need separate gradient/control-policy validation.
    # Constructor receives cosine-space tau values. The train driver may use
    # safe placeholders before one-time quantile calibration.
    tau_init_attn_qk: Optional[float] = None
    tau_init_attn_v: Optional[float] = None
    tau_init_rst: Optional[float] = None

    def setup(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})")
        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model, embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model, embedding_init=scaled_normal(0.02))
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst_eff,
            d_model=self.d_model, d_route=self.d_route)
        self.router = Router(
            d_model=self.d_model, d_route=self.d_route,
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst_eff,
            router_dropout=self.router_dropout,
            tau_init_attn_qk=self.tau_init_attn_qk,
            tau_init_attn_v=self.tau_init_attn_v,
            tau_init_rst=self.tau_init_rst)
        self.layers = [
            DAWNBlock(d_model=self.d_model, n_heads=self.n_heads,
                      dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, sharded_fns=None, analysis=False,
                 soft_gate_temperature=0.07,
                 soft_gate_t_final=0.07,
                 soft_gate_T_qk=None,
                 soft_gate_T_v=None,
                 soft_gate_T_rst=None,
                 soft_gate_boundary_power=2.0,
                 soft_gate_boundary_power_final=4.0,
                 admission_den_power=1.0,
                 execution_prune_eps=0.0,
                 minimal_train=False):
        """Run the shared-pool SRW Transformer forward pass.

        analysis=False is the train/eval path and returns only regular
        training metrics.  analysis=True enables extra observational stats
        such as distribution shape, selection diagnostics, entropy, tau stats,
        raw norms, and output-stability norms.
        """
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        soft_gate_T_qk = (
            soft_gate_temperature
            if soft_gate_T_qk is None else soft_gate_T_qk)
        soft_gate_T_v = (
            soft_gate_temperature
            if soft_gate_T_v is None else soft_gate_T_v)
        soft_gate_T_rst = (
            soft_gate_temperature
            if soft_gate_T_rst is None else soft_gate_T_rst)
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len")

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        if self.is_initializing():
            _z = jnp.float32(0.0)
            total_aux = _z
            attn_auxes = _z
            rst_auxes = _z
            rst_active_all = _z
            rst_raw_gmax_all = _z
            rst_sstd_all = _z
            rst_gsum_all = _z
            rst_active_n_mean_all = _z
            rst_strong_all = _z
            attn_qk_active_all = _z
            attn_v_active_all = _z
            attn_raw_gmax_all = _z
            attn_sstd_all = _z
            attn_gsum_all = _z
            attn_active_n_mean_all = _z
            attn_strong_all = _z
            attn_qk_strong_all = _z
            attn_v_strong_all = _z
            rst_positive_margin_active_all = _z
            attn_qk_positive_margin_active_all = _z
            attn_v_positive_margin_active_all = _z
            rst_op_key_n_all = _z
            k_read_n_all = _z
            k_write_n_all = _z
            rst_out_norm_all = _z
            attn_out_norm_all = _z
            attn_tau_mean_all = _z
            rst_tau_mean_all = _z
            attn_tau_abs_all = _z
            rst_tau_abs_all = _z
            attn_qk_op_key_n_mean_all = _z
            attn_v_op_key_n_mean_all = _z
            rst_op_key_n_std_all = _z
            attn_qk_op_key_n_min_all = _z
            attn_qk_op_key_n_std_all = _z
            attn_v_op_key_n_min_all = _z
            attn_v_op_key_n_std_all = _z
            rst_op_key_n_min_all = _z
            attn_dead_penalty_all = _z
            rst_dead_penalty_all = _z
            attn_dead_count_all = _z
            rst_dead_count_all = _z
            attn_tau_direct_all = _z
            rst_tau_direct_all = _z
            attn_no_active_direct_all = _z
            rst_no_active_direct_all = _z
            attn_int_max_all = _z
            rst_int_max_all = _z
            attn_den_cost_mean_all = _z
            rst_den_cost_mean_all = _z
            attn_selection_cost_mean_all = _z
            rst_selection_cost_mean_all = _z
            attn_current_cost_mean_all = _z
            rst_current_cost_mean_all = _z
            attn_gate_eff_n_all = _z
            attn_gate_eff_ratio_all = _z
            attn_top1_gate_frac_all = _z
            attn_top1_gate_frac_max_all = _z
            rst_gate_eff_n_all = _z
            rst_gate_eff_ratio_all = _z
            rst_top1_gate_frac_all = _z
            rst_top1_gate_frac_max_all = _z
            attn_qk_selection_residency_all = _z
            attn_v_selection_residency_all = _z
            rst_selection_residency_all = _z
            attn_qk_edge_margin_stat_all = _z
            attn_v_edge_margin_stat_all = _z
            rst_edge_margin_stat_all = _z
            attn_rho_mean_all = _z
            attn_rho_std_all = _z
            attn_rho_max_all = _z
            attn_tau_raw_mean_all = _z
            attn_tau_floor_mean_all = _z
            attn_tau_min_hit_frac_all = _z
            attn_tau_direct_mean_all = _z
            attn_tau_direct_min_all = _z
            attn_tau_direct_max_all = _z
            attn_selection_margin_mean_all = _z
            attn_positive_margin_mean_all = _z
            attn_positive_margin_max_all = _z
            attn_selected_frac_all = _z
            attn_no_active_frac_all = _z
            attn_angular_exposure_mean_all = _z
            attn_angular_exposure_min_all = _z
            attn_angular_exposure_max_all = _z
            attn_dead_exposure_frac_all = _z
            attn_weak_exposure_frac_all = _z
            attn_dead_exposure_target_all = _z
            rst_rho_mean_all = _z
            rst_rho_std_all = _z
            rst_rho_max_all = _z
            rst_tau_raw_mean_all = _z
            rst_tau_floor_mean_all = _z
            rst_tau_min_hit_frac_all = _z
            rst_tau_direct_mean_all = _z
            rst_tau_direct_min_all = _z
            rst_tau_direct_max_all = _z
            rst_selection_margin_mean_all = _z
            rst_positive_margin_mean_all = _z
            rst_positive_margin_max_all = _z
            rst_selected_frac_all = _z
            rst_no_active_frac_all = _z
            rst_angular_exposure_mean_all = _z
            rst_angular_exposure_min_all = _z
            rst_angular_exposure_max_all = _z
            rst_dead_exposure_frac_all = _z
            rst_weak_exposure_frac_all = _z
            rst_dead_exposure_target_all = _z
            attn_split_core_all = jnp.zeros(
                (1, ATTN_SPLIT_CORE_COUNT), dtype=jnp.float32)
            attn_qk_select_diag_all = jnp.zeros(
                (1, SELECT_DIAG_COUNT), dtype=jnp.float32)
            attn_v_select_diag_all = jnp.zeros(
                (1, SELECT_DIAG_COUNT), dtype=jnp.float32)
            attn_qk_exposure_diag_all = jnp.zeros(
                (1, DEAD_EXPOSURE_DIAG_COUNT), dtype=jnp.float32)
            attn_v_exposure_diag_all = jnp.zeros(
                (1, DEAD_EXPOSURE_DIAG_COUNT), dtype=jnp.float32)
            attn_qk_sparsity_diag_all = jnp.zeros(
                (1, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32)
            attn_v_sparsity_diag_all = jnp.zeros(
                (1, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32)
            attn_q_sparsity_diag_all = jnp.zeros(
                (1, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32)
            attn_k_sparsity_diag_all = jnp.zeros(
                (1, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32)
            rst_sparsity_diag_all = jnp.zeros(
                (1, GATE_SPARSITY_DIAG_COUNT), dtype=jnp.float32)
            # Trigger Flax param realization for all submodules (init-only).
            # The real forward runs through scan_body in the else branch and
            # accesses params by path, not via these module calls.
            _ = self.neuron_pool.attn_qk_read  # triggers NeuronPool.setup
            _ = self.neuron_pool.attn_qk_op_read_proj
            _ = self.neuron_pool.attn_qk_op_write_proj
            _ = self.neuron_pool.attn_v_op_read_proj
            _ = self.neuron_pool.attn_v_op_write_proj
            _ = self.neuron_pool.rst_op_read_proj
            _ = self.neuron_pool.rst_op_write_proj
            _ = self.router.proj_attn(x)
            _ = self.router.proj_rst(x)
            _ = self.router.raw_tau_attn(x)
            _ = self.router.raw_tau_rst(x)
            for layer in self.layers:
                _ = layer.norm1(x)
                _ = layer.norm2(x)
                _ = layer.attn.expand_O(x)
        else:
            all_params = self.variables['params']
            pool_params = _pool_params_with_operator_keys(
                all_params['neuron_pool'])
            pool_params = _pool_params_with_operator_page_tables(
                pool_params, sharded_fns)
            router_params = all_params['router']

            _sharded = sharded_fns

            block_params_list = [all_params[f'block_{i}']
                                 for i in range(self.n_layers)]
            stacked = jax.tree.map(
                lambda *arrays: jnp.stack(arrays), *block_params_list)

            base_rng = self.make_rng('dropout')
            layer_rngs = jax.random.split(base_rng, self.n_layers)

            if minimal_train:
                def scan_body_minimal(carry, xs):
                    x = carry
                    bp = xs['params']
                    rng = xs['rng']
                    rng, rng_attn, rng_rst = jax.random.split(rng, 3)

                    normed = _layer_norm(
                        x, bp['norm1']['scale'], bp['norm1']['bias'])
                    attn_out = _attn_forward_minimal(
                        normed, pool_params, router_params,
                        bp['attn']['expand_O']['kernel'], rng_attn,
                        self.n_qk, self.n_v,
                        self.n_heads, self.d_model, self.n_layers,
                        self.router_dropout, self.dropout_rate,
                        deterministic,
                        sharded_fns=_sharded,
                        soft_gate_temperature=soft_gate_temperature,
                        soft_gate_t_final=soft_gate_t_final,
                        soft_gate_T_qk=soft_gate_T_qk,
                        soft_gate_T_v=soft_gate_T_v,
                        soft_gate_boundary_power=soft_gate_boundary_power,
                        soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                        admission_den_power=admission_den_power,
                        execution_prune_eps=execution_prune_eps)
                    x = x + attn_out

                    normed = _layer_norm(
                        x, bp['norm2']['scale'], bp['norm2']['bias'])
                    rst_out = _rst_forward_minimal(
                        normed, pool_params, router_params, rng_rst,
                        self.router_dropout, self.dropout_rate,
                        deterministic,
                        sharded_fns=_sharded,
                        d_model=self.d_model,
                        n_layers=self.n_layers,
                        soft_gate_temperature=soft_gate_temperature,
                        soft_gate_t_final=soft_gate_t_final,
                        soft_gate_T_rst=soft_gate_T_rst,
                        soft_gate_boundary_power=soft_gate_boundary_power,
                        soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                        admission_den_power=admission_den_power,
                        execution_prune_eps=execution_prune_eps)
                    return x + rst_out, None

                if self.gradient_checkpointing:
                    scan_body_minimal = jax.checkpoint(scan_body_minimal)

                xs_minimal = {
                    'params': stacked,
                    'rng': layer_rngs,
                }
                x, _ = jax.lax.scan(scan_body_minimal, x, xs_minimal)
                x = self.norm(x)
                if labels is None:
                    return {'logits': self.token_emb.attend(x)}

                embedding_matrix = self.token_emb.embedding
                shift_x = x[:, :-1, :]
                shift_labels = labels[:, 1:].astype(jnp.int32)
                valid_mask = (shift_labels != -100)

                @jax.checkpoint
                def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                    logits = x_chunk @ emb.T
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    safe = jnp.where(vmask, labs, 0)
                    token_loss = -jnp.take_along_axis(
                        log_probs, safe[..., jnp.newaxis],
                        axis=-1).squeeze(-1)
                    loss = (
                        (token_loss * vmask).sum()
                        / (vmask.sum() + 1e-8))
                    preds = jnp.argmax(logits, axis=-1)
                    correct = jnp.sum((preds == labs) & vmask)
                    return loss, correct, jnp.sum(vmask)

                loss, correct, valid_count = compute_loss_and_acc(
                    shift_x, embedding_matrix, shift_labels, valid_mask)
                return {
                    'loss': loss,
                    'correct': correct,
                    'valid_count': valid_count,
                    'aux_loss': jnp.float32(0.0),
                }

            def scan_body(carry, xs):
                x = carry
                bp = xs['params']
                rng = xs['rng']
                layer_idx = xs['layer_idx']
                rng, rng_attn, rng_rst = jax.random.split(rng, 3)

                x_pre_attn = x
                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])
                attn_ret = _attn_forward(
                    normed, pool_params, router_params,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v,
                    self.n_heads, self.d_model, self.n_layers,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis,
                    soft_gate_temperature=soft_gate_temperature,
                    soft_gate_t_final=soft_gate_t_final,
                    soft_gate_T_qk=soft_gate_T_qk,
                    soft_gate_T_v=soft_gate_T_v,
                    soft_gate_boundary_power=soft_gate_boundary_power,
                    soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                    admission_den_power=admission_den_power,
                    execution_prune_eps=execution_prune_eps)
                (attn_out, attn_aux, a_qk_active, a_v_active, a_raw_gmax,
                 a_sstd, a_gsum, a_active_n_mean,
                 a_out_norm, a_tau_mean, a_strong,
                 a_qk_strong, a_v_strong,
                 a_qk_positive_margin_active, a_v_positive_margin_active,
                 a_tau_abs,
                 a_qk_op_key_n_mean, a_v_op_key_n_mean,
                 a_qk_op_key_n_min, a_qk_op_key_n_std,
                 a_v_op_key_n_min, a_v_op_key_n_std,
                 a_dead_penalty, a_dead_count,
                 a_tau_direct, a_no_active_direct,
                 a_int_max,
                 a_den_cost_mean, a_selection_cost_mean,
                 a_current_cost_mean,
                 a_gate_eff_n, a_gate_eff_ratio,
                 a_top1_gate_frac, a_top1_gate_frac_max,
                 a_qk_selection_residency,
                 a_v_selection_residency,
                 a_qk_edge_margin_stat,
                 a_v_edge_margin_stat,
                 a_rho_mean, a_rho_std, a_rho_max,
                 a_tau_raw_mean, a_tau_floor_mean, a_tau_min_hit_frac,
                 a_tau_direct_mean, a_tau_direct_min, a_tau_direct_max,
                 a_selection_margin_mean,
                 a_positive_margin_mean, a_positive_margin_max,
                 a_selected_frac, a_no_active_frac,
                 a_angular_exposure_mean, a_angular_exposure_min,
                 a_angular_exposure_max, a_dead_exposure_frac,
                 a_weak_exposure_frac, a_dead_exposure_target) = attn_ret[:58]
                (a_split_core,
                 a_qk_select_diag, a_v_select_diag,
                 a_qk_exposure_diag, a_v_exposure_diag,
                 a_qk_sparsity_diag, a_v_sparsity_diag,
                 a_q_sparsity_diag, a_k_sparsity_diag,
                 a_qk_page_diag, a_v_page_diag) = attn_ret[58:69]
                if analysis:
                    (a_qk_raw_norm, a_v_raw_norm,
                     a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                     a_qk_margin_band, a_v_margin_band,
                     a_tau_std, a_tau_kernel_norm,
                     a_margin_band_wide, a_margin_band_mid,
                     a_skew, a_apt_std, a_entropy,
                     a_den_cost, a_selection_cost, a_current_cost,
                     a_qk_op_key_n_max, a_v_op_key_n_max,
                     a_rho_kurt, a_int_cap_frac,
                     a_q_norm_std, a_q_norm_max,
                     a_k_norm_std, a_k_norm_max,
                     a_logit_mean, a_logit_std,
                     a_softmax_top1_mean, a_softmax_top1_max,
                     a_logit_gap_mean, a_logit_gap_max,
                     a_softmax_entropy_mean, a_softmax_entropy_min,
                     a_o_input_norm_max, a_o_out_norm_max) = attn_ret[69:106]
                x = x + attn_out
                x_post_attn = x
                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                rst_ret = _rst_forward(
                    normed, pool_params, router_params, rng_rst,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis,
                    d_model=self.d_model, n_layers=self.n_layers,
                    soft_gate_temperature=soft_gate_temperature,
                    soft_gate_t_final=soft_gate_t_final,
                    soft_gate_T_rst=soft_gate_T_rst,
                    soft_gate_boundary_power=soft_gate_boundary_power,
                    soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                    admission_den_power=admission_den_power,
                    execution_prune_eps=execution_prune_eps)
                (rst_out, rst_aux, k_active, k_raw_gmax, k_sstd, k_gsum,
                 k_active_n_mean, k_op_key_n, k_read_n, k_write_n, k_out_norm,
                 k_tau_mean, k_strong, k_positive_margin_active, k_tau_abs,
                 k_op_key_n_min, k_op_key_n_std,
                 k_dead_penalty, k_dead_count,
                 k_tau_direct, k_no_active_direct,
                 k_int_max,
                 k_den_cost_mean, k_selection_cost_mean,
                 k_current_cost_mean,
                 k_gate_eff_n, k_gate_eff_ratio,
                 k_top1_gate_frac, k_top1_gate_frac_max,
                 k_selection_residency,
                 k_edge_margin_stat,
                 k_rho_mean, k_rho_std, k_rho_max,
                 k_tau_raw_mean, k_tau_floor_mean, k_tau_min_hit_frac,
                 k_tau_direct_mean, k_tau_direct_min, k_tau_direct_max,
                 k_selection_margin_mean,
                 k_positive_margin_mean, k_positive_margin_max,
                 k_selected_frac, k_no_active_frac,
                 k_angular_exposure_mean, k_angular_exposure_min,
                 k_angular_exposure_max, k_dead_exposure_frac,
                 k_weak_exposure_frac, k_dead_exposure_target,
                 k_sparsity_diag,
                 k_page_diag) = rst_ret[:53]
                if analysis:
                    (k_raw_out_norm,
                     k_tau_std, k_tau_kernel_norm,
                     k_margin_band_wide, k_margin_band_mid,
                     k_skew, k_apt_std, k_entropy,
                     k_den_cost, k_selection_cost, k_current_cost,
                     k_op_key_n_max, k_rho_kurt, k_margin_band,
                     k_int_cap_frac) = rst_ret[53:68]
                x = x + rst_out
                x_post_rst = x
                slim_ys = (attn_aux, rst_aux,
                           k_active, k_raw_gmax, k_sstd, k_gsum, k_active_n_mean,
                           a_qk_active, a_v_active, a_raw_gmax, a_sstd, a_gsum, a_active_n_mean,
                           k_op_key_n, k_read_n, k_write_n,
                           k_out_norm,
                           a_out_norm, a_tau_mean, k_tau_mean,
                           k_strong, a_strong,
                           a_qk_strong, a_v_strong,
                           k_positive_margin_active,
                           a_qk_positive_margin_active,
                           a_v_positive_margin_active,
                           a_tau_abs, k_tau_abs,
                           a_qk_op_key_n_mean, a_v_op_key_n_mean,
                           k_op_key_n_std,
                           a_qk_op_key_n_min, a_qk_op_key_n_std,
                           a_v_op_key_n_min, a_v_op_key_n_std,
                           k_op_key_n_min,
                           a_dead_penalty, k_dead_penalty,
                           a_dead_count, k_dead_count,
                           a_tau_direct, k_tau_direct,
                           a_no_active_direct, k_no_active_direct,
                           a_int_max, k_int_max,
                           a_den_cost_mean, k_den_cost_mean,
                           a_selection_cost_mean, k_selection_cost_mean,
                           a_current_cost_mean, k_current_cost_mean,
                           a_gate_eff_n, a_gate_eff_ratio,
                           a_top1_gate_frac, a_top1_gate_frac_max,
                           k_gate_eff_n, k_gate_eff_ratio,
                           k_top1_gate_frac, k_top1_gate_frac_max,
                           a_qk_selection_residency,
                           a_v_selection_residency,
                           k_selection_residency,
                           a_qk_edge_margin_stat,
                           a_v_edge_margin_stat,
                           k_edge_margin_stat,
                           a_rho_mean, a_rho_std, a_rho_max,
                           a_tau_raw_mean, a_tau_floor_mean,
                           a_tau_min_hit_frac,
                           a_tau_direct_mean, a_tau_direct_min,
                           a_tau_direct_max,
                           a_selection_margin_mean,
                           a_positive_margin_mean, a_positive_margin_max,
                           a_selected_frac, a_no_active_frac,
                           a_angular_exposure_mean, a_angular_exposure_min,
                           a_angular_exposure_max, a_dead_exposure_frac,
                           a_weak_exposure_frac, a_dead_exposure_target,
                           k_rho_mean, k_rho_std, k_rho_max,
                           k_tau_raw_mean, k_tau_floor_mean,
                           k_tau_min_hit_frac,
                           k_tau_direct_mean, k_tau_direct_min,
                           k_tau_direct_max,
                           k_selection_margin_mean,
                           k_positive_margin_mean, k_positive_margin_max,
                           k_selected_frac, k_no_active_frac,
                           k_angular_exposure_mean, k_angular_exposure_min,
                           k_angular_exposure_max, k_dead_exposure_frac,
                           k_weak_exposure_frac, k_dead_exposure_target,
                           a_split_core,
                           a_qk_select_diag, a_v_select_diag,
                           a_qk_exposure_diag, a_v_exposure_diag,
                           a_qk_sparsity_diag, a_v_sparsity_diag,
                           a_q_sparsity_diag, a_k_sparsity_diag,
                           k_sparsity_diag,
                           a_qk_page_diag, a_v_page_diag, k_page_diag,
                           )
                if not analysis:
                    return x, slim_ys
                analysis_ys = slim_ys + (
                    a_qk_raw_norm, a_v_raw_norm, k_raw_out_norm,
                    a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                    k_margin_band, a_qk_margin_band, a_v_margin_band,
                    a_tau_std, k_tau_std,
                    a_tau_kernel_norm, k_tau_kernel_norm,
                    a_margin_band_wide, k_margin_band_wide,
                    a_margin_band_mid, k_margin_band_mid,
                    a_skew, k_skew,
                    a_apt_std, k_apt_std,
                    a_entropy, k_entropy,
                    a_den_cost, k_den_cost,
                    a_selection_cost, k_selection_cost,
                    a_current_cost, k_current_cost,
                    a_qk_op_key_n_max, a_v_op_key_n_max,
                    k_op_key_n_max,
                    a_rho_kurt, k_rho_kurt,
                    a_int_cap_frac, k_int_cap_frac,
                    a_q_norm_std, a_q_norm_max,
                    a_k_norm_std, a_k_norm_max,
                    a_logit_mean, a_logit_std,
                    a_softmax_top1_mean, a_softmax_top1_max,
                    a_logit_gap_mean, a_logit_gap_max,
                    a_softmax_entropy_mean, a_softmax_entropy_min,
                    a_o_input_norm_max, a_o_out_norm_max,
                )
                return x, analysis_ys

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {
                'params': stacked,
                'rng': layer_rngs,
                'layer_idx': jnp.arange(self.n_layers, dtype=jnp.int32),
            }
            x, scan_ys = jax.lax.scan(scan_body, x, xs)

            (attn_auxes, rst_auxes,
             rst_active_all, rst_raw_gmax_all, rst_sstd_all, rst_gsum_all, rst_active_n_mean_all,
             attn_qk_active_all, attn_v_active_all, attn_raw_gmax_all, attn_sstd_all, attn_gsum_all, attn_active_n_mean_all,
             rst_op_key_n_all, k_read_n_all, k_write_n_all,
             rst_out_norm_all,
             attn_out_norm_all, attn_tau_mean_all, rst_tau_mean_all,
             rst_strong_all, attn_strong_all,
             attn_qk_strong_all, attn_v_strong_all,
             rst_positive_margin_active_all,
             attn_qk_positive_margin_active_all,
             attn_v_positive_margin_active_all,
             attn_tau_abs_all, rst_tau_abs_all,
             attn_qk_op_key_n_mean_all, attn_v_op_key_n_mean_all,
             rst_op_key_n_std_all,
             attn_qk_op_key_n_min_all, attn_qk_op_key_n_std_all,
             attn_v_op_key_n_min_all, attn_v_op_key_n_std_all,
             rst_op_key_n_min_all,
            attn_dead_penalty_all, rst_dead_penalty_all,
            attn_dead_count_all, rst_dead_count_all,
            attn_tau_direct_all, rst_tau_direct_all,
            attn_no_active_direct_all, rst_no_active_direct_all,
            attn_int_max_all, rst_int_max_all,
            attn_den_cost_mean_all, rst_den_cost_mean_all,
            attn_selection_cost_mean_all, rst_selection_cost_mean_all,
            attn_current_cost_mean_all, rst_current_cost_mean_all,
            attn_gate_eff_n_all, attn_gate_eff_ratio_all,
            attn_top1_gate_frac_all, attn_top1_gate_frac_max_all,
            rst_gate_eff_n_all, rst_gate_eff_ratio_all,
            rst_top1_gate_frac_all, rst_top1_gate_frac_max_all,
            attn_qk_selection_residency_all,
            attn_v_selection_residency_all,
            rst_selection_residency_all,
            attn_qk_edge_margin_stat_all,
            attn_v_edge_margin_stat_all,
            rst_edge_margin_stat_all,
            attn_rho_mean_all, attn_rho_std_all, attn_rho_max_all,
            attn_tau_raw_mean_all, attn_tau_floor_mean_all,
            attn_tau_min_hit_frac_all,
            attn_tau_direct_mean_all, attn_tau_direct_min_all,
            attn_tau_direct_max_all,
            attn_selection_margin_mean_all,
            attn_positive_margin_mean_all, attn_positive_margin_max_all,
            attn_selected_frac_all, attn_no_active_frac_all,
            attn_angular_exposure_mean_all,
            attn_angular_exposure_min_all,
            attn_angular_exposure_max_all,
            attn_dead_exposure_frac_all,
            attn_weak_exposure_frac_all,
            attn_dead_exposure_target_all,
            rst_rho_mean_all, rst_rho_std_all, rst_rho_max_all,
            rst_tau_raw_mean_all, rst_tau_floor_mean_all,
            rst_tau_min_hit_frac_all,
            rst_tau_direct_mean_all, rst_tau_direct_min_all,
            rst_tau_direct_max_all,
            rst_selection_margin_mean_all,
            rst_positive_margin_mean_all, rst_positive_margin_max_all,
            rst_selected_frac_all, rst_no_active_frac_all,
            rst_angular_exposure_mean_all,
            rst_angular_exposure_min_all,
            rst_angular_exposure_max_all,
            rst_dead_exposure_frac_all,
            rst_weak_exposure_frac_all,
            rst_dead_exposure_target_all) = scan_ys[:107]
            _scan_offset = 107
            (attn_split_core_all,
             attn_qk_select_diag_all, attn_v_select_diag_all,
             attn_qk_exposure_diag_all, attn_v_exposure_diag_all,
             attn_qk_sparsity_diag_all, attn_v_sparsity_diag_all,
             attn_q_sparsity_diag_all, attn_k_sparsity_diag_all,
             rst_sparsity_diag_all,
             attn_qk_page_diag_all, attn_v_page_diag_all,
             rst_page_diag_all) = scan_ys[
                _scan_offset:_scan_offset + 13]
            _scan_offset += 13
            if analysis:
                (attn_qk_raw_norm_all, attn_v_raw_norm_all, rst_raw_out_norm_all,
                 attn_q_norm_all, attn_k_norm_all, attn_v_norm_dbg_all,
                 attn_logit_max_all, attn_o_input_norm_all,
                 rst_margin_band_all, attn_qk_margin_band_all, attn_v_margin_band_all,
                 attn_tau_std_all, rst_tau_std_all,
                 attn_tau_kernel_norm_all, rst_tau_kernel_norm_all,
                 attn_margin_band_wide_all, rst_margin_band_wide_all,
                 attn_margin_band_mid_all, rst_margin_band_mid_all,
                 attn_skew_all, rst_skew_all,
                 attn_apt_std_all, rst_apt_std_all,
                 attn_entropy_all, rst_entropy_all,
                 attn_den_cost_all, rst_den_cost_all,
                 attn_selection_cost_all, rst_selection_cost_all,
                 attn_current_cost_all, rst_current_cost_all,
                 attn_qk_op_key_n_max_all, attn_v_op_key_n_max_all,
                 rst_op_key_n_max_all,
                 attn_rho_kurt_all, rst_rho_kurt_all,
                 attn_int_cap_frac_all, rst_int_cap_frac_all,
                 attn_q_norm_std_all, attn_q_norm_max_all,
                 attn_k_norm_std_all, attn_k_norm_max_all,
                 attn_logit_mean_all, attn_logit_std_all,
                 attn_softmax_top1_mean_all, attn_softmax_top1_max_all,
                 attn_logit_gap_mean_all, attn_logit_gap_max_all,
                 attn_softmax_entropy_mean_all,
                 attn_softmax_entropy_min_all,
                 attn_o_input_norm_max_all,
                 attn_o_output_norm_max_all) = scan_ys[
                    _scan_offset:_scan_offset + 52]
                _scan_offset += 52
            total_aux = (attn_auxes + rst_auxes).mean()

        x_pre_final_norm = x
        x = self.norm(x)
        def _attn_core_mean(idx):
            return attn_split_core_all[:, idx].mean()

        def _attn_core_max(idx):
            return attn_split_core_all[:, idx].max()

        def _select_mean(diag_all, idx):
            return diag_all[:, idx].mean()

        def _select_min(diag_all, idx):
            return diag_all[:, idx].min()

        def _select_max(diag_all, idx):
            return diag_all[:, idx].max()

        def _exposure_mean(diag_all, idx):
            return diag_all[:, idx].mean()

        def _exposure_min(diag_all, idx):
            return diag_all[:, idx].min()

        def _exposure_max(diag_all, idx):
            return diag_all[:, idx].max()

        def _sparsity_mean(diag_all, idx):
            return diag_all[:, idx].mean()

        def _page_mean(diag_all, name):
            return diag_all[:, PAGE_DIAG_INDEX[name]].mean()

        result = {
            'aux_loss': total_aux,
            'soft_gate_T': jnp.asarray(soft_gate_T_qk, dtype=jnp.float32),
            'soft_gate_T_qk': jnp.asarray(soft_gate_T_qk, dtype=jnp.float32),
            'soft_gate_T_v': jnp.asarray(soft_gate_T_v, dtype=jnp.float32),
            'soft_gate_T_rst': jnp.asarray(soft_gate_T_rst, dtype=jnp.float32),
            'soft_gate_boundary_power': jnp.asarray(
                soft_gate_boundary_power, dtype=jnp.float32),
            'soft_gate_boundary_power_final': jnp.asarray(
                soft_gate_boundary_power_final, dtype=jnp.float32),
            'execution_prune_eps': jnp.asarray(execution_prune_eps, dtype=jnp.float32),
            'execution_gate_mass_retained': (
                (attn_den_cost_mean_all.mean() + rst_den_cost_mean_all.mean())
                / jnp.maximum(attn_current_cost_mean_all.mean()
                              + rst_current_cost_mean_all.mean(), 1.0e-8)),
            'execution_estimated_compute_frac': (
                (attn_active_n_mean_all.mean() + rst_active_n_mean_all.mean())
                / jnp.maximum(jnp.float32(self.n_qk + self.n_v)
                              + jnp.float32(n_rst_eff), 1.0)),
            'execution_prune_gate_den_mean': (
                attn_den_cost_mean_all.mean() + rst_den_cost_mean_all.mean()) / 2.0,
            'execution_prune_gate_den_min': jnp.minimum(
                attn_den_cost_mean_all.min(), rst_den_cost_mean_all.min()),
            'execution_prune_no_active_frac': (
                (jnp.mean((attn_qk_active_all <= 0.0).astype(jnp.float32))
                 + jnp.mean((attn_v_active_all <= 0.0).astype(jnp.float32))) / 2.0
                + jnp.mean((rst_active_all <= 0.0).astype(jnp.float32))) / 2.0,
            'execution_prune_unpruned_gate_den_mean': (
                attn_current_cost_mean_all.mean() + rst_current_cost_mean_all.mean()) / 2.0,
            'attn_aux': attn_auxes.mean(),
            'rst_aux': rst_auxes.mean(),
            'edge_margin_stat_qk': (
                attn_qk_edge_margin_stat_all.mean()),
            'edge_margin_stat_v': (
                attn_v_edge_margin_stat_all.mean()),
            'edge_margin_stat_rst': (
                rst_edge_margin_stat_all.mean()),

            'rst_active': rst_active_all.mean(),
            'rst_raw_gate_max': rst_raw_gmax_all.mean(),
            'rst_gate_sum': rst_gsum_all.mean(),
            'rst_active_n_mean': rst_active_n_mean_all.mean(),
            'rst_strong': rst_strong_all.mean(),
            'rst_positive_margin_mean_active': (
                rst_positive_margin_active_all.mean()),

            'attn_qk_active': attn_qk_active_all.mean(),
            'attn_q_active': _attn_core_mean(ATTN_SPLIT_Q_ACTIVE_FRAC),
            'attn_k_active': _attn_core_mean(ATTN_SPLIT_K_ACTIVE_FRAC),
            'attn_v_active': attn_v_active_all.mean(),
            'attn_raw_gate_max': attn_raw_gmax_all.mean(),
            'attn_gate_sum': attn_gsum_all.mean(),
            'attn_active_n_mean': attn_active_n_mean_all.mean(),
            'attn_q_active_n_mean': _attn_core_mean(ATTN_SPLIT_Q_ACTIVE_N_MEAN),
            'attn_k_active_n_mean': _attn_core_mean(ATTN_SPLIT_K_ACTIVE_N_MEAN),
            'attn_strong': attn_strong_all.mean(),
            'attn_qk_strong': attn_qk_strong_all.mean(),
            'attn_v_strong': attn_v_strong_all.mean(),
            'attn_qk_positive_margin_mean_active': (
                attn_qk_positive_margin_active_all.mean()),
            'attn_v_positive_margin_mean_active': (
                attn_v_positive_margin_active_all.mean()),

            'rst_op_key_norm': rst_op_key_n_all.mean(),
            'rst_read_norm': k_read_n_all.mean(),
            'rst_write_norm': k_write_n_all.mean(),

            'rst_out_norm': rst_out_norm_all.mean(),
            'attn_out_norm': attn_out_norm_all.mean(),
            'attn_tau_mean': attn_tau_mean_all.mean(),
            'rst_tau_mean': rst_tau_mean_all.mean(),
            'attn_tau_abs_mean': attn_tau_abs_all.mean(),
            'rst_tau_abs_mean': rst_tau_abs_all.mean(),
            'attn_rho_mean': attn_rho_mean_all.mean(),
            'attn_rho_std': attn_rho_std_all.mean(),
            'attn_rho_max': attn_rho_max_all.max(),
            'attn_tau_min': attn_tau_floor_mean_all.min(),
            'attn_tau_max': attn_tau_min_hit_frac_all.max(),
            'attn_raw_tau_mean': attn_tau_direct_mean_all.mean(),
            'attn_raw_tau_min': attn_tau_direct_min_all.min(),
            'attn_raw_tau_max': attn_tau_direct_max_all.max(),
            'attn_selection_margin_mean': attn_selection_margin_mean_all.mean(),
            'attn_positive_margin_mean': attn_positive_margin_mean_all.mean(),
            'attn_positive_margin_max': attn_positive_margin_max_all.max(),
            'attn_qk_rho_mean': _select_mean(
                attn_qk_select_diag_all, SELECT_RHO_MEAN),
            'attn_qk_rho_std': _select_mean(
                attn_qk_select_diag_all, SELECT_RHO_STD),
            'attn_qk_score_std': _select_mean(
                attn_qk_select_diag_all, SELECT_RHO_STD),
            'attn_qk_rho_max': _select_max(
                attn_qk_select_diag_all, SELECT_RHO_MAX),
            'attn_qk_tau_mean': _select_mean(
                attn_qk_select_diag_all, SELECT_TAU_MEAN),
            'attn_qk_tau_min': _select_min(
                attn_qk_select_diag_all, SELECT_TAU_MIN),
            'attn_qk_tau_max': _select_max(
                attn_qk_select_diag_all, SELECT_TAU_MAX),
            'attn_qk_raw_tau_mean': _select_mean(
                attn_qk_select_diag_all, SELECT_RAW_TAU_MEAN),
            'attn_qk_raw_tau_min': _select_min(
                attn_qk_select_diag_all, SELECT_RAW_TAU_MIN),
            'attn_qk_raw_tau_max': _select_max(
                attn_qk_select_diag_all, SELECT_RAW_TAU_MAX),
            'attn_qk_selection_margin_mean': _select_mean(
                attn_qk_select_diag_all, SELECT_SELECTION_MARGIN_MEAN),
            'attn_qk_positive_margin_mean': _select_mean(
                attn_qk_select_diag_all, SELECT_POSITIVE_MARGIN_MEAN),
            'attn_qk_positive_margin_max': _select_max(
                attn_qk_select_diag_all, SELECT_POSITIVE_MARGIN_MAX),
            'attn_v_rho_mean': _select_mean(
                attn_v_select_diag_all, SELECT_RHO_MEAN),
            'attn_v_rho_std': _select_mean(
                attn_v_select_diag_all, SELECT_RHO_STD),
            'attn_v_score_std': _select_mean(
                attn_v_select_diag_all, SELECT_RHO_STD),
            'attn_v_rho_max': _select_max(
                attn_v_select_diag_all, SELECT_RHO_MAX),
            'attn_v_tau_mean': _select_mean(
                attn_v_select_diag_all, SELECT_TAU_MEAN),
            'attn_v_tau_min': _select_min(
                attn_v_select_diag_all, SELECT_TAU_MIN),
            'attn_v_tau_max': _select_max(
                attn_v_select_diag_all, SELECT_TAU_MAX),
            'attn_v_raw_tau_mean': _select_mean(
                attn_v_select_diag_all, SELECT_RAW_TAU_MEAN),
            'attn_v_raw_tau_min': _select_min(
                attn_v_select_diag_all, SELECT_RAW_TAU_MIN),
            'attn_v_raw_tau_max': _select_max(
                attn_v_select_diag_all, SELECT_RAW_TAU_MAX),
            'attn_v_selection_margin_mean': _select_mean(
                attn_v_select_diag_all, SELECT_SELECTION_MARGIN_MEAN),
            'attn_v_positive_margin_mean': _select_mean(
                attn_v_select_diag_all, SELECT_POSITIVE_MARGIN_MEAN),
            'attn_v_positive_margin_max': _select_max(
                attn_v_select_diag_all, SELECT_POSITIVE_MARGIN_MAX),
            'rst_rho_mean': rst_rho_mean_all.mean(),
            'rst_rho_std': rst_rho_std_all.mean(),
            'rst_rho_max': rst_rho_max_all.max(),
            'rst_tau_min': rst_tau_floor_mean_all.min(),
            'rst_tau_max': rst_tau_min_hit_frac_all.max(),
            'rst_raw_tau_mean': rst_tau_direct_mean_all.mean(),
            'rst_raw_tau_min': rst_tau_direct_min_all.min(),
            'rst_raw_tau_max': rst_tau_direct_max_all.max(),
            'rst_selection_margin_mean': rst_selection_margin_mean_all.mean(),
            'rst_positive_margin_mean': rst_positive_margin_mean_all.mean(),
            'rst_positive_margin_max': rst_positive_margin_max_all.max(),
            'attn_soft_exposure_mean': attn_angular_exposure_mean_all.mean(),
            'attn_soft_exposure_min': attn_angular_exposure_min_all.min(),
            'attn_soft_exposure_max': attn_angular_exposure_max_all.max(),
            'attn_soft_dead_frac_eps_1e_6': attn_dead_exposure_frac_all.mean(),
            'attn_soft_dead_frac_eps_1e_5': attn_weak_exposure_frac_all.mean(),
            'attn_soft_dead_frac_eps_1e_4': attn_dead_exposure_target_all.mean(),
            'attn_qk_soft_exposure_mean': _exposure_mean(
                attn_qk_exposure_diag_all, DEAD_EXPOSURE_MEAN),
            'attn_qk_soft_exposure_min': _exposure_min(
                attn_qk_exposure_diag_all, DEAD_EXPOSURE_MIN),
            'attn_qk_soft_exposure_max': _exposure_max(
                attn_qk_exposure_diag_all, DEAD_EXPOSURE_MAX),
            'attn_qk_soft_dead_frac_eps_1e_6': _exposure_mean(
                attn_qk_exposure_diag_all, DEAD_EXPOSURE_DEAD_FRAC),
            'attn_qk_soft_dead_frac_eps_1e_5': _exposure_mean(
                attn_qk_exposure_diag_all, DEAD_EXPOSURE_WEAK_FRAC),
            'attn_qk_soft_dead_frac_eps_1e_4': _exposure_mean(
                attn_qk_exposure_diag_all, DEAD_EXPOSURE_TARGET),
            'attn_v_soft_exposure_mean': _exposure_mean(
                attn_v_exposure_diag_all, DEAD_EXPOSURE_MEAN),
            'attn_v_soft_exposure_min': _exposure_min(
                attn_v_exposure_diag_all, DEAD_EXPOSURE_MIN),
            'attn_v_soft_exposure_max': _exposure_max(
                attn_v_exposure_diag_all, DEAD_EXPOSURE_MAX),
            'attn_v_soft_dead_frac_eps_1e_6': _exposure_mean(
                attn_v_exposure_diag_all, DEAD_EXPOSURE_DEAD_FRAC),
            'attn_v_soft_dead_frac_eps_1e_5': _exposure_mean(
                attn_v_exposure_diag_all, DEAD_EXPOSURE_WEAK_FRAC),
            'attn_v_soft_dead_frac_eps_1e_4': _exposure_mean(
                attn_v_exposure_diag_all, DEAD_EXPOSURE_TARGET),
            'rst_soft_exposure_mean': rst_angular_exposure_mean_all.mean(),
            'rst_soft_exposure_min': rst_angular_exposure_min_all.min(),
            'rst_soft_exposure_max': rst_angular_exposure_max_all.max(),
            'rst_soft_dead_frac_eps_1e_6': rst_dead_exposure_frac_all.mean(),
            'rst_soft_dead_frac_eps_1e_5': rst_weak_exposure_frac_all.mean(),
            'rst_soft_dead_frac_eps_1e_4': rst_dead_exposure_target_all.mean(),
            'attn_qk_op_key_norm_mean': (
                attn_qk_op_key_n_mean_all.mean()),
            'attn_qk_op_key_norm_min': (
                attn_qk_op_key_n_min_all.min()),
            'attn_qk_op_key_norm_std': (
                attn_qk_op_key_n_std_all.mean()),
            'attn_v_op_key_norm_mean': attn_v_op_key_n_mean_all.mean(),
            'attn_v_op_key_norm_min': attn_v_op_key_n_min_all.min(),
            'attn_v_op_key_norm_std': attn_v_op_key_n_std_all.mean(),
            'rst_op_key_norm_mean': rst_op_key_n_all.mean(),
            'rst_op_key_norm_min': rst_op_key_n_min_all.min(),
            'rst_op_key_norm_std': rst_op_key_n_std_all.mean(),

            # Dead-only penalty is separate from aux and weighted in train_jax.
            # Mean across layers so the training weight is layer-count-agnostic.
            'attn_dead_penalty': attn_dead_penalty_all.mean(),
            'rst_dead_penalty': rst_dead_penalty_all.mean(),
            'dead_penalty': (attn_dead_penalty_all.mean()
                             + rst_dead_penalty_all.mean()),
            'attn_dead_count': attn_dead_count_all.mean(),
            'attn_qk_dead_penalty': _attn_core_mean(
                ATTN_SPLIT_QK_DEAD_PENALTY),
            'attn_v_dead_penalty': _attn_core_mean(
                ATTN_SPLIT_V_DEAD_PENALTY),
            'attn_qk_dead_count': _attn_core_mean(
                ATTN_SPLIT_QK_DEAD_COUNT),
            'attn_v_dead_count': _attn_core_mean(
                ATTN_SPLIT_V_DEAD_COUNT),
            'rst_dead_count': rst_dead_count_all.mean(),

            'per_layer_attn_out_norm': attn_out_norm_all,
            'per_layer_rst_out_norm': rst_out_norm_all,
            # Per-layer direct tau stacks retained for diagnostics.
            # Shapes: attn [L, B, S, 3], RST [L, B, S, 1].
            'attn_tau_direct': attn_tau_direct_all,
            'rst_tau_direct': rst_tau_direct_all,
            'attn_no_active_direct': jax.lax.stop_gradient(
                attn_no_active_direct_all),
            'rst_no_active_direct': jax.lax.stop_gradient(
                rst_no_active_direct_all),
            # v4164 aliases: old gate tuple slots carry paper
            # execution_weight statistics; denominator slots carry
            # admission-only sums.
            'attn_int_max': attn_int_max_all.max(),
            'attn_qk_int_max': _attn_core_max(ATTN_SPLIT_QK_INT_MAX),
            'attn_v_int_max': _attn_core_max(ATTN_SPLIT_V_INT_MAX),
            'rst_int_max': rst_int_max_all.max(),
            'attn_qk_raw_gate_max': _attn_core_mean(
                ATTN_SPLIT_QK_RAW_GATE_MAX),
            'attn_v_raw_gate_max': _attn_core_mean(
                ATTN_SPLIT_V_RAW_GATE_MAX),
            'attn_qk_gate_sum': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_SUM),
            'attn_v_gate_sum': _attn_core_mean(
                ATTN_SPLIT_V_GATE_SUM),
            'attn_qk_active_n_mean': _attn_core_mean(
                ATTN_SPLIT_QK_ACTIVE_N_MEAN),
            'attn_v_active_n_mean': _attn_core_mean(
                ATTN_SPLIT_V_ACTIVE_N_MEAN),
            'attn_qk_tau_abs_mean': _attn_core_mean(
                ATTN_SPLIT_QK_TAU_ABS_MEAN),
            'attn_v_tau_abs_mean': _attn_core_mean(
                ATTN_SPLIT_V_TAU_ABS_MEAN),
            'attn_gate_den_sum_mean': attn_den_cost_mean_all.mean(),
            'attn_qk_gate_den_sum_mean': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_DEN_SUM_MEAN),
            'attn_v_gate_den_sum_mean': _attn_core_mean(
                ATTN_SPLIT_V_GATE_DEN_SUM_MEAN),
            'rst_gate_den_sum_mean': rst_den_cost_mean_all.mean(),
            'attn_gate_eff_n': attn_gate_eff_n_all.mean(),
            'attn_gate_eff_ratio': attn_gate_eff_ratio_all.mean(),
            'attn_top1_gate_frac': attn_top1_gate_frac_all.mean(),
            'attn_top1_gate_frac_max': attn_top1_gate_frac_max_all.max(),
            'attn_qk_gate_eff_n': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_EFF_N),
            'attn_v_gate_eff_n': _attn_core_mean(
                ATTN_SPLIT_V_GATE_EFF_N),
            'attn_qk_gate_eff_ratio': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_EFF_RATIO),
            'attn_v_gate_eff_ratio': _attn_core_mean(
                ATTN_SPLIT_V_GATE_EFF_RATIO),
            'attn_qk_top1_gate_frac': _attn_core_mean(
                ATTN_SPLIT_QK_TOP1_GATE_FRAC),
            'attn_v_top1_gate_frac': _attn_core_mean(
                ATTN_SPLIT_V_TOP1_GATE_FRAC),
            'attn_qk_top1_gate_frac_max': _attn_core_max(
                ATTN_SPLIT_QK_TOP1_GATE_FRAC_MAX),
            'attn_v_top1_gate_frac_max': _attn_core_max(
                ATTN_SPLIT_V_TOP1_GATE_FRAC_MAX),
            'rst_gate_eff_n': rst_gate_eff_n_all.mean(),
            'rst_gate_eff_ratio': rst_gate_eff_ratio_all.mean(),
            'rst_top1_gate_frac': rst_top1_gate_frac_all.mean(),
            'rst_top1_gate_frac_max': rst_top1_gate_frac_max_all.max(),
            'admission_den_sum': (
                attn_den_cost_mean_all.mean() + rst_den_cost_mean_all.mean()),
            'attn_admission_den_sum': attn_den_cost_mean_all.mean(),
            'attn_qk_admission_den_sum': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_DEN_SUM_MEAN),
            'attn_v_admission_den_sum': _attn_core_mean(
                ATTN_SPLIT_V_GATE_DEN_SUM_MEAN),
            'rst_admission_den_sum': rst_den_cost_mean_all.mean(),
            'execution_mass_sum': attn_gsum_all.mean() + rst_gsum_all.mean(),
            'attn_execution_mass_sum': attn_gsum_all.mean(),
            'attn_qk_execution_mass_sum': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_SUM),
            'attn_v_execution_mass_sum': _attn_core_mean(
                ATTN_SPLIT_V_GATE_SUM),
            'rst_execution_mass_sum': rst_gsum_all.mean(),
            'drive_mean': (
                attn_current_cost_mean_all.mean()
                + rst_current_cost_mean_all.mean()) / jnp.float32(2.0),
            'attn_drive_mean': attn_current_cost_mean_all.mean(),
            'attn_qk_drive_mean': _attn_core_mean(
                ATTN_SPLIT_QK_DRIVE_MEAN),
            'attn_v_drive_mean': _attn_core_mean(
                ATTN_SPLIT_V_DRIVE_MEAN),
            'rst_drive_mean': rst_current_cost_mean_all.mean(),
            'drive_max': jnp.maximum(
                attn_int_max_all.max(), rst_int_max_all.max()),
            'attn_drive_max': attn_int_max_all.max(),
            'attn_qk_drive_max': _attn_core_max(
                ATTN_SPLIT_QK_INT_MAX),
            'attn_v_drive_max': _attn_core_max(
                ATTN_SPLIT_V_INT_MAX),
            'rst_drive_max': rst_int_max_all.max(),
            'execution_eff_n': (
                attn_gate_eff_n_all.mean() + rst_gate_eff_n_all.mean()
            ) / jnp.float32(2.0),
            'attn_execution_eff_n': attn_gate_eff_n_all.mean(),
            'attn_qk_execution_eff_n': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_EFF_N),
            'attn_v_execution_eff_n': _attn_core_mean(
                ATTN_SPLIT_V_GATE_EFF_N),
            'rst_execution_eff_n': rst_gate_eff_n_all.mean(),
            'execution_top1_frac': (
                attn_top1_gate_frac_all.mean()
                + rst_top1_gate_frac_all.mean()) / jnp.float32(2.0),
            'execution_top1_frac_max': jnp.maximum(
                attn_top1_gate_frac_all.max(),
                rst_top1_gate_frac_all.max()),
            'attn_execution_top1_frac': attn_top1_gate_frac_all.mean(),
            'attn_execution_top1_frac_max': attn_top1_gate_frac_all.max(),
            'attn_qk_execution_top1_frac': _attn_core_mean(
                ATTN_SPLIT_QK_TOP1_GATE_FRAC),
            'attn_qk_execution_top1_frac_max': _attn_core_max(
                ATTN_SPLIT_QK_TOP1_GATE_FRAC_MAX),
            'attn_v_execution_top1_frac': _attn_core_mean(
                ATTN_SPLIT_V_TOP1_GATE_FRAC),
            'attn_v_execution_top1_frac_max': _attn_core_max(
                ATTN_SPLIT_V_TOP1_GATE_FRAC_MAX),
            'rst_execution_top1_frac': rst_top1_gate_frac_all.mean(),
            'rst_execution_top1_frac_max': rst_top1_gate_frac_all.max(),
            # Always-on output diagnostics: cheap scalar reductions used by train logs.
            # These are kept outside the analysis-only block so out_diag never falls
            # back to misleading zeros during normal training.
            'residual_norm': jnp.linalg.norm(x, axis=-1).mean(),
            'residual_norm_max': jnp.linalg.norm(x, axis=-1).max(),
            'token_emb_norm': jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean(),
            'token_emb_norm_max': jnp.linalg.norm(self.token_emb.embedding, axis=-1).max(),
        }
        for _prefix, _diag_all in (
                ('attn_qk', attn_qk_sparsity_diag_all),
                ('attn_v', attn_v_sparsity_diag_all),
                ('attn_q', attn_q_sparsity_diag_all),
                ('attn_k', attn_k_sparsity_diag_all),
                ('rst', rst_sparsity_diag_all)):
            result.update({
                f'{_prefix}_{_name}': _sparsity_mean(_diag_all, _idx)
                for _idx, _name in enumerate(GATE_SPARSITY_DIAG_NAMES)
            })
        qk_pages_enabled = _operator_pages_pool_enabled_static(
            self.operator_pages_enabled, self.operator_pages_pools, 'qk')
        v_pages_enabled = _operator_pages_pool_enabled_static(
            self.operator_pages_enabled, self.operator_pages_pools, 'v')
        rst_pages_enabled = _operator_pages_pool_enabled_static(
            self.operator_pages_enabled, self.operator_pages_pools, 'rst')
        result.update(_operator_page_static_result(
            'attn_qk', qk_pages_enabled, self.n_qk,
            self.operator_page_size_qk, self.operator_page_capacity_qk,
            self.operator_page_fallback_pages,
            self.operator_page_random_pages))
        result.update(_operator_page_static_result(
            'attn_v', v_pages_enabled, self.n_v,
            self.operator_page_size_v, self.operator_page_capacity_v,
            self.operator_page_fallback_pages,
            self.operator_page_random_pages))
        result.update(_operator_page_static_result(
            'rst', rst_pages_enabled, n_rst_eff,
            self.operator_page_size_rst, self.operator_page_capacity_rst,
            self.operator_page_fallback_pages,
            self.operator_page_random_pages))
        if not self.is_initializing():
            for _prefix, _enabled, _diag_all in (
                    ('attn_qk', qk_pages_enabled, attn_qk_page_diag_all),
                    ('attn_v', v_pages_enabled, attn_v_page_diag_all),
                    ('rst', rst_pages_enabled, rst_page_diag_all)):
                if _enabled:
                    result.update({
                        f'{_prefix}_{_name}': _page_mean(_diag_all, _name)
                        for _name in PAGE_DIAG_NAMES
                    })
        result['selected_page_count'] = (
            result['attn_qk_page_count_effective']
            + result['attn_v_page_count_effective']
            + result['rst_page_count_effective'])
        result['estimated_compute_frac_page'] = (
            result['attn_qk_estimated_compute_frac_page']
            + result['attn_v_estimated_compute_frac_page']
            + result['rst_estimated_compute_frac_page']
        ) / jnp.float32(3.0)
        result['rst_candidate_den_mean'] = jnp.where(
            rst_pages_enabled,
            result['rst_gate_den_sum_mean'],
            result['rst_candidate_den_mean'])
        result['rst_candidate_execution_mass'] = jnp.where(
            rst_pages_enabled,
            result['rst_execution_mass_sum'],
            result['rst_candidate_execution_mass'])
        result['attn_qk_candidate_den_mean'] = jnp.where(
            qk_pages_enabled,
            result['attn_qk_gate_den_sum_mean'],
            result['attn_qk_candidate_den_mean'])
        result['attn_qk_candidate_execution_mass'] = jnp.where(
            qk_pages_enabled,
            result['attn_qk_execution_mass_sum'],
            result['attn_qk_candidate_execution_mass'])
        result['attn_v_candidate_den_mean'] = jnp.where(
            v_pages_enabled,
            result['attn_v_gate_den_sum_mean'],
            result['attn_v_candidate_den_mean'])
        result['attn_v_candidate_execution_mass'] = jnp.where(
            v_pages_enabled,
            result['attn_v_execution_mass_sum'],
            result['attn_v_candidate_execution_mass'])
        page_enabled_count = (
            jnp.float32(1.0 if qk_pages_enabled else 0.0)
            + jnp.float32(1.0 if v_pages_enabled else 0.0)
            + jnp.float32(1.0 if rst_pages_enabled else 0.0))
        page_cost_base = (
            jnp.float32(1.0 if qk_pages_enabled else 0.0)
            * result['attn_qk_candidate_frac']
            + jnp.float32(1.0 if v_pages_enabled else 0.0)
            * result['attn_v_candidate_frac']
            + jnp.float32(1.0 if rst_pages_enabled else 0.0)
            * result['rst_candidate_frac'])
        page_cost_base = page_cost_base / jnp.maximum(
            page_enabled_count, jnp.float32(1.0))
        operator_page_cost = (
            jnp.asarray(self.operator_page_cost_weight, dtype=jnp.float32)
            * page_cost_base)
        result['operator_page_cost'] = operator_page_cost
        result['aux_loss'] = result['aux_loss'] + operator_page_cost
        if analysis and not self.is_initializing():
            _residual_norm = jnp.linalg.norm(x, axis=-1).mean()
            _emb_norm = jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean()
            _o_proj_norm = jnp.linalg.norm(
                stacked['attn']['expand_O']['kernel'], axis=(-2, -1)).mean()
            _attn_logit_max_layer = jnp.argmax(attn_logit_max_all)
            result.update({
                'rst_margin_band': rst_margin_band_all.mean(),
                'attn_qk_margin_band': attn_qk_margin_band_all.mean(),
                'attn_v_margin_band': attn_v_margin_band_all.mean(),
                'attn_tau_std': attn_tau_std_all.mean(axis=0),
                'rst_tau_std': rst_tau_std_all.mean(),
                'attn_tau_kernel_norm': attn_tau_kernel_norm_all.mean(),
                'rst_tau_kernel_norm': rst_tau_kernel_norm_all.mean(),
                'attn_margin_band_wide': attn_margin_band_wide_all.mean(),
                'rst_margin_band_wide': rst_margin_band_wide_all.mean(),
                'attn_margin_band_mid': attn_margin_band_mid_all.mean(),
                'rst_margin_band_mid': rst_margin_band_mid_all.mean(),
                'attn_rho_skew': attn_skew_all.mean(),
                'rst_rho_skew': rst_skew_all.mean(),
                'attn_active_per_token_std': attn_apt_std_all.mean(),
                'rst_active_per_token_std': rst_apt_std_all.mean(),
                'attn_gate_entropy': attn_entropy_all.mean(),
                'rst_gate_entropy': rst_entropy_all.mean(),
                'attn_gate_den_sum': attn_den_cost_all.mean(),
                'rst_gate_den_sum': rst_den_cost_all.mean(),
                'attn_qk_op_key_norm_max': (
                    attn_qk_op_key_n_max_all.max()),
                'attn_v_op_key_norm_max': attn_v_op_key_n_max_all.max(),
                'rst_op_key_norm_max': rst_op_key_n_max_all.max(),
                'attn_rho_kurt': attn_rho_kurt_all.mean(),
                'rst_rho_kurt': rst_rho_kurt_all.mean(),
                'attn_qk_raw_norm': attn_qk_raw_norm_all.mean(),
                'attn_v_raw_norm': attn_v_raw_norm_all.mean(),
                'rst_raw_out_norm': rst_raw_out_norm_all.mean(),
                'residual_norm': _residual_norm,
                'token_emb_norm_analysis': _emb_norm,
                'o_proj_norm': _o_proj_norm,
                'q_norm': attn_q_norm_all.mean(),
                'k_norm': attn_k_norm_all.mean(),
                'v_norm': attn_v_norm_dbg_all.mean(),
                'attn_logit_max_mean': attn_logit_max_all.mean(),
                'o_input_norm': attn_o_input_norm_all.mean(),
                'attn_q_norm_mean': attn_q_norm_all.mean(),
                'attn_q_norm_std': attn_q_norm_std_all.mean(),
                'attn_q_norm_max': attn_q_norm_max_all.max(),
                'attn_k_norm_mean': attn_k_norm_all.mean(),
                'attn_k_norm_std': attn_k_norm_std_all.mean(),
                'attn_k_norm_max': attn_k_norm_max_all.max(),
                'attn_logit_mean': attn_logit_mean_all.mean(),
                'attn_logit_std': attn_logit_std_all.mean(),
                'attn_logit_max': attn_logit_max_all.max(),
                'attn_logit_max_layer': _attn_logit_max_layer,
                'attn_softmax_top1_mean': attn_softmax_top1_mean_all.mean(),
                'attn_softmax_top1_max': attn_softmax_top1_max_all.max(),
                'attn_logit_gap_top1_top2_mean': attn_logit_gap_mean_all.mean(),
                'attn_logit_gap_top1_top2_max': attn_logit_gap_max_all.max(),
                'attn_softmax_entropy_mean': attn_softmax_entropy_mean_all.mean(),
                'attn_softmax_entropy_min': attn_softmax_entropy_min_all.min(),
                'attn_o_input_norm_mean': attn_o_input_norm_all.mean(),
                'attn_o_input_norm_max': attn_o_input_norm_max_all.max(),
                'attn_o_output_norm_mean': attn_out_norm_all.mean(),
                'attn_o_output_norm_max': attn_o_output_norm_max_all.max(),
                'per_layer_attn_q_norm_mean': attn_q_norm_all,
                'per_layer_attn_q_norm_std': attn_q_norm_std_all,
                'per_layer_attn_q_norm_max': attn_q_norm_max_all,
                'per_layer_attn_k_norm_mean': attn_k_norm_all,
                'per_layer_attn_k_norm_std': attn_k_norm_std_all,
                'per_layer_attn_k_norm_max': attn_k_norm_max_all,
                'per_layer_attn_logit_mean': attn_logit_mean_all,
                'per_layer_attn_logit_std': attn_logit_std_all,
                'per_layer_attn_logit_max': attn_logit_max_all,
                'per_layer_attn_softmax_top1_mean': attn_softmax_top1_mean_all,
                'per_layer_attn_softmax_top1_max': attn_softmax_top1_max_all,
                'per_layer_attn_logit_gap_top1_top2_mean': attn_logit_gap_mean_all,
                'per_layer_attn_logit_gap_top1_top2_max': attn_logit_gap_max_all,
                'per_layer_attn_softmax_entropy_mean': attn_softmax_entropy_mean_all,
                'per_layer_attn_softmax_entropy_min': attn_softmax_entropy_min_all,
                'per_layer_attn_o_input_norm_mean': attn_o_input_norm_all,
                'per_layer_attn_o_input_norm_max': attn_o_input_norm_max_all,
                'per_layer_attn_o_output_norm_mean': attn_out_norm_all,
                'per_layer_attn_o_output_norm_max': attn_o_output_norm_max_all,
                'attn_int_cap_frac': attn_int_cap_frac_all.mean(),
                'rst_int_cap_frac': rst_int_cap_frac_all.mean(),
            })
        if labels is not None:
            embedding_matrix = self.token_emb.embedding
            shift_x = x[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = (shift_labels != -100)

            @jax.checkpoint
            def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                logits = x_chunk @ emb.T
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                safe = jnp.where(vmask, labs, 0)
                tl = -jnp.take_along_axis(
                    log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
                per_token_ce = tl * vmask            # [B, S-1], 0 on invalid
                loss = per_token_ce.sum() / (vmask.sum() + 1e-8)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum((preds == labs) & vmask)
                logits_f = logits.astype(jnp.float32)
                logit_abs_max = jnp.max(jnp.abs(logits_f))
                logit_norm_mean = jnp.linalg.norm(logits_f, axis=-1).mean()
                logit_mean = logits_f.mean()
                logit_std = logits_f.std()
                return (loss, per_token_ce, correct, jnp.sum(vmask),
                        logit_abs_max, logit_norm_mean, logit_mean, logit_std)

            (loss, per_token_ce, correct, valid_count,
             logit_abs_max, logit_norm_mean, logit_mean, logit_std) = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
            result['logit_max'] = logit_abs_max
            result['logit_norm_mean'] = logit_norm_mean
            result['logit_mean'] = logit_mean
            result['logit_std'] = logit_std
            result['per_token_ce'] = per_token_ce
            result['valid_mask'] = valid_mask
        else:
            result['logits'] = self.token_emb.attend(x)

        return result

    def get_config(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        cfg = {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'd_route': self.d_route,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_rst': n_rst_eff,
            'n_know': n_rst_eff,
            'operator_pages_enabled': self.operator_pages_enabled,
            'operator_pages_pools': self.operator_pages_pools,
            'operator_page_size_qk': self.operator_page_size_qk,
            'operator_page_size_v': self.operator_page_size_v,
            'operator_page_size_rst': self.operator_page_size_rst,
            'operator_page_capacity_qk': self.operator_page_capacity_qk,
            'operator_page_capacity_v': self.operator_page_capacity_v,
            'operator_page_capacity_rst': self.operator_page_capacity_rst,
            'operator_page_microgroup_sequences':
                self.operator_page_microgroup_sequences,
            'operator_page_score_mode': self.operator_page_score_mode,
            'operator_page_fallback_pages': self.operator_page_fallback_pages,
            'operator_page_random_pages': self.operator_page_random_pages,
            'operator_page_cost_weight': self.operator_page_cost_weight,
            'operator_pages_analysis_full_scan':
                self.operator_pages_analysis_full_scan,
        }
        return cfg

    def get_model_info(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        qk_scale, v_scale, rst_scale = _pool_output_scales(
            self.d_model, self.n_layers)
        return [
            f"DAWN-SRW ({self.__version__})",
            f"  d_model={self.d_model}, d_route={self.d_route}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  Attention-QK: {self.n_qk}, Attention-V: {self.n_v}, RST: {n_rst_eff}",
            "  Selection: live-gradient RW operator keys with fixed "
            "RW-matched operator queries",
            "  Pool scales: fixed depth-scaled "
            f"(qk={float(qk_scale):.6g}, v={float(v_scale):.6g}, "
            f"rst={float(rst_scale):.6g})",
            "  Global CEU operator pages: page-routed candidate RW execution "
            "over the live RW-derived operator-key atlas",
            "  Page defaults: all-pool top-k "
            f"(qk page_size={self.operator_page_size_qk}, "
            f"cap={self.operator_page_capacity_qk}, "
            f"v page_size={self.operator_page_size_v}, "
            f"cap={self.operator_page_capacity_v}, "
            f"rst page_size={self.operator_page_size_rst}, "
            f"cap={self.operator_page_capacity_rst}, "
            f"microgroup_sequences={self.operator_page_microgroup_sequences}, "
            f"fallback={self.operator_page_fallback_pages})",
        ]


DAWN = DAWN_SRW_V4167


# ================================================================
# 9. Inference API: KV-cache prefill + decode.
#    Pure functions only. Training code above is untouched.
# ================================================================

def _squeeze_params(params):
    """Remove leading singleton dim from all param arrays.

    Device-replicated checkpoints store params with shape (1, ...).
    Squeeze axis 0 when it is size 1 so indexing and matmul work correctly.
    """
    def _sq(x):
        if hasattr(x, 'ndim') and x.ndim >= 2 and x.shape[0] == 1:
            return x.squeeze(0)
        return x
    return jax.tree.map(_sq, params)


def _angular_execution_kwargs_from_model_cfg(model_cfg):
    """Extract v4167 drive boundary settings for inference."""
    # Pure generation/vectorized helpers do not receive the training schedule,
    # so they use explicit boundary defaults unless the caller supplies current
    # values through model_cfg.
    return {
        'soft_gate_temperature': float(
            model_cfg.get('soft_gate_temperature', 0.07)),
        'soft_gate_boundary_power': float(
            model_cfg.get('soft_gate_boundary_power', 4.0)),
        'admission_den_power': float(model_cfg.get(
            'admission_den_power',
            model_cfg.get('v4164_den_power', model_cfg.get('den_power', 1.0)))),
        'execution_prune_eps': float(model_cfg.get('execution_prune_eps', 0.0)),
        'soft_gate_effective_active_eps': float(
            model_cfg.get('soft_gate_effective_active_eps', 1.0e-6)),
    }


def _tau_init_calibration_scores(params, input_ids, max_tokens=128,
                                 operator_pages_enabled=False,
                                 operator_pages_pools=None,
                                 operator_page_size_qk=128,
                                 operator_page_size_v=128,
                                 operator_page_size_rst=128,
                                 operator_page_capacity_qk=8,
                                 operator_page_capacity_v=8,
                                 operator_page_capacity_rst=32,
                                 operator_page_microgroup_sequences=2,
                                 operator_page_score_mode='maxmean',
                                 operator_page_fallback_pages=0,
                                 operator_page_random_pages=0):
    """Sample fresh-init cosine scores without changing forward semantics.

    The sample uses the first block's freshly initialized normalized route
    states and the shared v4167 router/pool parameters. Rho follows the
    train path exactly: RW-matched operator queries against live-gradient
    RW-derived operator keys.
    """
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError(
            f"tau init calibration max_tokens must be > 0, got {max_tokens}")

    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim != 2:
        raise ValueError(
            "tau init calibration input_ids must be rank 2 [batch, seq], "
            f"got shape={input_ids.shape}")
    _, seq_len = input_ids.shape
    total_tokens = int(input_ids.size)
    token_count = min(max_tokens, total_tokens)
    token_idx = (
        jnp.arange(token_count, dtype=jnp.int32) * total_tokens
        // token_count)
    token_ids = input_ids.reshape(-1)[token_idx]
    positions = token_idx % seq_len

    x = (
        params['token_emb']['embedding'][token_ids]
        + params['pos_emb']['embedding'][positions]
    ).astype(jnp.float32)
    block0 = params['block_0']
    attn_x = _layer_norm(
        x, block0['norm1']['scale'], block0['norm1']['bias'])
    rst_x = _layer_norm(
        x, block0['norm2']['scale'], block0['norm2']['bias'])
    router = params['router']
    attn_read_query = (
        attn_x @ router['proj_attn']['kernel']
        + router['proj_attn']['bias'])
    q_read_query, k_read_query, v_read_query = jnp.split(
        attn_read_query, 3, axis=-1)
    rst_read_query = (
        rst_x @ router['proj_rst']['kernel']
        + router['proj_rst']['bias'])
    h_q = _read_write_operator_query(
        q_read_query, attn_x, router['q_op_write_query_proj'])
    h_k = _read_write_operator_query(
        k_read_query, attn_x, router['k_op_write_query_proj'])
    h_v = _read_write_operator_query(
        v_read_query, attn_x, router['v_op_write_query_proj'])
    h_rst = _read_write_operator_query(
        rst_read_query, rst_x, router['rst_op_write_query_proj'])

    def _selection_rho(h, op_key):
        q_unit = _forward_unit_direction(
            h.astype(jnp.float32)).astype(jnp.bfloat16)
        op_key_unit = op_key.astype(jnp.bfloat16)
        return (q_unit @ op_key_unit.T).astype(jnp.float32)

    pool = params['neuron_pool']
    op_keys = _ensure_pool_operator_keys(pool)
    qk_op_key = op_keys['attn_qk_op_key']
    v_op_key = op_keys['attn_v_op_key']
    rst_op_key = op_keys['rst_op_key']

    qk_pages_enabled = _operator_pages_pool_enabled_static(
        operator_pages_enabled, operator_pages_pools, 'qk')
    v_pages_enabled = _operator_pages_pool_enabled_static(
        operator_pages_enabled, operator_pages_pools, 'v')
    rst_pages_enabled = _operator_pages_pool_enabled_static(
        operator_pages_enabled, operator_pages_pools, 'rst')

    def _with_full_pool_meta(out, name, values, full_pool_size):
        out[name] = values
        out[f'{name}_pages_enabled'] = jnp.float32(0.0)
        out[f'{name}_candidate_valid_count'] = jnp.float32(full_pool_size)
        out[f'{name}_candidate_count'] = jnp.float32(full_pool_size)
        out[f'{name}_full_pool_size'] = jnp.float32(full_pool_size)
        out[f'{name}_candidate_group_count'] = jnp.float32(1.0)

    def _with_page_meta(out, name, rho, valid_mask, page_meta):
        out[name] = rho
        out[f'{name}_valid_mask'] = valid_mask
        out[f'{name}_pages_enabled'] = jnp.float32(1.0)
        out[f'{name}_candidate_valid_count'] = (
            page_meta['candidate_valid_count'])
        out[f'{name}_candidate_count'] = page_meta['candidate_count']
        out[f'{name}_full_pool_size'] = page_meta['full_pool_size']
        out[f'{name}_candidate_group_count'] = (
            page_meta['candidate_group_count'])

    out = {
        'tokens': jnp.asarray(token_count, dtype=jnp.int32),
    }
    if qk_pages_enabled or v_pages_enabled or rst_pages_enabled:
        if token_count == total_tokens:
            page_shape = (int(input_ids.shape[0]), int(seq_len))
        else:
            page_shape = (1, int(token_count))
        h_q_page = h_q.reshape(page_shape + (h_q.shape[-1],))
        h_k_page = h_k.reshape(page_shape + (h_k.shape[-1],))
        h_v_page = h_v.reshape(page_shape + (h_v.shape[-1],))
        h_rst_page = h_rst.reshape(page_shape + (h_rst.shape[-1],))
    else:
        h_q_page = h_k_page = h_v_page = h_rst_page = None

    if qk_pages_enabled:
        qk_page = _select_operator_pages_for_calibration(
            jnp.stack((h_q_page, h_k_page), axis=2),
            qk_op_key,
            'qk-calibration',
            operator_page_size_qk,
            operator_page_capacity_qk,
            operator_page_microgroup_sequences,
            operator_page_score_mode,
            operator_page_fallback_pages,
            operator_page_random_pages,
            paired=True)
        qk_rho = qk_page['rho']
        qk_valid = qk_page['valid_mask']
        _with_page_meta(
            out, 'q', qk_rho[:, :, 0, :], qk_valid[:, :, 0, :], qk_page)
        _with_page_meta(
            out, 'k', qk_rho[:, :, 1, :], qk_valid[:, :, 1, :], qk_page)
    else:
        _with_full_pool_meta(
            out, 'q', _selection_rho(h_q, qk_op_key), qk_op_key.shape[0])
        _with_full_pool_meta(
            out, 'k', _selection_rho(h_k, qk_op_key), qk_op_key.shape[0])

    if v_pages_enabled:
        v_page = _select_operator_pages_for_calibration(
            h_v_page,
            v_op_key,
            'v-calibration',
            operator_page_size_v,
            operator_page_capacity_v,
            operator_page_microgroup_sequences,
            operator_page_score_mode,
            operator_page_fallback_pages,
            operator_page_random_pages,
            paired=False)
        _with_page_meta(
            out, 'v', v_page['rho'], v_page['valid_mask'], v_page)
    else:
        _with_full_pool_meta(
            out, 'v', _selection_rho(h_v, v_op_key), v_op_key.shape[0])

    if rst_pages_enabled:
        rst_page = _select_operator_pages_for_calibration(
            h_rst_page,
            rst_op_key,
            'rst-calibration',
            operator_page_size_rst,
            operator_page_capacity_rst,
            operator_page_microgroup_sequences,
            operator_page_score_mode,
            operator_page_fallback_pages,
            operator_page_random_pages,
            paired=False)
        _with_page_meta(
            out, 'rst', rst_page['rho'], rst_page['valid_mask'], rst_page)
    else:
        _with_full_pool_meta(
            out, 'rst', _selection_rho(h_rst, rst_op_key),
            rst_op_key.shape[0])
    return out


def _angular_relation(h, op_key):
    q = _forward_unit_direction(h.astype(jnp.float32))
    op_key = _forward_unit_direction(op_key.astype(jnp.float32))
    return (q @ op_key.T).astype(jnp.float32)


def _angular_execution(h, op_key, raw_tau, raw_scan_offset=None,
                     soft_gate_temperature=0.07,
                     soft_gate_boundary_power=4.0,
                     execution_prune_eps=0.0,
                     soft_gate_effective_active_eps=1.0e-6):
    rho = _angular_relation(h, op_key)
    tau = _tau_from_param(raw_tau)
    return _compute_admission_drive(
        rho, tau, soft_gate_temperature,
        boundary_power=soft_gate_boundary_power,
        effective_active_eps=soft_gate_effective_active_eps,
        execution_prune_eps=execution_prune_eps)


def _angular_execution_weight(h, op_key, raw_tau, raw_scan_offset=None,
                  soft_gate_temperature=0.07,
                  soft_gate_boundary_power=4.0,
                  execution_prune_eps=0.0,
                  soft_gate_effective_active_eps=1.0e-6):
    """Canonical v4167 execution_weight for non-sharded inference helpers."""
    _, _, _, execution_weight, _ = _angular_execution(
        h, op_key, raw_tau, raw_scan_offset,
        soft_gate_temperature=soft_gate_temperature,
        soft_gate_boundary_power=soft_gate_boundary_power,
        execution_prune_eps=execution_prune_eps,
        soft_gate_effective_active_eps=soft_gate_effective_active_eps)
    return execution_weight.astype(jnp.float32)


def _read_write_attn_operator_queries(
        router_params, x, q_read_query, k_read_query, v_read_query):
    return (
        _read_write_operator_query(
            q_read_query, x, router_params['q_op_write_query_proj']),
        _read_write_operator_query(
            k_read_query, x, router_params['k_op_write_query_proj']),
        _read_write_operator_query(
            v_read_query, x, router_params['v_op_write_query_proj']),
    )


def _read_write_rst_operator_query(router_params, x, rst_read_query):
    return _read_write_operator_query(
        rst_read_query, x, router_params['rst_op_write_query_proj'])


def _split_admission_den_kwargs(angular_execution_kwargs):
    execution_kwargs = dict(angular_execution_kwargs)
    admission_den_power = execution_kwargs.pop('admission_den_power', 1.0)
    admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    return execution_kwargs, admission_den_power


def _srw_inference(x, h, op_key, raw_tau, raw_scan_offset, w_read, w_write,
                   **angular_execution_kwargs):
    """Non-chunked SRW for inference."""
    # v4.1.6.6: inference selection uses op keys; execution uses RW dirs.
    r_n = _forward_unit_direction(w_read.astype(jnp.float32))
    w_n = _forward_unit_direction(w_write.astype(jnp.float32))
    execution_kwargs, admission_den_power = _split_admission_den_kwargs(
        angular_execution_kwargs)
    _, admission, _, execution_weight, _ = _angular_execution(
        h, op_key, raw_tau, raw_scan_offset, **execution_kwargs)

    xr = x.astype(jnp.float32) @ r_n.T
    a = execution_weight * xr
    raw_out = a @ w_n
    admission_den = jnp.power(
        jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0),
        admission_den_power)
    out = raw_out.astype(jnp.float32) / admission_den
    return out.astype(jnp.float32)


def _srw_inference_with_gates(x, h, op_key, raw_tau, raw_scan_offset, w_read,
                              w_write, **angular_execution_kwargs):
    """Like _srw_inference but also returns gate and normalized gate."""
    # v4.1.6.6: analysis selection uses op keys; execution uses RW dirs.
    r_n = _forward_unit_direction(w_read.astype(jnp.float32))
    w_n = _forward_unit_direction(w_write.astype(jnp.float32))
    execution_kwargs, admission_den_power = _split_admission_den_kwargs(
        angular_execution_kwargs)
    _, admission, _, execution_weight, _ = _angular_execution(
        h, op_key, raw_tau, raw_scan_offset, **execution_kwargs)
    admission_den = jnp.power(
        jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0),
        admission_den_power)
    execution_weight_norm = execution_weight / jnp.maximum(
        admission_den, 1e-8)

    xr = x.astype(jnp.float32) @ r_n.T
    a = execution_weight * xr
    raw_out = a @ w_n
    out = raw_out.astype(jnp.float32) / admission_den
    return out.astype(jnp.float32), execution_weight, execution_weight_norm



def _attn_forward_cached(x, pool_params, router_params, expand_O_kernel,
                         n_heads, d_model, n_layers,
                         cache_K, cache_V, cache_len,
                         angular_execution_kwargs=None):
    """Cached attention decode step. x: [B, 1, D]."""
    B = x.shape[0]
    d_head = d_model // n_heads

    if angular_execution_kwargs is None:
        angular_execution_kwargs = {}
    pool_params = _ensure_pool_operator_keys(pool_params)
    qk_norm = pool_params['attn_qk_op_key']
    v_norm = pool_params['attn_v_op_key']
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
    h_Q, h_K, h_V = _read_write_attn_operator_queries(
        router_params, x, h_Q, h_K, h_V)
    tau_all = x @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
    raw_scan_offset_all = jnp.zeros_like(tau_all)

    Q = _srw_inference(x, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                       pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                       **angular_execution_kwargs)
    K_new = _srw_inference(x, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                           **angular_execution_kwargs)
    V_new = _srw_inference(x, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                           pool_params['attn_v_read'], pool_params['attn_v_write'],
                           **angular_execution_kwargs)
    _qk_s, _v_s, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)
    Q = Q * _qk_s
    K_new = K_new * _qk_s
    V_new = V_new * _v_s

    Q = Q.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)
    K_new_h = K_new.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)
    V_new_h = V_new.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)

    cache_K = cache_K.at[:, :, cache_len, :].set(K_new_h[:, :, 0, :])
    cache_V = cache_V.at[:, :, cache_len, :].set(V_new_h[:, :, 0, :])

    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum('bhqd,bhkd->bhqk', Q, cache_K) / scale
    pos_mask = jnp.arange(cache_K.shape[2]) < (cache_len + 1)
    attn_scores = jnp.where(pos_mask[None, None, None, :], attn_scores,
                            jnp.finfo(attn_scores.dtype).min)
    attn_w = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn_w, cache_V)

    out = out.transpose(0, 2, 1, 3).reshape(B, 1, d_model)
    out = out @ expand_O_kernel
    return out, cache_K, cache_V


def _rst_forward_inference(x, pool_params, router_params,
                           d_model=None, n_layers=None,
                           angular_execution_kwargs=None):
    """Inference-only RST Layer forward. No chunking, no LB, no dropout."""
    if angular_execution_kwargs is None:
        angular_execution_kwargs = {}
    pool_params = _ensure_pool_operator_keys(pool_params)
    rst_norm = pool_params['rst_op_key']
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    h = _read_write_rst_operator_query(router_params, x, h)
    tau = x @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
    raw_scan_offset = jnp.zeros_like(tau)
    out = _srw_inference(x, h, rst_norm, tau, raw_scan_offset,
                         pool_params['rst_read'], pool_params['rst_write'],
                         **angular_execution_kwargs)
    if d_model is None or n_layers is None:
        raise ValueError(
            "depth-scaled pool outputs require d_model and n_layers.")
    _, _, rst_scale = _pool_output_scales(d_model, n_layers)
    return out * rst_scale




def prefill(params, model_cfg, input_ids):
    """Run full forward on prompt, populate KV cache.

    Returns: logits [B,S,vocab], cache_K, cache_V [n_layers,B,H,max_seq,d_head], cache_len
    """
    params = _squeeze_params(params)
    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    angular_execution_kwargs = _angular_execution_kwargs_from_model_cfg(model_cfg)
    max_seq = model_cfg['max_seq_len']
    d_head = d_model // n_heads

    pool_params = _pool_params_with_operator_keys(params['neuron_pool'])
    router_params = params['router']
    qk_scale_eff, v_scale_eff, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]

    qk_norm = pool_params['attn_qk_op_key']
    v_norm = pool_params['attn_v_op_key']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    cache_K = jnp.zeros((n_layers, B, n_heads, max_seq, d_head))
    cache_V = jnp.zeros((n_layers, B, n_heads, max_seq, d_head))

    def prefill_layer(carry, xs):
        x, cK, cV = carry
        bp = xs['params']
        layer_idx = xs['layer_idx']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        h_Q, h_K, h_V = _read_write_attn_operator_queries(
            router_params, normed, h_Q, h_K, h_V)
        tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
        raw_scan_offset_all = jnp.zeros_like(tau_all)

        Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                           **angular_execution_kwargs)
        K_val = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               **angular_execution_kwargs)
        V_val = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'],
                               **angular_execution_kwargs)
        _qk_s = qk_scale_eff
        _v_s = v_scale_eff
        Q = Q * _qk_s
        K_val = K_val * _qk_s
        V_val = V_val * _v_s

        Q_h = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        K_h = K_val.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        V_h = V_val.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

        cK = cK.at[layer_idx, :, :, :S, :].set(K_h)
        cV = cV.at[layer_idx, :, :, :S, :].set(V_h)

        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Q_h, K_h) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V_h)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
        attn_out = attn_out @ bp['attn']['expand_O']['kernel']
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        rst_out = _rst_forward_inference(
            normed, pool_params, router_params,
            d_model=d_model, n_layers=n_layers,
            angular_execution_kwargs=angular_execution_kwargs)
        x = x + rst_out
        return (x, cK, cV), None

    xs = {'params': stacked, 'layer_idx': jnp.arange(n_layers)}
    (x, cache_K, cache_V), _ = jax.lax.scan(prefill_layer, (x, cache_K, cache_V), xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = x @ params['token_emb']['embedding'].T
    return logits, cache_K, cache_V, S


def decode_step(params, model_cfg, token_id, cache_K, cache_V, cache_len):
    """Single token decode with KV cache. Returns logits [B,vocab], updated cache."""
    params = _squeeze_params(params)
    token_id = token_id.reshape(-1, 1)
    B = token_id.shape[0]
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    angular_execution_kwargs = _angular_execution_kwargs_from_model_cfg(model_cfg)

    pool_params = _pool_params_with_operator_keys(params['neuron_pool'])
    router_params = params['router']

    x = (params['token_emb']['embedding'][token_id]
         + params['pos_emb']['embedding'][cache_len][jnp.newaxis, :])

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    def decode_layer(carry, xs):
        x, cK, cV, pos = carry
        bp = xs['params']
        layer_idx = xs['layer_idx']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        attn_out, new_cK, new_cV = _attn_forward_cached(
            normed, pool_params, router_params,
            bp['attn']['expand_O']['kernel'],
            n_heads, d_model, n_layers,
            cK[layer_idx], cV[layer_idx], pos,
            angular_execution_kwargs=angular_execution_kwargs)
        cK = cK.at[layer_idx].set(new_cK)
        cV = cV.at[layer_idx].set(new_cV)
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        rst_out = _rst_forward_inference(
            normed, pool_params, router_params,
            d_model=d_model, n_layers=n_layers,
            angular_execution_kwargs=angular_execution_kwargs)
        x = x + rst_out
        return (x, cK, cV, pos), None

    xs = {'params': stacked, 'layer_idx': jnp.arange(n_layers)}
    (x, cache_K, cache_V, _), _ = jax.lax.scan(
        decode_layer, (x, cache_K, cache_V, cache_len), xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = (x @ params['token_emb']['embedding'].T)[:, 0, :]
    return logits, cache_K, cache_V, cache_len + 1


# ================================================================
# 10. Vectorized analysis helpers (inference-only)
# ================================================================

def vectorized_eval(params, model_cfg, all_tokens, batch_size=32):
    """Validation without Python loops. all_tokens: [N_seqs, max_seq] on device.

    Uses jax.lax.scan over batches, _srw_inference per layer (no chunking).
    Returns jnp scalars: avg_loss, ppl, accuracy, total_valid.
    """
    params = _squeeze_params(params)
    n_seqs = all_tokens.shape[0]
    n_batches = n_seqs // batch_size
    tokens = all_tokens[:n_batches * batch_size].reshape(n_batches, batch_size, -1).astype(jnp.int32)

    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    max_seq = model_cfg['max_seq_len']
    angular_execution_kwargs = _angular_execution_kwargs_from_model_cfg(model_cfg)

    pool_params = _pool_params_with_operator_keys(params['neuron_pool'])
    router_params = params['router']
    qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
        pool_params, d_model, n_layers)
    norm_params = params['norm']
    emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    qk_norm = pool_params['attn_qk_op_key']
    v_norm = pool_params['attn_v_op_key']
    rst_norm = pool_params['rst_op_key']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    def forward_batch(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        def layer_fn(x, bp):
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            h_Q, h_K, h_V = _read_write_attn_operator_queries(
                router_params, normed, h_Q, h_K, h_V)
            tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
            raw_scan_offset_all = jnp.zeros_like(tau_all)

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               **angular_execution_kwargs)
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               **angular_execution_kwargs)
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'],
                               **angular_execution_kwargs)
            _qk_s = qk_scale_eff
            _v_s = v_scale_eff
            Q = Q * _qk_s
            K = K * _qk_s
            V = V * _v_s

            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
            h_k = _read_write_rst_operator_query(router_params, normed, h_k)
            tau_k = normed @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
            raw_scan_offset_k = jnp.zeros_like(tau_k)
            rst_out = _srw_inference(normed, h_k, rst_norm, tau_k, raw_scan_offset_k,
                                     pool_params['rst_read'], pool_params['rst_write'],
                                     **angular_execution_kwargs)
            x = x + rst_out * rst_scale_eff
            return x, None

        x, _ = jax.lax.scan(layer_fn, x, stacked)
        x = _layer_norm(x, norm_params['scale'], norm_params['bias'])

        shift_x = x[:, :-1, :]
        shift_labels = input_ids[:, 1:].astype(jnp.int32)
        valid_mask = shift_labels > 0

        logits = shift_x @ emb_matrix.T
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        safe_labels = jnp.where(valid_mask, shift_labels, 0)
        token_loss = -jnp.take_along_axis(
            log_probs, safe_labels[..., jnp.newaxis], axis=-1).squeeze(-1)
        total_loss = (token_loss * valid_mask).sum()
        preds = jnp.argmax(logits, axis=-1)
        correct = ((preds == shift_labels) & valid_mask).sum()
        valid_count = valid_mask.sum()
        return total_loss, correct, valid_count

    def scan_batches(carry, batch):
        tl, tc, tv = carry
        loss, correct, valid = forward_batch(batch)
        return (tl + loss, tc + correct, tv + valid), None

    init = (jnp.float32(0.0), jnp.int32(0), jnp.int32(0))
    (total_loss, total_correct, total_valid), _ = jax.lax.scan(
        scan_batches, init, tokens)

    avg_loss = total_loss / (total_valid + 1e-8)
    ppl = jnp.exp(avg_loss)
    acc = total_correct.astype(jnp.float32) / (total_valid + 1e-8) * 100.0
    return avg_loss, ppl, acc, total_valid


def vectorized_neuron_health(params):
    """All neuron health stats. Returns dict of jnp values (single device_get later)."""
    params = _squeeze_params(params)
    pool = _pool_params_with_operator_keys(params['neuron_pool'])
    results = {}
    for pool_name, op_key_key, read_key, write_key in [
            ('Attention-QK', 'attn_qk_op_key', 'attn_qk_read', 'attn_qk_write'),
            ('Attention-V', 'attn_v_op_key', 'attn_v_read', 'attn_v_write'),
            ('RST', 'rst_op_key', 'rst_read', 'rst_write')]:
        op_key = pool[op_key_key]
        read = pool[read_key]
        write = pool[write_key]
        op_key_n = jnp.linalg.norm(op_key, axis=-1)
        read_n = jnp.linalg.norm(read, axis=-1)
        write_n = jnp.linalg.norm(write, axis=-1)
        results[pool_name] = {
            'N': op_key.shape[0],
            'op_key_mean': op_key_n.mean(),
            'op_key_std': op_key_n.std(),
            'op_key_dead': (op_key_n < 1e-6).sum(),
            'read_mean': read_n.mean(), 'read_std': read_n.std(),
            'read_dead': (read_n < 1e-6).sum(),
            'write_mean': write_n.mean(), 'write_std': write_n.std(),
            'write_dead': (write_n < 1e-6).sum(),
        }
    results['raw_tau_attn_bias'] = params['router']['raw_tau_attn']['bias']
    results['raw_tau_rst_bias'] = params['router']['raw_tau_rst']['bias']
    return results


def vectorized_weight_analysis(params, max_sample=2048):
    """Weight analysis: effective rank + cosine sim. All on device."""
    params = _squeeze_params(params)
    pool = _pool_params_with_operator_keys(params['neuron_pool'])
    results = {}
    for pool_name, op_key_key in [
            ('Attention-QK', 'attn_qk_op_key'),
            ('Attention-V', 'attn_v_op_key'),
            ('RST', 'rst_op_key')]:
        op_key = pool[op_key_key]
        N, d = op_key.shape
        if N > max_sample:
            idx = jnp.linspace(0, N - 1, max_sample, dtype=jnp.int32)
            op_key_s = op_key[idx]
        else:
            op_key_s = op_key
        norms = jnp.linalg.norm(op_key_s, axis=-1, keepdims=True) + 1e-8
        op_key_normed = op_key_s / norms
        n_s = op_key_normed.shape[0]

        gram = op_key_normed @ op_key_normed.T
        gram = gram - jnp.eye(n_s) * gram
        mean_sim = jnp.abs(gram).sum() / (n_s * (n_s - 1))
        max_sim = jnp.abs(gram).max()

        sv = jnp.linalg.svd(op_key_s, compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-8)
        entropy = -(sv_norm * jnp.log(sv_norm + 1e-10)).sum()
        eff_rank = jnp.exp(entropy)

        results[pool_name] = {
            'N': N, 'd': d,
            'mean_cosine_sim': mean_sim,
            'max_cosine_sim': max_sim,
            'effective_rank': eff_rank,
            'top5_sv': sv[:5],
        }
    return results


def analysis_forward(params, model_cfg, input_ids, mode='full'):
    """Forward returning per-layer gate distributions + output norms.

    mode='full': returns gate + execution_weight_norm (R.1, P2, P3 etc.)
    mode='light': returns execution_weight_norm only.

    Returns:
        logits: [B, S, vocab]
        layer_info: dict with stacked arrays:
            gate_Q: [n_layers, B, S, n_qk]
            gate_K: [n_layers, B, S, n_qk]
            gate_V: [n_layers, B, S, n_v]
            gate_RST: [n_layers, B, S, n_rst]
            (mode='full' only) gate_Q_raw, gate_K_raw, gate_V_raw, gate_RST_raw
            attn_out_norm: [n_layers]
            rst_out_norm: [n_layers]
    """
    params = _squeeze_params(params)
    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    angular_execution_kwargs = _angular_execution_kwargs_from_model_cfg(model_cfg)

    pool_params = _pool_params_with_operator_keys(params['neuron_pool'])
    router_params = params['router']
    qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]

    qk_norm = pool_params['attn_qk_op_key']
    v_norm = pool_params['attn_v_op_key']
    rst_norm_w = pool_params['rst_op_key']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    _return_raw = (mode == 'full')

    def analysis_layer(carry, xs):
        x = carry
        bp = xs['params']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        h_Q, h_K, h_V = _read_write_attn_operator_queries(
            router_params, normed, h_Q, h_K, h_V)
        tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
        raw_scan_offset_all = jnp.zeros_like(tau_all)

        Q, gate_Q_raw, gate_Q = _srw_inference_with_gates(
            normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'],
            **angular_execution_kwargs)
        K, gate_K_raw, gate_K = _srw_inference_with_gates(
            normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'],
            **angular_execution_kwargs)
        V, gate_V_raw, gate_V = _srw_inference_with_gates(
            normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
            pool_params['attn_v_read'], pool_params['attn_v_write'],
            **angular_execution_kwargs)
        _qk_s = qk_scale_eff
        _v_s = v_scale_eff
        Q = Q * _qk_s
        K = K * _qk_s
        V = V * _v_s

        d_head = d_model // n_heads
        Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
        attn_out = attn_out @ bp['attn']['expand_O']['kernel']
        attn_out_norm = jnp.linalg.norm(attn_out, axis=-1).mean()
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        h_k = normed @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
        h_k = _read_write_rst_operator_query(router_params, normed, h_k)
        tau_k = normed @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
        raw_scan_offset_k = jnp.zeros_like(tau_k)
        rst_out, gate_RST_raw, gate_RST = _srw_inference_with_gates(
            normed, h_k, rst_norm_w, tau_k, raw_scan_offset_k,
            pool_params['rst_read'], pool_params['rst_write'],
            **angular_execution_kwargs)
        rst_out = rst_out * rst_scale_eff
        rst_out_norm = jnp.linalg.norm(rst_out, axis=-1).mean()
        x = x + rst_out

        info = {
            'gate_Q': gate_Q, 'gate_K': gate_K,
            'gate_V': gate_V, 'gate_RST': gate_RST,
            'attn_out_norm': attn_out_norm,
            'rst_out_norm': rst_out_norm,
        }
        if _return_raw:
            info['gate_Q_raw'] = gate_Q_raw
            info['gate_K_raw'] = gate_K_raw
            info['gate_V_raw'] = gate_V_raw
            info['gate_RST_raw'] = gate_RST_raw
        # Analysis aliases kept for downstream tooling compatibility.
        info['gate_Know'] = info['gate_RST']
        if _return_raw:
            info['gate_Know_raw'] = info['gate_RST_raw']
        return x, info

    xs = {'params': stacked}
    x, layer_info = jax.lax.scan(analysis_layer, x, xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = x @ params['token_emb']['embedding'].T
    return logits, layer_info


def build_suppressed_forward(params, model_cfg, suppress_masks):
    """Build forward with specific neurons suppressed (gate zeroed).

    suppress_masks: dict with 'qk':[n_qk] bool, 'v':[n_v], 'rst':[n_rst] bool.
    Compatibility key 'know' is still accepted.
    True = suppress.
    Returns: forward_fn(input_ids) -> logits [B, S, vocab]
    """
    params = _squeeze_params(params)
    params = jax.tree.map(jnp.asarray, params)
    angular_execution_kwargs = _angular_execution_kwargs_from_model_cfg(model_cfg)
    qk_mult = jnp.where(suppress_masks.get('qk', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'qk' in suppress_masks else None
    v_mult = jnp.where(suppress_masks.get('v', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'v' in suppress_masks else None
    rst_mask = suppress_masks.get('rst', suppress_masks.get('know', None))
    rst_mult = jnp.where(rst_mask, 0.0, 1.0) if rst_mask is not None else None

    def _srw_sup(x, h, op_key, tau_off, raw_scan_offset, w_read, w_write, mult):
        """SRW with optional gate suppression."""
        # v4.1.6.6: suppressed forward selects by op key and executes RW.
        r_n = _forward_unit_direction(w_read.astype(jnp.float32))
        w_n = _forward_unit_direction(w_write.astype(jnp.float32))
        execution_kwargs, admission_den_power = _split_admission_den_kwargs(
            angular_execution_kwargs)
        _, admission, _, execution_weight, _ = _angular_execution(
            h, op_key, tau_off, raw_scan_offset, **execution_kwargs)
        if mult is not None:
            execution_weight = execution_weight * mult[None, None, :]
            admission = admission * mult[None, None, :]
        xr = x.astype(jnp.float32) @ r_n.T
        a = execution_weight * xr
        out = a @ w_n
        admission_den = jnp.power(
            jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0),
            admission_den_power)
        return (out.astype(jnp.float32) / admission_den).astype(jnp.float32)

    def forward_fn(input_ids):
        B, S = input_ids.shape
        d_model = model_cfg['d_model']
        n_layers = model_cfg['n_layers']
        n_heads = model_cfg['n_heads']
        d_head = d_model // n_heads
        pp = _pool_params_with_operator_keys(params['neuron_pool'])
        rp = params['router']
        qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
            pp, d_model, n_layers)

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]
        qk_n = pp['attn_qk_op_key']
        v_n = pp['attn_v_op_key']
        kn_n = pp['rst_op_key']

        for i in range(n_layers):
            bp = params[f'block_{i}']
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ rp['proj_attn']['kernel'] + rp['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            h_Q, h_K, h_V = _read_write_attn_operator_queries(
                rp, normed, h_Q, h_K, h_V)
            tau_all = normed @ rp['raw_tau_attn']['kernel'] + rp['raw_tau_attn']['bias']
            raw_scan_offset_all = jnp.zeros_like(tau_all)

            Q = _srw_sup(normed, h_Q, qk_n, tau_all[:,:,0:1], raw_scan_offset_all[:,:,0:1], pp['attn_qk_read'], pp['attn_qk_write'], qk_mult)
            K = _srw_sup(normed, h_K, qk_n, tau_all[:,:,1:2], raw_scan_offset_all[:,:,1:2], pp['attn_qk_read'], pp['attn_qk_write'], qk_mult)
            V = _srw_sup(normed, h_V, v_n, tau_all[:,:,2:3], raw_scan_offset_all[:,:,2:3], pp['attn_v_read'], pp['attn_v_write'], v_mult)
            _qk_s = qk_scale_eff
            _v_s = v_scale_eff
            Q = Q * _qk_s
            K = K * _qk_s
            V = V * _v_s

            Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            sc = jnp.sqrt(jnp.float32(d_head))
            attn_s = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / sc
            causal = jnp.tril(jnp.ones((S,S), dtype=jnp.bool_))
            attn_s = jnp.where(causal, attn_s, jnp.finfo(attn_s.dtype).min)
            attn_w = jax.nn.softmax(attn_s, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0,2,1,3).reshape(B,S,d_model) @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed @ rp['proj_rst']['kernel'] + rp['proj_rst']['bias']
            h_k = _read_write_rst_operator_query(rp, normed, h_k)
            tau_k = normed @ rp['raw_tau_rst']['kernel'] + rp['raw_tau_rst']['bias']
            raw_scan_offset_k = jnp.zeros_like(tau_k)
            x = x + _srw_sup(normed, h_k, kn_n, tau_k, raw_scan_offset_k, pp['rst_read'], pp['rst_write'], rst_mult) * rst_scale_eff

        norm_p = params['norm']
        x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
        return x @ params['token_emb']['embedding'].T

    return forward_fn


def _rename_key_if_needed(d, old, new):
    """Rename old -> new in a mutable mapping, preserving an existing new key."""
    if old in d:
        if new not in d:
            d[new] = d[old]
        del d[old]
