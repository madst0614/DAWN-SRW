"""
DAWN-SRW v4.1.6.4 Angular-Depth Boundary DirectTau

Clean v4164 experimental model path.

Implemented concepts:
- cosine-space tau reference with bounded sigmoid min/max mapping
- one-sided generalized Gaussian boundary DirectTau admission
- boundary-relative angular-depth RW composition
- scheduled soft-gate boundary-scale input
- scheduled boundary power input
- tau movement controlled by optimizer-side tau_lr_mult
- delayed RPE/exploration weight input
- train-time effective gate statistics
- validation-time execution pruning through execution_prune_eps
"""


import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from typing import Optional, Dict
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
    'qk_angular_depth_mean',
    'v_angular_depth_mean',
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
    ATTN_SPLIT_QK_ANGULAR_DEPTH_MEAN,
    ATTN_SPLIT_V_ANGULAR_DEPTH_MEAN,
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
GATE_EPS_VALUE_TO_SUFFIX = {
    float(v): k for k, v in GATE_EPS_SUFFIX_TO_VALUE.items()
}


def _gate_eps_suffix(value):
    value_f = float(value)
    for known_value, suffix in GATE_EPS_VALUE_TO_SUFFIX.items():
        if math.isclose(value_f, known_value, rel_tol=0.0, abs_tol=known_value * 1.0e-6):
            return suffix
    raise ValueError(
        f"Unsupported v4164 sparsity eps={value!r}; supported values are "
        f"{sorted(GATE_EPS_VALUE_TO_SUFFIX)}.")


def _gate_eps_values_from_suffixes(suffixes):
    return tuple(float(GATE_EPS_SUFFIX_TO_VALUE[s]) for s in suffixes)


def _regular_diag_level(level):
    level = str(level or 'compact').lower()
    if level not in ('compact', 'full'):
        raise ValueError(
            f"regular_diagnostics_level={level!r}; expected 'compact' or 'full'.")
    return level


def _regular_eps_suffixes(values, default_suffixes, allowed_suffixes):
    if values is None:
        return tuple(default_suffixes)
    suffixes = tuple(_gate_eps_suffix(v) for v in values)
    bad = [s for s in suffixes if s not in allowed_suffixes]
    if bad:
        raise ValueError(
            f"Unsupported regular sparsity eps suffixes {bad}; allowed are "
            f"{tuple(allowed_suffixes)}.")
    return suffixes


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

    The learned scale parameters stay in the checkpoint tree for compatibility,
    but the active forward path ignores them.
    """
    return _pool_output_scales(d_model, n_layers)


# ================================================================
# V4.1.6.4 annealed generalized Gaussian DirectTau in cosine space.
#
#   rho              = cosine(q, signature)
#   raw_tau          = learned cosine-space reference
#   tau              = -1 + 2 * sigmoid(raw_tau)
#   margin, r        = rho - tau, margin / boundary_scale
#   d_neg            = lambda_neg * softplus(-r / lambda_neg)
#   admission        = exp(-(d_neg ** boundary_power) - gamma(p) * exp(-r / A(p)))
#   angular_depth    = softplus(margin / B) / softplus((1 - tau) / B)
#   compose_weight   = admission * angular_depth
#   den              = max(sum(admission), 1.0)
# ================================================================

DEFAULT_D_ROUTE = 64
RW_FORWARD_NORM_EPS = 1e-6     # forward-only read/write direction floor


# ================================================================
# 1. Helpers
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_rate, x.shape)
    dropped = jnp.where(mask, x / keep_rate, 0.0)
    # Eval path: return x unscaled. Previous version returned x/keep_rate
    # here (mask forced to ones but the where-branch still divided), which
    # inflated all eval tensors by 1/keep_rate and put a structural
    # offset into val_loss.
    return jnp.where(deterministic, x, dropped)


def _layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _forward_unit_direction(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True)
                + RW_FORWARD_NORM_EPS)


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


def _angular_depth_from_margin(margin, tau, depth_scale, eps=1.0e-6):
    depth_scale = jnp.maximum(
        jnp.asarray(depth_scale, dtype=jnp.float32),
        jnp.float32(1.0e-4))
    tau = jnp.asarray(tau, dtype=jnp.float32)
    max_margin = jnp.maximum(
        jnp.float32(1.0) - tau,
        jnp.float32(eps))

    numerator = jax.nn.softplus(margin / depth_scale)
    denominator = jax.nn.softplus(max_margin / depth_scale)
    return numerator / jnp.maximum(denominator, jnp.float32(eps))


def _boundary_gate_from_margin(margin, boundary_scale, boundary_power):
    """Compatibility name for projected admission diagnostics."""
    return _boundary_soft_weight_from_margin(
        margin, boundary_scale, boundary_power)


def _compute_admission_depth(score, tau, boundary_scale,
                             boundary_power=2.0,
                             effective_active_eps=1.0e-6,
                             execution_prune_eps=0.0):
    """v4164 admission plus boundary-relative angular-depth composition."""
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
    angular_depth = _angular_depth_from_margin(
        margin, tau, boundary_scale)
    compose_weight = admission * angular_depth
    active_mask = (
        admission >= jnp.asarray(effective_active_eps, dtype=jnp.float32))
    return margin, admission, angular_depth, compose_weight, active_mask


LOCAL_SPIKE_METRIC_COUNT = 11
LOCAL_SPIKE_TOP1_COUNT = 17
ATTN_LOCAL_METRIC_COUNT = 7
SPIKE_SRW_FIELD_COUNT = 26
SPIKE_ATTN_FIELD_COUNT = 14
SPIKE_TOKEN_FIELD_COUNT = 13
SPIKE_FOCUS_PATH_FIELD_COUNT = 28
SPIKE_FOCUS_ROUTE_FIELD_COUNT = 24
SPIKE_FOCUS_SRW_FIELD_COUNT = 31
SPIKE_FOCUS_ATTN_FIELD_COUNT = 18


def _topk_rows(candidates, topk, field_count):
    """Return top-k candidate rows sorted by score in column 0."""
    k = int(topk)
    size = int(candidates.shape[0])
    take_k = min(k, size)
    scores, idx = jax.lax.top_k(candidates[:, 0], take_k)
    rows = jnp.take(candidates, idx, axis=0)
    rows = rows.at[:, 0].set(scores)
    if take_k < k:
        pad = jnp.full((k - take_k, field_count), -jnp.inf, dtype=rows.dtype)
        rows = jnp.concatenate([rows, pad], axis=0)
    return rows


# ================================================================
# 2. shard_map based gate + sense_read_write
# ================================================================


# ================================================================
# 3. shard_map based gate + sense_read_write
#    Per-device code with explicit psum communication.
#    fori_loop inside for N_local chunking.
# ================================================================

def make_sharded_srw(mesh, max_chunk_size=2048,
                     analysis=False,
                     local_diagnostics=False,
                     spike_probe=False,
                     spike_probe_topk=8,
                     dead_exposure_target=0.1,
                     soft_gate_effective_active_eps=1.0e-6,
                     soft_gate_enabled=False,
                     regular_diagnostics_level='compact',
                     regular_current_eps=(1.0e-1, 1.0e-2, 1.0e-3),
                     regular_projected_eps=(1.0e-6,),
                     regular_mass_enabled=False,
                     regular_sparsity_enabled=False):
    """Create fused shard_map'd angular Select + SRW.

    Fast train path: one chunked pass computes rho, tau, gate, and SRW.
    Analysis path may compute rho distribution moments for diagnostics.

    One-sided generalized Gaussian boundary DirectTau admission:
        rho              = cosine(q, signature)
        margin           = rho - tau
        admission        = exp(-((max(tau - rho, 0) / B) ** p))
        angular_depth    = softplus(margin / B) / softplus((1 - tau) / B)
        compose_weight   = admission * angular_depth
        den              = max(sum(admission), 1.0)


    `analysis=False` (default, train path): returns the SLIM tuple plus
    four gate-concentration diagnostics, and skips distribution-shape stats
    (skew/kurt), selection-residency/entropy diagnostics and depth extrema.
    XLA DCE's the unused work.
    `local_diagnostics=True` appends a lightweight, scalar-only local-spike
    summary to either path. It is independent of `analysis=True` and is
    collected inline during the existing chunk scan.
    `analysis=True`: returns the SLIM/concentration tuple followed by
    observational scalars/arrays for route shape, gate concentration, and
    denominator diagnostics.
    Used by analysis_step at val time only.
    """
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']
    _dead_exposure_target = jnp.float32(dead_exposure_target)
    _soft_gate_effective_active_eps = jnp.float32(soft_gate_effective_active_eps)
    _local_diagnostics = bool(local_diagnostics)
    _spike_probe = bool(spike_probe)
    _spike_probe_topk = max(1, int(spike_probe_topk))
    _soft_gate_enabled = True
    _regular_level = _regular_diag_level(regular_diagnostics_level)
    _compact_regular_diag = (not bool(analysis) and _regular_level == 'compact')
    _sparsity_diag_enabled = bool(analysis) or bool(regular_sparsity_enabled)
    _regular_current_suffixes = _regular_eps_suffixes(
        regular_current_eps, ('1e_1', '1e_2', '1e_3'),
        GATE_EPS_NAME_SUFFIXES)
    _regular_projected_suffixes = _regular_eps_suffixes(
        regular_projected_eps, ('1e_6',),
        GATE_PROJECTED_EPS_NAME_SUFFIXES)
    _current_suffixes = (
        _regular_current_suffixes if _compact_regular_diag
        else GATE_EPS_NAME_SUFFIXES)
    _projected_suffixes = (
        _regular_projected_suffixes if _compact_regular_diag
        else GATE_PROJECTED_EPS_NAME_SUFFIXES)
    _current_eps_values = _gate_eps_values_from_suffixes(_current_suffixes)
    _projected_eps_values = _gate_eps_values_from_suffixes(_projected_suffixes)
    _compute_sparsity_mass = (
        (not _compact_regular_diag) or bool(regular_mass_enabled))
    _compact_margin_bands = bool(_compact_regular_diag)
    _angular_strong_margin = jnp.float32(0.05)

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
        P(),                     # int_max scalar (v4.1 diag)
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
    _local_diag_specs = (
        P(),                     # local_spike_values [1, metric]
        P(),                     # local_spike_locs [1, metric, b/t/neuron]
        P(),                     # top1_breakdown_values [1, field]
        P(),                     # top1_breakdown_locs [1, b/t/neuron]
    )
    _spike_probe_specs = (
        P(),                     # spike SRW top rows [1, K, field]
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs + _conc_out_specs)
    _out_specs = _out_specs + _select_diag_specs
    _out_specs = _out_specs + _dead_exposure_diag_specs
    _out_specs = _out_specs + _sparsity_diag_specs
    if _local_diagnostics:
        _out_specs = _out_specs + _local_diag_specs
    if _spike_probe:
        _out_specs = _out_specs + _spike_probe_specs

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),    # x [B,S,D]
                       P('data', None, None),    # h [B,S,d_route]
                       P('model', None),          # route emb [N_local,d_route]
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
    def fused_gate_srw(x, h, emb_local, raw_tau,
                       read_local, write_local,
                       soft_gate_temperature, soft_gate_t_final,
                       soft_gate_boundary_power,
                       soft_gate_boundary_power_final,
                       execution_prune_eps):
        N_local = emb_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        emb_bf = emb_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1 = jnp.zeros((B, S, 1))
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_pos_inf = jnp.float32(1.0e30)
        diag_vals_init = jnp.full(
            (1, LOCAL_SPIKE_METRIC_COUNT), diag_neg_inf)
        spike_rows_init = jnp.full(
            (_spike_probe_topk, SPIKE_SRW_FIELD_COUNT),
            diag_neg_inf, dtype=jnp.float32)
        spike_rows_init = spike_rows_init.at[:, 1:4].set(0.0)

        def route_emb_chunk(start):
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, start, cs, axis=0)
            return ec

        def route_relation(h_in, route):
            q_unit = _forward_unit_direction(
                h_in.astype(jnp.float32)).astype(jnp.bfloat16)
            route_unit = _forward_unit_direction(
                route.astype(jnp.float32)).astype(jnp.bfloat16)
            rho = (q_unit @ route_unit.T).astype(jnp.float32)
            rho_exposure = (
                jax.lax.stop_gradient(q_unit) @ route_unit.T
            ).astype(jnp.float32)
            return rho, rho_exposure

        def route_rw_chunk(start):
            ec = route_emb_chunk(start)
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
                route = route_emb_chunk(s)
                rho, _ = route_relation(h_bf, route)
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
            (selection_margin, admission, angular_depth, compose_weight,
             active_mask) = _compute_admission_depth(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            strong_mask = selection_margin > _angular_strong_margin
            return (
                selection_margin,
                admission,
                angular_depth,
                compose_weight,
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

        def gate_sparsity_parts(selection_margin, admission, compose_weight):
            margin_sg = jax.lax.stop_gradient(selection_margin)
            admission_sg = jax.lax.stop_gradient(admission)
            compose_sg = jax.lax.stop_gradient(compose_weight)
            active_tau = margin_sg > 0.0

            admission_active = admission_sg[..., None] > current_eps
            admission_active_count = admission_active.astype(jnp.float32).sum(
                axis=(0, 1, 2))
            current_active = compose_sg[..., None] > current_eps
            current_active_count = current_active.astype(jnp.float32).sum(
                axis=(0, 1, 2))
            if _compute_sparsity_mass:
                current_mass = (
                    compose_sg[..., None] * current_active.astype(jnp.float32)
                ).sum(axis=(0, 1, 2))
                gate_mass = compose_sg.sum()
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
            if _compact_regular_diag:
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

        def spike_chunk_rows(start, rho, raw_tau_chunk, tau_chunk,
                             selection_margin, admission, angular_depth,
                             compose_weight, xr_f, read_norm, write_norm):
            contrib_proxy = (
                jnp.abs(jax.lax.stop_gradient(compose_weight * xr_f))
                * write_norm[None, None, :])
            flat_score = contrib_proxy.reshape((-1,))
            top_scores, top_idx = jax.lax.top_k(
                flat_score,
                min(_spike_probe_topk, int(flat_score.shape[0])))
            b_idx = top_idx // (S * cs)
            rem = top_idx - b_idx * (S * cs)
            pos_idx = rem // cs
            n_idx = rem - pos_idx * cs
            read_norm_i = jnp.take(read_norm, n_idx, axis=0)
            write_norm_i = jnp.take(write_norm, n_idx, axis=0)
            raw_tau_i = raw_tau_chunk[b_idx, pos_idx, 0]
            tau_i = tau_chunk[b_idx, pos_idx, 0]
            margin_i = selection_margin[b_idx, pos_idx, n_idx]
            pos_margin_i = admission[b_idx, pos_idx, n_idx]
            angular_depth_i = angular_depth[b_idx, pos_idx, n_idx]
            gate_i = compose_weight[b_idx, pos_idx, n_idx]
            xr_i = xr_f[b_idx, pos_idx, n_idx]
            h_norm_i = jnp.linalg.norm(
                jax.lax.stop_gradient(h[b_idx, pos_idx].astype(jnp.float32)),
                axis=-1)
            resid_norm_i = jnp.linalg.norm(
                jax.lax.stop_gradient(x[b_idx, pos_idx].astype(jnp.float32)),
                axis=-1)
            rows = jnp.stack([
                top_scores,
                b_idx.astype(jnp.float32),
                pos_idx.astype(jnp.float32),
                (jnp.int32(start) + n_idx).astype(jnp.float32),
                rho[b_idx, pos_idx, n_idx],
                raw_tau_i,
                tau_i,
                margin_i,
                pos_margin_i,
                angular_depth_i,
                gate_i,
                jnp.zeros_like(top_scores),
                jnp.zeros_like(top_scores),
                jnp.zeros_like(top_scores),
                jnp.zeros_like(top_scores),
                jnp.zeros_like(top_scores),
                xr_i,
                jnp.abs(xr_i),
                read_norm_i,
                write_norm_i,
                read_norm_i * write_norm_i,
                top_scores,
                jnp.zeros_like(top_scores),
                jnp.zeros_like(top_scores),
                resid_norm_i,
                h_norm_i,
            ], axis=-1).astype(jnp.float32)
            if int(top_scores.shape[0]) < _spike_probe_topk:
                pad = jnp.full(
                    (_spike_probe_topk - int(top_scores.shape[0]),
                     SPIKE_SRW_FIELD_COUNT),
                    diag_neg_inf, dtype=jnp.float32)
                pad = pad.at[:, 1:4].set(0.0)
                rows = jnp.concatenate([rows, pad], axis=0)
            return rows

        def spike_merge_rows(rows_a, rows_b):
            return _topk_rows(
                jnp.concatenate([rows_a, rows_b], axis=0),
                _spike_probe_topk, SPIKE_SRW_FIELD_COUNT)

        def spike_finalize_rows(rows, raw_out, out, global_den_cost_m,
                                global_gate_max_m, global_active_m,
                                no_active_direct):
            local_b = rows[:, 1].astype(jnp.int32)
            pos = rows[:, 2].astype(jnp.int32)
            den_i = global_den_cost_m[local_b, pos, 0]
            top1_i = (
                global_gate_max_m[local_b, pos, 0]
                / jnp.maximum(den_i, 1e-8))
            active_i = global_active_m[local_b, pos, 0]
            no_active_i = no_active_direct[local_b, pos, 0]
            raw_out_i = jnp.linalg.norm(
                jax.lax.stop_gradient(raw_out[local_b, pos]), axis=-1)
            out_i = jnp.linalg.norm(
                jax.lax.stop_gradient(out[local_b, pos]), axis=-1)
            model_offset = jax.lax.axis_index('model') * N_local
            data_offset = jax.lax.axis_index('data') * B
            rows = rows.at[:, 1].set(rows[:, 1] + data_offset)
            rows = rows.at[:, 3].set(rows[:, 3] + model_offset)
            rows = rows.at[:, 11].set(den_i)
            rows = rows.at[:, 12].set(rows[:, 10] / jnp.maximum(den_i, 1e-8))
            rows = rows.at[:, 13].set(top1_i)
            rows = rows.at[:, 14].set(active_i)
            rows = rows.at[:, 15].set(no_active_i)
            rows = rows.at[:, 22].set(raw_out_i)
            rows = rows.at[:, 23].set(out_i)
            gathered = jax.lax.all_gather(rows, 'model', axis=0)
            gathered = jax.lax.all_gather(gathered, 'data', axis=0)
            return _topk_rows(
                gathered.reshape((-1, SPIKE_SRW_FIELD_COUNT)),
                _spike_probe_topk, SPIKE_SRW_FIELD_COUNT)[None, :, :]

        # --- Pass 2: admission + angular-depth SRW fused (scan + checkpoint) ---
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
                 sparsity_carry, select_diag_carry, diag_vals, spike_rows) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                rho, rho_exposure = route_relation(h_bf, route)
                (selection_margin, admission, angular_depth, compose_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = angular_depth.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, compose_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_depth.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = compose_weight * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = compose_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(compose_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(compose_weight * xr_f))
                        * write_norm[None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(selection_margin)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(angular_depth)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(xr_f))))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.max(jnp.linalg.norm(
                            jax.lax.stop_gradient(route.astype(jnp.float32)),
                            axis=-1)))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
                if _spike_probe:
                    read_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            read_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    write_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            write_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    spike_rows = spike_merge_rows(
                        spike_rows,
                        spike_chunk_rows(
                            s, rho, raw_tau, tau, selection_margin,
                            admission, angular_depth, compose_weight, xr_f,
                            read_norm, write_norm))
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_margin_band = jnp.zeros((B, S, 1), dtype=jnp.float32)
                chunk_margin_band_wide = jnp.zeros((B, S, 1), dtype=jnp.float32)
                chunk_margin_band_mid = jnp.zeros((B, S, 1), dtype=jnp.float32)
                g_safe = compose_weight + 1e-8
                chunk_g_log_g = (compose_weight * jnp.log(g_safe)).sum(
                    axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission))
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, compose_weight.max(axis=-1, keepdims=True)),
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
                        select_diag_carry,
                        diag_vals,
                        spike_rows), None

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
             select_diag_carry, diag_vals, spike_rows), _ = jax.lax.scan(
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
                 select_diag_carry0,
                 diag_vals_init,
                 spike_rows_init),
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
                 sparsity_carry, select_diag_carry, diag_vals, spike_rows) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                rho, rho_exposure = route_relation(h_bf, route)
                (selection_margin, admission, angular_depth, compose_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = angular_depth.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, compose_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_depth.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = compose_weight * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = compose_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(compose_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(compose_weight * xr_f))
                        * write_norm[None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(selection_margin)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(angular_depth)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(xr_f))))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.max(jnp.linalg.norm(
                            jax.lax.stop_gradient(route.astype(jnp.float32)),
                            axis=-1)))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
                if _spike_probe:
                    read_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            read_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    write_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            write_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    spike_rows = spike_merge_rows(
                        spike_rows,
                        spike_chunk_rows(
                            s, rho, raw_tau, tau, selection_margin,
                            admission, angular_depth, compose_weight, xr_f,
                            read_norm, write_norm))
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
                        jnp.maximum(total_gate_max, compose_weight.max(axis=-1, keepdims=True)),
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
                        select_diag_carry,
                        diag_vals,
                        spike_rows), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_selection_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_exposure_sum, total_exposure_min,
             total_exposure_max, total_weak_exposure_count,
             total_int_max, total_int_cap_count, total_selection_residency_sum,
             total_selection_residency_count,
             total_edge_margin_stat,
             sparsity_carry,
             select_diag_carry, diag_vals, spike_rows), _ = jax.lax.scan(
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
                 select_diag_carry0,
                 diag_vals_init,
                 spike_rows_init),
                jnp.arange(nc))


        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')  # sum(compose_weight)
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        # Denominator intentionally uses admission only, not compose_weight.
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        global_selection_cost = jax.lax.psum(total_selection_cost, 'model')
        global_current_cost = jax.lax.psum(total_current_cost, 'model')
        selection_residency_loss = jnp.float32(0.0)
        edge_margin_stat = (
            jax.lax.psum(total_edge_margin_stat, 'model')
            / jnp.float32(B * S * N_total))
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        den = jnp.maximum(global_den_cost, 1.0)
        out = raw_out / den
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
        # Action path above keeps global_den_cost/global_weighted_cost live for
        # the SRW denominator and output gradient.
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
        sparsity_diag_out = finalize_sparsity_diag(sparsity_carry)

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
        local_diag_out = ()
        if _local_diagnostics:
            tau_abs_max = jnp.max(jax.lax.stop_gradient(tau))
            top1_share_max = jnp.max(
                global_gate_max_m / jnp.maximum(global_den_cost_m, 1e-8))
            gate_den_sum_max = jnp.max(global_den_cost_m)
            local_out_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(out), axis=-1))
            residual_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(x), axis=-1))
            h_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(h), axis=-1))
            token_vals = jnp.stack([
                tau_abs_max, top1_share_max, gate_den_sum_max,
                local_out_norm_max, residual_norm_max, h_norm_max,
            ]).reshape((1, 6))
            token_slots = jnp.array([0, 2, 3, 7, 8, 9], dtype=jnp.int32)
            metric_vals = diag_vals.at[:, token_slots].set(token_vals)
            metric_vals = jax.lax.stop_gradient(metric_vals)
            metric_vals = jax.lax.pmax(metric_vals, 'model')
            metric_vals = jax.lax.pmax(metric_vals, 'data')
            metric_locs = jnp.full(
                (1, LOCAL_SPIKE_METRIC_COUNT, 3), -1, dtype=jnp.int32)
            top1_details = jnp.zeros(
                (1, LOCAL_SPIKE_TOP1_COUNT), dtype=jnp.float32)
            top1_details = top1_details.at[:, 0].set(metric_vals[:, 2])
            top1_details = top1_details.at[:, 4].set(metric_vals[:, 1])
            top1_details = top1_details.at[:, 6].set(metric_vals[:, 3])
            top1_details = top1_details.at[:, 7].set(metric_vals[:, 4])
            top1_details = top1_details.at[:, 8].set(metric_vals[:, 5])
            top1_details = top1_details.at[:, 12].set(metric_vals[:, 6])
            top1_details = top1_details.at[:, 13].set(metric_vals[:, 7])
            top1_details = top1_details.at[:, 15].set(metric_vals[:, 9])
            top1_details = top1_details.at[:, 16].set(metric_vals[:, 10])
            top1_locs = jnp.full((1, 3), -1, dtype=jnp.int32)
            local_diag_out = (
                metric_vals.astype(jnp.float32), metric_locs,
                top1_details, top1_locs)
        spike_probe_out = ()
        if _spike_probe:
            spike_probe_out = (
                spike_finalize_rows(
                    spike_rows, raw_out, out, global_den_cost_m,
                    global_gate_max_m, global_active_m,
                    no_active_direct).astype(jnp.float32),)
        if not analysis:
            return (slim_out + conc_out + select_diag_out
                    + dead_exposure_diag_out + (sparsity_diag_out,)
                    + local_diag_out + spike_probe_out)

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
        # local_diag_out is collected inline in pass 2 above; no replay path.
        return (slim_out + conc_out
                + (margin_band_frac, margin_band_wide_frac,
                   margin_band_mid_frac, rho_skew, active_per_token_std,
                   gate_entropy, den_cost_out, selection_cost_out,
                   current_cost_out, rho_kurt, int_cap_frac_out)
                + select_diag_out + dead_exposure_diag_out
                + (sparsity_diag_out,)
                + local_diag_out + spike_probe_out)

    return fused_gate_srw


def make_sharded_srw_paired(mesh, max_chunk_size=2048,
                            analysis=False,
                            local_diagnostics=False,
                            spike_probe=False,
                            spike_probe_topk=8,
                            dead_exposure_target=0.1,
                            soft_gate_effective_active_eps=1.0e-6,
                            soft_gate_enabled=False,
                            regular_diagnostics_level='compact',
                            regular_current_eps=(1.0e-1, 1.0e-2, 1.0e-3),
                            regular_projected_eps=(1.0e-6,),
                            regular_mass_enabled=False,
                            regular_sparsity_enabled=False):
    """Fused Q+K shard_map: two routes sharing same pool in one shard_map call.

    h is [B,S,2,d_route] (h_Q, h_K stacked on axis=2).
    raw_tau is [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].

    Boundary DirectTau admission with angular-depth composition:
        admission = exp(-((max(tau - rho, 0) / B) ** p))
        angular_depth = softplus((rho - tau) / B) / softplus((1 - tau) / B)
        compose_weight = admission * angular_depth
        den = max(sum(admission), 1.0)
    analysis/local_diagnostics: see make_sharded_srw docstring.
    """
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']
    _dead_exposure_target = jnp.float32(dead_exposure_target)
    _soft_gate_effective_active_eps = jnp.float32(soft_gate_effective_active_eps)
    _local_diagnostics = bool(local_diagnostics)
    _spike_probe = bool(spike_probe)
    _spike_probe_topk = max(1, int(spike_probe_topk))
    _regular_level = _regular_diag_level(regular_diagnostics_level)
    _compact_regular_diag = (not bool(analysis) and _regular_level == 'compact')
    _sparsity_diag_enabled = bool(analysis) or bool(regular_sparsity_enabled)
    _regular_current_suffixes = _regular_eps_suffixes(
        regular_current_eps, ('1e_1', '1e_2', '1e_3'),
        GATE_EPS_NAME_SUFFIXES)
    _regular_projected_suffixes = _regular_eps_suffixes(
        regular_projected_eps, ('1e_6',),
        GATE_PROJECTED_EPS_NAME_SUFFIXES)
    _current_suffixes = (
        _regular_current_suffixes if _compact_regular_diag
        else GATE_EPS_NAME_SUFFIXES)
    _projected_suffixes = (
        _regular_projected_suffixes if _compact_regular_diag
        else GATE_PROJECTED_EPS_NAME_SUFFIXES)
    _current_eps_values = _gate_eps_values_from_suffixes(_current_suffixes)
    _projected_eps_values = _gate_eps_values_from_suffixes(_projected_suffixes)
    _compute_sparsity_mass = (
        (not _compact_regular_diag) or bool(regular_mass_enabled))
    _compact_margin_bands = bool(_compact_regular_diag)
    _soft_gate_enabled = True
    _angular_strong_margin = jnp.float32(0.05)

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
        P(),                          # int_max scalar (v4.1 diag)
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
    _local_diag_specs = (
        P(),                          # local_spike_values [2, metric]
        P(),                          # local_spike_locs [2, metric, b/t/neuron]
        P(),                          # top1_breakdown_values [2, field]
        P(),                          # top1_breakdown_locs [2, b/t/neuron]
    )
    _spike_probe_specs = (
        P(),                          # spike SRW top rows [2, K, field]
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _route_split_out_specs
                  + _analysis_extra_specs
                  if analysis
                  else _slim_out_specs + _conc_out_specs + _route_split_out_specs)
    _out_specs = _out_specs + _select_diag_specs
    _out_specs = _out_specs + _dead_exposure_diag_specs
    _out_specs = _out_specs + _sparsity_diag_specs
    if _local_diagnostics:
        _out_specs = _out_specs + _local_diag_specs
    if _spike_probe:
        _out_specs = _out_specs + _spike_probe_specs

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),        # x [B,S,D]
                       P('data', None, None, None),  # h [B,S,2,d_route]
                       P('model', None),              # route emb [N_local,d_route]
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
    def fused_gate_srw_paired(x, h, emb_local, raw_tau,
                              read_local, write_local,
                              soft_gate_temperature, soft_gate_t_final,
                              soft_gate_boundary_power,
                              soft_gate_boundary_power_final,
                              execution_prune_eps):
        N_local = emb_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        # h: [B,S,2,d_route], raw_tau: [B,S,2,1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        emb_bf = emb_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_pos_inf = jnp.float32(1.0e30)
        diag_vals_init = jnp.full(
            (2, LOCAL_SPIKE_METRIC_COUNT), diag_neg_inf)
        spike_rows_init = jnp.full(
            (2, _spike_probe_topk, SPIKE_SRW_FIELD_COUNT),
            diag_neg_inf, dtype=jnp.float32)
        spike_rows_init = spike_rows_init.at[:, :, 1:4].set(0.0)

        def route_emb_chunk(start):
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, start, cs, axis=0)
            return ec

        def route_relation(h_in, route):
            q_unit = _forward_unit_direction(
                h_in.astype(jnp.float32)).astype(jnp.bfloat16)
            route_unit = _forward_unit_direction(
                route.astype(jnp.float32)).astype(jnp.bfloat16)
            rho = jnp.einsum(
                'bsrd,nd->bsrn', q_unit, route_unit).astype(jnp.float32)
            rho_exposure = jnp.einsum(
                'bsrd,nd->bsrn',
                jax.lax.stop_gradient(q_unit),
                route_unit).astype(jnp.float32)
            return rho, rho_exposure

        def route_rw_chunk(start):
            ec = route_emb_chunk(start)
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
                route = route_emb_chunk(s)
                rho, _ = route_relation(h_bf, route)
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
            (selection_margin, admission, angular_depth, compose_weight,
             active_mask) = _compute_admission_depth(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            strong_mask = selection_margin > _angular_strong_margin
            return (
                selection_margin,
                admission,
                angular_depth,
                compose_weight,
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

        def gate_sparsity_parts(selection_margin, admission, compose_weight):
            margin_sg = jax.lax.stop_gradient(selection_margin)
            admission_sg = jax.lax.stop_gradient(admission)
            compose_sg = jax.lax.stop_gradient(compose_weight)
            active_tau = margin_sg > 0.0

            admission_active = admission_sg[..., None] > current_eps
            admission_active_count = admission_active.astype(jnp.float32).sum(
                axis=(0, 1, 3))
            current_active = compose_sg[..., None] > current_eps
            current_active_count = current_active.astype(jnp.float32).sum(
                axis=(0, 1, 3))
            if _compute_sparsity_mass:
                current_mass = (
                    compose_sg[..., None] * current_active.astype(jnp.float32)
                ).sum(axis=(0, 1, 3))
                gate_mass = compose_sg.sum(axis=(0, 1, 3))
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
            if _compact_regular_diag:
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

        def spike_chunk_rows_paired(start, rho, raw_tau_chunk, tau_chunk,
                                    selection_margin, admission,
                                    angular_depth, compose_weight, xr_f,
                                    read_norm, write_norm):
            xr_r = xr_f[:, :, None, :]
            contrib_proxy = (
                jnp.abs(jax.lax.stop_gradient(compose_weight * xr_r))
                * write_norm[None, None, None, :])
            out_rows = []
            local_k = min(_spike_probe_topk, int(B * S * cs))
            for route_idx in range(2):
                flat_score = contrib_proxy[:, :, route_idx, :].reshape((-1,))
                top_scores, top_idx = jax.lax.top_k(flat_score, local_k)
                b_idx = top_idx // (S * cs)
                rem = top_idx - b_idx * (S * cs)
                pos_idx = rem // cs
                n_idx = rem - pos_idx * cs
                read_norm_i = jnp.take(read_norm, n_idx, axis=0)
                write_norm_i = jnp.take(write_norm, n_idx, axis=0)
                raw_tau_i = raw_tau_chunk[b_idx, pos_idx, route_idx, 0]
                tau_i = tau_chunk[b_idx, pos_idx, route_idx, 0]
                margin_i = selection_margin[b_idx, pos_idx, route_idx, n_idx]
                pos_margin_i = admission[b_idx, pos_idx, route_idx, n_idx]
                angular_depth_i = angular_depth[
                    b_idx, pos_idx, route_idx, n_idx]
                gate_i = compose_weight[b_idx, pos_idx, route_idx, n_idx]
                xr_i = xr_f[b_idx, pos_idx, n_idx]
                h_norm_i = jnp.linalg.norm(
                    jax.lax.stop_gradient(
                        h[b_idx, pos_idx, route_idx].astype(jnp.float32)),
                    axis=-1)
                resid_norm_i = jnp.linalg.norm(
                    jax.lax.stop_gradient(
                        x[b_idx, pos_idx].astype(jnp.float32)),
                    axis=-1)
                rows = jnp.stack([
                    top_scores,
                    b_idx.astype(jnp.float32),
                    pos_idx.astype(jnp.float32),
                    (jnp.int32(start) + n_idx).astype(jnp.float32),
                    rho[b_idx, pos_idx, route_idx, n_idx],
                    raw_tau_i,
                    tau_i,
                    margin_i,
                    pos_margin_i,
                    angular_depth_i,
                    gate_i,
                    jnp.zeros_like(top_scores),
                    jnp.zeros_like(top_scores),
                    jnp.zeros_like(top_scores),
                    jnp.zeros_like(top_scores),
                    jnp.zeros_like(top_scores),
                    xr_i,
                    jnp.abs(xr_i),
                    read_norm_i,
                    write_norm_i,
                    read_norm_i * write_norm_i,
                    top_scores,
                    jnp.zeros_like(top_scores),
                    jnp.zeros_like(top_scores),
                    resid_norm_i,
                    h_norm_i,
                ], axis=-1).astype(jnp.float32)
                if local_k < _spike_probe_topk:
                    pad = jnp.full(
                        (_spike_probe_topk - local_k,
                         SPIKE_SRW_FIELD_COUNT),
                        diag_neg_inf, dtype=jnp.float32)
                    pad = pad.at[:, 1:4].set(0.0)
                    rows = jnp.concatenate([rows, pad], axis=0)
                out_rows.append(rows)
            return jnp.stack(out_rows, axis=0)

        def spike_merge_rows_paired(rows_a, rows_b):
            merged = []
            for route_idx in range(2):
                merged.append(_topk_rows(
                    jnp.concatenate(
                        [rows_a[route_idx], rows_b[route_idx]], axis=0),
                    _spike_probe_topk, SPIKE_SRW_FIELD_COUNT))
            return jnp.stack(merged, axis=0)

        def spike_finalize_rows_paired(rows, raw_out, out,
                                       global_den_cost_m,
                                       global_gate_max_m, global_active_m,
                                       no_active_direct):
            finalized = []
            model_offset = jax.lax.axis_index('model') * N_local
            data_offset = jax.lax.axis_index('data') * B
            for route_idx in range(2):
                r = rows[route_idx]
                local_b = r[:, 1].astype(jnp.int32)
                pos = r[:, 2].astype(jnp.int32)
                den_i = global_den_cost_m[local_b, pos, route_idx, 0]
                top1_i = (
                    global_gate_max_m[local_b, pos, route_idx, 0]
                    / jnp.maximum(den_i, 1e-8))
                active_i = global_active_m[local_b, pos, route_idx, 0]
                no_active_i = no_active_direct[local_b, pos, route_idx, 0]
                raw_out_i = jnp.linalg.norm(
                    jax.lax.stop_gradient(
                        raw_out[local_b, pos, route_idx]),
                    axis=-1)
                out_i = jnp.linalg.norm(
                    jax.lax.stop_gradient(out[local_b, pos, route_idx]),
                    axis=-1)
                r = r.at[:, 1].set(r[:, 1] + data_offset)
                r = r.at[:, 3].set(r[:, 3] + model_offset)
                r = r.at[:, 11].set(den_i)
                r = r.at[:, 12].set(r[:, 10] / jnp.maximum(den_i, 1e-8))
                r = r.at[:, 13].set(top1_i)
                r = r.at[:, 14].set(active_i)
                r = r.at[:, 15].set(no_active_i)
                r = r.at[:, 22].set(raw_out_i)
                r = r.at[:, 23].set(out_i)
                gathered = jax.lax.all_gather(r, 'model', axis=0)
                gathered = jax.lax.all_gather(gathered, 'data', axis=0)
                finalized.append(_topk_rows(
                    gathered.reshape((-1, SPIKE_SRW_FIELD_COUNT)),
                    _spike_probe_topk, SPIKE_SRW_FIELD_COUNT))
            return jnp.stack(finalized, axis=0)

        # --- Pass 2: gate + srw fused ---
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
                 sparsity_carry, select_diag_carry, diag_vals, spike_rows) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                rho, rho_exposure = route_relation(h_bf, route)
                (selection_margin, admission, angular_depth, compose_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = angular_depth.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, compose_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_depth.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T  # [B,S,N]
                xr_f = xr.astype(jnp.float32)
                a = compose_weight * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = compose_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(compose_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    xr_r = xr_f[:, :, None, :]
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(compose_weight * xr_r))
                        * write_norm[None, None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(selection_margin),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(angular_depth),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(
                            xr_r + jnp.zeros_like(gate))), axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy, axis=(0, 1, 3)))
                    _route_norm_max = jnp.max(jnp.linalg.norm(
                        jax.lax.stop_gradient(route.astype(jnp.float32)),
                        axis=-1))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.repeat(_route_norm_max, 2))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
                if _spike_probe:
                    read_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            read_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    write_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            write_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    spike_rows = spike_merge_rows_paired(
                        spike_rows,
                        spike_chunk_rows_paired(
                            s, rho, raw_tau, tau, selection_margin,
                            admission, angular_depth, compose_weight, xr_f,
                            read_norm, write_norm))
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_margin_band = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
                chunk_margin_band_wide = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
                chunk_margin_band_mid = jnp.zeros((B, S, 2, 1), dtype=jnp.float32)
                g_safe = compose_weight + 1e-8
                chunk_g_log_g = (compose_weight * jnp.log(g_safe)).sum(
                    axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission))
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, compose_weight.max(axis=-1, keepdims=True)),
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
                        select_diag_carry,
                        diag_vals,
                        spike_rows), None

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
             select_diag_carry, diag_vals, spike_rows), _ = jax.lax.scan(
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
                 select_diag_carry0,
                 diag_vals_init,
                 spike_rows_init),
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
                 sparsity_carry, select_diag_carry, diag_vals, spike_rows) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                rho, rho_exposure = route_relation(h_bf, route)
                (selection_margin, admission, angular_depth, compose_weight,
                 active_mask, strong_mask) = angular_compose_parts(rho)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = angular_depth.mean(
                    axis=-1, keepdims=True)
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, compose_weight)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_depth.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = compose_weight * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = compose_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(compose_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission.sum(axis=-1, keepdims=True)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    xr_r = xr_f[:, :, None, :]
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(compose_weight * xr_r))
                        * write_norm[None, None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(selection_margin),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(angular_depth),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(
                            xr_r + jnp.zeros_like(gate))), axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy, axis=(0, 1, 3)))
                    _route_norm_max = jnp.max(jnp.linalg.norm(
                        jax.lax.stop_gradient(route.astype(jnp.float32)),
                        axis=-1))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.repeat(_route_norm_max, 2))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
                if _spike_probe:
                    read_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            read_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    write_norm = jnp.linalg.norm(
                        jax.lax.dynamic_slice_in_dim(
                            write_bf, s, cs, axis=0).astype(jnp.float32),
                        axis=-1)
                    spike_rows = spike_merge_rows_paired(
                        spike_rows,
                        spike_chunk_rows_paired(
                            s, rho, raw_tau, tau, selection_margin,
                            admission, angular_depth, compose_weight, xr_f,
                            read_norm, write_norm))
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
                        jnp.maximum(total_gate_max, compose_weight.max(axis=-1, keepdims=True)),
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
                        select_diag_carry,
                        diag_vals,
                        spike_rows), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_selection_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_exposure_sum, total_exposure_min,
             total_exposure_max, total_weak_exposure_count,
             total_int_max, total_int_cap_count, total_selection_residency_sum,
             total_selection_residency_count,
             total_edge_margin_stat,
             sparsity_carry,
             select_diag_carry, diag_vals, spike_rows), _ = jax.lax.scan(
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
                 select_diag_carry0,
                 diag_vals_init,
                 spike_rows_init),
                jnp.arange(nc))


        # Normalize per route independently
        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')   # sum(compose_weight)
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        global_selection_cost = jax.lax.psum(total_selection_cost, 'model')
        global_current_cost = jax.lax.psum(total_current_cost, 'model')
        selection_residency_loss = jnp.float32(0.0)
        edge_margin_stat = (
            jax.lax.psum(total_edge_margin_stat, 'model')
            / jnp.float32(B * S * 2 * N_total))
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        den = jnp.maximum(global_den_cost, 1.0)
        out = raw_out / den
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
        # Action path above keeps global_den_cost/global_weighted_cost live for
        # the SRW denominator and output gradient.
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
        sparsity_diag_out = finalize_sparsity_diag(sparsity_carry)
        local_diag_out = ()
        if _local_diagnostics:
            tau_abs_max = jnp.max(
                jax.lax.stop_gradient(tau[..., 0]),
                axis=(0, 1))
            top1_share_max = jnp.max(
                global_gate_max_m / jnp.maximum(global_den_cost_m, 1e-8),
                axis=(0, 1, 3))
            gate_den_sum_max = jnp.max(global_den_cost_m, axis=(0, 1, 3))
            local_out_norm_max = jnp.max(
                jnp.linalg.norm(jax.lax.stop_gradient(out), axis=-1),
                axis=(0, 1))
            residual_norm_max = jnp.repeat(
                jnp.max(jnp.linalg.norm(
                    jax.lax.stop_gradient(x), axis=-1)),
                2)
            h_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(h), axis=-1), axis=(0, 1))
            token_vals = jnp.stack([
                tau_abs_max, top1_share_max, gate_den_sum_max,
                local_out_norm_max, residual_norm_max, h_norm_max,
            ], axis=1)
            token_slots = jnp.array([0, 2, 3, 7, 8, 9], dtype=jnp.int32)
            metric_vals = diag_vals.at[:, token_slots].set(token_vals)
            metric_vals = jax.lax.stop_gradient(metric_vals)
            metric_vals = jax.lax.pmax(metric_vals, 'model')
            metric_vals = jax.lax.pmax(metric_vals, 'data')
            metric_locs = jnp.full(
                (2, LOCAL_SPIKE_METRIC_COUNT, 3), -1, dtype=jnp.int32)
            top1_details = jnp.zeros(
                (2, LOCAL_SPIKE_TOP1_COUNT), dtype=jnp.float32)
            top1_details = top1_details.at[:, 0].set(metric_vals[:, 2])
            top1_details = top1_details.at[:, 4].set(metric_vals[:, 1])
            top1_details = top1_details.at[:, 6].set(metric_vals[:, 3])
            top1_details = top1_details.at[:, 7].set(metric_vals[:, 4])
            top1_details = top1_details.at[:, 8].set(metric_vals[:, 5])
            top1_details = top1_details.at[:, 12].set(metric_vals[:, 6])
            top1_details = top1_details.at[:, 13].set(metric_vals[:, 7])
            top1_details = top1_details.at[:, 15].set(metric_vals[:, 9])
            top1_details = top1_details.at[:, 16].set(metric_vals[:, 10])
            top1_locs = jnp.full((2, 3), -1, dtype=jnp.int32)
            local_diag_out = (
                metric_vals.astype(jnp.float32), metric_locs,
                top1_details, top1_locs)
        spike_probe_out = ()
        if _spike_probe:
            spike_probe_out = (
                spike_finalize_rows_paired(
                    spike_rows, raw_out, out, global_den_cost_m,
                    global_gate_max_m, global_active_m,
                    no_active_direct).astype(jnp.float32),)
        if not analysis:
            return (slim_out + conc_out + route_split_out + select_diag_out
                    + dead_exposure_diag_out + (sparsity_diag_out,)
                    + local_diag_out + spike_probe_out)

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
        # local_diag_out is collected inline in pass 2 above; no replay path.
        return (slim_out + conc_out + route_split_out
                + (margin_band_frac_mean, margin_band_wide_frac,
                   margin_band_mid_frac, rho_skew, active_per_token_std,
                   gate_entropy, den_cost_out, selection_cost_out,
                   current_cost_out, rho_kurt, int_cap_frac_out)
                + select_diag_out + dead_exposure_diag_out
                + (sparsity_diag_out,)
                + local_diag_out + spike_probe_out)

    return fused_gate_srw_paired


# ================================================================
# 4. NeuronPool -- route emb + read/write operator vectors
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    d_model: int
    d_route: int
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Legacy alias accepted from older configs.

    def setup(self):
        db = self.d_route
        dm = self.d_model
        n_rst_eff = self.n_rst if self.n_rst is not None else self.n_know
        if n_rst_eff is None:
            raise ValueError("NeuronPool requires n_rst or legacy n_know.")

        # Learned route embeddings use the full d_route angular space.
        self.attn_qk_emb = self.param('attn_qk_emb', unit_norm_init(), (self.n_qk, db))
        self.attn_v_emb = self.param('attn_v_emb', unit_norm_init(), (self.n_v, db))
        self.rst_emb = self.param('rst_emb', unit_norm_init(), (n_rst_eff, db))

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
    n_know: Optional[int] = None  # Legacy alias accepted from older configs.
    router_dropout: float = 0.1
    # Constructor receives cosine-space tau values. The train driver may use
    # safe placeholders before one-time quantile calibration.
    tau_init_attn_qk: Optional[float] = None
    tau_init_attn_v: Optional[float] = None
    tau_init_rst: Optional[float] = None

    def setup(self):
        db = self.d_route
        n_rst_eff = self.n_rst if self.n_rst is not None else self.n_know
        if n_rst_eff is None:
            raise ValueError("Router requires n_rst or legacy n_know.")

        missing_tau = [
            name for name, value in (
                ('tau_init_attn_qk', self.tau_init_attn_qk),
                ('tau_init_attn_v', self.tau_init_attn_v),
                ('tau_init_rst', self.tau_init_rst),
            ) if value is None
        ]
        if missing_tau:
            raise ValueError(
                "v4164 requires explicit cosine-space tau_init_attn_qk/v/rst; "
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
        self.proj_attn = nn.Dense(db * 3, name='proj_attn')
        self.proj_rst = nn.Dense(db, name='proj_rst')
        self.raw_tau_attn = nn.Dense(3, name='raw_tau_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: raw_tau_attn_bias_init.astype(d))
        self.raw_tau_rst = nn.Dense(1, name='raw_tau_rst',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, raw_tau_rst_bias_init, d))


# ================================================================
# 6. Pure functions for scan body
# ================================================================

def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v,
                  n_heads, d_model, n_layers,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False,
                  local_diagnostics=False,
                  spike_probe=False,
                  spike_probe_topk=8,
                  focus_probe_enabled=False,
                  focus_b=None,
                  focus_pos=None,
                  focus_rank=None,
                  focus_target_emb=None,
                  focus_pred_emb=None,
                  soft_gate_temperature=0.07,
                  soft_gate_t_final=0.07,
                  soft_gate_T_qk=None,
                  soft_gate_T_v=None,
                  soft_gate_boundary_power=2.0,
                  soft_gate_boundary_power_final=4.0,
                  execution_prune_eps=0.0):
    """v4.1: sharded-only. sharded_fns=(fused_single, fused_paired) required.

    `analysis=False` (train path): returns the SLIM tuple. `analysis=True`:
    returns the SLIM tuple extended with observational ANALYSIS stats
    (see scan_body below for the full unpack shape).
    """
    B, S, D = x.shape
    soft_gate_T_qk = (
        soft_gate_temperature if soft_gate_T_qk is None else soft_gate_T_qk)
    soft_gate_T_v = (
        soft_gate_temperature if soft_gate_T_v is None else soft_gate_T_v)
    qk_emb = pool_params['attn_qk_emb']
    qk_read = pool_params['attn_qk_read']
    qk_write = pool_params['attn_qk_write']
    v_emb = pool_params['attn_v_emb']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']

    # Raw signature params are passed into the sharded SRW closure.
    # The closure can forward-normalize them for routing stability while
    # retaining raw parameter norms as diagnostics.
    qk_emb_unit = qk_emb
    v_emb_unit = v_emb

    # Emb-norm monitoring is observational only.
    _qk_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(qk_emb, axis=-1))
    attn_qk_emb_norm_mean = _qk_emb_norms.mean()
    attn_qk_emb_norm_min = _qk_emb_norms.min()
    attn_qk_emb_norm_std = _qk_emb_norms.std()
    _v_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(v_emb, axis=-1))
    attn_v_emb_norm_mean = _v_emb_norms.mean()
    attn_v_emb_norm_min = _v_emb_norms.min()
    attn_v_emb_norm_std = _v_emb_norms.std()
    if analysis:
        attn_qk_emb_norm_max = _qk_emb_norms.max()
        attn_v_emb_norm_max = _v_emb_norms.max()

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

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
    qk_ret = fused_paired(x, h_QK, qk_emb_unit, raw_tau_QK,
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
    if spike_probe:
        qk_spike_rows = qk_ret[-1]
    if local_diagnostics:
        _qk_local_tail = qk_ret[-5:-1] if spike_probe else qk_ret[-4:]
        (qk_local_values, qk_local_locs,
         qk_top1_values, qk_top1_locs) = _qk_local_tail
    Q = QK_out[:, :, 0, :] * qk_scale
    K = QK_out[:, :, 1, :] * qk_scale
    v_ret = fused_single_v(x, h_V, v_emb_unit, raw_tau_all[:, :, 2:3],
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
    v_sparsity_start = v_exposure_start + DEAD_EXPOSURE_DIAG_COUNT
    v_sparsity_diag = v_ret[v_sparsity_start]
    if spike_probe:
        v_spike_rows = v_ret[-1]
    if local_diagnostics:
        _v_local_tail = v_ret[-5:-1] if spike_probe else v_ret[-4:]
        (v_local_values, v_local_locs,
         v_top1_values, v_top1_locs) = _v_local_tail
    V = V * v_scale

    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    rng, rng_attn_drop = jax.random.split(rng)
    _spike_probe_topk = max(1, int(spike_probe_topk))
    _focus_probe_enabled = bool(focus_probe_enabled)
    _focus_take_k = _spike_probe_topk
    _focus_b = focus_b if focus_b is not None else jnp.zeros((_focus_take_k,), dtype=jnp.int32)
    _focus_pos = focus_pos if focus_pos is not None else jnp.zeros((_focus_take_k,), dtype=jnp.int32)
    _focus_rank = focus_rank if focus_rank is not None else jnp.arange(_focus_take_k, dtype=jnp.float32)
    _focus_target_emb = (focus_target_emb.astype(jnp.float32)
                         if focus_target_emb is not None
                         else jnp.zeros((_focus_take_k, D), dtype=jnp.float32))
    _focus_pred_emb = (focus_pred_emb.astype(jnp.float32)
                       if focus_pred_emb is not None
                       else jnp.zeros((_focus_take_k, D), dtype=jnp.float32))
    def _merge_focus_topk(candidates, score_col, field_count):
        scores = candidates[:, :, score_col]
        vals, idx = jax.lax.top_k(scores, min(_focus_take_k, int(candidates.shape[1])))
        rows = jnp.take_along_axis(candidates, idx[:, :, None], axis=1)
        rows = rows.at[:, :, score_col].set(vals)
        return rows.astype(jnp.float32)

    def _focus_srw_rows_for_pool(pool_id, x_in, h_in, emb, raw_tau, read, write, pool_scale):
        hf = h_in[_focus_b, _focus_pos].astype(jnp.float32)
        xf = x_in[_focus_b, _focus_pos].astype(jnp.float32)
        emb_f = emb.astype(jnp.float32)
        read_f = read.astype(jnp.float32)
        write_f = write.astype(jnp.float32)
        q = _forward_unit_direction(hf)
        e = _forward_unit_direction(emb_f)
        rho = q @ e.T
        tau = _tau_from_param(raw_tau[_focus_b, _focus_pos, 0].astype(jnp.float32))[:, None]
        pool_soft_gate_T = soft_gate_T_v if pool_id == 2 else soft_gate_T_qk
        (margin, admission, angular_depth, compose_weight,
         active_mask) = _compute_admission_depth(
            rho, tau, pool_soft_gate_T,
            boundary_power=soft_gate_boundary_power,
            execution_prune_eps=execution_prune_eps)
        den = jnp.maximum(admission.sum(axis=-1), 1.0)
        active_n = active_mask.astype(jnp.float32).sum(axis=-1)
        read_dir = _forward_unit_direction(read_f)
        write_dir = _forward_unit_direction(write_f)
        xr = xf @ read_dir.T
        weighted = compose_weight * xr
        raw_out = weighted @ write_dir
        out_vec = raw_out / den[:, None] * pool_scale
        raw_out_norm = jnp.linalg.norm(raw_out, axis=-1)
        out_norm = jnp.linalg.norm(out_vec, axis=-1)
        read_norm = jnp.linalg.norm(read_f, axis=-1)
        write_norm = jnp.linalg.norm(write_f, axis=-1)
        op_gain = read_norm * write_norm
        read_target = _focus_target_emb @ read_dir.T
        read_pred = _focus_pred_emb @ read_dir.T
        write_target = _focus_target_emb @ write_dir.T
        write_pred = _focus_pred_emb @ write_dir.T
        write_pred_minus_target = write_pred - write_target
        contrib_pred_minus_target = (weighted / jnp.maximum(den[:, None], 1e-8)
                                     * pool_scale * write_pred_minus_target)
        score = jnp.abs(weighted) * write_norm[None, :]
        vals, idx = jax.lax.top_k(score, min(_focus_take_k, int(score.shape[-1])))
        def take(a):
            return jnp.take_along_axis(a, idx, axis=1)
        K = vals.shape[1]
        return jnp.stack([
            jnp.broadcast_to(_focus_rank[:, None], vals.shape),
            jnp.broadcast_to(_focus_b[:, None].astype(jnp.float32), vals.shape),
            jnp.broadcast_to(_focus_pos[:, None].astype(jnp.float32), vals.shape),
            jnp.zeros_like(vals),  # layer filled by scan_body
            jnp.full(vals.shape, jnp.float32(pool_id)),
            idx.astype(jnp.float32),
            vals,
            take(rho),
            jnp.broadcast_to(tau, vals.shape),
            take(margin),
            take(admission),
            take(angular_depth),
            take(compose_weight),
            jnp.broadcast_to(den[:, None], vals.shape),
            take(compose_weight) / jnp.maximum(den[:, None], 1e-8),
            jnp.broadcast_to(active_n[:, None], vals.shape),
            take(xr),
            jnp.abs(take(xr)),
            jnp.take(read_norm, idx),
            jnp.take(write_norm, idx),
            jnp.take(op_gain, idx),
            jnp.broadcast_to(raw_out_norm[:, None], vals.shape),
            jnp.broadcast_to(out_norm[:, None], vals.shape),
            jnp.broadcast_to(jnp.linalg.norm(xf, axis=-1)[:, None], vals.shape),
            jnp.broadcast_to(jnp.linalg.norm(hf, axis=-1)[:, None], vals.shape),
            take(read_target),
            take(read_pred),
            take(write_target),
            take(write_pred),
            take(write_pred_minus_target),
            take(contrib_pred_minus_target),
        ], axis=-1).astype(jnp.float32)

    def _attention_focus_rows(attn_scores, attn_w, out_dbg):
        scores_sg = jax.lax.stop_gradient(attn_scores.astype(jnp.float32))
        score_floor = jnp.finfo(scores_sg.dtype).min
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        causal_4d = causal[None, None, :, :]
        masked_scores = jnp.where(causal_4d, scores_sg, score_floor)
        attn_w_sg = jax.lax.stop_gradient(attn_w.astype(jnp.float32))
        q_norms = jnp.linalg.norm(jax.lax.stop_gradient(Q), axis=-1)
        k_norms = jnp.linalg.norm(jax.lax.stop_gradient(K), axis=-1)
        v_norms = jnp.linalg.norm(jax.lax.stop_gradient(V), axis=-1)
        o_in_norm = jnp.linalg.norm(jax.lax.stop_gradient(out_dbg), axis=-1)
        entropy_terms = jnp.where(
            attn_w_sg > 0.0,
            attn_w_sg * jnp.log(jnp.maximum(attn_w_sg, 1e-30)),
            0.0)
        entropy = -jnp.sum(entropy_terms, axis=-1)
        top1_logits = jnp.max(masked_scores, axis=-1)
        top1_idx = jnp.argmax(masked_scores, axis=-1)
        attn_idx = jnp.arange(S)
        second_scores = jnp.where(
            attn_idx[None, None, None, :] == top1_idx[..., None],
            score_floor, masked_scores)
        top2_logits = jnp.max(second_scores, axis=-1)
        has_top2 = (jnp.arange(S) + 1) > 1
        top2_logits = jnp.where(has_top2[None, None, :], top2_logits, top1_logits)
        gap = top1_logits - top2_logits

        cand_logits = masked_scores[_focus_b, :, _focus_pos, :]  # [K,H,S]
        cand_weights = attn_w_sg[_focus_b, :, _focus_pos, :]
        # Focus attention rows must rank only causally-valid keys.  The
        # masked future logits are ~finfo.min; using abs(logit) accidentally
        # made those future keys look largest.  Rank by attention weight after
        # applying the causal valid-key mask so the printed rows are real keys.
        valid_keys = causal[_focus_pos, :][:, None, :]  # [K,1,S]
        cand_score = jnp.where(
            valid_keys,
            cand_weights,
            jnp.full_like(cand_weights, -jnp.inf),
        )
        flat_score = cand_score.reshape((_focus_take_k, -1))
        vals, flat_idx = jax.lax.top_k(flat_score, min(_focus_take_k, int(flat_score.shape[-1])))
        head_idx = flat_idx // S
        key_idx = flat_idx - head_idx * S
        q_idx = _focus_pos[:, None]
        batch_idx = _focus_b[:, None]
        gather_b = _focus_b[:, None]
        gather_q = _focus_pos[:, None]
        cand_logit = jnp.take_along_axis(cand_logits.reshape((_focus_take_k, -1)), flat_idx, axis=1)
        cand_weight = jnp.take_along_axis(cand_weights.reshape((_focus_take_k, -1)), flat_idx, axis=1)
        return jnp.stack([
            jnp.broadcast_to(_focus_rank[:, None], vals.shape),
            jnp.broadcast_to(batch_idx.astype(jnp.float32), vals.shape),
            jnp.broadcast_to(q_idx.astype(jnp.float32), vals.shape),
            jnp.zeros_like(vals),  # layer filled by scan_body
            head_idx.astype(jnp.float32),
            key_idx.astype(jnp.float32),
            vals,
            cand_logit,
            cand_weight,
            gap[_focus_b[:, None], head_idx, _focus_pos[:, None]],
            entropy[_focus_b[:, None], head_idx, _focus_pos[:, None]],
            q_norms[_focus_b[:, None], head_idx, _focus_pos[:, None]],
            k_norms[_focus_b[:, None], head_idx, key_idx],
            v_norms[_focus_b[:, None], head_idx, key_idx],
            o_in_norm[_focus_b[:, None], head_idx, _focus_pos[:, None]],
            jnp.zeros_like(vals),  # projected o_out_norm filled after expand_O
            (_focus_pos[:, None].astype(jnp.float32) - key_idx.astype(jnp.float32)),
            (key_idx == _focus_pos[:, None]).astype(jnp.float32),
        ], axis=-1).astype(jnp.float32)

    def _attention_spike_rows(attn_scores, attn_w, out_dbg):
        scores_sg = jax.lax.stop_gradient(attn_scores.astype(jnp.float32))
        score_floor = jnp.finfo(scores_sg.dtype).min
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        causal_4d = causal[None, None, :, :]
        masked_scores = jnp.where(causal_4d, scores_sg, score_floor)
        top1_logits = jnp.max(masked_scores, axis=-1)
        min_logits = jnp.min(jnp.where(causal_4d, scores_sg, jnp.inf), axis=-1)
        key_idx = jnp.argmax(masked_scores, axis=-1)
        attn_idx = jnp.arange(S)
        second_scores = jnp.where(
            attn_idx[None, None, None, :] == key_idx[..., None],
            score_floor,
            masked_scores)
        top2_logits = jnp.max(second_scores, axis=-1)
        has_top2 = (jnp.arange(S) + 1) > 1
        top2_logits = jnp.where(
            has_top2[None, None, :], top2_logits, top1_logits)
        gap = top1_logits - top2_logits
        attn_w_sg = jax.lax.stop_gradient(attn_w.astype(jnp.float32))
        top1_w = jnp.max(attn_w_sg, axis=-1)
        entropy_terms = jnp.where(
            attn_w_sg > 0.0,
            attn_w_sg * jnp.log(jnp.maximum(attn_w_sg, 1e-30)),
            0.0)
        entropy = -jnp.sum(entropy_terms, axis=-1)
        q_norms = jnp.linalg.norm(jax.lax.stop_gradient(Q), axis=-1)
        k_norms = jnp.linalg.norm(jax.lax.stop_gradient(K), axis=-1)
        v_norms = jnp.linalg.norm(jax.lax.stop_gradient(V), axis=-1)
        k_norm_i = jnp.take_along_axis(
            k_norms, key_idx.astype(jnp.int32), axis=-1)
        v_norm_i = jnp.take_along_axis(
            v_norms, key_idx.astype(jnp.int32), axis=-1)
        o_in_norm = jnp.linalg.norm(
            jax.lax.stop_gradient(out_dbg), axis=-1)
        score = jnp.maximum(
            jnp.abs(top1_logits),
            jnp.maximum(top1_w * jnp.float32(10.0), o_in_norm))
        flat_score = score.reshape((-1,))
        local_k = min(_spike_probe_topk, int(flat_score.shape[0]))
        top_scores, flat_idx = jax.lax.top_k(flat_score, local_k)
        b_idx = flat_idx // (n_heads * S)
        rem = flat_idx - b_idx * (n_heads * S)
        h_idx = rem // S
        q_idx = rem - h_idx * S
        rows = jnp.stack([
            top_scores,
            b_idx.astype(jnp.float32),
            q_idx.astype(jnp.float32),
            key_idx[b_idx, h_idx, q_idx].astype(jnp.float32),
            top1_logits[b_idx, h_idx, q_idx],
            min_logits[b_idx, h_idx, q_idx],
            gap[b_idx, h_idx, q_idx],
            top1_w[b_idx, h_idx, q_idx],
            entropy[b_idx, h_idx, q_idx],
            q_norms[b_idx, h_idx, q_idx],
            k_norm_i[b_idx, h_idx, q_idx],
            v_norm_i[b_idx, h_idx, q_idx],
            o_in_norm[b_idx, h_idx, q_idx],
            jnp.zeros_like(top_scores),
        ], axis=-1).astype(jnp.float32)
        if local_k < _spike_probe_topk:
            pad = jnp.full(
                (_spike_probe_topk - local_k, SPIKE_ATTN_FIELD_COUNT),
                -jnp.inf, dtype=jnp.float32)
            pad = pad.at[:, 1:4].set(0.0)
            rows = jnp.concatenate([rows, pad], axis=0)
        return rows

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
        elif local_diagnostics:
            attn_logit_max_dbg = jnp.max(attn_scores)
            softmax_top1_max = jnp.max(attn_w)
        attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_drop)
        out_dbg = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
        if spike_probe:
            attn_rows = _attention_spike_rows(attn_scores, attn_w, out_dbg)
            if _focus_probe_enabled:
                attn_focus_rows = _attention_focus_rows(attn_scores, attn_w, out_dbg)
        if analysis:
            return (
                out_dbg,
                attn_logit_mean, attn_logit_std,
                attn_logit_max_dbg,
                softmax_top1_mean, softmax_top1_max,
                logit_gap_mean, logit_gap_max,
                softmax_entropy_mean, softmax_entropy_min,
            )
        if local_diagnostics:
            return out_dbg, attn_logit_max_dbg, softmax_top1_max
        if spike_probe:
            if _focus_probe_enabled:
                return out_dbg, attn_rows, attn_focus_rows
            return out_dbg, attn_rows
        return out_dbg

    if analysis or local_diagnostics or spike_probe:
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
    elif local_diagnostics:
        out, attn_logit_max_actual, attn_softmax_top1_max = _attn_scores(
            Q, K, V, rng_attn_drop)
    elif spike_probe:
        if _focus_probe_enabled:
            out, attn_spike_rows, attn_focus_rows = _attn_scores(Q, K, V, rng_attn_drop)
        else:
            out, attn_spike_rows = _attn_scores(Q, K, V, rng_attn_drop)
    else:
        out = _attn_scores(Q, K, V, rng_attn_drop)
    if analysis or local_diagnostics or spike_probe:
        o_input_norm = jnp.linalg.norm(out, axis=-1).mean()
        if local_diagnostics and not analysis:
            q_norm_max = q_norms_dbg.max()
            k_norm_max = k_norms_dbg.max()
        v_norm_max = v_norms_dbg.max()
        o_input_norm_max = jnp.linalg.norm(out, axis=-1).max()
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    attn_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    if analysis or local_diagnostics or spike_probe:
        o_out_norm_max = jnp.linalg.norm(out, axis=-1).max()
    if spike_probe:
        _row_b = attn_spike_rows[:, 1].astype(jnp.int32)
        _row_q = attn_spike_rows[:, 2].astype(jnp.int32)
        _o_out_by_token = jnp.linalg.norm(out, axis=-1)
        attn_spike_rows = attn_spike_rows.at[:, 13].set(
            _o_out_by_token[_row_b, _row_q])
        if _focus_probe_enabled:
            attn_focus_rows = attn_focus_rows.at[:, :, 15].set(
                _o_out_by_token[_focus_b, _focus_pos][:, None])
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
    # Exploration/RPE feedback consumes per-layer direct tau/no-active stacks.
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
    if spike_probe and _focus_probe_enabled:
        _q_focus_rows = _focus_srw_rows_for_pool(
            0, x, h_Q, qk_emb_unit, raw_tau_all[:, :, 0:1],
            qk_read, qk_write, qk_scale)
        _k_focus_rows = _focus_srw_rows_for_pool(
            1, x, h_K, qk_emb_unit, raw_tau_all[:, :, 1:2],
            qk_read, qk_write, qk_scale)
        _v_focus_rows = _focus_srw_rows_for_pool(
            2, x, h_V, v_emb_unit, raw_tau_all[:, :, 2:3],
            v_read, v_write, v_scale)
        attn_focus_srw_rows = _merge_focus_topk(
            jnp.concatenate([_q_focus_rows, _k_focus_rows, _v_focus_rows], axis=1),
            6, SPIKE_FOCUS_SRW_FIELD_COUNT)
    slim_ret = (out, aux, qk_active.mean(), v_active.mean(), attn_raw_gmax,
                attn_rho_std_slim, attn_gate_sum, attn_active_n_mean,
                attn_out_norm, attn_tau_mean,
                attn_strong,
                qk_strong.mean(), v_strong.mean(),
                attn_qk_positive_margin_mean_active,
                attn_v_positive_margin_mean_active,
                attn_tau_abs_mean,
                attn_qk_emb_norm_mean, attn_v_emb_norm_mean,
                attn_qk_emb_norm_min, attn_qk_emb_norm_std,
                attn_v_emb_norm_min, attn_v_emb_norm_std,
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
                k_sparsity_diag)
    if not analysis:
        ret = slim_ret
        if local_diagnostics:
            attn_local_layer_values = jnp.stack([
                q_norm_max, k_norm_max, v_norm_max,
                attn_logit_max_actual, attn_softmax_top1_max,
                o_input_norm_max, o_out_norm_max,
            ])
            attn_local_values = jnp.concatenate(
                [qk_local_values, v_local_values], axis=0)
            attn_local_locs = jnp.concatenate(
                [qk_local_locs, v_local_locs], axis=0)
            attn_top1_values = jnp.concatenate(
                [qk_top1_values, v_top1_values], axis=0)
            attn_top1_locs = jnp.concatenate(
                [qk_top1_locs, v_top1_locs], axis=0)
            ret = ret + (
                attn_local_layer_values,
                attn_local_values, attn_local_locs,
                attn_top1_values, attn_top1_locs,
            )
        if spike_probe:
            if _focus_probe_enabled:
                ret = ret + (
                    jnp.concatenate([qk_spike_rows, v_spike_rows], axis=0),
                    attn_spike_rows,
                    attn_focus_srw_rows,
                    attn_focus_rows,
                    o_out_norm_max.astype(jnp.float32),
                )
            else:
                ret = ret + (
                    jnp.concatenate([qk_spike_rows, v_spike_rows], axis=0),
                    attn_spike_rows,
                    o_out_norm_max.astype(jnp.float32),
                )
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
        attn_qk_emb_norm_max, attn_v_emb_norm_max,
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
    if local_diagnostics:
        attn_local_layer_values = jnp.stack([
            q_norm_max, k_norm_max, v_norm_max,
            attn_logit_max_actual, attn_softmax_top1_max,
            o_input_norm_max, o_out_norm_max,
        ])
        attn_local_values = jnp.concatenate([qk_local_values, v_local_values], axis=0)
        attn_local_locs = jnp.concatenate([qk_local_locs, v_local_locs], axis=0)
        attn_top1_values = jnp.concatenate([qk_top1_values, v_top1_values], axis=0)
        attn_top1_locs = jnp.concatenate([qk_top1_locs, v_top1_locs], axis=0)
        analysis_ret = analysis_ret + (
            attn_local_layer_values,
            attn_local_values, attn_local_locs,
            attn_top1_values, attn_top1_locs,
        )
    if spike_probe:
        if _focus_probe_enabled:
            analysis_ret = analysis_ret + (
                jnp.concatenate([qk_spike_rows, v_spike_rows], axis=0),
                attn_spike_rows,
                attn_focus_srw_rows,
                attn_focus_rows,
                o_out_norm_max.astype(jnp.float32),
            )
        else:
            analysis_ret = analysis_ret + (
                jnp.concatenate([qk_spike_rows, v_spike_rows], axis=0),
                attn_spike_rows,
                o_out_norm_max.astype(jnp.float32),
            )
    return analysis_ret


def _rst_forward(x, pool_params, router_params, rng,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False,
                  local_diagnostics=False,
                  spike_probe=False,
                  spike_probe_topk=8,
                  d_model=None, n_layers=None,
                  focus_probe_enabled=False,
                  focus_b=None,
                  focus_pos=None,
                  focus_rank=None,
                  focus_target_emb=None,
                  focus_pred_emb=None,
                  soft_gate_temperature=0.07,
                  soft_gate_t_final=0.07,
                  soft_gate_T_rst=None,
                  soft_gate_boundary_power=2.0,
                  soft_gate_boundary_power_final=4.0,
                  execution_prune_eps=0.0):
    """v4.1: sharded-only. sharded_fns=(fused_single, fused_paired) required.

    `analysis` see _attn_forward docstring.
    """
    B, S, D = x.shape
    soft_gate_T_rst = (
        soft_gate_temperature
        if soft_gate_T_rst is None else soft_gate_T_rst)
    rst_emb = pool_params['rst_emb']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    # Raw signature params are passed into the sharded SRW closure.
    # The closure can forward-normalize them for routing stability while
    # retaining raw parameter norms as diagnostics.
    rst_emb_unit = rst_emb
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
    _focus_probe_enabled = bool(focus_probe_enabled)
    _focus_take_k = max(1, int(spike_probe_topk))
    _focus_b = focus_b if focus_b is not None else jnp.zeros((_focus_take_k,), dtype=jnp.int32)
    _focus_pos = focus_pos if focus_pos is not None else jnp.zeros((_focus_take_k,), dtype=jnp.int32)
    _focus_rank = focus_rank if focus_rank is not None else jnp.arange(_focus_take_k, dtype=jnp.float32)
    _focus_target_emb = (focus_target_emb.astype(jnp.float32)
                         if focus_target_emb is not None
                         else jnp.zeros((_focus_take_k, D), dtype=jnp.float32))
    _focus_pred_emb = (focus_pred_emb.astype(jnp.float32)
                       if focus_pred_emb is not None
                       else jnp.zeros((_focus_take_k, D), dtype=jnp.float32))
    def _focus_srw_rows_for_pool(pool_id, x_in, h_in, emb, raw_tau, read, write, pool_scale):
        hf = h_in[_focus_b, _focus_pos].astype(jnp.float32)
        xf = x_in[_focus_b, _focus_pos].astype(jnp.float32)
        emb_f = emb.astype(jnp.float32)
        read_f = read.astype(jnp.float32)
        write_f = write.astype(jnp.float32)
        q = _forward_unit_direction(hf)
        e = _forward_unit_direction(emb_f)
        rho = q @ e.T
        tau = _tau_from_param(raw_tau[_focus_b, _focus_pos, 0].astype(jnp.float32))[:, None]
        (margin, admission, angular_depth, compose_weight,
         active_mask) = _compute_admission_depth(
            rho, tau, soft_gate_T_rst,
            boundary_power=soft_gate_boundary_power,
            execution_prune_eps=execution_prune_eps)
        den = jnp.maximum(admission.sum(axis=-1), 1.0)
        active_n = active_mask.astype(jnp.float32).sum(axis=-1)
        read_dir = _forward_unit_direction(read_f)
        write_dir = _forward_unit_direction(write_f)
        xr = xf @ read_dir.T
        weighted = compose_weight * xr
        raw_out = weighted @ write_dir
        out_vec = raw_out / den[:, None] * pool_scale
        raw_out_norm = jnp.linalg.norm(raw_out, axis=-1)
        out_norm = jnp.linalg.norm(out_vec, axis=-1)
        read_norm = jnp.linalg.norm(read_f, axis=-1)
        write_norm = jnp.linalg.norm(write_f, axis=-1)
        op_gain = read_norm * write_norm
        read_target = _focus_target_emb @ read_dir.T
        read_pred = _focus_pred_emb @ read_dir.T
        write_target = _focus_target_emb @ write_dir.T
        write_pred = _focus_pred_emb @ write_dir.T
        write_pred_minus_target = write_pred - write_target
        contrib_pred_minus_target = (weighted / jnp.maximum(den[:, None], 1e-8)
                                     * pool_scale * write_pred_minus_target)
        score = jnp.abs(weighted) * write_norm[None, :]
        vals, idx = jax.lax.top_k(score, min(_focus_take_k, int(score.shape[-1])))
        def take(a):
            return jnp.take_along_axis(a, idx, axis=1)
        return jnp.stack([
            jnp.broadcast_to(_focus_rank[:, None], vals.shape),
            jnp.broadcast_to(_focus_b[:, None].astype(jnp.float32), vals.shape),
            jnp.broadcast_to(_focus_pos[:, None].astype(jnp.float32), vals.shape),
            jnp.zeros_like(vals),  # layer filled by scan_body
            jnp.full(vals.shape, jnp.float32(pool_id)),
            idx.astype(jnp.float32),
            vals,
            take(rho),
            jnp.broadcast_to(tau, vals.shape),
            take(margin),
            take(admission),
            take(angular_depth),
            take(compose_weight),
            jnp.broadcast_to(den[:, None], vals.shape),
            take(compose_weight) / jnp.maximum(den[:, None], 1e-8),
            jnp.broadcast_to(active_n[:, None], vals.shape),
            take(xr),
            jnp.abs(take(xr)),
            jnp.take(read_norm, idx),
            jnp.take(write_norm, idx),
            jnp.take(op_gain, idx),
            jnp.broadcast_to(raw_out_norm[:, None], vals.shape),
            jnp.broadcast_to(out_norm[:, None], vals.shape),
            jnp.broadcast_to(jnp.linalg.norm(xf, axis=-1)[:, None], vals.shape),
            jnp.broadcast_to(jnp.linalg.norm(hf, axis=-1)[:, None], vals.shape),
            take(read_target),
            take(read_pred),
            take(write_target),
            take(write_pred),
            take(write_pred_minus_target),
            take(contrib_pred_minus_target),
        ], axis=-1).astype(jnp.float32)

    if isinstance(sharded_fns, dict):
        fused_single = sharded_fns.get('rst_single', sharded_fns['single'])
    else:
        fused_single, _ = sharded_fns
    rst_ret = fused_single(x, h, rst_emb_unit, raw_tau,
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
    if spike_probe:
        rst_spike_rows = rst_ret[-1]
    if local_diagnostics:
        _rst_local_tail = rst_ret[-5:-1] if spike_probe else rst_ret[-4:]
        (rst_local_values, rst_local_locs,
         rst_top1_values, rst_top1_locs) = _rst_local_tail
    out = out * rst_scale
    rst_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rst_out_norm_max = jnp.linalg.norm(out, axis=-1).max()
    if spike_probe and _focus_probe_enabled:
        rst_focus_srw_rows = _focus_srw_rows_for_pool(
            3, x, h, rst_emb_unit, raw_tau, rst_read, rst_write, rst_scale)
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    _rst_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(rst_emb, axis=-1))
    emb_norm_val = _rst_emb_norms.mean()
    rst_emb_norm_min = _rst_emb_norms.min()
    rst_emb_norm_std = _rst_emb_norms.std()
    if analysis:
        rst_emb_norm_max = _rst_emb_norms.max()
    read_norm_val = jnp.linalg.norm(rst_read, axis=-1).mean()
    write_norm_val = jnp.linalg.norm(rst_write, axis=-1).mean()
    rst_tau_mean = tau.mean()
    rst_strong = strong_frac.mean()
    rst_positive_margin_mean_active = positive_margin_mean_active.mean()
    slim_ret = (out, aux, active_frac, raw_gate_max, rho_std_slim, gate_sum, active_n_mean,
                emb_norm_val, read_norm_val, write_norm_val, rst_out_norm,
                rst_tau_mean, rst_strong, rst_positive_margin_mean_active,
                rst_tau_abs_mean,
                rst_emb_norm_min, rst_emb_norm_std,
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
                rst_sparsity_diag)
    if not analysis:
        ret = slim_ret
        if local_diagnostics:
            ret = ret + (
                rst_local_values, rst_local_locs,
                rst_top1_values, rst_top1_locs,
            )
        if spike_probe:
            if _focus_probe_enabled:
                ret = ret + (
                    rst_spike_rows,
                    rst_focus_srw_rows,
                    rst_out_norm_max.astype(jnp.float32),
                )
            else:
                ret = ret + (
                    rst_spike_rows,
                    rst_out_norm_max.astype(jnp.float32),
                )
        return ret

    rst_margin_band = margin_band_frac.mean()
    analysis_ret = slim_ret + (
        rst_raw_out_norm,
        rst_tau_std, rst_tau_kernel_norm,
        rst_margin_band_wide_frac, rst_margin_band_mid_frac,
        rst_rho_skew, rst_active_per_token_std, rst_gate_entropy,
        rst_den_cost,
        rst_selection_cost, rst_current_cost,
        rst_emb_norm_max,
        rst_rho_kurt,
        rst_margin_band,
        rst_int_cap_frac,
    )
    if local_diagnostics:
        analysis_ret = analysis_ret + (
            rst_local_values, rst_local_locs,
            rst_top1_values, rst_top1_locs,
        )
    if spike_probe:
        if _focus_probe_enabled:
            analysis_ret = analysis_ret + (
                rst_spike_rows,
                rst_focus_srw_rows,
                rst_out_norm_max.astype(jnp.float32),
            )
        else:
            analysis_ret = analysis_ret + (
                rst_spike_rows,
                rst_out_norm_max.astype(jnp.float32),
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

class DAWN(nn.Module):
    """DAWN-SRW v4.1.6.4 with angular-depth boundary composition."""
    __version__ = "spatial-r1-v4.1.6.4"

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
    n_know: Optional[int] = None  # Legacy config alias; prefer n_rst in new configs.
    router_dropout: float = 0.1
    n_chunks_rst: Optional[int] = None
    n_chunks_know: int = 1    # Legacy config alias; prefer n_chunks_rst.
    n_chunks_qk: int = 1     # N-axis chunking for qk pool
    n_chunks_v: int = 1      # N-axis chunking for v pool
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
                 local_diagnostics=False, spike_probe=False,
                 spike_probe_topk=8,
                 spike_focus_bpos=None,
                 spike_focus_input_ids=None,
                 spike_focus_target_ids=None,
                 spike_focus_pred_ids=None,
                 soft_gate_temperature=0.07,
                 soft_gate_t_final=0.07,
                 soft_gate_T_qk=None,
                 soft_gate_T_v=None,
                 soft_gate_T_rst=None,
                 soft_gate_boundary_power=2.0,
                 soft_gate_boundary_power_final=4.0,
                 rpe_effective_weight=0.0,
                 execution_prune_eps=0.0):
        """Run the shared-pool SRW Transformer forward pass.

        analysis=False is the train/eval path and returns only regular
        training metrics.  analysis=True enables extra observational stats
        such as distribution shape, selection diagnostics, entropy, tau stats,
        raw norms, and debug norms.
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
        focus_probe_enabled = bool(spike_probe and spike_focus_bpos is not None)
        focus_k = int(spike_probe_topk)
        if focus_probe_enabled:
            _focus_b = jnp.clip(
                spike_focus_bpos[:, 0].astype(jnp.int32), 0, B - 1)
            _focus_pos = jnp.clip(
                spike_focus_bpos[:, 1].astype(jnp.int32), 0, S - 1)
            _focus_rank = jnp.arange(focus_k, dtype=jnp.float32)
            _focus_input_tok = spike_focus_input_ids.astype(jnp.int32)
            _focus_target_tok = spike_focus_target_ids.astype(jnp.int32)
            _focus_pred_tok = spike_focus_pred_ids.astype(jnp.int32)
            _focus_vocab_emb = self.token_emb.embedding.astype(jnp.float32)
            _focus_target_emb = _focus_vocab_emb[_focus_target_tok]
            _focus_pred_emb = _focus_vocab_emb[_focus_pred_tok]
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
            k_emb_n_all = _z
            k_read_n_all = _z
            k_write_n_all = _z
            rst_out_norm_all = _z
            attn_out_norm_all = _z
            attn_tau_mean_all = _z
            rst_tau_mean_all = _z
            attn_tau_abs_all = _z
            rst_tau_abs_all = _z
            attn_qk_emb_n_mean_all = _z
            attn_v_emb_n_mean_all = _z
            rst_emb_n_std_all = _z
            attn_qk_emb_n_min_all = _z
            attn_qk_emb_n_std_all = _z
            attn_v_emb_n_min_all = _z
            attn_v_emb_n_std_all = _z
            rst_emb_n_min_all = _z
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
            _ = self.neuron_pool.attn_qk_emb  # triggers NeuronPool.setup
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
            pool_params = all_params['neuron_pool']
            router_params = all_params['router']

            _sharded = sharded_fns

            block_params_list = [all_params[f'block_{i}']
                                 for i in range(self.n_layers)]
            stacked = jax.tree.map(
                lambda *arrays: jnp.stack(arrays), *block_params_list)

            base_rng = self.make_rng('dropout')
            layer_rngs = jax.random.split(base_rng, self.n_layers)

            def _focus_alignment_rows(layer_idx, stage_id, state, prev_state):
                """Rows [K, SPIKE_FOCUS_PATH_FIELD_COUNT] for focused CE tokens.

                stage_id convention:
                  0 pre_attn_resid, 1 norm1, 2 attn_out, 3 post_attn_resid,
                  4 norm2, 5 rst_out, 6 post_rst_resid, 7 final_norm.
                """
                hidden = state[_focus_b, _focus_pos].astype(jnp.float32)
                prev = prev_state[_focus_b, _focus_pos].astype(jnp.float32)
                delta = hidden - prev
                emb = self.token_emb.embedding.astype(jnp.float32)
                in_emb = emb[_focus_input_tok]
                tgt_emb = emb[_focus_target_tok]
                pred_emb = emb[_focus_pred_tok]
                h_norm = jnp.linalg.norm(hidden, axis=-1) + 1e-8
                d_norm = jnp.linalg.norm(delta, axis=-1)
                in_norm = jnp.linalg.norm(in_emb, axis=-1) + 1e-8
                tgt_norm = jnp.linalg.norm(tgt_emb, axis=-1) + 1e-8
                pred_norm = jnp.linalg.norm(pred_emb, axis=-1) + 1e-8
                in_dot = jnp.sum(hidden * in_emb, axis=-1)
                tgt_dot = jnp.sum(hidden * tgt_emb, axis=-1)
                pred_dot = jnp.sum(hidden * pred_emb, axis=-1)
                in_cos = in_dot / (h_norm * in_norm)
                tgt_cos = tgt_dot / (h_norm * tgt_norm)
                pred_cos = pred_dot / (h_norm * pred_norm)
                delta_norm_safe = d_norm + 1e-8
                delta_in_dot = jnp.sum(delta * in_emb, axis=-1)
                delta_tgt_dot = jnp.sum(delta * tgt_emb, axis=-1)
                delta_pred_dot = jnp.sum(delta * pred_emb, axis=-1)
                delta_in_cos = delta_in_dot / (delta_norm_safe * in_norm)
                delta_tgt_cos = delta_tgt_dot / (delta_norm_safe * tgt_norm)
                delta_pred_cos = delta_pred_dot / (delta_norm_safe * pred_norm)
                return jnp.stack([
                    _focus_rank,
                    _focus_b.astype(jnp.float32),
                    _focus_pos.astype(jnp.float32),
                    jnp.full((focus_k,), layer_idx.astype(jnp.float32)),
                    jnp.full((focus_k,), jnp.float32(stage_id)),
                    _focus_input_tok.astype(jnp.float32),
                    _focus_target_tok.astype(jnp.float32),
                    _focus_pred_tok.astype(jnp.float32),
                    h_norm,
                    d_norm,
                    in_cos,
                    tgt_cos,
                    pred_cos,
                    in_dot,
                    tgt_dot,
                    pred_dot,
                    pred_dot - tgt_dot,
                    pred_cos - tgt_cos,
                    tgt_dot - in_dot,
                    delta_tgt_dot,
                    delta_pred_dot,
                    delta_pred_dot - delta_tgt_dot,
                    delta_tgt_cos,
                    delta_pred_cos,
                    delta_pred_cos - delta_tgt_cos,
                    (_focus_pred_tok == _focus_input_tok).astype(jnp.float32),
                    (_focus_target_tok == _focus_input_tok).astype(jnp.float32),
                    (_focus_pred_tok == _focus_target_tok).astype(jnp.float32),
                ], axis=-1).astype(jnp.float32)

            def _focus_route_rows(layer_idx, attn_out, rst_out,
                                  pre_attn_x, post_attn_x, post_rst_x,
                                  a_qk_active, a_v_active, k_active,
                                  a_tau_direct, k_tau_direct,
                                  a_no_active_direct, k_no_active_direct,
                                  a_qk_positive_margin_active,
                                  a_v_positive_margin_active,
                                  k_positive_margin_active):
                attn_delta = attn_out[_focus_b, _focus_pos].astype(jnp.float32)
                rst_delta = rst_out[_focus_b, _focus_pos].astype(jnp.float32)
                pre = pre_attn_x[_focus_b, _focus_pos].astype(jnp.float32)
                post_a = post_attn_x[_focus_b, _focus_pos].astype(jnp.float32)
                post_r = post_rst_x[_focus_b, _focus_pos].astype(jnp.float32)
                q_tau = a_tau_direct[_focus_b, _focus_pos, 0].astype(jnp.float32)
                k_tau = a_tau_direct[_focus_b, _focus_pos, 1].astype(jnp.float32)
                v_tau = a_tau_direct[_focus_b, _focus_pos, 2].astype(jnp.float32)
                rst_tau = k_tau_direct[_focus_b, _focus_pos, 0].astype(jnp.float32)
                rst_active_focus = k_active[_focus_b, _focus_pos, 0].astype(jnp.float32)
                q_no = a_no_active_direct[_focus_b, _focus_pos, 0].astype(jnp.float32)
                k_no = a_no_active_direct[_focus_b, _focus_pos, 1].astype(jnp.float32)
                v_no = a_no_active_direct[_focus_b, _focus_pos, 2].astype(jnp.float32)
                rst_no = k_no_active_direct[_focus_b, _focus_pos, 0].astype(jnp.float32)
                return jnp.stack([
                    _focus_rank,
                    _focus_b.astype(jnp.float32),
                    _focus_pos.astype(jnp.float32),
                    jnp.full((focus_k,), layer_idx.astype(jnp.float32)),
                    jnp.linalg.norm(attn_delta, axis=-1),
                    jnp.linalg.norm(rst_delta, axis=-1),
                    jnp.linalg.norm(pre, axis=-1),
                    jnp.linalg.norm(post_a, axis=-1),
                    jnp.linalg.norm(post_r, axis=-1),
                    jnp.full((focus_k,), a_qk_active.astype(jnp.float32)),
                    jnp.full((focus_k,), a_v_active.astype(jnp.float32)),
                    rst_active_focus,
                    q_tau, k_tau, v_tau, rst_tau,
                    q_no, k_no, v_no, rst_no,
                    jnp.full((focus_k,), a_qk_positive_margin_active.astype(jnp.float32)),
                    jnp.full((focus_k,), a_v_positive_margin_active.astype(jnp.float32)),
                    jnp.full((focus_k,), k_positive_margin_active.astype(jnp.float32)),
                    jnp.linalg.norm(post_r - pre, axis=-1),
                ], axis=-1).astype(jnp.float32)

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
                    local_diagnostics=local_diagnostics,
                    spike_probe=spike_probe,
                    spike_probe_topk=spike_probe_topk,
                    focus_probe_enabled=focus_probe_enabled,
                    focus_b=_focus_b if focus_probe_enabled else None,
                    focus_pos=_focus_pos if focus_probe_enabled else None,
                    focus_rank=_focus_rank if focus_probe_enabled else None,
                    focus_target_emb=_focus_target_emb if focus_probe_enabled else None,
                    focus_pred_emb=_focus_pred_emb if focus_probe_enabled else None,
                    soft_gate_temperature=soft_gate_temperature,
                    soft_gate_t_final=soft_gate_t_final,
                    soft_gate_T_qk=soft_gate_T_qk,
                    soft_gate_T_v=soft_gate_T_v,
                    soft_gate_boundary_power=soft_gate_boundary_power,
                    soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                    execution_prune_eps=execution_prune_eps)
                (attn_out, attn_aux, a_qk_active, a_v_active, a_raw_gmax,
                 a_sstd, a_gsum, a_active_n_mean,
                 a_out_norm, a_tau_mean, a_strong,
                 a_qk_strong, a_v_strong,
                 a_qk_positive_margin_active, a_v_positive_margin_active,
                 a_tau_abs,
                 a_qk_emb_n_mean, a_v_emb_n_mean,
                 a_qk_emb_n_min, a_qk_emb_n_std,
                 a_v_emb_n_min, a_v_emb_n_std,
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
                 a_q_sparsity_diag, a_k_sparsity_diag) = attn_ret[58:67]
                if analysis:
                    (a_qk_raw_norm, a_v_raw_norm,
                     a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                     a_qk_margin_band, a_v_margin_band,
                     a_tau_std, a_tau_kernel_norm,
                     a_margin_band_wide, a_margin_band_mid,
                     a_skew, a_apt_std, a_entropy,
                     a_den_cost, a_selection_cost, a_current_cost,
                     a_qk_emb_n_max, a_v_emb_n_max,
                     a_rho_kurt, a_int_cap_frac,
                     a_q_norm_std, a_q_norm_max,
                     a_k_norm_std, a_k_norm_max,
                     a_logit_mean, a_logit_std,
                     a_softmax_top1_mean, a_softmax_top1_max,
                     a_logit_gap_mean, a_logit_gap_max,
                     a_softmax_entropy_mean, a_softmax_entropy_min,
                     a_o_input_norm_max, a_o_out_norm_max) = attn_ret[67:104]
                if local_diagnostics:
                    if spike_probe and focus_probe_enabled:
                        _a_local_tail = attn_ret[-10:-5]
                    elif spike_probe:
                        _a_local_tail = attn_ret[-8:-3]
                    else:
                        _a_local_tail = attn_ret[-5:]
                    (a_attn_local_layer_values,
                     a_attn_local_values, a_attn_local_locs,
                     a_attn_top1_values, a_attn_top1_locs) = _a_local_tail
                if spike_probe:
                    if focus_probe_enabled:
                        (a_spike_srw_rows,
                         a_spike_attn_rows,
                         a_focus_srw_rows,
                         a_focus_attn_rows,
                         a_out_norm_max) = attn_ret[-5:]
                        a_focus_srw_rows = a_focus_srw_rows.at[:, :, 3].set(
                            layer_idx.astype(jnp.float32))
                        a_focus_attn_rows = a_focus_attn_rows.at[:, :, 3].set(
                            layer_idx.astype(jnp.float32))
                    else:
                        (a_spike_srw_rows,
                         a_spike_attn_rows,
                         a_out_norm_max) = attn_ret[-3:]
                x = x + attn_out
                x_post_attn = x
                if spike_probe:
                    a_resid_norm_max = jnp.linalg.norm(
                        jax.lax.stop_gradient(x), axis=-1).max()

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                rst_ret = _rst_forward(
                    normed, pool_params, router_params, rng_rst,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis,
                    local_diagnostics=local_diagnostics,
                    spike_probe=spike_probe,
                    spike_probe_topk=spike_probe_topk,
                    d_model=self.d_model, n_layers=self.n_layers,
                    focus_probe_enabled=focus_probe_enabled,
                    focus_b=_focus_b if focus_probe_enabled else None,
                    focus_pos=_focus_pos if focus_probe_enabled else None,
                    focus_rank=_focus_rank if focus_probe_enabled else None,
                    focus_target_emb=_focus_target_emb if focus_probe_enabled else None,
                    focus_pred_emb=_focus_pred_emb if focus_probe_enabled else None,
                    soft_gate_temperature=soft_gate_temperature,
                    soft_gate_t_final=soft_gate_t_final,
                    soft_gate_T_rst=soft_gate_T_rst,
                    soft_gate_boundary_power=soft_gate_boundary_power,
                    soft_gate_boundary_power_final=soft_gate_boundary_power_final,
                    execution_prune_eps=execution_prune_eps)
                (rst_out, rst_aux, k_active, k_raw_gmax, k_sstd, k_gsum,
                 k_active_n_mean, k_emb_n, k_read_n, k_write_n, k_out_norm,
                 k_tau_mean, k_strong, k_positive_margin_active, k_tau_abs,
                 k_emb_n_min, k_emb_n_std,
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
                 k_sparsity_diag) = rst_ret[:52]
                if analysis:
                    (k_raw_out_norm,
                     k_tau_std, k_tau_kernel_norm,
                     k_margin_band_wide, k_margin_band_mid,
                     k_skew, k_apt_std, k_entropy,
                     k_den_cost, k_selection_cost, k_current_cost,
                     k_emb_n_max, k_rho_kurt, k_margin_band,
                     k_int_cap_frac) = rst_ret[52:67]
                if local_diagnostics:
                    if spike_probe and focus_probe_enabled:
                        _k_local_tail = rst_ret[-7:-3]
                    elif spike_probe:
                        _k_local_tail = rst_ret[-6:-2]
                    else:
                        _k_local_tail = rst_ret[-4:]
                    (k_local_values, k_local_locs,
                     k_top1_values, k_top1_locs) = _k_local_tail
                if spike_probe:
                    if focus_probe_enabled:
                        (k_spike_srw_rows,
                         k_focus_srw_rows,
                         k_out_norm_max) = rst_ret[-3:]
                        k_focus_srw_rows = k_focus_srw_rows.at[:, :, 3].set(
                            layer_idx.astype(jnp.float32))
                    else:
                        (k_spike_srw_rows,
                         k_out_norm_max) = rst_ret[-2:]
                x = x + rst_out
                x_post_rst = x
                if spike_probe:
                    k_resid_norm_max = jnp.linalg.norm(
                        jax.lax.stop_gradient(x), axis=-1).max()
                if focus_probe_enabled:
                    _zero_ref = jnp.zeros_like(x_pre_attn)
                    focus_path_rows = jnp.stack([
                        _focus_alignment_rows(layer_idx, 0, x_pre_attn, x_pre_attn),
                        _focus_alignment_rows(layer_idx, 1, normed, x_pre_attn),
                        _focus_alignment_rows(layer_idx, 2, attn_out, _zero_ref),
                        _focus_alignment_rows(layer_idx, 3, x_post_attn, x_pre_attn),
                        _focus_alignment_rows(layer_idx, 4, normed, x_post_attn),
                        _focus_alignment_rows(layer_idx, 5, rst_out, _zero_ref),
                        _focus_alignment_rows(layer_idx, 6, x_post_rst, x_post_attn),
                    ], axis=1)
                    focus_route_rows = _focus_route_rows(
                        layer_idx, attn_out, rst_out,
                        x_pre_attn, x_post_attn, x_post_rst,
                        a_qk_active, a_v_active, k_active,
                        a_tau_direct, k_tau_direct,
                        a_no_active_direct, k_no_active_direct,
                        a_qk_positive_margin_active,
                        a_v_positive_margin_active,
                        k_positive_margin_active)

                slim_ys = (attn_aux, rst_aux,
                           k_active, k_raw_gmax, k_sstd, k_gsum, k_active_n_mean,
                           a_qk_active, a_v_active, a_raw_gmax, a_sstd, a_gsum, a_active_n_mean,
                           k_emb_n, k_read_n, k_write_n,
                           k_out_norm,
                           a_out_norm, a_tau_mean, k_tau_mean,
                           k_strong, a_strong,
                           a_qk_strong, a_v_strong,
                           k_positive_margin_active,
                           a_qk_positive_margin_active,
                           a_v_positive_margin_active,
                           a_tau_abs, k_tau_abs,
                           a_qk_emb_n_mean, a_v_emb_n_mean,
                           k_emb_n_std,
                           a_qk_emb_n_min, a_qk_emb_n_std,
                           a_v_emb_n_min, a_v_emb_n_std,
                           k_emb_n_min,
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
                           )
                if not analysis:
                    if local_diagnostics:
                        slim_ys = slim_ys + (
                            a_attn_local_layer_values,
                            a_attn_local_values, a_attn_local_locs,
                            a_attn_top1_values, a_attn_top1_locs,
                            k_local_values, k_local_locs,
                            k_top1_values, k_top1_locs,
                        )
                    if spike_probe:
                        slim_ys = slim_ys + (
                            a_spike_srw_rows,
                            a_spike_attn_rows,
                            k_spike_srw_rows,
                            a_out_norm_max,
                            k_out_norm_max,
                            a_resid_norm_max,
                            k_resid_norm_max,
                        )
                    if focus_probe_enabled:
                        focus_srw_candidates = jnp.concatenate(
                            [a_focus_srw_rows, k_focus_srw_rows], axis=1)
                        _fs_scores = focus_srw_candidates[:, :, 6]
                        _fs_vals, _fs_idx = jax.lax.top_k(
                            _fs_scores, min(focus_k, int(_fs_scores.shape[1])))
                        focus_srw_rows = jnp.take_along_axis(
                            focus_srw_candidates, _fs_idx[:, :, None], axis=1)
                        focus_srw_rows = focus_srw_rows.at[:, :, 6].set(_fs_vals)
                        slim_ys = slim_ys + (
                            focus_path_rows,
                            focus_route_rows,
                            focus_srw_rows,
                            a_focus_attn_rows,
                        )
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
                    a_qk_emb_n_max, a_v_emb_n_max,
                    k_emb_n_max,
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
                if local_diagnostics:
                    analysis_ys = analysis_ys + (
                        a_attn_local_layer_values,
                        a_attn_local_values, a_attn_local_locs,
                        a_attn_top1_values, a_attn_top1_locs,
                        k_local_values, k_local_locs,
                        k_top1_values, k_top1_locs,
                    )
                if spike_probe:
                    analysis_ys = analysis_ys + (
                        a_spike_srw_rows,
                        a_spike_attn_rows,
                        k_spike_srw_rows,
                        a_out_norm_max,
                        k_out_norm_max,
                        a_resid_norm_max,
                        k_resid_norm_max,
                    )
                if focus_probe_enabled:
                    focus_srw_candidates = jnp.concatenate(
                        [a_focus_srw_rows, k_focus_srw_rows], axis=1)
                    _fs_scores = focus_srw_candidates[:, :, 6]
                    _fs_vals, _fs_idx = jax.lax.top_k(
                        _fs_scores, min(focus_k, int(_fs_scores.shape[1])))
                    focus_srw_rows = jnp.take_along_axis(
                        focus_srw_candidates, _fs_idx[:, :, None], axis=1)
                    focus_srw_rows = focus_srw_rows.at[:, :, 6].set(_fs_vals)
                    analysis_ys = analysis_ys + (
                        focus_path_rows,
                        focus_route_rows,
                        focus_srw_rows,
                        a_focus_attn_rows,
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
             k_emb_n_all, k_read_n_all, k_write_n_all,
             rst_out_norm_all,
             attn_out_norm_all, attn_tau_mean_all, rst_tau_mean_all,
             rst_strong_all, attn_strong_all,
             attn_qk_strong_all, attn_v_strong_all,
             rst_positive_margin_active_all,
             attn_qk_positive_margin_active_all,
             attn_v_positive_margin_active_all,
             attn_tau_abs_all, rst_tau_abs_all,
             attn_qk_emb_n_mean_all, attn_v_emb_n_mean_all,
             rst_emb_n_std_all,
             attn_qk_emb_n_min_all, attn_qk_emb_n_std_all,
             attn_v_emb_n_min_all, attn_v_emb_n_std_all,
             rst_emb_n_min_all,
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
             rst_sparsity_diag_all) = scan_ys[
                _scan_offset:_scan_offset + 10]
            _scan_offset += 10
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
                 attn_qk_emb_n_max_all, attn_v_emb_n_max_all,
                 rst_emb_n_max_all,
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
            if local_diagnostics:
                (attn_local_layer_values_all,
                 attn_local_values_all, attn_local_locs_all,
                 attn_top1_values_all, attn_top1_locs_all,
                 rst_local_values_all, rst_local_locs_all,
                 rst_top1_values_all, rst_top1_locs_all) = scan_ys[
                    _scan_offset:_scan_offset + 9]
                _scan_offset += 9
            if spike_probe:
                (spike_attn_srw_rows_all,
                 spike_attn_rows_all,
                 spike_rst_srw_rows_all,
                 spike_layer_attn_out_norm_max_all,
                 spike_layer_rst_out_norm_max_all,
                 spike_layer_resid_norm_max_after_attn_all,
                 spike_layer_resid_norm_max_after_rst_all) = scan_ys[
                    _scan_offset:_scan_offset + 7]
                _scan_offset += 7
            if focus_probe_enabled:
                (spike_focus_path_trace_all,
                 spike_focus_route_trace_all,
                 spike_focus_srw_top_all,
                 spike_focus_attention_top_all) = scan_ys[
                    _scan_offset:_scan_offset + 4]
                _scan_offset += 4
            # Aux is averaged over layers after attention and RST terms are
            # collected.  Attention keeps historical Q/K/V scaling upstream.
            total_aux = (attn_auxes + rst_auxes).mean()

        x_pre_final_norm = x
        x = self.norm(x)
        if focus_probe_enabled and not self.is_initializing():
            spike_focus_final_trace = _focus_alignment_rows(
                jnp.asarray(self.n_layers, dtype=jnp.int32),
                7, x, x_pre_final_norm)[:, None, :]

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
            'rpe_effective_weight': jnp.asarray(rpe_effective_weight, dtype=jnp.float32),
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

            'rst_emb_norm': k_emb_n_all.mean(),
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
            'attn_qk_emb_norm_mean': attn_qk_emb_n_mean_all.mean(),
            'attn_qk_emb_norm_min': attn_qk_emb_n_min_all.min(),
            'attn_qk_emb_norm_std': attn_qk_emb_n_std_all.mean(),
            'attn_v_emb_norm_mean': attn_v_emb_n_mean_all.mean(),
            'attn_v_emb_norm_min': attn_v_emb_n_min_all.min(),
            'attn_v_emb_norm_std': attn_v_emb_n_std_all.mean(),
            'rst_emb_norm_min': rst_emb_n_min_all.min(),
            'rst_emb_norm_std': rst_emb_n_std_all.mean(),

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
            # Per-layer direct tau stacks. RPE/exploration is disabled in
            # v4.1.6.1, but these remain diagnostics and keep 4160 parity.
            # Shapes: attn [L, B, S, 3], RST [L, B, S, 1].
            'attn_tau_direct': attn_tau_direct_all,
            'rst_tau_direct': rst_tau_direct_all,
            'attn_no_active_direct': jax.lax.stop_gradient(
                attn_no_active_direct_all),
            'rst_no_active_direct': jax.lax.stop_gradient(
                rst_no_active_direct_all),
            # v4164 aliases: old gate slots carry compose-weight statistics;
            # denominator slots carry admission-only sums.
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
            'compose_mass_sum': attn_gsum_all.mean() + rst_gsum_all.mean(),
            'attn_compose_mass_sum': attn_gsum_all.mean(),
            'attn_qk_compose_mass_sum': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_SUM),
            'attn_v_compose_mass_sum': _attn_core_mean(
                ATTN_SPLIT_V_GATE_SUM),
            'rst_compose_mass_sum': rst_gsum_all.mean(),
            'angular_depth_mean': (
                attn_current_cost_mean_all.mean()
                + rst_current_cost_mean_all.mean()) / jnp.float32(2.0),
            'attn_angular_depth_mean': attn_current_cost_mean_all.mean(),
            'attn_qk_angular_depth_mean': _attn_core_mean(
                ATTN_SPLIT_QK_ANGULAR_DEPTH_MEAN),
            'attn_v_angular_depth_mean': _attn_core_mean(
                ATTN_SPLIT_V_ANGULAR_DEPTH_MEAN),
            'rst_angular_depth_mean': rst_current_cost_mean_all.mean(),
            'angular_depth_max': jnp.maximum(
                attn_int_max_all.max(), rst_int_max_all.max()),
            'attn_angular_depth_max': attn_int_max_all.max(),
            'attn_qk_angular_depth_max': _attn_core_max(
                ATTN_SPLIT_QK_INT_MAX),
            'attn_v_angular_depth_max': _attn_core_max(
                ATTN_SPLIT_V_INT_MAX),
            'rst_angular_depth_max': rst_int_max_all.max(),
            'compose_eff_n': (
                attn_gate_eff_n_all.mean() + rst_gate_eff_n_all.mean()
            ) / jnp.float32(2.0),
            'attn_compose_eff_n': attn_gate_eff_n_all.mean(),
            'attn_qk_compose_eff_n': _attn_core_mean(
                ATTN_SPLIT_QK_GATE_EFF_N),
            'attn_v_compose_eff_n': _attn_core_mean(
                ATTN_SPLIT_V_GATE_EFF_N),
            'rst_compose_eff_n': rst_gate_eff_n_all.mean(),
            'compose_top1_frac': (
                attn_top1_gate_frac_all.mean()
                + rst_top1_gate_frac_all.mean()) / jnp.float32(2.0),
            'compose_top1_frac_max': jnp.maximum(
                attn_top1_gate_frac_all.max(),
                rst_top1_gate_frac_all.max()),
            'attn_compose_top1_frac': attn_top1_gate_frac_all.mean(),
            'attn_compose_top1_frac_max': attn_top1_gate_frac_all.max(),
            'attn_qk_compose_top1_frac': _attn_core_mean(
                ATTN_SPLIT_QK_TOP1_GATE_FRAC),
            'attn_qk_compose_top1_frac_max': _attn_core_max(
                ATTN_SPLIT_QK_TOP1_GATE_FRAC_MAX),
            'attn_v_compose_top1_frac': _attn_core_mean(
                ATTN_SPLIT_V_TOP1_GATE_FRAC),
            'attn_v_compose_top1_frac_max': _attn_core_max(
                ATTN_SPLIT_V_TOP1_GATE_FRAC_MAX),
            'rst_compose_top1_frac': rst_top1_gate_frac_all.mean(),
            'rst_compose_top1_frac_max': rst_top1_gate_frac_all.max(),
            # Always-on output diagnostics: cheap scalar reductions used by train logs.
            # These are kept outside the analysis-only block so out_diag never falls
            # back to misleading zeros during normal training.
            'debug_residual_norm': jnp.linalg.norm(x, axis=-1).mean(),
            'debug_residual_norm_max': jnp.linalg.norm(x, axis=-1).max(),
            'debug_token_emb_norm': jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean(),
            'debug_token_emb_norm_max': jnp.linalg.norm(self.token_emb.embedding, axis=-1).max(),
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
                'attn_qk_emb_norm_max': attn_qk_emb_n_max_all.max(),
                'attn_v_emb_norm_max': attn_v_emb_n_max_all.max(),
                'rst_emb_norm_max': rst_emb_n_max_all.max(),
                'attn_rho_kurt': attn_rho_kurt_all.mean(),
                'rst_rho_kurt': rst_rho_kurt_all.mean(),
                'attn_qk_raw_norm': attn_qk_raw_norm_all.mean(),
                'attn_v_raw_norm': attn_v_raw_norm_all.mean(),
                'rst_raw_out_norm': rst_raw_out_norm_all.mean(),
                'debug_residual_norm': _residual_norm,
                'debug_token_emb_norm_analysis': _emb_norm,
                'debug_o_proj_norm': _o_proj_norm,
                'debug_q_norm': attn_q_norm_all.mean(),
                'debug_k_norm': attn_k_norm_all.mean(),
                'debug_v_norm': attn_v_norm_dbg_all.mean(),
                'debug_attn_logit_max_mean': attn_logit_max_all.mean(),
                'debug_o_input_norm': attn_o_input_norm_all.mean(),
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
        if local_diagnostics and not self.is_initializing():
            result.update({
                'attn_local_layer_values': attn_local_layer_values_all,
                'local_spike_values': jnp.concatenate([
                    attn_local_values_all,
                    rst_local_values_all,
                ], axis=1),
                'local_spike_locs': jnp.concatenate([
                    attn_local_locs_all,
                    rst_local_locs_all,
                ], axis=1),
                'local_top1_values': jnp.concatenate([
                    attn_top1_values_all,
                    rst_top1_values_all,
                ], axis=1),
                'local_top1_locs': jnp.concatenate([
                    attn_top1_locs_all,
                    rst_top1_locs_all,
                ], axis=1),
            })
        if spike_probe and not self.is_initializing():
            result.update({
                'spike_srw_top': jnp.concatenate([
                    spike_attn_srw_rows_all,
                    spike_rst_srw_rows_all,
                ], axis=1),
                'spike_attention_top': spike_attn_rows_all,
                'spike_layer_attn_out_norm_max': (
                    spike_layer_attn_out_norm_max_all),
                'spike_layer_rst_out_norm_max': (
                    spike_layer_rst_out_norm_max_all),
                'spike_layer_resid_norm_max_after_attn': (
                    spike_layer_resid_norm_max_after_attn_all),
                'spike_layer_resid_norm_max_after_rst': (
                    spike_layer_resid_norm_max_after_rst_all),
                'spike_layer_lm_logit_abs_proxy_if_available': jnp.zeros(
                    (self.n_layers,), dtype=jnp.float32),
            })
        if focus_probe_enabled and not self.is_initializing():
            _focus_final_block = jnp.full(
                (focus_k, 7, SPIKE_FOCUS_PATH_FIELD_COUNT),
                -jnp.inf, dtype=jnp.float32)
            _focus_final_block = _focus_final_block.at[:, 0, :].set(
                spike_focus_final_trace[:, 0, :])
            result.update({
                'spike_focus_path_trace': jnp.concatenate([
                    spike_focus_path_trace_all,
                    _focus_final_block[None, :, :, :],
                ], axis=0),
                'spike_focus_route_trace': spike_focus_route_trace_all,
                'spike_focus_srw_top': spike_focus_srw_top_all,
                'spike_focus_attention_top': spike_focus_attention_top_all,
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
            result['debug_logit_max'] = logit_abs_max
            result['debug_logit_norm_mean'] = logit_norm_mean
            result['debug_logit_mean'] = logit_mean
            result['debug_logit_std'] = logit_std
            result['per_token_ce'] = per_token_ce
            result['valid_mask'] = valid_mask
            if spike_probe:
                logits_f = (shift_x @ embedding_matrix.T).astype(jnp.float32)
                pred_ids = jnp.argmax(logits_f, axis=-1).astype(jnp.int32)
                max_logits = jnp.max(logits_f, axis=-1)
                min_logits = jnp.min(logits_f, axis=-1)
                pred_logits = jnp.take_along_axis(
                    logits_f, pred_ids[..., None], axis=-1).squeeze(-1)
                safe_targets = jnp.where(valid_mask, shift_labels, 0)
                target_logits = jnp.take_along_axis(
                    logits_f, safe_targets[..., None], axis=-1).squeeze(-1)
                score = jnp.where(
                    valid_mask, per_token_ce, jnp.float32(-jnp.inf))
                flat_score = score.reshape((-1,))
                _tok_k = min(max(1, int(spike_probe_topk)),
                             int(flat_score.shape[0]))
                top_scores, flat_idx = jax.lax.top_k(flat_score, _tok_k)
                tok_b = flat_idx // (S - 1)
                tok_pos = flat_idx - tok_b * (S - 1)
                rows = jnp.stack([
                    jnp.arange(_tok_k, dtype=jnp.float32),
                    tok_b.astype(jnp.float32),
                    tok_pos.astype(jnp.float32),
                    input_ids[tok_b, tok_pos].astype(jnp.float32),
                    safe_targets[tok_b, tok_pos].astype(jnp.float32),
                    pred_ids[tok_b, tok_pos].astype(jnp.float32),
                    top_scores,
                    target_logits[tok_b, tok_pos],
                    pred_logits[tok_b, tok_pos],
                    max_logits[tok_b, tok_pos],
                    min_logits[tok_b, tok_pos],
                    jnp.maximum(
                        jnp.abs(max_logits[tok_b, tok_pos]),
                        jnp.abs(min_logits[tok_b, tok_pos])),
                    pred_logits[tok_b, tok_pos]
                    - target_logits[tok_b, tok_pos],
                ], axis=-1).astype(jnp.float32)
                if _tok_k < int(spike_probe_topk):
                    pad = jnp.full(
                        (int(spike_probe_topk) - _tok_k,
                         SPIKE_TOKEN_FIELD_COUNT),
                        -jnp.inf, dtype=jnp.float32)
                    pad = pad.at[:, 1:6].set(0.0)
                    rows = jnp.concatenate([rows, pad], axis=0)
                result['spike_top_token_ce'] = rows
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
            f"  Route: unified angular d_route={self.d_route}",
            "  Pool scales: fixed depth-scaled "
            f"(qk={float(qk_scale):.6g}, v={float(v_scale):.6g}, "
            f"rst={float(rst_scale):.6g})",
        ]


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


def _angular_gate_kwargs_from_model_cfg(model_cfg):
    """Extract v4164 angular-depth boundary settings for inference."""
    # Pure generation/vectorized helpers do not receive the training schedule,
    # so they use explicit boundary defaults unless the caller supplies current
    # values through model_cfg.
    return {
        'soft_gate_temperature': float(
            model_cfg.get('soft_gate_temperature', 0.07)),
        'soft_gate_boundary_power': float(
            model_cfg.get('soft_gate_boundary_power', 4.0)),
        'execution_prune_eps': float(model_cfg.get('execution_prune_eps', 0.0)),
        'soft_gate_effective_active_eps': float(
            model_cfg.get('soft_gate_effective_active_eps', 1.0e-6)),
    }


def _tau_init_calibration_scores(params, input_ids, max_tokens=128):
    """Sample fresh-init cosine scores without changing forward semantics.

    The sample uses the first block's freshly initialized normalized route
    states and the shared v4164 router/pool parameters. Rho follows the
    sharded train path exactly: normalize the full d_route vector, cast
    directions to bf16, then compute cosine dot products.
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
    h_all = (
        attn_x @ router['proj_attn']['kernel']
        + router['proj_attn']['bias'])
    h_q, h_k, h_v = jnp.split(h_all, 3, axis=-1)
    h_rst = (
        rst_x @ router['proj_rst']['kernel']
        + router['proj_rst']['bias'])

    def _selection_rho(h, emb):
        q_unit = _forward_unit_direction(
            h.astype(jnp.float32)).astype(jnp.bfloat16)
        route_unit = _forward_unit_direction(
            emb.astype(jnp.float32)).astype(jnp.bfloat16)
        return (q_unit @ route_unit.T).astype(jnp.float32)

    pool = params['neuron_pool']
    qk_emb = pool['attn_qk_emb']
    v_emb = pool['attn_v_emb']
    rst_emb = pool['rst_emb']
    return {
        'q': _selection_rho(h_q, qk_emb),
        'k': _selection_rho(h_k, qk_emb),
        'v': _selection_rho(h_v, v_emb),
        'rst': _selection_rho(h_rst, rst_emb),
    }


def _angular_relation(h, emb):
    q = _forward_unit_direction(h.astype(jnp.float32))
    route = _forward_unit_direction(emb.astype(jnp.float32))
    return (q @ route.T).astype(jnp.float32)


def _angular_compose(h, emb, raw_tau, raw_scan_offset=None,
                     soft_gate_temperature=0.07,
                     soft_gate_boundary_power=4.0,
                     execution_prune_eps=0.0,
                     soft_gate_effective_active_eps=1.0e-6):
    rho = _angular_relation(h, emb)
    tau = _tau_from_param(raw_tau)
    return _compute_admission_depth(
        rho, tau, soft_gate_temperature,
        boundary_power=soft_gate_boundary_power,
        effective_active_eps=soft_gate_effective_active_eps,
        execution_prune_eps=execution_prune_eps)


def _angular_gate(h, emb, raw_tau, raw_scan_offset=None,
                  soft_gate_temperature=0.07,
                  soft_gate_boundary_power=4.0,
                  execution_prune_eps=0.0,
                  soft_gate_effective_active_eps=1.0e-6):
    """Canonical v4164 compose weight for non-sharded inference helpers."""
    _, _, _, compose_weight, _ = _angular_compose(
        h, emb, raw_tau, raw_scan_offset,
        soft_gate_temperature=soft_gate_temperature,
        soft_gate_boundary_power=soft_gate_boundary_power,
        execution_prune_eps=execution_prune_eps,
        soft_gate_effective_active_eps=soft_gate_effective_active_eps)
    return compose_weight.astype(jnp.float32)


def _srw_inference(x, h, emb, raw_tau, raw_scan_offset, w_read, w_write,
                   **angular_gate_kwargs):
    """Non-chunked SRW for inference."""
    # v4.1.6.4: inference uses read/write directions; params stay raw.
    r_n = _forward_unit_direction(w_read.astype(jnp.float32))
    w_n = _forward_unit_direction(w_write.astype(jnp.float32))
    _, admission, _, compose_weight, _ = _angular_compose(
        h, emb, raw_tau, raw_scan_offset, **angular_gate_kwargs)

    xr = x.astype(jnp.float32) @ r_n.T
    a = compose_weight * xr
    raw_out = a @ w_n
    den = jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32)


def _srw_inference_with_gates(x, h, emb, raw_tau, raw_scan_offset, w_read,
                              w_write, **angular_gate_kwargs):
    """Like _srw_inference but also returns gate and normalized gate."""
    # v4.1.6.4: analysis inference uses read/write directions; params stay raw.
    r_n = _forward_unit_direction(w_read.astype(jnp.float32))
    w_n = _forward_unit_direction(w_write.astype(jnp.float32))
    _, admission, _, compose_weight, _ = _angular_compose(
        h, emb, raw_tau, raw_scan_offset, **angular_gate_kwargs)
    gate_norm = compose_weight / jnp.maximum(
        admission.sum(axis=-1, keepdims=True), 1e-8)

    xr = x.astype(jnp.float32) @ r_n.T
    a = compose_weight * xr
    raw_out = a @ w_n
    den = jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32), compose_weight, gate_norm



def _attn_forward_cached(x, pool_params, router_params, expand_O_kernel,
                         n_heads, d_model, n_layers,
                         cache_K, cache_V, cache_len,
                         angular_gate_kwargs=None):
    """Cached attention decode step. x: [B, 1, D]."""
    B = x.shape[0]
    d_head = d_model // n_heads

    if angular_gate_kwargs is None:
        angular_gate_kwargs = {}
    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
    tau_all = x @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
    raw_scan_offset_all = jnp.zeros_like(tau_all)

    Q = _srw_inference(x, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                       pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                       **angular_gate_kwargs)
    K_new = _srw_inference(x, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                           **angular_gate_kwargs)
    V_new = _srw_inference(x, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                           pool_params['attn_v_read'], pool_params['attn_v_write'],
                           **angular_gate_kwargs)
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
                           angular_gate_kwargs=None):
    """Inference-only RST Layer forward. No chunking, no LB, no dropout."""
    if angular_gate_kwargs is None:
        angular_gate_kwargs = {}
    # emb used as-is (matches training path).
    rst_norm = pool_params['rst_emb']
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    tau = x @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
    raw_scan_offset = jnp.zeros_like(tau)
    out = _srw_inference(x, h, rst_norm, tau, raw_scan_offset,
                         pool_params['rst_read'], pool_params['rst_write'],
                         **angular_gate_kwargs)
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
    angular_gate_kwargs = _angular_gate_kwargs_from_model_cfg(model_cfg)
    max_seq = model_cfg['max_seq_len']
    d_head = d_model // n_heads

    pool_params = params['neuron_pool']
    router_params = params['router']
    qk_scale_eff, v_scale_eff, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']

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
        tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
        raw_scan_offset_all = jnp.zeros_like(tau_all)

        Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                           **angular_gate_kwargs)
        K_val = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               **angular_gate_kwargs)
        V_val = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'],
                               **angular_gate_kwargs)
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
            angular_gate_kwargs=angular_gate_kwargs)
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
    angular_gate_kwargs = _angular_gate_kwargs_from_model_cfg(model_cfg)

    pool_params = params['neuron_pool']
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
            angular_gate_kwargs=angular_gate_kwargs)
        cK = cK.at[layer_idx].set(new_cK)
        cV = cV.at[layer_idx].set(new_cV)
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        rst_out = _rst_forward_inference(
            normed, pool_params, router_params,
            d_model=d_model, n_layers=n_layers,
            angular_gate_kwargs=angular_gate_kwargs)
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
    angular_gate_kwargs = _angular_gate_kwargs_from_model_cfg(model_cfg)

    pool_params = params['neuron_pool']
    router_params = params['router']
    qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
        pool_params, d_model, n_layers)
    norm_params = params['norm']
    emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']
    rst_norm = pool_params['rst_emb']

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
            tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
            raw_scan_offset_all = jnp.zeros_like(tau_all)

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               **angular_gate_kwargs)
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               **angular_gate_kwargs)
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'],
                               **angular_gate_kwargs)
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
            tau_k = normed @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
            raw_scan_offset_k = jnp.zeros_like(tau_k)
            rst_out = _srw_inference(normed, h_k, rst_norm, tau_k, raw_scan_offset_k,
                                     pool_params['rst_read'], pool_params['rst_write'],
                                     **angular_gate_kwargs)
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
    pool = params['neuron_pool']
    results = {}
    for pool_name, emb_key in [
            ('Attention-QK', 'attn_qk_emb'),
            ('Attention-V', 'attn_v_emb'),
            ('RST', 'rst_emb')]:
        emb = pool[emb_key]
        read = pool[emb_key.replace('emb', 'read')]
        write = pool[emb_key.replace('emb', 'write')]
        emb_n = jnp.linalg.norm(emb, axis=-1)
        read_n = jnp.linalg.norm(read, axis=-1)
        write_n = jnp.linalg.norm(write, axis=-1)
        results[pool_name] = {
            'N': emb.shape[0],
            'emb_mean': emb_n.mean(), 'emb_std': emb_n.std(),
            'emb_dead': (emb_n < 1e-6).sum(),
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
    pool = params['neuron_pool']
    results = {}
    for pool_name, emb_key in [
            ('Attention-QK', 'attn_qk_emb'),
            ('Attention-V', 'attn_v_emb'),
            ('RST', 'rst_emb')]:
        emb = pool[emb_key]
        N, d = emb.shape
        if N > max_sample:
            idx = jnp.linspace(0, N - 1, max_sample, dtype=jnp.int32)
            emb_s = emb[idx]
        else:
            emb_s = emb
        norms = jnp.linalg.norm(emb_s, axis=-1, keepdims=True) + 1e-8
        emb_normed = emb_s / norms
        n_s = emb_normed.shape[0]

        gram = emb_normed @ emb_normed.T
        gram = gram - jnp.eye(n_s) * gram
        mean_sim = jnp.abs(gram).sum() / (n_s * (n_s - 1))
        max_sim = jnp.abs(gram).max()

        sv = jnp.linalg.svd(emb_s, compute_uv=False)
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

    mode='full': returns gate + gate_norm (R.1, P2, P3 etc.)
    mode='light': returns gate_norm only.

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
    angular_gate_kwargs = _angular_gate_kwargs_from_model_cfg(model_cfg)

    pool_params = params['neuron_pool']
    router_params = params['router']
    qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']
    rst_norm_w = pool_params['rst_emb']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    _return_raw = (mode == 'full')

    def analysis_layer(carry, xs):
        x = carry
        bp = xs['params']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
        raw_scan_offset_all = jnp.zeros_like(tau_all)

        Q, gate_Q_raw, gate_Q = _srw_inference_with_gates(
            normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'],
            **angular_gate_kwargs)
        K, gate_K_raw, gate_K = _srw_inference_with_gates(
            normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'],
            **angular_gate_kwargs)
        V, gate_V_raw, gate_V = _srw_inference_with_gates(
            normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
            pool_params['attn_v_read'], pool_params['attn_v_write'],
            **angular_gate_kwargs)
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
        tau_k = normed @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
        raw_scan_offset_k = jnp.zeros_like(tau_k)
        rst_out, gate_RST_raw, gate_RST = _srw_inference_with_gates(
            normed, h_k, rst_norm_w, tau_k, raw_scan_offset_k,
            pool_params['rst_read'], pool_params['rst_write'],
            **angular_gate_kwargs)
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
        # Legacy analysis aliases; remove after downstream analysis is migrated.
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
    Legacy key 'know' is still accepted.
    True = suppress.
    Returns: forward_fn(input_ids) -> logits [B, S, vocab]
    """
    params = _squeeze_params(params)
    params = jax.tree.map(jnp.asarray, params)
    angular_gate_kwargs = _angular_gate_kwargs_from_model_cfg(model_cfg)
    qk_mult = jnp.where(suppress_masks.get('qk', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'qk' in suppress_masks else None
    v_mult = jnp.where(suppress_masks.get('v', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'v' in suppress_masks else None
    rst_mask = suppress_masks.get('rst', suppress_masks.get('know', None))
    rst_mult = jnp.where(rst_mask, 0.0, 1.0) if rst_mask is not None else None

    def _srw_sup(x, h, emb, tau_off, raw_scan_offset, w_read, w_write, mult):
        """SRW with optional gate suppression."""
        # v4.1.6.4: suppressed forward uses read/write directions.
        r_n = _forward_unit_direction(w_read.astype(jnp.float32))
        w_n = _forward_unit_direction(w_write.astype(jnp.float32))
        _, admission, _, compose_weight, _ = _angular_compose(
            h, emb, tau_off, raw_scan_offset, **angular_gate_kwargs)
        if mult is not None:
            compose_weight = compose_weight * mult[None, None, :]
            admission = admission * mult[None, None, :]
        xr = x.astype(jnp.float32) @ r_n.T
        a = compose_weight * xr
        out = a @ w_n
        den = jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0)
        return (out.astype(jnp.float32) / den).astype(jnp.float32)

    def forward_fn(input_ids):
        B, S = input_ids.shape
        d_model = model_cfg['d_model']
        n_layers = model_cfg['n_layers']
        n_heads = model_cfg['n_heads']
        d_head = d_model // n_heads
        pp = params['neuron_pool']
        rp = params['router']
        qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
            pp, d_model, n_layers)

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]
        # Route embeddings are used as-is, matching the training path.
        qk_n = pp['attn_qk_emb']
        v_n = pp['attn_v_emb']
        kn_n = pp['rst_emb']

        for i in range(n_layers):
            bp = params[f'block_{i}']
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ rp['proj_attn']['kernel'] + rp['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
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


def migrate_legacy_v4155_params(params):
    """
    Convert legacy v4.1.5.5 parameter names to the new DAWN-SRW/RST names.

    Legacy:
        qk_* -> attn_qk_*
        v_* -> attn_v_*
        know_* -> rst_*
        proj_know -> proj_rst
        tau_know -> tau_rst
        scan_bias_attn -> raw_scan_offset_attn
        scan_bias_know -> raw_scan_offset_rst

    The function is safe to call on already-migrated params. It does not mutate
    the input in-place, preserves unknown keys, and handles plain dicts and
    Flax FrozenDict parameter trees. If old and new keys both exist, the new
    key is preferred.
    """
    was_frozen = isinstance(params, FrozenDict)
    tree = unfreeze(params) if was_frozen else jax.tree.map(
        lambda x: x, params, is_leaf=lambda x: not isinstance(x, dict))

    def _migrate_container(container):
        if not isinstance(container, dict):
            return
        if 'neuron_pool' in container and isinstance(container['neuron_pool'], dict):
            pool = container['neuron_pool']
            for old, new in (
                    ('qk_emb', 'attn_qk_emb'),
                    ('qk_read', 'attn_qk_read'),
                    ('qk_write', 'attn_qk_write'),
                    ('qk_scale', 'attn_qk_scale'),
                    ('v_emb', 'attn_v_emb'),
                    ('v_read', 'attn_v_read'),
                    ('v_write', 'attn_v_write'),
                    ('v_scale', 'attn_v_scale'),
                    ('know_emb', 'rst_emb'),
                    ('know_read', 'rst_read'),
                    ('know_write', 'rst_write'),
                    ('know_scale', 'rst_scale'),
            ):
                _rename_key_if_needed(pool, old, new)
        if 'router' in container and isinstance(container['router'], dict):
            router = container['router']
            for old, new in (
                    ('proj_know', 'proj_rst'),
                    ('tau_know', 'tau_rst'),
                    ('scan_bias_attn', 'raw_scan_offset_attn'),
                    ('scan_bias_know', 'raw_scan_offset_rst'),
            ):
                _rename_key_if_needed(router, old, new)

    def _walk(node):
        if isinstance(node, dict):
            _migrate_container(node)
            for value in node.values():
                _walk(value)

    _walk(tree)
    return freeze(tree) if was_frozen else tree
