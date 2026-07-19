"""
DAWN-SRW v4.1.7.x shared canonical SRW core

The v4171 learned-address model remains the default. The v4172 wrapper fixes
the same core to live generalized coordinate-wise bilinear RW addresses.

Implemented concepts:
- cosine-space tau reference with bounded sigmoid min/max mapping
- statically selected learned or generalized live-gradient operator addresses
- direct state-to-operation queries
- linear angular-depth gate after DirectTau
- selectable linear-angular, quadratic, or heat-energy composition
- tau movement controlled by optimizer-side tau_lr_mult
- train-time effective gate statistics
- validation-time execution pruning through execution_prune_eps
"""


import jax
import jax.numpy as jnp
import flax.linen as nn
import math
import numbers
import numpy as np
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

    v4171 has no learned pool scale parameters; scale is fixed by d_model/layers.
    """
    return _pool_output_scales(d_model, n_layers)


# ================================================================
# V4.1.7.1 canonical DirectTau selection and SRW composition.
#
#   rho              = cosine(operator_query, operator_keys)
#   raw_tau          = learned cosine-space reference
#   tau              = -1 + 2 * sigmoid(raw_tau)
#   margin           = rho - tau
#   angular_amplitude = clip(margin / max(1 - tau, 1e-4), 0, 1)
#   admission_weight  = mode-specific compact-cap composition weight
#   execution_weight  = admission_weight, optionally eval-pruned
#   den                = mode-specific function of unpruned admission mass
# ================================================================

DEFAULT_D_ROUTE = 64
RW_FORWARD_NORM_EPS = 1e-6     # forward-only read/write direction floor
MODEL_VERSION = "spatial-r1-v4.1.7.1"
ANALYSIS_INTERVENTION_NAME = "production_core_execution_suppression"
OPERATOR_KEY_MODE_LEARNED = "learned_operator_embedding"
OPERATOR_KEY_MODE_GENERALIZED_BILINEAR = "generalized_bilinear_rw"
OPERATOR_KEY_MODE = OPERATOR_KEY_MODE_LEARNED
OPERATOR_QUERY_MODE = "direct_state_projection"
DEFAULT_ADMISSION_DEN_POWER = 1.0
DEFAULT_SRW_COMPOSITION_MODE = "linear_angular"
DEFAULT_HEAT_KERNEL_BETA = 2.0
QUADRATIC_DEN_EPS = 1.0e-6
HEAT_ENERGY_DEN_EPS = 1.0e-6
_V4171_SRW_COMPOSITION_MODES = frozenset((
    DEFAULT_SRW_COMPOSITION_MODE,
    "quadratic",
    "heat_energy",
))
_V417X_OPERATOR_KEY_MODES = frozenset((
    OPERATOR_KEY_MODE_LEARNED,
    OPERATOR_KEY_MODE_GENERALIZED_BILINEAR,
))


def _validate_operator_key_mode(value, *, context="v417x"):
    """Validate the static Python-side operator-address selector."""
    if isinstance(value, jax.core.Tracer):
        raise ValueError(
            f"{context} operator_key_mode must be a static Python string")
    if not isinstance(value, str) or value not in _V417X_OPERATOR_KEY_MODES:
        raise ValueError(
            f"{context} unsupported operator_key_mode={value!r}; expected "
            f"one of {sorted(_V417X_OPERATOR_KEY_MODES)}")
    return value


def _validate_v4171_srw_composition_mode(value, *, context="v4171"):
    """Validate the static Python-side v4171 SRW composition selector."""
    if isinstance(value, jax.core.Tracer):
        raise ValueError(
            f"{context} srw_composition_mode must be a static Python string; "
            "v4171 does not support a traced or dynamic composition mode")
    if not isinstance(value, str):
        raise ValueError(
            f"{context} srw_composition_mode must be one of "
            f"{sorted(_V4171_SRW_COMPOSITION_MODES)}, got {value!r}")
    if value not in _V4171_SRW_COMPOSITION_MODES:
        raise ValueError(
            f"{context} unsupported srw_composition_mode={value!r}; "
            f"expected one of {sorted(_V4171_SRW_COMPOSITION_MODES)}")
    return value


def _validate_v4171_admission_den_power(value, *, context="v4171"):
    """Validate a static, non-negative composition power before tracing."""
    if isinstance(value, jax.core.Tracer):
        raise ValueError(
            f"{context} admission_den_power must be a static Python scalar; "
            "v4171 does not support a traced or scheduled denominator power")
    if isinstance(value, bool):
        raise ValueError(
            f"{context} admission_den_power must be numeric, not bool")
    if not isinstance(value, numbers.Real):
        raise ValueError(
            f"{context} admission_den_power must be a static numeric Python "
            f"scalar, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{context} admission_den_power must be finite and >= 0.0, "
            f"got {value}")
    return value


def _resolve_v417x_admission_den_powers(
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_power_qk=None,
        admission_den_power_v=None,
        admission_den_power_rst=None, *, context="v417x"):
    """Resolve validated legacy fallback plus QK/V/RST static powers."""
    legacy = _validate_v4171_admission_den_power(
        admission_den_power, context=f"{context}.admission_den_power")
    resolved = []
    for pool, value in (
            ("qk", admission_den_power_qk),
            ("v", admission_den_power_v),
            ("rst", admission_den_power_rst)):
        resolved.append(_validate_v4171_admission_den_power(
            legacy if value is None else value,
            context=f"{context}.admission_den_power_{pool}"))
    return legacy, *resolved


def _validate_v4171_heat_kernel_beta(value, *, context="v4171"):
    """Validate compact spherical heat amplitude sharpness before tracing."""
    if isinstance(value, jax.core.Tracer):
        raise ValueError(
            f"{context} heat_kernel_beta must be a static Python scalar")
    if isinstance(value, bool):
        raise ValueError(
            f"{context} heat_kernel_beta must be numeric, not bool")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} heat_kernel_beta must be finite and > 0, "
            f"got {value!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{context} heat_kernel_beta must be finite and > 0, "
            f"got {value}")
    return value


def _validate_v4171_composition_settings(
        srw_composition_mode, admission_den_power,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA, *, context="v4171"):
    mode = _validate_v4171_srw_composition_mode(
        srw_composition_mode, context=context)
    power = _validate_v4171_admission_den_power(
        admission_den_power, context=context)
    beta = _validate_v4171_heat_kernel_beta(
        heat_kernel_beta, context=context)
    return mode, power, beta


def _validate_v4171_admission_den_grad_scale(value, *, context="v4171"):
    if isinstance(value, jax.core.Tracer):
        raise ValueError(
            f"{context} admission_den_grad_scale must be a static Python scalar")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"v417x requires admission_den_grad_scale=1.0, got {value!r}") from exc
    if not math.isfinite(value) or value != 1.0:
        raise ValueError(
            f"v417x requires admission_den_grad_scale=1.0, got {value}")
    return value


def _mark_v4171_srw_factory_output(
        fn, admission_den_power,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    srw_composition_mode, admission_den_power, heat_kernel_beta = (
        _validate_v4171_composition_settings(
            srw_composition_mode, admission_den_power,
            heat_kernel_beta,
            context="v4171 sharded factory metadata"))
    fn._v4171_srw_composition_mode = srw_composition_mode
    fn._v4171_admission_den_power = float(admission_den_power)
    fn._v4171_heat_kernel_beta = float(heat_kernel_beta)
    fn._v4171_admission_den_grad_scale = 1.0
    return fn


def _validate_v4171_sharded_fns(
        sharded_fns, expected_power,
        expected_mode=DEFAULT_SRW_COMPOSITION_MODE,
        expected_heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA, *,
        expected_power_qk=None, expected_power_v=None,
        expected_power_rst=None):
    if sharded_fns is None:
        return
    if not isinstance(sharded_fns, dict):
        raise ValueError("v417x requires dict-style canonical sharded_fns")
    expected_mode, expected_power, expected_heat_kernel_beta = (
        _validate_v4171_composition_settings(
            expected_mode, expected_power, expected_heat_kernel_beta,
            context="v4171 model/sharded closure"))
    _, qk_power, v_power, rst_power = (
        _resolve_v417x_admission_den_powers(
            expected_power, expected_power_qk, expected_power_v,
            expected_power_rst, context="v4171 model/sharded closure"))
    wrapper_pools = {
        'single': ('v', v_power),
        'attn_v_single': ('v', v_power),
        'v_single': ('v', v_power),
        'attn_v_single_minimal': ('v', v_power),
        'attn_v_single_suppression_minimal': ('v', v_power),
        'attn_v_single_trajectory_minimal': ('v', v_power),
        'rst_single': ('rst', rst_power),
        'rst_single_minimal': ('rst', rst_power),
        'rst_single_suppression_minimal': ('rst', rst_power),
        'rst_single_trajectory_minimal': ('rst', rst_power),
        'paired': ('qk', qk_power),
        'attn_qk_paired': ('qk', qk_power),
        'qk_paired': ('qk', qk_power),
        'attn_qk_single_minimal': ('qk', qk_power),
        'attn_qk_paired_minimal': ('qk', qk_power),
        'attn_qk_paired_suppression_minimal': ('qk', qk_power),
        'attn_qk_paired_trajectory_minimal': ('qk', qk_power),
    }
    for name, (pool, route_power) in wrapper_pools.items():
        fn = sharded_fns.get(name)
        if fn is None:
            continue
        actual_mode = getattr(fn, '_v4171_srw_composition_mode', None)
        if actual_mode is None:
            raise ValueError(
                f"v4171 sharded function {name!r} is missing canonical "
                "srw_composition_mode metadata")
        actual_mode = _validate_v4171_srw_composition_mode(
            actual_mode, context=f"v4171 sharded_fns[{name!r}]")
        if actual_mode != expected_mode:
            raise ValueError(
                "v4171 model/sharded closure srw_composition_mode mismatch: "
                f"sharded_fns[{name!r}]={actual_mode!r}, "
                f"model={expected_mode!r}")
        actual = getattr(fn, '_v4171_admission_den_power', None)
        if actual is None:
            raise ValueError(
                f"v4171 sharded function {name!r} is missing canonical "
                "admission_den_power metadata")
        if float(actual) != float(route_power):
            raise ValueError(
                "v4171 closure/runtime admission_den_power mismatch for "
                f"{pool} pool: sharded_fns[{name!r}]={actual}, "
                f"runtime={route_power}")
        actual_beta = getattr(fn, '_v4171_heat_kernel_beta', None)
        if actual_beta is None:
            raise ValueError(
                f"v4171 sharded function {name!r} is missing canonical "
                "heat_kernel_beta metadata")
        actual_beta = _validate_v4171_heat_kernel_beta(
            actual_beta, context=f"v4171 sharded_fns[{name!r}]")
        if actual_beta != expected_heat_kernel_beta:
            raise ValueError(
                "v4171 model/sharded closure heat_kernel_beta mismatch: "
                f"sharded_fns[{name!r}]={actual_beta}, "
                f"model={expected_heat_kernel_beta}")
    for production_name, suppression_name in (
            ('attn_v_single_minimal',
             'attn_v_single_suppression_minimal'),
            ('rst_single_minimal', 'rst_single_suppression_minimal'),
            ('attn_qk_paired_minimal',
             'attn_qk_paired_suppression_minimal')):
        production_fn = sharded_fns.get(production_name)
        suppression_fn = sharded_fns.get(suppression_name)
        if production_fn is None or suppression_fn is None:
            continue
        production_kernel = getattr(
            production_fn, '_v4171_canonical_shard_map_kernel', None)
        suppression_kernel = getattr(
            suppression_fn, '_v4171_canonical_shard_map_kernel', None)
        if production_kernel is None or production_kernel is not suppression_kernel:
            raise ValueError(
                "v4171 production/suppression wrappers must share one "
                f"canonical shard-map kernel: {production_name!r} vs "
                f"{suppression_name!r}")


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


def _unit_direction_max_floor(x, eps=RW_FORWARD_NORM_EPS):
    x = jnp.asarray(x, dtype=jnp.float32)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, jnp.asarray(eps, dtype=jnp.float32))


def _generalized_bilinear_operator_key_components(
        read_vectors, write_vectors, read_probes, write_probes,
        eps=RW_FORWARD_NORM_EPS):
    read_vectors = jnp.asarray(read_vectors, dtype=jnp.float32)
    write_vectors = jnp.asarray(write_vectors, dtype=jnp.float32)
    read_probes = jnp.asarray(read_probes, dtype=jnp.float32)
    write_probes = jnp.asarray(write_probes, dtype=jnp.float32)
    if (read_vectors.ndim != 2 or write_vectors.ndim != 2
            or read_vectors.shape != write_vectors.shape):
        raise ValueError(
            "generalized bilinear read/write must have matching rank-2 "
            f"[N, d_model] shapes, got read={read_vectors.shape}, "
            f"write={write_vectors.shape}")
    if read_probes.ndim != 2 or write_probes.ndim != 2:
        raise ValueError(
            "generalized bilinear probes must have rank 2 [d_model, d_route], "
            f"got read_probes={read_probes.shape}, "
            f"write_probes={write_probes.shape}")
    expected_model = int(read_vectors.shape[1])
    if (int(read_probes.shape[0]) != expected_model
            or int(write_probes.shape[0]) != expected_model
            or tuple(read_probes.shape) != tuple(write_probes.shape)):
        raise ValueError(
            "generalized bilinear probe shape mismatch: expected matching "
            f"[{expected_model}, d_route], got read_probes={read_probes.shape}, "
            f"write_probes={write_probes.shape}")
    read_directions = _unit_direction_max_floor(read_vectors, eps)
    write_directions = _unit_direction_max_floor(write_vectors, eps)
    read_features = read_directions @ read_probes
    write_features = write_directions @ write_probes
    read_feature_directions = _unit_direction_max_floor(read_features, eps)
    write_feature_directions = _unit_direction_max_floor(write_features, eps)
    raw_operator_keys = read_feature_directions * write_feature_directions
    operator_keys = _unit_direction_max_floor(raw_operator_keys, eps)
    return operator_keys, read_features, write_features, raw_operator_keys


def materialize_generalized_bilinear_operator_keys(
        read_vectors, write_vectors, read_probes, write_probes,
        eps=RW_FORWARD_NORM_EPS):
    """Materialize live coordinate-wise bilinear RW addresses for one pool."""
    operator_keys, _, _, _ = _generalized_bilinear_operator_key_components(
        read_vectors, write_vectors, read_probes, write_probes, eps)
    return operator_keys


def generalized_bilinear_operator_key_diagnostics(
        read_vectors, write_vectors, read_probes, write_probes,
        eps=RW_FORWARD_NORM_EPS):
    """Return lightweight generated-key diagnostics for analysis cadence."""
    operator_keys, read_features, write_features, raw_operator_keys = (
        _generalized_bilinear_operator_key_components(
            read_vectors, write_vectors, read_probes, write_probes, eps))

    def _norm_stats(value, prefix):
        norms = jnp.linalg.norm(value, axis=-1)
        return {
            f'{prefix}_mean': norms.mean(),
            f'{prefix}_min': norms.min(),
            f'{prefix}_max': norms.max(),
            f'{prefix}_std': norms.std(),
        }

    out = {}
    out.update(_norm_stats(operator_keys, 'key_norm'))
    out.update(_norm_stats(raw_operator_keys, 'raw_product_norm'))
    out.update(_norm_stats(read_features, 'read_projected_norm'))
    out.update(_norm_stats(write_features, 'write_projected_norm'))
    return out


def symbolic_parameter_count(model_cfg):
    """Return the exact parameter breakdown for the shared v417x core."""
    if (isinstance(model_cfg, dict) and 'model' in model_cfg
            and isinstance(model_cfg['model'], dict)):
        model_cfg = model_cfg['model']
    if not isinstance(model_cfg, dict):
        raise ValueError("symbolic parameter count requires a model config dict")

    def _positive_int(name, fallback=None):
        value = model_cfg.get(name, fallback)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value <= 0):
            raise ValueError(
                f"model.{name} must be a positive integer, got {value!r}")
        return int(value)

    vocab_size = _positive_int(
        'vocab_size_padded', model_cfg.get('vocab_size'))
    max_seq_len = _positive_int('max_seq_len')
    d_model = _positive_int('d_model')
    d_route = _positive_int('d_route')
    n_layers = _positive_int('n_layers')
    n_qk = _positive_int('n_qk')
    n_v = _positive_int('n_v')
    n_rst = _positive_int('n_rst', model_cfg.get('n_know'))
    operator_count = n_qk + n_v + n_rst
    mode = _validate_operator_key_mode(
        model_cfg.get('operator_key_mode', OPERATOR_KEY_MODE_LEARNED),
        context="symbolic parameter count")
    learned_keys = (
        operator_count * d_route
        if mode == OPERATOR_KEY_MODE_LEARNED else 0)
    bilinear_probes = (
        2 * d_model * d_route
        if mode == OPERATOR_KEY_MODE_GENERALIZED_BILINEAR else 0)
    counts = {
        'token_embedding': vocab_size * d_model,
        'position_embedding': max_seq_len * d_model,
        'layer_stack': n_layers * (d_model * d_model + 4 * d_model),
        'router': (
            4 * d_model * d_route + 4 * d_route + 4 * d_model + 4),
        'read_write_pools': 2 * operator_count * d_model,
        'learned_key_tables': learned_keys,
        'bilinear_probe_matrices': bilinear_probes,
        'final_norm': 2 * d_model,
    }
    counts['total'] = sum(counts.values())
    return counts


_LEARNED_OPERATOR_KEY_NAMES = (
    'attn_qk_op_key', 'attn_v_op_key', 'rst_op_key')
_BILINEAR_PROBE_NAMES = ('rw_key_read_probe', 'rw_key_write_probe')


def _resolve_operator_key_mode(pool_params, operator_key_mode=None):
    if operator_key_mode is not None:
        return _validate_operator_key_mode(
            operator_key_mode, context="v417x neuron_pool")
    learned_present = tuple(
        name for name in _LEARNED_OPERATOR_KEY_NAMES if name in pool_params)
    probes_present = tuple(
        name for name in _BILINEAR_PROBE_NAMES if name in pool_params)
    if learned_present and probes_present:
        raise ValueError(
            "v417x neuron_pool mixes learned operator key tables with "
            "generalized bilinear probes")
    if learned_present:
        return OPERATOR_KEY_MODE_LEARNED
    if probes_present:
        return OPERATOR_KEY_MODE_GENERALIZED_BILINEAR
    raise ValueError(
        "v417x neuron_pool has neither learned operator key tables nor "
        "generalized bilinear probes")


def _pool_operator_keys(pool_params, operator_key_mode=None):
    mode = _resolve_operator_key_mode(pool_params, operator_key_mode)
    learned_present = tuple(
        name for name in _LEARNED_OPERATOR_KEY_NAMES if name in pool_params)
    probes_present = tuple(
        name for name in _BILINEAR_PROBE_NAMES if name in pool_params)
    if mode == OPERATOR_KEY_MODE_LEARNED:
        if probes_present:
            raise ValueError(
                "learned_operator_embedding neuron_pool must not contain "
                "generalized bilinear probes: " + ", ".join(probes_present))
        missing = tuple(
            key for key in _LEARNED_OPERATOR_KEY_NAMES if key not in pool_params)
        if missing:
            raise ValueError(
                "v4171 neuron_pool is missing learned operator embeddings: "
                + ", ".join(missing))
        keys = {
            name: pool_params[name] for name in _LEARNED_OPERATOR_KEY_NAMES
        }
    else:
        if learned_present:
            raise ValueError(
                "generalized_bilinear_rw neuron_pool must not contain stored "
                "operator key tables: " + ", ".join(learned_present))
        missing = tuple(
            key for key in _BILINEAR_PROBE_NAMES if key not in pool_params)
        if missing:
            raise ValueError(
                "generalized_bilinear_rw neuron_pool is missing shared probes: "
                + ", ".join(missing))
        read_probe = pool_params['rw_key_read_probe']
        write_probe = pool_params['rw_key_write_probe']
        keys = {}
        for prefix in ('attn_qk', 'attn_v', 'rst'):
            read_name = f'{prefix}_read'
            write_name = f'{prefix}_write'
            missing_rw = tuple(
                name for name in (read_name, write_name)
                if name not in pool_params)
            if missing_rw:
                raise ValueError(
                    "generalized_bilinear_rw neuron_pool is missing RW "
                    "parameters: " + ", ".join(missing_rw))
            keys[f'{prefix}_op_key'] = (
                materialize_generalized_bilinear_operator_keys(
                pool_params[read_name], pool_params[write_name],
                read_probe, write_probe))
    d_route = None
    for prefix, read_key in (
            ('attn_qk', 'attn_qk_read'),
            ('attn_v', 'attn_v_read'),
            ('rst', 'rst_read')):
        operator_keys = keys[f'{prefix}_op_key']
        if operator_keys.ndim != 2:
            raise ValueError(
                f"v417x {prefix}_op_key must have rank 2 [N, d_route], "
                f"got {operator_keys.shape}")
        if read_key in pool_params:
            expected_rows = int(pool_params[read_key].shape[0])
            if int(operator_keys.shape[0]) != expected_rows:
                raise ValueError(
                    f"v417x {prefix}_op_key shape mismatch: expected "
                    f"[{expected_rows}, d_route], got {operator_keys.shape}")
        if d_route is None:
            d_route = int(operator_keys.shape[1])
            if d_route <= 0:
                raise ValueError(
                    f"v417x {prefix}_op_key must have d_route > 0, got "
                    f"{operator_keys.shape}")
        elif int(operator_keys.shape[1]) != d_route:
            raise ValueError(
                f"v417x {prefix}_op_key route width mismatch: expected "
                f"d_route={d_route}, got {operator_keys.shape}")
    return keys


def _composition_den(
        admission_mass, admission_den_power,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE):
    """Live-gradient canonical SRW composition denominator."""
    # Mode/power compatibility is validated at config, model, and factory
    # boundaries. Keep this numerical helper trace-safe because the already
    # validated power is represented as a scalar JAX value in compiled paths.
    srw_composition_mode = _validate_v4171_srw_composition_mode(
        srw_composition_mode, context="v4171 composition denominator")
    admission_mass = jnp.asarray(admission_mass, dtype=jnp.float32)
    if srw_composition_mode in ("quadratic", "heat_energy"):
        den_eps = (
            HEAT_ENERGY_DEN_EPS
            if srw_composition_mode == "heat_energy"
            else QUADRATIC_DEN_EPS)
        den_floor_mass = jnp.float32(den_eps ** 2)
    else:
        den_floor_mass = jnp.float32(1.0)
    return jnp.power(
        jnp.maximum(admission_mass, den_floor_mass),
        jnp.asarray(admission_den_power, dtype=jnp.float32))


def _composition_den_floor_mass(
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE):
    mode = _validate_v4171_srw_composition_mode(
        srw_composition_mode, context="v4171 composition denominator floor")
    if mode in ("quadratic", "heat_energy"):
        if mode == "heat_energy":
            return HEAT_ENERGY_DEN_EPS ** 2
        return QUADRATIC_DEN_EPS ** 2
    return 1.0


def _analysis_int32_array(value, *, name):
    """Convert an integer analysis input without accepting lossy casts."""
    source_dtype = getattr(value, "dtype", None)
    if (source_dtype is not None
            and not jnp.issubdtype(source_dtype, jnp.integer)):
        raise TypeError(
            f"{name} must have an integer dtype, got {source_dtype}")
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(
            f"{name} must have an integer dtype, got {array.dtype}")
    return array.astype(jnp.int32)


def _analysis_sorted_operator_membership(
        global_operator_ids, sorted_selected_operator_ids):
    """Match a pool chunk without materializing ``pool x selected``."""
    def row_membership(row):
        insertion = jnp.searchsorted(
            row, global_operator_ids, side="left")
        insertion = jnp.clip(insertion, 0, row.shape[0] - 1)
        return row[insertion] == global_operator_ids

    if sorted_selected_operator_ids.ndim == 1:
        return (
            global_operator_ids[None, None, :]
            == sorted_selected_operator_ids[:, None, None])
    if sorted_selected_operator_ids.ndim == 2:
        return jax.vmap(row_membership)(
            sorted_selected_operator_ids)[:, None, :]
    if sorted_selected_operator_ids.ndim == 3:
        return jax.vmap(jax.vmap(row_membership))(
            sorted_selected_operator_ids)
    raise ValueError(
        "selected operator ids must have shape [B], [B, M], or "
        "[B, route, M], got "
        f"{sorted_selected_operator_ids.shape}")


def analysis_operator_membership(global_operator_ids, selected_operator_ids):
    """Build a chunk-local analysis membership mask.

    The existing single-id shape is ``[B]``.  Group analysis uses one static,
    ``-1`` padded ``[B, M]`` shape for every requested group size. Membership
    uses sorted indexed lookup, not a dense ``B x pool x M`` one-hot tensor.
    This changes only execution numerators; admission and every production
    denominator/statistic remain untouched.
    """
    global_operator_ids = _analysis_int32_array(
        global_operator_ids, name="global_operator_ids")
    selected_operator_ids = _analysis_int32_array(
        selected_operator_ids, name="selected_operator_ids")
    sorted_ids = (
        jnp.sort(selected_operator_ids, axis=-1)
        if selected_operator_ids.ndim == 2 else selected_operator_ids)
    return _analysis_sorted_operator_membership(
        global_operator_ids, sorted_ids)


def _analysis_group_nonempty(selected_operator_ids):
    """Return one membership bit per example for ``-1`` padded groups."""
    selected_operator_ids = _analysis_int32_array(
        selected_operator_ids, name="selected_operator_ids")
    if selected_operator_ids.ndim == 1:
        return selected_operator_ids >= jnp.int32(0)
    if selected_operator_ids.ndim == 2:
        return jnp.any(
            selected_operator_ids >= jnp.int32(0), axis=-1)
    raise ValueError(
        "selected operator ids must have shape [B] or [B, M], got "
        f"{selected_operator_ids.shape}")


TRAJECTORY_PATCH_STAGES = {
    "q": 0,
    "k": 1,
    "v": 2,
    "rst": 3,
    "residual_input": 4,
    "post_attention": 5,
    "post_rst": 6,
}

TRAJECTORY_TRACE_FIELDS = (
    "production_output",
    "selected_replay_output",
    "production_precast_output",
    "selected_replay_precast_output",
    "query",
    "tau",
    "admission_mass",
    "denominator",
    "numerator_active_count",
    "denominator_active_count",
    "operator_id",
    "operator_valid",
    "read_scalar_bf16_bits",
    "prewrite_amplitude_bf16_bits",
    "execution_weight",
    "admission",
    "margin",
    "rho",
    "position_valid",
)


def _analysis_apply_trajectory_patch(
        value, *, layer_index, stage_code, patch_layers, patch_positions,
        patch_stages, patch_enabled, patch_values):
    """Apply one fixed-width batch-native trajectory patch schedule.

    Every schedule has shape ``[B, P]`` and every replacement has shape
    ``[B, P, D]``.  Host validation rejects duplicate enabled sites, so the
    sum below selects either exactly one replacement or the untouched value.
    """
    B, S, D = value.shape
    slot_match = (
        jnp.asarray(patch_enabled, dtype=jnp.bool_)
        & (jnp.asarray(patch_layers, dtype=jnp.int32)
           == jnp.asarray(layer_index, dtype=jnp.int32))
        & (jnp.asarray(patch_stages, dtype=jnp.int32)
           == jnp.int32(stage_code)))
    position_match = (
        jnp.arange(S, dtype=jnp.int32)[None, None, :]
        == jnp.asarray(patch_positions, dtype=jnp.int32)[:, :, None])
    site_mask = slot_match[:, :, None] & position_match
    replacement = jnp.einsum(
        "bps,bpd->bsd", site_mask.astype(jnp.float32),
        jnp.asarray(patch_values, dtype=jnp.float32))
    apply_mask = jnp.any(site_mask, axis=1)[:, :, None]
    return jnp.where(apply_mask, replacement, value)


def _analysis_replace_trajectory_positions(
        value, replacements, positions, position_valid):
    """Replace unique fixed-width positions without a Python-side loop."""
    _, S, _ = value.shape
    position_mask = (
        jnp.arange(S, dtype=jnp.int32)[None, None, :]
        == jnp.asarray(positions, dtype=jnp.int32)[:, :, None])
    position_mask &= jnp.asarray(position_valid, dtype=jnp.bool_)[:, :, None]
    replacement = jnp.einsum(
        "bts,btd->bsd", position_mask.astype(jnp.float32),
        jnp.asarray(replacements, dtype=jnp.float32))
    return jnp.where(
        jnp.any(position_mask, axis=1)[:, :, None], replacement, value)


def _validate_concrete_analysis_trajectory(
        *, positions, position_valid, selected_ids_by_route,
        selected_valid_by_route, patch_layers, patch_positions, patch_stages,
        patch_enabled, patch_values, batch_size, sequence_length, n_layers,
        route_pool_sizes):
    """Fail closed on host-visible trajectory schedules before tracing."""
    values = (
        positions, position_valid, patch_layers, patch_positions,
        patch_stages, patch_enabled, patch_values,
        *selected_ids_by_route.values(), *selected_valid_by_route.values())
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return
    positions_host = np.asarray(jax.device_get(positions), dtype=np.int64)
    position_valid_host = np.asarray(
        jax.device_get(position_valid), dtype=np.bool_)
    if positions_host.ndim != 2 or positions_host.shape[0] != batch_size:
        raise ValueError("trajectory positions must have shape [B, T]")
    if position_valid_host.shape != positions_host.shape:
        raise ValueError("trajectory position validity shape mismatch")
    for row_positions, row_valid in zip(
            positions_host, position_valid_host):
        active = row_positions[row_valid]
        if np.any((active < 0) | (active >= sequence_length)):
            raise ValueError("trajectory position is outside the sequence")
        if len(set(int(value) for value in active.tolist())) != active.size:
            raise ValueError("trajectory positions must be unique per example")
    expected_prefix = (n_layers, batch_size, positions_host.shape[1])
    for route, ids in selected_ids_by_route.items():
        ids_host = np.asarray(jax.device_get(ids), dtype=np.int64)
        valid_host = np.asarray(
            jax.device_get(selected_valid_by_route[route]), dtype=np.bool_)
        if ids_host.ndim != 4 or ids_host.shape[:3] != expected_prefix:
            raise ValueError(
                f"trajectory {route} ids must have shape [L,B,T,K]")
        if valid_host.shape != ids_host.shape:
            raise ValueError(f"trajectory {route} validity shape mismatch")
        pool_size = int(route_pool_sizes[route])
        if np.any((ids_host[valid_host] < 0)
                  | (ids_host[valid_host] >= pool_size)):
            raise ValueError(f"trajectory {route} operator id is out of range")
    patch_layers_host = np.asarray(
        jax.device_get(patch_layers), dtype=np.int64)
    patch_positions_host = np.asarray(
        jax.device_get(patch_positions), dtype=np.int64)
    patch_stages_host = np.asarray(
        jax.device_get(patch_stages), dtype=np.int64)
    patch_enabled_host = np.asarray(
        jax.device_get(patch_enabled), dtype=np.bool_)
    patch_values_host = np.asarray(jax.device_get(patch_values))
    if patch_layers_host.ndim != 2 or patch_layers_host.shape[0] != batch_size:
        raise ValueError("trajectory patch schedule must have shape [B, P]")
    patch_shape = patch_layers_host.shape
    if any(value.shape != patch_shape for value in (
            patch_positions_host, patch_stages_host, patch_enabled_host)):
        raise ValueError("trajectory patch schedule shapes disagree")
    if patch_values_host.shape[:2] != patch_shape:
        raise ValueError("trajectory patch values shape mismatch")
    known_stages = set(TRAJECTORY_PATCH_STAGES.values())
    for row in range(batch_size):
        sites = []
        for slot in np.flatnonzero(patch_enabled_host[row]):
            layer = int(patch_layers_host[row, slot])
            position = int(patch_positions_host[row, slot])
            stage = int(patch_stages_host[row, slot])
            if not 0 <= layer < n_layers:
                raise ValueError("trajectory patch layer is out of range")
            if not 0 <= position < sequence_length:
                raise ValueError("trajectory patch position is out of range")
            if stage not in known_stages:
                raise ValueError("trajectory patch stage is unknown")
            sites.append((layer, position, stage))
        if len(set(sites)) != len(sites):
            raise ValueError("trajectory patch schedule contains duplicates")


def _validate_concrete_trajectory_patches(
        *, patch_layers, patch_positions, patch_stages, patch_enabled,
        patch_values, batch_size, sequence_length, n_layers):
    values = (
        patch_layers, patch_positions, patch_stages, patch_enabled,
        patch_values)
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return
    layers = np.asarray(jax.device_get(patch_layers), dtype=np.int64)
    positions = np.asarray(jax.device_get(patch_positions), dtype=np.int64)
    stages = np.asarray(jax.device_get(patch_stages), dtype=np.int64)
    enabled = np.asarray(jax.device_get(patch_enabled), dtype=np.bool_)
    patch_values_host = np.asarray(jax.device_get(patch_values))
    if layers.ndim != 2 or layers.shape[0] != batch_size:
        raise ValueError("trajectory patch schedule must have shape [B, P]")
    if any(value.shape != layers.shape for value in (
            positions, stages, enabled)):
        raise ValueError("trajectory patch schedule shapes disagree")
    if patch_values_host.shape[:2] != layers.shape:
        raise ValueError("trajectory patch values shape mismatch")
    known_stages = set(TRAJECTORY_PATCH_STAGES.values())
    for row in range(batch_size):
        sites = []
        for slot in np.flatnonzero(enabled[row]):
            layer = int(layers[row, slot])
            position = int(positions[row, slot])
            stage = int(stages[row, slot])
            if not 0 <= layer < n_layers:
                raise ValueError("trajectory patch layer is out of range")
            if not 0 <= position < sequence_length:
                raise ValueError("trajectory patch position is out of range")
            if stage not in known_stages:
                raise ValueError("trajectory patch stage is unknown")
            sites.append((layer, position, stage))
        if len(set(sites)) != len(sites):
            raise ValueError("trajectory patch schedule contains duplicates")


def _validate_concrete_analysis_interchange(
        target_positions, target_layer, target_route,
        selected_operator_ids, *, batch_size, sequence_length, n_layers,
        n_qk, n_v, n_rst, enabled):
    """Validate values before JIT; traced callers use their host validator."""
    values = (
        enabled, target_positions, target_layer, target_route,
        selected_operator_ids)
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return
    if not bool(jax.device_get(enabled)):
        return
    positions = tuple(
        int(value) for value in
        jax.device_get(target_positions).reshape(-1).tolist())
    layer = int(jax.device_get(target_layer))
    route = int(jax.device_get(target_route))
    operator_ids = tuple(
        int(value) for value in
        jax.device_get(selected_operator_ids).reshape(-1).tolist())
    if len(positions) != int(batch_size):
        raise ValueError(
            "analysis target position batch mismatch: "
            f"expected={batch_size} actual={len(positions)}")
    if any(not 0 <= value < int(sequence_length) for value in positions):
        raise ValueError(
            "analysis target position is out of range: "
            f"positions={positions}")
    if not 0 <= layer < int(n_layers):
        raise ValueError(
            f"analysis target layer is out of range: layer={layer}")
    if not 0 <= route < 4:
        raise ValueError(
            f"analysis target route is out of range: route={route}")
    pool_size = (n_qk if route <= 1 else n_v if route == 2 else n_rst)
    if any(
            value != -1 and not 0 <= value < int(pool_size)
            for value in operator_ids):
        raise ValueError(
            "analysis operator group contains an invalid id for its route: "
            f"route={route} operator_ids={operator_ids}")


def _validate_concrete_analysis_program(
        *, program_mode, target_positions,
        selected_ids_q, selected_ids_k, selected_ids_v, selected_ids_rst,
        selected_valid_q, selected_valid_k, selected_valid_v,
        selected_valid_rst, batch_size, sequence_length, n_layers,
        n_qk, n_v, n_rst, enabled):
    """Fail loudly on concrete dynamic-program values before compilation."""
    values = (
        program_mode, target_positions,
        selected_ids_q, selected_ids_k, selected_ids_v, selected_ids_rst,
        selected_valid_q, selected_valid_k, selected_valid_v,
        selected_valid_rst,
    )
    if not enabled or any(isinstance(value, jax.core.Tracer) for value in values):
        return
    mode = int(jax.device_get(program_mode))
    if not 0 <= mode <= 4:
        raise ValueError(f"analysis program mode is out of range: {mode}")
    positions = np.asarray(jax.device_get(target_positions), dtype=np.int64)
    if positions.shape != (int(batch_size),):
        raise ValueError("analysis program target-position batch mismatch")
    if np.any(positions < 0) or np.any(positions >= int(sequence_length)):
        raise ValueError("analysis program target position is out of range")
    route_values = (
        ("q", selected_ids_q, selected_valid_q, n_qk),
        ("k", selected_ids_k, selected_valid_k, n_qk),
        ("v", selected_ids_v, selected_valid_v, n_v),
        ("rst", selected_ids_rst, selected_valid_rst, n_rst),
    )
    for route, ids_value, valid_value, pool_size in route_values:
        ids = np.asarray(jax.device_get(ids_value))
        valid = np.asarray(jax.device_get(valid_value))
        if ids.shape[:2] != (int(n_layers), int(batch_size)):
            raise ValueError(f"analysis program {route} shape mismatch")
        if np.any(ids[~valid] != 0):
            raise ValueError(
                f"analysis program {route} invalid padding must use id 0")
        selected = ids[valid]
        if selected.size and (
                int(selected.min()) < 0 or int(selected.max()) >= int(pool_size)):
            raise ValueError(
                f"analysis program {route} contains an out-of-range id")
        for layer in range(ids.shape[0]):
            for batch_index in range(ids.shape[1]):
                row = ids[layer, batch_index, valid[layer, batch_index]]
                if len(set(int(value) for value in row.tolist())) != row.size:
                    raise ValueError(
                        f"analysis program {route} contains duplicate ids")


def _pool_params_with_operator_keys(pool_params, operator_key_mode=None):
    """Materialize live operator keys once before the layer scan."""
    mode = _resolve_operator_key_mode(pool_params, operator_key_mode)
    keys = _pool_operator_keys(pool_params, mode)
    if mode == OPERATOR_KEY_MODE_LEARNED:
        return pool_params
    materialized = {
        key: value for key, value in pool_params.items()
        if key not in _BILINEAR_PROBE_NAMES
    }
    materialized.update(keys)
    return materialized


def _ensure_pool_operator_keys(pool_params):
    # The top-level model forward has already validated/materialized this
    # schema. Keep the layer scan free of repeated key-helper calls.
    if (all(name in pool_params for name in _LEARNED_OPERATOR_KEY_NAMES)
            and not any(name in pool_params for name in _BILINEAR_PROBE_NAMES)):
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


def _chunked_ce_loss_and_acc(shift_x, embedding_matrix, shift_labels,
                             valid_mask, token_chunk_size=32768,
                             compute_accuracy=True,
                             logical_vocab_size=None,
                             compute_logit_stats=True):
    token_chunk_size = int(token_chunk_size)
    if token_chunk_size <= 0:
        raise ValueError(
            f"token_chunk_size must be > 0, got {token_chunk_size}")
    compute_accuracy = bool(compute_accuracy)
    compute_logit_stats = bool(compute_logit_stats)
    logical_vocab_size = (
        int(logical_vocab_size)
        if logical_vocab_size is not None
        else int(embedding_matrix.shape[0]))
    if logical_vocab_size <= 0:
        raise ValueError(
            f"logical_vocab_size must be > 0, got {logical_vocab_size}")
    if int(embedding_matrix.shape[0]) < logical_vocab_size:
        raise ValueError(
            f"embedding vocab size {embedding_matrix.shape[0]} is smaller "
            f"than logical_vocab_size={logical_vocab_size}")
    B, T, D = shift_x.shape
    flat_x = shift_x.reshape(B * T, D)
    flat_labels = shift_labels.reshape(B * T)
    flat_valid = valid_mask.reshape(B * T)

    n_tokens = flat_x.shape[0]
    pad = (-n_tokens) % token_chunk_size
    flat_x = jnp.pad(flat_x, ((0, pad), (0, 0)))
    flat_labels = jnp.pad(flat_labels, ((0, pad),), constant_values=0)
    flat_valid = jnp.pad(flat_valid, ((0, pad),), constant_values=False)

    flat_x = flat_x.reshape(-1, token_chunk_size, D)
    flat_labels = flat_labels.reshape(-1, token_chunk_size)
    flat_valid = flat_valid.reshape(-1, token_chunk_size)
    vocab_ids = jnp.arange(embedding_matrix.shape[0], dtype=jnp.int32)
    valid_vocab = vocab_ids < logical_vocab_size
    neg_inf = jnp.finfo(jnp.float32).min

    @jax.checkpoint
    def step(carry, xs):
        del carry
        x_c, labels_c, valid_c = xs

        logits = (x_c @ embedding_matrix.T).astype(jnp.float32)
        logits = jnp.where(valid_vocab[None, :], logits, neg_inf)
        safe_labels = jnp.where(valid_c, labels_c, 0)
        target_logits = jnp.take_along_axis(
            logits, safe_labels[..., None], axis=-1).squeeze(-1)
        token_loss = jax.nn.logsumexp(logits, axis=-1) - target_logits
        token_loss = token_loss.astype(jnp.float32)
        token_loss = jnp.where(valid_c, token_loss, 0.0)

        if compute_accuracy:
            preds = jnp.argmax(logits, axis=-1)
            correct_delta = ((preds == labels_c) & valid_c).astype(
                jnp.int32).sum()
        else:
            correct_delta = jnp.array(0, dtype=jnp.int32)
        if compute_logit_stats:
            valid_2d = valid_c[:, None] & valid_vocab[None, :]
            logits_for_sum = jnp.where(valid_2d, logits, 0.0)
            local_logit_sum = jnp.sum(logits_for_sum)
            local_logit_sumsq = jnp.sum(logits_for_sum * logits_for_sum)
            local_abs_max = jnp.max(jnp.where(
                valid_2d, jnp.abs(logits), 0.0))
            token_sumsq = jnp.sum(
                jnp.where(valid_vocab[None, :], logits * logits, 0.0),
                axis=-1)
            logit_norm_sum = jnp.sum(
                jnp.where(
                    valid_c,
                    jnp.sqrt(jnp.maximum(token_sumsq, 0.0)),
                    0.0))
        else:
            local_logit_sum = jnp.array(0.0, dtype=jnp.float32)
            local_logit_sumsq = jnp.array(0.0, dtype=jnp.float32)
            local_abs_max = jnp.array(0.0, dtype=jnp.float32)
            logit_norm_sum = jnp.array(0.0, dtype=jnp.float32)
        return None, (
            token_loss,
            correct_delta,
            valid_c.astype(jnp.int32).sum(),
            token_loss.sum(),
            local_abs_max,
            local_logit_sum,
            local_logit_sumsq,
            logit_norm_sum,
        )

    _, ys = jax.lax.scan(step, None, (flat_x, flat_labels, flat_valid))
    (token_loss_chunks, correct_chunks, valid_chunks, loss_chunks,
     abs_max_chunks, logit_sum_chunks, logit_sumsq_chunks,
     logit_norm_chunks) = ys
    per_token_ce_flat = token_loss_chunks.reshape(-1)[:n_tokens]
    per_token_ce = per_token_ce_flat.reshape(B, T)
    loss_sum = jnp.sum(loss_chunks)
    correct = jnp.sum(correct_chunks)
    valid_count = jnp.sum(valid_chunks)
    loss = loss_sum / (valid_count.astype(jnp.float32) + 1e-8)
    logit_abs_max = jnp.max(abs_max_chunks)
    diag_count = (
        valid_count.astype(jnp.float32)
        * jnp.asarray(logical_vocab_size, dtype=jnp.float32))
    logit_sum = jnp.sum(logit_sum_chunks)
    logit_sumsq = jnp.sum(logit_sumsq_chunks)
    logit_mean = logit_sum / (diag_count + 1e-8)
    logit_var = logit_sumsq / (diag_count + 1e-8) - logit_mean * logit_mean
    logit_std = jnp.sqrt(jnp.maximum(logit_var, 0.0))
    logit_norm_mean = (
        jnp.sum(logit_norm_chunks)
        / (valid_count.astype(jnp.float32) + 1e-8))
    if not compute_logit_stats:
        logit_abs_max = jnp.array(0.0, dtype=jnp.float32)
        logit_norm_mean = jnp.array(0.0, dtype=jnp.float32)
        logit_mean = jnp.array(0.0, dtype=jnp.float32)
        logit_std = jnp.array(0.0, dtype=jnp.float32)
    return (loss, per_token_ce, correct, valid_count,
            logit_abs_max, logit_norm_mean, logit_mean, logit_std)


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


def _linear_angular_depth_from_margin(margin, tau, eps=1.0e-4):
    tau = jnp.asarray(tau, dtype=jnp.float32)
    margin = jnp.asarray(margin, dtype=jnp.float32)
    den = jnp.maximum(jnp.float32(1.0) - tau, jnp.float32(eps))
    return jnp.clip(margin / den, jnp.float32(0.0), jnp.float32(1.0))


def _compact_heat_kernel_from_amplitude(
        angular_amplitude, heat_kernel_beta):
    """Compact spherical heat amplitude on the DirectTau cap.

    This overflow-safe form is algebraically equal to
    expm1(beta * amplitude) / expm1(beta). The cap amplitude already makes
    the amplitude exactly zero at and outside the DirectTau boundary.
    """
    a = jnp.clip(
        jnp.asarray(angular_amplitude, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    beta = jnp.minimum(
        jnp.asarray(heat_kernel_beta, dtype=jnp.float32),
        jnp.finfo(jnp.float32).max)
    denominator = -jnp.expm1(-beta)
    numerator = (
        jnp.exp(beta * (a - jnp.float32(1.0)))
        * (-jnp.expm1(-beta * a))
    )
    ratio = numerator / jnp.maximum(
        denominator, jnp.finfo(jnp.float32).tiny)
    return jnp.where(beta < jnp.float32(1.0e-4), a, ratio)


def _boundary_gate_from_margin(margin, tau, boundary_power=None):
    """Compatibility hook for diagnostics; v4171 has no projected gate."""
    del boundary_power
    return _linear_angular_depth_from_margin(margin, tau)


def _compute_admission_drive(score, tau, boundary_scale,
                             boundary_power=2.0,
                             effective_active_eps=1.0e-6,
                             execution_prune_eps=0.0,
                             srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
                             heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """v4171 canonical angular selection and mode-specific composition.

    The historical tuple slots are kept for scan compatibility:
    selection margin, unpruned composition admission, linear angular
    amplitude, pruned composition execution weight, and active mask.
    """
    del boundary_scale, boundary_power, effective_active_eps
    srw_composition_mode = _validate_v4171_srw_composition_mode(
        srw_composition_mode, context="v4171 admission drive")
    execution_prune_eps = jnp.asarray(execution_prune_eps, dtype=jnp.float32)
    score = jnp.clip(jnp.asarray(score, dtype=jnp.float32), -1.0, 1.0)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    selection_margin = score - tau
    angular_amplitude = _linear_angular_depth_from_margin(
        selection_margin, tau)
    if srw_composition_mode == "quadratic":
        admission_weight = angular_amplitude * angular_amplitude
    elif srw_composition_mode == "heat_energy":
        heat_amplitude = _compact_heat_kernel_from_amplitude(
            angular_amplitude, heat_kernel_beta)
        admission_weight = heat_amplitude * heat_amplitude
    else:
        admission_weight = angular_amplitude
    execution_weight = jnp.where(
        execution_prune_eps > 0.0,
        jnp.where(
            admission_weight >= execution_prune_eps,
            admission_weight, 0.0),
        admission_weight)
    active_mask = selection_margin > jnp.float32(0.0)
    return (
        selection_margin,
        admission_weight,
        angular_amplitude,
        execution_weight,
        active_mask,
    )




def make_sharded_srw(mesh, max_chunk_size=2048,
                     analysis=False,
                     dead_exposure_target=0.1,
                     soft_gate_effective_active_eps=1.0e-6,
                     admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                     admission_den_grad_scale=1.0,
                     srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
                     heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create fused shard_map'd angular Select + SRW.

    Fast train path: one chunked pass computes rho, tau, gate, and SRW.
    Analysis path may compute rho distribution moments for diagnostics.

    v4171 canonical DirectTau execution:
        rho               = cosine(operator_query, operator_keys)
        margin            = rho - tau
        angular_amplitude = clip(margin / max(1 - tau, 1e-4), 0, 1)
        admission_weight  = mode-specific unpruned composition weight
        execution_weight  = pruned admission_weight
        den               = mode-specific unpruned-admission normalization


    `analysis=False` (default, train path): returns the SLIM tuple plus
    four gate-concentration diagnostics, and skips distribution-shape stats
    (skew/kurt), selection-residency/entropy diagnostics and drive extrema.
    XLA DCE's the unused work.
    `analysis=True`: returns the SLIM/concentration tuple followed by
    observational scalars/arrays for route shape, gate concentration, and
    denominator diagnostics.
    Used by analysis_step at val time only.
    """
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
    srw_composition_mode, admission_den_power, heat_kernel_beta = (
        _validate_v4171_composition_settings(
            srw_composition_mode, admission_den_power,
            heat_kernel_beta,
            context="make_sharded_srw"))
    _validate_v4171_admission_den_grad_scale(
        admission_den_grad_scale, context="make_sharded_srw")
    _admission_den_power = jnp.float32(admission_den_power)
    _srw_composition_mode = srw_composition_mode
    _heat_kernel_beta = jnp.float32(heat_kernel_beta)

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
                       P('data', None, None),    # operator_query [B,S,d_route]
                       P('model', None),          # operator keys [N_local,d_route]
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
    def fused_gate_srw(x, operator_query, operator_keys_local, raw_tau,
                       read_vectors_local, write_vectors_local,
                       soft_gate_temperature, soft_gate_t_final,
                       soft_gate_boundary_power,
                       soft_gate_boundary_power_final,
                       execution_prune_eps):
        N_local = operator_keys_local.shape[0]
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, D = x.shape
        x_bf = x.astype(jnp.bfloat16)
        operator_keys_padded = jnp.pad(operator_keys_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_vectors_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_vectors_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        operator_query_unit_bf = _forward_unit_direction(
            operator_query.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        operator_key_directions_bf = _forward_unit_direction(
            operator_keys_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir_bf = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir_bf = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        z1 = jnp.zeros((B, S, 1))
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_pos_inf = jnp.float32(1.0e30)

        def operator_keys_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                operator_key_directions_bf, start, cs, axis=0)

        def operator_scores_from_keys(operator_keys):
            # Cosine between the d_route operator query and operator keys.
            rho = (operator_query_unit_bf @ operator_keys.T).astype(jnp.float32)
            rho_exposure = (
                jax.lax.stop_gradient(operator_query_unit_bf) @ operator_keys.T
            ).astype(jnp.float32)
            return rho, rho_exposure

        def operator_keys_rw_chunk(start):
            ec = operator_keys_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_dir_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_dir_bf, start, cs, axis=0)
            vc = jax.lax.dynamic_slice_in_dim(valid_padded, start, cs, axis=0)
            return ec, rc, wc, vc

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
                operator_keys, _, _, valid_chunk = operator_keys_rw_chunk(s)
                valid_bsn = valid_chunk[None, None, :]
                rho_raw, _ = operator_scores_from_keys(operator_keys)
                rho = jnp.where(valid_bsn, rho_raw, 0.0)
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

        # Load-balance over rho distribution is disabled in the fast path;
        # v4171 does not require a rho-statistics pass for regular train.
        rho_lb = jnp.float32(0.0)

        def edge_margin_stat_terms(rho):
            positive_selection_margin = jax.nn.relu(rho - tau_ref)
            return jnp.square(positive_selection_margin)

        def angular_compose_parts(rho, valid_mask):
            (selection_margin, admission_weight, angular_amplitude,
             execution_weight,
             active_mask) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps,
                srw_composition_mode=_srw_composition_mode,
                heat_kernel_beta=_heat_kernel_beta)
            strong_mask = selection_margin > _angular_strong_margin
            selection_margin = jnp.where(valid_mask, selection_margin, 0.0)
            admission_weight = jnp.where(
                valid_mask, admission_weight, 0.0)
            angular_amplitude = jnp.where(
                valid_mask, angular_amplitude, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            active_mask = active_mask & valid_mask
            strong_mask = strong_mask & valid_mask
            return (
                selection_margin,
                admission_weight,
                angular_amplitude,
                execution_weight,
                active_mask,
                strong_mask,
            )

        def update_select_diag(carry, rho, selection_margin,
                               angular_amplitude,
                               valid_mask):
            (total_selected, total_selection_margin_sum,
             total_positive_margin_sum, total_positive_margin_max,
             total_rho_max, total_selection_margin_max) = carry
            selected = ((selection_margin > 0.0) & valid_mask).astype(jnp.float32)
            selection_margin_sum = jnp.where(
                valid_mask, selection_margin, 0.0).sum()
            selection_margin_max = jnp.where(
                valid_mask, selection_margin, diag_neg_inf)
            positive_margin_valid = jnp.where(
                valid_mask, angular_amplitude, 0.0)
            return (
                total_selected + selected.sum(axis=-1, keepdims=True),
                total_selection_margin_sum + selection_margin_sum,
                total_positive_margin_sum + positive_margin_valid.sum(),
                jnp.maximum(total_positive_margin_max,
                            positive_margin_valid.max()),
                jnp.maximum(total_rho_max, rho.max()),
                jnp.maximum(total_selection_margin_max,
                            selection_margin_max.max(axis=-1, keepdims=True)),
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

        def gate_sparsity_parts(selection_margin, admission, execution_weight,
                                valid_mask):
            margin_sg = jax.lax.stop_gradient(selection_margin)
            admission_sg = jax.lax.stop_gradient(admission)
            execution_sg = jax.lax.stop_gradient(execution_weight)
            active_tau = (margin_sg > 0.0) & valid_mask

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

            projected_gate = jnp.where(valid_mask, execution_sg, 0.0)
            projected_active = (
                (projected_gate[..., None] > projected_eps)
                & valid_mask[..., None])
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
                    (((margin_sg >= -0.03) & (margin_sg <= 0.0) & valid_mask)
                     .astype(jnp.float32).sum()),
                    ((margin_sg < -0.10) & valid_mask).astype(jnp.float32).sum(),
                )).astype(jnp.float32)
            else:
                margin_bands = jnp.stack((
                    active_tau.astype(jnp.float32).sum(),
                    (((margin_sg >= -0.01) & (margin_sg <= 0.0) & valid_mask)
                     .astype(jnp.float32).sum()),
                    (((margin_sg >= -0.03) & (margin_sg < -0.01) & valid_mask)
                     .astype(jnp.float32).sum()),
                    (((margin_sg >= -0.10) & (margin_sg < -0.03) & valid_mask)
                     .astype(jnp.float32).sum()),
                    ((margin_sg < -0.10) & valid_mask).astype(jnp.float32).sum(),
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

        def soft_gate_exposure_parts(gate_unpruned, valid_chunk):
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
            local_soft_exposure = jnp.where(
                valid_chunk, local_soft_exposure, 0.0)
            soft_exposure = jax.lax.all_gather(
                local_soft_exposure, 'data', axis=0).max(axis=0)
            valid_exposure = valid_chunk.astype(jnp.bool_)
            soft_dead_1e6 = (
                (soft_exposure <= jnp.float32(1.0e-6)) & valid_exposure)
            soft_dead_1e5 = (
                (soft_exposure <= jnp.float32(1.0e-5)) & valid_exposure)
            soft_dead_1e4 = (
                (soft_exposure <= jnp.float32(1.0e-4)) & valid_exposure)
            soft_exposure_sum = jnp.where(
                valid_exposure, soft_exposure, 0.0).sum()
            soft_exposure_min = jnp.where(
                valid_exposure, soft_exposure, diag_pos_inf).min()
            soft_exposure_max = jnp.where(
                valid_exposure, soft_exposure, 0.0).max()
            return (
                jnp.float32(0.0),
                soft_dead_1e6.astype(jnp.float32).sum(),
                soft_exposure_sum,
                soft_exposure_min,
                soft_exposure_max,
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
                operator_keys, rc, wc, valid_chunk = operator_keys_rw_chunk(s)
                valid_bsn = valid_chunk[None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_scores_from_keys(operator_keys)
                rho = jnp.where(valid_bsn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsn, rho_raw, tau)
                (selection_margin, admission_weight, angular_amplitude,
                 execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin,
                    angular_amplitude,
                    valid_bsn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    angular_amplitude.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission_weight, execution_weight,
                        valid_bsn)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_amplitude.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission_weight.sum(
                    axis=-1, keepdims=True)
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
                    soft_gate_exposure_parts(admission_weight, valid_chunk))
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
                operator_keys, rc, wc, valid_chunk = operator_keys_rw_chunk(s)
                valid_bsn = valid_chunk[None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_scores_from_keys(operator_keys)
                rho = jnp.where(valid_bsn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsn, rho_raw, tau)
                (selection_margin, admission_weight, angular_amplitude,
                 execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin,
                    angular_amplitude,
                    valid_bsn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    angular_amplitude.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission_weight, execution_weight,
                        valid_bsn)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_amplitude.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission_weight.sum(
                    axis=-1, keepdims=True)
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission_weight, valid_chunk))
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
        composition_den = _composition_den(
            global_den_cost, _admission_den_power,
            _srw_composition_mode)
        out = raw_out / composition_den
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
        tau_direct = tau
        # Measurement path: the copies below are detached for diagnostics only.
        # The forward denominator above intentionally remains live-gradient,
        # and numerator paths through execution_weight also remain live.
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
        tau_abs_mean = jnp.abs(jax.lax.stop_gradient(tau)).mean()
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

    return _mark_v4171_srw_factory_output(
        fused_gate_srw, admission_den_power, _srw_composition_mode,
        heat_kernel_beta)


def make_sharded_srw_paired(mesh, max_chunk_size=2048,
                            analysis=False,
                            dead_exposure_target=0.1,
                            soft_gate_effective_active_eps=1.0e-6,
                            admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                            admission_den_grad_scale=1.0,
                            srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
                            heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Fused attention_q+attention_k shard_map: two routes sharing same pool in one shard_map call.

    operator_query is [B,S,2,d_route] (q_operator_query, k_operator_query stacked on axis=2).
    raw_tau is [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].

    v4171 canonical execution uses the same statically selected composition
    kernel and unpruned-mass denominator as the single-route factory. attention_q and attention_k
    accumulate and normalize their kernel masses independently.
    analysis: see make_sharded_srw docstring.
    """
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
    srw_composition_mode, admission_den_power, heat_kernel_beta = (
        _validate_v4171_composition_settings(
            srw_composition_mode, admission_den_power,
            heat_kernel_beta,
            context="make_sharded_srw_paired"))
    _validate_v4171_admission_den_grad_scale(
        admission_den_grad_scale, context="make_sharded_srw_paired")
    _admission_den_power = jnp.float32(admission_den_power)
    _srw_composition_mode = srw_composition_mode
    _heat_kernel_beta = jnp.float32(heat_kernel_beta)

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
                       P('data', None, None, None),  # operator_query [B,S,2,d_route]
                       P('model', None),              # operator keys [N_local,d_route]
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
    def fused_gate_srw_paired(x, operator_query, operator_keys_local, raw_tau,
                              read_vectors_local, write_vectors_local,
                              soft_gate_temperature, soft_gate_t_final,
                              soft_gate_boundary_power,
                              soft_gate_boundary_power_final,
                              execution_prune_eps):
        N_local = operator_keys_local.shape[0]
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, D = x.shape
        # operator_query: [B,S,2,d_route], raw_tau: [B,S,2,1]
        x_bf = x.astype(jnp.bfloat16)
        operator_keys_padded = jnp.pad(operator_keys_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_vectors_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_vectors_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        operator_query_unit_bf = _forward_unit_direction(
            operator_query.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        operator_key_directions_bf = _forward_unit_direction(
            operator_keys_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir_bf = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir_bf = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_pos_inf = jnp.float32(1.0e30)

        def operator_keys_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                operator_key_directions_bf, start, cs, axis=0)

        def operator_scores_from_keys(operator_keys):
            # Cosine between the d_route operator query and operator keys.
            rho = jnp.einsum(
                'bsrd,nd->bsrn', operator_query_unit_bf, operator_keys).astype(jnp.float32)
            rho_exposure = jnp.einsum(
                'bsrd,nd->bsrn',
                jax.lax.stop_gradient(operator_query_unit_bf),
                operator_keys).astype(jnp.float32)
            return rho, rho_exposure

        def operator_keys_rw_chunk(start):
            ec = operator_keys_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_dir_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_dir_bf, start, cs, axis=0)
            vc = jax.lax.dynamic_slice_in_dim(valid_padded, start, cs, axis=0)
            return ec, rc, wc, vc

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
                operator_keys, _, _, valid_chunk = operator_keys_rw_chunk(s)
                valid_bsrn = valid_chunk[None, None, None, :]
                rho_raw, _ = operator_scores_from_keys(operator_keys)
                rho = jnp.where(valid_bsrn, rho_raw, 0.0)
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

        # Load-balance over rho distribution is disabled in the fast path;
        # v4171 does not require a rho-statistics pass for regular train.
        rho_lb = jnp.float32(0.0)

        def edge_margin_stat_terms(rho):
            positive_selection_margin = jax.nn.relu(rho - tau_ref)
            return jnp.square(positive_selection_margin)

        def angular_compose_parts(rho, valid_mask):
            (selection_margin, admission_weight, angular_amplitude,
             execution_weight,
             active_mask) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps,
                srw_composition_mode=_srw_composition_mode,
                heat_kernel_beta=_heat_kernel_beta)
            strong_mask = selection_margin > _angular_strong_margin
            selection_margin = jnp.where(valid_mask, selection_margin, 0.0)
            admission_weight = jnp.where(
                valid_mask, admission_weight, 0.0)
            angular_amplitude = jnp.where(
                valid_mask, angular_amplitude, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            active_mask = active_mask & valid_mask
            strong_mask = strong_mask & valid_mask
            return (
                selection_margin,
                admission_weight,
                angular_amplitude,
                execution_weight,
                active_mask,
                strong_mask,
            )

        def update_select_diag(carry, rho, selection_margin,
                               angular_amplitude,
                               valid_mask):
            (total_selected, total_selection_margin_sum,
             total_positive_margin_sum, total_positive_margin_max,
             total_rho_max, total_selection_margin_max) = carry
            selected = ((selection_margin > 0.0) & valid_mask).astype(jnp.float32)
            selection_margin_sum = jnp.where(
                valid_mask, selection_margin, 0.0).sum()
            selection_margin_max = jnp.where(
                valid_mask, selection_margin, diag_neg_inf)
            positive_margin_valid = jnp.where(
                valid_mask, angular_amplitude, 0.0)
            return (
                total_selected + selected.sum(axis=-1, keepdims=True),
                total_selection_margin_sum + selection_margin_sum,
                total_positive_margin_sum + positive_margin_valid.sum(),
                jnp.maximum(total_positive_margin_max,
                            positive_margin_valid.max()),
                jnp.maximum(total_rho_max, rho.max()),
                jnp.maximum(total_selection_margin_max,
                            selection_margin_max.max(axis=-1, keepdims=True)),
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

        def gate_sparsity_parts(selection_margin, admission, execution_weight,
                                valid_mask):
            margin_sg = jax.lax.stop_gradient(selection_margin)
            admission_sg = jax.lax.stop_gradient(admission)
            execution_sg = jax.lax.stop_gradient(execution_weight)
            active_tau = (margin_sg > 0.0) & valid_mask

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

            projected_gate = jnp.where(valid_mask, execution_sg, 0.0)
            projected_active = (
                (projected_gate[..., None] > projected_eps)
                & valid_mask[..., None])
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
                    (((margin_sg >= -0.03) & (margin_sg <= 0.0) & valid_mask)
                     .astype(jnp.float32).sum(axis=(0, 1, 3))),
                    (((margin_sg < -0.10) & valid_mask)
                     .astype(jnp.float32).sum(axis=(0, 1, 3))),
                ), axis=1).astype(jnp.float32)
            else:
                margin_bands = jnp.stack((
                    active_tau.astype(jnp.float32).sum(axis=(0, 1, 3)),
                    (((margin_sg >= -0.01) & (margin_sg <= 0.0) & valid_mask)
                     .astype(jnp.float32).sum(axis=(0, 1, 3))),
                    (((margin_sg >= -0.03) & (margin_sg < -0.01) & valid_mask)
                     .astype(jnp.float32).sum(axis=(0, 1, 3))),
                    (((margin_sg >= -0.10) & (margin_sg < -0.03) & valid_mask)
                     .astype(jnp.float32).sum(axis=(0, 1, 3))),
                    (((margin_sg < -0.10) & valid_mask)
                     .astype(jnp.float32).sum(axis=(0, 1, 3))),
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

        def soft_gate_exposure_parts(gate_unpruned, valid_chunk):
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
            local_soft_exposure = jnp.where(
                valid_chunk, local_soft_exposure, 0.0)
            soft_exposure = jax.lax.all_gather(
                local_soft_exposure, 'data', axis=0).max(axis=0)
            valid_exposure = valid_chunk.astype(jnp.bool_)
            soft_dead_1e6 = (
                (soft_exposure <= jnp.float32(1.0e-6)) & valid_exposure)
            soft_dead_1e5 = (
                (soft_exposure <= jnp.float32(1.0e-5)) & valid_exposure)
            soft_dead_1e4 = (
                (soft_exposure <= jnp.float32(1.0e-4)) & valid_exposure)
            soft_exposure_sum = jnp.where(
                valid_exposure, soft_exposure, 0.0).sum()
            soft_exposure_min = jnp.where(
                valid_exposure, soft_exposure, diag_pos_inf).min()
            soft_exposure_max = jnp.where(
                valid_exposure, soft_exposure, 0.0).max()
            return (
                jnp.float32(0.0),
                soft_dead_1e6.astype(jnp.float32).sum(),
                soft_exposure_sum,
                soft_exposure_min,
                soft_exposure_max,
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
                operator_keys, rc, wc, valid_chunk = operator_keys_rw_chunk(s)
                valid_bsrn = valid_chunk[None, None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_scores_from_keys(operator_keys)
                rho = jnp.where(valid_bsrn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsrn, rho_raw, tau)
                (selection_margin, admission_weight, angular_amplitude,
                 execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsrn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin,
                    angular_amplitude,
                    valid_bsrn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    angular_amplitude.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission_weight, execution_weight,
                        valid_bsrn)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_amplitude.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T  # [B,S,N]
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission_weight.sum(
                    axis=-1, keepdims=True)
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
                    soft_gate_exposure_parts(admission_weight, valid_chunk))
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
                operator_keys, rc, wc, valid_chunk = operator_keys_rw_chunk(s)
                valid_bsrn = valid_chunk[None, None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_scores_from_keys(operator_keys)
                rho = jnp.where(valid_bsrn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsrn, rho_raw, tau)
                (selection_margin, admission_weight, angular_amplitude,
                 execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsrn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin,
                    angular_amplitude,
                    valid_bsrn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    angular_amplitude.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission_weight, execution_weight,
                        valid_bsrn)
                else:
                    chunk_sparsity = sparsity_carry0
                chunk_int_max = angular_amplitude.max()
                chunk_int_cap_count = jnp.float32(0.0)
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = execution_weight * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = execution_weight.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(execution_weight).sum(
                    axis=-1, keepdims=True)
                chunk_den_cost = admission_weight.sum(
                    axis=-1, keepdims=True)
                chunk_active = active_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = strong_mask.astype(jnp.float32).sum(axis=-1, keepdims=True)
                (chunk_dead_penalty, chunk_dead_count,
                 chunk_exposure_sum, chunk_exposure_min,
                 chunk_exposure_max, chunk_weak_exposure_count,
                 chunk_soft_dead_1e4_count) = (
                    soft_gate_exposure_parts(admission_weight, valid_chunk))
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
        composition_den = _composition_den(
            global_den_cost, _admission_den_power,
            _srw_composition_mode)
        out = raw_out / composition_den
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
        tau_direct = tau
        # Measurement path: the copies below are detached for diagnostics only.
        # The forward denominator above intentionally remains live-gradient,
        # and numerator paths through execution_weight also remain live.
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
        tau_abs_mean = jnp.abs(jax.lax.stop_gradient(tau)).mean()
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

    return _mark_v4171_srw_factory_output(
        fused_gate_srw_paired, admission_den_power, _srw_composition_mode,
        heat_kernel_beta)


_V4171_MINIMAL_KERNEL_BUNDLES = {}


def _v4171_minimal_bundle_key(
        route_kind, mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode, heat_kernel_beta,
        trajectory_capture_width=0):
    return (
        str(route_kind), id(mesh), int(max_chunk_size),
        float(dead_exposure_target), float(soft_gate_effective_active_eps),
        float(admission_den_power), float(admission_den_grad_scale),
        str(srw_composition_mode), float(heat_kernel_beta),
        int(trajectory_capture_width),
    )


def _cached_v4171_minimal_bundle(route_kind, builder, *factory_args):
    key = _v4171_minimal_bundle_key(route_kind, *factory_args)
    bundle = _V4171_MINIMAL_KERNEL_BUNDLES.get(key)
    if bundle is None:
        bundle = builder(*factory_args)
        _V4171_MINIMAL_KERNEL_BUNDLES[key] = bundle
    return bundle


def _make_sharded_srw_minimal_impl(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA,
        trajectory_capture_width=0):
    """Build production, suppression, and exact-trajectory wrappers."""
    srw_composition_mode, admission_den_power, heat_kernel_beta = (
        _validate_v4171_composition_settings(
            srw_composition_mode, admission_den_power,
            heat_kernel_beta,
            context="make_sharded_srw_minimal"))
    _validate_v4171_admission_den_grad_scale(
        admission_den_grad_scale, context="make_sharded_srw_minimal")
    _admission_den_power = jnp.float32(admission_den_power)
    _srw_composition_mode = srw_composition_mode
    _heat_kernel_beta = jnp.float32(heat_kernel_beta)
    _composition_floor_mass = jnp.float32(
        _composition_den_floor_mass(_srw_composition_mode))
    del dead_exposure_target
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _trajectory_capture_width = int(trajectory_capture_width)
    if _trajectory_capture_width < 0:
        raise ValueError("trajectory_capture_width must be >= 0")

    def _sharded_srw_minimal_core(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, retain_mask_local,
            position_mask, retention_mode, *, capture_trajectory=False,
            trajectory_positions=None, trajectory_position_valid=None,
            trajectory_selected_ids=None, trajectory_selected_valid=None):
        del soft_gate_t_final, soft_gate_boundary_power_final
        N_local = operator_keys_local.shape[0]
        selected_global_operator_id = jnp.asarray(
            selected_global_operator_id, dtype=jnp.int32)
        if selected_global_operator_id.ndim == 0:
            selected_global_operator_id = jnp.broadcast_to(
                selected_global_operator_id, (x.shape[0],))
        selected_global_operator_id_sorted = (
            jnp.sort(selected_global_operator_id, axis=-1)
            if selected_global_operator_id.ndim == 2
            else selected_global_operator_id)
        target_positions = jnp.asarray(target_positions, dtype=jnp.int32)
        if target_positions.ndim == 0:
            target_positions = jnp.broadcast_to(
                target_positions, (x.shape[0],))
        if capture_trajectory:
            trajectory_positions = jnp.asarray(
                trajectory_positions, dtype=jnp.int32)
            trajectory_position_valid = jnp.asarray(
                trajectory_position_valid, dtype=jnp.bool_)
            trajectory_selected_ids = jnp.asarray(
                trajectory_selected_ids, dtype=jnp.int32)
            trajectory_selected_valid = jnp.asarray(
                trajectory_selected_valid, dtype=jnp.bool_)
            if _trajectory_capture_width <= 0:
                raise ValueError(
                    "trajectory wrapper requires a positive capture width")
        retain_mask_local = jnp.asarray(retain_mask_local, dtype=jnp.bool_)
        position_mask = jnp.asarray(position_mask, dtype=jnp.bool_)
        retention_mode = jnp.asarray(retention_mode, dtype=jnp.int32)
        global_start = (
            jax.lax.axis_index('model').astype(jnp.int32)
            * jnp.int32(N_local))
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, D = x.shape
        retain_padded = jnp.pad(
            retain_mask_local, ((0, pad_n),), constant_values=False)
        x_bf = x.astype(jnp.bfloat16)
        operator_keys_padded = jnp.pad(operator_keys_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_vectors_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_vectors_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        operator_query_unit_bf = _forward_unit_direction(
            operator_query.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        operator_key_directions_bf = _forward_unit_direction(
            operator_keys_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir_bf = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir_bf = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        tau = _tau_from_param(raw_tau)

        def operator_keys_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                operator_key_directions_bf, start, cs, axis=0)

        def operator_scores_from_keys(operator_keys):
            return (operator_query_unit_bf @ operator_keys.T).astype(jnp.float32)

        def operator_keys_rw_chunk(start):
            ec = operator_keys_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_dir_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_dir_bf, start, cs, axis=0)
            vc = jax.lax.dynamic_slice_in_dim(valid_padded, start, cs, axis=0)
            return ec, rc, wc, vc

        def angular_compose_parts(rho, valid_mask):
            (margin, admission, angular_amplitude, execution_weight,
             _) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps,
                srw_composition_mode=_srw_composition_mode,
                heat_kernel_beta=_heat_kernel_beta)
            admission = jnp.where(valid_mask, admission, 0.0)
            angular_amplitude = jnp.where(
                valid_mask, angular_amplitude, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            active_count = (
                ((margin > jnp.float32(0.0)) & valid_mask)
                .astype(jnp.float32)
                .sum(axis=-1, keepdims=True))
            return (margin, admission, angular_amplitude, execution_weight,
                    active_count)

        @jax.checkpoint
        def gate_srw_step(carry, i):
            (raw_out, total_gate_mass, total_gate_sq,
             total_gate_max, total_active_count,
             total_angular_amplitude, selected_raw_out) = carry[:7]
            if capture_trajectory:
                (trace_score, trace_ids, trace_read_bits,
                 trace_amplitude_bits, trace_execution_weight,
                 trace_admission, trace_margin, trace_rho,
                 trace_numerator_count, trace_denominator_count,
                 trace_selected_raw) = carry[7:]
            s = i * cs
            operator_keys, rc, wc, valid_chunk = operator_keys_rw_chunk(s)
            valid_bsn = valid_chunk[None, None, :]
            rho_raw = operator_scores_from_keys(operator_keys)
            rho_compute = jnp.where(valid_bsn, rho_raw, tau)
            (margin, admission, angular_amplitude, execution_weight,
             chunk_active_count) = angular_compose_parts(
                 rho_compute, valid_bsn)
            global_ids = global_start + s + jnp.arange(cs, dtype=jnp.int32)
            token_match = (
                jnp.arange(S, dtype=jnp.int32)[None, :, None]
                == target_positions[:, None, None])
            operator_match = _analysis_sorted_operator_membership(
                global_ids, selected_global_operator_id_sorted)
            route_match = jnp.bool_(True)
            suppress_mask = (
                jnp.asarray(apply_suppression, dtype=jnp.bool_)
                & token_match & operator_match & route_match & valid_bsn)
            retain_chunk = jax.lax.dynamic_slice_in_dim(
                retain_padded, s, cs, axis=0)[None, None, :]
            position_selected = position_mask[:, :, None]
            effective_keep = (
                ~position_selected | retain_chunk) & valid_bsn
            # A route with no removed local operator is production, exactly.
            # Disable retention before the numerator/denominator selects so
            # an all-ones mask cannot lower to a different masked reduction
            # on TPU.  Other model shards may still remove their own local
            # operators; preserving this shard is the correct global result.
            route_has_removal = ~jnp.all(retain_mask_local)
            retention_enabled = (
                (retention_mode > jnp.int32(0)) & route_has_removal)
            autonomous_retention = (
                (retention_mode == jnp.int32(2)) & route_has_removal)
            numerator_keep = (~retention_enabled) | effective_keep
            execution_for_numerator = jnp.where(
                suppress_mask | ~numerator_keep,
                jnp.float32(0.0), execution_weight)
            admission_for_den = jnp.where(
                autonomous_retention & ~effective_keep,
                jnp.float32(0.0), admission)
            xr = x_bf @ rc.T
            a = execution_for_numerator * xr.astype(jnp.float32)
            c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
            selected_execution = jnp.where(
                token_match & operator_match & valid_bsn,
                execution_weight, jnp.float32(0.0))
            selected_a = selected_execution * xr.astype(jnp.float32)
            selected_c_out = (
                selected_a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
            chunk_gate_mass = admission_for_den.sum(axis=-1, keepdims=True)
            chunk_gate_sq = jnp.square(admission_for_den).sum(
                axis=-1, keepdims=True)
            chunk_gate_max = admission_for_den.max(axis=-1, keepdims=True)
            next_carry = (
                raw_out + c_out,
                total_gate_mass + chunk_gate_mass,
                total_gate_sq + chunk_gate_sq,
                jnp.maximum(total_gate_max, chunk_gate_max),
                total_active_count + chunk_active_count,
                total_angular_amplitude + angular_amplitude.sum(
                    axis=-1, keepdims=True),
                selected_raw_out + selected_c_out)
            if capture_trajectory:
                trace_batch = jnp.arange(B, dtype=jnp.int32)[:, None]
                trace_position = jnp.clip(
                    trajectory_positions, 0, S - 1)

                def gather_target(value):
                    return value[trace_batch, trace_position, :]

                rho_target = gather_target(rho_compute)
                margin_target = gather_target(margin)
                admission_target = gather_target(admission)
                execution_target = gather_target(execution_weight)
                xr_target = gather_target(xr)
                amplitude_target = (
                    execution_target * xr_target.astype(jnp.float32))
                target_valid = trajectory_position_valid[:, :, None]
                valid_target = valid_chunk[None, None, :] & target_valid
                numerator_active = (
                    valid_target
                    & (execution_target != jnp.float32(0.0)))
                denominator_active = (
                    valid_target & (admission_target != jnp.float32(0.0)))
                chunk_score = jnp.where(
                    numerator_active, jnp.abs(amplitude_target),
                    -jnp.inf)
                chunk_ids = jnp.broadcast_to(
                    global_ids[None, None, :], chunk_score.shape)
                candidate_score = jnp.concatenate(
                    (trace_score, chunk_score), axis=-1)
                candidate_fields = tuple(jnp.concatenate(
                    (previous, current), axis=-1) for previous, current in (
                        (trace_ids, chunk_ids),
                        (trace_read_bits, jax.lax.bitcast_convert_type(
                            xr_target.astype(jnp.bfloat16), jnp.uint16)),
                        (trace_amplitude_bits, jax.lax.bitcast_convert_type(
                            amplitude_target.astype(jnp.bfloat16),
                            jnp.uint16)),
                        (trace_execution_weight, execution_target),
                        (trace_admission, admission_target),
                        (trace_margin, margin_target),
                        (trace_rho, rho_target),
                    ))
                trace_score, top_index = jax.lax.top_k(
                    candidate_score, _trajectory_capture_width)
                top_fields = tuple(jnp.take_along_axis(
                    field, top_index, axis=-1)
                    for field in candidate_fields)
                selected_ids = jnp.where(
                    trajectory_selected_valid, trajectory_selected_ids,
                    jnp.int32(N_local * jax.device_count() + 1))
                selected_ids = jnp.sort(selected_ids, axis=-1)
                selected_match = _analysis_sorted_operator_membership(
                    global_ids, selected_ids)
                selected_execution = jnp.where(
                    selected_match & valid_target, execution_target,
                    jnp.float32(0.0))
                selected_amplitude = (
                    selected_execution * xr_target.astype(jnp.float32))
                selected_chunk_raw = (
                    selected_amplitude.astype(jnp.bfloat16) @ wc
                ).astype(jnp.float32)
                next_carry += (
                    trace_score, *top_fields,
                    trace_numerator_count + numerator_active.astype(
                        jnp.int32).sum(axis=-1),
                    trace_denominator_count + denominator_active.astype(
                        jnp.int32).sum(axis=-1),
                    trace_selected_raw + selected_chunk_raw,
                )
            return next_carry, None

        initial_carry = (
            jnp.zeros((B, S, D), dtype=jnp.float32),
            jnp.zeros((B, S, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 1), dtype=jnp.float32),
            jnp.zeros((B, S, D), dtype=jnp.float32))
        if capture_trajectory:
            trace_shape = (
                B, trajectory_positions.shape[1],
                _trajectory_capture_width)
            initial_carry += (
                jnp.full(trace_shape, -jnp.inf, dtype=jnp.float32),
                jnp.full(trace_shape, -1, dtype=jnp.int32),
                jnp.zeros(trace_shape, dtype=jnp.uint16),
                jnp.zeros(trace_shape, dtype=jnp.uint16),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape[:2], dtype=jnp.int32),
                jnp.zeros(trace_shape[:2], dtype=jnp.int32),
                jnp.zeros(trace_shape[:2] + (D,), dtype=jnp.float32),
            )
        scan_carry, _ = jax.lax.scan(
            gate_srw_step, initial_carry, jnp.arange(nc))
        (raw_out, total_gate_mass, total_gate_sq,
         total_gate_max, total_active_count,
         total_angular_amplitude, selected_raw_out) = scan_carry[:7]

        global_gate_mass = jax.lax.psum(total_gate_mass, 'model')
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        global_gate_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_gate_max), 'model')
        global_active_count = jax.lax.stop_gradient(
            jax.lax.psum(total_active_count, 'model'))
        global_angular_amplitude = jax.lax.stop_gradient(
            jax.lax.psum(total_angular_amplitude, 'model'))
        global_operator_count = jax.lax.stop_gradient(
            jax.lax.psum(
                valid_padded.astype(jnp.float32).sum(), 'model'))
        gate_den = _composition_den(
            global_gate_mass, _admission_den_power,
            _srw_composition_mode)
        active_n_mean = global_active_count.mean()
        active_frac = active_n_mean / jnp.maximum(global_operator_count, 1.0)
        gate_mass_mean = global_gate_mass.mean()
        gate_den_mean = gate_den.mean()
        depth_active_mean = jnp.where(
            global_active_count > 0.0,
            global_angular_amplitude / jnp.maximum(global_active_count, 1.0),
            0.0).mean()
        gate_eff_n_mean = jnp.where(
            global_gate_mass > 0.0,
            jnp.square(global_gate_mass) / (global_gate_sq + 1.0e-8),
            0.0).mean()
        top1_gate_frac_mean = jnp.where(
            global_gate_mass > 0.0,
            global_gate_max / jnp.maximum(global_gate_mass, 1.0e-8),
            0.0).mean()
        den_floor_frac = (
            (global_gate_mass < _composition_floor_mass)
            if _srw_composition_mode == "linear_angular"
            else (global_gate_mass <= _composition_floor_mass)
        ).astype(jnp.float32).mean()
        tau_mean = jax.lax.stop_gradient(tau).mean()
        out = raw_out / gate_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')
        selected_out = jax.lax.psum(
            (selected_raw_out / gate_den).astype(jnp.bfloat16), 'model')
        selected_target = selected_out[
            jnp.arange(B, dtype=jnp.int32),
            target_positions, :]
        raw_out_global = jax.lax.psum(
            jax.lax.stop_gradient(raw_out).astype(jnp.bfloat16), 'model'
        ).astype(jnp.float32)
        admission_mass_max = global_gate_mass.max()
        composition_den_min = gate_den.min()
        composition_den_max = gate_den.max()
        raw_srw_out_norm = jnp.linalg.norm(
            raw_out_global, axis=-1).mean()
        normalized_srw_out_norm = jnp.linalg.norm(
            jax.lax.stop_gradient(out).astype(jnp.float32), axis=-1).mean()
        base_result = (
            out.astype(jnp.float32),
            jax.lax.stop_gradient(active_frac.astype(jnp.float32)),
            jax.lax.stop_gradient(active_n_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(gate_mass_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(gate_den_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(depth_active_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(gate_eff_n_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(top1_gate_frac_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(den_floor_frac.astype(jnp.float32)),
            jax.lax.stop_gradient(tau_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(admission_mass_max.astype(jnp.float32)),
            jax.lax.stop_gradient(composition_den_min.astype(jnp.float32)),
            jax.lax.stop_gradient(composition_den_max.astype(jnp.float32)),
            jax.lax.stop_gradient(raw_srw_out_norm.astype(jnp.float32)),
            jax.lax.stop_gradient(normalized_srw_out_norm.astype(jnp.float32)),
            jax.lax.stop_gradient(selected_target.astype(jnp.float32)),
        )
        if not capture_trajectory:
            return base_result

        (trace_score, trace_ids, trace_read_bits,
         trace_amplitude_bits, trace_execution_weight,
         trace_admission, trace_margin, trace_rho,
         trace_numerator_count, trace_denominator_count,
         trace_selected_raw) = scan_carry[7:]

        def gather_model_axis(value):
            return jax.lax.all_gather(
                value, 'model', axis=-1, tiled=True)

        gathered_score = gather_model_axis(trace_score)
        global_score, global_index = jax.lax.top_k(
            gathered_score, _trajectory_capture_width)
        gathered_fields = tuple(gather_model_axis(value) for value in (
            trace_ids, trace_read_bits, trace_amplitude_bits,
            trace_execution_weight, trace_admission, trace_margin,
            trace_rho))
        (global_ids, global_read_bits, global_amplitude_bits,
         global_execution_weight, global_admission, global_margin,
         global_rho) = tuple(jnp.take_along_axis(
             value, global_index, axis=-1) for value in gathered_fields)
        global_valid = (
            jnp.isfinite(global_score)
            & trajectory_position_valid[:, :, None])
        global_ids = jnp.where(global_valid, global_ids, jnp.int32(-1))
        global_numerator_count = jax.lax.psum(
            trace_numerator_count, 'model')
        global_denominator_count = jax.lax.psum(
            trace_denominator_count, 'model')
        trace_batch = jnp.arange(B, dtype=jnp.int32)[:, None]
        trace_position = jnp.clip(trajectory_positions, 0, S - 1)
        denominator_target = gate_den[
            trace_batch, trace_position, :]
        admission_mass_target = global_gate_mass[
            trace_batch, trace_position, :]
        production_target = out[
            trace_batch, trace_position, :].astype(jnp.float32)
        local_production_precast = raw_out[
            trace_batch, trace_position, :] / denominator_target
        production_precast_target = jax.lax.psum(
            local_production_precast, 'model')
        local_replay_precast = trace_selected_raw / denominator_target
        replay_precast_target = jax.lax.psum(
            local_replay_precast, 'model')
        replay_target = jax.lax.psum(
            local_replay_precast.astype(jnp.bfloat16), 'model'
        ).astype(jnp.float32)
        query_target = operator_query[
            trace_batch, trace_position, :].astype(jnp.float32)
        tau_target = tau[
            trace_batch, trace_position, :].astype(jnp.float32)
        valid_vector = trajectory_position_valid[:, :, None]
        trace_result = tuple(jax.lax.stop_gradient(value) for value in (
            jnp.where(valid_vector, production_target, 0.0),
            jnp.where(valid_vector, replay_target, 0.0),
            jnp.where(valid_vector, production_precast_target, 0.0),
            jnp.where(valid_vector, replay_precast_target, 0.0),
            jnp.where(valid_vector, query_target, 0.0),
            jnp.where(valid_vector, tau_target, 0.0),
            jnp.where(valid_vector, admission_mass_target, 0.0),
            jnp.where(valid_vector, denominator_target, 0.0),
            jnp.where(trajectory_position_valid, global_numerator_count, 0),
            jnp.where(trajectory_position_valid, global_denominator_count, 0),
            global_ids,
            global_valid,
            jnp.where(global_valid, global_read_bits, jnp.uint16(0)),
            jnp.where(global_valid, global_amplitude_bits, jnp.uint16(0)),
            jnp.where(global_valid, global_execution_weight, 0.0),
            jnp.where(global_valid, global_admission, 0.0),
            jnp.where(global_valid, global_margin, 0.0),
            jnp.where(global_valid, global_rho, 0.0),
            trajectory_position_valid,
        ))
        return base_result, trace_result

    common_in_specs = (
        P('data', None, None), P('data', None, None), P('model', None),
        P('data', None, None), P('model', None), P('model', None),
        P(), P(), P(), P(), P())
    out_specs = (
        P('data', None, None), P(), P(), P(), P(), P(), P(), P(), P(), P(),
        P(), P(), P(), P(), P(), P('data', None))
    @partial(
        shard_map, mesh=mesh,
        in_specs=common_in_specs + (
            P('data'), P('data'), P(), P('model'), P('data', None), P()),
        out_specs=out_specs, check_rep=False)
    def canonical_single_kernel(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, retain_mask_local,
            position_mask, retention_mode):
        return _sharded_srw_minimal_core(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, retain_mask_local,
            position_mask, retention_mode)

    trajectory_out_specs = (
        P('data', None, None), P('data', None, None),
        P('data', None, None), P('data', None, None),
        P('data', None, None), P('data', None, None),
        P('data', None, None), P('data', None, None),
        P('data', None), P('data', None),
        P('data', None, None), P('data', None, None),
        P('data', None, None), P('data', None, None),
        P('data', None, None), P('data', None, None),
        P('data', None, None), P('data', None, None),
        P('data', None),
    )

    @partial(
        shard_map, mesh=mesh,
        in_specs=common_in_specs + (
            P('data', None), P('data', None),
            P('data', None, None), P('data', None, None)),
        out_specs=(out_specs, trajectory_out_specs), check_rep=False)
    def canonical_single_trajectory_kernel(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, trajectory_positions,
            trajectory_position_valid, trajectory_selected_ids,
            trajectory_selected_valid):
        batch_size = x.shape[0]
        return _sharded_srw_minimal_core(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps,
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.bool_(False),
            jnp.ones((operator_keys_local.shape[0],), dtype=jnp.bool_),
            jnp.ones(x.shape[:2], dtype=jnp.bool_), jnp.int32(0),
            capture_trajectory=True,
            trajectory_positions=trajectory_positions,
            trajectory_position_valid=trajectory_position_valid,
            trajectory_selected_ids=trajectory_selected_ids,
            trajectory_selected_valid=trajectory_selected_valid)

    def fused_gate_srw_minimal(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        batch_size = x.shape[0]
        return canonical_single_kernel(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps,
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.bool_(False),
            jnp.ones((operator_keys_local.shape[0],), dtype=jnp.bool_),
            jnp.ones(x.shape[:2], dtype=jnp.bool_),
            jnp.int32(0))

    def fused_gate_srw_suppression_minimal(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, retain_mask_local=None,
            position_mask=None, retention_mode=0):
        if retain_mask_local is None:
            retain_mask_local = jnp.ones(
                (operator_keys_local.shape[0],), dtype=jnp.bool_)
        if position_mask is None:
            position_mask = jnp.ones(x.shape[:2], dtype=jnp.bool_)
        return canonical_single_kernel(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, retain_mask_local,
            position_mask, retention_mode)

    def fused_gate_srw_trajectory_minimal(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, trajectory_positions,
            trajectory_position_valid, trajectory_selected_ids,
            trajectory_selected_valid):
        return canonical_single_trajectory_kernel(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, trajectory_positions,
            trajectory_position_valid, trajectory_selected_ids,
            trajectory_selected_valid)

    factory_token = object()
    for wrapper in (
            fused_gate_srw_minimal,
            fused_gate_srw_suppression_minimal,
            fused_gate_srw_trajectory_minimal):
        _mark_v4171_srw_factory_output(
            wrapper, admission_den_power, _srw_composition_mode,
            heat_kernel_beta)
        wrapper._v4171_canonical_shard_map_kernel = canonical_single_kernel
        wrapper._v4171_canonical_factory_token = factory_token
    fused_gate_srw_trajectory_minimal._v4171_trajectory_shard_map_kernel = (
        canonical_single_trajectory_kernel)
    fused_gate_srw_trajectory_minimal._v4171_trajectory_capture_width = (
        _trajectory_capture_width)
    fused_gate_srw_minimal._v4171_suppression_wrapper = (
        fused_gate_srw_suppression_minimal)
    fused_gate_srw_suppression_minimal._v4171_production_wrapper = (
        fused_gate_srw_minimal)
    fused_gate_srw_trajectory_minimal._v4171_production_wrapper = (
        fused_gate_srw_minimal)
    return (fused_gate_srw_minimal, fused_gate_srw_suppression_minimal,
            fused_gate_srw_trajectory_minimal)


def make_sharded_srw_minimal(mesh, max_chunk_size=2048,
                             dead_exposure_target=0.1,
                             soft_gate_effective_active_eps=1.0e-6,
                             admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                             admission_den_grad_scale=1.0,
                             srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
                             heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create the production single-route minimal SRW kernel."""
    return _cached_v4171_minimal_bundle(
        "single", _make_sharded_srw_minimal_impl,
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta)[0]


def make_sharded_srw_suppression_minimal(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create the analysis-only exact execution-suppression kernel."""
    return _cached_v4171_minimal_bundle(
        "single", _make_sharded_srw_minimal_impl,
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta)[1]


def make_sharded_srw_trajectory_minimal(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA,
        trajectory_capture_width=1024):
    """Create the analysis-only exact active-operator trace kernel."""
    if int(trajectory_capture_width) <= 0:
        raise ValueError("trajectory_capture_width must be positive")
    return _cached_v4171_minimal_bundle(
        "single", _make_sharded_srw_minimal_impl,
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta, int(trajectory_capture_width))[2]


def _make_sharded_srw_paired_minimal_impl(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA,
        trajectory_capture_width=0):
    """Build production, suppression, and exact paired-trajectory wrappers."""
    srw_composition_mode, admission_den_power, heat_kernel_beta = (
        _validate_v4171_composition_settings(
            srw_composition_mode, admission_den_power,
            heat_kernel_beta,
            context="make_sharded_srw_paired_minimal"))
    _validate_v4171_admission_den_grad_scale(
        admission_den_grad_scale, context="make_sharded_srw_paired_minimal")
    _admission_den_power = jnp.float32(admission_den_power)
    _srw_composition_mode = srw_composition_mode
    _heat_kernel_beta = jnp.float32(heat_kernel_beta)
    _composition_floor_mass = jnp.float32(
        _composition_den_floor_mass(_srw_composition_mode))
    del dead_exposure_target
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _trajectory_capture_width = int(trajectory_capture_width)
    if _trajectory_capture_width < 0:
        raise ValueError("trajectory_capture_width must be >= 0")

    def _sharded_srw_paired_minimal_core(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, route_selector,
            retain_mask_local, position_mask, retention_mode, *,
            capture_trajectory=False, trajectory_positions=None,
            trajectory_position_valid=None, trajectory_selected_ids=None,
            trajectory_selected_valid=None):
        del soft_gate_t_final, soft_gate_boundary_power_final
        N_local = operator_keys_local.shape[0]
        selected_global_operator_id = jnp.asarray(
            selected_global_operator_id, dtype=jnp.int32)
        if selected_global_operator_id.ndim == 0:
            selected_global_operator_id = jnp.broadcast_to(
                selected_global_operator_id, (x.shape[0],))
        selected_global_operator_id_sorted = (
            jnp.sort(selected_global_operator_id, axis=-1)
            if selected_global_operator_id.ndim == 2
            else selected_global_operator_id)
        target_positions = jnp.asarray(target_positions, dtype=jnp.int32)
        if target_positions.ndim == 0:
            target_positions = jnp.broadcast_to(
                target_positions, (x.shape[0],))
        if capture_trajectory:
            trajectory_positions = jnp.asarray(
                trajectory_positions, dtype=jnp.int32)
            trajectory_position_valid = jnp.asarray(
                trajectory_position_valid, dtype=jnp.bool_)
            trajectory_selected_ids = jnp.asarray(
                trajectory_selected_ids, dtype=jnp.int32)
            trajectory_selected_valid = jnp.asarray(
                trajectory_selected_valid, dtype=jnp.bool_)
            if _trajectory_capture_width <= 0:
                raise ValueError(
                    "trajectory wrapper requires a positive capture width")
        retain_mask_local = jnp.asarray(retain_mask_local, dtype=jnp.bool_)
        position_mask = jnp.asarray(position_mask, dtype=jnp.bool_)
        retention_mode = jnp.asarray(retention_mode, dtype=jnp.int32)
        global_start = (
            jax.lax.axis_index('model').astype(jnp.int32)
            * jnp.int32(N_local))
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, _, _ = operator_query.shape
        D = x.shape[-1]
        x_bf = x.astype(jnp.bfloat16)
        operator_keys_padded = jnp.pad(operator_keys_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_vectors_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_vectors_local, ((0, pad_n), (0, 0)))
        retain_padded = jnp.pad(
            retain_mask_local, ((0, 0), (0, pad_n)),
            constant_values=False)
        valid_padded = jnp.arange(N_pad) < N_local
        operator_query_unit_bf = _forward_unit_direction(
            operator_query.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        operator_key_directions_bf = _forward_unit_direction(
            operator_keys_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir_bf = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir_bf = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        tau = _tau_from_param(raw_tau)

        def operator_keys_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                operator_key_directions_bf, start, cs, axis=0)

        def operator_scores_from_keys(operator_keys):
            return jnp.einsum(
                'bsrd,nd->bsrn', operator_query_unit_bf, operator_keys).astype(jnp.float32)

        def operator_keys_rw_chunk(start):
            ec = operator_keys_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_dir_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_dir_bf, start, cs, axis=0)
            vc = jax.lax.dynamic_slice_in_dim(valid_padded, start, cs, axis=0)
            return ec, rc, wc, vc

        def angular_compose_parts(rho, valid_mask):
            (margin, admission, angular_amplitude, execution_weight,
             _) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps,
                srw_composition_mode=_srw_composition_mode,
                heat_kernel_beta=_heat_kernel_beta)
            admission = jnp.where(valid_mask, admission, 0.0)
            angular_amplitude = jnp.where(
                valid_mask, angular_amplitude, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            active_count = (
                ((margin > jnp.float32(0.0)) & valid_mask)
                .astype(jnp.float32)
                .sum(axis=-1, keepdims=True))
            return (margin, admission, angular_amplitude, execution_weight,
                    active_count)

        @jax.checkpoint
        def gate_srw_step(carry, i):
            (raw_out, total_gate_mass, total_gate_sq,
             total_gate_max, total_active_count,
             total_angular_amplitude, selected_raw_out) = carry[:7]
            if capture_trajectory:
                (trace_score, trace_ids, trace_read_bits,
                 trace_amplitude_bits, trace_execution_weight,
                 trace_admission, trace_margin, trace_rho,
                 trace_numerator_count, trace_denominator_count,
                 trace_selected_raw) = carry[7:]
            s = i * cs
            operator_keys, rc, wc, valid_chunk = operator_keys_rw_chunk(s)
            valid_bsrn = valid_chunk[None, None, None, :]
            rho_raw = operator_scores_from_keys(operator_keys)
            rho_compute = jnp.where(valid_bsrn, rho_raw, tau)
            (margin, admission, angular_amplitude, execution_weight,
             chunk_active_count) = angular_compose_parts(
                rho_compute, valid_bsrn)
            global_ids = global_start + s + jnp.arange(cs, dtype=jnp.int32)
            token_match = (
                jnp.arange(S, dtype=jnp.int32)[None, :, None, None]
                == target_positions[:, None, None, None])
            if selected_global_operator_id_sorted.ndim == 3:
                operator_match = jnp.stack((
                    _analysis_sorted_operator_membership(
                        global_ids,
                        selected_global_operator_id_sorted[:, 0, :]),
                    _analysis_sorted_operator_membership(
                        global_ids,
                        selected_global_operator_id_sorted[:, 1, :]),
                ), axis=2)
                route_match = jnp.ones(
                    (1, 1, 2, 1), dtype=jnp.bool_)
            else:
                operator_match = _analysis_sorted_operator_membership(
                    global_ids,
                    selected_global_operator_id_sorted)[:, :, None, :]
                route_match = (
                    jnp.arange(2, dtype=jnp.int32)[None, None, :, None]
                    == jnp.asarray(route_selector, dtype=jnp.int32))
            suppress_mask = (
                jnp.asarray(apply_suppression, dtype=jnp.bool_)
                & token_match & operator_match & route_match & valid_bsrn)
            retain_chunk = jax.lax.dynamic_slice_in_dim(
                retain_padded, s, cs, axis=1)[None, None, :, :]
            position_selected = position_mask[:, :, None, None]
            effective_keep = (
                ~position_selected | retain_chunk) & valid_bsrn
            # Q and K have independent retention contracts even though they
            # share one paired kernel.  A fully retained sibling route must
            # remain the exact production route when the other is reduced.
            route_has_removal = (
                ~jnp.all(retain_mask_local, axis=-1))[None, None, :, None]
            retention_enabled = (
                (retention_mode > jnp.int32(0)) & route_has_removal)
            autonomous_retention = (
                (retention_mode == jnp.int32(2)) & route_has_removal)
            numerator_keep = (~retention_enabled) | effective_keep
            execution_for_numerator = jnp.where(
                suppress_mask | ~numerator_keep,
                jnp.float32(0.0), execution_weight)
            admission_for_den = jnp.where(
                autonomous_retention & ~effective_keep,
                jnp.float32(0.0), admission)
            xr = x_bf @ rc.T
            a = execution_for_numerator * xr.astype(jnp.float32)[:, :, None, :]
            c_out = jnp.einsum(
                'bsrn,nd->bsrd',
                a.astype(jnp.bfloat16),
                wc).astype(jnp.float32)
            selected_execution = jnp.where(
                token_match & operator_match & route_match & valid_bsrn,
                execution_weight, jnp.float32(0.0))
            selected_a = (
                selected_execution * xr.astype(jnp.float32)[:, :, None, :])
            selected_c_out = jnp.einsum(
                'bsrn,nd->bsrd', selected_a.astype(jnp.bfloat16),
                wc).astype(jnp.float32)
            chunk_gate_mass = admission_for_den.sum(axis=-1, keepdims=True)
            chunk_gate_sq = jnp.square(admission_for_den).sum(
                axis=-1, keepdims=True)
            chunk_gate_max = admission_for_den.max(axis=-1, keepdims=True)
            next_carry = (
                raw_out + c_out,
                total_gate_mass + chunk_gate_mass,
                total_gate_sq + chunk_gate_sq,
                jnp.maximum(total_gate_max, chunk_gate_max),
                total_active_count + chunk_active_count,
                total_angular_amplitude + angular_amplitude.sum(
                    axis=-1, keepdims=True),
                selected_raw_out + selected_c_out)
            if capture_trajectory:
                trace_batch = jnp.arange(B, dtype=jnp.int32)[:, None]
                trace_position = jnp.clip(
                    trajectory_positions, 0, S - 1)

                def gather_target(value):
                    return value[trace_batch, trace_position, ...]

                rho_target = gather_target(rho_compute)
                margin_target = gather_target(margin)
                admission_target = gather_target(admission)
                execution_target = gather_target(execution_weight)
                xr_target = gather_target(xr)[:, :, None, :]
                amplitude_target = (
                    execution_target * xr_target.astype(jnp.float32))
                target_valid = trajectory_position_valid[:, :, None, None]
                valid_target = (
                    valid_chunk[None, None, None, :] & target_valid)
                numerator_active = (
                    valid_target
                    & (execution_target != jnp.float32(0.0)))
                denominator_active = (
                    valid_target & (admission_target != jnp.float32(0.0)))
                chunk_score = jnp.where(
                    numerator_active, jnp.abs(amplitude_target), -jnp.inf)
                chunk_ids = jnp.broadcast_to(
                    global_ids[None, None, None, :], chunk_score.shape)
                candidate_score = jnp.concatenate(
                    (trace_score, chunk_score), axis=-1)
                candidate_fields = tuple(jnp.concatenate(
                    (previous, current), axis=-1) for previous, current in (
                        (trace_ids, chunk_ids),
                        (trace_read_bits, jnp.broadcast_to(
                            jax.lax.bitcast_convert_type(
                                xr_target.astype(jnp.bfloat16), jnp.uint16),
                            chunk_score.shape)),
                        (trace_amplitude_bits, jax.lax.bitcast_convert_type(
                            amplitude_target.astype(jnp.bfloat16),
                            jnp.uint16)),
                        (trace_execution_weight, execution_target),
                        (trace_admission, admission_target),
                        (trace_margin, margin_target),
                        (trace_rho, rho_target),
                    ))
                trace_score, top_index = jax.lax.top_k(
                    candidate_score, _trajectory_capture_width)
                top_fields = tuple(jnp.take_along_axis(
                    field, top_index, axis=-1)
                    for field in candidate_fields)
                selected_ids = jnp.where(
                    trajectory_selected_valid, trajectory_selected_ids,
                    jnp.int32(N_local * jax.device_count() + 1))
                selected_ids = jnp.sort(selected_ids, axis=-1)
                selected_match = jnp.stack((
                    _analysis_sorted_operator_membership(
                        global_ids, selected_ids[:, :, 0, :]),
                    _analysis_sorted_operator_membership(
                        global_ids, selected_ids[:, :, 1, :]),
                ), axis=2)
                selected_execution = jnp.where(
                    selected_match & valid_target, execution_target,
                    jnp.float32(0.0))
                selected_amplitude = (
                    selected_execution * xr_target.astype(jnp.float32))
                selected_chunk_raw = jnp.einsum(
                    'btrn,nd->btrd',
                    selected_amplitude.astype(jnp.bfloat16), wc
                ).astype(jnp.float32)
                next_carry += (
                    trace_score, *top_fields,
                    trace_numerator_count + numerator_active.astype(
                        jnp.int32).sum(axis=-1),
                    trace_denominator_count + denominator_active.astype(
                        jnp.int32).sum(axis=-1),
                    trace_selected_raw + selected_chunk_raw,
                )
            return next_carry, None

        initial_carry = (
            jnp.zeros((B, S, 2, D), dtype=jnp.float32),
            jnp.zeros((B, S, 2, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 2, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 2, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 2, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 2, 1), dtype=jnp.float32),
            jnp.zeros((B, S, 2, D), dtype=jnp.float32))
        if capture_trajectory:
            trace_shape = (
                B, trajectory_positions.shape[1], 2,
                _trajectory_capture_width)
            initial_carry += (
                jnp.full(trace_shape, -jnp.inf, dtype=jnp.float32),
                jnp.full(trace_shape, -1, dtype=jnp.int32),
                jnp.zeros(trace_shape, dtype=jnp.uint16),
                jnp.zeros(trace_shape, dtype=jnp.uint16),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape, dtype=jnp.float32),
                jnp.zeros(trace_shape[:3], dtype=jnp.int32),
                jnp.zeros(trace_shape[:3], dtype=jnp.int32),
                jnp.zeros(trace_shape[:3] + (D,), dtype=jnp.float32),
            )
        scan_carry, _ = jax.lax.scan(
            gate_srw_step, initial_carry, jnp.arange(nc))
        (raw_out, total_gate_mass, total_gate_sq,
         total_gate_max, total_active_count,
         total_angular_amplitude, selected_raw_out) = scan_carry[:7]

        global_gate_mass = jax.lax.psum(total_gate_mass, 'model')
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        global_gate_max = jax.lax.pmax(
            jax.lax.stop_gradient(total_gate_max), 'model')
        global_active_count = jax.lax.stop_gradient(
            jax.lax.psum(total_active_count, 'model'))
        global_angular_amplitude = jax.lax.stop_gradient(
            jax.lax.psum(total_angular_amplitude, 'model'))
        global_operator_count = jax.lax.stop_gradient(
            jax.lax.psum(
                valid_padded.astype(jnp.float32).sum(), 'model'))

        def route_mean(value, route_index):
            return value[:, :, route_index, :].mean()

        gate_den = _composition_den(
            global_gate_mass, _admission_den_power,
            _srw_composition_mode)
        depth_active = jnp.where(
            global_active_count > 0.0,
            global_angular_amplitude / jnp.maximum(global_active_count, 1.0),
            0.0)
        gate_eff_n = jnp.where(
            global_gate_mass > 0.0,
            jnp.square(global_gate_mass) / (global_gate_sq + 1.0e-8),
            0.0)
        top1_gate_frac = jnp.where(
            global_gate_mass > 0.0,
            global_gate_max / jnp.maximum(global_gate_mass, 1.0e-8),
            0.0)
        den_floor = (
            (global_gate_mass < _composition_floor_mass)
            if _srw_composition_mode == "linear_angular"
            else (global_gate_mass <= _composition_floor_mass)
        ).astype(jnp.float32)
        q_active_n_mean = route_mean(global_active_count, 0)
        k_active_n_mean = route_mean(global_active_count, 1)
        q_active_frac = q_active_n_mean / jnp.maximum(global_operator_count, 1.0)
        k_active_frac = k_active_n_mean / jnp.maximum(global_operator_count, 1.0)
        q_gate_mass_mean = route_mean(global_gate_mass, 0)
        k_gate_mass_mean = route_mean(global_gate_mass, 1)
        q_gate_den_mean = route_mean(gate_den, 0)
        k_gate_den_mean = route_mean(gate_den, 1)
        q_depth_active_mean = route_mean(depth_active, 0)
        k_depth_active_mean = route_mean(depth_active, 1)
        q_gate_eff_n_mean = route_mean(gate_eff_n, 0)
        k_gate_eff_n_mean = route_mean(gate_eff_n, 1)
        q_top1_gate_frac_mean = route_mean(top1_gate_frac, 0)
        k_top1_gate_frac_mean = route_mean(top1_gate_frac, 1)
        q_den_floor_frac = route_mean(den_floor, 0)
        k_den_floor_frac = route_mean(den_floor, 1)
        tau_sg = jax.lax.stop_gradient(tau)
        q_tau_mean = tau_sg[:, :, 0, :].mean()
        k_tau_mean = tau_sg[:, :, 1, :].mean()
        out = raw_out / gate_den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')
        selected_out = jax.lax.psum(
            (selected_raw_out / gate_den).astype(jnp.bfloat16), 'model')
        selected_target = selected_out[
            jnp.arange(B, dtype=jnp.int32),
            target_positions, :, :]
        raw_out_global = jax.lax.psum(
            jax.lax.stop_gradient(raw_out).astype(jnp.bfloat16), 'model'
        ).astype(jnp.float32)
        raw_norm_by_route = jnp.linalg.norm(
            raw_out_global, axis=-1).mean(axis=(0, 1))
        normalized_norm_by_route = jnp.linalg.norm(
            jax.lax.stop_gradient(out).astype(jnp.float32), axis=-1
        ).mean(axis=(0, 1))

        def route_max(value, route_index):
            return value[:, :, route_index, :].max()

        def route_min(value, route_index):
            return value[:, :, route_index, :].min()

        base_result = (
            out.astype(jnp.float32),
            jax.lax.stop_gradient(q_active_frac.astype(jnp.float32)),
            jax.lax.stop_gradient(k_active_frac.astype(jnp.float32)),
            jax.lax.stop_gradient(q_active_n_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_active_n_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(q_gate_mass_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_gate_mass_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(q_gate_den_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_gate_den_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(q_depth_active_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_depth_active_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(q_gate_eff_n_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_gate_eff_n_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(q_top1_gate_frac_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_top1_gate_frac_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(q_den_floor_frac.astype(jnp.float32)),
            jax.lax.stop_gradient(k_den_floor_frac.astype(jnp.float32)),
            jax.lax.stop_gradient(q_tau_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(k_tau_mean.astype(jnp.float32)),
            jax.lax.stop_gradient(route_max(global_gate_mass, 0)),
            jax.lax.stop_gradient(route_max(global_gate_mass, 1)),
            jax.lax.stop_gradient(route_min(gate_den, 0)),
            jax.lax.stop_gradient(route_min(gate_den, 1)),
            jax.lax.stop_gradient(route_max(gate_den, 0)),
            jax.lax.stop_gradient(route_max(gate_den, 1)),
            jax.lax.stop_gradient(raw_norm_by_route[0]),
            jax.lax.stop_gradient(raw_norm_by_route[1]),
            jax.lax.stop_gradient(normalized_norm_by_route[0]),
            jax.lax.stop_gradient(normalized_norm_by_route[1]),
            jax.lax.stop_gradient(selected_target.astype(jnp.float32)),
        )
        if not capture_trajectory:
            return base_result

        (trace_score, trace_ids, trace_read_bits,
         trace_amplitude_bits, trace_execution_weight,
         trace_admission, trace_margin, trace_rho,
         trace_numerator_count, trace_denominator_count,
         trace_selected_raw) = scan_carry[7:]

        def gather_model_axis(value):
            return jax.lax.all_gather(
                value, 'model', axis=-1, tiled=True)

        gathered_score = gather_model_axis(trace_score)
        global_score, global_index = jax.lax.top_k(
            gathered_score, _trajectory_capture_width)
        gathered_fields = tuple(gather_model_axis(value) for value in (
            trace_ids, trace_read_bits, trace_amplitude_bits,
            trace_execution_weight, trace_admission, trace_margin,
            trace_rho))
        (global_ids, global_read_bits, global_amplitude_bits,
         global_execution_weight, global_admission, global_margin,
         global_rho) = tuple(jnp.take_along_axis(
             value, global_index, axis=-1) for value in gathered_fields)
        global_valid = (
            jnp.isfinite(global_score)
            & trajectory_position_valid[:, :, None, None])
        global_ids = jnp.where(global_valid, global_ids, jnp.int32(-1))
        global_numerator_count = jax.lax.psum(
            trace_numerator_count, 'model')
        global_denominator_count = jax.lax.psum(
            trace_denominator_count, 'model')
        trace_batch = jnp.arange(B, dtype=jnp.int32)[:, None]
        trace_position = jnp.clip(trajectory_positions, 0, S - 1)
        denominator_target = gate_den[
            trace_batch, trace_position, :, :]
        admission_mass_target = global_gate_mass[
            trace_batch, trace_position, :, :]
        production_target = out[
            trace_batch, trace_position, :, :].astype(jnp.float32)
        local_production_precast = raw_out[
            trace_batch, trace_position, :, :] / denominator_target
        production_precast_target = jax.lax.psum(
            local_production_precast, 'model')
        local_replay_precast = trace_selected_raw / denominator_target
        replay_precast_target = jax.lax.psum(
            local_replay_precast, 'model')
        replay_target = jax.lax.psum(
            local_replay_precast.astype(jnp.bfloat16), 'model'
        ).astype(jnp.float32)
        query_target = operator_query[
            trace_batch, trace_position, :, :].astype(jnp.float32)
        tau_target = tau[
            trace_batch, trace_position, :, :].astype(jnp.float32)
        valid_vector = trajectory_position_valid[:, :, None, None]
        trace_result = tuple(jax.lax.stop_gradient(value) for value in (
            jnp.where(valid_vector, production_target, 0.0),
            jnp.where(valid_vector, replay_target, 0.0),
            jnp.where(valid_vector, production_precast_target, 0.0),
            jnp.where(valid_vector, replay_precast_target, 0.0),
            jnp.where(valid_vector, query_target, 0.0),
            jnp.where(valid_vector, tau_target, 0.0),
            jnp.where(valid_vector, admission_mass_target, 0.0),
            jnp.where(valid_vector, denominator_target, 0.0),
            jnp.where(
                trajectory_position_valid[:, :, None],
                global_numerator_count, 0),
            jnp.where(
                trajectory_position_valid[:, :, None],
                global_denominator_count, 0),
            global_ids,
            global_valid,
            jnp.where(global_valid, global_read_bits, jnp.uint16(0)),
            jnp.where(global_valid, global_amplitude_bits, jnp.uint16(0)),
            jnp.where(global_valid, global_execution_weight, 0.0),
            jnp.where(global_valid, global_admission, 0.0),
            jnp.where(global_valid, global_margin, 0.0),
            jnp.where(global_valid, global_rho, 0.0),
            trajectory_position_valid,
        ))
        return base_result, trace_result

    common_in_specs = (
        P('data', None, None), P('data', None, None, None), P('model', None),
        P('data', None, None, None), P('model', None), P('model', None),
        P(), P(), P(), P(), P())
    out_specs = (
        P('data', None, None, None), P(), P(), P(), P(), P(), P(), P(), P(),
        P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(), P(),
        P(), P(), P(), P(), P(), P(), P(), P('data', None, None))
    @partial(
        shard_map, mesh=mesh,
        in_specs=common_in_specs + (
            P('data'), P('data'), P(), P(), P(None, 'model'),
            P('data', None), P()),
        out_specs=out_specs, check_rep=False)
    def canonical_paired_kernel(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, route_selector,
            retain_mask_local, position_mask, retention_mode):
        return _sharded_srw_paired_minimal_core(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, route_selector,
            retain_mask_local, position_mask, retention_mode)

    trajectory_out_specs = (
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None), P('data', None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None, None, None), P('data', None, None, None),
        P('data', None),
    )

    @partial(
        shard_map, mesh=mesh,
        in_specs=common_in_specs + (
            P('data', None), P('data', None),
            P('data', None, None, None),
            P('data', None, None, None)),
        out_specs=(out_specs, trajectory_out_specs), check_rep=False)
    def canonical_paired_trajectory_kernel(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, trajectory_positions,
            trajectory_position_valid, trajectory_selected_ids,
            trajectory_selected_valid):
        batch_size = x.shape[0]
        return _sharded_srw_paired_minimal_core(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps,
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.bool_(False), jnp.int32(-1),
            jnp.ones(
                (2, operator_keys_local.shape[0]), dtype=jnp.bool_),
            jnp.ones(x.shape[:2], dtype=jnp.bool_), jnp.int32(0),
            capture_trajectory=True,
            trajectory_positions=trajectory_positions,
            trajectory_position_valid=trajectory_position_valid,
            trajectory_selected_ids=trajectory_selected_ids,
            trajectory_selected_valid=trajectory_selected_valid)

    def fused_gate_srw_paired_minimal(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        batch_size = x.shape[0]
        return canonical_paired_kernel(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps,
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.full((batch_size,), -1, dtype=jnp.int32),
            jnp.bool_(False), jnp.int32(-1),
            jnp.ones((2, operator_keys_local.shape[0]), dtype=jnp.bool_),
            jnp.ones(x.shape[:2], dtype=jnp.bool_), jnp.int32(0))

    def fused_gate_srw_paired_suppression_minimal(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, route_selector,
            retain_mask_local=None, position_mask=None, retention_mode=0):
        if retain_mask_local is None:
            retain_mask_local = jnp.ones(
                (2, operator_keys_local.shape[0]), dtype=jnp.bool_)
        if position_mask is None:
            position_mask = jnp.ones(x.shape[:2], dtype=jnp.bool_)
        return canonical_paired_kernel(
            x, operator_query, operator_keys_local, raw_tau, read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, selected_global_operator_id,
            target_positions, apply_suppression, route_selector,
            retain_mask_local, position_mask, retention_mode)

    def fused_gate_srw_paired_trajectory_minimal(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, trajectory_positions,
            trajectory_position_valid, trajectory_selected_ids,
            trajectory_selected_valid):
        return canonical_paired_trajectory_kernel(
            x, operator_query, operator_keys_local, raw_tau,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, trajectory_positions,
            trajectory_position_valid, trajectory_selected_ids,
            trajectory_selected_valid)

    factory_token = object()
    for wrapper in (
            fused_gate_srw_paired_minimal,
            fused_gate_srw_paired_suppression_minimal,
            fused_gate_srw_paired_trajectory_minimal):
        _mark_v4171_srw_factory_output(
            wrapper, admission_den_power, _srw_composition_mode,
            heat_kernel_beta)
        wrapper._v4171_canonical_shard_map_kernel = canonical_paired_kernel
        wrapper._v4171_canonical_factory_token = factory_token
    fused_gate_srw_paired_trajectory_minimal._v4171_trajectory_shard_map_kernel = (
        canonical_paired_trajectory_kernel)
    fused_gate_srw_paired_trajectory_minimal._v4171_trajectory_capture_width = (
        _trajectory_capture_width)
    fused_gate_srw_paired_minimal._v4171_suppression_wrapper = (
        fused_gate_srw_paired_suppression_minimal)
    fused_gate_srw_paired_suppression_minimal._v4171_production_wrapper = (
        fused_gate_srw_paired_minimal)
    fused_gate_srw_paired_trajectory_minimal._v4171_production_wrapper = (
        fused_gate_srw_paired_minimal)
    return (fused_gate_srw_paired_minimal,
            fused_gate_srw_paired_suppression_minimal,
            fused_gate_srw_paired_trajectory_minimal)


def make_sharded_srw_paired_minimal(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create the production paired attention_q/attention_k minimal SRW kernel."""
    return _cached_v4171_minimal_bundle(
        "paired", _make_sharded_srw_paired_minimal_impl,
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta)[0]


def make_sharded_srw_paired_suppression_minimal(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create the analysis-only route-selective attention_q/attention_k suppression kernel."""
    return _cached_v4171_minimal_bundle(
        "paired", _make_sharded_srw_paired_minimal_impl,
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta)[1]


def make_sharded_srw_paired_trajectory_minimal(
        mesh, max_chunk_size=2048, dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA,
        trajectory_capture_width=1024):
    """Create the analysis-only exact Q/K active-operator trace kernel."""
    if int(trajectory_capture_width) <= 0:
        raise ValueError("trajectory_capture_width must be positive")
    return _cached_v4171_minimal_bundle(
        "paired", _make_sharded_srw_paired_minimal_impl,
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta, int(trajectory_capture_width))[2]


# ================================================================
# 4. NeuronPool -- RW execution directions + static address parameterization
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    d_model: int
    d_route: int
    operator_key_mode: str = OPERATOR_KEY_MODE_LEARNED
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Checkpoint/config alias for rst pool size.

    def setup(self):
        dm = self.d_model
        d_route = int(self.d_route)
        operator_key_mode = _validate_operator_key_mode(
            self.operator_key_mode, context="NeuronPool")
        if d_route <= 0:
            raise ValueError(
                f"v4171 model.d_route must be > 0, got {d_route}")
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

        if operator_key_mode == OPERATOR_KEY_MODE_LEARNED:
            # Address parameters are independent from the RW execution vectors.
            # Their raw norms remain unconstrained; selection uses directions.
            self.attn_qk_op_key = self.param(
                'attn_qk_op_key', unit_norm_init(), (self.n_qk, d_route))
            self.attn_v_op_key = self.param(
                'attn_v_op_key', unit_norm_init(), (self.n_v, d_route))
            self.rst_operator_keys = self.param(
                'rst_op_key', unit_norm_init(), (n_rst_eff, d_route))
        else:
            probe_init = nn.initializers.orthogonal(scale=1.0)
            self.rw_key_read_probe = self.param(
                'rw_key_read_probe', probe_init, (dm, d_route))
            self.rw_key_write_probe = self.param(
                'rw_key_write_probe', probe_init, (dm, d_route))

        # No learned pool strength params; output strength is fixed by
        # sqrt(d_model / n_layers) in the forward path.


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
                "v417x requires explicit cosine-space tau_init_attn_qk/v/rst; "
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
                          admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                          execution_prune_eps=0.0,
                          analysis_selected_operator_id=None,
                          analysis_layer_index=0,
                          analysis_target_layer=-1,
                          analysis_target_positions=0,
                          analysis_target_route=-1,
                          analysis_intervention_enabled=False,
                          analysis_keep_qk=None,
                          analysis_keep_v=None,
                          analysis_position_mask=None,
                          analysis_retention_mode=0,
                          analysis_interchange_source=None,
                          analysis_interchange_enabled=False,
                          analysis_program_enabled=False,
                          analysis_program_ids_q=None,
                          analysis_program_ids_k=None,
                          analysis_program_ids_v=None,
                          analysis_program_target_positions=None,
                          analysis_program_mode=0,
                          analysis_program_source_q=None,
                          analysis_program_source_k=None,
                          analysis_program_source_v=None,
                          analysis_trajectory_enabled=False,
                          analysis_trajectory_positions=None,
                          analysis_trajectory_position_valid=None,
                          analysis_trajectory_ids_q=None,
                          analysis_trajectory_ids_k=None,
                          analysis_trajectory_ids_v=None,
                          analysis_trajectory_valid_q=None,
                          analysis_trajectory_valid_k=None,
                          analysis_trajectory_valid_v=None,
                          analysis_trajectory_replay_enabled=False,
                          analysis_trajectory_patch_schedule_enabled=False,
                          analysis_trajectory_patch_layers=None,
                          analysis_trajectory_patch_positions=None,
                          analysis_trajectory_patch_stages=None,
                          analysis_trajectory_patch_enabled=None,
                          analysis_trajectory_patch_values=None,
                          parity_debug=False):
    """Canonical shared v417x minimal attention path."""
    del n_qk, n_v
    admission_den_power = jnp.asarray(admission_den_power, dtype=jnp.float32)
    B, S, D = x.shape
    soft_gate_T_qk = (
        soft_gate_temperature if soft_gate_T_qk is None else soft_gate_T_qk)
    soft_gate_T_v = (
        soft_gate_temperature if soft_gate_T_v is None else soft_gate_T_v)
    pool_params = _ensure_pool_operator_keys(pool_params)

    qk_operator_keys = pool_params['attn_qk_op_key']
    qk_read = pool_params['attn_qk_read']
    qk_write = pool_params['attn_qk_write']
    v_operator_keys = pool_params['attn_v_op_key']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']
    if analysis_keep_qk is None:
        analysis_keep_qk = jnp.ones(
            (2, qk_operator_keys.shape[0]), dtype=jnp.bool_)
    if analysis_keep_v is None:
        analysis_keep_v = jnp.ones(
            (v_operator_keys.shape[0],), dtype=jnp.bool_)
    if analysis_position_mask is None:
        analysis_position_mask = jnp.ones((B, S), dtype=jnp.bool_)
    if analysis_interchange_source is None:
        analysis_interchange_source = jnp.zeros(
            (B, D), dtype=jnp.float32)
    if analysis_program_enabled:
        if any(value is None for value in (
                analysis_program_ids_q, analysis_program_ids_k,
                analysis_program_ids_v, analysis_program_target_positions,
                analysis_program_source_q, analysis_program_source_k,
                analysis_program_source_v)):
            raise ValueError("attention program schedule is incomplete")

    rng, rng_drop = jax.random.split(rng)
    attn_operator_queries = (
        x @ router_params['proj_attn']['kernel']
        + router_params['proj_attn']['bias'])
    attn_operator_queries = safe_dropout(
        attn_operator_queries, router_dropout, deterministic, rng_drop)
    q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)

    raw_tau_all = (
        x @ router_params['raw_tau_attn']['kernel']
        + router_params['raw_tau_attn']['bias'])
    qk_scale, v_scale, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    trajectory_paired = None
    trajectory_single_v = None
    if isinstance(sharded_fns, dict):
        fused_paired = sharded_fns.get(
            'attn_qk_paired_minimal',
            sharded_fns.get('attn_qk_paired', sharded_fns['paired']))
        fused_single_v = sharded_fns.get(
            'attn_v_single_minimal',
            sharded_fns.get('attn_v_single', sharded_fns['single']))
        trajectory_paired = sharded_fns.get(
            'attn_qk_paired_trajectory_minimal')
        trajectory_single_v = sharded_fns.get(
            'attn_v_single_trajectory_minimal')
    else:
        fused_single_v, fused_paired = sharded_fns
    canonical_paired = getattr(
        fused_paired, '_v4171_canonical_shard_map_kernel', None)
    canonical_single_v = getattr(
        fused_single_v, '_v4171_canonical_shard_map_kernel', None)
    if canonical_paired is None or canonical_single_v is None:
        raise ValueError(
            "minimal attention requires canonical v4171 shard-map kernels")
    if analysis_trajectory_enabled and (
            trajectory_paired is None or trajectory_single_v is None):
        raise ValueError(
            "trajectory analysis requires canonical QK and V trace kernels")

    qk_operator_queries = jnp.stack([q_operator_query, k_operator_query], axis=2)
    raw_tau_QK = jnp.stack(
        [raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2]], axis=2)
    apply_qk = (
        jnp.asarray(analysis_intervention_enabled, dtype=jnp.bool_)
        & (jnp.asarray(analysis_layer_index, dtype=jnp.int32)
           == jnp.asarray(analysis_target_layer, dtype=jnp.int32))
        & (jnp.asarray(analysis_target_route, dtype=jnp.int32) < 2))
    qk_selected_ids = analysis_selected_operator_id
    qk_selected_positions = analysis_target_positions
    qk_apply_suppression = apply_qk
    qk_selected_route = analysis_target_route
    if analysis_program_enabled:
        qk_program_width = max(
            analysis_program_ids_q.shape[-1],
            analysis_program_ids_k.shape[-1])
        program_ids_q = jnp.pad(
            analysis_program_ids_q,
            ((0, 0), (0, qk_program_width
                      - analysis_program_ids_q.shape[-1])),
            constant_values=qk_operator_keys.shape[0])
        program_ids_k = jnp.pad(
            analysis_program_ids_k,
            ((0, 0), (0, qk_program_width
                      - analysis_program_ids_k.shape[-1])),
            constant_values=qk_operator_keys.shape[0])
        qk_selected_ids = jnp.stack(
            (program_ids_q, program_ids_k), axis=1)
        qk_selected_positions = analysis_program_target_positions
        qk_apply_suppression = jnp.bool_(False)
        qk_selected_route = jnp.int32(-1)
    qk_trajectory_trace = None
    if analysis_trajectory_enabled:
        qk_trajectory_ids = jnp.stack(
            (analysis_trajectory_ids_q, analysis_trajectory_ids_k), axis=2)
        qk_trajectory_valid = jnp.stack(
            (analysis_trajectory_valid_q, analysis_trajectory_valid_k),
            axis=2)
        qk_result, qk_trajectory_trace = trajectory_paired(
            x, qk_operator_queries, qk_operator_keys, raw_tau_QK,
            qk_read, qk_write, soft_gate_T_qk, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, analysis_trajectory_positions,
            analysis_trajectory_position_valid, qk_trajectory_ids,
            qk_trajectory_valid)
    else:
        qk_result = canonical_paired(
            x, qk_operator_queries, qk_operator_keys, raw_tau_QK,
            qk_read, qk_write, soft_gate_T_qk, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, qk_selected_ids, qk_selected_positions,
            qk_apply_suppression, qk_selected_route, analysis_keep_qk,
            analysis_position_mask, analysis_retention_mode)
    (qk_state_transitions,
     q_active_frac, k_active_frac,
     q_active_n_mean, k_active_n_mean,
     q_gate_mass_mean, k_gate_mass_mean,
     q_gate_den_mean, k_gate_den_mean,
     q_depth_active_mean, k_depth_active_mean,
     q_gate_eff_n_mean, k_gate_eff_n_mean,
     q_top1_gate_frac_mean, k_top1_gate_frac_mean,
     q_den_floor_frac, k_den_floor_frac,
     q_tau_mean, k_tau_mean,
     q_admission_mass_max, k_admission_mass_max,
     q_composition_den_min, k_composition_den_min,
     q_composition_den_max, k_composition_den_max,
     q_raw_srw_out_norm, k_raw_srw_out_norm,
     q_normalized_srw_out_norm, k_normalized_srw_out_norm,
     qk_selected_target) = qk_result
    if analysis_trajectory_enabled:
        replayed_q = _analysis_replace_trajectory_positions(
            qk_state_transitions[:, :, 0, :],
            qk_trajectory_trace[1][:, :, 0, :],
            analysis_trajectory_positions,
            analysis_trajectory_position_valid)
        replayed_k = _analysis_replace_trajectory_positions(
            qk_state_transitions[:, :, 1, :],
            qk_trajectory_trace[1][:, :, 1, :],
            analysis_trajectory_positions,
            analysis_trajectory_position_valid)
        replayed_qk = jnp.stack((replayed_q, replayed_k), axis=2)
        qk_state_transitions = jnp.where(
            jnp.asarray(
                analysis_trajectory_replay_enabled, dtype=jnp.bool_),
            replayed_qk, qk_state_transitions)
    apply_v = (
        jnp.asarray(analysis_intervention_enabled, dtype=jnp.bool_)
        & (jnp.asarray(analysis_layer_index, dtype=jnp.int32)
           == jnp.asarray(analysis_target_layer, dtype=jnp.int32))
        & (jnp.asarray(analysis_target_route, dtype=jnp.int32) == 2))
    v_selected_ids = analysis_selected_operator_id
    v_selected_positions = analysis_target_positions
    v_apply_suppression = apply_v
    if analysis_program_enabled:
        v_selected_ids = analysis_program_ids_v
        v_selected_positions = analysis_program_target_positions
        v_apply_suppression = jnp.bool_(False)
    v_trajectory_trace = None
    if analysis_trajectory_enabled:
        v_result, v_trajectory_trace = trajectory_single_v(
            x, v_operator_query, v_operator_keys,
            raw_tau_all[:, :, 2:3], v_read, v_write,
            soft_gate_T_v, soft_gate_t_final, soft_gate_boundary_power,
            soft_gate_boundary_power_final, execution_prune_eps,
            analysis_trajectory_positions,
            analysis_trajectory_position_valid,
            analysis_trajectory_ids_v, analysis_trajectory_valid_v)
    else:
        v_result = canonical_single_v(
            x, v_operator_query, v_operator_keys,
            raw_tau_all[:, :, 2:3], v_read, v_write,
            soft_gate_T_v, soft_gate_t_final, soft_gate_boundary_power,
            soft_gate_boundary_power_final, execution_prune_eps,
            v_selected_ids, v_selected_positions, v_apply_suppression,
            analysis_keep_v, analysis_position_mask,
            analysis_retention_mode)
    (attention_v, v_active_frac, v_active_n_mean, v_gate_mass_mean, v_gate_den_mean,
     v_depth_active_mean, v_gate_eff_n_mean, v_top1_gate_frac_mean,
     v_den_floor_frac, v_tau_mean,
     v_admission_mass_max, v_composition_den_min,
     v_composition_den_max, v_raw_srw_out_norm,
     v_normalized_srw_out_norm, v_selected_target) = v_result
    if analysis_trajectory_enabled:
        replayed_v = _analysis_replace_trajectory_positions(
            attention_v, v_trajectory_trace[1],
            analysis_trajectory_positions,
            analysis_trajectory_position_valid)
        attention_v = jnp.where(
            jnp.asarray(
                analysis_trajectory_replay_enabled, dtype=jnp.bool_),
            replayed_v, attention_v)
    program_selected_q = jnp.zeros((B, D), dtype=jnp.float32)
    program_selected_k = jnp.zeros((B, D), dtype=jnp.float32)
    program_selected_v = jnp.zeros((B, D), dtype=jnp.float32)
    if analysis_program_enabled:
        program_selected_q = qk_selected_target[:, 0, :] * qk_scale
        program_selected_k = qk_selected_target[:, 1, :] * qk_scale
        program_selected_v = v_selected_target * v_scale

    capture_at_layer = (
        jnp.asarray(analysis_layer_index, dtype=jnp.int32)
        == jnp.asarray(analysis_target_layer, dtype=jnp.int32))
    target_route = jnp.asarray(analysis_target_route, dtype=jnp.int32)
    group_nonempty = _analysis_group_nonempty(
        analysis_selected_operator_id)
    effective_reference_enabled = (
        capture_at_layer
        & ~jnp.asarray(analysis_intervention_enabled, dtype=jnp.bool_)
        & jnp.any(group_nonempty))

    # Capture the effective contribution at the exact production precision:
    # production route minus the same route with only this family suppressed.
    # Computing the suppression counterfactual only at the requested analysis
    # layer avoids changing or slowing the disabled production path.
    def qk_suppression_reference(_):
        return canonical_paired(
            x, qk_operator_queries, qk_operator_keys, raw_tau_QK,
            qk_read, qk_write, soft_gate_T_qk, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, analysis_selected_operator_id,
            analysis_target_positions, jnp.bool_(True), target_route,
            analysis_keep_qk, analysis_position_mask,
            analysis_retention_mode)[0]

    qk_suppressed_transitions = jax.lax.cond(
        effective_reference_enabled & (target_route < jnp.int32(2)),
        qk_suppression_reference,
        lambda _: qk_state_transitions,
        operand=None)

    def v_suppression_reference(_):
        return canonical_single_v(
            x, v_operator_query, v_operator_keys,
            raw_tau_all[:, :, 2:3], v_read, v_write,
            soft_gate_T_v, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, analysis_selected_operator_id,
            analysis_target_positions, jnp.bool_(True), analysis_keep_v,
            analysis_position_mask, analysis_retention_mode)[0]

    v_suppressed_transition = jax.lax.cond(
        effective_reference_enabled & (target_route == jnp.int32(2)),
        v_suppression_reference,
        lambda _: attention_v,
        operand=None)

    attention_q = qk_state_transitions[:, :, 0, :] * qk_scale
    attention_k = qk_state_transitions[:, :, 1, :] * qk_scale
    attention_v = attention_v * v_scale
    suppressed_q = qk_suppressed_transitions[:, :, 0, :] * qk_scale
    suppressed_k = qk_suppressed_transitions[:, :, 1, :] * qk_scale
    suppressed_v = v_suppressed_transition * v_scale
    batch_indices = jnp.arange(B, dtype=jnp.int32)
    target_positions = jnp.asarray(
        analysis_target_positions, dtype=jnp.int32)
    effective_q_target = (
        attention_q[batch_indices, target_positions, :]
        - suppressed_q[batch_indices, target_positions, :])
    effective_k_target = (
        attention_k[batch_indices, target_positions, :]
        - suppressed_k[batch_indices, target_positions, :])
    effective_v_target = (
        attention_v[batch_indices, target_positions, :]
        - suppressed_v[batch_indices, target_positions, :])
    q_selected_target = jnp.where(
        effective_reference_enabled, effective_q_target,
        qk_selected_target[:, 0, :] * qk_scale)
    k_selected_target = jnp.where(
        effective_reference_enabled, effective_k_target,
        qk_selected_target[:, 1, :] * qk_scale)
    v_selected_target = jnp.where(
        effective_reference_enabled, effective_v_target,
        v_selected_target * v_scale)
    q_selected_target = jnp.where(
        capture_at_layer & (target_route == jnp.int32(0)),
        q_selected_target, jnp.zeros_like(q_selected_target))
    k_selected_target = jnp.where(
        capture_at_layer & (target_route == jnp.int32(1)),
        k_selected_target, jnp.zeros_like(k_selected_target))
    v_selected_target = jnp.where(
        capture_at_layer & (target_route == jnp.int32(2)),
        v_selected_target, jnp.zeros_like(v_selected_target))

    interchange_at_layer = (
        jnp.asarray(analysis_interchange_enabled, dtype=jnp.bool_)
        & (jnp.asarray(analysis_layer_index, dtype=jnp.int32)
           == jnp.asarray(analysis_target_layer, dtype=jnp.int32)))
    interchange_token_mask = (
        jnp.arange(S, dtype=jnp.int32)[None, :]
        == jnp.asarray(analysis_target_positions, dtype=jnp.int32)[:, None])

    def interchange_route(
            route_value, suppressed_value, selected_value, route_index):
        apply_route = (
            interchange_at_layer
            & (jnp.asarray(analysis_target_route, dtype=jnp.int32)
               == jnp.int32(route_index)))
        source_value = jnp.asarray(analysis_interchange_source)
        selected_value = selected_value.astype(jnp.float32)
        source_matches_base = jnp.all(
            source_value == selected_value, axis=-1)
        apply_mask = (
            apply_route
            & group_nonempty[:, None, None]
            & ~source_matches_base[:, None, None]
            & interchange_token_mask[:, :, None])
        # suppressed_value is the exact production implementation of
        # ``base - selected(base)``. Insert the captured source contribution
        # at this route-native site; zero source therefore equals suppression.
        patched = (
            suppressed_value.astype(jnp.float32)
            + source_value[:, None, :])
        return jnp.where(apply_mask, patched, route_value)

    attention_q = interchange_route(
        attention_q, suppressed_q, q_selected_target, 0)
    attention_k = interchange_route(
        attention_k, suppressed_k, k_selected_target, 1)
    attention_v = interchange_route(
        attention_v, suppressed_v, v_selected_target, 2)

    if analysis_program_enabled:
        program_mode = jnp.asarray(
            analysis_program_mode, dtype=jnp.int32)
        program_positions = jnp.asarray(
            analysis_program_target_positions, dtype=jnp.int32)
        program_position_mask = (
            jnp.arange(S, dtype=jnp.int32)[None, :]
            == program_positions[:, None])
        program_batch = jnp.arange(B, dtype=jnp.int32)

        def apply_program_route(
                production_value, selected_value, source_value):
            production_target = production_value[
                program_batch, program_positions, :]
            suppressed_target = (
                production_target - selected_value.astype(jnp.float32))
            source_value = jnp.asarray(source_value, dtype=jnp.float32)
            replay_mode = (
                (program_mode == jnp.int32(1))
                | (program_mode == jnp.int32(3)))
            patched_target = jnp.where(
                replay_mode, selected_value, production_target)
            patched_target = jnp.where(
                program_mode == jnp.int32(2),
                suppressed_target, patched_target)
            transplanted_target = suppressed_target + source_value
            source_matches_base = jnp.all(
                source_value == selected_value, axis=-1)
            transplanted_target = jnp.where(
                source_matches_base[:, None],
                production_target, transplanted_target)
            patched_target = jnp.where(
                program_mode == jnp.int32(4),
                transplanted_target, patched_target)
            return jnp.where(
                program_position_mask[:, :, None],
                patched_target[:, None, :], production_value)

        attention_q = apply_program_route(
            attention_q, program_selected_q,
            analysis_program_source_q)
        attention_k = apply_program_route(
            attention_k, program_selected_k,
            analysis_program_source_k)
        attention_v = apply_program_route(
            attention_v, program_selected_v,
            analysis_program_source_v)
    if analysis_trajectory_patch_schedule_enabled:
        patch_kwargs = {
            "layer_index": analysis_layer_index,
            "patch_layers": analysis_trajectory_patch_layers,
            "patch_positions": analysis_trajectory_patch_positions,
            "patch_stages": analysis_trajectory_patch_stages,
            "patch_enabled": analysis_trajectory_patch_enabled,
            "patch_values": analysis_trajectory_patch_values,
        }
        attention_q = _analysis_apply_trajectory_patch(
            attention_q, stage_code=TRAJECTORY_PATCH_STAGES["q"],
            **patch_kwargs)
        attention_k = _analysis_apply_trajectory_patch(
            attention_k, stage_code=TRAJECTORY_PATCH_STAGES["k"],
            **patch_kwargs)
        attention_v = _analysis_apply_trajectory_patch(
            attention_v, stage_code=TRAJECTORY_PATCH_STAGES["v"],
            **patch_kwargs)
    q_debug = attention_q
    k_debug = attention_k
    v_debug = attention_v

    d_head = d_model // n_heads
    attention_q = attention_q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    attention_k = attention_k.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    attention_v = attention_v.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    rng, rng_attn_drop = jax.random.split(rng)

    @jax.checkpoint
    def _attn_scores(attention_q, attention_k, attention_v, rng_drop):
        attn_scores = jnp.einsum('bhsd,bhtd->bhst', attention_q, attention_k) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        attn_scores = jnp.where(
            causal, attn_scores, jnp.finfo(attn_scores.dtype).min)
        attn_w = jax.nn.softmax(attn_scores, axis=-1)
        attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_drop)
        return jnp.einsum('bhst,bhtd->bhsd', attn_w, attention_v)

    out = _attn_scores(attention_q, attention_k, attention_v, rng_attn_drop)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)
    attn_out_norm = jnp.linalg.norm(out.astype(jnp.float32), axis=-1).mean()
    qk_admission_mass_max = jnp.maximum(
        q_admission_mass_max, k_admission_mass_max)
    qk_composition_den_min = jnp.minimum(
        q_composition_den_min, k_composition_den_min)
    qk_composition_den_max = jnp.maximum(
        q_composition_den_max, k_composition_den_max)
    qk_raw_srw_out_norm = jnp.float32(0.5) * (
        q_raw_srw_out_norm + k_raw_srw_out_norm)
    qk_normalized_srw_out_norm = jnp.float32(0.5) * (
        q_normalized_srw_out_norm + k_normalized_srw_out_norm)
    result = (
        out,
        q_active_frac,
        k_active_frac,
        v_active_frac,
        q_active_n_mean,
        k_active_n_mean,
        v_active_n_mean,
        q_gate_mass_mean,
        k_gate_mass_mean,
        v_gate_mass_mean,
        q_gate_den_mean,
        k_gate_den_mean,
        v_gate_den_mean,
        q_depth_active_mean,
        k_depth_active_mean,
        v_depth_active_mean,
        q_gate_eff_n_mean,
        k_gate_eff_n_mean,
        v_gate_eff_n_mean,
        q_top1_gate_frac_mean,
        k_top1_gate_frac_mean,
        v_top1_gate_frac_mean,
        q_den_floor_frac,
        k_den_floor_frac,
        v_den_floor_frac,
        q_tau_mean,
        k_tau_mean,
        v_tau_mean,
        jax.lax.stop_gradient(attn_out_norm.astype(jnp.float32)),
        qk_admission_mass_max,
        qk_composition_den_min,
        qk_composition_den_max,
        qk_raw_srw_out_norm,
        qk_normalized_srw_out_norm,
        qk_normalized_srw_out_norm * qk_scale,
        v_admission_mass_max,
        v_composition_den_min,
        v_composition_den_max,
        v_raw_srw_out_norm,
        v_normalized_srw_out_norm,
        v_normalized_srw_out_norm * v_scale,
        q_selected_target,
        k_selected_target,
        v_selected_target,
        program_selected_q,
        program_selected_k,
        program_selected_v,
    )
    if analysis_trajectory_enabled:
        result += ((
            qk_trajectory_trace,
            v_trajectory_trace,
            jax.lax.stop_gradient(qk_scale.astype(jnp.float32)),
            jax.lax.stop_gradient(v_scale.astype(jnp.float32)),
        ),)
    if parity_debug:
        return result + ((
            q_debug, k_debug, v_debug, out, x,
            q_operator_query, k_operator_query, v_operator_query,
            raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2],
            raw_tau_all[:, :, 2:3],
        ),)
    return result


def _rst_forward_minimal(x, pool_params, router_params, rng,
                         router_dropout, dropout_rate, deterministic,
                         sharded_fns,
                         d_model=None, n_layers=None,
                         soft_gate_temperature=0.07,
                         soft_gate_t_final=0.07,
                         soft_gate_T_rst=None,
                         soft_gate_boundary_power=2.0,
                         soft_gate_boundary_power_final=4.0,
                         admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                         execution_prune_eps=0.0,
                         analysis_selected_operator_id=None,
                         analysis_layer_index=0,
                         analysis_target_layer=-1,
                         analysis_target_positions=0,
                         analysis_target_route=-1,
                         analysis_intervention_enabled=False,
                         analysis_keep_rst=None,
                         analysis_position_mask=None,
                         analysis_retention_mode=0,
                         analysis_interchange_source=None,
                         analysis_interchange_enabled=False,
                         analysis_program_enabled=False,
                         analysis_program_ids_rst=None,
                         analysis_program_target_positions=None,
                         analysis_program_mode=0,
                         analysis_program_source_rst=None,
                         analysis_trajectory_enabled=False,
                         analysis_trajectory_positions=None,
                         analysis_trajectory_position_valid=None,
                         analysis_trajectory_ids_rst=None,
                         analysis_trajectory_valid_rst=None,
                         analysis_trajectory_replay_enabled=False,
                         analysis_trajectory_patch_schedule_enabled=False,
                         analysis_trajectory_patch_layers=None,
                         analysis_trajectory_patch_positions=None,
                         analysis_trajectory_patch_stages=None,
                         analysis_trajectory_patch_enabled=None,
                         analysis_trajectory_patch_values=None,
                         parity_debug=False):
    """Canonical shared v417x minimal RST path."""
    admission_den_power = jnp.asarray(admission_den_power, dtype=jnp.float32)
    if d_model is None or n_layers is None:
        raise ValueError(
            "depth-scaled pool outputs require d_model and n_layers.")
    soft_gate_T_rst = (
        soft_gate_temperature if soft_gate_T_rst is None else soft_gate_T_rst)
    pool_params = _ensure_pool_operator_keys(pool_params)
    rst_operator_keys = pool_params['rst_op_key']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']
    if analysis_keep_rst is None:
        analysis_keep_rst = jnp.ones(
            (rst_operator_keys.shape[0],), dtype=jnp.bool_)
    if analysis_position_mask is None:
        analysis_position_mask = jnp.ones(x.shape[:2], dtype=jnp.bool_)
    if analysis_interchange_source is None:
        analysis_interchange_source = jnp.zeros(
            (x.shape[0], x.shape[-1]), dtype=jnp.float32)
    if analysis_program_enabled and any(value is None for value in (
            analysis_program_ids_rst, analysis_program_target_positions,
            analysis_program_source_rst)):
        raise ValueError("RST program schedule is incomplete")

    rng, rng_drop = jax.random.split(rng)
    operator_query = (
        x @ router_params['proj_rst']['kernel']
        + router_params['proj_rst']['bias'])
    operator_query = safe_dropout(operator_query, router_dropout, deterministic, rng_drop)
    raw_tau = (
        x @ router_params['raw_tau_rst']['kernel']
        + router_params['raw_tau_rst']['bias'])
    _, _, rst_scale = _pool_output_scales(d_model, n_layers)
    trajectory_single = None
    if isinstance(sharded_fns, dict):
        fused_single = sharded_fns.get(
            'rst_single_minimal',
            sharded_fns.get('rst_single', sharded_fns['single']))
        trajectory_single = sharded_fns.get(
            'rst_single_trajectory_minimal')
    else:
        fused_single, _ = sharded_fns
    canonical_single = getattr(
        fused_single, '_v4171_canonical_shard_map_kernel', None)
    if canonical_single is None:
        raise ValueError(
            "minimal RST requires a canonical v4171 shard-map kernel")
    if analysis_trajectory_enabled and trajectory_single is None:
        raise ValueError(
            "trajectory analysis requires a canonical RST trace kernel")

    apply_rst = (
        jnp.asarray(analysis_intervention_enabled, dtype=jnp.bool_)
        & (jnp.asarray(analysis_layer_index, dtype=jnp.int32)
           == jnp.asarray(analysis_target_layer, dtype=jnp.int32))
        & (jnp.asarray(analysis_target_route, dtype=jnp.int32) == 3))
    rst_selected_ids = analysis_selected_operator_id
    rst_selected_positions = analysis_target_positions
    rst_apply_suppression = apply_rst
    if analysis_program_enabled:
        rst_selected_ids = analysis_program_ids_rst
        rst_selected_positions = analysis_program_target_positions
        rst_apply_suppression = jnp.bool_(False)
    rst_trajectory_trace = None
    if analysis_trajectory_enabled:
        rst_result, rst_trajectory_trace = trajectory_single(
            x, operator_query, rst_operator_keys, raw_tau,
            rst_read, rst_write, soft_gate_T_rst, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, analysis_trajectory_positions,
            analysis_trajectory_position_valid,
            analysis_trajectory_ids_rst, analysis_trajectory_valid_rst)
    else:
        rst_result = canonical_single(
            x, operator_query, rst_operator_keys, raw_tau,
            rst_read, rst_write, soft_gate_T_rst, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, rst_selected_ids,
            rst_selected_positions, rst_apply_suppression,
            analysis_keep_rst, analysis_position_mask,
            analysis_retention_mode)
    (out, rst_active_frac, rst_active_n_mean, rst_gate_mass_mean,
     rst_gate_den_mean, rst_depth_active_mean, rst_gate_eff_n_mean,
     rst_top1_gate_frac_mean, rst_den_floor_frac, rst_tau_mean,
     rst_admission_mass_max, rst_composition_den_min,
     rst_composition_den_max, rst_raw_srw_out_norm,
     rst_normalized_srw_out_norm, rst_selected_target) = rst_result
    if analysis_trajectory_enabled:
        replayed_rst = _analysis_replace_trajectory_positions(
            out, rst_trajectory_trace[1],
            analysis_trajectory_positions,
            analysis_trajectory_position_valid)
        out = jnp.where(
            jnp.asarray(
                analysis_trajectory_replay_enabled, dtype=jnp.bool_),
            replayed_rst, out)
    program_selected_rst = jnp.zeros(
        (x.shape[0], x.shape[-1]), dtype=jnp.float32)
    if analysis_program_enabled:
        program_selected_rst = rst_selected_target * rst_scale
    capture_at_layer = (
        jnp.asarray(analysis_layer_index, dtype=jnp.int32)
        == jnp.asarray(analysis_target_layer, dtype=jnp.int32))
    target_route = jnp.asarray(analysis_target_route, dtype=jnp.int32)
    group_nonempty = _analysis_group_nonempty(
        analysis_selected_operator_id)
    effective_reference_enabled = (
        capture_at_layer
        & ~jnp.asarray(analysis_intervention_enabled, dtype=jnp.bool_)
        & jnp.any(group_nonempty)
        & (target_route == jnp.int32(3)))

    def rst_suppression_reference(_):
        return canonical_single(
            x, operator_query, rst_operator_keys, raw_tau, rst_read,
            rst_write, soft_gate_T_rst, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, analysis_selected_operator_id,
            analysis_target_positions, jnp.bool_(True), analysis_keep_rst,
            analysis_position_mask, analysis_retention_mode)[0]

    suppressed_out = jax.lax.cond(
        effective_reference_enabled,
        rst_suppression_reference,
        lambda _: out,
        operand=None)
    out = out * rst_scale
    suppressed_out = suppressed_out * rst_scale
    batch_indices = jnp.arange(x.shape[0], dtype=jnp.int32)
    target_positions = jnp.asarray(
        analysis_target_positions, dtype=jnp.int32)
    effective_rst_target = (
        out[batch_indices, target_positions, :]
        - suppressed_out[batch_indices, target_positions, :])
    rst_selected_target = jnp.where(
        effective_reference_enabled,
        effective_rst_target,
        rst_selected_target * rst_scale)
    rst_selected_target = jnp.where(
        capture_at_layer & (target_route == jnp.int32(3)),
        rst_selected_target, jnp.zeros_like(rst_selected_target))
    apply_interchange = (
        jnp.asarray(analysis_interchange_enabled, dtype=jnp.bool_)
        & (jnp.asarray(analysis_layer_index, dtype=jnp.int32)
           == jnp.asarray(analysis_target_layer, dtype=jnp.int32))
        & (jnp.asarray(analysis_target_route, dtype=jnp.int32) == jnp.int32(3)))
    interchange_token_mask = (
        jnp.arange(x.shape[1], dtype=jnp.int32)[None, :]
        == jnp.asarray(analysis_target_positions, dtype=jnp.int32)[:, None])
    source_value = jnp.asarray(analysis_interchange_source)
    rst_selected_target = rst_selected_target.astype(jnp.float32)
    source_matches_base = jnp.all(
        source_value == rst_selected_target, axis=-1)
    apply_mask = (
        apply_interchange
        & group_nonempty[:, None, None]
        & ~source_matches_base[:, None, None]
        & interchange_token_mask[:, :, None])
    patched_out = (
        suppressed_out.astype(jnp.float32)
        + source_value[:, None, :])
    out = jnp.where(apply_mask, patched_out, out)
    if analysis_program_enabled:
        program_mode = jnp.asarray(
            analysis_program_mode, dtype=jnp.int32)
        program_positions = jnp.asarray(
            analysis_program_target_positions, dtype=jnp.int32)
        program_batch = jnp.arange(x.shape[0], dtype=jnp.int32)
        production_target = out[program_batch, program_positions, :]
        suppressed_target = production_target - program_selected_rst
        program_source = jnp.asarray(
            analysis_program_source_rst, dtype=jnp.float32)
        replay_mode = (
            (program_mode == jnp.int32(1))
            | (program_mode == jnp.int32(3)))
        patched_target = jnp.where(
            replay_mode, program_selected_rst, production_target)
        patched_target = jnp.where(
            program_mode == jnp.int32(2),
            suppressed_target, patched_target)
        transplanted_target = suppressed_target + program_source
        source_matches_base = jnp.all(
            program_source == program_selected_rst, axis=-1)
        transplanted_target = jnp.where(
            source_matches_base[:, None],
            production_target, transplanted_target)
        patched_target = jnp.where(
            program_mode == jnp.int32(4),
            transplanted_target, patched_target)
        program_position_mask = (
            jnp.arange(x.shape[1], dtype=jnp.int32)[None, :]
            == program_positions[:, None])
        out = jnp.where(
            program_position_mask[:, :, None],
            patched_target[:, None, :], out)
    if analysis_trajectory_patch_schedule_enabled:
        out = _analysis_apply_trajectory_patch(
            out, layer_index=analysis_layer_index,
            stage_code=TRAJECTORY_PATCH_STAGES["rst"],
            patch_layers=analysis_trajectory_patch_layers,
            patch_positions=analysis_trajectory_patch_positions,
            patch_stages=analysis_trajectory_patch_stages,
            patch_enabled=analysis_trajectory_patch_enabled,
            patch_values=analysis_trajectory_patch_values)
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)
    rst_out_norm = jnp.linalg.norm(out.astype(jnp.float32), axis=-1).mean()
    result = (
        out,
        rst_active_frac,
        rst_active_n_mean,
        rst_gate_mass_mean,
        rst_gate_den_mean,
        rst_depth_active_mean,
        rst_gate_eff_n_mean,
        rst_top1_gate_frac_mean,
        rst_den_floor_frac,
        rst_tau_mean,
        jax.lax.stop_gradient(rst_out_norm.astype(jnp.float32)),
        rst_admission_mass_max,
        rst_composition_den_min,
        rst_composition_den_max,
        rst_raw_srw_out_norm,
        rst_normalized_srw_out_norm,
        rst_normalized_srw_out_norm * rst_scale,
        rst_selected_target,
        program_selected_rst,
    )
    if analysis_trajectory_enabled:
        result += ((
            rst_trajectory_trace,
            jax.lax.stop_gradient(rst_scale.astype(jnp.float32)),
        ),)
    if parity_debug:
        return result + ((out, x, operator_query, raw_tau),)
    return result


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
                  admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                  execution_prune_eps=0.0):
    """Shared v417x sharded analysis path; canonical fused functions required.

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
    qk_operator_keys = pool_params['attn_qk_op_key']
    qk_read = pool_params['attn_qk_read']
    qk_write = pool_params['attn_qk_write']
    v_operator_keys = pool_params['attn_v_op_key']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']

    # Learned operator embeddings are passed into the sharded SRW closure.
    # The closure forward-normalizes them for selection stability.
    qk_operator_keys_unit = qk_operator_keys
    v_operator_keys_unit = v_operator_keys

    _qk_op_key_norms = jax.lax.stop_gradient(
        jnp.linalg.norm(qk_operator_keys, axis=-1))
    attn_qk_op_key_norm_mean = _qk_op_key_norms.mean()
    attn_qk_op_key_norm_min = _qk_op_key_norms.min()
    attn_qk_op_key_norm_std = _qk_op_key_norms.std()
    _v_op_key_norms = jax.lax.stop_gradient(
        jnp.linalg.norm(v_operator_keys, axis=-1))
    attn_v_op_key_norm_mean = _v_op_key_norms.mean()
    attn_v_op_key_norm_min = _v_op_key_norms.min()
    attn_v_op_key_norm_std = _v_op_key_norms.std()
    if analysis:
        attn_qk_op_key_norm_max = _qk_op_key_norms.max()
        attn_v_op_key_norm_max = _v_op_key_norms.max()

    rng, rng_drop = jax.random.split(rng)
    # Direct state-to-operation query projection.
    attn_operator_queries = (
        x @ router_params['proj_attn']['kernel']
        + router_params['proj_attn']['bias'])
    attn_operator_queries = safe_dropout(
        attn_operator_queries, router_dropout, deterministic, rng_drop)
    q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)

    raw_tau_all = (
        x @ router_params['raw_tau_attn']['kernel']
        + router_params['raw_tau_attn']['bias'])
    tau_all = _tau_from_param(raw_tau_all)
    if analysis:
        _tau_all_sg = jax.lax.stop_gradient(tau_all)
        attn_tau_std = _tau_all_sg.std(axis=(0, 1))  # [3] attention_q/attention_k/attention_v
        attn_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['raw_tau_attn']['kernel']) ** 2) + 1e-12)

    qk_scale, v_scale, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    if isinstance(sharded_fns, dict):
        fused_paired = sharded_fns.get('attn_qk_paired', sharded_fns.get('qk_paired', sharded_fns['paired']))
        fused_single_v = sharded_fns.get('attn_v_single', sharded_fns.get('v_single', sharded_fns['single']))
    else:
        fused_single_v, fused_paired = sharded_fns
    qk_operator_queries = jnp.stack([q_operator_query, k_operator_query], axis=2)
    raw_tau_QK = jnp.stack([raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2]], axis=2)
    qk_ret = fused_paired(x, qk_operator_queries, qk_operator_keys_unit, raw_tau_QK,
                           qk_read, qk_write,
                           soft_gate_T_qk, soft_gate_t_final,
                           soft_gate_boundary_power,
                           soft_gate_boundary_power_final,
                           execution_prune_eps)
    (qk_state_transitions, qk_active, qk_raw_gmax, qk_lb, qk_sstd, qk_es, qk_anm,
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
        qk_raw_norm = jnp.linalg.norm(qk_state_transitions, axis=-1).mean()
    qk_select_start = qk_offset
    qk_select_diag = qk_ret[qk_select_start:qk_select_start + SELECT_DIAG_COUNT]
    qk_exposure_start = qk_select_start + SELECT_DIAG_COUNT
    qk_exposure_diag = qk_ret[
        qk_exposure_start:qk_exposure_start + DEAD_EXPOSURE_DIAG_COUNT]
    attention_q = qk_state_transitions[:, :, 0, :] * qk_scale
    attention_k = qk_state_transitions[:, :, 1, :] * qk_scale
    v_ret = fused_single_v(x, v_operator_query, v_operator_keys_unit, raw_tau_all[:, :, 2:3],
                           v_read, v_write,
                           soft_gate_T_v, soft_gate_t_final,
                           soft_gate_boundary_power,
                           soft_gate_boundary_power_final,
                           execution_prune_eps)
    (attention_v, v_active, v_raw_gmax, v_lb, v_sstd, v_es, v_anm,
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
        v_raw_norm = jnp.linalg.norm(attention_v, axis=-1).mean()
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
    attention_v = attention_v * v_scale

    d_head = d_model // n_heads
    attention_q = attention_q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    attention_k = attention_k.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    attention_v = attention_v.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    rng, rng_attn_drop = jax.random.split(rng)
    @jax.checkpoint
    def _attn_scores(attention_q, attention_k, attention_v, rng_drop):
        attn_scores = jnp.einsum('bhsd,bhtd->bhst', attention_q, attention_k) / scale
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
        out_dbg = jnp.einsum('bhst,bhtd->bhsd', attn_w, attention_v)
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
        q_norms_dbg = jnp.linalg.norm(attention_q, axis=-1)
        k_norms_dbg = jnp.linalg.norm(attention_k, axis=-1)
        v_norms_dbg = jnp.linalg.norm(attention_v, axis=-1)
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
            attention_q, attention_k, attention_v, rng_attn_drop)
    else:
        out = _attn_scores(attention_q, attention_k, attention_v, rng_attn_drop)
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
    # attention_q/attention_k share the qk pool, while attention_v has its own pool.  Keep the historical
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
                k_sparsity_diag)
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
                  admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
                  execution_prune_eps=0.0):
    """Shared v417x sharded analysis path; canonical fused functions required.

    `analysis` see _attn_forward docstring.
    """
    B, S, D = x.shape
    soft_gate_T_rst = (
        soft_gate_temperature
        if soft_gate_T_rst is None else soft_gate_T_rst)
    pool_params = _ensure_pool_operator_keys(pool_params)
    rst_operator_keys = pool_params['rst_op_key']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']

    rng, rng_drop = jax.random.split(rng)
    operator_query = (
        x @ router_params['proj_rst']['kernel']
        + router_params['proj_rst']['bias'])
    operator_query = safe_dropout(operator_query, router_dropout, deterministic, rng_drop)

    # Learned operator embeddings are passed into the sharded SRW closure.
    # The closure forward-normalizes them for selection stability.
    rst_operator_keys_unit = rst_operator_keys
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
    rst_ret = fused_single(x, operator_query, rst_operator_keys_unit, raw_tau,
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
    out = out * rst_scale
    rst_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rst_out_norm_max = jnp.linalg.norm(out, axis=-1).max()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    _rst_op_key_norms = jax.lax.stop_gradient(
        jnp.linalg.norm(rst_operator_keys, axis=-1))
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
                rst_sparsity_diag)
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
    attention-v pools to construct attention_q/attention_k/attention_v, then applies causal self-attention
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

class DAWN_SRW_V4171(nn.Module):
    """DAWN-SRW v4.1.7.1 with learned operator-address embeddings."""
    __version__ = MODEL_VERSION

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False
    logical_vocab_size: Optional[int] = None
    vocab_size_padded: Optional[int] = None

    d_route: int = DEFAULT_D_ROUTE
    operator_key_mode: str = OPERATOR_KEY_MODE_LEARNED
    admission_den_power: float = DEFAULT_ADMISSION_DEN_POWER
    admission_den_power_qk: Optional[float] = None
    admission_den_power_v: Optional[float] = None
    admission_den_power_rst: Optional[float] = None
    srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE
    heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA
    n_qk: int = 1580
    n_v: int = 2600
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Checkpoint/config alias; n_rst is canonical.
    router_dropout: float = 0.1
    n_chunks_rst: Optional[int] = None
    n_chunks_know: int = 1    # Config alias; n_chunks_rst is canonical.
    n_chunks_qk: int = 1     # N-axis chunking for qk pool
    n_chunks_v: int = 1      # N-axis chunking for v pool
    # Constructor receives cosine-space tau values. The train driver may use
    # safe placeholders before one-time quantile calibration.
    tau_init_attn_qk: Optional[float] = None
    tau_init_attn_v: Optional[float] = None
    tau_init_rst: Optional[float] = None

    def _vocab_sizes(self):
        logical = (
            int(self.logical_vocab_size)
            if self.logical_vocab_size is not None
            else int(self.vocab_size)
        )
        embedding = (
            int(self.vocab_size_padded)
            if self.vocab_size_padded is not None
            else int(self.vocab_size)
        )
        if logical <= 0:
            raise ValueError(f"logical_vocab_size must be > 0, got {logical}")
        if embedding < logical:
            raise ValueError(
                f"embedding vocab size {embedding} is smaller than "
                f"logical_vocab_size={logical}"
            )
        return logical, embedding

    def setup(self):
        operator_key_mode = _validate_operator_key_mode(
            self.operator_key_mode, context="DAWN_SRW_V4171 constructor")
        legacy_power, _, _, _ = _resolve_v417x_admission_den_powers(
            self.admission_den_power,
            self.admission_den_power_qk,
            self.admission_den_power_v,
            self.admission_den_power_rst,
            context="DAWN_SRW_V4171 constructor")
        _validate_v4171_composition_settings(
            self.srw_composition_mode,
            legacy_power,
            self.heat_kernel_beta,
            context="DAWN_SRW_V4171 constructor")
        if int(self.d_route) <= 0:
            raise ValueError(
                f"v4171 model.d_route must be > 0, got {self.d_route}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})")
        _, embedding_vocab_size = self._vocab_sizes()
        self.token_emb = nn.Embed(
            embedding_vocab_size,
            self.d_model,
            embedding_init=scaled_normal(0.02),
        )
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model, embedding_init=scaled_normal(0.02))
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst_eff,
            d_model=self.d_model, d_route=self.d_route,
            operator_key_mode=operator_key_mode)
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
                 admission_den_power=None,
                 srw_composition_mode=None,
                 heat_kernel_beta=None,
                 execution_prune_eps=0.0,
                 minimal_train=False,
                 ce_token_chunk_size=32768,
                 compute_accuracy=True,
                 analysis_contribution=None,
                 analysis_target_layer=-1,
                 analysis_target_positions=0,
                 analysis_target_route=-1,
                 analysis_intervention_enabled=False,
                 analysis_return_residual=False,
                 analysis_return_logits=False,
                 analysis_parity_debug=False,
                 analysis_causal_trace=False,
                 analysis_keep_qk=None,
                 analysis_keep_v=None,
                 analysis_keep_rst=None,
                 analysis_position_mask=None,
                 analysis_retention_mode=0,
                 analysis_capture_contribution=False,
                 analysis_interchange_source=None,
                 analysis_interchange_enabled=False,
                 analysis_program_ids_q=None,
                 analysis_program_ids_k=None,
                 analysis_program_ids_v=None,
                 analysis_program_ids_rst=None,
                 analysis_program_valid_q=None,
                 analysis_program_valid_k=None,
                 analysis_program_valid_v=None,
                 analysis_program_valid_rst=None,
                 analysis_program_target_positions=None,
                 analysis_program_mode=0,
                 analysis_program_source_q=None,
                 analysis_program_source_k=None,
                 analysis_program_source_v=None,
                 analysis_program_source_rst=None,
                 analysis_program_capture_contribution=False,
                 analysis_trajectory_positions=None,
                 analysis_trajectory_position_valid=None,
                 analysis_trajectory_ids_q=None,
                 analysis_trajectory_ids_k=None,
                 analysis_trajectory_ids_v=None,
                 analysis_trajectory_ids_rst=None,
                 analysis_trajectory_valid_q=None,
                 analysis_trajectory_valid_k=None,
                 analysis_trajectory_valid_v=None,
                 analysis_trajectory_valid_rst=None,
                 analysis_trajectory_replay_enabled=False,
                 analysis_trajectory_patch_layers=None,
                 analysis_trajectory_patch_positions=None,
                 analysis_trajectory_patch_stages=None,
                 analysis_trajectory_patch_enabled=None,
                 analysis_trajectory_patch_values=None):
        """Run the shared-pool SRW Transformer forward pass.

        analysis=False is the train/eval path and returns only regular
        training metrics.  analysis=True enables extra observational stats
        such as distribution shape, selection diagnostics, entropy, tau stats,
        raw norms, and output-stability norms.
        """
        operator_key_mode = _validate_operator_key_mode(
            self.operator_key_mode, context="DAWN_SRW_V4171 forward")
        (model_admission_den_power, model_admission_den_power_qk,
         model_admission_den_power_v, model_admission_den_power_rst) = (
            _resolve_v417x_admission_den_powers(
                self.admission_den_power,
                self.admission_den_power_qk,
                self.admission_den_power_v,
                self.admission_den_power_rst,
                context="DAWN_SRW_V4171 constructor"))
        (model_srw_composition_mode, model_admission_den_power,
         model_heat_kernel_beta) = (
            _validate_v4171_composition_settings(
                self.srw_composition_mode,
                model_admission_den_power,
                self.heat_kernel_beta,
                context="DAWN_SRW_V4171 constructor"))
        (runtime_srw_composition_mode, runtime_admission_den_power,
         runtime_heat_kernel_beta) = (
            _validate_v4171_composition_settings(
                model_srw_composition_mode
                if srw_composition_mode is None else srw_composition_mode,
                (model_admission_den_power
                 if admission_den_power is None else admission_den_power),
                (model_heat_kernel_beta
                 if heat_kernel_beta is None else heat_kernel_beta),
                context="DAWN_SRW_V4171 forward"))
        if runtime_srw_composition_mode != model_srw_composition_mode:
            raise ValueError(
                "v4171 constructor/forward srw_composition_mode mismatch: "
                f"model={model_srw_composition_mode!r}, "
                f"runtime={runtime_srw_composition_mode!r}")
        if runtime_admission_den_power != model_admission_den_power:
            raise ValueError(
                "v4171 constructor/forward admission_den_power mismatch: "
                f"model={model_admission_den_power}, "
                f"runtime={runtime_admission_den_power}")
        if runtime_heat_kernel_beta != model_heat_kernel_beta:
            raise ValueError(
                "v4171 constructor/forward heat_kernel_beta mismatch: "
                f"model={model_heat_kernel_beta}, "
                f"runtime={runtime_heat_kernel_beta}")
        _validate_v4171_sharded_fns(
            sharded_fns, model_admission_den_power,
            model_srw_composition_mode, model_heat_kernel_beta,
            expected_power_qk=model_admission_den_power_qk,
            expected_power_v=model_admission_den_power_v,
            expected_power_rst=model_admission_den_power_rst)
        admission_den_power = model_admission_den_power
        admission_den_power_qk = model_admission_den_power_qk
        admission_den_power_v = model_admission_den_power_v
        admission_den_power_rst = model_admission_den_power_rst
        heat_kernel_beta = model_heat_kernel_beta
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

        analysis_program_values = (
            analysis_program_ids_q, analysis_program_ids_k,
            analysis_program_ids_v, analysis_program_ids_rst,
            analysis_program_valid_q, analysis_program_valid_k,
            analysis_program_valid_v, analysis_program_valid_rst,
            analysis_program_target_positions,
        )
        analysis_program_enabled = any(
            value is not None for value in analysis_program_values)
        analysis_trajectory_capture_values = (
            analysis_trajectory_positions,
            analysis_trajectory_position_valid,
            analysis_trajectory_ids_q, analysis_trajectory_ids_k,
            analysis_trajectory_ids_v, analysis_trajectory_ids_rst,
            analysis_trajectory_valid_q, analysis_trajectory_valid_k,
            analysis_trajectory_valid_v, analysis_trajectory_valid_rst,
        )
        analysis_trajectory_patch_values_all = (
            analysis_trajectory_patch_layers,
            analysis_trajectory_patch_positions,
            analysis_trajectory_patch_stages,
            analysis_trajectory_patch_enabled,
            analysis_trajectory_patch_values,
        )
        analysis_trajectory_enabled = any(
            value is not None for value in analysis_trajectory_capture_values)
        analysis_trajectory_patch_schedule_enabled = any(
            value is not None
            for value in analysis_trajectory_patch_values_all)
        any_trajectory_analysis = (
            analysis_trajectory_enabled
            or analysis_trajectory_patch_schedule_enabled)
        if any_trajectory_analysis and not minimal_train:
            raise ValueError(
                "trajectory analysis requires the canonical minimal scan")
        if analysis_trajectory_enabled and any(
                value is None
                for value in analysis_trajectory_capture_values):
            raise ValueError(
                "trajectory capture requires positions and per-route ids "
                "with validity")
        if analysis_trajectory_patch_schedule_enabled and any(
                value is None
                for value in analysis_trajectory_patch_values_all):
            raise ValueError(
                "trajectory patching requires one complete fixed schedule")
        if analysis_program_enabled and not minimal_train:
            raise ValueError(
                "analysis operator programs require the canonical minimal scan")
        if analysis_program_enabled and any(
                value is None for value in analysis_program_values):
            raise ValueError(
                "analysis operator program requires ids, validity, and target "
                "positions for q,k,v,rst")
        if (analysis_program_capture_contribution
                and not analysis_program_enabled):
            raise ValueError(
                "program contribution capture requires a program schedule")
        if (not analysis_program_enabled
                and isinstance(analysis_program_mode, numbers.Integral)
                and not isinstance(analysis_program_mode, bool)
                and int(analysis_program_mode) != 0):
            raise ValueError("non-production program mode lacks a schedule")

        if minimal_train:
            if analysis_trajectory_enabled:
                analysis_trajectory_positions = _analysis_int32_array(
                    analysis_trajectory_positions,
                    name="analysis_trajectory_positions")
                analysis_trajectory_position_valid = jnp.asarray(
                    analysis_trajectory_position_valid)
                if analysis_trajectory_position_valid.dtype != jnp.bool_:
                    raise TypeError(
                        "analysis_trajectory_position_valid must be bool")
                if (analysis_trajectory_positions.ndim != 2
                        or analysis_trajectory_positions.shape[0] != B
                        or analysis_trajectory_positions.shape[1] <= 0):
                    raise ValueError(
                        "analysis_trajectory_positions must have shape "
                        "[B, positive_T]")
                if analysis_trajectory_position_valid.shape != (
                        analysis_trajectory_positions.shape):
                    raise ValueError(
                        "analysis trajectory position validity mismatch")
                trajectory_prefix = (
                    self.n_layers, B,
                    analysis_trajectory_positions.shape[1])

                def normalize_trajectory_route(ids, valid, route):
                    ids = _analysis_int32_array(
                        ids, name=f"analysis_trajectory_ids_{route}")
                    valid = jnp.asarray(valid)
                    if valid.dtype != jnp.bool_:
                        raise TypeError(
                            f"analysis_trajectory_valid_{route} must be bool")
                    if (ids.ndim != 4 or ids.shape[:3] != trajectory_prefix
                            or ids.shape[3] <= 0):
                        raise ValueError(
                            f"analysis_trajectory_ids_{route} must have shape "
                            "[L,B,T,positive_K]")
                    if valid.shape != ids.shape:
                        raise ValueError(
                            f"analysis trajectory {route} validity mismatch")
                    return ids, valid

                (analysis_trajectory_ids_q,
                 analysis_trajectory_valid_q) = normalize_trajectory_route(
                     analysis_trajectory_ids_q,
                     analysis_trajectory_valid_q, "q")
                (analysis_trajectory_ids_k,
                 analysis_trajectory_valid_k) = normalize_trajectory_route(
                     analysis_trajectory_ids_k,
                     analysis_trajectory_valid_k, "k")
                (analysis_trajectory_ids_v,
                 analysis_trajectory_valid_v) = normalize_trajectory_route(
                     analysis_trajectory_ids_v,
                     analysis_trajectory_valid_v, "v")
                (analysis_trajectory_ids_rst,
                 analysis_trajectory_valid_rst) = normalize_trajectory_route(
                     analysis_trajectory_ids_rst,
                     analysis_trajectory_valid_rst, "rst")
                analysis_trajectory_replay_enabled = jnp.asarray(
                    analysis_trajectory_replay_enabled, dtype=jnp.bool_)
                if analysis_trajectory_replay_enabled.shape != ():
                    raise ValueError(
                        "trajectory replay flag must be a scalar")
            if analysis_trajectory_patch_schedule_enabled:
                analysis_trajectory_patch_layers = _analysis_int32_array(
                    analysis_trajectory_patch_layers,
                    name="analysis_trajectory_patch_layers")
                analysis_trajectory_patch_positions = _analysis_int32_array(
                    analysis_trajectory_patch_positions,
                    name="analysis_trajectory_patch_positions")
                analysis_trajectory_patch_stages = _analysis_int32_array(
                    analysis_trajectory_patch_stages,
                    name="analysis_trajectory_patch_stages")
                analysis_trajectory_patch_enabled = jnp.asarray(
                    analysis_trajectory_patch_enabled)
                if analysis_trajectory_patch_enabled.dtype != jnp.bool_:
                    raise TypeError(
                        "analysis_trajectory_patch_enabled must be bool")
                patch_shape = analysis_trajectory_patch_layers.shape
                if (len(patch_shape) != 2 or patch_shape[0] != B
                        or patch_shape[1] <= 0):
                    raise ValueError(
                        "trajectory patch schedule must have shape "
                        "[B, positive_P]")
                if any(value.shape != patch_shape for value in (
                        analysis_trajectory_patch_positions,
                        analysis_trajectory_patch_stages,
                        analysis_trajectory_patch_enabled)):
                    raise ValueError("trajectory patch schedule shape mismatch")
                patch_dtype = getattr(
                    analysis_trajectory_patch_values, "dtype", None)
                if patch_dtype != jnp.float32:
                    raise TypeError(
                        "analysis_trajectory_patch_values must be explicit "
                        "float32")
                analysis_trajectory_patch_values = jnp.asarray(
                    analysis_trajectory_patch_values)
                if analysis_trajectory_patch_values.shape != (
                        patch_shape + (self.d_model,)):
                    raise ValueError(
                        "trajectory patch values must have shape [B,P,D]")
                _validate_concrete_trajectory_patches(
                    patch_layers=analysis_trajectory_patch_layers,
                    patch_positions=analysis_trajectory_patch_positions,
                    patch_stages=analysis_trajectory_patch_stages,
                    patch_enabled=analysis_trajectory_patch_enabled,
                    patch_values=analysis_trajectory_patch_values,
                    batch_size=B, sequence_length=S,
                    n_layers=self.n_layers)
            if analysis_trajectory_enabled:
                if analysis_trajectory_patch_schedule_enabled:
                    validation_patch_layers = (
                        analysis_trajectory_patch_layers)
                    validation_patch_positions = (
                        analysis_trajectory_patch_positions)
                    validation_patch_stages = (
                        analysis_trajectory_patch_stages)
                    validation_patch_enabled = (
                        analysis_trajectory_patch_enabled)
                    validation_patch_values = (
                        analysis_trajectory_patch_values)
                else:
                    validation_patch_layers = jnp.zeros(
                        (B, 1), dtype=jnp.int32)
                    validation_patch_positions = jnp.zeros(
                        (B, 1), dtype=jnp.int32)
                    validation_patch_stages = jnp.zeros(
                        (B, 1), dtype=jnp.int32)
                    validation_patch_enabled = jnp.zeros(
                        (B, 1), dtype=jnp.bool_)
                    validation_patch_values = jnp.zeros(
                        (B, 1, self.d_model), dtype=jnp.float32)
                _validate_concrete_analysis_trajectory(
                    positions=analysis_trajectory_positions,
                    position_valid=analysis_trajectory_position_valid,
                    selected_ids_by_route={
                        "q": analysis_trajectory_ids_q,
                        "k": analysis_trajectory_ids_k,
                        "v": analysis_trajectory_ids_v,
                        "rst": analysis_trajectory_ids_rst,
                    },
                    selected_valid_by_route={
                        "q": analysis_trajectory_valid_q,
                        "k": analysis_trajectory_valid_k,
                        "v": analysis_trajectory_valid_v,
                        "rst": analysis_trajectory_valid_rst,
                    },
                    patch_layers=validation_patch_layers,
                    patch_positions=validation_patch_positions,
                    patch_stages=validation_patch_stages,
                    patch_enabled=validation_patch_enabled,
                    patch_values=validation_patch_values,
                    batch_size=B, sequence_length=S,
                    n_layers=self.n_layers,
                    route_pool_sizes={
                        "q": self.n_qk, "k": self.n_qk,
                        "v": self.n_v, "rst": n_rst_eff,
                    })
            if analysis_program_enabled:
                def normalize_program_route(ids, valid, route):
                    ids = _analysis_int32_array(
                        ids, name=f"analysis_program_ids_{route}")
                    valid = jnp.asarray(valid)
                    if valid.dtype != jnp.bool_:
                        raise TypeError(
                            f"analysis_program_valid_{route} must have bool "
                            f"dtype, got {valid.dtype}")
                    if (ids.ndim != 3 or ids.shape[:2] != (
                            self.n_layers, B) or ids.shape[2] <= 0):
                        raise ValueError(
                            f"analysis_program_ids_{route} must have shape "
                            "[layers,batch,positive_width], got "
                            f"{ids.shape}")
                    if valid.shape != ids.shape:
                        raise ValueError(
                            f"analysis_program_valid_{route} shape mismatch")
                    return ids, valid

                (analysis_program_ids_q,
                 analysis_program_valid_q) = normalize_program_route(
                     analysis_program_ids_q, analysis_program_valid_q, "q")
                (analysis_program_ids_k,
                 analysis_program_valid_k) = normalize_program_route(
                     analysis_program_ids_k, analysis_program_valid_k, "k")
                (analysis_program_ids_v,
                 analysis_program_valid_v) = normalize_program_route(
                     analysis_program_ids_v, analysis_program_valid_v, "v")
                (analysis_program_ids_rst,
                 analysis_program_valid_rst) = normalize_program_route(
                     analysis_program_ids_rst, analysis_program_valid_rst,
                     "rst")
                analysis_program_target_positions = _analysis_int32_array(
                    analysis_program_target_positions,
                    name="analysis_program_target_positions")
                if analysis_program_target_positions.shape != (B,):
                    raise ValueError(
                        "analysis_program_target_positions shape mismatch: "
                        f"expected={(B,)} actual="
                        f"{analysis_program_target_positions.shape}")
                analysis_program_mode = _analysis_int32_array(
                    analysis_program_mode, name="analysis_program_mode")
                if analysis_program_mode.shape != ():
                    raise ValueError("analysis_program_mode must be a scalar")

                def normalize_program_source(value, route):
                    if value is None:
                        return jnp.zeros(
                            (self.n_layers, B, self.d_model),
                            dtype=jnp.float32)
                    source_dtype = getattr(value, "dtype", None)
                    if source_dtype is None or source_dtype != jnp.float32:
                        raise TypeError(
                            f"analysis_program_source_{route} must be an "
                            f"explicit float32 array, got {source_dtype}")
                    source = jnp.asarray(value)
                    expected = (self.n_layers, B, self.d_model)
                    if source.shape != expected:
                        raise ValueError(
                            f"analysis_program_source_{route} shape mismatch: "
                            f"expected={expected} actual={source.shape}")
                    return source

                analysis_program_source_q = normalize_program_source(
                    analysis_program_source_q, "q")
                analysis_program_source_k = normalize_program_source(
                    analysis_program_source_k, "k")
                analysis_program_source_v = normalize_program_source(
                    analysis_program_source_v, "v")
                analysis_program_source_rst = normalize_program_source(
                    analysis_program_source_rst, "rst")
                _validate_concrete_analysis_program(
                    program_mode=analysis_program_mode,
                    target_positions=analysis_program_target_positions,
                    selected_ids_q=analysis_program_ids_q,
                    selected_ids_k=analysis_program_ids_k,
                    selected_ids_v=analysis_program_ids_v,
                    selected_ids_rst=analysis_program_ids_rst,
                    selected_valid_q=analysis_program_valid_q,
                    selected_valid_k=analysis_program_valid_k,
                    selected_valid_v=analysis_program_valid_v,
                    selected_valid_rst=analysis_program_valid_rst,
                    batch_size=B,
                    sequence_length=S,
                    n_layers=self.n_layers,
                    n_qk=self.n_qk,
                    n_v=self.n_v,
                    n_rst=n_rst_eff,
                    enabled=True)
            if analysis_contribution is None:
                analysis_contribution = jnp.full(
                    (B,), -1, dtype=jnp.int32)
                analysis_target_positions = jnp.full(
                    (B,), -1, dtype=jnp.int32)
                analysis_target_layer = jnp.asarray(-1, dtype=jnp.int32)
                analysis_target_route = jnp.asarray(-1, dtype=jnp.int32)
                analysis_intervention_enabled = jnp.asarray(
                    False, dtype=jnp.bool_)
            else:
                analysis_contribution = _analysis_int32_array(
                    analysis_contribution,
                    name="analysis_contribution")
                if analysis_contribution.ndim == 0:
                    analysis_contribution = jnp.full(
                        (B,), analysis_contribution, dtype=jnp.int32)
                if (analysis_contribution.ndim not in (1, 2)
                        or analysis_contribution.shape[0] != B
                        or (analysis_contribution.ndim == 2
                            and analysis_contribution.shape[1] <= 0)):
                    raise ValueError(
                        "analysis_contribution must have shape [B] or "
                        "non-empty fixed-width [B, M], got "
                        f"{analysis_contribution.shape}")
                analysis_target_positions = _analysis_int32_array(
                    analysis_target_positions,
                    name="analysis_target_positions")
                if analysis_target_positions.ndim == 0:
                    analysis_target_positions = jnp.full(
                        (B,), analysis_target_positions, dtype=jnp.int32)
                if analysis_target_positions.shape != (B,):
                    raise ValueError(
                        "analysis_target_positions shape mismatch: "
                        f"expected={(B,)} "
                        f"actual={analysis_target_positions.shape}")
                analysis_target_layer = _analysis_int32_array(
                    analysis_target_layer,
                    name="analysis_target_layer")
                analysis_target_route = _analysis_int32_array(
                    analysis_target_route,
                    name="analysis_target_route")
                if analysis_target_layer.shape != ():
                    raise ValueError(
                        "analysis_target_layer must be a scalar, got "
                        f"{analysis_target_layer.shape}")
                if analysis_target_route.shape != ():
                    raise ValueError(
                        "analysis_target_route must be a scalar, got "
                        f"{analysis_target_route.shape}")
                analysis_intervention_enabled = jnp.asarray(
                    analysis_intervention_enabled, dtype=jnp.bool_)
                if analysis_intervention_enabled.shape != ():
                    raise ValueError(
                        "analysis_intervention_enabled must be a scalar")
            if analysis_keep_qk is None:
                analysis_keep_qk = jnp.ones(
                    (self.n_layers, 2, self.n_qk), dtype=jnp.bool_)
            else:
                analysis_keep_qk = jnp.asarray(analysis_keep_qk)
                if analysis_keep_qk.dtype != jnp.bool_:
                    raise TypeError("analysis_keep_qk must have bool dtype")
            if analysis_keep_v is None:
                analysis_keep_v = jnp.ones(
                    (self.n_layers, self.n_v), dtype=jnp.bool_)
            else:
                analysis_keep_v = jnp.asarray(analysis_keep_v)
                if analysis_keep_v.dtype != jnp.bool_:
                    raise TypeError("analysis_keep_v must have bool dtype")
            if analysis_keep_rst is None:
                analysis_keep_rst = jnp.ones(
                    (self.n_layers, n_rst_eff), dtype=jnp.bool_)
            else:
                analysis_keep_rst = jnp.asarray(analysis_keep_rst)
                if analysis_keep_rst.dtype != jnp.bool_:
                    raise TypeError("analysis_keep_rst must have bool dtype")
            expected_qk = (self.n_layers, 2, self.n_qk)
            expected_v = (self.n_layers, self.n_v)
            expected_rst = (self.n_layers, n_rst_eff)
            if analysis_keep_qk.shape != expected_qk:
                raise ValueError(
                    "analysis_keep_qk shape mismatch: "
                    f"expected={expected_qk} actual={analysis_keep_qk.shape}")
            if analysis_keep_v.shape != expected_v:
                raise ValueError(
                    "analysis_keep_v shape mismatch: "
                    f"expected={expected_v} actual={analysis_keep_v.shape}")
            if analysis_keep_rst.shape != expected_rst:
                raise ValueError(
                    "analysis_keep_rst shape mismatch: "
                    f"expected={expected_rst} actual={analysis_keep_rst.shape}")
            if analysis_position_mask is None:
                analysis_position_mask = jnp.ones((B, S), dtype=jnp.bool_)
            else:
                analysis_position_mask = jnp.asarray(analysis_position_mask)
                if analysis_position_mask.dtype != jnp.bool_:
                    raise TypeError(
                        "analysis_position_mask must have bool dtype")
            if analysis_position_mask.shape != (B, S):
                raise ValueError(
                    "analysis_position_mask shape mismatch: "
                    f"expected={(B, S)} actual={analysis_position_mask.shape}")
            analysis_retention_mode = jnp.asarray(
                analysis_retention_mode, dtype=jnp.int32)
            if analysis_interchange_source is None:
                analysis_interchange_source = jnp.zeros(
                    (B, self.d_model), dtype=jnp.float32)
            else:
                source_dtype = getattr(
                    analysis_interchange_source, "dtype", None)
                if source_dtype is None or source_dtype != jnp.float32:
                    raise TypeError(
                        "analysis_interchange_source must be an explicit "
                        "float32 array, got "
                        f"{source_dtype}")
                analysis_interchange_source = jnp.asarray(
                    analysis_interchange_source)
                if analysis_interchange_source.dtype != jnp.float32:
                    raise TypeError(
                        "analysis_interchange_source must have float32 "
                        "dtype, got "
                        f"{analysis_interchange_source.dtype}")
            if analysis_interchange_source.shape != (B, self.d_model):
                raise ValueError(
                    "analysis_interchange_source shape mismatch: "
                    f"expected={(B, self.d_model)} "
                    f"actual={analysis_interchange_source.shape}")
            analysis_interchange_enabled = jnp.asarray(
                analysis_interchange_enabled, dtype=jnp.bool_)
            if analysis_interchange_enabled.shape != ():
                raise ValueError(
                    "analysis_interchange_enabled must be a scalar")
            selected_sidecar_enabled = (
                analysis_intervention_enabled
                | jnp.asarray(
                    analysis_capture_contribution, dtype=jnp.bool_)
                | analysis_interchange_enabled)
            interchange_contract_enabled = (
                jnp.asarray(
                    analysis_capture_contribution, dtype=jnp.bool_)
                | analysis_interchange_enabled)

            _validate_concrete_analysis_interchange(
                analysis_target_positions,
                analysis_target_layer,
                analysis_target_route,
                analysis_contribution,
                batch_size=B,
                sequence_length=S,
                n_layers=self.n_layers,
                n_qk=self.n_qk,
                n_v=self.n_v,
                n_rst=n_rst_eff,
                enabled=interchange_contract_enabled)
            analysis_contribution = jnp.where(
                selected_sidecar_enabled,
                analysis_contribution,
                jnp.full_like(analysis_contribution, -1))

        positions = jnp.arange(S)[jnp.newaxis, :]
        vp_embed = (
            sharded_fns.get("vocab_parallel_embedding")
            if isinstance(sharded_fns, dict)
            else None
        )
        if vp_embed is not None:
            x = vp_embed(input_ids, self.token_emb.embedding)
        else:
            x = self.token_emb(input_ids)
        x = x + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        _eval_stats_enabled = (
            isinstance(sharded_fns, dict)
            and sharded_fns.get("vocab_eval_stats") is not None)

        def _compute_vocab_ce(final_x):
            _ce_token_chunk_size = int(ce_token_chunk_size)
            if _ce_token_chunk_size <= 0:
                raise ValueError(
                    "ce_token_chunk_size must be > 0, got "
                    f"{_ce_token_chunk_size}")
            _compute_accuracy = bool(compute_accuracy)
            embedding_matrix = self.token_emb.embedding
            shift_x = final_x[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = shift_labels != -100

            eval_stats = (
                sharded_fns.get("vocab_eval_stats")
                if isinstance(sharded_fns, dict)
                else None
            )
            vp_ce = (
                sharded_fns.get("vocab_ce")
                if isinstance(sharded_fns, dict)
                else None
            )
            if eval_stats is not None:
                per_token_ce, per_token_correct = eval_stats(
                    shift_x, embedding_matrix, shift_labels, valid_mask)
                valid_f = valid_mask.astype(jnp.float32)
                valid_count = valid_mask.astype(jnp.int32).sum()
                loss = (per_token_ce * valid_f).sum() / jnp.maximum(
                    valid_count.astype(jnp.float32), 1.0)
                correct = per_token_correct.astype(jnp.int32).sum()
                logit_abs_max = jnp.float32(0.0)
                logit_norm_mean = jnp.float32(0.0)
                logit_mean = jnp.float32(0.0)
                logit_std = jnp.float32(0.0)
            elif vp_ce is not None:
                (
                    loss,
                    per_token_ce,
                    correct,
                    valid_count,
                    logit_abs_max,
                    logit_norm_mean,
                    logit_mean,
                    logit_std,
                ) = vp_ce(
                    shift_x, embedding_matrix, shift_labels, valid_mask)
            else:
                logical_vocab_size, _ = self._vocab_sizes()
                (
                    loss,
                    per_token_ce,
                    correct,
                    valid_count,
                    logit_abs_max,
                    logit_norm_mean,
                    logit_mean,
                    logit_std,
                ) = _chunked_ce_loss_and_acc(
                    shift_x,
                    embedding_matrix,
                    shift_labels,
                    valid_mask,
                    token_chunk_size=_ce_token_chunk_size,
                    compute_accuracy=_compute_accuracy,
                    logical_vocab_size=logical_vocab_size,
                )
                per_token_correct = None
            if eval_stats is None and vp_ce is not None:
                per_token_correct = None
            return (
                loss,
                per_token_ce,
                correct,
                valid_count,
                logit_abs_max,
                logit_norm_mean,
                logit_mean,
                logit_std,
                valid_mask,
                per_token_correct,
            )

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
            if operator_key_mode == OPERATOR_KEY_MODE_LEARNED:
                _ = self.neuron_pool.attn_qk_op_key
                _ = self.neuron_pool.attn_v_op_key
                _ = self.neuron_pool.rst_operator_keys
            else:
                _ = self.neuron_pool.rw_key_read_probe
                _ = self.neuron_pool.rw_key_write_probe
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
                all_params['neuron_pool'], operator_key_mode)
            router_params = all_params['router']

            _sharded = sharded_fns

            block_params_list = [all_params[f'block_{i}']
                                 for i in range(self.n_layers)]
            stacked = jax.tree.map(
                lambda *arrays: jnp.stack(arrays), *block_params_list)

            base_rng = self.make_rng('dropout')
            layer_rngs = jax.random.split(base_rng, self.n_layers)

            if minimal_train:
                trace_minimal_layers = bool(
                    analysis_parity_debug or analysis_causal_trace
                    or analysis_trajectory_enabled)

                def scan_body_minimal(carry, xs):
                    x = carry
                    pre_layer_residual = x
                    bp = xs['params']
                    rng = xs['rng']
                    layer_index = xs['layer_index']
                    if analysis_trajectory_patch_schedule_enabled:
                        x = _analysis_apply_trajectory_patch(
                            x, layer_index=layer_index,
                            stage_code=TRAJECTORY_PATCH_STAGES[
                                "residual_input"],
                            patch_layers=analysis_trajectory_patch_layers,
                            patch_positions=(
                                analysis_trajectory_patch_positions),
                            patch_stages=analysis_trajectory_patch_stages,
                            patch_enabled=analysis_trajectory_patch_enabled,
                            patch_values=analysis_trajectory_patch_values)
                        pre_layer_residual = x
                    if analysis_program_enabled:
                        # The public schedule uses logical id 0 plus a false
                        # validity bit. Indexed membership receives a sorted
                        # out-of-pool sentinel so padding cannot match id 0.
                        program_ids_q = jnp.where(
                            xs['program_valid_q'], xs['program_ids_q'],
                            jnp.int32(self.n_qk))
                        program_ids_k = jnp.where(
                            xs['program_valid_k'], xs['program_ids_k'],
                            jnp.int32(self.n_qk))
                        program_ids_v = jnp.where(
                            xs['program_valid_v'], xs['program_ids_v'],
                            jnp.int32(self.n_v))
                        program_ids_rst = jnp.where(
                            xs['program_valid_rst'], xs['program_ids_rst'],
                            jnp.int32(n_rst_eff))
                        program_source_q = xs['program_source_q']
                        program_source_k = xs['program_source_k']
                        program_source_v = xs['program_source_v']
                        program_source_rst = xs['program_source_rst']
                    else:
                        program_ids_q = None
                        program_ids_k = None
                        program_ids_v = None
                        program_ids_rst = None
                        program_source_q = None
                        program_source_k = None
                        program_source_v = None
                        program_source_rst = None
                    rng, rng_attn, rng_rst = jax.random.split(rng, 3)

                    normed = _layer_norm(
                        x, bp['norm1']['scale'], bp['norm1']['bias'])
                    attn_result = _attn_forward_minimal(
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
                        admission_den_power=admission_den_power_qk,
                        execution_prune_eps=execution_prune_eps,
                        analysis_selected_operator_id=analysis_contribution,
                        analysis_layer_index=layer_index,
                        analysis_target_layer=analysis_target_layer,
                        analysis_target_positions=analysis_target_positions,
                        analysis_target_route=analysis_target_route,
                        analysis_intervention_enabled=(
                            analysis_intervention_enabled),
                        analysis_keep_qk=analysis_keep_qk[layer_index],
                        analysis_keep_v=analysis_keep_v[layer_index],
                        analysis_position_mask=analysis_position_mask,
                        analysis_retention_mode=analysis_retention_mode,
                        analysis_interchange_source=(
                            analysis_interchange_source),
                        analysis_interchange_enabled=(
                            analysis_interchange_enabled),
                        analysis_program_enabled=analysis_program_enabled,
                        analysis_program_ids_q=program_ids_q,
                        analysis_program_ids_k=program_ids_k,
                        analysis_program_ids_v=program_ids_v,
                        analysis_program_target_positions=(
                            analysis_program_target_positions),
                        analysis_program_mode=analysis_program_mode,
                        analysis_program_source_q=program_source_q,
                        analysis_program_source_k=program_source_k,
                        analysis_program_source_v=program_source_v,
                        analysis_trajectory_enabled=(
                            analysis_trajectory_enabled),
                        analysis_trajectory_positions=(
                            analysis_trajectory_positions),
                        analysis_trajectory_position_valid=(
                            analysis_trajectory_position_valid),
                        analysis_trajectory_ids_q=(
                            xs.get('trajectory_ids_q')),
                        analysis_trajectory_ids_k=(
                            xs.get('trajectory_ids_k')),
                        analysis_trajectory_ids_v=(
                            xs.get('trajectory_ids_v')),
                        analysis_trajectory_valid_q=(
                            xs.get('trajectory_valid_q')),
                        analysis_trajectory_valid_k=(
                            xs.get('trajectory_valid_k')),
                        analysis_trajectory_valid_v=(
                            xs.get('trajectory_valid_v')),
                        analysis_trajectory_replay_enabled=(
                            analysis_trajectory_replay_enabled),
                        analysis_trajectory_patch_schedule_enabled=(
                            analysis_trajectory_patch_schedule_enabled),
                        analysis_trajectory_patch_layers=(
                            analysis_trajectory_patch_layers),
                        analysis_trajectory_patch_positions=(
                            analysis_trajectory_patch_positions),
                        analysis_trajectory_patch_stages=(
                            analysis_trajectory_patch_stages),
                        analysis_trajectory_patch_enabled=(
                            analysis_trajectory_patch_enabled),
                        analysis_trajectory_patch_values=(
                            analysis_trajectory_patch_values),
                        parity_debug=trace_minimal_layers)
                    if trace_minimal_layers:
                        attn_values = attn_result[:-1]
                        attn_debug = attn_result[-1]
                    else:
                        attn_values = attn_result
                    attn_trajectory = None
                    if analysis_trajectory_enabled:
                        attn_trajectory = attn_values[-1]
                        attn_values = attn_values[:-1]
                    (attn_out, q_active_frac, k_active_frac, v_active_frac,
                     q_active_n_mean, k_active_n_mean, v_active_n_mean,
                     q_gate_mass_mean, k_gate_mass_mean, v_gate_mass_mean,
                     q_gate_den_mean, k_gate_den_mean, v_gate_den_mean,
                     q_depth_active_mean, k_depth_active_mean,
                     v_depth_active_mean,
                     q_gate_eff_n_mean, k_gate_eff_n_mean,
                     v_gate_eff_n_mean,
                     q_top1_gate_frac_mean, k_top1_gate_frac_mean,
                     v_top1_gate_frac_mean,
                     q_den_floor_frac, k_den_floor_frac, v_den_floor_frac,
                     q_tau_mean, k_tau_mean, v_tau_mean,
                     attn_out_norm,
                     qk_admission_mass_max, qk_composition_den_min,
                     qk_composition_den_max, qk_raw_srw_out_norm,
                     qk_normalized_srw_out_norm,
                     qk_pool_scaled_srw_out_norm,
                     v_admission_mass_max, v_composition_den_min,
                     v_composition_den_max, v_raw_srw_out_norm,
                     v_normalized_srw_out_norm,
                     v_pool_scaled_srw_out_norm,
                     selected_q_contribution,
                     selected_k_contribution,
                     selected_v_contribution,
                     program_selected_q_contribution,
                     program_selected_k_contribution,
                     program_selected_v_contribution) = attn_values
                    x = x + attn_out
                    if analysis_trajectory_patch_schedule_enabled:
                        x = _analysis_apply_trajectory_patch(
                            x, layer_index=layer_index,
                            stage_code=TRAJECTORY_PATCH_STAGES[
                                "post_attention"],
                            patch_layers=analysis_trajectory_patch_layers,
                            patch_positions=(
                                analysis_trajectory_patch_positions),
                            patch_stages=analysis_trajectory_patch_stages,
                            patch_enabled=analysis_trajectory_patch_enabled,
                            patch_values=analysis_trajectory_patch_values)
                    post_attention_residual = x

                    normed = _layer_norm(
                        x, bp['norm2']['scale'], bp['norm2']['bias'])
                    rst_result = _rst_forward_minimal(
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
                        admission_den_power=admission_den_power_rst,
                        execution_prune_eps=execution_prune_eps,
                        analysis_selected_operator_id=analysis_contribution,
                        analysis_layer_index=layer_index,
                        analysis_target_layer=analysis_target_layer,
                        analysis_target_positions=analysis_target_positions,
                        analysis_target_route=analysis_target_route,
                        analysis_intervention_enabled=(
                            analysis_intervention_enabled),
                        analysis_keep_rst=analysis_keep_rst[layer_index],
                        analysis_position_mask=analysis_position_mask,
                        analysis_retention_mode=analysis_retention_mode,
                        analysis_interchange_source=(
                            analysis_interchange_source),
                        analysis_interchange_enabled=(
                            analysis_interchange_enabled),
                        analysis_program_enabled=analysis_program_enabled,
                        analysis_program_ids_rst=program_ids_rst,
                        analysis_program_target_positions=(
                            analysis_program_target_positions),
                        analysis_program_mode=analysis_program_mode,
                        analysis_program_source_rst=program_source_rst,
                        analysis_trajectory_enabled=(
                            analysis_trajectory_enabled),
                        analysis_trajectory_positions=(
                            analysis_trajectory_positions),
                        analysis_trajectory_position_valid=(
                            analysis_trajectory_position_valid),
                        analysis_trajectory_ids_rst=(
                            xs.get('trajectory_ids_rst')),
                        analysis_trajectory_valid_rst=(
                            xs.get('trajectory_valid_rst')),
                        analysis_trajectory_replay_enabled=(
                            analysis_trajectory_replay_enabled),
                        analysis_trajectory_patch_schedule_enabled=(
                            analysis_trajectory_patch_schedule_enabled),
                        analysis_trajectory_patch_layers=(
                            analysis_trajectory_patch_layers),
                        analysis_trajectory_patch_positions=(
                            analysis_trajectory_patch_positions),
                        analysis_trajectory_patch_stages=(
                            analysis_trajectory_patch_stages),
                        analysis_trajectory_patch_enabled=(
                            analysis_trajectory_patch_enabled),
                        analysis_trajectory_patch_values=(
                            analysis_trajectory_patch_values),
                        parity_debug=trace_minimal_layers)
                    if trace_minimal_layers:
                        rst_values = rst_result[:-1]
                        rst_debug = rst_result[-1]
                    else:
                        rst_values = rst_result
                    rst_trajectory = None
                    if analysis_trajectory_enabled:
                        rst_trajectory = rst_values[-1]
                        rst_values = rst_values[:-1]
                    (rst_out, rst_active_frac, rst_active_n_mean,
                     rst_gate_mass_mean, rst_gate_den_mean,
                     rst_depth_active_mean, rst_gate_eff_n_mean,
                     rst_top1_gate_frac_mean, rst_den_floor_frac,
                     rst_tau_mean, rst_out_norm,
                     rst_admission_mass_max, rst_composition_den_min,
                     rst_composition_den_max, rst_raw_srw_out_norm,
                     rst_normalized_srw_out_norm,
                     rst_pool_scaled_srw_out_norm,
                     selected_rst_contribution,
                     program_selected_rst_contribution) = rst_values
                    x_next = x + rst_out
                    if analysis_trajectory_patch_schedule_enabled:
                        x_next = _analysis_apply_trajectory_patch(
                            x_next, layer_index=layer_index,
                            stage_code=TRAJECTORY_PATCH_STAGES["post_rst"],
                            patch_layers=analysis_trajectory_patch_layers,
                            patch_positions=(
                                analysis_trajectory_patch_positions),
                            patch_stages=analysis_trajectory_patch_stages,
                            patch_enabled=analysis_trajectory_patch_enabled,
                            patch_values=analysis_trajectory_patch_values)
                    residual_norm = jnp.linalg.norm(
                        x_next.astype(jnp.float32), axis=-1).mean()
                    layer_stats = (
                        q_active_frac,
                        k_active_frac,
                        v_active_frac,
                        rst_active_frac,
                        q_active_n_mean,
                        k_active_n_mean,
                        v_active_n_mean,
                        rst_active_n_mean,
                        q_gate_mass_mean,
                        k_gate_mass_mean,
                        v_gate_mass_mean,
                        rst_gate_mass_mean,
                        q_gate_den_mean,
                        k_gate_den_mean,
                        v_gate_den_mean,
                        rst_gate_den_mean,
                        q_depth_active_mean,
                        k_depth_active_mean,
                        v_depth_active_mean,
                        rst_depth_active_mean,
                        q_gate_eff_n_mean,
                        k_gate_eff_n_mean,
                        v_gate_eff_n_mean,
                        rst_gate_eff_n_mean,
                        q_top1_gate_frac_mean,
                        k_top1_gate_frac_mean,
                        v_top1_gate_frac_mean,
                        rst_top1_gate_frac_mean,
                        q_den_floor_frac,
                        k_den_floor_frac,
                        v_den_floor_frac,
                        rst_den_floor_frac,
                        q_tau_mean,
                        k_tau_mean,
                        v_tau_mean,
                        rst_tau_mean,
                        attn_out_norm,
                        rst_out_norm,
                        qk_admission_mass_max,
                        qk_composition_den_min,
                        qk_composition_den_max,
                        qk_raw_srw_out_norm,
                        qk_normalized_srw_out_norm,
                        qk_pool_scaled_srw_out_norm,
                        v_admission_mass_max,
                        v_composition_den_min,
                        v_composition_den_max,
                        v_raw_srw_out_norm,
                        v_normalized_srw_out_norm,
                        v_pool_scaled_srw_out_norm,
                        rst_admission_mass_max,
                        rst_composition_den_min,
                        rst_composition_den_max,
                        rst_raw_srw_out_norm,
                        rst_normalized_srw_out_norm,
                        rst_pool_scaled_srw_out_norm,
                        jax.lax.stop_gradient(
                            residual_norm.astype(jnp.float32)),
                    )
                    if analysis_capture_contribution:
                        layer_stats += (
                            selected_q_contribution,
                            selected_k_contribution,
                            selected_v_contribution,
                            selected_rst_contribution,
                        )
                    if analysis_program_capture_contribution:
                        layer_stats += (
                            program_selected_q_contribution,
                            program_selected_k_contribution,
                            program_selected_v_contribution,
                            program_selected_rst_contribution,
                        )
                    if analysis_trajectory_enabled:
                        layer_stats += (
                            attn_trajectory,
                            rst_trajectory,
                        )
                    if trace_minimal_layers:
                        trace_values = (
                            pre_layer_residual,
                            attn_debug[0],
                            attn_debug[1],
                            attn_debug[2],
                            attn_debug[3],
                            attn_debug[4],
                            attn_debug[5],
                            attn_debug[6],
                            attn_debug[7],
                            attn_debug[8],
                            attn_debug[9],
                            attn_debug[10],
                            post_attention_residual,
                            rst_debug[0],
                            rst_debug[1],
                            rst_debug[2],
                            rst_debug[3],
                            x_next,
                        )
                        if analysis_trajectory_enabled:
                            trace_batch = jnp.arange(
                                B, dtype=jnp.int32)[:, None]
                            trace_positions = jnp.clip(
                                analysis_trajectory_positions, 0, S - 1)

                            def trajectory_debug(value):
                                return value[
                                    trace_batch, trace_positions, :]

                            trace_values = tuple(
                                trajectory_debug(value)
                                for value in trace_values)
                        elif (analysis_causal_trace
                              and not analysis_parity_debug):
                            trace_batch = jnp.arange(B, dtype=jnp.int32)
                            trace_positions = jnp.clip(
                                jnp.asarray(
                                    analysis_target_positions,
                                    dtype=jnp.int32),
                                0, S - 1)

                            def target_debug(value):
                                return value[
                                    trace_batch, trace_positions, :]

                            trace_values = tuple(
                                target_debug(value) for value in trace_values)
                        layer_stats += trace_values
                    return x_next, layer_stats

                if self.gradient_checkpointing:
                    scan_body_minimal = jax.checkpoint(scan_body_minimal)

                xs_minimal = {
                    'params': stacked,
                    'rng': layer_rngs,
                    'layer_index': jnp.arange(self.n_layers, dtype=jnp.int32),
                }
                if analysis_program_enabled:
                    xs_minimal.update({
                        'program_ids_q': analysis_program_ids_q,
                        'program_ids_k': analysis_program_ids_k,
                        'program_ids_v': analysis_program_ids_v,
                        'program_ids_rst': analysis_program_ids_rst,
                        'program_valid_q': analysis_program_valid_q,
                        'program_valid_k': analysis_program_valid_k,
                        'program_valid_v': analysis_program_valid_v,
                        'program_valid_rst': analysis_program_valid_rst,
                        'program_source_q': analysis_program_source_q,
                        'program_source_k': analysis_program_source_k,
                        'program_source_v': analysis_program_source_v,
                        'program_source_rst': analysis_program_source_rst,
                    })
                if analysis_trajectory_enabled:
                    xs_minimal.update({
                        'trajectory_ids_q': analysis_trajectory_ids_q,
                        'trajectory_ids_k': analysis_trajectory_ids_k,
                        'trajectory_ids_v': analysis_trajectory_ids_v,
                        'trajectory_ids_rst': analysis_trajectory_ids_rst,
                        'trajectory_valid_q': analysis_trajectory_valid_q,
                        'trajectory_valid_k': analysis_trajectory_valid_k,
                        'trajectory_valid_v': analysis_trajectory_valid_v,
                        'trajectory_valid_rst': analysis_trajectory_valid_rst,
                    })
                x, minimal_stats = jax.lax.scan(
                    scan_body_minimal, x, xs_minimal)
                if trace_minimal_layers:
                    minimal_values = minimal_stats[:-18]
                    (parity_pre_layer_residual_all,
                     parity_q_all, parity_k_all, parity_v_all,
                     parity_attention_update_all,
                     parity_attention_router_input_all,
                     parity_query_q_all, parity_query_k_all,
                     parity_query_v_all,
                     parity_raw_tau_q_all, parity_raw_tau_k_all,
                     parity_raw_tau_v_all,
                     parity_post_attention_residual_all, parity_rst_all,
                     parity_rst_router_input_all, parity_query_rst_all,
                     parity_raw_tau_rst_all,
                     parity_post_layer_residual_all) = minimal_stats[-18:]
                else:
                    minimal_values = minimal_stats
                operator_trajectory_layers = None
                if analysis_trajectory_enabled:
                    operator_trajectory_layers = {
                        'attention': minimal_values[-2],
                        'rst': minimal_values[-1],
                    }
                    minimal_values = minimal_values[:-2]
                program_route_contributions = None
                if analysis_program_capture_contribution:
                    program_route_contributions = {
                        'q': minimal_values[-4],
                        'k': minimal_values[-3],
                        'v': minimal_values[-2],
                        'rst': minimal_values[-1],
                    }
                    minimal_values = minimal_values[:-4]
                route_contributions = None
                if analysis_capture_contribution:
                    route_contributions = {
                        'q': minimal_values[-4],
                        'k': minimal_values[-3],
                        'v': minimal_values[-2],
                        'rst': minimal_values[-1],
                    }
                    minimal_values = minimal_values[:-4]
                (q_active_all, k_active_all, v_active_all, rst_active_all,
                 q_active_n_all, k_active_n_all, v_active_n_all,
                 rst_active_n_all,
                 q_gate_mass_all, k_gate_mass_all, v_gate_mass_all,
                 rst_gate_mass_all,
                 q_gate_den_all, k_gate_den_all, v_gate_den_all,
                 rst_gate_den_all,
                 q_depth_active_all, k_depth_active_all, v_depth_active_all,
                 rst_depth_active_all,
                 q_gate_eff_n_all, k_gate_eff_n_all, v_gate_eff_n_all,
                 rst_gate_eff_n_all,
                 q_top1_gate_frac_all, k_top1_gate_frac_all,
                 v_top1_gate_frac_all, rst_top1_gate_frac_all,
                 q_den_floor_all, k_den_floor_all, v_den_floor_all,
                 rst_den_floor_all,
                 q_tau_all, k_tau_all, v_tau_all, rst_tau_all,
                 attn_out_norm_all, rst_out_norm_all,
                 qk_admission_mass_max_all,
                 qk_composition_den_min_all, qk_composition_den_max_all,
                 qk_raw_srw_out_norm_all, qk_normalized_srw_out_norm_all,
                 qk_pool_scaled_srw_out_norm_all,
                 v_admission_mass_max_all,
                 v_composition_den_min_all, v_composition_den_max_all,
                 v_raw_srw_out_norm_all, v_normalized_srw_out_norm_all,
                 v_pool_scaled_srw_out_norm_all,
                 rst_admission_mass_max_all,
                 rst_composition_den_min_all, rst_composition_den_max_all,
                 rst_raw_srw_out_norm_all, rst_normalized_srw_out_norm_all,
                 rst_pool_scaled_srw_out_norm_all,
                 residual_norm_all) = minimal_values
                qk_active_all = jnp.float32(0.5) * (
                    q_active_all + k_active_all)
                qk_active_n_all = jnp.float32(0.5) * (
                    q_active_n_all + k_active_n_all)
                qk_gate_mass_all = jnp.float32(0.5) * (
                    q_gate_mass_all + k_gate_mass_all)
                qk_gate_den_all = jnp.float32(0.5) * (
                    q_gate_den_all + k_gate_den_all)
                qk_depth_active_all = jnp.float32(0.5) * (
                    q_depth_active_all + k_depth_active_all)
                qk_gate_eff_n_all = jnp.float32(0.5) * (
                    q_gate_eff_n_all + k_gate_eff_n_all)
                qk_top1_gate_frac_all = jnp.float32(0.5) * (
                    q_top1_gate_frac_all + k_top1_gate_frac_all)
                qk_den_floor_all = jnp.float32(0.5) * (
                    q_den_floor_all + k_den_floor_all)
                qk_tau_all = jnp.float32(0.5) * (q_tau_all + k_tau_all)
                attn_tau_all = (
                    q_tau_all + k_tau_all + v_tau_all) / jnp.float32(3.0)
                if analysis_causal_trace:
                    trace_batch = jnp.arange(B, dtype=jnp.int32)
                    trace_positions = jnp.clip(
                        jnp.asarray(
                            analysis_target_positions, dtype=jnp.int32),
                        0, S - 1)

                    def target_trace(value):
                        if value.ndim == 3:
                            return value
                        return value[:, trace_batch, trace_positions, :]

                    causal_trace = {
                        'pre_layer_residual': target_trace(
                            parity_pre_layer_residual_all),
                        'attention_router_input': target_trace(
                            parity_attention_router_input_all),
                        'query_q': target_trace(parity_query_q_all),
                        'query_k': target_trace(parity_query_k_all),
                        'query_v': target_trace(parity_query_v_all),
                        'raw_tau_q': target_trace(parity_raw_tau_q_all),
                        'raw_tau_k': target_trace(parity_raw_tau_k_all),
                        'raw_tau_v': target_trace(parity_raw_tau_v_all),
                        'route_transition_q': target_trace(parity_q_all),
                        'route_transition_k': target_trace(parity_k_all),
                        'route_transition_v': target_trace(parity_v_all),
                        'post_attention_residual': target_trace(
                            parity_post_attention_residual_all),
                        'attention_update': target_trace(
                            parity_attention_update_all),
                        'rst_router_input': target_trace(
                            parity_rst_router_input_all),
                        'query_rst': target_trace(parity_query_rst_all),
                        'raw_tau_rst': target_trace(parity_raw_tau_rst_all),
                        'route_transition_rst': target_trace(parity_rst_all),
                        'rst_update': target_trace(parity_rst_all),
                        'post_layer_residual': target_trace(
                            parity_post_layer_residual_all),
                    }
                if analysis_trajectory_enabled:
                    attention_trajectory = operator_trajectory_layers[
                        'attention']
                    rst_trajectory = operator_trajectory_layers['rst']
                    qk_fields = attention_trajectory[0]
                    v_fields = attention_trajectory[1]
                    rst_fields = rst_trajectory[0]
                    trajectory_positions = jnp.clip(
                        jnp.asarray(
                            analysis_trajectory_positions,
                            dtype=jnp.int32),
                        0, S - 1)
                    trajectory_batch = jnp.arange(
                        B, dtype=jnp.int32)[:, None]
                    trajectory_valid = jnp.asarray(
                        analysis_trajectory_position_valid,
                        dtype=jnp.bool_)

                    def trajectory_target_trace(value):
                        gathered = value[
                            :, trajectory_batch, trajectory_positions, :]
                        return jnp.where(
                            trajectory_valid[None, :, :, None],
                            gathered, jnp.zeros_like(gathered))

                    def paired_route_fields(route_index):
                        output_fields = {}
                        for field_index, field_name in enumerate(
                                TRAJECTORY_TRACE_FIELDS):
                            value = qk_fields[field_index]
                            if field_index <= 7:
                                value = value[:, :, :, route_index, :]
                            elif field_index <= 9:
                                value = value[:, :, :, route_index]
                            elif field_index <= 17:
                                value = value[:, :, :, route_index, :]
                            output_fields[field_name] = value
                        output_fields['scale'] = attention_trajectory[2]
                        return output_fields

                    def single_route_fields(fields, scale):
                        output_fields = {
                            field_name: fields[field_index]
                            for field_index, field_name in enumerate(
                                TRAJECTORY_TRACE_FIELDS)
                        }
                        output_fields['scale'] = scale
                        return output_fields

                    operator_trajectory_trace = {
                        'positions': analysis_trajectory_positions,
                        'position_valid': (
                            analysis_trajectory_position_valid),
                        'states': {
                            'residual_input': (
                                trajectory_target_trace(
                                    parity_pre_layer_residual_all)),
                            'post_attention': (
                                trajectory_target_trace(
                                    parity_post_attention_residual_all)),
                            'post_rst': trajectory_target_trace(
                                parity_post_layer_residual_all),
                        },
                        'routes': {
                            'q': paired_route_fields(0),
                            'k': paired_route_fields(1),
                            'v': single_route_fields(
                                v_fields, attention_trajectory[3]),
                            'rst': single_route_fields(
                                rst_fields, rst_trajectory[1]),
                        },
                    }
                x = self.norm(x)
                if labels is None:
                    vocab_argmax = (
                        sharded_fns.get("vocab_argmax")
                        if isinstance(sharded_fns, dict) else None)
                    if vocab_argmax is not None:
                        output = {
                            'argmax_token_ids': vocab_argmax(
                                x, self.token_emb.embedding)}
                        if analysis_return_residual:
                            output['final_residual'] = x
                        if analysis_parity_debug:
                            output['parity_debug'] = {
                                'residual_input': (
                                    parity_pre_layer_residual_all),
                                'q': parity_q_all,
                                'k': parity_k_all,
                                'v': parity_v_all,
                                'attention_update': (
                                    parity_attention_update_all),
                                'rst': parity_rst_all,
                                'post_attention': (
                                    parity_post_attention_residual_all),
                                'post_layer_residual': (
                                    parity_post_layer_residual_all),
                            }
                        if analysis_causal_trace:
                            output['causal_trace'] = causal_trace
                        if analysis_trajectory_enabled:
                            output['operator_trajectory_trace'] = (
                                operator_trajectory_trace)
                        if analysis_capture_contribution:
                            output['operator_route_contributions'] = (
                                route_contributions)
                        if analysis_program_capture_contribution:
                            output['operator_program_contributions'] = (
                                program_route_contributions)
                        return output
                    if vp_embed is not None:
                        raise NotImplementedError(
                            "Full logits are disabled on the "
                            "vocab-parallel v4166 path. Pass labels or run "
                            "without sharded_fns.")
                    logical_vocab_size, embedding_vocab_size = (
                        self._vocab_sizes())
                    logits = self.token_emb.attend(x)
                    if embedding_vocab_size != logical_vocab_size:
                        logits = logits[..., :logical_vocab_size]
                    output = {'logits': logits}
                    if analysis_return_residual:
                        output['final_residual'] = x
                    if analysis_parity_debug:
                        output['parity_debug'] = {
                            'residual_input': parity_pre_layer_residual_all,
                            'q': parity_q_all,
                            'k': parity_k_all,
                            'v': parity_v_all,
                            'attention_update': parity_attention_update_all,
                            'rst': parity_rst_all,
                            'post_attention': (
                                parity_post_attention_residual_all),
                            'post_layer_residual': (
                                parity_post_layer_residual_all),
                        }
                    if analysis_causal_trace:
                        output['causal_trace'] = causal_trace
                    if analysis_trajectory_enabled:
                        output['operator_trajectory_trace'] = (
                            operator_trajectory_trace)
                    if analysis_capture_contribution:
                        output['operator_route_contributions'] = (
                            route_contributions)
                    if analysis_program_capture_contribution:
                        output['operator_program_contributions'] = (
                            program_route_contributions)
                    return output

                (loss, per_token_ce, correct, valid_count,
                 logit_abs_max, logit_norm_mean, logit_mean,
                 logit_std, valid_mask,
                 per_token_correct) = _compute_vocab_ce(x)
                def _sg_mean(value):
                    return jax.lax.stop_gradient(value.mean())

                output = {
                    'loss': loss,
                    'correct': correct,
                    'valid_count': valid_count,
                    'logit_max': logit_abs_max,
                    'logit_norm_mean': logit_norm_mean,
                    'logit_mean': logit_mean,
                    'logit_std': logit_std,
                    'per_token_ce': per_token_ce,
                    'valid_mask': valid_mask,
                    'aux_loss': jnp.float32(0.0),
                    **({'per_token_correct': per_token_correct}
                       if _eval_stats_enabled else {}),
                    'attn_q_active': jax.lax.stop_gradient(
                        q_active_all.mean()),
                    'attn_k_active': jax.lax.stop_gradient(
                        k_active_all.mean()),
                    'attn_qk_active': jax.lax.stop_gradient(
                        qk_active_all.mean()),
                    'attn_v_active': jax.lax.stop_gradient(
                        v_active_all.mean()),
                    'rst_active': jax.lax.stop_gradient(
                        rst_active_all.mean()),
                    'attn_q_active_n_mean': jax.lax.stop_gradient(
                        q_active_n_all.mean()),
                    'attn_k_active_n_mean': jax.lax.stop_gradient(
                        k_active_n_all.mean()),
                    'attn_qk_active_n_mean': jax.lax.stop_gradient(
                        qk_active_n_all.mean()),
                    'attn_v_active_n_mean': jax.lax.stop_gradient(
                        v_active_n_all.mean()),
                    'rst_active_n_mean': jax.lax.stop_gradient(
                        rst_active_n_all.mean()),
                    'attn_q_active_tau_frac': jax.lax.stop_gradient(
                        q_active_all.mean()),
                    'attn_k_active_tau_frac': jax.lax.stop_gradient(
                        k_active_all.mean()),
                    'attn_qk_active_tau_frac': jax.lax.stop_gradient(
                        qk_active_all.mean()),
                    'attn_v_active_tau_frac': jax.lax.stop_gradient(
                        v_active_all.mean()),
                    'rst_active_tau_frac': jax.lax.stop_gradient(
                        rst_active_all.mean()),
                    'attn_q_active_tau_count': jax.lax.stop_gradient(
                        q_active_n_all.mean()),
                    'attn_k_active_tau_count': jax.lax.stop_gradient(
                        k_active_n_all.mean()),
                    'attn_qk_active_tau_count': jax.lax.stop_gradient(
                        qk_active_n_all.mean()),
                    'attn_v_active_tau_count': jax.lax.stop_gradient(
                        v_active_n_all.mean()),
                    'rst_active_tau_count': jax.lax.stop_gradient(
                        rst_active_n_all.mean()),
                    'attn_q_gate_mass': _sg_mean(q_gate_mass_all),
                    'attn_k_gate_mass': _sg_mean(k_gate_mass_all),
                    'attn_qk_gate_mass': _sg_mean(qk_gate_mass_all),
                    'attn_v_gate_mass': _sg_mean(v_gate_mass_all),
                    'rst_gate_mass': _sg_mean(rst_gate_mass_all),
                    'attn_q_gate_den': _sg_mean(q_gate_den_all),
                    'attn_k_gate_den': _sg_mean(k_gate_den_all),
                    'attn_qk_gate_den': _sg_mean(qk_gate_den_all),
                    'attn_v_gate_den': _sg_mean(v_gate_den_all),
                    'rst_gate_den': _sg_mean(rst_gate_den_all),
                    'attn_q_depth_active': _sg_mean(q_depth_active_all),
                    'attn_k_depth_active': _sg_mean(k_depth_active_all),
                    'attn_qk_depth_active': _sg_mean(qk_depth_active_all),
                    'attn_v_depth_active': _sg_mean(v_depth_active_all),
                    'rst_depth_active': _sg_mean(rst_depth_active_all),
                    'attn_q_gate_eff_n': _sg_mean(q_gate_eff_n_all),
                    'attn_k_gate_eff_n': _sg_mean(k_gate_eff_n_all),
                    'attn_qk_gate_eff_n': _sg_mean(qk_gate_eff_n_all),
                    'attn_v_gate_eff_n': _sg_mean(v_gate_eff_n_all),
                    'rst_gate_eff_n': _sg_mean(rst_gate_eff_n_all),
                    'attn_q_top1_gate_frac': _sg_mean(q_top1_gate_frac_all),
                    'attn_k_top1_gate_frac': _sg_mean(k_top1_gate_frac_all),
                    'attn_qk_top1_gate_frac': _sg_mean(
                        qk_top1_gate_frac_all),
                    'attn_v_top1_gate_frac': _sg_mean(
                        v_top1_gate_frac_all),
                    'rst_top1_gate_frac': _sg_mean(rst_top1_gate_frac_all),
                    'attn_q_den_floor_frac': _sg_mean(q_den_floor_all),
                    'attn_k_den_floor_frac': _sg_mean(k_den_floor_all),
                    'attn_qk_den_floor_frac': _sg_mean(qk_den_floor_all),
                    'attn_v_den_floor_frac': _sg_mean(v_den_floor_all),
                    'rst_den_floor_frac': _sg_mean(rst_den_floor_all),
                    'attn_q_tau_mean': jax.lax.stop_gradient(
                        q_tau_all.mean()),
                    'attn_k_tau_mean': jax.lax.stop_gradient(
                        k_tau_all.mean()),
                    'attn_v_tau_mean': jax.lax.stop_gradient(
                        v_tau_all.mean()),
                    'attn_qk_tau_mean': jax.lax.stop_gradient(
                        qk_tau_all.mean()),
                    'rst_tau_mean': jax.lax.stop_gradient(
                        rst_tau_all.mean()),
                    'attn_tau_mean': jax.lax.stop_gradient(
                        attn_tau_all.mean()),
                    'attn_out_norm': _sg_mean(attn_out_norm_all),
                    'rst_out_norm': _sg_mean(rst_out_norm_all),
                    'd_route': jnp.int32(self.d_route),
                    'admission_den_power': jnp.float32(admission_den_power),
                    'heat_kernel_beta': jnp.float32(heat_kernel_beta),
                    'attn_qk_admission_mass_mean': _sg_mean(qk_gate_mass_all),
                    'attn_qk_admission_mass_max': jax.lax.stop_gradient(
                        qk_admission_mass_max_all.max()),
                    'attn_qk_composition_den_mean': _sg_mean(qk_gate_den_all),
                    'attn_qk_composition_den_min': jax.lax.stop_gradient(
                        qk_composition_den_min_all.min()),
                    'attn_qk_composition_den_max': jax.lax.stop_gradient(
                        qk_composition_den_max_all.max()),
                    'attn_qk_composition_den_floor_frac': _sg_mean(
                        qk_den_floor_all),
                    'attn_qk_raw_srw_out_norm': _sg_mean(
                        qk_raw_srw_out_norm_all),
                    'attn_qk_normalized_srw_out_norm': _sg_mean(
                        qk_normalized_srw_out_norm_all),
                    'attn_qk_pool_scaled_srw_out_norm': _sg_mean(
                        qk_pool_scaled_srw_out_norm_all),
                    'attn_v_admission_mass_mean': _sg_mean(v_gate_mass_all),
                    'attn_v_admission_mass_max': jax.lax.stop_gradient(
                        v_admission_mass_max_all.max()),
                    'attn_v_composition_den_mean': _sg_mean(v_gate_den_all),
                    'attn_v_composition_den_min': jax.lax.stop_gradient(
                        v_composition_den_min_all.min()),
                    'attn_v_composition_den_max': jax.lax.stop_gradient(
                        v_composition_den_max_all.max()),
                    'attn_v_composition_den_floor_frac': _sg_mean(
                        v_den_floor_all),
                    'attn_v_raw_srw_out_norm': _sg_mean(
                        v_raw_srw_out_norm_all),
                    'attn_v_normalized_srw_out_norm': _sg_mean(
                        v_normalized_srw_out_norm_all),
                    'attn_v_pool_scaled_srw_out_norm': _sg_mean(
                        v_pool_scaled_srw_out_norm_all),
                    'rst_admission_mass_mean': _sg_mean(rst_gate_mass_all),
                    'rst_admission_mass_max': jax.lax.stop_gradient(
                        rst_admission_mass_max_all.max()),
                    'rst_composition_den_mean': _sg_mean(rst_gate_den_all),
                    'rst_composition_den_min': jax.lax.stop_gradient(
                        rst_composition_den_min_all.min()),
                    'rst_composition_den_max': jax.lax.stop_gradient(
                        rst_composition_den_max_all.max()),
                    'rst_composition_den_floor_frac': _sg_mean(
                        rst_den_floor_all),
                    'rst_raw_srw_out_norm': _sg_mean(
                        rst_raw_srw_out_norm_all),
                    'rst_normalized_srw_out_norm': _sg_mean(
                        rst_normalized_srw_out_norm_all),
                    'rst_pool_scaled_srw_out_norm': _sg_mean(
                        rst_pool_scaled_srw_out_norm_all),
                    'residual_norm': _sg_mean(residual_norm_all),
                }
                if analysis_return_residual:
                    output['final_residual'] = x
                if analysis_return_logits:
                    logical_vocab_size, embedding_vocab_size = (
                        self._vocab_sizes())
                    logits = self.token_emb.attend(x)
                    if embedding_vocab_size != logical_vocab_size:
                        logits = logits[..., :logical_vocab_size]
                    output['logits'] = logits
                if analysis_parity_debug:
                    output['parity_debug'] = {
                        'residual_input': parity_pre_layer_residual_all,
                        'q': parity_q_all,
                        'k': parity_k_all,
                        'v': parity_v_all,
                        'attention_update': parity_attention_update_all,
                        'rst': parity_rst_all,
                        'post_attention': (
                            parity_post_attention_residual_all),
                        'post_layer_residual': (
                            parity_post_layer_residual_all),
                    }
                if analysis_causal_trace:
                    output['causal_trace'] = causal_trace
                if analysis_trajectory_enabled:
                    output['operator_trajectory_trace'] = (
                        operator_trajectory_trace)
                if analysis_capture_contribution:
                    output['operator_route_contributions'] = (
                        route_contributions)
                if analysis_program_capture_contribution:
                    output['operator_program_contributions'] = (
                        program_route_contributions)
                return output

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
                    admission_den_power=admission_den_power_qk,
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
                 a_q_sparsity_diag, a_k_sparsity_diag) = attn_ret[58:67]
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
                     a_o_input_norm_max, a_o_out_norm_max) = attn_ret[67:104]
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
                    admission_den_power=admission_den_power_rst,
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
                 k_sparsity_diag) = rst_ret[:52]
                if analysis:
                    (k_raw_out_norm,
                     k_tau_std, k_tau_kernel_norm,
                     k_margin_band_wide, k_margin_band_mid,
                     k_skew, k_apt_std, k_entropy,
                     k_den_cost, k_selection_cost, k_current_cost,
                     k_op_key_n_max, k_rho_kurt, k_margin_band,
                     k_int_cap_frac) = rst_ret[52:67]
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

        def _sparsity_layer(diag_all, name):
            return diag_all[:, GATE_SPARSITY_DIAG_INDEX[name]]

        def _select_layer(diag_all, idx):
            return diag_all[:, idx]

        def _attn_route_tau_mean(route_idx):
            ndim = getattr(attn_tau_direct_all, 'ndim', 0)
            if ndim >= 5:
                return attn_tau_direct_all[..., route_idx, :].mean()
            if ndim >= 4:
                return attn_tau_direct_all[..., route_idx].mean()
            return _select_mean(attn_qk_select_diag_all, SELECT_TAU_MEAN)

        result = {
            'aux_loss': total_aux,
            'heat_kernel_beta': jnp.float32(heat_kernel_beta),
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
            'attn_q_tau_mean': _attn_route_tau_mean(0),
            'attn_k_tau_mean': _attn_route_tau_mean(1),
            'rst_tau_mean': rst_tau_mean_all.mean(),
            'attn_tau_abs_mean': jnp.abs(attn_tau_direct_all).mean(),
            'rst_tau_abs_mean': jnp.abs(rst_tau_direct_all).mean(),
            'attn_rho_mean': attn_rho_mean_all.mean(),
            'attn_rho_std': attn_rho_std_all.mean(),
            'attn_rho_max': attn_rho_max_all.max(),
            'attn_tau_min': attn_tau_direct_all.min(),
            'attn_tau_max': attn_tau_direct_all.max(),
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
            'rst_tau_min': rst_tau_direct_all.min(),
            'rst_tau_max': rst_tau_direct_all.max(),
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
        q_active_frac = _attn_core_mean(ATTN_SPLIT_Q_ACTIVE_FRAC)
        k_active_frac = _attn_core_mean(ATTN_SPLIT_K_ACTIVE_FRAC)
        q_active_count = _attn_core_mean(ATTN_SPLIT_Q_ACTIVE_N_MEAN)
        k_active_count = _attn_core_mean(ATTN_SPLIT_K_ACTIVE_N_MEAN)
        result.update({
            'attn_q_active_tau_frac': q_active_frac,
            'attn_k_active_tau_frac': k_active_frac,
            'attn_qk_active_tau_frac': jnp.float32(0.5) * (
                q_active_frac + k_active_frac),
            'attn_v_active_tau_frac': attn_v_active_all.mean(),
            'rst_active_tau_frac': rst_active_all.mean(),
            'attn_q_active_tau_count': q_active_count,
            'attn_k_active_tau_count': k_active_count,
            'attn_qk_active_tau_count': jnp.float32(0.5) * (
                q_active_count + k_active_count),
            'attn_v_active_tau_count': _attn_core_mean(
                ATTN_SPLIT_V_ACTIVE_N_MEAN),
            'rst_active_tau_count': rst_active_n_mean_all.mean(),
        })
        if analysis and not self.is_initializing():
            _residual_norm = jnp.linalg.norm(x, axis=-1).mean()
            _emb_norm = jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean()
            _o_proj_norm = jnp.linalg.norm(
                stacked['attn']['expand_O']['kernel'], axis=(-2, -1)).mean()
            _attn_logit_max_layer = jnp.argmax(attn_logit_max_all)
            result.update({
                'per_layer_attn_qk_active_tau_frac': _sparsity_layer(
                    attn_qk_sparsity_diag_all, 'active_tau_frac'),
                'per_layer_attn_qk_admission_active_eps_1e_2_frac': _sparsity_layer(
                    attn_qk_sparsity_diag_all, 'admission_active_eps_1e_2_frac'),
                'per_layer_attn_qk_active_eps_1e_2_frac': _sparsity_layer(
                    attn_qk_sparsity_diag_all, 'active_eps_1e_2_frac'),
                'per_layer_attn_qk_mass_eps_1e_2': _sparsity_layer(
                    attn_qk_sparsity_diag_all, 'mass_eps_1e_2'),
                'per_layer_attn_qk_margin_band_pos': _sparsity_layer(
                    attn_qk_sparsity_diag_all, 'margin_band_pos'),
                'per_layer_attn_qk_active_n_mean': attn_split_core_all[
                    :, ATTN_SPLIT_QK_ACTIVE_N_MEAN],
                'per_layer_attn_qk_gate_eff_n': attn_split_core_all[
                    :, ATTN_SPLIT_QK_GATE_EFF_N],
                'per_layer_attn_qk_gate_eff_ratio': attn_split_core_all[
                    :, ATTN_SPLIT_QK_GATE_EFF_RATIO],
                'per_layer_attn_qk_execution_top1_frac': attn_split_core_all[
                    :, ATTN_SPLIT_QK_TOP1_GATE_FRAC],
                'per_layer_attn_qk_execution_top1_frac_max': attn_split_core_all[
                    :, ATTN_SPLIT_QK_TOP1_GATE_FRAC_MAX],
                'per_layer_attn_qk_rho_mean': _select_layer(
                    attn_qk_select_diag_all, SELECT_RHO_MEAN),
                'per_layer_attn_qk_rho_std': _select_layer(
                    attn_qk_select_diag_all, SELECT_RHO_STD),
                'per_layer_attn_qk_rho_max': _select_layer(
                    attn_qk_select_diag_all, SELECT_RHO_MAX),
                'per_layer_attn_qk_tau_mean': _select_layer(
                    attn_qk_select_diag_all, SELECT_TAU_MEAN),
                'per_layer_attn_qk_tau_min': _select_layer(
                    attn_qk_select_diag_all, SELECT_TAU_MIN),
                'per_layer_attn_qk_tau_max': _select_layer(
                    attn_qk_select_diag_all, SELECT_TAU_MAX),
                'per_layer_attn_qk_selection_margin_mean': _select_layer(
                    attn_qk_select_diag_all, SELECT_SELECTION_MARGIN_MEAN),
                'per_layer_attn_qk_positive_margin_mean': _select_layer(
                    attn_qk_select_diag_all, SELECT_POSITIVE_MARGIN_MEAN),
                'per_layer_attn_qk_selected_frac': _select_layer(
                    attn_qk_select_diag_all, SELECT_SELECTED_FRAC),
                'per_layer_attn_q_active_tau_frac': _sparsity_layer(
                    attn_q_sparsity_diag_all, 'active_tau_frac'),
                'per_layer_attn_q_admission_active_eps_1e_2_frac': _sparsity_layer(
                    attn_q_sparsity_diag_all, 'admission_active_eps_1e_2_frac'),
                'per_layer_attn_q_active_eps_1e_2_frac': _sparsity_layer(
                    attn_q_sparsity_diag_all, 'active_eps_1e_2_frac'),
                'per_layer_attn_q_active_n_mean': attn_split_core_all[
                    :, ATTN_SPLIT_Q_ACTIVE_N_MEAN],
                'per_layer_attn_k_active_tau_frac': _sparsity_layer(
                    attn_k_sparsity_diag_all, 'active_tau_frac'),
                'per_layer_attn_k_admission_active_eps_1e_2_frac': _sparsity_layer(
                    attn_k_sparsity_diag_all, 'admission_active_eps_1e_2_frac'),
                'per_layer_attn_k_active_eps_1e_2_frac': _sparsity_layer(
                    attn_k_sparsity_diag_all, 'active_eps_1e_2_frac'),
                'per_layer_attn_k_active_n_mean': attn_split_core_all[
                    :, ATTN_SPLIT_K_ACTIVE_N_MEAN],
                'per_layer_attn_v_active_tau_frac': _sparsity_layer(
                    attn_v_sparsity_diag_all, 'active_tau_frac'),
                'per_layer_attn_v_admission_active_eps_1e_2_frac': _sparsity_layer(
                    attn_v_sparsity_diag_all, 'admission_active_eps_1e_2_frac'),
                'per_layer_attn_v_active_eps_1e_2_frac': _sparsity_layer(
                    attn_v_sparsity_diag_all, 'active_eps_1e_2_frac'),
                'per_layer_attn_v_mass_eps_1e_2': _sparsity_layer(
                    attn_v_sparsity_diag_all, 'mass_eps_1e_2'),
                'per_layer_attn_v_margin_band_pos': _sparsity_layer(
                    attn_v_sparsity_diag_all, 'margin_band_pos'),
                'per_layer_attn_v_active_n_mean': attn_split_core_all[
                    :, ATTN_SPLIT_V_ACTIVE_N_MEAN],
                'per_layer_attn_v_gate_eff_n': attn_split_core_all[
                    :, ATTN_SPLIT_V_GATE_EFF_N],
                'per_layer_attn_v_gate_eff_ratio': attn_split_core_all[
                    :, ATTN_SPLIT_V_GATE_EFF_RATIO],
                'per_layer_attn_v_execution_top1_frac': attn_split_core_all[
                    :, ATTN_SPLIT_V_TOP1_GATE_FRAC],
                'per_layer_attn_v_execution_top1_frac_max': attn_split_core_all[
                    :, ATTN_SPLIT_V_TOP1_GATE_FRAC_MAX],
                'per_layer_attn_v_rho_mean': _select_layer(
                    attn_v_select_diag_all, SELECT_RHO_MEAN),
                'per_layer_attn_v_rho_std': _select_layer(
                    attn_v_select_diag_all, SELECT_RHO_STD),
                'per_layer_attn_v_rho_max': _select_layer(
                    attn_v_select_diag_all, SELECT_RHO_MAX),
                'per_layer_attn_v_tau_mean': _select_layer(
                    attn_v_select_diag_all, SELECT_TAU_MEAN),
                'per_layer_attn_v_tau_min': _select_layer(
                    attn_v_select_diag_all, SELECT_TAU_MIN),
                'per_layer_attn_v_tau_max': _select_layer(
                    attn_v_select_diag_all, SELECT_TAU_MAX),
                'per_layer_attn_v_selection_margin_mean': _select_layer(
                    attn_v_select_diag_all, SELECT_SELECTION_MARGIN_MEAN),
                'per_layer_attn_v_positive_margin_mean': _select_layer(
                    attn_v_select_diag_all, SELECT_POSITIVE_MARGIN_MEAN),
                'per_layer_attn_v_selected_frac': _select_layer(
                    attn_v_select_diag_all, SELECT_SELECTED_FRAC),
                'per_layer_rst_active_tau_frac': _sparsity_layer(
                    rst_sparsity_diag_all, 'active_tau_frac'),
                'per_layer_rst_admission_active_eps_1e_2_frac': _sparsity_layer(
                    rst_sparsity_diag_all, 'admission_active_eps_1e_2_frac'),
                'per_layer_rst_active_eps_1e_2_frac': _sparsity_layer(
                    rst_sparsity_diag_all, 'active_eps_1e_2_frac'),
                'per_layer_rst_mass_eps_1e_2': _sparsity_layer(
                    rst_sparsity_diag_all, 'mass_eps_1e_2'),
                'per_layer_rst_margin_band_pos': _sparsity_layer(
                    rst_sparsity_diag_all, 'margin_band_pos'),
                'per_layer_rst_active_n_mean': rst_active_n_mean_all,
                'per_layer_rst_gate_eff_n': rst_gate_eff_n_all,
                'per_layer_rst_gate_eff_ratio': rst_gate_eff_ratio_all,
                'per_layer_rst_execution_top1_frac': rst_top1_gate_frac_all,
                'per_layer_rst_execution_top1_frac_max': rst_top1_gate_frac_max_all,
                'per_layer_rst_rho_mean': rst_rho_mean_all,
                'per_layer_rst_rho_std': rst_rho_std_all,
                'per_layer_rst_rho_max': rst_rho_max_all,
                'per_layer_rst_tau_mean': rst_tau_mean_all,
                'per_layer_rst_tau_min': rst_tau_direct_all.reshape(
                    (rst_tau_direct_all.shape[0], -1)).min(axis=1),
                'per_layer_rst_tau_max': rst_tau_direct_all.reshape(
                    (rst_tau_direct_all.shape[0], -1)).max(axis=1),
                'per_layer_rst_tau_floor_mean': rst_tau_floor_mean_all,
                'per_layer_rst_tau_min_hit_frac': rst_tau_min_hit_frac_all,
                'per_layer_rst_selection_margin_mean': rst_selection_margin_mean_all,
                'per_layer_rst_positive_margin_mean': rst_positive_margin_mean_all,
                'per_layer_rst_selected_frac': rst_selected_frac_all,
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
            (loss, per_token_ce, correct, valid_count,
             logit_abs_max, logit_norm_mean, logit_mean, logit_std,
             valid_mask, per_token_correct) = _compute_vocab_ce(x)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
            result['logit_max'] = logit_abs_max
            result['logit_norm_mean'] = logit_norm_mean
            result['logit_mean'] = logit_mean
            result['logit_std'] = logit_std
            result['per_token_ce'] = per_token_ce
            if _eval_stats_enabled:
                result['per_token_correct'] = per_token_correct
            result['valid_mask'] = valid_mask
        else:
            vocab_argmax = (
                sharded_fns.get("vocab_argmax")
                if isinstance(sharded_fns, dict) else None)
            if vocab_argmax is not None:
                result['argmax_token_ids'] = vocab_argmax(
                    x, self.token_emb.embedding)
            elif vp_embed is not None:
                raise NotImplementedError(
                    "Full logits are disabled on the vocab-parallel v4166 "
                    "path. Pass labels or run without sharded_fns.")
            else:
                logical_vocab_size, embedding_vocab_size = self._vocab_sizes()
                logits = self.token_emb.attend(x)
                if embedding_vocab_size != logical_vocab_size:
                    logits = logits[..., :logical_vocab_size]
                result['logits'] = logits

        if analysis_return_residual:
            result['final_residual'] = x
        return result

    def analysis_forward_with_operator_suppression(
            self, input_ids, selected_global_operator_id, target_layer,
            target_positions, route_selector, *, labels=None,
            attention_mask=None, apply_suppression=True,
            return_residual=True, **production_kwargs):
        """Suppress one execution numerator inside the production SRW core.

        Admission and its denominator remain unmodified.  Operator ids and
        target positions are per-example arrays; ``route_selector`` is 0=attention_q,
        1=attention_k, 2=attention_v, or 3=RST.
        """
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_contribution=selected_global_operator_id,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=route_selector,
            analysis_intervention_enabled=apply_suppression,
            analysis_return_residual=return_residual,
            **production_kwargs,
        )

    def analysis_forward_with_operator_group_suppression(
            self, input_ids, selected_global_operator_ids, target_layer,
            target_positions, route_selector, *, labels=None,
            attention_mask=None, apply_suppression=True,
            return_residual=True, **production_kwargs):
        """Suppress one fixed-width, ``-1`` padded operator-id group.

        Every group size, including the all-``-1`` size-zero baseline, uses
        the same ``[B, M]`` input shape and therefore the same compiled graph.
        attention_q/attention_k route selection and all production admission/denominator semantics
        are identical to single-operator suppression.
        """
        selected_global_operator_ids = _analysis_int32_array(
            selected_global_operator_ids,
            name="selected_global_operator_ids")
        if selected_global_operator_ids.ndim != 2:
            raise ValueError(
                "group operator ids must have shape [B, M], got "
                f"{selected_global_operator_ids.shape}")
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_contribution=selected_global_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=route_selector,
            analysis_intervention_enabled=apply_suppression,
            analysis_return_residual=return_residual,
            **production_kwargs,
        )

    def analysis_forward_with_circuit_retention(
            self, input_ids, keep_qk, keep_v, keep_rst, *,
            mode, position_mask=None, labels=None, attention_mask=None,
            return_residual=True, **production_kwargs):
        """Retain a circuit across every layer and route in one forward.

        ``conditional_execution_sufficiency`` keeps the selected execution
        numerator while retaining the production admission denominator.
        ``autonomous_subcircuit_sufficiency`` restricts both numerator and
        admission denominator to the selected circuit.  Operator masks are
        dense boolean arrays with shapes ``[L,2,Nqk]``, ``[L,Nv]``, and
        ``[L,Nrst]``.  ``position_mask`` selects positions where retention is
        active; unselected positions execute the full model.
        """
        modes = {
            "conditional_execution_sufficiency": 1,
            "autonomous_subcircuit_sufficiency": 2,
        }
        if mode not in modes:
            raise ValueError(
                f"Unsupported circuit retention mode={mode!r}; "
                f"known={','.join(modes)}")
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_keep_qk=keep_qk,
            analysis_keep_v=keep_v,
            analysis_keep_rst=keep_rst,
            analysis_position_mask=position_mask,
            analysis_retention_mode=modes[mode],
            analysis_return_residual=return_residual,
            **production_kwargs,
        )

    def analysis_capture_operator_group_contribution(
            self, input_ids, selected_global_operator_ids, target_layer,
            target_positions, route_selector, *, labels=None,
            attention_mask=None, return_residual=False,
            **production_kwargs):
        """Capture the selected route-native post-denominator contribution.

        The returned ``operator_route_contributions`` are defined by the exact
        production route minus its same-denominator group-suppressed
        counterfactual. They are float32, already multiplied by the learned
        pool scale, and taken before attention head reshaping or residual
        addition.
        """
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_contribution=selected_global_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=route_selector,
            analysis_intervention_enabled=False,
            analysis_capture_contribution=True,
            analysis_return_residual=return_residual,
            **production_kwargs,
        )

    def analysis_forward_with_operator_interchange(
            self, input_ids, selected_global_operator_ids, target_layer,
            target_positions, route_selector, source_contribution, *,
            labels=None, attention_mask=None, return_residual=True,
            **production_kwargs):
        """Patch ``base - selected(base) + selected(source)`` in production.

        ``source_contribution`` must come from
        :meth:`analysis_capture_operator_group_contribution` for the same
        layer, route, operator group, and benchmark-audited token position.
        The first two terms use the exact production suppression
        counterfactual, so an explicit zero source is suppression, not no-op.
        """
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_contribution=selected_global_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=route_selector,
            analysis_intervention_enabled=False,
            analysis_interchange_source=source_contribution,
            analysis_interchange_enabled=True,
            analysis_return_residual=return_residual,
            **production_kwargs,
        )

    def analysis_capture_operator_program_contributions(
            self, input_ids, *, selected_ids_q, selected_ids_k,
            selected_ids_v, selected_ids_rst, selected_valid_q,
            selected_valid_k, selected_valid_v, selected_valid_rst,
            target_positions, labels=None, attention_mask=None,
            **production_kwargs):
        """Capture every layer/route selected contribution in one scan.

        IDs are evaluated against each layer's current unpatched production
        state. Invalid padding is represented by id 0 plus a false validity
        bit and therefore contributes exactly zero.
        """
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_program_ids_q=selected_ids_q,
            analysis_program_ids_k=selected_ids_k,
            analysis_program_ids_v=selected_ids_v,
            analysis_program_ids_rst=selected_ids_rst,
            analysis_program_valid_q=selected_valid_q,
            analysis_program_valid_k=selected_valid_k,
            analysis_program_valid_v=selected_valid_v,
            analysis_program_valid_rst=selected_valid_rst,
            analysis_program_target_positions=target_positions,
            analysis_program_mode=jnp.int32(0),
            analysis_program_capture_contribution=True,
            **production_kwargs,
        )

    def analysis_forward_with_operator_program(
            self, input_ids, *, selected_ids_q, selected_ids_k,
            selected_ids_v, selected_ids_rst, selected_valid_q,
            selected_valid_k, selected_valid_v, selected_valid_rst,
            target_positions, program_mode,
            source_contribution_q=None, source_contribution_k=None,
            source_contribution_v=None, source_contribution_rst=None,
            labels=None, attention_mask=None, return_residual=False,
            **production_kwargs):
        """Execute dynamic replay, ablation, or transplant in production.

        ``program_mode`` is a dynamic scalar shared by one compiled forward:
        0 production, 1 own-ID replay, 2 own-ID ablation, 3 source-ID replay,
        and 4 source-contribution transplant. Only each example's target
        answer position is patched; all other positions remain production.
        """
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_program_ids_q=selected_ids_q,
            analysis_program_ids_k=selected_ids_k,
            analysis_program_ids_v=selected_ids_v,
            analysis_program_ids_rst=selected_ids_rst,
            analysis_program_valid_q=selected_valid_q,
            analysis_program_valid_k=selected_valid_k,
            analysis_program_valid_v=selected_valid_v,
            analysis_program_valid_rst=selected_valid_rst,
            analysis_program_target_positions=target_positions,
            analysis_program_mode=program_mode,
            analysis_program_source_q=source_contribution_q,
            analysis_program_source_k=source_contribution_k,
            analysis_program_source_v=source_contribution_v,
            analysis_program_source_rst=source_contribution_rst,
            analysis_return_residual=return_residual,
            **production_kwargs,
        )

    def analysis_forward_with_paired_operator_trajectory(
            self, input_ids, *, trajectory_positions,
            trajectory_position_valid, selected_ids_q, selected_ids_k,
            selected_ids_v, selected_ids_rst, selected_valid_q,
            selected_valid_k, selected_valid_v, selected_valid_rst,
            replay_full_active=False, patch_layers, patch_positions,
            patch_stages, patch_enabled, patch_values, labels=None,
            attention_mask=None, return_residual=True,
            return_logits=False, **production_kwargs):
        """Capture or replay an exact S2-to-answer paired trajectory.

        The per-route ID tensors are fixed ``[L,B,T,K]`` schedules.  A first
        pass supplies all-false validity and captures numerator-active IDs;
        the fail-closed replay pass supplies those exact IDs and sets
        ``replay_full_active=True``.  State and route interventions share one
        fixed ``[B,P]`` patch interface with the stage codes exported in
        :data:`TRAJECTORY_PATCH_STAGES`.
        """
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_return_residual=return_residual,
            analysis_return_logits=return_logits,
            analysis_trajectory_positions=trajectory_positions,
            analysis_trajectory_position_valid=trajectory_position_valid,
            analysis_trajectory_ids_q=selected_ids_q,
            analysis_trajectory_ids_k=selected_ids_k,
            analysis_trajectory_ids_v=selected_ids_v,
            analysis_trajectory_ids_rst=selected_ids_rst,
            analysis_trajectory_valid_q=selected_valid_q,
            analysis_trajectory_valid_k=selected_valid_k,
            analysis_trajectory_valid_v=selected_valid_v,
            analysis_trajectory_valid_rst=selected_valid_rst,
            analysis_trajectory_replay_enabled=replay_full_active,
            analysis_trajectory_patch_layers=patch_layers,
            analysis_trajectory_patch_positions=patch_positions,
            analysis_trajectory_patch_stages=patch_stages,
            analysis_trajectory_patch_enabled=patch_enabled,
            analysis_trajectory_patch_values=patch_values,
            **production_kwargs,
        )

    def analysis_forward_with_trajectory_patches(
            self, input_ids, *, patch_layers, patch_positions,
            patch_stages, patch_enabled, patch_values, labels=None,
            attention_mask=None, return_residual=False,
            return_logits=False, **production_kwargs):
        """Apply a fixed batch-native patch schedule without trace capture."""
        return self(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            minimal_train=True,
            analysis_return_residual=return_residual,
            analysis_return_logits=return_logits,
            analysis_trajectory_patch_layers=patch_layers,
            analysis_trajectory_patch_positions=patch_positions,
            analysis_trajectory_patch_stages=patch_stages,
            analysis_trajectory_patch_enabled=patch_enabled,
            analysis_trajectory_patch_values=patch_values,
            **production_kwargs,
        )

    def get_config(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        logical_vocab_size, embedding_vocab_size = self._vocab_sizes()
        (legacy_power, qk_power, v_power, rst_power) = (
            _resolve_v417x_admission_den_powers(
                self.admission_den_power,
                self.admission_den_power_qk,
                self.admission_den_power_v,
                self.admission_den_power_rst,
                context="DAWN_SRW_V4171 config serialization"))
        cfg = {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'logical_vocab_size': logical_vocab_size,
            'vocab_size_padded': embedding_vocab_size,
            'd_route': self.d_route,
            'operator_key_mode': self.operator_key_mode,
            'operator_query_mode': OPERATOR_QUERY_MODE,
            'admission_den_power': legacy_power,
            'admission_den_power_qk': qk_power,
            'admission_den_power_v': v_power,
            'admission_den_power_rst': rst_power,
            'srw_composition_mode': self.srw_composition_mode,
            'heat_kernel_beta': self.heat_kernel_beta,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_rst': n_rst_eff,
            'n_know': n_rst_eff,
        }
        return cfg

    def get_model_info(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        logical_vocab_size, embedding_vocab_size = self._vocab_sizes()
        qk_scale, v_scale, rst_scale = _pool_output_scales(
            self.d_model, self.n_layers)
        (admission_den_power, qk_den_power, v_den_power,
         rst_den_power) = _resolve_v417x_admission_den_powers(
            self.admission_den_power,
            self.admission_den_power_qk,
            self.admission_den_power_v,
            self.admission_den_power_rst,
            context="DAWN_SRW_V4171 model info")
        mode, admission_den_power, heat_kernel_beta = (
            _validate_v4171_composition_settings(
            self.srw_composition_mode,
            self.admission_den_power,
            self.heat_kernel_beta,
            context="DAWN_SRW_V4171 model info"))
        if mode == "quadratic":
            composition_info = [
                f"  mode={mode}",
                "  angular_amplitude=linear_cap_depth",
                "  admission_weight=amplitude^2",
                "  total_weight=sum(unpruned_admission)",
                ("  den=max(total_weight,1e-12)^"
                 "admission_den_power"),
                "  numerator=pruned_execution_weight",
                "  live_den_gradient=true",
            ]
        elif mode == "heat_energy":
            composition_info = [
                f"  mode={mode}",
                "  support=rho>tau",
                ("  cap_amplitude=clip((rho-tau)/"
                 "max(1-tau,1e-4),0,1)"),
                ("  heat_amplitude=(exp(beta*cap_amplitude)-1)/"
                 "(exp(beta)-1)"),
                f"  beta={heat_kernel_beta:g}",
                "  energy_weight=heat_amplitude^2",
                "  total_energy=sum(unpruned_energy_weight)",
                "  numerator=pruned_energy_weight",
                ("  denominator=max(total_energy,1e-12)^"
                 "admission_den_power"),
                ("  beta_to_zero_limit=quadratic"),
                "  live_den_gradient=true",
            ]
        else:
            composition_info = [
                f"  mode={mode}",
                "  angular_amplitude=linear_cap_depth",
                "  admission_weight=amplitude",
                ("  den=max(sum(unpruned_admission),1)^"
                 "admission_den_power"),
                "  numerator=pruned_execution_weight",
                "  live_den_gradient=true",
            ]
        return [
            f"DAWN-SRW {self.__version__}",
            f"  d_model={self.d_model}, d_route={self.d_route}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            "Operator address:",
            f"  mode={self.operator_key_mode}",
            f"  d_route={self.d_route}",
            ("  independent_per_operator=true"
             if self.operator_key_mode == OPERATOR_KEY_MODE_LEARNED
             else "  probe_scope=shared_across_qk_v_rst"),
            "  live_gradient=true",
            "  full_rw_execution=true",
            f"  vocab logical/padded={logical_vocab_size}/{embedding_vocab_size}",
            f"  Attention-QK: {self.n_qk}, Attention-V: {self.n_v}, RST: {n_rst_eff}",
            ("  Selection: cosine(direct state query, learned operator embedding)"
             if self.operator_key_mode == OPERATOR_KEY_MODE_LEARNED
             else "  Selection: cosine(direct state query, live bilinear RW key)"),
            "Execution:",
            "  full rank-1 read/write operator",
            "Composition:",
            *composition_info,
            ("  den_power["
             f"qk={qk_den_power:g} v={v_den_power:g} "
             f"rst={rst_den_power:g}]"),
            f"  heat_kernel_beta={heat_kernel_beta:g}",
            "  runtime_source=model",
            "  Pool scales: fixed depth-scaled "
            f"(qk={float(qk_scale):.6g}, v={float(v_scale):.6g}, "
            f"rst={float(rst_scale):.6g})",
        ]


DAWN = DAWN_SRW_V4171


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


def _logical_vocab_size_from_model_cfg(model_cfg, fallback):
    if hasattr(model_cfg, 'get'):
        return int(model_cfg.get('logical_vocab_size', fallback))
    return int(fallback)


def _slice_logits_to_logical_vocab(logits, model_cfg):
    logical = _logical_vocab_size_from_model_cfg(model_cfg, logits.shape[-1])
    return logits[..., :logical]


def _angular_execution_kwargs_from_model_cfg(model_cfg):
    """Extract v4171 canonical execution settings for inference."""
    (admission_den_power, admission_den_power_qk,
     admission_den_power_v, admission_den_power_rst) = (
        _resolve_v417x_admission_den_powers(
            model_cfg.get(
                'admission_den_power', DEFAULT_ADMISSION_DEN_POWER),
            model_cfg.get('admission_den_power_qk'),
            model_cfg.get('admission_den_power_v'),
            model_cfg.get('admission_den_power_rst'),
            context="v417x inference model config"))
    (srw_composition_mode, admission_den_power,
     heat_kernel_beta) = (
        _validate_v4171_composition_settings(
            model_cfg.get(
                'srw_composition_mode', DEFAULT_SRW_COMPOSITION_MODE),
            admission_den_power,
            model_cfg.get('heat_kernel_beta', DEFAULT_HEAT_KERNEL_BETA),
            context="v4171 inference model config"))
    return {
        'soft_gate_temperature': float(
            model_cfg.get('soft_gate_temperature', 0.07)),
        'soft_gate_boundary_power': float(
            model_cfg.get('soft_gate_boundary_power', 4.0)),
        'admission_den_power': admission_den_power,
        'admission_den_power_qk': admission_den_power_qk,
        'admission_den_power_v': admission_den_power_v,
        'admission_den_power_rst': admission_den_power_rst,
        'srw_composition_mode': srw_composition_mode,
        'heat_kernel_beta': heat_kernel_beta,
        'execution_prune_eps': float(model_cfg.get('execution_prune_eps', 0.0)),
        'soft_gate_effective_active_eps': float(
            model_cfg.get('soft_gate_effective_active_eps', 1.0e-6)),
    }


def _query_geometry_stats(operator_query, prefix, max_sample=4096):
    operator_query = _forward_unit_direction(operator_query.astype(jnp.float32))
    flat = operator_query.reshape((-1, operator_query.shape[-1]))
    stride = max(1, flat.shape[0] // int(max_sample))
    sampled = flat[::stride][:int(max_sample)]
    norms = jnp.linalg.norm(sampled, axis=-1)
    centered = sampled - sampled.mean(axis=0, keepdims=True)
    singular = jnp.linalg.svd(
        centered, full_matrices=False, compute_uv=False)
    energy = jnp.sum(jnp.square(singular))
    effective_rank = energy / (jnp.max(jnp.square(singular)) + 1e-8)
    return {
        f'{prefix}_query_norm_mean': norms.mean(),
        f'{prefix}_query_norm_min': norms.min(),
        f'{prefix}_query_norm_max': norms.max(),
        f'{prefix}_query_effective_rank': effective_rank,
    }


def _tau_init_calibration_scores(params, input_ids, max_tokens=128):
    """Sample fresh-init cosine scores without changing forward semantics.

    The sample uses the first block's freshly initialized normalized route
    states and the shared v4171 router/pool parameters. Rho follows the
    train path exactly: direct state projections against independent learned
    live-gradient operator-address embeddings.
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
    attn_operator_queries = (
        attn_x @ router['proj_attn']['kernel']
        + router['proj_attn']['bias'])
    q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)
    rst_operator_query = (
        rst_x @ router['proj_rst']['kernel']
        + router['proj_rst']['bias'])

    def _selection_rho(operator_query, operator_keys):
        q_unit = _forward_unit_direction(
            operator_query.astype(jnp.float32)).astype(jnp.bfloat16)
        op_key_unit = _forward_unit_direction(
            operator_keys.astype(jnp.float32)).astype(jnp.bfloat16)
        return (q_unit @ op_key_unit.T).astype(jnp.float32)

    pool = params['neuron_pool']
    operator_keys_by_pool = _ensure_pool_operator_keys(pool)
    qk_operator_keys = operator_keys_by_pool['attn_qk_op_key']
    v_operator_keys = operator_keys_by_pool['attn_v_op_key']
    rst_operator_keys = operator_keys_by_pool['rst_op_key']
    return {
        'q': _selection_rho(q_operator_query, qk_operator_keys),
        'k': _selection_rho(k_operator_query, qk_operator_keys),
        'v': _selection_rho(v_operator_query, v_operator_keys),
        'rst': _selection_rho(rst_operator_query, rst_operator_keys),
    }


def _query_geometry_diagnostics(params, input_ids, max_tokens=4096):
    """Heavy v4171 query geometry on deterministic first-block tokens."""
    max_tokens = int(max_tokens)
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim != 2 or max_tokens <= 0:
        raise ValueError(
            "query geometry requires rank-2 input_ids and max_tokens > 0; "
            f"shape={input_ids.shape}, max_tokens={max_tokens}")
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
    q_operator_query, k_operator_query, v_operator_query = jnp.split(
        attn_x @ router['proj_attn']['kernel']
        + router['proj_attn']['bias'], 3, axis=-1)
    rst_operator_query = (
        rst_x @ router['proj_rst']['kernel']
        + router['proj_rst']['bias'])
    out = {}
    out.update(_query_geometry_stats(q_operator_query, 'q', max_tokens))
    out.update(_query_geometry_stats(k_operator_query, 'k', max_tokens))
    out.update(_query_geometry_stats(v_operator_query, 'v', max_tokens))
    out.update(_query_geometry_stats(rst_operator_query, 'rst', max_tokens))
    return out


def _angular_relation(operator_query, operator_keys):
    operator_query_direction = _forward_unit_direction(
        operator_query.astype(jnp.float32))
    operator_key_directions = _forward_unit_direction(
        operator_keys.astype(jnp.float32))
    return (operator_query_direction @ operator_key_directions.T).astype(
        jnp.float32)


def _angular_execution(operator_query, operator_keys, raw_tau, raw_scan_offset=None,
                     soft_gate_temperature=0.07,
                     soft_gate_boundary_power=4.0,
                     execution_prune_eps=0.0,
                     soft_gate_effective_active_eps=1.0e-6,
                     srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
                     heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    rho = _angular_relation(operator_query, operator_keys)
    tau = _tau_from_param(raw_tau)
    return _compute_admission_drive(
        rho, tau, soft_gate_temperature,
        boundary_power=soft_gate_boundary_power,
        effective_active_eps=soft_gate_effective_active_eps,
        execution_prune_eps=execution_prune_eps,
        srw_composition_mode=srw_composition_mode,
        heat_kernel_beta=heat_kernel_beta)


def _angular_execution_weight(operator_query, operator_keys, raw_tau, raw_scan_offset=None,
                  soft_gate_temperature=0.07,
                  soft_gate_boundary_power=4.0,
                  execution_prune_eps=0.0,
                  soft_gate_effective_active_eps=1.0e-6,
                  srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
                  heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Canonical v4166 execution_weight for non-sharded inference helpers."""
    _, _, _, execution_weight, _ = _angular_execution(
        operator_query, operator_keys, raw_tau, raw_scan_offset,
        soft_gate_temperature=soft_gate_temperature,
        soft_gate_boundary_power=soft_gate_boundary_power,
        execution_prune_eps=execution_prune_eps,
        soft_gate_effective_active_eps=soft_gate_effective_active_eps,
        srw_composition_mode=srw_composition_mode,
        heat_kernel_beta=heat_kernel_beta)
    return execution_weight.astype(jnp.float32)


def _split_admission_den_kwargs(
        angular_execution_kwargs, admission_den_pool=None):
    execution_kwargs = dict(angular_execution_kwargs)
    legacy_power = execution_kwargs.pop(
        'admission_den_power', DEFAULT_ADMISSION_DEN_POWER)
    pool_powers = {
        pool: execution_kwargs.pop(f'admission_den_power_{pool}', legacy_power)
        for pool in ('qk', 'v', 'rst')
    }
    if admission_den_pool is not None and admission_den_pool not in pool_powers:
        raise ValueError(
            f"unsupported admission denominator pool {admission_den_pool!r}")
    admission_den_power = jnp.float32(
        legacy_power if admission_den_pool is None
        else pool_powers[admission_den_pool])
    return execution_kwargs, admission_den_power


def _srw_inference(
        state, operator_query, operator_keys, raw_tau, raw_scan_offset,
        read_vectors, write_vectors, admission_den_pool=None,
        **angular_execution_kwargs):
    """Non-chunked SRW for inference."""
    # Selection uses d_route operator keys; execution uses d_model RW dirs.
    read_directions = _forward_unit_direction(
        read_vectors.astype(jnp.float32))
    write_directions = _forward_unit_direction(
        write_vectors.astype(jnp.float32))
    execution_kwargs, admission_den_power = _split_admission_den_kwargs(
        angular_execution_kwargs, admission_den_pool)
    _, admission, _, execution_weight, _ = _angular_execution(
        operator_query, operator_keys, raw_tau, raw_scan_offset, **execution_kwargs)

    read_activations = state.astype(jnp.float32) @ read_directions.T
    weighted_read_activations = execution_weight * read_activations
    raw_state_transition = weighted_read_activations @ write_directions
    composition_den = _composition_den(
        admission.sum(axis=-1, keepdims=True), admission_den_power,
        execution_kwargs.get(
            'srw_composition_mode', DEFAULT_SRW_COMPOSITION_MODE))
    state_transition = raw_state_transition.astype(jnp.float32) / composition_den
    return state_transition.astype(jnp.float32)


def _srw_inference_with_gates(
        state, operator_query, operator_keys, raw_tau, raw_scan_offset,
        read_vectors, write_vectors, admission_den_pool=None,
        **angular_execution_kwargs):
    """Like _srw_inference but also returns gate and normalized gate."""
    # Selection uses d_route operator keys; execution uses d_model RW dirs.
    read_directions = _forward_unit_direction(
        read_vectors.astype(jnp.float32))
    write_directions = _forward_unit_direction(
        write_vectors.astype(jnp.float32))
    execution_kwargs, admission_den_power = _split_admission_den_kwargs(
        angular_execution_kwargs, admission_den_pool)
    _, admission, _, execution_weight, _ = _angular_execution(
        operator_query, operator_keys, raw_tau, raw_scan_offset, **execution_kwargs)
    composition_den = _composition_den(
        admission.sum(axis=-1, keepdims=True), admission_den_power,
        execution_kwargs.get(
            'srw_composition_mode', DEFAULT_SRW_COMPOSITION_MODE))
    execution_weight_norm = execution_weight / jnp.maximum(
        composition_den, 1e-8)

    read_activations = state.astype(jnp.float32) @ read_directions.T
    weighted_read_activations = execution_weight * read_activations
    raw_state_transition = weighted_read_activations @ write_directions
    state_transition = raw_state_transition.astype(jnp.float32) / composition_den
    return (state_transition.astype(jnp.float32), execution_weight,
            execution_weight_norm)



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
    qk_operator_keys = pool_params['attn_qk_op_key']
    v_operator_keys = pool_params['attn_v_op_key']
    attn_operator_queries = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)
    tau_all = x @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
    raw_scan_offset_all = jnp.zeros_like(tau_all)

    attention_q = _srw_inference(x, q_operator_query, qk_operator_keys, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                       pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                       admission_den_pool='qk',
                       **angular_execution_kwargs)
    attention_k_new = _srw_inference(x, k_operator_query, qk_operator_keys, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                           admission_den_pool='qk',
                           **angular_execution_kwargs)
    attention_v_new = _srw_inference(x, v_operator_query, v_operator_keys, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                           pool_params['attn_v_read'], pool_params['attn_v_write'],
                           admission_den_pool='v',
                           **angular_execution_kwargs)
    _qk_s, _v_s, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)
    attention_q = attention_q * _qk_s
    attention_k_new = attention_k_new * _qk_s
    attention_v_new = attention_v_new * _v_s

    attention_q = attention_q.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)
    attention_k_new_heads = attention_k_new.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)
    attention_v_new_heads = attention_v_new.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)

    cache_K = cache_K.at[:, :, cache_len, :].set(attention_k_new_heads[:, :, 0, :])
    cache_V = cache_V.at[:, :, cache_len, :].set(attention_v_new_heads[:, :, 0, :])

    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum('bhqd,bhkd->bhqk', attention_q, cache_K) / scale
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
    rst_operator_keys = pool_params['rst_op_key']
    operator_query = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    tau = x @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
    raw_scan_offset = jnp.zeros_like(tau)
    out = _srw_inference(x, operator_query, rst_operator_keys, tau, raw_scan_offset,
                         pool_params['rst_read'], pool_params['rst_write'],
                         admission_den_pool='rst',
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

    qk_operator_keys = pool_params['attn_qk_op_key']
    v_operator_keys = pool_params['attn_v_op_key']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    cache_K = jnp.zeros((n_layers, B, n_heads, max_seq, d_head))
    cache_V = jnp.zeros((n_layers, B, n_heads, max_seq, d_head))

    def prefill_layer(carry, xs):
        x, cK, cV = carry
        bp = xs['params']
        layer_idx = xs['layer_idx']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        attn_operator_queries = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)
        tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
        raw_scan_offset_all = jnp.zeros_like(tau_all)

        attention_q = _srw_inference(normed, q_operator_query, qk_operator_keys, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                           admission_den_pool='qk',
                           **angular_execution_kwargs)
        attention_k = _srw_inference(normed, k_operator_query, qk_operator_keys, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                                     pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                                     admission_den_pool='qk',
                                     **angular_execution_kwargs)
        attention_v = _srw_inference(normed, v_operator_query, v_operator_keys, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                                     pool_params['attn_v_read'], pool_params['attn_v_write'],
                                     admission_den_pool='v',
                                     **angular_execution_kwargs)
        _qk_s = qk_scale_eff
        _v_s = v_scale_eff
        attention_q = attention_q * _qk_s
        attention_k = attention_k * _qk_s
        attention_v = attention_v * _v_s

        attention_q_heads = attention_q.reshape(
            B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_k_heads = attention_k.reshape(
            B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_v_heads = attention_v.reshape(
            B, S, n_heads, d_head).transpose(0, 2, 1, 3)

        cK = cK.at[layer_idx, :, :, :S, :].set(attention_k_heads)
        cV = cV.at[layer_idx, :, :, :S, :].set(attention_v_heads)

        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum(
            'bhsd,bhtd->bhst', attention_q_heads,
            attention_k_heads) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum(
            'bhst,bhtd->bhsd', attn_w, attention_v_heads)
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
    logits = _slice_logits_to_logical_vocab(logits, model_cfg)
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
    logits = _slice_logits_to_logical_vocab(logits, model_cfg)
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

    qk_operator_keys = pool_params['attn_qk_op_key']
    v_operator_keys = pool_params['attn_v_op_key']
    rst_operator_keys = pool_params['rst_op_key']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    def forward_batch(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        def layer_fn(x, bp):
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            attn_operator_queries = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)
            tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
            raw_scan_offset_all = jnp.zeros_like(tau_all)

            attention_q = _srw_inference(normed, q_operator_query, qk_operator_keys, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               admission_den_pool='qk',
                               **angular_execution_kwargs)
            attention_k = _srw_inference(normed, k_operator_query, qk_operator_keys, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'],
                               admission_den_pool='qk',
                               **angular_execution_kwargs)
            attention_v = _srw_inference(normed, v_operator_query, v_operator_keys, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'],
                               admission_den_pool='v',
                               **angular_execution_kwargs)
            _qk_s = qk_scale_eff
            _v_s = v_scale_eff
            attention_q = attention_q * _qk_s
            attention_k = attention_k * _qk_s
            attention_v = attention_v * _v_s

            d_head = d_model // n_heads
            attention_q_heads = attention_q.reshape(
                B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            attention_k_heads = attention_k.reshape(
                B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            attention_v_heads = attention_v.reshape(
                B, S, n_heads, d_head).transpose(0, 2, 1, 3)

            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum(
                'bhsd,bhtd->bhst', attention_q_heads,
                attention_k_heads) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum(
                'bhst,bhtd->bhsd', attn_w, attention_v_heads)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            rst_operator_query = normed @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
            tau_k = normed @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
            raw_scan_offset_k = jnp.zeros_like(tau_k)
            rst_out = _srw_inference(normed, rst_operator_query, rst_operator_keys, tau_k, raw_scan_offset_k,
                                     pool_params['rst_read'], pool_params['rst_write'],
                                     admission_den_pool='rst',
                                     **angular_execution_kwargs)
            x = x + rst_out * rst_scale_eff
            return x, None

        x, _ = jax.lax.scan(layer_fn, x, stacked)
        x = _layer_norm(x, norm_params['scale'], norm_params['bias'])

        shift_x = x[:, :-1, :]
        shift_labels = input_ids[:, 1:].astype(jnp.int32)
        valid_mask = shift_labels > 0

        logits = shift_x @ emb_matrix.T
        logits = _slice_logits_to_logical_vocab(logits, model_cfg)
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
        operator_keys = pool[op_key_key]
        read = pool[read_key]
        write = pool[write_key]
        op_key_n = jnp.linalg.norm(operator_keys, axis=-1)
        read_n = jnp.linalg.norm(read, axis=-1)
        write_n = jnp.linalg.norm(write, axis=-1)
        results[pool_name] = {
            'N': operator_keys.shape[0],
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
        operator_keys = pool[op_key_key]
        N, d = operator_keys.shape
        if N > max_sample:
            idx = jnp.linspace(0, N - 1, max_sample, dtype=jnp.int32)
            op_key_s = operator_keys[idx]
        else:
            op_key_s = operator_keys
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

    qk_operator_keys = pool_params['attn_qk_op_key']
    v_operator_keys = pool_params['attn_v_op_key']
    rst_operator_keys = pool_params['rst_op_key']

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    _return_raw = (mode == 'full')

    def analysis_layer(carry, xs):
        x = carry
        bp = xs['params']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        attn_operator_queries = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)
        tau_all = normed @ router_params['raw_tau_attn']['kernel'] + router_params['raw_tau_attn']['bias']
        raw_scan_offset_all = jnp.zeros_like(tau_all)

        attention_q, gate_Q_raw, gate_Q = _srw_inference_with_gates(
            normed, q_operator_query, qk_operator_keys, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'],
            admission_den_pool='qk',
            **angular_execution_kwargs)
        attention_k, gate_K_raw, gate_K = _srw_inference_with_gates(
            normed, k_operator_query, qk_operator_keys, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'],
            admission_den_pool='qk',
            **angular_execution_kwargs)
        attention_v, gate_V_raw, gate_V = _srw_inference_with_gates(
            normed, v_operator_query, v_operator_keys, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
            pool_params['attn_v_read'], pool_params['attn_v_write'],
            admission_den_pool='v',
            **angular_execution_kwargs)
        _qk_s = qk_scale_eff
        _v_s = v_scale_eff
        attention_q = attention_q * _qk_s
        attention_k = attention_k * _qk_s
        attention_v = attention_v * _v_s

        d_head = d_model // n_heads
        attention_q_heads = attention_q.reshape(
            B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_k_heads = attention_k.reshape(
            B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_v_heads = attention_v.reshape(
            B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum(
            'bhsd,bhtd->bhst', attention_q_heads,
            attention_k_heads) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum(
            'bhst,bhtd->bhsd', attn_w, attention_v_heads)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
        attn_out = attn_out @ bp['attn']['expand_O']['kernel']
        attn_out_norm = jnp.linalg.norm(attn_out, axis=-1).mean()
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        rst_operator_query = normed @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
        tau_k = normed @ router_params['raw_tau_rst']['kernel'] + router_params['raw_tau_rst']['bias']
        raw_scan_offset_k = jnp.zeros_like(tau_k)
        rst_out, gate_RST_raw, gate_RST = _srw_inference_with_gates(
            normed, rst_operator_query, rst_operator_keys, tau_k, raw_scan_offset_k,
            pool_params['rst_read'], pool_params['rst_write'],
            admission_den_pool='rst',
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
        info.update(_query_geometry_stats(q_operator_query, 'q'))
        info.update(_query_geometry_stats(k_operator_query, 'k'))
        info.update(_query_geometry_stats(v_operator_query, 'v'))
        info.update(_query_geometry_stats(rst_operator_query, 'rst'))
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
    logits = _slice_logits_to_logical_vocab(logits, model_cfg)
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

    def _srw_sup(
            state, operator_query, operator_keys, tau_off, raw_scan_offset,
            read_vectors, write_vectors, mult, admission_den_pool):
        """SRW with optional gate suppression."""
        # Suppressed forward selects by d_route keys and executes d_model RW.
        read_directions = _forward_unit_direction(
            read_vectors.astype(jnp.float32))
        write_directions = _forward_unit_direction(
            write_vectors.astype(jnp.float32))
        execution_kwargs, admission_den_power = _split_admission_den_kwargs(
            angular_execution_kwargs, admission_den_pool)
        _, admission, _, execution_weight, _ = _angular_execution(
            operator_query, operator_keys, tau_off, raw_scan_offset, **execution_kwargs)
        if mult is not None:
            execution_weight = execution_weight * mult[None, None, :]
            admission = admission * mult[None, None, :]
        read_activations = state.astype(jnp.float32) @ read_directions.T
        weighted_read_activations = execution_weight * read_activations
        state_transition = weighted_read_activations @ write_directions
        composition_den = _composition_den(
            admission.sum(axis=-1, keepdims=True), admission_den_power,
            execution_kwargs.get(
                'srw_composition_mode', DEFAULT_SRW_COMPOSITION_MODE))
        return (state_transition.astype(jnp.float32) / composition_den).astype(
            jnp.float32)

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
        qk_operator_keys = pp['attn_qk_op_key']
        v_operator_keys = pp['attn_v_op_key']
        rst_operator_keys = pp['rst_op_key']

        for i in range(n_layers):
            bp = params[f'block_{i}']
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            attn_operator_queries = normed @ rp['proj_attn']['kernel'] + rp['proj_attn']['bias']
            q_operator_query, k_operator_query, v_operator_query = jnp.split(attn_operator_queries, 3, axis=-1)
            tau_all = normed @ rp['raw_tau_attn']['kernel'] + rp['raw_tau_attn']['bias']
            raw_scan_offset_all = jnp.zeros_like(tau_all)

            attention_q = _srw_sup(normed, q_operator_query, qk_operator_keys, tau_all[:,:,0:1], raw_scan_offset_all[:,:,0:1], pp['attn_qk_read'], pp['attn_qk_write'], qk_mult, 'qk')
            attention_k = _srw_sup(normed, k_operator_query, qk_operator_keys, tau_all[:,:,1:2], raw_scan_offset_all[:,:,1:2], pp['attn_qk_read'], pp['attn_qk_write'], qk_mult, 'qk')
            attention_v = _srw_sup(normed, v_operator_query, v_operator_keys, tau_all[:,:,2:3], raw_scan_offset_all[:,:,2:3], pp['attn_v_read'], pp['attn_v_write'], v_mult, 'v')
            _qk_s = qk_scale_eff
            _v_s = v_scale_eff
            attention_q = attention_q * _qk_s
            attention_k = attention_k * _qk_s
            attention_v = attention_v * _v_s

            attention_q_heads = attention_q.reshape(
                B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            attention_k_heads = attention_k.reshape(
                B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            attention_v_heads = attention_v.reshape(
                B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            sc = jnp.sqrt(jnp.float32(d_head))
            attn_s = jnp.einsum(
                'bhsd,bhtd->bhst', attention_q_heads,
                attention_k_heads) / sc
            causal = jnp.tril(jnp.ones((S,S), dtype=jnp.bool_))
            attn_s = jnp.where(causal, attn_s, jnp.finfo(attn_s.dtype).min)
            attn_w = jax.nn.softmax(attn_s, axis=-1)
            attn_out = jnp.einsum(
                'bhst,bhtd->bhsd', attn_w, attention_v_heads)
            attn_out = attn_out.transpose(0,2,1,3).reshape(B,S,d_model) @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            rst_operator_query = normed @ rp['proj_rst']['kernel'] + rp['proj_rst']['bias']
            tau_k = normed @ rp['raw_tau_rst']['kernel'] + rp['raw_tau_rst']['bias']
            raw_scan_offset_k = jnp.zeros_like(tau_k)
            x = x + _srw_sup(normed, rst_operator_query, rst_operator_keys, tau_k, raw_scan_offset_k, pp['rst_read'], pp['rst_write'], rst_mult, 'rst') * rst_scale_eff

        norm_p = params['norm']
        x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
        logits = x @ params['token_emb']['embedding'].T
        return _slice_logits_to_logical_vocab(logits, model_cfg)

    return forward_fn


def _rename_key_if_needed(d, old, new):
    """Rename old -> new in a mutable mapping, preserving an existing new key."""
    if old in d:
        if new not in d:
            d[new] = d[old]
        del d[old]
