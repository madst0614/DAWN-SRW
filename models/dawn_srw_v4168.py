"""
DAWN-SRW v4.1.6.8 operation-space SRW.

When operation_space is enabled:
- Region/Block/Operator Atlas selection is the only operation-space routing.
- RST uses sparse_region_block execution over selected blocks.
- QK and V use dense execution with region/block masks.
- QK/V/RST do not use DirectTau, admission, drive, or selection calibration in
  the active output path.
- Legacy DirectTau/sector code remains only for disabled operation_space or
  compatibility.

Region/Block Operation Atlas:
RW operator keys define a spherical operation-space atlas. Regions are coarse
visibility groups, blocks are dense RW execution units. Tokens select a small
render set of visible blocks via region-level hard visibility and block-level
local selection. The old lane terminology is legacy only.
"""


import jax
import jax.numpy as jnp
import flax.linen as nn
import math
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze
from typing import Optional, Dict
from functools import partial
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    _PALLAS_AVAILABLE = True
except Exception:
    pl = None
    pltpu = None
    _PALLAS_AVAILABLE = False

_PALLAS_REQUIRED_API = (
    'BlockSpec', 'dot', 'load', 'pallas_call', 'store')
_PALLAS_TPU_REQUIRED_API = ('CompilerParams', 'PrefetchScalarGridSpec')
_PALLAS_API_AVAILABLE = (
    _PALLAS_AVAILABLE
    and all(hasattr(pl, name) for name in _PALLAS_REQUIRED_API)
    and pltpu is not None
    and all(hasattr(pltpu, name) for name in _PALLAS_TPU_REQUIRED_API))


def _pallas_unavailable_message():
    if not _PALLAS_AVAILABLE:
        return "jax.experimental.pallas is unavailable"
    missing = [
        name for name in _PALLAS_REQUIRED_API if not hasattr(pl, name)
    ]
    if pltpu is None:
        missing.extend(f'tpu.{name}' for name in _PALLAS_TPU_REQUIRED_API)
    else:
        missing.extend(
            f'tpu.{name}' for name in _PALLAS_TPU_REQUIRED_API
            if not hasattr(pltpu, name))
    return (
        "jax.experimental.pallas is missing required APIs for "
        "block_bucketed_pallas_final: " + ", ".join(missing))

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

SECTOR_RUNTIME_DIAG_NAMES = (
    'sector_fill_mean',
    'sector_fill_max',
    'sector_overflow_count',
    'selected_sector_frac',
    'effective_operator_frac',
)
SECTOR_RUNTIME_DIAG_COUNT = len(SECTOR_RUNTIME_DIAG_NAMES)
(
    SECTOR_FILL_MEAN,
    SECTOR_FILL_MAX,
    SECTOR_OVERFLOW_COUNT,
    SECTOR_SELECTED_SECTOR_FRAC,
    SECTOR_EFFECTIVE_OPERATOR_FRAC,
) = range(SECTOR_RUNTIME_DIAG_COUNT)

OPSPACE_RUNTIME_DIAG_NAMES = (
    'enabled',
    'visible_regions',
    'visible_ops_per_token',
    'valid_exec_slots_mean',
    'bucket_capacity',
    'bucket_fill_mean',
    'selected_region_top1_frac',
    'primary_accept_frac',
    'bucket_fill_skew',
    'reroute_frac',
    'relu_gate_count_mean',
    'gate_denominator_mean',
    'gate_mass_mean',
    'effective_ops',
    'padded_exec_slots_mean',
    'overflow_frac',
    'no_nan',
    'owner_model_axis',
    'selected_requests',
    'processed_requests',
    'all_processed',
    'factorized_tile_layout_ok',
)
OPSPACE_RUNTIME_DIAG_COUNT = len(OPSPACE_RUNTIME_DIAG_NAMES)
(
    OPSPACE_ENABLED,
    OPSPACE_VISIBLE_REGIONS,
    OPSPACE_VISIBLE_OPS_PER_TOKEN,
    OPSPACE_VALID_EXEC_SLOTS_MEAN,
    OPSPACE_BUCKET_CAPACITY,
    OPSPACE_BUCKET_FILL_MEAN,
    OPSPACE_SELECTED_REGION_TOP1_FRAC,
    OPSPACE_PRIMARY_ACCEPT_FRAC,
    OPSPACE_BUCKET_FILL_SKEW,
    OPSPACE_REROUTE_FRAC,
    OPSPACE_RELU_GATE_COUNT_MEAN,
    OPSPACE_GATE_DENOMINATOR_MEAN,
    OPSPACE_GATE_MASS_MEAN,
    OPSPACE_EFFECTIVE_OPS,
    OPSPACE_PADDED_EXEC_SLOTS_MEAN,
    OPSPACE_OVERFLOW_FRAC,
    OPSPACE_NO_NAN,
    OPSPACE_OWNER_MODEL_AXIS,
    OPSPACE_SELECTED_REQUESTS,
    OPSPACE_PROCESSED_REQUESTS,
    OPSPACE_ALL_PROCESSED,
    OPSPACE_FACTORIZE_TILE_LAYOUT_OK,
) = range(OPSPACE_RUNTIME_DIAG_COUNT)

OPSPACE_VRST_GATE = "relu_power"
OPSPACE_VRST_GATE_POWER = 2.0
OPSPACE_REPACK_STD_MULT = 2.0
OPSPACE_REPACK_MIN_ASSIGNMENT_GAIN = 0.03
OPSPACE_REPACK_MIN_SWAP_GAIN = 0.0
OPSPACE_OWNER_AXIS = "model_axis"
OPSPACE_NO_DROP = True
OPSPACE_BUCKET_OVERFLOW_SEMANTICS = True
OPSPACE_FINAL_OUTPUT_TILE_SIZE = 128
OPSPACE_FINAL_D_TILE_SIZE_POLICY = 128
OPSPACE_FINAL_OP_TILE_SIZE_POLICY = 128
OPSPACE_RST_XLA_REGION_GROUP_SIZE = 4
OPSPACE_RST_XLA_TOKEN_COMBINE_CHUNK_SIZE = 1024
OPSPACE_RST_REGION_CAPACITY_ROUND_MULTIPLE = 1024
OPSPACE_RST_BLOCK_CAPACITY_ROUND_MULTIPLE = 128

OPSPACE_FINAL_RUNTIME_DIAG_NAMES = (
    'assignment_score_best_mean',
    'assignment_score_chosen_mean',
    'assignment_score_loss_mean',
    'assignment_score_loss_p95',
    'assignment_score_loss_max',
    'assignment_regret_mean',
    'assignment_regret_p95',
    'assignment_regret_max',
    'low_regret_spill_frac',
    'high_regret_spill_frac',
    'unresolved_count',
    'assignment_collision_count',
    'overflow_slowpath_frac',
    'semantic_drop_frac',
    'bucket_fill_p50',
    'bucket_fill_p95',
    'bucket_fill_p99',
    'bucket_fill_max',
    'bucket_capacity_util_p50',
    'bucket_capacity_util_p95',
    'bucket_capacity_util_p99',
    'bucket_capacity_util_max',
    'pallas_logical_executed_chunks',
    'pallas_padding_frac',
    'pallas_valid_token_frac',
    'pallas_backend_available',
    'pallas_logical_empty_chunks',
    'pallas_actual_launched_chunks',
    'pallas_actual_skipped_chunks',
    'actual_vs_logical_chunk_ratio',
    'output_tile_capacity',
    'output_tile_fill_p50',
    'output_tile_fill_p95',
    'output_tile_fill_p99',
    'output_tile_fill_max',
    'output_tile_overflow_frac',
    'output_tile_padding_frac',
    'd_tile_size',
    'output_tile_size',
    'output_tiled_backend_active',
    'region_load_p99_over_mean',
    'block_load_p99_over_mean',
    'block_compactness_mean',
    'block_radius_mean',
    'block_radius_max',
    'region_count',
    'regions_per_owner',
    'blocks_per_region',
    'operators_per_block',
    'visible_regions',
    'visible_blocks_per_region',
    'region_capacity',
    'region_capacity_factor',
    'region_load_mean',
    'region_load_p50',
    'region_load_p95',
    'region_load_p99',
    'region_load_max',
    'region_capacity_util_p50',
    'region_capacity_util_p95',
    'region_capacity_util_p99',
    'region_capacity_util_max',
    'region_unresolved_count',
    'region_unresolved_frac',
    'region_reroute_frac',
    'region_low_regret_spill_frac',
    'region_high_regret_spill_frac',
    'region_score_best_mean',
    'region_score_chosen_mean',
    'region_score_loss_mean',
    'region_score_loss_p95',
    'region_score_loss_max',
    'block_capacity',
    'block_capacity_factor',
    'block_load_mean',
    'block_load_p50',
    'block_load_p95',
    'block_load_p99',
    'block_load_max',
    'block_capacity_util_p50',
    'block_capacity_util_p95',
    'block_capacity_util_p99',
    'block_capacity_util_max',
    'block_unresolved_count',
    'block_unresolved_frac',
    'block_reroute_frac',
    'block_low_regret_spill_frac',
    'block_high_regret_spill_frac',
    'block_score_best_mean',
    'block_score_chosen_mean',
    'block_score_loss_mean',
    'block_score_loss_p95',
    'block_score_loss_max',
    'visible_blocks_per_token',
    'visible_ops_per_token',
    'physical_visible_ops_per_token',
    'compute_frac_vs_dense',
    'logical_compute_frac_vs_dense',
    'physical_compute_frac_vs_dense',
    'region_all_requests_processed',
    'block_all_requests_processed',
    'selected_requests',
    'processed_requests',
    'all_processed',
    'no_nan',
)
OPSPACE_FINAL_RUNTIME_DIAG_COUNT = len(OPSPACE_FINAL_RUNTIME_DIAG_NAMES)
(
    OPSPACE_FINAL_ASSIGNMENT_SCORE_BEST_MEAN,
    OPSPACE_FINAL_ASSIGNMENT_SCORE_CHOSEN_MEAN,
    OPSPACE_FINAL_ASSIGNMENT_SCORE_LOSS_MEAN,
    OPSPACE_FINAL_ASSIGNMENT_SCORE_LOSS_P95,
    OPSPACE_FINAL_ASSIGNMENT_SCORE_LOSS_MAX,
    OPSPACE_FINAL_ASSIGNMENT_REGRET_MEAN,
    OPSPACE_FINAL_ASSIGNMENT_REGRET_P95,
    OPSPACE_FINAL_ASSIGNMENT_REGRET_MAX,
    OPSPACE_FINAL_LOW_REGRET_SPILL_FRAC,
    OPSPACE_FINAL_HIGH_REGRET_SPILL_FRAC,
    OPSPACE_FINAL_UNRESOLVED_COUNT,
    OPSPACE_FINAL_ASSIGNMENT_COLLISION_COUNT,
    OPSPACE_FINAL_OVERFLOW_SLOWPATH_FRAC,
    OPSPACE_FINAL_SEMANTIC_DROP_FRAC,
    OPSPACE_FINAL_BUCKET_FILL_P50,
    OPSPACE_FINAL_BUCKET_FILL_P95,
    OPSPACE_FINAL_BUCKET_FILL_P99,
    OPSPACE_FINAL_BUCKET_FILL_MAX,
    OPSPACE_FINAL_BUCKET_CAPACITY_UTIL_P50,
    OPSPACE_FINAL_BUCKET_CAPACITY_UTIL_P95,
    OPSPACE_FINAL_BUCKET_CAPACITY_UTIL_P99,
    OPSPACE_FINAL_BUCKET_CAPACITY_UTIL_MAX,
    OPSPACE_FINAL_PALLAS_LOGICAL_EXECUTED_CHUNKS,
    OPSPACE_FINAL_PALLAS_PADDING_FRAC,
    OPSPACE_FINAL_PALLAS_VALID_TOKEN_FRAC,
    OPSPACE_FINAL_PALLAS_BACKEND_AVAILABLE,
    OPSPACE_FINAL_PALLAS_LOGICAL_EMPTY_CHUNKS,
    OPSPACE_FINAL_PALLAS_ACTUAL_LAUNCHED_CHUNKS,
    OPSPACE_FINAL_PALLAS_ACTUAL_SKIPPED_CHUNKS,
    OPSPACE_FINAL_ACTUAL_VS_LOGICAL_CHUNK_RATIO,
    OPSPACE_FINAL_OUTPUT_TILE_CAPACITY,
    OPSPACE_FINAL_OUTPUT_TILE_FILL_P50,
    OPSPACE_FINAL_OUTPUT_TILE_FILL_P95,
    OPSPACE_FINAL_OUTPUT_TILE_FILL_P99,
    OPSPACE_FINAL_OUTPUT_TILE_FILL_MAX,
    OPSPACE_FINAL_OUTPUT_TILE_OVERFLOW_FRAC,
    OPSPACE_FINAL_OUTPUT_TILE_PADDING_FRAC,
    OPSPACE_FINAL_D_TILE_SIZE_DIAG,
    OPSPACE_FINAL_OUTPUT_TILE_SIZE_DIAG,
    OPSPACE_FINAL_OUTPUT_TILED_BACKEND_ACTIVE,
    OPSPACE_FINAL_REGION_LOAD_P99_OVER_MEAN,
    OPSPACE_FINAL_BLOCK_LOAD_P99_OVER_MEAN,
    OPSPACE_FINAL_BLOCK_COMPACTNESS_MEAN,
    OPSPACE_FINAL_BLOCK_RADIUS_MEAN,
    OPSPACE_FINAL_BLOCK_RADIUS_MAX,
    OPSPACE_FINAL_REGION_COUNT,
    OPSPACE_FINAL_REGIONS_PER_OWNER,
    OPSPACE_FINAL_BLOCKS_PER_REGION,
    OPSPACE_FINAL_OPERATORS_PER_BLOCK,
    OPSPACE_FINAL_VISIBLE_REGIONS,
    OPSPACE_FINAL_VISIBLE_BLOCKS_PER_REGION,
    OPSPACE_FINAL_REGION_CAPACITY,
    OPSPACE_FINAL_REGION_CAPACITY_FACTOR,
    OPSPACE_FINAL_REGION_LOAD_MEAN,
    OPSPACE_FINAL_REGION_LOAD_P50,
    OPSPACE_FINAL_REGION_LOAD_P95,
    OPSPACE_FINAL_REGION_LOAD_P99,
    OPSPACE_FINAL_REGION_LOAD_MAX,
    OPSPACE_FINAL_REGION_CAPACITY_UTIL_P50,
    OPSPACE_FINAL_REGION_CAPACITY_UTIL_P95,
    OPSPACE_FINAL_REGION_CAPACITY_UTIL_P99,
    OPSPACE_FINAL_REGION_CAPACITY_UTIL_MAX,
    OPSPACE_FINAL_REGION_UNRESOLVED_COUNT,
    OPSPACE_FINAL_REGION_UNRESOLVED_FRAC,
    OPSPACE_FINAL_REGION_REROUTE_FRAC,
    OPSPACE_FINAL_REGION_LOW_REGRET_SPILL_FRAC,
    OPSPACE_FINAL_REGION_HIGH_REGRET_SPILL_FRAC,
    OPSPACE_FINAL_REGION_SCORE_BEST_MEAN,
    OPSPACE_FINAL_REGION_SCORE_CHOSEN_MEAN,
    OPSPACE_FINAL_REGION_SCORE_LOSS_MEAN,
    OPSPACE_FINAL_REGION_SCORE_LOSS_P95,
    OPSPACE_FINAL_REGION_SCORE_LOSS_MAX,
    OPSPACE_FINAL_BLOCK_CAPACITY,
    OPSPACE_FINAL_BLOCK_CAPACITY_FACTOR,
    OPSPACE_FINAL_BLOCK_LOAD_MEAN,
    OPSPACE_FINAL_BLOCK_LOAD_P50,
    OPSPACE_FINAL_BLOCK_LOAD_P95,
    OPSPACE_FINAL_BLOCK_LOAD_P99,
    OPSPACE_FINAL_BLOCK_LOAD_MAX,
    OPSPACE_FINAL_BLOCK_CAPACITY_UTIL_P50,
    OPSPACE_FINAL_BLOCK_CAPACITY_UTIL_P95,
    OPSPACE_FINAL_BLOCK_CAPACITY_UTIL_P99,
    OPSPACE_FINAL_BLOCK_CAPACITY_UTIL_MAX,
    OPSPACE_FINAL_BLOCK_UNRESOLVED_COUNT,
    OPSPACE_FINAL_BLOCK_UNRESOLVED_FRAC,
    OPSPACE_FINAL_BLOCK_REROUTE_FRAC,
    OPSPACE_FINAL_BLOCK_LOW_REGRET_SPILL_FRAC,
    OPSPACE_FINAL_BLOCK_HIGH_REGRET_SPILL_FRAC,
    OPSPACE_FINAL_BLOCK_SCORE_BEST_MEAN,
    OPSPACE_FINAL_BLOCK_SCORE_CHOSEN_MEAN,
    OPSPACE_FINAL_BLOCK_SCORE_LOSS_MEAN,
    OPSPACE_FINAL_BLOCK_SCORE_LOSS_P95,
    OPSPACE_FINAL_BLOCK_SCORE_LOSS_MAX,
    OPSPACE_FINAL_VISIBLE_BLOCKS_PER_TOKEN,
    OPSPACE_FINAL_VISIBLE_OPS_PER_TOKEN,
    OPSPACE_FINAL_PHYSICAL_VISIBLE_OPS_PER_TOKEN,
    OPSPACE_FINAL_COMPUTE_FRAC_VS_DENSE,
    OPSPACE_FINAL_LOGICAL_COMPUTE_FRAC_VS_DENSE,
    OPSPACE_FINAL_PHYSICAL_COMPUTE_FRAC_VS_DENSE,
    OPSPACE_FINAL_REGION_ALL_REQUESTS_PROCESSED,
    OPSPACE_FINAL_BLOCK_ALL_REQUESTS_PROCESSED,
    OPSPACE_FINAL_SELECTED_REQUESTS,
    OPSPACE_FINAL_PROCESSED_REQUESTS,
    OPSPACE_FINAL_ALL_PROCESSED,
    OPSPACE_FINAL_NO_NAN,
) = range(OPSPACE_FINAL_RUNTIME_DIAG_COUNT)

BENCHMARK_SECTOR_RUNTIME_DIAG_NAMES = (
    'bucket_fill_mean',
    'bucket_fill_p50',
    'bucket_fill_p90',
    'bucket_fill_p95',
    'bucket_fill_p99',
    'bucket_fill_max',
    'overflow_count',
    'overflow_frac',
    'bucket_capacity',
    'bucket_capacity_util_p50',
    'bucket_capacity_util_p95',
    'bucket_capacity_util_p99',
    'bucket_capacity_util_max',
    'expected_selected_pair_count',
    'executed_pair_count',
    'executed_selected_pair_frac',
    'batch_union_selected_sector_frac',
    'batch_union_effective_operator_frac',
    'per_token_selected_sector_count',
    'per_token_selected_ops',
    'per_token_effective_operator_frac',
    'semantic_work_frac_vs_dense',
    'padded_work_frac_vs_dense',
    'hot_sector_skew_p99_over_mean',
    'attempted_fill_mean',
    'attempted_fill_p50',
    'attempted_fill_p95',
    'attempted_fill_p99',
    'attempted_fill_max',
    'required_capacity_no_overflow',
    'current_capacity',
    'capacity_shortfall',
)
BENCHMARK_SECTOR_RUNTIME_DIAG_COUNT = len(
    BENCHMARK_SECTOR_RUNTIME_DIAG_NAMES)
(
    BENCHMARK_SECTOR_BUCKET_FILL_MEAN,
    BENCHMARK_SECTOR_BUCKET_FILL_P50,
    BENCHMARK_SECTOR_BUCKET_FILL_P90,
    BENCHMARK_SECTOR_BUCKET_FILL_P95,
    BENCHMARK_SECTOR_BUCKET_FILL_P99,
    BENCHMARK_SECTOR_BUCKET_FILL_MAX,
    BENCHMARK_SECTOR_OVERFLOW_COUNT,
    BENCHMARK_SECTOR_OVERFLOW_FRAC,
    BENCHMARK_SECTOR_BUCKET_CAPACITY,
    BENCHMARK_SECTOR_BUCKET_CAPACITY_UTIL_P50,
    BENCHMARK_SECTOR_BUCKET_CAPACITY_UTIL_P95,
    BENCHMARK_SECTOR_BUCKET_CAPACITY_UTIL_P99,
    BENCHMARK_SECTOR_BUCKET_CAPACITY_UTIL_MAX,
    BENCHMARK_SECTOR_EXPECTED_SELECTED_PAIR_COUNT,
    BENCHMARK_SECTOR_EXECUTED_PAIR_COUNT,
    BENCHMARK_SECTOR_EXECUTED_SELECTED_PAIR_FRAC,
    BENCHMARK_SECTOR_BATCH_UNION_SELECTED_SECTOR_FRAC,
    BENCHMARK_SECTOR_BATCH_UNION_EFFECTIVE_OPERATOR_FRAC,
    BENCHMARK_SECTOR_PER_TOKEN_SELECTED_SECTOR_COUNT,
    BENCHMARK_SECTOR_PER_TOKEN_SELECTED_OPS,
    BENCHMARK_SECTOR_PER_TOKEN_EFFECTIVE_OPERATOR_FRAC,
    BENCHMARK_SECTOR_SEMANTIC_WORK_FRAC_VS_DENSE,
    BENCHMARK_SECTOR_PADDED_WORK_FRAC_VS_DENSE,
    BENCHMARK_SECTOR_HOT_SECTOR_SKEW_P99_OVER_MEAN,
    BENCHMARK_SECTOR_ATTEMPTED_FILL_MEAN,
    BENCHMARK_SECTOR_ATTEMPTED_FILL_P50,
    BENCHMARK_SECTOR_ATTEMPTED_FILL_P95,
    BENCHMARK_SECTOR_ATTEMPTED_FILL_P99,
    BENCHMARK_SECTOR_ATTEMPTED_FILL_MAX,
    BENCHMARK_SECTOR_REQUIRED_CAPACITY_NO_OVERFLOW,
    BENCHMARK_SECTOR_CURRENT_CAPACITY,
    BENCHMARK_SECTOR_CAPACITY_SHORTFALL,
) = range(BENCHMARK_SECTOR_RUNTIME_DIAG_COUNT)


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
MODEL_VERSION = "spatial-r1-v4.1.6.8"


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
    """Compressed live-gradient RW operator identity for v4166 selection.

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


def _chunked_ce_loss_and_acc(shift_x, embedding_matrix, shift_labels,
                             valid_mask, token_chunk_size=8192):
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

    @jax.checkpoint
    def step(carry, xs):
        loss_sum, correct_sum, valid_sum = carry
        x_c, labels_c, valid_c = xs

        logits = x_c @ embedding_matrix.T
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        safe_labels = jnp.where(valid_c, labels_c, 0)
        token_loss = -jnp.take_along_axis(
            log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
        token_loss = token_loss.astype(jnp.float32)
        valid_f = valid_c.astype(jnp.float32)

        preds = jnp.argmax(logits, axis=-1)
        loss_sum = loss_sum + (token_loss * valid_f).sum()
        correct_sum = (
            correct_sum
            + ((preds == labels_c) & valid_c).astype(jnp.int32).sum())
        valid_sum = valid_sum + valid_c.astype(jnp.int32).sum()
        return (loss_sum, correct_sum, valid_sum), None

    init = (
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )
    (loss_sum, correct, valid_count), _ = jax.lax.scan(
        step, init, (flat_x, flat_labels, flat_valid))
    loss = loss_sum / (valid_count.astype(jnp.float32) + 1e-8)
    return loss, correct, valid_count


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
                     admission_den_grad_scale=1.0):
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
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, D = x.shape
        x_bf = x.astype(jnp.bfloat16)
        op_key_padded = jnp.pad(op_key_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        op_key_dir_bf = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
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

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_dir_bf, start, cs, axis=0)

        def operator_relation(op_key):
            # Cosine between operator query and RW-derived operator key.
            rho = (h_unit_bf @ op_key.T).astype(jnp.float32)
            rho_exposure = (
                jax.lax.stop_gradient(h_unit_bf) @ op_key.T
            ).astype(jnp.float32)
            return rho, rho_exposure

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
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
                op_key, _, _, valid_chunk = op_key_rw_chunk(s)
                valid_bsn = valid_chunk[None, None, :]
                rho_raw, _ = operator_relation(op_key)
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

        # Load-balance over rho distribution is disabled in the fast path for
        # v4.1.6.4 does not require a rho-statistics pass for regular train.
        rho_lb = jnp.float32(0.0)

        def edge_margin_stat_terms(rho):
            positive_selection_margin = jax.nn.relu(rho - tau_ref)
            return jnp.square(positive_selection_margin)

        def angular_compose_parts(rho, valid_mask):
            (selection_margin, admission, drive, execution_weight,
             active_mask) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            strong_mask = selection_margin > _angular_strong_margin
            selection_margin = jnp.where(valid_mask, selection_margin, 0.0)
            admission = jnp.where(valid_mask, admission, 0.0)
            drive = jnp.where(valid_mask, drive, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            active_mask = active_mask & valid_mask
            strong_mask = strong_mask & valid_mask
            return (
                selection_margin,
                admission,
                drive,
                execution_weight,
                active_mask,
                strong_mask,
            )

        def update_select_diag(carry, rho, selection_margin, positive_margin,
                               valid_mask):
            (total_selected, total_selection_margin_sum,
             total_positive_margin_sum, total_positive_margin_max,
             total_rho_max, total_selection_margin_max) = carry
            selected = ((selection_margin > 0.0) & valid_mask).astype(jnp.float32)
            selection_margin_sum = jnp.where(
                valid_mask, selection_margin, 0.0).sum()
            selection_margin_max = jnp.where(
                valid_mask, selection_margin, diag_neg_inf)
            positive_margin_valid = jnp.where(valid_mask, positive_margin, 0.0)
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

            projected_gate = _boundary_gate_from_margin(
                margin_sg, soft_gate_t_final, soft_gate_boundary_power_final)
            projected_gate = jnp.where(valid_mask, projected_gate, 0.0)
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
                op_key, rc, wc, valid_chunk = op_key_rw_chunk(s)
                valid_bsn = valid_chunk[None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_relation(op_key)
                rho = jnp.where(valid_bsn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsn, rho_raw, tau)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission,
                    valid_bsn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    drive.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight,
                        valid_bsn)
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
                    soft_gate_exposure_parts(admission, valid_chunk))
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
                op_key, rc, wc, valid_chunk = op_key_rw_chunk(s)
                valid_bsn = valid_chunk[None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_relation(op_key)
                rho = jnp.where(valid_bsn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsn, rho_raw, tau)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission,
                    valid_bsn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    drive.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight,
                        valid_bsn)
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
                    soft_gate_exposure_parts(admission, valid_chunk))
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
                            admission_den_grad_scale=1.0):
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
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, D = x.shape
        # h: [B,S,2,d_route], raw_tau: [B,S,2,1]
        x_bf = x.astype(jnp.bfloat16)
        op_key_padded = jnp.pad(op_key_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        op_key_dir_bf = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
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

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_dir_bf, start, cs, axis=0)

        def operator_relation(op_key):
            # Cosine between operator query and RW-derived operator key.
            rho = jnp.einsum(
                'bsrd,nd->bsrn', h_unit_bf, op_key).astype(jnp.float32)
            rho_exposure = jnp.einsum(
                'bsrd,nd->bsrn',
                jax.lax.stop_gradient(h_unit_bf),
                op_key).astype(jnp.float32)
            return rho, rho_exposure

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
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
                op_key, _, _, valid_chunk = op_key_rw_chunk(s)
                valid_bsrn = valid_chunk[None, None, None, :]
                rho_raw, _ = operator_relation(op_key)
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

        # Load-balance over rho distribution is disabled in the fast path for
        # v4.1.6.4 does not require a rho-statistics pass for regular train.
        rho_lb = jnp.float32(0.0)

        def edge_margin_stat_terms(rho):
            positive_selection_margin = jax.nn.relu(rho - tau_ref)
            return jnp.square(positive_selection_margin)

        def angular_compose_parts(rho, valid_mask):
            (selection_margin, admission, drive, execution_weight,
             active_mask) = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            strong_mask = selection_margin > _angular_strong_margin
            selection_margin = jnp.where(valid_mask, selection_margin, 0.0)
            admission = jnp.where(valid_mask, admission, 0.0)
            drive = jnp.where(valid_mask, drive, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            active_mask = active_mask & valid_mask
            strong_mask = strong_mask & valid_mask
            return (
                selection_margin,
                admission,
                drive,
                execution_weight,
                active_mask,
                strong_mask,
            )

        def update_select_diag(carry, rho, selection_margin, positive_margin,
                               valid_mask):
            (total_selected, total_selection_margin_sum,
             total_positive_margin_sum, total_positive_margin_max,
             total_rho_max, total_selection_margin_max) = carry
            selected = ((selection_margin > 0.0) & valid_mask).astype(jnp.float32)
            selection_margin_sum = jnp.where(
                valid_mask, selection_margin, 0.0).sum()
            selection_margin_max = jnp.where(
                valid_mask, selection_margin, diag_neg_inf)
            positive_margin_valid = jnp.where(valid_mask, positive_margin, 0.0)
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

            projected_gate = _boundary_gate_from_margin(
                margin_sg, soft_gate_t_final, soft_gate_boundary_power_final)
            projected_gate = jnp.where(valid_mask, projected_gate, 0.0)
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
                op_key, rc, wc, valid_chunk = op_key_rw_chunk(s)
                valid_bsrn = valid_chunk[None, None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_relation(op_key)
                rho = jnp.where(valid_bsrn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsrn, rho_raw, tau)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsrn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission,
                    valid_bsrn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    drive.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight,
                        valid_bsrn)
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
                    soft_gate_exposure_parts(admission, valid_chunk))
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
                op_key, rc, wc, valid_chunk = op_key_rw_chunk(s)
                valid_bsrn = valid_chunk[None, None, None, :]
                valid_count = valid_chunk.astype(jnp.float32).sum()
                rho_raw, rho_exposure = operator_relation(op_key)
                rho = jnp.where(valid_bsrn, rho_raw, diag_neg_inf)
                rho_compute = jnp.where(valid_bsrn, rho_raw, tau)
                (selection_margin, admission, drive, execution_weight,
                 active_mask, strong_mask) = angular_compose_parts(
                    rho_compute, valid_bsrn)
                select_diag_carry = update_select_diag(
                    select_diag_carry, rho, selection_margin, admission,
                    valid_bsrn)
                chunk_edge_margin_stat = edge_margin_stat_terms(rho).sum()
                chunk_selection_residency_sum = jnp.float32(0.0)
                chunk_selection_residency_count = jnp.float32(0.0)
                chunk_current_cost = (
                    drive.sum(axis=-1, keepdims=True)
                    / jnp.maximum(valid_count, 1.0))
                if _sparsity_diag_enabled:
                    chunk_sparsity = gate_sparsity_parts(
                        selection_margin, admission, execution_weight,
                        valid_bsrn)
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
                    soft_gate_exposure_parts(admission, valid_chunk))
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


def make_sharded_srw_dense_minimal(mesh, max_chunk_size=2048,
                                   dead_exposure_target=0.1,
                                   soft_gate_effective_active_eps=1.0e-6,
                                   admission_den_power=1.0,
                                   admission_den_grad_scale=1.0):
    """Create a shard_map'd single-route SRW kernel that returns only output."""
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
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, D = x.shape
        x_bf = x.astype(jnp.bfloat16)
        op_key_padded = jnp.pad(op_key_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        op_key_dir_bf = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir_bf = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir_bf = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        tau = _tau_from_param(raw_tau)

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_dir_bf, start, cs, axis=0)

        def operator_relation(op_key):
            return (h_unit_bf @ op_key.T).astype(jnp.float32)

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_dir_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_dir_bf, start, cs, axis=0)
            vc = jax.lax.dynamic_slice_in_dim(valid_padded, start, cs, axis=0)
            return ec, rc, wc, vc

        def angular_compose_parts(rho, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        @jax.checkpoint
        def gate_srw_step(carry, i):
            raw_out, total_den_cost = carry
            s = i * cs
            op_key, rc, wc, valid_chunk = op_key_rw_chunk(s)
            valid_bsn = valid_chunk[None, None, :]
            rho_raw = operator_relation(op_key)
            rho_compute = jnp.where(valid_bsn, rho_raw, tau)
            admission, execution_weight = angular_compose_parts(
                rho_compute, valid_bsn)
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


def make_sharded_srw_paired_dense_minimal(mesh, max_chunk_size=2048,
                                          dead_exposure_target=0.1,
                                          soft_gate_effective_active_eps=1.0e-6,
                                          admission_den_power=1.0,
                                          admission_den_grad_scale=1.0,
                                          operation_space_mode='direct_tau_dense',
                                          operation_space_routing_mode=None,
                                          operation_space_execution_mode=None,
                                          operation_space_execution_backend=None,
                                          opspace_lanes=8,
                                          opspace_tiles_per_lane=4,
                                          opspace_exec_tiles_per_block=1,
                                          opspace_blocks_per_lane=None,
                                          opspace_block_size=None,
                                          opspace_padded_ops=0,
                                          opspace_tile_size=128,
                                          opspace_k_exec=4,
                                          opspace_num_regions=8,
                                          opspace_blocks_per_region=2,
                                          opspace_operators_per_block=128,
                                          opspace_visible_regions=2,
                                          opspace_visible_blocks_per_region=1,
                                          opspace_token_chunk_size=256):
    """Create a shard_map'd Q/K SRW kernel that returns only paired output."""
    execution_backend = str(
        operation_space_execution_backend
        if operation_space_execution_backend is not None
        else '').lower()
    if execution_backend == 'dense':
        if int(opspace_visible_blocks_per_region) != 1:
            raise ValueError(
                "QK dense Region/Block backend currently requires "
                "visible_blocks_per_region=1.")
        return make_sharded_srw_paired_tau_free_relu_dense_masked_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            lanes=opspace_num_regions,
            tiles_per_lane=opspace_blocks_per_region,
            padded_ops=opspace_padded_ops,
            tile_size=opspace_operators_per_block,
            k_exec=opspace_visible_regions,
            token_chunk_size=opspace_token_chunk_size)
    if execution_backend == 'sparse_region_block':
        raise ValueError(
            "QK operation_space execution_backend=sparse_region_block is "
            "not supported in this patch; use dense.")
    routing_mode = str(
        operation_space_routing_mode
        if operation_space_routing_mode is not None
        else operation_space_mode).lower()
    execution_mode = str(
        operation_space_execution_mode
        if operation_space_execution_mode is not None
        else 'dense_masked').lower()
    if routing_mode == 'factorized_lane_mean':
        if execution_mode == 'dense_masked':
            return make_sharded_srw_paired_tau_free_relu_dense_masked_minimal(
                mesh, max_chunk_size=max_chunk_size,
                dead_exposure_target=dead_exposure_target,
                soft_gate_effective_active_eps=soft_gate_effective_active_eps,
                admission_den_power=admission_den_power,
                admission_den_grad_scale=admission_den_grad_scale,
                lanes=opspace_lanes,
                tiles_per_lane=opspace_tiles_per_lane,
                padded_ops=opspace_padded_ops,
                tile_size=opspace_tile_size,
                k_exec=opspace_k_exec,
                token_chunk_size=opspace_token_chunk_size)
        if execution_mode == 'block_bucketed_dense':
            return (
                make_sharded_srw_paired_tau_free_relu_block_bucketed_dense_minimal(
                    mesh, max_chunk_size=max_chunk_size,
                    dead_exposure_target=dead_exposure_target,
                    soft_gate_effective_active_eps=soft_gate_effective_active_eps,
                    admission_den_power=admission_den_power,
                    admission_den_grad_scale=admission_den_grad_scale,
                    lanes=opspace_lanes,
                    tiles_per_lane=opspace_tiles_per_lane,
                    padded_ops=opspace_padded_ops,
                    tile_size=opspace_tile_size,
                    k_exec=opspace_k_exec,
                    exec_tiles_per_block=opspace_exec_tiles_per_block,
                    blocks_per_lane=opspace_blocks_per_lane,
                    block_size=opspace_block_size,
                    token_chunk_size=opspace_token_chunk_size))
        raise ValueError(
            "operation_space paired QK execution_mode must be "
            f"'dense_masked' or 'block_bucketed_dense', got "
            f"{execution_mode!r}.")
    del operation_space_mode, operation_space_routing_mode
    del operation_space_execution_mode
    del opspace_lanes, opspace_tiles_per_lane
    del opspace_exec_tiles_per_block, opspace_blocks_per_lane
    del opspace_block_size
    del opspace_padded_ops, opspace_tile_size, opspace_k_exec
    del opspace_token_chunk_size
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
        cs = min(int(max_chunk_size), int(N_local))
        nc = (int(N_local) + cs - 1) // cs
        N_pad = nc * cs
        pad_n = N_pad - int(N_local)

        B, S, _, _ = h.shape
        D = x.shape[-1]
        x_bf = x.astype(jnp.bfloat16)
        op_key_padded = jnp.pad(op_key_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(N_pad) < N_local
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        op_key_dir_bf = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir_bf = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir_bf = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        tau = _tau_from_param(raw_tau)

        def op_key_chunk(start):
            return jax.lax.dynamic_slice_in_dim(
                op_key_dir_bf, start, cs, axis=0)

        def operator_relation(op_key):
            return jnp.einsum(
                'bsrd,nd->bsrn', h_unit_bf, op_key).astype(jnp.float32)

        def op_key_rw_chunk(start):
            ec = op_key_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_dir_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_dir_bf, start, cs, axis=0)
            vc = jax.lax.dynamic_slice_in_dim(valid_padded, start, cs, axis=0)
            return ec, rc, wc, vc

        def angular_compose_parts(rho, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        @jax.checkpoint
        def gate_srw_step(carry, i):
            raw_out, total_den_cost = carry
            s = i * cs
            op_key, rc, wc, valid_chunk = op_key_rw_chunk(s)
            valid_bsrn = valid_chunk[None, None, None, :]
            rho_raw = operator_relation(op_key)
            rho_compute = jnp.where(valid_bsrn, rho_raw, tau)
            admission, execution_weight = angular_compose_parts(
                rho_compute, valid_bsrn)
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


def make_sharded_srw_tau_free_relu_dense_masked_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=8,
        tiles_per_lane=12,
        padded_ops=0,
        tile_size=128,
        k_exec=5,
        token_chunk_size=256):
    """Create tau-free relu2 tile-mask SRW using dense local matmuls."""
    del max_chunk_size, dead_exposure_target, padded_ops, token_chunk_size
    del soft_gate_effective_active_eps, admission_den_grad_scale
    layout = _opspace_model_axis_layout(
        mesh, lanes, tiles_per_lane, tile_size)
    _lanes = layout['lanes']
    _tiles_per_lane = layout['tiles_per_lane']
    _tile_size = layout['tile_size']
    _model_axis_size = layout['model_axis_size']
    _local_lanes = layout['local_lanes']
    _local_tile_count = layout['local_tile_count']
    _local_padded_ops = layout['local_padded_ops']
    _k_exec = max(1, min(int(k_exec), _lanes))
    _exec_slots = _k_exec * _tile_size
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None), P(), P()),
             check_rep=False)
    def fused_gate_srw_tau_free_relu_dense_masked(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens):
        del raw_tau, soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power, soft_gate_boundary_power_final
        del execution_prune_eps, training_tokens
        B, S, D = x.shape
        N_local = int(op_key_local.shape[0])
        if N_local > _local_padded_ops:
            raise ValueError(
                "operation-space dense-masked local rows exceed lane-block "
                f"slots: rows={N_local}, slots={_local_padded_ops}")
        (slot_to_row_np, valid_slot_np, valid_counts_np) = (
            _opspace_balanced_slot_layout_np(
                N_local, _local_tile_count, _tile_size))
        slot_to_row = jnp.asarray(slot_to_row_np, dtype=jnp.int32)
        valid_slot = jnp.asarray(valid_slot_np, dtype=jnp.bool_)
        valid_counts_local = jnp.asarray(
            valid_counts_np.reshape(_local_lanes, _tiles_per_lane),
            dtype=jnp.float32)
        model_axis_index = jax.lax.axis_index('model')

        op_key_padded = op_key_local[slot_to_row]
        read_padded = read_local[slot_to_row]
        write_padded = write_local[slot_to_row]
        op_key_padded = jnp.where(
            valid_slot[:, None], op_key_padded, jnp.zeros_like(op_key_padded))
        read_padded = jnp.where(
            valid_slot[:, None], read_padded, jnp.zeros_like(read_padded))
        write_padded = jnp.where(
            valid_slot[:, None], write_padded, jnp.zeros_like(write_padded))

        op_key_dir = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)

        key_tiles = op_key_dir.reshape(
            _local_lanes, _tiles_per_lane, _tile_size, op_key_dir.shape[-1])
        valid_tiles = valid_slot.reshape(
            _local_lanes, _tiles_per_lane, _tile_size)
        valid_f = valid_tiles.astype(jnp.float32)[..., None]
        tile_count = jnp.maximum(valid_f.sum(axis=2), 1.0)
        tile_mean_raw = (
            key_tiles.astype(jnp.float32) * valid_f).sum(axis=2) / tile_count
        tile_mean_local = jax.lax.stop_gradient(
            _forward_unit_direction(tile_mean_raw)).astype(jnp.bfloat16)
        tile_mean_all = jax.lax.all_gather(
            tile_mean_local, 'model', axis=0, tiled=True)
        valid_counts_all = jax.lax.all_gather(
            valid_counts_local, 'model', axis=0, tiled=True)

        h_unit = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
        scores = jnp.einsum(
            'bsd,ltd->bslt', h_unit, tile_mean_all).astype(jnp.float32)
        best_score_by_lane = scores.max(axis=-1)
        best_tile_by_lane = jnp.argmax(scores, axis=-1).astype(jnp.int32)
        _selected_scores, selected_lanes = jax.lax.top_k(
            best_score_by_lane, _k_exec)
        selected_tile = jnp.take_along_axis(
            best_tile_by_lane, selected_lanes, axis=2)

        lane_ids = jnp.arange(_lanes, dtype=jnp.int32)
        tile_ids = jnp.arange(_tiles_per_lane, dtype=jnp.int32)
        lane_match = (
            lane_ids[None, None, :, None] == selected_lanes[:, :, None, :])
        tile_match = (
            tile_ids[None, None, None, :, None]
            == selected_tile[:, :, None, None, :])
        selected_mask_global = jnp.any(
            jnp.logical_and(lane_match[:, :, :, None, :], tile_match),
            axis=-1)
        local_lane_start = model_axis_index * _local_lanes
        selected_mask_local = jax.lax.dynamic_slice_in_dim(
            selected_mask_global, local_lane_start, _local_lanes, axis=2)
        row_mask = jnp.broadcast_to(
            selected_mask_local[..., None],
            (B, S, _local_lanes, _tiles_per_lane, _tile_size)
        ).reshape(B, S, _local_padded_ops)
        row_mask = jnp.logical_and(row_mask, valid_slot[None, None, :])

        rho_local = jnp.einsum(
            'bsd,nd->bsn', h_unit, op_key_dir).astype(jnp.float32)
        read_value = jnp.einsum(
            'bsd,nd->bsn', x.astype(jnp.bfloat16), read_dir).astype(jnp.float32)
        gate_local = jnp.square(jax.nn.relu(rho_local)).astype(jnp.float32)
        gate_local = jnp.where(row_mask, gate_local, 0.0)
        raw_out_local = jnp.einsum(
            'bsn,nd->bsd',
            (gate_local * read_value).astype(jnp.bfloat16),
            write_dir).astype(jnp.float32)
        gate_mass_local = gate_local.sum(axis=-1, keepdims=True)
        relu_gate_count_local = (gate_local > 0.0).astype(jnp.float32).sum(
            axis=-1)

        raw_out = jax.lax.psum(raw_out_local, 'model')
        gate_mass = jax.lax.psum(gate_mass_local, 'model')
        relu_gate_count = jax.lax.psum(relu_gate_count_local, 'model')
        gate_denominator = jnp.power(
            jnp.maximum(gate_mass, 1.0), _admission_den_power)
        out = raw_out / gate_denominator

        def mesh_mean(v):
            return jax.lax.pmean(jax.lax.pmean(v, 'data'), 'model')

        def data_sum(v):
            return jax.lax.pmean(jax.lax.psum(v, 'data'), 'model')

        token_count = jnp.maximum(
            data_sum(jnp.asarray(B * S, dtype=jnp.float32)), 1.0)
        score_sorted = jnp.sort(scores.reshape(-1))
        tile_score_p50 = mesh_mean(
            _v4168_percentile_from_sorted(score_sorted, 50))
        tile_score_p95 = mesh_mean(
            _v4168_percentile_from_sorted(score_sorted, 95))

        tile_hist = selected_mask_global.astype(jnp.float32).sum(axis=(0, 1))
        tile_hist = jax.lax.pmean(jax.lax.psum(tile_hist, 'data'), 'model')
        tile_hist_flat = tile_hist.reshape(-1)
        tile_hist_sorted = jnp.sort(tile_hist_flat)
        tile_mean = jnp.maximum(tile_hist_flat.mean(), 1.0e-6)
        selected_tile_top1_frac = tile_hist_flat.max() / jnp.maximum(
            tile_hist_flat.sum(), 1.0)
        selected_tile_p99_over_mean = (
            _v4168_percentile_from_sorted(tile_hist_sorted, 99) / tile_mean)

        owner_ids = jnp.arange(_model_axis_size, dtype=jnp.int32)
        tile_owner = jnp.repeat(
            lane_ids // jnp.asarray(_local_lanes, dtype=jnp.int32),
            _tiles_per_lane)
        owner_load = (
            (tile_owner[:, None] == owner_ids[None, :]).astype(jnp.float32)
            * tile_hist_flat[:, None]).sum(axis=0)
        owner_load_sorted = jnp.sort(owner_load)
        owner_load_p99_over_mean = (
            _v4168_percentile_from_sorted(owner_load_sorted, 99)
            / jnp.maximum(owner_load.mean(), 1.0e-6))

        lane_hist = (
            selected_lanes[:, :, :, None] == lane_ids[None, None, None, :]
        ).astype(jnp.float32).sum(axis=(0, 1, 2))
        lane_hist = jax.lax.pmean(jax.lax.psum(lane_hist, 'data'), 'model')
        selected_lane_top1_frac = lane_hist.max() / jnp.maximum(
            lane_hist.sum(), 1.0)

        valid_exec_slots = (
            selected_mask_global.astype(jnp.float32)
            * valid_counts_all[None, None, :, :]).sum(axis=(2, 3))
        valid_exec_slots_mean = data_sum(valid_exec_slots.sum()) / token_count
        relu_gate_count_mean = data_sum(relu_gate_count.sum()) / token_count
        gate_denominator_mean = data_sum(gate_denominator.sum()) / token_count
        gate_mass_mean = data_sum(gate_mass.sum()) / token_count
        selected_request_count = token_count * jnp.float32(_k_exec)
        processed_request_count = selected_request_count
        no_nan = jnp.minimum(
            jnp.all(jnp.isfinite(scores)).astype(jnp.float32),
            jnp.all(jnp.isfinite(out)).astype(jnp.float32))
        no_nan = jax.lax.pmin(jax.lax.pmin(no_nan, 'data'), 'model')
        padded_exec_slots_mean = (
            jnp.float32(_exec_slots) - valid_exec_slots_mean)
        layout_ok = jnp.asarray(
            1.0 if _local_padded_ops == _local_tile_count * _tile_size
            else 0.0,
            dtype=jnp.float32)
        diag = jnp.asarray((
            jnp.float32(1.0),
            jnp.float32(_k_exec),
            jnp.float32(_exec_slots),
            valid_exec_slots_mean,
            jnp.float32(0.0),
            jnp.float32(0.0),
            selected_lane_top1_frac,
            jnp.float32(1.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            relu_gate_count_mean,
            gate_denominator_mean,
            gate_mass_mean,
            relu_gate_count_mean,
            padded_exec_slots_mean,
            jnp.float32(0.0),
            no_nan,
            jnp.float32(1.0),
            selected_request_count,
            processed_request_count,
            jnp.float32(1.0),
            layout_ok,
        ), dtype=jnp.float32)
        return out.astype(jnp.float32), diag, jnp.float32(0.0)

    fused_gate_srw_tau_free_relu_dense_masked._v4168_accepts_training_tokens = True
    return fused_gate_srw_tau_free_relu_dense_masked


def make_sharded_srw_paired_tau_free_relu_dense_masked_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=8,
        tiles_per_lane=4,
        padded_ops=0,
        tile_size=128,
        k_exec=4,
        token_chunk_size=256):
    """Create tau-free relu2 dense-masked paired Q/K SRW."""
    del max_chunk_size, dead_exposure_target, padded_ops, token_chunk_size
    del soft_gate_effective_active_eps, admission_den_grad_scale
    layout = _opspace_model_axis_layout(
        mesh, lanes, tiles_per_lane, tile_size)
    _lanes = layout['lanes']
    _tiles_per_lane = layout['tiles_per_lane']
    _tile_size = layout['tile_size']
    _local_lanes = layout['local_lanes']
    _local_tile_count = layout['local_tile_count']
    _local_padded_ops = layout['local_padded_ops']
    _k_exec = max(1, min(int(k_exec), _lanes))
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))

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
    def fused_gate_srw_paired_tau_free_relu_dense_masked(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del raw_tau, soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power, soft_gate_boundary_power_final
        del execution_prune_eps
        B, S, R, _ = h.shape
        D = x.shape[-1]
        N_local = int(op_key_local.shape[0])
        if N_local > _local_padded_ops:
            raise ValueError(
                "QK operation-space dense-masked local rows exceed "
                f"lane-block slots: rows={N_local}, slots={_local_padded_ops}")
        (slot_to_row_np, valid_slot_np, _valid_counts_np) = (
            _opspace_balanced_slot_layout_np(
                N_local, _local_tile_count, _tile_size))
        slot_to_row = jnp.asarray(slot_to_row_np, dtype=jnp.int32)
        valid_slot = jnp.asarray(valid_slot_np, dtype=jnp.bool_)
        model_axis_index = jax.lax.axis_index('model')

        op_key_padded = op_key_local[slot_to_row]
        read_padded = read_local[slot_to_row]
        write_padded = write_local[slot_to_row]
        op_key_padded = jnp.where(
            valid_slot[:, None], op_key_padded, jnp.zeros_like(op_key_padded))
        read_padded = jnp.where(
            valid_slot[:, None], read_padded, jnp.zeros_like(read_padded))
        write_padded = jnp.where(
            valid_slot[:, None], write_padded, jnp.zeros_like(write_padded))
        op_key_dir = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)

        key_tiles = op_key_dir.reshape(
            _local_lanes, _tiles_per_lane, _tile_size, op_key_dir.shape[-1])
        valid_tiles = valid_slot.reshape(
            _local_lanes, _tiles_per_lane, _tile_size)
        valid_f = valid_tiles.astype(jnp.float32)[..., None]
        tile_count = jnp.maximum(valid_f.sum(axis=2), 1.0)
        tile_mean_raw = (
            key_tiles.astype(jnp.float32) * valid_f).sum(axis=2) / tile_count
        tile_mean_local = jax.lax.stop_gradient(
            _forward_unit_direction(tile_mean_raw)).astype(jnp.bfloat16)
        tile_mean_all = jax.lax.all_gather(
            tile_mean_local, 'model', axis=0, tiled=True)

        h_unit = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
        scores = jnp.einsum(
            'bsrd,ltd->bsrlt', h_unit, tile_mean_all).astype(jnp.float32)
        best_score_by_lane = scores.max(axis=-1)
        best_tile_by_lane = jnp.argmax(scores, axis=-1).astype(jnp.int32)
        _selected_scores, selected_lanes = jax.lax.top_k(
            best_score_by_lane, _k_exec)
        selected_tile = jnp.take_along_axis(
            best_tile_by_lane, selected_lanes, axis=3)

        lane_ids = jnp.arange(_lanes, dtype=jnp.int32)
        tile_ids = jnp.arange(_tiles_per_lane, dtype=jnp.int32)
        lane_match = (
            lane_ids[None, None, None, :, None]
            == selected_lanes[:, :, :, None, :])
        tile_match = (
            tile_ids[None, None, None, None, :, None]
            == selected_tile[:, :, :, None, None, :])
        selected_mask_global = jnp.any(
            jnp.logical_and(lane_match[:, :, :, :, None, :], tile_match),
            axis=-1)
        selected_mask_local = jax.lax.dynamic_slice_in_dim(
            selected_mask_global, model_axis_index * _local_lanes,
            _local_lanes, axis=3)
        row_mask = jnp.broadcast_to(
            selected_mask_local[..., None],
            (B, S, R, _local_lanes, _tiles_per_lane, _tile_size)
        ).reshape(B, S, R, _local_padded_ops)
        row_mask = jnp.logical_and(row_mask, valid_slot[None, None, None, :])

        rho_local = jnp.einsum(
            'bsrd,nd->bsrn', h_unit, op_key_dir).astype(jnp.float32)
        read_value = jnp.einsum(
            'bsd,nd->bsn', x.astype(jnp.bfloat16), read_dir).astype(jnp.float32)
        gate_local = jnp.square(jax.nn.relu(rho_local)).astype(jnp.float32)
        gate_local = jnp.where(row_mask, gate_local, 0.0)
        raw_out_local = jnp.einsum(
            'bsrn,nd->bsrd',
            (gate_local * read_value[:, :, None, :]).astype(jnp.bfloat16),
            write_dir).astype(jnp.float32)
        gate_mass_local = gate_local.sum(axis=-1, keepdims=True)
        raw_out = jax.lax.psum(raw_out_local, 'model')
        gate_mass = jax.lax.psum(gate_mass_local, 'model')
        gate_denominator = jnp.power(
            jnp.maximum(gate_mass, 1.0), _admission_den_power)
        return (raw_out / gate_denominator).astype(jnp.float32)

    return fused_gate_srw_paired_tau_free_relu_dense_masked


def _opspace_block_bucketed_layout(mesh, lanes, tiles_per_lane, tile_size,
                                   exec_tiles_per_block, blocks_per_lane,
                                   block_size):
    _exec_tiles_per_block = max(1, int(exec_tiles_per_block))
    _block_size = (
        int(block_size)
        if block_size is not None else int(tile_size) * _exec_tiles_per_block)
    if _block_size != int(tile_size) * _exec_tiles_per_block:
        raise ValueError(
            "block_bucketed_dense requires block_size == "
            "tile_size * exec_tiles_per_block; got "
            f"block_size={_block_size}, tile_size={int(tile_size)}, "
            f"exec_tiles_per_block={_exec_tiles_per_block}")
    if int(tiles_per_lane) % _exec_tiles_per_block != 0:
        raise ValueError(
            "block_bucketed_dense requires exec_tiles_per_block to divide "
            f"tiles_per_lane; got tiles_per_lane={int(tiles_per_lane)}, "
            f"exec_tiles_per_block={_exec_tiles_per_block}")
    _blocks_per_lane = int(tiles_per_lane) // _exec_tiles_per_block
    if blocks_per_lane is not None and int(blocks_per_lane) != _blocks_per_lane:
        raise ValueError(
            "block_bucketed_dense blocks_per_lane mismatch: "
            f"config={int(blocks_per_lane)} derived={_blocks_per_lane}")
    layout = _opspace_model_axis_layout(
        mesh, lanes, tiles_per_lane, tile_size)
    layout.update({
        'exec_tiles_per_block': _exec_tiles_per_block,
        'blocks_per_lane': _blocks_per_lane,
        'block_size': _block_size,
    })
    return layout


def _opspace_tau_free_relu_block_bucketed_dense_core(
        flat_x, flat_h, op_key_local, read_local, write_local, *,
        lanes, tiles_per_lane, tile_size, local_lanes, local_tile_count,
        local_padded_ops, blocks_per_lane, exec_tiles_per_block, block_size,
        k_exec, exec_slots, admission_den_power, bucket_capacity_factor=1.25):
    """Flattened operation-space block executor shared by single and paired."""
    del exec_tiles_per_block
    T = int(flat_x.shape[0])
    D = int(flat_x.shape[-1])
    d_route = int(flat_h.shape[-1])
    N_local = int(op_key_local.shape[0])
    if N_local > local_padded_ops:
        raise ValueError(
            "operation-space block_bucketed_dense local rows exceed "
            f"lane-block slots: rows={N_local}, slots={local_padded_ops}")
    bucket_capacity = max(
        1, int(math.ceil(float(bucket_capacity_factor) * float(T)
                         / float(blocks_per_lane))))
    bucket_chunk_size = min(1024, bucket_capacity)
    bucket_chunk_count = (
        bucket_capacity + bucket_chunk_size - 1) // bucket_chunk_size
    bucket_padded_capacity = bucket_chunk_count * bucket_chunk_size
    bucket_capacity_i32 = jnp.asarray(bucket_capacity, dtype=jnp.int32)
    block_ids = jnp.arange(blocks_per_lane, dtype=jnp.int32)

    (slot_to_row_np, valid_slot_np, _valid_counts_np) = (
        _opspace_balanced_slot_layout_np(
            N_local, local_tile_count, tile_size))
    slot_to_row = jnp.asarray(slot_to_row_np, dtype=jnp.int32)
    valid_slot = jnp.asarray(valid_slot_np, dtype=jnp.bool_)
    model_axis_index = jax.lax.axis_index('model')

    op_key_padded = op_key_local[slot_to_row]
    read_padded = read_local[slot_to_row]
    write_padded = write_local[slot_to_row]
    op_key_padded = jnp.where(
        valid_slot[:, None], op_key_padded, jnp.zeros_like(op_key_padded))
    read_padded = jnp.where(
        valid_slot[:, None], read_padded, jnp.zeros_like(read_padded))
    write_padded = jnp.where(
        valid_slot[:, None], write_padded, jnp.zeros_like(write_padded))

    op_key_dir = _forward_unit_direction(
        op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    read_dir = _forward_unit_direction(
        read_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    write_dir = _forward_unit_direction(
        write_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)

    key_tiles = op_key_dir.reshape(
        local_lanes, tiles_per_lane, tile_size, op_key_dir.shape[-1])
    read_tiles = read_dir.reshape(
        local_lanes, tiles_per_lane, tile_size, read_dir.shape[-1])
    write_tiles = write_dir.reshape(
        local_lanes, tiles_per_lane, tile_size, write_dir.shape[-1])
    valid_tiles = valid_slot.reshape(local_lanes, tiles_per_lane, tile_size)

    key_blocks = key_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size,
        key_tiles.shape[-1]).reshape(
            local_lanes, blocks_per_lane, block_size, key_tiles.shape[-1])
    read_blocks = read_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size,
        read_tiles.shape[-1]).reshape(
            local_lanes, blocks_per_lane, block_size, read_tiles.shape[-1])
    write_blocks = write_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size,
        write_tiles.shape[-1]).reshape(
            local_lanes, blocks_per_lane, block_size, write_tiles.shape[-1])
    valid_blocks = valid_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size).reshape(
            local_lanes, blocks_per_lane, block_size)

    valid_f = valid_blocks.astype(jnp.float32)[..., None]
    block_count = jnp.maximum(valid_f.sum(axis=2), 1.0)
    block_center_raw = (
        key_blocks.astype(jnp.float32) * valid_f).sum(axis=2) / block_count
    block_center_local = jax.lax.stop_gradient(
        _forward_unit_direction(block_center_raw)).astype(jnp.bfloat16)
    block_center_all = jax.lax.all_gather(
        block_center_local, 'model', axis=0, tiled=True)

    flat_h = _forward_unit_direction(
        flat_h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    flat_x = flat_x.astype(jnp.bfloat16)
    scores_all = jnp.einsum(
        'td,lbd->tlb', flat_h, block_center_all).astype(jnp.float32)
    best_score_by_lane = scores_all.max(axis=-1)
    _selected_lane_scores, selected_global_lanes = jax.lax.top_k(
        best_score_by_lane, k_exec)
    lane_ids_global = jnp.arange(lanes, dtype=jnp.int32)
    selected_lane_mask_global = jnp.any(
        selected_global_lanes[:, :, None] == lane_ids_global[None, None, :],
        axis=1)
    local_lane_start = model_axis_index * local_lanes
    selected_lane_mask_local = jax.lax.dynamic_slice_in_dim(
        selected_lane_mask_global, local_lane_start, local_lanes, axis=1)
    scores_local = jax.lax.dynamic_slice_in_dim(
        scores_all, local_lane_start, local_lanes, axis=1)
    token_ids_all = jnp.arange(T, dtype=jnp.int32)

    def assign_lane(scores_lane, lane_active):
        _candidate_scores, candidate_blocks = jax.lax.top_k(
            scores_lane, blocks_per_lane)
        del _candidate_scores

        def candidate_pass(carry, pass_i):
            (assigned_block, assigned_slot, unresolved, bucket_counts,
             primary_accepts, reroute_accepts) = carry
            candidate = jax.lax.dynamic_index_in_dim(
                candidate_blocks, pass_i, axis=1, keepdims=False)
            candidate_match = candidate[:, None] == block_ids[None, :]
            pass_mask = jnp.logical_and(unresolved[:, None], candidate_match)
            pass_ranks = jnp.cumsum(pass_mask.astype(jnp.int32), axis=0) - 1
            rank_for_token = jnp.sum(
                jnp.where(pass_mask, pass_ranks, 0), axis=1)
            slot = bucket_counts[candidate] + rank_for_token
            accept = jnp.logical_and(unresolved, slot < bucket_capacity_i32)
            accepted_by_block = (
                candidate_match.astype(jnp.int32)
                * accept[:, None].astype(jnp.int32)).sum(axis=0)
            bucket_counts = bucket_counts + accepted_by_block
            assigned_block = jnp.where(accept, candidate, assigned_block)
            assigned_slot = jnp.where(accept, slot, assigned_slot)
            unresolved = jnp.logical_and(unresolved, jnp.logical_not(accept))
            accept_count = accept.astype(jnp.float32).sum()
            primary_accepts = primary_accepts + jnp.where(
                pass_i == jnp.asarray(0, dtype=jnp.int32),
                accept_count, jnp.float32(0.0))
            reroute_accepts = reroute_accepts + jnp.where(
                pass_i == jnp.asarray(0, dtype=jnp.int32),
                jnp.float32(0.0), accept_count)
            return (
                assigned_block, assigned_slot, unresolved, bucket_counts,
                primary_accepts, reroute_accepts), None

        init = (
            jnp.full((T,), -1, dtype=jnp.int32),
            jnp.zeros((T,), dtype=jnp.int32),
            lane_active.astype(jnp.bool_),
            jnp.zeros((blocks_per_lane,), dtype=jnp.int32),
            jnp.float32(0.0),
            jnp.float32(0.0),
        )
        (assigned_block, assigned_slot, unresolved, bucket_counts,
         primary_accepts, reroute_accepts) = jax.lax.scan(
            candidate_pass, init,
            jnp.arange(blocks_per_lane, dtype=jnp.int32))[0]

        def fallback_pass(carry, block_i):
            (assigned_block, assigned_slot, unresolved, bucket_counts,
             reroute_accepts) = carry
            rank_for_token = jnp.cumsum(unresolved.astype(jnp.int32)) - 1
            slot = bucket_counts[block_i] + rank_for_token
            accept = jnp.logical_and(unresolved, slot < bucket_capacity_i32)
            bucket_counts = bucket_counts.at[block_i].add(
                accept.astype(jnp.int32).sum())
            assigned_block = jnp.where(accept, block_i, assigned_block)
            assigned_slot = jnp.where(accept, slot, assigned_slot)
            unresolved = jnp.logical_and(unresolved, jnp.logical_not(accept))
            reroute_accepts = reroute_accepts + accept.astype(jnp.float32).sum()
            return (
                assigned_block, assigned_slot, unresolved, bucket_counts,
                reroute_accepts), None

        (assigned_block, assigned_slot, unresolved, bucket_counts,
         reroute_accepts) = jax.lax.scan(
            fallback_pass,
            (assigned_block, assigned_slot, unresolved, bucket_counts,
             reroute_accepts),
            jnp.arange(blocks_per_lane, dtype=jnp.int32))[0]
        unresolved_count = unresolved.astype(jnp.float32).sum()
        return (assigned_block, assigned_slot, unresolved_count, bucket_counts,
                primary_accepts, reroute_accepts)

    @jax.checkpoint
    def lane_step(carry, lane_i):
        (flat_raw_out, flat_gate_mass, flat_relu_gate_count,
         processed_count, valid_exec_slots_sum, primary_accept_count,
         reroute_accept_count, overflow_count, bucket_fill_sum,
         bucket_fill_max, finite_acc) = carry
        scores_lane = scores_local[:, lane_i, :]
        lane_active = selected_lane_mask_local[:, lane_i]
        (assigned_block, assigned_slot, unresolved_count, bucket_counts,
         primary_accepts, reroute_accepts) = assign_lane(
            scores_lane, lane_active)
        valid_assignment = assigned_block >= 0
        safe_block = jnp.where(valid_assignment, assigned_block, 0)
        safe_slot = jnp.where(valid_assignment, assigned_slot, 0)
        valid_f_token = valid_assignment.astype(jnp.float32)

        bucket_valid_count = jnp.zeros(
            (blocks_per_lane, bucket_padded_capacity), dtype=jnp.float32)
        token_id_bucket = jnp.zeros(
            (blocks_per_lane, bucket_padded_capacity), dtype=jnp.int32)
        bucket_valid_count = bucket_valid_count.at[
            safe_block, safe_slot].add(valid_f_token, mode='drop')
        token_id_bucket = token_id_bucket.at[safe_block, safe_slot].max(
            token_ids_all * valid_assignment.astype(jnp.int32), mode='drop')
        assignment_collision_count = jnp.maximum(
            bucket_valid_count - jnp.float32(1.0), jnp.float32(0.0)).sum()
        bucket_valid = bucket_valid_count == jnp.float32(1.0)

        key_lane = key_blocks[lane_i]
        read_lane = read_blocks[lane_i]
        write_lane = write_blocks[lane_i]
        valid_lane = valid_blocks[lane_i]

        def chunk_step(chunk_carry, chunk_i):
            flat_raw_out, flat_gate_mass, flat_relu_gate_count, finite_acc = (
                chunk_carry)
            start = chunk_i * bucket_chunk_size
            token_chunk = jax.lax.dynamic_slice(
                token_id_bucket, (0, start),
                (blocks_per_lane, bucket_chunk_size))
            valid_chunk = jax.lax.dynamic_slice(
                bucket_valid, (0, start),
                (blocks_per_lane, bucket_chunk_size))
            safe_token = jnp.where(valid_chunk, token_chunk, 0)
            h_chunk = flat_h[safe_token]
            x_chunk = flat_x[safe_token]
            valid_chunk_f = valid_chunk.astype(jnp.float32)
            h_chunk = h_chunk * valid_chunk_f[..., None].astype(jnp.bfloat16)
            x_chunk = x_chunk * valid_chunk_f[..., None].astype(jnp.bfloat16)
            rho = jnp.einsum(
                'bcd,bnd->bcn', h_chunk, key_lane).astype(jnp.float32)
            read_value = jnp.einsum(
                'bcd,bnd->bcn', x_chunk, read_lane).astype(jnp.float32)
            gate = jnp.square(jax.nn.relu(rho)).astype(jnp.float32)
            gate = gate * valid_chunk_f[..., None]
            gate = gate * valid_lane.astype(jnp.float32)[:, None, :]
            raw_out_bucket = jnp.einsum(
                'bcn,bnd->bcd',
                (gate * read_value).astype(jnp.bfloat16),
                write_lane).astype(jnp.float32)
            gate_mass_bucket = gate.sum(axis=-1, keepdims=True)
            relu_count_bucket = (gate > 0.0).astype(jnp.float32).sum(axis=-1)

            bucket_valid_f = valid_chunk.reshape(-1).astype(jnp.float32)
            scatter_token = safe_token.reshape(-1)
            flat_raw_out = flat_raw_out.at[scatter_token].add(
                raw_out_bucket.reshape(-1, D) * bucket_valid_f[:, None],
                mode='drop')
            flat_gate_mass = flat_gate_mass.at[scatter_token].add(
                gate_mass_bucket.reshape(-1, 1) * bucket_valid_f[:, None],
                mode='drop')
            flat_relu_gate_count = flat_relu_gate_count.at[scatter_token].add(
                relu_count_bucket.reshape(-1) * bucket_valid_f,
                mode='drop')
            finite = jnp.logical_and(
                jnp.all(jnp.isfinite(gate)),
                jnp.all(jnp.isfinite(raw_out_bucket)))
            finite_acc = jnp.minimum(finite_acc, finite.astype(jnp.float32))
            return (
                flat_raw_out, flat_gate_mass, flat_relu_gate_count,
                finite_acc), None

        (flat_raw_out, flat_gate_mass, flat_relu_gate_count,
         finite_acc) = jax.lax.scan(
            chunk_step,
            (flat_raw_out, flat_gate_mass, flat_relu_gate_count, finite_acc),
            jnp.arange(bucket_chunk_count, dtype=jnp.int32))[0]

        valid_counts_by_block = valid_lane.astype(jnp.float32).sum(axis=1)
        unique_by_block = bucket_valid.astype(jnp.float32).sum(axis=1)
        processed_now = unique_by_block.sum()
        valid_exec_slots_sum = valid_exec_slots_sum + jnp.sum(
            unique_by_block * valid_counts_by_block)
        processed_count = processed_count + processed_now
        primary_accept_count = primary_accept_count + primary_accepts
        reroute_accept_count = reroute_accept_count + reroute_accepts
        assignment_error_count = unresolved_count + assignment_collision_count
        overflow_count = overflow_count + assignment_error_count
        bucket_fill_sum = bucket_fill_sum + unique_by_block.sum()
        bucket_fill_max = jnp.maximum(bucket_fill_max, unique_by_block.max())
        return (
            flat_raw_out, flat_gate_mass, flat_relu_gate_count,
            processed_count, valid_exec_slots_sum, primary_accept_count,
            reroute_accept_count, overflow_count, bucket_fill_sum,
            bucket_fill_max, finite_acc), None

    init_carry = (
        jnp.zeros((T, D), dtype=jnp.float32),
        jnp.zeros((T, 1), dtype=jnp.float32),
        jnp.zeros((T,), dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    (flat_raw_out_local, flat_gate_mass_local, flat_relu_gate_count_local,
     processed_count_local, valid_exec_slots_sum_local,
     primary_accept_count_local, reroute_accept_count_local,
     overflow_count_local, bucket_fill_sum_local, bucket_fill_max_local,
     compute_no_nan) = jax.lax.scan(
        lane_step, init_carry,
        jnp.arange(local_lanes, dtype=jnp.int32))[0]

    global_raw_out = jax.lax.psum(flat_raw_out_local, 'model')
    global_gate_mass = jax.lax.psum(flat_gate_mass_local, 'model')
    global_relu_gate_count = jax.lax.psum(
        flat_relu_gate_count_local, 'model')
    global_processed_count = jax.lax.psum(processed_count_local, 'model')
    global_valid_exec_slots_sum = jax.lax.psum(
        valid_exec_slots_sum_local, 'model')
    global_primary_accept_count = jax.lax.psum(
        primary_accept_count_local, 'model')
    global_reroute_accept_count = jax.lax.psum(
        reroute_accept_count_local, 'model')
    global_overflow_count = jax.lax.psum(overflow_count_local, 'model')
    global_bucket_fill_sum = jax.lax.psum(bucket_fill_sum_local, 'model')
    global_bucket_fill_max = jax.lax.pmax(bucket_fill_max_local, 'model')

    gate_denominator = jnp.power(
        jnp.maximum(global_gate_mass, 1.0), admission_den_power)
    flat_out = global_raw_out / gate_denominator

    def data_sum(v):
        return jax.lax.pmean(jax.lax.psum(v, 'data'), 'model')

    token_count = jnp.maximum(
        data_sum(jnp.asarray(T, dtype=jnp.float32)), 1.0)
    selected_request_count = token_count * jnp.float32(k_exec)
    processed_request_count = data_sum(global_processed_count)
    valid_exec_slots_mean = (
        data_sum(global_valid_exec_slots_sum) / token_count)
    relu_gate_count_mean = (
        data_sum(global_relu_gate_count.sum()) / token_count)
    gate_denominator_mean = (
        data_sum(gate_denominator.sum()) / token_count)
    gate_mass_mean = data_sum(global_gate_mass.sum()) / token_count
    primary_accept_frac = (
        data_sum(global_primary_accept_count)
        / jnp.maximum(selected_request_count, 1.0))
    reroute_frac = (
        data_sum(global_reroute_accept_count)
        / jnp.maximum(selected_request_count, 1.0))
    overflow_count_global = data_sum(global_overflow_count)
    overflow_frac = overflow_count_global / jnp.maximum(
        selected_request_count, 1.0)
    bucket_count_global = jnp.float32(lanes * blocks_per_lane)
    bucket_fill_mean = (
        data_sum(global_bucket_fill_sum)
        / jnp.maximum(bucket_count_global, 1.0))
    bucket_fill_max = jax.lax.pmax(
        jax.lax.pmax(global_bucket_fill_max, 'data'), 'model')
    bucket_fill_skew = bucket_fill_max / jnp.maximum(bucket_fill_mean, 1.0)

    region_hist = selected_lane_mask_global.astype(jnp.float32).sum(axis=0)
    region_hist = jax.lax.pmean(jax.lax.psum(region_hist, 'data'), 'model')
    selected_region_top1_frac = region_hist.max() / jnp.maximum(
        region_hist.sum(), 1.0)
    region_hist_sorted = jnp.sort(region_hist)
    region_load_mean = jnp.maximum(region_hist.mean(), jnp.float32(1.0e-6))
    region_load_p99_over_mean = (
        _opspace_percentile(region_hist_sorted, 99.0) / region_load_mean)
    score_no_nan = jnp.all(jnp.isfinite(scores_all)).astype(jnp.float32)
    score_no_nan = jax.lax.pmin(
        jax.lax.pmin(score_no_nan, 'data'), 'model')
    compute_no_nan = jax.lax.pmin(
        jax.lax.pmin(compute_no_nan, 'data'), 'model')
    out_no_nan = jnp.all(jnp.isfinite(flat_out)).astype(jnp.float32)
    out_no_nan = jax.lax.pmin(jax.lax.pmin(out_no_nan, 'data'), 'model')
    no_nan = jnp.minimum(score_no_nan, jnp.minimum(compute_no_nan, out_no_nan))
    padded_exec_slots_mean = jnp.float32(exec_slots) - valid_exec_slots_mean
    all_requests_processed = jnp.logical_and(
        processed_request_count >= selected_request_count - 0.5,
        overflow_count_global <= 0.0).astype(jnp.float32)
    layout_ok = jnp.asarray(
        1.0 if local_padded_ops == (
            local_lanes * blocks_per_lane * block_size) else 0.0,
        dtype=jnp.float32)

    diag = jnp.asarray((
        jnp.float32(1.0),
        jnp.float32(k_exec),
        jnp.float32(exec_slots),
        valid_exec_slots_mean,
        jnp.float32(bucket_capacity),
        bucket_fill_mean,
        selected_region_top1_frac,
        primary_accept_frac,
        bucket_fill_skew,
        reroute_frac,
        relu_gate_count_mean,
        gate_denominator_mean,
        gate_mass_mean,
        relu_gate_count_mean,
        padded_exec_slots_mean,
        overflow_frac,
        no_nan,
        jnp.float32(1.0),
        selected_request_count,
        processed_request_count,
        all_requests_processed,
        layout_ok,
    ), dtype=jnp.float32)
    return flat_out.astype(jnp.float32), global_gate_mass, diag


def _opspace_percentile(values, q):
    flat = values.reshape(-1).astype(jnp.float32)
    count = flat.shape[0]
    if count <= 0:
        return jnp.float32(0.0)
    sorted_values = jnp.sort(flat)
    idx = int(math.ceil((float(q) / 100.0) * float(count - 1)))
    idx = max(0, min(count - 1, idx))
    return sorted_values[idx]


def _opspace_masked_percentile(values, mask, q):
    flat_values = values.reshape(-1).astype(jnp.float32)
    flat_mask = mask.reshape(-1).astype(jnp.bool_)
    count = flat_mask.astype(jnp.int32).sum()
    sorted_values = jnp.sort(
        jnp.where(flat_mask, flat_values, jnp.float32(1.0e9)))
    max_idx = jnp.maximum(count - 1, 0)
    idx_f = jnp.ceil(
        (jnp.asarray(q, dtype=jnp.float32) / jnp.float32(100.0))
        * max_idx.astype(jnp.float32))
    idx = jnp.minimum(idx_f.astype(jnp.int32), max_idx)
    return jnp.where(
        count > 0, sorted_values[idx], jnp.float32(0.0))


def _opspace_region_first_capacity_admission(
        region_scores, *, visible_regions, region_capacity,
        high_regret_threshold=0.05):
    """Admit token requests to capacity-bounded regions before block routing."""
    T = int(region_scores.shape[0])
    region_count = int(region_scores.shape[1])
    visible_regions = max(1, min(int(visible_regions), region_count))
    region_capacity_i32 = jnp.asarray(region_capacity, dtype=jnp.int32)
    region_ids = jnp.arange(region_count, dtype=jnp.int32)
    # Every region is a candidate. Capacity-aware low-regret rerouting happens
    # across the whole atlas instead of a truncated top-k candidate set.
    candidate_scores, candidate_regions = jax.lax.top_k(
        region_scores.astype(jnp.float32), region_count)

    def candidate_pass(carry, pass_i):
        (accepted_regions, accepted_scores, accepted_best_scores,
         accepted_rank, accepted_count, region_counts) = carry
        candidate = jax.lax.dynamic_index_in_dim(
            candidate_regions, pass_i, axis=1, keepdims=False)
        curr_score = jax.lax.dynamic_index_in_dim(
            candidate_scores, pass_i, axis=1, keepdims=False)
        unresolved = accepted_count < jnp.asarray(
            visible_regions, dtype=jnp.int32)
        candidate_match = jnp.logical_and(
            unresolved[:, None],
            candidate[:, None] == region_ids[None, :])
        rank_in_region = (
            jnp.cumsum(candidate_match.astype(jnp.int32), axis=0) - 1)
        available = jnp.maximum(region_capacity_i32 - region_counts, 0)
        accept_by_region = jnp.logical_and(
            candidate_match, rank_in_region < available[None, :])
        accept = accept_by_region.any(axis=1)
        safe_slot = jnp.minimum(
            accepted_count,
            jnp.asarray(visible_regions - 1, dtype=jnp.int32))
        best_score = jnp.take_along_axis(
            candidate_scores, safe_slot[:, None], axis=1).reshape((T,))
        slot_mask = jax.nn.one_hot(
            safe_slot, visible_regions, dtype=jnp.int32).astype(jnp.bool_)
        write_mask = jnp.logical_and(accept[:, None], slot_mask)
        accepted_regions = jnp.where(
            write_mask, candidate[:, None], accepted_regions)
        accepted_scores = jnp.where(
            write_mask, curr_score[:, None], accepted_scores)
        accepted_best_scores = jnp.where(
            write_mask, best_score[:, None], accepted_best_scores)
        accepted_rank = jnp.where(
            write_mask, pass_i, accepted_rank)
        accepted_count = accepted_count + accept.astype(jnp.int32)
        region_counts = region_counts + accept_by_region.astype(
            jnp.int32).sum(axis=0)
        return (
            accepted_regions, accepted_scores, accepted_best_scores,
            accepted_rank, accepted_count, region_counts), None

    init = (
        jnp.full((T, visible_regions), -1, dtype=jnp.int32),
        jnp.zeros((T, visible_regions), dtype=jnp.float32),
        jnp.zeros((T, visible_regions), dtype=jnp.float32),
        jnp.full((T, visible_regions), -1, dtype=jnp.int32),
        jnp.zeros((T,), dtype=jnp.int32),
        jnp.zeros((region_count,), dtype=jnp.int32),
    )
    (accepted_regions, accepted_scores, accepted_best_scores,
     accepted_rank, accepted_count, region_counts) = jax.lax.scan(
        candidate_pass, init,
        jnp.arange(region_count, dtype=jnp.int32))[0]

    accepted_mask = accepted_regions >= 0
    accepted_f = accepted_mask.astype(jnp.float32)
    selected_request_count = jnp.float32(T * visible_regions)
    accepted_request_count = accepted_f.sum()
    unresolved_count = selected_request_count - accepted_request_count
    slot_ids = jnp.arange(visible_regions, dtype=jnp.int32)[None, :]
    score_loss = jnp.where(
        accepted_mask,
        accepted_best_scores - accepted_scores,
        jnp.float32(0.0))
    spill_mask = jnp.logical_and(accepted_mask, accepted_rank > slot_ids)
    high_regret_threshold_f = jnp.asarray(
        high_regret_threshold, dtype=jnp.float32)
    low_regret_spill = jnp.logical_and(
        spill_mask, score_loss <= high_regret_threshold_f)
    high_regret_spill = jnp.logical_and(
        spill_mask, score_loss > high_regret_threshold_f)
    requested_best_score_sum = candidate_scores[:, :visible_regions].sum()
    chosen_score_sum = (accepted_scores * accepted_f).sum()
    score_loss_sum = (score_loss * accepted_f).sum()
    score_loss_max = jnp.max(jnp.where(
        accepted_mask, score_loss, jnp.float32(0.0)))
    score_loss_p95 = score_loss_max

    return {
        'accepted_regions': accepted_regions,
        'accepted_mask': accepted_mask,
        'accepted_rank': accepted_rank,
        'accepted_count_per_token': accepted_count,
        'region_load': region_counts.astype(jnp.float32),
        'accepted_request_count': accepted_request_count,
        'unresolved_count': unresolved_count,
        'reroute_count': spill_mask.astype(jnp.float32).sum(),
        'low_regret_spill_count': low_regret_spill.astype(jnp.float32).sum(),
        'high_regret_spill_count': high_regret_spill.astype(jnp.float32).sum(),
        'best_score_sum': requested_best_score_sum,
        'chosen_score_sum': chosen_score_sum,
        'score_loss_sum': score_loss_sum,
        'score_loss_p95': score_loss_p95,
        'score_loss_max': score_loss_max,
    }


def _opspace_low_regret_assign_region_blocks(
        scores_region, region_admitted, *, blocks_per_region, block_capacity,
        visible_blocks_per_region=1, high_regret_threshold=0.05):
    """Assign admitted requests to sibling blocks inside one region."""
    del high_regret_threshold
    T = int(scores_region.shape[0])
    visible_blocks = max(
        1, min(int(visible_blocks_per_region), int(blocks_per_region)))
    block_capacity_i32 = jnp.asarray(block_capacity, dtype=jnp.int32)
    block_ids = jnp.arange(blocks_per_region, dtype=jnp.int32)
    candidate_scores, candidate_blocks = jax.lax.top_k(
        scores_region.astype(jnp.float32), blocks_per_region)

    def block_pass(carry, pass_i):
        (assigned_block, assigned_rank, chosen_score, best_score,
         accepted_count, block_counts) = carry
        candidate = jax.lax.dynamic_index_in_dim(
            candidate_blocks, pass_i, axis=1, keepdims=False)
        curr_score = jax.lax.dynamic_index_in_dim(
            candidate_scores, pass_i, axis=1, keepdims=False)
        unresolved = jnp.logical_and(
            region_admitted.astype(jnp.bool_),
            accepted_count < jnp.asarray(visible_blocks, dtype=jnp.int32))
        candidate_match = jnp.logical_and(
            unresolved[:, None],
            candidate[:, None] == block_ids[None, :])
        rank_in_block = (
            jnp.cumsum(candidate_match.astype(jnp.int32), axis=0) - 1)
        available = jnp.maximum(block_capacity_i32 - block_counts, 0)
        accept_by_block = jnp.logical_and(
            candidate_match, rank_in_block < available[None, :])
        accept = accept_by_block.any(axis=1)
        safe_slot = jnp.minimum(
            accepted_count,
            jnp.asarray(visible_blocks - 1, dtype=jnp.int32))
        slot_mask = jax.nn.one_hot(
            safe_slot, visible_blocks, dtype=jnp.int32).astype(jnp.bool_)
        write_mask = jnp.logical_and(accept[:, None], slot_mask)
        requested_best_score = jnp.take_along_axis(
            candidate_scores, safe_slot[:, None], axis=1).reshape((T,))
        assigned_block = jnp.where(
            write_mask, candidate[:, None], assigned_block)
        assigned_rank = jnp.where(
            write_mask, pass_i, assigned_rank)
        chosen_score = jnp.where(
            write_mask, curr_score[:, None], chosen_score)
        best_score = jnp.where(
            write_mask, requested_best_score[:, None], best_score)
        accepted_count = accepted_count + accept.astype(jnp.int32)
        block_counts = block_counts + accept_by_block.astype(
            jnp.int32).sum(axis=0)
        return (
            assigned_block, assigned_rank, chosen_score, best_score,
            accepted_count, block_counts), None

    init = (
        jnp.full((T, visible_blocks), -1, dtype=jnp.int32),
        jnp.full((T, visible_blocks), -1, dtype=jnp.int32),
        jnp.zeros((T, visible_blocks), dtype=jnp.float32),
        jnp.zeros((T, visible_blocks), dtype=jnp.float32),
        jnp.zeros((T,), dtype=jnp.int32),
        jnp.zeros((blocks_per_region,), dtype=jnp.int32),
    )
    (assigned_block, assigned_rank, chosen_score, best_score, accepted_count,
     block_counts) = jax.lax.scan(
        block_pass, init,
        jnp.arange(blocks_per_region, dtype=jnp.int32))[0]
    unresolved = jnp.logical_and(
        region_admitted[:, None].astype(jnp.bool_),
        assigned_block < 0)
    primary_regret = jnp.where(
        assigned_block >= 0, best_score - chosen_score, jnp.float32(0.0))
    return (
        assigned_block, assigned_rank, chosen_score, best_score,
        primary_regret, unresolved, block_counts,
        candidate_blocks[:, :visible_blocks])


def _opspace_select_region_blocks(
        flat_h, op_key_region_blocks, valid_region_blocks, *,
        num_regions, local_regions, blocks_per_region, operators_per_block,
        visible_regions, visible_blocks_per_region,
        region_score_temperature=0.25, capacity_policy=None):
    """Shared Region/Block/Operator Atlas selector."""
    del operators_per_block
    flat_h_unit = _forward_unit_direction(
        flat_h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    valid_f = valid_region_blocks.astype(jnp.float32)[..., None]
    block_count = jnp.maximum(valid_f.sum(axis=2), 1.0)
    block_center_raw = (
        op_key_region_blocks.astype(jnp.float32) * valid_f).sum(
            axis=2) / block_count
    block_center_local = jax.lax.stop_gradient(
        _forward_unit_direction(block_center_raw)).astype(jnp.bfloat16)
    block_center_all = jax.lax.all_gather(
        block_center_local, 'model', axis=0, tiled=True)
    scores_all = jnp.einsum(
        'td,rbd->trb', flat_h_unit, block_center_all).astype(jnp.float32)
    temperature = jnp.maximum(
        jnp.asarray(region_score_temperature, dtype=jnp.float32),
        jnp.float32(1.0e-6))
    region_score = (
        temperature
        * (jax.nn.logsumexp(scores_all / temperature, axis=-1)
           - jnp.log(jnp.float32(blocks_per_region))))

    policy = capacity_policy if isinstance(capacity_policy, dict) else {}
    region_capacity = policy.get('region_capacity', None)
    if region_capacity is None:
        _selected_scores, selected_regions_global = jax.lax.top_k(
            region_score, int(visible_regions))
        selected_mask_global = jnp.any(
            selected_regions_global[:, :, None]
            == jnp.arange(num_regions, dtype=jnp.int32)[None, None, :],
            axis=1)
        admission = {
            'accepted_regions': selected_regions_global,
            'accepted_request_count': jnp.float32(
                int(flat_h.shape[0]) * int(visible_regions)),
            'unresolved_count': jnp.float32(0.0),
            'reroute_count': jnp.float32(0.0),
            'low_regret_spill_count': jnp.float32(0.0),
            'high_regret_spill_count': jnp.float32(0.0),
            'best_score_sum': _selected_scores.sum(),
            'chosen_score_sum': _selected_scores.sum(),
            'score_loss_sum': jnp.float32(0.0),
            'score_loss_p95': jnp.float32(0.0),
            'score_loss_max': jnp.float32(0.0),
            'region_load': selected_mask_global.astype(jnp.float32).sum(
                axis=0),
        }
    else:
        admission = _opspace_region_first_capacity_admission(
            region_score,
            visible_regions=visible_regions,
            region_capacity=int(region_capacity),
            high_regret_threshold=float(
                policy.get('high_regret_threshold', 0.05)))
        selected_regions_global = admission['accepted_regions']
        selected_mask_global = jnp.any(
            selected_regions_global[:, :, None]
            == jnp.arange(num_regions, dtype=jnp.int32)[None, None, :],
            axis=1)

    model_axis_index = jax.lax.axis_index('model')
    local_region_start = model_axis_index * int(local_regions)
    selected_region_mask_local_t = jax.lax.dynamic_slice_in_dim(
        selected_mask_global, local_region_start, int(local_regions), axis=1)
    scores_local = jax.lax.dynamic_slice_in_dim(
        scores_all, local_region_start, int(local_regions), axis=1)

    block_capacity = policy.get('block_capacity', None)
    if block_capacity is None:
        _block_scores, blocks_tlr = jax.lax.top_k(
            scores_local, int(blocks_per_region))
        selected_blocks_local = jnp.transpose(
            blocks_tlr[:, :, :int(visible_blocks_per_region)], (1, 0, 2))
        selected_blocks_local = jnp.where(
            selected_region_mask_local_t.T[:, :, None],
            selected_blocks_local,
            jnp.full_like(selected_blocks_local, -1))
        selected_rank = jnp.broadcast_to(
            jnp.arange(int(visible_blocks_per_region), dtype=jnp.int32),
            selected_blocks_local.shape)
        chosen_score = jnp.transpose(
            _block_scores[:, :, :int(visible_blocks_per_region)], (1, 0, 2))
        best_score = chosen_score
        primary_regret = jnp.zeros_like(chosen_score)
        unresolved = selected_blocks_local < 0
        block_load = jnp.zeros(
            (int(local_regions), int(blocks_per_region)), dtype=jnp.float32)
        primary_block = selected_blocks_local
    else:
        def assign_one(scores_region, region_admitted):
            return _opspace_low_regret_assign_region_blocks(
                scores_region, region_admitted,
                blocks_per_region=int(blocks_per_region),
                block_capacity=int(block_capacity),
                visible_blocks_per_region=int(visible_blocks_per_region),
                high_regret_threshold=float(
                    policy.get('high_regret_threshold', 0.05)))

        (selected_blocks_local, selected_rank, chosen_score, best_score,
         primary_regret, unresolved, block_load, primary_block) = jax.vmap(
            assign_one, in_axes=(1, 1), out_axes=0)(
                scores_local, selected_region_mask_local_t)
        block_load = block_load.astype(jnp.float32)

    block_ids = jnp.arange(blocks_per_region, dtype=jnp.int32)
    selected_block_mask_local = jnp.logical_and(
        selected_region_mask_local_t[:, :, None],
        jnp.any(
            selected_blocks_local.transpose(1, 0, 2)[:, :, :, None]
            == block_ids[None, None, None, :],
            axis=2))
    return {
        'selected_regions_global': selected_regions_global,
        'selected_region_mask_local': selected_region_mask_local_t.T,
        'selected_blocks_local': selected_blocks_local,
        'selected_block_mask_local': selected_block_mask_local,
        'scores_all': scores_all,
        'scores_local': scores_local,
        'region_score': region_score,
        'admission': admission,
        'assigned_rank': selected_rank,
        'chosen_score': chosen_score,
        'best_score': best_score,
        'primary_regret': primary_regret,
        'unresolved': unresolved,
        'block_load': block_load,
        'primary_block': primary_block,
    }


def _opspace_selected_block_bucket_capacity(
        token_count, *, num_regions, blocks_per_region, visible_regions,
        visible_blocks_per_region, block_capacity_factor):
    expected = (
        float(token_count) * float(visible_regions)
        * float(visible_blocks_per_region)
        / float(max(1, int(num_regions) * int(blocks_per_region))))
    capacity = max(1, int(math.ceil(expected * float(block_capacity_factor))))
    round_multiple = max(1, int(OPSPACE_RST_BLOCK_CAPACITY_ROUND_MULTIPLE))
    return int(
        math.ceil(float(capacity) / float(round_multiple)) * round_multiple)


def _opspace_build_selected_block_buckets(
        selected_blocks_local, *, blocks_per_region, block_capacity):
    """Create fixed selected-block token buckets without region outputs."""
    local_regions = int(selected_blocks_local.shape[0])
    T = int(selected_blocks_local.shape[1])
    visible_blocks = int(selected_blocks_local.shape[2])
    flat_count = local_regions * T * visible_blocks
    request_region = jnp.repeat(
        jnp.arange(local_regions, dtype=jnp.int32), T * visible_blocks)
    request_token = jnp.tile(
        jnp.repeat(jnp.arange(T, dtype=jnp.int32), visible_blocks),
        local_regions)
    request_block = selected_blocks_local.reshape(flat_count).astype(jnp.int32)
    request_valid = request_block >= 0

    region_ids = jnp.arange(local_regions, dtype=jnp.int32)
    block_ids = jnp.arange(blocks_per_region, dtype=jnp.int32)
    request_match = jnp.logical_and(
        request_valid[:, None, None],
        jnp.logical_and(
            request_region[:, None, None] == region_ids[None, :, None],
            request_block[:, None, None] == block_ids[None, None, :]))
    rank_by_block = jnp.cumsum(request_match.astype(jnp.int32), axis=0) - 1
    rank = jnp.sum(
        jnp.where(request_match, rank_by_block, 0), axis=(1, 2)).astype(
            jnp.int32)
    in_capacity = jnp.logical_and(
        request_valid,
        rank < jnp.asarray(block_capacity, dtype=jnp.int32))
    safe_region = jnp.where(
        in_capacity, request_region,
        jnp.asarray(local_regions, dtype=jnp.int32))
    safe_block = jnp.where(
        in_capacity, request_block,
        jnp.asarray(blocks_per_region, dtype=jnp.int32))
    safe_rank = jnp.where(
        in_capacity, rank,
        jnp.asarray(block_capacity, dtype=jnp.int32))
    bucket_valid_count = jnp.zeros(
        (local_regions, blocks_per_region, block_capacity),
        dtype=jnp.float32).at[safe_region, safe_block, safe_rank].add(
            in_capacity.astype(jnp.float32), mode='drop')
    token_id_bucket = jnp.zeros(
        (local_regions, blocks_per_region, block_capacity),
        dtype=jnp.int32).at[safe_region, safe_block, safe_rank].max(
            request_token * in_capacity.astype(jnp.int32), mode='drop')
    bucket_valid = bucket_valid_count == jnp.float32(1.0)
    bucket_fill = bucket_valid.astype(jnp.float32).sum(axis=-1)
    assignment_collision_count = jnp.maximum(
        bucket_valid_count - jnp.float32(1.0), jnp.float32(0.0)).sum()
    overflow_count = jnp.logical_and(
        request_valid, jnp.logical_not(in_capacity)).astype(jnp.float32).sum()
    return {
        'token_id_bucket': token_id_bucket,
        'bucket_valid': bucket_valid,
        'bucket_fill': bucket_fill,
        'processed_request_count': bucket_valid.astype(jnp.float32).sum(),
        'assignment_collision_count': assignment_collision_count,
        'overflow_request_count': overflow_count,
    }


def _opspace_execute_rst_selected_block_sparse(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, *, admission_den_power):
    """Execute only selected Region/Block buckets and scatter-add by token."""
    T = int(flat_x.shape[0])
    D = int(flat_x.shape[-1])
    local_regions = int(key_blocks.shape[0])
    blocks_per_region = int(key_blocks.shape[1])
    block_capacity = int(token_id_bucket.shape[-1])
    block_count = local_regions * blocks_per_region
    flat_h = _forward_unit_direction(
        flat_h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    flat_x = flat_x.astype(jnp.bfloat16)

    def block_step(carry, block_i):
        flat_raw_out, flat_gate_mass, flat_relu_gate_count, finite_acc = carry
        region_i = block_i // jnp.asarray(blocks_per_region, dtype=jnp.int32)
        block_j = block_i - region_i * jnp.asarray(
            blocks_per_region, dtype=jnp.int32)
        token_ids = token_id_bucket[region_i, block_j]
        valid = bucket_valid[region_i, block_j]
        safe_token_ids = jnp.where(valid, token_ids, 0)
        valid_f = valid.astype(jnp.float32)
        valid_bf = valid_f.astype(jnp.bfloat16)
        x_b = flat_x[safe_token_ids] * valid_bf[:, None]
        h_b = flat_h[safe_token_ids] * valid_bf[:, None]
        key_b = key_blocks[region_i, block_j].astype(jnp.bfloat16)
        read_b = read_blocks[region_i, block_j].astype(jnp.bfloat16)
        write_b = write_blocks[region_i, block_j].astype(jnp.bfloat16)
        valid_ops = valid_blocks[region_i, block_j].astype(jnp.float32)

        rho = jnp.einsum('cd,od->co', h_b, key_b).astype(jnp.float32)
        read_value = jnp.einsum('cd,od->co', x_b, read_b).astype(jnp.float32)
        gate = jnp.square(jax.nn.relu(rho)).astype(jnp.float32)
        gate = gate * valid_f[:, None]
        gate = gate * valid_ops[None, :]
        raw_b = jnp.einsum(
            'co,od->cd',
            (gate * read_value).astype(jnp.bfloat16),
            write_b).astype(jnp.float32)
        mass_b = gate.sum(axis=-1, keepdims=True)
        relu_b = (gate > 0.0).astype(jnp.float32).sum(axis=-1)

        flat_raw_out = flat_raw_out.at[safe_token_ids].add(
            raw_b * valid_f[:, None], mode='drop')
        flat_gate_mass = flat_gate_mass.at[safe_token_ids].add(
            mass_b * valid_f[:, None], mode='drop')
        flat_relu_gate_count = flat_relu_gate_count.at[safe_token_ids].add(
            relu_b * valid_f, mode='drop')
        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(gate)), jnp.all(jnp.isfinite(raw_b)))
        finite_acc = jnp.minimum(finite_acc, finite.astype(jnp.float32))
        return (
            flat_raw_out, flat_gate_mass, flat_relu_gate_count,
            finite_acc), None

    init = (
        jnp.zeros((T, D), dtype=jnp.float32),
        jnp.zeros((T, 1), dtype=jnp.float32),
        jnp.zeros((T,), dtype=jnp.float32),
        jnp.float32(1.0),
    )
    (flat_raw_out_local, flat_gate_mass_local, flat_relu_gate_count_local,
     compute_no_nan) = jax.lax.scan(
        block_step, init, jnp.arange(block_count, dtype=jnp.int32))[0]
    return (
        *_opspace_finalize_bucketed_output(
            flat_raw_out_local, flat_gate_mass_local,
            flat_relu_gate_count_local, admission_den_power),
        compute_no_nan)


def _opspace_build_region_stationary_buckets(
        scores_local, admitted_region_mask_local, *, blocks_per_region,
        block_capacity, region_capacity, visible_blocks_per_region,
        visible_regions, high_regret_threshold=0.05):
    """Build fixed region-owned token buckets after region/block routing."""
    T = int(scores_local.shape[0])
    local_regions = int(scores_local.shape[1])
    token_ids = jnp.arange(T, dtype=jnp.int32)
    visible_blocks = int(visible_blocks_per_region)
    region_capacity_i32 = jnp.asarray(region_capacity, dtype=jnp.int32)

    def assign_one(scores_region, region_admitted):
        return _opspace_low_regret_assign_region_blocks(
            scores_region, region_admitted,
            blocks_per_region=blocks_per_region,
            block_capacity=block_capacity,
            visible_blocks_per_region=visible_blocks_per_region,
            high_regret_threshold=high_regret_threshold)

    (assigned_block, assigned_rank, chosen_score, best_score,
     primary_regret, unresolved, block_load, primary_block) = jax.vmap(
        assign_one, in_axes=(1, 1), out_axes=0)(
            scores_local, admitted_region_mask_local)
    valid_assignment = assigned_block >= 0
    valid_region_request = jnp.any(valid_assignment, axis=-1)
    region_slot = (
        jnp.cumsum(valid_region_request.astype(jnp.int32), axis=1) - 1)
    bucket_write = jnp.logical_and(
        valid_region_request, region_slot < region_capacity_i32)
    safe_slot = jnp.where(bucket_write, region_slot, 0)
    region_ids = jnp.arange(local_regions, dtype=jnp.int32)[:, None]
    token_ids_l = jnp.broadcast_to(token_ids[None, :], (local_regions, T))
    write_f = bucket_write.astype(jnp.float32)
    bucket_valid_count = jnp.zeros(
        (local_regions, region_capacity),
        dtype=jnp.float32).at[region_ids, safe_slot].add(write_f, mode='drop')
    token_id_bucket = jnp.zeros(
        (local_regions, region_capacity),
        dtype=jnp.int32).at[region_ids, safe_slot].max(
            token_ids_l * bucket_write.astype(jnp.int32), mode='drop')
    block_slot_ids = jnp.arange(visible_blocks, dtype=jnp.int32)[None, None, :]
    region_ids_b = region_ids[:, :, None]
    safe_slot_b = safe_slot[:, :, None]
    block_id_bucket = jnp.full(
        (local_regions, region_capacity, visible_blocks),
        -1, dtype=jnp.int32).at[
            region_ids_b, safe_slot_b, block_slot_ids].max(
                jnp.where(bucket_write[:, :, None], assigned_block, -1),
                mode='drop')
    bucket_valid = bucket_valid_count == jnp.float32(1.0)
    bucket_fill = bucket_valid.astype(jnp.float32).sum(axis=-1)
    assignment_collision_count = jnp.maximum(
        bucket_valid_count - jnp.float32(1.0), jnp.float32(0.0)).sum()
    overflow_region = jnp.logical_and(
        valid_region_request, region_slot >= region_capacity_i32)
    bucket_overflow_request_count = (
        valid_assignment.astype(jnp.float32)
        * overflow_region[:, :, None].astype(jnp.float32)).sum()
    processed_request_count = (
        valid_assignment.astype(jnp.float32)
        * bucket_write[:, :, None].astype(jnp.float32)).sum()
    max_local_visible_regions = int(visible_regions)
    valid_TL = bucket_write.T
    slot_TL = region_slot.T
    rank_TL = jnp.cumsum(valid_TL.astype(jnp.int32), axis=1) - 1
    write_inverse = jnp.logical_and(
        valid_TL, rank_TL < jnp.asarray(
            max_local_visible_regions, dtype=jnp.int32))
    token_ids_t = jnp.arange(T, dtype=jnp.int32)[:, None]
    local_region_ids = jnp.arange(local_regions, dtype=jnp.int32)[None, :]
    safe_rank = jnp.where(
        write_inverse, rank_TL,
        jnp.asarray(max_local_visible_regions, dtype=jnp.int32))
    token_local_region_ids = jnp.zeros(
        (T, max_local_visible_regions), dtype=jnp.int32).at[
            token_ids_t, safe_rank].set(local_region_ids, mode='drop')
    token_region_slots = jnp.zeros(
        (T, max_local_visible_regions), dtype=jnp.int32).at[
            token_ids_t, safe_rank].set(slot_TL, mode='drop')
    token_region_valid = jnp.zeros(
        (T, max_local_visible_regions), dtype=jnp.bool_).at[
            token_ids_t, safe_rank].set(True, mode='drop')
    return {
        'assigned_block': assigned_block,
        'assigned_rank': assigned_rank,
        'chosen_score': chosen_score,
        'best_score': best_score,
        'primary_regret': primary_regret,
        'unresolved': unresolved,
        'block_load': block_load.astype(jnp.float32),
        'bucket_fill': bucket_fill,
        'token_id_bucket': token_id_bucket,
        'bucket_valid': bucket_valid,
        'block_id_bucket': block_id_bucket,
        'token_local_region_ids': token_local_region_ids,
        'token_region_slots': token_region_slots,
        'token_region_valid': token_region_valid,
        'primary_block': primary_block,
        'assignment_collision_count': assignment_collision_count,
        'bucket_overflow_request_count': bucket_overflow_request_count,
        'processed_request_count': processed_request_count,
    }


def _opspace_low_regret_assign_lane_output_tiled(
        scores_lane, lane_active, token_output_tile, token_local_slot, *,
        blocks_per_lane, output_tile_count, output_tile_capacity,
        regret_weight=1.0, spill_penalty=0.01):
    """Assign one lane into output-tile buckets with local-token slots."""
    del regret_weight, spill_penalty
    T = int(scores_lane.shape[0])
    tile_capacity_i32 = jnp.asarray(output_tile_capacity, dtype=jnp.int32)
    candidate_scores, candidate_blocks = jax.lax.top_k(
        scores_lane.astype(jnp.float32), blocks_per_lane)
    very_low = jnp.asarray(-1.0e9, dtype=jnp.float32)
    best_score = candidate_scores[:, 0]
    second_score = (
        candidate_scores[:, 1]
        if blocks_per_lane > 1 else jnp.full_like(best_score, very_low))
    primary_regret = best_score - second_score

    def candidate_pass(carry, pass_i):
        (assigned_block, assigned_slot, assigned_rank, chosen_score,
         unresolved, bucket_counts) = carry
        candidate = jax.lax.dynamic_index_in_dim(
            candidate_blocks, pass_i, axis=1, keepdims=False)
        curr_score = jax.lax.dynamic_index_in_dim(
            candidate_scores, pass_i, axis=1, keepdims=False)

        active = unresolved.astype(jnp.bool_)
        slot = token_local_slot
        accept = jnp.logical_and(
            active,
            jnp.logical_and(
                candidate < blocks_per_lane,
                jnp.logical_and(
                    token_output_tile < output_tile_count,
                    slot < tile_capacity_i32)))
        bucket_counts = bucket_counts.at[
            candidate, token_output_tile].add(
                accept.astype(jnp.int32), mode='drop')
        assigned_block = jnp.where(accept, candidate, assigned_block)
        assigned_slot = jnp.where(accept, slot, assigned_slot)
        assigned_rank = jnp.where(accept, pass_i, assigned_rank)
        chosen_score = jnp.where(accept, curr_score, chosen_score)
        unresolved = jnp.logical_and(unresolved, jnp.logical_not(accept))
        return (
            assigned_block, assigned_slot, assigned_rank, chosen_score,
            unresolved, bucket_counts), None

    init = (
        jnp.full((T,), -1, dtype=jnp.int32),
        jnp.zeros((T,), dtype=jnp.int32),
        jnp.full((T,), -1, dtype=jnp.int32),
        jnp.zeros((T,), dtype=jnp.float32),
        lane_active.astype(jnp.bool_),
        jnp.zeros((blocks_per_lane, output_tile_count), dtype=jnp.int32),
    )
    (assigned_block, assigned_slot, assigned_rank, chosen_score,
     unresolved, bucket_counts) = jax.lax.scan(
        candidate_pass, init,
        jnp.arange(blocks_per_lane, dtype=jnp.int32))[0]
    return (
        assigned_block, assigned_slot, assigned_rank, chosen_score,
        best_score, primary_regret, unresolved, bucket_counts,
        candidate_blocks[:, 0])


def _opspace_build_lane_buckets(
        scores_local, selected_lane_mask_local, *, blocks_per_lane,
        output_tile_size, output_tile_count, output_tile_capacity,
        regret_weight=1.0, spill_penalty=0.01):
    """Build output-tile-local fixed-shape buckets for all local lanes."""
    T = int(scores_local.shape[0])
    local_lanes = int(scores_local.shape[1])
    token_ids = jnp.arange(T, dtype=jnp.int32)
    token_output_tile = jnp.minimum(
        token_ids // jnp.asarray(output_tile_size, dtype=jnp.int32),
        jnp.asarray(output_tile_count - 1, dtype=jnp.int32))
    token_local_slot = (
        token_ids % jnp.asarray(output_tile_size, dtype=jnp.int32))

    def assign_one(scores_lane, lane_active):
        return _opspace_low_regret_assign_lane_output_tiled(
            scores_lane, lane_active, token_output_tile, token_local_slot,
            blocks_per_lane=blocks_per_lane,
            output_tile_count=output_tile_count,
            output_tile_capacity=output_tile_capacity,
            regret_weight=regret_weight, spill_penalty=spill_penalty)

    (assigned_block, assigned_slot, assigned_rank, chosen_score, best_score,
     primary_regret, unresolved, bucket_fill, primary_block) = jax.vmap(
        assign_one, in_axes=(1, 1), out_axes=0)(
            scores_local, selected_lane_mask_local)
    valid_assignment = assigned_block >= 0
    safe_block = jnp.where(valid_assignment, assigned_block, 0)
    safe_slot = jnp.where(valid_assignment, assigned_slot, 0)
    safe_tile = jnp.broadcast_to(
        token_output_tile[None, :], (local_lanes, T))
    lane_ids = jnp.arange(local_lanes, dtype=jnp.int32)[:, None]
    token_ids_l = jnp.broadcast_to(token_ids[None, :], (local_lanes, T))
    valid_f = valid_assignment.astype(jnp.float32)
    bucket_valid_count = jnp.zeros(
        (local_lanes, blocks_per_lane, output_tile_count,
         output_tile_capacity),
        dtype=jnp.float32).at[lane_ids, safe_block, safe_tile, safe_slot].add(
            valid_f, mode='drop')
    token_id_bucket = jnp.zeros(
        (local_lanes, blocks_per_lane, output_tile_count,
         output_tile_capacity),
        dtype=jnp.int32).at[lane_ids, safe_block, safe_tile, safe_slot].max(
            token_ids_l * valid_assignment.astype(jnp.int32), mode='drop')
    bucket_valid = bucket_valid_count == jnp.float32(1.0)
    bucket_fill = bucket_valid.astype(jnp.int32).sum(axis=-1)
    assignment_collision_count = jnp.maximum(
        bucket_valid_count - jnp.float32(1.0), jnp.float32(0.0)).sum()
    return {
        'assigned_block': assigned_block,
        'assigned_slot': assigned_slot,
        'assigned_rank': assigned_rank,
        'chosen_score': chosen_score,
        'best_score': best_score,
        'primary_regret': primary_regret,
        'unresolved': unresolved,
        'bucket_fill': bucket_fill,
        'token_id_bucket': token_id_bucket,
        'bucket_valid': bucket_valid,
        'primary_block': primary_block,
        'assignment_collision_count': assignment_collision_count,
    }


def _opspace_execute_output_tiled_pallas(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill, *,
        output_tile_size, d_tile_size, output_tile_count, d_tile_count,
        output_tile_capacity, interpret=False):
    """Pallas executor whose programs own one complete output tile."""
    if not _PALLAS_API_AVAILABLE:
        raise RuntimeError(_pallas_unavailable_message())
    del token_id_bucket, bucket_fill, output_tile_capacity
    local_lanes = int(bucket_valid.shape[0])
    blocks_per_lane = int(bucket_valid.shape[1])
    T = int(flat_x.shape[0])
    D = int(flat_x.shape[-1])
    d_route = int(flat_h.shape[-1])
    block_size = int(key_blocks.shape[2])
    op_tile_size = min(block_size, int(OPSPACE_FINAL_OP_TILE_SIZE_POLICY))
    if op_tile_size <= 0 or block_size % op_tile_size != 0:
        raise ValueError(
            "block_bucketed_pallas_final requires block_size to be divisible "
            f"by op_tile_size; got block_size={block_size}, "
            f"op_tile_size={op_tile_size}.")
    op_tile_count = block_size // op_tile_size
    padded_T = output_tile_count * output_tile_size
    padded_D = d_tile_count * d_tile_size
    pad_rows = padded_T - T
    flat_x_padded = jnp.pad(flat_x, ((0, pad_rows), (0, 0)))
    flat_h_padded = jnp.pad(flat_h, ((0, pad_rows), (0, 0)))

    def output_tile_kernel(
            flat_x_ref, flat_h_ref, key_blocks_ref, read_blocks_ref,
            write_blocks_ref, valid_blocks_ref, bucket_valid_ref,
            raw_tiles_ref, gate_mass_tiles_ref, relu_count_tiles_ref):
        output_tile_i = pl.program_id(0)
        output_start = output_tile_i * output_tile_size

        scratch_raw_tiles = [
            jnp.zeros((output_tile_size, d_tile_size), dtype=jnp.float32)
            for _ in range(d_tile_count)
        ]
        scratch_mass = jnp.zeros((output_tile_size,), dtype=jnp.float32)
        scratch_relu = jnp.zeros((output_tile_size,), dtype=jnp.float32)
        rows = jnp.arange(output_tile_size, dtype=jnp.int32)
        token_ids = output_start + rows
        tile_row_valid = token_ids < T
        tile_row_valid_f = tile_row_valid.astype(jnp.float32)
        h_rows_all = pl.load(
            flat_h_ref,
            (pl.dslice(output_start, output_tile_size),
             pl.dslice(0, d_route)),
            other=jnp.asarray(0.0, dtype=flat_h.dtype))
        h_rows_all = (
            h_rows_all.astype(jnp.float32)
            * tile_row_valid_f[:, None])

        for lane_i in range(local_lanes):
            for block_i in range(blocks_per_lane):
                lane_idx = jnp.asarray(lane_i, dtype=jnp.int32)
                block_idx = jnp.asarray(block_i, dtype=jnp.int32)
                valid_rows = pl.load(
                    bucket_valid_ref,
                    (lane_idx, block_idx, output_tile_i,
                     pl.dslice(0, output_tile_size)),
                    other=jnp.asarray(False, dtype=jnp.bool_))
                valid_rows = jnp.logical_and(valid_rows, tile_row_valid)
                valid_rows_f = valid_rows.astype(jnp.float32)
                h_rows = (
                    h_rows_all
                    * valid_rows_f[:, None])
                for op_tile_i in range(op_tile_count):
                    op_start = op_tile_i * op_tile_size
                    key_local = pl.load(
                        key_blocks_ref,
                        (lane_idx, block_idx,
                         pl.dslice(op_start, op_tile_size),
                         pl.dslice(0, d_route)),
                        other=jnp.asarray(0.0, dtype=key_blocks.dtype))
                    valid_ops = pl.load(
                        valid_blocks_ref,
                        (lane_idx, block_idx,
                         pl.dslice(op_start, op_tile_size)),
                        other=jnp.asarray(False, dtype=jnp.bool_))
                    rho = pl.dot(
                        h_rows.astype(jnp.bfloat16),
                        key_local.astype(jnp.bfloat16),
                        trans_b=True).astype(jnp.float32)
                    read_value = jnp.zeros(
                        (output_tile_size, op_tile_size), dtype=jnp.float32)
                    for read_tile_i in range(d_tile_count):
                        read_start = (
                            jnp.asarray(read_tile_i, dtype=jnp.int32)
                            * d_tile_size)
                        x_read_tile = pl.load(
                            flat_x_ref,
                            (pl.dslice(output_start, output_tile_size),
                             pl.dslice(read_start, d_tile_size)),
                            other=jnp.asarray(0.0, dtype=flat_x.dtype))
                        read_local = pl.load(
                            read_blocks_ref,
                            (lane_idx, block_idx,
                             pl.dslice(op_start, op_tile_size),
                             pl.dslice(read_start, d_tile_size)),
                            other=jnp.asarray(0.0, dtype=read_blocks.dtype))
                        x_read_tile = (
                            x_read_tile.astype(jnp.float32)
                            * valid_rows_f[:, None])
                        read_value = read_value + pl.dot(
                            x_read_tile.astype(jnp.bfloat16),
                            read_local.astype(jnp.bfloat16),
                            trans_b=True).astype(jnp.float32)
                    gate = jnp.square(jax.nn.relu(rho)).astype(jnp.float32)
                    gate = gate * valid_rows_f[:, None]
                    gate = gate * valid_ops.astype(jnp.float32)[None, :]
                    mixed = gate * read_value
                    scratch_mass = scratch_mass + gate.sum(axis=-1)
                    scratch_relu = scratch_relu + (
                        (gate > 0.0).astype(jnp.float32).sum(axis=-1))

                    for out_d_tile_i in range(d_tile_count):
                        out_d_start = (
                            jnp.asarray(out_d_tile_i, dtype=jnp.int32)
                            * d_tile_size)
                        write_local = pl.load(
                            write_blocks_ref,
                            (lane_idx, block_idx,
                             pl.dslice(op_start, op_tile_size),
                             pl.dslice(out_d_start, d_tile_size)),
                            other=jnp.asarray(0.0, dtype=write_blocks.dtype))
                        raw_rows = pl.dot(
                            mixed.astype(jnp.bfloat16),
                            write_local.astype(jnp.bfloat16)).astype(
                                jnp.float32)
                        scratch_raw_tiles[out_d_tile_i] = (
                            scratch_raw_tiles[out_d_tile_i] + raw_rows)

        for out_d_tile_i in range(d_tile_count):
            pl.store(
                raw_tiles_ref,
                (output_tile_i, pl.dslice(0, output_tile_size),
                 pl.dslice(out_d_tile_i * d_tile_size, d_tile_size)),
                scratch_raw_tiles[out_d_tile_i])
        pl.store(
            gate_mass_tiles_ref,
            (output_tile_i, pl.dslice(0, output_tile_size)),
            scratch_mass)
        pl.store(
            relu_count_tiles_ref,
            (output_tile_i, pl.dslice(0, output_tile_size)),
            scratch_relu)

    raw_tiles, gate_mass_tiles, relu_count_tiles = pl.pallas_call(
        output_tile_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (output_tile_count, output_tile_size, padded_D), jnp.float32),
            jax.ShapeDtypeStruct(
                (output_tile_count, output_tile_size), jnp.float32),
            jax.ShapeDtypeStruct(
                (output_tile_count, output_tile_size), jnp.float32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",)),
        grid=(output_tile_count,),
        interpret=bool(interpret),
        name='opspace_output_tiled_pallas_final')(
            flat_x_padded, flat_h_padded, key_blocks, read_blocks,
            write_blocks, valid_blocks, bucket_valid)

    flat_raw_out = raw_tiles.reshape(padded_T, padded_D)[:T, :D]
    flat_gate_mass = gate_mass_tiles.reshape(padded_T, 1)[:T]
    flat_relu_count = relu_count_tiles.reshape(padded_T)[:T]
    return flat_raw_out, flat_gate_mass, flat_relu_count


def _opspace_execute_output_tiled_pallas_trainable_impl(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill,
        output_tile_size, d_tile_size, output_tile_count, d_tile_count,
        output_tile_capacity, interpret):
    return _opspace_execute_output_tiled_pallas(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill,
        output_tile_size=output_tile_size,
        d_tile_size=d_tile_size,
        output_tile_count=output_tile_count,
        d_tile_count=d_tile_count,
        output_tile_capacity=output_tile_capacity,
        interpret=interpret)


_opspace_execute_output_tiled_pallas_trainable = jax.custom_vjp(
    _opspace_execute_output_tiled_pallas_trainable_impl,
    nondiff_argnums=(9, 10, 11, 12, 13, 14))


def _opspace_execute_output_tiled_pallas_trainable_fwd(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill,
        output_tile_size, d_tile_size, output_tile_count, d_tile_count,
        output_tile_capacity, interpret):
    out = _opspace_execute_output_tiled_pallas_trainable_impl(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill,
        output_tile_size, d_tile_size, output_tile_count, d_tile_count,
        output_tile_capacity, interpret)
    return out, (
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill)


def _opspace_execute_output_tiled_manual_bwd(
        flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
        token_id_bucket, bucket_valid, bucket_fill, cotangent, *,
        output_tile_size, d_tile_size, output_tile_count, d_tile_count,
        output_tile_capacity):
    """Manual output-tile VJP for the final Pallas executor."""
    del token_id_bucket, bucket_fill, output_tile_capacity
    d_raw_out, d_gate_mass, _d_relu_count = cotangent
    local_lanes = int(bucket_valid.shape[0])
    blocks_per_lane = int(bucket_valid.shape[1])
    T = int(flat_x.shape[0])
    D = int(flat_x.shape[-1])
    d_route = int(flat_h.shape[-1])
    block_size = int(key_blocks.shape[2])
    del d_route
    op_tile_size = min(block_size, int(OPSPACE_FINAL_OP_TILE_SIZE_POLICY))
    if op_tile_size <= 0 or block_size % op_tile_size != 0:
        raise ValueError(
            "block_bucketed_pallas_final manual VJP requires block_size to "
            "be divisible by op_tile_size; got "
            f"block_size={block_size}, op_tile_size={op_tile_size}.")
    op_tile_count = block_size // op_tile_size

    def tile_step(carry, output_tile_i):
        grad_x, grad_h, grad_key, grad_read, grad_write = carry
        token_offsets = (
            output_tile_i * output_tile_size
            + jnp.arange(output_tile_size, dtype=jnp.int32))
        token_valid = token_offsets < T
        safe_output_token = jnp.where(token_valid, token_offsets, 0)
        d_mass_tile = d_gate_mass[safe_output_token, 0]
        d_mass_tile = d_mass_tile * token_valid.astype(jnp.float32)

        for lane_i in range(local_lanes):
            for block_i in range(blocks_per_lane):
                valid_rows = jax.lax.dynamic_slice(
                    bucket_valid,
                    (lane_i, block_i, output_tile_i, 0),
                    (1, 1, 1, output_tile_size)).reshape(
                        output_tile_size)
                valid_rows = jnp.logical_and(valid_rows, token_valid)
                valid_rows_f = valid_rows.astype(jnp.float32)
                safe_token = jnp.where(valid_rows, token_offsets, 0)

                h_rows = flat_h[safe_token]
                h_rows = (
                    h_rows * valid_rows_f[:, None].astype(jnp.bfloat16))
                d_mass = d_mass_tile * valid_rows_f

                for op_tile_i in range(op_tile_count):
                    op_start = op_tile_i * op_tile_size
                    op_end = op_start + op_tile_size
                    key_tile = key_blocks[lane_i, block_i, op_start:op_end]
                    valid_ops = valid_blocks[
                        lane_i, block_i, op_start:op_end]
                    valid_ops_f = valid_ops.astype(jnp.float32)
                    rho = jnp.einsum(
                        'rd,nd->rn', h_rows, key_tile).astype(jnp.float32)
                    read_value = jnp.zeros(
                        (output_tile_size, op_tile_size), dtype=jnp.float32)
                    for read_tile_i in range(d_tile_count):
                        read_offsets = (
                            jnp.asarray(read_tile_i, dtype=jnp.int32)
                            * d_tile_size
                            + jnp.arange(d_tile_size, dtype=jnp.int32))
                        read_valid = read_offsets < D
                        safe_read_d = jnp.where(read_valid, read_offsets, 0)
                        x_read = flat_x[
                            safe_token[:, None], safe_read_d[None, :]]
                        x_read = x_read * (
                            valid_rows_f[:, None]
                            * read_valid[None, :]).astype(jnp.bfloat16)
                        read_block_tile = jnp.take(
                            read_blocks[lane_i, block_i, op_start:op_end],
                            safe_read_d, axis=1)
                        read_block_tile = read_block_tile * (
                            read_valid[None, :].astype(
                                read_block_tile.dtype))
                        read_value = read_value + jnp.einsum(
                            'rd,nd->rn', x_read,
                            read_block_tile).astype(jnp.float32)
                    gate = jnp.square(jax.nn.relu(rho)).astype(jnp.float32)
                    gate = gate * valid_rows_f[:, None]
                    gate = gate * valid_ops_f[None, :]
                    mixed = gate * read_value

                    d_mixed = jnp.zeros(
                        (output_tile_size, op_tile_size), dtype=jnp.float32)
                    for out_d_tile_i in range(d_tile_count):
                        d_offsets = (
                            jnp.asarray(out_d_tile_i, dtype=jnp.int32)
                            * d_tile_size
                            + jnp.arange(d_tile_size, dtype=jnp.int32))
                        d_valid = d_offsets < D
                        safe_d = jnp.where(d_valid, d_offsets, 0)
                        d_y_tile = d_raw_out[
                            safe_token[:, None], safe_d[None, :]]
                        d_y_tile = d_y_tile * (
                            valid_rows_f[:, None]
                            * d_valid[None, :]).astype(jnp.float32)
                        d_y_bf16 = d_y_tile.astype(jnp.bfloat16)
                        write_tile = jnp.take(
                            write_blocks[
                                lane_i, block_i, op_start:op_end],
                            safe_d, axis=1)
                        write_tile = write_tile * d_valid[None, :].astype(
                            write_tile.dtype)
                        d_mixed = d_mixed + jnp.einsum(
                            'rd,nd->rn',
                            d_y_bf16, write_tile).astype(jnp.float32)
                        d_write = jnp.einsum(
                            'rn,rd->nd',
                            mixed.astype(jnp.bfloat16),
                            d_y_bf16).astype(jnp.float32)
                        grad_write = grad_write.at[
                            lane_i, block_i, op_start:op_end, safe_d].add(
                                (d_write * d_valid[None, :]).T, mode='drop')

                    d_read_value = d_mixed * gate
                    d_gate = d_mixed * read_value
                    d_gate = d_gate + d_mass[:, None]
                    d_read_value_bf16 = d_read_value.astype(jnp.bfloat16)

                    relu_rho = jax.nn.relu(rho)
                    d_rho = d_gate * (jnp.float32(2.0) * relu_rho)
                    d_rho = d_rho * (rho > 0.0).astype(jnp.float32)
                    d_rho = d_rho * valid_rows_f[:, None]
                    d_rho = d_rho * valid_ops_f[None, :]
                    d_rho_bf16 = d_rho.astype(jnp.bfloat16)
                    d_h = jnp.einsum(
                        'rn,nd->rd', d_rho_bf16,
                        key_tile).astype(jnp.float32)
                    d_key = jnp.einsum(
                        'rn,rd->nd', d_rho_bf16,
                        h_rows).astype(jnp.float32)

                    for read_tile_i in range(d_tile_count):
                        read_offsets = (
                            jnp.asarray(read_tile_i, dtype=jnp.int32)
                            * d_tile_size
                            + jnp.arange(d_tile_size, dtype=jnp.int32))
                        read_valid = read_offsets < D
                        safe_read_d = jnp.where(read_valid, read_offsets, 0)
                        x_read = flat_x[
                            safe_token[:, None], safe_read_d[None, :]]
                        x_read = x_read * (
                            valid_rows_f[:, None]
                            * read_valid[None, :]).astype(jnp.bfloat16)
                        read_block_tile = jnp.take(
                            read_blocks[
                                lane_i, block_i, op_start:op_end],
                            safe_read_d, axis=1)
                        read_block_tile = read_block_tile * (
                            read_valid[None, :].astype(
                                read_block_tile.dtype))
                        d_x = jnp.einsum(
                            'rn,nd->rd', d_read_value_bf16,
                            read_block_tile).astype(jnp.float32)
                        d_read = jnp.einsum(
                            'rn,rd->nd', d_read_value_bf16,
                            x_read).astype(jnp.float32)
                        grad_x = grad_x.at[
                            safe_token[:, None], safe_read_d[None, :]].add(
                                (d_x * valid_rows_f[:, None]
                                 * read_valid[None, :]).astype(flat_x.dtype),
                                mode='drop')
                        grad_read = grad_read.at[
                            lane_i, block_i, op_start:op_end,
                            safe_read_d].add(
                                (d_read * read_valid[None, :]).T,
                                mode='drop')
                    grad_h = grad_h.at[safe_token].add(
                        (d_h * valid_rows_f[:, None]).astype(flat_h.dtype),
                        mode='drop')
                    grad_key = grad_key.at[
                        lane_i, block_i, op_start:op_end].add(d_key)
        return (grad_x, grad_h, grad_key, grad_read, grad_write), None

    init = (
        jnp.zeros_like(flat_x),
        jnp.zeros_like(flat_h),
        jnp.zeros_like(key_blocks, dtype=jnp.float32),
        jnp.zeros_like(read_blocks, dtype=jnp.float32),
        jnp.zeros_like(write_blocks, dtype=jnp.float32),
    )
    grad_x, grad_h, grad_key, grad_read, grad_write = jax.lax.scan(
        tile_step, init, jnp.arange(output_tile_count, dtype=jnp.int32))[0]
    return (
        grad_x.astype(flat_x.dtype),
        grad_h.astype(flat_h.dtype),
        grad_key.astype(key_blocks.dtype),
        grad_read.astype(read_blocks.dtype),
        grad_write.astype(write_blocks.dtype),
    )


def _opspace_execute_output_tiled_pallas_trainable_bwd(
        output_tile_size, d_tile_size, output_tile_count, d_tile_count,
        output_tile_capacity, interpret, residual, cotangent):
    del interpret
    (flat_x, flat_h, key_blocks, read_blocks, write_blocks, valid_blocks,
     token_id_bucket, bucket_valid, bucket_fill) = residual
    grad_x, grad_h, grad_key, grad_read, grad_write = (
        _opspace_execute_output_tiled_manual_bwd(
            flat_x, flat_h, key_blocks, read_blocks, write_blocks,
            valid_blocks, token_id_bucket, bucket_valid, bucket_fill,
            cotangent, output_tile_size=output_tile_size,
            d_tile_size=d_tile_size, output_tile_count=output_tile_count,
            d_tile_count=d_tile_count,
            output_tile_capacity=output_tile_capacity))
    return (
        grad_x, grad_h, grad_key, grad_read, grad_write,
        None, None, None, None)


_opspace_execute_output_tiled_pallas_trainable.defvjp(
    _opspace_execute_output_tiled_pallas_trainable_fwd,
    _opspace_execute_output_tiled_pallas_trainable_bwd)


def _opspace_execute_region_grouped_xla_dense(
        flat_x, flat_h, key_regions, read_regions, write_regions,
        valid_regions, token_id_bucket, bucket_valid, block_id_bucket,
        token_local_region_ids, token_region_slots, token_region_valid, *,
        blocks_per_region, operators_per_block,
        region_group_size=OPSPACE_RST_XLA_REGION_GROUP_SIZE,
        token_combine_chunk_size=OPSPACE_RST_XLA_TOKEN_COMBINE_CHUNK_SIZE):
    """Execute RST region/block buckets as XLA-friendly region groups."""
    T = int(flat_x.shape[0])
    D = int(flat_x.shape[-1])
    local_regions = int(token_id_bucket.shape[0])
    region_capacity = int(token_id_bucket.shape[1])
    max_local_visible_regions = int(token_local_region_ids.shape[1])
    group_size = max(1, int(region_group_size))
    token_chunk_size = max(1, int(token_combine_chunk_size))
    group_count = max(1, (local_regions + group_size - 1) // group_size)
    padded_T = max(
        token_chunk_size,
        ((T + token_chunk_size - 1) // token_chunk_size) * token_chunk_size)
    token_chunk_count = padded_T // token_chunk_size
    token_pad = padded_T - T
    region_ops = int(blocks_per_region) * int(operators_per_block)
    op_block_ids = (
        jnp.arange(region_ops, dtype=jnp.int32)
        // jnp.asarray(operators_per_block, dtype=jnp.int32))

    token_local_region_ids_p = jnp.pad(
        token_local_region_ids, ((0, token_pad), (0, 0)),
        constant_values=0)
    token_region_slots_p = jnp.pad(
        token_region_slots, ((0, token_pad), (0, 0)),
        constant_values=0)
    token_region_valid_p = jnp.pad(
        token_region_valid, ((0, token_pad), (0, 0)),
        constant_values=False)

    def group_step(carry, group_i):
        flat_raw_out, flat_gate_mass, flat_relu_gate_count, finite_acc = carry
        group_start = group_i * jnp.asarray(group_size, dtype=jnp.int32)
        group_offsets = jnp.arange(group_size, dtype=jnp.int32)
        region_ids = group_start + group_offsets
        group_valid = region_ids < jnp.asarray(local_regions, dtype=jnp.int32)
        safe_region_ids = jnp.where(group_valid, region_ids, 0)
        token_ids_g = token_id_bucket[safe_region_ids]
        valid_g = jnp.logical_and(
            bucket_valid[safe_region_ids], group_valid[:, None])
        block_ids_g = block_id_bucket[safe_region_ids]
        safe_token_ids_g = jnp.where(valid_g, token_ids_g, 0)
        valid_g_f32 = valid_g.astype(jnp.float32)
        valid_g_bf16 = valid_g_f32.astype(jnp.bfloat16)

        x_g = flat_x[safe_token_ids_g].astype(jnp.bfloat16)
        h_g = flat_h[safe_token_ids_g].astype(jnp.bfloat16)
        x_g = x_g * valid_g_bf16[..., None]
        h_g = h_g * valid_g_bf16[..., None]
        key_g = key_regions[safe_region_ids].reshape(
            group_size, region_ops, key_regions.shape[-1])
        read_g = read_regions[safe_region_ids].reshape(
            group_size, region_ops, D)
        write_g = write_regions[safe_region_ids].reshape(
            group_size, region_ops, D)
        valid_o = jnp.logical_and(
            valid_regions[safe_region_ids].reshape(group_size, region_ops),
            group_valid[:, None])

        rho = jnp.einsum(
            'gcd,god->gco', h_g, key_g.astype(jnp.bfloat16)).astype(
                jnp.float32)
        read_value = jnp.einsum(
            'gcd,god->gco', x_g, read_g.astype(jnp.bfloat16)).astype(
                jnp.float32)
        gate = jnp.square(jax.nn.relu(rho)).astype(jnp.float32)
        block_mask = jnp.any(
            op_block_ids[None, None, None, :] == block_ids_g[:, :, :, None],
            axis=2)
        gate = gate * block_mask.astype(jnp.float32)
        gate = gate * valid_o[:, None, :].astype(jnp.float32)
        gate = gate * valid_g_f32[:, :, None]
        raw_g = jnp.einsum(
            'gco,god->gcd',
            (gate * read_value).astype(jnp.bfloat16),
            write_g.astype(jnp.bfloat16)).astype(jnp.float32)
        mass_g = gate.sum(axis=-1, keepdims=True)
        relu_g = (gate > 0.0).astype(jnp.float32).sum(axis=-1)
        raw_g_store = raw_g.astype(jnp.bfloat16)

        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(gate)),
            jnp.all(jnp.isfinite(raw_g)))
        finite_acc = jnp.minimum(finite_acc, finite.astype(jnp.float32))

        def token_chunk_step(chunk_carry, chunk_i):
            flat_raw_out, flat_gate_mass, flat_relu_gate_count = chunk_carry
            chunk_start = chunk_i * jnp.asarray(
                token_chunk_size, dtype=jnp.int32)
            local_region_ids_c = jax.lax.dynamic_slice(
                token_local_region_ids_p, (chunk_start, 0),
                (token_chunk_size, max_local_visible_regions))
            slots_c = jax.lax.dynamic_slice(
                token_region_slots_p, (chunk_start, 0),
                (token_chunk_size, max_local_visible_regions))
            valid_c = jax.lax.dynamic_slice(
                token_region_valid_p, (chunk_start, 0),
                (token_chunk_size, max_local_visible_regions))
            in_group = jnp.logical_and(
                valid_c,
                jnp.logical_and(
                    local_region_ids_c >= group_start,
                    local_region_ids_c < (
                        group_start
                        + jnp.asarray(group_size, dtype=jnp.int32))))
            group_pos = jnp.clip(local_region_ids_c - group_start, 0,
                                 group_size - 1)
            safe_slots = jnp.clip(slots_c, 0, region_capacity - 1)
            raw_sel = raw_g_store[group_pos, safe_slots]
            mass_sel = mass_g[group_pos, safe_slots]
            relu_sel = relu_g[group_pos, safe_slots]
            mask = in_group.astype(jnp.float32)
            raw_chunk = (
                raw_sel.astype(jnp.float32) * mask[..., None]).sum(axis=1)
            mass_chunk = (mass_sel * mask[..., None]).sum(axis=1)
            relu_chunk = (relu_sel * mask).sum(axis=1)
            old_raw_chunk = jax.lax.dynamic_slice(
                flat_raw_out, (chunk_start, 0), (token_chunk_size, D))
            old_mass_chunk = jax.lax.dynamic_slice(
                flat_gate_mass, (chunk_start, 0), (token_chunk_size, 1))
            old_relu_chunk = jax.lax.dynamic_slice(
                flat_relu_gate_count, (chunk_start,), (token_chunk_size,))
            flat_raw_out = jax.lax.dynamic_update_slice(
                flat_raw_out, old_raw_chunk + raw_chunk, (chunk_start, 0))
            flat_gate_mass = jax.lax.dynamic_update_slice(
                flat_gate_mass, old_mass_chunk + mass_chunk, (chunk_start, 0))
            flat_relu_gate_count = jax.lax.dynamic_update_slice(
                flat_relu_gate_count, old_relu_chunk + relu_chunk,
                (chunk_start,))
            return (
                flat_raw_out, flat_gate_mass, flat_relu_gate_count), None

        (flat_raw_out, flat_gate_mass, flat_relu_gate_count) = jax.lax.scan(
            token_chunk_step,
            (flat_raw_out, flat_gate_mass, flat_relu_gate_count),
            jnp.arange(token_chunk_count, dtype=jnp.int32))[0]
        return (
            flat_raw_out, flat_gate_mass, flat_relu_gate_count,
            finite_acc), None

    init = (
        jnp.zeros((padded_T, D), dtype=jnp.float32),
        jnp.zeros((padded_T, 1), dtype=jnp.float32),
        jnp.zeros((padded_T,), dtype=jnp.float32),
        jnp.float32(1.0),
    )
    (flat_raw_out, flat_gate_mass, flat_relu_count,
     compute_no_nan) = jax.lax.scan(
        group_step, init,
        jnp.arange(group_count, dtype=jnp.int32))[0]
    return (
        flat_raw_out[:T],
        flat_gate_mass[:T],
        flat_relu_count[:T],
        compute_no_nan)


def _opspace_finalize_bucketed_output(
        flat_raw_out_local, flat_gate_mass_local,
        flat_relu_gate_count_local, admission_den_power):
    global_raw_out = jax.lax.psum(flat_raw_out_local, 'model')
    global_gate_mass = jax.lax.psum(flat_gate_mass_local, 'model')
    global_relu_gate_count = jax.lax.psum(
        flat_relu_gate_count_local, 'model')
    gate_denominator = jnp.power(
        jnp.maximum(global_gate_mass, 1.0), admission_den_power)
    flat_out = global_raw_out / gate_denominator
    return flat_out, global_gate_mass, global_relu_gate_count, gate_denominator


def _opspace_tau_free_relu_region_stationary_dense_core(
        flat_x, flat_h, op_key_local, read_local, write_local, *,
        lanes, tiles_per_lane, tile_size, local_lanes, local_tile_count,
        local_padded_ops, blocks_per_lane, exec_tiles_per_block, block_size,
        k_exec, exec_slots, admission_den_power, bucket_capacity_factor=1.5,
        bucket_chunk_size=128, high_regret_threshold=0.05,
        assignment_policy='low_regret',
        visible_blocks_per_region=1, region_score_pooling='smoothmax',
        region_score_temperature=0.25, region_capacity_factor=1.25,
        block_capacity_factor=None, use_pallas=True,
        pallas_interpret=False):
    """RST region-stationary masked-dense backend.

    Region-Block Operation Atlas

    RW operator keys define a spherical operation-space atlas. A token
    projects to an operation coordinate z. Regions are hard visibility and
    communication boundaries. Blocks are hard dense RW execution units inside
    regions. Region admission uses all regions as candidates and applies
    capacity-aware low-regret assignment, so hot regions are capped and excess
    requests are rerouted to the nearest available regions by score loss.
    After a request is admitted to a region, block assignment remains inside
    that region. Blocks do not mix softly. They only determine which RW
    operator bundle is visible. The only continuous mixing/intensity occurs at
    the RW operator gate inside each visible block.
    """
    del bucket_chunk_size, exec_tiles_per_block, use_pallas, pallas_interpret
    if str(assignment_policy).lower() != 'low_regret':
        raise ValueError(
            "region_stationary_dense requires assignment_policy="
            f"'low_regret', got {assignment_policy!r}.")
    if int(blocks_per_lane) <= 0:
        raise ValueError(
            "region_stationary_dense requires blocks_per_region > 0, got "
            f"{blocks_per_lane}.")
    if int(block_size) <= 0:
        raise ValueError(
            "region_stationary_dense requires operators_per_block > 0, got "
            f"{block_size}.")
    _visible_blocks_per_region = int(visible_blocks_per_region)
    if not (1 <= _visible_blocks_per_region <= int(blocks_per_lane)):
        raise ValueError(
            "region_stationary_dense visible_blocks_per_region must be in "
            f"[1, blocks_per_region={blocks_per_lane}], got "
            f"{visible_blocks_per_region}.")
    if not (1 <= int(k_exec) <= int(lanes)):
        raise ValueError(
            "region_stationary_dense visible_regions must be in "
            f"[1, num_regions={lanes}], got {k_exec}.")
    _region_score_pooling = str(region_score_pooling).strip().lower()
    if _region_score_pooling != 'smoothmax':
        raise ValueError(
            "region_stationary_dense requires region_score_pooling='smoothmax', "
            f"got {region_score_pooling!r}.")
    _region_score_temperature = max(
        float(region_score_temperature), 1.0e-6)
    _region_capacity_factor = float(region_capacity_factor)
    if _region_capacity_factor <= 0.0:
        raise ValueError(
            "region_stationary_dense region_capacity_factor must be > 0, got "
            f"{region_capacity_factor}.")
    _block_capacity_factor = (
        float(bucket_capacity_factor)
        if block_capacity_factor is None else float(block_capacity_factor))
    if _block_capacity_factor <= 0.0:
        raise ValueError(
            "region_stationary_dense block_capacity_factor must be > 0, got "
            f"{_block_capacity_factor}.")
    T = int(flat_x.shape[0])
    D = int(flat_x.shape[-1])
    N_local = int(op_key_local.shape[0])
    if N_local > local_padded_ops:
        raise ValueError(
            "operation-space region_stationary_dense local rows exceed "
            f"lane-block slots: rows={N_local}, slots={local_padded_ops}")
    expected_local_ops = int(local_lanes) * int(blocks_per_lane) * int(block_size)
    if int(local_padded_ops) != expected_local_ops:
        raise ValueError(
            "region_stationary_dense local_padded_ops invariant failed: "
            f"local_padded_ops={local_padded_ops}, expected={expected_local_ops}.")
    selected_region_requests = int(T) * int(k_exec)
    raw_region_capacity = max(
        1, int(math.ceil(
            (float(selected_region_requests) / float(lanes))
            * _region_capacity_factor)))
    region_capacity_round_multiple = max(
        1, int(OPSPACE_RST_REGION_CAPACITY_ROUND_MULTIPLE))
    region_capacity = max(
        raw_region_capacity,
        int(math.ceil(
            float(raw_region_capacity) / float(region_capacity_round_multiple))
            * region_capacity_round_multiple))
    if int(lanes) * int(region_capacity) < int(selected_region_requests):
        raise ValueError(
            "region_stationary_dense region capacity invariant failed: "
            f"num_regions={lanes}, region_capacity={region_capacity}, "
            f"selected_region_requests={selected_region_requests}.")
    block_capacity = _opspace_selected_block_bucket_capacity(
        T,
        num_regions=lanes,
        blocks_per_region=blocks_per_lane,
        visible_regions=k_exec,
        visible_blocks_per_region=_visible_blocks_per_region,
        block_capacity_factor=_block_capacity_factor)
    if (int(blocks_per_lane) * int(block_capacity)
            < int(region_capacity) * int(_visible_blocks_per_region)):
        raise ValueError(
            "selected-block sparse block capacity invariant failed: "
            f"blocks_per_region={blocks_per_lane}, "
            f"block_capacity={block_capacity}, "
            f"region_capacity={region_capacity}, "
            f"visible_blocks_per_region={_visible_blocks_per_region}.")
    bucket_capacity = block_capacity

    (slot_to_row_np, valid_slot_np, _valid_counts_np) = (
        _opspace_balanced_slot_layout_np(
            N_local, local_tile_count, tile_size))
    slot_to_row = jnp.asarray(slot_to_row_np, dtype=jnp.int32)
    valid_slot = jnp.asarray(valid_slot_np, dtype=jnp.bool_)
    model_axis_index = jax.lax.axis_index('model')

    op_key_padded = op_key_local[slot_to_row]
    read_padded = read_local[slot_to_row]
    write_padded = write_local[slot_to_row]
    op_key_padded = jnp.where(
        valid_slot[:, None], op_key_padded, jnp.zeros_like(op_key_padded))
    read_padded = jnp.where(
        valid_slot[:, None], read_padded, jnp.zeros_like(read_padded))
    write_padded = jnp.where(
        valid_slot[:, None], write_padded, jnp.zeros_like(write_padded))

    op_key_dir = _forward_unit_direction(
        op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    read_dir = _forward_unit_direction(
        read_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    write_dir = _forward_unit_direction(
        write_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)

    key_tiles = op_key_dir.reshape(
        local_lanes, tiles_per_lane, tile_size, op_key_dir.shape[-1])
    read_tiles = read_dir.reshape(
        local_lanes, tiles_per_lane, tile_size, read_dir.shape[-1])
    write_tiles = write_dir.reshape(
        local_lanes, tiles_per_lane, tile_size, write_dir.shape[-1])
    valid_tiles = valid_slot.reshape(local_lanes, tiles_per_lane, tile_size)

    key_blocks = key_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size,
        key_tiles.shape[-1]).reshape(
            local_lanes, blocks_per_lane, block_size, key_tiles.shape[-1])
    read_blocks = read_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size,
        read_tiles.shape[-1]).reshape(
            local_lanes, blocks_per_lane, block_size, read_tiles.shape[-1])
    write_blocks = write_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size,
        write_tiles.shape[-1]).reshape(
            local_lanes, blocks_per_lane, block_size, write_tiles.shape[-1])
    valid_blocks = valid_tiles.reshape(
        local_lanes, blocks_per_lane, -1, tile_size).reshape(
            local_lanes, blocks_per_lane, block_size)

    selection = _opspace_select_region_blocks(
        flat_h, key_blocks, valid_blocks,
        num_regions=lanes,
        local_regions=local_lanes,
        blocks_per_region=blocks_per_lane,
        operators_per_block=block_size,
        visible_regions=k_exec,
        visible_blocks_per_region=_visible_blocks_per_region,
        region_score_temperature=_region_score_temperature,
        capacity_policy={
            'region_capacity': region_capacity,
            'block_capacity': block_capacity,
            'high_regret_threshold': high_regret_threshold,
        })
    scores_all = selection['scores_all']
    scores_local = selection['scores_local']
    admission = selection['admission']
    admitted_region_mask_local = selection['selected_region_mask_local'].T
    assigned_rank = selection['assigned_rank']
    assigned_block = selection['selected_blocks_local']
    chosen_score = selection['chosen_score']
    best_score = selection['best_score']
    primary_regret = selection['primary_regret']
    unresolved = selection['unresolved']
    primary_block = selection['primary_block']

    block_buckets = _opspace_build_selected_block_buckets(
        assigned_block,
        blocks_per_region=blocks_per_lane,
        block_capacity=block_capacity)
    bucket_valid = block_buckets['bucket_valid']
    bucket_fill = block_buckets['bucket_fill']
    assignment_collision_count = block_buckets[
        'assignment_collision_count']

    (flat_out, global_gate_mass, global_relu_gate_count,
     gate_denominator, compute_no_nan) = (
        _opspace_execute_rst_selected_block_sparse(
            flat_x, flat_h, key_blocks, read_blocks, write_blocks,
            valid_blocks, block_buckets['token_id_bucket'], bucket_valid,
            admission_den_power=admission_den_power))

    def data_sum(v):
        return jax.lax.pmean(jax.lax.psum(v, 'data'), 'model')

    def metric_pmax(v, axis_name):
        return jax.lax.pmax(jax.lax.stop_gradient(v), axis_name)

    def metric_pmin(v, axis_name):
        return jax.lax.pmin(jax.lax.stop_gradient(v), axis_name)

    token_count = jnp.maximum(
        data_sum(jnp.asarray(T, dtype=jnp.float32)), 1.0)
    visible_blocks_per_region_f = jnp.float32(_visible_blocks_per_region)
    selected_region_request_count = token_count * jnp.float32(k_exec)
    selected_request_count = (
        selected_region_request_count * visible_blocks_per_region_f)
    valid_counts_by_block = valid_blocks.astype(jnp.float32).sum(axis=2)
    block_bucket_fill = bucket_fill.astype(jnp.float32)
    block_load_total = block_bucket_fill
    processed_count_local = block_buckets['processed_request_count']
    valid_exec_slots_sum_local = jnp.sum(
        block_bucket_fill * valid_counts_by_block)
    region_selected_local = admitted_region_mask_local.T
    selected_f = jnp.broadcast_to(
        region_selected_local[:, :, None], assigned_block.shape).astype(
            jnp.float32)
    assigned_valid = assigned_block >= 0
    assigned_valid_f = assigned_valid.astype(jnp.float32)
    block_slot_ids = jnp.arange(
        _visible_blocks_per_region, dtype=jnp.int32)[None, None, :]
    primary_accept_count_local = jnp.logical_and(
        assigned_valid, assigned_rank == block_slot_ids).astype(
            jnp.float32).sum()
    reroute_accept_count_local = jnp.logical_and(
        assigned_valid, assigned_rank > block_slot_ids).astype(
            jnp.float32).sum()
    block_unresolved_count_local = unresolved.astype(jnp.float32).sum()
    unprocessed_count_local = (
        block_unresolved_count_local
        + assignment_collision_count
        + block_buckets['overflow_request_count'])
    bucket_fill_sum_local = block_bucket_fill.sum()
    bucket_fill_max_local = block_bucket_fill.max()
    block_fill_max_local = block_load_total.max()

    global_processed_count = jax.lax.psum(processed_count_local, 'model')
    global_valid_exec_slots_sum = jax.lax.psum(
        valid_exec_slots_sum_local, 'model')
    global_primary_accept_count = jax.lax.psum(
        primary_accept_count_local, 'model')
    global_reroute_accept_count = jax.lax.psum(
        reroute_accept_count_local, 'model')
    global_unprocessed_count = jax.lax.psum(unprocessed_count_local, 'model')
    global_block_unresolved_count = jax.lax.psum(
        block_unresolved_count_local, 'model')
    global_assignment_collision_count = jax.lax.psum(
        assignment_collision_count, 'model')
    global_bucket_fill_sum = jax.lax.psum(bucket_fill_sum_local, 'model')
    global_bucket_fill_max = metric_pmax(bucket_fill_max_local, 'model')
    global_block_fill_max = metric_pmax(block_fill_max_local, 'model')

    accepted_region_request_count = data_sum(
        admission['accepted_request_count'])
    accepted_block_request_count = (
        accepted_region_request_count * visible_blocks_per_region_f)
    region_unresolved_count = data_sum(admission['unresolved_count'])
    region_reroute_count = data_sum(admission['reroute_count'])
    region_low_regret_spill_count = data_sum(
        admission['low_regret_spill_count'])
    region_high_regret_spill_count = data_sum(
        admission['high_regret_spill_count'])
    block_unresolved_count = data_sum(global_block_unresolved_count)
    block_reroute_count = data_sum(global_reroute_accept_count)
    processed_request_count = data_sum(global_processed_count)
    valid_exec_slots_mean = (
        data_sum(global_valid_exec_slots_sum) / token_count)
    relu_gate_count_mean = (
        data_sum(global_relu_gate_count.sum()) / token_count)
    gate_denominator_mean = (
        data_sum(gate_denominator.sum()) / token_count)
    gate_mass_mean = data_sum(global_gate_mass.sum()) / token_count
    primary_accept_frac = (
        data_sum(global_primary_accept_count)
        / jnp.maximum(selected_request_count, 1.0))
    reroute_frac = (
        ((region_reroute_count * visible_blocks_per_region_f)
         + block_reroute_count)
        / jnp.maximum(selected_request_count, 1.0))
    assignment_collision_count_global = data_sum(
        global_assignment_collision_count)
    unprocessed_count_global = data_sum(global_unprocessed_count)
    semantic_drop_count = (
        region_unresolved_count * visible_blocks_per_region_f
        + unprocessed_count_global)
    overflow_frac = semantic_drop_count / jnp.maximum(
        selected_request_count, 1.0)
    semantic_drop_frac = semantic_drop_count / jnp.maximum(
        selected_request_count, 1.0)
    bucket_count_global = jnp.float32(lanes * blocks_per_lane)
    bucket_fill_mean = (
        data_sum(global_bucket_fill_sum)
        / jnp.maximum(bucket_count_global, 1.0))
    bucket_fill_max = metric_pmax(
        metric_pmax(global_bucket_fill_max, 'data'), 'model')
    block_fill_max = metric_pmax(
        metric_pmax(global_block_fill_max, 'data'), 'model')
    bucket_fill_skew = bucket_fill_max / jnp.maximum(bucket_fill_mean, 1.0)

    region_load_max = metric_pmax(
        metric_pmax(jnp.max(admission['region_load']), 'data'), 'model')
    selected_region_top1_frac = region_load_max / jnp.maximum(
        accepted_region_request_count, 1.0)
    region_load_mean = jnp.maximum(
        accepted_region_request_count / jnp.maximum(jnp.float32(lanes), 1.0),
        jnp.float32(1.0e-6))
    region_load_p50 = region_load_mean
    region_load_p95 = region_load_max
    region_load_p99 = region_load_max
    region_load_p99_over_mean = (
        region_load_p99 / region_load_mean)
    region_capacity_util_p50 = (
        region_load_mean / jnp.maximum(jnp.float32(region_capacity), 1.0))
    region_capacity_util_p95 = (
        region_load_max / jnp.maximum(jnp.float32(region_capacity), 1.0))
    region_capacity_util_p99 = region_capacity_util_p95
    region_capacity_util_max = region_capacity_util_p95
    region_unresolved_frac = region_unresolved_count / jnp.maximum(
        selected_region_request_count, 1.0)
    region_reroute_frac = region_reroute_count / jnp.maximum(
        selected_region_request_count, 1.0)
    region_low_regret_spill_frac = (
        region_low_regret_spill_count
        / jnp.maximum(selected_region_request_count, 1.0))
    region_high_regret_spill_frac = (
        region_high_regret_spill_count
        / jnp.maximum(selected_region_request_count, 1.0))
    score_no_nan = jnp.all(jnp.isfinite(scores_all)).astype(jnp.float32)
    score_no_nan = metric_pmin(
        metric_pmin(score_no_nan, 'data'), 'model')
    compute_no_nan = metric_pmin(
        metric_pmin(compute_no_nan, 'data'), 'model')
    out_no_nan = jnp.all(jnp.isfinite(flat_out)).astype(jnp.float32)
    out_no_nan = metric_pmin(metric_pmin(out_no_nan, 'data'), 'model')
    no_nan = jnp.minimum(score_no_nan, jnp.minimum(compute_no_nan, out_no_nan))
    padded_exec_slots_mean = jnp.float32(exec_slots) - valid_exec_slots_mean
    all_requests_processed = jnp.logical_and(
        processed_request_count >= selected_request_count - 0.5,
        semantic_drop_count <= 0.0).astype(jnp.float32)
    region_all_requests_processed = (
        region_unresolved_count <= 0.0).astype(jnp.float32)
    block_all_requests_processed = jnp.logical_and(
        block_unresolved_count <= 0.0,
        jnp.logical_and(
            assignment_collision_count_global <= 0.0,
            unprocessed_count_global <= 0.0)).astype(jnp.float32)
    layout_ok = jnp.asarray(
        1.0 if local_padded_ops == (
            local_lanes * blocks_per_lane * block_size) else 0.0,
        dtype=jnp.float32)

    score_loss = jnp.where(
        assigned_valid, best_score - chosen_score, jnp.float32(0.0))
    high_regret_threshold_f = jnp.asarray(
        high_regret_threshold, dtype=jnp.float32)
    spill_mask = jnp.logical_and(assigned_valid, assigned_rank > 0)
    high_regret_spill = jnp.logical_and(
        spill_mask, score_loss > high_regret_threshold_f)
    low_regret_spill = jnp.logical_and(
        spill_mask, score_loss <= high_regret_threshold_f)
    selected_score_best_sum = (best_score * selected_f).sum()
    chosen_score_sum = (chosen_score * assigned_valid_f).sum()
    score_loss_sum = (score_loss * assigned_valid_f).sum()
    primary_regret_sum = (primary_regret * selected_f).sum()
    global_best_score_sum = jax.lax.psum(selected_score_best_sum, 'model')
    global_chosen_score_sum = jax.lax.psum(chosen_score_sum, 'model')
    global_score_loss_sum = jax.lax.psum(score_loss_sum, 'model')
    global_primary_regret_sum = jax.lax.psum(primary_regret_sum, 'model')
    global_assigned_den = jax.lax.psum(assigned_valid_f.sum(), 'model')
    block_assigned_den = data_sum(global_assigned_den)
    assignment_score_best_mean = (
        data_sum(global_best_score_sum)
        / jnp.maximum(accepted_block_request_count, 1.0))
    assignment_score_chosen_mean = (
        data_sum(global_chosen_score_sum)
        / jnp.maximum(block_assigned_den, 1.0))
    assignment_score_loss_mean = (
        data_sum(global_score_loss_sum)
        / jnp.maximum(block_assigned_den, 1.0))
    assignment_regret_mean = (
        data_sum(global_primary_regret_sum)
        / jnp.maximum(accepted_block_request_count, 1.0))
    region_score_best_mean = (
        data_sum(admission['best_score_sum'])
        / jnp.maximum(selected_region_request_count, 1.0))
    region_score_chosen_mean = (
        data_sum(admission['chosen_score_sum'])
        / jnp.maximum(accepted_region_request_count, 1.0))
    region_score_loss_mean = (
        data_sum(admission['score_loss_sum'])
        / jnp.maximum(accepted_region_request_count, 1.0))
    region_score_loss_p95 = metric_pmax(
        metric_pmax(admission['score_loss_p95'], 'model'), 'data')
    region_score_loss_max = metric_pmax(
        metric_pmax(admission['score_loss_max'], 'model'), 'data')
    assigned_mask = assigned_valid_f > 0.0
    score_loss_metric = jax.lax.stop_gradient(score_loss)
    primary_regret_metric = jax.lax.stop_gradient(primary_regret)
    local_score_loss_max = jnp.max(jnp.where(
        assigned_mask, score_loss_metric, jnp.float32(0.0)))
    local_regret_max = jnp.max(jnp.where(
        selected_f > 0.0, primary_regret_metric, jnp.float32(0.0)))
    assignment_score_loss_max = metric_pmax(
        metric_pmax(local_score_loss_max, 'model'), 'data')
    assignment_score_loss_p95 = assignment_score_loss_max
    assignment_regret_max = metric_pmax(
        metric_pmax(local_regret_max, 'model'), 'data')
    assignment_regret_p95 = assignment_regret_max
    low_regret_spill_frac = (
        data_sum(jax.lax.psum(
            low_regret_spill.astype(jnp.float32).sum(), 'model'))
        / jnp.maximum(accepted_block_request_count, 1.0))
    high_regret_spill_frac = (
        data_sum(jax.lax.psum(
            high_regret_spill.astype(jnp.float32).sum(), 'model'))
        / jnp.maximum(accepted_block_request_count, 1.0))

    bucket_fill_p50 = bucket_fill_mean
    bucket_fill_p95 = bucket_fill_max
    bucket_fill_p99 = bucket_fill_max
    bucket_capacity_util_p50 = (
        bucket_fill_mean / jnp.maximum(jnp.float32(bucket_capacity), 1.0))
    bucket_capacity_util_p95 = (
        bucket_fill_max / jnp.maximum(jnp.float32(bucket_capacity), 1.0))
    bucket_capacity_util_p99 = bucket_capacity_util_p95
    bucket_capacity_util_max = bucket_capacity_util_p95
    block_load_mean = jnp.maximum(
        processed_request_count / jnp.maximum(
            jnp.float32(lanes * blocks_per_lane), 1.0),
        jnp.float32(1.0e-6))
    block_load_p50 = block_load_mean
    block_load_p95 = block_fill_max
    block_load_p99 = block_fill_max
    block_load_max = block_fill_max
    block_capacity_util_p50 = (
        block_load_mean / jnp.maximum(jnp.float32(block_capacity), 1.0))
    block_capacity_util_p95 = (
        block_load_max / jnp.maximum(jnp.float32(block_capacity), 1.0))
    block_capacity_util_p99 = block_capacity_util_p95
    block_capacity_util_max = block_capacity_util_p95
    block_unresolved_frac = block_unresolved_count / jnp.maximum(
        accepted_block_request_count, 1.0)
    block_reroute_frac = block_reroute_count / jnp.maximum(
        accepted_block_request_count, 1.0)
    block_load_p99_over_mean = (
        block_load_p99 / block_load_mean)
    block_compactness_mean = jnp.float32(0.0)
    block_radius_mean = jnp.float32(0.0)
    block_radius_max = jnp.float32(0.0)

    pallas_logical_executed_chunks = jnp.float32(0.0)
    pallas_logical_empty_chunks = jnp.float32(0.0)
    pallas_actual_launched_chunks = jnp.float32(0.0)
    pallas_actual_skipped_chunks = jnp.float32(0.0)
    actual_vs_logical_chunk_ratio = jnp.float32(1.0)
    pallas_valid_token_frac = jnp.float32(0.0)
    pallas_padding_frac = jnp.float32(0.0)
    pallas_backend_available = jnp.float32(0.0)
    output_tile_capacity_metric = jnp.float32(0.0)
    output_tile_fill_p50 = jnp.float32(0.0)
    output_tile_fill_p95 = jnp.float32(0.0)
    output_tile_fill_p99 = jnp.float32(0.0)
    output_tile_fill_max = jnp.float32(0.0)
    output_tile_overflow_frac = jnp.float32(0.0)
    output_tile_padding_frac = jnp.float32(0.0)
    d_tile_size_metric = jnp.float32(0.0)
    output_tile_size_metric = jnp.float32(0.0)
    output_tiled_backend_active = jnp.float32(0.0)
    visible_blocks_per_token = jnp.float32(
        int(k_exec) * int(_visible_blocks_per_region))
    visible_ops_per_token = (
        visible_blocks_per_token * jnp.float32(block_size))
    physical_visible_ops_per_token = visible_ops_per_token
    compute_frac_vs_dense = (
        visible_ops_per_token
        / jnp.maximum(
            jnp.float32(lanes * blocks_per_lane * block_size), 1.0))
    logical_compute_frac_vs_dense = compute_frac_vs_dense
    physical_compute_frac_vs_dense = (
        physical_visible_ops_per_token
        / jnp.maximum(
            jnp.float32(lanes * blocks_per_lane * block_size), 1.0))

    diag = jnp.asarray((
        jnp.float32(1.0),
        jnp.float32(k_exec),
        visible_ops_per_token,
        valid_exec_slots_mean,
        jnp.float32(bucket_capacity),
        bucket_fill_mean,
        selected_region_top1_frac,
        primary_accept_frac,
        bucket_fill_skew,
        reroute_frac,
        relu_gate_count_mean,
        gate_denominator_mean,
        gate_mass_mean,
        relu_gate_count_mean,
        padded_exec_slots_mean,
        overflow_frac,
        no_nan,
        jnp.float32(1.0),
        selected_request_count,
        processed_request_count,
        all_requests_processed,
        layout_ok,
    ), dtype=jnp.float32)
    final_diag = jnp.asarray((
        assignment_score_best_mean,
        assignment_score_chosen_mean,
        assignment_score_loss_mean,
        assignment_score_loss_p95,
        assignment_score_loss_max,
        assignment_regret_mean,
        assignment_regret_p95,
        assignment_regret_max,
        low_regret_spill_frac,
        high_regret_spill_frac,
        semantic_drop_count,
        assignment_collision_count_global,
        jnp.float32(0.0),
        semantic_drop_frac,
        bucket_fill_p50,
        bucket_fill_p95,
        bucket_fill_p99,
        bucket_fill_max,
        bucket_capacity_util_p50,
        bucket_capacity_util_p95,
        bucket_capacity_util_p99,
        bucket_capacity_util_max,
        pallas_logical_executed_chunks,
        pallas_padding_frac,
        pallas_valid_token_frac,
        pallas_backend_available,
        pallas_logical_empty_chunks,
        pallas_actual_launched_chunks,
        pallas_actual_skipped_chunks,
        actual_vs_logical_chunk_ratio,
        output_tile_capacity_metric,
        output_tile_fill_p50,
        output_tile_fill_p95,
        output_tile_fill_p99,
        output_tile_fill_max,
        output_tile_overflow_frac,
        output_tile_padding_frac,
        d_tile_size_metric,
        output_tile_size_metric,
        output_tiled_backend_active,
        region_load_p99_over_mean,
        block_load_p99_over_mean,
        block_compactness_mean,
        block_radius_mean,
        block_radius_max,
        jnp.float32(lanes),
        jnp.float32(local_lanes),
        jnp.float32(blocks_per_lane),
        jnp.float32(block_size),
        jnp.float32(k_exec),
        jnp.float32(_visible_blocks_per_region),
        jnp.float32(region_capacity),
        jnp.float32(_region_capacity_factor),
        region_load_mean,
        region_load_p50,
        region_load_p95,
        region_load_p99,
        region_load_max,
        region_capacity_util_p50,
        region_capacity_util_p95,
        region_capacity_util_p99,
        region_capacity_util_max,
        region_unresolved_count,
        region_unresolved_frac,
        region_reroute_frac,
        region_low_regret_spill_frac,
        region_high_regret_spill_frac,
        region_score_best_mean,
        region_score_chosen_mean,
        region_score_loss_mean,
        region_score_loss_p95,
        region_score_loss_max,
        jnp.float32(block_capacity),
        jnp.float32(_block_capacity_factor),
        block_load_mean,
        block_load_p50,
        block_load_p95,
        block_load_p99,
        block_load_max,
        block_capacity_util_p50,
        block_capacity_util_p95,
        block_capacity_util_p99,
        block_capacity_util_max,
        block_unresolved_count,
        block_unresolved_frac,
        block_reroute_frac,
        low_regret_spill_frac,
        high_regret_spill_frac,
        assignment_score_best_mean,
        assignment_score_chosen_mean,
        assignment_score_loss_mean,
        assignment_score_loss_p95,
        assignment_score_loss_max,
        visible_blocks_per_token,
        visible_ops_per_token,
        physical_visible_ops_per_token,
        compute_frac_vs_dense,
        logical_compute_frac_vs_dense,
        physical_compute_frac_vs_dense,
        region_all_requests_processed,
        block_all_requests_processed,
        selected_request_count,
        processed_request_count,
        all_requests_processed,
        no_nan,
    ), dtype=jnp.float32)
    aux = {
        'aux_loss': jnp.float32(0.0),
        'final_diag': final_diag,
    }
    del primary_block
    return flat_out.astype(jnp.float32), global_gate_mass, diag, aux


def make_sharded_srw_tau_free_relu_block_bucketed_dense_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=32,
        tiles_per_lane=8,
        padded_ops=0,
        tile_size=128,
        k_exec=32,
        exec_tiles_per_block=2,
        blocks_per_lane=None,
        block_size=None,
        token_chunk_size=256):
    """Create tau-free relu2 regional block-bucketed dense SRW."""
    del max_chunk_size, dead_exposure_target, padded_ops, token_chunk_size
    del soft_gate_effective_active_eps, admission_den_grad_scale
    layout = _opspace_block_bucketed_layout(
        mesh, lanes, tiles_per_lane, tile_size, exec_tiles_per_block,
        blocks_per_lane, block_size)
    _lanes = layout['lanes']
    _tiles_per_lane = layout['tiles_per_lane']
    _tile_size = layout['tile_size']
    _local_lanes = layout['local_lanes']
    _local_tile_count = layout['local_tile_count']
    _local_padded_ops = layout['local_padded_ops']
    _exec_tiles_per_block = layout['exec_tiles_per_block']
    _blocks_per_lane = layout['blocks_per_lane']
    _block_size = layout['block_size']
    _k_exec = max(1, min(int(k_exec), _lanes))
    _exec_slots = _k_exec * _block_size
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None), P(), P()),
             check_rep=False)
    def fused_gate_srw_tau_free_relu_block_bucketed_dense(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens):
        del raw_tau, soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power, soft_gate_boundary_power_final
        del execution_prune_eps, training_tokens
        B, S, D = x.shape
        flat_out, _flat_gate_mass, diag = (
            _opspace_tau_free_relu_block_bucketed_dense_core(
                x.reshape(int(B) * int(S), D),
                h.reshape(int(B) * int(S), h.shape[-1]),
                op_key_local, read_local, write_local,
                lanes=_lanes,
                tiles_per_lane=_tiles_per_lane,
                tile_size=_tile_size,
                local_lanes=_local_lanes,
                local_tile_count=_local_tile_count,
                local_padded_ops=_local_padded_ops,
                blocks_per_lane=_blocks_per_lane,
                exec_tiles_per_block=_exec_tiles_per_block,
                block_size=_block_size,
                k_exec=_k_exec,
                exec_slots=_exec_slots,
                admission_den_power=_admission_den_power))
        return flat_out.reshape(B, S, D), diag, jnp.float32(0.0)

    fused_gate_srw_tau_free_relu_block_bucketed_dense._v4168_accepts_training_tokens = True
    return fused_gate_srw_tau_free_relu_block_bucketed_dense


def make_sharded_srw_tau_free_relu_block_bucketed_pallas_final_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=32,
        tiles_per_lane=8,
        padded_ops=0,
        tile_size=128,
        k_exec=32,
        exec_tiles_per_block=2,
        blocks_per_lane=None,
        block_size=None,
        token_chunk_size=256,
        bucket_capacity_factor=1.5,
        bucket_chunk_size=128,
        assignment_policy='low_regret',
        high_regret_threshold=0.05,
        visible_blocks_per_region=1,
        region_score_pooling='smoothmax',
        region_score_temperature=0.25,
        region_capacity_factor=1.25,
        block_capacity_factor=None,
        use_pallas=True,
        pallas_interpret=False):
    """Create tau-free relu2 region-stationary masked-dense RST SRW.

    The public name is retained for older call sites; it no longer constructs
    an output-tiled Pallas executor.
    """
    del max_chunk_size, dead_exposure_target, padded_ops, token_chunk_size
    del soft_gate_effective_active_eps, admission_den_grad_scale
    del use_pallas, pallas_interpret
    layout = _opspace_block_bucketed_layout(
        mesh, lanes, tiles_per_lane, tile_size, exec_tiles_per_block,
        blocks_per_lane, block_size)
    _lanes = layout['lanes']
    _tiles_per_lane = layout['tiles_per_lane']
    _tile_size = layout['tile_size']
    _local_lanes = layout['local_lanes']
    _local_tile_count = layout['local_tile_count']
    _local_padded_ops = layout['local_padded_ops']
    _exec_tiles_per_block = layout['exec_tiles_per_block']
    _blocks_per_lane = layout['blocks_per_lane']
    _block_size = layout['block_size']
    _k_exec = max(1, min(int(k_exec), _lanes))
    _visible_blocks_per_region = int(visible_blocks_per_region)
    _exec_slots = _k_exec * _visible_blocks_per_region * _block_size
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _bucket_capacity_factor = float(bucket_capacity_factor)
    _bucket_chunk_size = int(bucket_chunk_size)
    _assignment_policy = str(assignment_policy).lower()
    _high_regret_threshold = float(high_regret_threshold)
    _region_score_pooling = str(region_score_pooling).lower()
    _region_score_temperature = float(region_score_temperature)
    _region_capacity_factor = float(region_capacity_factor)
    _block_capacity_factor = (
        None if block_capacity_factor is None else float(block_capacity_factor))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None), P(),
                        {'aux_loss': P(), 'final_diag': P()}),
             check_rep=False)
    def fused_gate_srw_tau_free_relu_region_stationary_dense(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens):
        del raw_tau, soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power, soft_gate_boundary_power_final
        del execution_prune_eps, training_tokens
        B, S, D = x.shape
        flat_out, _flat_gate_mass, diag, aux = (
            _opspace_tau_free_relu_region_stationary_dense_core(
                x.reshape(int(B) * int(S), D),
                h.reshape(int(B) * int(S), h.shape[-1]),
                op_key_local, read_local, write_local,
                lanes=_lanes,
                tiles_per_lane=_tiles_per_lane,
                tile_size=_tile_size,
                local_lanes=_local_lanes,
                local_tile_count=_local_tile_count,
                local_padded_ops=_local_padded_ops,
                blocks_per_lane=_blocks_per_lane,
                exec_tiles_per_block=_exec_tiles_per_block,
                block_size=_block_size,
                k_exec=_k_exec,
                exec_slots=_exec_slots,
                admission_den_power=_admission_den_power,
                bucket_capacity_factor=_bucket_capacity_factor,
                bucket_chunk_size=_bucket_chunk_size,
                high_regret_threshold=_high_regret_threshold,
                assignment_policy=_assignment_policy,
                visible_blocks_per_region=_visible_blocks_per_region,
                region_score_pooling=_region_score_pooling,
                region_score_temperature=_region_score_temperature,
                region_capacity_factor=_region_capacity_factor,
                block_capacity_factor=_block_capacity_factor))
        return flat_out.reshape(B, S, D), diag, aux

    fused_gate_srw_tau_free_relu_region_stationary_dense._v4168_accepts_training_tokens = True
    return fused_gate_srw_tau_free_relu_region_stationary_dense


def make_sharded_srw_tau_free_relu_region_block_sparse_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        num_regions=32,
        blocks_per_region=2,
        operators_per_block=512,
        visible_regions=4,
        visible_blocks_per_region=1,
        region_score_pooling='smoothmax',
        region_score_temperature=0.25,
        tile_size=128,
        padded_ops=0,
        token_chunk_size=256,
        bucket_capacity_factor=1.5,
        bucket_chunk_size=128,
        region_capacity_factor=1.25,
        block_capacity_factor=1.25,
        assignment_policy='low_regret',
        high_regret_threshold=0.05,
        use_pallas=True,
        pallas_interpret=False):
    """Create the RST selected-block sparse Region/Block atlas executor."""
    del use_pallas, pallas_interpret
    if int(operators_per_block) % int(tile_size) != 0:
        raise ValueError(
            "selected-block sparse RST requires operators_per_block divisible by "
            f"tile_size; got operators_per_block={operators_per_block}, "
            f"tile_size={tile_size}.")
    _visible_regions = int(visible_regions)
    _exec_tiles_per_block = int(operators_per_block) // int(tile_size)
    _tiles_per_region = int(blocks_per_region) * _exec_tiles_per_block
    return make_sharded_srw_tau_free_relu_block_bucketed_pallas_final_minimal(
        mesh, max_chunk_size=max_chunk_size,
        dead_exposure_target=dead_exposure_target,
        soft_gate_effective_active_eps=soft_gate_effective_active_eps,
        admission_den_power=admission_den_power,
        admission_den_grad_scale=admission_den_grad_scale,
        lanes=int(num_regions),
        tiles_per_lane=_tiles_per_region,
        padded_ops=padded_ops,
        tile_size=int(tile_size),
        k_exec=_visible_regions,
        exec_tiles_per_block=_exec_tiles_per_block,
        blocks_per_lane=int(blocks_per_region),
        block_size=int(operators_per_block),
        token_chunk_size=token_chunk_size,
        bucket_capacity_factor=bucket_capacity_factor,
        bucket_chunk_size=bucket_chunk_size,
        region_capacity_factor=region_capacity_factor,
        block_capacity_factor=block_capacity_factor,
        assignment_policy=assignment_policy,
        high_regret_threshold=high_regret_threshold,
        visible_blocks_per_region=visible_blocks_per_region,
        region_score_pooling=region_score_pooling,
        region_score_temperature=region_score_temperature,
        use_pallas=False,
        pallas_interpret=False)


def make_sharded_srw_tau_free_relu_region_block_pallas_minimal(*args,
                                                               **kwargs):
    return make_sharded_srw_tau_free_relu_region_block_sparse_minimal(
        *args, **kwargs)


def make_sharded_srw_paired_tau_free_relu_block_bucketed_dense_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=8,
        tiles_per_lane=4,
        padded_ops=0,
        tile_size=128,
        k_exec=4,
        exec_tiles_per_block=1,
        blocks_per_lane=None,
        block_size=None,
        token_chunk_size=256):
    """Create tau-free relu2 paired Q/K block-bucketed dense SRW."""
    del max_chunk_size, dead_exposure_target, padded_ops, token_chunk_size
    del soft_gate_effective_active_eps, admission_den_grad_scale
    layout = _opspace_block_bucketed_layout(
        mesh, lanes, tiles_per_lane, tile_size, exec_tiles_per_block,
        blocks_per_lane, block_size)
    _lanes = layout['lanes']
    _tiles_per_lane = layout['tiles_per_lane']
    _tile_size = layout['tile_size']
    _local_lanes = layout['local_lanes']
    _local_tile_count = layout['local_tile_count']
    _local_padded_ops = layout['local_padded_ops']
    _exec_tiles_per_block = layout['exec_tiles_per_block']
    _blocks_per_lane = layout['blocks_per_lane']
    _block_size = layout['block_size']
    _k_exec = max(1, min(int(k_exec), _lanes))
    _exec_slots = _k_exec * _block_size
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None, None),
                       P('model', None),
                       P('data', None, None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P(), P()),
             out_specs=P('data', None, None, None),
             check_rep=False)
    def fused_gate_srw_paired_tau_free_relu_block_bucketed_dense(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens):
        del raw_tau, soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power, soft_gate_boundary_power_final
        del execution_prune_eps, training_tokens
        B, S, R, d_route = h.shape
        D = x.shape[-1]
        if int(R) != 2:
            raise ValueError(
                "operation-space paired QK block_bucketed_dense expects "
                f"route dimension 2, got {int(R)}.")
        x_route = jnp.broadcast_to(x[:, :, None, :], (B, S, R, D))
        flat_out, _flat_gate_mass, _diag = (
            _opspace_tau_free_relu_block_bucketed_dense_core(
                x_route.reshape(int(B) * int(S) * int(R), D),
                h.reshape(int(B) * int(S) * int(R), d_route),
                op_key_local, read_local, write_local,
                lanes=_lanes,
                tiles_per_lane=_tiles_per_lane,
                tile_size=_tile_size,
                local_lanes=_local_lanes,
                local_tile_count=_local_tile_count,
                local_padded_ops=_local_padded_ops,
                blocks_per_lane=_blocks_per_lane,
                exec_tiles_per_block=_exec_tiles_per_block,
                block_size=_block_size,
                k_exec=_k_exec,
                exec_slots=_exec_slots,
                admission_den_power=_admission_den_power))
        return flat_out.reshape(B, S, R, D)

    fused_gate_srw_paired_tau_free_relu_block_bucketed_dense._v4168_accepts_training_tokens = True
    return fused_gate_srw_paired_tau_free_relu_block_bucketed_dense


_HARDWARE_REPACK_POOLS = (
    ('attn_qk', 'attn_qk_op_key', 'attn_qk_read', 'attn_qk_write',
     'qk_block_size', 'qk_top_blocks'),
    ('attn_v', 'attn_v_op_key', 'attn_v_read', 'attn_v_write',
     'v_block_size', 'v_top_blocks'),
    ('rst', 'rst_op_key', 'rst_read', 'rst_write',
     'rst_block_size', 'rst_top_blocks'),
)
_HARDWARE_VQ_REPACK_POOLS = tuple(
    entry for entry in _HARDWARE_REPACK_POOLS if entry[0] != 'attn_qk')
_V4168_VQ_REPACK_ITERATIONS = 4
_V4168_VQ_REPACK_MAX_MOVE_FRAC = 0.08
_V4168_VQ_REPACK_STRATEGY = 'balanced_vq'


def _np_unit_direction(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (
        np.linalg.norm(x, axis=-1, keepdims=True) + RW_FORWARD_NORM_EPS)


def _validate_region_block_permutation_np(perm, n_valid, total_capacity,
                                          pool=''):
    perm = np.asarray(perm, dtype=np.int32)
    n_valid = int(n_valid)
    total_capacity = int(total_capacity)
    if int(perm.size) != total_capacity:
        raise RuntimeError(
            f"region/block atlas {pool} permutation length {perm.size} "
            f"!= total_capacity {total_capacity}")
    valid_perm = perm[perm >= 0]
    if int(valid_perm.size) != n_valid:
        raise RuntimeError(
            f"region/block atlas {pool} valid row count {valid_perm.size} "
            f"!= expected {n_valid}")
    if not np.array_equal(
            np.sort(valid_perm), np.arange(n_valid, dtype=np.int32)):
        raise RuntimeError(
            f"region/block atlas {pool} planner produced duplicate or "
            "missing valid row ids")
    invalid = perm[perm < 0]
    if invalid.size and not np.all(invalid == -1):
        raise RuntimeError(
            f"region/block atlas {pool} padded slots must be -1")
    return perm


def _plan_region_block_atlas_np(op_keys, *, num_regions,
                                blocks_per_region, operators_per_block,
                                owner_count=1, pool=''):
    keys = _np_unit_direction(op_keys)
    n_valid = int(keys.shape[0])
    d_route = int(keys.shape[-1]) if keys.ndim == 2 else 0
    num_regions = max(1, int(num_regions))
    blocks_per_region = max(1, int(blocks_per_region))
    operators_per_block = max(1, int(operators_per_block))
    owner_count = max(1, int(owner_count))
    block_count = num_regions * blocks_per_region
    total_capacity = block_count * operators_per_block
    if n_valid > total_capacity:
        raise RuntimeError(
            f"region/block atlas {pool} capacity {total_capacity} "
            f"is smaller than valid rows {n_valid}")
    perm = np.full((total_capacity,), -1, dtype=np.int32)
    perm[:n_valid] = np.arange(n_valid, dtype=np.int32)
    _validate_region_block_permutation_np(
        perm, n_valid, total_capacity, pool=pool)

    block_center = np.zeros((block_count, d_route), dtype=np.float32)
    block_compactness = np.zeros((block_count,), dtype=np.float32)
    block_radius_mean = np.zeros((block_count,), dtype=np.float32)
    block_radius_max = np.zeros((block_count,), dtype=np.float32)
    valid_count_per_block = np.zeros((block_count,), dtype=np.int32)
    for block_i in range(block_count):
        start = block_i * operators_per_block
        stop = start + operators_per_block
        valid_rows = perm[start:stop]
        valid_rows = valid_rows[valid_rows >= 0]
        valid_count_per_block[block_i] = int(valid_rows.size)
        if valid_rows.size == 0:
            continue
        block_keys = keys[valid_rows]
        center = _np_unit_direction(block_keys.mean(axis=0, keepdims=True))[0]
        block_center[block_i] = center
        cos = block_keys @ center
        dist = 1.0 - cos
        block_compactness[block_i] = float(np.mean(cos))
        block_radius_mean[block_i] = float(np.mean(dist))
        block_radius_max[block_i] = float(np.max(dist))

    region_id_per_block = (
        np.arange(block_count, dtype=np.int32) // blocks_per_region)
    owner_id_per_region = (
        np.arange(num_regions, dtype=np.int32) % owner_count)
    owner_id_per_block = owner_id_per_region[region_id_per_block]
    valid_blocks = valid_count_per_block > 0
    diagnostics = {
        'valid_rows': float(n_valid),
        'total_capacity': float(total_capacity),
        'invalid_padded_slots': float(total_capacity - n_valid),
        'block_compactness_mean': float(
            np.mean(block_compactness[valid_blocks])
            if np.any(valid_blocks) else 0.0),
        'block_radius_mean': float(
            np.mean(block_radius_mean[valid_blocks])
            if np.any(valid_blocks) else 0.0),
        'block_radius_max': float(
            np.max(block_radius_max[valid_blocks])
            if np.any(valid_blocks) else 0.0),
        'perm_checksum': float(_hardware_perm_checksum(
            np.where(perm >= 0, perm, 0))),
    }
    return {
        'perm': perm,
        'region_id_per_block': region_id_per_block,
        'owner_id_per_region': owner_id_per_region,
        'owner_id_per_block': owner_id_per_block,
        'block_center': block_center,
        'valid_count_per_block': valid_count_per_block,
        'diagnostics': diagnostics,
    }


def _apply_region_block_permutation_to_params_and_opt_state(
        params, opt_state, pool_name, atlas):
    perm = np.asarray(atlas.get('perm', ()), dtype=np.int32)
    valid_perm = perm[perm >= 0]
    _validate_region_block_permutation_np(
        perm, int(valid_perm.size), int(perm.size), pool=pool_name)
    return _apply_pool_permutations_to_params_and_opt_state(
        params, opt_state, {pool_name: valid_perm.astype(np.int32)})


def _sector_sums_np(keys, block_size):
    keys = _np_unit_direction(keys)
    n_ops = int(keys.shape[0])
    if n_ops == 0:
        return (
            np.zeros((0, keys.shape[-1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    n_sectors = (n_ops + int(block_size) - 1) // int(block_size)
    sums = np.zeros((n_sectors, keys.shape[-1]), dtype=np.float32)
    counts = np.zeros((n_sectors,), dtype=np.int32)
    for sector in range(n_sectors):
        start = sector * int(block_size)
        stop = min(start + int(block_size), n_ops)
        if start >= stop:
            continue
        sums[sector] = keys[start:stop].sum(axis=0)
        counts[sector] = stop - start
    return sums, counts


def _sector_compactness_and_radius_np(keys, block_size):
    keys = _np_unit_direction(keys)
    sums, counts = _sector_sums_np(keys, block_size)
    compactness = np.linalg.norm(sums, axis=-1).astype(np.float32)
    centers = _np_unit_direction(sums)
    n_ops = int(keys.shape[0])
    n_sectors = int(sums.shape[0])
    radius_mean = np.zeros((n_sectors,), dtype=np.float32)
    radius_max = np.zeros((n_sectors,), dtype=np.float32)
    for sector in range(n_sectors):
        start = sector * int(block_size)
        stop = min(start + int(block_size), n_ops)
        if start >= stop:
            continue
        dist = 1.0 - keys[start:stop] @ centers[sector]
        radius_mean[sector] = np.mean(dist).astype(np.float32)
        radius_max[sector] = np.max(dist).astype(np.float32)
    valid = counts > 0
    if not np.any(valid):
        return compactness, np.float32(0.0), np.float32(0.0)
    return (
        compactness,
        np.mean(radius_mean[valid]).astype(np.float32),
        np.max(radius_max[valid]).astype(np.float32),
    )


def _hardware_perm_checksum(perm):
    perm64 = np.asarray(perm, dtype=np.int64)
    weights = np.arange(1, int(perm64.size) + 1, dtype=np.int64)
    return np.int64(np.sum(weights * (perm64 + np.int64(1))))


def _hardware_perm_checksum_meta(checksum):
    """Signed int32 checksum value stable through TPU multihost allgather."""
    value = int(checksum) & 0xffffffff
    if value >= 0x80000000:
        value -= 0x100000000
    return int(value)


def _validate_hardware_permutation_np(perm, n_ops=None, pool=''):
    perm = np.asarray(perm, dtype=np.int32)
    expected_n = int(perm.size if n_ops is None else n_ops)
    if int(perm.size) != expected_n:
        raise RuntimeError(
            f"hardware repack {pool} permutation length {perm.size} "
            f"!= expected {expected_n}")
    if not np.array_equal(np.sort(perm), np.arange(expected_n, dtype=np.int32)):
        raise RuntimeError(
            f"hardware repack {pool} planner produced duplicate, missing, "
            "or padded row ids")
    return perm


def _identity_hardware_repack_metrics(perm, compact_before,
                                      radius_before_mean,
                                      radius_before_max,
                                      candidate_count=0.0):
    compact_mean = (
        float(np.mean(compact_before))
        if np.asarray(compact_before).size else 0.0)
    return {
        'moved_count': 0.0,
        'moved_frac': 0.0,
        'candidate_count': float(candidate_count),
        'mean_gain': 0.0,
        'max_gain': 0.0,
        'compactness_before_mean': compact_mean,
        'compactness_after_mean': compact_mean,
        'radius_before_mean': float(radius_before_mean),
        'radius_after_mean': float(radius_before_mean),
        'radius_before_max': float(radius_before_max),
        'radius_after_max': float(radius_before_max),
        'perm_checksum': float(_hardware_perm_checksum(perm)),
        'perm_broadcast_ok': 1.0,
    }


def _sector_layout_quality_np(keys, block_size):
    keys = _np_unit_direction(keys)
    n_ops = int(keys.shape[0])
    if n_ops == 0:
        return {
            'mean_compactness_cos': 0.0,
            'mean_sector_radius': 0.0,
            'max_sector_radius': 0.0,
            'min_sector_size': 0.0,
            'max_sector_size': 0.0,
        }
    sums, counts = _sector_sums_np(keys, block_size)
    centers = _np_unit_direction(sums)
    n_sectors = int(sums.shape[0])
    compact = np.zeros((n_sectors,), dtype=np.float32)
    radius_mean = np.zeros((n_sectors,), dtype=np.float32)
    radius_max = np.zeros((n_sectors,), dtype=np.float32)
    for sector in range(n_sectors):
        start = sector * int(block_size)
        stop = min(start + int(block_size), n_ops)
        if start >= stop:
            continue
        cos = keys[start:stop] @ centers[sector]
        dist = 1.0 - cos
        compact[sector] = np.mean(cos).astype(np.float32)
        radius_mean[sector] = np.mean(dist).astype(np.float32)
        radius_max[sector] = np.max(dist).astype(np.float32)
    valid = counts > 0
    if not np.any(valid):
        return {
            'mean_compactness_cos': 0.0,
            'mean_sector_radius': 0.0,
            'max_sector_radius': 0.0,
            'min_sector_size': 0.0,
            'max_sector_size': 0.0,
        }
    return {
        'mean_compactness_cos': float(np.mean(compact[valid])),
        'mean_sector_radius': float(np.mean(radius_mean[valid])),
        'max_sector_radius': float(np.max(radius_max[valid])),
        'min_sector_size': float(np.min(counts[valid])),
        'max_sector_size': float(np.max(counts[valid])),
    }


def _vq_target_sector_counts(n_ops, block_size):
    n_ops = int(n_ops)
    block_size = int(block_size)
    if n_ops <= 0:
        return np.zeros((0,), dtype=np.int32)
    n_sectors = (n_ops + block_size - 1) // block_size
    counts = np.full((n_sectors,), block_size, dtype=np.int32)
    remainder = n_ops % block_size
    if remainder:
        counts[-1] = remainder
    return counts


def _balanced_vq_assign_np(keys, centroids, target_counts):
    """Deterministic greedy fixed-capacity assignment."""
    n_ops = int(keys.shape[0])
    n_sectors = int(centroids.shape[0])
    if n_ops == 0:
        return np.zeros((0,), dtype=np.int32)
    sim = (keys @ centroids.T).astype(np.float32)
    flat_sim = sim.reshape(-1)
    flat_ops = np.repeat(np.arange(n_ops, dtype=np.int32), n_sectors)
    flat_sectors = np.tile(np.arange(n_sectors, dtype=np.int32), n_ops)
    order = np.lexsort((flat_sectors, flat_ops, -flat_sim))
    remaining = np.asarray(target_counts, dtype=np.int32).copy()
    assignment = np.full((n_ops,), -1, dtype=np.int32)
    assigned_count = 0
    for flat_idx in order:
        op_i = int(flat_ops[flat_idx])
        if assignment[op_i] >= 0:
            continue
        sector = int(flat_sectors[flat_idx])
        if remaining[sector] <= 0:
            continue
        assignment[op_i] = sector
        remaining[sector] -= 1
        assigned_count += 1
        if assigned_count == n_ops:
            break
    if assigned_count != n_ops or np.any(remaining != 0):
        raise RuntimeError(
            "balanced VQ assignment failed to fill fixed sector capacities")
    return assignment


def _balanced_vq_recompute_centroids_np(keys, assignment, n_sectors,
                                        previous_centroids):
    sums = np.zeros((n_sectors, keys.shape[-1]), dtype=np.float32)
    np.add.at(sums, assignment, keys)
    empty = np.linalg.norm(sums, axis=-1) <= 1.0e-12
    centroids = _np_unit_direction(sums)
    if np.any(empty):
        centroids[empty] = previous_centroids[empty]
    return _np_unit_direction(centroids)


def _decompose_balanced_assignment_cycles(current_assignment,
                                          desired_assignment,
                                          gains):
    moved = np.flatnonzero(desired_assignment != current_assignment)
    remaining = set(int(x) for x in moved.tolist())
    by_src = {}
    for op_i in remaining:
        by_src.setdefault(int(current_assignment[op_i]), []).append(op_i)
    for src, ops in by_src.items():
        ops.sort(key=lambda op_i: (-float(gains[op_i]), int(op_i),
                                   int(desired_assignment[op_i])))
    cycles = []
    while remaining:
        start = min(
            remaining,
            key=lambda op_i: (-float(gains[op_i]), int(op_i)))
        src0 = int(current_assignment[start])
        visited = {src0: 0}
        path_ops = []
        sector = src0
        while True:
            candidates = [op_i for op_i in by_src.get(sector, [])
                          if op_i in remaining]
            if not candidates:
                raise RuntimeError(
                    "balanced VQ movement limiter lost sector flow balance")
            op_i = candidates[0]
            path_ops.append(op_i)
            next_sector = int(desired_assignment[op_i])
            if next_sector in visited:
                cycle_ops = path_ops[visited[next_sector]:]
                for cycle_op in cycle_ops:
                    remaining.remove(cycle_op)
                cycles.append(tuple(cycle_ops))
                break
            visited[next_sector] = len(path_ops)
            sector = next_sector
    cycles.sort(key=lambda cyc: (
        -float(np.sum(gains[np.asarray(cyc, dtype=np.int32)])),
        -float(np.mean(gains[np.asarray(cyc, dtype=np.int32)])),
        len(cyc),
        min(cyc)))
    return cycles


def _assignment_to_limited_hardware_perm(current_assignment,
                                         desired_assignment,
                                         gains,
                                         block_size,
                                         max_move_frac,
                                         gain_eps):
    n_ops = int(current_assignment.size)
    if n_ops == 0:
        return np.arange(0, dtype=np.int32), np.zeros((0,), dtype=np.int32)
    max_move_frac = max(0.0, float(max_move_frac))
    move_budget = int(math.floor(float(n_ops) * max_move_frac))
    if max_move_frac > 0.0 and move_budget == 0:
        move_budget = 1
    selected = np.zeros((n_ops,), dtype=np.bool_)
    moved_so_far = 0
    for cycle in _decompose_balanced_assignment_cycles(
            current_assignment, desired_assignment, gains):
        cycle_arr = np.asarray(cycle, dtype=np.int32)
        cycle_gain = float(np.sum(gains[cycle_arr]))
        if cycle_gain <= float(gain_eps):
            continue
        if moved_so_far + int(cycle_arr.size) > move_budget:
            continue
        selected[cycle_arr] = True
        moved_so_far += int(cycle_arr.size)
    final_assignment = np.array(current_assignment, copy=True)
    final_assignment[selected] = desired_assignment[selected]

    perm = np.arange(n_ops, dtype=np.int32)
    n_sectors = int(np.max(current_assignment)) + 1
    for sector in range(n_sectors):
        holes = np.flatnonzero(
            np.logical_and(current_assignment == sector, selected))
        incoming = np.flatnonzero(
            np.logical_and(final_assignment == sector,
                           current_assignment != sector))
        holes = np.asarray(sorted(holes.tolist()), dtype=np.int32)
        incoming = np.asarray(sorted(incoming.tolist()), dtype=np.int32)
        if int(holes.size) != int(incoming.size):
            raise RuntimeError(
                "balanced VQ movement limiter produced unbalanced sector flow")
        if holes.size:
            perm[holes] = incoming

    return _validate_hardware_permutation_np(perm, n_ops), final_assignment


def _plan_hardware_sector_balanced_vq_repack(op_keys, block_size,
                                             iterations=4,
                                             max_move_frac=0.08,
                                             gain_eps=1.0e-3):
    """Plan VQ/IVF physical row packing for one operator pool.

    Returns perm[new_slot] = old_slot.  The planner is host-side, deterministic,
    and uses only current operator keys plus the existing physical layout.
    """
    block_size = int(block_size)
    iterations = max(1, int(iterations))
    gain_eps = float(gain_eps)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    keys = _np_unit_direction(op_keys)
    n_ops = int(keys.shape[0])
    perm = np.arange(n_ops, dtype=np.int32)
    compact_before, radius_before_mean, radius_before_max = (
        _sector_compactness_and_radius_np(keys, block_size))
    if n_ops == 0:
        return perm, _identity_hardware_repack_metrics(
            perm, compact_before, radius_before_mean, radius_before_max)

    target_counts = _vq_target_sector_counts(n_ops, block_size)
    n_sectors = int(target_counts.size)
    current_assignment = (
        np.arange(n_ops, dtype=np.int32) // np.int32(block_size))
    sums, _counts = _sector_sums_np(keys, block_size)
    centroids = _np_unit_direction(sums)

    desired_assignment = np.array(current_assignment, copy=True)
    for _ in range(iterations):
        desired_assignment = _balanced_vq_assign_np(
            keys, centroids, target_counts)
        centroids = _balanced_vq_recompute_centroids_np(
            keys, desired_assignment, n_sectors, centroids)

    sim = keys @ centroids.T
    gains = (
        sim[np.arange(n_ops), desired_assignment]
        - sim[np.arange(n_ops), current_assignment])
    perm, final_assignment = _assignment_to_limited_hardware_perm(
        current_assignment, desired_assignment, gains, block_size,
        max_move_frac=max_move_frac, gain_eps=gain_eps)
    moved_mask = perm != np.arange(n_ops, dtype=np.int32)
    moved_count = int(moved_mask.sum())
    new_keys = keys[perm]
    compact_after, radius_after_mean, radius_after_max = (
        _sector_compactness_and_radius_np(new_keys, block_size))
    before_quality = _sector_layout_quality_np(keys, block_size)
    after_quality = _sector_layout_quality_np(new_keys, block_size)
    selected_gains = gains[perm[moved_mask]] if moved_count else np.zeros(
        (0,), dtype=np.float32)
    full_moved = int(np.count_nonzero(desired_assignment != current_assignment))
    metrics = {
        'moved_count': float(moved_count),
        'moved_frac': float(moved_count) / float(max(n_ops, 1)),
        'candidate_count': float(full_moved),
        'full_vq_moved_count': float(full_moved),
        'full_vq_moved_frac': float(full_moved) / float(max(n_ops, 1)),
        'mean_gain': float(np.mean(selected_gains)) if moved_count else 0.0,
        'max_gain': float(np.max(selected_gains)) if moved_count else 0.0,
        'compactness_before_mean': float(compact_before.mean()),
        'compactness_after_mean': float(compact_after.mean()),
        'radius_before_mean': float(radius_before_mean),
        'radius_after_mean': float(radius_after_mean),
        'radius_before_max': float(radius_before_max),
        'radius_after_max': float(radius_after_max),
        'mean_compactness_cos': after_quality['mean_compactness_cos'],
        'mean_compactness_cos_before':
            before_quality['mean_compactness_cos'],
        'mean_sector_radius': after_quality['mean_sector_radius'],
        'max_sector_radius': after_quality['max_sector_radius'],
        'min_sector_size': after_quality['min_sector_size'],
        'max_sector_size': after_quality['max_sector_size'],
        'max_move_frac': float(max_move_frac),
        'vq_iterations': float(iterations),
        'strategy_balanced_vq': 1.0,
        'perm_checksum': float(_hardware_perm_checksum(perm)),
    }
    expected_counts = np.bincount(
        final_assignment, minlength=n_sectors).astype(np.int32)
    if not np.array_equal(expected_counts, target_counts):
        raise RuntimeError(
            "balanced VQ repack did not preserve fixed sector sizes")
    return perm, metrics


def _plan_hardware_sector_swap_repack(op_keys, block_size,
                                      farthest_per_sector=10,
                                      gain_eps=1.0e-3):
    """Plan a physical row permutation for one operator pool.

    Returns perm[new_slot] = old_slot.  Planning is intentionally host-side and
    uses only the compact RW-derived operator coordinates.
    """
    block_size = int(block_size)
    farthest_per_sector = int(farthest_per_sector)
    gain_eps = float(gain_eps)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    keys = _np_unit_direction(op_keys)
    n_ops = int(keys.shape[0])
    perm = np.arange(n_ops, dtype=np.int32)
    compact_before = np.zeros((0,), dtype=np.float32)
    radius_before_mean = np.float32(0.0)
    radius_before_max = np.float32(0.0)
    if n_ops == 0:
        return perm, _identity_hardware_repack_metrics(
            perm, compact_before, radius_before_mean, radius_before_max)

    n_sectors = (n_ops + block_size - 1) // block_size
    sums, counts = _sector_sums_np(keys, block_size)
    centers = _np_unit_direction(sums)
    compact_before, radius_before_mean, radius_before_max = (
        _sector_compactness_and_radius_np(keys, block_size))

    if farthest_per_sector <= 0:
        return perm, _identity_hardware_repack_metrics(
            perm, compact_before, radius_before_mean, radius_before_max)

    candidate_indices = []
    if farthest_per_sector > 0:
        for sector in range(n_sectors):
            start = sector * block_size
            stop = min(start + block_size, n_ops)
            if start >= stop:
                continue
            idx = np.arange(start, stop, dtype=np.int32)
            dots = keys[start:stop] @ centers[sector]
            order = np.lexsort((idx, dots))
            take_n = min(farthest_per_sector, int(idx.size))
            candidate_indices.extend(idx[order[:take_n]].tolist())

    candidate_indices = np.asarray(sorted(set(candidate_indices)),
                                   dtype=np.int32)
    candidate_count = int(candidate_indices.size)
    if candidate_count == 0:
        return perm, _identity_hardware_repack_metrics(
            perm, compact_before, radius_before_mean, radius_before_max)

    candidate_sectors = candidate_indices // block_size
    sums_base = np.array(sums, copy=True)
    holes_by_sector = [[] for _ in range(n_sectors)]
    for old_idx, sector in zip(candidate_indices, candidate_sectors):
        sums_base[sector] -= keys[old_idx]
        holes_by_sector[sector].append(int(old_idx))

    capacities = np.asarray([len(h) for h in holes_by_sector], dtype=np.int32)
    assigned_by_sector = [[] for _ in range(n_sectors)]
    assigned_dst = {}
    move_gains = {}

    possible_moves = []
    for old_idx in candidate_indices:
        old_idx_i = int(old_idx)
        src = old_idx_i // block_size
        k = keys[old_idx_i]
        for dst in range(n_sectors):
            if dst == src or len(holes_by_sector[dst]) == 0:
                continue
            src_sum = sums_base[src] + k
            dst_sum = sums_base[dst]
            gain = (
                np.linalg.norm(src_sum - k)
                + np.linalg.norm(dst_sum + k)
                - np.linalg.norm(src_sum)
                - np.linalg.norm(dst_sum)
            )
            if gain > gain_eps:
                possible_moves.append(
                    (-float(gain), old_idx_i, int(dst), float(gain)))

    possible_moves.sort()

    for _neg_gain, old_idx_i, dst, gain in possible_moves:
        if old_idx_i in assigned_dst or capacities[dst] <= 0:
            continue
        assigned_dst[old_idx_i] = dst
        move_gains[old_idx_i] = gain
        capacities[dst] -= 1
        assigned_by_sector[dst].append(old_idx_i)

    def undo_lowest_gain_incoming(sector):
        incoming = [
            old_idx_i for old_idx_i in assigned_by_sector[sector]
            if (old_idx_i // block_size) != sector
        ]
        if not incoming:
            return False
        undo_old = sorted(
            incoming,
            key=lambda old_idx_i: (move_gains.get(old_idx_i, 0.0),
                                   old_idx_i))[0]
        assigned_by_sector[sector].remove(undo_old)
        capacities[sector] += 1
        assigned_dst.pop(undo_old, None)
        move_gains.pop(undo_old, None)
        return True

    # Non-moved candidates keep their original physical-sector hole whenever
    # possible. If a greedy positive move consumed that capacity, undo the
    # lowest-gain incoming move instead of forcing a no-gain candidate out.
    while True:
        progress = False
        for old_idx in candidate_indices:
            old_idx_i = int(old_idx)
            if old_idx_i in assigned_dst:
                continue
            src = old_idx_i // block_size
            if capacities[src] <= 0:
                progress = undo_lowest_gain_incoming(src) or progress
                if capacities[src] <= 0:
                    continue
            assigned_dst[old_idx_i] = src
            capacities[src] -= 1
            assigned_by_sector[src].append(old_idx_i)
            progress = True
        if len(assigned_dst) == candidate_count or not progress:
            break

    # Capacity is fixed. If accepted positive moves consumed a sector's holes,
    # fill remaining holes deterministically with the lowest old indices.
    for old_idx in candidate_indices:
        old_idx_i = int(old_idx)
        if old_idx_i in assigned_dst:
            continue
        available = np.flatnonzero(capacities > 0)
        if available.size == 0:
            raise RuntimeError(
                "hardware repack planner exhausted candidate holes")
        dst = int(available[0])
        assigned_dst[old_idx_i] = dst
        capacities[dst] -= 1
        assigned_by_sector[dst].append(old_idx_i)

    if np.any(capacities != 0):
        raise RuntimeError(
            "hardware repack planner did not fill every candidate hole")

    new_perm = np.arange(n_ops, dtype=np.int32)
    for sector, assigned_old in enumerate(assigned_by_sector):
        holes = sorted(holes_by_sector[sector])
        assigned_old = sorted(assigned_old)
        if len(holes) != len(assigned_old):
            raise RuntimeError(
                "hardware repack planner produced mismatched hole capacity")
        for new_slot, old_slot in zip(holes, assigned_old):
            new_perm[new_slot] = old_slot

    new_perm = _validate_hardware_permutation_np(new_perm, n_ops)

    moved_mask = new_perm != np.arange(n_ops, dtype=np.int32)
    moved_count = int(moved_mask.sum())
    new_keys = keys[new_perm]
    compact_after, radius_after_mean, radius_after_max = (
        _sector_compactness_and_radius_np(new_keys, block_size))
    moved_gains = [
        move_gains.get(int(old_slot), 0.0)
        for old_slot in new_perm[moved_mask]
    ]
    mean_gain = float(np.mean(moved_gains)) if moved_gains else 0.0
    max_gain = float(np.max(moved_gains)) if moved_gains else 0.0
    metrics = {
        'moved_count': float(moved_count),
        'moved_frac': float(moved_count) / float(max(n_ops, 1)),
        'candidate_count': float(candidate_count),
        'mean_gain': mean_gain,
        'max_gain': max_gain,
        'compactness_before_mean': float(compact_before.mean()),
        'compactness_after_mean': float(compact_after.mean()),
        'radius_before_mean': float(radius_before_mean),
        'radius_after_mean': float(radius_after_mean),
        'radius_before_max': float(radius_before_max),
        'radius_after_max': float(radius_after_max),
        'perm_checksum': float(_hardware_perm_checksum(new_perm)),
    }
    return new_perm, metrics


def _plan_hardware_sector_repack(op_keys, block_size,
                                 farthest_per_sector=10,
                                 gain_eps=1.0e-3,
                                 strategy=_V4168_VQ_REPACK_STRATEGY,
                                 max_move_frac=_V4168_VQ_REPACK_MAX_MOVE_FRAC,
                                 vq_iterations=_V4168_VQ_REPACK_ITERATIONS):
    strategy = str(strategy or _V4168_VQ_REPACK_STRATEGY).lower()
    if strategy == 'balanced_vq':
        return _plan_hardware_sector_balanced_vq_repack(
            op_keys, block_size,
            iterations=vq_iterations,
            max_move_frac=max_move_frac,
            gain_eps=gain_eps)
    if strategy in ('sector_swap', 'legacy_swap', 'legacy'):
        return _plan_hardware_sector_swap_repack(
            op_keys, block_size,
            farthest_per_sector=farthest_per_sector,
            gain_eps=gain_eps)
    raise ValueError(
        "hardware_repack_strategy must be 'balanced_vq' or 'sector_swap', "
        f"got {strategy!r}")


def _path_key_tuple(path):
    out = []
    for entry in path:
        if hasattr(entry, 'key'):
            out.append(entry.key)
        elif hasattr(entry, 'idx'):
            out.append(entry.idx)
        elif hasattr(entry, 'name'):
            out.append(entry.name)
        else:
            out.append(str(entry))
    return tuple(out)


def _path_endswith(path_tuple, suffix):
    if len(path_tuple) < len(suffix):
        return False
    return tuple(path_tuple[-len(suffix):]) == tuple(suffix)


def _permute_pool_axis_leaf(leaf, perm):
    if not hasattr(leaf, 'shape') or getattr(leaf, 'ndim', 0) < 1:
        return leaf
    if int(leaf.shape[0]) != int(perm.shape[0]):
        return leaf
    return jnp.take(leaf, perm, axis=0)


def _apply_pool_permutations_to_params_and_opt_state(params, opt_state,
                                                     pool_perms):
    """Apply physical pool permutations to params and matching optimizer rows."""
    if not pool_perms:
        return params, opt_state
    jax_perms = {
        pool: jnp.asarray(perm, dtype=jnp.int32)
        for pool, perm in pool_perms.items()
    }
    pool_leaf_to_perm = {}
    for pool, _op_key, read_key, write_key, _block_key, _top_key in (
            _HARDWARE_REPACK_POOLS):
        if pool in jax_perms:
            pool_leaf_to_perm[_op_key] = jax_perms[pool]
            pool_leaf_to_perm[read_key] = jax_perms[pool]
            pool_leaf_to_perm[write_key] = jax_perms[pool]

    params_was_frozen = isinstance(params, FrozenDict)
    mutable_params = unfreeze(params) if params_was_frozen else dict(params)
    pool_params = mutable_params.get('neuron_pool', {})
    if isinstance(pool_params, FrozenDict):
        pool_params = unfreeze(pool_params)
    else:
        pool_params = dict(pool_params)
    for leaf_key, perm in pool_leaf_to_perm.items():
        if leaf_key in pool_params:
            pool_params[leaf_key] = _permute_pool_axis_leaf(
                pool_params[leaf_key], perm)
    mutable_params['neuron_pool'] = pool_params
    new_params = freeze(mutable_params) if params_was_frozen else mutable_params

    suffix_to_perm = {
        ('neuron_pool', leaf_key): perm
        for leaf_key, perm in pool_leaf_to_perm.items()
    }

    def opt_leaf(path, leaf):
        path_tuple = _path_key_tuple(path)
        for suffix, perm in suffix_to_perm.items():
            if _path_endswith(path_tuple, suffix):
                return _permute_pool_axis_leaf(leaf, perm)
        return leaf

    new_opt_state = jax.tree_util.tree_map_with_path(opt_leaf, opt_state)
    return new_params, new_opt_state


def hardware_sector_static_metrics(model_config, model_axis_size=1,
                                   batch_size=None, max_seq_len=None,
                                   data_axis_size=1,
                                   bucketed_execution_enabled=True):
    """Cheap exact-global-sector-topK bucketed execution cost metrics."""
    metrics = {}
    model_axis_size = max(1, int(model_axis_size))
    data_axis_size = max(1, int(data_axis_size))
    bucketed_execution_enabled = bool(bucketed_execution_enabled)
    if batch_size is not None and max_seq_len is not None:
        local_batch = max(1, int(batch_size) // data_axis_size)
        token_pair_span = max(1, local_batch * int(max_seq_len))
    else:
        token_pair_span = None
    for pool, _op_key, _read_key, _write_key, block_key, top_key in (
            _HARDWARE_REPACK_POOLS):
        if pool == 'attn_qk':
            n_key = 'n_qk'
            route_count = 2
        elif pool == 'attn_v':
            n_key = 'n_v'
            route_count = 1
        else:
            n_key = 'n_rst'
            route_count = 1
        n_ops = int(model_config.get(n_key, model_config.get('n_know', 0)))
        sector_size = int(model_config.get(block_key, 256))
        topk = int(model_config.get(top_key, 2))
        n_local_ops = max(
            1, (n_ops + model_axis_size - 1) // model_axis_size)
        n_local_sectors = max(
            1, (n_local_ops + sector_size - 1) // sector_size)
        n_sectors = n_local_sectors * model_axis_size
        topk_eff = min(max(1, topk), n_sectors)
        semantic_selected = topk_eff * sector_size
        token_gather_spmd_slots = model_axis_size * semantic_selected
        dense_ops = max(1, n_ops)
        valid_est = min(n_ops, semantic_selected)
        metrics[f'sector/{pool}/topk'] = float(topk_eff)
        metrics[f'sector/{pool}/sector_size'] = float(sector_size)
        metrics[f'sector/{pool}/num_global_sectors'] = float(n_sectors)
        metrics[f'sector/{pool}/num_local_sectors'] = float(n_local_sectors)
        metrics[f'sector/{pool}/model_axis_size'] = float(model_axis_size)
        metrics[f'sector/{pool}/route_count'] = float(route_count)
        metrics[f'sector/{pool}/semantic_selected_ops_per_token'] = float(
            semantic_selected)
        metrics[f'sector/{pool}/token_gather_spmd_slots_per_token'] = float(
            token_gather_spmd_slots)
        metrics[f'sector/{pool}/spmd_tile_slots_per_token'] = float(
            token_gather_spmd_slots)
        metrics[f'sector/{pool}/useful_local_tile_slots_per_token'] = float(
            semantic_selected)
        metrics[f'sector/{pool}/remote_dummy_tile_slot_frac'] = (
            1.0 - (
                float(semantic_selected)
                / float(max(token_gather_spmd_slots, 1))))
        metrics[f'sector/{pool}/estimated_dense_ops_per_token'] = float(
            n_ops)
        metrics[f'sector/{pool}/semantic_compute_frac_vs_dense'] = float(
            semantic_selected) / float(dense_ops)
        metrics[f'sector/{pool}/spmd_compute_frac_vs_dense'] = float(
            token_gather_spmd_slots) / float(dense_ops)
        metrics[f'sector/{pool}/selected_valid_ops_mean'] = float(valid_est)
        metrics[f'sector/{pool}/selected_sector_count_mean'] = float(topk_eff)
        metrics[f'sector/{pool}/useful_local_sector_slots_mean'] = float(
            topk_eff)
        metrics[f'sector/{pool}/local_selected_sector_count_mean'] = (
            float(topk_eff) / float(model_axis_size))
        metrics[f'sector/{pool}/remote_dummy_sector_slot_frac'] = (
            1.0 - (float(topk_eff) / float(max(model_axis_size * topk_eff, 1))))
        pool_bucketed_execution_enabled = (
            bool(bucketed_execution_enabled) and pool != 'attn_qk')
        metrics[f'sector/{pool}/bucketed_execution_enabled'] = float(
            pool_bucketed_execution_enabled)
        metrics[f'sector/{pool}/bucket_fallback_used'] = 0.0
        metrics[f'sector/{pool}/bucket_overflow_count'] = 0.0
        metrics[f'sector/{pool}/bucket_overflow_frac'] = 0.0
        if token_pair_span is not None:
            pair_span = token_pair_span * route_count
            bucket_capacity, bucket_avg = _v4168_sector_bucket_capacity(
                pair_span, topk_eff, n_sectors)
            bucket_total_slots = n_local_sectors * bucket_capacity
            bucket_expected_used = (
                float(pair_span * topk_eff) / float(model_axis_size))
            bucket_fill = (
                bucket_expected_used / float(max(bucket_total_slots, 1)))
            metrics[f'sector/{pool}/bucket_pair_span'] = float(pair_span)
            metrics[f'sector/{pool}/bucket_avg_expected'] = float(bucket_avg)
            metrics[f'sector/{pool}/bucket_capacity'] = float(bucket_capacity)
            metrics[f'sector/{pool}/bucket_total_slots'] = float(
                bucket_total_slots)
            metrics[f'sector/{pool}/bucket_used_slots'] = float(
                bucket_expected_used)
            metrics[f'sector/{pool}/bucket_expected_used_slots'] = float(
                bucket_expected_used)
            metrics[f'sector/{pool}/bucket_fill_frac'] = float(bucket_fill)
            metrics[f'sector/{pool}/bucket_max_fill'] = float(bucket_capacity)
            metrics[f'sector/{pool}/bucket_mean_fill'] = float(
                bucket_expected_used / float(max(n_local_sectors, 1)))
            metrics[f'sector/{pool}/sector_fill_mean'] = float(
                bucket_expected_used / float(max(n_local_sectors, 1)))
            metrics[f'sector/{pool}/sector_fill_max'] = float(
                bucket_capacity)
            metrics[f'sector/{pool}/sector_overflow_count'] = 0.0
            metrics[f'sector/{pool}/selected_sector_frac'] = float(
                topk_eff) / float(max(n_sectors, 1))
            metrics[f'sector/{pool}/effective_operator_frac'] = float(
                valid_est) / float(dense_ops)
    return metrics


def _hardware_repack_enabled(config):
    if not isinstance(config, dict):
        return False
    return bool(config.get('hardware_repack_enabled', False))


def _global_jax_array_to_host_np(x, dtype=np.float32):
    if bool(getattr(x, 'is_fully_addressable', True)):
        return np.asarray(jax.device_get(x), dtype=dtype)

    local = np.zeros(tuple(x.shape), dtype=dtype)
    local_mask = np.zeros((int(x.shape[0]),), dtype=np.int32)
    for shard in getattr(x, 'addressable_shards', ()):
        idx = getattr(shard, 'index', None)
        if not isinstance(idx, tuple) or not idx:
            continue
        row_idx = idx[0]
        if not isinstance(row_idx, slice):
            continue
        shard_arr = np.asarray(jax.device_get(shard.data), dtype=dtype)
        local[idx] = shard_arr
        start = 0 if row_idx.start is None else int(row_idx.start)
        stop = int(x.shape[0]) if row_idx.stop is None else int(row_idx.stop)
        local_mask[start:stop] = 1

    try:
        from jax.experimental.multihost_utils import process_allgather
        gathered = np.asarray(process_allgather(local), dtype=dtype)
        gathered_mask = np.asarray(
            process_allgather(local_mask), dtype=np.int32)
        if gathered.ndim == local.ndim:
            return gathered
        mask_f = gathered_mask.astype(dtype)
        while mask_f.ndim < gathered.ndim:
            mask_f = mask_f[..., None]
        denom = np.maximum(mask_f.sum(axis=0), dtype(1.0))
        return (gathered * mask_f).sum(axis=0) / denom
    except Exception:
        if np.all(local_mask > 0):
            return local
        raise


def _canonical_hardware_permutation_across_hosts(perm, pool=''):
    """Return the process-0 permutation on every host, with loud validation."""
    perm = _validate_hardware_permutation_np(perm, pool=pool)
    checksum = _hardware_perm_checksum(perm)
    checksum_meta = _hardware_perm_checksum_meta(checksum)
    if jax.process_count() <= 1:
        return perm, checksum, 1.0

    from jax.experimental.multihost_utils import process_allgather

    local_meta = np.asarray(
        [int(perm.size), int(checksum_meta)], dtype=np.int32)
    all_meta = np.asarray(process_allgather(local_meta)).reshape(-1, 2)
    if not np.all(all_meta[:, 0] == int(perm.size)):
        raise RuntimeError(
            f"hardware repack {pool} permutation length mismatch across "
            f"hosts: {all_meta[:, 0].tolist()}")

    try:
        from jax.experimental.multihost_utils import broadcast_one_to_all
        perm = np.asarray(
            broadcast_one_to_all(np.asarray(perm, dtype=np.int32)),
            dtype=np.int32)
        perm = _validate_hardware_permutation_np(perm, pool=pool)
        checksum = _hardware_perm_checksum(perm)
        checksum_meta = _hardware_perm_checksum_meta(checksum)
    except Exception as exc:
        if not np.all(all_meta[:, 1] == int(checksum_meta)):
            raise RuntimeError(
                f"hardware repack {pool} permutation checksum mismatch "
                f"across hosts and broadcast failed: "
                f"{all_meta[:, 1].tolist()}") from exc
        perm = _validate_hardware_permutation_np(perm, pool=pool)

    final_meta = np.asarray(
        process_allgather(
            np.asarray([int(perm.size), int(checksum_meta)], dtype=np.int32))
    ).reshape(-1, 2)
    if (not np.all(final_meta[:, 0] == int(perm.size))
            or not np.all(final_meta[:, 1] == int(checksum_meta))):
        raise RuntimeError(
            f"hardware repack {pool} broadcast produced inconsistent "
            f"permutation metadata: {final_meta.tolist()}")
    return perm, checksum, 1.0


def maybe_hardware_repack(params, opt_state, model_config, mesh, step, config):
    """Physically repack v4168 SRW pool rows and matching optimizer slots."""
    if not _hardware_repack_enabled(config):
        return params, opt_state, {}

    strategy = str(config.get(
        'hardware_repack_strategy',
        _V4168_VQ_REPACK_STRATEGY)).lower()
    farthest_per_sector = int(
        config.get('hardware_repack_farthest_per_sector', 10))
    gain_eps = float(config.get('hardware_repack_gain_eps', 1.0e-3))
    max_move_frac = float(config.get(
        'hardware_repack_max_move_frac',
        _V4168_VQ_REPACK_MAX_MOVE_FRAC))
    vq_iterations = int(config.get(
        'hardware_repack_vq_iterations',
        _V4168_VQ_REPACK_ITERATIONS))

    pool_params = params['neuron_pool']
    op_keys = _pool_operator_keys(pool_params)

    model_axis_size = int(getattr(mesh, 'shape', {}).get('model', 1))
    pool_perms = {}
    metrics = {}
    total_moved = 0.0
    for pool, op_key_key, _read_key, _write_key, block_key, _top_key in (
            _HARDWARE_VQ_REPACK_POOLS):
        block_size = int(model_config.get(block_key, 256))
        key_device = _forward_unit_direction(
            op_keys[op_key_key].astype(jnp.float32))
        key_host = _global_jax_array_to_host_np(key_device, dtype=np.float32)
        perm, pool_metrics = _plan_hardware_sector_repack(
            key_host, block_size,
            farthest_per_sector=farthest_per_sector,
            gain_eps=gain_eps,
            strategy=strategy,
            max_move_frac=max_move_frac,
            vq_iterations=vq_iterations)
        perm, perm_checksum, perm_broadcast_ok = (
            _canonical_hardware_permutation_across_hosts(perm, pool=pool))
        moved_mask = perm != np.arange(int(perm.size), dtype=np.int32)
        pool_metrics['moved_count'] = float(int(moved_mask.sum()))
        pool_metrics['moved_frac'] = (
            float(pool_metrics['moved_count']) / float(max(int(perm.size), 1)))
        pool_metrics['perm_checksum'] = float(perm_checksum)
        pool_metrics['perm_broadcast_ok'] = float(perm_broadcast_ok)
        moved = int(pool_metrics['moved_count'])
        total_moved += float(moved)
        if moved > 0:
            pool_perms[pool] = perm
        for name, value in pool_metrics.items():
            metrics[f'repack/{pool}/{name}'] = float(value)

    metrics['repack/total_moved_count'] = float(total_moved)
    metrics['repack/step'] = float(int(step))
    metrics['repack/drift_snapshot_refreshed'] = 0.0
    metrics['repack/strategy_balanced_vq'] = float(
        1.0 if strategy == 'balanced_vq' else 0.0)
    metrics['repack/max_move_frac'] = float(max_move_frac)
    metrics['repack/vq_iterations'] = float(vq_iterations)
    metrics.update(hardware_sector_static_metrics(
        model_config, model_axis_size=model_axis_size,
        bucketed_execution_enabled=bool(
            config.get('hardware_sector_execution_enabled', False))))
    if total_moved == 0.0:
        return params, opt_state, metrics

    new_params, new_opt_state = _apply_pool_permutations_to_params_and_opt_state(
        params, opt_state, pool_perms)
    return new_params, new_opt_state, metrics


def apply_operator_bundle_swaps(train_state, pool_name, layer_idx, swaps):
    """Apply valid-row operator bundle swaps to (params, opt_state)."""
    del layer_idx
    if not isinstance(train_state, tuple) or len(train_state) < 2:
        raise ValueError(
            "apply_operator_bundle_swaps expects train_state as "
            "(params, opt_state[, ...]) in this v4168 trainer.")
    params, opt_state = train_state[0], train_state[1]
    pool_name = str(pool_name)
    if pool_name == 'v':
        pool_name = 'attn_v'
    if pool_name not in ('attn_v', 'rst', 'attn_qk'):
        raise ValueError(f"unsupported operation-space pool {pool_name!r}")
    pool_params = params['neuron_pool']
    n_rows = None
    for pool, _op_key, read_key, _write_key, _block_key, _top_key in (
            _HARDWARE_REPACK_POOLS):
        if pool == pool_name:
            n_rows = int(pool_params[read_key].shape[0])
            break
    if n_rows is None:
        raise ValueError(f"unsupported operation-space pool {pool_name!r}")
    perm = np.arange(n_rows, dtype=np.int32)
    for a, b in swaps:
        a_i = int(a)
        b_i = int(b)
        if not (0 <= a_i < n_rows and 0 <= b_i < n_rows):
            raise ValueError(
                f"swap ({a_i}, {b_i}) outside pool {pool_name} rows={n_rows}")
        perm[a_i] = b_i
        perm[b_i] = a_i
    new_params, new_opt_state = _apply_pool_permutations_to_params_and_opt_state(
        params, opt_state, {pool_name: perm})
    if len(train_state) == 2:
        return new_params, new_opt_state
    return (new_params, new_opt_state, *train_state[2:])


def operation_space_static_metrics(model_config, config=None,
                                   model_axis_size=1):
    """Static Region/Block/Operator Atlas layout metrics for diagnostics."""
    config = config if isinstance(config, dict) else {}
    pool_cfgs = config.get('operation_space_pool_layouts', {})
    if not isinstance(pool_cfgs, dict):
        pool_cfgs = {}
    model_axis_size = max(1, int(model_axis_size))
    out = {
        'opspace/mode_tau_free_relu': 1.0,
        'opspace/owner_model_axis': 1.0,
        'opspace/no_selected_request_drop': 1.0,
        'opspace/bucket_overflow_semantics': 0.0,
    }
    defaults = {
        'qk': ('n_qk', 'dense'),
        'v': ('n_v', 'dense'),
        'rst': ('n_rst', 'sparse_region_block'),
    }
    for label, (n_key, backend_d) in defaults.items():
        pcfg = pool_cfgs.get(label, {})
        if not isinstance(pcfg, dict):
            pcfg = {}
        execution_backend = str(pcfg.get(
            'execution_backend', backend_d)).lower()
        num_regions = int(pcfg.get('num_regions', 1))
        blocks_per_region = int(pcfg.get('blocks_per_region', 1))
        operators_per_block = int(pcfg.get('operators_per_block', 1))
        visible_regions = int(pcfg.get(
            'visible_regions', 1))
        visible_blocks_per_region = int(pcfg.get(
            'visible_blocks_per_region', 1))
        region_capacity_factor = float(pcfg.get(
            'region_capacity_factor', 1.25))
        block_capacity_factor = float(pcfg.get(
            'block_capacity_factor',
            pcfg.get('bucket_capacity_factor', 1.25)))
        owner_count = int(pcfg.get('num_devices', model_axis_size))
        owner_count = max(1, owner_count)
        n_ops = int(model_config.get(n_key, 0))
        total_capacity = max(
            1, num_regions * blocks_per_region * operators_per_block)
        visible_blocks_per_token = visible_regions * visible_blocks_per_region
        visible_ops_per_token = (
            visible_blocks_per_token * operators_per_block)
        physical_visible_ops_per_token = visible_ops_per_token
        out[f'opspace/{label}/visible_regions'] = float(visible_regions)
        out[f'opspace/{label}/visible_ops_per_token'] = float(
            visible_ops_per_token)
        out[f'opspace/{label}/num_regions'] = float(num_regions)
        out[f'opspace/{label}/regions_per_owner'] = (
            float(num_regions) / float(owner_count))
        out[f'opspace/{label}/blocks_per_region'] = float(blocks_per_region)
        out[f'opspace/{label}/operators_per_block'] = float(
            operators_per_block)
        out[f'opspace/{label}/visible_regions'] = float(visible_regions)
        out[f'opspace/{label}/region_capacity_factor'] = float(
            region_capacity_factor)
        out[f'opspace/{label}/block_capacity_factor'] = float(
            block_capacity_factor)
        out[f'opspace/{label}/visible_blocks_per_region'] = float(
            visible_blocks_per_region)
        out[f'opspace/{label}/visible_blocks_per_token'] = float(
            visible_blocks_per_token)
        out[f'opspace/{label}/visible_ops_per_token'] = float(
            visible_ops_per_token)
        out[f'opspace/{label}/physical_visible_ops_per_token'] = float(
            physical_visible_ops_per_token)
        out[f'opspace/{label}/compute_frac_vs_dense'] = float(
            visible_ops_per_token) / float(total_capacity)
        out[f'opspace/{label}/logical_compute_frac_vs_dense'] = float(
            visible_ops_per_token) / float(total_capacity)
        out[f'opspace/{label}/physical_compute_frac_vs_dense'] = float(
            physical_visible_ops_per_token) / float(total_capacity)
        out[f'opspace/{label}/execution_backend_dense'] = float(
            1.0 if execution_backend == 'dense' else 0.0)
        out[f'opspace/{label}/execution_backend_sparse_region_block'] = float(
            1.0 if execution_backend == 'sparse_region_block' else 0.0)
        if execution_backend == 'sparse_region_block':
            out[f'opspace/{label}/bucket_capacity_factor'] = float(
                pcfg.get('bucket_capacity_factor', block_capacity_factor))
            out[f'opspace/{label}/bucket_chunk_size'] = float(
                pcfg.get('bucket_chunk_size', 128))
            out[f'opspace/{label}/high_regret_threshold'] = float(
                pcfg.get('high_regret_threshold', 0.05))
    return out


def _operation_space_repack_enabled(config):
    if not isinstance(config, dict):
        return False
    return bool(config.get('operation_space_repack_enabled', False))


def maybe_operation_space_repack(params, opt_state, model_config, mesh, step,
                                 config):
    """Run fixed-K operation-space positive-gain bundle repack."""
    if not _operation_space_repack_enabled(config):
        return params, opt_state, {}

    pool_layouts = config.get('operation_space_pool_layouts', {})
    if not isinstance(pool_layouts, dict):
        pool_layouts = {}
    repack_pools = config.get('operation_space_repack_pools', {})
    if not isinstance(repack_pools, dict):
        repack_pools = {}
    std_mult = OPSPACE_REPACK_STD_MULT
    min_assignment_gain = OPSPACE_REPACK_MIN_ASSIGNMENT_GAIN
    min_swap_gain = OPSPACE_REPACK_MIN_SWAP_GAIN

    pool_params = params['neuron_pool']
    op_keys = _pool_operator_keys(pool_params)
    pool_perms = {}
    metrics = {}
    total_swaps = 0.0
    total_moved = 0.0
    model_axis_size = int(getattr(mesh, 'shape', {}).get('model', 1))
    for pool, op_key_key, _read_key, _write_key, _block_key, _top_key in (
            _HARDWARE_REPACK_POOLS):
        label = (
            'qk' if pool == 'attn_qk'
            else ('v' if pool == 'attn_v' else 'rst'))
        pcfg = pool_layouts.get(label, {})
        if not isinstance(pcfg, dict):
            pcfg = {}
        rcfg = repack_pools.get(label, {})
        if not isinstance(rcfg, dict):
            rcfg = {}
        if not bool(rcfg.get('enabled', label == 'rst')):
            continue
        max_swaps = int(rcfg.get(
            'max_swaps_per_repack', 256 if label == 'rst' else 0))
        if max_swaps <= 0:
            continue
        key_device = _forward_unit_direction(
            op_keys[op_key_key].astype(jnp.float32))
        key_host = _global_jax_array_to_host_np(key_device, dtype=np.float32)
        perm, pool_metrics = _plan_operation_space_threshold_repack(
            key_host,
            lanes=int(pcfg.get('lanes', 32 if label == 'rst' else 8)),
            tiles_per_lane=int(pcfg.get(
                'tiles_per_lane', 8 if label == 'rst' else 4)),
            tile_size=int(pcfg.get('tile_size', 128)),
            max_swaps_per_repack=max_swaps,
            std_mult=std_mult,
            min_assignment_gain=min_assignment_gain,
            min_swap_gain=min_swap_gain)
        perm, perm_checksum, perm_broadcast_ok = (
            _canonical_hardware_permutation_across_hosts(
                perm, pool=f'opspace/{label}'))
        moved_mask = perm != np.arange(int(perm.size), dtype=np.int32)
        moved_count = int(moved_mask.sum())
        swap_count = int(pool_metrics.get('swaps', 0.0))
        pool_metrics['moved_count'] = float(moved_count)
        pool_metrics['moved_frac'] = float(moved_count) / float(
            max(int(perm.size), 1))
        pool_metrics['perm_checksum'] = float(perm_checksum)
        pool_metrics['perm_broadcast_ok'] = float(perm_broadcast_ok)
        pool_metrics['optimizer_state_swapped'] = 1.0
        pool_metrics['tau_values_changed'] = 0.0
        total_swaps += float(swap_count)
        total_moved += float(moved_count)
        if moved_count > 0:
            pool_perms[pool] = perm
        for name, value in pool_metrics.items():
            metrics[f'repack/{label}/{name}'] = float(value)

    metrics['repack/total_swaps'] = float(total_swaps)
    metrics['repack/total_moved_count'] = float(total_moved)
    metrics['repack/step'] = float(int(step))
    metrics['repack/strategy_threshold_positive_gain_swap'] = 1.0
    metrics.update(operation_space_static_metrics(
        model_config, config=config, model_axis_size=model_axis_size))
    if total_moved == 0.0:
        return params, opt_state, metrics

    new_params, new_opt_state = _apply_pool_permutations_to_params_and_opt_state(
        params, opt_state, pool_perms)
    return new_params, new_opt_state, metrics


def _v4168_block_sparse_config(block_size, top_blocks):
    block_size = int(block_size)
    top_blocks = int(top_blocks)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if top_blocks < 1:
        raise ValueError(f"top_blocks must be >= 1, got {top_blocks}")
    return block_size, top_blocks


_V4168_BUCKET_CHUNK_SIZE = 512
_V4168_EXECUTION_PAIR_PRUNE_EPS = 1.0e-6
_V4168_SECTOR_BUCKET_CAPACITY_MULT = 1.5
_V4168_SECTOR_BUCKET_MIN_CAPACITY = 32
_V4168_SECTOR_BUCKET_CAPACITY_ROUND_MULTIPLE = 128
_V4168_SECTOR_BUCKET_ROUND_POWER_OF_TWO = False


def _v4168_next_power_of_two(x):
    x = max(1, int(x))
    return 1 << (x - 1).bit_length()


def _v4168_sector_bucket_capacity(pair_span, topk, global_n_sectors):
    """Static sector-bucket capacity policy for token/route-sector pairs."""
    pair_span = max(1, int(pair_span))
    topk = max(1, int(topk))
    global_n_sectors = max(1, int(global_n_sectors))
    selected_pairs = pair_span * topk
    avg_pairs = int(math.ceil(float(selected_pairs) / float(global_n_sectors)))
    capacity = max(
        int(_V4168_SECTOR_BUCKET_MIN_CAPACITY),
        int(math.ceil(
            float(avg_pairs) * float(_V4168_SECTOR_BUCKET_CAPACITY_MULT))),
    )
    if _V4168_SECTOR_BUCKET_ROUND_POWER_OF_TWO:
        capacity = _v4168_next_power_of_two(capacity)
    else:
        round_multiple = max(
            1, int(_V4168_SECTOR_BUCKET_CAPACITY_ROUND_MULTIPLE))
        capacity = int(
            math.ceil(float(capacity) / float(round_multiple))
            * round_multiple)
    capacity = min(max(1, capacity), selected_pairs)
    return capacity, avg_pairs


def _v4168_build_sector_buckets(local_sector_ids, selected_here, pair_span,
                                topk, n_local_sectors, bucket_capacity):
    """Build fixed-size local-sector buckets from exact selected sectors."""
    pair_span = int(pair_span)
    topk = int(topk)
    n_local_sectors = int(n_local_sectors)
    bucket_capacity = int(bucket_capacity)
    flat_count = pair_span * topk

    flat_pair_ids = (
        jnp.arange(flat_count, dtype=jnp.int32)
        // jnp.asarray(topk, dtype=jnp.int32))
    flat_sector = local_sector_ids.reshape(flat_count).astype(jnp.int32)
    flat_selected = selected_here.reshape(flat_count).astype(jnp.bool_)

    sector_ids = jnp.arange(n_local_sectors, dtype=jnp.int32)
    sector_onehot = jnp.logical_and(
        flat_selected[:, None],
        flat_sector[:, None] == sector_ids[None, :])
    attempted_fill = sector_onehot.astype(jnp.int32).sum(axis=0)
    rank_by_sector = jnp.cumsum(
        sector_onehot.astype(jnp.int32), axis=0) - 1
    rank = jnp.sum(
        jnp.where(sector_onehot, rank_by_sector, 0), axis=1).astype(jnp.int32)
    in_capacity = jnp.logical_and(
        flat_selected,
        rank < jnp.asarray(bucket_capacity, dtype=jnp.int32))
    overflow = jnp.logical_and(flat_selected, jnp.logical_not(in_capacity))

    safe_sector = jnp.where(
        in_capacity, flat_sector,
        jnp.asarray(n_local_sectors, dtype=jnp.int32))
    safe_rank = jnp.where(
        in_capacity, rank,
        jnp.asarray(bucket_capacity, dtype=jnp.int32))
    pair_buckets = jnp.zeros(
        (n_local_sectors, bucket_capacity), dtype=jnp.int32).at[
            safe_sector, safe_rank
        ].set(flat_pair_ids, mode='drop')
    bucket_valid = jnp.zeros(
        (n_local_sectors, bucket_capacity), dtype=jnp.bool_).at[
            safe_sector, safe_rank
        ].set(in_capacity, mode='drop')
    bucket_fill = bucket_valid.astype(jnp.int32).sum(axis=1)
    return (
        pair_buckets,
        bucket_valid,
        overflow.astype(jnp.int32).sum(),
        bucket_fill.max(),
        bucket_fill.astype(jnp.float32).mean(),
        attempted_fill,
    )


def _v4168_percentile_from_sorted(sorted_values, pct):
    n = int(sorted_values.shape[0])
    if n <= 1:
        return sorted_values[0]
    idx = int(math.ceil((float(pct) / 100.0) * float(n))) - 1
    idx = min(max(idx, 0), n - 1)
    return sorted_values[idx]


def _v4168_sector_core_runtime_diag(bucket_fill, overflow_count,
                                    global_n_sectors,
                                    selected_sector_count,
                                    selected_real_ops, total_real_ops):
    bucket_fill_f = bucket_fill.astype(jnp.float32)
    global_sector_fill_sum = jax.lax.psum(bucket_fill_f.sum(), 'model')
    global_sector_fill_max = jax.lax.pmax(
        jax.lax.stop_gradient(bucket_fill.max()).astype(jnp.float32),
        'model')
    return jnp.asarray((
        global_sector_fill_sum / jnp.float32(max(global_n_sectors, 1)),
        global_sector_fill_max,
        overflow_count.astype(jnp.float32),
        selected_sector_count / jnp.float32(max(global_n_sectors, 1)),
        selected_real_ops / jnp.maximum(total_real_ops, 1.0),
    ), dtype=jnp.float32)


def _v4168_sector_benchmark_runtime_diag(bucket_fill, attempted_fill,
                                         overflow_count,
                                         bucket_capacity, pair_span, topk,
                                         global_n_sectors, sector_size,
                                         selected_sector_count,
                                         selected_real_ops, total_real_ops):
    """Replicated runtime diagnostics for exact sector bucket execution."""
    bucket_fill_f = bucket_fill.astype(jnp.float32)
    global_bucket_fill = jax.lax.all_gather(
        bucket_fill_f, 'model', axis=0, tiled=True)
    sorted_fill = jnp.sort(global_bucket_fill)
    bucket_fill_mean = global_bucket_fill.mean()
    bucket_fill_p50 = _v4168_percentile_from_sorted(sorted_fill, 50)
    bucket_fill_p90 = _v4168_percentile_from_sorted(sorted_fill, 90)
    bucket_fill_p95 = _v4168_percentile_from_sorted(sorted_fill, 95)
    bucket_fill_p99 = _v4168_percentile_from_sorted(sorted_fill, 99)
    bucket_fill_max = sorted_fill[-1]
    attempted_fill_f = attempted_fill.astype(jnp.float32)
    global_attempted_fill = jax.lax.all_gather(
        attempted_fill_f, 'model', axis=0, tiled=True)
    sorted_attempted_fill = jnp.sort(global_attempted_fill)
    attempted_fill_mean = global_attempted_fill.mean()
    attempted_fill_p50 = _v4168_percentile_from_sorted(
        sorted_attempted_fill, 50)
    attempted_fill_p95 = _v4168_percentile_from_sorted(
        sorted_attempted_fill, 95)
    attempted_fill_p99 = _v4168_percentile_from_sorted(
        sorted_attempted_fill, 99)
    attempted_fill_max = sorted_attempted_fill[-1]

    bucket_capacity_f = jnp.asarray(bucket_capacity, dtype=jnp.float32)
    expected_pair_count = jnp.asarray(
        max(1, int(pair_span) * int(topk)), dtype=jnp.float32)
    executed_pair_count = global_bucket_fill.sum()
    overflow_count = overflow_count.astype(jnp.float32)
    overflow_frac = overflow_count / expected_pair_count
    executed_selected_pair_frac = executed_pair_count / expected_pair_count

    selected_sector_frac = (
        selected_sector_count
        / jnp.asarray(max(global_n_sectors, 1), dtype=jnp.float32))
    effective_operator_frac = selected_real_ops / jnp.maximum(total_real_ops, 1.0)

    per_token_selected_ops = jnp.asarray(
        int(topk) * int(sector_size), dtype=jnp.float32)
    per_token_operator_frac = per_token_selected_ops / jnp.maximum(
        total_real_ops, 1.0)
    dense_work = jnp.maximum(
        jnp.asarray(max(1, int(pair_span)), dtype=jnp.float32)
        * jnp.maximum(total_real_ops, 1.0),
        1.0)
    semantic_work_frac = (
        expected_pair_count * jnp.asarray(sector_size, dtype=jnp.float32)
        / dense_work)
    padded_pair_slots = jnp.asarray(
        max(1, int(global_n_sectors) * int(bucket_capacity)),
        dtype=jnp.float32)
    padded_work_frac = (
        padded_pair_slots * jnp.asarray(sector_size, dtype=jnp.float32)
        / dense_work)
    hot_sector_skew = bucket_fill_p99 / jnp.maximum(
        bucket_fill_mean, jnp.float32(1.0e-6))
    required_capacity_no_overflow = attempted_fill_max
    capacity_shortfall = (
        attempted_fill_max / jnp.maximum(bucket_capacity_f, 1.0))

    return jnp.asarray((
        bucket_fill_mean,
        bucket_fill_p50,
        bucket_fill_p90,
        bucket_fill_p95,
        bucket_fill_p99,
        bucket_fill_max,
        overflow_count,
        overflow_frac,
        bucket_capacity_f,
        bucket_fill_p50 / jnp.maximum(bucket_capacity_f, 1.0),
        bucket_fill_p95 / jnp.maximum(bucket_capacity_f, 1.0),
        bucket_fill_p99 / jnp.maximum(bucket_capacity_f, 1.0),
        bucket_fill_max / jnp.maximum(bucket_capacity_f, 1.0),
        expected_pair_count,
        executed_pair_count,
        executed_selected_pair_frac,
        selected_sector_frac,
        effective_operator_frac,
        jnp.asarray(topk, dtype=jnp.float32),
        per_token_selected_ops,
        per_token_operator_frac,
        semantic_work_frac,
        padded_work_frac,
        hot_sector_skew,
        attempted_fill_mean,
        attempted_fill_p50,
        attempted_fill_p95,
        attempted_fill_p99,
        attempted_fill_max,
        required_capacity_no_overflow,
        bucket_capacity_f,
        capacity_shortfall,
    ), dtype=jnp.float32)


def _sector_runtime_metric_dict(prefix, diag):
    out = {
        f'{prefix}/{name}': diag[i]
        for i, name in enumerate(BENCHMARK_SECTOR_RUNTIME_DIAG_NAMES)
    }
    out[f'{prefix}/sector_fill_mean'] = (
        diag[BENCHMARK_SECTOR_BUCKET_FILL_MEAN])
    out[f'{prefix}/sector_fill_max'] = (
        diag[BENCHMARK_SECTOR_BUCKET_FILL_MAX])
    out[f'{prefix}/sector_overflow_count'] = (
        diag[BENCHMARK_SECTOR_OVERFLOW_COUNT])
    out[f'{prefix}/selected_sector_frac'] = (
        diag[BENCHMARK_SECTOR_BATCH_UNION_SELECTED_SECTOR_FRAC])
    out[f'{prefix}/effective_operator_frac'] = (
        diag[BENCHMARK_SECTOR_BATCH_UNION_EFFECTIVE_OPERATOR_FRAC])
    return out


def _sector_core_runtime_metric_dict(prefix, diag):
    return {
        f'{prefix}/sector_fill_mean': diag[SECTOR_FILL_MEAN],
        f'{prefix}/sector_fill_max': diag[SECTOR_FILL_MAX],
        f'{prefix}/sector_overflow_count': diag[SECTOR_OVERFLOW_COUNT],
        f'{prefix}/selected_sector_frac':
            diag[SECTOR_SELECTED_SECTOR_FRAC],
        f'{prefix}/effective_operator_frac':
            diag[SECTOR_EFFECTIVE_OPERATOR_FRAC],
    }


def _sector_benchmark_runtime_metric_dict(prefix, diag):
    out = _sector_runtime_metric_dict(prefix, diag)
    return out


def _opspace_runtime_metric_dict(prefix, diag):
    return {
        f'{prefix}/{name}': diag[i]
        for i, name in enumerate(OPSPACE_RUNTIME_DIAG_NAMES)
    }


def _opspace_aux_loss(aux):
    if isinstance(aux, dict):
        return aux.get('aux_loss', jnp.float32(0.0))
    return aux


def _opspace_final_runtime_metric_dict(prefix, aux):
    if not isinstance(aux, dict) or 'final_diag' not in aux:
        return {}
    diag = aux['final_diag']
    out = {
        f'{prefix}/final/{name}': diag[i]
        for i, name in enumerate(OPSPACE_FINAL_RUNTIME_DIAG_NAMES)
    }
    for name in (
            'semantic_drop_frac',
            'assignment_collision_count',
            'pallas_backend_available',
            'output_tiled_backend_active',
            'region_count',
            'regions_per_owner',
            'blocks_per_region',
            'operators_per_block',
            'visible_regions',
            'visible_blocks_per_region',
            'region_capacity',
            'region_capacity_factor',
            'region_load_mean',
            'region_load_p50',
            'region_load_p95',
            'region_load_p99',
            'region_load_max',
            'region_capacity_util_p50',
            'region_capacity_util_p95',
            'region_capacity_util_p99',
            'region_capacity_util_max',
            'region_unresolved_count',
            'region_unresolved_frac',
            'region_reroute_frac',
            'region_low_regret_spill_frac',
            'region_high_regret_spill_frac',
            'region_score_best_mean',
            'region_score_chosen_mean',
            'region_score_loss_mean',
            'region_score_loss_p95',
            'region_score_loss_max',
            'block_capacity',
            'block_capacity_factor',
            'block_load_mean',
            'block_load_p50',
            'block_load_p95',
            'block_load_p99',
            'block_load_max',
            'block_capacity_util_p50',
            'block_capacity_util_p95',
            'block_capacity_util_p99',
            'block_capacity_util_max',
            'block_unresolved_count',
            'block_unresolved_frac',
            'block_reroute_frac',
            'block_low_regret_spill_frac',
            'block_high_regret_spill_frac',
            'block_score_best_mean',
            'block_score_chosen_mean',
            'block_score_loss_mean',
            'block_score_loss_p95',
            'block_score_loss_max',
            'visible_blocks_per_token',
            'visible_ops_per_token',
            'physical_visible_ops_per_token',
            'compute_frac_vs_dense',
            'logical_compute_frac_vs_dense',
            'physical_compute_frac_vs_dense',
            'region_all_requests_processed',
            'block_all_requests_processed',
            'selected_requests',
            'processed_requests',
            'all_processed',
            'no_nan',
            'region_load_p99_over_mean',
            'block_load_p99_over_mean',
            'block_compactness_mean',
            'block_radius_mean',
            'block_radius_max'):
        final_key = f'{prefix}/final/{name}'
        if final_key in out:
            out[f'{prefix}/{name}'] = out[final_key]
    return out


def _v4168_block_execution_upper_bound(block_score_ub, tau, boundary_scale,
                                       boundary_power):
    """DirectTau execution upper bound used to open Blocks."""
    margin = block_score_ub - tau
    admission = _boundary_soft_weight_from_margin(
        margin, boundary_scale, boundary_power)
    drive = _drive_from_margin(margin, tau, boundary_scale)
    return admission * drive


def _v4168_block_sparse_blocks(op_key_local, read_local, write_local,
                               block_size):
    """Contiguous local operator Blocks plus current op-key metadata."""
    N_local = int(op_key_local.shape[0])
    N_pad = ((N_local + block_size - 1) // block_size) * block_size
    pad_n = N_pad - N_local
    n_blocks = N_pad // block_size

    op_key_padded = jnp.pad(op_key_local, ((0, pad_n), (0, 0)))
    read_padded = jnp.pad(read_local, ((0, pad_n), (0, 0)))
    write_padded = jnp.pad(write_local, ((0, pad_n), (0, 0)))
    valid_padded = jnp.arange(N_pad) < N_local

    op_key_dir = _forward_unit_direction(
        op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    read_dir = _forward_unit_direction(
        read_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    write_dir = _forward_unit_direction(
        write_padded.astype(jnp.bfloat16).astype(jnp.float32)
    ).astype(jnp.bfloat16)

    op_key_blocks = op_key_dir.reshape(
        n_blocks, block_size, op_key_dir.shape[-1])
    read_blocks = read_dir.reshape(n_blocks, block_size, read_dir.shape[-1])
    write_blocks = write_dir.reshape(
        n_blocks, block_size, write_dir.shape[-1])
    valid_blocks = valid_padded.reshape(n_blocks, block_size)

    valid_f = valid_blocks.astype(jnp.float32)[..., None]
    count = jnp.maximum(valid_f.sum(axis=1), jnp.float32(1.0))
    center_raw = (
        op_key_blocks.astype(jnp.float32) * valid_f
    ).sum(axis=1) / count
    block_center = _forward_unit_direction(center_raw).astype(jnp.bfloat16)
    delta = (
        op_key_blocks.astype(jnp.float32)
        - block_center[:, None, :].astype(jnp.float32))
    spread = jnp.linalg.norm(delta, axis=-1)
    spread = jnp.where(valid_blocks, spread, 0.0)
    block_radius = spread.max(axis=1).astype(jnp.float32)
    return (op_key_blocks, read_blocks, write_blocks, valid_blocks,
            block_center, block_radius, n_blocks)


def _opspace_piecewise(tokens, start_tokens, mid_tokens, end_tokens,
                       v_start, v_mid, v_end):
    tokens = jnp.asarray(tokens, dtype=jnp.float32)
    start_tokens = jnp.asarray(start_tokens, dtype=jnp.float32)
    mid_tokens = jnp.maximum(
        jnp.asarray(mid_tokens, dtype=jnp.float32), start_tokens + 1.0)
    end_tokens = jnp.maximum(
        jnp.asarray(end_tokens, dtype=jnp.float32), mid_tokens + 1.0)
    first = jnp.clip(
        (tokens - start_tokens) / (mid_tokens - start_tokens), 0.0, 1.0)
    second = jnp.clip(
        (tokens - mid_tokens) / (end_tokens - mid_tokens), 0.0, 1.0)
    v_start = jnp.asarray(v_start, dtype=jnp.float32)
    v_mid = jnp.asarray(v_mid, dtype=jnp.float32)
    v_end = jnp.asarray(v_end, dtype=jnp.float32)
    return jnp.where(
        tokens <= mid_tokens,
        v_start + (v_mid - v_start) * first,
        v_mid + (v_end - v_mid) * second)


def _opspace_lb_weight(tokens, peak_weight, warmup_tokens,
                       peak_start_tokens, peak_end_tokens,
                       decay_end_tokens, final_weight_frac):
    tokens = jnp.asarray(tokens, dtype=jnp.float32)
    peak_weight = jnp.asarray(peak_weight, dtype=jnp.float32)
    warmup_tokens = jnp.asarray(warmup_tokens, dtype=jnp.float32)
    peak_start_tokens = jnp.maximum(
        jnp.asarray(peak_start_tokens, dtype=jnp.float32),
        warmup_tokens + 1.0)
    peak_end_tokens = jnp.maximum(
        jnp.asarray(peak_end_tokens, dtype=jnp.float32),
        peak_start_tokens + 1.0)
    decay_end_tokens = jnp.maximum(
        jnp.asarray(decay_end_tokens, dtype=jnp.float32),
        peak_end_tokens + 1.0)
    final_weight = peak_weight * jnp.asarray(
        final_weight_frac, dtype=jnp.float32)
    ramp = jnp.clip(
        (tokens - warmup_tokens) / (peak_start_tokens - warmup_tokens),
        0.0, 1.0)
    decay = jnp.clip(
        (tokens - peak_end_tokens) / (decay_end_tokens - peak_end_tokens),
        0.0, 1.0)
    rising = peak_weight * ramp
    falling = peak_weight + (final_weight - peak_weight) * decay
    return jnp.where(
        tokens < warmup_tokens,
        jnp.float32(0.0),
        jnp.where(tokens <= peak_end_tokens, rising, falling))


def _opspace_capped_simplex_gate(selected_scores, candidate_mask, K_budget,
                                 T_gate):
    candidate_mask_f = candidate_mask.astype(jnp.float32)
    m_eff = jnp.maximum(candidate_mask_f.sum(axis=-1, keepdims=True), 1.0)
    K_budget = jnp.clip(
        jnp.asarray(K_budget, dtype=jnp.float32), 1.0, m_eff)
    T_gate = jnp.maximum(jnp.asarray(T_gate, dtype=jnp.float32), 1.0e-4)
    q = selected_scores.astype(jnp.float32) / T_gate
    q = jnp.where(candidate_mask, q, -1.0e6)
    lo = jnp.min(jnp.where(candidate_mask, q, 1.0e6), axis=-1, keepdims=True) - 1.0
    hi = jnp.max(q, axis=-1, keepdims=True) + 1.0

    def body(_i, state):
        lo, hi = state
        mid = (lo + hi) * 0.5
        gate = jnp.clip(q - mid, 0.0, 1.0) * candidate_mask_f
        too_much = gate.sum(axis=-1, keepdims=True) > K_budget
        lo = jnp.where(too_much, mid, lo)
        hi = jnp.where(too_much, hi, mid)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 20, body, (lo, hi))
    lam = (lo + hi) * 0.5
    gate = jnp.clip(q - lam, 0.0, 1.0) * candidate_mask_f
    all_one_gate = candidate_mask_f
    return jnp.where(K_budget >= m_eff - 1.0e-5, all_one_gate, gate)


def _opspace_balanced_slot_layout_np(n_valid, total_tiles, tile_size):
    """Map compact valid rows into fixed tiles with invalid slots spread out."""
    n_valid = int(n_valid)
    total_tiles = max(1, int(total_tiles))
    tile_size = max(1, int(tile_size))
    total_slots = total_tiles * tile_size
    if n_valid > total_slots:
        raise ValueError(
            f"operation-space layout has {n_valid} valid rows but only "
            f"{total_slots} padded slots")
    counts = np.full((total_tiles,), n_valid // total_tiles, dtype=np.int32)
    counts[:n_valid % total_tiles] += 1
    if np.any(counts > tile_size):
        raise ValueError(
            "operation-space balanced padding requires tile_size large enough "
            f"for max valid count {int(counts.max())}")
    slot_to_row = np.zeros((total_slots,), dtype=np.int32)
    valid = np.zeros((total_slots,), dtype=np.bool_)
    row = 0
    for tile_i, count in enumerate(counts.tolist()):
        start = tile_i * tile_size
        stop = start + int(count)
        if count > 0:
            slot_to_row[start:stop] = np.arange(
                row, row + int(count), dtype=np.int32)
            valid[start:stop] = True
        row += int(count)
    return slot_to_row, valid, counts


def _opspace_model_axis_layout(mesh, lanes, tiles_per_lane, tile_size):
    # Mesh interpretation:
    # The mesh is data x model. The data axis shards token rows. The model
    # axis owns operation regions inside each data group. A region is a
    # communication boundary; a block is a local dense RW execution unit.
    lanes = max(1, int(lanes))
    tiles_per_lane = max(1, int(tiles_per_lane))
    tile_size = max(1, int(tile_size))
    try:
        model_axis_size = int(mesh.shape.get('model', 1))
    except AttributeError:
        model_axis_size = int(dict(getattr(mesh, 'shape', {})).get('model', 1))
    model_axis_size = max(1, model_axis_size)
    if lanes % model_axis_size != 0:
        raise ValueError(
            "v4168 Region-Block operation-space requires num_regions "
            "divisible by mesh_model; got num_regions="
            f"{lanes}, mesh_model={model_axis_size}")
    local_lanes = lanes // model_axis_size
    local_tile_count = local_lanes * tiles_per_lane
    return {
        'lanes': lanes,
        'tiles_per_lane': tiles_per_lane,
        'tile_size': tile_size,
        'model_axis_size': model_axis_size,
        'local_lanes': local_lanes,
        'local_tile_count': local_tile_count,
        'global_tile_count': lanes * tiles_per_lane,
        'local_padded_ops': local_tile_count * tile_size,
    }


def _opspace_region_block_layout(
        mesh, *, num_regions, blocks_per_region, operators_per_block,
        n_valid_operators):
    """Region/Block/Operator Atlas layout on the model axis."""
    num_regions = max(1, int(num_regions))
    blocks_per_region = max(1, int(blocks_per_region))
    operators_per_block = max(1, int(operators_per_block))
    n_valid_operators = max(0, int(n_valid_operators))
    try:
        mesh_model = int(mesh.shape.get('model', 1))
    except AttributeError:
        mesh_model = int(dict(getattr(mesh, 'shape', {})).get('model', 1))
    mesh_model = max(1, mesh_model)
    if num_regions % mesh_model != 0:
        raise ValueError(
            "v4168 Region/Block operation-space requires num_regions "
            "divisible by mesh_model; got num_regions="
            f"{num_regions}, mesh_model={mesh_model}")
    local_regions = num_regions // mesh_model
    operators_per_region = blocks_per_region * operators_per_block
    global_operator_capacity = num_regions * operators_per_region
    local_operator_capacity = local_regions * operators_per_region
    if n_valid_operators > global_operator_capacity:
        raise ValueError(
            "v4168 Region/Block operation-space capacity is smaller than "
            f"the pool: capacity={global_operator_capacity}, "
            f"n_valid_operators={n_valid_operators}")
    return {
        "num_regions": num_regions,
        "local_regions": local_regions,
        "blocks_per_region": blocks_per_region,
        "operators_per_block": operators_per_block,
        "operators_per_region": operators_per_region,
        "global_operator_capacity": global_operator_capacity,
        "local_operator_capacity": local_operator_capacity,
        "invalid_padded_operators": (
            global_operator_capacity - n_valid_operators),
    }


def _opspace_tile_drift_percentiles_np(keys, row_to_tile, tile_means):
    if int(keys.shape[0]) == 0:
        return 0.0, 0.0, 0.0
    current_cos = np.sum(keys * tile_means[row_to_tile], axis=-1)
    drift = (1.0 - current_cos).astype(np.float32)
    return (
        float(np.percentile(drift, 50)),
        float(np.percentile(drift, 95)),
        float(np.percentile(drift, 99)),
    )


def _plan_operation_space_threshold_repack(
        op_keys, lanes, tiles_per_lane, tile_size, max_swaps_per_repack,
        std_mult=2.0, min_assignment_gain=0.03, min_swap_gain=0.0):
    """Plan deterministic positive-gain valid-row swaps for one pool."""
    lanes = max(1, int(lanes))
    tiles_per_lane = max(1, int(tiles_per_lane))
    tile_size = max(1, int(tile_size))
    total_tiles = lanes * tiles_per_lane
    max_swaps_per_repack = max(0, int(max_swaps_per_repack))
    keys = _np_unit_direction(op_keys)
    n_valid = int(keys.shape[0])
    perm = np.arange(n_valid, dtype=np.int32)
    if n_valid == 0 or max_swaps_per_repack <= 0:
        return perm, {
            'layers_touched': 0.0,
            'candidates': 0.0,
            'swaps': 0.0,
            'mean_assignment_gain': 0.0,
            'mean_swap_gain': 0.0,
            'max_swap_gain': 0.0,
            'drift_p50_before': 0.0,
            'drift_p95_before': 0.0,
            'drift_p99_before': 0.0,
            'drift_p95_after': 0.0,
            'drift_p99_after': 0.0,
            'perm_checksum': float(_hardware_perm_checksum(perm)),
        }

    slot_to_row, slot_valid, counts = _opspace_balanced_slot_layout_np(
        n_valid, total_tiles, tile_size)
    row_to_tile = np.zeros((n_valid,), dtype=np.int32)
    tile_rows = []
    for tile_i in range(total_tiles):
        start = tile_i * tile_size
        stop = start + tile_size
        rows = slot_to_row[start:stop][slot_valid[start:stop]]
        rows = np.asarray(rows, dtype=np.int32)
        tile_rows.append(rows)
        row_to_tile[rows] = tile_i

    sums = np.zeros((total_tiles, keys.shape[-1]), dtype=np.float32)
    for tile_i, rows in enumerate(tile_rows):
        if rows.size:
            sums[tile_i] = keys[rows].sum(axis=0)
    tile_means = _np_unit_direction(sums)
    empty = counts <= 0
    if np.any(empty):
        tile_means[empty] = 0.0

    current_cos = np.sum(keys * tile_means[row_to_tile], axis=-1)
    drift = (1.0 - current_cos).astype(np.float32)
    thresholds = np.zeros((total_tiles,), dtype=np.float32)
    for tile_i, rows in enumerate(tile_rows):
        if rows.size:
            d = drift[rows]
            thresholds[tile_i] = (
                float(np.mean(d)) + float(std_mult) * float(np.std(d)))
        else:
            thresholds[tile_i] = np.inf

    all_cos = keys @ tile_means.T
    best_tile = np.argmax(all_cos, axis=-1).astype(np.int32)
    best_cos = all_cos[np.arange(n_valid), best_tile]
    assignment_gain = (best_cos - current_cos).astype(np.float32)
    candidate_mask = (
        (drift > thresholds[row_to_tile])
        & (assignment_gain > float(min_assignment_gain))
        & (best_tile != row_to_tile))
    candidates = np.flatnonzero(candidate_mask).astype(np.int32)
    candidate_order = sorted(
        candidates.tolist(),
        key=lambda row: (-float(assignment_gain[row]), int(row),
                         int(best_tile[row])))

    proposals = []
    for row_i in candidate_order:
        src = int(row_to_tile[row_i])
        dst = int(best_tile[row_i])
        rows_b = tile_rows[dst]
        if rows_b.size == 0:
            continue
        i_to_src = float(all_cos[row_i, src])
        i_to_dst = float(all_cos[row_i, dst])
        victim_best = None
        victim_gain = None
        for row_j in rows_b.tolist():
            if int(row_j) == int(row_i):
                continue
            swap_gain = (
                i_to_dst + float(all_cos[row_j, src])
                - i_to_src - float(all_cos[row_j, dst]))
            if swap_gain <= float(min_swap_gain):
                continue
            key = (float(swap_gain), -int(row_i), -int(row_j))
            if victim_best is None or key > victim_best:
                victim_best = key
                victim_gain = float(swap_gain)
                victim_row = int(row_j)
        if victim_best is None:
            continue
        proposals.append((
            -float(victim_gain),
            -float(assignment_gain[row_i]),
            int(row_i),
            int(victim_row),
            src,
            dst,
            float(victim_gain),
        ))
    proposals.sort()

    used_rows = set()
    accepted = []
    for _neg_swap, _neg_assign, row_i, row_j, src, dst, swap_gain in proposals:
        if row_i in used_rows or row_j in used_rows:
            continue
        used_rows.add(row_i)
        used_rows.add(row_j)
        accepted.append((row_i, row_j, src, dst, swap_gain))
        if len(accepted) >= max_swaps_per_repack:
            break

    for row_i, row_j, _src, _dst, _swap_gain in accepted:
        perm[row_i] = row_j
        perm[row_j] = row_i
    perm = _validate_hardware_permutation_np(
        perm, n_valid, pool='operation_space')

    new_keys = keys[perm]
    new_sums = np.zeros_like(sums)
    for tile_i, rows in enumerate(tile_rows):
        if rows.size:
            new_sums[tile_i] = new_keys[rows].sum(axis=0)
    new_tile_means = _np_unit_direction(new_sums)
    if np.any(empty):
        new_tile_means[empty] = 0.0
    p50_before, p95_before, p99_before = _opspace_tile_drift_percentiles_np(
        keys, row_to_tile, tile_means)
    _p50_after, p95_after, p99_after = _opspace_tile_drift_percentiles_np(
        new_keys, row_to_tile, new_tile_means)
    assignment_vals = [
        float(assignment_gain[row_i]) for row_i, *_rest in accepted
    ]
    swap_vals = [float(item[-1]) for item in accepted]
    return perm, {
        'layers_touched': 1.0,
        'candidates': float(int(candidates.size)),
        'swaps': float(len(accepted)),
        'mean_assignment_gain': (
            float(np.mean(assignment_vals)) if assignment_vals else 0.0),
        'mean_swap_gain': float(np.mean(swap_vals)) if swap_vals else 0.0,
        'max_swap_gain': float(np.max(swap_vals)) if swap_vals else 0.0,
        'drift_p50_before': p50_before,
        'drift_p95_before': p95_before,
        'drift_p99_before': p99_before,
        'drift_p95_after': p95_after,
        'drift_p99_after': p99_after,
        'perm_checksum': float(_hardware_perm_checksum(perm)),
    }


def make_sharded_srw_factorized_lane_mean_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=4,
        tiles_per_lane=48,
        padded_ops=0,
        tile_size=64,
        budget_start_tokens=0.0,
        budget_mid_tokens=15000000000.0,
        budget_end_tokens=30000000000.0,
        K_start=3.8,
        K_mid=3.0,
        K_final=2.0,
        T_gate_start=1.0,
        T_gate_mid=0.5,
        T_gate_final=0.18,
        T_lb_start=1.0,
        T_lb_mid=0.5,
        T_lb_final=0.20,
        lb_enabled=True,
        lb_capacity_factor=1.35,
        lb_device_capacity_factor=1.25,
        lb_tile_weight_peak=0.01,
        lb_device_weight_peak=0.003,
        lb_warmup_tokens=3000000000.0,
        lb_peak_start_tokens=8000000000.0,
        lb_peak_end_tokens=22000000000.0,
        lb_decay_end_tokens=30000000000.0,
        lb_final_weight_frac=0.10):
    """Create v4168 factorized lane/tile-mean SRW for V/RST pools."""
    del max_chunk_size, dead_exposure_target
    _tile_size = int(tile_size)
    _tiles_per_lane = max(1, int(tiles_per_lane))
    _lanes = max(1, int(lanes))
    _padded_ops = max(0, int(padded_ops))
    _max_exec_lanes = max(
        1, min(_lanes, int(math.ceil(max(K_start, K_mid, K_final)))))
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    try:
        _model_axis_size = int(mesh.shape.get('model', 1))
    except AttributeError:
        _model_axis_size = int(dict(getattr(mesh, 'shape', {})).get('model', 1))
    _model_axis_size = max(1, _model_axis_size)
    _configured_local_padded_ops = (
        _padded_ops // _model_axis_size if _padded_ops > 0 else 0)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None), P(), P()),
             check_rep=False)
    def fused_gate_srw_factorized_lane_mean_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens):
        del soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power_final
        B, S, D = x.shape
        N_local = int(op_key_local.shape[0])
        min_local_padded_ops = (
            (N_local + _tile_size - 1) // _tile_size) * _tile_size
        local_padded_ops = max(
            _configured_local_padded_ops, min_local_padded_ops)
        local_padded_ops = (
            (local_padded_ops + _tile_size - 1) // _tile_size) * _tile_size
        local_tile_count = max(1, local_padded_ops // _tile_size)
        local_lanes = max(
            1, local_tile_count // _tiles_per_lane)
        local_tile_count = local_lanes * _tiles_per_lane
        local_padded_ops = local_tile_count * _tile_size
        global_lanes = local_lanes * _model_axis_size
        max_exec = min(_max_exec_lanes, global_lanes)
        pad_n = local_padded_ops - N_local
        model_axis_index = jax.lax.axis_index('model')

        training_tokens = jnp.asarray(training_tokens, dtype=jnp.float32)
        K_budget = _opspace_piecewise(
            training_tokens, budget_start_tokens, budget_mid_tokens,
            budget_end_tokens, K_start, K_mid, K_final)
        T_gate = _opspace_piecewise(
            training_tokens, budget_start_tokens, budget_mid_tokens,
            budget_end_tokens, T_gate_start, T_gate_mid, T_gate_final)
        T_lb = _opspace_piecewise(
            training_tokens, budget_start_tokens, budget_mid_tokens,
            budget_end_tokens, T_lb_start, T_lb_mid, T_lb_final)
        tile_weight = _opspace_lb_weight(
            training_tokens, lb_tile_weight_peak, lb_warmup_tokens,
            lb_peak_start_tokens, lb_peak_end_tokens, lb_decay_end_tokens,
            lb_final_weight_frac)
        device_weight = _opspace_lb_weight(
            training_tokens, lb_device_weight_peak, lb_warmup_tokens,
            lb_peak_start_tokens, lb_peak_end_tokens, lb_decay_end_tokens,
            lb_final_weight_frac)
        if not lb_enabled:
            tile_weight = jnp.float32(0.0)
            device_weight = jnp.float32(0.0)

        op_key_padded = jnp.pad(op_key_local, ((0, pad_n), (0, 0)))
        read_padded = jnp.pad(read_local, ((0, pad_n), (0, 0)))
        write_padded = jnp.pad(write_local, ((0, pad_n), (0, 0)))
        valid_padded = jnp.arange(local_padded_ops) < N_local

        op_key_dir = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)

        key_tiles = op_key_dir.reshape(
            local_lanes, _tiles_per_lane, _tile_size, op_key_dir.shape[-1])
        read_tiles = read_dir.reshape(
            local_lanes, _tiles_per_lane, _tile_size, read_dir.shape[-1])
        write_tiles = write_dir.reshape(
            local_lanes, _tiles_per_lane, _tile_size, write_dir.shape[-1])
        valid_tiles = valid_padded.reshape(
            local_lanes, _tiles_per_lane, _tile_size)

        valid_f = valid_tiles.astype(jnp.float32)[..., None]
        tile_count = jnp.maximum(valid_f.sum(axis=2), 1.0)
        tile_mean_raw = (
            key_tiles.astype(jnp.float32) * valid_f).sum(axis=2) / tile_count
        tile_mean = jax.lax.stop_gradient(
            _forward_unit_direction(tile_mean_raw)).astype(jnp.bfloat16)

        h_unit = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        scores_local = jnp.einsum(
            'bsd,ltd->bslt', h_unit, tile_mean).astype(jnp.float32)
        best_score_local = scores_local.max(axis=-1)
        best_tile_local = jnp.argmax(scores_local, axis=-1).astype(jnp.int32)
        global_best_score = jax.lax.all_gather(
            best_score_local, 'model', axis=2, tiled=True)
        global_best_tile = jax.lax.all_gather(
            best_tile_local, 'model', axis=2, tiled=True)
        selected_scores, selected_global_lanes = jax.lax.top_k(
            global_best_score, max_exec)
        selected_tile = jnp.take_along_axis(
            global_best_tile, selected_global_lanes, axis=2)

        M_exec = jnp.clip(
            jnp.ceil(K_budget).astype(jnp.int32), 1, max_exec)
        candidate_mask = (
            jnp.arange(max_exec, dtype=jnp.int32)[None, None, :] < M_exec)
        tile_gate = _opspace_capped_simplex_gate(
            selected_scores, candidate_mask, K_budget, T_gate)

        owner = selected_global_lanes // local_lanes
        selected_here = owner == model_axis_index
        local_lane = selected_global_lanes - model_axis_index * local_lanes
        safe_lane = jnp.clip(local_lane, 0, local_lanes - 1)
        flat_tile_id = safe_lane * _tiles_per_lane + selected_tile
        flat_key_tiles = key_tiles.reshape(
            local_tile_count, _tile_size, key_tiles.shape[-1])
        flat_read_tiles = read_tiles.reshape(
            local_tile_count, _tile_size, read_tiles.shape[-1])
        flat_write_tiles = write_tiles.reshape(
            local_tile_count, _tile_size, write_tiles.shape[-1])
        flat_valid_tiles = valid_tiles.reshape(local_tile_count, _tile_size)

        selected_key = flat_key_tiles[flat_tile_id]
        selected_read = flat_read_tiles[flat_tile_id]
        selected_write = flat_write_tiles[flat_tile_id]
        selected_valid = flat_valid_tiles[flat_tile_id]
        selected_valid = jnp.logical_and(
            selected_valid,
            jnp.logical_and(selected_here, candidate_mask)[..., None])
        selected_gate = tile_gate * selected_here.astype(jnp.float32)

        tau = _tau_from_param(raw_tau).astype(jnp.float32)
        rho = jnp.einsum(
            'bsd,bsmtd->bsmt', h_unit, selected_key).astype(jnp.float32)
        rho_compute = jnp.where(selected_valid, rho, tau[:, :, None, :])
        _, admission, _drive, execution_weight, _ = _compute_admission_drive(
            rho_compute, tau[:, :, None, :], soft_gate_boundary_power,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=_soft_gate_effective_active_eps,
            execution_prune_eps=execution_prune_eps)
        selected_gate_b = selected_gate[..., None]
        admission = jnp.where(
            selected_valid, admission * selected_gate_b, 0.0)
        execution_weight = jnp.where(
            selected_valid, execution_weight * selected_gate_b, 0.0)

        xr = jnp.einsum(
            'bsd,bsmtd->bsmt', x_bf, selected_read).astype(jnp.float32)
        weighted = execution_weight * xr
        raw_out = jnp.einsum(
            'bsmt,bsmtd->bsd',
            weighted.astype(jnp.bfloat16),
            selected_write).astype(jnp.float32)
        total_den_cost = admission.sum(axis=(2, 3), keepdims=False)[..., None]
        global_raw_out = jax.lax.psum(raw_out, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = global_raw_out / admission_den

        T_lb_safe = jnp.maximum(T_lb, jnp.float32(1.0e-4))
        p_tile_local = jax.nn.softmax(scores_local / T_lb_safe, axis=-1)
        lane_score_local = (
            T_lb_safe
            * jax.nn.logsumexp(scores_local / T_lb_safe, axis=-1))
        lane_score_global = jax.lax.all_gather(
            lane_score_local, 'model', axis=2, tiled=True)
        p_lane_global = jax.nn.softmax(lane_score_global / T_lb_safe, axis=-1)
        p_lane_local = jax.lax.dynamic_slice_in_dim(
            p_lane_global, model_axis_index * local_lanes,
            local_lanes, axis=2)
        tile_load_local = (
            K_budget
            * p_lane_local[..., None]
            * p_tile_local).sum(axis=(0, 1))
        tile_load_local = jax.lax.psum(tile_load_local, 'data')
        tile_load_global = jax.lax.all_gather(
            tile_load_local.reshape(-1), 'model', axis=0, tiled=True)
        total_tokens = jax.lax.psum(
            jnp.asarray(B * S, dtype=jnp.float32), 'data')
        avg_tile_load = (
            total_tokens * K_budget
            / jnp.maximum(jnp.asarray(tile_load_global.size, dtype=jnp.float32), 1.0))
        tile_capacity = avg_tile_load * jnp.asarray(
            lb_capacity_factor, dtype=jnp.float32)
        tile_over = jax.nn.relu(
            tile_load_global / jnp.maximum(tile_capacity, 1.0e-6) - 1.0)
        lb_tile = (tile_over * tile_over).mean()

        device_load = tile_load_global.reshape(_model_axis_size, -1).sum(axis=1)
        avg_device_load = device_load.mean()
        device_capacity = avg_device_load * jnp.asarray(
            lb_device_capacity_factor, dtype=jnp.float32)
        device_over = jax.nn.relu(
            device_load / jnp.maximum(device_capacity, 1.0e-6) - 1.0)
        lb_device = (device_over * device_over).mean()
        lb_total = tile_weight * lb_tile + device_weight * lb_device

        def mesh_mean(v):
            return jax.lax.pmean(jax.lax.pmean(v, 'data'), 'model')

        def mesh_sum(v):
            return jax.lax.psum(jax.lax.psum(v, 'data'), 'model')

        selected_mask_f = candidate_mask.astype(jnp.float32)
        gate_sum = tile_gate.sum(axis=-1)
        gate_tail = jnp.min(
            jnp.where(candidate_mask, tile_gate, 1.0), axis=-1)
        gate_sum_error = jnp.abs(gate_sum - jnp.minimum(K_budget, M_exec))
        active_ops_local = (
            admission >= _soft_gate_effective_active_eps).astype(
                jnp.float32).sum(axis=(2, 3))
        effective_ops_local = (
            selected_gate_b
            * selected_valid.astype(jnp.float32)).sum(axis=(2, 3))
        active_ops = jax.lax.psum(active_ops_local, 'model')
        effective_ops = jax.lax.psum(effective_ops_local, 'model')
        invalid_exec = jnp.where(
            selected_valid, 0.0, jnp.abs(execution_weight)).sum()
        invalid_exec = mesh_sum(invalid_exec)

        lane_ids = jnp.arange(global_lanes, dtype=jnp.int32)
        lane_hist = (
            (selected_global_lanes[..., None] == lane_ids)
            .astype(jnp.float32)
            * selected_mask_f[..., None]).sum(axis=(0, 1, 2))
        lane_hist = jax.lax.psum(lane_hist, 'data')
        lane_total = jnp.maximum(lane_hist.sum(), 1.0)
        lane_prob = lane_hist / lane_total
        lane_entropy = -jnp.sum(jnp.where(
            lane_prob > 0.0, lane_prob * jnp.log(lane_prob + 1.0e-8), 0.0))
        lane_top1_frac = lane_prob.max()

        sorted_tile_load = jnp.sort(tile_load_global)
        sorted_device_load = jnp.sort(device_load)
        tile_load_mean = jnp.maximum(tile_load_global.mean(), 1.0e-6)
        device_load_mean = jnp.maximum(device_load.mean(), 1.0e-6)
        tile_load_p95 = _v4168_percentile_from_sorted(sorted_tile_load, 95)
        tile_load_p99 = _v4168_percentile_from_sorted(sorted_tile_load, 99)
        device_load_p99 = _v4168_percentile_from_sorted(sorted_device_load, 99)

        no_nan_gate = jax.lax.pmin(
            jax.lax.pmin(
                jnp.all(jnp.isfinite(tile_gate)).astype(jnp.float32), 'data'),
            'model')
        no_nan_load = jax.lax.pmin(
            jax.lax.pmin(
                jnp.all(jnp.isfinite(tile_load_global)).astype(jnp.float32),
                'data'),
            'model')
        selected_count_ok = (
            jnp.abs(selected_mask_f.sum(axis=-1).mean()
                    - M_exec.astype(jnp.float32)) < 1.0e-5).astype(jnp.float32)
        layout_ok = jnp.asarray(
            1.0 if (
                local_padded_ops == local_lanes * _tiles_per_lane * _tile_size
                and local_tile_count > 0) else 0.0,
            dtype=jnp.float32)

        diag = jnp.asarray((
            jnp.float32(1.0),
            K_budget,
            M_exec.astype(jnp.float32),
            T_gate,
            T_lb,
            tile_weight * lb_tile,
            device_weight * lb_device,
            lb_total,
            M_exec.astype(jnp.float32),
            mesh_mean(gate_sum.mean()),
            mesh_mean(tile_gate.max(axis=-1).mean()),
            mesh_mean(gate_tail.mean()),
            tile_load_p95,
            tile_load_p99,
            tile_load_global.max() / tile_load_mean,
            device_load_p99 / device_load_mean,
            lane_entropy,
            lane_top1_frac,
            mesh_mean(gate_sum_error.mean()),
            no_nan_gate,
            no_nan_load,
            (invalid_exec <= 1.0e-8).astype(jnp.float32),
            selected_count_ok,
            layout_ok,
            mesh_mean(active_ops.mean()),
            mesh_mean(effective_ops.mean()),
        ), dtype=jnp.float32)
        return out.astype(jnp.float32), diag, lb_total.astype(jnp.float32)

    fused_gate_srw_factorized_lane_mean_minimal._v4168_accepts_training_tokens = True
    return fused_gate_srw_factorized_lane_mean_minimal


def make_sharded_srw_fixed_k4_repack_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        lanes=4,
        tiles_per_lane=48,
        padded_ops=0,
        tile_size=64,
        k_exec=4,
        token_chunk_size=256):
    """Create v4168 tau-free ReLU RST tile-major sparse grouped SRW."""
    del max_chunk_size, dead_exposure_target, padded_ops
    del soft_gate_effective_active_eps, admission_den_grad_scale
    layout = _opspace_model_axis_layout(
        mesh, lanes, tiles_per_lane, tile_size)
    _lanes = layout['lanes']
    _tiles_per_lane = layout['tiles_per_lane']
    _tile_size = layout['tile_size']
    _model_axis_size = layout['model_axis_size']
    _local_lanes = layout['local_lanes']
    _local_tile_count = layout['local_tile_count']
    _local_padded_ops = layout['local_padded_ops']
    _local_groups = _local_tile_count
    _k_exec = max(1, min(int(k_exec), _lanes))
    if _k_exec != _lanes:
        raise ValueError(
            "RST legacy fixed_k4_repack requires one best tile "
            f"per lane: k_exec must equal lanes; got k_exec={_k_exec}, "
            f"lanes={_lanes}")
    _token_chunk_size = max(1, int(token_chunk_size))
    _global_tile_count = layout['global_tile_count']
    _exec_slots = _k_exec * _tile_size
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None), P(), P()),
             check_rep=False)
    def fused_gate_srw_fixed_k4_repack_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens):
        del raw_tau, soft_gate_temperature, soft_gate_t_final
        del soft_gate_boundary_power, soft_gate_boundary_power_final
        del execution_prune_eps, training_tokens
        B, S, D = x.shape
        N_local = int(op_key_local.shape[0])
        if N_local > _local_padded_ops:
            raise ValueError(
                "operation-space local valid rows exceed fixed local tile "
                f"slots: rows={N_local}, slots={_local_padded_ops}")
        (slot_to_row_np, valid_slot_np, valid_counts_np) = (
            _opspace_balanced_slot_layout_np(
                N_local, _local_tile_count, _tile_size))
        slot_to_row = jnp.asarray(slot_to_row_np, dtype=jnp.int32)
        valid_slot = jnp.asarray(valid_slot_np, dtype=jnp.bool_)
        valid_counts_local = jnp.asarray(
            valid_counts_np.reshape(_local_lanes, _tiles_per_lane),
            dtype=jnp.float32)
        model_axis_index = jax.lax.axis_index('model')

        op_key_padded = op_key_local[slot_to_row]
        read_padded = read_local[slot_to_row]
        write_padded = write_local[slot_to_row]
        zero_key = jnp.zeros_like(op_key_padded)
        zero_read = jnp.zeros_like(read_padded)
        zero_write = jnp.zeros_like(write_padded)
        op_key_padded = jnp.where(valid_slot[:, None], op_key_padded, zero_key)
        read_padded = jnp.where(valid_slot[:, None], read_padded, zero_read)
        write_padded = jnp.where(valid_slot[:, None], write_padded, zero_write)

        op_key_dir = _forward_unit_direction(
            op_key_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        read_dir = _forward_unit_direction(
            read_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        write_dir = _forward_unit_direction(
            write_padded.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)

        key_tiles = op_key_dir.reshape(
            _local_lanes, _tiles_per_lane, _tile_size, op_key_dir.shape[-1])
        read_tiles = read_dir.reshape(
            _local_lanes, _tiles_per_lane, _tile_size, read_dir.shape[-1])
        write_tiles = write_dir.reshape(
            _local_lanes, _tiles_per_lane, _tile_size, write_dir.shape[-1])
        valid_tiles = valid_slot.reshape(
            _local_lanes, _tiles_per_lane, _tile_size)

        valid_f = valid_tiles.astype(jnp.float32)[..., None]
        tile_count = jnp.maximum(valid_f.sum(axis=2), 1.0)
        tile_mean_raw = (
            key_tiles.astype(jnp.float32) * valid_f).sum(axis=2) / tile_count
        tile_mean_local = jax.lax.stop_gradient(
            _forward_unit_direction(tile_mean_raw)).astype(jnp.bfloat16)

        T = int(B) * int(S)
        h_unit = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
        flat_h = h_unit.reshape(T, h_unit.shape[-1])
        flat_x = x.reshape(T, D).astype(jnp.bfloat16)

        scores_local = jnp.einsum(
            'td,lud->tlu', flat_h, tile_mean_local).astype(jnp.float32)
        best_tile_local = jnp.argmax(scores_local, axis=-1).astype(jnp.int32)

        request_count = T * _local_lanes
        request_token = jnp.repeat(
            jnp.arange(T, dtype=jnp.int32), _local_lanes)
        request_lane = jnp.tile(
            jnp.arange(_local_lanes, dtype=jnp.int32), T)
        request_tile = best_tile_local.reshape(request_count)
        request_group = request_lane * _tiles_per_lane + request_tile
        request_ids = jnp.arange(request_count, dtype=jnp.int32)
        group_counts = jnp.bincount(
            request_group, length=_local_groups).astype(jnp.int32)
        group_starts = jnp.cumsum(group_counts) - group_counts
        request_sort_key = request_group * jnp.asarray(
            request_count, dtype=jnp.int32) + request_ids
        sorted_request_ids = jnp.argsort(request_sort_key)
        sorted_request_ids = jnp.concatenate([
            sorted_request_ids,
            jnp.zeros((_token_chunk_size,), dtype=jnp.int32),
        ], axis=0)
        max_group_chunks = max(
            1, (request_count + _token_chunk_size - 1) // _token_chunk_size)
        token_chunk_size_i32 = jnp.asarray(_token_chunk_size, dtype=jnp.int32)
        chunk_offsets = jnp.arange(_token_chunk_size, dtype=jnp.int32)

        @jax.checkpoint
        def group_step(carry, group_i):
            (flat_raw_out, flat_gate_mass, flat_relu_gate_count,
             processed_count, valid_exec_slots_sum, padded_gate_mass,
             finite_acc) = carry
            lane_i = group_i // _tiles_per_lane
            tile_i = group_i - lane_i * _tiles_per_lane
            group_start = group_starts[group_i]
            group_count = group_counts[group_i]
            op_key_tile = key_tiles[lane_i, tile_i]
            read_tile = read_tiles[lane_i, tile_i]
            write_tile = write_tiles[lane_i, tile_i]
            valid_ops = valid_tiles[lane_i, tile_i]
            valid_ops_f = valid_ops.astype(jnp.float32)
            valid_count = valid_counts_local[lane_i, tile_i]

            def chunk_step(state, chunk_i):
                (flat_raw_out, flat_gate_mass, flat_relu_gate_count,
                 processed_count, valid_exec_slots_sum, padded_gate_mass,
                 finite_acc) = state
                offset = chunk_i * token_chunk_size_i32
                safe_offset = jnp.minimum(
                    offset, jnp.maximum(group_count - 1, 0))
                req_ids = jax.lax.dynamic_slice_in_dim(
                    sorted_request_ids, group_start + safe_offset,
                    _token_chunk_size, axis=0)
                valid_req = chunk_offsets < (group_count - offset)
                token_ids = request_token[req_ids]
                x_b = flat_x[token_ids]
                h_b = flat_h[token_ids]
                rho_raw = (
                    h_b @ op_key_tile.T).astype(jnp.float32)
                gate = jnp.square(jax.nn.relu(rho_raw)).astype(jnp.float32)
                gate = gate * valid_ops_f[None, :]
                gate = jnp.where(valid_req[:, None], gate, 0.0)
                xr = (x_b @ read_tile.T).astype(jnp.float32)
                raw_out = (
                    (gate * xr).astype(jnp.bfloat16) @ write_tile
                ).astype(jnp.float32)
                gate_mass = gate.sum(axis=1, keepdims=True)
                relu_gate_count = (gate > 0.0).astype(jnp.float32).sum(axis=1)
                valid_req_f = valid_req.astype(jnp.float32)
                raw_out = raw_out * valid_req_f[:, None]
                gate_mass = gate_mass * valid_req_f[:, None]
                relu_gate_count = relu_gate_count * valid_req_f
                flat_raw_out = flat_raw_out.at[token_ids].add(
                    raw_out, mode='drop')
                flat_gate_mass = flat_gate_mass.at[token_ids].add(
                    gate_mass, mode='drop')
                flat_relu_gate_count = flat_relu_gate_count.at[token_ids].add(
                    relu_gate_count, mode='drop')
                processed_count = processed_count + valid_req_f.sum()
                valid_exec_slots_sum = (
                    valid_exec_slots_sum + valid_count * valid_req_f.sum())
                finite = jnp.logical_and(
                    jnp.all(jnp.isfinite(gate)),
                    jnp.all(jnp.isfinite(raw_out)))
                finite_acc = jnp.minimum(
                    finite_acc, finite.astype(jnp.float32))
                return (
                    flat_raw_out, flat_gate_mass, flat_relu_gate_count,
                    processed_count, valid_exec_slots_sum,
                    padded_gate_mass, finite_acc), None

            (flat_raw_out, flat_gate_mass, flat_relu_gate_count,
             processed_count, valid_exec_slots_sum, padded_gate_mass,
             finite_acc) = jax.lax.scan(
                chunk_step,
                (flat_raw_out, flat_gate_mass, flat_relu_gate_count,
                 processed_count, valid_exec_slots_sum, padded_gate_mass,
                 finite_acc),
                jnp.arange(max_group_chunks, dtype=jnp.int32))[0]
            return (
                flat_raw_out, flat_gate_mass, flat_relu_gate_count,
                processed_count, valid_exec_slots_sum, padded_gate_mass,
                finite_acc), None

        init_carry = (
            jnp.zeros((T, D), dtype=jnp.float32),
            jnp.zeros((T, 1), dtype=jnp.float32),
            jnp.zeros((T,), dtype=jnp.float32),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        (flat_raw_out_local, flat_gate_mass_local, flat_relu_gate_count_local,
         processed_count_local, valid_exec_slots_sum_local,
         padded_gate_mass_local,
         chunk_no_nan) = jax.lax.scan(
            group_step,
            init_carry,
            jnp.arange(_local_groups, dtype=jnp.int32))[0]
        global_raw_out = jax.lax.psum(flat_raw_out_local, 'model')
        global_gate_mass = jax.lax.psum(flat_gate_mass_local, 'model')
        global_relu_gate_count = jax.lax.psum(
            flat_relu_gate_count_local, 'model')
        global_processed_count = jax.lax.psum(processed_count_local, 'model')
        global_valid_exec_slots_sum = jax.lax.psum(
            valid_exec_slots_sum_local, 'model')
        global_padded_gate_mass = jax.lax.psum(padded_gate_mass_local, 'model')
        gate_denominator = jnp.power(
            jnp.maximum(global_gate_mass, 1.0), _admission_den_power)
        flat_out = global_raw_out / gate_denominator
        out = flat_out.reshape(B, S, D)

        def mesh_mean(v):
            return jax.lax.pmean(jax.lax.pmean(v, 'data'), 'model')

        def data_sum(v):
            return jax.lax.pmean(jax.lax.psum(v, 'data'), 'model')

        token_count = jnp.maximum(
            data_sum(jnp.asarray(T, dtype=jnp.float32)), 1.0)
        score_sorted = jnp.sort(scores_local.reshape(-1))
        tile_score_p50 = mesh_mean(
            _v4168_percentile_from_sorted(score_sorted, 50))
        tile_score_p95 = mesh_mean(
            _v4168_percentile_from_sorted(score_sorted, 95))

        tile_hist_local = jax.lax.psum(
            group_counts.astype(jnp.float32), 'data')
        tile_hist = jax.lax.all_gather(
            tile_hist_local, 'model', axis=0, tiled=True)
        tile_hist_sorted = jnp.sort(tile_hist)
        tile_mean = jnp.maximum(tile_hist.mean(), 1.0e-6)
        selected_tile_top1_frac = tile_hist.max() / jnp.maximum(
            tile_hist.sum(), 1.0)
        selected_tile_p99 = _v4168_percentile_from_sorted(
            tile_hist_sorted, 99)
        selected_tile_p99_over_mean = selected_tile_p99 / tile_mean

        owner_load_local = jax.lax.psum(
            group_counts.astype(jnp.float32).sum(), 'data')
        owner_load = jax.lax.all_gather(
            owner_load_local.reshape((1,)), 'model', axis=0, tiled=True)
        owner_load_sorted = jnp.sort(owner_load)
        owner_load_mean = jnp.maximum(owner_load.mean(), 1.0e-6)
        owner_load_p99_over_mean = (
            _v4168_percentile_from_sorted(owner_load_sorted, 99)
            / owner_load_mean)

        lane_hist_local = group_counts.reshape(
            _local_lanes, _tiles_per_lane).sum(axis=1).astype(jnp.float32)
        lane_hist_local = jax.lax.psum(lane_hist_local, 'data')
        lane_hist = jax.lax.all_gather(
            lane_hist_local, 'model', axis=0, tiled=True)
        selected_lane_top1_frac = lane_hist.max() / jnp.maximum(
            lane_hist.sum(), 1.0)

        valid_exec_slots_mean = (
            data_sum(global_valid_exec_slots_sum) / token_count)
        relu_gate_count_mean = (
            data_sum(global_relu_gate_count.sum()) / token_count)
        gate_denominator_mean = (
            data_sum(gate_denominator.sum()) / token_count)
        gate_mass_mean = data_sum(global_gate_mass.sum()) / token_count
        padded_gate_mass = data_sum(global_padded_gate_mass)
        processed_request_count = data_sum(global_processed_count)
        selected_request_count = token_count * jnp.float32(_lanes)
        all_requests_processed = (
            processed_request_count >= selected_request_count - 0.5
        ).astype(jnp.float32)
        chunk_no_nan = jax.lax.pmin(jax.lax.pmin(chunk_no_nan, 'data'), 'model')
        score_no_nan = jnp.all(jnp.isfinite(scores_local)).astype(jnp.float32)
        score_no_nan = jax.lax.pmin(
            jax.lax.pmin(score_no_nan, 'data'), 'model')
        no_nan = jnp.minimum(score_no_nan, chunk_no_nan)
        padded_exec_slots_mean = (
            jnp.float32(_exec_slots) - valid_exec_slots_mean)
        layout_ok = jnp.asarray(
            1.0 if _local_padded_ops == _local_tile_count * _tile_size
            else 0.0,
            dtype=jnp.float32)

        diag = jnp.asarray((
            jnp.float32(1.0),
            jnp.float32(_k_exec),
            jnp.float32(_exec_slots),
            valid_exec_slots_mean,
            tile_score_p50,
            tile_score_p95,
            selected_lane_top1_frac,
            selected_tile_top1_frac,
            selected_tile_p99_over_mean,
            owner_load_p99_over_mean,
            relu_gate_count_mean,
            gate_denominator_mean,
            gate_mass_mean,
            relu_gate_count_mean,
            padded_exec_slots_mean,
            padded_gate_mass,
            no_nan,
            jnp.float32(1.0),
            selected_request_count,
            processed_request_count,
            all_requests_processed,
            layout_ok,
        ), dtype=jnp.float32)
        return out.astype(jnp.float32), diag, jnp.float32(0.0)

    fused_gate_srw_fixed_k4_repack_minimal._v4168_accepts_training_tokens = True
    return fused_gate_srw_fixed_k4_repack_minimal


make_sharded_srw_tau_free_relu_opspace_minimal = (
    make_sharded_srw_fixed_k4_repack_minimal)
make_sharded_srw_factorized_lane_mean_minimal = (
    make_sharded_srw_tau_free_relu_opspace_minimal)


def make_sharded_srw_block_sparse_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0):
    """Create v4168 single-route Block-sparse SRW returning only output."""
    del max_chunk_size, dead_exposure_target, block_margin
    _block_size, _top_blocks = _v4168_block_sparse_config(
        block_size, top_blocks)
    _bucket_chunk_size = int(_V4168_BUCKET_CHUNK_SIZE)
    _execution_pair_prune_eps = jnp.float32(
        _V4168_EXECUTION_PAIR_PRUNE_EPS)
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
             out_specs=(P('data', None, None), P()),
             check_rep=False)
    def fused_gate_srw_block_sparse_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        B, S, D = x.shape
        (op_key_blocks, read_blocks, write_blocks, valid_blocks,
         block_center, block_radius, n_blocks) = _v4168_block_sparse_blocks(
            op_key_local, read_local, write_local, _block_size)

        T = int(B) * int(S)
        token_capacity_i = jnp.asarray(T, dtype=jnp.int32)
        token_pos = jnp.arange(T, dtype=jnp.int32)
        max_token_chunks = max(
            1, (T + _bucket_chunk_size - 1) // _bucket_chunk_size)

        flat_x_bf = x.reshape(T, D).astype(jnp.bfloat16)
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        flat_h_bf = h_unit_bf.reshape(T, h_unit_bf.shape[-1])
        tau = _tau_from_param(raw_tau)
        flat_tau = tau.reshape(T, 1)
        flat_tau_scalar = tau.reshape(T)

        def angular_compose_parts(rho, tau_b, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau_b, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        @jax.checkpoint
        def block_step(carry, block_i):
            flat_raw_out, flat_den_cost = carry
            op_key_b = op_key_blocks[block_i]
            read_b = read_blocks[block_i]
            write_b = write_blocks[block_i]
            valid_ops = valid_blocks[block_i]

            block_score_ub = (
                flat_h_bf @ block_center[block_i]
            ).astype(jnp.float32) + block_radius[block_i]
            block_execution_ub = _v4168_block_execution_upper_bound(
                block_score_ub, flat_tau_scalar, soft_gate_temperature,
                soft_gate_boundary_power)
            candidate_mask = block_execution_ub >= _execution_pair_prune_eps
            candidate_i = candidate_mask.astype(jnp.int32)
            candidate_count = candidate_i.sum()
            candidate_rank = jnp.cumsum(candidate_i) - 1
            candidate_scatter_rank = jnp.where(
                candidate_mask, candidate_rank, token_capacity_i)
            candidate_tokens = jnp.zeros(
                (T,), dtype=jnp.int32).at[
                    candidate_scatter_rank
                ].set(token_pos, mode='drop')
            candidate_chunk_count = (
                (candidate_count + _bucket_chunk_size - 1)
                // _bucket_chunk_size)

            @jax.checkpoint
            def gate_candidate_chunk(keep_candidate, chunk_i):
                def process_chunk(keep_candidate):
                    start = chunk_i * _bucket_chunk_size
                    offsets = start + jnp.arange(
                        _bucket_chunk_size, dtype=jnp.int32)
                    safe_offsets = jnp.minimum(
                        offsets, jnp.maximum(candidate_count - 1, 0))
                    valid_pairs = offsets < candidate_count
                    token_ids = jnp.where(
                        valid_pairs, candidate_tokens[safe_offsets], 0)

                    h_b = flat_h_bf[token_ids]
                    tau_b = flat_tau[token_ids]
                    valid_mask = jnp.logical_and(
                        valid_pairs[:, None], valid_ops[None, :])

                    rho_raw = (h_b @ op_key_b.T).astype(jnp.float32)
                    rho_compute = jnp.where(
                        valid_ops[None, :], rho_raw, tau_b)
                    _admission, execution_weight = angular_compose_parts(
                        rho_compute, tau_b, valid_mask)
                    pair_exec_max = execution_weight.max(axis=-1)
                    keep_chunk = jnp.logical_and(
                        valid_pairs,
                        pair_exec_max >= _execution_pair_prune_eps)
                    keep_candidate = keep_candidate.at[offsets].set(
                        keep_chunk, mode='drop')
                    return keep_candidate

                keep_candidate = jax.lax.cond(
                    chunk_i < candidate_chunk_count,
                    process_chunk,
                    lambda k: k,
                    keep_candidate)
                return keep_candidate, None

            keep_candidate, _ = jax.lax.scan(
                gate_candidate_chunk,
                jnp.zeros((T,), dtype=jnp.bool_),
                jnp.arange(max_token_chunks, dtype=jnp.int32))

            kept_i = keep_candidate.astype(jnp.int32)
            kept_count = kept_i.sum()
            kept_rank = jnp.cumsum(kept_i) - 1
            kept_scatter_rank = jnp.where(
                keep_candidate, kept_rank, token_capacity_i)
            kept_tokens = jnp.zeros(
                (T,), dtype=jnp.int32).at[
                    kept_scatter_rank
                ].set(candidate_tokens, mode='drop')
            kept_chunk_count = (
                (kept_count + _bucket_chunk_size - 1)
                // _bucket_chunk_size)

            @jax.checkpoint
            def kept_chunk_step(carry, chunk_i):
                def process_chunk(carry):
                    flat_raw_out, flat_den_cost = carry
                    start = chunk_i * _bucket_chunk_size
                    offsets = start + jnp.arange(
                        _bucket_chunk_size, dtype=jnp.int32)
                    safe_offsets = jnp.minimum(
                        offsets, jnp.maximum(kept_count - 1, 0))
                    valid_pairs = offsets < kept_count
                    fallback_token = kept_tokens[0]
                    token_ids = jnp.where(
                        valid_pairs, kept_tokens[safe_offsets],
                        fallback_token)

                    x_b = flat_x_bf[token_ids]
                    h_b = flat_h_bf[token_ids]
                    tau_b = flat_tau[token_ids]
                    valid_mask = jnp.logical_and(
                        valid_pairs[:, None], valid_ops[None, :])

                    rho_raw = (h_b @ op_key_b.T).astype(jnp.float32)
                    rho_compute = jnp.where(
                        valid_ops[None, :], rho_raw, tau_b)
                    admission, execution_weight = angular_compose_parts(
                        rho_compute, tau_b, valid_mask)
                    pair_exec_max = execution_weight.max(
                        axis=-1, keepdims=True)
                    pair_active = pair_exec_max >= _execution_pair_prune_eps
                    admission = jnp.where(pair_active, admission, 0.0)
                    execution_weight = jnp.where(
                        pair_active, execution_weight, 0.0)
                    xr = (x_b @ read_b.T).astype(jnp.float32)
                    weighted = execution_weight * xr
                    out_b = (weighted.astype(jnp.bfloat16) @ write_b).astype(
                        jnp.float32)
                    den_b = admission.sum(axis=-1, keepdims=True)
                    flat_raw_out = flat_raw_out.at[token_ids].add(out_b)
                    flat_den_cost = flat_den_cost.at[token_ids].add(den_b)
                    return flat_raw_out, flat_den_cost

                carry = jax.lax.cond(
                    chunk_i < kept_chunk_count,
                    process_chunk,
                    lambda c: c,
                    carry)
                return carry, None

            (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
                kept_chunk_step,
                (flat_raw_out, flat_den_cost),
                jnp.arange(max_token_chunks, dtype=jnp.int32))
            return (flat_raw_out, flat_den_cost), None

        (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
            block_step,
            (jnp.zeros((T, D), dtype=jnp.float32),
             jnp.zeros((T, 1), dtype=jnp.float32)),
            jnp.arange(n_blocks, dtype=jnp.int32))

        raw_out = flat_raw_out.reshape(B, S, D)
        total_den_cost = flat_den_cost.reshape(B, S, 1)
        global_raw_out = jax.lax.psum(raw_out, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = global_raw_out / admission_den
        return out.astype(jnp.float32)

    return fused_gate_srw_block_sparse_minimal


def make_sharded_srw_paired_block_sparse_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0):
    """Create v4168 paired Q/K Block-sparse SRW returning only output."""
    del max_chunk_size, dead_exposure_target, block_margin
    _block_size, _top_blocks = _v4168_block_sparse_config(
        block_size, top_blocks)
    _bucket_chunk_size = int(_V4168_BUCKET_CHUNK_SIZE)
    _execution_pair_prune_eps = jnp.float32(
        _V4168_EXECUTION_PAIR_PRUNE_EPS)
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
             out_specs=(P('data', None, None, None), P()),
             check_rep=False)
    def fused_gate_srw_paired_block_sparse_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        B, S, D = x.shape
        R = int(h.shape[2])
        (op_key_blocks, read_blocks, write_blocks, valid_blocks,
         block_center, block_radius, n_blocks) = _v4168_block_sparse_blocks(
            op_key_local, read_local, write_local, _block_size)

        T = int(B) * int(S)
        route_token_span = int(R) * int(T)
        pair_capacity_i = jnp.asarray(route_token_span, dtype=jnp.int32)
        pair_pos = jnp.arange(route_token_span, dtype=jnp.int32)
        max_pair_chunks = max(
            1, (route_token_span + _bucket_chunk_size - 1)
            // _bucket_chunk_size)

        flat_x_bf = x.reshape(T, D).astype(jnp.bfloat16)
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        flat_h_bf = h_unit_bf.reshape(T, R, h_unit_bf.shape[-1])
        tau = _tau_from_param(raw_tau)
        flat_tau = tau.reshape(T, R, 1)
        flat_tau_scalar = tau.reshape(T, R)

        def angular_compose_parts(rho, tau_b, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau_b, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        @jax.checkpoint
        def block_step(carry, block_i):
            flat_raw_out, flat_den_cost = carry
            op_key_b = op_key_blocks[block_i]
            read_b = read_blocks[block_i]
            write_b = write_blocks[block_i]
            valid_ops = valid_blocks[block_i]

            block_score_ub = jnp.einsum(
                'trd,d->tr', flat_h_bf, block_center[block_i]
            ).astype(jnp.float32) + block_radius[block_i]
            block_execution_ub = _v4168_block_execution_upper_bound(
                block_score_ub, flat_tau_scalar, soft_gate_temperature,
                soft_gate_boundary_power)
            candidate_mask = block_execution_ub >= _execution_pair_prune_eps
            candidate_flat = candidate_mask.T.reshape(route_token_span)
            candidate_i = candidate_flat.astype(jnp.int32)
            candidate_count = candidate_i.sum()
            candidate_rank = jnp.cumsum(candidate_i) - 1
            candidate_scatter_rank = jnp.where(
                candidate_flat, candidate_rank, pair_capacity_i)
            candidate_pairs = jnp.zeros(
                (route_token_span,), dtype=jnp.int32).at[
                    candidate_scatter_rank
                ].set(pair_pos, mode='drop')
            candidate_chunk_count = (
                (candidate_count + _bucket_chunk_size - 1)
                // _bucket_chunk_size)

            @jax.checkpoint
            def gate_candidate_chunk(keep_candidate, chunk_i):
                def process_chunk(keep_candidate):
                    start = chunk_i * _bucket_chunk_size
                    offsets = start + jnp.arange(
                        _bucket_chunk_size, dtype=jnp.int32)
                    safe_offsets = jnp.minimum(
                        offsets, jnp.maximum(candidate_count - 1, 0))
                    valid_pairs = offsets < candidate_count
                    pair_ids = jnp.where(
                        valid_pairs, candidate_pairs[safe_offsets], 0)
                    route_ids = pair_ids // int(T)
                    token_ids = pair_ids % int(T)

                    h_b = flat_h_bf[token_ids, route_ids]
                    tau_b = flat_tau[token_ids, route_ids]
                    valid_mask = jnp.logical_and(
                        valid_pairs[:, None], valid_ops[None, :])

                    rho_raw = (h_b @ op_key_b.T).astype(jnp.float32)
                    rho_compute = jnp.where(
                        valid_ops[None, :], rho_raw, tau_b)
                    _admission, execution_weight = angular_compose_parts(
                        rho_compute, tau_b, valid_mask)
                    pair_exec_max = execution_weight.max(axis=-1)
                    keep_chunk = jnp.logical_and(
                        valid_pairs,
                        pair_exec_max >= _execution_pair_prune_eps)
                    keep_candidate = keep_candidate.at[offsets].set(
                        keep_chunk, mode='drop')
                    return keep_candidate

                keep_candidate = jax.lax.cond(
                    chunk_i < candidate_chunk_count,
                    process_chunk,
                    lambda k: k,
                    keep_candidate)
                return keep_candidate, None

            keep_candidate, _ = jax.lax.scan(
                gate_candidate_chunk,
                jnp.zeros((route_token_span,), dtype=jnp.bool_),
                jnp.arange(max_pair_chunks, dtype=jnp.int32))

            kept_i = keep_candidate.astype(jnp.int32)
            kept_count = kept_i.sum()
            kept_rank = jnp.cumsum(kept_i) - 1
            kept_scatter_rank = jnp.where(
                keep_candidate, kept_rank, pair_capacity_i)
            kept_pairs = jnp.zeros(
                (route_token_span,), dtype=jnp.int32).at[
                    kept_scatter_rank
                ].set(candidate_pairs, mode='drop')
            kept_chunk_count = (
                (kept_count + _bucket_chunk_size - 1)
                // _bucket_chunk_size)

            @jax.checkpoint
            def kept_chunk_step(carry, chunk_i):
                def process_chunk(carry):
                    flat_raw_out, flat_den_cost = carry
                    start = chunk_i * _bucket_chunk_size
                    offsets = start + jnp.arange(
                        _bucket_chunk_size, dtype=jnp.int32)
                    safe_offsets = jnp.minimum(
                        offsets, jnp.maximum(kept_count - 1, 0))
                    valid_pairs = offsets < kept_count
                    fallback_pair = kept_pairs[0]
                    pair_ids = jnp.where(
                        valid_pairs, kept_pairs[safe_offsets], fallback_pair)
                    route_ids = pair_ids // int(T)
                    token_ids = pair_ids % int(T)

                    x_b = flat_x_bf[token_ids]
                    h_b = flat_h_bf[token_ids, route_ids]
                    tau_b = flat_tau[token_ids, route_ids]
                    valid_mask = jnp.logical_and(
                        valid_pairs[:, None], valid_ops[None, :])

                    rho_raw = (h_b @ op_key_b.T).astype(jnp.float32)
                    rho_compute = jnp.where(
                        valid_ops[None, :], rho_raw, tau_b)
                    admission, execution_weight = angular_compose_parts(
                        rho_compute, tau_b, valid_mask)
                    pair_exec_max = execution_weight.max(
                        axis=-1, keepdims=True)
                    pair_active = pair_exec_max >= _execution_pair_prune_eps
                    admission = jnp.where(pair_active, admission, 0.0)
                    execution_weight = jnp.where(
                        pair_active, execution_weight, 0.0)
                    xr = (x_b @ read_b.T).astype(jnp.float32)
                    weighted = execution_weight * xr
                    out_b = (weighted.astype(jnp.bfloat16) @ write_b).astype(
                        jnp.float32)
                    den_b = admission.sum(axis=-1, keepdims=True)
                    flat_raw_out = flat_raw_out.at[token_ids, route_ids].add(
                        out_b)
                    flat_den_cost = flat_den_cost.at[token_ids, route_ids].add(
                        den_b)
                    return flat_raw_out, flat_den_cost

                carry = jax.lax.cond(
                    chunk_i < kept_chunk_count,
                    process_chunk,
                    lambda c: c,
                    carry)
                return carry, None

            (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
                kept_chunk_step,
                (flat_raw_out, flat_den_cost),
                jnp.arange(max_pair_chunks, dtype=jnp.int32))
            return (flat_raw_out, flat_den_cost), None

        (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
            block_step,
            (jnp.zeros((T, R, D), dtype=jnp.float32),
             jnp.zeros((T, R, 1), dtype=jnp.float32)),
            jnp.arange(n_blocks, dtype=jnp.int32))

        raw_out = flat_raw_out.reshape(B, S, R, D)
        total_den_cost = flat_den_cost.reshape(B, S, R, 1)
        global_raw_out = jax.lax.psum(raw_out, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = global_raw_out / admission_den
        return out.astype(jnp.float32)

    return fused_gate_srw_paired_block_sparse_minimal


# Legacy block-sparse candidate execution is retained only as a fallback/debug
# path. The hardware-sector path below executes exact global topK fixed tiles.
make_sharded_srw_block_sparse_minimal_fallback = (
    make_sharded_srw_block_sparse_minimal)
make_sharded_srw_paired_block_sparse_minimal_fallback = (
    make_sharded_srw_paired_block_sparse_minimal)


def _v4168_sector_topk_blocks(op_key_local, read_local, write_local,
                              sector_size):
    """Contiguous physical sector tiles plus normalized sector centroids."""
    (op_key_blocks, read_blocks, write_blocks, valid_blocks,
     sector_center, _block_radius, n_sectors) = _v4168_block_sparse_blocks(
        op_key_local, read_local, write_local, sector_size)
    sector_center = jax.lax.stop_gradient(sector_center)
    return (
        op_key_blocks,
        read_blocks,
        write_blocks,
        valid_blocks,
        sector_center,
        n_sectors,
    )


def make_sharded_srw_sector_topk_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0):
    """Create exact token-gather sector-topK fallback single-route SRW.

    This is semantically exact global sector topK: local sector centers are
    all-gathered across the model axis, global topK sector ids are selected per
    token, and each model shard masks out selected sectors it does not own.
    The shape cost is token-centric and dynamically gathers sector tiles, so it
    is retained as a correctness/debug fallback for the bucketed path.
    """
    del max_chunk_size, dead_exposure_target, block_margin
    _sector_size, _top_sectors = _v4168_block_sparse_config(
        block_size, top_blocks)
    _bucket_chunk_size = min(64, int(_V4168_BUCKET_CHUNK_SIZE))
    _model_axis_size = int(mesh.shape['model'])
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
    def fused_gate_srw_sector_topk_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        B, S, D = x.shape
        (op_key_blocks, read_blocks, write_blocks, valid_blocks,
         sector_center, n_sectors) = _v4168_sector_topk_blocks(
            op_key_local, read_local, write_local, _sector_size)
        topk = min(_top_sectors, int(n_sectors) * _model_axis_size)
        all_sector_center = jax.lax.all_gather(
            sector_center, 'model', axis=0, tiled=True)
        local_sector_offset = (
            jax.lax.axis_index('model') * jnp.asarray(n_sectors, jnp.int32))

        T = int(B) * int(S)
        max_token_chunks = max(
            1, (T + _bucket_chunk_size - 1) // _bucket_chunk_size)
        token_pos = jnp.arange(T, dtype=jnp.int32)
        flat_x_bf = x.reshape(T, D).astype(jnp.bfloat16)
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        flat_h_bf = h_unit_bf.reshape(T, h_unit_bf.shape[-1])
        tau = _tau_from_param(raw_tau)
        flat_tau = tau.reshape(T, 1)

        sector_scores = (
            flat_h_bf @ all_sector_center.T
        ).astype(jnp.float32)
        _top_scores, top_sector_ids = jax.lax.top_k(sector_scores, topk)
        del _top_scores

        def angular_compose_parts(rho, tau_b, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau_b, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        @jax.checkpoint
        def token_chunk_step(carry, chunk_i):
            flat_raw_out, flat_den_cost = carry
            start = chunk_i * _bucket_chunk_size
            offsets = start + jnp.arange(
                _bucket_chunk_size, dtype=jnp.int32)
            valid_tokens = offsets < T
            safe_offsets = jnp.minimum(offsets, T - 1)
            token_ids = jnp.where(valid_tokens, token_pos[safe_offsets], 0)

            x_b = flat_x_bf[token_ids]
            h_b = flat_h_bf[token_ids]
            tau_b = flat_tau[token_ids]
            sector_ids = top_sector_ids[token_ids]
            local_sector_ids = sector_ids - local_sector_offset
            selected_here = jnp.logical_and(
                local_sector_ids >= 0,
                local_sector_ids < int(n_sectors))
            safe_sector_ids = jnp.clip(
                local_sector_ids, 0, int(n_sectors) - 1).astype(jnp.int32)
            op_key_b = op_key_blocks[safe_sector_ids]
            read_b = read_blocks[safe_sector_ids]
            write_b = write_blocks[safe_sector_ids]
            valid_ops = jnp.logical_and(
                valid_blocks[safe_sector_ids],
                selected_here[:, :, None])
            valid_mask = jnp.logical_and(
                valid_tokens[:, None, None], valid_ops)

            rho_raw = jnp.einsum(
                'cd,cknd->ckn', h_b, op_key_b).astype(jnp.float32)
            rho_compute = jnp.where(valid_ops, rho_raw, tau_b[:, None, :])
            admission, execution_weight = angular_compose_parts(
                rho_compute, tau_b[:, None, :], valid_mask)
            xr = jnp.einsum(
                'cd,cknd->ckn', x_b, read_b).astype(jnp.float32)
            weighted = execution_weight * xr
            out_b = jnp.einsum(
                'ckn,cknd->cd',
                weighted.astype(jnp.bfloat16),
                write_b).astype(jnp.float32)
            den_b = admission.sum(axis=(1, 2), keepdims=False)[:, None]
            flat_raw_out = flat_raw_out.at[token_ids].add(
                out_b, mode='drop')
            flat_den_cost = flat_den_cost.at[token_ids].add(
                den_b, mode='drop')
            return (flat_raw_out, flat_den_cost), None

        (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
            token_chunk_step,
            (jnp.zeros((T, D), dtype=jnp.float32),
             jnp.zeros((T, 1), dtype=jnp.float32)),
            jnp.arange(max_token_chunks, dtype=jnp.int32))

        raw_out = flat_raw_out.reshape(B, S, D)
        total_den_cost = flat_den_cost.reshape(B, S, 1)
        global_raw_out = jax.lax.psum(raw_out, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = global_raw_out / admission_den
        return out.astype(jnp.float32)

    return fused_gate_srw_sector_topk_minimal


def make_sharded_srw_paired_sector_topk_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0):
    """Create exact token-gather sector-topK fallback paired Q/K SRW.

    This has the same exact global topK semantics and replicated SPMD shape
    cost as make_sharded_srw_sector_topk_minimal, with route-local Q/K
    denominators and outputs preserved. It is retained as a correctness/debug
    fallback for the bucketed path.
    """
    del max_chunk_size, dead_exposure_target, block_margin
    _sector_size, _top_sectors = _v4168_block_sparse_config(
        block_size, top_blocks)
    _bucket_chunk_size = min(64, int(_V4168_BUCKET_CHUNK_SIZE))
    _model_axis_size = int(mesh.shape['model'])
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
    def fused_gate_srw_paired_sector_topk_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        B, S, D = x.shape
        R = int(h.shape[2])
        (op_key_blocks, read_blocks, write_blocks, valid_blocks,
         sector_center, n_sectors) = _v4168_sector_topk_blocks(
            op_key_local, read_local, write_local, _sector_size)
        topk = min(_top_sectors, int(n_sectors) * _model_axis_size)
        all_sector_center = jax.lax.all_gather(
            sector_center, 'model', axis=0, tiled=True)
        local_sector_offset = (
            jax.lax.axis_index('model') * jnp.asarray(n_sectors, jnp.int32))

        T = int(B) * int(S)
        pair_span = int(R) * int(T)
        max_pair_chunks = max(
            1, (pair_span + _bucket_chunk_size - 1) // _bucket_chunk_size)
        pair_pos = jnp.arange(pair_span, dtype=jnp.int32)
        flat_x_bf = x.reshape(T, D).astype(jnp.bfloat16)
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        flat_h_pair_bf = jnp.transpose(
            h_unit_bf.reshape(T, R, h_unit_bf.shape[-1]),
            (1, 0, 2)).reshape(pair_span, h_unit_bf.shape[-1])
        tau = _tau_from_param(raw_tau)
        flat_tau_pair = jnp.transpose(
            tau.reshape(T, R, 1), (1, 0, 2)).reshape(pair_span, 1)

        sector_scores = (
            flat_h_pair_bf @ all_sector_center.T
        ).astype(jnp.float32)
        _top_scores, top_sector_ids = jax.lax.top_k(sector_scores, topk)
        del _top_scores

        def angular_compose_parts(rho, tau_b, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau_b, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        @jax.checkpoint
        def pair_chunk_step(carry, chunk_i):
            flat_raw_out, flat_den_cost = carry
            start = chunk_i * _bucket_chunk_size
            offsets = start + jnp.arange(
                _bucket_chunk_size, dtype=jnp.int32)
            valid_pairs = offsets < pair_span
            safe_offsets = jnp.minimum(offsets, pair_span - 1)
            pair_ids = jnp.where(valid_pairs, pair_pos[safe_offsets], 0)
            route_ids = pair_ids // int(T)
            token_ids = pair_ids % int(T)

            x_b = flat_x_bf[token_ids]
            h_b = flat_h_pair_bf[pair_ids]
            tau_b = flat_tau_pair[pair_ids]
            sector_ids = top_sector_ids[pair_ids]
            local_sector_ids = sector_ids - local_sector_offset
            selected_here = jnp.logical_and(
                local_sector_ids >= 0,
                local_sector_ids < int(n_sectors))
            safe_sector_ids = jnp.clip(
                local_sector_ids, 0, int(n_sectors) - 1).astype(jnp.int32)
            op_key_b = op_key_blocks[safe_sector_ids]
            read_b = read_blocks[safe_sector_ids]
            write_b = write_blocks[safe_sector_ids]
            valid_ops = jnp.logical_and(
                valid_blocks[safe_sector_ids],
                selected_here[:, :, None])
            valid_mask = jnp.logical_and(
                valid_pairs[:, None, None], valid_ops)

            rho_raw = jnp.einsum(
                'cd,cknd->ckn', h_b, op_key_b).astype(jnp.float32)
            rho_compute = jnp.where(valid_ops, rho_raw, tau_b[:, None, :])
            admission, execution_weight = angular_compose_parts(
                rho_compute, tau_b[:, None, :], valid_mask)
            xr = jnp.einsum(
                'cd,cknd->ckn', x_b, read_b).astype(jnp.float32)
            weighted = execution_weight * xr
            out_b = jnp.einsum(
                'ckn,cknd->cd',
                weighted.astype(jnp.bfloat16),
                write_b).astype(jnp.float32)
            den_b = admission.sum(axis=(1, 2), keepdims=False)[:, None]
            flat_raw_out = flat_raw_out.at[token_ids, route_ids].add(
                out_b, mode='drop')
            flat_den_cost = flat_den_cost.at[token_ids, route_ids].add(
                den_b, mode='drop')
            return (flat_raw_out, flat_den_cost), None

        (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
            pair_chunk_step,
            (jnp.zeros((T, R, D), dtype=jnp.float32),
             jnp.zeros((T, R, 1), dtype=jnp.float32)),
            jnp.arange(max_pair_chunks, dtype=jnp.int32))

        raw_out = flat_raw_out.reshape(B, S, R, D)
        total_den_cost = flat_den_cost.reshape(B, S, R, 1)
        global_raw_out = jax.lax.psum(raw_out, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        admission_den_base = jnp.maximum(global_den_cost, 1.0)
        admission_den_forward = jnp.power(
            admission_den_base, _admission_den_power)
        admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
        admission_den = (
            admission_den_sg
            + _admission_den_grad_scale
            * (admission_den_forward - admission_den_sg))
        out = global_raw_out / admission_den
        return out.astype(jnp.float32)

    return fused_gate_srw_paired_sector_topk_minimal


make_sharded_srw_sector_topk_token_gather_minimal_fallback = (
    make_sharded_srw_sector_topk_minimal)
make_sharded_srw_paired_sector_topk_token_gather_minimal_fallback = (
    make_sharded_srw_paired_sector_topk_minimal)


def make_sharded_srw_sector_bucketed_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0,
        benchmark_runtime_metrics=False,
        _debug_bucket_capacity_override=None):
    """Create exact global-sector-topK owner-local bucketed single-route SRW."""
    del max_chunk_size, dead_exposure_target, block_margin
    _sector_size, _top_sectors = _v4168_block_sparse_config(
        block_size, top_blocks)
    _token_gather_chunk_size = min(64, int(_V4168_BUCKET_CHUNK_SIZE))
    _model_axis_size = int(mesh.shape['model'])
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    _benchmark_runtime_metrics = bool(benchmark_runtime_metrics)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None),
                       P('model', None),
                       P('data', None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None), P()),
             check_rep=False)
    def fused_gate_srw_sector_bucketed_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        B, S, D = x.shape
        (op_key_blocks, read_blocks, write_blocks, valid_blocks,
         sector_center, n_sectors) = _v4168_sector_topk_blocks(
            op_key_local, read_local, write_local, _sector_size)
        n_local_sectors = int(n_sectors)
        topk = min(_top_sectors, n_local_sectors * _model_axis_size)
        global_n_sectors = n_local_sectors * _model_axis_size
        all_sector_center = jax.lax.all_gather(
            sector_center, 'model', axis=0, tiled=True)
        local_sector_offset = (
            jax.lax.axis_index('model')
            * jnp.asarray(n_local_sectors, jnp.int32))

        T = int(B) * int(S)
        max_token_chunks = max(
            1, (T + _token_gather_chunk_size - 1)
            // _token_gather_chunk_size)
        token_pos = jnp.arange(T, dtype=jnp.int32)
        flat_x_bf = x.reshape(T, D).astype(jnp.bfloat16)
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        flat_h_bf = h_unit_bf.reshape(T, h_unit_bf.shape[-1])
        tau = _tau_from_param(raw_tau)
        flat_tau = tau.reshape(T, 1)

        sector_scores = (
            flat_h_bf @ all_sector_center.T
        ).astype(jnp.float32)
        _top_scores, top_sector_ids = jax.lax.top_k(sector_scores, topk)
        del _top_scores

        local_sector_ids = top_sector_ids - local_sector_offset
        selected_here = jnp.logical_and(
            local_sector_ids >= 0,
            local_sector_ids < n_local_sectors)
        if _debug_bucket_capacity_override is None:
            bucket_capacity, _avg_pairs_per_sector = (
                _v4168_sector_bucket_capacity(T, topk, global_n_sectors))
        else:
            bucket_capacity = max(1, int(_debug_bucket_capacity_override))
        (sector_token_ids, sector_bucket_valid, overflow_count,
         _bucket_max_fill, _bucket_mean_fill,
         attempted_fill) = _v4168_build_sector_buckets(
            local_sector_ids, selected_here, T, topk, n_local_sectors,
            bucket_capacity)
        global_overflow_count = jax.lax.psum(overflow_count, 'model')
        bucket_fill = sector_bucket_valid.astype(jnp.int32).sum(axis=1)
        selected_sector_local = (bucket_fill > 0).astype(jnp.float32)
        selected_sector_count = jax.lax.psum(
            selected_sector_local.sum(), 'model')
        selected_real_ops_local = (
            selected_sector_local[:, None]
            * valid_blocks.astype(jnp.float32)).sum()
        selected_real_ops = jax.lax.psum(selected_real_ops_local, 'model')
        total_real_ops = jax.lax.psum(
            valid_blocks.astype(jnp.float32).sum(), 'model')
        if _benchmark_runtime_metrics:
            sector_diag = _v4168_sector_benchmark_runtime_diag(
                bucket_fill, attempted_fill, global_overflow_count,
                bucket_capacity,
                T, topk, global_n_sectors, _sector_size,
                selected_sector_count, selected_real_ops, total_real_ops)
        else:
            sector_diag = _v4168_sector_core_runtime_diag(
                bucket_fill, global_overflow_count, global_n_sectors,
                selected_sector_count, selected_real_ops, total_real_ops)

        def angular_compose_parts(rho, tau_b, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau_b, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        def finish(flat_raw_out, flat_den_cost):
            raw_out = flat_raw_out.reshape(B, S, D)
            total_den_cost = flat_den_cost.reshape(B, S, 1)
            global_raw_out = jax.lax.psum(raw_out, 'model')
            global_den_cost = jax.lax.psum(total_den_cost, 'model')
            admission_den_base = jnp.maximum(global_den_cost, 1.0)
            admission_den_forward = jnp.power(
                admission_den_base, _admission_den_power)
            admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
            admission_den = (
                admission_den_sg
                + _admission_den_grad_scale
                * (admission_den_forward - admission_den_sg))
            return (global_raw_out / admission_den).astype(jnp.float32)

        def bucket_execute(_):
            @jax.checkpoint
            def sector_step(carry, sector_i):
                flat_raw_out, flat_den_cost = carry
                token_ids = sector_token_ids[sector_i]
                valid_t = sector_bucket_valid[sector_i]
                op_key_b = op_key_blocks[sector_i]
                read_b = read_blocks[sector_i]
                write_b = write_blocks[sector_i]
                valid_ops = valid_blocks[sector_i]

                x_b = flat_x_bf[token_ids]
                h_b = flat_h_bf[token_ids]
                tau_b = flat_tau[token_ids]
                valid_mask = jnp.logical_and(
                    valid_t[:, None], valid_ops[None, :])

                rho_raw = (h_b @ op_key_b.T).astype(jnp.float32)
                rho_compute = jnp.where(
                    valid_ops[None, :], rho_raw, tau_b)
                admission, execution_weight = angular_compose_parts(
                    rho_compute, tau_b, valid_mask)
                xr = (x_b @ read_b.T).astype(jnp.float32)
                weighted = execution_weight * xr
                out_b = (weighted.astype(jnp.bfloat16) @ write_b).astype(
                    jnp.float32)
                den_b = admission.sum(axis=-1, keepdims=True)
                flat_raw_out = flat_raw_out.at[token_ids].add(
                    out_b, mode='drop')
                flat_den_cost = flat_den_cost.at[token_ids].add(
                    den_b, mode='drop')
                return (flat_raw_out, flat_den_cost), None

            (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
                sector_step,
                (jnp.zeros((T, D), dtype=jnp.float32),
                 jnp.zeros((T, 1), dtype=jnp.float32)),
                jnp.arange(n_local_sectors, dtype=jnp.int32))
            return finish(flat_raw_out, flat_den_cost)

        def token_gather_execute(_):
            @jax.checkpoint
            def token_chunk_step(carry, chunk_i):
                flat_raw_out, flat_den_cost = carry
                start = chunk_i * _token_gather_chunk_size
                offsets = start + jnp.arange(
                    _token_gather_chunk_size, dtype=jnp.int32)
                valid_tokens = offsets < T
                safe_offsets = jnp.minimum(offsets, T - 1)
                token_ids = jnp.where(valid_tokens, token_pos[safe_offsets], 0)

                x_b = flat_x_bf[token_ids]
                h_b = flat_h_bf[token_ids]
                tau_b = flat_tau[token_ids]
                sector_ids = top_sector_ids[token_ids]
                chunk_local_sector_ids = sector_ids - local_sector_offset
                chunk_selected_here = jnp.logical_and(
                    chunk_local_sector_ids >= 0,
                    chunk_local_sector_ids < n_local_sectors)
                safe_sector_ids = jnp.clip(
                    chunk_local_sector_ids, 0,
                    n_local_sectors - 1).astype(jnp.int32)
                op_key_b = op_key_blocks[safe_sector_ids]
                read_b = read_blocks[safe_sector_ids]
                write_b = write_blocks[safe_sector_ids]
                valid_ops = jnp.logical_and(
                    valid_blocks[safe_sector_ids],
                    chunk_selected_here[:, :, None])
                valid_mask = jnp.logical_and(
                    valid_tokens[:, None, None], valid_ops)

                rho_raw = jnp.einsum(
                    'cd,cknd->ckn', h_b, op_key_b).astype(jnp.float32)
                rho_compute = jnp.where(valid_ops, rho_raw, tau_b[:, None, :])
                admission, execution_weight = angular_compose_parts(
                    rho_compute, tau_b[:, None, :], valid_mask)
                xr = jnp.einsum(
                    'cd,cknd->ckn', x_b, read_b).astype(jnp.float32)
                weighted = execution_weight * xr
                out_b = jnp.einsum(
                    'ckn,cknd->cd',
                    weighted.astype(jnp.bfloat16),
                    write_b).astype(jnp.float32)
                den_b = admission.sum(axis=(1, 2), keepdims=False)[:, None]
                flat_raw_out = flat_raw_out.at[token_ids].add(
                    out_b, mode='drop')
                flat_den_cost = flat_den_cost.at[token_ids].add(
                    den_b, mode='drop')
                return (flat_raw_out, flat_den_cost), None

            (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
                token_chunk_step,
                (jnp.zeros((T, D), dtype=jnp.float32),
                 jnp.zeros((T, 1), dtype=jnp.float32)),
                jnp.arange(max_token_chunks, dtype=jnp.int32))
            return finish(flat_raw_out, flat_den_cost)

        del token_gather_execute
        return bucket_execute(None), sector_diag

    return fused_gate_srw_sector_bucketed_minimal


def make_sharded_srw_paired_sector_bucketed_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0,
        benchmark_runtime_metrics=False,
        _debug_bucket_capacity_override=None):
    """Create exact global-sector-topK owner-local bucketed paired Q/K SRW."""
    del max_chunk_size, dead_exposure_target, block_margin
    _sector_size, _top_sectors = _v4168_block_sparse_config(
        block_size, top_blocks)
    _token_gather_chunk_size = min(64, int(_V4168_BUCKET_CHUNK_SIZE))
    _model_axis_size = int(mesh.shape['model'])
    _soft_gate_effective_active_eps = jnp.float32(
        soft_gate_effective_active_eps)
    _admission_den_power = jnp.maximum(
        jnp.asarray(admission_den_power, dtype=jnp.float32),
        jnp.float32(0.0))
    _admission_den_grad_scale = jnp.clip(
        jnp.asarray(admission_den_grad_scale, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0))
    _benchmark_runtime_metrics = bool(benchmark_runtime_metrics)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),
                       P('data', None, None, None),
                       P('model', None),
                       P('data', None, None, None),
                       P('model', None),
                       P('model', None),
                       P(), P(), P(), P(), P()),
             out_specs=(P('data', None, None, None), P()),
             check_rep=False)
    def fused_gate_srw_paired_sector_bucketed_minimal(
            x, h, op_key_local, raw_tau, read_local, write_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        B, S, D = x.shape
        R = int(h.shape[2])
        (op_key_blocks, read_blocks, write_blocks, valid_blocks,
         sector_center, n_sectors) = _v4168_sector_topk_blocks(
            op_key_local, read_local, write_local, _sector_size)
        n_local_sectors = int(n_sectors)
        topk = min(_top_sectors, n_local_sectors * _model_axis_size)
        global_n_sectors = n_local_sectors * _model_axis_size
        all_sector_center = jax.lax.all_gather(
            sector_center, 'model', axis=0, tiled=True)
        local_sector_offset = (
            jax.lax.axis_index('model')
            * jnp.asarray(n_local_sectors, jnp.int32))

        T = int(B) * int(S)
        pair_span = int(R) * int(T)
        max_pair_chunks = max(
            1, (pair_span + _token_gather_chunk_size - 1)
            // _token_gather_chunk_size)
        pair_pos = jnp.arange(pair_span, dtype=jnp.int32)
        flat_x_bf = x.reshape(T, D).astype(jnp.bfloat16)
        h_unit_bf = _forward_unit_direction(
            h.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)
        flat_h_pair_bf = jnp.transpose(
            h_unit_bf.reshape(T, R, h_unit_bf.shape[-1]),
            (1, 0, 2)).reshape(pair_span, h_unit_bf.shape[-1])
        tau = _tau_from_param(raw_tau)
        flat_tau_pair = jnp.transpose(
            tau.reshape(T, R, 1), (1, 0, 2)).reshape(pair_span, 1)

        sector_scores = (
            flat_h_pair_bf @ all_sector_center.T
        ).astype(jnp.float32)
        _top_scores, top_sector_ids = jax.lax.top_k(sector_scores, topk)
        del _top_scores

        local_sector_ids = top_sector_ids - local_sector_offset
        selected_here = jnp.logical_and(
            local_sector_ids >= 0,
            local_sector_ids < n_local_sectors)
        if _debug_bucket_capacity_override is None:
            bucket_capacity, _avg_pairs_per_sector = (
                _v4168_sector_bucket_capacity(
                    pair_span, topk, global_n_sectors))
        else:
            bucket_capacity = max(1, int(_debug_bucket_capacity_override))
        (sector_pair_ids, sector_bucket_valid, overflow_count,
         _bucket_max_fill, _bucket_mean_fill,
         attempted_fill) = _v4168_build_sector_buckets(
            local_sector_ids, selected_here, pair_span, topk, n_local_sectors,
            bucket_capacity)
        global_overflow_count = jax.lax.psum(overflow_count, 'model')
        bucket_fill = sector_bucket_valid.astype(jnp.int32).sum(axis=1)
        selected_sector_local = (bucket_fill > 0).astype(jnp.float32)
        selected_sector_count = jax.lax.psum(
            selected_sector_local.sum(), 'model')
        selected_real_ops_local = (
            selected_sector_local[:, None]
            * valid_blocks.astype(jnp.float32)).sum()
        selected_real_ops = jax.lax.psum(selected_real_ops_local, 'model')
        total_real_ops = jax.lax.psum(
            valid_blocks.astype(jnp.float32).sum(), 'model')
        if _benchmark_runtime_metrics:
            sector_diag = _v4168_sector_benchmark_runtime_diag(
                bucket_fill, attempted_fill, global_overflow_count,
                bucket_capacity,
                pair_span, topk, global_n_sectors, _sector_size,
                selected_sector_count, selected_real_ops, total_real_ops)
        else:
            sector_diag = _v4168_sector_core_runtime_diag(
                bucket_fill, global_overflow_count, global_n_sectors,
                selected_sector_count, selected_real_ops, total_real_ops)

        def angular_compose_parts(rho, tau_b, valid_mask):
            _, admission, _drive, execution_weight, _ = _compute_admission_drive(
                rho, tau_b, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=_soft_gate_effective_active_eps,
                execution_prune_eps=execution_prune_eps)
            admission = jnp.where(valid_mask, admission, 0.0)
            execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
            return admission, execution_weight

        def finish(flat_raw_out, flat_den_cost):
            raw_out = flat_raw_out.reshape(B, S, R, D)
            total_den_cost = flat_den_cost.reshape(B, S, R, 1)
            global_raw_out = jax.lax.psum(raw_out, 'model')
            global_den_cost = jax.lax.psum(total_den_cost, 'model')
            admission_den_base = jnp.maximum(global_den_cost, 1.0)
            admission_den_forward = jnp.power(
                admission_den_base, _admission_den_power)
            admission_den_sg = jax.lax.stop_gradient(admission_den_forward)
            admission_den = (
                admission_den_sg
                + _admission_den_grad_scale
                * (admission_den_forward - admission_den_sg))
            return (global_raw_out / admission_den).astype(jnp.float32)

        def bucket_execute(_):
            @jax.checkpoint
            def sector_step(carry, sector_i):
                flat_raw_out, flat_den_cost = carry
                pair_ids = sector_pair_ids[sector_i]
                valid_p = sector_bucket_valid[sector_i]
                route_ids = pair_ids // int(T)
                token_ids = pair_ids % int(T)
                op_key_b = op_key_blocks[sector_i]
                read_b = read_blocks[sector_i]
                write_b = write_blocks[sector_i]
                valid_ops = valid_blocks[sector_i]

                x_b = flat_x_bf[token_ids]
                h_b = flat_h_pair_bf[pair_ids]
                tau_b = flat_tau_pair[pair_ids]
                valid_mask = jnp.logical_and(
                    valid_p[:, None], valid_ops[None, :])

                rho_raw = (h_b @ op_key_b.T).astype(jnp.float32)
                rho_compute = jnp.where(
                    valid_ops[None, :], rho_raw, tau_b)
                admission, execution_weight = angular_compose_parts(
                    rho_compute, tau_b, valid_mask)
                xr = (x_b @ read_b.T).astype(jnp.float32)
                weighted = execution_weight * xr
                out_b = (weighted.astype(jnp.bfloat16) @ write_b).astype(
                    jnp.float32)
                den_b = admission.sum(axis=-1, keepdims=True)
                flat_raw_out = flat_raw_out.at[token_ids, route_ids].add(
                    out_b, mode='drop')
                flat_den_cost = flat_den_cost.at[token_ids, route_ids].add(
                    den_b, mode='drop')
                return (flat_raw_out, flat_den_cost), None

            (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
                sector_step,
                (jnp.zeros((T, R, D), dtype=jnp.float32),
                 jnp.zeros((T, R, 1), dtype=jnp.float32)),
                jnp.arange(n_local_sectors, dtype=jnp.int32))
            return finish(flat_raw_out, flat_den_cost)

        def token_gather_execute(_):
            @jax.checkpoint
            def pair_chunk_step(carry, chunk_i):
                flat_raw_out, flat_den_cost = carry
                start = chunk_i * _token_gather_chunk_size
                offsets = start + jnp.arange(
                    _token_gather_chunk_size, dtype=jnp.int32)
                valid_pairs = offsets < pair_span
                safe_offsets = jnp.minimum(offsets, pair_span - 1)
                pair_ids = jnp.where(valid_pairs, pair_pos[safe_offsets], 0)
                route_ids = pair_ids // int(T)
                token_ids = pair_ids % int(T)

                x_b = flat_x_bf[token_ids]
                h_b = flat_h_pair_bf[pair_ids]
                tau_b = flat_tau_pair[pair_ids]
                sector_ids = top_sector_ids[pair_ids]
                chunk_local_sector_ids = sector_ids - local_sector_offset
                chunk_selected_here = jnp.logical_and(
                    chunk_local_sector_ids >= 0,
                    chunk_local_sector_ids < n_local_sectors)
                safe_sector_ids = jnp.clip(
                    chunk_local_sector_ids, 0,
                    n_local_sectors - 1).astype(jnp.int32)
                op_key_b = op_key_blocks[safe_sector_ids]
                read_b = read_blocks[safe_sector_ids]
                write_b = write_blocks[safe_sector_ids]
                valid_ops = jnp.logical_and(
                    valid_blocks[safe_sector_ids],
                    chunk_selected_here[:, :, None])
                valid_mask = jnp.logical_and(
                    valid_pairs[:, None, None], valid_ops)

                rho_raw = jnp.einsum(
                    'cd,cknd->ckn', h_b, op_key_b).astype(jnp.float32)
                rho_compute = jnp.where(valid_ops, rho_raw, tau_b[:, None, :])
                admission, execution_weight = angular_compose_parts(
                    rho_compute, tau_b[:, None, :], valid_mask)
                xr = jnp.einsum(
                    'cd,cknd->ckn', x_b, read_b).astype(jnp.float32)
                weighted = execution_weight * xr
                out_b = jnp.einsum(
                    'ckn,cknd->cd',
                    weighted.astype(jnp.bfloat16),
                    write_b).astype(jnp.float32)
                den_b = admission.sum(axis=(1, 2), keepdims=False)[:, None]
                flat_raw_out = flat_raw_out.at[token_ids, route_ids].add(
                    out_b, mode='drop')
                flat_den_cost = flat_den_cost.at[token_ids, route_ids].add(
                    den_b, mode='drop')
                return (flat_raw_out, flat_den_cost), None

            (flat_raw_out, flat_den_cost), _ = jax.lax.scan(
                pair_chunk_step,
                (jnp.zeros((T, R, D), dtype=jnp.float32),
                 jnp.zeros((T, R, 1), dtype=jnp.float32)),
                jnp.arange(max_pair_chunks, dtype=jnp.int32))
            return finish(flat_raw_out, flat_den_cost)

        del token_gather_execute
        return bucket_execute(None), sector_diag

    return fused_gate_srw_paired_sector_bucketed_minimal


def make_sharded_srw_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0,
        hardware_sector_execution_enabled=False,
        hardware_sector_debug_token_gather_fallback=False,
        benchmark_runtime_metrics=False,
        operation_space_mode='block_sparse',
        operation_space_routing_mode=None,
        operation_space_execution_mode=None,
        operation_space_execution_backend=None,
        opspace_lanes=8,
        opspace_tiles_per_lane=12,
        opspace_exec_tiles_per_block=1,
        opspace_blocks_per_lane=None,
        opspace_block_size=None,
        opspace_padded_ops=0,
        opspace_tile_size=128,
        opspace_bucket_capacity_factor=1.5,
        opspace_bucket_chunk_size=128,
        opspace_region_capacity_factor=1.25,
        opspace_block_capacity_factor=1.25,
        opspace_assignment_policy='low_regret',
        opspace_high_regret_threshold=0.05,
        opspace_num_regions=32,
        opspace_blocks_per_region=2,
        opspace_operators_per_block=512,
        opspace_visible_regions=4,
        opspace_visible_blocks_per_region=1,
        opspace_region_score_pooling='smoothmax',
        opspace_region_score_temperature=0.25,
        opspace_budget_start_tokens=0.0,
        opspace_budget_mid_tokens=15000000000.0,
        opspace_budget_end_tokens=30000000000.0,
        opspace_K_start=3.8,
        opspace_K_mid=3.0,
        opspace_K_final=2.0,
        opspace_k_exec=5,
        opspace_token_chunk_size=256,
        opspace_T_gate_start=1.0,
        opspace_T_gate_mid=0.5,
        opspace_T_gate_final=0.18,
        opspace_T_lb_start=1.0,
        opspace_T_lb_mid=0.5,
        opspace_T_lb_final=0.20,
        opspace_lb_enabled=True,
        opspace_lb_capacity_factor=1.35,
        opspace_lb_device_capacity_factor=1.25,
        opspace_lb_tile_weight_peak=0.01,
        opspace_lb_device_weight_peak=0.003,
        opspace_lb_warmup_tokens=3000000000.0,
        opspace_lb_peak_start_tokens=8000000000.0,
        opspace_lb_peak_end_tokens=22000000000.0,
        opspace_lb_decay_end_tokens=30000000000.0,
        opspace_lb_final_weight_frac=0.10):
    execution_backend = str(
        operation_space_execution_backend
        if operation_space_execution_backend is not None
        else '').lower()
    if execution_backend == 'dense':
        if int(opspace_visible_blocks_per_region) != 1:
            raise ValueError(
                "Dense Region/Block backend currently requires "
                "visible_blocks_per_region=1.")
        return make_sharded_srw_tau_free_relu_dense_masked_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            lanes=opspace_num_regions,
            tiles_per_lane=opspace_blocks_per_region,
            padded_ops=opspace_padded_ops,
            tile_size=opspace_operators_per_block,
            k_exec=opspace_visible_regions,
            token_chunk_size=opspace_token_chunk_size)
    if execution_backend == 'sparse_region_block':
        return make_sharded_srw_tau_free_relu_region_block_sparse_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            num_regions=opspace_num_regions,
            blocks_per_region=opspace_blocks_per_region,
            operators_per_block=opspace_operators_per_block,
            visible_regions=opspace_visible_regions,
            visible_blocks_per_region=opspace_visible_blocks_per_region,
            region_score_pooling=opspace_region_score_pooling,
            region_score_temperature=opspace_region_score_temperature,
            tile_size=opspace_operators_per_block,
            padded_ops=opspace_padded_ops,
            token_chunk_size=opspace_token_chunk_size,
            bucket_capacity_factor=opspace_bucket_capacity_factor,
            bucket_chunk_size=opspace_bucket_chunk_size,
            region_capacity_factor=opspace_region_capacity_factor,
            block_capacity_factor=opspace_block_capacity_factor,
            assignment_policy=opspace_assignment_policy,
            high_regret_threshold=opspace_high_regret_threshold)
    routing_mode = str(
        operation_space_routing_mode
        if operation_space_routing_mode is not None
        else operation_space_mode).lower()
    execution_mode = str(
        operation_space_execution_mode
        if operation_space_execution_mode is not None
        else 'dense_masked').lower()
    if routing_mode == 'region_block':
        return make_sharded_srw_tau_free_relu_region_block_pallas_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            num_regions=opspace_num_regions,
            blocks_per_region=opspace_blocks_per_region,
            operators_per_block=opspace_operators_per_block,
            visible_regions=opspace_visible_regions,
            visible_blocks_per_region=opspace_visible_blocks_per_region,
            region_score_pooling=opspace_region_score_pooling,
            region_score_temperature=opspace_region_score_temperature,
            tile_size=opspace_tile_size,
            padded_ops=opspace_padded_ops,
            token_chunk_size=opspace_token_chunk_size,
            bucket_capacity_factor=opspace_bucket_capacity_factor,
            bucket_chunk_size=opspace_bucket_chunk_size,
            region_capacity_factor=opspace_region_capacity_factor,
            block_capacity_factor=opspace_block_capacity_factor,
            assignment_policy=opspace_assignment_policy,
            high_regret_threshold=opspace_high_regret_threshold)
    if routing_mode == 'factorized_lane_mean':
        if execution_mode == 'dense_masked':
            return make_sharded_srw_tau_free_relu_dense_masked_minimal(
                mesh, max_chunk_size=max_chunk_size,
                dead_exposure_target=dead_exposure_target,
                soft_gate_effective_active_eps=soft_gate_effective_active_eps,
                admission_den_power=admission_den_power,
                admission_den_grad_scale=admission_den_grad_scale,
                lanes=opspace_lanes,
                tiles_per_lane=opspace_tiles_per_lane,
                padded_ops=opspace_padded_ops,
                tile_size=opspace_tile_size,
                k_exec=opspace_k_exec,
                token_chunk_size=opspace_token_chunk_size)
        if execution_mode == 'block_bucketed_dense':
            return make_sharded_srw_tau_free_relu_block_bucketed_dense_minimal(
                mesh, max_chunk_size=max_chunk_size,
                dead_exposure_target=dead_exposure_target,
                soft_gate_effective_active_eps=soft_gate_effective_active_eps,
                admission_den_power=admission_den_power,
                admission_den_grad_scale=admission_den_grad_scale,
                lanes=opspace_lanes,
                tiles_per_lane=opspace_tiles_per_lane,
                padded_ops=opspace_padded_ops,
                tile_size=opspace_tile_size,
                k_exec=opspace_k_exec,
                exec_tiles_per_block=opspace_exec_tiles_per_block,
                blocks_per_lane=opspace_blocks_per_lane,
                block_size=opspace_block_size,
                token_chunk_size=opspace_token_chunk_size)
        raise ValueError(
            "operation_space execution_mode must be 'dense_masked', "
            "or 'block_bucketed_dense' for factorized_lane_mean routing, got "
            f"{execution_mode!r}.")
    if hardware_sector_execution_enabled:
        if hardware_sector_debug_token_gather_fallback:
            return make_sharded_srw_sector_topk_minimal(
                mesh, max_chunk_size=max_chunk_size,
                dead_exposure_target=dead_exposure_target,
                soft_gate_effective_active_eps=soft_gate_effective_active_eps,
                admission_den_power=admission_den_power,
                admission_den_grad_scale=admission_den_grad_scale,
                block_size=block_size,
                top_blocks=top_blocks,
                block_margin=block_margin)
        return make_sharded_srw_sector_bucketed_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            block_size=block_size,
            top_blocks=top_blocks,
            block_margin=block_margin,
            benchmark_runtime_metrics=benchmark_runtime_metrics)
    return make_sharded_srw_block_sparse_minimal(
        mesh, max_chunk_size=max_chunk_size,
        dead_exposure_target=dead_exposure_target,
        soft_gate_effective_active_eps=soft_gate_effective_active_eps,
        admission_den_power=admission_den_power,
        admission_den_grad_scale=admission_den_grad_scale,
        block_size=block_size,
        top_blocks=top_blocks,
        block_margin=block_margin)


def make_sharded_srw_paired_minimal(
        mesh, max_chunk_size=2048,
        dead_exposure_target=0.1,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=1.0,
        admission_den_grad_scale=1.0,
        block_size=256,
        top_blocks=2,
        block_margin=0.0,
        hardware_sector_execution_enabled=False,
        hardware_sector_debug_token_gather_fallback=False,
        benchmark_runtime_metrics=False):
    if hardware_sector_execution_enabled:
        if hardware_sector_debug_token_gather_fallback:
            return make_sharded_srw_paired_sector_topk_minimal(
                mesh, max_chunk_size=max_chunk_size,
                dead_exposure_target=dead_exposure_target,
                soft_gate_effective_active_eps=soft_gate_effective_active_eps,
                admission_den_power=admission_den_power,
                admission_den_grad_scale=admission_den_grad_scale,
                block_size=block_size,
                top_blocks=top_blocks,
                block_margin=block_margin)
        return make_sharded_srw_paired_sector_bucketed_minimal(
            mesh, max_chunk_size=max_chunk_size,
            dead_exposure_target=dead_exposure_target,
            soft_gate_effective_active_eps=soft_gate_effective_active_eps,
            admission_den_power=admission_den_power,
            admission_den_grad_scale=admission_den_grad_scale,
            block_size=block_size,
            top_blocks=top_blocks,
            block_margin=block_margin,
            benchmark_runtime_metrics=benchmark_runtime_metrics)
    return make_sharded_srw_paired_block_sparse_minimal(
        mesh, max_chunk_size=max_chunk_size,
        dead_exposure_target=dead_exposure_target,
        soft_gate_effective_active_eps=soft_gate_effective_active_eps,
        admission_den_power=admission_den_power,
        admission_den_grad_scale=admission_den_grad_scale,
        block_size=block_size,
        top_blocks=top_blocks,
        block_margin=block_margin)


make_sharded_srw_exact_global_sector_topk_replicated_spmd_minimal = (
    make_sharded_srw_sector_topk_minimal)
make_sharded_srw_paired_exact_global_sector_topk_replicated_spmd_minimal = (
    make_sharded_srw_paired_sector_topk_minimal)


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
                "v4168 requires explicit cosine-space tau_init_attn_qk/v/rst; "
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

def _call_minimal_srw(fused, x, h, op_key, raw_tau, read, write,
                      soft_gate_temperature, soft_gate_t_final,
                      soft_gate_boundary_power,
                      soft_gate_boundary_power_final,
                      execution_prune_eps, training_tokens):
    if getattr(fused, '_v4168_accepts_training_tokens', False):
        return fused(
            x, h, op_key, raw_tau, read, write,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps, training_tokens)
    return fused(
        x, h, op_key, raw_tau, read, write,
        soft_gate_temperature, soft_gate_t_final,
        soft_gate_boundary_power, soft_gate_boundary_power_final,
        execution_prune_eps)


def _split_minimal_srw_return(ret):
    zero_sector = jnp.zeros((SECTOR_RUNTIME_DIAG_COUNT,), dtype=jnp.float32)
    zero_opspace = jnp.zeros((OPSPACE_RUNTIME_DIAG_COUNT,), dtype=jnp.float32)
    zero_aux = jnp.float32(0.0)
    if not isinstance(ret, tuple):
        return ret, zero_sector, zero_opspace, zero_aux
    if len(ret) >= 4:
        return ret[0], ret[1], ret[3], ret[2]
    if len(ret) >= 3:
        return ret[0], zero_sector, ret[1], ret[2]
    if len(ret) >= 2:
        return ret[0], ret[1], zero_opspace, zero_aux
    return ret[0], zero_sector, zero_opspace, zero_aux


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
                          execution_prune_eps=0.0,
                          training_tokens=0.0):
    """Minimal v4168 attention path: tau-free opspace or legacy SRW."""
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
    v_op_key = pool_params['attn_v_op_key']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']

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

    opspace_tau_free = (
        isinstance(sharded_fns, dict)
        and bool(sharded_fns.get('operation_space_tau_free', False)))
    if opspace_tau_free:
        raw_tau_QK = jnp.zeros((B, S, 2, 1), dtype=x.dtype)
        raw_tau_Q = raw_tau_QK[:, :, 0, :]
        raw_tau_K = raw_tau_QK[:, :, 1, :]
        raw_tau_V = jnp.zeros((B, S, 1), dtype=x.dtype)
    else:
        raw_tau_all = (
            x @ router_params['raw_tau_attn']['kernel']
            + router_params['raw_tau_attn']['bias'])
        raw_tau_QK = jnp.stack(
            [raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2]], axis=2)
        raw_tau_Q = raw_tau_all[:, :, 0:1]
        raw_tau_K = raw_tau_all[:, :, 1:2]
        raw_tau_V = raw_tau_all[:, :, 2:3]
    qk_scale, v_scale, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers)

    if isinstance(sharded_fns, dict):
        if opspace_tau_free:
            fused_paired = sharded_fns.get('attn_qk_paired_minimal', None)
            fused_single_qk = None
            fused_single_v = sharded_fns.get('attn_v_single_minimal', None)
            if fused_paired is None or fused_single_v is None:
                raise RuntimeError(
                    "operation_space enabled but required tau-free QK/V/RST "
                    "executor is missing")
        else:
            fused_paired = sharded_fns.get(
                'attn_qk_paired_minimal',
                sharded_fns.get(
                    'attn_qk_paired', sharded_fns.get('paired', None)))
            fused_single_qk = None if fused_paired is not None else (
                sharded_fns.get(
                    'attn_qk_single_minimal',
                    sharded_fns.get('attn_qk_single', None)))
            fused_single_v = sharded_fns.get(
                'attn_v_single_minimal',
                sharded_fns.get('attn_v_single', sharded_fns['single']))
    else:
        fused_single_qk = None
        fused_single_v, fused_paired = sharded_fns

    if fused_paired is not None:
        h_QK = jnp.stack([h_Q, h_K], axis=2)
        qk_ret = _call_minimal_srw(
            fused_paired,
            x, h_QK, qk_op_key, raw_tau_QK, qk_read, qk_write,
            soft_gate_T_qk, soft_gate_t_final, soft_gate_boundary_power,
            soft_gate_boundary_power_final, execution_prune_eps,
            training_tokens)
        QK_out = qk_ret[0] if isinstance(qk_ret, tuple) else qk_ret
        Q = QK_out[:, :, 0, :]
        K = QK_out[:, :, 1, :]
    elif fused_single_qk is not None:
        Q = _call_minimal_srw(
            fused_single_qk,
            x, h_Q, qk_op_key, raw_tau_Q, qk_read, qk_write,
            soft_gate_T_qk, soft_gate_t_final, soft_gate_boundary_power,
            soft_gate_boundary_power_final, execution_prune_eps,
            training_tokens)
        K = _call_minimal_srw(
            fused_single_qk,
            x, h_K, qk_op_key, raw_tau_K, qk_read, qk_write,
            soft_gate_T_qk, soft_gate_t_final, soft_gate_boundary_power,
            soft_gate_boundary_power_final, execution_prune_eps,
            training_tokens)
    else:
        raise RuntimeError("v4168 minimal attention requires a Q/K SRW executor.")
    Q = Q * qk_scale
    K = K * qk_scale
    v_ret = _call_minimal_srw(
        fused_single_v,
        x, h_V, v_op_key, raw_tau_V, v_read, v_write,
        soft_gate_T_v, soft_gate_t_final, soft_gate_boundary_power,
        soft_gate_boundary_power_final, execution_prune_eps,
        training_tokens)
    V, v_sector_diag, v_opspace_diag, v_aux = _split_minimal_srw_return(v_ret)
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
    return (safe_dropout(out, dropout_rate, deterministic, rng_out),
            v_sector_diag, v_opspace_diag, v_aux)


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
                         execution_prune_eps=0.0,
                         training_tokens=0.0):
    """Minimal v4168 RST path: tau-free opspace or legacy SRW."""
    del admission_den_power
    if d_model is None or n_layers is None:
        raise ValueError(
            "depth-scaled pool outputs require d_model and n_layers.")
    B, S = x.shape[:2]
    soft_gate_T_rst = (
        soft_gate_temperature if soft_gate_T_rst is None else soft_gate_T_rst)
    pool_params = _ensure_pool_operator_keys(pool_params)
    rst_op_key = pool_params['rst_op_key']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']

    rng, rng_drop = jax.random.split(rng)
    rst_read_query = (
        x @ router_params['proj_rst']['kernel']
        + router_params['proj_rst']['bias'])
    rst_read_query = safe_dropout(
        rst_read_query, router_dropout, deterministic, rng_drop)
    h = _read_write_operator_query(
        rst_read_query, x, router_params['rst_op_write_query_proj'])
    opspace_tau_free = (
        isinstance(sharded_fns, dict)
        and bool(sharded_fns.get('operation_space_tau_free', False)))
    if opspace_tau_free:
        raw_tau = jnp.zeros((B, S, 1), dtype=x.dtype)
    else:
        raw_tau = (
            x @ router_params['raw_tau_rst']['kernel']
            + router_params['raw_tau_rst']['bias'])
    _, _, rst_scale = _pool_output_scales(d_model, n_layers)
    if isinstance(sharded_fns, dict):
        if opspace_tau_free:
            fused_single = sharded_fns.get('rst_single_minimal', None)
            if fused_single is None:
                raise RuntimeError(
                    "operation_space enabled but required tau-free QK/V/RST "
                    "executor is missing")
        else:
            fused_single = sharded_fns.get(
                'rst_single_minimal',
                sharded_fns.get('rst_single', sharded_fns['single']))
    else:
        fused_single, _ = sharded_fns
    rst_ret = _call_minimal_srw(
        fused_single,
        x, h, rst_op_key, raw_tau, rst_read, rst_write,
        soft_gate_T_rst, soft_gate_t_final, soft_gate_boundary_power,
        soft_gate_boundary_power_final, execution_prune_eps,
        training_tokens)
    out, rst_sector_diag, rst_opspace_diag, rst_aux = (
        _split_minimal_srw_return(rst_ret))
    out = out * rst_scale
    rng, rng_out = jax.random.split(rng)
    return (safe_dropout(out, dropout_rate, deterministic, rng_out),
            rst_sector_diag, rst_opspace_diag, rst_aux)


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
    """v4168 dense diagnostic path. sharded_fns=(fused_single, fused_paired) required.

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
    v_op_key = pool_params['attn_v_op_key']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']

    # RW-derived operator keys are passed into the sharded SRW closure.
    # The closure forward-normalizes them for selection stability.
    qk_op_key_unit = qk_op_key
    v_op_key_unit = v_op_key

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
    v_sparsity_start = v_exposure_start + DEAD_EXPOSURE_DIAG_COUNT
    v_sparsity_diag = v_ret[v_sparsity_start]
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
                  admission_den_power=1.0,
                  execution_prune_eps=0.0):
    """v4168 dense diagnostic path. sharded_fns=(fused_single, fused_paired) required.

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
    rst_op_key_unit = rst_op_key
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

class DAWN_SRW_V4168(nn.Module):
    """DAWN-SRW v4.1.6.8 with Block-sparse minimal SRW execution."""
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
    qk_block_size: int = 256
    v_block_size: int = 256
    rst_block_size: int = 256
    qk_top_blocks: int = 2
    v_top_blocks: int = 2
    rst_top_blocks: int = 2
    block_margin: float = 0.0
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
                 minimal_train=False,
                 benchmark_runtime_metrics=False,
                 training_tokens=0.0):
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
                    (attn_out, attn_v_sector_diag, attn_v_opspace_diag,
                     attn_v_opspace_aux) = _attn_forward_minimal(
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
                        execution_prune_eps=execution_prune_eps,
                        training_tokens=training_tokens)
                    x = x + attn_out

                    normed = _layer_norm(
                        x, bp['norm2']['scale'], bp['norm2']['bias'])
                    (rst_out, rst_sector_diag, rst_opspace_diag,
                     rst_opspace_aux) = _rst_forward_minimal(
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
                        execution_prune_eps=execution_prune_eps,
                        training_tokens=training_tokens)
                    return x + rst_out, (
                        attn_v_sector_diag, rst_sector_diag,
                        attn_v_opspace_diag, rst_opspace_diag,
                        attn_v_opspace_aux, rst_opspace_aux)

                if self.gradient_checkpointing:
                    scan_body_minimal = jax.checkpoint(scan_body_minimal)

                xs_minimal = {
                    'params': stacked,
                    'rng': layer_rngs,
                }
                x, sector_ys = jax.lax.scan(scan_body_minimal, x, xs_minimal)
                attn_v_sector_diag = jnp.mean(sector_ys[0], axis=0)
                rst_sector_diag = jnp.mean(sector_ys[1], axis=0)
                attn_v_opspace_diag = jnp.mean(sector_ys[2], axis=0)
                rst_opspace_diag = jnp.mean(sector_ys[3], axis=0)
                attn_v_opspace_aux = jax.tree.map(
                    lambda v: jnp.mean(v, axis=0), sector_ys[4])
                rst_opspace_aux = jax.tree.map(
                    lambda v: jnp.mean(v, axis=0), sector_ys[5])
                del sector_ys
                opspace_aux = (
                    _opspace_aux_loss(attn_v_opspace_aux)
                    + _opspace_aux_loss(rst_opspace_aux))
                x = self.norm(x)
                if labels is None:
                    return {'logits': self.token_emb.attend(x)}

                embedding_matrix = self.token_emb.embedding
                shift_x = x[:, :-1, :]
                shift_labels = labels[:, 1:].astype(jnp.int32)
                valid_mask = (shift_labels != -100)

                loss, correct, valid_count = _chunked_ce_loss_and_acc(
                    shift_x, embedding_matrix, shift_labels, valid_mask)
                sector_metrics = {}
                if benchmark_runtime_metrics:
                    sector_metrics.update(_sector_benchmark_runtime_metric_dict(
                        'sector/attn_v', attn_v_sector_diag))
                    sector_metrics.update(_sector_benchmark_runtime_metric_dict(
                        'sector/rst', rst_sector_diag))
                else:
                    sector_metrics.update(_sector_core_runtime_metric_dict(
                        'sector/attn_v', attn_v_sector_diag))
                    sector_metrics.update(_sector_core_runtime_metric_dict(
                        'sector/rst', rst_sector_diag))
                opspace_metrics = {}
                opspace_metrics.update(_opspace_runtime_metric_dict(
                    'opspace/attn_v', attn_v_opspace_diag))
                opspace_metrics.update(_opspace_runtime_metric_dict(
                    'opspace/rst', rst_opspace_diag))
                opspace_metrics.update(_opspace_final_runtime_metric_dict(
                    'opspace/attn_v', attn_v_opspace_aux))
                opspace_metrics.update(_opspace_final_runtime_metric_dict(
                    'opspace/rst', rst_opspace_aux))
                return {
                    'loss': loss,
                    'correct': correct,
                    'valid_count': valid_count,
                    'aux_loss': opspace_aux,
                    **sector_metrics,
                    **opspace_metrics,
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
            'qk_block_size': self.qk_block_size,
            'v_block_size': self.v_block_size,
            'rst_block_size': self.rst_block_size,
            'qk_top_blocks': self.qk_top_blocks,
            'v_top_blocks': self.v_top_blocks,
            'rst_top_blocks': self.rst_top_blocks,
            'block_margin': self.block_margin,
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
        ]


DAWN = DAWN_SRW_V4168


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
    """Extract v4166 drive boundary settings for inference."""
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


def _tau_init_calibration_scores(params, input_ids, max_tokens=128):
    """Sample fresh-init cosine scores without changing forward semantics.

    The sample uses the first block's freshly initialized normalized route
    states and the shared v4166 router/pool parameters. Rho follows the
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
    return {
        'q': _selection_rho(h_q, qk_op_key),
        'k': _selection_rho(h_k, qk_op_key),
        'v': _selection_rho(h_v, v_op_key),
        'rst': _selection_rho(h_rst, rst_op_key),
    }


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
    """Canonical v4166 execution_weight for non-sharded inference helpers."""
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
