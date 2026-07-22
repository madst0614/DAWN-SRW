"""DAWN-SRW v4.1.7.4: operator fields over shared local state spaces.

An operation space is an independent local state space.  Given a shared state
``x``, space ``m`` owns exactly one encoder ``P_m`` and decoder ``U_m``::

    z_m = P_m x
    y_r = sum_m alpha[r, m] U_m F[r, m](z_m, c_r)

Q, K, V, and RST consume the same ``z_m``.  They differ in their global
operator query, RW bank/gate/tau, composition, and downstream use.  A space is
selected from the distribution of its live generalized-bilinear RW operator
keys, never from a learned representative key.  Production routing uses a
deterministic FAVOR+-style positive orthogonal sketch; exact log-mean-exp is
kept as the analysis reference.

Semantic routing is hard top-k.  Physical execution remains all-space dense.
The fused decoder contracts ``[M,T,R]`` directly into ``[T,D]`` and never
materializes ``[M,T,D]``.  The canonical 400M-equivalent geometry is
``D=2048, M=8, R=256``.  The stacked encoder is initialized orthogonally and
the independent trainable decoder starts as its transpose.

This is the sole v4.1.7.4 architecture.  Earlier checkpoint, optimizer,
parameter-path, and config schemas are intentionally unsupported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache, wraps
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from models.dawn_srw_v4173 import (
    DAWNBlock as _SharedDAWNBlock,
    DEFAULT_ADMISSION_DEN_POWER,
    DEFAULT_HEAT_KERNEL_BETA,
    DEFAULT_SRW_COMPOSITION_MODE,
    OPERATOR_KEY_MODE,
    RW_FORWARD_NORM_EPS,
    _composition_den as _shared_composition_den,
    _composition_den_floor_mass as _shared_composition_den_floor_mass,
    _compute_admission_drive as _shared_compute_admission_drive,
    _forward_unit_direction as _shared_forward_unit_direction,
    _layer_norm as _shared_layer_norm,
    _pool_output_scales as _shared_pool_output_scales,
    _raw_tau_init_from_cosine_tau,
    _tau_from_param as _shared_tau_from_param,
    _validate_v4173_sharded_fns as _validate_shared_rw_sharded_fns,
    generalized_bilinear_operator_key_diagnostics as
        _shared_operator_key_diagnostics,
    make_sharded_rst_multispace_dense_diagnostics as
        _shared_make_space_dense_diagnostics,
    make_sharded_rst_multispace_dense_minimal as
        _shared_make_space_dense_minimal,
    make_sharded_srw as _shared_make_sharded_srw,
    make_sharded_srw_diagnostics_minimal as
        _shared_make_sharded_srw_diagnostics_minimal,
    make_sharded_srw_minimal as _shared_make_sharded_srw_minimal,
    make_sharded_srw_paired as _shared_make_sharded_srw_paired,
    make_sharded_srw_paired_diagnostics_minimal as
        _shared_make_sharded_srw_paired_diagnostics_minimal,
    make_sharded_srw_paired_minimal as
        _shared_make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_retention_minimal as
        _shared_make_sharded_srw_paired_retention_minimal,
    make_sharded_srw_paired_suppression_minimal as
        _shared_make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_paired_trajectory_minimal as
        _shared_make_sharded_srw_paired_trajectory_minimal,
    make_sharded_srw_retention_minimal as
        _shared_make_sharded_srw_retention_minimal,
    make_sharded_srw_suppression_minimal as
        _shared_make_sharded_srw_suppression_minimal,
    make_sharded_srw_trajectory_minimal as
        _shared_make_sharded_srw_trajectory_minimal,
    materialize_generalized_bilinear_operator_keys as
        _shared_materialize_operator_keys,
    safe_dropout as _shared_safe_dropout,
    scaled_normal as _shared_scaled_normal,
    unit_norm_init as _shared_unit_norm_init,
)


MODEL_VERSION = "spatial-r1-v4.1.7.4"
ROUTES = ("q", "k", "v", "rst")
SPACE_KERNEL_FEATURE_MULTIPLIER = 2
SPACE_KERNEL_SEED = 4174
SPACE_KERNEL_EPS = 1.0e-20
SPACE_METRIC_SUFFIXES = (
    "dominant_space_id",
    "space_top1_rate",
    "space_usage_min",
    "space_usage_max",
    "space_usage_std",
    "space_selected_weight_top1",
    "space_selected_entropy",
    "space_dead_frac",
)


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"model.{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_operation_space_config(model_cfg: Mapping[str, Any]) -> tuple[int, int]:
    """Validate the only supported v4174 operation-space schema."""
    n_spaces = _positive_int(
        "n_operation_spaces", model_cfg.get("n_operation_spaces"))
    top_k = _positive_int(
        "operation_space_top_k", model_cfg.get("operation_space_top_k"))
    if top_k > n_spaces:
        raise ValueError(
            "model.operation_space_top_k must be <= model.n_operation_spaces, "
            f"got {top_k} > {n_spaces}")
    return n_spaces, top_k


def materialize_operation_space_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and materialize the canonical fresh-run schema in-place."""
    n_spaces, top_k = resolve_operation_space_config(model_cfg)
    model_cfg["n_operation_spaces"] = n_spaces
    model_cfg["operation_space_top_k"] = top_k
    d_model = _positive_int("d_model", model_cfg.get("d_model"))
    d_route = _positive_int("d_route", model_cfg.get("d_route"))
    if d_model != n_spaces * d_route:
        raise ValueError(
            "v4174 requires model.d_model == model.n_operation_spaces * "
            f"model.d_route, got {d_model} != {n_spaces} * {d_route}")
    for pool_name in ("n_qk", "n_v", "n_rst"):
        value = _positive_int(pool_name, model_cfg.get(
            pool_name, model_cfg.get("n_know") if pool_name == "n_rst" else None))
        if value % n_spaces:
            raise ValueError(
                f"model.{pool_name} must be divisible by "
                f"model.n_operation_spaces, got {value} % {n_spaces}")
    for name in (
            "space_kernel_beta_qk", "space_kernel_beta_v",
            "space_kernel_beta_rst"):
        value = model_cfg.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"model.{name} must be a materialized positive float")
        model_cfg[name] = float(value)
    return model_cfg


def symbolic_parameter_count(model_cfg: Mapping[str, Any]) -> dict[str, int]:
    """Count the exact canonical v4174 parameter tree."""
    if "model" in model_cfg and isinstance(model_cfg["model"], Mapping):
        model_cfg = model_cfg["model"]
    cfg = dict(model_cfg)
    n_spaces, _ = resolve_operation_space_config(cfg)
    vocab = _positive_int("vocab_size_padded", cfg.get(
        "vocab_size_padded", cfg.get("vocab_size")))
    max_seq = _positive_int("max_seq_len", cfg.get("max_seq_len"))
    d_model = _positive_int("d_model", cfg.get("d_model"))
    d_route = _positive_int("d_route", cfg.get("d_route"))
    n_layers = _positive_int("n_layers", cfg.get("n_layers"))
    if d_model != n_spaces * d_route:
        raise ValueError("v4174 parameter count requires D == M * R")
    counts_by_pool = {
        name: _positive_int(name, cfg.get(
            name, cfg.get("n_know") if name == "n_rst" else None))
        for name in ("n_qk", "n_v", "n_rst")}
    for name, value in counts_by_pool.items():
        if value % n_spaces:
            raise ValueError(f"model.{name} must be divisible by M")
    shared_space_state = 2 * n_spaces * d_model * d_route
    global_route_queries = 4 * d_model * d_route
    route_tau = 4 * (d_route + 1)
    counts = {
        "token_embedding": vocab * d_model,
        "position_embedding": max_seq * d_model,
        "layer_stack": n_layers * (d_model * d_model + 4 * d_model),
        "router": shared_space_state + global_route_queries + route_tau,
        "shared_space_state": shared_space_state,
        "global_route_queries": global_route_queries,
        "read_write_pools": sum(counts_by_pool.values()) * 2 * d_route,
        "learned_key_tables": 0,
        "bilinear_probe_matrices": 2 * d_route * d_route,
        "final_norm": 2 * d_model,
    }
    counts["total"] = sum(
        value for key, value in counts.items()
        if key not in ("shared_space_state", "global_route_queries"))
    return counts


def _orthogonal_space_projection_init(
        key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
    """Initialize ``[M,D,R]`` from row blocks of one orthogonal ``[D,D]``."""
    m, d_model, d_route = map(int, shape)
    if m * d_route != d_model:
        raise ValueError("orthogonal block initialization requires M * R == D")
    orthogonal = nn.initializers.orthogonal(scale=1.0)(
        key, (d_model, d_model), jnp.float32)
    blocks = orthogonal.T.reshape((m, d_route, d_model))
    return jnp.swapaxes(blocks, -1, -2).astype(dtype)


def _space_operator_read_init(
        key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
    """Initialize each space's live RW field around a distinct ORF direction."""
    m, n_operator, d_route = map(int, shape)
    centers = _positive_orthogonal_projection(d_route)[:m]
    centers = _shared_forward_unit_direction(centers)
    noise = (jnp.float32(0.005) / math.sqrt(d_route)) * jax.random.normal(
        key, (m, n_operator, d_route), dtype=jnp.float32)
    return _shared_forward_unit_direction(
        centers[:, None, :] + noise).astype(dtype)


def _space_operator_write_init(
        key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
    """Use a positive carrier so bilinear keys retain the space field center."""
    m, n_operator, d_route = map(int, shape)
    carrier = jnp.full(
        (m, n_operator, d_route), 1.0 / math.sqrt(d_route), jnp.float32)
    noise = (jnp.float32(0.0005) / math.sqrt(d_route)) * jax.random.normal(
        key, (m, n_operator, d_route), dtype=jnp.float32)
    return _shared_forward_unit_direction(carrier + noise).astype(dtype)


def _identity_probe_init(
        key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
    del key
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator-key probes must be square")
    return jnp.eye(shape[0], dtype=dtype)


class OperationSpaceNeuronPool(nn.Module):
    """Space-indexed RW banks; only the per-space operator axis is sharded."""
    n_qk: int
    n_v: int
    n_rst: int
    d_route: int
    n_operation_spaces: int

    def setup(self):
        m, r = int(self.n_operation_spaces), int(self.d_route)
        for name, count in (
                ("qk", self.n_qk), ("v", self.n_v), ("rst", self.n_rst)):
            shape = (m, int(count) // m, r)
            setattr(self, f"{name}_read_vectors", self.param(
                f"{name}_read_vectors", _space_operator_read_init, shape))
            setattr(self, f"{name}_write_vectors", self.param(
                f"{name}_write_vectors", _space_operator_write_init, shape))
        self.operator_key_read_probe = self.param(
            "operator_key_read_probe", _identity_probe_init, (r, r))
        self.operator_key_write_probe = self.param(
            "operator_key_write_probe", _identity_probe_init, (r, r))


class OperationSpaceRouter(nn.Module):
    """Shared local-state basis plus global route queries and route tau."""
    d_model: int
    d_route: int
    n_operation_spaces: int
    tau_init_attn_qk: float
    tau_init_attn_v: float
    tau_init_rst: float

    def setup(self):
        m, d, r = (int(self.n_operation_spaces), int(self.d_model),
                   int(self.d_route))
        self.space_state_proj = self.param(
            "space_state_proj", _orthogonal_space_projection_init, (m, d, r))
        self.space_state_writeback = self.param(
            "space_state_writeback",
            lambda key, shape, dtype=jnp.float32: jnp.swapaxes(
                self.space_state_proj, -1, -2).astype(dtype),
            (m, r, d))
        tau_values = {
            "q": self.tau_init_attn_qk,
            "k": self.tau_init_attn_qk,
            "v": self.tau_init_attn_v,
            "rst": self.tau_init_rst,
        }
        for route in ROUTES:
            setattr(self, f"{route}_operator_query_proj", nn.Dense(
                r, use_bias=False, name=f"{route}_operator_query_proj",
                kernel_init=nn.initializers.orthogonal(scale=1.0)))
            tau_value = min(max(float(tau_values[route]), -0.9998), 0.9998)
            probability = (tau_value + 1.0) * 0.5
            raw_tau = math.log(probability) - math.log1p(-probability)
            setattr(self, f"{route}_operator_tau_proj", nn.Dense(
                1, name=f"{route}_operator_tau_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=lambda key, shape, dtype, value=raw_tau: jnp.full(
                    shape, value, dtype)))


def _linear(params: Mapping[str, jax.Array], state: jax.Array) -> jax.Array:
    return state @ params["kernel"] + params.get("bias", 0.0)


def _unit_normalize(value: jax.Array, eps: float = 1.0e-8) -> jax.Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    return value / jnp.maximum(
        jnp.linalg.norm(value, axis=-1, keepdims=True), jnp.float32(eps))


def _project_space_local_states(
        state: jax.Array, space_state_proj: jax.Array) -> jax.Array:
    """Project shared ``[T,D]`` state once into canonical ``[M,T,R]``."""
    if state.ndim != 2 or space_state_proj.ndim != 3:
        raise ValueError("state/projection must have shapes [T,D] and [M,D,R]")
    return jnp.einsum("td,mdr->mtr", state, space_state_proj)


@lru_cache(maxsize=None)
def _positive_orthogonal_projection(
        d_route: int, seed: int = SPACE_KERNEL_SEED) -> jax.Array:
    """Return deterministic paired orthogonal Gaussian directions ``[2R,R]``."""
    d_route = int(d_route)
    rng = np.random.default_rng(int(seed))
    matrix = rng.standard_normal((d_route, d_route), dtype=np.float64)
    q, r = np.linalg.qr(matrix)
    q = q * np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    rows = np.concatenate((q.T, -q.T), axis=0) * math.sqrt(d_route)
    return jnp.asarray(rows, dtype=jnp.float32)


def _build_positive_kernel_features(
        vectors: jax.Array, beta: float,
        projection: Optional[jax.Array] = None) -> jax.Array:
    """Positive orthogonal features approximating ``exp(beta q^T k)``."""
    vectors = jnp.asarray(vectors, dtype=jnp.float32)
    if projection is None:
        projection = _positive_orthogonal_projection(int(vectors.shape[-1]))
    projection = jnp.asarray(projection, dtype=jnp.float32)
    beta_array = jnp.asarray(beta, dtype=jnp.float32)
    log_features = (
        jnp.sqrt(beta_array) * jnp.einsum(
            "...r,hr->...h", vectors, projection)
        - 0.5 * beta_array * jnp.sum(
            jnp.square(vectors), axis=-1, keepdims=True)
        - 0.5 * jnp.log(jnp.float32(projection.shape[0])))
    return jnp.exp(jnp.clip(log_features, -80.0, 80.0))


def _build_space_kernel_sketches(
        operator_keys: Mapping[str, jax.Array],
        betas: Mapping[str, float]) -> dict[str, jax.Array]:
    """Materialize one live positive operator-field sketch per pool and space."""
    result = {}
    for pool_name in ("qk", "v", "rst"):
        keys = _unit_normalize(
            operator_keys[f"{pool_name}_operator_keys"])
        features = _build_positive_kernel_features(
            keys, float(betas[pool_name]))
        result[f"{pool_name}_space_kernel_sketch"] = features.mean(axis=-2)
    return result


def _exact_space_log_field(
        operator_query: jax.Array, operator_keys: jax.Array, beta: float,
        valid_operator_mask: Optional[jax.Array] = None) -> jax.Array:
    """Reference log-mean-exp field over each space's valid operators."""
    query = _unit_normalize(operator_query)
    keys = _unit_normalize(operator_keys)
    cosine = jnp.einsum("...tr,mnr->...tmn", query, keys)
    if valid_operator_mask is None:
        valid_operator_mask = jnp.ones(keys.shape[:-1], dtype=jnp.bool_)
    mask = jnp.asarray(valid_operator_mask, dtype=jnp.bool_)
    valid_count = jnp.maximum(mask.sum(axis=-1), 1)
    values = jnp.where(
        mask, jnp.float32(beta) * (cosine - 1.0), -jnp.inf)
    return (jax.scipy.special.logsumexp(values, axis=-1)
            - jnp.log(valid_count.astype(jnp.float32)))


def _sketch_space_log_field(
        operator_query: jax.Array, space_kernel_sketch: jax.Array,
        beta: float, eps: float = SPACE_KERNEL_EPS) -> jax.Array:
    """Production positive-sketch log field, returned as ``[...,T,M]``."""
    query = _unit_normalize(operator_query)
    query_features = _build_positive_kernel_features(query, float(beta))
    density = jnp.einsum(
        "...th,mh->...tm", query_features,
        jnp.asarray(space_kernel_sketch, dtype=jnp.float32))
    return jnp.log(jnp.maximum(density, jnp.float32(eps))) - jnp.float32(beta)


def _select_operation_spaces(
        space_log_scores: jax.Array,
        operation_space_top_k: int) -> tuple[jax.Array, jax.Array]:
    """Hard top-k IDs with differentiable selected-score softmax weights."""
    selected_scores, selected_ids = jax.lax.top_k(
        space_log_scores, int(operation_space_top_k))
    return selected_ids, jax.nn.softmax(selected_scores, axis=-1)


def _dense_space_weights(
        selected_space_ids: jax.Array, selected_space_weights: jax.Array,
        n_operation_spaces: int) -> jax.Array:
    """Scatter selected semantic weights into exact-zero dense ``[T,M]``."""
    return jnp.sum(
        jax.nn.one_hot(
            selected_space_ids, int(n_operation_spaces),
            dtype=selected_space_weights.dtype)
        * selected_space_weights[..., None], axis=-2).reshape(
            (-1, int(n_operation_spaces)))


def _space_weighted_writeback(
        space_results: jax.Array, dense_space_weights: jax.Array,
        space_state_writeback: jax.Array,
        route_scale: jax.Array | float) -> jax.Array:
    """Fuse semantic weighting and shared decoder without ``[M,T,D]``."""
    if space_results.ndim != 3:
        raise ValueError("space_results must have shape [M,T,R]")
    weighted_local = (
        space_results
        * jnp.swapaxes(dense_space_weights, 0, 1)[..., None]
        * jnp.asarray(route_scale))
    return jnp.einsum(
        "mtr,mrd->td", weighted_local, space_state_writeback)


def _materialize_operator_keys(
        read_vectors: jax.Array, write_vectors: jax.Array,
        read_probe: jax.Array, write_probe: jax.Array) -> jax.Array:
    """Generate unit operator keys with live gradients to both RW vectors."""
    if read_vectors.ndim != 3:
        raise ValueError("space-indexed RW vectors must have shape [M,N,R]")
    return jax.vmap(lambda read, write: _shared_materialize_operator_keys(
        read, write, read_probe, write_probe))(read_vectors, write_vectors)


def _space_read_scalar(
        space_local_states: jax.Array, read_vectors: jax.Array) -> jax.Array:
    """Compute ``r_i^T z_m`` for all spaces and operators."""
    read_unit = _shared_forward_unit_direction(
        read_vectors.astype(jnp.float32)).astype(jnp.bfloat16)
    return jnp.einsum(
        "mtr,mnr->mtn", space_local_states.astype(jnp.bfloat16), read_unit
    ).astype(jnp.float32)


def _rw_compose_space_dense(
        operator_query: jax.Array, operator_keys: jax.Array,
        raw_operator_tau: jax.Array, space_local_states: jax.Array,
        read_vectors: jax.Array, write_vectors: jax.Array, *,
        shared_read_scalar: Optional[jax.Array] = None,
        soft_gate_temperature: float, soft_gate_boundary_power: float,
        admission_den_power: float, srw_composition_mode: str,
        heat_kernel_beta: float, execution_prune_eps: float = 0.0,
        max_chunk_size: int = 2048, diagnostics: bool = False):
    """Execute one route across every space while chunking only operators."""
    if operator_query.ndim != 2:
        raise ValueError("operator_query must have shape [T,R]")
    if space_local_states.ndim != 3 or operator_keys.ndim != 3:
        raise ValueError("local states/keys must have shapes [M,T,R]/[M,N,R]")
    m, token_count, d_route = space_local_states.shape
    n_operator = int(operator_keys.shape[1])
    chunk = min(max(1, int(max_chunk_size)), n_operator)
    n_chunks = math.ceil(n_operator / chunk)
    padded = n_chunks * chunk
    padding = ((0, 0), (0, padded - n_operator), (0, 0))
    keys = jnp.pad(operator_keys, padding)
    reads = jnp.pad(read_vectors, padding)
    writes = jnp.pad(write_vectors, padding)
    read_scalars = (None if shared_read_scalar is None else jnp.pad(
        shared_read_scalar, ((0, 0), (0, 0), (0, padded - n_operator))))
    valid = jnp.arange(padded) < n_operator
    query_unit = _shared_forward_unit_direction(
        operator_query.astype(jnp.float32)).astype(jnp.bfloat16)
    key_unit = _shared_forward_unit_direction(
        keys.astype(jnp.float32)).astype(jnp.bfloat16)
    read_unit = _shared_forward_unit_direction(
        reads.astype(jnp.float32)).astype(jnp.bfloat16)
    write_unit = _shared_forward_unit_direction(
        writes.astype(jnp.float32)).astype(jnp.bfloat16)
    tau = _shared_tau_from_param(raw_operator_tau)
    aggregate_shape = (m, token_count, 1)
    carry = (
        jnp.zeros((m, token_count, d_route), dtype=jnp.float32),
        jnp.zeros(aggregate_shape, dtype=jnp.float32),
        jnp.zeros(aggregate_shape, dtype=jnp.float32),
        jnp.zeros(aggregate_shape, dtype=jnp.float32),
        jnp.zeros(aggregate_shape, dtype=jnp.float32),
        jnp.zeros(aggregate_shape, dtype=jnp.float32),
    )

    def step(carry_value, chunk_index):
        raw_out, gate_mass, gate_sq, gate_max, active_count, depth_sum = carry_value
        start = chunk_index * chunk
        key = jax.lax.dynamic_slice_in_dim(key_unit, start, chunk, axis=1)
        read = jax.lax.dynamic_slice_in_dim(read_unit, start, chunk, axis=1)
        write = jax.lax.dynamic_slice_in_dim(write_unit, start, chunk, axis=1)
        row_valid = jax.lax.dynamic_slice_in_dim(valid, start, chunk, axis=0)
        valid_mask = row_valid[None, None, :]
        rho = jnp.einsum("tr,mnr->mtn", query_unit, key).astype(jnp.float32)
        rho = jnp.where(valid_mask, rho, tau)
        margin, gate, depth, execution_weight, _ = _shared_compute_admission_drive(
            rho, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=jnp.float32(1.0e-6),
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=srw_composition_mode,
            heat_kernel_beta=jnp.float32(heat_kernel_beta))
        gate = jnp.where(valid_mask, gate, 0.0)
        depth = jnp.where(valid_mask, depth, 0.0)
        execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
        if read_scalars is None:
            read_value = jnp.einsum(
                "mtr,mnr->mtn", space_local_states.astype(jnp.bfloat16), read
            ).astype(jnp.float32)
        else:
            read_value = jax.lax.dynamic_slice_in_dim(
                read_scalars, start, chunk, axis=2).astype(jnp.float32)
        chunk_out = jnp.einsum(
            "mtn,mnr->mtr",
            (execution_weight * read_value).astype(jnp.bfloat16), write
        ).astype(jnp.float32)
        return (
            raw_out + chunk_out,
            gate_mass + gate.sum(axis=-1, keepdims=True),
            gate_sq + jnp.square(gate).sum(axis=-1, keepdims=True),
            jnp.maximum(gate_max, gate.max(axis=-1, keepdims=True)),
            active_count + ((margin > 0) & valid_mask).sum(
                axis=-1, keepdims=True).astype(jnp.float32),
            depth_sum + depth.sum(axis=-1, keepdims=True),
        ), None

    (raw_out, gate_mass, gate_sq, gate_max, active_count,
     depth_sum), _ = jax.lax.scan(step, carry, jnp.arange(n_chunks))
    gate_den = _shared_composition_den(
        gate_mass, jnp.float32(admission_den_power), srw_composition_mode)
    space_results = (raw_out / gate_den).astype(jnp.bfloat16).astype(jnp.float32)
    if not diagnostics:
        return space_results
    return (space_results, active_count, gate_mass, gate_sq, gate_max,
            depth_sum, tau, gate_den)


def _qk_shared_read_compose(
        q_operator_query: jax.Array, k_operator_query: jax.Array,
        qk_operator_keys: jax.Array, q_raw_tau: jax.Array,
        k_raw_tau: jax.Array, space_local_states: jax.Array,
        qk_read_vectors: jax.Array, qk_write_vectors: jax.Array,
        **execution_kwargs):
    """Execute Q/K independent gates using one shared read-scalar tensor."""
    qk_read_scalar = _space_read_scalar(space_local_states, qk_read_vectors)
    q_result = _rw_compose_space_dense(
        q_operator_query, qk_operator_keys, q_raw_tau, space_local_states,
        qk_read_vectors, qk_write_vectors,
        shared_read_scalar=qk_read_scalar, **execution_kwargs)
    k_result = _rw_compose_space_dense(
        k_operator_query, qk_operator_keys, k_raw_tau, space_local_states,
        qk_read_vectors, qk_write_vectors,
        shared_read_scalar=qk_read_scalar, **execution_kwargs)
    return q_result, k_result


def _space_selector_metrics(
        selected_ids: jax.Array, selected_weights: jax.Array,
        n_operation_spaces: int, route: str) -> dict[str, jax.Array]:
    m = int(n_operation_spaces)
    ids = selected_ids.reshape((-1, selected_ids.shape[-1]))
    weights = selected_weights.reshape((-1, selected_weights.shape[-1]))
    top1_usage = jax.nn.one_hot(
        ids[:, 0], m, dtype=jnp.float32).mean(axis=0)
    inclusion = jax.nn.one_hot(
        ids, m, dtype=jnp.float32).sum(axis=1).mean(axis=0)
    entropy = -(weights * jnp.log(jnp.maximum(weights, 1.0e-8))).sum(axis=-1)
    dominant = jnp.argmax(top1_usage)
    values = {
        "dominant_space_id": dominant.astype(jnp.float32),
        "space_top1_rate": top1_usage[dominant],
        "space_usage_min": inclusion.min(),
        "space_usage_max": inclusion.max(),
        "space_usage_std": inclusion.std(),
        "space_selected_weight_top1": weights[:, 0].mean(),
        "space_selected_entropy": entropy.mean(),
        "space_dead_frac": (inclusion == 0).astype(jnp.float32).mean(),
    }
    return {f"{route}_{name}": jax.lax.stop_gradient(value)
            for name, value in values.items()}


def _space_router_metrics(
        route_assignments: Mapping[str, tuple[jax.Array, jax.Array]],
        n_operation_spaces: int) -> dict[str, jax.Array]:
    """Regular selector diagnostics from already materialized top-k tensors."""
    out = {}
    for route, (ids, weights) in route_assignments.items():
        out.update(_space_selector_metrics(
            ids, weights, n_operation_spaces, route))

    def agreement(left: str, right: str, name: str):
        left_ids, right_ids = (
            route_assignments[left][0], route_assignments[right][0])
        out[f"{name}_dominant_space_agreement"] = jax.lax.stop_gradient(
            (left_ids[..., 0] == right_ids[..., 0]).astype(jnp.float32).mean())
        matches = (left_ids[..., :, None] == right_ids[..., None, :]).sum(
            axis=(-2, -1)).astype(jnp.float32)
        out[f"{name}_topk_set_agreement"] = jax.lax.stop_gradient(
            (matches / float(left_ids.shape[-1])).mean())

    agreement("q", "k", "qk")
    agreement("q", "v", "qv")
    agreement("v", "rst", "v_rst")
    qk_pair = (
        route_assignments["q"][0][..., 0] * int(n_operation_spaces)
        + route_assignments["k"][0][..., 0])
    pair_hist = jax.nn.one_hot(
        qk_pair, int(n_operation_spaces) ** 2, dtype=jnp.float32
    ).reshape((-1, int(n_operation_spaces) ** 2)).mean(axis=0)
    out["topk_pair_concentration"] = jax.lax.stop_gradient(pair_hist.max())
    all_top1 = jnp.concatenate(tuple(
        route_assignments[route][0][..., 0].reshape(-1)
        for route in ROUTES))
    dominant = jnp.argmax(jax.nn.one_hot(
        all_top1, int(n_operation_spaces), dtype=jnp.float32).sum(axis=0))
    second_index = min(1, int(route_assignments["q"][0].shape[-1]) - 1)
    all_second = jnp.concatenate(tuple(
        route_assignments[route][0][..., second_index].reshape(-1)
        for route in ROUTES))
    second_counts = jax.nn.one_hot(
        all_second, int(n_operation_spaces), dtype=jnp.float32).sum(axis=0)
    second_counts = second_counts.at[dominant].set(0.0)
    second_probability = second_counts / jnp.maximum(second_counts.sum(), 1.0)
    out["hub_excluded_second_slot_entropy"] = jax.lax.stop_gradient(
        -jnp.sum(second_probability * jnp.log(jnp.maximum(
            second_probability, 1.0e-8))))
    return out


def _aggregate_operator_diagnostics(
        diagnostics_values: tuple[jax.Array, ...],
        selected_ids: jax.Array, selected_weights: jax.Array,
        n_operators_per_space: int, route: str,
        srw_composition_mode: str,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Reduce space-local gate aggregates on the selected semantic spaces."""
    (space_results, active_count, gate_mass, gate_sq, gate_max,
     depth_sum, tau, gate_den) = diagnostics_values
    ids = selected_ids.reshape((-1, selected_ids.shape[-1]))
    weights = selected_weights.reshape((-1, selected_weights.shape[-1]))

    def selected(value):
        return jnp.take_along_axis(
            jnp.swapaxes(value[..., 0], 0, 1), ids, axis=1)

    active, mass, square, maximum, depth, tau_selected, den = tuple(
        selected(value) for value in (
            active_count, gate_mass, gate_sq, gate_max, depth_sum, tau, gate_den))
    weighted_mean = lambda value: (value * weights).sum(axis=-1).mean()
    floor_mass = _shared_composition_den_floor_mass(srw_composition_mode)
    floor_compare = (jnp.less if srw_composition_mode == "linear_angular"
                     else jnp.less_equal)
    metrics = {
        f"{route}_operator_active_tau_frac": (
            active / float(n_operators_per_space)).mean(),
        f"{route}_operator_active_tau_count": active.sum(axis=-1).mean(),
        f"{route}_operator_tau_mean": tau_selected.mean(),
        f"{route}_operator_gate_mass_mean": weighted_mean(mass),
        f"{route}_operator_gate_den_mean": weighted_mean(den),
        f"{route}_operator_depth_active_mean": weighted_mean(
            depth / jnp.maximum(active, 1.0)),
        f"{route}_operator_gate_eff_n_mean": weighted_mean(
            jnp.square(mass) / jnp.maximum(square, 1.0e-8)),
        f"{route}_operator_top1_gate_frac_mean": weighted_mean(
            maximum / jnp.maximum(mass, 1.0e-8)),
        f"{route}_operator_den_floor_frac": weighted_mean(
            floor_compare(mass, floor_mass).astype(jnp.float32)),
    }
    per_space = {
        f"{route}_per_space_active_frac": (
            active_count[..., 0] / float(n_operators_per_space)).mean(axis=1),
        f"{route}_per_space_active_count": active_count[..., 0].mean(axis=1),
        f"{route}_per_space_tau_mean": tau[..., 0].mean(axis=1),
        f"{route}_per_space_gate_mass": gate_mass[..., 0].mean(axis=1),
        f"{route}_per_space_gate_den": gate_den[..., 0].mean(axis=1),
        f"{route}_per_space_output_norm": jnp.linalg.norm(
            space_results.astype(jnp.float32), axis=-1).mean(axis=1),
    }
    return (
        {key: jax.lax.stop_gradient(value) for key, value in metrics.items()},
        {key: jax.lax.stop_gradient(value) for key, value in per_space.items()})


def _canonical_regular_operator_metrics(
        metrics: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    """Map route-local observations into the trainer's pool-level schema."""
    qk = lambda suffix: jnp.float32(0.5) * (
        metrics[f"q_operator_{suffix}"] + metrics[f"k_operator_{suffix}"])
    out = {
        "attn_qk_admission_mass_mean": qk("gate_mass_mean"),
        "attn_v_admission_mass_mean": metrics["v_operator_gate_mass_mean"],
        "rst_admission_mass_mean": metrics["rst_operator_gate_mass_mean"],
        "attn_qk_composition_den_mean": qk("gate_den_mean"),
        "attn_v_composition_den_mean": metrics["v_operator_gate_den_mean"],
        "rst_composition_den_mean": metrics["rst_operator_gate_den_mean"],
        "attn_qk_composition_den_floor_frac": qk("den_floor_frac"),
        "attn_v_composition_den_floor_frac": metrics[
            "v_operator_den_floor_frac"],
        "rst_composition_den_floor_frac": metrics[
            "rst_operator_den_floor_frac"],
        "attn_qk_pool_scaled_srw_out_norm": jnp.float32(0.5) * (
            metrics["q_route_output_norm"] + metrics["k_route_output_norm"]),
        "attn_v_pool_scaled_srw_out_norm": metrics["v_route_output_norm"],
        "rst_pool_scaled_srw_out_norm": metrics["rst_route_update_norm"],
    }
    return {key: jax.lax.stop_gradient(value) for key, value in out.items()}


def _rank_values(value: jax.Array) -> jax.Array:
    order = jnp.argsort(value, axis=-1)
    return jnp.argsort(order, axis=-1).astype(jnp.float32)


def _correlation(left: jax.Array, right: jax.Array) -> jax.Array:
    left = left.reshape(-1).astype(jnp.float32)
    right = right.reshape(-1).astype(jnp.float32)
    left = left - left.mean()
    right = right - right.mean()
    return jnp.sum(left * right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), 1.0e-8)


def _kernel_sketch_reference_metrics(
        operator_query: jax.Array, operator_keys: jax.Array,
        space_kernel_sketch: jax.Array, beta: float,
        operation_space_top_k: int) -> dict[str, jax.Array]:
    """Compare production sketch scores against the exact operator field."""
    exact = _exact_space_log_field(operator_query, operator_keys, beta)
    sketch = _sketch_space_log_field(
        operator_query, space_kernel_sketch, beta)
    exact_ids, exact_weights = _select_operation_spaces(
        exact, operation_space_top_k)
    sketch_ids, sketch_weights = _select_operation_spaces(
        sketch, operation_space_top_k)
    topk_matches = (exact_ids[..., :, None] == sketch_ids[..., None, :]).sum(
        axis=(-2, -1)).astype(jnp.float32) / float(operation_space_top_k)
    error = jnp.abs(sketch - exact)
    return {
        "pearson": _correlation(sketch, exact),
        "rank_correlation": _correlation(
            _rank_values(sketch), _rank_values(exact)),
        "top1_agreement": (
            sketch_ids[..., 0] == exact_ids[..., 0]).astype(jnp.float32).mean(),
        "topk_set_agreement": topk_matches.mean(),
        "selected_weight_abs_difference": jnp.abs(
            sketch_weights - exact_weights).mean(),
        "maximum_log_score_error": error.max(),
        "mean_log_score_error": error.mean(),
        "all_scores_finite": jnp.all(jnp.isfinite(sketch)).astype(jnp.float32),
    }


def _space_geometry_diagnostics(
        space_state_proj: jax.Array,
        space_state_writeback: jax.Array,
        state: Optional[jax.Array] = None) -> dict[str, jax.Array]:
    """Rare-cadence projection rank, overlap, and local-state geometry."""
    projection = jnp.asarray(space_state_proj, dtype=jnp.float32)
    stacked = jnp.swapaxes(projection, -1, -2).reshape(
        (-1, projection.shape[1]))
    singular = jnp.linalg.svd(stacked, compute_uv=False)
    probability = singular / jnp.maximum(singular.sum(), 1.0e-8)
    effective_rank = jnp.exp(-jnp.sum(
        probability * jnp.log(jnp.maximum(probability, 1.0e-8))))
    basis_overlap = jnp.einsum("mdr,nds->mnrs", projection, projection)
    principal_cosines = jax.vmap(lambda row: jax.vmap(
        lambda cross: jnp.linalg.svd(cross, compute_uv=False))(row))(
            basis_overlap)
    principal_angles = jnp.arccos(jnp.clip(principal_cosines, -1.0, 1.0))
    frobenius = jnp.einsum("mdr,ndr->mn", projection, projection)
    diagonal = jnp.sqrt(jnp.maximum(jnp.diag(frobenius), 1.0e-8))
    overlap = frobenius / (diagonal[:, None] * diagonal[None, :])
    mask = ~jnp.eye(projection.shape[0], dtype=jnp.bool_)
    out = {
        "stacked_projection_singular_values": singular,
        "stacked_projection_rank": jnp.linalg.matrix_rank(stacked),
        "stacked_projection_effective_rank": effective_rank,
        "space_projection_pairwise_overlap": overlap[mask],
        "space_subspace_principal_angles": principal_angles[mask],
        "encoder_writeback_init_error": jnp.max(jnp.abs(
            jnp.swapaxes(projection, -1, -2) - space_state_writeback)),
    }
    if state is not None:
        local = _project_space_local_states(state, projection)
        covariance = jnp.einsum("mtr,nts->mnrs", local, local) / jnp.maximum(
            state.shape[0], 1)
        covariance_norm = jnp.linalg.norm(covariance, axis=(-2, -1))
        covariance_diag = jnp.sqrt(jnp.maximum(
            jnp.diag(covariance_norm), 1.0e-8))
        covariance_overlap = covariance_norm / (
            covariance_diag[:, None] * covariance_diag[None, :])
        variance = jnp.var(local, axis=1).sum(axis=-1)
        out.update({
            "space_local_state_covariance_overlap": covariance_overlap[mask],
            "space_local_state_norm": jnp.linalg.norm(
                local, axis=-1).mean(axis=-1),
            "space_explained_variance": variance / jnp.maximum(
                variance.sum(), 1.0e-8),
        })
    return out


def _operator_field_geometry_diagnostics(
        operator_keys: Mapping[str, jax.Array],
        sketches: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    """Rare-cadence live operator-field geometry by pool and space."""
    out = {}
    for pool in ("qk", "v", "rst"):
        keys = operator_keys[f"{pool}_operator_keys"].astype(jnp.float32)
        covariance = jnp.einsum("mnr,mns->mrs", keys, keys) / keys.shape[1]
        eigenvalues = jnp.linalg.eigvalsh(covariance)
        probability = eigenvalues / jnp.maximum(
            eigenvalues.sum(axis=-1, keepdims=True), 1.0e-8)
        effective_rank = jnp.exp(-jnp.sum(
            probability * jnp.log(jnp.maximum(probability, 1.0e-8)), axis=-1))
        sketch = sketches[f"{pool}_space_kernel_sketch"]
        sketch_similarity = jnp.einsum("mh,nh->mn", sketch, sketch)
        sketch_norm = jnp.linalg.norm(sketch, axis=-1)
        sketch_similarity = sketch_similarity / jnp.maximum(
            sketch_norm[:, None] * sketch_norm[None, :], 1.0e-8)
        sample_count = min(256, int(keys.shape[1]))
        sample_index = jnp.linspace(
            0, int(keys.shape[1]) - 1, sample_count).astype(jnp.int32)
        sampled_keys = keys[:, sample_index]
        cross_similarity = jnp.einsum(
            "mir,njr->mnij", sampled_keys, sampled_keys)
        cross_nearest = cross_similarity.max(axis=(-2, -1))
        cross_mask = ~jnp.eye(keys.shape[0], dtype=jnp.bool_)
        out.update({
            f"{pool}_operator_key_covariance_effective_rank": effective_rank,
            f"{pool}_space_kernel_field_mass": sketch.sum(axis=-1),
            f"{pool}_space_field_pairwise_similarity": sketch_similarity,
            f"{pool}_cross_space_nearest_operator_similarity": cross_nearest[
                cross_mask],
            f"{pool}_operator_key_mean_norm": jnp.linalg.norm(
                keys, axis=-1).mean(axis=-1),
            f"{pool}_operator_key_angular_spread": jnp.std(
                jnp.einsum("mnr,mr->mn", keys, keys.mean(axis=1)), axis=-1),
        })
    return out


class DAWN_SRW_V4174(nn.Module):
    """Canonical shared-local-state, operator-field-routed DAWN-SRW."""
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
    d_route: int = 128
    operator_key_mode: str = OPERATOR_KEY_MODE
    admission_den_power: float = DEFAULT_ADMISSION_DEN_POWER
    admission_den_power_qk: Optional[float] = None
    admission_den_power_v: Optional[float] = None
    admission_den_power_rst: Optional[float] = None
    srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE
    heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA
    n_qk: int = 1581
    n_v: int = 2601
    n_rst: Optional[int] = None
    n_know: Optional[int] = None
    n_operation_spaces: int = 3
    operation_space_top_k: int = 2
    space_kernel_beta_qk: float = 1.0
    space_kernel_beta_v: float = 1.0
    space_kernel_beta_rst: float = 1.0
    router_dropout: float = 0.1
    n_chunks_rst: Optional[int] = None
    n_chunks_know: int = 1
    n_chunks_qk: int = 1
    n_chunks_v: int = 1
    tau_init_attn_qk: Optional[float] = None
    tau_init_attn_v: Optional[float] = None
    tau_init_rst: Optional[float] = None

    def _vocab_sizes(self) -> tuple[int, int]:
        logical = int(self.logical_vocab_size or self.vocab_size)
        embedding = int(self.vocab_size_padded or self.vocab_size)
        if logical <= 0 or embedding < logical:
            raise ValueError(
                f"invalid logical/embedding vocab sizes {logical}/{embedding}")
        return logical, embedding

    def setup(self):
        if self.operator_key_mode != OPERATOR_KEY_MODE:
            raise ValueError(f"v4174 requires operator_key_mode={OPERATOR_KEY_MODE!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        m = _positive_int("n_operation_spaces", self.n_operation_spaces)
        k = _positive_int("operation_space_top_k", self.operation_space_top_k)
        if k > m or self.d_model != m * self.d_route:
            raise ValueError("v4174 requires K <= M and D == M * R")
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        for name, value in (
                ("n_qk", self.n_qk), ("n_v", self.n_v), ("n_rst", n_rst)):
            if int(value) % m:
                raise ValueError(f"{name} must be divisible by n_operation_spaces")
        if any(value is None for value in (
                self.tau_init_attn_qk, self.tau_init_attn_v,
                self.tau_init_rst)):
            raise ValueError("v4174 requires explicit initial operator tau values")
        for name, value in (
                ("qk", self.space_kernel_beta_qk),
                ("v", self.space_kernel_beta_v),
                ("rst", self.space_kernel_beta_rst)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"space kernel beta {name} must be positive")
        _, embedding_vocab = self._vocab_sizes()
        self.token_emb = nn.Embed(
            embedding_vocab, self.d_model,
            embedding_init=_shared_scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model,
            embedding_init=_shared_scaled_normal(0.02))
        self.neuron_pool = OperationSpaceNeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst,
            d_route=self.d_route, n_operation_spaces=m)
        self.router = OperationSpaceRouter(
            d_model=self.d_model, d_route=self.d_route,
            n_operation_spaces=m,
            tau_init_attn_qk=float(self.tau_init_attn_qk),
            tau_init_attn_v=float(self.tau_init_attn_v),
            tau_init_rst=float(self.tau_init_rst))
        self.layers = [
            _SharedDAWNBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                dropout_rate=self.dropout_rate, name=f"block_{index}")
            for index in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def _realize_parameters(self, state: jax.Array) -> None:
        _ = self.neuron_pool.qk_read_vectors
        local = jnp.einsum(
            "bsd,mdr->mbsr", state, self.router.space_state_proj)
        for route in ROUTES:
            _ = getattr(self.router, f"{route}_operator_query_proj")(state)
            _ = getattr(self.router, f"{route}_operator_tau_proj")(local)
        _ = self.router.space_state_writeback
        for layer in self.layers:
            _ = layer.norm1(state)
            _ = layer.norm2(state)
            _ = layer.attn.expand_O(state)
        _ = self.norm(state)

    def __call__(
            self, input_ids, labels=None, attention_mask=None,
            deterministic=False, sharded_fns=None, analysis=False,
            soft_gate_temperature=0.07, soft_gate_t_final=0.07,
            soft_gate_T_qk=None, soft_gate_T_v=None, soft_gate_T_rst=None,
            soft_gate_boundary_power=2.0,
            soft_gate_boundary_power_final=4.0,
            admission_den_power=None, srw_composition_mode=None,
            heat_kernel_beta=None, execution_prune_eps=0.0,
            minimal_train=False, minimal_runtime_profile="training",
            ce_token_chunk_size=32768, compute_accuracy=True,
            **analysis_kwargs):
        """Run the single canonical all-space dense architecture."""
        del (attention_mask, soft_gate_t_final,
             soft_gate_boundary_power_final, admission_den_power,
             heat_kernel_beta, minimal_train, ce_token_chunk_size,
             analysis_kwargs)
        m, top_k = int(self.n_operation_spaces), int(self.operation_space_top_k)
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        composition_mode = str(srw_composition_mode or self.srw_composition_mode)
        den_powers = {
            "qk": float(self.admission_den_power_qk
                        if self.admission_den_power_qk is not None
                        else self.admission_den_power),
            "v": float(self.admission_den_power_v
                       if self.admission_den_power_v is not None
                       else self.admission_den_power),
            "rst": float(self.admission_den_power_rst
                         if self.admission_den_power_rst is not None
                         else self.admission_den_power),
        }
        temperatures = {
            "qk": float(soft_gate_temperature if soft_gate_T_qk is None
                        else soft_gate_T_qk),
            "v": float(soft_gate_temperature if soft_gate_T_v is None
                       else soft_gate_T_v),
            "rst": float(soft_gate_temperature if soft_gate_T_rst is None
                         else soft_gate_T_rst),
        }
        positions = jnp.arange(input_ids.shape[1])[None, :]
        vocab_embed = (sharded_fns.get("vocab_parallel_embedding")
                       if isinstance(sharded_fns, dict) else None)
        state = (vocab_embed(input_ids, self.token_emb.embedding)
                 if vocab_embed is not None else self.token_emb(input_ids))
        state = state + self.pos_emb(positions)
        state = _shared_safe_dropout(
            state, self.dropout_rate, deterministic, self.make_rng("dropout"))
        if self.is_mutable_collection("params"):
            self._realize_parameters(state)
            state = self.norm(state)
            logical, embedding = self._vocab_sizes()
            logits = self.token_emb.attend(state)
            return {"logits": logits[..., :logical]
                    if embedding != logical else logits}

        params = self.variables["params"]
        pool, router = params["neuron_pool"], params["router"]
        operator_keys = _pool_operator_keys(pool)
        betas = {
            "qk": float(self.space_kernel_beta_qk),
            "v": float(self.space_kernel_beta_v),
            "rst": float(self.space_kernel_beta_rst),
        }
        sketches = _build_space_kernel_sketches(operator_keys, betas)
        bank_values = {
            name: (pool[f"{name}_read_vectors"],
                   pool[f"{name}_write_vectors"],
                   operator_keys[f"{name}_operator_keys"])
            for name in ("qk", "v", "rst")}
        qk_scale, v_scale, rst_scale = _shared_pool_output_scales(
            self.d_model, self.n_layers)
        layer_rngs = jax.random.split(self.make_rng("dropout"), self.n_layers)
        diagnostics_enabled = (
            minimal_runtime_profile == "diagnostics" or analysis)
        regular_lists: dict[str, list[jax.Array]] = {}
        analysis_lists: dict[str, list[jax.Array]] = {}
        last_local_input = None

        def append(target, values):
            for key, value in values.items():
                target.setdefault(key, []).append(value)

        def execute(route, query, raw_tau, local_states):
            pool_name = "qk" if route in ("q", "k") else route
            read, write, keys = bank_values[pool_name]
            return _rw_compose_space_dense(
                query, keys, raw_tau, local_states, read, write,
                soft_gate_temperature=temperatures[pool_name],
                soft_gate_boundary_power=soft_gate_boundary_power,
                admission_den_power=den_powers[pool_name],
                srw_composition_mode=composition_mode,
                heat_kernel_beta=self.heat_kernel_beta,
                execution_prune_eps=execution_prune_eps,
                diagnostics=diagnostics_enabled)

        for layer_index, layer in enumerate(self.layers):
            rng, rng_attn, rng_rst = jax.random.split(
                layer_rngs[layer_index], 3)
            normalized = layer.norm1(state)
            batch_size, sequence_length = normalized.shape[:2]
            flat_state = normalized.reshape((-1, self.d_model))
            last_local_input = flat_state
            space_local_states = _project_space_local_states(
                flat_state, router["space_state_proj"])
            queries = {
                route: _shared_forward_unit_direction(_linear(
                    router[f"{route}_operator_query_proj"], flat_state
                ).astype(jnp.float32))
                for route in ("q", "k", "v")}
            tau = {
                route: _linear(
                    router[f"{route}_operator_tau_proj"], space_local_states)
                for route in ROUTES}
            assignments = {}
            dense_weights = {}
            score_values = {}
            for route in ("q", "k", "v"):
                pool_name = "qk" if route in ("q", "k") else route
                scores = _sketch_space_log_field(
                    queries[route],
                    sketches[f"{pool_name}_space_kernel_sketch"],
                    betas[pool_name])
                ids, weights = _select_operation_spaces(scores, top_k)
                assignments[route] = (
                    ids.reshape((batch_size, sequence_length, top_k)),
                    weights.reshape((batch_size, sequence_length, top_k)))
                dense_weights[route] = _dense_space_weights(ids, weights, m)
                score_values[route] = scores

            qk_kwargs = dict(
                soft_gate_temperature=temperatures["qk"],
                soft_gate_boundary_power=soft_gate_boundary_power,
                admission_den_power=den_powers["qk"],
                srw_composition_mode=composition_mode,
                heat_kernel_beta=self.heat_kernel_beta,
                execution_prune_eps=execution_prune_eps,
                diagnostics=diagnostics_enabled)
            q_result, k_result = _qk_shared_read_compose(
                queries["q"], queries["k"], bank_values["qk"][2],
                tau["q"], tau["k"], space_local_states,
                bank_values["qk"][0], bank_values["qk"][1], **qk_kwargs)
            if diagnostics_enabled:
                q_space_outputs, k_space_outputs = q_result[0], k_result[0]
                for route, diag in (("q", q_result), ("k", k_result)):
                    scalar, arrays = _aggregate_operator_diagnostics(
                        diag, *assignments[route], self.n_qk // m,
                        route, composition_mode)
                    append(regular_lists, scalar)
                    if analysis:
                        append(analysis_lists, arrays)
            else:
                q_space_outputs, k_space_outputs = q_result, k_result
            shared_writeback = router["space_state_writeback"]
            q_route_output = _space_weighted_writeback(
                q_space_outputs, dense_weights["q"], shared_writeback,
                qk_scale).reshape(normalized.shape)
            k_route_output = _space_weighted_writeback(
                k_space_outputs, dense_weights["k"], shared_writeback,
                qk_scale).reshape(normalized.shape)

            v_result = execute("v", queries["v"], tau["v"], space_local_states)
            if diagnostics_enabled:
                v_space_outputs = v_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    v_result, *assignments["v"], self.n_v // m,
                    "v", composition_mode)
                append(regular_lists, scalar)
                if analysis:
                    append(analysis_lists, arrays)
            else:
                v_space_outputs = v_result
            v_route_output = _space_weighted_writeback(
                v_space_outputs, dense_weights["v"], shared_writeback,
                v_scale).reshape(normalized.shape)

            d_head = self.d_model // self.n_heads
            query = q_route_output.reshape(
                batch_size, sequence_length, self.n_heads, d_head).transpose(
                    0, 2, 1, 3)
            key = k_route_output.reshape(
                batch_size, sequence_length, self.n_heads, d_head).transpose(
                    0, 2, 1, 3)
            value = v_route_output.reshape(
                batch_size, sequence_length, self.n_heads, d_head).transpose(
                    0, 2, 1, 3)
            attention_scores = jnp.einsum(
                "bhsd,bhtd->bhst", query, key) / jnp.sqrt(jnp.float32(d_head))
            causal = jnp.tril(jnp.ones(
                (sequence_length, sequence_length), dtype=jnp.bool_))
            attention_scores = jnp.where(
                causal, attention_scores,
                jnp.finfo(attention_scores.dtype).min)
            attention_weights = jax.nn.softmax(attention_scores, axis=-1)
            attention_weights = _shared_safe_dropout(
                attention_weights, self.dropout_rate, deterministic, rng_attn)
            attention_output = jnp.einsum(
                "bhst,bhtd->bhsd", attention_weights, value)
            attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
                batch_size, sequence_length, self.d_model)
            attention_output = layer.attn.expand_O(attention_output)
            state = state + _shared_safe_dropout(
                attention_output, self.dropout_rate, deterministic, rng)

            rst_state = layer.norm2(state).reshape((-1, self.d_model))
            rst_query = _shared_forward_unit_direction(_linear(
                router["rst_operator_query_proj"], rst_state).astype(jnp.float32))
            queries["rst"] = rst_query
            rst_scores = _sketch_space_log_field(
                rst_query, sketches["rst_space_kernel_sketch"], betas["rst"])
            rst_ids, rst_weights = _select_operation_spaces(rst_scores, top_k)
            assignments["rst"] = (
                rst_ids.reshape((batch_size, sequence_length, top_k)),
                rst_weights.reshape((batch_size, sequence_length, top_k)))
            dense_weights["rst"] = _dense_space_weights(
                rst_ids, rst_weights, m)
            score_values["rst"] = rst_scores
            rst_result = execute(
                "rst", rst_query, tau["rst"], space_local_states)
            if diagnostics_enabled:
                rst_space_updates = rst_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    rst_result, *assignments["rst"], n_rst // m,
                    "rst", composition_mode)
                append(regular_lists, scalar)
                if analysis:
                    append(analysis_lists, arrays)
            else:
                rst_space_updates = rst_result
            rst_route_update = _space_weighted_writeback(
                rst_space_updates, dense_weights["rst"], shared_writeback,
                rst_scale).reshape(normalized.shape)
            state = state + _shared_safe_dropout(
                rst_route_update, self.dropout_rate, deterministic, rng_rst)

            append(regular_lists, _space_router_metrics(assignments, m))
            if diagnostics_enabled:
                append(regular_lists, {
                    "attn_out_norm": jnp.linalg.norm(
                        attention_output.astype(jnp.float32), axis=-1).mean(),
                    "rst_out_norm": jnp.linalg.norm(
                        rst_route_update.astype(jnp.float32), axis=-1).mean(),
                    "residual_norm": jnp.linalg.norm(
                        state.astype(jnp.float32), axis=-1).mean(),
                    "q_route_output_norm": jnp.linalg.norm(
                        q_route_output.astype(jnp.float32), axis=-1).mean(),
                    "k_route_output_norm": jnp.linalg.norm(
                        k_route_output.astype(jnp.float32), axis=-1).mean(),
                    "v_route_output_norm": jnp.linalg.norm(
                        v_route_output.astype(jnp.float32), axis=-1).mean(),
                    "rst_route_update_norm": jnp.linalg.norm(
                        rst_route_update.astype(jnp.float32), axis=-1).mean(),
                })
            if analysis:
                for route in ROUTES:
                    pool_name = "qk" if route in ("q", "k") else route
                    reference = _kernel_sketch_reference_metrics(
                        queries[route], bank_values[pool_name][2],
                        sketches[f"{pool_name}_space_kernel_sketch"],
                        betas[pool_name], top_k)
                    append(analysis_lists, {
                        f"{route}_kernel_sketch_{name}": value
                        for name, value in reference.items()})

        state = self.norm(state)
        result: dict[str, jax.Array] = {}
        if labels is None:
            argmax_fn = (sharded_fns.get("vocab_argmax")
                         if isinstance(sharded_fns, dict) else None)
            if argmax_fn is not None:
                result["argmax_token_ids"] = argmax_fn(
                    state, self.token_emb.embedding)
            else:
                logical, embedding = self._vocab_sizes()
                logits = self.token_emb.attend(state)
                result["logits"] = (logits[..., :logical]
                                    if embedding != logical else logits)
        else:
            shift_state = state[:, :-1]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid = shift_labels != -100
            vocab_loss = (sharded_fns.get("vocab_ce_loss")
                          if isinstance(sharded_fns, dict) else None)
            if vocab_loss is not None:
                loss = vocab_loss(
                    shift_state, self.token_emb.embedding, shift_labels, valid)
                correct = jnp.int32(0)
            else:
                logical, _ = self._vocab_sizes()
                logits = self.token_emb.attend(shift_state)[..., :logical]
                safe_labels = jnp.where(valid, shift_labels, 0)
                token_loss = -jax.nn.log_softmax(logits).reshape(
                    (-1, logical))[jnp.arange(safe_labels.size),
                                    safe_labels.reshape(-1)]
                loss = (token_loss.reshape(safe_labels.shape) * valid).sum(
                    ) / jnp.maximum(valid.sum(), 1)
                predictions = jnp.argmax(logits, axis=-1)
                correct = ((predictions == safe_labels) & valid).sum().astype(
                    jnp.int32)
            result.update({
                "loss": loss, "aux_loss": jnp.float32(0.0),
                "correct": correct if compute_accuracy else jnp.int32(0),
                "valid_count": valid.sum().astype(jnp.int32),
            })
        for key, values in regular_lists.items():
            result[key] = jnp.stack(values).mean()
        if diagnostics_enabled:
            result.update(_canonical_regular_operator_metrics(result))
            result.update({
                "heat_kernel_beta": jnp.float32(self.heat_kernel_beta),
                "space_kernel_beta_qk": jnp.float32(self.space_kernel_beta_qk),
                "space_kernel_beta_v": jnp.float32(self.space_kernel_beta_v),
                "space_kernel_beta_rst": jnp.float32(self.space_kernel_beta_rst),
            })
        if analysis:
            for key, values in analysis_lists.items():
                result[key] = jnp.stack(values).mean(axis=0)
            result.update(_space_geometry_diagnostics(
                router["space_state_proj"], router["space_state_writeback"],
                last_local_input))
            result.update(_operator_field_geometry_diagnostics(
                operator_keys, sketches))
        return result

    def get_model_info(self) -> list[str]:
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        m = int(self.n_operation_spaces)
        return [
            f"DAWN-SRW v4.1.7.4 ({MODEL_VERSION})",
            "operation space: one shared local state projection and one shared "
            "decoder basis consumed by Q/K/V/RST",
            f"canonical geometry: D={self.d_model}, M={m}, R={self.d_route}; "
            f"D=M*R, top_k={self.operation_space_top_k}",
            "selection: live RW operator-distribution positive kernel field; "
            "the same global route query selects spaces and exact operators",
            "Q/K share RW read/write rows, live keys, local state, and one "
            "read-scalar computation; query/gate/tau remain independent",
            f"operators per space: qk={int(self.n_qk)//m}, "
            f"v={int(self.n_v)//m}, rst={n_rst//m}",
            "execution: semantic hard top-k, physical all-space dense, fused "
            "shared writeback without an [M,T,D] intermediate",
        ]


def _pool_operator_keys(
        pool_params: Mapping[str, jax.Array],
        operator_key_mode: Optional[str] = None) -> dict[str, jax.Array]:
    """Materialize canonical live-gradient QK, V, and RST operator keys."""
    if operator_key_mode not in (None, OPERATOR_KEY_MODE):
        raise ValueError(f"unsupported operator_key_mode={operator_key_mode!r}")
    read_probe = pool_params["operator_key_read_probe"]
    write_probe = pool_params["operator_key_write_probe"]
    return {
        f"{name}_operator_keys": _materialize_operator_keys(
            pool_params[f"{name}_read_vectors"],
            pool_params[f"{name}_write_vectors"], read_probe, write_probe)
        for name in ("qk", "v", "rst")}


def generalized_bilinear_operator_key_diagnostics(
        read_vectors, write_vectors, read_probes, write_probes,
        eps=RW_FORWARD_NORM_EPS):
    """Expose generalized-bilinear key diagnostics for a space-indexed bank."""
    flattened_read = read_vectors.reshape((-1, read_vectors.shape[-1]))
    flattened_write = write_vectors.reshape((-1, write_vectors.shape[-1]))
    return _shared_operator_key_diagnostics(
        flattened_read, flattened_write, read_probes, write_probes, eps)


def _sampled_layer_states(params, input_ids, max_tokens):
    max_tokens = _positive_int("max_tokens", int(max_tokens))
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    token_count = min(max_tokens, int(input_ids.size))
    token_index = jnp.arange(token_count) * int(input_ids.size) // token_count
    positions = token_index % input_ids.shape[1]
    token_ids = input_ids.reshape(-1)[token_index]
    state = (params["token_emb"]["embedding"][token_ids]
             + params["pos_emb"]["embedding"][positions]).astype(jnp.float32)
    block = params["block_0"]
    attention_state = _shared_layer_norm(
        state, block["norm1"]["scale"], block["norm1"]["bias"])
    rst_state = _shared_layer_norm(
        state, block["norm2"]["scale"], block["norm2"]["bias"])
    return attention_state, rst_state


def _tau_init_calibration_scores(params, input_ids, max_tokens=128):
    """Return per-route ``[M,T,N]`` cosine tables for independent tau fits."""
    attention_state, rst_state = _sampled_layer_states(
        params, input_ids, max_tokens)
    router, keys = params["router"], _pool_operator_keys(params["neuron_pool"])

    def score(route, state, pool_name):
        query = _shared_forward_unit_direction(_linear(
            router[f"{route}_operator_query_proj"], state).astype(jnp.float32))
        return jnp.einsum(
            "tr,mnr->mtn", query, keys[f"{pool_name}_operator_keys"])

    return {
        "q": score("q", attention_state, "qk"),
        "k": score("k", attention_state, "qk"),
        "v": score("v", attention_state, "v"),
        "rst": score("rst", rst_state, "rst"),
    }


def calibrate_operator_tau_per_space(
        scores_by_route: Mapping[str, jax.Array], *,
        target_qk_frac: float, target_v_frac: float,
        target_rst_frac: float) -> dict[str, jax.Array]:
    """Compute one exact cosine quantile per route and operation space."""
    targets = {
        "q": float(target_qk_frac), "k": float(target_qk_frac),
        "v": float(target_v_frac), "rst": float(target_rst_frac)}
    result = {}
    for route in ROUTES:
        scores = jnp.asarray(scores_by_route[route], dtype=jnp.float32)
        target = targets[route]
        if scores.ndim != 3 or not 0.0 < target < 1.0:
            raise ValueError("tau calibration requires [M,T,N] and target in (0,1)")
        result[route] = jnp.quantile(
            scores.reshape((scores.shape[0], -1)), 1.0 - target, axis=1)
    return result


def calibrate_space_kernel_betas(
        scores_by_route: Mapping[str, jax.Array], *,
        target_qk_frac: float, target_v_frac: float,
        target_rst_frac: float, eps: float = 1.0e-6) -> dict[str, float]:
    """Calibrate fixed pool betas from near and four-times-broader quantiles."""
    route_and_target = {
        "qk": (("q", "k"), float(target_qk_frac)),
        "v": (("v",), float(target_v_frac)),
        "rst": (("rst",), float(target_rst_frac)),
    }
    result = {}
    for pool, (routes, near_fraction) in route_and_target.items():
        if not 0.0 < near_fraction < 1.0:
            raise ValueError("kernel calibration targets must be in (0,1)")
        values = jnp.concatenate(tuple(
            jnp.asarray(scores_by_route[route], dtype=jnp.float32).reshape(-1)
            for route in routes))
        broad_fraction = min(4.0 * near_fraction, 1.0)
        rho_near = jnp.quantile(values, 1.0 - near_fraction)
        rho_broad = (values.min() if broad_fraction >= 1.0 else
                     jnp.quantile(values, 1.0 - broad_fraction))
        beta = math.log(2.0) / max(
            float(rho_near - rho_broad), float(eps))
        result[f"space_kernel_beta_{pool}"] = float(beta)
    return result


def _query_geometry_diagnostics(params, input_ids, max_tokens=4096):
    scores = _tau_init_calibration_scores(params, input_ids, max_tokens)
    return {
        f"{route}_operator_score_{name}": value
        for route, route_scores in scores.items()
        for name, value in (
            ("mean", route_scores.mean()), ("std", route_scores.std()),
            ("max", route_scores.max()))}


def initialization_diagnostics_from_params(
        params: Mapping[str, Any], input_ids: jax.Array,
        operation_space_top_k: int) -> dict[str, float]:
    """Host-side one-shot orthogonal basis diagnostics."""
    del operation_space_top_k
    attention_state, _ = _sampled_layer_states(params, input_ids, 4096)
    router = params["router"]
    diagnostics = _space_geometry_diagnostics(
        router["space_state_proj"], router["space_state_writeback"],
        attention_state)
    out = {}
    for key, value in diagnostics.items():
        value = jnp.asarray(value, dtype=jnp.float32)
        if value.ndim == 0:
            out[key] = float(value)
        else:
            out[f"{key}_min"] = float(value.min())
            out[f"{key}_max"] = float(value.max())
            out[f"{key}_mean"] = float(value.mean())
    return out


def analysis_forward(params, model_cfg, input_ids, mode="full"):
    """Run the canonical module with rare-cadence diagnostics enabled."""
    del mode
    cfg = dict(model_cfg)
    model = DAWN_SRW_V4174(
        vocab_size=cfg.get("vocab_size", 30000),
        logical_vocab_size=cfg.get("logical_vocab_size"),
        vocab_size_padded=cfg.get("vocab_size_padded"),
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], max_seq_len=cfg.get("max_seq_len", 512),
        dropout_rate=0.0, router_dropout=0.0,
        d_route=cfg["d_route"], n_qk=cfg["n_qk"], n_v=cfg["n_v"],
        n_rst=cfg.get("n_rst", cfg.get("n_know")),
        n_operation_spaces=cfg["n_operation_spaces"],
        operation_space_top_k=cfg["operation_space_top_k"],
        space_kernel_beta_qk=cfg["space_kernel_beta_qk"],
        space_kernel_beta_v=cfg["space_kernel_beta_v"],
        space_kernel_beta_rst=cfg["space_kernel_beta_rst"],
        admission_den_power=cfg.get("admission_den_power", 1.0),
        admission_den_power_qk=cfg.get("admission_den_power_qk"),
        admission_den_power_v=cfg.get("admission_den_power_v"),
        admission_den_power_rst=cfg.get("admission_den_power_rst"),
        srw_composition_mode=cfg.get(
            "srw_composition_mode", DEFAULT_SRW_COMPOSITION_MODE),
        heat_kernel_beta=cfg.get("heat_kernel_beta", DEFAULT_HEAT_KERNEL_BETA),
        tau_init_attn_qk=cfg.get("tau_init_attn_qk", 0.0),
        tau_init_attn_v=cfg.get("tau_init_attn_v", 0.0),
        tau_init_rst=cfg.get("tau_init_rst", 0.0))
    return model.apply(
        {"params": params}, input_ids, deterministic=True, analysis=True,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.PRNGKey(0)})


def _factory_profile_wrapper(factory: Callable, profile: str) -> Callable:
    @wraps(factory)
    def wrapped(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        kernel._v4174_kernel_profile = profile
        return kernel
    return wrapped


# Vocab and generic analysis profiles retain the proven v4173 shard-map
# factories.  Canonical all-space RW execution is expressed directly in the
# model so GSPMD can shard only the per-space operator axis of live parameters.
make_sharded_srw = _shared_make_sharded_srw
make_sharded_srw_paired = _shared_make_sharded_srw_paired
make_sharded_srw_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_minimal, "production")
make_sharded_srw_paired_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_paired_minimal, "production")
make_sharded_srw_diagnostics_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_diagnostics_minimal, "production_diagnostics")
make_sharded_srw_paired_diagnostics_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_paired_diagnostics_minimal,
    "production_diagnostics")
make_sharded_srw_retention_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_retention_minimal, "retention")
make_sharded_srw_paired_retention_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_paired_retention_minimal, "retention")
make_sharded_srw_suppression_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_suppression_minimal, "suppression")
make_sharded_srw_paired_suppression_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_paired_suppression_minimal, "suppression")
make_sharded_srw_trajectory_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_trajectory_minimal, "trajectory")
make_sharded_srw_paired_trajectory_minimal = _factory_profile_wrapper(
    _shared_make_sharded_srw_paired_trajectory_minimal, "trajectory")


@wraps(_shared_make_space_dense_minimal)
def make_sharded_space_dense_minimal(*args, **kwargs):
    kernel = _shared_make_space_dense_minimal(*args, **kwargs)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "all_spaces"
    return kernel


@wraps(_shared_make_space_dense_diagnostics)
def make_sharded_space_dense_diagnostics(*args, **kwargs):
    kernel = _shared_make_space_dense_diagnostics(*args, **kwargs)
    kernel._v4174_kernel_profile = "production_diagnostics"
    kernel._v4174_dense_grouped_diagnostics = "all_spaces"
    return kernel


make_sharded_qk_space_dense_minimal = make_sharded_space_dense_minimal
make_sharded_qk_space_dense_diagnostics = make_sharded_space_dense_diagnostics


def _validate_v4174_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs):
    return _validate_shared_rw_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs)


def get_model_version() -> str:
    return MODEL_VERSION
