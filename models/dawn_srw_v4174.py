"""DAWN-SRW v4.1.7.4: route execution over shared operation spaces.

The model starts from a shared state representation ``x`` in
``[B, S, d_model]``.  It defines one shared operation-coordinate system and
``M`` operation spaces.  Each operation space is represented by one learned
space key ``k_m`` in that coordinate system.  Q, K, V, and RST independently
select from the same ``M`` spaces and execute their route-specific components
within the selected spaces.  The same space index aligns the Q/K/V/RST
execution components belonging to that operation space.

For route ``r`` and operation-space index ``m`` the canonical computation is

``c_r(x) = C_r x``
``alpha_{r,m}(x) = TopKSoftmax_m(cos(c_r(x), k_m))``
``z_{r,m} = P_{r,m} x``
``u_{r,m} = RWCompose_{r,m}(z_{r,m})``
``y_r = sum_m alpha_{r,m}(x) U_{r,m} u_{r,m}``

The operation-space keys are shared.  Space queries, operation projections, tau
projections, and execution are route-specific.  Q and K alone share their RW
read/write rows and state writeback.  V and RST have independent banks.
Semantic selection is sparse top-k, while physical execution is deliberately
all-space dense.  Production activations stop at ``[M,T,R]`` (or paired
``[M,T,2,R]``); writeback contracts directly to ``[T,D]`` and never creates
``[M,T,D]``.  Native v4174 with ``M=1`` uses implicit space weight one.

The five historical ``operation_address_*`` serialized names are retained
solely for exact params/optimizer-state resume compatibility with native
v4174 checkpoints already in training.  They are not v4173 migration aliases
and never trigger cross-version conversion.  A space key may be described as
the learned address of its operation space in the shared coordinate system.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import wraps
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

V4174_SERIALIZED_SPACE_PARAM_NAMES = {
    "operation_space_keys": "operation_address_keys",
    "q_space_query_proj": "q_address_query_proj",
    "k_space_query_proj": "k_address_query_proj",
    "v_space_query_proj": "v_address_query_proj",
    "rst_space_query_proj": "rst_address_query_proj",
}

SPACE_METRIC_SUFFIXES = (
    "space_weight_top1_mean",
    "space_weight_entropy_mean",
    "space_usage_min",
    "space_usage_max",
    "space_usage_std",
    "space_dead_frac",
    "space_top1_usage_max",
)
ROUTES = ("q", "k", "v", "rst")


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"model.{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_operation_space_config(model_cfg: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve canonical and native-v4174 compatibility config fields.

    ``n_operation_addresses`` and ``operation_address_top_k`` are accepted
    only because native v4174 runs created before the terminology correction
    serialized those config fields.  They do not imply v4173 compatibility.
    """
    canonical_n = model_cfg.get("n_operation_spaces")
    canonical_k = model_cfg.get("operation_space_top_k")
    serialized_n = model_cfg.get("n_operation_addresses")
    serialized_k = model_cfg.get("operation_address_top_k")

    def resolve_pair(canonical_name, canonical_value,
                     serialized_name, serialized_value, default):
        canonical = (None if canonical_value is None else
                     _positive_int(canonical_name, canonical_value))
        serialized = (None if serialized_value is None else
                      _positive_int(serialized_name, serialized_value))
        if canonical is not None and serialized is not None and canonical != serialized:
            raise ValueError(
                f"model.{canonical_name}={canonical} conflicts with native-v4174 "
                f"compatibility field model.{serialized_name}={serialized}")
        return canonical if canonical is not None else (
            serialized if serialized is not None else default)

    n_spaces = resolve_pair(
        "n_operation_spaces", canonical_n,
        "n_operation_addresses", serialized_n, 1)
    top_k = resolve_pair(
        "operation_space_top_k", canonical_k,
        "operation_address_top_k", serialized_k, 1)
    if top_k > n_spaces:
        raise ValueError(
            "model.operation_space_top_k must be <= "
            f"model.n_operation_spaces, got {top_k} > {n_spaces}"
        )
    return n_spaces, top_k


def materialize_operation_space_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and add canonical fields while preserving raw v4174 aliases."""
    n_spaces, top_k = resolve_operation_space_config(model_cfg)
    model_cfg["n_operation_spaces"] = n_spaces
    model_cfg["operation_space_top_k"] = top_k
    for pool_name in ("n_qk", "n_v", "n_rst"):
        value = _positive_int(pool_name, model_cfg.get(
            pool_name, model_cfg.get("n_know") if pool_name == "n_rst" else None))
        if value % n_spaces:
            raise ValueError(
                f"model.{pool_name} must be divisible by "
                f"model.n_operation_spaces, got {value} % {n_spaces}"
            )
    return model_cfg


def _serialized_space_param(params: Mapping[str, Any], canonical_name: str):
    """Read a native-v4174 parameter through the centralized ABI mapping."""
    serialized_name = V4174_SERIALIZED_SPACE_PARAM_NAMES.get(
        canonical_name, canonical_name)
    return params[serialized_name]


def symbolic_parameter_count(model_cfg: Mapping[str, Any]) -> dict[str, int]:
    """Count the exact v4174 tree, including the shared space codebook."""
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
    counts_by_route = {
        name: _positive_int(name, cfg.get(name, cfg.get("n_know")
                                          if name == "n_rst" else None))
        for name in ("n_qk", "n_v", "n_rst")
    }
    for name, value in counts_by_route.items():
        if value % n_spaces:
            raise ValueError(
                f"model.{name}={value} must be divisible by "
                f"n_operation_spaces={n_spaces}")
    operator_count = sum(counts_by_route.values())
    if n_spaces == 1:
        router = 7 * d_model * d_route + 4 * d_model + 4 * d_route + 4
        space_lookup = 0
    else:
        route_local = (
            7 * n_spaces * d_model * d_route
            + 4 * n_spaces * d_model
            + 4 * n_spaces * d_route
            + 4 * n_spaces
        )
        space_lookup = 4 * (d_model * d_route + d_route) + n_spaces * d_route
        router = route_local + space_lookup
    counts = {
        "token_embedding": vocab * d_model,
        "position_embedding": max_seq * d_model,
        "layer_stack": n_layers * (d_model * d_model + 4 * d_model),
        "router": router,
        "operation_space_lookup": space_lookup,
        "read_write_pools": operator_count * 2 * d_route,
        "learned_key_tables": 0,
        "bilinear_probe_matrices": 2 * d_route * d_route,
        "final_norm": 2 * d_model,
    }
    counts["total"] = sum(
        value for key, value in counts.items()
        if key != "operation_space_lookup")
    return counts


def _stacked_initializer(initializer: Callable) -> Callable:
    def init(key, shape, dtype=jnp.float32):
        keys = jax.random.split(key, int(shape[0]))
        return jax.vmap(lambda k: initializer(k, shape[1:], dtype))(keys)
    return init


class SpaceDense(nn.Module):
    """Route-local ``P_{r,m}`` or tau projection with replicated space axis."""
    n_operation_spaces: int
    features: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, state):
        kernel = self.param(
            "kernel", _stacked_initializer(self.kernel_init),
            (self.n_operation_spaces, state.shape[-1], self.features))
        result = jnp.einsum("...d,mdr->m...r", state, kernel)
        if self.use_bias:
            bias = self.param(
                "bias", _stacked_initializer(self.bias_init),
                (self.n_operation_spaces, self.features))
            result = result + bias[(slice(None),) + (None,) * (state.ndim - 1)]
        return result


class OperationSpaceNeuronPool(nn.Module):
    """Space-indexed RW banks; only the operation-row axis is sharded."""
    n_qk: int
    n_v: int
    n_rst: int
    d_route: int
    n_operation_spaces: int

    def setup(self):
        m = int(self.n_operation_spaces)
        r = int(self.d_route)
        shapes = {
            "qk": (self.n_qk, r) if m == 1 else (m, self.n_qk // m, r),
            "v": (self.n_v, r) if m == 1 else (m, self.n_v // m, r),
            "rst": (self.n_rst, r) if m == 1 else (m, self.n_rst // m, r),
        }
        for name in ("qk", "v", "rst"):
            setattr(self, f"{name}_read_vectors", self.param(
                f"{name}_read_vectors", _shared_unit_norm_init(), shapes[name]))
            setattr(self, f"{name}_write_vectors", self.param(
                f"{name}_write_vectors", _shared_unit_norm_init(), shapes[name]))
        probe_init = nn.initializers.orthogonal(scale=1.0)
        self.operator_key_read_probe = self.param(
            "operator_key_read_probe", probe_init, (r, r))
        self.operator_key_write_probe = self.param(
            "operator_key_write_probe", probe_init, (r, r))


class OperationSpaceRouter(nn.Module):
    """Shared space codebook plus route-specific projections and tau."""
    d_model: int
    d_route: int
    n_operation_spaces: int
    tau_init_attn_qk: float
    tau_init_attn_v: float
    tau_init_rst: float

    def setup(self):
        m, r = int(self.n_operation_spaces), int(self.d_route)
        projection_init = nn.initializers.orthogonal(scale=1.0)
        tau_values = {
            "q": self.tau_init_attn_qk,
            "k": self.tau_init_attn_qk,
            "v": self.tau_init_attn_v,
            "rst": self.tau_init_rst,
        }
        projection_cls = nn.Dense if m == 1 else SpaceDense
        projection_args = ({"features": r} if m == 1 else {
            "n_operation_spaces": m, "features": r})
        tau_args = ({"features": 1} if m == 1 else {
            "n_operation_spaces": m, "features": 1})
        for route in ROUTES:
            setattr(self, f"{route}_operation_proj", projection_cls(
                name=f"{route}_operation_proj", kernel_init=projection_init,
                bias_init=nn.initializers.zeros, **projection_args))
            tau_value = min(max(float(tau_values[route]), -0.9998), 0.9998)
            probability = (tau_value + 1.0) * 0.5
            raw_tau = math.log(probability) - math.log1p(-probability)
            setattr(self, f"{route}_operator_tau_proj", projection_cls(
                name=f"{route}_operator_tau_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=lambda k, s, d, value=raw_tau: jnp.full(s, value, d),
                **tau_args))
        writeback_args = ({"features": self.d_model, "use_bias": False}
                          if m == 1 else {
                              "n_operation_spaces": m,
                              "features": self.d_model, "use_bias": False})
        for name in ("qk", "v", "rst"):
            setattr(self, f"{name}_state_writeback", projection_cls(
                name=f"{name}_state_writeback", kernel_init=projection_init,
                **writeback_args))
        if m > 1:
            # Native v4174 checkpoints already in training serialize these
            # tensors under operation_address_* names.  Python attributes use
            # corrected operation-space terminology while the serialized ABI
            # remains byte-for-byte compatible for params and optimizer slots.
            self.operation_space_keys = self.param(
                V4174_SERIALIZED_SPACE_PARAM_NAMES["operation_space_keys"],
                _shared_unit_norm_init(), (m, r))
            for route in ROUTES:
                setattr(self, f"{route}_space_query_proj", nn.Dense(
                    r, name=V4174_SERIALIZED_SPACE_PARAM_NAMES[
                        f"{route}_space_query_proj"],
                    kernel_init=projection_init,
                    bias_init=nn.initializers.zeros))


def _linear(params: Mapping[str, jax.Array], state: jax.Array) -> jax.Array:
    return state @ params["kernel"] + params.get("bias", 0.0)


def _select_operation_spaces(
        state: jax.Array,
        space_query_params: Mapping[str, jax.Array],
        operation_space_keys: jax.Array,
        operation_space_top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select semantic spaces without changing physical dense execution.

    ``state`` is ``[B,S,D]``; the route-specific query projection maps it to
    ``[B,S,R]``.  Shared keys are ``[M,R]``.  Returned IDs and selected
    weights are ``[B,S,K]`` and scores are ``[B,S,M]``.  The dense execution
    path later scatters these K weights into ``[T,M]``; it never dispatches or
    gathers tokens by space.
    """
    query = _linear(space_query_params, state)
    query = _shared_forward_unit_direction(query.astype(jnp.float32))
    keys = _shared_forward_unit_direction(
        jnp.asarray(operation_space_keys, dtype=jnp.float32))
    scores = jnp.einsum("bsr,mr->bsm", query, keys)
    selected_scores, selected_ids = jax.lax.top_k(
        scores, int(operation_space_top_k))
    selected_weights = jax.nn.softmax(selected_scores, axis=-1)
    return selected_ids, selected_weights, scores


def _dense_space_weights(
        selected_space_ids: jax.Array,
        selected_space_weights: jax.Array,
        n_operation_spaces: int,
) -> jax.Array:
    """Return exact-zero unselected weights as a flattened ``[T,M]`` matrix."""
    dense = jnp.sum(
        jax.nn.one_hot(
            selected_space_ids, int(n_operation_spaces),
            dtype=selected_space_weights.dtype)
        * selected_space_weights[..., None],
        axis=-2)
    return dense.reshape((-1, int(n_operation_spaces)))


def _space_weighted_state_writeback(
        space_results: jax.Array,
        dense_space_weights: jax.Array,
        state_writeback_kernel: jax.Array,
        route_scale: jax.Array | float,
) -> jax.Array:
    """Fuse semantic space weighting, space sum, and ``R -> D`` writeback.

    Inputs are ``space_results[M,T,R]``, weights ``[T,M]``, and the
    route-specific kernel ``[M,R,D]``.  The only weighted intermediate is
    ``[M,T,R]``.  The einsum contracts directly to ``[T,D]``; a forbidden
    ``[M,T,D]`` tensor is never materialized.
    """
    space_results = jnp.asarray(space_results)
    if space_results.ndim != 3:
        raise ValueError(
            "space_results must have shape [M,T,R], got "
            f"{space_results.shape}")
    weighted_results = (
        space_results
        * jnp.swapaxes(dense_space_weights, 0, 1)[..., None])
    return jnp.einsum(
        "mtr,mrd->td", weighted_results * jnp.asarray(route_scale),
        state_writeback_kernel)


def _materialize_space_operator_keys(
        read_vectors: jax.Array, write_vectors: jax.Array,
        read_probe: jax.Array, write_probe: jax.Array) -> jax.Array:
    """Generate live-gradient operator keys independently inside each space."""
    if read_vectors.ndim == 2:
        return _shared_materialize_operator_keys(
            read_vectors, write_vectors, read_probe, write_probe)
    if read_vectors.ndim != 3:
        raise ValueError(
            "space-indexed read/write vectors must be [M,N,R], got "
            f"{read_vectors.shape}")
    return jax.vmap(
        lambda read, write: _shared_materialize_operator_keys(
            read, write, read_probe, write_probe))(read_vectors, write_vectors)


def _rw_compose_space_dense(
        operator_query: jax.Array, operator_keys: jax.Array,
        raw_operator_tau: jax.Array, read_vectors: jax.Array,
        write_vectors: jax.Array, *, soft_gate_temperature: float,
        soft_gate_boundary_power: float, admission_den_power: float,
        srw_composition_mode: str, heat_kernel_beta: float,
        execution_prune_eps: float = 0.0, max_chunk_size: int = 2048,
        diagnostics: bool = False):
    """Execute every space densely while chunking only the operator axis.

    The physical input/output is ``[M,T,R]`` and each space owns
    ``[N_per_space,R]`` operator rows.  Operator gates are transient
    ``[M,T,chunk]`` values.  Production returns only local results;
    diagnostics additionally returns seven ``[M,T,1]`` aggregates.
    """
    if operator_query.ndim == 2:
        operator_query = operator_query[None, ...]
    if operator_keys.ndim == 2:
        operator_keys = operator_keys[None, ...]
    if raw_operator_tau.ndim == 2:
        raw_operator_tau = raw_operator_tau[None, ...]
    if read_vectors.ndim == 2:
        read_vectors = read_vectors[None, ...]
    if write_vectors.ndim == 2:
        write_vectors = write_vectors[None, ...]
    m, token_count, d_route = operator_query.shape
    n_operator = int(operator_keys.shape[1])
    chunk = min(int(max_chunk_size), n_operator)
    n_chunks = math.ceil(n_operator / chunk)
    padded = n_chunks * chunk
    padding = ((0, 0), (0, padded - n_operator), (0, 0))
    keys = jnp.pad(operator_keys, padding)
    reads = jnp.pad(read_vectors, padding)
    writes = jnp.pad(write_vectors, padding)
    valid = jnp.arange(padded) < n_operator
    query_unit = _shared_forward_unit_direction(
        operator_query.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    key_unit = _shared_forward_unit_direction(
        keys.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    read_unit = _shared_forward_unit_direction(
        reads.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    write_unit = _shared_forward_unit_direction(
        writes.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
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
        rho = jnp.einsum("mtr,mnr->mtn", query_unit, key).astype(jnp.float32)
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
        read_value = jnp.einsum(
            "mtr,mnr->mtn", operator_query.astype(jnp.bfloat16), read
        ).astype(jnp.float32)
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
    # Match the canonical shard-map boundary: model shards contribute BF16
    # local results before the replicated FP32 route-local output is formed.
    space_results = (raw_out / gate_den).astype(jnp.bfloat16).astype(jnp.float32)
    if not diagnostics:
        return space_results
    return (space_results, active_count, gate_mass, gate_sq, gate_max,
            depth_sum, tau, gate_den)


def _paired_qk_space_execution(
        stacked_qk_operator_query: jax.Array,
        qk_operator_keys: jax.Array,
        stacked_qk_raw_tau: jax.Array,
        qk_read_vectors: jax.Array,
        qk_write_vectors: jax.Array,
        **execution_kwargs) -> jax.Array:
    """Execute independent Q/K gates against one shared space-indexed bank.

    Q and K arrive as ``[M,T,2,R]`` and return the same shape.  The route axis
    is paired before attention; it is never retained by the attention core.
    Selection, operation projection, and tau remain independent.
    """
    q_result = _rw_compose_space_dense(
        stacked_qk_operator_query[:, :, 0, :], qk_operator_keys,
        stacked_qk_raw_tau[:, :, 0, :], qk_read_vectors, qk_write_vectors,
        **execution_kwargs)
    k_result = _rw_compose_space_dense(
        stacked_qk_operator_query[:, :, 1, :], qk_operator_keys,
        stacked_qk_raw_tau[:, :, 1, :], qk_read_vectors, qk_write_vectors,
        **execution_kwargs)
    return jnp.stack((q_result, k_result), axis=2)


def _space_selector_metrics(
        selected_ids: jax.Array, selected_weights: jax.Array,
        n_operation_spaces: int, route: str) -> dict[str, jax.Array]:
    """Small regular metrics computed only from already-selected IDs/weights."""
    m = int(n_operation_spaces)
    flat_ids = selected_ids.reshape((-1, selected_ids.shape[-1]))
    flat_weights = selected_weights.reshape((-1, selected_weights.shape[-1]))
    top1 = flat_ids[:, 0]
    top1_usage = jax.nn.one_hot(top1, m, dtype=jnp.float32).mean(axis=0)
    usage = jax.nn.one_hot(flat_ids, m, dtype=jnp.float32).sum(axis=1).mean(axis=0)
    entropy = -(flat_weights * jnp.log(jnp.maximum(flat_weights, 1e-8))).sum(axis=-1)
    suffix_values = {
        "weight_top1_mean": flat_weights[:, 0].mean(),
        "weight_entropy_mean": entropy.mean(),
        "usage_min": usage.min(),
        "usage_max": usage.max(),
        "usage_std": usage.std(),
        "dead_frac": (usage == 0).astype(jnp.float32).mean(),
        "top1_usage_max": top1_usage.max(),
    }
    values = {}
    for suffix, value in suffix_values.items():
        value = jax.lax.stop_gradient(value)
        values[f"{route}_space_{suffix}"] = value
        # Deprecated compatibility alias for native v4174 run log consumers.
        values[f"{route}_address_{suffix}"] = value
    return values


def operation_space_initialization_diagnostics(
        operation_space_keys: jax.Array,
        route_assignments: Mapping[str, tuple[jax.Array, jax.Array, jax.Array]],
) -> dict[str, float]:
    """Host-side, one-shot key and initial token-assignment diagnostics."""
    keys = jax.device_get(jnp.asarray(operation_space_keys, dtype=jnp.float32))
    norms = jnp.linalg.norm(keys, axis=-1)
    if (not bool(jnp.all(jnp.isfinite(keys)))
            or bool(jnp.any(norms <= 0))):
        raise ValueError("operation_space_keys contain non-finite or zero-norm rows")
    unit = keys / norms[:, None]
    cosine = unit @ unit.T
    offdiag = cosine[~jnp.eye(cosine.shape[0], dtype=jnp.bool_)]
    if bool(jnp.any(offdiag > 0.95)):
        raise ValueError("operation_space_keys contain duplicate or near-identical rows")
    nearest = jnp.max(jnp.where(
        jnp.eye(cosine.shape[0], dtype=jnp.bool_), -jnp.inf, cosine), axis=-1)
    out = {
        "operation_space_pair_cosine_mean": float(offdiag.mean()),
        "operation_space_pair_cosine_abs_mean": float(jnp.abs(offdiag).mean()),
        "operation_space_pair_cosine_max": float(offdiag.max()),
        "operation_space_nearest_neighbor_cosine_mean": float(nearest.mean()),
        "operation_space_nearest_neighbor_cosine_max": float(nearest.max()),
    }
    m = int(keys.shape[0])
    for route, (ids, weights, scores) in route_assignments.items():
        ids = jnp.asarray(ids)
        weights = jnp.asarray(weights)
        scores = jnp.asarray(scores)
        top1_usage = jax.nn.one_hot(ids[..., 0], m).reshape((-1, m)).mean(axis=0)
        topk_usage = jax.nn.one_hot(ids, m).sum(axis=-2).reshape((-1, m)).mean(axis=0)
        sorted_scores = jnp.sort(scores, axis=-1)
        gap = sorted_scores[..., -1] - sorted_scores[..., -2]
        for label, values in (("top1_usage", top1_usage),
                              ("topk_usage", topk_usage)):
            out[f"{route}_{label}_min"] = float(values.min())
            out[f"{route}_{label}_max"] = float(values.max())
            out[f"{route}_{label}_std"] = float(values.std())
        out[f"{route}_dead_space_frac"] = float((topk_usage == 0).mean())
        out[f"{route}_top1_weight_mean"] = float(weights[..., 0].mean())
        out[f"{route}_weight_entropy"] = float(
            (-(weights * jnp.log(jnp.maximum(weights, 1e-8))).sum(axis=-1)).mean())
        out[f"{route}_top1_top2_score_gap"] = float(gap.mean())
    return out


def initialization_diagnostics_from_params(
        params: Mapping[str, Any], input_ids: jax.Array,
        operation_space_top_k: int) -> dict[str, float]:
    """Build the one-shot host diagnostic outside the training-step graph."""
    router = params["router"]
    serialized_key_name = V4174_SERIALIZED_SPACE_PARAM_NAMES[
        "operation_space_keys"]
    if serialized_key_name not in router:
        return {}
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    positions = jnp.arange(input_ids.shape[1])[None, :]
    state = (params["token_emb"]["embedding"][input_ids]
             + params["pos_emb"]["embedding"][positions])
    block = params["block_0"]
    attention_state = _shared_layer_norm(
        state, block["norm1"]["scale"], block["norm1"]["bias"])
    rst_state = _shared_layer_norm(
        state, block["norm2"]["scale"], block["norm2"]["bias"])
    assignments = {}
    operation_space_keys = _serialized_space_param(
        router, "operation_space_keys")
    for route in ROUTES:
        route_state = rst_state if route == "rst" else attention_state
        assignments[route] = _select_operation_spaces(
            route_state, _serialized_space_param(
                router, f"{route}_space_query_proj"),
            operation_space_keys, operation_space_top_k)
    return operation_space_initialization_diagnostics(
        operation_space_keys, assignments)


def _space_projection(
        state: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
    """Apply a canonical projection and normalize its space-leading shape."""
    kernel = params["kernel"]
    if kernel.ndim == 2:
        return (_linear(params, state))[None, ...]
    result = jnp.einsum("...d,mdr->m...r", state, kernel)
    bias = params.get("bias", 0.0)
    return result + bias.reshape(
        (bias.shape[0],) + (1,) * (result.ndim - 2) + (bias.shape[-1],))


def _canonical_writeback_kernel(params: Mapping[str, jax.Array]) -> jax.Array:
    kernel = params["kernel"]
    return kernel[None, ...] if kernel.ndim == 2 else kernel


def _aggregate_operator_diagnostics(
        diagnostics_values: tuple[jax.Array, ...],
        selected_ids: jax.Array, selected_weights: jax.Array,
        n_operators_per_space: int, route: str,
        srw_composition_mode: str,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Reduce space-local aggregates according to the reporting contract."""
    (space_results, active_count, gate_mass, gate_sq, gate_max,
     depth_sum, tau, gate_den) = diagnostics_values
    ids = selected_ids.reshape((-1, selected_ids.shape[-1]))
    weights = selected_weights.reshape((-1, selected_weights.shape[-1]))

    def selected(value):
        token_major = jnp.swapaxes(value[..., 0], 0, 1)
        return jnp.take_along_axis(token_major, ids, axis=1)

    active = selected(active_count)
    mass = selected(gate_mass)
    square = selected(gate_sq)
    maximum = selected(gate_max)
    depth = selected(depth_sum)
    tau_selected = selected(tau)
    den = selected(gate_den)
    weighted_mean = lambda value: (value * weights).sum(axis=-1).mean()
    floor_mass = _shared_composition_den_floor_mass(srw_composition_mode)
    floor_compare = jnp.less if srw_composition_mode == "linear_angular" else jnp.less_equal
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
    m = int(space_results.shape[0])
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    selected_counts = jax.nn.one_hot(flat_ids, m).sum(axis=0)
    selected_weight_sum = (
        jax.nn.one_hot(flat_ids, m) * flat_weights[:, None]).sum(axis=0)
    per_space = {
        f"{route}_per_space_selection_frac": selected_counts / ids.shape[0],
        f"{route}_per_space_mean_selected_weight": (
            selected_weight_sum / jnp.maximum(selected_counts, 1.0)),
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
        {key: jax.lax.stop_gradient(value) for key, value in per_space.items()},
    )


def _canonical_regular_operator_metrics(
        metrics: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    """Map v4174 space diagnostics to the shared DirectTau log contract."""
    route_prefixes = {
        "q": "attn_q", "k": "attn_k", "v": "attn_v", "rst": "rst"}
    suffixes = {
        "active_tau_frac": "active_tau_frac",
        "active_tau_count": "active_tau_count",
        "tau_mean": "tau_mean",
        "gate_mass_mean": "gate_mass",
        "gate_den_mean": "gate_den",
        "depth_active_mean": "depth_active",
        "gate_eff_n_mean": "gate_eff_n",
        "top1_gate_frac_mean": "top1_gate_frac",
        "den_floor_frac": "den_floor_frac",
    }
    required = {
        f"{route}_operator_{source_suffix}"
        for route in route_prefixes for source_suffix in suffixes}
    required.update({
        "q_route_output_norm", "k_route_output_norm",
        "v_route_output_norm", "rst_route_update_norm",
        "attn_out_norm", "rst_out_norm", "residual_norm",
    })
    missing = tuple(sorted(required.difference(metrics)))
    if missing:
        raise KeyError(
            "v4174 canonical regular diagnostics are incomplete: "
            + ", ".join(missing))

    out: dict[str, jax.Array] = {}
    for route, prefix in route_prefixes.items():
        for source_suffix, canonical_suffix in suffixes.items():
            out[f"{prefix}_{canonical_suffix}"] = metrics[
                f"{route}_operator_{source_suffix}"]
    for canonical_suffix in suffixes.values():
        out[f"attn_qk_{canonical_suffix}"] = jnp.float32(0.5) * (
            out[f"attn_q_{canonical_suffix}"]
            + out[f"attn_k_{canonical_suffix}"])

    out.update({
        "attn_qk_admission_mass_mean": out["attn_qk_gate_mass"],
        "attn_v_admission_mass_mean": out["attn_v_gate_mass"],
        "rst_admission_mass_mean": out["rst_gate_mass"],
        "attn_qk_composition_den_mean": out["attn_qk_gate_den"],
        "attn_v_composition_den_mean": out["attn_v_gate_den"],
        "rst_composition_den_mean": out["rst_gate_den"],
        "attn_qk_composition_den_floor_frac": out[
            "attn_qk_den_floor_frac"],
        "attn_v_composition_den_floor_frac": out[
            "attn_v_den_floor_frac"],
        "rst_composition_den_floor_frac": out["rst_den_floor_frac"],
        "attn_qk_pool_scaled_srw_out_norm": jnp.float32(0.5) * (
            metrics["q_route_output_norm"]
            + metrics["k_route_output_norm"]),
        "attn_v_pool_scaled_srw_out_norm": metrics["v_route_output_norm"],
        "rst_pool_scaled_srw_out_norm": metrics["rst_route_update_norm"],
    })
    return {
        key: jax.lax.stop_gradient(value) for key, value in out.items()}


class DAWN_SRW_V4174(nn.Module):
    """Transformer whose four routes share one operation-space codebook."""
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
    n_qk: int = 1580
    n_v: int = 2600
    n_rst: Optional[int] = None
    n_know: Optional[int] = None
    n_operation_spaces: int = 1
    operation_space_top_k: int = 1
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
            raise ValueError(
                f"v4174 requires operator_key_mode={OPERATOR_KEY_MODE!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        m = _positive_int("n_operation_spaces", self.n_operation_spaces)
        k = _positive_int("operation_space_top_k", self.operation_space_top_k)
        if k > m:
            raise ValueError("operation_space_top_k exceeds n_operation_spaces")
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        for name, value in (("n_qk", self.n_qk), ("n_v", self.n_v),
                            ("n_rst", n_rst)):
            if int(value) % m:
                raise ValueError(f"{name} must be divisible by n_operation_spaces")
        if any(value is None for value in (
                self.tau_init_attn_qk, self.tau_init_attn_v,
                self.tau_init_rst)):
            raise ValueError("v4174 requires explicit initial operator tau values")
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
        """Create every canonical parameter during ``init`` without a heavy RW run."""
        _ = self.neuron_pool.qk_read_vectors
        for route in ROUTES:
            _ = getattr(self.router, f"{route}_operation_proj")(state)
            _ = getattr(self.router, f"{route}_operator_tau_proj")(state)
        if int(self.n_operation_spaces) > 1:
            _ = self.router.operation_space_keys
            for route in ROUTES:
                _ = getattr(self.router, f"{route}_space_query_proj")(state)
        local_zero = jnp.zeros((*state.shape[:-1], self.d_route), state.dtype)
        for name in ("qk", "v", "rst"):
            _ = getattr(self.router, f"{name}_state_writeback")(local_zero)
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
        """Run all-space dense routing; heavy arrays require analysis cadence."""
        del (attention_mask, soft_gate_t_final,
             soft_gate_boundary_power_final, admission_den_power,
             heat_kernel_beta, minimal_train, ce_token_chunk_size,
             analysis_kwargs)
        m = int(self.n_operation_spaces)
        top_k = int(self.operation_space_top_k)
        n_rst_effective = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        composition_mode = str(srw_composition_mode or self.srw_composition_mode)
        qk_power = float(self.admission_den_power_qk
                         if self.admission_den_power_qk is not None
                         else self.admission_den_power)
        v_power = float(self.admission_den_power_v
                        if self.admission_den_power_v is not None
                        else self.admission_den_power)
        rst_power = float(self.admission_den_power_rst
                          if self.admission_den_power_rst is not None
                          else self.admission_den_power)
        soft_gate_T_qk = float(soft_gate_temperature if soft_gate_T_qk is None
                               else soft_gate_T_qk)
        soft_gate_T_v = float(soft_gate_temperature if soft_gate_T_v is None
                              else soft_gate_T_v)
        soft_gate_T_rst = float(soft_gate_temperature if soft_gate_T_rst is None
                                else soft_gate_T_rst)
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
        pool = params["neuron_pool"]
        router = params["router"]
        read_probe = pool["operator_key_read_probe"]
        write_probe = pool["operator_key_write_probe"]
        bank_values: dict[str, tuple[jax.Array, jax.Array, jax.Array]] = {}
        for name in ("qk", "v", "rst"):
            read = pool[f"{name}_read_vectors"]
            write = pool[f"{name}_write_vectors"]
            bank_values[name] = (
                read, write, _materialize_space_operator_keys(
                    read, write, read_probe, write_probe))
        qk_scale, v_scale, rst_scale = _shared_pool_output_scales(
            self.d_model, self.n_layers)
        base_rng = self.make_rng("dropout")
        layer_rngs = jax.random.split(base_rng, self.n_layers)
        selector_metrics: dict[str, list[jax.Array]] = {}
        operator_metrics: dict[str, list[jax.Array]] = {}
        analysis_arrays: dict[str, list[jax.Array]] = {}
        diagnostics_enabled = (
            minimal_runtime_profile == "diagnostics" or analysis)

        def execute_space_route(
                route_key, operator_query, operator_keys, raw_tau,
                read_vectors, write_vectors, *, temperature, den_power):
            kernel = (sharded_fns.get(route_key)
                      if isinstance(sharded_fns, dict) else None)
            if kernel is not None and m > 1:
                token_valid = jnp.ones(
                    operator_query.shape[:2], dtype=jnp.bool_)
                return kernel(
                    operator_query, operator_keys, raw_tau, token_valid,
                    read_vectors, write_vectors,
                    temperature, temperature,
                    soft_gate_boundary_power, soft_gate_boundary_power,
                    execution_prune_eps)
            return _rw_compose_space_dense(
                operator_query, operator_keys, raw_tau,
                read_vectors, write_vectors,
                soft_gate_temperature=temperature,
                soft_gate_boundary_power=soft_gate_boundary_power,
                admission_den_power=den_power,
                srw_composition_mode=composition_mode,
                heat_kernel_beta=self.heat_kernel_beta,
                execution_prune_eps=execution_prune_eps,
                diagnostics=diagnostics_enabled)

        for layer_index, layer in enumerate(self.layers):
            rng, rng_attn, rng_rst = jax.random.split(layer_rngs[layer_index], 3)
            normalized = layer.norm1(state)
            route_ids: dict[str, jax.Array] = {}
            route_weights: dict[str, jax.Array] = {}
            dense_weights: dict[str, jax.Array] = {}
            if m == 1:
                shape = (*normalized.shape[:2], 1)
                for route in ROUTES:
                    route_ids[route] = jnp.zeros(shape, dtype=jnp.int32)
                    route_weights[route] = jnp.ones(shape, dtype=jnp.float32)
                    dense_weights[route] = jnp.ones(
                        (normalized.shape[0] * normalized.shape[1], 1),
                        dtype=jnp.float32)
            else:
                operation_space_keys = _serialized_space_param(
                    router, "operation_space_keys")
                for route in ("q", "k", "v"):
                    ids, weights, _ = _select_operation_spaces(
                        normalized, _serialized_space_param(
                            router, f"{route}_space_query_proj"),
                        operation_space_keys, top_k)
                    route_ids[route], route_weights[route] = ids, weights
                    dense_weights[route] = _dense_space_weights(ids, weights, m)
            for route in ("q", "k", "v"):
                for key, value in _space_selector_metrics(
                        route_ids[route], route_weights[route], m, route).items():
                    selector_metrics.setdefault(key, []).append(value)
            flat_attention_state = normalized.reshape((-1, self.d_model))
            query_values = {
                route: _space_projection(
                    normalized, router[f"{route}_operation_proj"]
                ).reshape((m, -1, self.d_route))
                for route in ("q", "k", "v")}
            tau_values = {
                route: _space_projection(
                    normalized, router[f"{route}_operator_tau_proj"]
                ).reshape((m, -1, 1))
                for route in ("q", "k", "v")}
            qk_execution_kwargs = dict(
                soft_gate_temperature=soft_gate_T_qk,
                soft_gate_boundary_power=soft_gate_boundary_power,
                admission_den_power=qk_power,
                srw_composition_mode=composition_mode,
                heat_kernel_beta=self.heat_kernel_beta,
                execution_prune_eps=execution_prune_eps,
                diagnostics=diagnostics_enabled)
            qk_paired = jnp.stack((query_values["q"], query_values["k"]), axis=2)
            qk_tau = jnp.stack((tau_values["q"], tau_values["k"]), axis=2)
            qk_read, qk_write, qk_keys = bank_values["qk"]
            paired_space_kernel = (
                sharded_fns.get("qk_space_dense")
                if isinstance(sharded_fns, dict) and m > 1 else None)
            if paired_space_kernel is not None:
                paired_result = paired_space_kernel(
                    qk_paired, qk_keys, qk_tau,
                    jnp.ones(qk_paired.shape[:2], dtype=jnp.bool_),
                    qk_read, qk_write,
                    soft_gate_T_qk, soft_gate_T_qk,
                    soft_gate_boundary_power, soft_gate_boundary_power,
                    execution_prune_eps)
                if diagnostics_enabled:
                    q_diag, k_diag = paired_result
                else:
                    q_space_outputs = paired_result[:, :, 0]
                    k_space_outputs = paired_result[:, :, 1]
            elif diagnostics_enabled:
                q_diag = execute_space_route(
                    "q_space_dense", qk_paired[:, :, 0], qk_keys,
                    qk_tau[:, :, 0], qk_read, qk_write,
                    temperature=soft_gate_T_qk, den_power=qk_power)
                k_diag = execute_space_route(
                    "k_space_dense", qk_paired[:, :, 1], qk_keys,
                    qk_tau[:, :, 1], qk_read, qk_write,
                    temperature=soft_gate_T_qk, den_power=qk_power)
            if diagnostics_enabled:
                q_space_outputs, k_space_outputs = q_diag[0], k_diag[0]
                for route, diag in (("q", q_diag), ("k", k_diag)):
                    scalar, arrays = _aggregate_operator_diagnostics(
                        diag, route_ids[route], route_weights[route],
                        self.n_qk // m, route, composition_mode)
                    for key, value in scalar.items():
                        operator_metrics.setdefault(key, []).append(value)
                    if analysis:
                        for key, value in arrays.items():
                            analysis_arrays.setdefault(key, []).append(value)
            elif paired_space_kernel is None:
                paired_result = _paired_qk_space_execution(
                    qk_paired, qk_keys, qk_tau, qk_read, qk_write,
                    **qk_execution_kwargs)
                q_space_outputs = paired_result[:, :, 0]
                k_space_outputs = paired_result[:, :, 1]
            q_route_output = _space_weighted_state_writeback(
                q_space_outputs, dense_weights["q"],
                _canonical_writeback_kernel(router["qk_state_writeback"]),
                qk_scale).reshape(normalized.shape)
            k_route_output = _space_weighted_state_writeback(
                k_space_outputs, dense_weights["k"],
                _canonical_writeback_kernel(router["qk_state_writeback"]),
                qk_scale).reshape(normalized.shape)
            v_read, v_write, v_keys = bank_values["v"]
            v_execution_kwargs = dict(qk_execution_kwargs)
            v_execution_kwargs["soft_gate_temperature"] = soft_gate_T_v
            v_execution_kwargs["admission_den_power"] = v_power
            v_result = execute_space_route(
                "v_space_dense", query_values["v"], v_keys,
                tau_values["v"], v_read, v_write,
                temperature=soft_gate_T_v, den_power=v_power)
            if diagnostics_enabled:
                v_space_outputs = v_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    v_result, route_ids["v"], route_weights["v"],
                    self.n_v // m, "v", composition_mode)
                for key, value in scalar.items():
                    operator_metrics.setdefault(key, []).append(value)
                if analysis:
                    for key, value in arrays.items():
                        analysis_arrays.setdefault(key, []).append(value)
            else:
                v_space_outputs = v_result
            v_route_output = _space_weighted_state_writeback(
                v_space_outputs, dense_weights["v"],
                _canonical_writeback_kernel(router["v_state_writeback"]),
                v_scale).reshape(normalized.shape)
            batch_size, sequence_length = normalized.shape[:2]
            d_head = self.d_model // self.n_heads
            query = q_route_output.reshape(
                batch_size, sequence_length, self.n_heads, d_head).transpose(0, 2, 1, 3)
            key = k_route_output.reshape(
                batch_size, sequence_length, self.n_heads, d_head).transpose(0, 2, 1, 3)
            value = v_route_output.reshape(
                batch_size, sequence_length, self.n_heads, d_head).transpose(0, 2, 1, 3)
            attention_scores = jnp.einsum(
                "bhsd,bhtd->bhst", query, key) / jnp.sqrt(jnp.float32(d_head))
            causal = jnp.tril(jnp.ones(
                (sequence_length, sequence_length), dtype=jnp.bool_))
            attention_scores = jnp.where(
                causal, attention_scores, jnp.finfo(attention_scores.dtype).min)
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

            normalized = layer.norm2(state)
            if m > 1:
                ids, weights, _ = _select_operation_spaces(
                    normalized, _serialized_space_param(
                        router, "rst_space_query_proj"),
                    operation_space_keys, top_k)
                route_ids["rst"], route_weights["rst"] = ids, weights
                dense_weights["rst"] = _dense_space_weights(ids, weights, m)
            for key, value in _space_selector_metrics(
                    route_ids["rst"], route_weights["rst"], m, "rst").items():
                selector_metrics.setdefault(key, []).append(value)
            rst_query = _space_projection(
                normalized, router["rst_operation_proj"]).reshape(
                    (m, -1, self.d_route))
            rst_tau = _space_projection(
                normalized, router["rst_operator_tau_proj"]).reshape((m, -1, 1))
            rst_read_vectors, rst_write_vectors, rst_keys = bank_values["rst"]
            rst_result = execute_space_route(
                "rst_space_dense", rst_query, rst_keys, rst_tau,
                rst_read_vectors, rst_write_vectors,
                temperature=soft_gate_T_rst, den_power=rst_power)
            if diagnostics_enabled:
                rst_space_updates = rst_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    rst_result, route_ids["rst"], route_weights["rst"],
                    n_rst_effective // m, "rst", composition_mode)
                for key, value in scalar.items():
                    operator_metrics.setdefault(key, []).append(value)
                if analysis:
                    for key, value in arrays.items():
                        analysis_arrays.setdefault(key, []).append(value)
            else:
                rst_space_updates = rst_result
            rst_route_update = _space_weighted_state_writeback(
                rst_space_updates, dense_weights["rst"],
                _canonical_writeback_kernel(router["rst_state_writeback"]),
                rst_scale).reshape(normalized.shape)
            state = state + _shared_safe_dropout(
                rst_route_update, self.dropout_rate, deterministic, rng_rst)
            if diagnostics_enabled:
                operator_metrics.setdefault("attn_out_norm", []).append(
                    jax.lax.stop_gradient(jnp.linalg.norm(
                        attention_output.astype(jnp.float32), axis=-1).mean()))
                operator_metrics.setdefault("rst_out_norm", []).append(
                    jax.lax.stop_gradient(jnp.linalg.norm(
                        rst_route_update.astype(jnp.float32), axis=-1).mean()))
                operator_metrics.setdefault("residual_norm", []).append(
                    jax.lax.stop_gradient(jnp.linalg.norm(
                        state.astype(jnp.float32), axis=-1).mean()))
                operator_metrics.setdefault("q_route_output_norm", []).append(
                    jnp.linalg.norm(q_route_output.astype(jnp.float32), axis=-1).mean())
                operator_metrics.setdefault("k_route_output_norm", []).append(
                    jnp.linalg.norm(k_route_output.astype(jnp.float32), axis=-1).mean())
                operator_metrics.setdefault("v_route_output_norm", []).append(
                    jnp.linalg.norm(v_route_output.astype(jnp.float32), axis=-1).mean())
                rst_route_update_norm = jnp.linalg.norm(
                    rst_route_update.astype(jnp.float32), axis=-1).mean()
                operator_metrics.setdefault("rst_route_update_norm", []).append(
                    rst_route_update_norm)
                # Deprecated native-v4174 log compatibility alias.
                operator_metrics.setdefault("rst_route_output_norm", []).append(
                    rst_route_update_norm)

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
                result["logits"] = logits[..., :logical] if embedding != logical else logits
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
                    (-1, logical))[jnp.arange(safe_labels.size), safe_labels.reshape(-1)]
                loss = (token_loss.reshape(safe_labels.shape) * valid).sum() / jnp.maximum(valid.sum(), 1)
                predictions = jnp.argmax(logits, axis=-1)
                correct = ((predictions == safe_labels) & valid).sum().astype(jnp.int32)
            result.update({
                "loss": loss, "aux_loss": jnp.float32(0.0),
                "correct": correct if compute_accuracy else jnp.int32(0),
                "valid_count": valid.sum().astype(jnp.int32),
            })
        for key, values in selector_metrics.items():
            result[key] = jnp.stack(values).mean()
        if diagnostics_enabled:
            for key, values in operator_metrics.items():
                result[key] = jnp.stack(values).mean()
            result.update(_canonical_regular_operator_metrics(result))
            result["heat_kernel_beta"] = jnp.float32(self.heat_kernel_beta)
        if analysis:
            for key, values in analysis_arrays.items():
                value = jnp.stack(values).mean(axis=0)
                result[key] = value
                if "_per_space_" in key:
                    # Deprecated native-v4174 analysis compatibility alias.
                    result[key.replace("_per_space_", "_per_address_")] = value
        return result

    def get_model_info(self) -> list[str]:
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        m = int(self.n_operation_spaces)
        return [
            f"DAWN-SRW v4.1.7.4 ({MODEL_VERSION})",
            "architecture: shared state -> shared operation-space lookup -> "
            "route-local RW execution -> space-weighted state writeback",
            f"spaces: count={m}, top_k={int(self.operation_space_top_k)}, "
            f"d_space=d_route={int(self.d_route)}",
            "Q/K share qk_read_vectors, qk_write_vectors, generated keys, and "
            "qk_state_writeback; routing/projection/tau remain independent",
            f"operators per space: qk={int(self.n_qk)//m}, "
            f"v={int(self.n_v)//m}, rst={n_rst//m}",
            "execution: semantic top-k with physical all-space dense kernels; "
            "no [M,T,D] writeback intermediate",
        ]


def _pool_operator_keys(pool_params: Mapping[str, jax.Array],
                        operator_key_mode: Optional[str] = None) -> dict[str, jax.Array]:
    """Materialize canonical live-gradient keys for QK, V, and RST banks."""
    if operator_key_mode not in (None, OPERATOR_KEY_MODE):
        raise ValueError(f"unsupported operator_key_mode={operator_key_mode!r}")
    read_probe = pool_params["operator_key_read_probe"]
    write_probe = pool_params["operator_key_write_probe"]
    return {
        f"{name}_operator_keys": _materialize_space_operator_keys(
            pool_params[f"{name}_read_vectors"],
            pool_params[f"{name}_write_vectors"], read_probe, write_probe)
        for name in ("qk", "v", "rst")
    }


def generalized_bilinear_operator_key_diagnostics(
        read_vectors, write_vectors, read_probes, write_probes,
        eps=RW_FORWARD_NORM_EPS):
    """Expose existing bilinear-key diagnostics for one space or bank."""
    if read_vectors.ndim == 2:
        return _shared_operator_key_diagnostics(
            read_vectors, write_vectors, read_probes, write_probes, eps)
    flattened_read = read_vectors.reshape((-1, read_vectors.shape[-1]))
    flattened_write = write_vectors.reshape((-1, write_vectors.shape[-1]))
    return _shared_operator_key_diagnostics(
        flattened_read, flattened_write, read_probes, write_probes, eps)


def _tau_init_calibration_scores(params, input_ids, max_tokens=128):
    """Return all-token × every-space score tables for independent tau fits.

    Space selectors are deliberately absent: operator tau calibrates the
    space-local RW bank, not the semantic space lookup.  Q and K share
    operator rows but receive independent score tables and therefore
    independent quantiles under the common QK target.
    """
    max_tokens = _positive_int("max_tokens", int(max_tokens))
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    seq_len = input_ids.shape[1]
    token_count = min(max_tokens, int(input_ids.size))
    token_index = jnp.arange(token_count) * int(input_ids.size) // token_count
    token_ids = input_ids.reshape(-1)[token_index]
    positions = token_index % seq_len
    state = (params["token_emb"]["embedding"][token_ids]
             + params["pos_emb"]["embedding"][positions]).astype(jnp.float32)
    block = params["block_0"]
    attention_state = _shared_layer_norm(
        state, block["norm1"]["scale"], block["norm1"]["bias"])
    rst_state = _shared_layer_norm(
        state, block["norm2"]["scale"], block["norm2"]["bias"])
    router = params["router"]
    pool = params["neuron_pool"]
    keys = _pool_operator_keys(pool)

    def score(route_state, projection, operator_keys):
        query = _space_projection(route_state, projection)
        if operator_keys.ndim == 2:
            operator_keys = operator_keys[None, ...]
        query = _shared_forward_unit_direction(query.astype(jnp.float32))
        operator_keys = _shared_forward_unit_direction(
            operator_keys.astype(jnp.float32))
        return jnp.einsum("mtr,mnr->mtn", query, operator_keys)

    return {
        "q": score(attention_state, router["q_operation_proj"],
                   keys["qk_operator_keys"]),
        "k": score(attention_state, router["k_operation_proj"],
                   keys["qk_operator_keys"]),
        "v": score(attention_state, router["v_operation_proj"],
                   keys["v_operator_keys"]),
        "rst": score(rst_state, router["rst_operation_proj"],
                     keys["rst_operator_keys"]),
    }


def calibrate_operator_tau_per_space(
        scores_by_route: Mapping[str, jax.Array],
        *, target_qk_frac: float, target_v_frac: float,
        target_rst_frac: float) -> dict[str, jax.Array]:
    """Compute one quantile per route/space from all calibration tokens."""
    targets = {
        "q": float(target_qk_frac), "k": float(target_qk_frac),
        "v": float(target_v_frac), "rst": float(target_rst_frac)}
    result = {}
    for route in ROUTES:
        scores = jnp.asarray(scores_by_route[route], dtype=jnp.float32)
        if scores.ndim != 3:
            raise ValueError(f"{route} scores must have shape [M,T,N]")
        target = targets[route]
        if not 0.0 < target < 1.0:
            raise ValueError("tau target fractions must be in (0,1)")
        result[route] = jnp.quantile(
            scores.reshape((scores.shape[0], -1)), 1.0 - target, axis=1)
    return result


def _query_geometry_diagnostics(params, input_ids, max_tokens=4096):
    """Rare-cadence route-query geometry with canonical space terminology."""
    scores = _tau_init_calibration_scores(params, input_ids, max_tokens)
    out = {}
    for route, value in scores.items():
        out[f"{route}_operator_score_mean"] = value.mean()
        out[f"{route}_operator_score_std"] = value.std()
        out[f"{route}_operator_score_max"] = value.max()
    return out


def analysis_forward(params, model_cfg, input_ids, mode="full"):
    """Run the canonical module in analysis cadence without full state outputs."""
    del mode
    cfg = dict(model_cfg)
    model = DAWN_SRW_V4174(
        vocab_size=cfg.get("vocab_size", 30000),
        logical_vocab_size=cfg.get("logical_vocab_size"),
        vocab_size_padded=cfg.get("vocab_size_padded"),
        d_model=cfg.get("d_model", 384), n_layers=cfg.get("n_layers", 12),
        n_heads=cfg.get("n_heads", 6), max_seq_len=cfg.get("max_seq_len", 512),
        dropout_rate=0.0, router_dropout=0.0,
        d_route=cfg.get("d_route", 128), n_qk=cfg["n_qk"], n_v=cfg["n_v"],
        n_rst=cfg.get("n_rst", cfg.get("n_know")),
        n_operation_spaces=cfg.get("n_operation_spaces", 1),
        operation_space_top_k=cfg.get("operation_space_top_k", 1),
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
    variables = {"params": params}
    return model.apply(
        variables, input_ids, deterministic=True, analysis=True,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.PRNGKey(0)})


def _factory_profile_wrapper(factory: Callable, profile: str) -> Callable:
    @wraps(factory)
    def wrapped(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        kernel._v4174_kernel_profile = profile
        return kernel
    return wrapped


# Native v4174 reuses the proven, numerically identical RW shard-map kernels.
# M=1 and M>1 both keep the native v4174 parameter tree; the grouped wrappers
# below add only the leading operation-space axis for M>1.
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
    """Create a production ``[M,T,R]`` grouped-dense space kernel."""
    kernel = _shared_make_space_dense_minimal(*args, **kwargs)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "all_spaces"
    return kernel


@wraps(_shared_make_space_dense_diagnostics)
def make_sharded_space_dense_diagnostics(*args, **kwargs):
    """Create the matching observational kernel with ``[M,T,1]`` aggregates."""
    kernel = _shared_make_space_dense_diagnostics(*args, **kwargs)
    kernel._v4174_kernel_profile = "production_diagnostics"
    kernel._v4174_dense_grouped_diagnostics = "all_spaces"
    return kernel


@wraps(_shared_make_space_dense_minimal)
def make_sharded_qk_space_dense_minimal(*args, **kwargs):
    """Create the paired Q/K ``[M,T,2,R]`` production interface.

    Both routes consume the same sharded QK bank closure.  Their query/tau
    slices remain independent, and the paired route axis is eliminated before
    causal attention.
    """
    single = make_sharded_space_dense_minimal(*args, **kwargs)

    def paired(stacked_query, operator_keys, stacked_tau, token_valid,
               read_vectors, write_vectors, *runtime_scalars):
        q_output = single(
            stacked_query[:, :, 0], operator_keys, stacked_tau[:, :, 0],
            token_valid, read_vectors, write_vectors, *runtime_scalars)
        k_output = single(
            stacked_query[:, :, 1], operator_keys, stacked_tau[:, :, 1],
            token_valid, read_vectors, write_vectors, *runtime_scalars)
        return jnp.stack((q_output, k_output), axis=2)

    paired._v4174_kernel_profile = "production"
    paired._v4174_paired_qk_execution = "all_spaces"
    return paired


@wraps(_shared_make_space_dense_diagnostics)
def make_sharded_qk_space_dense_diagnostics(*args, **kwargs):
    """Create paired Q/K observational execution without full operator tensors."""
    single = make_sharded_space_dense_diagnostics(*args, **kwargs)

    def paired(stacked_query, operator_keys, stacked_tau, token_valid,
               read_vectors, write_vectors, *runtime_scalars):
        return (
            single(stacked_query[:, :, 0], operator_keys,
                   stacked_tau[:, :, 0], token_valid, read_vectors,
                   write_vectors, *runtime_scalars),
            single(stacked_query[:, :, 1], operator_keys,
                   stacked_tau[:, :, 1], token_valid, read_vectors,
                   write_vectors, *runtime_scalars),
        )

    paired._v4174_kernel_profile = "production_diagnostics"
    paired._v4174_paired_qk_execution = "all_spaces"
    return paired


def _validate_v4174_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs):
    """Validate common RW metadata while allowing v4174 space wrappers."""
    return _validate_shared_rw_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs)

def get_model_version() -> str:
    return MODEL_VERSION
