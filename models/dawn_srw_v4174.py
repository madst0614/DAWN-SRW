"""DAWN-SRW v4.1.7.4: explicit direct-read operation spaces.

The model owns one shared ``D -> R`` space-routing projection and one explicit
read vector per operation space.  Cosine scores select semantic support with a
hard top-k mask; the canonical space gate is ``ReLU(score)^2`` divided by the
square root of its mass and is never normalized with softmax.

Each operation space owns one encoder ``P_m`` and one decoder ``U_m``.  The
projected local state ``z_m = P_m x`` directly matches forward-normalized RW
read vectors: the read vector itself is the operator key.  Q, K, V, and RST
have fully separate read/write banks and tau projections.  Q/K/V share the
attention space gate, while RST recomputes routing and local state after the
attention residual update.  Semantic execution is hard top-k.  The canonical
dense executor remains available as a reference, while the production bundle
executor compactly gathers tokens into fixed four-space physical blocks.

This is the sole v4.1.7.4 architecture.  Earlier checkpoint, optimizer,
parameter-path, and config schemas are intentionally unsupported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import ad_checkpoint
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from models.dawn_srw_v4173 import (
    DAWNBlock as _SharedDAWNBlock,
    DEFAULT_ADMISSION_DEN_POWER,
    DEFAULT_HEAT_KERNEL_BETA,
    DEFAULT_SRW_COMPOSITION_MODE,
    RW_FORWARD_NORM_EPS,
    _composition_den as _shared_composition_den,
    _composition_den_floor_mass as _shared_composition_den_floor_mass,
    _compute_admission_drive as _shared_compute_admission_drive,
    _forward_unit_direction as _shared_forward_unit_direction,
    _layer_norm as _shared_layer_norm,
    _pool_output_scales as _shared_pool_output_scales,
    _raw_tau_init_from_cosine_tau,
    _tau_from_param as _shared_tau_from_param,
    safe_dropout as _shared_safe_dropout,
    scaled_normal as _shared_scaled_normal,
)


MODEL_VERSION = "spatial-r1-v4.1.7.4"
ROUTES = ("q", "k", "v", "rst")
POOLS = ROUTES
ATTENTION_CORE_NAME = "causal_dot_product_fp32_softmax"
LAYER_EXECUTION_NAME = "rematerialized"
OPERATION_SPACE_EXECUTION_MODES = ("dense_all_space", "bundle_dense")
BUNDLE_DENSE_SIZE = 4
DEFAULT_BUNDLE_TOKEN_BLOCK_SIZE = 1024
_BUNDLE_PACKING_CHECKPOINT_NAME = "v4174_bundle_packing_metadata"


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"model.{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_operation_space_config(model_cfg: Mapping[str, Any]) -> tuple[int, int]:
    """Validate the canonical operation-space routing fields."""
    n_spaces = _positive_int(
        "n_operation_spaces", model_cfg.get("n_operation_spaces"))
    top_k = _positive_int(
        "operation_space_top_k", model_cfg.get("operation_space_top_k"))
    if top_k > n_spaces:
        raise ValueError(
            "model.operation_space_top_k must be <= model.n_operation_spaces, "
            f"got {top_k} > {n_spaces}")
    return n_spaces, top_k


def materialize_operation_space_config(
        model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the fresh-run v4174 schema without compatibility conversion."""
    forbidden = tuple(
        name for name in (
            "n_qk", "n_know", "operator_key_mode", "operator_query_mode",
            "space_kernel_beta_qk", "space_kernel_beta_v",
            "space_kernel_beta_rst", "router_dropout", "tau_init_attn_qk")
        if name in model_cfg)
    if forbidden:
        raise ValueError(
            "v4174 does not accept removed model fields: "
            + ", ".join(forbidden))
    n_spaces, top_k = resolve_operation_space_config(model_cfg)
    execution_mode = str(model_cfg.get(
        "operation_space_execution_mode", "dense_all_space")).strip()
    if execution_mode not in OPERATION_SPACE_EXECUTION_MODES:
        raise ValueError(
            "model.operation_space_execution_mode must be one of "
            f"{OPERATION_SPACE_EXECUTION_MODES}, got {execution_mode!r}")
    if execution_mode == "bundle_dense":
        bundle_size = _positive_int(
            "operation_space_bundle_size",
            model_cfg.get("operation_space_bundle_size"))
        token_block_size = _positive_int(
            "operation_space_bundle_token_block_size",
            model_cfg.get("operation_space_bundle_token_block_size"))
        if bundle_size != BUNDLE_DENSE_SIZE:
            raise ValueError(
                "v4174 bundle_dense supports "
                f"model.operation_space_bundle_size={BUNDLE_DENSE_SIZE} "
                f"only, got {bundle_size}")
        if n_spaces != 24:
            raise ValueError(
                "v4174 bundle_dense production requires "
                f"model.n_operation_spaces=24, got {n_spaces}")
        if n_spaces % bundle_size:
            raise ValueError(
                "model.n_operation_spaces must be divisible by "
                "model.operation_space_bundle_size, got "
                f"{n_spaces} % {bundle_size}")
        if top_k != 2:
            raise ValueError(
                "v4174 bundle_dense requires "
                f"model.operation_space_top_k=2, got {top_k}")
        model_cfg["operation_space_bundle_size"] = bundle_size
        model_cfg[
            "operation_space_bundle_token_block_size"] = token_block_size
    elif (
            "operation_space_bundle_size" in model_cfg
            or "operation_space_bundle_token_block_size" in model_cfg):
        raise ValueError(
            "model.operation_space_bundle_* fields require "
            "model.operation_space_execution_mode='bundle_dense'")
    d_model = _positive_int("d_model", model_cfg.get("d_model"))
    d_route = _positive_int("d_route", model_cfg.get("d_route"))
    if d_route > d_model:
        raise ValueError(
            "v4174 requires the local route dimension to be no larger than "
            f"the model dimension, got d_route={d_route} > d_model={d_model}")
    if n_spaces > d_route:
        raise ValueError(
            "v4174 orthogonal space reads require "
            "model.n_operation_spaces <= model.d_route")
    for pool_name in ("n_q", "n_k", "n_v", "n_rst"):
        value = _positive_int(pool_name, model_cfg.get(pool_name))
        if value % n_spaces:
            raise ValueError(
                f"model.{pool_name} must be divisible by "
                f"model.n_operation_spaces, got {value} % {n_spaces}")
    for tau_name in (
            "tau_init_attn_q", "tau_init_attn_k",
            "tau_init_attn_v", "tau_init_rst"):
        if tau_name in model_cfg:
            value = float(model_cfg[tau_name])
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"model.{tau_name} must be a finite cosine in [-1, 1]")
            model_cfg[tau_name] = value
    model_cfg["n_operation_spaces"] = n_spaces
    model_cfg["operation_space_top_k"] = top_k
    model_cfg["operation_space_execution_mode"] = execution_mode
    return model_cfg


def symbolic_parameter_count(model_cfg: Mapping[str, Any]) -> dict[str, int]:
    """Count the exact canonical v4174 parameter tree."""
    if "model" in model_cfg and isinstance(model_cfg["model"], Mapping):
        model_cfg = model_cfg["model"]
    cfg = dict(model_cfg)
    materialize_operation_space_config(cfg)
    vocab = _positive_int(
        "vocab_size_padded",
        cfg.get("vocab_size_padded", cfg.get("vocab_size")))
    max_seq = _positive_int("max_seq_len", cfg.get("max_seq_len"))
    d_model = int(cfg["d_model"])
    d_route = int(cfg["d_route"])
    n_spaces = int(cfg["n_operation_spaces"])
    n_layers = _positive_int("n_layers", cfg.get("n_layers"))
    pool_total = sum(int(cfg[name]) for name in ("n_q", "n_k", "n_v", "n_rst"))
    shared_space_state = 2 * n_spaces * d_model * d_route
    space_router = d_model * d_route + n_spaces * d_route
    operator_tau = 4 * (d_route + 1)
    read_write_pools = 2 * d_route * pool_total
    counts = {
        "token_embedding": vocab * d_model,
        "position_embedding": max_seq * d_model,
        "layer_stack": n_layers * (d_model * d_model + 4 * d_model),
        "shared_space_state": shared_space_state,
        "space_router": space_router,
        "operator_tau": operator_tau,
        "read_write_pools": read_write_pools,
        "learned_key_tables": 0,
        "bilinear_probe_matrices": 0,
        "operator_query_projections": 0,
        "final_norm": 2 * d_model,
    }
    counts["router"] = shared_space_state + space_router + operator_tau
    counts["total"] = (
        counts["token_embedding"]
        + counts["position_embedding"]
        + counts["layer_stack"]
        + counts["router"]
        + read_write_pools
        + counts["final_norm"])
    return counts


def _independent_space_projection_init(
        key: jax.Array, shape: tuple[int, ...],
        dtype=jnp.float32) -> jax.Array:
    """Initialize distinct semi-orthogonal ``D -> R`` maps for every space."""
    n_spaces, d_model, d_route = map(int, shape)
    if d_route > d_model:
        raise ValueError(
            "space projection initialization requires d_route <= d_model")
    basis_key, sign_key = jax.random.split(key)
    base = nn.initializers.orthogonal(scale=1.0)(
        basis_key, (d_model, d_route), jnp.float32)
    signs = jnp.where(
        jax.random.bernoulli(
            sign_key, shape=(n_spaces, d_model, 1)),
        jnp.float32(1.0),
        jnp.float32(-1.0))
    return (base[None, :, :] * signs).astype(dtype)


def _orthogonal_space_read_init(
        key: jax.Array, shape: tuple[int, ...],
        dtype=jnp.float32) -> jax.Array:
    """Use the first M rows of an orthogonal ``[R,R]`` matrix."""
    n_spaces, d_route = map(int, shape)
    if n_spaces > d_route:
        raise ValueError("orthogonal space-read initialization requires M <= R")
    matrix = nn.initializers.orthogonal(scale=1.0)(
        key, (d_route, d_route), jnp.float32)
    return matrix[:n_spaces].astype(dtype)


def _isotropic_unit_init(
        key: jax.Array, shape: tuple[int, ...],
        dtype=jnp.float32) -> jax.Array:
    """Independent isotropic unit directions for each RW row."""
    d_route = int(shape[-1])
    values = jax.random.normal(key, shape, dtype=jnp.float32)
    values = values / math.sqrt(d_route)
    return _shared_forward_unit_direction(values).astype(dtype)


def _constant_tau_bias(cosine_tau: float) -> Callable:
    value = _raw_tau_init_from_cosine_tau(
        min(max(float(cosine_tau), -0.9998), 0.9998))

    def initialize(key, shape, dtype=jnp.float32):
        del key
        return jnp.full(shape, value, dtype)

    return initialize


class OperationSpaceNeuronPool(nn.Module):
    """Fully separate, space-indexed Q/K/V/RST read/write banks."""
    n_q: int
    n_k: int
    n_v: int
    n_rst: int
    d_route: int
    n_operation_spaces: int

    def setup(self):
        n_spaces = int(self.n_operation_spaces)
        d_route = int(self.d_route)
        for name, count in (
                ("q", self.n_q), ("k", self.n_k),
                ("v", self.n_v), ("rst", self.n_rst)):
            shape = (n_spaces, int(count) // n_spaces, d_route)
            setattr(self, f"{name}_read_vectors", self.param(
                f"{name}_read_vectors", _isotropic_unit_init, shape))
            setattr(self, f"{name}_write_vectors", self.param(
                f"{name}_write_vectors", _isotropic_unit_init, shape))


class OperationSpaceRouter(nn.Module):
    """Shared routing/local geometry and four independent local tau maps."""
    d_model: int
    d_route: int
    n_operation_spaces: int
    tau_init_attn_q: float
    tau_init_attn_k: float
    tau_init_attn_v: float
    tau_init_rst: float

    def setup(self):
        n_spaces = int(self.n_operation_spaces)
        d_model = int(self.d_model)
        d_route = int(self.d_route)
        self.space_route_proj = nn.Dense(
            d_route, use_bias=False, name="space_route_proj",
            kernel_init=nn.initializers.orthogonal(scale=1.0))
        self.space_read_vectors = self.param(
            "space_read_vectors", _orthogonal_space_read_init,
            (n_spaces, d_route))
        self.space_state_proj = self.param(
            "space_state_proj", _independent_space_projection_init,
            (n_spaces, d_model, d_route))
        self.space_state_writeback = self.param(
            "space_state_writeback",
            lambda key, shape, dtype=jnp.float32: jnp.swapaxes(
                self.space_state_proj, -1, -2).astype(dtype),
            (n_spaces, d_route, d_model))
        tau_values = {
            "q": self.tau_init_attn_q,
            "k": self.tau_init_attn_k,
            "v": self.tau_init_attn_v,
            "rst": self.tau_init_rst,
        }
        for route in ROUTES:
            setattr(self, f"{route}_operator_tau_proj", nn.Dense(
                1, name=f"{route}_operator_tau_proj",
                kernel_init=nn.initializers.zeros,
                bias_init=_constant_tau_bias(tau_values[route])))


def _linear(params: Mapping[str, jax.Array], state: jax.Array) -> jax.Array:
    return state @ params["kernel"] + params.get("bias", 0.0)


def throughput_dot_bf16_f32(
        lhs: jax.Array,
        rhs: jax.Array, *,
        dimension_numbers,
        precision=jax.lax.Precision.DEFAULT) -> jax.Array:
    """Run a throughput dot with BF16 operands and an FP32 result."""
    return jax.lax.dot_general(
        jnp.asarray(lhs).astype(jnp.bfloat16),
        jnp.asarray(rhs).astype(jnp.bfloat16),
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=jnp.float32)


def control_dot_f32(
        lhs: jax.Array,
        rhs: jax.Array, *,
        dimension_numbers) -> jax.Array:
    """Run a routing/control dot entirely in FP32."""
    return jax.lax.dot_general(
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(rhs, dtype=jnp.float32),
        dimension_numbers=dimension_numbers,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32)


def _throughput_einsum_bf16_f32(
        equation: str,
        lhs: jax.Array,
        rhs: jax.Array) -> jax.Array:
    return jnp.einsum(
        equation,
        jnp.asarray(lhs).astype(jnp.bfloat16),
        jnp.asarray(rhs).astype(jnp.bfloat16),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32)


def _control_einsum_f32(
        equation: str,
        lhs: jax.Array,
        rhs: jax.Array) -> jax.Array:
    return jnp.einsum(
        equation,
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(rhs, dtype=jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32)


def _throughput_linear_bf16_f32(
        params: Mapping[str, jax.Array],
        state: jax.Array) -> jax.Array:
    result = throughput_dot_bf16_f32(
        state, params["kernel"],
        dimension_numbers=(
            ((state.ndim - 1,), (0,)),
            ((), ())))
    return result + jnp.asarray(params.get("bias", 0.0), dtype=jnp.float32)


def _control_linear_f32(
        params: Mapping[str, jax.Array],
        state: jax.Array) -> jax.Array:
    result = control_dot_f32(
        state, params["kernel"],
        dimension_numbers=(
            ((state.ndim - 1,), (0,)),
            ((), ())))
    return result + jnp.asarray(params.get("bias", 0.0), dtype=jnp.float32)


def forward_unit_direction(value: jax.Array) -> jax.Array:
    """Forward-normalize a state/read tensor in FP32."""
    return _shared_forward_unit_direction(jnp.asarray(value, dtype=jnp.float32))


def _project_space_local_states(
        state: jax.Array, space_state_proj: jax.Array) -> jax.Array:
    """Project a shared ``[T,D]`` state into ``[M,T,R]``."""
    if state.ndim != 2 or space_state_proj.ndim != 3:
        raise ValueError("state/projection must have shapes [T,D] and [M,D,R]")
    return jnp.einsum("td,mdr->mtr", state, space_state_proj)


def _compute_space_routing(
        flat_state: jax.Array,
        space_route_kernel: jax.Array,
        space_read_vectors: jax.Array,
        operation_space_top_k: int) -> dict[str, jax.Array]:
    """Direct cosine routing with hard top-k ReLU-squared non-softmax gates."""
    route_state = control_dot_f32(
        flat_state, space_route_kernel,
        dimension_numbers=(((1,), (0,)), ((), ())))
    normalized_state = forward_unit_direction(route_state)
    normalized_reads = forward_unit_direction(space_read_vectors)
    space_scores = control_dot_f32(
        normalized_state, normalized_reads,
        dimension_numbers=(((1,), (1,)), ((), ())))
    gate_values = _space_gate_from_scores(
        space_scores, operation_space_top_k)
    return {
        "route_state": route_state,
        "space_scores": space_scores,
        **gate_values,
    }


def _space_gate_from_scores(
        space_scores: jax.Array,
        operation_space_top_k: int) -> dict[str, jax.Array]:
    """Apply the fixed hard-top-k ReLU-squared sqrt-mass space gate."""
    selected_scores, selected_ids = jax.lax.top_k(
        space_scores, int(operation_space_top_k))
    top_k_mask = jnp.sum(
        jax.nn.one_hot(
            selected_ids, int(space_scores.shape[-1]),
            dtype=jnp.float32),
        axis=-2)
    raw_space_gate = jnp.square(jax.nn.relu(space_scores))
    space_gate = raw_space_gate * top_k_mask
    space_gate_mass = space_gate.sum(axis=-1, keepdims=True)
    space_gate_den = jnp.sqrt(jnp.maximum(
        space_gate_mass, jnp.float32(1.0)))
    dense_space_weights = space_gate / space_gate_den
    return {
        "selected_scores": selected_scores,
        "selected_ids": selected_ids,
        "space_gate": space_gate,
        "space_gate_mass": space_gate_mass,
        "space_gate_den": space_gate_den,
        "dense_space_weights": dense_space_weights,
    }


def _direct_read_match(
        space_local_states: jax.Array,
        read_vectors: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reuse one read matmul for the scalar read and direct cosine score."""
    read_unit = forward_unit_direction(
        read_vectors.astype(jnp.float32))
    read_value = jnp.einsum(
        "mtr,mnr->mtn",
        space_local_states.astype(jnp.bfloat16),
        read_unit.astype(jnp.bfloat16),
    ).astype(jnp.float32)
    local_norm = jnp.linalg.norm(
        space_local_states.astype(jnp.float32),
        axis=-1, keepdims=True)
    rho = read_value / jnp.maximum(
        local_norm, jnp.float32(RW_FORWARD_NORM_EPS))
    return read_value, jnp.clip(rho, -1.0, 1.0), local_norm


def _rw_compose_space_dense(
        space_local_states: jax.Array,
        read_vectors: jax.Array,
        write_vectors: jax.Array,
        raw_operator_tau: jax.Array, *,
        soft_gate_temperature: float,
        soft_gate_boundary_power: float,
        admission_den_power: float,
        srw_composition_mode: str,
        heat_kernel_beta: float,
        execution_prune_eps: float = 0.0,
        max_chunk_size: int = 2048,
        diagnostics: bool = False):
    """Execute direct-read RW composition densely over all spaces."""
    if space_local_states.ndim != 3:
        raise ValueError("space_local_states must have shape [M,T,R]")
    if read_vectors.ndim != 3 or write_vectors.ndim != 3:
        raise ValueError("read/write vectors must have shape [M,N,R]")
    n_spaces, token_count, d_route = map(int, space_local_states.shape)
    n_operator = int(read_vectors.shape[1])
    chunk = min(max(1, int(max_chunk_size)), n_operator)
    n_chunks = math.ceil(n_operator / chunk)
    padded = n_chunks * chunk
    padding = ((0, 0), (0, padded - n_operator), (0, 0))
    reads = jnp.pad(read_vectors, padding)
    writes = jnp.pad(write_vectors, padding)
    valid = jnp.arange(padded) < n_operator
    read_unit = forward_unit_direction(reads).astype(jnp.bfloat16)
    write_unit = forward_unit_direction(writes).astype(jnp.bfloat16)
    local_bf16 = space_local_states.astype(jnp.bfloat16)
    local_norm = jnp.linalg.norm(
        space_local_states.astype(jnp.float32),
        axis=-1, keepdims=True)
    local_norm = jnp.maximum(
        local_norm, jnp.float32(RW_FORWARD_NORM_EPS))
    tau = _shared_tau_from_param(raw_operator_tau)
    aggregate_shape = (n_spaces, token_count, 1)
    carry = (
        jnp.zeros((n_spaces, token_count, d_route), jnp.float32),
        jnp.zeros(aggregate_shape, jnp.float32),
        jnp.zeros(aggregate_shape, jnp.float32),
        jnp.zeros(aggregate_shape, jnp.float32),
        jnp.zeros(aggregate_shape, jnp.float32),
        jnp.zeros(aggregate_shape, jnp.float32),
    )

    def step(carry_value, chunk_index):
        raw_out, gate_mass, gate_sq, gate_max, active_count, depth_sum = (
            carry_value)
        start = chunk_index * chunk
        read = jax.lax.dynamic_slice_in_dim(
            read_unit, start, chunk, axis=1)
        write = jax.lax.dynamic_slice_in_dim(
            write_unit, start, chunk, axis=1)
        row_valid = jax.lax.dynamic_slice_in_dim(
            valid, start, chunk, axis=0)
        valid_mask = row_valid[None, None, :]
        read_value = jnp.einsum(
            "mtr,mnr->mtn", local_bf16, read).astype(jnp.float32)
        rho = jnp.clip(read_value / local_norm, -1.0, 1.0)
        rho = jnp.where(valid_mask, rho, tau)
        margin, gate, depth, execution_weight, _ = (
            _shared_compute_admission_drive(
                rho, tau, soft_gate_temperature,
                boundary_power=soft_gate_boundary_power,
                effective_active_eps=jnp.float32(1.0e-6),
                execution_prune_eps=execution_prune_eps,
                srw_composition_mode=srw_composition_mode,
                heat_kernel_beta=jnp.float32(heat_kernel_beta)))
        gate = jnp.where(valid_mask, gate, 0.0)
        depth = jnp.where(valid_mask, depth, 0.0)
        execution_weight = jnp.where(valid_mask, execution_weight, 0.0)
        chunk_out = jnp.einsum(
            "mtn,mnr->mtr",
            (execution_weight * read_value).astype(jnp.bfloat16),
            write).astype(jnp.float32)
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
    return (
        space_results, active_count, gate_mass, gate_sq,
        gate_max, depth_sum, tau, gate_den)


def _space_weighted_writeback(
        space_results: jax.Array,
        dense_space_weights: jax.Array,
        space_state_writeback: jax.Array,
        route_scale: jax.Array | float) -> jax.Array:
    """Fuse space weights and ``U_m`` without materializing ``[M,T,D]``."""
    weighted_local = (
        space_results
        * jnp.swapaxes(dense_space_weights, 0, 1)[..., None]
        * jnp.asarray(route_scale))
    return jnp.einsum(
        "mtr,mrd->td", weighted_local, space_state_writeback)


def _space_routing_metrics(
        routing: Mapping[str, jax.Array],
        prefix: str) -> dict[str, jax.Array]:
    gate = routing["space_gate"].astype(jnp.float32)
    mass = routing["space_gate_mass"][..., 0].astype(jnp.float32)
    den = routing["space_gate_den"][..., 0].astype(jnp.float32)
    selected_ids = routing["selected_ids"]
    n_spaces = int(gate.shape[-1])
    active = gate > 0.0
    top1_usage = jax.nn.one_hot(
        selected_ids[..., 0], n_spaces, dtype=jnp.float32
    ).reshape((-1, n_spaces)).mean(axis=0)
    usage = active.astype(jnp.float32).reshape((-1, n_spaces)).mean(axis=0)
    values = {
        f"{prefix}_space_gate_mass_mean": mass.mean(),
        f"{prefix}_space_gate_den_mean": den.mean(),
        f"{prefix}_space_active_count_mean": (
            active.astype(jnp.float32).sum(axis=-1).mean()),
        f"{prefix}_space_zero_gate_frac": (
            mass <= 0.0).astype(jnp.float32).mean(),
        f"{prefix}_space_top1_rate": top1_usage.max(),
        f"{prefix}_space_usage_min": usage.min(),
        f"{prefix}_space_usage_max": usage.max(),
        f"{prefix}_space_usage_std": usage.std(),
    }
    diagnostic_probability = gate / jnp.maximum(
        mass[..., None], jnp.float32(1.0e-8))
    values[f"{prefix}_space_selected_entropy"] = (
        -diagnostic_probability
        * jnp.log(jnp.maximum(
            diagnostic_probability, jnp.float32(1.0e-8)))
    ).sum(axis=-1).mean()
    return {
        key: jax.lax.stop_gradient(value)
        for key, value in values.items()}


def _compact_space_routing_metrics(
        routing: Mapping[str, jax.Array],
        prefix: str) -> dict[str, jax.Array]:
    """Return only scalar routing metrics used by regular training."""
    gate = routing["space_gate"].astype(jnp.float32)
    mass = routing["space_gate_mass"][..., 0].astype(jnp.float32)
    selected_ids = routing["selected_ids"]
    n_spaces = int(gate.shape[-1])
    top1_usage = jax.nn.one_hot(
        selected_ids[..., 0], n_spaces, dtype=jnp.float32
    ).reshape((-1, n_spaces)).mean(axis=0)
    values = {
        f"{prefix}_space_gate_mass_mean": mass.mean(),
        f"{prefix}_space_active_count_mean": (
            (gate > 0.0).astype(jnp.float32).sum(axis=-1).mean()),
        f"{prefix}_space_zero_gate_frac": (
            mass <= 0.0).astype(jnp.float32).mean(),
        f"{prefix}_space_top1_rate": top1_usage.max(),
    }
    return {
        key: jax.lax.stop_gradient(value)
        for key, value in values.items()}


def _aggregate_operator_diagnostics(
        diagnostics_values: tuple[jax.Array, ...],
        n_operators_per_space: int,
        route: str,
        srw_composition_mode: str,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    (space_results, active_count, gate_mass, gate_sq, gate_max,
     depth_sum, tau, gate_den) = diagnostics_values
    floor_mass = _shared_composition_den_floor_mass(srw_composition_mode)
    floor_compare = (
        jnp.less if srw_composition_mode == "linear_angular"
        else jnp.less_equal)
    metrics = {
        f"{route}_operator_active_tau_frac": (
            active_count / float(n_operators_per_space)).mean(),
        f"{route}_operator_active_tau_count": active_count.mean(),
        f"{route}_operator_tau_mean": tau.mean(),
        f"{route}_operator_gate_mass_mean": gate_mass.mean(),
        f"{route}_operator_gate_den_mean": gate_den.mean(),
        f"{route}_operator_depth_active_mean": (
            depth_sum / jnp.maximum(active_count, 1.0)).mean(),
        f"{route}_operator_gate_eff_n_mean": (
            jnp.square(gate_mass) / jnp.maximum(gate_sq, 1.0e-8)).mean(),
        f"{route}_operator_top1_gate_frac_mean": (
            gate_max / jnp.maximum(gate_mass, 1.0e-8)).mean(),
        f"{route}_operator_den_floor_frac": floor_compare(
            gate_mass, floor_mass).astype(jnp.float32).mean(),
    }
    per_space = {
        f"{route}_per_space_active_frac": (
            active_count[..., 0] / float(n_operators_per_space)).mean(axis=1),
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
    """Map compact native route scalars to the stable regular-log schema."""
    qk = lambda suffix: jnp.float32(0.5) * (
        metrics[f"q_operator_{suffix}"] + metrics[f"k_operator_{suffix}"])
    source_suffixes = {
        "active_tau_frac": "active_tau_frac",
        "active_tau_count": "active_tau_count",
        "gate_mass": "gate_mass_mean",
        "gate_den": "gate_den_mean",
        "depth_active": "depth_active_mean",
        "gate_eff_n": "gate_eff_n_mean",
        "top1_gate_frac": "top1_gate_frac_mean",
        "den_floor_frac": "den_floor_frac",
        "tau_mean": "tau_mean",
    }
    prefixes = {
        "q": "attn_q",
        "k": "attn_k",
        "v": "attn_v",
        "rst": "rst",
    }
    out = {
        f"{prefix}_{destination}": metrics[f"{route}_operator_{source}"]
        for route, prefix in prefixes.items()
        for destination, source in source_suffixes.items()
    }
    for destination in source_suffixes:
        out[f"attn_qk_{destination}"] = jnp.float32(0.5) * (
            out[f"attn_q_{destination}"]
            + out[f"attn_k_{destination}"])
    out.update({
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
    })
    return {
        key: jax.lax.stop_gradient(value)
        for key, value in out.items()}


def _off_diagonal_entries(value: jax.Array) -> jax.Array:
    """Return the leading square matrix's off-diagonal entries JIT-safely."""
    size = int(value.shape[0])
    if size <= 1:
        return value.reshape((size * size,) + value.shape[2:])[:0]
    flat_index = jnp.arange(size * (size - 1), dtype=jnp.int32)
    row = flat_index // (size - 1)
    column = flat_index % (size - 1)
    column = column + (column >= row)
    return value[row, column]


def _space_geometry_diagnostics(
        space_state_proj: jax.Array,
        space_state_writeback: jax.Array,
        state: Optional[jax.Array] = None) -> dict[str, jax.Array]:
    """Projection rank, overlap, principal angles, and local covariance."""
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
    out = {
        "stacked_projection_singular_values": singular,
        "stacked_projection_rank": jnp.linalg.matrix_rank(stacked),
        "stacked_projection_effective_rank": effective_rank,
        "space_projection_pairwise_overlap": _off_diagonal_entries(overlap),
        "space_subspace_principal_angles": _off_diagonal_entries(
            principal_angles),
        "encoder_writeback_init_error": jnp.max(jnp.abs(
            jnp.swapaxes(projection, -1, -2) - space_state_writeback)),
    }
    if state is not None:
        local = _project_space_local_states(state, projection)
        covariance = jnp.einsum(
            "mtr,nts->mnrs", local, local) / jnp.maximum(state.shape[0], 1)
        covariance_norm = jnp.linalg.norm(covariance, axis=(-2, -1))
        covariance_diag = jnp.sqrt(jnp.maximum(
            jnp.diag(covariance_norm), 1.0e-8))
        covariance_overlap = covariance_norm / (
            covariance_diag[:, None] * covariance_diag[None, :])
        variance = jnp.var(local, axis=1).sum(axis=-1)
        out.update({
            "space_local_state_covariance_overlap": _off_diagonal_entries(
                covariance_overlap),
            "space_local_state_norm": jnp.linalg.norm(
                local, axis=-1).mean(axis=-1),
            "space_explained_variance": variance / jnp.maximum(
                variance.sum(), 1.0e-8),
        })
    return out


def _read_bank_geometry_diagnostics(
        pool_params: Mapping[str, jax.Array],
        space_read_vectors: jax.Array,
        route_state: jax.Array) -> dict[str, jax.Array]:
    """Direct-read geometry for explicit space reads and all four RW pools."""
    space_reads = forward_unit_direction(space_read_vectors)
    pairwise = space_reads @ space_reads.T
    out = {
        "space_read_pairwise_cosine": _off_diagonal_entries(pairwise),
        "space_read_norm": jnp.linalg.norm(
            space_read_vectors.astype(jnp.float32), axis=-1),
        "space_route_state_norm": jnp.linalg.norm(
            route_state.astype(jnp.float32), axis=-1).mean(),
    }
    for pool in POOLS:
        read = forward_unit_direction(pool_params[f"{pool}_read_vectors"])
        write = forward_unit_direction(pool_params[f"{pool}_write_vectors"])
        covariance = jnp.einsum(
            "mnr,mns->mrs", read, read) / read.shape[1]
        eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(covariance), 0.0)
        probability = eigenvalues / jnp.maximum(
            eigenvalues.sum(axis=-1, keepdims=True), 1.0e-8)
        effective_rank = jnp.exp(-jnp.sum(
            probability * jnp.log(jnp.maximum(probability, 1.0e-8)),
            axis=-1))
        sample_count = min(128, int(read.shape[1]))
        sampled = read[:, :sample_count]
        cross = jnp.einsum("mir,njr->mnij", sampled, sampled)
        cross_nearest = cross.max(axis=(-2, -1))
        out.update({
            f"{pool}_read_vector_covariance_effective_rank": effective_rank,
            f"{pool}_cross_space_nearest_read_similarity": (
                _off_diagonal_entries(cross_nearest)),
            f"{pool}_read_norm": jnp.linalg.norm(
                pool_params[f"{pool}_read_vectors"].astype(jnp.float32),
                axis=-1).mean(axis=-1),
            f"{pool}_write_norm": jnp.linalg.norm(
                pool_params[f"{pool}_write_vectors"].astype(jnp.float32),
                axis=-1).mean(axis=-1),
        })
    return out


def _causal_attention_core(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        dropout_rate: float,
        deterministic: bool,
        dropout_rng: jax.Array, *,
        throughput_bf16: bool = False) -> jax.Array:
    """Production causal attention with FP32 score/softmax semantics."""
    d_head = int(query.shape[-1])
    sequence_length = int(query.shape[-2])
    if throughput_bf16:
        scores = _throughput_einsum_bf16_f32(
            "bhsd,bhtd->bhst", query, key)
    else:
        scores = jnp.einsum(
            "bhsd,bhtd->bhst", query, key)
    scores = scores.astype(jnp.float32) / jnp.sqrt(jnp.float32(d_head))
    causal = jnp.tril(jnp.ones(
        (sequence_length, sequence_length), dtype=jnp.bool_))
    scores = jnp.where(
        causal, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    weights = _shared_safe_dropout(
        weights, dropout_rate, deterministic, dropout_rng)
    if throughput_bf16:
        return _throughput_einsum_bf16_f32(
            "bhst,bhtd->bhsd", weights, value).astype(jnp.float32)
    return jnp.einsum(
        "bhst,bhtd->bhsd", weights, value).astype(jnp.float32)


class DAWN_SRW_V4174(nn.Module):
    """Canonical explicit-space-read, direct-operator-read DAWN-SRW."""
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
    admission_den_power: float = DEFAULT_ADMISSION_DEN_POWER
    admission_den_power_qk: Optional[float] = None
    admission_den_power_v: Optional[float] = None
    admission_den_power_rst: Optional[float] = None
    srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE
    heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA
    n_q: int = 792
    n_k: int = 792
    n_v: int = 2601
    n_rst: int = 25200
    n_operation_spaces: int = 3
    operation_space_top_k: int = 2
    operation_space_execution_mode: str = "dense_all_space"
    operation_space_bundle_size: Optional[int] = None
    operation_space_bundle_token_block_size: Optional[int] = None
    n_chunks_rst: int = 1
    n_chunks_q: int = 1
    n_chunks_k: int = 1
    n_chunks_v: int = 1
    tau_init_attn_q: Optional[float] = None
    tau_init_attn_k: Optional[float] = None
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
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        config = {
            "d_model": self.d_model,
            "d_route": self.d_route,
            "n_operation_spaces": self.n_operation_spaces,
            "operation_space_top_k": self.operation_space_top_k,
            "operation_space_execution_mode":
                self.operation_space_execution_mode,
            "n_q": self.n_q,
            "n_k": self.n_k,
            "n_v": self.n_v,
            "n_rst": self.n_rst,
        }
        if self.operation_space_bundle_size is not None:
            config["operation_space_bundle_size"] = (
                self.operation_space_bundle_size)
        if self.operation_space_bundle_token_block_size is not None:
            config["operation_space_bundle_token_block_size"] = (
                self.operation_space_bundle_token_block_size)
        materialize_operation_space_config(config)
        tau_values = (
            self.tau_init_attn_q, self.tau_init_attn_k,
            self.tau_init_attn_v, self.tau_init_rst)
        if any(value is None for value in tau_values):
            raise ValueError(
                "v4174 requires explicit q/k/v/rst initial operator tau values")
        _, embedding_vocab = self._vocab_sizes()
        self.token_emb = nn.Embed(
            embedding_vocab, self.d_model,
            embedding_init=_shared_scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model,
            embedding_init=_shared_scaled_normal(0.02))
        self.neuron_pool = OperationSpaceNeuronPool(
            n_q=self.n_q, n_k=self.n_k, n_v=self.n_v, n_rst=self.n_rst,
            d_route=self.d_route,
            n_operation_spaces=self.n_operation_spaces)
        self.router = OperationSpaceRouter(
            d_model=self.d_model, d_route=self.d_route,
            n_operation_spaces=self.n_operation_spaces,
            tau_init_attn_q=float(self.tau_init_attn_q),
            tau_init_attn_k=float(self.tau_init_attn_k),
            tau_init_attn_v=float(self.tau_init_attn_v),
            tau_init_rst=float(self.tau_init_rst))
        self.layers = [
            _SharedDAWNBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                dropout_rate=self.dropout_rate, name=f"block_{index}")
            for index in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def _realize_parameters(self, state: jax.Array) -> None:
        _ = self.neuron_pool.q_read_vectors
        _ = self.neuron_pool.k_read_vectors
        _ = self.neuron_pool.v_read_vectors
        _ = self.neuron_pool.rst_read_vectors
        _ = self.router.space_route_proj(state)
        local = jnp.einsum(
            "bsd,mdr->mbsr", state, self.router.space_state_proj)
        for route in ROUTES:
            _ = getattr(self.router, f"{route}_operator_tau_proj")(local)
        _ = self.router.space_read_vectors
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
            collect_train_metrics=True,
            **analysis_kwargs):
        """Run v4174 with the statically selected physical executor."""
        del (
            attention_mask, soft_gate_t_final,
            soft_gate_boundary_power_final, admission_den_power,
            heat_kernel_beta, minimal_train, ce_token_chunk_size,
            analysis_kwargs)
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        n_spaces = int(self.n_operation_spaces)
        top_k = int(self.operation_space_top_k)
        composition_mode = str(
            srw_composition_mode or self.srw_composition_mode)
        den_qk = float(
            self.admission_den_power_qk
            if self.admission_den_power_qk is not None
            else self.admission_den_power)
        den_powers = {
            "q": den_qk,
            "k": den_qk,
            "v": float(
                self.admission_den_power_v
                if self.admission_den_power_v is not None
                else self.admission_den_power),
            "rst": float(
                self.admission_den_power_rst
                if self.admission_den_power_rst is not None
                else self.admission_den_power),
        }
        temp_qk = jnp.asarray(
            soft_gate_temperature
            if soft_gate_T_qk is None else soft_gate_T_qk,
            dtype=jnp.float32)
        temperatures = {
            "q": temp_qk,
            "k": temp_qk,
            "v": jnp.asarray(
                soft_gate_temperature
                if soft_gate_T_v is None else soft_gate_T_v,
                dtype=jnp.float32),
            "rst": jnp.asarray(
                soft_gate_temperature
                if soft_gate_T_rst is None else soft_gate_T_rst,
                dtype=jnp.float32),
        }
        positions = jnp.arange(input_ids.shape[1])[None, :]
        vocab_embed = (
            sharded_fns.get("vocab_parallel_embedding")
            if isinstance(sharded_fns, dict) else None)
        state = (
            vocab_embed(input_ids, self.token_emb.embedding)
            if vocab_embed is not None else self.token_emb(input_ids))
        state = state + self.pos_emb(positions)
        state = _shared_safe_dropout(
            state, self.dropout_rate, deterministic,
            self.make_rng("dropout"))
        if self.is_mutable_collection("params"):
            self._realize_parameters(state)
            state = self.norm(state)
            logical, embedding = self._vocab_sizes()
            logits = self.token_emb.attend(state)
            return {
                "logits": logits[..., :logical]
                if embedding != logical else logits}

        params = self.variables["params"]
        pool = params["neuron_pool"]
        router = params["router"]
        route_counts = {
            "q": int(self.n_q),
            "k": int(self.n_k),
            "v": int(self.n_v),
            "rst": int(self.n_rst),
        }
        route_chunks = {
            "q": max(1, math.ceil(
                route_counts["q"] / n_spaces / max(1, int(self.n_chunks_q)))),
            "k": max(1, math.ceil(
                route_counts["k"] / n_spaces / max(1, int(self.n_chunks_k)))),
            "v": max(1, math.ceil(
                route_counts["v"] / n_spaces / max(1, int(self.n_chunks_v)))),
            "rst": max(1, math.ceil(
                route_counts["rst"] / n_spaces
                / max(1, int(self.n_chunks_rst)))),
        }
        qk_scale, v_scale, rst_scale = _shared_pool_output_scales(
            self.d_model, self.n_layers)
        route_scales = {
            "q": qk_scale, "k": qk_scale,
            "v": v_scale, "rst": rst_scale}
        layer_rngs = jax.random.split(
            self.make_rng("dropout"), self.n_layers)
        diagnostics_enabled = (
            minimal_runtime_profile == "diagnostics" or analysis)
        collect_regular_metrics = (
            jnp.asarray(True, dtype=jnp.bool_)
            if diagnostics_enabled
            else jnp.asarray(
                collect_train_metrics, dtype=jnp.bool_).reshape(()))
        attention_dense = (
            sharded_fns.get("attention_space_dense")
            if isinstance(sharded_fns, dict) else None)
        rst_dense = (
            sharded_fns.get("rst_space_dense")
            if isinstance(sharded_fns, dict) else None)
        fused_production = (
            not diagnostics_enabled
            and attention_dense is not None)
        if fused_production and rst_dense is None:
            raise ValueError(
                "v4174 production requires both attention_space_dense and "
                "rst_space_dense executors")
        fused_throughput_precision = (
            getattr(
                attention_dense, "_v4174_throughput_precision", None)
            if fused_production else None)
        if fused_production and fused_throughput_precision not in (
                "bf16_operands_f32_accum", "fp32_reference"):
            raise ValueError(
                "v4174 fused attention executor is missing its static "
                "throughput precision contract")
        if (fused_production
                and getattr(
                    rst_dense, "_v4174_throughput_precision", None)
                != fused_throughput_precision):
            raise ValueError(
                "v4174 attention/RST executors require identical static "
                "throughput precision")
        fused_execution_mode = (
            getattr(
                attention_dense, "_v4174_execution_mode",
                "dense_all_space")
            if fused_production else None)
        if (fused_production
                and getattr(
                    rst_dense, "_v4174_execution_mode",
                    "dense_all_space")
                != fused_execution_mode):
            raise ValueError(
                "v4174 attention/RST executors require identical static "
                "operation-space execution modes")
        if (fused_production
                and fused_execution_mode
                != self.operation_space_execution_mode):
            raise ValueError(
                "v4174 configured/factory operation-space execution mode "
                f"mismatch: {self.operation_space_execution_mode!r} != "
                f"{fused_execution_mode!r}")
        fused_throughput_bf16 = (
            fused_throughput_precision == "bf16_operands_f32_accum")
        regular_lists: dict[str, list[jax.Array]] = {}
        analysis_lists: dict[str, list[jax.Array]] = {}
        last_local_input = None
        last_route_state = None
        scanned_regular_metrics: dict[str, jax.Array] = {}

        def append(target, values):
            for key, value in values.items():
                target.setdefault(key, []).append(value)

        def execute(route, local_states, raw_tau, pool_params):
            read = pool_params[f"{route}_read_vectors"]
            write = pool_params[f"{route}_write_vectors"]
            direct_kernel = (
                sharded_fns.get(f"{route}_space_dense")
                if isinstance(sharded_fns, dict) else None)
            if direct_kernel is not None:
                token_valid = jnp.ones(
                    local_states.shape[:2], dtype=jnp.bool_)
                return direct_kernel(
                    local_states, raw_tau, token_valid, read, write,
                    temperatures[route], temperatures[route],
                    soft_gate_boundary_power, soft_gate_boundary_power,
                    execution_prune_eps)
            return _rw_compose_space_dense(
                local_states, read, write, raw_tau,
                soft_gate_temperature=temperatures[route],
                soft_gate_boundary_power=soft_gate_boundary_power,
                admission_den_power=den_powers[route],
                srw_composition_mode=composition_mode,
                heat_kernel_beta=self.heat_kernel_beta,
                execution_prune_eps=execution_prune_eps,
                max_chunk_size=route_chunks[route],
                diagnostics=True)

        def route_and_local(flat_state, router_params):
            routing = _compute_space_routing(
                flat_state,
                router_params["space_route_proj"]["kernel"],
                router_params["space_read_vectors"],
                top_k)
            local_states = _project_space_local_states(
                flat_state, router_params["space_state_proj"])
            return routing, local_states

        def layer_forward(
                current_state, block_params, pool_params,
                router_params, layer_rng):
            """One complete checkpointable Q/K/V-attention-RST layer."""
            regular_metrics: dict[str, jax.Array] = {}
            analysis_metrics: dict[str, jax.Array] = {}
            rng, rng_attn, rng_rst = jax.random.split(
                layer_rng, 3)
            normalized = _shared_layer_norm(
                current_state,
                block_params["norm1"]["scale"],
                block_params["norm1"]["bias"])
            batch_size, sequence_length = normalized.shape[:2]
            flat_attention_state = normalized.reshape((-1, self.d_model))
            attention_routing = None
            if fused_production:
                q_output, k_output, v_output, attention_metrics = (
                    attention_dense(
                        flat_attention_state,
                        router_params["space_route_proj"]["kernel"],
                        router_params["space_read_vectors"],
                        router_params["space_state_proj"],
                        router_params["space_state_writeback"],
                        router_params["q_operator_tau_proj"]["kernel"],
                        router_params["q_operator_tau_proj"]["bias"],
                        router_params["k_operator_tau_proj"]["kernel"],
                        router_params["k_operator_tau_proj"]["bias"],
                        router_params["v_operator_tau_proj"]["kernel"],
                        router_params["v_operator_tau_proj"]["bias"],
                        pool_params["q_read_vectors"],
                        pool_params["q_write_vectors"],
                        pool_params["k_read_vectors"],
                        pool_params["k_write_vectors"],
                        pool_params["v_read_vectors"],
                        pool_params["v_write_vectors"],
                        temperatures["q"], temperatures["v"],
                        soft_gate_boundary_power, execution_prune_eps,
                        route_scales["q"], route_scales["v"],
                        collect_regular_metrics))
                route_outputs = {
                    "q": q_output.reshape(normalized.shape),
                    "k": k_output.reshape(normalized.shape),
                    "v": v_output.reshape(normalized.shape),
                }
                regular_metrics.update(attention_metrics)
            else:
                attention_routing, attention_local = route_and_local(
                    flat_attention_state, router_params)
                attention_tau = {
                    route: _linear(
                        router_params[f"{route}_operator_tau_proj"],
                        attention_local)
                    for route in ("q", "k", "v")}
                route_outputs = {}
                for route in ("q", "k", "v"):
                    route_result = execute(
                        route, attention_local, attention_tau[route],
                        pool_params)
                    if isinstance(route_result, tuple):
                        local_output = route_result[0]
                        scalar, arrays = _aggregate_operator_diagnostics(
                            route_result, route_counts[route] // n_spaces,
                            route, composition_mode)
                        regular_metrics.update(scalar)
                        if analysis:
                            analysis_metrics.update(arrays)
                    else:
                        local_output = route_result
                    route_outputs[route] = _space_weighted_writeback(
                        local_output,
                        attention_routing["dense_space_weights"],
                        router_params["space_state_writeback"],
                        route_scales[route]).reshape(normalized.shape)

            d_head = self.d_model // self.n_heads
            query = route_outputs["q"].reshape(
                batch_size, sequence_length, self.n_heads, d_head
            ).transpose(0, 2, 1, 3)
            key = route_outputs["k"].reshape(
                batch_size, sequence_length, self.n_heads, d_head
            ).transpose(0, 2, 1, 3)
            value = route_outputs["v"].reshape(
                batch_size, sequence_length, self.n_heads, d_head
            ).transpose(0, 2, 1, 3)
            attention_output = _causal_attention_core(
                query, key, value, self.dropout_rate,
                deterministic, rng_attn,
                throughput_bf16=fused_throughput_bf16)
            attention_output = attention_output.transpose(
                0, 2, 1, 3).reshape(
                    batch_size, sequence_length, self.d_model)
            attention_output = (
                _throughput_linear_bf16_f32(
                    block_params["attn"]["expand_O"], attention_output)
                if fused_throughput_bf16 else _linear(
                    block_params["attn"]["expand_O"], attention_output))
            current_state = current_state + _shared_safe_dropout(
                attention_output, self.dropout_rate, deterministic, rng)

            rst_normalized = _shared_layer_norm(
                current_state,
                block_params["norm2"]["scale"],
                block_params["norm2"]["bias"])
            flat_rst_state = rst_normalized.reshape((-1, self.d_model))
            rst_routing = None
            if fused_production:
                rst_flat_update, rst_metrics = rst_dense(
                    flat_rst_state,
                    router_params["space_route_proj"]["kernel"],
                    router_params["space_read_vectors"],
                    router_params["space_state_proj"],
                    router_params["space_state_writeback"],
                    router_params["rst_operator_tau_proj"]["kernel"],
                    router_params["rst_operator_tau_proj"]["bias"],
                    pool_params["rst_read_vectors"],
                    pool_params["rst_write_vectors"],
                    temperatures["rst"], soft_gate_boundary_power,
                    execution_prune_eps, route_scales["rst"],
                    collect_regular_metrics)
                rst_route_update = rst_flat_update.reshape(
                    rst_normalized.shape)
                regular_metrics.update(rst_metrics)
            else:
                rst_routing, rst_local = route_and_local(
                    flat_rst_state, router_params)
                rst_tau = _linear(
                    router_params["rst_operator_tau_proj"], rst_local)
                rst_result = execute(
                    "rst", rst_local, rst_tau, pool_params)
                if isinstance(rst_result, tuple):
                    rst_space_output = rst_result[0]
                    scalar, arrays = _aggregate_operator_diagnostics(
                        rst_result, route_counts["rst"] // n_spaces,
                        "rst", composition_mode)
                    regular_metrics.update(scalar)
                    if analysis:
                        analysis_metrics.update(arrays)
                else:
                    rst_space_output = rst_result
                rst_route_update = _space_weighted_writeback(
                    rst_space_output,
                    rst_routing["dense_space_weights"],
                    router_params["space_state_writeback"],
                    route_scales["rst"]).reshape(rst_normalized.shape)
            current_state = current_state + _shared_safe_dropout(
                rst_route_update, self.dropout_rate, deterministic, rng_rst)

            if not fused_production:
                routing_metric_fn = (
                    _space_routing_metrics
                    if diagnostics_enabled
                    else _compact_space_routing_metrics)
                regular_metrics.update(routing_metric_fn(
                    attention_routing, "attention"))
                regular_metrics.update(routing_metric_fn(
                    rst_routing, "rst"))
            norm_metric_names = (
                "attn_out_norm",
                "rst_out_norm",
                "residual_norm",
                "q_route_output_norm",
                "k_route_output_norm",
                "v_route_output_norm",
                "rst_route_update_norm",
            )

            def compute_norm_metrics(_):
                values = {
                    "attn_out_norm": jnp.linalg.norm(
                        attention_output.astype(jnp.float32),
                        axis=-1).mean(),
                    "rst_out_norm": jnp.linalg.norm(
                        rst_route_update.astype(jnp.float32),
                        axis=-1).mean(),
                    "residual_norm": jnp.linalg.norm(
                        current_state.astype(jnp.float32), axis=-1).mean(),
                    "q_route_output_norm": jnp.linalg.norm(
                        route_outputs["q"].astype(jnp.float32),
                        axis=-1).mean(),
                    "k_route_output_norm": jnp.linalg.norm(
                        route_outputs["k"].astype(jnp.float32),
                        axis=-1).mean(),
                    "v_route_output_norm": jnp.linalg.norm(
                        route_outputs["v"].astype(jnp.float32),
                        axis=-1).mean(),
                    "rst_route_update_norm": jnp.linalg.norm(
                        rst_route_update.astype(jnp.float32),
                        axis=-1).mean(),
                }
                return {
                    key: jax.lax.stop_gradient(value)
                    for key, value in values.items()}

            if diagnostics_enabled:
                norm_metrics = compute_norm_metrics(None)
            else:
                norm_metrics = jax.lax.cond(
                    collect_regular_metrics,
                    compute_norm_metrics,
                    lambda _: {
                        name: jnp.float32(0.0)
                        for name in norm_metric_names},
                    operand=None)
            regular_metrics.update(norm_metrics)
            layer_aux = {"regular": regular_metrics}
            if analysis:
                layer_aux.update({
                    "analysis": analysis_metrics,
                    "last_local_input": flat_rst_state,
                    "last_route_state": rst_routing["route_state"],
                })
            return current_state, layer_aux

        layer_impl = layer_forward
        if self.gradient_checkpointing:
            checkpoint_kwargs = {"prevent_cse": False}
            if self.operation_space_execution_mode == "bundle_dense":
                checkpoint_kwargs["policy"] = (
                    jax.checkpoint_policies.save_only_these_names(
                        _BUNDLE_PACKING_CHECKPOINT_NAME))
            layer_impl = jax.checkpoint(layer_forward, **checkpoint_kwargs)

        block_params = [
            params[f"block_{index}"] for index in range(self.n_layers)]
        stacked_block_params = jax.tree.map(
            lambda *values: jnp.stack(values), *block_params)
        if not diagnostics_enabled:
            def scan_body(current_state, layer_inputs):
                return layer_impl(
                    current_state,
                    layer_inputs["params"],
                    pool,
                    router,
                    layer_inputs["rng"])

            state, scan_aux = jax.lax.scan(
                scan_body,
                state,
                {
                    "params": stacked_block_params,
                    "rng": layer_rngs,
                })
            scanned_regular_metrics = {
                key: values.mean()
                for key, values in scan_aux["regular"].items()}
        else:
            for layer_index in range(self.n_layers):
                layer_block_params = jax.tree.map(
                    lambda values: values[layer_index],
                    stacked_block_params)
                state, layer_aux = layer_impl(
                    state,
                    layer_block_params,
                    pool,
                    router,
                    layer_rngs[layer_index])
                append(regular_lists, layer_aux["regular"])
                if analysis:
                    append(analysis_lists, layer_aux["analysis"])
                    last_local_input = layer_aux["last_local_input"]
                    last_route_state = layer_aux["last_route_state"]

        state = self.norm(state)
        result: dict[str, jax.Array] = {}
        if labels is None:
            argmax_fn = (
                sharded_fns.get("vocab_argmax")
                if isinstance(sharded_fns, dict) else None)
            if argmax_fn is not None:
                result["argmax_token_ids"] = argmax_fn(
                    state, self.token_emb.embedding)
            else:
                logical, embedding = self._vocab_sizes()
                logits = self.token_emb.attend(state)
                result["logits"] = (
                    logits[..., :logical]
                    if embedding != logical else logits)
        else:
            shift_state = state[:, :-1]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid = shift_labels != -100
            vocab_loss = (
                sharded_fns.get("vocab_ce_loss")
                if isinstance(sharded_fns, dict) else None)
            if vocab_loss is not None:
                loss = vocab_loss(
                    shift_state, self.token_emb.embedding,
                    shift_labels, valid)
                correct = jnp.int32(0)
            else:
                logical, _ = self._vocab_sizes()
                logits = self.token_emb.attend(shift_state)[..., :logical]
                safe_labels = jnp.where(valid, shift_labels, 0)
                token_loss = -jax.nn.log_softmax(logits).reshape(
                    (-1, logical))[
                        jnp.arange(safe_labels.size),
                        safe_labels.reshape(-1)]
                loss = (
                    token_loss.reshape(safe_labels.shape) * valid
                ).sum() / jnp.maximum(valid.sum(), 1)
                predictions = jnp.argmax(logits, axis=-1)
                correct = (
                    (predictions == safe_labels) & valid
                ).sum().astype(jnp.int32)
            result.update({
                "loss": loss,
                "aux_loss": jnp.float32(0.0),
                "correct": correct if compute_accuracy else jnp.int32(0),
                "valid_count": valid.sum().astype(jnp.int32),
            })
        for key, values in regular_lists.items():
            result[key] = jnp.stack(values).mean()
        result.update(scanned_regular_metrics)
        operator_metric_sentinel = "q_operator_active_tau_frac"
        if operator_metric_sentinel in result:
            result.update(_canonical_regular_operator_metrics(result))
        result["train_metrics_collected"] = jax.lax.stop_gradient(
            collect_regular_metrics.astype(jnp.float32))
        if diagnostics_enabled:
            result["heat_kernel_beta"] = jnp.float32(self.heat_kernel_beta)
        if analysis:
            for key, values in analysis_lists.items():
                result[key] = jnp.stack(values).mean(axis=0)
            result.update(_space_geometry_diagnostics(
                router["space_state_proj"],
                router["space_state_writeback"],
                last_local_input))
            result.update(_read_bank_geometry_diagnostics(
                pool, router["space_read_vectors"], last_route_state))
        return result

    def get_model_info(self) -> list[str]:
        n_spaces = int(self.n_operation_spaces)
        return [
            f"DAWN-SRW v4.1.7.4 ({MODEL_VERSION})",
            "routing: one shared D->R projection directly matches explicit "
            "space read vectors",
            f"canonical geometry: D={self.d_model}, M={n_spaces}, "
            f"R={self.d_route}; independent D->R coordinates, "
            f"top_k={self.operation_space_top_k}",
            "space gate: hard top-k ReLU^2, non-softmax, sqrt-mass "
            "composition denominator",
            "local execution: z_m directly matches each pool's read vectors; "
            "the read vector itself is the operator key",
            "banks: Q/K/V/RST are fully separate; Q/K/V share the attention "
            "space gate and RST recomputes routing after attention",
            f"operators per space: q={int(self.n_q)//n_spaces}, "
            f"k={int(self.n_k)//n_spaces}, "
            f"v={int(self.n_v)//n_spaces}, "
            f"rst={int(self.n_rst)//n_spaces}",
            (
                "execution: semantic hard top-k, physical compact fixed "
                f"{self.operation_space_bundle_size}-space bundles with "
                f"token blocks of "
                f"{self.operation_space_bundle_token_block_size}"
                if self.operation_space_execution_mode == "bundle_dense"
                else (
                    "execution: semantic hard top-k, physical all-space "
                    "dense, one shared P_m/U_m coordinate system and fused "
                    "writeback")),
            "layer execution: lax.scan with "
            + ("full-layer rematerialization"
               if self.gradient_checkpointing
               else "uncheckpointed scan"),
            f"attention core: {ATTENTION_CORE_NAME}",
        ]


def _sampled_layer_states(
        params, input_ids, max_tokens, *,
        n_heads=6, n_layers=12, operation_space_top_k=2,
        soft_gate_temperature=0.07, soft_gate_boundary_power=2.0,
        soft_gate_T_qk=None, soft_gate_T_v=None,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_power_qk=None, admission_den_power_v=None,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA,
        production_rst=True):
    """Sample block-0 Norm1 and production post-attention Norm2 states."""
    max_tokens = _positive_int("max_tokens", int(max_tokens))
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    sequence_length = min(int(input_ids.shape[1]), max_tokens)
    batch_count = min(
        int(input_ids.shape[0]), max(1, max_tokens // sequence_length))
    batch_index = (
        jnp.arange(batch_count) * int(input_ids.shape[0]) // batch_count)
    token_ids = input_ids[batch_index, :sequence_length]
    positions = jnp.arange(sequence_length)[None, :]
    state = (
        params["token_emb"]["embedding"][token_ids]
        + params["pos_emb"]["embedding"][positions]
    ).astype(jnp.float32)
    block = params["block_0"]
    attention_state = _shared_layer_norm(
        state, block["norm1"]["scale"], block["norm1"]["bias"])
    flat_attention_state = attention_state.reshape(
        (-1, attention_state.shape[-1]))
    if not production_rst:
        return flat_attention_state, flat_attention_state
    router = params["router"]
    pool = params["neuron_pool"]
    routing = _compute_space_routing(
        flat_attention_state,
        router["space_route_proj"]["kernel"],
        router["space_read_vectors"],
        operation_space_top_k)
    local = _project_space_local_states(
        flat_attention_state, router["space_state_proj"])
    den_qk = (
        admission_den_power
        if admission_den_power_qk is None else admission_den_power_qk)
    den_v = (
        admission_den_power
        if admission_den_power_v is None else admission_den_power_v)
    qk_scale, v_scale, _ = _shared_pool_output_scales(
        int(attention_state.shape[-1]), int(n_layers))
    route_scales = {"q": qk_scale, "k": qk_scale, "v": v_scale}
    route_outputs = {}
    for route in ("q", "k", "v"):
        route_temperature = (
            soft_gate_T_qk
            if route in ("q", "k") and soft_gate_T_qk is not None
            else (
                soft_gate_T_v
                if route == "v" and soft_gate_T_v is not None
                else soft_gate_temperature))
        raw_tau = _linear(
            router[f"{route}_operator_tau_proj"], local)
        local_output = _rw_compose_space_dense(
            local,
            pool[f"{route}_read_vectors"],
            pool[f"{route}_write_vectors"],
            raw_tau,
            soft_gate_temperature=route_temperature,
            soft_gate_boundary_power=soft_gate_boundary_power,
            admission_den_power=den_qk if route in ("q", "k") else den_v,
            srw_composition_mode=srw_composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            max_chunk_size=int(pool[f"{route}_read_vectors"].shape[1]))
        route_outputs[route] = _space_weighted_writeback(
            local_output,
            routing["dense_space_weights"],
            router["space_state_writeback"],
            route_scales[route]).reshape(attention_state.shape)

    d_model = int(attention_state.shape[-1])
    n_heads = _positive_int("n_heads", int(n_heads))
    if d_model % n_heads:
        raise ValueError("calibration d_model must be divisible by n_heads")
    d_head = d_model // n_heads

    def split_heads(value):
        return value.reshape(
            batch_count, sequence_length, n_heads, d_head
        ).transpose(0, 2, 1, 3)

    query = split_heads(route_outputs["q"])
    key = split_heads(route_outputs["k"])
    value = split_heads(route_outputs["v"])
    attention_scores = jnp.einsum(
        "bhsd,bhtd->bhst", query, key
    ) / jnp.sqrt(jnp.float32(d_head))
    causal = jnp.tril(jnp.ones(
        (sequence_length, sequence_length), dtype=jnp.bool_))
    attention_scores = jnp.where(
        causal, attention_scores, jnp.finfo(attention_scores.dtype).min)
    attention_weights = jax.nn.softmax(attention_scores, axis=-1)
    attention_output = jnp.einsum(
        "bhst,bhtd->bhsd", attention_weights, value)
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
        batch_count, sequence_length, d_model)
    attention_output = _linear(
        block["attn"]["expand_O"], attention_output)
    post_attention_state = state + attention_output
    rst_state = _shared_layer_norm(
        post_attention_state,
        block["norm2"]["scale"],
        block["norm2"]["bias"])
    return flat_attention_state, rst_state.reshape((-1, d_model))


def _tau_init_calibration_scores(
        params, input_ids, max_tokens=128, **production_kwargs):
    """Return direct-read per-route ``[M,T,N]`` cosine tables."""
    attention_state, rst_state = _sampled_layer_states(
        params, input_ids, max_tokens, **production_kwargs)
    router = params["router"]
    pool = params["neuron_pool"]

    def score(local, route):
        _, rho, _ = _direct_read_match(
            local, pool[f"{route}_read_vectors"])
        return rho

    attention_local = _project_space_local_states(
        attention_state, router["space_state_proj"])
    production_rst = bool(production_kwargs.get("production_rst", True))
    rst_local = (
        _project_space_local_states(
            rst_state, router["space_state_proj"])
        if production_rst else None)
    return {
        "q": score(attention_local, "q"),
        "k": score(attention_local, "k"),
        "v": score(attention_local, "v"),
        "rst": (
            score(rst_local, "rst")
            if production_rst else jnp.zeros(
                (
                    attention_local.shape[0],
                    attention_local.shape[1],
                    pool["rst_read_vectors"].shape[1],
                ),
                dtype=jnp.float32)),
    }


def calibrate_operator_tau_per_space(
        scores_by_route: Mapping[str, jax.Array], *,
        target_qk_frac: float,
        target_v_frac: float,
        target_rst_frac: float) -> dict[str, jax.Array]:
    """Fit independent Q/K/V/RST cosine quantiles for every space."""
    targets = {
        "q": float(target_qk_frac),
        "k": float(target_qk_frac),
        "v": float(target_v_frac),
        "rst": float(target_rst_frac),
    }
    result = {}
    for route in ROUTES:
        scores = jnp.asarray(scores_by_route[route], dtype=jnp.float32)
        target = targets[route]
        if scores.ndim != 3 or not 0.0 < target < 1.0:
            raise ValueError(
                "tau calibration requires [M,T,N] and target in (0,1)")
        result[route] = jnp.quantile(
            scores.reshape((scores.shape[0], -1)),
            1.0 - target, axis=1)
    return result


def _direct_read_geometry_diagnostics(
        params, input_ids, max_tokens=128, **production_kwargs):
    scores = _tau_init_calibration_scores(
        params, input_ids, max_tokens, **production_kwargs)
    return {
        f"{route}_direct_read_score_{name}": value
        for route, route_scores in scores.items()
        for name, value in (
            ("mean", route_scores.mean()),
            ("std", route_scores.std()),
            ("max", route_scores.max()))
    }


def initialization_diagnostics_from_params(
        params: Mapping[str, Any], input_ids: jax.Array,
        operation_space_top_k: int) -> dict[str, float]:
    """Host-side one-shot routing/local-geometry diagnostics."""
    attention_state, _ = _sampled_layer_states(
        params, input_ids, 4096, production_rst=False)
    router = params["router"]
    routing = _compute_space_routing(
        attention_state,
        router["space_route_proj"]["kernel"],
        router["space_read_vectors"],
        operation_space_top_k)
    diagnostics = _space_geometry_diagnostics(
        router["space_state_proj"],
        router["space_state_writeback"],
        attention_state)
    diagnostics.update(_read_bank_geometry_diagnostics(
        params["neuron_pool"],
        router["space_read_vectors"],
        routing["route_state"]))
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
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg.get("max_seq_len", 512),
        dropout_rate=0.0,
        d_route=cfg["d_route"],
        n_q=cfg["n_q"],
        n_k=cfg["n_k"],
        n_v=cfg["n_v"],
        n_rst=cfg["n_rst"],
        n_operation_spaces=cfg["n_operation_spaces"],
        operation_space_top_k=cfg["operation_space_top_k"],
        admission_den_power=cfg.get("admission_den_power", 1.0),
        admission_den_power_qk=cfg.get("admission_den_power_qk"),
        admission_den_power_v=cfg.get("admission_den_power_v"),
        admission_den_power_rst=cfg.get("admission_den_power_rst"),
        srw_composition_mode=cfg.get(
            "srw_composition_mode", DEFAULT_SRW_COMPOSITION_MODE),
        heat_kernel_beta=cfg.get(
            "heat_kernel_beta", DEFAULT_HEAT_KERNEL_BETA),
        tau_init_attn_q=cfg.get("tau_init_attn_q", 0.0),
        tau_init_attn_k=cfg.get("tau_init_attn_k", 0.0),
        tau_init_attn_v=cfg.get("tau_init_attn_v", 0.0),
        tau_init_rst=cfg.get("tau_init_rst", 0.0))
    return model.apply(
        {"params": params}, input_ids,
        deterministic=True, analysis=True,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.PRNGKey(0)})


def _make_sharded_space_dense_direct(
        mesh, max_chunk_size=2048, dead_exposure_target=0.0,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA,
        *, diagnostics: bool):
    """Build the operator-axis sharded one-matmul direct-read executor."""
    del dead_exposure_target, admission_den_grad_scale
    if int(max_chunk_size) <= 0:
        raise ValueError("max_chunk_size must be positive")
    den_power = jnp.float32(admission_den_power)
    effective_active_eps = jnp.float32(soft_gate_effective_active_eps)
    composition_mode = str(srw_composition_mode)
    beta = jnp.float32(heat_kernel_beta)

    def direct_core(
            space_local_states, raw_tau, token_valid,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        n_spaces, token_capacity, d_route = space_local_states.shape
        n_local = int(read_vectors_local.shape[1])
        chunk_size = min(int(max_chunk_size), n_local)
        n_chunks = math.ceil(n_local / chunk_size)
        n_padded = n_chunks * chunk_size
        pad_n = n_padded - n_local
        pad_spec = ((0, 0), (0, pad_n), (0, 0))
        reads = jnp.pad(read_vectors_local, pad_spec)
        writes = jnp.pad(write_vectors_local, pad_spec)
        valid_rows = jnp.arange(n_padded) < n_local
        read_unit = forward_unit_direction(reads).astype(jnp.bfloat16)
        write_unit = forward_unit_direction(writes).astype(jnp.bfloat16)
        local_bf16 = space_local_states.astype(jnp.bfloat16)
        local_norm = jnp.maximum(
            jnp.linalg.norm(
                space_local_states.astype(jnp.float32),
                axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))
        tau = _shared_tau_from_param(raw_tau)
        token_valid = jnp.asarray(token_valid, dtype=jnp.bool_)
        aggregate_shape = (n_spaces, token_capacity, 1)
        carry = (
            jnp.zeros(
                (n_spaces, token_capacity, d_route), dtype=jnp.float32),
            jnp.zeros(aggregate_shape, dtype=jnp.float32),
            jnp.zeros(aggregate_shape, dtype=jnp.float32),
            jnp.zeros(aggregate_shape, dtype=jnp.float32),
            jnp.zeros(aggregate_shape, dtype=jnp.float32),
            jnp.zeros(aggregate_shape, dtype=jnp.float32),
        )

        @jax.checkpoint
        def direct_step(carry_value, chunk_index):
            (raw_out, gate_mass, gate_sq, gate_max,
             active_count, depth_sum) = carry_value
            start = chunk_index * chunk_size
            read = jax.lax.dynamic_slice_in_dim(
                read_unit, start, chunk_size, axis=1)
            write = jax.lax.dynamic_slice_in_dim(
                write_unit, start, chunk_size, axis=1)
            row_valid = jax.lax.dynamic_slice_in_dim(
                valid_rows, start, chunk_size, axis=0)
            valid = token_valid[:, :, None] & row_valid[None, None, :]
            # This is the sole local-state/read-vector matmul.  Its result is
            # reused as both the actual read scalar and the cosine numerator.
            read_value = jnp.einsum(
                "mtr,mnr->mtn", local_bf16, read).astype(jnp.float32)
            rho = jnp.clip(read_value / local_norm, -1.0, 1.0)
            rho_compute = jnp.where(valid, rho, tau)
            margin, gate, depth, execution_weight, _ = (
                _shared_compute_admission_drive(
                    rho_compute, tau, soft_gate_temperature,
                    boundary_power=soft_gate_boundary_power,
                    effective_active_eps=effective_active_eps,
                    execution_prune_eps=execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=beta))
            gate = jnp.where(valid, gate, 0.0)
            depth = jnp.where(valid, depth, 0.0)
            execution_weight = jnp.where(valid, execution_weight, 0.0)
            chunk_out = jnp.einsum(
                "mtn,mnr->mtr",
                (execution_weight * read_value).astype(jnp.bfloat16),
                write).astype(jnp.float32)
            return (
                raw_out + chunk_out,
                gate_mass + gate.sum(axis=-1, keepdims=True),
                gate_sq + jnp.square(gate).sum(axis=-1, keepdims=True),
                jnp.maximum(gate_max, gate.max(axis=-1, keepdims=True)),
                active_count + ((margin > 0.0) & valid).sum(
                    axis=-1, keepdims=True).astype(jnp.float32),
                depth_sum + depth.sum(axis=-1, keepdims=True),
            ), None

        (raw_out, gate_mass, gate_sq, gate_max,
         active_count, depth_sum), _ = jax.lax.scan(
            direct_step, carry, jnp.arange(n_chunks))
        gate_mass = jax.lax.psum(gate_mass, "model")
        gate_den = _shared_composition_den(
            gate_mass, den_power, composition_mode)
        space_results = jax.lax.psum(
            (raw_out / gate_den).astype(jnp.bfloat16),
            "model").astype(jnp.float32)
        if not diagnostics:
            return space_results
        return (
            space_results,
            jax.lax.psum(active_count, "model"),
            gate_mass,
            jax.lax.psum(gate_sq, "model"),
            jax.lax.pmax(gate_max, "model"),
            jax.lax.psum(depth_sum, "model"),
            tau,
            gate_den,
        )

    common_specs = (
        P(None, "data", None),
        P(None, "data", None),
        P(None, "data"),
        P(None, "model", None),
        P(None, "model", None),
        P(), P(), P(), P(), P(),
    )
    result_specs = P(None, "data", None)
    if diagnostics:
        result_specs = (
            result_specs,
            P(None, "data", None),
            P(None, "data", None),
            P(None, "data", None),
            P(None, "data", None),
            P(None, "data", None),
            P(None, "data", None),
            P(None, "data", None),
        )
    kernel = shard_map(
        direct_core, mesh=mesh,
        in_specs=common_specs, out_specs=result_specs,
        check_rep=False)
    kernel._v4174_kernel_profile = (
        "production_diagnostics" if diagnostics else "production")
    kernel._v4174_dense_grouped_execution = "all_spaces"
    kernel._v4174_direct_read_matmuls = 1
    return kernel


def make_sharded_space_dense_minimal(
        mesh, max_chunk_size=2048, dead_exposure_target=0.0,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create the production direct-read all-space dense shard kernel."""
    return _make_sharded_space_dense_direct(
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta, diagnostics=False)


def make_sharded_space_dense_diagnostics(
        mesh, max_chunk_size=2048, dead_exposure_target=0.0,
        soft_gate_effective_active_eps=1.0e-6,
        admission_den_power=DEFAULT_ADMISSION_DEN_POWER,
        admission_den_grad_scale=1.0,
        srw_composition_mode=DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta=DEFAULT_HEAT_KERNEL_BETA):
    """Create the diagnostic direct-read all-space dense shard kernel."""
    return _make_sharded_space_dense_direct(
        mesh, max_chunk_size, dead_exposure_target,
        soft_gate_effective_active_eps, admission_den_power,
        admission_den_grad_scale, srw_composition_mode,
        heat_kernel_beta, diagnostics=True)


_FUSED_OPERATOR_METRIC_SUFFIXES = (
    "active_tau_frac",
    "active_tau_count",
    "tau_mean",
    "gate_mass_mean",
    "gate_den_mean",
    "depth_active_mean",
    "gate_eff_n_mean",
    "top1_gate_frac_mean",
    "den_floor_frac",
)

_BUNDLE_PACKING_METRIC_SUFFIXES = (
    "valid_entries",
    "entries_per_token",
    "same_top2_frac",
    "physical_spaces_per_token",
    "dense_compute_fraction",
    "padding_entries",
    "padding_fraction",
    "token_count_min",
    "token_count_mean",
    "token_count_max",
    "token_drop_count",
)


def _bundle_packing_metric_names(stage: str) -> tuple[str, ...]:
    return tuple(
        f"{stage}_bundle_{suffix}"
        for suffix in _BUNDLE_PACKING_METRIC_SUFFIXES)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _metric_only_cond(
        metric_fn: Callable,
        metric_count: int,
        collect_metrics: jax.Array,
        operands) -> jax.Array:
    """Run an observational metric branch with an explicitly zero tangent."""
    return jax.lax.cond(
        jnp.asarray(collect_metrics, dtype=jnp.bool_).reshape(()),
        metric_fn,
        lambda _: jnp.zeros((metric_count,), dtype=jnp.float32),
        operands)


def _metric_only_cond_fwd(
        metric_fn: Callable,
        metric_count: int,
        collect_metrics: jax.Array,
        operands):
    return (
        _metric_only_cond(
            metric_fn, metric_count, collect_metrics, operands),
        None)


def _metric_only_cond_bwd(
        metric_fn: Callable,
        metric_count: int,
        residual,
        cotangent):
    del metric_fn, metric_count, residual, cotangent
    return None, None


_metric_only_cond.defvjp(
    _metric_only_cond_fwd,
    _metric_only_cond_bwd)


def _prefix_pack_top2_bundle_arrays_impl(
        selected_ids: jax.Array,
        dense_weights: jax.Array,
        bundle_size: int,
        token_block_size: int):
    """Build exact fixed-bundle packing with six prefix counters."""
    token_count, n_spaces = map(int, dense_weights.shape)
    n_bundles = n_spaces // bundle_size
    selected_weights = jnp.take_along_axis(
        dense_weights, selected_ids, axis=1)
    bundle_ids = selected_ids // bundle_size
    local_ids = selected_ids % bundle_size
    same_bundle = bundle_ids[:, 0] == bundle_ids[:, 1]

    local0 = jax.nn.one_hot(
        local_ids[:, 0], bundle_size, dtype=jnp.float32)
    local1 = jax.nn.one_hot(
        local_ids[:, 1], bundle_size, dtype=jnp.float32)
    membership0 = (
        local0 + jnp.where(same_bundle[:, None], local1, 0.0)) > 0.0
    membership1 = jnp.where(
        same_bundle[:, None], 0.0, local1) > 0.0
    weight0 = (
        local0 * selected_weights[:, 0, None]
        + jnp.where(
            same_bundle[:, None],
            local1 * selected_weights[:, 1, None],
            0.0))
    weight1 = jnp.where(
        same_bundle[:, None],
        0.0,
        local1 * selected_weights[:, 1, None])

    token_ids = jnp.arange(token_count, dtype=jnp.int32)
    sentinel = jnp.int32(n_bundles)
    candidate_bundle = jnp.concatenate((
        bundle_ids[:, 0],
        jnp.where(same_bundle, sentinel, bundle_ids[:, 1]),
    ))
    candidate_token = jnp.concatenate((token_ids, token_ids))
    candidate_membership = jnp.concatenate(
        (membership0, membership1), axis=0)
    candidate_weight = jnp.concatenate((weight0, weight1), axis=0)
    candidate_valid = candidate_bundle < sentinel

    bundle_onehot = (
        jax.nn.one_hot(
            candidate_bundle, n_bundles, dtype=jnp.int32)
        * candidate_valid[:, None].astype(jnp.int32))
    rank_per_bundle = (
        jnp.cumsum(bundle_onehot, axis=0, dtype=jnp.int32)
        - bundle_onehot)
    local_rank = jnp.sum(
        rank_per_bundle * bundle_onehot, axis=1, dtype=jnp.int32)
    counts = bundle_onehot.sum(axis=0, dtype=jnp.int32)
    padded_counts = (
        (counts + token_block_size - 1) // token_block_size
        * token_block_size)
    packed_offsets = jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.cumsum(padded_counts[:-1], dtype=jnp.int32),
    ))
    safe_bundle = jnp.minimum(candidate_bundle, n_bundles - 1)

    entry_capacity = (
        2 * token_count
        + n_bundles * (token_block_size - 1))
    scan_blocks = math.ceil(entry_capacity / token_block_size)
    scan_capacity = scan_blocks * token_block_size
    destination = packed_offsets[safe_bundle] + local_rank
    destination = jnp.where(
        candidate_valid, destination, jnp.int32(scan_capacity))

    packed_bundle = jnp.full(
        (scan_capacity,), sentinel, dtype=jnp.int32
    ).at[destination].set(candidate_bundle, mode="drop")
    packed_token = jnp.zeros(
        (scan_capacity,), dtype=jnp.int32
    ).at[destination].set(candidate_token, mode="drop")
    packed_membership = jnp.zeros(
        (scan_capacity, bundle_size), dtype=jnp.bool_
    ).at[destination].set(candidate_membership, mode="drop")
    packed_weight = jnp.zeros(
        (scan_capacity, bundle_size), dtype=jnp.float32
    ).at[destination].set(candidate_weight, mode="drop")
    packed_valid = jnp.zeros(
        (scan_capacity,), dtype=jnp.bool_
    ).at[destination].set(candidate_valid, mode="drop")
    return (
        packed_bundle,
        packed_token,
        packed_membership,
        packed_weight,
        packed_valid,
        counts,
        padded_counts,
        same_bundle,
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _prefix_pack_top2_bundle_arrays(
        selected_ids: jax.Array,
        dense_weights: jax.Array,
        bundle_size: int,
        token_block_size: int,
        operation_space_count: int,
        token_count: int):
    """Pack once and preserve compact routing metadata for its VJP."""
    del operation_space_count, token_count
    return _prefix_pack_top2_bundle_arrays_impl(
        selected_ids, dense_weights, bundle_size, token_block_size)


def _prefix_pack_top2_bundle_arrays_fwd(
        selected_ids: jax.Array,
        dense_weights: jax.Array,
        bundle_size: int,
        token_block_size: int,
        operation_space_count: int,
        token_count: int):
    del operation_space_count, token_count
    packed = _prefix_pack_top2_bundle_arrays_impl(
        selected_ids, dense_weights, bundle_size, token_block_size)
    residual = tuple(
        ad_checkpoint.checkpoint_name(
            packed[index], name=_BUNDLE_PACKING_CHECKPOINT_NAME)
        for index in (0, 1, 2, 4)
    )
    return packed, residual


def _prefix_pack_top2_bundle_arrays_bwd(
        bundle_size: int,
        token_block_size: int,
        operation_space_count: int,
        token_count: int,
        residual,
        cotangent):
    del token_block_size
    (packed_bundle, packed_token,
     packed_membership, packed_valid) = residual
    packed_weight_cotangent = jnp.asarray(
        cotangent[3], dtype=jnp.float32)
    packed_space_ids = (
        packed_bundle[:, None] * int(bundle_size)
        + jnp.arange(int(bundle_size), dtype=jnp.int32)[None, :])
    valid_membership = packed_membership & packed_valid[:, None]
    dense_grad = jnp.zeros(
        (int(token_count),
         int(operation_space_count)),
        dtype=jnp.float32)
    dense_grad = dense_grad.at[
        packed_token[:, None], packed_space_ids
    ].add(
        jnp.where(
            valid_membership, packed_weight_cotangent, 0.0),
        mode="drop")
    return None, dense_grad


_prefix_pack_top2_bundle_arrays.defvjp(
    _prefix_pack_top2_bundle_arrays_fwd,
    _prefix_pack_top2_bundle_arrays_bwd)


def _pack_top2_bundle_entries_sharded(
        routing: Mapping[str, jax.Array], *,
        bundle_size: int,
        token_block_size: int,
        stage: str) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Prefix-pack one or two fixed-bundle entries for every routed token."""
    selected_ids = routing["selected_ids"].astype(jnp.int32)
    if selected_ids.ndim != 2 or int(selected_ids.shape[1]) != 2:
        raise ValueError("bundle_dense routing requires selected_ids[T,2]")
    dense_weights = routing["dense_space_weights"].astype(jnp.float32)
    token_count, n_spaces = map(int, dense_weights.shape)
    bundle_size = int(bundle_size)
    token_block_size = int(token_block_size)
    if bundle_size != BUNDLE_DENSE_SIZE:
        raise ValueError(
            f"bundle_dense supports bundle_size={BUNDLE_DENSE_SIZE} only")
    if n_spaces % bundle_size:
        raise ValueError("operation-space count must be divisible by bundle size")
    n_bundles = n_spaces // bundle_size
    (packed_bundle,
     packed_token,
     packed_membership,
     packed_weight,
     packed_valid,
     counts,
     padded_counts,
     same_bundle) = _prefix_pack_top2_bundle_arrays(
         selected_ids, dense_weights, bundle_size, token_block_size,
         n_spaces, token_count)

    requested_entries = counts.astype(jnp.float32).sum()
    valid_entries = packed_valid.astype(jnp.float32).sum()
    dropped_entries = jnp.maximum(
        requested_entries - valid_entries, jnp.float32(0.0))
    padding_entries = (
        padded_counts.sum() - counts.sum()).astype(jnp.float32)
    tokens_f32 = jnp.float32(token_count)
    valid_mean = jax.lax.pmean(valid_entries, "data")
    dropped_mean = jax.lax.pmean(dropped_entries, "data")
    padding_mean = jax.lax.pmean(padding_entries, "data")
    same_mean = jax.lax.pmean(
        same_bundle.astype(jnp.float32).sum(), "data")
    count_f32 = counts.astype(jnp.float32)
    count_min = jax.lax.pmin(count_f32.min(), "data")
    count_mean = jax.lax.pmean(count_f32.mean(), "data")
    count_max = jax.lax.pmax(count_f32.max(), "data")
    entries_per_token = valid_mean / tokens_f32
    physical_spaces = jnp.float32(bundle_size) * entries_per_token
    metric_values = (
        valid_mean,
        entries_per_token,
        same_mean / tokens_f32,
        physical_spaces,
        physical_spaces / jnp.float32(n_spaces),
        padding_mean,
        padding_mean / jnp.maximum(
            valid_mean + padding_mean, jnp.float32(1.0)),
        count_min,
        count_mean,
        count_max,
        dropped_mean,
    )
    metrics = {
        name: jax.lax.stop_gradient(value)
        for name, value in zip(
            _bundle_packing_metric_names(stage), metric_values)}
    return ({
        "bundle_id": packed_bundle,
        "token_id": packed_token,
        "membership_mask": packed_membership,
        "routing_weight": packed_weight,
        "token_valid": packed_valid,
    }, metrics)


def _dense_rw_output_sharded(
        local_f32: jax.Array,
        local_norm: jax.Array,
        token_valid: jax.Array,
        read_vectors: jax.Array,
        write_vectors: jax.Array,
        raw_tau: jax.Array, *,
        max_chunk_size: int,
        soft_gate_temperature: jax.Array,
        soft_gate_boundary_power: jax.Array,
        execution_prune_eps: jax.Array,
        srw_composition_mode: str,
        heat_kernel_beta: float,
        effective_active_eps: float,
        throughput_bf16: bool) -> tuple[jax.Array, jax.Array]:
    """Compute production RW output with multi-chunk backward remat."""
    n_routes, n_spaces, token_capacity, d_route = map(
        int, (read_vectors.shape[0], local_f32.shape[0],
              local_f32.shape[1], local_f32.shape[2]))
    n_local = int(read_vectors.shape[2])
    chunk_size = min(max(1, int(max_chunk_size)), n_local)
    n_chunks = math.ceil(n_local / chunk_size)
    n_padded = n_chunks * chunk_size
    pad_n = n_padded - n_local
    pad_spec = ((0, 0), (0, 0), (0, pad_n), (0, 0))
    reads = forward_unit_direction(jnp.pad(read_vectors, pad_spec))
    writes = forward_unit_direction(jnp.pad(write_vectors, pad_spec))
    valid_rows = jnp.arange(n_padded) < n_local
    tau = _shared_tau_from_param(raw_tau)
    aggregate_shape = (n_routes, n_spaces, token_capacity, 1)
    carry = (
        jnp.zeros(
            (n_routes, n_spaces, token_capacity, d_route),
            dtype=jnp.float32),
        jnp.zeros(aggregate_shape, dtype=jnp.float32),
    )
    beta = jnp.float32(heat_kernel_beta)
    active_eps = jnp.float32(effective_active_eps)
    rw_dot = (
        _throughput_einsum_bf16_f32
        if throughput_bf16 else _control_einsum_f32)

    def production_step(carry_value, chunk_index):
        raw_out, gate_mass = carry_value
        start = chunk_index * chunk_size
        read = jax.lax.dynamic_slice_in_dim(
            reads, start, chunk_size, axis=2)
        write = jax.lax.dynamic_slice_in_dim(
            writes, start, chunk_size, axis=2)
        row_valid = jax.lax.dynamic_slice_in_dim(
            valid_rows, start, chunk_size, axis=0)
        valid = (
            token_valid[None, :, :, None]
            & row_valid[None, None, None, :])
        read_value = rw_dot(
            "mtr,amnr->amtn", local_f32, read).astype(jnp.float32)
        rho = jnp.clip(read_value / local_norm[None, ...], -1.0, 1.0)
        rho_compute = jnp.where(valid, rho, tau)
        _, gate, _, execution_weight, _ = _shared_compute_admission_drive(
            rho_compute, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=active_eps,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=srw_composition_mode,
            heat_kernel_beta=beta)
        gate = jnp.where(valid, gate, 0.0)
        execution_weight = jnp.where(valid, execution_weight, 0.0)
        chunk_out = rw_dot(
            "amtn,amnr->amtr",
            execution_weight * read_value,
            write).astype(jnp.float32)
        return (
            raw_out + chunk_out,
            gate_mass + gate.sum(axis=-1, keepdims=True),
        ), None

    scan_step = jax.checkpoint(production_step, prevent_cse=False)
    (raw_out, gate_mass), _ = jax.lax.scan(
        scan_step, carry, jnp.arange(n_chunks))
    return raw_out, gate_mass


def _global_dense_rw_den_sharded(
        gate_mass: jax.Array,
        admission_den_power: jax.Array,
        srw_composition_mode: str) -> tuple[jax.Array, jax.Array]:
    """Form the live global denominator without reducing the RW numerator."""
    global_gate_mass = jax.lax.psum(
        gate_mass.astype(jnp.float32), "model")
    gate_den = _shared_composition_den(
        global_gate_mass, admission_den_power, srw_composition_mode)
    return global_gate_mass, gate_den


def _psum_dense_rw_representation_sharded(
        local_output: jax.Array,
        collective_bf16: bool) -> jax.Array:
    """Reduce a completed representation update at the configured boundary."""
    collective_dtype = jnp.bfloat16 if collective_bf16 else jnp.float32
    return jax.lax.psum(
        local_output.astype(collective_dtype), "model").astype(jnp.float32)


def _dense_rw_metric_stats_sharded(
        local_f32: jax.Array,
        local_norm: jax.Array,
        token_valid: jax.Array,
        read_vectors: jax.Array,
        raw_tau: jax.Array, *,
        max_chunk_size: int,
        soft_gate_temperature: jax.Array,
        soft_gate_boundary_power: jax.Array,
        execution_prune_eps: jax.Array,
        srw_composition_mode: str,
        heat_kernel_beta: float,
        effective_active_eps: float,
        throughput_bf16: bool) -> tuple[jax.Array, ...]:
    """Recompute stop-gradient read/gate observations without RW writeback."""
    metric_local = jax.lax.stop_gradient(local_f32)
    metric_norm = jax.lax.stop_gradient(local_norm)
    metric_reads = jax.lax.stop_gradient(read_vectors)
    metric_tau = jax.lax.stop_gradient(raw_tau)
    n_routes, n_spaces, token_capacity = map(
        int, (metric_reads.shape[0], metric_local.shape[0],
              metric_local.shape[1]))
    n_local = int(metric_reads.shape[2])
    chunk_size = min(max(1, int(max_chunk_size)), n_local)
    n_chunks = math.ceil(n_local / chunk_size)
    n_padded = n_chunks * chunk_size
    pad_n = n_padded - n_local
    reads = forward_unit_direction(jnp.pad(
        metric_reads, ((0, 0), (0, 0), (0, pad_n), (0, 0))))
    valid_rows = jnp.arange(n_padded) < n_local
    tau = _shared_tau_from_param(metric_tau)
    aggregate_zero = jnp.zeros(
        (n_routes, n_spaces, token_capacity, 1), dtype=jnp.float32)
    carry = (
        aggregate_zero,
        aggregate_zero,
        aggregate_zero,
        aggregate_zero,
        aggregate_zero,
    )
    beta = jnp.float32(heat_kernel_beta)
    active_eps = jnp.float32(effective_active_eps)
    rw_read_dot = (
        _throughput_einsum_bf16_f32
        if throughput_bf16 else _control_einsum_f32)

    def metric_step(carry_value, chunk_index):
        gate_mass, gate_sq, gate_max, active_count, depth_sum = carry_value
        start = chunk_index * chunk_size
        read = jax.lax.dynamic_slice_in_dim(
            reads, start, chunk_size, axis=2)
        row_valid = jax.lax.dynamic_slice_in_dim(
            valid_rows, start, chunk_size, axis=0)
        valid = (
            token_valid[None, :, :, None]
            & row_valid[None, None, None, :])
        read_value = rw_read_dot(
            "mtr,amnr->amtn", metric_local, read).astype(jnp.float32)
        rho = jnp.clip(read_value / metric_norm[None, ...], -1.0, 1.0)
        rho_compute = jnp.where(valid, rho, tau)
        margin, gate, depth, _, _ = _shared_compute_admission_drive(
            rho_compute, tau, soft_gate_temperature,
            boundary_power=soft_gate_boundary_power,
            effective_active_eps=active_eps,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=srw_composition_mode,
            heat_kernel_beta=beta)
        gate = jnp.where(valid, gate, 0.0)
        depth = jnp.where(valid, depth, 0.0)
        return (
            gate_mass + gate.sum(axis=-1, keepdims=True),
            gate_sq + jnp.square(gate).sum(axis=-1, keepdims=True),
            jnp.maximum(gate_max, gate.max(axis=-1, keepdims=True)),
            active_count + ((margin > 0.0) & valid).sum(
                axis=-1, keepdims=True).astype(jnp.float32),
            depth_sum + depth.sum(axis=-1, keepdims=True),
        ), None

    metric_stats, _ = jax.lax.scan(
        metric_step, carry, jnp.arange(n_chunks))
    return (*metric_stats, tau)


def _dense_rw_metric_operator_sums_sharded(
        metric_stats: tuple[jax.Array, ...],
        token_valid: jax.Array,
        admission_den_powers: jax.Array,
        srw_composition_mode: str) -> jax.Array:
    """Collapse one four-space token block to the canonical operator sums."""
    (gate_mass, gate_sq, gate_max,
     active_count, depth_sum, tau) = metric_stats
    additive = jax.lax.psum(
        jnp.stack(
            (active_count, gate_mass, gate_sq, depth_sum), axis=0),
        "model")
    active_count, gate_mass, gate_sq, depth_sum = additive
    gate_max = jax.lax.pmax(
        jax.lax.stop_gradient(gate_max), "model")
    gate_den = _shared_composition_den(
        gate_mass, admission_den_powers, srw_composition_mode)
    floor_mass = _shared_composition_den_floor_mass(
        srw_composition_mode)
    floor_compare = (
        jnp.less if srw_composition_mode == "linear_angular"
        else jnp.less_equal)
    valid = token_valid[None, ..., None].astype(jnp.float32)
    tau = tau * valid
    return jnp.stack((
        active_count.astype(jnp.float32).sum(axis=(1, 2, 3)),
        tau.astype(jnp.float32).sum(axis=(1, 2, 3)),
        gate_mass.astype(jnp.float32).sum(axis=(1, 2, 3)),
        (gate_den * valid).astype(jnp.float32).sum(axis=(1, 2, 3)),
        (depth_sum / jnp.maximum(active_count, 1.0)).astype(
            jnp.float32).sum(axis=(1, 2, 3)),
        (jnp.square(gate_mass) / jnp.maximum(
            gate_sq, 1.0e-8)).astype(
                jnp.float32).sum(axis=(1, 2, 3)),
        (gate_max / jnp.maximum(gate_mass, 1.0e-8)).astype(
            jnp.float32).sum(axis=(1, 2, 3)),
        (floor_compare(gate_mass, floor_mass) & (valid > 0.0)).astype(
            jnp.float32).sum(axis=(1, 2, 3)),
    ), axis=-1)


def _finish_blocked_dense_rw_metric_vector_sharded(
        operator_sums: jax.Array,
        route_specs: tuple[tuple[str, int], ...],
        routing: Mapping[str, jax.Array]) -> jax.Array:
    """Finish exact all-space metrics accumulated in four-space blocks."""
    global_sums = jax.lax.psum(operator_sums, "data")
    n_spaces = int(routing["space_gate"].shape[-1])
    token_count = int(routing["space_gate"].shape[0])
    global_positions = jax.lax.psum(
        jnp.float32(n_spaces * token_count), "data")
    global_positions = jnp.maximum(
        global_positions, jnp.float32(1.0))
    metric_values = []
    for route_index, (_, n_operators_per_space) in enumerate(route_specs):
        route_sum = global_sums[route_index]
        active_sum = route_sum[0]
        metric_values.extend((
            active_sum / global_positions / float(n_operators_per_space),
            active_sum / global_positions,
            route_sum[1] / global_positions,
            route_sum[2] / global_positions,
            route_sum[3] / global_positions,
            route_sum[4] / global_positions,
            route_sum[5] / global_positions,
            route_sum[6] / global_positions,
            route_sum[7] / global_positions,
        ))

    gate = jax.lax.stop_gradient(
        routing["space_gate"].astype(jnp.float32))
    mass = jax.lax.stop_gradient(
        routing["space_gate_mass"][..., 0].astype(jnp.float32))
    selected_ids = jax.lax.stop_gradient(routing["selected_ids"])
    top1_counts = jax.nn.one_hot(
        selected_ids[..., 0], n_spaces, dtype=jnp.float32).sum(axis=0)
    space_active_count = (
        (gate > 0.0).astype(jnp.float32).sum(axis=-1))
    routing_sums = jnp.concatenate((
        jnp.stack((
            mass.sum(),
            space_active_count.sum(),
            (mass <= 0.0).astype(jnp.float32).sum(),
        )),
        top1_counts,
        jnp.asarray((token_count,), dtype=jnp.float32),
    ))
    global_routing = jax.lax.psum(routing_sums, "data")
    global_tokens = jnp.maximum(
        global_routing[-1], jnp.float32(1.0))
    top1_offset = 3
    metric_values.extend((
        global_routing[0] / global_tokens,
        global_routing[1] / global_tokens,
        global_routing[2] / global_tokens,
        (global_routing[top1_offset:top1_offset + n_spaces]
         / global_tokens).max(),
    ))
    return jax.lax.stop_gradient(jnp.stack(metric_values))


def _collect_dense_rw_metric_vector_sharded(
        route_stats: tuple[tuple[jax.Array, ...], ...],
        route_specs: tuple[tuple[str, int], ...],
        admission_den_powers: jax.Array,
        routing: Mapping[str, jax.Array],
        srw_composition_mode: str) -> jax.Array:
    """Reduce metric-only scans to one compact stop-gradient FP32 vector."""
    (gate_mass, gate_sq, gate_max,
     active_count, depth_sum, tau) = (
         tuple(jnp.concatenate(values, axis=0)
               for values in zip(*route_stats)))
    additive_stats = jax.lax.psum(
        jnp.stack(
            (active_count, gate_mass, gate_sq, depth_sum), axis=0),
        "model")
    active_count, gate_mass, gate_sq, depth_sum = additive_stats
    gate_max = jax.lax.pmax(
        jax.lax.stop_gradient(gate_max), "model")
    gate_den = _shared_composition_den(
        gate_mass, admission_den_powers, srw_composition_mode)
    floor_mass = _shared_composition_den_floor_mass(
        srw_composition_mode)
    floor_compare = (
        jnp.less if srw_composition_mode == "linear_angular"
        else jnp.less_equal)
    operator_sums = []
    for route_index in range(len(route_specs)):
        route_active = active_count[route_index]
        route_mass = gate_mass[route_index]
        route_sq = gate_sq[route_index]
        route_max = gate_max[route_index]
        route_depth = depth_sum[route_index]
        route_tau = tau[route_index]
        route_den = gate_den[route_index]
        operator_sums.extend((
            route_active.astype(jnp.float32).sum(),
            route_tau.astype(jnp.float32).sum(),
            route_mass.astype(jnp.float32).sum(),
            route_den.astype(jnp.float32).sum(),
            (route_depth / jnp.maximum(route_active, 1.0)).astype(
                jnp.float32).sum(),
            (jnp.square(route_mass) / jnp.maximum(
                route_sq, 1.0e-8)).astype(jnp.float32).sum(),
            (route_max / jnp.maximum(
                route_mass, 1.0e-8)).astype(jnp.float32).sum(),
            floor_compare(
                route_mass, floor_mass).astype(jnp.float32).sum(),
        ))

    gate = jax.lax.stop_gradient(
        routing["space_gate"].astype(jnp.float32))
    mass = jax.lax.stop_gradient(
        routing["space_gate_mass"][..., 0].astype(jnp.float32))
    selected_ids = jax.lax.stop_gradient(routing["selected_ids"])
    n_spaces = int(gate.shape[-1])
    top1_counts = jax.nn.one_hot(
        selected_ids[..., 0], n_spaces, dtype=jnp.float32).sum(axis=0)
    space_active_count = (
        (gate > 0.0).astype(jnp.float32).sum(axis=-1))
    local_positions = jnp.float32(
        int(active_count.shape[1]) * int(active_count.shape[2]))
    local_tokens = jnp.float32(gate.shape[0])
    packed_metrics = jnp.concatenate((
        jnp.stack(operator_sums),
        jnp.stack((
            mass.sum(),
            space_active_count.sum(),
            (mass <= 0.0).astype(jnp.float32).sum(),
        )),
        top1_counts,
        jnp.stack((local_positions, local_tokens)),
    ))
    global_metrics = jax.lax.psum(packed_metrics, "data")

    operator_width = 8
    space_offset = operator_width * len(route_specs)
    top1_offset = space_offset + 3
    global_positions = jnp.maximum(
        global_metrics[-2], jnp.float32(1.0))
    global_tokens = jnp.maximum(
        global_metrics[-1], jnp.float32(1.0))
    metric_values = []
    for route_index, (_, n_operators_per_space) in enumerate(route_specs):
        offset = route_index * operator_width
        active_sum = global_metrics[offset]
        metric_values.extend((
            active_sum / global_positions / float(n_operators_per_space),
            active_sum / global_positions,
            global_metrics[offset + 1] / global_positions,
            global_metrics[offset + 2] / global_positions,
            global_metrics[offset + 3] / global_positions,
            global_metrics[offset + 4] / global_positions,
            global_metrics[offset + 5] / global_positions,
            global_metrics[offset + 6] / global_positions,
            global_metrics[offset + 7] / global_positions,
        ))
    metric_values.extend((
        global_metrics[space_offset] / global_tokens,
        global_metrics[space_offset + 1] / global_tokens,
        global_metrics[space_offset + 2] / global_tokens,
        (global_metrics[top1_offset:top1_offset + n_spaces]
         / global_tokens).max(),
    ))
    return jax.lax.stop_gradient(jnp.stack(metric_values))


def _make_sharded_attention_space_dense(
        mesh, *,
        max_chunk_size_qk: int,
        max_chunk_size_v: int,
        operation_space_top_k: int,
        admission_den_power_qk: float,
        admission_den_power_v: float,
        srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA,
        soft_gate_effective_active_eps: float = 1.0e-6,
        throughput_bf16: bool,
        output_collective_bf16: bool):
    """Create the single-boundary production Q/K/V dense executor."""
    composition_mode = str(srw_composition_mode)
    top_k = int(operation_space_top_k)
    model_axis_size = int(mesh.shape["model"])
    metric_names = tuple(
        f"{route}_operator_{suffix}"
        for route in ("q", "k", "v")
        for suffix in _FUSED_OPERATOR_METRIC_SUFFIXES
    ) + tuple(
        f"attention_space_{suffix}"
        for suffix in (
            "gate_mass_mean", "active_count_mean",
            "zero_gate_frac", "top1_rate"))

    def attention_core(
            flat_state,
            space_route_kernel, space_read_vectors,
            space_state_proj, space_state_writeback,
            q_tau_kernel, q_tau_bias,
            k_tau_kernel, k_tau_bias,
            v_tau_kernel, v_tau_bias,
            q_read, q_write, k_read, k_write, v_read, v_write,
            qk_temperature, v_temperature,
            boundary_power, execution_prune_eps,
            qk_scale, v_scale, collect_metrics):
        routing = _compute_space_routing(
            flat_state, space_route_kernel, space_read_vectors, top_k)
        if throughput_bf16:
            local = jnp.swapaxes(
                throughput_dot_bf16_f32(
                    flat_state, space_state_proj,
                    dimension_numbers=(
                        ((1,), (1,)),
                        ((), ()))),
                0, 1)
        else:
            local = _control_einsum_f32(
                "td,mdr->mtr", flat_state, space_state_proj)
        local = local.astype(jnp.float32)
        local_norm = jnp.maximum(
            jnp.linalg.norm(local, axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))
        token_valid = jnp.ones(local.shape[:2], dtype=jnp.bool_)
        qk_tau_kernel = jnp.stack((q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_bias = jnp.stack((q_tau_bias, k_tau_bias), axis=0)
        qk_raw_tau = (
            _control_einsum_f32(
                "mtr,ari->amti", local, qk_tau_kernel)
            + qk_tau_bias[:, None, None, :])
        qk_local_output = _dense_rw_output_sharded(
            local, local_norm, token_valid,
            jnp.stack((q_read, k_read), axis=0),
            jnp.stack((q_write, k_write), axis=0),
            qk_raw_tau,
            max_chunk_size=max_chunk_size_qk,
            soft_gate_temperature=qk_temperature,
            soft_gate_boundary_power=boundary_power,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            effective_active_eps=soft_gate_effective_active_eps,
            throughput_bf16=throughput_bf16)

        v_raw_tau = (
            _control_einsum_f32(
                "mtr,ri->mti", local, v_tau_kernel)
            + v_tau_bias)
        v_local_output = _dense_rw_output_sharded(
            local, local_norm, token_valid,
            v_read[None, ...], v_write[None, ...],
            v_raw_tau[None, ...],
            max_chunk_size=max_chunk_size_v,
            soft_gate_temperature=v_temperature,
            soft_gate_boundary_power=boundary_power,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            effective_active_eps=soft_gate_effective_active_eps,
            throughput_bf16=throughput_bf16)
        qk_raw_out, qk_gate_mass = qk_local_output
        v_raw_out, v_gate_mass = v_local_output
        grouped_raw_out = jnp.concatenate(
            (qk_raw_out, v_raw_out), axis=0)
        grouped_gate_mass = jnp.concatenate(
            (qk_gate_mass, v_gate_mass), axis=0)
        grouped_den_powers = jnp.asarray(
            (
                admission_den_power_qk,
                admission_den_power_qk,
                admission_den_power_v,
            ),
            dtype=jnp.float32).reshape((3, 1, 1, 1))
        _, grouped_gate_den = _global_dense_rw_den_sharded(
            grouped_gate_mass,
            grouped_den_powers,
            composition_mode)
        local_grouped_space_results = (
            grouped_raw_out / grouped_gate_den).astype(jnp.float32)
        space_weights = jnp.swapaxes(
            routing["dense_space_weights"], 0, 1)[None, ..., None]
        route_scales = jnp.asarray(
            (qk_scale, qk_scale, v_scale), dtype=jnp.float32
        ).reshape((3, 1, 1, 1))
        writeback_dot = (
            _throughput_einsum_bf16_f32
            if throughput_bf16 else _control_einsum_f32)
        local_grouped_output = writeback_dot(
            "amtr,mrd->atd",
            local_grouped_space_results * space_weights * route_scales,
            space_state_writeback).astype(jnp.float32)
        grouped_output = _psum_dense_rw_representation_sharded(
            local_grouped_output, output_collective_bf16)

        q_per_space = int(q_read.shape[1]) * model_axis_size
        k_per_space = int(k_read.shape[1]) * model_axis_size
        v_per_space = int(v_read.shape[1]) * model_axis_size
        route_specs = (
            ("q", q_per_space),
            ("k", k_per_space),
            ("v", v_per_space),
        )
        metric_local = jax.lax.stop_gradient(local)
        metric_norm = jax.lax.stop_gradient(local_norm)
        metric_qk_reads = jax.lax.stop_gradient(
            jnp.stack((q_read, k_read), axis=0))
        metric_v_reads = jax.lax.stop_gradient(v_read[None, ...])
        metric_qk_tau = jax.lax.stop_gradient(qk_raw_tau)
        metric_v_tau = jax.lax.stop_gradient(v_raw_tau[None, ...])
        metric_routing = jax.tree.map(
            jax.lax.stop_gradient, routing)
        metric_qk_temperature = jax.lax.stop_gradient(qk_temperature)
        metric_v_temperature = jax.lax.stop_gradient(v_temperature)
        metric_boundary_power = jax.lax.stop_gradient(boundary_power)
        metric_execution_prune_eps = jax.lax.stop_gradient(
            execution_prune_eps)
        metric_operands = (
            metric_local,
            metric_norm,
            token_valid,
            metric_qk_reads,
            metric_qk_tau,
            metric_v_reads,
            metric_v_tau,
            metric_routing,
            metric_qk_temperature,
            metric_v_temperature,
            metric_boundary_power,
            metric_execution_prune_eps,
            grouped_den_powers,
        )

        def collect_metric_vector(operands):
            (metric_local_value,
             metric_norm_value,
             metric_token_valid,
             metric_qk_reads_value,
             metric_qk_tau_value,
             metric_v_reads_value,
             metric_v_tau_value,
             metric_routing_value,
             metric_qk_temperature_value,
             metric_v_temperature_value,
             metric_boundary_power_value,
             metric_execution_prune_eps_value,
             metric_den_powers_value) = operands
            qk_metric_stats = _dense_rw_metric_stats_sharded(
                metric_local_value, metric_norm_value, metric_token_valid,
                metric_qk_reads_value,
                metric_qk_tau_value,
                max_chunk_size=max_chunk_size_qk,
                soft_gate_temperature=metric_qk_temperature_value,
                soft_gate_boundary_power=metric_boundary_power_value,
                execution_prune_eps=metric_execution_prune_eps_value,
                srw_composition_mode=composition_mode,
                heat_kernel_beta=heat_kernel_beta,
                effective_active_eps=soft_gate_effective_active_eps,
                throughput_bf16=throughput_bf16)
            v_metric_stats = _dense_rw_metric_stats_sharded(
                metric_local_value, metric_norm_value, metric_token_valid,
                metric_v_reads_value, metric_v_tau_value,
                max_chunk_size=max_chunk_size_v,
                soft_gate_temperature=metric_v_temperature_value,
                soft_gate_boundary_power=metric_boundary_power_value,
                execution_prune_eps=metric_execution_prune_eps_value,
                srw_composition_mode=composition_mode,
                heat_kernel_beta=heat_kernel_beta,
                effective_active_eps=soft_gate_effective_active_eps,
                throughput_bf16=throughput_bf16)
            return _collect_dense_rw_metric_vector_sharded(
                (qk_metric_stats, v_metric_stats),
                route_specs,
                metric_den_powers_value,
                metric_routing_value,
                composition_mode)

        metric_vector = _metric_only_cond(
            collect_metric_vector,
            len(metric_names),
            collect_metrics,
            metric_operands)
        metrics = {
            name: metric_vector[index]
            for index, name in enumerate(metric_names)}
        return (
            grouped_output[0], grouped_output[1], grouped_output[2], metrics)

    kernel = shard_map(
        attention_core,
        mesh=mesh,
        in_specs=(
            P("data", None),
            P(), P(), P(), P(),
            P(), P(), P(), P(), P(), P(),
            P(None, "model", None), P(None, "model", None),
            P(None, "model", None), P(None, "model", None),
            P(None, "model", None), P(None, "model", None),
            P(), P(), P(), P(), P(), P(),
            P(),
        ),
        out_specs=(
            P("data", None), P("data", None), P("data", None),
            {name: P() for name in metric_names}),
        check_rep=False)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "attention_qkv"
    kernel._v4174_execution_mode = "dense_all_space"
    kernel._v4174_qk_paired = True
    kernel._v4174_dynamic_metric_flag = True
    kernel._v4174_chunk_remat_policy = "always"
    kernel._v4174_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4174_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4174_output_contract = ("q[T,D]", "k[T,D]", "v[T,D]", "scalars")
    return kernel


def make_sharded_attention_space_dense_minimal(mesh, **kwargs):
    """Create the canonical mixed-precision production attention executor."""
    return _make_sharded_attention_space_dense(
        mesh, throughput_bf16=True, output_collective_bf16=True, **kwargs)


def make_sharded_attention_space_dense_fp32_collective_reference(
        mesh, **kwargs):
    """Keep production GEMMs while reducing the final representation in FP32."""
    return _make_sharded_attention_space_dense(
        mesh, throughput_bf16=True, output_collective_bf16=False, **kwargs)


def make_sharded_attention_space_dense_fp32_reference(mesh, **kwargs):
    """Create a fused FP32 executor for structural/numerical tests only."""
    return _make_sharded_attention_space_dense(
        mesh, throughput_bf16=False, output_collective_bf16=False, **kwargs)


def _make_sharded_attention_space_bundle_dense(
        mesh, *,
        max_chunk_size_qk: int,
        max_chunk_size_v: int,
        operation_space_top_k: int,
        admission_den_power_qk: float,
        admission_den_power_v: float,
        operation_space_bundle_size: int,
        operation_space_bundle_token_block_size: int,
        srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA,
        soft_gate_effective_active_eps: float = 1.0e-6,
        throughput_bf16: bool,
        output_collective_bf16: bool):
    """Create compact fixed-four-space production Q/K/V execution."""
    composition_mode = str(srw_composition_mode)
    top_k = int(operation_space_top_k)
    bundle_size = int(operation_space_bundle_size)
    token_block_size = int(operation_space_bundle_token_block_size)
    if top_k != 2:
        raise ValueError("bundle_dense attention requires top_k=2")
    if bundle_size != BUNDLE_DENSE_SIZE:
        raise ValueError(
            f"bundle_dense attention requires bundle_size={BUNDLE_DENSE_SIZE}")
    if token_block_size <= 0:
        raise ValueError("bundle_dense token_block_size must be positive")
    model_axis_size = int(mesh.shape["model"])
    operator_metric_names = tuple(
        f"{route}_operator_{suffix}"
        for route in ("q", "k", "v")
        for suffix in _FUSED_OPERATOR_METRIC_SUFFIXES
    ) + tuple(
        f"attention_space_{suffix}"
        for suffix in (
            "gate_mass_mean", "active_count_mean",
            "zero_gate_frac", "top1_rate"))
    bundle_metric_names = _bundle_packing_metric_names("attention")

    def attention_core(
            flat_state,
            space_route_kernel, space_read_vectors,
            space_state_proj, space_state_writeback,
            q_tau_kernel, q_tau_bias,
            k_tau_kernel, k_tau_bias,
            v_tau_kernel, v_tau_bias,
            q_read, q_write, k_read, k_write, v_read, v_write,
            qk_temperature, v_temperature,
            boundary_power, execution_prune_eps,
            qk_scale, v_scale, collect_metrics):
        routing = _compute_space_routing(
            flat_state, space_route_kernel, space_read_vectors, top_k)
        packing, packing_metrics = _pack_top2_bundle_entries_sharded(
            routing,
            bundle_size=bundle_size,
            token_block_size=token_block_size,
            stage="attention")
        token_count = int(flat_state.shape[0])
        d_model = int(flat_state.shape[1])
        n_spaces = int(space_state_proj.shape[0])
        if n_spaces % bundle_size:
            raise ValueError(
                "bundle_dense attention space count is not divisible by 4")
        n_bundles = n_spaces // bundle_size
        scan_blocks = (
            int(packing["token_id"].shape[0]) // token_block_size)
        qk_reads = jnp.stack((q_read, k_read), axis=0)
        qk_writes = jnp.stack((q_write, k_write), axis=0)
        qk_tau_kernels = jnp.stack(
            (q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_biases = jnp.stack(
            (q_tau_bias, k_tau_bias), axis=0)
        den_powers = jnp.asarray(
            (
                admission_den_power_qk,
                admission_den_power_qk,
                admission_den_power_v,
            ),
            dtype=jnp.float32).reshape((3, 1, 1, 1))
        route_scales = jnp.asarray(
            (qk_scale, qk_scale, v_scale), dtype=jnp.float32
        ).reshape((3, 1, 1, 1))
        projection_dot = (
            _throughput_einsum_bf16_f32
            if throughput_bf16 else _control_einsum_f32)
        writeback_dot = projection_dot
        initial_output = jnp.zeros(
            (3, token_count, d_model), dtype=jnp.float32)

        def block_step(local_output, block_index):
            packed_start = block_index * token_block_size
            block_bundle = jax.lax.dynamic_index_in_dim(
                packing["bundle_id"], packed_start,
                axis=0, keepdims=False)

            def execute_block(output_value):
                space_start = block_bundle * bundle_size
                block_token = jax.lax.dynamic_slice_in_dim(
                    packing["token_id"], packed_start,
                    token_block_size, axis=0)
                block_valid = jax.lax.dynamic_slice_in_dim(
                    packing["token_valid"], packed_start,
                    token_block_size, axis=0)
                block_membership = jax.lax.dynamic_slice_in_dim(
                    packing["membership_mask"], packed_start,
                    token_block_size, axis=0)
                block_weight = jax.lax.dynamic_slice_in_dim(
                    packing["routing_weight"], packed_start,
                    token_block_size, axis=0)
                packed_state = flat_state[block_token]
                projection = jax.lax.dynamic_slice_in_dim(
                    space_state_proj, space_start, bundle_size, axis=0)
                local = projection_dot(
                    "bd,mdr->mbr", packed_state, projection
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                token_space_valid = (
                    jnp.swapaxes(block_membership, 0, 1)
                    & block_valid[None, :])

                qk_raw_tau = (
                    _control_einsum_f32(
                        "mbr,ari->ambi", local, qk_tau_kernels)
                    + qk_tau_biases[:, None, None, :])
                block_qk_read = jax.lax.dynamic_slice_in_dim(
                    qk_reads, space_start, bundle_size, axis=1)
                block_qk_write = jax.lax.dynamic_slice_in_dim(
                    qk_writes, space_start, bundle_size, axis=1)
                qk_raw_out, qk_gate_mass = _dense_rw_output_sharded(
                    local, local_norm, token_space_valid,
                    block_qk_read, block_qk_write, qk_raw_tau,
                    max_chunk_size=max_chunk_size_qk,
                    soft_gate_temperature=qk_temperature,
                    soft_gate_boundary_power=boundary_power,
                    execution_prune_eps=execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)

                v_raw_tau = (
                    _control_einsum_f32(
                        "mbr,ri->mbi", local, v_tau_kernel)
                    + v_tau_bias)
                block_v_read = jax.lax.dynamic_slice_in_dim(
                    v_read, space_start, bundle_size, axis=0)
                block_v_write = jax.lax.dynamic_slice_in_dim(
                    v_write, space_start, bundle_size, axis=0)
                v_raw_out, v_gate_mass = _dense_rw_output_sharded(
                    local, local_norm, token_space_valid,
                    block_v_read[None, ...], block_v_write[None, ...],
                    v_raw_tau[None, ...],
                    max_chunk_size=max_chunk_size_v,
                    soft_gate_temperature=v_temperature,
                    soft_gate_boundary_power=boundary_power,
                    execution_prune_eps=execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)
                grouped_raw_out = jnp.concatenate(
                    (qk_raw_out, v_raw_out), axis=0)
                grouped_gate_mass = jnp.concatenate(
                    (qk_gate_mass, v_gate_mass), axis=0)
                _, grouped_gate_den = _global_dense_rw_den_sharded(
                    grouped_gate_mass, den_powers, composition_mode)
                local_results = (
                    grouped_raw_out / grouped_gate_den).astype(jnp.float32)
                weights = jnp.swapaxes(
                    block_weight, 0, 1)[None, ..., None]
                writeback = jax.lax.dynamic_slice_in_dim(
                    space_state_writeback,
                    space_start, bundle_size, axis=0)
                block_output = writeback_dot(
                    "ambr,mrd->abd",
                    local_results * weights * route_scales,
                    writeback).astype(jnp.float32)
                block_output = jnp.where(
                    block_valid[None, :, None], block_output, 0.0)
                return output_value.at[
                    :, block_token, :].add(block_output)

            return jax.lax.cond(
                block_bundle < n_bundles,
                execute_block,
                lambda output_value: output_value,
                local_output), None

        scan_step = jax.checkpoint(block_step, prevent_cse=False)
        local_grouped_output, _ = jax.lax.scan(
            scan_step, initial_output, jnp.arange(scan_blocks))
        grouped_output = _psum_dense_rw_representation_sharded(
            local_grouped_output, output_collective_bf16)

        q_per_space = int(q_read.shape[1]) * model_axis_size
        k_per_space = int(k_read.shape[1]) * model_axis_size
        v_per_space = int(v_read.shape[1]) * model_axis_size
        route_specs = (
            ("q", q_per_space),
            ("k", k_per_space),
            ("v", v_per_space),
        )
        metric_operands = (
            jax.lax.stop_gradient(flat_state),
            jax.lax.stop_gradient(space_state_proj),
            jax.lax.stop_gradient(qk_tau_kernels),
            jax.lax.stop_gradient(qk_tau_biases),
            jax.lax.stop_gradient(v_tau_kernel),
            jax.lax.stop_gradient(v_tau_bias),
            jax.lax.stop_gradient(qk_reads),
            jax.lax.stop_gradient(v_read),
            jax.tree.map(jax.lax.stop_gradient, routing),
            jax.lax.stop_gradient(qk_temperature),
            jax.lax.stop_gradient(v_temperature),
            jax.lax.stop_gradient(boundary_power),
            jax.lax.stop_gradient(execution_prune_eps),
        )

        def collect_metric_vector(operands):
            (metric_state,
             metric_projection,
             metric_qk_tau_kernels,
             metric_qk_tau_biases,
             metric_v_tau_kernel,
             metric_v_tau_bias,
             metric_qk_reads,
             metric_v_reads,
             metric_routing,
             metric_qk_temperature,
             metric_v_temperature,
             metric_boundary_power,
             metric_execution_prune_eps) = operands
            metric_token_blocks = math.ceil(
                token_count / token_block_size)
            metric_capacity = metric_token_blocks * token_block_size
            metric_state = jnp.pad(
                metric_state,
                ((0, metric_capacity - token_count), (0, 0)))
            initial_sums = jnp.zeros((3, 8), dtype=jnp.float32)

            def metric_block_step(operator_sums, block_index):
                bundle_id = block_index // metric_token_blocks
                token_block = block_index % metric_token_blocks
                token_start = token_block * token_block_size
                space_start = bundle_id * bundle_size
                state_block = jax.lax.dynamic_slice_in_dim(
                    metric_state, token_start,
                    token_block_size, axis=0)
                token_valid = (
                    jnp.arange(token_block_size) + token_start
                    < token_count)
                token_space_valid = jnp.broadcast_to(
                    token_valid[None, :],
                    (bundle_size, token_block_size))
                projection_block = jax.lax.dynamic_slice_in_dim(
                    metric_projection, space_start,
                    bundle_size, axis=0)
                local = projection_dot(
                    "bd,mdr->mbr", state_block, projection_block
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                qk_tau = (
                    _control_einsum_f32(
                        "mbr,ari->ambi",
                        local, metric_qk_tau_kernels)
                    + metric_qk_tau_biases[:, None, None, :])
                qk_read_block = jax.lax.dynamic_slice_in_dim(
                    metric_qk_reads, space_start,
                    bundle_size, axis=1)
                qk_stats = _dense_rw_metric_stats_sharded(
                    local, local_norm, token_space_valid,
                    qk_read_block, qk_tau,
                    max_chunk_size=max_chunk_size_qk,
                    soft_gate_temperature=metric_qk_temperature,
                    soft_gate_boundary_power=metric_boundary_power,
                    execution_prune_eps=metric_execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)
                qk_sums = _dense_rw_metric_operator_sums_sharded(
                    qk_stats, token_space_valid,
                    den_powers[:2], composition_mode)

                v_tau = (
                    _control_einsum_f32(
                        "mbr,ri->mbi", local,
                        metric_v_tau_kernel)
                    + metric_v_tau_bias)
                v_read_block = jax.lax.dynamic_slice_in_dim(
                    metric_v_reads, space_start,
                    bundle_size, axis=0)
                v_stats = _dense_rw_metric_stats_sharded(
                    local, local_norm, token_space_valid,
                    v_read_block[None, ...], v_tau[None, ...],
                    max_chunk_size=max_chunk_size_v,
                    soft_gate_temperature=metric_v_temperature,
                    soft_gate_boundary_power=metric_boundary_power,
                    execution_prune_eps=metric_execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)
                v_sums = _dense_rw_metric_operator_sums_sharded(
                    v_stats, token_space_valid,
                    den_powers[2:3], composition_mode)
                return operator_sums + jnp.concatenate(
                    (qk_sums, v_sums), axis=0), None

            operator_sums, _ = jax.lax.scan(
                metric_block_step,
                initial_sums,
                jnp.arange(n_bundles * metric_token_blocks))
            return _finish_blocked_dense_rw_metric_vector_sharded(
                operator_sums, route_specs, metric_routing)

        metric_vector = _metric_only_cond(
            collect_metric_vector,
            len(operator_metric_names),
            collect_metrics,
            metric_operands)
        metrics = {
            name: metric_vector[index]
            for index, name in enumerate(operator_metric_names)}
        metrics.update(packing_metrics)
        return (
            grouped_output[0], grouped_output[1], grouped_output[2], metrics)

    all_metric_names = (*operator_metric_names, *bundle_metric_names)
    kernel = shard_map(
        attention_core,
        mesh=mesh,
        in_specs=(
            P("data", None),
            P(), P(), P(), P(),
            P(), P(), P(), P(), P(), P(),
            P(None, "model", None), P(None, "model", None),
            P(None, "model", None), P(None, "model", None),
            P(None, "model", None), P(None, "model", None),
            P(), P(), P(), P(), P(), P(),
            P(),
        ),
        out_specs=(
            P("data", None), P("data", None), P("data", None),
            {name: P() for name in all_metric_names}),
        check_rep=False)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "attention_qkv_bundle4"
    kernel._v4174_execution_mode = "bundle_dense"
    kernel._v4174_bundle_size = bundle_size
    kernel._v4174_bundle_token_block_size = token_block_size
    kernel._v4174_packed_entry_capacity = (
        "2*T + n_bundles*(token_block_size-1)")
    kernel._v4174_bundle_packing = "prefix_count_exact"
    kernel._v4174_bundle_packing_vjp = "saved_metadata_scatter"
    kernel._v4174_attention_packing_count = 1
    kernel._v4174_qk_paired = True
    kernel._v4174_dynamic_metric_flag = True
    kernel._v4174_chunk_remat_policy = "always"
    kernel._v4174_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4174_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4174_output_contract = (
        "q[T,D]", "k[T,D]", "v[T,D]", "scalars")
    return kernel


def make_sharded_attention_space_bundle_dense_minimal(mesh, **kwargs):
    """Create the mixed-precision compact four-space attention executor."""
    return _make_sharded_attention_space_bundle_dense(
        mesh, throughput_bf16=True, output_collective_bf16=True, **kwargs)


def make_sharded_attention_space_bundle_dense_fp32_reference(mesh, **kwargs):
    """Create the compact FP32 attention executor for parity checks."""
    return _make_sharded_attention_space_bundle_dense(
        mesh, throughput_bf16=False, output_collective_bf16=False, **kwargs)


def _make_sharded_rst_space_dense(
        mesh, *,
        max_chunk_size: int,
        operation_space_top_k: int,
        admission_den_power: float,
        srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA,
        soft_gate_effective_active_eps: float = 1.0e-6,
        throughput_bf16: bool,
        output_collective_bf16: bool):
    """Create the end-to-end production RST dense executor."""
    composition_mode = str(srw_composition_mode)
    top_k = int(operation_space_top_k)
    model_axis_size = int(mesh.shape["model"])
    metric_names = tuple(
        f"rst_operator_{suffix}"
        for suffix in _FUSED_OPERATOR_METRIC_SUFFIXES
    ) + tuple(
        f"rst_space_{suffix}"
        for suffix in (
            "gate_mass_mean", "active_count_mean",
            "zero_gate_frac", "top1_rate"))

    def rst_core(
            flat_state,
            space_route_kernel, space_read_vectors,
            space_state_proj, space_state_writeback,
            tau_kernel, tau_bias,
            read_vectors, write_vectors,
            temperature, boundary_power, execution_prune_eps,
            route_scale, collect_metrics):
        routing = _compute_space_routing(
            flat_state, space_route_kernel, space_read_vectors, top_k)
        if throughput_bf16:
            local = jnp.swapaxes(
                throughput_dot_bf16_f32(
                    flat_state, space_state_proj,
                    dimension_numbers=(
                        ((1,), (1,)),
                        ((), ()))),
                0, 1)
        else:
            local = _control_einsum_f32(
                "td,mdr->mtr", flat_state, space_state_proj)
        local = local.astype(jnp.float32)
        local_norm = jnp.maximum(
            jnp.linalg.norm(local, axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))
        raw_tau = (
            _control_einsum_f32(
                "mtr,ri->mti", local, tau_kernel)
            + tau_bias)
        local_output = _dense_rw_output_sharded(
            local, local_norm,
            jnp.ones(local.shape[:2], dtype=jnp.bool_),
            read_vectors[None, ...], write_vectors[None, ...],
            raw_tau[None, ...],
            max_chunk_size=max_chunk_size,
            soft_gate_temperature=temperature,
            soft_gate_boundary_power=boundary_power,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            effective_active_eps=soft_gate_effective_active_eps,
            throughput_bf16=throughput_bf16)
        raw_out, gate_mass = local_output
        _, gate_den = _global_dense_rw_den_sharded(
            gate_mass, jnp.float32(admission_den_power),
            composition_mode)
        local_space_result = (raw_out / gate_den).astype(jnp.float32)
        local_weighted = (
            local_space_result[0]
            * jnp.swapaxes(
                routing["dense_space_weights"], 0, 1)[..., None]
            * jnp.asarray(route_scale))
        writeback_dot = (
            _throughput_einsum_bf16_f32
            if throughput_bf16 else _control_einsum_f32)
        local_update = writeback_dot(
            "mtr,mrd->td", local_weighted,
            space_state_writeback).astype(jnp.float32)
        update = _psum_dense_rw_representation_sharded(
            local_update, output_collective_bf16)
        n_per_space = int(read_vectors.shape[1]) * model_axis_size
        route_specs = (("rst", n_per_space),)
        metric_local = jax.lax.stop_gradient(local)
        metric_norm = jax.lax.stop_gradient(local_norm)
        metric_reads = jax.lax.stop_gradient(read_vectors[None, ...])
        metric_tau = jax.lax.stop_gradient(raw_tau[None, ...])
        metric_routing = jax.tree.map(
            jax.lax.stop_gradient, routing)
        metric_temperature = jax.lax.stop_gradient(temperature)
        metric_boundary_power = jax.lax.stop_gradient(boundary_power)
        metric_execution_prune_eps = jax.lax.stop_gradient(
            execution_prune_eps)
        metric_den_powers = jnp.float32(
            admission_den_power).reshape((1, 1, 1, 1))
        metric_operands = (
            metric_local,
            metric_norm,
            jnp.ones(local.shape[:2], dtype=jnp.bool_),
            metric_reads,
            metric_tau,
            metric_routing,
            metric_temperature,
            metric_boundary_power,
            metric_execution_prune_eps,
            metric_den_powers,
        )

        def collect_metric_vector(operands):
            (metric_local_value,
             metric_norm_value,
             metric_token_valid,
             metric_reads_value,
             metric_tau_value,
             metric_routing_value,
             metric_temperature_value,
             metric_boundary_power_value,
             metric_execution_prune_eps_value,
             metric_den_powers_value) = operands
            metric_stats = _dense_rw_metric_stats_sharded(
                metric_local_value, metric_norm_value,
                metric_token_valid,
                metric_reads_value, metric_tau_value,
                max_chunk_size=max_chunk_size,
                soft_gate_temperature=metric_temperature_value,
                soft_gate_boundary_power=metric_boundary_power_value,
                execution_prune_eps=metric_execution_prune_eps_value,
                srw_composition_mode=composition_mode,
                heat_kernel_beta=heat_kernel_beta,
                effective_active_eps=soft_gate_effective_active_eps,
                throughput_bf16=throughput_bf16)
            return _collect_dense_rw_metric_vector_sharded(
                (metric_stats,),
                route_specs,
                metric_den_powers_value,
                metric_routing_value,
                composition_mode)

        metric_vector = _metric_only_cond(
            collect_metric_vector,
            len(metric_names),
            collect_metrics,
            metric_operands)
        metrics = {
            name: metric_vector[index]
            for index, name in enumerate(metric_names)}
        return update, metrics

    kernel = shard_map(
        rst_core,
        mesh=mesh,
        in_specs=(
            P("data", None),
            P(), P(), P(), P(), P(), P(),
            P(None, "model", None), P(None, "model", None),
            P(), P(), P(), P(),
            P(),
        ),
        out_specs=(
            P("data", None),
            {name: P() for name in metric_names}),
        check_rep=False)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "rst_end_to_end"
    kernel._v4174_execution_mode = "dense_all_space"
    kernel._v4174_dynamic_metric_flag = True
    kernel._v4174_chunk_remat_policy = "always"
    kernel._v4174_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4174_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4174_output_contract = ("rst[T,D]", "scalars")
    return kernel


def make_sharded_rst_space_dense_minimal(mesh, **kwargs):
    """Create the canonical mixed-precision production RST executor."""
    return _make_sharded_rst_space_dense(
        mesh, throughput_bf16=True, output_collective_bf16=True, **kwargs)


def make_sharded_rst_space_dense_fp32_collective_reference(mesh, **kwargs):
    """Keep production GEMMs while reducing the final representation in FP32."""
    return _make_sharded_rst_space_dense(
        mesh, throughput_bf16=True, output_collective_bf16=False, **kwargs)


def make_sharded_rst_space_dense_fp32_reference(mesh, **kwargs):
    """Create a fused FP32 RST executor for numerical tests only."""
    return _make_sharded_rst_space_dense(
        mesh, throughput_bf16=False, output_collective_bf16=False, **kwargs)


def _make_sharded_rst_space_bundle_dense(
        mesh, *,
        max_chunk_size: int,
        operation_space_top_k: int,
        admission_den_power: float,
        operation_space_bundle_size: int,
        operation_space_bundle_token_block_size: int,
        srw_composition_mode: str = DEFAULT_SRW_COMPOSITION_MODE,
        heat_kernel_beta: float = DEFAULT_HEAT_KERNEL_BETA,
        soft_gate_effective_active_eps: float = 1.0e-6,
        throughput_bf16: bool,
        output_collective_bf16: bool):
    """Create compact fixed-four-space production RST execution."""
    composition_mode = str(srw_composition_mode)
    top_k = int(operation_space_top_k)
    bundle_size = int(operation_space_bundle_size)
    token_block_size = int(operation_space_bundle_token_block_size)
    if top_k != 2:
        raise ValueError("bundle_dense RST requires top_k=2")
    if bundle_size != BUNDLE_DENSE_SIZE:
        raise ValueError(
            f"bundle_dense RST requires bundle_size={BUNDLE_DENSE_SIZE}")
    if token_block_size <= 0:
        raise ValueError("bundle_dense token_block_size must be positive")
    model_axis_size = int(mesh.shape["model"])
    operator_metric_names = tuple(
        f"rst_operator_{suffix}"
        for suffix in _FUSED_OPERATOR_METRIC_SUFFIXES
    ) + tuple(
        f"rst_space_{suffix}"
        for suffix in (
            "gate_mass_mean", "active_count_mean",
            "zero_gate_frac", "top1_rate"))
    bundle_metric_names = _bundle_packing_metric_names("rst")

    def rst_core(
            flat_state,
            space_route_kernel, space_read_vectors,
            space_state_proj, space_state_writeback,
            tau_kernel, tau_bias,
            read_vectors, write_vectors,
            temperature, boundary_power, execution_prune_eps,
            route_scale, collect_metrics):
        routing = _compute_space_routing(
            flat_state, space_route_kernel, space_read_vectors, top_k)
        packing, packing_metrics = _pack_top2_bundle_entries_sharded(
            routing,
            bundle_size=bundle_size,
            token_block_size=token_block_size,
            stage="rst")
        token_count = int(flat_state.shape[0])
        d_model = int(flat_state.shape[1])
        n_spaces = int(space_state_proj.shape[0])
        if n_spaces % bundle_size:
            raise ValueError(
                "bundle_dense RST space count is not divisible by 4")
        n_bundles = n_spaces // bundle_size
        scan_blocks = (
            int(packing["token_id"].shape[0]) // token_block_size)
        den_power = jnp.float32(
            admission_den_power).reshape((1, 1, 1, 1))
        projection_dot = (
            _throughput_einsum_bf16_f32
            if throughput_bf16 else _control_einsum_f32)
        writeback_dot = projection_dot
        initial_output = jnp.zeros(
            (token_count, d_model), dtype=jnp.float32)

        def block_step(local_output, block_index):
            packed_start = block_index * token_block_size
            block_bundle = jax.lax.dynamic_index_in_dim(
                packing["bundle_id"], packed_start,
                axis=0, keepdims=False)

            def execute_block(output_value):
                space_start = block_bundle * bundle_size
                block_token = jax.lax.dynamic_slice_in_dim(
                    packing["token_id"], packed_start,
                    token_block_size, axis=0)
                block_valid = jax.lax.dynamic_slice_in_dim(
                    packing["token_valid"], packed_start,
                    token_block_size, axis=0)
                block_membership = jax.lax.dynamic_slice_in_dim(
                    packing["membership_mask"], packed_start,
                    token_block_size, axis=0)
                block_weight = jax.lax.dynamic_slice_in_dim(
                    packing["routing_weight"], packed_start,
                    token_block_size, axis=0)
                packed_state = flat_state[block_token]
                projection = jax.lax.dynamic_slice_in_dim(
                    space_state_proj, space_start, bundle_size, axis=0)
                local = projection_dot(
                    "bd,mdr->mbr", packed_state, projection
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                token_space_valid = (
                    jnp.swapaxes(block_membership, 0, 1)
                    & block_valid[None, :])
                raw_tau = (
                    _control_einsum_f32(
                        "mbr,ri->mbi", local, tau_kernel)
                    + tau_bias)
                block_read = jax.lax.dynamic_slice_in_dim(
                    read_vectors, space_start, bundle_size, axis=0)
                block_write = jax.lax.dynamic_slice_in_dim(
                    write_vectors, space_start, bundle_size, axis=0)
                raw_out, gate_mass = _dense_rw_output_sharded(
                    local, local_norm, token_space_valid,
                    block_read[None, ...], block_write[None, ...],
                    raw_tau[None, ...],
                    max_chunk_size=max_chunk_size,
                    soft_gate_temperature=temperature,
                    soft_gate_boundary_power=boundary_power,
                    execution_prune_eps=execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)
                _, gate_den = _global_dense_rw_den_sharded(
                    gate_mass, den_power, composition_mode)
                local_results = (raw_out / gate_den).astype(jnp.float32)
                weights = jnp.swapaxes(
                    block_weight, 0, 1)[None, ..., None]
                writeback = jax.lax.dynamic_slice_in_dim(
                    space_state_writeback,
                    space_start, bundle_size, axis=0)
                block_output = writeback_dot(
                    "ambr,mrd->abd",
                    local_results
                    * weights
                    * jnp.asarray(route_scale, dtype=jnp.float32),
                    writeback)[0].astype(jnp.float32)
                block_output = jnp.where(
                    block_valid[:, None], block_output, 0.0)
                return output_value.at[
                    block_token, :].add(block_output)

            return jax.lax.cond(
                block_bundle < n_bundles,
                execute_block,
                lambda output_value: output_value,
                local_output), None

        scan_step = jax.checkpoint(block_step, prevent_cse=False)
        local_update, _ = jax.lax.scan(
            scan_step, initial_output, jnp.arange(scan_blocks))
        update = _psum_dense_rw_representation_sharded(
            local_update, output_collective_bf16)

        n_per_space = int(read_vectors.shape[1]) * model_axis_size
        route_specs = (("rst", n_per_space),)
        metric_operands = (
            jax.lax.stop_gradient(flat_state),
            jax.lax.stop_gradient(space_state_proj),
            jax.lax.stop_gradient(tau_kernel),
            jax.lax.stop_gradient(tau_bias),
            jax.lax.stop_gradient(read_vectors),
            jax.tree.map(jax.lax.stop_gradient, routing),
            jax.lax.stop_gradient(temperature),
            jax.lax.stop_gradient(boundary_power),
            jax.lax.stop_gradient(execution_prune_eps),
        )

        def collect_metric_vector(operands):
            (metric_state,
             metric_projection,
             metric_tau_kernel,
             metric_tau_bias,
             metric_reads,
             metric_routing,
             metric_temperature,
             metric_boundary_power,
             metric_execution_prune_eps) = operands
            metric_token_blocks = math.ceil(
                token_count / token_block_size)
            metric_capacity = metric_token_blocks * token_block_size
            metric_state = jnp.pad(
                metric_state,
                ((0, metric_capacity - token_count), (0, 0)))
            initial_sums = jnp.zeros((1, 8), dtype=jnp.float32)

            def metric_block_step(operator_sums, block_index):
                bundle_id = block_index // metric_token_blocks
                token_block = block_index % metric_token_blocks
                token_start = token_block * token_block_size
                space_start = bundle_id * bundle_size
                state_block = jax.lax.dynamic_slice_in_dim(
                    metric_state, token_start,
                    token_block_size, axis=0)
                token_valid = (
                    jnp.arange(token_block_size) + token_start
                    < token_count)
                token_space_valid = jnp.broadcast_to(
                    token_valid[None, :],
                    (bundle_size, token_block_size))
                projection_block = jax.lax.dynamic_slice_in_dim(
                    metric_projection, space_start,
                    bundle_size, axis=0)
                local = projection_dot(
                    "bd,mdr->mbr", state_block, projection_block
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                raw_tau = (
                    _control_einsum_f32(
                        "mbr,ri->mbi", local,
                        metric_tau_kernel)
                    + metric_tau_bias)
                read_block = jax.lax.dynamic_slice_in_dim(
                    metric_reads, space_start,
                    bundle_size, axis=0)
                stats = _dense_rw_metric_stats_sharded(
                    local, local_norm, token_space_valid,
                    read_block[None, ...], raw_tau[None, ...],
                    max_chunk_size=max_chunk_size,
                    soft_gate_temperature=metric_temperature,
                    soft_gate_boundary_power=metric_boundary_power,
                    execution_prune_eps=metric_execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)
                block_sums = _dense_rw_metric_operator_sums_sharded(
                    stats, token_space_valid,
                    den_power, composition_mode)
                return operator_sums + block_sums, None

            operator_sums, _ = jax.lax.scan(
                metric_block_step,
                initial_sums,
                jnp.arange(n_bundles * metric_token_blocks))
            return _finish_blocked_dense_rw_metric_vector_sharded(
                operator_sums, route_specs, metric_routing)

        metric_vector = _metric_only_cond(
            collect_metric_vector,
            len(operator_metric_names),
            collect_metrics,
            metric_operands)
        metrics = {
            name: metric_vector[index]
            for index, name in enumerate(operator_metric_names)}
        metrics.update(packing_metrics)
        return update, metrics

    all_metric_names = (*operator_metric_names, *bundle_metric_names)
    kernel = shard_map(
        rst_core,
        mesh=mesh,
        in_specs=(
            P("data", None),
            P(), P(), P(), P(), P(), P(),
            P(None, "model", None), P(None, "model", None),
            P(), P(), P(), P(),
            P(),
        ),
        out_specs=(
            P("data", None),
            {name: P() for name in all_metric_names}),
        check_rep=False)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "rst_end_to_end_bundle4"
    kernel._v4174_execution_mode = "bundle_dense"
    kernel._v4174_bundle_size = bundle_size
    kernel._v4174_bundle_token_block_size = token_block_size
    kernel._v4174_packed_entry_capacity = (
        "2*T + n_bundles*(token_block_size-1)")
    kernel._v4174_bundle_packing = "prefix_count_exact"
    kernel._v4174_bundle_packing_vjp = "saved_metadata_scatter"
    kernel._v4174_rst_packing_count = 1
    kernel._v4174_dynamic_metric_flag = True
    kernel._v4174_chunk_remat_policy = "always"
    kernel._v4174_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4174_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4174_output_contract = ("rst[T,D]", "scalars")
    return kernel


def make_sharded_rst_space_bundle_dense_minimal(mesh, **kwargs):
    """Create the mixed-precision compact four-space RST executor."""
    return _make_sharded_rst_space_bundle_dense(
        mesh, throughput_bf16=True, output_collective_bf16=True, **kwargs)


def make_sharded_rst_space_bundle_dense_fp32_reference(mesh, **kwargs):
    """Create the compact FP32 RST executor for parity checks."""
    return _make_sharded_rst_space_bundle_dense(
        mesh, throughput_bf16=False, output_collective_bf16=False, **kwargs)


def _validate_v4174_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs):
    """Validate only the static profile used by v4174's internal executor."""
    del (
        admission_den_power, srw_composition_mode,
        heat_kernel_beta, kwargs)
    if not isinstance(sharded_fns, dict):
        return
    profile = sharded_fns.get("_v4174_kernel_profile")
    if profile is not None and profile not in (
            "production", "production_diagnostics",
            "retention", "suppression", "trajectory"):
        raise ValueError(f"unsupported v4174 kernel profile {profile!r}")
    attention = sharded_fns.get("attention_space_dense")
    rst = sharded_fns.get("rst_space_dense")
    if attention is None and rst is None:
        return
    if attention is None or rst is None:
        raise ValueError(
            "v4174 production requires both attention and RST executors")
    attention_mode = getattr(
        attention, "_v4174_execution_mode", "dense_all_space")
    rst_mode = getattr(rst, "_v4174_execution_mode", "dense_all_space")
    if attention_mode != rst_mode:
        raise ValueError(
            "v4174 attention/RST execution mode mismatch: "
            f"{attention_mode!r} != {rst_mode!r}")
    if attention_mode == "bundle_dense":
        for name, kernel in (("attention", attention), ("rst", rst)):
            if getattr(kernel, "_v4174_bundle_size", None) != BUNDLE_DENSE_SIZE:
                raise ValueError(
                    f"v4174 {name} bundle executor must use bundle size 4")
            if int(getattr(
                    kernel, "_v4174_bundle_token_block_size", 0)) <= 0:
                raise ValueError(
                    f"v4174 {name} bundle token block size must be positive")


def get_model_version() -> str:
    return MODEL_VERSION
