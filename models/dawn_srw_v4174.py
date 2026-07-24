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
attention residual update.  Semantic execution is hard top-k; physical
operator execution remains all-space dense and writeback is fused without an
``[M,T,D]`` intermediate.

This is the sole v4.1.7.4 architecture.  Earlier checkpoint, optimizer,
parameter-path, and config schemas are intentionally unsupported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
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
    d_model = _positive_int("d_model", model_cfg.get("d_model"))
    d_route = _positive_int("d_route", model_cfg.get("d_route"))
    if d_model != n_spaces * d_route:
        raise ValueError(
            "v4174 requires model.d_model == model.n_operation_spaces * "
            f"model.d_route, got {d_model} != {n_spaces} * {d_route}")
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


def _orthogonal_space_projection_init(
        key: jax.Array, shape: tuple[int, ...],
        dtype=jnp.float32) -> jax.Array:
    """Initialize ``[M,D,R]`` from row blocks of one orthogonal ``[D,D]``."""
    n_spaces, d_model, d_route = map(int, shape)
    if n_spaces * d_route != d_model:
        raise ValueError("orthogonal block initialization requires M * R == D")
    orthogonal = nn.initializers.orthogonal(scale=1.0)(
        key, (d_model, d_model), jnp.float32)
    blocks = orthogonal.T.reshape((n_spaces, d_route, d_model))
    return jnp.swapaxes(blocks, -1, -2).astype(dtype)


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
    value = float(_raw_tau_init_from_cosine_tau(
        min(max(float(cosine_tau), -0.9998), 0.9998)))

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
            "space_state_proj", _orthogonal_space_projection_init,
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
    route_state = jnp.asarray(flat_state, jnp.float32) @ jnp.asarray(
        space_route_kernel, jnp.float32)
    normalized_state = forward_unit_direction(route_state)
    normalized_reads = forward_unit_direction(space_read_vectors)
    space_scores = jnp.einsum(
        "tr,mr->tm", normalized_state, normalized_reads)
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
    """Retain dashboard-only QK aggregates without sharing execution."""
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
            "n_q": self.n_q,
            "n_k": self.n_k,
            "n_v": self.n_v,
            "n_rst": self.n_rst,
        }
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
            **analysis_kwargs):
        """Run the sole all-space-dense v4174 architecture."""
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
        bank_values = {
            route: (
                pool[f"{route}_read_vectors"],
                pool[f"{route}_write_vectors"])
            for route in ROUTES}
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
        regular_lists: dict[str, list[jax.Array]] = {}
        analysis_lists: dict[str, list[jax.Array]] = {}
        last_local_input = None
        last_route_state = None

        def append(target, values):
            for key, value in values.items():
                target.setdefault(key, []).append(value)

        def execute(route, local_states, raw_tau):
            read, write = bank_values[route]
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
                diagnostics=diagnostics_enabled)

        def route_and_local(flat_state):
            routing = _compute_space_routing(
                flat_state,
                router["space_route_proj"]["kernel"],
                router["space_read_vectors"],
                top_k)
            local_states = _project_space_local_states(
                flat_state, router["space_state_proj"])
            return routing, local_states

        for layer_index, layer in enumerate(self.layers):
            rng, rng_attn, rng_rst = jax.random.split(
                layer_rngs[layer_index], 3)
            normalized = layer.norm1(state)
            batch_size, sequence_length = normalized.shape[:2]
            flat_attention_state = normalized.reshape((-1, self.d_model))
            attention_routing, attention_local = route_and_local(
                flat_attention_state)
            attention_tau = {
                route: _linear(
                    router[f"{route}_operator_tau_proj"], attention_local)
                for route in ("q", "k", "v")}
            route_outputs = {}
            for route in ("q", "k", "v"):
                route_result = execute(
                    route, attention_local, attention_tau[route])
                if diagnostics_enabled:
                    local_output = route_result[0]
                    scalar, arrays = _aggregate_operator_diagnostics(
                        route_result, route_counts[route] // n_spaces,
                        route, composition_mode)
                    append(regular_lists, scalar)
                    if analysis:
                        append(analysis_lists, arrays)
                else:
                    local_output = route_result
                route_outputs[route] = _space_weighted_writeback(
                    local_output,
                    attention_routing["dense_space_weights"],
                    router["space_state_writeback"],
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
            attention_scores = jnp.einsum(
                "bhsd,bhtd->bhst", query, key
            ) / jnp.sqrt(jnp.float32(d_head))
            causal = jnp.tril(jnp.ones(
                (sequence_length, sequence_length), dtype=jnp.bool_))
            attention_scores = jnp.where(
                causal, attention_scores,
                jnp.finfo(attention_scores.dtype).min)
            attention_weights = jax.nn.softmax(
                attention_scores, axis=-1)
            attention_weights = _shared_safe_dropout(
                attention_weights, self.dropout_rate,
                deterministic, rng_attn)
            attention_output = jnp.einsum(
                "bhst,bhtd->bhsd", attention_weights, value)
            attention_output = attention_output.transpose(
                0, 2, 1, 3).reshape(
                    batch_size, sequence_length, self.d_model)
            attention_output = layer.attn.expand_O(attention_output)
            state = state + _shared_safe_dropout(
                attention_output, self.dropout_rate, deterministic, rng)

            rst_normalized = layer.norm2(state)
            flat_rst_state = rst_normalized.reshape((-1, self.d_model))
            rst_routing, rst_local = route_and_local(flat_rst_state)
            rst_tau = _linear(
                router["rst_operator_tau_proj"], rst_local)
            rst_result = execute("rst", rst_local, rst_tau)
            if diagnostics_enabled:
                rst_space_output = rst_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    rst_result, route_counts["rst"] // n_spaces,
                    "rst", composition_mode)
                append(regular_lists, scalar)
                if analysis:
                    append(analysis_lists, arrays)
            else:
                rst_space_output = rst_result
            rst_route_update = _space_weighted_writeback(
                rst_space_output,
                rst_routing["dense_space_weights"],
                router["space_state_writeback"],
                route_scales["rst"]).reshape(rst_normalized.shape)
            state = state + _shared_safe_dropout(
                rst_route_update, self.dropout_rate, deterministic, rng_rst)

            append(regular_lists, _space_routing_metrics(
                attention_routing, "attention"))
            append(regular_lists, _space_routing_metrics(
                rst_routing, "rst"))
            if diagnostics_enabled:
                append(regular_lists, {
                    "attn_out_norm": jnp.linalg.norm(
                        attention_output.astype(jnp.float32),
                        axis=-1).mean(),
                    "rst_out_norm": jnp.linalg.norm(
                        rst_route_update.astype(jnp.float32),
                        axis=-1).mean(),
                    "residual_norm": jnp.linalg.norm(
                        state.astype(jnp.float32), axis=-1).mean(),
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
                })
            last_local_input = flat_rst_state
            last_route_state = rst_routing["route_state"]

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
        if diagnostics_enabled:
            result.update(_canonical_regular_operator_metrics(result))
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
            f"R={self.d_route}; D=M*R, top_k={self.operation_space_top_k}",
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
            "execution: semantic hard top-k, physical all-space dense, one "
            "shared P_m/U_m coordinate system and fused writeback",
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


def get_model_version() -> str:
    return MODEL_VERSION
