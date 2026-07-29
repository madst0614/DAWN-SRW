"""DAWN-SRW v4.1.7.5: explicit operation-space interfaces.

A token-level global representation produces independent Q, K, V, and RST
space queries.  Every route matches its query against the same learned
operation-space keys and independently selects from one shared atlas of
operation spaces.  A space-owned read projection maps the global
representation into that space's private coordinates; the corresponding
space-owned write projection returns local outputs to the global
representation.  Q, K, V, and RST apply independent RW operator banks in those
shared space coordinates.

Q/K/V independently select attention spaces from the same pre-attention state.
RST recomputes its own selection after the attention residual.  Every route
owns a distinct query projection and tau map for every operation space, so
selection remains route-specific while a space's address and coordinate system
remain intrinsic to that shared space.
Selected-space weights are L1-normalized across the hard top-k support; if
their total ReLU-squared mass is at or below ``SPACE_GATE_L1_EPS``, the
highest-ranked space receives a deterministic one-hot fallback.  Absolute
update scale therefore belongs to the within-space RW composition and
residual/output scales, not the selector.
The final attention ``expand_O`` remains the global interface that combines
attention heads before the residual update.

This is the sole v4.1.7.5 architecture.  Earlier checkpoint, optimizer, and
parameter-path schemas are intentionally unsupported.
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


MODEL_VERSION = "spatial-r1-v4.1.7.5"
ROUTES = ("q", "k", "v", "rst")
ROUTE_INDEX = {route: index for index, route in enumerate(ROUTES)}
POOLS = ROUTES
ATTENTION_CORE_NAME = "causal_dot_product_fp32_softmax"
LAYER_EXECUTION_NAME = "rematerialized"
OPERATION_SPACE_EXECUTION_MODES = ("dense_all_space",)
BUNDLE_DENSE_SIZE = 4
DEFAULT_BUNDLE_TOKEN_BLOCK_SIZE = 1024
SPACE_GATE_L1_EPS = 1.0e-8
_BUNDLE_PACKING_CHECKPOINT_NAME = "v4175_bundle_packing_metadata"
_SPACE_ROUTING_COMPACT_METRIC_SUFFIXES = (
    "gate_mass_mean",
    "active_count_mean",
    "exact_zero_mass_frac",
    "fallback_frac",
    "zero_gate_frac",
    "top1_rate",
)


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
    """Validate the fresh-run v4175 schema without compatibility conversion."""
    forbidden = tuple(
        name for name in (
            "n_qk", "n_know", "operator_key_mode", "operator_query_mode",
            "space_kernel_beta_qk", "space_kernel_beta_v",
            "space_kernel_beta_rst", "router_dropout", "tau_init_attn_qk")
        if name in model_cfg)
    if forbidden:
        raise ValueError(
            "v4175 does not accept removed model fields: "
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
                "v4175 bundle_dense supports "
                f"model.operation_space_bundle_size={BUNDLE_DENSE_SIZE} "
                f"only, got {bundle_size}")
        if n_spaces != 24:
            raise ValueError(
                "v4175 bundle_dense production requires "
                f"model.n_operation_spaces=24, got {n_spaces}")
        if n_spaces % bundle_size:
            raise ValueError(
                "model.n_operation_spaces must be divisible by "
                "model.operation_space_bundle_size, got "
                f"{n_spaces} % {bundle_size}")
        if top_k != 2:
            raise ValueError(
                "v4175 bundle_dense requires "
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
            "v4175 requires the local route dimension to be no larger than "
            f"the model dimension, got d_route={d_route} > d_model={d_model}")
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
    """Count the exact canonical v4175 parameter tree."""
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
    operation_space_interface = 2 * n_spaces * d_model * d_route
    space_addressing = (
        len(ROUTES) * d_model * d_route
        + n_spaces * d_route)
    space_operator_control = 4 * n_spaces * (d_route + 1)
    operator_bank = 2 * d_route * pool_total
    counts = {
        "token_embedding": vocab * d_model,
        "position_embedding": max_seq * d_model,
        "layer_stack": n_layers * (d_model * d_model + 4 * d_model),
        "operation_space_interface": operation_space_interface,
        "space_addressing": space_addressing,
        "space_operator_control": space_operator_control,
        "operator_bank": operator_bank,
        "learned_key_tables": 0,
        "bilinear_probe_matrices": 0,
        "operator_query_projections": 0,
        "final_norm": 2 * d_model,
    }
    counts["operation_space_system"] = (
        operation_space_interface
        + space_addressing
        + space_operator_control)
    counts["total"] = (
        counts["token_embedding"]
        + counts["position_embedding"]
        + counts["layer_stack"]
        + counts["operation_space_system"]
        + operator_bank
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
    space_keys = jax.random.split(key, n_spaces)
    orthogonal = nn.initializers.orthogonal(scale=1.0)
    projections = jax.vmap(
        lambda space_key: orthogonal(
            space_key, (d_model, d_route), jnp.float32)
    )(space_keys)
    return projections.astype(dtype)


def _unit_operation_space_key_init(
        key: jax.Array, shape: tuple[int, ...],
        dtype=jnp.float32) -> jax.Array:
    """Initialize shared learned unit keys without an ``M <= R`` limit."""
    values = jax.random.normal(key, shape, dtype=jnp.float32)
    return _shared_forward_unit_direction(values).astype(dtype)


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


class OperationSpaceOperatorBank(nn.Module):
    """Fully separate, space-indexed Q/K/V/RST RW operator banks."""
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


class OperationSpaceSelector(nn.Module):
    """Own route queries and shared address keys for one operation-space atlas."""
    d_model: int
    d_route: int
    n_operation_spaces: int

    def setup(self):
        n_spaces = int(self.n_operation_spaces)
        d_model = int(self.d_model)
        d_route = int(self.d_route)
        self.route_query_proj = self.param(
            "route_query_proj", _independent_space_projection_init,
            (len(ROUTES), d_model, d_route))
        self.operation_space_keys = self.param(
            "operation_space_keys", _unit_operation_space_key_init,
            (n_spaces, d_route))


def _route_selector_parameters(
        space_selector_params: Mapping[str, jax.Array],
        route: str) -> tuple[jax.Array, jax.Array]:
    """Return one route's query projection and the atlas' shared address keys."""
    try:
        route_index = ROUTE_INDEX[str(route)]
    except KeyError as exc:
        raise ValueError(f"unsupported operation-space route {route!r}") from exc
    return (
        space_selector_params["route_query_proj"][route_index],
        space_selector_params["operation_space_keys"],
    )


class OperationSpaceInterface(nn.Module):
    """Read global representations into spaces and write outputs back."""
    d_model: int
    d_route: int
    n_operation_spaces: int

    def setup(self):
        n_spaces = int(self.n_operation_spaces)
        d_model = int(self.d_model)
        d_route = int(self.d_route)
        self.space_read_proj = self.param(
            "space_read_proj", _independent_space_projection_init,
            (n_spaces, d_model, d_route))
        self.space_write_proj = self.param(
            "space_write_proj",
            lambda key, shape, dtype=jnp.float32: jnp.swapaxes(
                self.space_read_proj, -1, -2).astype(dtype),
            (n_spaces, d_route, d_model))


class OperationGateController(nn.Module):
    """Own one coordinate-sensitive tau map per route and operation space."""
    d_route: int
    n_operation_spaces: int
    tau_init_attn_q: float
    tau_init_attn_k: float
    tau_init_attn_v: float
    tau_init_rst: float

    def setup(self):
        n_spaces = int(self.n_operation_spaces)
        d_route = int(self.d_route)
        tau_values = {
            "q": self.tau_init_attn_q,
            "k": self.tau_init_attn_k,
            "v": self.tau_init_attn_v,
            "rst": self.tau_init_rst,
        }
        for route in ROUTES:
            setattr(self, f"{route}_tau_kernel", self.param(
                f"{route}_tau_kernel",
                nn.initializers.zeros,
                (n_spaces, d_route, 1)))
            setattr(self, f"{route}_tau_bias", self.param(
                f"{route}_tau_bias",
                _constant_tau_bias(tau_values[route]),
                (n_spaces, 1)))


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


def _read_operation_space_states(
        global_state: jax.Array, space_read_proj: jax.Array) -> jax.Array:
    """Read a global ``[T,D]`` representation into ``[M,T,R]`` spaces."""
    if global_state.ndim != 2 or space_read_proj.ndim != 3:
        raise ValueError(
            "global_state/space_read_proj must have shapes [T,D] and [M,D,R]")
    return jnp.einsum("td,mdr->mtr", global_state, space_read_proj)


def _select_operation_spaces(
        global_state: jax.Array,
        space_query_kernel: jax.Array,
        operation_space_keys: jax.Array,
        operation_space_top_k: int) -> dict[str, jax.Array]:
    """Direct cosine routing with hard top-k ReLU-squared non-softmax gates."""
    space_query = control_dot_f32(
        global_state, space_query_kernel,
        dimension_numbers=(((1,), (0,)), ((), ())))
    normalized_query = forward_unit_direction(space_query)
    normalized_keys = forward_unit_direction(operation_space_keys)
    space_scores = control_dot_f32(
        normalized_query, normalized_keys,
        dimension_numbers=(((1,), (1,)), ((), ())))
    gate_values = _space_gate_from_scores(
        space_scores, operation_space_top_k)
    return {
        "space_query": space_query,
        "space_scores": space_scores,
        **gate_values,
    }


def _apply_operation_space_tau_map(
        space_states: jax.Array,
        tau_kernel: jax.Array,
        tau_bias: jax.Array) -> jax.Array:
    """Apply independent ``[R,1]`` tau maps in every operation space."""
    if (space_states.ndim != 3
            or tau_kernel.ndim != 3
            or tau_bias.ndim != 2):
        raise ValueError(
            "space tau map requires [M,T,R], [M,R,1], and [M,1]")
    if (space_states.shape[0] != tau_kernel.shape[0]
            or space_states.shape[0] != tau_bias.shape[0]
            or space_states.shape[-1] != tau_kernel.shape[1]
            or tau_kernel.shape[-1] != 1
            or tau_bias.shape[-1] != 1):
        raise ValueError(
            "operation-space tau map shapes are inconsistent: "
            f"states={space_states.shape}, kernel={tau_kernel.shape}, "
            f"bias={tau_bias.shape}")
    return (
        _control_einsum_f32("mtr,mro->mto", space_states, tau_kernel)
        + tau_bias[:, None, :])


def _select_and_read_operation_spaces(
        global_state: jax.Array,
        space_query_kernel: jax.Array,
        operation_space_keys: jax.Array,
        space_read_proj: jax.Array,
        operation_space_top_k: int
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Select operation spaces, then read their states as a separate step."""
    space_routing = _select_operation_spaces(
        global_state,
        space_query_kernel,
        operation_space_keys,
        operation_space_top_k)
    space_states = _read_operation_space_states(
        global_state, space_read_proj)
    return space_routing, space_states


def _space_gate_from_scores(
        space_scores: jax.Array,
        operation_space_top_k: int) -> dict[str, jax.Array]:
    """L1-normalize hard-top-k ReLU-squared weights across selected spaces."""
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
    fallback_mask = (
        space_gate_mass <= jnp.float32(SPACE_GATE_L1_EPS))
    has_positive_mass = jnp.logical_not(fallback_mask)
    space_gate_den = jnp.where(
        has_positive_mass, space_gate_mass, jnp.float32(1.0))
    normalized_space_weights = space_gate / space_gate_den
    top1_fallback = jax.nn.one_hot(
        selected_ids[..., 0],
        int(space_scores.shape[-1]),
        dtype=jnp.float32)
    dense_space_weights = jnp.where(
        has_positive_mass, normalized_space_weights, top1_fallback)
    return {
        "selected_scores": selected_scores,
        "selected_ids": selected_ids,
        "space_gate": space_gate,
        "space_gate_mass": space_gate_mass,
        "space_gate_den": space_gate_den,
        "fallback_mask": fallback_mask,
        "dense_space_weights": dense_space_weights,
    }


def _direct_read_match(
        space_states: jax.Array,
        read_vectors: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reuse one read matmul for the scalar read and direct cosine score."""
    read_unit = forward_unit_direction(
        read_vectors.astype(jnp.float32))
    read_value = jnp.einsum(
        "mtr,mnr->mtn",
        space_states.astype(jnp.bfloat16),
        read_unit.astype(jnp.bfloat16),
    ).astype(jnp.float32)
    local_norm = jnp.linalg.norm(
        space_states.astype(jnp.float32),
        axis=-1, keepdims=True)
    rho = read_value / jnp.maximum(
        local_norm, jnp.float32(RW_FORWARD_NORM_EPS))
    return read_value, jnp.clip(rho, -1.0, 1.0), local_norm


def _rw_compose_space_dense(
        space_states: jax.Array,
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
    if space_states.ndim != 3:
        raise ValueError("space_states must have shape [M,T,R]")
    if read_vectors.ndim != 3 or write_vectors.ndim != 3:
        raise ValueError("read/write vectors must have shape [M,N,R]")
    n_spaces, token_count, d_route = map(int, space_states.shape)
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
    local_bf16 = space_states.astype(jnp.bfloat16)
    local_norm = jnp.linalg.norm(
        space_states.astype(jnp.float32),
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


def _write_space_outputs_to_global(
        space_output: jax.Array,
        space_weights: jax.Array,
        space_write_proj: jax.Array,
        output_scale: jax.Array | float) -> jax.Array:
    """Write weighted operation-space outputs to the global representation."""
    weighted_space_output = (
        space_output
        * jnp.swapaxes(space_weights, 0, 1)[..., None]
        * jnp.asarray(output_scale))
    return jnp.einsum(
        "mtr,mrd->td", weighted_space_output, space_write_proj)


def _space_routing_metrics(
        routing: Mapping[str, jax.Array],
        prefix: str) -> dict[str, jax.Array]:
    gate = routing["space_gate"].astype(jnp.float32)
    weights = routing["dense_space_weights"].astype(jnp.float32)
    mass = routing["space_gate_mass"][..., 0].astype(jnp.float32)
    den = routing["space_gate_den"][..., 0].astype(jnp.float32)
    fallback = routing["fallback_mask"][..., 0]
    selected_ids = routing["selected_ids"]
    n_spaces = int(gate.shape[-1])
    active = weights > 0.0
    top1_usage = jax.nn.one_hot(
        selected_ids[..., 0], n_spaces, dtype=jnp.float32
    ).reshape((-1, n_spaces)).mean(axis=0)
    usage = active.astype(jnp.float32).reshape((-1, n_spaces)).mean(axis=0)
    values = {
        f"{prefix}_space_gate_mass_mean": mass.mean(),
        f"{prefix}_space_gate_den_mean": den.mean(),
        f"{prefix}_space_active_count_mean": (
            active.astype(jnp.float32).sum(axis=-1).mean()),
        f"{prefix}_space_exact_zero_mass_frac": (
            mass <= 0.0).astype(jnp.float32).mean(),
        f"{prefix}_space_fallback_frac": (
            fallback.astype(jnp.float32).mean()),
        # Temporary compatibility alias for existing log consumers.
        f"{prefix}_space_zero_gate_frac": (
            mass <= 0.0).astype(jnp.float32).mean(),
        f"{prefix}_space_top1_rate": top1_usage.max(),
        f"{prefix}_space_usage_min": usage.min(),
        f"{prefix}_space_usage_max": usage.max(),
        f"{prefix}_space_usage_std": usage.std(),
    }
    diagnostic_probability = weights
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
    weights = routing["dense_space_weights"].astype(jnp.float32)
    mass = routing["space_gate_mass"][..., 0].astype(jnp.float32)
    fallback = routing["fallback_mask"][..., 0]
    selected_ids = routing["selected_ids"]
    n_spaces = int(gate.shape[-1])
    top1_usage = jax.nn.one_hot(
        selected_ids[..., 0], n_spaces, dtype=jnp.float32
    ).reshape((-1, n_spaces)).mean(axis=0)
    values = {
        f"{prefix}_space_gate_mass_mean": mass.mean(),
        f"{prefix}_space_active_count_mean": (
            (weights > 0.0).astype(jnp.float32).sum(axis=-1).mean()),
        f"{prefix}_space_exact_zero_mass_frac": (
            mass <= 0.0).astype(jnp.float32).mean(),
        f"{prefix}_space_fallback_frac": (
            fallback.astype(jnp.float32).mean()),
        # Temporary compatibility alias for existing log consumers.
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
            metrics["q_global_interface_norm"]
            + metrics["k_global_interface_norm"]),
        "attn_v_pool_scaled_srw_out_norm": metrics[
            "v_global_interface_norm"],
        "rst_pool_scaled_srw_out_norm": metrics["rst_global_update_norm"],
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


def _effective_rank_from_singular_values(
        singular_values: jax.Array) -> jax.Array:
    probability = singular_values / jnp.maximum(
        singular_values.sum(axis=-1, keepdims=True), 1.0e-8)
    return jnp.exp(-jnp.sum(
        probability * jnp.log(jnp.maximum(probability, 1.0e-8)),
        axis=-1))


def _operation_space_geometry_diagnostics(
        space_read_proj: jax.Array,
        space_write_proj: jax.Array,
        global_state: Optional[jax.Array] = None) -> dict[str, jax.Array]:
    """Measure read geometry, read/write round trips, and cross-space links."""
    read_projection = jnp.asarray(space_read_proj, dtype=jnp.float32)
    write_projection = jnp.asarray(space_write_proj, dtype=jnp.float32)
    stacked = jnp.swapaxes(read_projection, -1, -2).reshape(
        (-1, read_projection.shape[1]))
    singular = jnp.linalg.svd(stacked, compute_uv=False)
    basis_overlap = jnp.einsum(
        "mdr,nds->mnrs", read_projection, read_projection)
    principal_cosines = jax.vmap(lambda row: jax.vmap(
        lambda cross: jnp.linalg.svd(cross, compute_uv=False))(row))(
            basis_overlap)
    principal_angles = jnp.arccos(jnp.clip(principal_cosines, -1.0, 1.0))
    frobenius = jnp.einsum(
        "mdr,ndr->mn", read_projection, read_projection)
    diagonal = jnp.sqrt(jnp.maximum(jnp.diag(frobenius), 1.0e-8))
    overlap = frobenius / (diagonal[:, None] * diagonal[None, :])

    d_route = int(read_projection.shape[-1])
    identity = jnp.eye(d_route, dtype=jnp.float32)
    space_roundtrip = jnp.einsum(
        "mrd,mds->mrs", write_projection, read_projection)
    roundtrip_singular = jnp.linalg.svd(
        space_roundtrip, compute_uv=False)
    cross_space_transfer = jnp.einsum(
        "mrd,nds->mnrs", write_projection, read_projection)
    cross_space_principal_values = jnp.linalg.svd(
        cross_space_transfer, compute_uv=False)
    out = {
        "stacked_space_read_projection_singular_values": singular,
        "stacked_space_read_projection_rank": jnp.linalg.matrix_rank(stacked),
        "stacked_space_read_projection_effective_rank":
            _effective_rank_from_singular_values(singular),
        "space_read_projection_overlap": _off_diagonal_entries(overlap),
        "space_read_subspace_principal_angles": _off_diagonal_entries(
            principal_angles),
        "space_read_write_transpose_deviation": jnp.linalg.norm(
            jnp.swapaxes(read_projection, -1, -2) - write_projection,
            axis=(-2, -1)) / jnp.sqrt(jnp.float32(
                write_projection.shape[-2] * write_projection.shape[-1])),
        "space_roundtrip_identity_error": jnp.linalg.norm(
            space_roundtrip - identity[None, ...],
            axis=(-2, -1)) / jnp.sqrt(jnp.float32(d_route)),
        "space_roundtrip_singular_values": roundtrip_singular,
        "space_roundtrip_effective_rank":
            _effective_rank_from_singular_values(roundtrip_singular),
        "cross_space_transfer_norm": _off_diagonal_entries(
            jnp.linalg.norm(cross_space_transfer, axis=(-2, -1))),
        "cross_space_transfer_effective_rank": _off_diagonal_entries(
            _effective_rank_from_singular_values(
                cross_space_principal_values)),
        "cross_space_principal_values": _off_diagonal_entries(
            cross_space_principal_values),
    }
    if global_state is not None:
        space_states = _read_operation_space_states(
            global_state, read_projection)
        covariance = jnp.einsum(
            "mtr,nts->mnrs", space_states, space_states
        ) / jnp.maximum(global_state.shape[0], 1)
        covariance_norm = jnp.linalg.norm(covariance, axis=(-2, -1))
        covariance_diag = jnp.sqrt(jnp.maximum(
            jnp.diag(covariance_norm), 1.0e-8))
        covariance_overlap = covariance_norm / (
            covariance_diag[:, None] * covariance_diag[None, :])
        variance = jnp.var(space_states, axis=1).sum(axis=-1)
        out.update({
            "operation_space_state_covariance_overlap":
                _off_diagonal_entries(covariance_overlap),
            "operation_space_state_norm": jnp.linalg.norm(
                space_states, axis=-1).mean(axis=-1),
            "operation_space_explained_variance":
                variance / jnp.maximum(variance.sum(), 1.0e-8),
        })
    return out


def _operator_bank_geometry_diagnostics(
        operator_bank_params: Mapping[str, jax.Array],
        operation_space_keys: jax.Array,
        space_query: jax.Array) -> dict[str, jax.Array]:
    """Address-key geometry and direct-read geometry for all RW banks."""
    normalized_keys = forward_unit_direction(operation_space_keys)
    pairwise = normalized_keys @ normalized_keys.T
    pairwise_values = _off_diagonal_entries(pairwise)
    query_norm = jnp.linalg.norm(
        space_query.astype(jnp.float32), axis=-1).mean(axis=-1)
    out = {
        "operation_space_key_pairwise_cosine": pairwise_values,
        "operation_space_key_norm": jnp.linalg.norm(
            operation_space_keys.astype(jnp.float32), axis=-1),
        "space_query_norm": query_norm,
    }
    for pool in POOLS:
        read = forward_unit_direction(
            operator_bank_params[f"{pool}_read_vectors"])
        write = forward_unit_direction(
            operator_bank_params[f"{pool}_write_vectors"])
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
                operator_bank_params[
                    f"{pool}_read_vectors"].astype(jnp.float32),
                axis=-1).mean(axis=-1),
            f"{pool}_write_norm": jnp.linalg.norm(
                operator_bank_params[
                    f"{pool}_write_vectors"].astype(jnp.float32),
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


class DAWN_SRW_V4175(nn.Module):
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
                "v4175 requires explicit q/k/v/rst initial operator tau values")
        _, embedding_vocab = self._vocab_sizes()
        self.token_emb = nn.Embed(
            embedding_vocab, self.d_model,
            embedding_init=_shared_scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model,
            embedding_init=_shared_scaled_normal(0.02))
        self.operator_bank = OperationSpaceOperatorBank(
            n_q=self.n_q, n_k=self.n_k, n_v=self.n_v, n_rst=self.n_rst,
            d_route=self.d_route,
            n_operation_spaces=self.n_operation_spaces)
        self.space_selector = OperationSpaceSelector(
            d_model=self.d_model, d_route=self.d_route,
            n_operation_spaces=self.n_operation_spaces)
        self.space_interface = OperationSpaceInterface(
            d_model=self.d_model, d_route=self.d_route,
            n_operation_spaces=self.n_operation_spaces)
        self.operator_controller = OperationGateController(
            d_route=self.d_route,
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
        _ = self.operator_bank.q_read_vectors
        _ = self.operator_bank.k_read_vectors
        _ = self.operator_bank.v_read_vectors
        _ = self.operator_bank.rst_read_vectors
        _ = self.space_selector.route_query_proj
        local = jnp.einsum(
            "bsd,mdr->mbsr", state, self.space_interface.space_read_proj)
        for route in ROUTES:
            _ = getattr(self.operator_controller, f"{route}_tau_kernel")
            _ = getattr(self.operator_controller, f"{route}_tau_bias")
        _ = local
        _ = self.space_selector.operation_space_keys
        _ = self.space_interface.space_write_proj
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
        """Run v4175 with the statically selected physical executor."""
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
        operator_bank = params["operator_bank"]
        space_selector = params["space_selector"]
        space_interface = params["space_interface"]
        operator_controller = params["operator_controller"]
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
        if attention_dense is not None:
            raise ValueError(
                "v4175 reference execution requires route-specific Q/K/V "
                "space kernels; fused shared-routing attention is unsupported")
        fused_production = False
        if fused_production and rst_dense is None:
            raise ValueError(
                "v4175 production requires both attention_space_dense and "
                "rst_space_dense executors")
        fused_throughput_precision = (
            getattr(
                attention_dense, "_v4175_throughput_precision", None)
            if fused_production else None)
        if fused_production and fused_throughput_precision not in (
                "bf16_operands_f32_accum", "fp32_reference"):
            raise ValueError(
                "v4175 fused attention executor is missing its static "
                "throughput precision contract")
        if (fused_production
                and getattr(
                    rst_dense, "_v4175_throughput_precision", None)
                != fused_throughput_precision):
            raise ValueError(
                "v4175 attention/RST executors require identical static "
                "throughput precision")
        fused_execution_mode = (
            getattr(
                attention_dense, "_v4175_execution_mode",
                "dense_all_space")
            if fused_production else None)
        if (fused_production
                and getattr(
                    rst_dense, "_v4175_execution_mode",
                    "dense_all_space")
                != fused_execution_mode):
            raise ValueError(
                "v4175 attention/RST executors require identical static "
                "operation-space execution modes")
        if (fused_production
                and fused_execution_mode
                != self.operation_space_execution_mode):
            raise ValueError(
                "v4175 configured/factory operation-space execution mode "
                f"mismatch: {self.operation_space_execution_mode!r} != "
                f"{fused_execution_mode!r}")
        fused_throughput_bf16 = (
            fused_throughput_precision == "bf16_operands_f32_accum")
        regular_lists: dict[str, list[jax.Array]] = {}
        analysis_lists: dict[str, list[jax.Array]] = {}
        last_global_state_for_space_diagnostics = None
        last_space_query = None
        scanned_regular_metrics: dict[str, jax.Array] = {}

        def append(target, values):
            for key, value in values.items():
                target.setdefault(key, []).append(value)

        def execute(route, space_states, raw_tau, operator_bank_params):
            read = operator_bank_params[f"{route}_read_vectors"]
            write = operator_bank_params[f"{route}_write_vectors"]
            direct_kernel = (
                sharded_fns.get(f"{route}_space_dense")
                if isinstance(sharded_fns, dict) else None)
            if direct_kernel is not None:
                token_valid = jnp.ones(
                    space_states.shape[:2], dtype=jnp.bool_)
                return direct_kernel(
                    space_states, raw_tau, token_valid, read, write,
                    temperatures[route], temperatures[route],
                    soft_gate_boundary_power, soft_gate_boundary_power,
                    execution_prune_eps)
            return _rw_compose_space_dense(
                space_states, read, write, raw_tau,
                soft_gate_temperature=temperatures[route],
                soft_gate_boundary_power=soft_gate_boundary_power,
                admission_den_power=den_powers[route],
                srw_composition_mode=composition_mode,
                heat_kernel_beta=self.heat_kernel_beta,
                execution_prune_eps=execution_prune_eps,
                max_chunk_size=route_chunks[route],
                diagnostics=True)

        def layer_forward(
                global_state, block_params, operator_bank_params,
                space_selector_params, space_interface_params,
                operator_controller_params, layer_rng):
            """One complete checkpointable Q/K/V-attention-RST layer."""
            regular_metrics: dict[str, jax.Array] = {}
            analysis_metrics: dict[str, jax.Array] = {}
            rng, rng_attn, rng_rst = jax.random.split(
                layer_rng, 3)
            attn_global_state = _shared_layer_norm(
                global_state,
                block_params["norm1"]["scale"],
                block_params["norm1"]["bias"])
            batch_size, sequence_length = attn_global_state.shape[:2]
            attn_global_state_flat = attn_global_state.reshape(
                (-1, self.d_model))
            attention_space_routing = {}
            if fused_production:
                (q_global_interface_flat,
                 k_global_interface_flat,
                 v_global_interface_flat,
                 attention_metrics) = (
                    attention_dense(
                        attn_global_state_flat,
                        space_selector_params[
                            "space_query_proj"]["kernel"],
                        space_selector_params["operation_space_keys"],
                        space_interface_params["space_read_proj"],
                        space_interface_params["space_write_proj"],
                        operator_controller_params["q_tau_kernel"],
                        operator_controller_params["q_tau_bias"],
                        operator_controller_params["k_tau_kernel"],
                        operator_controller_params["k_tau_bias"],
                        operator_controller_params["v_tau_kernel"],
                        operator_controller_params["v_tau_bias"],
                        operator_bank_params["q_read_vectors"],
                        operator_bank_params["q_write_vectors"],
                        operator_bank_params["k_read_vectors"],
                        operator_bank_params["k_write_vectors"],
                        operator_bank_params["v_read_vectors"],
                        operator_bank_params["v_write_vectors"],
                        temperatures["q"], temperatures["v"],
                        soft_gate_boundary_power, execution_prune_eps,
                        route_scales["q"], route_scales["v"],
                        collect_regular_metrics))
                global_interfaces = {
                    "q": q_global_interface_flat.reshape(
                        attn_global_state.shape),
                    "k": k_global_interface_flat.reshape(
                        attn_global_state.shape),
                    "v": v_global_interface_flat.reshape(
                        attn_global_state.shape),
                }
                regular_metrics.update(attention_metrics)
            else:
                attention_space_states = _read_operation_space_states(
                    attn_global_state_flat,
                    space_interface_params["space_read_proj"])
                attention_tau = {
                    route: _apply_operation_space_tau_map(
                        attention_space_states,
                        operator_controller_params[
                            f"{route}_tau_kernel"],
                        operator_controller_params[
                            f"{route}_tau_bias"])
                    for route in ("q", "k", "v")}
                global_interfaces = {}
                for route in ("q", "k", "v"):
                    route_query_proj, operation_space_keys = (
                        _route_selector_parameters(
                            space_selector_params, route))
                    route_routing = _select_operation_spaces(
                        attn_global_state_flat,
                        route_query_proj,
                        operation_space_keys,
                        top_k)
                    attention_space_routing[route] = route_routing
                    route_result = execute(
                        route, attention_space_states, attention_tau[route],
                        operator_bank_params)
                    if isinstance(route_result, tuple):
                        space_output = route_result[0]
                        scalar, arrays = _aggregate_operator_diagnostics(
                            route_result, route_counts[route] // n_spaces,
                            route, composition_mode)
                        regular_metrics.update(scalar)
                        if analysis:
                            analysis_metrics.update(arrays)
                    else:
                        space_output = route_result
                    global_interfaces[route] = (
                        _write_space_outputs_to_global(
                            space_output,
                            route_routing[
                                "dense_space_weights"],
                            space_interface_params["space_write_proj"],
                            route_scales[route])
                        .reshape(attn_global_state.shape))

            d_head = self.d_model // self.n_heads
            q_global_interface = global_interfaces["q"]
            k_global_interface = global_interfaces["k"]
            v_global_interface = global_interfaces["v"]
            query = q_global_interface.reshape(
                batch_size, sequence_length, self.n_heads, d_head
            ).transpose(0, 2, 1, 3)
            key = k_global_interface.reshape(
                batch_size, sequence_length, self.n_heads, d_head
            ).transpose(0, 2, 1, 3)
            value = v_global_interface.reshape(
                batch_size, sequence_length, self.n_heads, d_head
            ).transpose(0, 2, 1, 3)
            attention_global_update = _causal_attention_core(
                query, key, value, self.dropout_rate,
                deterministic, rng_attn,
                throughput_bf16=fused_throughput_bf16)
            attention_global_update = attention_global_update.transpose(
                0, 2, 1, 3).reshape(
                    batch_size, sequence_length, self.d_model)
            attention_global_update = (
                _throughput_linear_bf16_f32(
                    block_params["attn"]["expand_O"],
                    attention_global_update)
                if fused_throughput_bf16 else _linear(
                    block_params["attn"]["expand_O"],
                    attention_global_update))
            global_state = global_state + _shared_safe_dropout(
                attention_global_update,
                self.dropout_rate, deterministic, rng)

            rst_global_state = _shared_layer_norm(
                global_state,
                block_params["norm2"]["scale"],
                block_params["norm2"]["bias"])
            rst_global_state_flat = rst_global_state.reshape(
                (-1, self.d_model))
            rst_space_routing = None
            if fused_production:
                rst_global_update_flat, rst_metrics = rst_dense(
                    rst_global_state_flat,
                    space_selector_params[
                        "space_query_proj"]["kernel"],
                    space_selector_params["operation_space_keys"],
                    space_interface_params["space_read_proj"],
                    space_interface_params["space_write_proj"],
                    operator_controller_params["rst_tau_kernel"],
                    operator_controller_params["rst_tau_bias"],
                    operator_bank_params["rst_read_vectors"],
                    operator_bank_params["rst_write_vectors"],
                    temperatures["rst"], soft_gate_boundary_power,
                    execution_prune_eps, route_scales["rst"],
                    collect_regular_metrics)
                rst_global_update = rst_global_update_flat.reshape(
                    rst_global_state.shape)
                regular_metrics.update(rst_metrics)
            else:
                (rst_query_proj,
                 operation_space_keys) = _route_selector_parameters(
                     space_selector_params, "rst")
                (rst_space_routing,
                 rst_space_states) = _select_and_read_operation_spaces(
                    rst_global_state_flat,
                    rst_query_proj,
                    operation_space_keys,
                    space_interface_params["space_read_proj"],
                    top_k)
                rst_tau = _apply_operation_space_tau_map(
                    rst_space_states,
                    operator_controller_params["rst_tau_kernel"],
                    operator_controller_params["rst_tau_bias"])
                rst_result = execute(
                    "rst", rst_space_states, rst_tau,
                    operator_bank_params)
                if isinstance(rst_result, tuple):
                    rst_space_update = rst_result[0]
                    scalar, arrays = _aggregate_operator_diagnostics(
                        rst_result, route_counts["rst"] // n_spaces,
                        "rst", composition_mode)
                    regular_metrics.update(scalar)
                    if analysis:
                        analysis_metrics.update(arrays)
                else:
                    rst_space_update = rst_result
                rst_global_update = _write_space_outputs_to_global(
                    rst_space_update,
                    rst_space_routing["dense_space_weights"],
                    space_interface_params["space_write_proj"],
                    route_scales["rst"]).reshape(rst_global_state.shape)
            global_state = global_state + _shared_safe_dropout(
                rst_global_update,
                self.dropout_rate, deterministic, rng_rst)

            if not fused_production:
                routing_metric_fn = (
                    _space_routing_metrics
                    if diagnostics_enabled
                    else _compact_space_routing_metrics)
                for route in ("q", "k", "v"):
                    regular_metrics.update(routing_metric_fn(
                        attention_space_routing[route], route))
                regular_metrics.update(routing_metric_fn(
                    rst_space_routing, "rst"))
            norm_metric_names = (
                "attn_out_norm",
                "rst_out_norm",
                "residual_norm",
                "q_global_interface_norm",
                "k_global_interface_norm",
                "v_global_interface_norm",
                "rst_global_update_norm",
                "q_route_output_norm",
                "k_route_output_norm",
                "v_route_output_norm",
                "rst_route_update_norm",
            )

            def compute_norm_metrics(_):
                q_global_interface_norm = jnp.linalg.norm(
                    q_global_interface.astype(jnp.float32),
                    axis=-1).mean()
                k_global_interface_norm = jnp.linalg.norm(
                    k_global_interface.astype(jnp.float32),
                    axis=-1).mean()
                v_global_interface_norm = jnp.linalg.norm(
                    v_global_interface.astype(jnp.float32),
                    axis=-1).mean()
                rst_global_update_norm = jnp.linalg.norm(
                    rst_global_update.astype(jnp.float32),
                    axis=-1).mean()
                values = {
                    "attn_out_norm": jnp.linalg.norm(
                        attention_global_update.astype(jnp.float32),
                        axis=-1).mean(),
                    "rst_out_norm": jnp.linalg.norm(
                        rst_global_update.astype(jnp.float32),
                        axis=-1).mean(),
                    "residual_norm": jnp.linalg.norm(
                        global_state.astype(jnp.float32), axis=-1).mean(),
                    "q_global_interface_norm": q_global_interface_norm,
                    "k_global_interface_norm": k_global_interface_norm,
                    "v_global_interface_norm": v_global_interface_norm,
                    "rst_global_update_norm": rst_global_update_norm,
                    # Temporary compatibility aliases for existing logs.
                    "q_route_output_norm": q_global_interface_norm,
                    "k_route_output_norm": k_global_interface_norm,
                    "v_route_output_norm": v_global_interface_norm,
                    "rst_route_update_norm": rst_global_update_norm,
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
                    "last_global_state_for_space_diagnostics":
                        rst_global_state_flat,
                    "last_space_query":
                        jnp.stack((
                            attention_space_routing["q"]["space_query"],
                            attention_space_routing["k"]["space_query"],
                            attention_space_routing["v"]["space_query"],
                            rst_space_routing["space_query"],
                        ), axis=0),
                })
            return global_state, layer_aux

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
            def scan_body(global_state, layer_inputs):
                return layer_impl(
                    global_state,
                    layer_inputs["params"],
                    operator_bank,
                    space_selector,
                    space_interface,
                    operator_controller,
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
                    operator_bank,
                    space_selector,
                    space_interface,
                    operator_controller,
                    layer_rngs[layer_index])
                append(regular_lists, layer_aux["regular"])
                if analysis:
                    append(analysis_lists, layer_aux["analysis"])
                    last_global_state_for_space_diagnostics = layer_aux[
                        "last_global_state_for_space_diagnostics"]
                    last_space_query = layer_aux["last_space_query"]

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
            result.update(_operation_space_geometry_diagnostics(
                space_interface["space_read_proj"],
                space_interface["space_write_proj"],
                last_global_state_for_space_diagnostics))
            result.update(_operator_bank_geometry_diagnostics(
                operator_bank,
                space_selector["operation_space_keys"],
                last_space_query))
        return result

    def get_model_info(self) -> list[str]:
        n_spaces = int(self.n_operation_spaces)
        return [
            f"DAWN-SRW v4.1.7.5 ({MODEL_VERSION})",
            "global representation state: token-level persistent d_model "
            "state between layers",
            "space addressing: independent Q/K/V/RST queries select against "
            "shared operation-space keys in one atlas",
            f"canonical geometry: D={self.d_model}, M={n_spaces}, "
            f"R={self.d_route}; space-specific D->R read coordinates, "
            f"top_k={self.operation_space_top_k}",
            "space gate: hard top-k ReLU^2, non-softmax, selected-space "
            "L1 sum 1 with deterministic top-1 epsilon-mass fallback",
            "operation-space state: each selected space reads its own state "
            "from the global representation",
            "local transition: route-specific RW composition and a distinct "
            "tau map in every operation space",
            "operator banks: Q/K/V/RST are fully separate; Q/K/V independently "
            "select spaces from the pre-attention state and RST reselects "
            "after attention",
            "space write: an independently learned projection returns local "
            "outputs or updates to the global representation",
            "attention output boundary: dense expand_O combines heads into "
            "the global residual representation",
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
                    "dense with fused operation-space writeback")),
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
    global_state = (
        params["token_emb"]["embedding"][token_ids]
        + params["pos_emb"]["embedding"][positions]
    ).astype(jnp.float32)
    block = params["block_0"]
    attn_global_state = _shared_layer_norm(
        global_state, block["norm1"]["scale"], block["norm1"]["bias"])
    attn_global_state_flat = attn_global_state.reshape(
        (-1, attn_global_state.shape[-1]))
    if not production_rst:
        return attn_global_state_flat, attn_global_state_flat
    space_selector = params["space_selector"]
    space_interface = params["space_interface"]
    operator_controller = params["operator_controller"]
    operator_bank = params["operator_bank"]
    attention_space_states = _read_operation_space_states(
        attn_global_state_flat, space_interface["space_read_proj"])
    den_qk = (
        admission_den_power
        if admission_den_power_qk is None else admission_den_power_qk)
    den_v = (
        admission_den_power
        if admission_den_power_v is None else admission_den_power_v)
    qk_scale, v_scale, _ = _shared_pool_output_scales(
        int(attn_global_state.shape[-1]), int(n_layers))
    route_scales = {"q": qk_scale, "k": qk_scale, "v": v_scale}
    global_interfaces = {}
    for route in ("q", "k", "v"):
        route_query_proj, operation_space_keys = _route_selector_parameters(
            space_selector, route)
        route_routing = _select_operation_spaces(
            attn_global_state_flat,
            route_query_proj,
            operation_space_keys,
            operation_space_top_k)
        route_temperature = (
            soft_gate_T_qk
            if route in ("q", "k") and soft_gate_T_qk is not None
            else (
                soft_gate_T_v
                if route == "v" and soft_gate_T_v is not None
                else soft_gate_temperature))
        raw_tau = _apply_operation_space_tau_map(
            attention_space_states,
            operator_controller[f"{route}_tau_kernel"],
            operator_controller[f"{route}_tau_bias"])
        space_output = _rw_compose_space_dense(
            attention_space_states,
            operator_bank[f"{route}_read_vectors"],
            operator_bank[f"{route}_write_vectors"],
            raw_tau,
            soft_gate_temperature=route_temperature,
            soft_gate_boundary_power=soft_gate_boundary_power,
            admission_den_power=den_qk if route in ("q", "k") else den_v,
            srw_composition_mode=srw_composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            max_chunk_size=int(
                operator_bank[f"{route}_read_vectors"].shape[1]))
        global_interfaces[route] = _write_space_outputs_to_global(
            space_output,
            route_routing["dense_space_weights"],
            space_interface["space_write_proj"],
            route_scales[route]).reshape(attn_global_state.shape)

    d_model = int(attn_global_state.shape[-1])
    n_heads = _positive_int("n_heads", int(n_heads))
    if d_model % n_heads:
        raise ValueError("calibration d_model must be divisible by n_heads")
    d_head = d_model // n_heads

    def split_heads(value):
        return value.reshape(
            batch_count, sequence_length, n_heads, d_head
        ).transpose(0, 2, 1, 3)

    query = split_heads(global_interfaces["q"])
    key = split_heads(global_interfaces["k"])
    value = split_heads(global_interfaces["v"])
    attention_scores = jnp.einsum(
        "bhsd,bhtd->bhst", query, key
    ) / jnp.sqrt(jnp.float32(d_head))
    causal = jnp.tril(jnp.ones(
        (sequence_length, sequence_length), dtype=jnp.bool_))
    attention_scores = jnp.where(
        causal, attention_scores, jnp.finfo(attention_scores.dtype).min)
    attention_weights = jax.nn.softmax(attention_scores, axis=-1)
    attention_global_update = jnp.einsum(
        "bhst,bhtd->bhsd", attention_weights, value)
    attention_global_update = attention_global_update.transpose(
        0, 2, 1, 3).reshape(batch_count, sequence_length, d_model)
    attention_global_update = _linear(
        block["attn"]["expand_O"], attention_global_update)
    global_state = global_state + attention_global_update
    rst_global_state = _shared_layer_norm(
        global_state,
        block["norm2"]["scale"],
        block["norm2"]["bias"])
    return attn_global_state_flat, rst_global_state.reshape((-1, d_model))


def _tau_init_calibration_scores(
        params, input_ids, max_tokens=128, **production_kwargs):
    """Return direct-read per-route ``[M,T,N]`` cosine tables."""
    attn_global_state, rst_global_state = _sampled_layer_states(
        params, input_ids, max_tokens, **production_kwargs)
    space_interface = params["space_interface"]
    operator_bank = params["operator_bank"]

    def score(space_states, route):
        _, rho, _ = _direct_read_match(
            space_states, operator_bank[f"{route}_read_vectors"])
        return rho

    attention_space_states = _read_operation_space_states(
        attn_global_state, space_interface["space_read_proj"])
    production_rst = bool(production_kwargs.get("production_rst", True))
    rst_space_states = (
        _read_operation_space_states(
            rst_global_state, space_interface["space_read_proj"])
        if production_rst else None)
    return {
        "q": score(attention_space_states, "q"),
        "k": score(attention_space_states, "k"),
        "v": score(attention_space_states, "v"),
        "rst": (
            score(rst_space_states, "rst")
            if production_rst else jnp.zeros(
                (
                    attention_space_states.shape[0],
                    attention_space_states.shape[1],
                    operator_bank["rst_read_vectors"].shape[1],
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
    attn_global_state, _ = _sampled_layer_states(
        params, input_ids, 4096, production_rst=False)
    space_selector = params["space_selector"]
    space_interface = params["space_interface"]
    route_routings = {}
    for route in ROUTES:
        route_query_proj, operation_space_keys = _route_selector_parameters(
            space_selector, route)
        route_routings[route] = _select_operation_spaces(
            attn_global_state,
            route_query_proj,
            operation_space_keys,
            operation_space_top_k)
    diagnostics = _operation_space_geometry_diagnostics(
        space_interface["space_read_proj"],
        space_interface["space_write_proj"],
        attn_global_state)
    diagnostics.update(_operator_bank_geometry_diagnostics(
        params["operator_bank"],
        space_selector["operation_space_keys"],
        jnp.stack(tuple(
            route_routings[route]["space_query"] for route in ROUTES),
            axis=0)))
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
    model = DAWN_SRW_V4175(
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
            space_states, raw_tau, token_valid,
            read_vectors_local, write_vectors_local,
            soft_gate_temperature, soft_gate_t_final,
            soft_gate_boundary_power, soft_gate_boundary_power_final,
            execution_prune_eps):
        del soft_gate_t_final, soft_gate_boundary_power_final
        n_spaces, token_capacity, d_route = space_states.shape
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
        local_bf16 = space_states.astype(jnp.bfloat16)
        local_norm = jnp.maximum(
            jnp.linalg.norm(
                space_states.astype(jnp.float32),
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
            jax.lax.pmax(jax.lax.stop_gradient(gate_max), "model"),
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
    kernel._v4175_kernel_profile = (
        "production_diagnostics" if diagnostics else "production")
    kernel._v4175_dense_grouped_execution = "all_spaces"
    kernel._v4175_direct_read_matmuls = 1
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
_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME = "v4175_bundle_rw_raw_out"
_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME = "v4175_bundle_rw_gate_mass"
_EXACT_SPACE_BACKWARD_BUCKET_CAPACITY = 3072
_EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE = 4
_BUNDLE_BLOCK_CHECKPOINT_POLICY = (
    jax.checkpoint_policies.save_from_both_policies(
        jax.checkpoint_policies.dots_saveable,
        jax.checkpoint_policies.save_only_these_names(
            _BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME,
            _BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)))

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


def _prefix_pack_top2_space_buckets(
        selected_ids: jax.Array,
        selected_weights: jax.Array, *,
        operation_space_count: int,
        bucket_capacity: int,
        task_group_size: int) -> dict[str, jax.Array]:
    """Pack exact pairs into primary buckets and compact overflow tasks."""
    token_count = int(selected_ids.shape[0])
    n_spaces = int(operation_space_count)
    bucket_size = int(bucket_capacity)
    group_size = int(task_group_size)
    if selected_ids.shape != (token_count, 2):
        raise ValueError("exact-space packing requires selected_ids[T,2]")
    if selected_weights.shape != (token_count, 2):
        raise ValueError("exact-space packing requires selected_weights[T,2]")
    if n_spaces <= 0 or bucket_size <= 0 or group_size <= 0:
        raise ValueError("exact-space packing requires positive capacities")

    token_ids = jnp.arange(token_count, dtype=jnp.int32)
    candidate_space = jnp.concatenate(
        (selected_ids[:, 0], selected_ids[:, 1])).astype(jnp.int32)
    candidate_token = jnp.concatenate((token_ids, token_ids))
    candidate_weight = jnp.concatenate(
        (selected_weights[:, 0], selected_weights[:, 1])
    ).astype(jnp.float32)

    space_onehot = jax.nn.one_hot(
        candidate_space, n_spaces, dtype=jnp.int32)
    rank_per_space = (
        jnp.cumsum(space_onehot, axis=0, dtype=jnp.int32)
        - space_onehot)
    local_rank = jnp.sum(
        rank_per_space * space_onehot, axis=1, dtype=jnp.int32)
    counts = space_onehot.sum(axis=0, dtype=jnp.int32)

    primary_valid = local_rank < bucket_size
    primary_token = jnp.zeros(
        (n_spaces, bucket_size), dtype=jnp.int32
    ).at[candidate_space, local_rank].set(
        candidate_token, mode="drop")
    primary_weight = jnp.zeros(
        (n_spaces, bucket_size), dtype=jnp.float32
    ).at[candidate_space, local_rank].set(
        candidate_weight, mode="drop")
    primary_mask = jnp.zeros(
        (n_spaces, bucket_size), dtype=jnp.bool_
    ).at[candidate_space, local_rank].set(
        primary_valid, mode="drop")

    overflow_valid = ~primary_valid
    overflow_counts = jnp.maximum(
        counts - jnp.int32(bucket_size), jnp.int32(0))
    max_chunks_per_space = max(
        1, math.ceil(token_count / bucket_size) - 1)
    raw_task_capacity = max(
        1, math.ceil(2 * token_count / bucket_size))
    task_capacity = (
        math.ceil(raw_task_capacity / group_size) * group_size)
    overflow_chunk = jnp.maximum(
        local_rank // jnp.int32(bucket_size) - jnp.int32(1),
        jnp.int32(0))
    overflow_slot = local_rank % jnp.int32(bucket_size)
    destination_chunk = jnp.where(
        overflow_valid,
        overflow_chunk,
        jnp.int32(max_chunks_per_space))

    space_chunk_token = jnp.zeros(
        (n_spaces, max_chunks_per_space, bucket_size),
        dtype=jnp.int32
    ).at[
        candidate_space, destination_chunk, overflow_slot
    ].set(candidate_token, mode="drop")
    space_chunk_weight = jnp.zeros(
        (n_spaces, max_chunks_per_space, bucket_size),
        dtype=jnp.float32
    ).at[
        candidate_space, destination_chunk, overflow_slot
    ].set(candidate_weight, mode="drop")
    space_chunk_valid = jnp.zeros(
        (n_spaces, max_chunks_per_space, bucket_size),
        dtype=jnp.bool_
    ).at[
        candidate_space, destination_chunk, overflow_slot
    ].set(overflow_valid, mode="drop")

    chunk_ids = jnp.arange(
        max_chunks_per_space, dtype=jnp.int32)[None, :]
    task_exists = (
        chunk_ids * jnp.int32(bucket_size) < overflow_counts[:, None])
    flat_task_exists = task_exists.reshape((-1,))
    task_rank = (
        jnp.cumsum(
            flat_task_exists.astype(jnp.int32), dtype=jnp.int32)
        - flat_task_exists.astype(jnp.int32))
    task_destination = jnp.where(
        flat_task_exists, task_rank, jnp.int32(task_capacity))
    task_space_source = jnp.broadcast_to(
        jnp.arange(n_spaces, dtype=jnp.int32)[:, None],
        task_exists.shape).reshape((-1,))
    task_chunk_source = jnp.broadcast_to(
        jnp.arange(max_chunks_per_space, dtype=jnp.int32)[None, :],
        task_exists.shape).reshape((-1,))
    sentinel = jnp.int32(n_spaces)
    task_space = jnp.full(
        (task_capacity,), sentinel, dtype=jnp.int32
    ).at[task_destination].set(task_space_source, mode="drop")
    task_chunk = jnp.zeros(
        (task_capacity,), dtype=jnp.int32
    ).at[task_destination].set(task_chunk_source, mode="drop")
    task_valid = task_space < sentinel
    safe_task_space = jnp.minimum(task_space, n_spaces - 1)
    task_token = space_chunk_token[safe_task_space, task_chunk]
    task_weight = space_chunk_weight[safe_task_space, task_chunk]
    task_mask = (
        space_chunk_valid[safe_task_space, task_chunk]
        & task_valid[:, None])
    return {
        "primary_token_id": primary_token,
        "primary_routing_weight": primary_weight,
        "primary_token_valid": primary_mask,
        "overflow_task_space_id": task_space,
        "overflow_task_token_id": task_token,
        "overflow_task_routing_weight": task_weight,
        "overflow_task_token_valid": task_mask,
        "space_counts": counts,
    }


def _exact_space_bucket_weight_cotangents_to_dense(
        packed: Mapping[str, jax.Array],
        primary_weight_cotangent: jax.Array,
        overflow_weight_cotangent: jax.Array, *,
        token_count: int,
        operation_space_count: int) -> jax.Array:
    """Scatter exact packed-pair weight gradients to dense routing weights."""
    n_spaces = int(operation_space_count)
    primary_space = jnp.broadcast_to(
        jnp.arange(n_spaces, dtype=jnp.int32)[:, None],
        packed["primary_token_id"].shape)
    dense_grad = jnp.zeros(
        (int(token_count), int(operation_space_count)),
        dtype=jnp.float32)
    dense_grad = dense_grad.at[
        packed["primary_token_id"], primary_space
    ].add(
        jnp.where(
            packed["primary_token_valid"],
            primary_weight_cotangent,
            0.0),
        mode="drop")
    return dense_grad.at[
        packed["overflow_task_token_id"],
        packed["overflow_task_space_id"][:, None]
    ].add(
        jnp.where(
            packed["overflow_task_token_valid"],
            overflow_weight_cotangent,
            0.0),
        mode="drop")


def _dense_rw_output_sharded_autodiff(
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


def _closed_interval_clip_vjp_multiplier(
        value: jax.Array,
        lower: float,
        upper: float) -> jax.Array:
    """Match JAX's half-gradient convention at clip boundaries."""
    value = jnp.asarray(value, dtype=jnp.float32)
    lower_value = jnp.float32(lower)
    upper_value = jnp.float32(upper)
    interior = (value > lower_value) & (value < upper_value)
    boundary = (value == lower_value) | (value == upper_value)
    return (
        interior.astype(jnp.float32)
        + jnp.float32(0.5) * boundary.astype(jnp.float32))


def _maximum_floor_vjp_multiplier(
        value: jax.Array,
        floor: jax.Array | float) -> jax.Array:
    """Match JAX's half-gradient convention for ``maximum(value, floor)``."""
    value_f32 = jnp.asarray(value, dtype=jnp.float32)
    floor_f32 = jnp.asarray(floor, dtype=jnp.float32)
    return jnp.where(
        value_f32 > floor_f32,
        jnp.float32(1.0),
        jnp.where(
            value_f32 == floor_f32,
            jnp.float32(0.5),
            jnp.float32(0.0)))


def _forward_unit_direction_vjp(
        raw_value: jax.Array,
        unit_cotangent: jax.Array) -> jax.Array:
    """Analytic pullback for x / (||x|| + eps), independently per row."""
    raw_f32 = jnp.asarray(raw_value, dtype=jnp.float32)
    cotangent_f32 = jnp.asarray(unit_cotangent, dtype=jnp.float32)
    norm = jnp.linalg.norm(raw_f32, axis=-1, keepdims=True)
    denominator = norm + jnp.float32(RW_FORWARD_NORM_EPS)
    denominator_cotangent = -(
        cotangent_f32 * raw_f32
    ).sum(axis=-1, keepdims=True) / jnp.square(denominator)
    norm_direction = jnp.where(
        norm > 0.0,
        raw_f32 / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny),
        0.0)
    return (
        cotangent_f32 / denominator
        + denominator_cotangent * norm_direction)


def _dense_rw_vjp_dot(
        equation: str,
        lhs: jax.Array,
        rhs: jax.Array, *,
        throughput_bf16: bool) -> jax.Array:
    """Transpose one RW dot while matching the primal cast boundary."""
    if throughput_bf16:
        result = jnp.einsum(
            equation,
            lhs,
            rhs,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32)
        return result.astype(jnp.bfloat16).astype(jnp.float32)
    return jnp.einsum(
        equation,
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(rhs, dtype=jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32)


@partial(
    jax.custom_vjp,
    nondiff_argnums=(6, 10, 11, 12),
)
def _dense_rw_output_sharded_linear_analytic(
        local_f32: jax.Array,
        local_norm: jax.Array,
        token_valid: jax.Array,
        read_vectors: jax.Array,
        write_vectors: jax.Array,
        raw_tau: jax.Array,
        max_chunk_size: int,
        soft_gate_temperature: jax.Array,
        soft_gate_boundary_power: jax.Array,
        execution_prune_eps: jax.Array,
        heat_kernel_beta: float,
        effective_active_eps: float,
        throughput_bf16: bool) -> tuple[jax.Array, jax.Array]:
    """Keep the accepted dense primal and replace only its RW pullback."""
    return _dense_rw_output_sharded_autodiff(
        local_f32,
        local_norm,
        token_valid,
        read_vectors,
        write_vectors,
        raw_tau,
        max_chunk_size=max_chunk_size,
        soft_gate_temperature=soft_gate_temperature,
        soft_gate_boundary_power=soft_gate_boundary_power,
        execution_prune_eps=execution_prune_eps,
        srw_composition_mode="linear_angular",
        heat_kernel_beta=heat_kernel_beta,
        effective_active_eps=effective_active_eps,
        throughput_bf16=throughput_bf16)


def _dense_rw_output_sharded_linear_analytic_fwd(
        local_f32: jax.Array,
        local_norm: jax.Array,
        token_valid: jax.Array,
        read_vectors: jax.Array,
        write_vectors: jax.Array,
        raw_tau: jax.Array,
        max_chunk_size: int,
        soft_gate_temperature: jax.Array,
        soft_gate_boundary_power: jax.Array,
        execution_prune_eps: jax.Array,
        heat_kernel_beta: float,
        effective_active_eps: float,
        throughput_bf16: bool):
    output = _dense_rw_output_sharded_autodiff(
        local_f32,
        local_norm,
        token_valid,
        read_vectors,
        write_vectors,
        raw_tau,
        max_chunk_size=max_chunk_size,
        soft_gate_temperature=soft_gate_temperature,
        soft_gate_boundary_power=soft_gate_boundary_power,
        execution_prune_eps=execution_prune_eps,
        srw_composition_mode="linear_angular",
        heat_kernel_beta=heat_kernel_beta,
        effective_active_eps=effective_active_eps,
        throughput_bf16=throughput_bf16)
    residual = (
        local_f32,
        local_norm,
        token_valid,
        read_vectors,
        write_vectors,
        raw_tau,
        soft_gate_temperature,
        soft_gate_boundary_power,
        execution_prune_eps,
    )
    return output, residual


def _dense_rw_output_sharded_linear_analytic_bwd(
        max_chunk_size: int,
        heat_kernel_beta: float,
        effective_active_eps: float,
        throughput_bf16: bool,
        residual,
        output_cotangent):
    del heat_kernel_beta, effective_active_eps
    (local_f32,
     local_norm,
     token_valid,
     read_vectors,
     write_vectors,
     raw_tau,
     soft_gate_temperature,
     soft_gate_boundary_power,
     execution_prune_eps) = residual
    raw_out_cotangent, gate_mass_cotangent = output_cotangent
    n_routes, n_spaces, token_capacity, d_route = map(
        int, (read_vectors.shape[0], local_f32.shape[0],
              local_f32.shape[1], local_f32.shape[2]))
    n_local = int(read_vectors.shape[2])
    chunk_size = min(max(1, int(max_chunk_size)), n_local)
    n_chunks = math.ceil(n_local / chunk_size)
    n_padded = n_chunks * chunk_size
    pad_n = n_padded - n_local
    pad_spec = ((0, 0), (0, 0), (0, pad_n), (0, 0))
    padded_reads = jnp.pad(read_vectors, pad_spec)
    padded_writes = jnp.pad(write_vectors, pad_spec)
    normalized_reads = forward_unit_direction(padded_reads)
    normalized_writes = forward_unit_direction(padded_writes)
    valid_rows = jnp.arange(n_padded) < n_local
    sigmoid_tau = jax.nn.sigmoid(
        jnp.asarray(raw_tau, dtype=jnp.float32))
    tau = (
        jnp.float32(-1.0)
        + jnp.float32(2.0) * sigmoid_tau)
    tau_param_multiplier = (
        jnp.float32(2.0)
        * sigmoid_tau
        * (jnp.float32(1.0) - sigmoid_tau))
    prune_eps = jnp.asarray(
        execution_prune_eps, dtype=jnp.float32)
    raw_out_ct = jnp.asarray(
        raw_out_cotangent, dtype=jnp.float32)
    gate_mass_ct = jnp.asarray(
        gate_mass_cotangent, dtype=jnp.float32)
    initial_carry = (
        jnp.zeros(
            (n_spaces, token_capacity, d_route),
            dtype=jnp.float32),
        jnp.zeros(
            (n_spaces, token_capacity, 1),
            dtype=jnp.float32),
        jnp.zeros_like(tau, dtype=jnp.float32),
    )

    def chunk_pullback(carry_value, chunk_index):
        local_cotangent, norm_cotangent, tau_cotangent = carry_value
        start = chunk_index * chunk_size
        raw_read = jax.lax.dynamic_slice_in_dim(
            padded_reads, start, chunk_size, axis=2)
        raw_write = jax.lax.dynamic_slice_in_dim(
            padded_writes, start, chunk_size, axis=2)
        read = jax.lax.dynamic_slice_in_dim(
            normalized_reads, start, chunk_size, axis=2)
        write = jax.lax.dynamic_slice_in_dim(
            normalized_writes, start, chunk_size, axis=2)
        row_valid = jax.lax.dynamic_slice_in_dim(
            valid_rows, start, chunk_size, axis=0)
        valid = (
            token_valid[None, :, :, None]
            & row_valid[None, None, None, :])
        read_value = (
            _throughput_einsum_bf16_f32(
                "mtr,amnr->amtn", local_f32, read)
            if throughput_bf16
            else _control_einsum_f32(
                "mtr,amnr->amtn", local_f32, read)
        ).astype(jnp.float32)
        rho_unclipped = read_value / local_norm[None, ...]
        rho = jnp.clip(rho_unclipped, -1.0, 1.0)
        rho_compute = jnp.where(valid, rho, tau)
        score = jnp.clip(rho_compute, -1.0, 1.0)
        margin = score - tau
        tau_distance = jnp.float32(1.0) - tau
        angular_den = jnp.maximum(
            tau_distance, jnp.float32(1.0e-4))
        angular_unclipped = margin / angular_den
        admission = jnp.clip(angular_unclipped, 0.0, 1.0)
        execution_enabled = (
            (prune_eps <= 0.0) | (admission >= prune_eps))
        execution_weight = jnp.where(
            execution_enabled, admission, 0.0)
        execution_weight = jnp.where(
            valid, execution_weight, 0.0)
        weighted_read = execution_weight * read_value

        write_operand = (
            write.astype(jnp.bfloat16)
            if throughput_bf16 else write)
        weighted_operand = (
            weighted_read.astype(jnp.bfloat16)
            if throughput_bf16 else weighted_read)
        weighted_cotangent = _dense_rw_vjp_dot(
            "amtr,amnr->amtn",
            raw_out_ct,
            write_operand,
            throughput_bf16=throughput_bf16)
        write_unit_cotangent = _dense_rw_vjp_dot(
            "amtn,amtr->amnr",
            weighted_operand,
            raw_out_ct,
            throughput_bf16=throughput_bf16)
        execution_cotangent = weighted_cotangent * read_value
        read_value_cotangent = (
            weighted_cotangent * execution_weight)

        admission_cotangent = jnp.where(
            valid,
            gate_mass_ct + jnp.where(
                execution_enabled, execution_cotangent, 0.0),
            0.0)
        angular_unclipped_cotangent = (
            admission_cotangent
            * _closed_interval_clip_vjp_multiplier(
                angular_unclipped, 0.0, 1.0))
        margin_cotangent = (
            angular_unclipped_cotangent / angular_den)
        angular_den_cotangent = -(
            angular_unclipped_cotangent
            * margin
            / jnp.square(angular_den))
        tau_distance_multiplier = jnp.where(
            tau_distance > jnp.float32(1.0e-4),
            jnp.float32(1.0),
            jnp.where(
                tau_distance == jnp.float32(1.0e-4),
                jnp.float32(0.5),
                jnp.float32(0.0)))
        chunk_tau_cotangent = (
            -margin_cotangent
            - angular_den_cotangent * tau_distance_multiplier)
        score_cotangent = margin_cotangent
        rho_compute_cotangent = (
            score_cotangent
            * _closed_interval_clip_vjp_multiplier(
                rho_compute, -1.0, 1.0))
        chunk_tau_cotangent = (
            chunk_tau_cotangent
            + jnp.where(valid, 0.0, rho_compute_cotangent))
        rho_cotangent = jnp.where(
            valid, rho_compute_cotangent, 0.0)
        rho_unclipped_cotangent = (
            rho_cotangent
            * _closed_interval_clip_vjp_multiplier(
                rho_unclipped, -1.0, 1.0))
        read_value_cotangent = (
            read_value_cotangent
            + rho_unclipped_cotangent / local_norm[None, ...])
        chunk_norm_cotangent = -(
            rho_unclipped_cotangent
            * read_value
            / jnp.square(local_norm[None, ...]))

        read_operand = (
            read.astype(jnp.bfloat16)
            if throughput_bf16 else read)
        local_operand = (
            local_f32.astype(jnp.bfloat16)
            if throughput_bf16 else local_f32)
        chunk_local_cotangent = _dense_rw_vjp_dot(
            "amtn,amnr->mtr",
            read_value_cotangent,
            read_operand,
            throughput_bf16=throughput_bf16)
        read_unit_cotangent = _dense_rw_vjp_dot(
            "amtn,mtr->amnr",
            read_value_cotangent,
            local_operand,
            throughput_bf16=throughput_bf16)
        raw_read_cotangent = _forward_unit_direction_vjp(
            raw_read, read_unit_cotangent)
        raw_write_cotangent = _forward_unit_direction_vjp(
            raw_write, write_unit_cotangent)
        return (
            local_cotangent + chunk_local_cotangent,
            norm_cotangent + chunk_norm_cotangent.sum(
                axis=(0, 3))[:, :, None],
            tau_cotangent + chunk_tau_cotangent.sum(
                axis=-1, keepdims=True),
        ), (raw_read_cotangent, raw_write_cotangent)

    (local_cotangent,
     norm_cotangent,
     tau_cotangent), (
         read_chunk_cotangents,
         write_chunk_cotangents,
     ) = jax.lax.scan(
        chunk_pullback, initial_carry, jnp.arange(n_chunks))
    read_cotangent = jnp.transpose(
        read_chunk_cotangents, (1, 2, 0, 3, 4)
    ).reshape((n_routes, n_spaces, n_padded, d_route))[:, :, :n_local, :]
    write_cotangent = jnp.transpose(
        write_chunk_cotangents, (1, 2, 0, 3, 4)
    ).reshape((n_routes, n_spaces, n_padded, d_route))[:, :, :n_local, :]
    raw_tau_cotangent = tau_cotangent * tau_param_multiplier
    return (
        local_cotangent,
        norm_cotangent,
        None,
        read_cotangent,
        write_cotangent,
        raw_tau_cotangent,
        jnp.zeros_like(soft_gate_temperature),
        jnp.zeros_like(soft_gate_boundary_power),
        jnp.zeros_like(execution_prune_eps),
    )


_dense_rw_output_sharded_linear_analytic.defvjp(
    _dense_rw_output_sharded_linear_analytic_fwd,
    _dense_rw_output_sharded_linear_analytic_bwd)


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
    """Use the analytic dense RW pullback for canonical linear-angular runs."""
    if str(srw_composition_mode) == "linear_angular":
        return _dense_rw_output_sharded_linear_analytic(
            local_f32,
            local_norm,
            token_valid,
            read_vectors,
            write_vectors,
            raw_tau,
            max_chunk_size,
            soft_gate_temperature,
            soft_gate_boundary_power,
            execution_prune_eps,
            heat_kernel_beta,
            effective_active_eps,
            throughput_bf16)
    return _dense_rw_output_sharded_autodiff(
        local_f32,
        local_norm,
        token_valid,
        read_vectors,
        write_vectors,
        raw_tau,
        max_chunk_size=max_chunk_size,
        soft_gate_temperature=soft_gate_temperature,
        soft_gate_boundary_power=soft_gate_boundary_power,
        execution_prune_eps=execution_prune_eps,
        srw_composition_mode=srw_composition_mode,
        heat_kernel_beta=heat_kernel_beta,
        effective_active_eps=effective_active_eps,
        throughput_bf16=throughput_bf16)


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


def _composition_den_mass_pullback(
        global_gate_mass: jax.Array,
        gate_den_cotangent: jax.Array,
        admission_den_power: jax.Array,
        srw_composition_mode: str) -> jax.Array:
    """Analytic gradient of the live global composition denominator."""
    floor_mass = jnp.float32(
        _shared_composition_den_floor_mass(srw_composition_mode))
    mass_f32 = jnp.asarray(global_gate_mass, dtype=jnp.float32)
    power_f32 = jnp.asarray(admission_den_power, dtype=jnp.float32)
    base = jnp.maximum(mass_f32, floor_mass)
    base_cotangent = (
        jnp.asarray(gate_den_cotangent, dtype=jnp.float32)
        * power_f32
        * jnp.power(base, power_f32 - jnp.float32(1.0)))
    return (
        base_cotangent
        * _maximum_floor_vjp_multiplier(mass_f32, floor_mass))


def _attention_exact_space_linear_group_pullback(
        packed_state: jax.Array,
        token_valid: jax.Array,
        routing_weight: jax.Array,
        state_params: tuple[jax.Array, ...],
        operator_params: tuple[jax.Array, ...],
        controls: tuple[jax.Array, ...],
        block_output_cotangent: jax.Array, *,
        max_chunk_size_qk: int,
        max_chunk_size_v: int,
        den_powers: jax.Array,
        heat_kernel_beta: float,
        soft_gate_effective_active_eps: float,
        throughput_bf16: bool
) -> tuple[
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array]:
    """Pull back one exact-space Q/K/V group without an outer AD graph."""
    (space_read_proj,
     space_write_proj,
     qk_tau_kernels,
     qk_tau_biases,
     v_tau_kernel,
     v_tau_bias) = state_params
    qk_reads, qk_writes, v_read, v_write = operator_params
    (qk_temperature,
     v_temperature,
     boundary_power,
     execution_prune_eps,
     qk_scale,
     v_scale) = controls
    projection_dot = (
        _throughput_einsum_bf16_f32
        if throughput_bf16 else _control_einsum_f32)

    local = projection_dot(
        "mtd,mdr->mtr", packed_state, space_read_proj
    ).astype(jnp.float32)
    raw_local_norm = jnp.linalg.norm(
        local, axis=-1, keepdims=True)
    local_norm = jnp.maximum(
        raw_local_norm, jnp.float32(RW_FORWARD_NORM_EPS))
    qk_raw_tau = (
        _control_einsum_f32(
            "mtr,amri->amti", local, qk_tau_kernels)
        + qk_tau_biases[:, :, None, :])
    v_raw_tau = _apply_operation_space_tau_map(
        local, v_tau_kernel, v_tau_bias)

    (qk_raw_out,
     qk_gate_mass), qk_rw_residual = (
        _dense_rw_output_sharded_linear_analytic_fwd(
            local,
            local_norm,
            token_valid,
            qk_reads,
            qk_writes,
            qk_raw_tau,
            max_chunk_size_qk,
            qk_temperature,
            boundary_power,
            execution_prune_eps,
            heat_kernel_beta,
            soft_gate_effective_active_eps,
            throughput_bf16))
    (v_raw_out,
     v_gate_mass), v_rw_residual = (
        _dense_rw_output_sharded_linear_analytic_fwd(
            local,
            local_norm,
            token_valid,
            v_read[None, ...],
            v_write[None, ...],
            v_raw_tau[None, ...],
            max_chunk_size_v,
            v_temperature,
            boundary_power,
            execution_prune_eps,
            heat_kernel_beta,
            soft_gate_effective_active_eps,
            throughput_bf16))
    grouped_raw_out = jnp.concatenate(
        (qk_raw_out, v_raw_out), axis=0)
    grouped_gate_mass = jnp.concatenate(
        (qk_gate_mass, v_gate_mass), axis=0)
    global_gate_mass, grouped_gate_den = (
        _global_dense_rw_den_sharded(
            grouped_gate_mass,
            den_powers,
            "linear_angular"))
    local_results = (
        grouped_raw_out / grouped_gate_den).astype(jnp.float32)

    route_scales = jnp.asarray(
        (qk_scale, qk_scale, v_scale),
        dtype=jnp.float32).reshape((3, 1, 1, 1))
    weight_f32 = jnp.asarray(
        routing_weight, dtype=jnp.float32)[None, ..., None]
    weighted_results = local_results * weight_f32 * route_scales
    valid_output_cotangent = jnp.where(
        token_valid[None, ..., None],
        jnp.asarray(block_output_cotangent, dtype=jnp.float32),
        jnp.float32(0.0))
    writeback_operand = (
        space_write_proj.astype(jnp.bfloat16)
        if throughput_bf16 else space_write_proj)
    weighted_operand = (
        weighted_results.astype(jnp.bfloat16)
        if throughput_bf16 else weighted_results)
    weighted_results_cotangent = _dense_rw_vjp_dot(
        "amtd,mrd->amtr",
        valid_output_cotangent,
        writeback_operand,
        throughput_bf16=throughput_bf16)
    space_write_proj_cotangent = _dense_rw_vjp_dot(
        "amtr,amtd->mrd",
        weighted_operand,
        valid_output_cotangent,
        throughput_bf16=throughput_bf16)

    local_results_cotangent = (
        weighted_results_cotangent * weight_f32 * route_scales)
    routing_weight_cotangent = (
        weighted_results_cotangent
        * local_results
        * route_scales
    ).sum(axis=(0, 3))
    route_scale_cotangent = (
        weighted_results_cotangent
        * local_results
        * weight_f32
    ).sum(axis=(1, 2, 3))
    qk_scale_cotangent = (
        route_scale_cotangent[0] + route_scale_cotangent[1])
    v_scale_cotangent = route_scale_cotangent[2]

    grouped_raw_out_cotangent = (
        local_results_cotangent / grouped_gate_den)
    grouped_gate_den_cotangent = -(
        local_results_cotangent
        * grouped_raw_out
        / jnp.square(grouped_gate_den)
    ).sum(axis=-1, keepdims=True)
    global_gate_mass_cotangent = _composition_den_mass_pullback(
        global_gate_mass,
        grouped_gate_den_cotangent,
        den_powers,
        "linear_angular")
    grouped_gate_mass_cotangent = jax.lax.psum(
        global_gate_mass_cotangent, "model")

    qk_rw_cotangents = (
        grouped_raw_out_cotangent[:2],
        grouped_gate_mass_cotangent[:2])
    v_rw_cotangents = (
        grouped_raw_out_cotangent[2:],
        grouped_gate_mass_cotangent[2:])
    (qk_local_cotangent,
     qk_local_norm_cotangent,
     _,
     qk_reads_cotangent,
     qk_writes_cotangent,
     qk_raw_tau_cotangent,
     qk_temperature_cotangent,
     qk_boundary_cotangent,
     qk_prune_cotangent) = (
        _dense_rw_output_sharded_linear_analytic_bwd(
            max_chunk_size_qk,
            heat_kernel_beta,
            soft_gate_effective_active_eps,
            throughput_bf16,
            qk_rw_residual,
            qk_rw_cotangents))
    (v_local_cotangent,
     v_local_norm_cotangent,
     _,
     v_reads_cotangent,
     v_writes_cotangent,
     v_raw_tau_cotangent,
     v_temperature_cotangent,
     v_boundary_cotangent,
     v_prune_cotangent) = (
        _dense_rw_output_sharded_linear_analytic_bwd(
            max_chunk_size_v,
            heat_kernel_beta,
            soft_gate_effective_active_eps,
            throughput_bf16,
            v_rw_residual,
            v_rw_cotangents))

    qk_tau_kernel_cotangent = _control_einsum_f32(
        "mtr,amti->amri", local, qk_raw_tau_cotangent)
    qk_tau_bias_cotangent = qk_raw_tau_cotangent.sum(axis=2)
    qk_tau_local_cotangent = _control_einsum_f32(
        "amti,amri->mtr",
        qk_raw_tau_cotangent,
        qk_tau_kernels)
    v_raw_tau_cotangent = v_raw_tau_cotangent[0]
    v_tau_kernel_cotangent = _control_einsum_f32(
        "mtr,mti->mri", local, v_raw_tau_cotangent)
    v_tau_bias_cotangent = v_raw_tau_cotangent.sum(axis=1)
    v_tau_local_cotangent = _control_einsum_f32(
        "mti,mri->mtr",
        v_raw_tau_cotangent,
        v_tau_kernel)

    local_norm_cotangent = (
        qk_local_norm_cotangent + v_local_norm_cotangent)
    norm_direction = jnp.where(
        raw_local_norm > 0.0,
        local / jnp.maximum(
            raw_local_norm, jnp.finfo(jnp.float32).tiny),
        jnp.float32(0.0))
    local_cotangent = (
        qk_local_cotangent
        + v_local_cotangent
        + qk_tau_local_cotangent
        + v_tau_local_cotangent
        + local_norm_cotangent
        * _maximum_floor_vjp_multiplier(
            raw_local_norm, RW_FORWARD_NORM_EPS)
        * norm_direction)

    projection_operand = (
        space_read_proj.astype(jnp.bfloat16)
        if throughput_bf16 else space_read_proj)
    state_operand = (
        packed_state.astype(jnp.bfloat16)
        if throughput_bf16 else packed_state)
    packed_state_cotangent = _dense_rw_vjp_dot(
        "mtr,mdr->mtd",
        local_cotangent,
        projection_operand,
        throughput_bf16=throughput_bf16)
    space_read_proj_cotangent = _dense_rw_vjp_dot(
        "mtd,mtr->mdr",
        state_operand,
        local_cotangent,
        throughput_bf16=throughput_bf16)

    return (
        packed_state_cotangent,
        (
            space_read_proj_cotangent,
            space_write_proj_cotangent,
            qk_tau_kernel_cotangent,
            qk_tau_bias_cotangent,
            v_tau_kernel_cotangent,
            v_tau_bias_cotangent,
        ),
        (
            qk_reads_cotangent,
            qk_writes_cotangent,
            v_reads_cotangent[0],
            v_writes_cotangent[0],
        ),
        (
            qk_temperature_cotangent,
            v_temperature_cotangent,
            qk_boundary_cotangent + v_boundary_cotangent,
            qk_prune_cotangent + v_prune_cotangent,
            qk_scale_cotangent,
            v_scale_cotangent,
        ),
        jnp.where(
            token_valid,
            routing_weight_cotangent,
            jnp.float32(0.0)))


def _rst_exact_space_linear_group_pullback(
        packed_state: jax.Array,
        token_valid: jax.Array,
        routing_weight: jax.Array,
        state_params: tuple[jax.Array, ...],
        operator_params: tuple[jax.Array, ...],
        controls: tuple[jax.Array, ...],
        block_output_cotangent: jax.Array, *,
        max_chunk_size: int,
        den_power: jax.Array,
        heat_kernel_beta: float,
        soft_gate_effective_active_eps: float,
        throughput_bf16: bool
) -> tuple[
        jax.Array,
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array]:
    """Pull back one exact-space RST group without an outer AD graph."""
    (space_read_proj,
     space_write_proj,
     tau_kernel,
     tau_bias) = state_params
    read_vectors, write_vectors = operator_params
    (temperature,
     boundary_power,
     execution_prune_eps,
     route_scale) = controls
    projection_dot = (
        _throughput_einsum_bf16_f32
        if throughput_bf16 else _control_einsum_f32)

    local = projection_dot(
        "mtd,mdr->mtr", packed_state, space_read_proj
    ).astype(jnp.float32)
    raw_local_norm = jnp.linalg.norm(
        local, axis=-1, keepdims=True)
    local_norm = jnp.maximum(
        raw_local_norm, jnp.float32(RW_FORWARD_NORM_EPS))
    raw_tau = _apply_operation_space_tau_map(
        local, tau_kernel, tau_bias)
    (raw_out,
     gate_mass), rw_residual = (
        _dense_rw_output_sharded_linear_analytic_fwd(
            local,
            local_norm,
            token_valid,
            read_vectors[None, ...],
            write_vectors[None, ...],
            raw_tau[None, ...],
            max_chunk_size,
            temperature,
            boundary_power,
            execution_prune_eps,
            heat_kernel_beta,
            soft_gate_effective_active_eps,
            throughput_bf16))
    global_gate_mass, gate_den = _global_dense_rw_den_sharded(
        gate_mass, den_power, "linear_angular")
    local_results = (raw_out / gate_den).astype(jnp.float32)

    weight_f32 = jnp.asarray(
        routing_weight, dtype=jnp.float32)[..., None]
    route_scale_f32 = jnp.asarray(route_scale, dtype=jnp.float32)
    weighted_results = (
        local_results[0] * weight_f32 * route_scale_f32)
    valid_output_cotangent = jnp.where(
        token_valid[..., None],
        jnp.asarray(block_output_cotangent, dtype=jnp.float32),
        jnp.float32(0.0))
    writeback_operand = (
        space_write_proj.astype(jnp.bfloat16)
        if throughput_bf16 else space_write_proj)
    weighted_operand = (
        weighted_results.astype(jnp.bfloat16)
        if throughput_bf16 else weighted_results)
    weighted_results_cotangent = _dense_rw_vjp_dot(
        "mtd,mrd->mtr",
        valid_output_cotangent,
        writeback_operand,
        throughput_bf16=throughput_bf16)
    space_write_proj_cotangent = _dense_rw_vjp_dot(
        "mtr,mtd->mrd",
        weighted_operand,
        valid_output_cotangent,
        throughput_bf16=throughput_bf16)

    local_results_cotangent = (
        weighted_results_cotangent * weight_f32 * route_scale_f32)
    routing_weight_cotangent = (
        weighted_results_cotangent
        * local_results[0]
        * route_scale_f32
    ).sum(axis=-1)
    route_scale_cotangent = (
        weighted_results_cotangent
        * local_results[0]
        * weight_f32
    ).sum()

    raw_out_cotangent = local_results_cotangent[None, ...] / gate_den
    gate_den_cotangent = -(
        local_results_cotangent[None, ...]
        * raw_out
        / jnp.square(gate_den)
    ).sum(axis=-1, keepdims=True)
    global_gate_mass_cotangent = _composition_den_mass_pullback(
        global_gate_mass,
        gate_den_cotangent,
        den_power,
        "linear_angular")
    gate_mass_cotangent = jax.lax.psum(
        global_gate_mass_cotangent, "model")

    (local_cotangent,
     local_norm_cotangent,
     _,
     reads_cotangent,
     writes_cotangent,
     raw_tau_cotangent,
     temperature_cotangent,
     boundary_cotangent,
     prune_cotangent) = (
        _dense_rw_output_sharded_linear_analytic_bwd(
            max_chunk_size,
            heat_kernel_beta,
            soft_gate_effective_active_eps,
            throughput_bf16,
            rw_residual,
            (raw_out_cotangent, gate_mass_cotangent)))

    raw_tau_cotangent = raw_tau_cotangent[0]
    tau_kernel_cotangent = _control_einsum_f32(
        "mtr,mti->mri", local, raw_tau_cotangent)
    tau_bias_cotangent = raw_tau_cotangent.sum(axis=1)
    tau_local_cotangent = _control_einsum_f32(
        "mti,mri->mtr", raw_tau_cotangent, tau_kernel)

    norm_direction = jnp.where(
        raw_local_norm > 0.0,
        local / jnp.maximum(
            raw_local_norm, jnp.finfo(jnp.float32).tiny),
        jnp.float32(0.0))
    local_cotangent = (
        local_cotangent
        + tau_local_cotangent
        + local_norm_cotangent
        * _maximum_floor_vjp_multiplier(
            raw_local_norm, RW_FORWARD_NORM_EPS)
        * norm_direction)

    projection_operand = (
        space_read_proj.astype(jnp.bfloat16)
        if throughput_bf16 else space_read_proj)
    state_operand = (
        packed_state.astype(jnp.bfloat16)
        if throughput_bf16 else packed_state)
    packed_state_cotangent = _dense_rw_vjp_dot(
        "mtr,mdr->mtd",
        local_cotangent,
        projection_operand,
        throughput_bf16=throughput_bf16)
    space_read_proj_cotangent = _dense_rw_vjp_dot(
        "mtd,mtr->mdr",
        state_operand,
        local_cotangent,
        throughput_bf16=throughput_bf16)

    return (
        packed_state_cotangent,
        (
            space_read_proj_cotangent,
            space_write_proj_cotangent,
            tau_kernel_cotangent,
            tau_bias_cotangent,
        ),
        (
            reads_cotangent[0],
            writes_cotangent[0],
        ),
        (
            temperature_cotangent,
            boundary_cotangent,
            prune_cotangent,
            route_scale_cotangent,
        ),
        jnp.where(
            token_valid,
            routing_weight_cotangent,
            jnp.float32(0.0)))


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
    fallback = jax.lax.stop_gradient(
        routing["fallback_mask"][..., 0].astype(jnp.float32))
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
            fallback.sum(),
        )),
        top1_counts,
        jnp.asarray((token_count,), dtype=jnp.float32),
    ))
    global_routing = jax.lax.psum(routing_sums, "data")
    global_tokens = jnp.maximum(
        global_routing[-1], jnp.float32(1.0))
    top1_offset = 4
    exact_zero_mass_frac = global_routing[2] / global_tokens
    metric_values.extend((
        global_routing[0] / global_tokens,
        global_routing[1] / global_tokens,
        exact_zero_mass_frac,
        global_routing[3] / global_tokens,
        exact_zero_mass_frac,
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
    fallback = jax.lax.stop_gradient(
        routing["fallback_mask"][..., 0].astype(jnp.float32))
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
            fallback.sum(),
        )),
        top1_counts,
        jnp.stack((local_positions, local_tokens)),
    ))
    global_metrics = jax.lax.psum(packed_metrics, "data")

    operator_width = 8
    space_offset = operator_width * len(route_specs)
    top1_offset = space_offset + 4
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
    exact_zero_mass_frac = (
        global_metrics[space_offset + 2] / global_tokens)
    metric_values.extend((
        global_metrics[space_offset] / global_tokens,
        global_metrics[space_offset + 1] / global_tokens,
        exact_zero_mass_frac,
        global_metrics[space_offset + 3] / global_tokens,
        exact_zero_mass_frac,
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
        for suffix in _SPACE_ROUTING_COMPACT_METRIC_SUFFIXES)

    def attention_core(
            global_state,
            space_query_kernel, operation_space_keys,
            space_read_proj, space_write_proj,
            q_tau_kernel, q_tau_bias,
            k_tau_kernel, k_tau_bias,
            v_tau_kernel, v_tau_bias,
            q_read, q_write, k_read, k_write, v_read, v_write,
            qk_temperature, v_temperature,
            boundary_power, execution_prune_eps,
            qk_scale, v_scale, collect_metrics):
        routing = _select_operation_spaces(
            global_state, space_query_kernel, operation_space_keys, top_k)
        if throughput_bf16:
            space_states = jnp.swapaxes(
                throughput_dot_bf16_f32(
                    global_state, space_read_proj,
                    dimension_numbers=(
                        ((1,), (1,)),
                        ((), ()))),
                0, 1)
        else:
            space_states = _control_einsum_f32(
                "td,mdr->mtr", global_state, space_read_proj)
        space_states = space_states.astype(jnp.float32)
        space_state_norm = jnp.maximum(
            jnp.linalg.norm(space_states, axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))
        token_valid = jnp.ones(space_states.shape[:2], dtype=jnp.bool_)
        qk_tau_kernels = jnp.stack((q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_biases = jnp.stack((q_tau_bias, k_tau_bias), axis=0)
        qk_raw_tau = (
            _control_einsum_f32(
                "mtr,amri->amti", space_states, qk_tau_kernels)
            + qk_tau_biases[:, :, None, :])
        qk_local_output = _dense_rw_output_sharded(
            space_states, space_state_norm, token_valid,
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

        v_raw_tau = _apply_operation_space_tau_map(
            space_states, v_tau_kernel, v_tau_bias)
        v_local_output = _dense_rw_output_sharded(
            space_states, space_state_norm, token_valid,
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
        grouped_space_results = (
            grouped_raw_out / grouped_gate_den).astype(jnp.float32)
        space_weights = jnp.swapaxes(
            routing["dense_space_weights"], 0, 1)[None, ..., None]
        route_scales = jnp.asarray(
            (qk_scale, qk_scale, v_scale), dtype=jnp.float32
        ).reshape((3, 1, 1, 1))
        writeback_dot = (
            _throughput_einsum_bf16_f32
            if throughput_bf16 else _control_einsum_f32)
        grouped_global_interfaces = writeback_dot(
            "amtr,mrd->atd",
            grouped_space_results * space_weights * route_scales,
            space_write_proj).astype(jnp.float32)
        grouped_output = _psum_dense_rw_representation_sharded(
            grouped_global_interfaces, output_collective_bf16)

        q_per_space = int(q_read.shape[1]) * model_axis_size
        k_per_space = int(k_read.shape[1]) * model_axis_size
        v_per_space = int(v_read.shape[1]) * model_axis_size
        route_specs = (
            ("q", q_per_space),
            ("k", k_per_space),
            ("v", v_per_space),
        )
        metric_local = jax.lax.stop_gradient(space_states)
        metric_norm = jax.lax.stop_gradient(space_state_norm)
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
    kernel._v4175_kernel_profile = "production"
    kernel._v4175_dense_grouped_execution = "attention_qkv"
    kernel._v4175_execution_mode = "dense_all_space"
    kernel._v4175_qk_paired = True
    kernel._v4175_dynamic_metric_flag = True
    kernel._v4175_chunk_remat_policy = "always"
    kernel._v4175_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4175_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4175_output_contract = ("q[T,D]", "k[T,D]", "v[T,D]", "scalars")
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
        for suffix in _SPACE_ROUTING_COMPACT_METRIC_SUFFIXES)
    bundle_metric_names = _bundle_packing_metric_names("attention")
    projection_dot = (
        _throughput_einsum_bf16_f32
        if throughput_bf16 else _control_einsum_f32)
    writeback_dot = projection_dot
    den_powers = jnp.asarray(
        (
            admission_den_power_qk,
            admission_den_power_qk,
            admission_den_power_v,
        ),
        dtype=jnp.float32).reshape((3, 1, 1, 1))

    def bundle_execute(
            global_state,
            state_params,
            operator_params,
            controls,
            packing):
        (space_read_proj, space_write_proj,
         q_tau_kernel, q_tau_bias,
         k_tau_kernel, k_tau_bias,
         v_tau_kernel, v_tau_bias) = state_params
        (q_read, q_write, k_read, k_write,
         v_read, v_write) = operator_params
        (qk_temperature, v_temperature,
         boundary_power, execution_prune_eps,
         qk_scale, v_scale) = controls
        token_count = int(global_state.shape[0])
        d_model = int(global_state.shape[1])
        n_spaces = int(space_read_proj.shape[0])
        n_bundles = n_spaces // bundle_size
        scan_blocks = (
            int(packing["token_id"].shape[0]) // token_block_size)
        qk_reads = jnp.stack((q_read, k_read), axis=0)
        qk_writes = jnp.stack((q_write, k_write), axis=0)
        qk_tau_kernels = jnp.stack(
            (q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_biases = jnp.stack(
            (q_tau_bias, k_tau_bias), axis=0)
        route_scales = jnp.asarray(
            (qk_scale, qk_scale, v_scale), dtype=jnp.float32
        ).reshape((3, 1, 1, 1))
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
                packed_state = global_state[block_token]
                projection = jax.lax.dynamic_slice_in_dim(
                    space_read_proj, space_start, bundle_size, axis=0)
                local = projection_dot(
                    "bd,mdr->mbr", packed_state, projection
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                token_space_valid = (
                    jnp.swapaxes(block_membership, 0, 1)
                    & block_valid[None, :])

                block_qk_tau_kernels = jax.lax.dynamic_slice_in_dim(
                    qk_tau_kernels, space_start, bundle_size, axis=1)
                block_qk_tau_biases = jax.lax.dynamic_slice_in_dim(
                    qk_tau_biases, space_start, bundle_size, axis=1)
                qk_raw_tau = (
                    _control_einsum_f32(
                        "mbr,amri->ambi", local, block_qk_tau_kernels)
                    + block_qk_tau_biases[:, :, None, :])
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

                block_v_tau_kernel = jax.lax.dynamic_slice_in_dim(
                    v_tau_kernel, space_start, bundle_size, axis=0)
                block_v_tau_bias = jax.lax.dynamic_slice_in_dim(
                    v_tau_bias, space_start, bundle_size, axis=0)
                v_raw_tau = _apply_operation_space_tau_map(
                    local, block_v_tau_kernel, block_v_tau_bias)
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
                grouped_raw_out = ad_checkpoint.checkpoint_name(
                    grouped_raw_out,
                    name=_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME)
                grouped_gate_mass = ad_checkpoint.checkpoint_name(
                    grouped_gate_mass,
                    name=_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)
                _, grouped_gate_den = _global_dense_rw_den_sharded(
                    grouped_gate_mass, den_powers, composition_mode)
                local_results = (
                    grouped_raw_out / grouped_gate_den).astype(jnp.float32)
                weights = jnp.swapaxes(
                    block_weight, 0, 1)[None, ..., None]
                writeback = jax.lax.dynamic_slice_in_dim(
                    space_write_proj,
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

        scan_step = jax.checkpoint(
            block_step,
            prevent_cse=False,
            policy=_BUNDLE_BLOCK_CHECKPOINT_POLICY)
        local_grouped_output, _ = jax.lax.scan(
            scan_step, initial_output, jnp.arange(scan_blocks))
        return _psum_dense_rw_representation_sharded(
            local_grouped_output, output_collective_bf16)

    def exact_space_execute(
            global_state,
            state_params,
            operator_params,
            controls,
            packed_metadata,
            primary_weight,
            overflow_weight):
        (space_read_proj, space_write_proj,
         q_tau_kernel, q_tau_bias,
         k_tau_kernel, k_tau_bias,
         v_tau_kernel, v_tau_bias) = state_params
        (q_read, q_write, k_read, k_write,
         v_read, v_write) = operator_params
        (qk_temperature, v_temperature,
         boundary_power, execution_prune_eps,
         qk_scale, v_scale) = controls
        token_count = int(global_state.shape[0])
        d_model = int(global_state.shape[1])
        n_spaces = int(space_read_proj.shape[0])
        task_group_size = _EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE
        task_capacity = int(
            packed_metadata["overflow_task_space_id"].shape[0])
        task_groups = task_capacity // task_group_size
        qk_reads = jnp.stack((q_read, k_read), axis=0)
        qk_writes = jnp.stack((q_write, k_write), axis=0)
        qk_tau_kernels = jnp.stack(
            (q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_biases = jnp.stack(
            (q_tau_bias, k_tau_bias), axis=0)
        route_scales = jnp.asarray(
            (qk_scale, qk_scale, v_scale), dtype=jnp.float32
        ).reshape((3, 1, 1))
        primary_token = packed_metadata["primary_token_id"]
        primary_valid = packed_metadata["primary_token_valid"]
        primary_state = global_state[primary_token]
        primary_local = projection_dot(
            "mtd,mdr->mtr", primary_state, space_read_proj
        ).astype(jnp.float32)
        primary_local_norm = jnp.maximum(
            jnp.linalg.norm(primary_local, axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))

        primary_qk_tau = (
            _control_einsum_f32(
                "mtr,amri->amti", primary_local, qk_tau_kernels)
            + qk_tau_biases[:, :, None, :])
        qk_raw_out, qk_gate_mass = _dense_rw_output_sharded(
            primary_local,
            primary_local_norm,
            primary_valid,
            qk_reads,
            qk_writes,
            primary_qk_tau,
            max_chunk_size=max_chunk_size_qk,
            soft_gate_temperature=qk_temperature,
            soft_gate_boundary_power=boundary_power,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            effective_active_eps=soft_gate_effective_active_eps,
            throughput_bf16=throughput_bf16)

        primary_v_tau = _apply_operation_space_tau_map(
            primary_local, v_tau_kernel, v_tau_bias)
        v_raw_out, v_gate_mass = _dense_rw_output_sharded(
            primary_local,
            primary_local_norm,
            primary_valid,
            v_read[None, ...],
            v_write[None, ...],
            primary_v_tau[None, ...],
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
        grouped_raw_out = ad_checkpoint.checkpoint_name(
            grouped_raw_out,
            name=_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME)
        grouped_gate_mass = ad_checkpoint.checkpoint_name(
            grouped_gate_mass,
            name=_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)
        _, grouped_gate_den = _global_dense_rw_den_sharded(
            grouped_gate_mass, den_powers, composition_mode)
        primary_results = (
            grouped_raw_out / grouped_gate_den).astype(jnp.float32)
        primary_block_output = writeback_dot(
            "amtr,mrd->amtd",
            primary_results
            * primary_weight[None, ..., None]
            * route_scales[:, None, ...],
            space_write_proj).astype(jnp.float32)
        primary_block_output = jnp.where(
            primary_valid[None, ..., None],
            primary_block_output,
            0.0)
        initial_output = jnp.zeros(
            (3, token_count, d_model), dtype=jnp.float32
        ).at[:, primary_token, :].add(primary_block_output)

        def task_group_step(local_output, group_index):
            task_start = group_index * task_group_size
            task_space = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_space_id"],
                task_start, task_group_size, axis=0)
            task_token = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_id"],
                task_start, task_group_size, axis=0)
            task_valid = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_valid"],
                task_start, task_group_size, axis=0)
            task_weight = jax.lax.dynamic_slice_in_dim(
                overflow_weight,
                task_start, task_group_size, axis=0)
            safe_task_space = jnp.minimum(
                task_space, jnp.int32(n_spaces - 1))

            def execute_group(output_value):
                packed_state = global_state[task_token]
                projection = space_read_proj[safe_task_space]
                local = projection_dot(
                    "gtd,gdr->gtr", packed_state, projection
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))

                task_qk_tau_kernels = qk_tau_kernels[:, safe_task_space]
                task_qk_tau_biases = qk_tau_biases[:, safe_task_space]
                qk_raw_tau = (
                    _control_einsum_f32(
                        "gtr,agri->agti", local, task_qk_tau_kernels)
                    + task_qk_tau_biases[:, :, None, :])
                task_qk_read = qk_reads[:, safe_task_space]
                task_qk_write = qk_writes[:, safe_task_space]
                qk_raw_out, qk_gate_mass = _dense_rw_output_sharded(
                    local,
                    local_norm,
                    task_valid,
                    task_qk_read,
                    task_qk_write,
                    qk_raw_tau,
                    max_chunk_size=max_chunk_size_qk,
                    soft_gate_temperature=qk_temperature,
                    soft_gate_boundary_power=boundary_power,
                    execution_prune_eps=execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)

                task_v_tau_kernel = v_tau_kernel[safe_task_space]
                task_v_tau_bias = v_tau_bias[safe_task_space]
                v_raw_tau = (
                    _control_einsum_f32(
                        "gtr,gri->gti", local, task_v_tau_kernel)
                    + task_v_tau_bias[:, None, :])
                task_v_read = v_read[safe_task_space]
                task_v_write = v_write[safe_task_space]
                v_raw_out, v_gate_mass = _dense_rw_output_sharded(
                    local,
                    local_norm,
                    task_valid,
                    task_v_read[None, ...],
                    task_v_write[None, ...],
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
                grouped_raw_out = ad_checkpoint.checkpoint_name(
                    grouped_raw_out,
                    name=_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME)
                grouped_gate_mass = ad_checkpoint.checkpoint_name(
                    grouped_gate_mass,
                    name=_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)
                _, grouped_gate_den = _global_dense_rw_den_sharded(
                    grouped_gate_mass, den_powers, composition_mode)
                local_results = (
                    grouped_raw_out / grouped_gate_den).astype(jnp.float32)
                writeback = space_write_proj[safe_task_space]
                block_output = writeback_dot(
                    "agtr,grd->agtd",
                    local_results
                    * task_weight[None, ..., None]
                    * route_scales[:, None, ...],
                    writeback).astype(jnp.float32)
                block_output = jnp.where(
                    task_valid[None, ..., None], block_output, 0.0)
                return output_value.at[
                    :, task_token, :].add(block_output)

            return jax.lax.cond(
                jnp.any(task_valid),
                execute_group,
                lambda output_value: output_value,
                local_output), None

        scan_step = jax.checkpoint(
            task_group_step,
            prevent_cse=False,
            policy=_BUNDLE_BLOCK_CHECKPOINT_POLICY)
        def execute_overflow(output_value):
            return jax.lax.scan(
                scan_step, output_value, jnp.arange(task_groups))[0]

        local_grouped_output = jax.lax.cond(
            jnp.any(packed_metadata["overflow_task_token_valid"]),
            execute_overflow,
            lambda output_value: output_value,
            initial_output)
        return _psum_dense_rw_representation_sharded(
            local_grouped_output, output_collective_bf16)

    def exact_space_linear_analytic_pullback(
            global_state,
            state_params,
            operator_params,
            controls,
            packed_metadata,
            primary_weight,
            overflow_weight,
            grouped_output_cotangent):
        """Pull back exact Q/K/V P-RW-U groups with explicit outer algebra."""
        (space_read_proj, space_write_proj,
         q_tau_kernel, q_tau_bias,
         k_tau_kernel, k_tau_bias,
         v_tau_kernel, v_tau_bias) = state_params
        (q_read, q_write, k_read, k_write,
         v_read, v_write) = operator_params
        token_count = int(global_state.shape[0])
        n_spaces = int(space_read_proj.shape[0])
        task_group_size = _EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE
        task_capacity = int(
            packed_metadata["overflow_task_space_id"].shape[0])
        task_groups = task_capacity // task_group_size
        qk_tau_kernels = jnp.stack(
            (q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_biases = jnp.stack(
            (q_tau_bias, k_tau_bias), axis=0)
        qk_reads = jnp.stack((q_read, k_read), axis=0)
        qk_writes = jnp.stack((q_write, k_write), axis=0)
        local_output_cotangent = (
            _psum_dense_rw_representation_sharded(
                grouped_output_cotangent,
                output_collective_bf16))

        primary_token = packed_metadata["primary_token_id"]
        primary_valid = packed_metadata["primary_token_valid"]
        primary_state = global_state[primary_token]
        primary_output_cotangent = jnp.take(
            local_output_cotangent, primary_token, axis=1)
        (primary_state_cotangent,
         primary_state_params_cotangent,
         primary_operator_params_cotangent,
         primary_controls_cotangent,
         primary_weight_cotangent) = (
            _attention_exact_space_linear_group_pullback(
                primary_state,
                primary_valid,
                primary_weight,
                (
                    space_read_proj,
                    space_write_proj,
                    qk_tau_kernels,
                    qk_tau_biases,
                    v_tau_kernel,
                    v_tau_bias,
                ),
                (
                    qk_reads,
                    qk_writes,
                    v_read,
                    v_write,
                ),
                controls,
                primary_output_cotangent,
                max_chunk_size_qk=max_chunk_size_qk,
                max_chunk_size_v=max_chunk_size_v,
                den_powers=den_powers,
                heat_kernel_beta=heat_kernel_beta,
                soft_gate_effective_active_eps=(
                    soft_gate_effective_active_eps),
                throughput_bf16=throughput_bf16))
        primary_state_cotangent = jnp.where(
            primary_valid[..., None],
            primary_state_cotangent,
            jnp.float32(0.0))
        global_state_cotangent = jnp.zeros_like(
            global_state, dtype=jnp.float32
        ).at[primary_token].add(primary_state_cotangent)
        (space_read_proj_cotangent,
         space_write_proj_cotangent,
         qk_tau_kernels_cotangent,
         qk_tau_biases_cotangent,
         v_tau_kernel_cotangent,
         v_tau_bias_cotangent) = primary_state_params_cotangent
        (qk_reads_cotangent,
         qk_writes_cotangent,
         v_read_cotangent,
         v_write_cotangent) = primary_operator_params_cotangent
        (qk_temperature_cotangent,
         v_temperature_cotangent,
         boundary_power_cotangent,
         execution_prune_eps_cotangent,
         qk_scale_cotangent,
         v_scale_cotangent) = primary_controls_cotangent
        overflow_weight_cotangent = jnp.zeros_like(
            overflow_weight, dtype=jnp.float32)

        initial_carry = (
            global_state_cotangent,
            space_read_proj_cotangent,
            space_write_proj_cotangent,
            qk_tau_kernels_cotangent,
            qk_tau_biases_cotangent,
            v_tau_kernel_cotangent,
            v_tau_bias_cotangent,
            qk_reads_cotangent,
            qk_writes_cotangent,
            v_read_cotangent,
            v_write_cotangent,
            qk_temperature_cotangent,
            v_temperature_cotangent,
            boundary_power_cotangent,
            execution_prune_eps_cotangent,
            qk_scale_cotangent,
            v_scale_cotangent,
            overflow_weight_cotangent,
        )

        def task_group_step(carry_value, group_index):
            task_start = group_index * task_group_size
            task_space = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_space_id"],
                task_start, task_group_size, axis=0)
            task_token = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_id"],
                task_start, task_group_size, axis=0)
            task_valid = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_valid"],
                task_start, task_group_size, axis=0)
            task_weight = jax.lax.dynamic_slice_in_dim(
                overflow_weight,
                task_start, task_group_size, axis=0)
            safe_task_space = jnp.minimum(
                task_space, jnp.int32(n_spaces - 1))

            def execute_group(accumulator):
                (global_state_ct,
                 space_read_proj_ct,
                 space_write_proj_ct,
                 qk_tau_kernels_ct,
                 qk_tau_biases_ct,
                 v_tau_kernel_ct,
                 v_tau_bias_ct,
                 qk_reads_ct,
                 qk_writes_ct,
                 v_read_ct,
                 v_write_ct,
                 qk_temperature_ct,
                 v_temperature_ct,
                 boundary_power_ct,
                 execution_prune_eps_ct,
                 qk_scale_ct,
                 v_scale_ct,
                 overflow_weight_ct) = accumulator
                task_state = global_state[task_token]
                task_output_cotangent = jnp.take(
                    local_output_cotangent, task_token, axis=1)
                (task_state_cotangent,
                 task_state_params_cotangent,
                 task_operator_params_cotangent,
                 task_controls_cotangent,
                 task_weight_cotangent) = (
                    _attention_exact_space_linear_group_pullback(
                        task_state,
                        task_valid,
                        task_weight,
                        (
                            space_read_proj[safe_task_space],
                            space_write_proj[safe_task_space],
                            qk_tau_kernels[:, safe_task_space],
                            qk_tau_biases[:, safe_task_space],
                            v_tau_kernel[safe_task_space],
                            v_tau_bias[safe_task_space],
                        ),
                        (
                            qk_reads[:, safe_task_space],
                            qk_writes[:, safe_task_space],
                            v_read[safe_task_space],
                            v_write[safe_task_space],
                        ),
                        controls,
                        task_output_cotangent,
                        max_chunk_size_qk=max_chunk_size_qk,
                        max_chunk_size_v=max_chunk_size_v,
                        den_powers=den_powers,
                        heat_kernel_beta=heat_kernel_beta,
                        soft_gate_effective_active_eps=(
                            soft_gate_effective_active_eps),
                        throughput_bf16=throughput_bf16))
                (task_space_read_proj_ct,
                 task_space_write_proj_ct,
                 task_qk_tau_kernels_ct,
                 task_qk_tau_biases_ct,
                 task_v_tau_kernel_ct,
                 task_v_tau_bias_ct) = (
                    task_state_params_cotangent)
                (task_qk_reads_ct,
                 task_qk_writes_ct,
                 task_v_read_ct,
                 task_v_write_ct) = task_operator_params_cotangent
                (task_qk_temperature_ct,
                 task_v_temperature_ct,
                 task_boundary_power_ct,
                 task_execution_prune_eps_ct,
                 task_qk_scale_ct,
                 task_v_scale_ct) = task_controls_cotangent
                task_active = jnp.any(task_valid, axis=1)
                task_state_cotangent = jnp.where(
                    task_valid[..., None],
                    task_state_cotangent,
                    jnp.float32(0.0))
                space_matrix_mask = task_active[:, None, None]
                qk_space_matrix_mask = task_active[None, :, None, None]
                qk_space_bias_mask = task_active[None, :, None]
                task_space_read_proj_ct = jnp.where(
                    space_matrix_mask,
                    task_space_read_proj_ct,
                    jnp.float32(0.0))
                task_space_write_proj_ct = jnp.where(
                    space_matrix_mask,
                    task_space_write_proj_ct,
                    jnp.float32(0.0))
                task_qk_tau_kernels_ct = jnp.where(
                    qk_space_matrix_mask,
                    task_qk_tau_kernels_ct,
                    jnp.float32(0.0))
                task_qk_tau_biases_ct = jnp.where(
                    qk_space_bias_mask,
                    task_qk_tau_biases_ct,
                    jnp.float32(0.0))
                task_v_tau_kernel_ct = jnp.where(
                    space_matrix_mask,
                    task_v_tau_kernel_ct,
                    jnp.float32(0.0))
                task_v_tau_bias_ct = jnp.where(
                    task_active[:, None],
                    task_v_tau_bias_ct,
                    jnp.float32(0.0))
                task_qk_reads_ct = jnp.where(
                    qk_space_matrix_mask,
                    task_qk_reads_ct,
                    jnp.float32(0.0))
                task_qk_writes_ct = jnp.where(
                    qk_space_matrix_mask,
                    task_qk_writes_ct,
                    jnp.float32(0.0))
                task_v_read_ct = jnp.where(
                    space_matrix_mask,
                    task_v_read_ct,
                    jnp.float32(0.0))
                task_v_write_ct = jnp.where(
                    space_matrix_mask,
                    task_v_write_ct,
                    jnp.float32(0.0))
                task_weight_cotangent = jnp.where(
                    task_valid,
                    task_weight_cotangent,
                    jnp.float32(0.0))

                global_state_ct = global_state_ct.at[
                    task_token
                ].add(task_state_cotangent)
                space_read_proj_ct = space_read_proj_ct.at[
                    safe_task_space
                ].add(task_space_read_proj_ct)
                space_write_proj_ct = space_write_proj_ct.at[
                    safe_task_space
                ].add(task_space_write_proj_ct)
                qk_tau_kernels_ct = qk_tau_kernels_ct.at[
                    :, safe_task_space
                ].add(task_qk_tau_kernels_ct)
                qk_tau_biases_ct = qk_tau_biases_ct.at[
                    :, safe_task_space
                ].add(task_qk_tau_biases_ct)
                v_tau_kernel_ct = v_tau_kernel_ct.at[
                    safe_task_space
                ].add(task_v_tau_kernel_ct)
                v_tau_bias_ct = v_tau_bias_ct.at[
                    safe_task_space
                ].add(task_v_tau_bias_ct)
                qk_reads_ct = qk_reads_ct.at[
                    :, safe_task_space
                ].add(task_qk_reads_ct)
                qk_writes_ct = qk_writes_ct.at[
                    :, safe_task_space
                ].add(task_qk_writes_ct)
                v_read_ct = v_read_ct.at[
                    safe_task_space
                ].add(task_v_read_ct)
                v_write_ct = v_write_ct.at[
                    safe_task_space
                ].add(task_v_write_ct)
                overflow_weight_ct = (
                    jax.lax.dynamic_update_slice_in_dim(
                        overflow_weight_ct,
                        task_weight_cotangent,
                        task_start,
                        axis=0))
                return (
                    global_state_ct,
                    space_read_proj_ct,
                    space_write_proj_ct,
                    qk_tau_kernels_ct,
                    qk_tau_biases_ct,
                    v_tau_kernel_ct,
                    v_tau_bias_ct,
                    qk_reads_ct,
                    qk_writes_ct,
                    v_read_ct,
                    v_write_ct,
                    qk_temperature_ct + task_qk_temperature_ct,
                    v_temperature_ct + task_v_temperature_ct,
                    boundary_power_ct + task_boundary_power_ct,
                    execution_prune_eps_ct
                    + task_execution_prune_eps_ct,
                    qk_scale_ct + task_qk_scale_ct,
                    v_scale_ct + task_v_scale_ct,
                    overflow_weight_ct,
                )

            return jax.lax.cond(
                jnp.any(task_valid),
                execute_group,
                lambda accumulator: accumulator,
                carry_value), None

        def execute_overflow(carry_value):
            return jax.lax.scan(
                task_group_step,
                carry_value,
                jnp.arange(task_groups))[0]

        final_carry = jax.lax.cond(
            jnp.any(packed_metadata["overflow_task_token_valid"]),
            execute_overflow,
            lambda carry_value: carry_value,
            initial_carry)
        (global_state_cotangent,
         space_read_proj_cotangent,
         space_write_proj_cotangent,
         qk_tau_kernels_cotangent,
         qk_tau_biases_cotangent,
         v_tau_kernel_cotangent,
         v_tau_bias_cotangent,
         qk_reads_cotangent,
         qk_writes_cotangent,
         v_read_cotangent,
         v_write_cotangent,
         qk_temperature_cotangent,
         v_temperature_cotangent,
         boundary_power_cotangent,
         execution_prune_eps_cotangent,
         qk_scale_cotangent,
         v_scale_cotangent,
         overflow_weight_cotangent) = final_carry
        state_params_cotangent = (
            space_read_proj_cotangent,
            space_write_proj_cotangent,
            qk_tau_kernels_cotangent[0],
            qk_tau_biases_cotangent[0],
            qk_tau_kernels_cotangent[1],
            qk_tau_biases_cotangent[1],
            v_tau_kernel_cotangent,
            v_tau_bias_cotangent,
        )
        operator_params_cotangent = (
            qk_reads_cotangent[0],
            qk_writes_cotangent[0],
            qk_reads_cotangent[1],
            qk_writes_cotangent[1],
            v_read_cotangent,
            v_write_cotangent,
        )
        controls_cotangent = (
            qk_temperature_cotangent,
            v_temperature_cotangent,
            boundary_power_cotangent,
            execution_prune_eps_cotangent,
            qk_scale_cotangent,
            v_scale_cotangent,
        )
        return (
            global_state_cotangent,
            state_params_cotangent,
            operator_params_cotangent,
            controls_cotangent,
            primary_weight_cotangent,
            overflow_weight_cotangent,
        )

    @jax.custom_vjp
    def execute_with_exact_space_backward(
            flat_state,
            state_params,
            operator_params,
            controls,
            packing,
            selected_ids,
            dense_space_weights):
        del selected_ids, dense_space_weights
        return bundle_execute(
            flat_state, state_params, operator_params, controls, packing)

    def execute_with_exact_space_backward_fwd(
            flat_state,
            state_params,
            operator_params,
            controls,
            packing,
            selected_ids,
            dense_space_weights):
        grouped_output = bundle_execute(
            flat_state, state_params, operator_params, controls, packing)
        selected_weights = jnp.take_along_axis(
            dense_space_weights, selected_ids, axis=1)
        residual = (
            flat_state,
            state_params,
            operator_params,
            controls,
            selected_ids,
            selected_weights,
        )
        return grouped_output, residual

    def execute_with_exact_space_backward_bwd(
            residual,
            grouped_output_cotangent):
        (flat_state,
         state_params,
         operator_params,
         controls,
         selected_ids,
         selected_weights) = residual
        token_count = int(flat_state.shape[0])
        n_spaces = int(state_params[0].shape[0])
        bucket_capacity = min(
            _EXACT_SPACE_BACKWARD_BUCKET_CAPACITY, token_count)
        packed = _prefix_pack_top2_space_buckets(
            selected_ids, selected_weights,
            operation_space_count=n_spaces,
            bucket_capacity=bucket_capacity,
            task_group_size=_EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE)
        packed_metadata = {
            "primary_token_id": packed["primary_token_id"],
            "primary_token_valid": packed["primary_token_valid"],
            "overflow_task_space_id": packed["overflow_task_space_id"],
            "overflow_task_token_id": packed["overflow_task_token_id"],
            "overflow_task_token_valid": packed[
                "overflow_task_token_valid"],
        }

        def exact_primal(
                exact_state,
                exact_state_params,
                exact_operator_params,
                exact_controls,
                exact_primary_weight,
                exact_overflow_weight):
            return exact_space_execute(
                exact_state,
                exact_state_params,
                exact_operator_params,
                exact_controls,
                packed_metadata,
                exact_primary_weight,
                exact_overflow_weight)

        if composition_mode == "linear_angular":
            (flat_state_grad,
             state_params_grad,
             operator_params_grad,
             controls_grad,
             primary_weight_grad,
             overflow_weight_grad) = (
                exact_space_linear_analytic_pullback(
                    flat_state,
                    state_params,
                    operator_params,
                    controls,
                    packed_metadata,
                    packed["primary_routing_weight"],
                    packed["overflow_task_routing_weight"],
                    grouped_output_cotangent))
        else:
            _, pullback = jax.vjp(
                exact_primal,
                flat_state,
                state_params,
                operator_params,
                controls,
                packed["primary_routing_weight"],
                packed["overflow_task_routing_weight"])
            (flat_state_grad,
             state_params_grad,
             operator_params_grad,
             controls_grad,
             primary_weight_grad,
             overflow_weight_grad) = pullback(
                 grouped_output_cotangent)
        dense_space_weights_grad = (
            _exact_space_bucket_weight_cotangents_to_dense(
                packed,
                primary_weight_grad,
                overflow_weight_grad,
                token_count=token_count,
                operation_space_count=n_spaces))
        return (
            flat_state_grad,
            state_params_grad,
            operator_params_grad,
            controls_grad,
            None,
            None,
            dense_space_weights_grad,
        )

    execute_with_exact_space_backward.defvjp(
        execute_with_exact_space_backward_fwd,
        execute_with_exact_space_backward_bwd)

    def attention_core(
            global_state,
            space_query_kernel, operation_space_keys,
            space_read_proj, space_write_proj,
            q_tau_kernel, q_tau_bias,
            k_tau_kernel, k_tau_bias,
            v_tau_kernel, v_tau_bias,
            q_read, q_write, k_read, k_write, v_read, v_write,
            qk_temperature, v_temperature,
            boundary_power, execution_prune_eps,
            qk_scale, v_scale, collect_metrics):
        routing = _select_operation_spaces(
            global_state, space_query_kernel, operation_space_keys, top_k)
        packing, packing_metrics = _pack_top2_bundle_entries_sharded(
            routing,
            bundle_size=bundle_size,
            token_block_size=token_block_size,
            stage="attention")
        token_count = int(global_state.shape[0])
        n_spaces = int(space_read_proj.shape[0])
        if n_spaces % bundle_size:
            raise ValueError(
                "bundle_dense attention space count is not divisible by 4")
        n_bundles = n_spaces // bundle_size
        qk_reads = jnp.stack((q_read, k_read), axis=0)
        qk_tau_kernels = jnp.stack(
            (q_tau_kernel, k_tau_kernel), axis=0)
        qk_tau_biases = jnp.stack(
            (q_tau_bias, k_tau_bias), axis=0)
        state_params = (
            space_read_proj, space_write_proj,
            q_tau_kernel, q_tau_bias,
            k_tau_kernel, k_tau_bias,
            v_tau_kernel, v_tau_bias,
        )
        operator_params = (
            q_read, q_write, k_read, k_write, v_read, v_write)
        controls = (
            qk_temperature, v_temperature,
            boundary_power, execution_prune_eps,
            qk_scale, v_scale,
        )
        grouped_output = execute_with_exact_space_backward(
            global_state,
            state_params,
            operator_params,
            controls,
            packing,
            routing["selected_ids"].astype(jnp.int32),
            routing["dense_space_weights"].astype(jnp.float32))

        q_per_space = int(q_read.shape[1]) * model_axis_size
        k_per_space = int(k_read.shape[1]) * model_axis_size
        v_per_space = int(v_read.shape[1]) * model_axis_size
        route_specs = (
            ("q", q_per_space),
            ("k", k_per_space),
            ("v", v_per_space),
        )
        metric_operands = (
            jax.lax.stop_gradient(global_state),
            jax.lax.stop_gradient(space_read_proj),
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
                qk_tau_kernel_block = jax.lax.dynamic_slice_in_dim(
                    metric_qk_tau_kernels,
                    space_start, bundle_size, axis=1)
                qk_tau_bias_block = jax.lax.dynamic_slice_in_dim(
                    metric_qk_tau_biases,
                    space_start, bundle_size, axis=1)
                qk_tau = (
                    _control_einsum_f32(
                        "mbr,amri->ambi",
                        local, qk_tau_kernel_block)
                    + qk_tau_bias_block[:, :, None, :])
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

                v_tau_kernel_block = jax.lax.dynamic_slice_in_dim(
                    metric_v_tau_kernel,
                    space_start, bundle_size, axis=0)
                v_tau_bias_block = jax.lax.dynamic_slice_in_dim(
                    metric_v_tau_bias,
                    space_start, bundle_size, axis=0)
                v_tau = _apply_operation_space_tau_map(
                    local, v_tau_kernel_block, v_tau_bias_block)
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
    kernel._v4175_kernel_profile = "production"
    kernel._v4175_dense_grouped_execution = "attention_qkv_bundle4"
    kernel._v4175_execution_mode = "bundle_dense"
    kernel._v4175_bundle_size = bundle_size
    kernel._v4175_bundle_token_block_size = token_block_size
    kernel._v4175_packed_entry_capacity = (
        "2*T + n_bundles*(token_block_size-1)")
    kernel._v4175_bundle_packing = "prefix_count_exact"
    kernel._v4175_bundle_packing_vjp = "saved_metadata_scatter"
    kernel._v4175_attention_packing_count = 1
    kernel._v4175_qk_paired = True
    kernel._v4175_dynamic_metric_flag = True
    kernel._v4175_chunk_remat_policy = "always"
    kernel._v4175_block_remat_policy = (
        "dots_and_compact_rw_outputs_saveable")
    kernel._v4175_backward_execution_mode = (
        "exact_top2_space_batched_dense_with_overflow")
    kernel._v4175_backward_bucket_capacity = (
        _EXACT_SPACE_BACKWARD_BUCKET_CAPACITY)
    kernel._v4175_backward_overflow_task_group_size = (
        _EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE)
    kernel._v4175_backward_packed_entry_capacity = (
        "n_spaces*min(T,3072) + compact exact overflow tasks")
    kernel._v4175_qkv_shared_projection_vjp = True
    kernel._v4175_attention_p_rw_u_pullback = (
        "linear_analytic_exact_group")
    kernel._v4175_attention_p_rw_u_generic_fallback = (
        "non_linear_composition_modes")
    kernel._v4175_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4175_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4175_output_contract = (
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
        for suffix in _SPACE_ROUTING_COMPACT_METRIC_SUFFIXES)

    def rst_core(
            global_state,
            space_query_kernel, operation_space_keys,
            space_read_proj, space_write_proj,
            tau_kernel, tau_bias,
            read_vectors, write_vectors,
            temperature, boundary_power, execution_prune_eps,
            route_scale, collect_metrics):
        routing = _select_operation_spaces(
            global_state, space_query_kernel, operation_space_keys, top_k)
        if throughput_bf16:
            space_states = jnp.swapaxes(
                throughput_dot_bf16_f32(
                    global_state, space_read_proj,
                    dimension_numbers=(
                        ((1,), (1,)),
                        ((), ()))),
                0, 1)
        else:
            space_states = _control_einsum_f32(
                "td,mdr->mtr", global_state, space_read_proj)
        space_states = space_states.astype(jnp.float32)
        space_state_norm = jnp.maximum(
            jnp.linalg.norm(space_states, axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))
        raw_tau = _apply_operation_space_tau_map(
            space_states, tau_kernel, tau_bias)
        space_output = _dense_rw_output_sharded(
            space_states, space_state_norm,
            jnp.ones(space_states.shape[:2], dtype=jnp.bool_),
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
        raw_out, gate_mass = space_output
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
            space_write_proj).astype(jnp.float32)
        update = _psum_dense_rw_representation_sharded(
            local_update, output_collective_bf16)
        n_per_space = int(read_vectors.shape[1]) * model_axis_size
        route_specs = (("rst", n_per_space),)
        metric_local = jax.lax.stop_gradient(space_states)
        metric_norm = jax.lax.stop_gradient(space_state_norm)
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
            jnp.ones(space_states.shape[:2], dtype=jnp.bool_),
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
    kernel._v4175_kernel_profile = "production"
    kernel._v4175_dense_grouped_execution = "rst_end_to_end"
    kernel._v4175_execution_mode = "dense_all_space"
    kernel._v4175_dynamic_metric_flag = True
    kernel._v4175_chunk_remat_policy = "always"
    kernel._v4175_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4175_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4175_output_contract = ("rst[T,D]", "scalars")
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
        for suffix in _SPACE_ROUTING_COMPACT_METRIC_SUFFIXES)
    bundle_metric_names = _bundle_packing_metric_names("rst")
    den_power = jnp.float32(
        admission_den_power).reshape((1, 1, 1, 1))
    projection_dot = (
        _throughput_einsum_bf16_f32
        if throughput_bf16 else _control_einsum_f32)
    writeback_dot = projection_dot

    def bundle_execute(
            global_state,
            state_params,
            operator_params,
            controls,
            packing):
        (space_read_proj, space_write_proj,
         tau_kernel, tau_bias) = state_params
        read_vectors, write_vectors = operator_params
        (temperature, boundary_power,
         execution_prune_eps, route_scale) = controls
        token_count = int(global_state.shape[0])
        d_model = int(global_state.shape[1])
        n_spaces = int(space_read_proj.shape[0])
        n_bundles = n_spaces // bundle_size
        scan_blocks = (
            int(packing["token_id"].shape[0]) // token_block_size)
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
                packed_state = global_state[block_token]
                projection = jax.lax.dynamic_slice_in_dim(
                    space_read_proj, space_start, bundle_size, axis=0)
                local = projection_dot(
                    "bd,mdr->mbr", packed_state, projection
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                token_space_valid = (
                    jnp.swapaxes(block_membership, 0, 1)
                    & block_valid[None, :])
                block_tau_kernel = jax.lax.dynamic_slice_in_dim(
                    tau_kernel, space_start, bundle_size, axis=0)
                block_tau_bias = jax.lax.dynamic_slice_in_dim(
                    tau_bias, space_start, bundle_size, axis=0)
                raw_tau = _apply_operation_space_tau_map(
                    local, block_tau_kernel, block_tau_bias)
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
                raw_out = ad_checkpoint.checkpoint_name(
                    raw_out, name=_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME)
                gate_mass = ad_checkpoint.checkpoint_name(
                    gate_mass, name=_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)
                _, gate_den = _global_dense_rw_den_sharded(
                    gate_mass, den_power, composition_mode)
                local_results = (raw_out / gate_den).astype(jnp.float32)
                weights = jnp.swapaxes(
                    block_weight, 0, 1)[None, ..., None]
                writeback = jax.lax.dynamic_slice_in_dim(
                    space_write_proj,
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

        scan_step = jax.checkpoint(
            block_step,
            prevent_cse=False,
            policy=_BUNDLE_BLOCK_CHECKPOINT_POLICY)
        local_update, _ = jax.lax.scan(
            scan_step, initial_output, jnp.arange(scan_blocks))
        return _psum_dense_rw_representation_sharded(
            local_update, output_collective_bf16)

    def exact_space_execute(
            global_state,
            state_params,
            operator_params,
            controls,
            packed_metadata,
            primary_weight,
            overflow_weight):
        (space_read_proj, space_write_proj,
         tau_kernel, tau_bias) = state_params
        read_vectors, write_vectors = operator_params
        (temperature, boundary_power,
         execution_prune_eps, route_scale) = controls
        token_count = int(global_state.shape[0])
        d_model = int(global_state.shape[1])
        n_spaces = int(space_read_proj.shape[0])
        task_group_size = _EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE
        task_capacity = int(
            packed_metadata["overflow_task_space_id"].shape[0])
        task_groups = task_capacity // task_group_size
        primary_token = packed_metadata["primary_token_id"]
        primary_valid = packed_metadata["primary_token_valid"]
        primary_state = global_state[primary_token]
        primary_local = projection_dot(
            "mtd,mdr->mtr", primary_state, space_read_proj
        ).astype(jnp.float32)
        primary_local_norm = jnp.maximum(
            jnp.linalg.norm(primary_local, axis=-1, keepdims=True),
            jnp.float32(RW_FORWARD_NORM_EPS))
        primary_tau = _apply_operation_space_tau_map(
            primary_local, tau_kernel, tau_bias)
        raw_out, gate_mass = _dense_rw_output_sharded(
            primary_local,
            primary_local_norm,
            primary_valid,
            read_vectors[None, ...],
            write_vectors[None, ...],
            primary_tau[None, ...],
            max_chunk_size=max_chunk_size,
            soft_gate_temperature=temperature,
            soft_gate_boundary_power=boundary_power,
            execution_prune_eps=execution_prune_eps,
            srw_composition_mode=composition_mode,
            heat_kernel_beta=heat_kernel_beta,
            effective_active_eps=soft_gate_effective_active_eps,
            throughput_bf16=throughput_bf16)
        raw_out = ad_checkpoint.checkpoint_name(
            raw_out, name=_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME)
        gate_mass = ad_checkpoint.checkpoint_name(
            gate_mass, name=_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)
        _, gate_den = _global_dense_rw_den_sharded(
            gate_mass, den_power, composition_mode)
        primary_results = (raw_out / gate_den).astype(jnp.float32)
        primary_block_output = writeback_dot(
            "mtr,mrd->mtd",
            primary_results[0]
            * primary_weight[..., None]
            * jnp.asarray(route_scale, dtype=jnp.float32),
            space_write_proj).astype(jnp.float32)
        primary_block_output = jnp.where(
            primary_valid[..., None], primary_block_output, 0.0)
        initial_output = jnp.zeros(
            (token_count, d_model), dtype=jnp.float32
        ).at[primary_token, :].add(primary_block_output)

        def task_group_step(local_output, group_index):
            task_start = group_index * task_group_size
            task_space = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_space_id"],
                task_start, task_group_size, axis=0)
            task_token = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_id"],
                task_start, task_group_size, axis=0)
            task_valid = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_valid"],
                task_start, task_group_size, axis=0)
            task_weight = jax.lax.dynamic_slice_in_dim(
                overflow_weight,
                task_start, task_group_size, axis=0)
            safe_task_space = jnp.minimum(
                task_space, jnp.int32(n_spaces - 1))

            def execute_group(output_value):
                packed_state = global_state[task_token]
                projection = space_read_proj[safe_task_space]
                local = projection_dot(
                    "gtd,gdr->gtr", packed_state, projection
                ).astype(jnp.float32)
                local_norm = jnp.maximum(
                    jnp.linalg.norm(local, axis=-1, keepdims=True),
                    jnp.float32(RW_FORWARD_NORM_EPS))
                task_tau_kernel = tau_kernel[safe_task_space]
                task_tau_bias = tau_bias[safe_task_space]
                raw_tau = (
                    _control_einsum_f32(
                        "gtr,gri->gti", local, task_tau_kernel)
                    + task_tau_bias[:, None, :])
                task_read = read_vectors[safe_task_space]
                task_write = write_vectors[safe_task_space]
                raw_out, gate_mass = _dense_rw_output_sharded(
                    local,
                    local_norm,
                    task_valid,
                    task_read[None, ...],
                    task_write[None, ...],
                    raw_tau[None, ...],
                    max_chunk_size=max_chunk_size,
                    soft_gate_temperature=temperature,
                    soft_gate_boundary_power=boundary_power,
                    execution_prune_eps=execution_prune_eps,
                    srw_composition_mode=composition_mode,
                    heat_kernel_beta=heat_kernel_beta,
                    effective_active_eps=soft_gate_effective_active_eps,
                    throughput_bf16=throughput_bf16)
                raw_out = ad_checkpoint.checkpoint_name(
                    raw_out, name=_BUNDLE_RW_RAW_OUT_CHECKPOINT_NAME)
                gate_mass = ad_checkpoint.checkpoint_name(
                    gate_mass, name=_BUNDLE_RW_GATE_MASS_CHECKPOINT_NAME)
                _, gate_den = _global_dense_rw_den_sharded(
                    gate_mass, den_power, composition_mode)
                local_results = (raw_out / gate_den).astype(jnp.float32)
                writeback = space_write_proj[safe_task_space]
                block_output = writeback_dot(
                    "gtr,grd->gtd",
                    local_results[0]
                    * task_weight[..., None]
                    * jnp.asarray(route_scale, dtype=jnp.float32),
                    writeback).astype(jnp.float32)
                block_output = jnp.where(
                    task_valid[..., None], block_output, 0.0)
                return output_value.at[
                    task_token, :].add(block_output)

            return jax.lax.cond(
                jnp.any(task_valid),
                execute_group,
                lambda output_value: output_value,
                local_output), None

        scan_step = jax.checkpoint(
            task_group_step,
            prevent_cse=False,
            policy=_BUNDLE_BLOCK_CHECKPOINT_POLICY)
        def execute_overflow(output_value):
            return jax.lax.scan(
                scan_step, output_value, jnp.arange(task_groups))[0]

        local_update = jax.lax.cond(
            jnp.any(packed_metadata["overflow_task_token_valid"]),
            execute_overflow,
            lambda output_value: output_value,
            initial_output)
        return _psum_dense_rw_representation_sharded(
            local_update, output_collective_bf16)

    def exact_space_linear_analytic_pullback(
            global_state,
            state_params,
            operator_params,
            controls,
            packed_metadata,
            primary_weight,
            overflow_weight,
            output_cotangent):
        """Pull back exact RST P-RW-U groups with explicit outer algebra."""
        (space_read_proj, space_write_proj,
         tau_kernel, tau_bias) = state_params
        read_vectors, write_vectors = operator_params
        token_count = int(global_state.shape[0])
        n_spaces = int(space_read_proj.shape[0])
        task_group_size = _EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE
        task_capacity = int(
            packed_metadata["overflow_task_space_id"].shape[0])
        task_groups = task_capacity // task_group_size
        local_output_cotangent = (
            _psum_dense_rw_representation_sharded(
                output_cotangent,
                output_collective_bf16))

        primary_token = packed_metadata["primary_token_id"]
        primary_valid = packed_metadata["primary_token_valid"]
        primary_state = global_state[primary_token]
        primary_output_cotangent = local_output_cotangent[primary_token]
        (primary_state_cotangent,
         primary_state_params_cotangent,
         primary_operator_params_cotangent,
         primary_controls_cotangent,
         primary_weight_cotangent) = (
            _rst_exact_space_linear_group_pullback(
                primary_state,
                primary_valid,
                primary_weight,
                (
                    space_read_proj,
                    space_write_proj,
                    tau_kernel,
                    tau_bias,
                ),
                (
                    read_vectors,
                    write_vectors,
                ),
                controls,
                primary_output_cotangent,
                max_chunk_size=max_chunk_size,
                den_power=den_power,
                heat_kernel_beta=heat_kernel_beta,
                soft_gate_effective_active_eps=(
                    soft_gate_effective_active_eps),
                throughput_bf16=throughput_bf16))
        primary_state_cotangent = jnp.where(
            primary_valid[..., None],
            primary_state_cotangent,
            jnp.float32(0.0))
        global_state_cotangent = jnp.zeros_like(
            global_state, dtype=jnp.float32
        ).at[primary_token].add(primary_state_cotangent)
        (space_read_proj_cotangent,
         space_write_proj_cotangent,
         tau_kernel_cotangent,
         tau_bias_cotangent) = primary_state_params_cotangent
        (read_vectors_cotangent,
         write_vectors_cotangent) = primary_operator_params_cotangent
        (temperature_cotangent,
         boundary_power_cotangent,
         execution_prune_eps_cotangent,
         route_scale_cotangent) = primary_controls_cotangent
        overflow_weight_cotangent = jnp.zeros_like(
            overflow_weight, dtype=jnp.float32)

        initial_carry = (
            global_state_cotangent,
            space_read_proj_cotangent,
            space_write_proj_cotangent,
            tau_kernel_cotangent,
            tau_bias_cotangent,
            read_vectors_cotangent,
            write_vectors_cotangent,
            temperature_cotangent,
            boundary_power_cotangent,
            execution_prune_eps_cotangent,
            route_scale_cotangent,
            overflow_weight_cotangent,
        )

        def task_group_step(carry_value, group_index):
            task_start = group_index * task_group_size
            task_space = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_space_id"],
                task_start, task_group_size, axis=0)
            task_token = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_id"],
                task_start, task_group_size, axis=0)
            task_valid = jax.lax.dynamic_slice_in_dim(
                packed_metadata["overflow_task_token_valid"],
                task_start, task_group_size, axis=0)
            task_weight = jax.lax.dynamic_slice_in_dim(
                overflow_weight,
                task_start, task_group_size, axis=0)
            safe_task_space = jnp.minimum(
                task_space, jnp.int32(n_spaces - 1))

            def execute_group(accumulator):
                (global_state_ct,
                 space_read_proj_ct,
                 space_write_proj_ct,
                 tau_kernel_ct,
                 tau_bias_ct,
                 read_vectors_ct,
                 write_vectors_ct,
                 temperature_ct,
                 boundary_power_ct,
                 execution_prune_eps_ct,
                 route_scale_ct,
                 overflow_weight_ct) = accumulator
                task_state = global_state[task_token]
                task_output_cotangent = local_output_cotangent[task_token]
                (task_state_cotangent,
                 task_state_params_cotangent,
                 task_operator_params_cotangent,
                 task_controls_cotangent,
                 task_weight_cotangent) = (
                    _rst_exact_space_linear_group_pullback(
                        task_state,
                        task_valid,
                        task_weight,
                        (
                            space_read_proj[safe_task_space],
                            space_write_proj[safe_task_space],
                            tau_kernel[safe_task_space],
                            tau_bias[safe_task_space],
                        ),
                        (
                            read_vectors[safe_task_space],
                            write_vectors[safe_task_space],
                        ),
                        controls,
                        task_output_cotangent,
                        max_chunk_size=max_chunk_size,
                        den_power=den_power,
                        heat_kernel_beta=heat_kernel_beta,
                        soft_gate_effective_active_eps=(
                            soft_gate_effective_active_eps),
                        throughput_bf16=throughput_bf16))
                (task_space_read_proj_ct,
                 task_space_write_proj_ct,
                 task_tau_kernel_ct,
                 task_tau_bias_ct) = task_state_params_cotangent
                (task_read_vectors_ct,
                 task_write_vectors_ct) = task_operator_params_cotangent
                (task_temperature_ct,
                 task_boundary_power_ct,
                 task_execution_prune_eps_ct,
                 task_route_scale_ct) = task_controls_cotangent
                task_active = jnp.any(task_valid, axis=1)
                task_state_cotangent = jnp.where(
                    task_valid[..., None],
                    task_state_cotangent,
                    jnp.float32(0.0))
                task_space_read_proj_ct = jnp.where(
                    task_active[:, None, None],
                    task_space_read_proj_ct,
                    jnp.float32(0.0))
                task_space_write_proj_ct = jnp.where(
                    task_active[:, None, None],
                    task_space_write_proj_ct,
                    jnp.float32(0.0))
                task_tau_kernel_ct = jnp.where(
                    task_active[:, None, None],
                    task_tau_kernel_ct,
                    jnp.float32(0.0))
                task_tau_bias_ct = jnp.where(
                    task_active[:, None],
                    task_tau_bias_ct,
                    jnp.float32(0.0))
                task_read_vectors_ct = jnp.where(
                    task_active[:, None, None],
                    task_read_vectors_ct,
                    jnp.float32(0.0))
                task_write_vectors_ct = jnp.where(
                    task_active[:, None, None],
                    task_write_vectors_ct,
                    jnp.float32(0.0))
                task_weight_cotangent = jnp.where(
                    task_valid,
                    task_weight_cotangent,
                    jnp.float32(0.0))

                global_state_ct = global_state_ct.at[
                    task_token
                ].add(task_state_cotangent)
                space_read_proj_ct = space_read_proj_ct.at[
                    safe_task_space
                ].add(task_space_read_proj_ct)
                space_write_proj_ct = space_write_proj_ct.at[
                    safe_task_space
                ].add(task_space_write_proj_ct)
                tau_kernel_ct = tau_kernel_ct.at[
                    safe_task_space
                ].add(task_tau_kernel_ct)
                tau_bias_ct = tau_bias_ct.at[
                    safe_task_space
                ].add(task_tau_bias_ct)
                read_vectors_ct = read_vectors_ct.at[
                    safe_task_space
                ].add(task_read_vectors_ct)
                write_vectors_ct = write_vectors_ct.at[
                    safe_task_space
                ].add(task_write_vectors_ct)
                overflow_weight_ct = (
                    jax.lax.dynamic_update_slice_in_dim(
                        overflow_weight_ct,
                        task_weight_cotangent,
                        task_start,
                        axis=0))
                return (
                    global_state_ct,
                    space_read_proj_ct,
                    space_write_proj_ct,
                    tau_kernel_ct,
                    tau_bias_ct,
                    read_vectors_ct,
                    write_vectors_ct,
                    temperature_ct + task_temperature_ct,
                    boundary_power_ct + task_boundary_power_ct,
                    execution_prune_eps_ct
                    + task_execution_prune_eps_ct,
                    route_scale_ct + task_route_scale_ct,
                    overflow_weight_ct,
                )

            return jax.lax.cond(
                jnp.any(task_valid),
                execute_group,
                lambda accumulator: accumulator,
                carry_value), None

        def execute_overflow(carry_value):
            return jax.lax.scan(
                task_group_step,
                carry_value,
                jnp.arange(task_groups))[0]

        final_carry = jax.lax.cond(
            jnp.any(packed_metadata["overflow_task_token_valid"]),
            execute_overflow,
            lambda carry_value: carry_value,
            initial_carry)
        (global_state_cotangent,
         space_read_proj_cotangent,
         space_write_proj_cotangent,
         tau_kernel_cotangent,
         tau_bias_cotangent,
         read_vectors_cotangent,
         write_vectors_cotangent,
         temperature_cotangent,
         boundary_power_cotangent,
         execution_prune_eps_cotangent,
         route_scale_cotangent,
         overflow_weight_cotangent) = final_carry
        state_params_cotangent = (
            space_read_proj_cotangent,
            space_write_proj_cotangent,
            tau_kernel_cotangent,
            tau_bias_cotangent,
        )
        operator_params_cotangent = (
            read_vectors_cotangent,
            write_vectors_cotangent,
        )
        controls_cotangent = (
            temperature_cotangent,
            boundary_power_cotangent,
            execution_prune_eps_cotangent,
            route_scale_cotangent,
        )
        return (
            global_state_cotangent,
            state_params_cotangent,
            operator_params_cotangent,
            controls_cotangent,
            primary_weight_cotangent,
            overflow_weight_cotangent,
        )

    @jax.custom_vjp
    def execute_with_exact_space_backward(
            flat_state,
            state_params,
            operator_params,
            controls,
            packing,
            selected_ids,
            dense_space_weights):
        del selected_ids, dense_space_weights
        return bundle_execute(
            flat_state, state_params, operator_params, controls, packing)

    def execute_with_exact_space_backward_fwd(
            flat_state,
            state_params,
            operator_params,
            controls,
            packing,
            selected_ids,
            dense_space_weights):
        update = bundle_execute(
            flat_state, state_params, operator_params, controls, packing)
        selected_weights = jnp.take_along_axis(
            dense_space_weights, selected_ids, axis=1)
        residual = (
            flat_state,
            state_params,
            operator_params,
            controls,
            selected_ids,
            selected_weights,
        )
        return update, residual

    def execute_with_exact_space_backward_bwd(
            residual,
            update_cotangent):
        (flat_state,
         state_params,
         operator_params,
         controls,
         selected_ids,
         selected_weights) = residual
        token_count = int(flat_state.shape[0])
        n_spaces = int(state_params[0].shape[0])
        bucket_capacity = min(
            _EXACT_SPACE_BACKWARD_BUCKET_CAPACITY, token_count)
        packed = _prefix_pack_top2_space_buckets(
            selected_ids, selected_weights,
            operation_space_count=n_spaces,
            bucket_capacity=bucket_capacity,
            task_group_size=_EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE)
        packed_metadata = {
            "primary_token_id": packed["primary_token_id"],
            "primary_token_valid": packed["primary_token_valid"],
            "overflow_task_space_id": packed["overflow_task_space_id"],
            "overflow_task_token_id": packed["overflow_task_token_id"],
            "overflow_task_token_valid": packed[
                "overflow_task_token_valid"],
        }

        def exact_primal(
                exact_state,
                exact_state_params,
                exact_operator_params,
                exact_controls,
                exact_primary_weight,
                exact_overflow_weight):
            return exact_space_execute(
                exact_state,
                exact_state_params,
                exact_operator_params,
                exact_controls,
                packed_metadata,
                exact_primary_weight,
                exact_overflow_weight)

        if composition_mode == "linear_angular":
            (flat_state_grad,
             state_params_grad,
             operator_params_grad,
             controls_grad,
             primary_weight_grad,
             overflow_weight_grad) = (
                exact_space_linear_analytic_pullback(
                    flat_state,
                    state_params,
                    operator_params,
                    controls,
                    packed_metadata,
                    packed["primary_routing_weight"],
                    packed["overflow_task_routing_weight"],
                    update_cotangent))
        else:
            _, pullback = jax.vjp(
                exact_primal,
                flat_state,
                state_params,
                operator_params,
                controls,
                packed["primary_routing_weight"],
                packed["overflow_task_routing_weight"])
            (flat_state_grad,
             state_params_grad,
             operator_params_grad,
             controls_grad,
             primary_weight_grad,
             overflow_weight_grad) = pullback(update_cotangent)
        dense_space_weights_grad = (
            _exact_space_bucket_weight_cotangents_to_dense(
                packed,
                primary_weight_grad,
                overflow_weight_grad,
                token_count=token_count,
                operation_space_count=n_spaces))
        return (
            flat_state_grad,
            state_params_grad,
            operator_params_grad,
            controls_grad,
            None,
            None,
            dense_space_weights_grad,
        )

    execute_with_exact_space_backward.defvjp(
        execute_with_exact_space_backward_fwd,
        execute_with_exact_space_backward_bwd)

    def rst_core(
            global_state,
            space_query_kernel, operation_space_keys,
            space_read_proj, space_write_proj,
            tau_kernel, tau_bias,
            read_vectors, write_vectors,
            temperature, boundary_power, execution_prune_eps,
            route_scale, collect_metrics):
        routing = _select_operation_spaces(
            global_state, space_query_kernel, operation_space_keys, top_k)
        packing, packing_metrics = _pack_top2_bundle_entries_sharded(
            routing,
            bundle_size=bundle_size,
            token_block_size=token_block_size,
            stage="rst")
        token_count = int(global_state.shape[0])
        n_spaces = int(space_read_proj.shape[0])
        if n_spaces % bundle_size:
            raise ValueError(
                "bundle_dense RST space count is not divisible by 4")
        n_bundles = n_spaces // bundle_size
        state_params = (
            space_read_proj, space_write_proj,
            tau_kernel, tau_bias,
        )
        operator_params = (read_vectors, write_vectors)
        controls = (
            temperature, boundary_power,
            execution_prune_eps, route_scale,
        )
        update = execute_with_exact_space_backward(
            global_state,
            state_params,
            operator_params,
            controls,
            packing,
            routing["selected_ids"].astype(jnp.int32),
            routing["dense_space_weights"].astype(jnp.float32))

        n_per_space = int(read_vectors.shape[1]) * model_axis_size
        route_specs = (("rst", n_per_space),)
        metric_operands = (
            jax.lax.stop_gradient(global_state),
            jax.lax.stop_gradient(space_read_proj),
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
                tau_kernel_block = jax.lax.dynamic_slice_in_dim(
                    metric_tau_kernel,
                    space_start, bundle_size, axis=0)
                tau_bias_block = jax.lax.dynamic_slice_in_dim(
                    metric_tau_bias,
                    space_start, bundle_size, axis=0)
                raw_tau = _apply_operation_space_tau_map(
                    local, tau_kernel_block, tau_bias_block)
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
    kernel._v4175_kernel_profile = "production"
    kernel._v4175_dense_grouped_execution = "rst_end_to_end_bundle4"
    kernel._v4175_execution_mode = "bundle_dense"
    kernel._v4175_bundle_size = bundle_size
    kernel._v4175_bundle_token_block_size = token_block_size
    kernel._v4175_packed_entry_capacity = (
        "2*T + n_bundles*(token_block_size-1)")
    kernel._v4175_bundle_packing = "prefix_count_exact"
    kernel._v4175_bundle_packing_vjp = "saved_metadata_scatter"
    kernel._v4175_rst_packing_count = 1
    kernel._v4175_dynamic_metric_flag = True
    kernel._v4175_chunk_remat_policy = "always"
    kernel._v4175_block_remat_policy = (
        "dots_and_compact_rw_outputs_saveable")
    kernel._v4175_backward_execution_mode = (
        "exact_top2_space_batched_dense_with_overflow")
    kernel._v4175_backward_bucket_capacity = (
        _EXACT_SPACE_BACKWARD_BUCKET_CAPACITY)
    kernel._v4175_backward_overflow_task_group_size = (
        _EXACT_SPACE_BACKWARD_TASK_GROUP_SIZE)
    kernel._v4175_backward_packed_entry_capacity = (
        "n_spaces*min(T,3072) + compact exact overflow tasks")
    kernel._v4175_rst_p_rw_u_pullback = (
        "linear_analytic_exact_group")
    kernel._v4175_rst_p_rw_u_generic_fallback = (
        "non_linear_composition_modes")
    kernel._v4175_throughput_precision = (
        "bf16_operands_f32_accum"
        if throughput_bf16 else "fp32_reference")
    kernel._v4175_output_collective_dtype = (
        "bf16" if output_collective_bf16 else "fp32")
    kernel._v4175_output_contract = ("rst[T,D]", "scalars")
    return kernel


def make_sharded_rst_space_bundle_dense_minimal(mesh, **kwargs):
    """Create the mixed-precision compact four-space RST executor."""
    return _make_sharded_rst_space_bundle_dense(
        mesh, throughput_bf16=True, output_collective_bf16=True, **kwargs)


def make_sharded_rst_space_bundle_dense_fp32_reference(mesh, **kwargs):
    """Create the compact FP32 RST executor for parity checks."""
    return _make_sharded_rst_space_bundle_dense(
        mesh, throughput_bf16=False, output_collective_bf16=False, **kwargs)


def _validate_v4175_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs):
    """Validate the route-separated reference executor used by v4175."""
    del (
        admission_den_power, srw_composition_mode,
        heat_kernel_beta, kwargs)
    if not isinstance(sharded_fns, dict):
        return
    profile = sharded_fns.get("_v4175_kernel_profile")
    if profile is not None and profile not in (
            "production", "production_diagnostics",
            "retention", "suppression", "trajectory"):
        raise ValueError(f"unsupported v4175 kernel profile {profile!r}")
    attention = sharded_fns.get("attention_space_dense")
    if attention is not None:
        raise ValueError(
            "v4175 does not accept the shared-routing fused attention executor")
    route_kernels = {
        route: sharded_fns.get(f"{route}_space_dense")
        for route in ROUTES
    }
    present = tuple(
        route for route, kernel in route_kernels.items()
        if kernel is not None)
    if not present:
        return
    missing = tuple(route for route in ROUTES if route not in present)
    if missing:
        raise ValueError(
            "v4175 requires independent dense route kernels for "
            + ", ".join(missing))


def get_model_version() -> str:
    return MODEL_VERSION
