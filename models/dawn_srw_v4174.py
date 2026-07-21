"""DAWN-SRW v4.1.7.4: shared operation addresses for every route.

The model starts from a shared state representation ``x`` in the state space
``[B, S, d_model]``.  Q, K, V, and RST compare independent route queries to
one learned operation-address codebook.  An address is not an operator and it
does not name a shared operation space: it is a latent index that connects a
route to an address-local projection, operator-bank slice, tau projection,
and state-writeback matrix.

For route ``r`` and address ``m`` the canonical computation is

``c_r(x) = C_r x``
``alpha_r(x) = TopKSoftmax(cos(c_r(x), A))``
``z_{r,m} = P_{r,m} x``
``u_{r,m} = RWCompose_{r,m}(z_{r,m})``
``y_r = sum_m alpha_{r,m}(x) U_{r,m} u_{r,m}``

The address keys are shared.  Address queries, operation projections, tau
projections, and execution are route-specific.  Q and K alone share their RW
read/write rows and state writeback.  V and RST have independent banks.
Semantic selection is sparse top-k, while physical execution is deliberately
all-address dense.  Production activations stop at ``[M,T,R]`` (or paired
``[M,T,2,R]``); writeback contracts directly to ``[T,D]`` and never creates
``[M,T,D]``.  A converted single-address v4.1.7.3 model omits the address
lookup entirely and uses implicit address weight one.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable, Optional

import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from models import dawn_srw_v4173 as _v4173


MODEL_VERSION = "spatial-r1-v4.1.7.4"
LEGACY_MODEL_VERSION = _v4173.MODEL_VERSION
OPERATOR_KEY_MODE = _v4173.OPERATOR_KEY_MODE
DEFAULT_ADMISSION_DEN_POWER = _v4173.DEFAULT_ADMISSION_DEN_POWER
DEFAULT_SRW_COMPOSITION_MODE = _v4173.DEFAULT_SRW_COMPOSITION_MODE
DEFAULT_HEAT_KERNEL_BETA = _v4173.DEFAULT_HEAT_KERNEL_BETA
_raw_tau_init_from_cosine_tau = _v4173._raw_tau_init_from_cosine_tau

ADDRESS_METRIC_SUFFIXES = (
    "address_weight_top1_mean",
    "address_weight_entropy_mean",
    "address_usage_min",
    "address_usage_max",
    "address_usage_std",
    "address_dead_frac",
    "address_top1_usage_max",
)
ROUTES = ("q", "k", "v", "rst")


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"model.{name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_operation_address_config(model_cfg: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the v4174 address schema and its narrow legacy alias.

    New configs use only ``n_operation_addresses`` and
    ``operation_address_top_k``.  The historical fields are accepted only
    when both describe an implicit single address.  RST-only multi-address
    checkpoints have different semantics and are rejected rather than
    silently reinterpreted.
    """
    new_n = model_cfg.get("n_operation_addresses")
    new_k = model_cfg.get("operation_address_top_k")
    old_n = model_cfg.get("n_operation_spaces")
    old_k = model_cfg.get("operation_space_top_k")
    legacy_n = 1 if old_n is None else _positive_int("n_operation_spaces", old_n)
    legacy_k = 1 if old_k is None else _positive_int("operation_space_top_k", old_k)
    if legacy_n != 1 or legacy_k != 1:
        raise ValueError(
            "v4174 supports automatic compatibility only for legacy single-address "
            "v4173 checkpoints; RST-only multi-space checkpoints are not compatible."
        )
    n_addresses = 1 if new_n is None else _positive_int(
        "n_operation_addresses", new_n)
    top_k = 1 if new_k is None else _positive_int(
        "operation_address_top_k", new_k)
    if (old_n is not None or old_k is not None) and n_addresses != 1:
        raise ValueError(
            "v4174 canonical address fields conflict with legacy single-address aliases"
        )
    if top_k > n_addresses:
        raise ValueError(
            "model.operation_address_top_k must be <= "
            f"model.n_operation_addresses, got {top_k} > {n_addresses}"
        )
    return n_addresses, top_k


def materialize_operation_address_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and store canonical address fields without preserving aliases."""
    n_addresses, top_k = resolve_operation_address_config(model_cfg)
    for old_name in ("n_operation_spaces", "operation_space_top_k"):
        model_cfg.pop(old_name, None)
    model_cfg["n_operation_addresses"] = n_addresses
    model_cfg["operation_address_top_k"] = top_k
    for pool_name in ("n_qk", "n_v", "n_rst"):
        value = _positive_int(pool_name, model_cfg.get(
            pool_name, model_cfg.get("n_know") if pool_name == "n_rst" else None))
        if value % n_addresses:
            raise ValueError(
                f"model.{pool_name} must be divisible by "
                f"model.n_operation_addresses, got {value} % {n_addresses}"
            )
    return model_cfg


def symbolic_parameter_count(model_cfg: Mapping[str, Any]) -> dict[str, int]:
    """Count the exact v4174 tree, including the shared address codebook."""
    if "model" in model_cfg and isinstance(model_cfg["model"], Mapping):
        model_cfg = model_cfg["model"]
    cfg = dict(model_cfg)
    n_addresses, _ = resolve_operation_address_config(cfg)
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
        if value % n_addresses:
            raise ValueError(
                f"model.{name}={value} must be divisible by "
                f"n_operation_addresses={n_addresses}")
    operator_count = sum(counts_by_route.values())
    if n_addresses == 1:
        router = 7 * d_model * d_route + 4 * d_model + 4 * d_route + 4
        address_lookup = 0
    else:
        route_local = (
            7 * n_addresses * d_model * d_route
            + 4 * n_addresses * d_model
            + 4 * n_addresses * d_route
            + 4 * n_addresses
        )
        address_lookup = 4 * (d_model * d_route + d_route) + n_addresses * d_route
        router = route_local + address_lookup
    counts = {
        "token_embedding": vocab * d_model,
        "position_embedding": max_seq * d_model,
        "layer_stack": n_layers * (d_model * d_model + 4 * d_model),
        "router": router,
        "operation_address_lookup": address_lookup,
        "read_write_pools": operator_count * 2 * d_route,
        "learned_key_tables": 0,
        "bilinear_probe_matrices": 2 * d_route * d_route,
        "final_norm": 2 * d_model,
    }
    counts["total"] = sum(
        value for key, value in counts.items()
        if key != "operation_address_lookup")
    return counts


def search_parameter_matched_operator_counts(
        baseline_model_cfg: Mapping[str, Any],
        address_model_cfg: Mapping[str, Any],
        *, search_radius: int = 96) -> dict[str, Any]:
    """Jointly match QK/V/RST counts under address divisibility.

    The objective is lexicographic: absolute parameter difference, deviation
    from the baseline QK:V:RST proportions, then larger total operator count.
    Because every RW row has the same width, only a narrow neighborhood around
    the analytically optimal total needs enumeration.
    """
    baseline = dict(baseline_model_cfg.get("model", baseline_model_cfg))
    target = dict(address_model_cfg.get("model", address_model_cfg))
    m, top_k = resolve_operation_address_config(target)
    if m <= 1:
        raise ValueError("parameter matching requires multiple operation addresses")
    baseline_total = int(_v4173.symbolic_parameter_count(baseline)["total"])
    base_counts = np.asarray([
        int(baseline["n_qk"]), int(baseline["n_v"]),
        int(baseline.get("n_rst", baseline.get("n_know")))], dtype=np.float64)
    base_sum = int(base_counts.sum())
    fixed_cfg = dict(target)
    fixed_cfg.update({"n_qk": m, "n_v": m, "n_rst": m})
    fixed_total = symbolic_parameter_count(fixed_cfg)["total"] - 3 * m * 2 * int(target["d_route"])
    row_cost = 2 * int(target["d_route"])
    ideal_total = max(3 * m, round((baseline_total - fixed_total) / row_cost / m) * m)
    proportions = base_counts / base_counts.sum()
    candidates: list[tuple[Any, ...]] = []
    for total in range(max(3 * m, ideal_total - search_radius * m),
                       ideal_total + search_radius * m + 1, m):
        units = total // m
        q_center = int(round(float(proportions[0]) * units))
        v_center = int(round(float(proportions[1]) * units))
        for q_units in range(max(1, q_center - 8), q_center + 9):
            for v_units in range(max(1, v_center - 8), v_center + 9):
                r_units = units - q_units - v_units
                if r_units <= 0:
                    continue
                counts = (q_units * m, v_units * m, r_units * m)
                candidate_cfg = dict(target)
                candidate_cfg.update(dict(zip(("n_qk", "n_v", "n_rst"), counts)))
                candidate_total = symbolic_parameter_count(candidate_cfg)["total"]
                ratio = np.asarray(counts, dtype=np.float64) / float(sum(counts))
                ratio_error = float(np.square(ratio - proportions).sum())
                candidates.append((abs(candidate_total - baseline_total),
                                   ratio_error, -sum(counts), counts,
                                   candidate_total))
    absolute_difference, ratio_error, _, counts, model_total = min(candidates)
    return {
        "baseline_params": baseline_total,
        "v4174_params": model_total,
        "absolute_difference": absolute_difference,
        "relative_difference": absolute_difference / float(baseline_total),
        "baseline_counts": tuple(int(v) for v in base_counts),
        "operator_counts": counts,
        "per_address_counts": tuple(v // m for v in counts),
        "n_operation_addresses": m,
        "operation_address_top_k": top_k,
        "operator_ratio_error": ratio_error,
    }


def _tree_has_key(tree: Mapping[str, Any], key: str) -> bool:
    if key in tree:
        return True
    return any(_tree_has_key(v, key) for v in tree.values()
               if isinstance(v, Mapping))


def _convert_v4173_single_address_params_to_v4174(
        params: Mapping[str, Any], *, source_model_cfg: Mapping[str, Any],
        return_metadata: bool = False):
    """Convert one legacy single-address tree at the loader boundary.

    Combined attention projections are sliced into Q/K/V modules, legacy RW
    tensors are renamed, and no address selector is created.  The converter is
    intentionally the only location that understands historical parameter
    names.  Optimizer slots cannot be structurally renamed safely, so callers
    must initialize a fresh optimizer after conversion.
    """
    legacy_n = int(source_model_cfg.get("n_operation_spaces", 1))
    legacy_k = int(source_model_cfg.get("operation_space_top_k", 1))
    mutable = unfreeze(params) if isinstance(params, FrozenDict) else copy.deepcopy(params)
    if legacy_n != 1 or legacy_k != 1 or _tree_has_key(mutable, "rst_space_keys"):
        raise ValueError(
            "v4174 supports automatic compatibility only for legacy single-address "
            "v4173 checkpoints; RST-only multi-space checkpoints are not compatible."
        )
    pool = mutable["neuron_pool"]
    router = mutable["router"]
    new_pool = {
        "qk_read_vectors": pool["attn_qk_read"],
        "qk_write_vectors": pool["attn_qk_write"],
        "v_read_vectors": pool["attn_v_read"],
        "v_write_vectors": pool["attn_v_write"],
        "rst_read_vectors": pool["rst_read"],
        "rst_write_vectors": pool["rst_write"],
        "operator_key_read_probe": pool["rw_key_read_probe"],
        "operator_key_write_probe": pool["rw_key_write_probe"],
    }
    combined_projection = router["proj_attn"]
    combined_tau = router["raw_tau_attn"]
    r = combined_projection["kernel"].shape[-1] // 3
    new_router: dict[str, Any] = {}
    for index, route in enumerate(("q", "k", "v")):
        new_router[f"{route}_operation_proj"] = {
            "kernel": combined_projection["kernel"][:, index * r:(index + 1) * r],
            "bias": combined_projection["bias"][index * r:(index + 1) * r],
        }
        new_router[f"{route}_operator_tau_proj"] = {
            "kernel": combined_tau["kernel"][:, index:index + 1],
            "bias": combined_tau["bias"][index:index + 1],
        }
    new_router["rst_operation_proj"] = router["proj_rst"]
    new_router["rst_operator_tau_proj"] = router["raw_tau_rst"]
    new_router["qk_state_writeback"] = router["up_qk"]
    new_router["v_state_writeback"] = router["up_v"]
    new_router["rst_state_writeback"] = router["up_rst"]
    mutable["neuron_pool"] = new_pool
    mutable["router"] = new_router
    converted = freeze(mutable) if isinstance(params, FrozenDict) else mutable
    metadata = {
        "model_version": MODEL_VERSION,
        "converted_source_version": LEGACY_MODEL_VERSION,
        "optimizer_restore_policy": "fresh_optimizer",
        "n_operation_addresses": 1,
        "operation_address_top_k": 1,
    }
    return (converted, metadata) if return_metadata else converted


def _stacked_initializer(initializer: Callable) -> Callable:
    def init(key, shape, dtype=jnp.float32):
        keys = jax.random.split(key, int(shape[0]))
        return jax.vmap(lambda k: initializer(k, shape[1:], dtype))(keys)
    return init


class AddressDense(nn.Module):
    """Route-local ``P_{r,m}`` or tau projection with replicated address axis."""
    n_operation_addresses: int
    features: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, state):
        kernel = self.param(
            "kernel", _stacked_initializer(self.kernel_init),
            (self.n_operation_addresses, state.shape[-1], self.features))
        result = jnp.einsum("...d,mdr->m...r", state, kernel)
        if self.use_bias:
            bias = self.param(
                "bias", _stacked_initializer(self.bias_init),
                (self.n_operation_addresses, self.features))
            result = result + bias[(slice(None),) + (None,) * (state.ndim - 1)]
        return result


class OperationAddressNeuronPool(nn.Module):
    """Address-indexed RW banks; only the operation-row axis is sharded."""
    n_qk: int
    n_v: int
    n_rst: int
    d_route: int
    n_operation_addresses: int

    def setup(self):
        m = int(self.n_operation_addresses)
        r = int(self.d_route)
        shapes = {
            "qk": (self.n_qk, r) if m == 1 else (m, self.n_qk // m, r),
            "v": (self.n_v, r) if m == 1 else (m, self.n_v // m, r),
            "rst": (self.n_rst, r) if m == 1 else (m, self.n_rst // m, r),
        }
        for name in ("qk", "v", "rst"):
            setattr(self, f"{name}_read_vectors", self.param(
                f"{name}_read_vectors", _v4173.unit_norm_init(), shapes[name]))
            setattr(self, f"{name}_write_vectors", self.param(
                f"{name}_write_vectors", _v4173.unit_norm_init(), shapes[name]))
        probe_init = nn.initializers.orthogonal(scale=1.0)
        self.operator_key_read_probe = self.param(
            "operator_key_read_probe", probe_init, (r, r))
        self.operator_key_write_probe = self.param(
            "operator_key_write_probe", probe_init, (r, r))


class OperationAddressRouter(nn.Module):
    """Shared address codebook plus route-specific projections and tau."""
    d_model: int
    d_route: int
    n_operation_addresses: int
    tau_init_attn_qk: float
    tau_init_attn_v: float
    tau_init_rst: float

    def setup(self):
        m, r = int(self.n_operation_addresses), int(self.d_route)
        projection_init = nn.initializers.orthogonal(scale=1.0)
        tau_values = {
            "q": self.tau_init_attn_qk,
            "k": self.tau_init_attn_qk,
            "v": self.tau_init_attn_v,
            "rst": self.tau_init_rst,
        }
        projection_cls = nn.Dense if m == 1 else AddressDense
        projection_args = ({"features": r} if m == 1 else {
            "n_operation_addresses": m, "features": r})
        tau_args = ({"features": 1} if m == 1 else {
            "n_operation_addresses": m, "features": 1})
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
                              "n_operation_addresses": m,
                              "features": self.d_model, "use_bias": False})
        for name in ("qk", "v", "rst"):
            setattr(self, f"{name}_state_writeback", projection_cls(
                name=f"{name}_state_writeback", kernel_init=projection_init,
                **writeback_args))
        if m > 1:
            self.operation_address_keys = self.param(
                "operation_address_keys", _v4173.unit_norm_init(), (m, r))
            for route in ROUTES:
                setattr(self, f"{route}_address_query_proj", nn.Dense(
                    r, name=f"{route}_address_query_proj",
                    kernel_init=projection_init,
                    bias_init=nn.initializers.zeros))


def _linear(params: Mapping[str, jax.Array], state: jax.Array) -> jax.Array:
    return state @ params["kernel"] + params.get("bias", 0.0)


def _select_operation_addresses(
        state: jax.Array,
        address_query_params: Mapping[str, jax.Array],
        operation_address_keys: jax.Array,
        operation_address_top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select semantic addresses without changing physical dense execution.

    ``state`` is ``[B,S,D]``; the route-specific query projection maps it to
    ``[B,S,R]``.  Shared keys are ``[M,R]``.  Returned IDs and selected
    weights are ``[B,S,K]`` and scores are ``[B,S,M]``.  The dense execution
    path later scatters these K weights into ``[T,M]``; it never dispatches or
    gathers tokens by address.
    """
    query = _linear(address_query_params, state)
    query = _v4173._forward_unit_direction(query.astype(jnp.float32))
    keys = _v4173._forward_unit_direction(
        jnp.asarray(operation_address_keys, dtype=jnp.float32))
    scores = jnp.einsum("bsr,mr->bsm", query, keys)
    selected_scores, selected_ids = jax.lax.top_k(
        scores, int(operation_address_top_k))
    selected_weights = jax.nn.softmax(selected_scores, axis=-1)
    return selected_ids, selected_weights, scores


def _dense_address_weights(
        selected_address_ids: jax.Array,
        selected_address_weights: jax.Array,
        n_operation_addresses: int,
) -> jax.Array:
    """Return exact-zero unselected weights as a flattened ``[T,M]`` matrix."""
    dense = jnp.sum(
        jax.nn.one_hot(
            selected_address_ids, int(n_operation_addresses),
            dtype=selected_address_weights.dtype)
        * selected_address_weights[..., None],
        axis=-2)
    return dense.reshape((-1, int(n_operation_addresses)))


def _weighted_state_writeback(
        address_results: jax.Array,
        dense_address_weights: jax.Array,
        state_writeback_kernel: jax.Array,
        route_scale: jax.Array | float,
) -> jax.Array:
    """Fuse semantic address weighting, address sum, and ``R -> D`` writeback.

    Inputs are ``address_results[M,T,R]``, weights ``[T,M]``, and the
    route-specific kernel ``[M,R,D]``.  The only weighted intermediate is
    ``[M,T,R]``.  The einsum contracts directly to ``[T,D]``; a forbidden
    ``[M,T,D]`` tensor is never materialized.
    """
    address_results = jnp.asarray(address_results)
    if address_results.ndim != 3:
        raise ValueError(
            "address_results must have shape [M,T,R], got "
            f"{address_results.shape}")
    weighted_results = (
        address_results
        * jnp.swapaxes(dense_address_weights, 0, 1)[..., None])
    return jnp.einsum(
        "mtr,mrd->td", weighted_results * jnp.asarray(route_scale),
        state_writeback_kernel)


def _materialize_address_operator_keys(
        read_vectors: jax.Array, write_vectors: jax.Array,
        read_probe: jax.Array, write_probe: jax.Array) -> jax.Array:
    """Generate live-gradient operator keys independently inside each address."""
    if read_vectors.ndim == 2:
        return _v4173.materialize_generalized_bilinear_operator_keys(
            read_vectors, write_vectors, read_probe, write_probe)
    if read_vectors.ndim != 3:
        raise ValueError(
            "address-indexed read/write vectors must be [M,N,R], got "
            f"{read_vectors.shape}")
    return jax.vmap(
        lambda read, write: _v4173.materialize_generalized_bilinear_operator_keys(
            read, write, read_probe, write_probe))(read_vectors, write_vectors)


def _rw_compose_address_dense(
        operator_query: jax.Array, operator_keys: jax.Array,
        raw_operator_tau: jax.Array, read_vectors: jax.Array,
        write_vectors: jax.Array, *, soft_gate_temperature: float,
        soft_gate_boundary_power: float, admission_den_power: float,
        srw_composition_mode: str, heat_kernel_beta: float,
        execution_prune_eps: float = 0.0, max_chunk_size: int = 2048,
        diagnostics: bool = False):
    """Execute every address densely while chunking only the operator axis.

    The physical input/output is ``[M,T,R]`` and each address owns
    ``[N_per_address,R]`` operator rows.  Operator gates are transient
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
    query_unit = _v4173._forward_unit_direction(
        operator_query.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    key_unit = _v4173._forward_unit_direction(
        keys.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    read_unit = _v4173._forward_unit_direction(
        reads.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    write_unit = _v4173._forward_unit_direction(
        writes.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    tau = _v4173._tau_from_param(raw_operator_tau)
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
        margin, gate, depth, execution_weight, _ = _v4173._compute_admission_drive(
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
    gate_den = _v4173._composition_den(
        gate_mass, jnp.float32(admission_den_power), srw_composition_mode)
    # Match the canonical shard-map boundary: model shards contribute BF16
    # local results before the replicated FP32 route-local output is formed.
    address_results = (raw_out / gate_den).astype(jnp.bfloat16).astype(jnp.float32)
    if not diagnostics:
        return address_results
    return (address_results, active_count, gate_mass, gate_sq, gate_max,
            depth_sum, tau, gate_den)


def _paired_qk_address_execution(
        stacked_qk_operator_query: jax.Array,
        qk_operator_keys: jax.Array,
        stacked_qk_raw_tau: jax.Array,
        qk_read_vectors: jax.Array,
        qk_write_vectors: jax.Array,
        **execution_kwargs) -> jax.Array:
    """Execute independent Q/K gates against one shared address-indexed bank.

    Q and K arrive as ``[M,T,2,R]`` and return the same shape.  The route axis
    is paired before attention; it is never retained by the attention core.
    Selection, operation projection, and tau remain independent.
    """
    q_result = _rw_compose_address_dense(
        stacked_qk_operator_query[:, :, 0, :], qk_operator_keys,
        stacked_qk_raw_tau[:, :, 0, :], qk_read_vectors, qk_write_vectors,
        **execution_kwargs)
    k_result = _rw_compose_address_dense(
        stacked_qk_operator_query[:, :, 1, :], qk_operator_keys,
        stacked_qk_raw_tau[:, :, 1, :], qk_read_vectors, qk_write_vectors,
        **execution_kwargs)
    return jnp.stack((q_result, k_result), axis=2)


def _address_selector_metrics(
        selected_ids: jax.Array, selected_weights: jax.Array,
        n_operation_addresses: int, route: str) -> dict[str, jax.Array]:
    """Small regular metrics computed only from already-selected IDs/weights."""
    m = int(n_operation_addresses)
    flat_ids = selected_ids.reshape((-1, selected_ids.shape[-1]))
    flat_weights = selected_weights.reshape((-1, selected_weights.shape[-1]))
    top1 = flat_ids[:, 0]
    top1_usage = jax.nn.one_hot(top1, m, dtype=jnp.float32).mean(axis=0)
    usage = jax.nn.one_hot(flat_ids, m, dtype=jnp.float32).sum(axis=1).mean(axis=0)
    entropy = -(flat_weights * jnp.log(jnp.maximum(flat_weights, 1e-8))).sum(axis=-1)
    values = {
        f"{route}_address_weight_top1_mean": flat_weights[:, 0].mean(),
        f"{route}_address_weight_entropy_mean": entropy.mean(),
        f"{route}_address_usage_min": usage.min(),
        f"{route}_address_usage_max": usage.max(),
        f"{route}_address_usage_std": usage.std(),
        f"{route}_address_dead_frac": (usage == 0).astype(jnp.float32).mean(),
        f"{route}_address_top1_usage_max": top1_usage.max(),
    }
    return {key: jax.lax.stop_gradient(value) for key, value in values.items()}


def operation_address_initialization_diagnostics(
        operation_address_keys: jax.Array,
        route_assignments: Mapping[str, tuple[jax.Array, jax.Array, jax.Array]],
) -> dict[str, float]:
    """Host-side, one-shot key and initial token-assignment diagnostics."""
    keys = jax.device_get(jnp.asarray(operation_address_keys, dtype=jnp.float32))
    norms = jnp.linalg.norm(keys, axis=-1)
    if (not bool(jnp.all(jnp.isfinite(keys)))
            or bool(jnp.any(norms <= 0))):
        raise ValueError("operation_address_keys contain non-finite or zero-norm rows")
    unit = keys / norms[:, None]
    cosine = unit @ unit.T
    offdiag = cosine[~jnp.eye(cosine.shape[0], dtype=jnp.bool_)]
    if bool(jnp.any(offdiag > 0.95)):
        raise ValueError("operation_address_keys contain duplicate or near-identical rows")
    nearest = jnp.max(jnp.where(
        jnp.eye(cosine.shape[0], dtype=jnp.bool_), -jnp.inf, cosine), axis=-1)
    out = {
        "operation_address_pair_cosine_mean": float(offdiag.mean()),
        "operation_address_pair_cosine_abs_mean": float(jnp.abs(offdiag).mean()),
        "operation_address_pair_cosine_max": float(offdiag.max()),
        "operation_address_nearest_neighbor_cosine_mean": float(nearest.mean()),
        "operation_address_nearest_neighbor_cosine_max": float(nearest.max()),
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
        out[f"{route}_dead_address_frac"] = float((topk_usage == 0).mean())
        out[f"{route}_top1_weight_mean"] = float(weights[..., 0].mean())
        out[f"{route}_weight_entropy"] = float(
            (-(weights * jnp.log(jnp.maximum(weights, 1e-8))).sum(axis=-1)).mean())
        out[f"{route}_top1_top2_score_gap"] = float(gap.mean())
    return out


def initialization_diagnostics_from_params(
        params: Mapping[str, Any], input_ids: jax.Array,
        operation_address_top_k: int) -> dict[str, float]:
    """Build the one-shot host diagnostic outside the training-step graph."""
    router = params["router"]
    if "operation_address_keys" not in router:
        return {}
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    positions = jnp.arange(input_ids.shape[1])[None, :]
    state = (params["token_emb"]["embedding"][input_ids]
             + params["pos_emb"]["embedding"][positions])
    block = params["block_0"]
    attention_state = _v4173._layer_norm(
        state, block["norm1"]["scale"], block["norm1"]["bias"])
    rst_state = _v4173._layer_norm(
        state, block["norm2"]["scale"], block["norm2"]["bias"])
    assignments = {}
    for route in ROUTES:
        route_state = rst_state if route == "rst" else attention_state
        assignments[route] = _select_operation_addresses(
            route_state, router[f"{route}_address_query_proj"],
            router["operation_address_keys"], operation_address_top_k)
    return operation_address_initialization_diagnostics(
        router["operation_address_keys"], assignments)


def _address_projection(
        state: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
    """Apply a canonical projection and normalize its address-leading shape."""
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
        n_operators_per_address: int, route: str,
        srw_composition_mode: str,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Reduce address-local aggregates according to the reporting contract."""
    (address_results, active_count, gate_mass, gate_sq, gate_max,
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
    floor_mass = _v4173._composition_den_floor_mass(srw_composition_mode)
    floor_compare = jnp.less if srw_composition_mode == "linear_angular" else jnp.less_equal
    metrics = {
        f"{route}_operator_active_tau_frac": (
            active / float(n_operators_per_address)).mean(),
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
    m = int(address_results.shape[0])
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    selected_counts = jax.nn.one_hot(flat_ids, m).sum(axis=0)
    selected_weight_sum = (
        jax.nn.one_hot(flat_ids, m) * flat_weights[:, None]).sum(axis=0)
    per_address = {
        f"{route}_per_address_selection_frac": selected_counts / ids.shape[0],
        f"{route}_per_address_mean_selected_weight": (
            selected_weight_sum / jnp.maximum(selected_counts, 1.0)),
        f"{route}_per_address_active_frac": (
            active_count[..., 0] / float(n_operators_per_address)).mean(axis=1),
        f"{route}_per_address_active_count": active_count[..., 0].mean(axis=1),
        f"{route}_per_address_tau_mean": tau[..., 0].mean(axis=1),
        f"{route}_per_address_gate_mass": gate_mass[..., 0].mean(axis=1),
        f"{route}_per_address_gate_den": gate_den[..., 0].mean(axis=1),
        f"{route}_per_address_output_norm": jnp.linalg.norm(
            address_results.astype(jnp.float32), axis=-1).mean(axis=1),
    }
    return (
        {key: jax.lax.stop_gradient(value) for key, value in metrics.items()},
        {key: jax.lax.stop_gradient(value) for key, value in per_address.items()},
    )


class DAWN_SRW_V4174(nn.Module):
    """Transformer whose four routes share one operation-address codebook."""
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
    n_operation_addresses: int = 1
    operation_address_top_k: int = 1
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
        m = _positive_int("n_operation_addresses", self.n_operation_addresses)
        k = _positive_int("operation_address_top_k", self.operation_address_top_k)
        if k > m:
            raise ValueError("operation_address_top_k exceeds n_operation_addresses")
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        for name, value in (("n_qk", self.n_qk), ("n_v", self.n_v),
                            ("n_rst", n_rst)):
            if int(value) % m:
                raise ValueError(f"{name} must be divisible by n_operation_addresses")
        if any(value is None for value in (
                self.tau_init_attn_qk, self.tau_init_attn_v,
                self.tau_init_rst)):
            raise ValueError("v4174 requires explicit initial operator tau values")
        _, embedding_vocab = self._vocab_sizes()
        self.token_emb = nn.Embed(
            embedding_vocab, self.d_model,
            embedding_init=_v4173.scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model,
            embedding_init=_v4173.scaled_normal(0.02))
        self.neuron_pool = OperationAddressNeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst,
            d_route=self.d_route, n_operation_addresses=m)
        self.router = OperationAddressRouter(
            d_model=self.d_model, d_route=self.d_route,
            n_operation_addresses=m,
            tau_init_attn_qk=float(self.tau_init_attn_qk),
            tau_init_attn_v=float(self.tau_init_attn_v),
            tau_init_rst=float(self.tau_init_rst))
        self.layers = [
            _v4173.DAWNBlock(
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
        if int(self.n_operation_addresses) > 1:
            _ = self.router.operation_address_keys
            for route in ROUTES:
                _ = getattr(self.router, f"{route}_address_query_proj")(state)
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
        """Run all-address dense routing; heavy arrays require analysis cadence."""
        del (attention_mask, soft_gate_t_final,
             soft_gate_boundary_power_final, admission_den_power,
             heat_kernel_beta, minimal_train, ce_token_chunk_size,
             analysis_kwargs)
        m = int(self.n_operation_addresses)
        top_k = int(self.operation_address_top_k)
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
        state = _v4173.safe_dropout(
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
                read, write, _materialize_address_operator_keys(
                    read, write, read_probe, write_probe))
        qk_scale, v_scale, rst_scale = _v4173._pool_output_scales(
            self.d_model, self.n_layers)
        base_rng = self.make_rng("dropout")
        layer_rngs = jax.random.split(base_rng, self.n_layers)
        selector_metrics: dict[str, list[jax.Array]] = {
            f"{route}_{suffix}": [] for route in ROUTES
            for suffix in ADDRESS_METRIC_SUFFIXES}
        operator_metrics: dict[str, list[jax.Array]] = {}
        analysis_arrays: dict[str, list[jax.Array]] = {}
        diagnostics_enabled = (
            minimal_runtime_profile == "diagnostics" or analysis)

        def execute_address_route(
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
            return _rw_compose_address_dense(
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
                keys = router["operation_address_keys"]
                for route in ("q", "k", "v"):
                    ids, weights, _ = _select_operation_addresses(
                        normalized, router[f"{route}_address_query_proj"],
                        keys, top_k)
                    route_ids[route], route_weights[route] = ids, weights
                    dense_weights[route] = _dense_address_weights(ids, weights, m)
            for route in ("q", "k", "v"):
                for key, value in _address_selector_metrics(
                        route_ids[route], route_weights[route], m, route).items():
                    selector_metrics[key].append(value)
            flat_attention_state = normalized.reshape((-1, self.d_model))
            query_values = {
                route: _address_projection(
                    normalized, router[f"{route}_operation_proj"]
                ).reshape((m, -1, self.d_route))
                for route in ("q", "k", "v")}
            tau_values = {
                route: _address_projection(
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
            paired_address_kernel = (
                sharded_fns.get("qk_address_dense")
                if isinstance(sharded_fns, dict) and m > 1 else None)
            if paired_address_kernel is not None:
                paired_result = paired_address_kernel(
                    qk_paired, qk_keys, qk_tau,
                    jnp.ones(qk_paired.shape[:2], dtype=jnp.bool_),
                    qk_read, qk_write,
                    soft_gate_T_qk, soft_gate_T_qk,
                    soft_gate_boundary_power, soft_gate_boundary_power,
                    execution_prune_eps)
                if diagnostics_enabled:
                    q_diag, k_diag = paired_result
                else:
                    q_address_outputs = paired_result[:, :, 0]
                    k_address_outputs = paired_result[:, :, 1]
            elif diagnostics_enabled:
                q_diag = execute_address_route(
                    "q_address_dense", qk_paired[:, :, 0], qk_keys,
                    qk_tau[:, :, 0], qk_read, qk_write,
                    temperature=soft_gate_T_qk, den_power=qk_power)
                k_diag = execute_address_route(
                    "k_address_dense", qk_paired[:, :, 1], qk_keys,
                    qk_tau[:, :, 1], qk_read, qk_write,
                    temperature=soft_gate_T_qk, den_power=qk_power)
            if diagnostics_enabled:
                q_address_outputs, k_address_outputs = q_diag[0], k_diag[0]
                for route, diag in (("q", q_diag), ("k", k_diag)):
                    scalar, arrays = _aggregate_operator_diagnostics(
                        diag, route_ids[route], route_weights[route],
                        self.n_qk // m, route, composition_mode)
                    for key, value in scalar.items():
                        operator_metrics.setdefault(key, []).append(value)
                    if analysis:
                        for key, value in arrays.items():
                            analysis_arrays.setdefault(key, []).append(value)
            elif paired_address_kernel is None:
                paired_result = _paired_qk_address_execution(
                    qk_paired, qk_keys, qk_tau, qk_read, qk_write,
                    **qk_execution_kwargs)
                q_address_outputs = paired_result[:, :, 0]
                k_address_outputs = paired_result[:, :, 1]
            q_route_output = _weighted_state_writeback(
                q_address_outputs, dense_weights["q"],
                _canonical_writeback_kernel(router["qk_state_writeback"]),
                qk_scale).reshape(normalized.shape)
            k_route_output = _weighted_state_writeback(
                k_address_outputs, dense_weights["k"],
                _canonical_writeback_kernel(router["qk_state_writeback"]),
                qk_scale).reshape(normalized.shape)
            v_read, v_write, v_keys = bank_values["v"]
            v_execution_kwargs = dict(qk_execution_kwargs)
            v_execution_kwargs["soft_gate_temperature"] = soft_gate_T_v
            v_execution_kwargs["admission_den_power"] = v_power
            v_result = execute_address_route(
                "v_address_dense", query_values["v"], v_keys,
                tau_values["v"], v_read, v_write,
                temperature=soft_gate_T_v, den_power=v_power)
            if diagnostics_enabled:
                v_address_outputs = v_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    v_result, route_ids["v"], route_weights["v"],
                    self.n_v // m, "v", composition_mode)
                for key, value in scalar.items():
                    operator_metrics.setdefault(key, []).append(value)
                if analysis:
                    for key, value in arrays.items():
                        analysis_arrays.setdefault(key, []).append(value)
            else:
                v_address_outputs = v_result
            v_route_output = _weighted_state_writeback(
                v_address_outputs, dense_weights["v"],
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
            attention_weights = _v4173.safe_dropout(
                attention_weights, self.dropout_rate, deterministic, rng_attn)
            attention_output = jnp.einsum(
                "bhst,bhtd->bhsd", attention_weights, value)
            attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
                batch_size, sequence_length, self.d_model)
            attention_output = layer.attn.expand_O(attention_output)
            state = state + _v4173.safe_dropout(
                attention_output, self.dropout_rate, deterministic, rng)

            normalized = layer.norm2(state)
            if m > 1:
                ids, weights, _ = _select_operation_addresses(
                    normalized, router["rst_address_query_proj"],
                    router["operation_address_keys"], top_k)
                route_ids["rst"], route_weights["rst"] = ids, weights
                dense_weights["rst"] = _dense_address_weights(ids, weights, m)
            for key, value in _address_selector_metrics(
                    route_ids["rst"], route_weights["rst"], m, "rst").items():
                selector_metrics[key].append(value)
            rst_query = _address_projection(
                normalized, router["rst_operation_proj"]).reshape(
                    (m, -1, self.d_route))
            rst_tau = _address_projection(
                normalized, router["rst_operator_tau_proj"]).reshape((m, -1, 1))
            rst_read_vectors, rst_write_vectors, rst_keys = bank_values["rst"]
            rst_result = execute_address_route(
                "rst_address_dense", rst_query, rst_keys, rst_tau,
                rst_read_vectors, rst_write_vectors,
                temperature=soft_gate_T_rst, den_power=rst_power)
            if diagnostics_enabled:
                rst_address_updates = rst_result[0]
                scalar, arrays = _aggregate_operator_diagnostics(
                    rst_result, route_ids["rst"], route_weights["rst"],
                    n_rst_effective // m, "rst", composition_mode)
                for key, value in scalar.items():
                    operator_metrics.setdefault(key, []).append(value)
                if analysis:
                    for key, value in arrays.items():
                        analysis_arrays.setdefault(key, []).append(value)
            else:
                rst_address_updates = rst_result
            rst_route_update = _weighted_state_writeback(
                rst_address_updates, dense_weights["rst"],
                _canonical_writeback_kernel(router["rst_state_writeback"]),
                rst_scale).reshape(normalized.shape)
            state = state + _v4173.safe_dropout(
                rst_route_update, self.dropout_rate, deterministic, rng_rst)
            if diagnostics_enabled:
                operator_metrics.setdefault("q_route_output_norm", []).append(
                    jnp.linalg.norm(q_route_output.astype(jnp.float32), axis=-1).mean())
                operator_metrics.setdefault("k_route_output_norm", []).append(
                    jnp.linalg.norm(k_route_output.astype(jnp.float32), axis=-1).mean())
                operator_metrics.setdefault("v_route_output_norm", []).append(
                    jnp.linalg.norm(v_route_output.astype(jnp.float32), axis=-1).mean())
                operator_metrics.setdefault("rst_route_output_norm", []).append(
                    jnp.linalg.norm(rst_route_update.astype(jnp.float32), axis=-1).mean())

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
        if analysis:
            for key, values in analysis_arrays.items():
                result[key] = jnp.stack(values).mean(axis=0)
        return result

    def get_model_info(self) -> list[str]:
        n_rst = int(self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200))
        m = int(self.n_operation_addresses)
        return [
            f"DAWN-SRW v4.1.7.4 ({MODEL_VERSION})",
            "architecture: shared state -> shared operation-address lookup -> "
            "route-local RW execution -> address-weighted state writeback",
            f"addresses: count={m}, top_k={int(self.operation_address_top_k)}, "
            f"d_address=d_route={int(self.d_route)}",
            "Q/K share qk_read_vectors, qk_write_vectors, generated keys, and "
            "qk_state_writeback; routing/projection/tau remain independent",
            f"operators per address: qk={int(self.n_qk)//m}, "
            f"v={int(self.n_v)//m}, rst={n_rst//m}",
            "execution: semantic top-k with physical all-address dense kernels; "
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
        f"{name}_operator_keys": _materialize_address_operator_keys(
            pool_params[f"{name}_read_vectors"],
            pool_params[f"{name}_write_vectors"], read_probe, write_probe)
        for name in ("qk", "v", "rst")
    }


def generalized_bilinear_operator_key_diagnostics(
        read_vectors, write_vectors, read_probes, write_probes,
        eps=_v4173.RW_FORWARD_NORM_EPS):
    """Expose existing bilinear-key diagnostics for one address or bank."""
    if read_vectors.ndim == 2:
        return _v4173.generalized_bilinear_operator_key_diagnostics(
            read_vectors, write_vectors, read_probes, write_probes, eps)
    flattened_read = read_vectors.reshape((-1, read_vectors.shape[-1]))
    flattened_write = write_vectors.reshape((-1, write_vectors.shape[-1]))
    return _v4173.generalized_bilinear_operator_key_diagnostics(
        flattened_read, flattened_write, read_probes, write_probes, eps)


def _tau_init_calibration_scores(params, input_ids, max_tokens=128):
    """Return all-token × every-address score tables for independent tau fits.

    Address selectors are deliberately absent: operator tau calibrates the
    address-local RW bank, not the semantic address lookup.  Q and K share
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
    attention_state = _v4173._layer_norm(
        state, block["norm1"]["scale"], block["norm1"]["bias"])
    rst_state = _v4173._layer_norm(
        state, block["norm2"]["scale"], block["norm2"]["bias"])
    router = params["router"]
    pool = params["neuron_pool"]
    keys = _pool_operator_keys(pool)

    def score(route_state, projection, operator_keys):
        query = _address_projection(route_state, projection)
        if operator_keys.ndim == 2:
            operator_keys = operator_keys[None, ...]
        query = _v4173._forward_unit_direction(query.astype(jnp.float32))
        operator_keys = _v4173._forward_unit_direction(
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


def calibrate_operator_tau_per_address(
        scores_by_route: Mapping[str, jax.Array],
        *, target_qk_frac: float, target_v_frac: float,
        target_rst_frac: float) -> dict[str, jax.Array]:
    """Compute one quantile per route/address from all calibration tokens."""
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
    """Rare-cadence route-query geometry with canonical address terminology."""
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
        n_operation_addresses=cfg.get("n_operation_addresses", 1),
        operation_address_top_k=cfg.get("operation_address_top_k", 1),
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


# Single-address compatibility keeps the proven v4173 shard-map math.  The
# v4174 trainer adds address-leading dense factories for M>1.
make_sharded_srw = _v4173.make_sharded_srw
make_sharded_srw_paired = _v4173.make_sharded_srw_paired
make_sharded_srw_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_minimal, "production")
make_sharded_srw_paired_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_paired_minimal, "production")
make_sharded_srw_diagnostics_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_diagnostics_minimal, "production_diagnostics")
make_sharded_srw_paired_diagnostics_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_paired_diagnostics_minimal,
    "production_diagnostics")
make_sharded_srw_retention_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_retention_minimal, "retention")
make_sharded_srw_paired_retention_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_paired_retention_minimal, "retention")
make_sharded_srw_suppression_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_suppression_minimal, "suppression")
make_sharded_srw_paired_suppression_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_paired_suppression_minimal, "suppression")
make_sharded_srw_trajectory_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_trajectory_minimal, "trajectory")
make_sharded_srw_paired_trajectory_minimal = _factory_profile_wrapper(
    _v4173.make_sharded_srw_paired_trajectory_minimal, "trajectory")


@wraps(_v4173.make_sharded_rst_multispace_dense_minimal)
def make_sharded_address_dense_minimal(*args, **kwargs):
    """Create a production ``[M,T,R]`` grouped-dense address kernel."""
    kernel = _v4173.make_sharded_rst_multispace_dense_minimal(*args, **kwargs)
    kernel._v4174_kernel_profile = "production"
    kernel._v4174_dense_grouped_execution = "all_addresses"
    return kernel


@wraps(_v4173.make_sharded_rst_multispace_dense_diagnostics)
def make_sharded_address_dense_diagnostics(*args, **kwargs):
    """Create the matching observational kernel with ``[M,T,1]`` aggregates."""
    kernel = _v4173.make_sharded_rst_multispace_dense_diagnostics(*args, **kwargs)
    kernel._v4174_kernel_profile = "production_diagnostics"
    kernel._v4174_dense_grouped_diagnostics = "all_addresses"
    return kernel


@wraps(_v4173.make_sharded_rst_multispace_dense_minimal)
def make_sharded_qk_address_dense_minimal(*args, **kwargs):
    """Create the paired Q/K ``[M,T,2,R]`` production interface.

    Both routes consume the same sharded QK bank closure.  Their query/tau
    slices remain independent, and the paired route axis is eliminated before
    causal attention.
    """
    single = make_sharded_address_dense_minimal(*args, **kwargs)

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
    paired._v4174_paired_qk_execution = "all_addresses"
    return paired


@wraps(_v4173.make_sharded_rst_multispace_dense_diagnostics)
def make_sharded_qk_address_dense_diagnostics(*args, **kwargs):
    """Create paired Q/K observational execution without full operator tensors."""
    single = make_sharded_address_dense_diagnostics(*args, **kwargs)

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
    paired._v4174_paired_qk_execution = "all_addresses"
    return paired


def _validate_v4174_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs):
    """Validate common RW metadata while allowing v4174 address wrappers."""
    return _v4173._validate_v4173_sharded_fns(
        sharded_fns, admission_den_power, srw_composition_mode,
        heat_kernel_beta, **kwargs)


def get_model_version() -> str:
    return MODEL_VERSION
