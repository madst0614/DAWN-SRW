"""Function-first operator-space analysis with address held out of discovery."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import analysis_model_module


def candidate_pool_vectors(
        ctx: Any, pool_name: str,
        operator_ids: Sequence[int]) -> dict[str, np.ndarray]:
    """Read live RW/address vectors from the checkpoint-declared key mode."""
    prefix = {"qk": "attn_qk", "v": "attn_v", "rst": "rst"}.get(
        str(pool_name))
    if prefix is None:
        raise ValueError(f"unknown operator pool {pool_name!r}")
    ids = np.asarray(operator_ids, dtype=np.int32)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("operator_ids must be a non-empty vector")
    module = analysis_model_module(ctx.model_cfg)
    params = module._squeeze_params(ctx.params)
    pool = module._pool_params_with_operator_keys(
        params["neuron_pool"], ctx.model_cfg.get("operator_key_mode"))
    index = jnp.asarray(ids, dtype=jnp.int32)
    return {
        "operator_ids": ids,
        "read": np.asarray(jax.device_get(pool[f"{prefix}_read"][index])),
        "write": np.asarray(jax.device_get(pool[f"{prefix}_write"][index])),
        "address": np.asarray(jax.device_get(pool[f"{prefix}_op_key"][index])),
    }


def operator_pool_provenance(ctx: Any) -> dict[str, Any]:
    """Report whether addresses are learned tables or live RW/probe functions."""
    module = analysis_model_module(ctx.model_cfg)
    params = module._squeeze_params(ctx.params)
    pool = params["neuron_pool"]
    mode = str(ctx.model_cfg.get(
        "operator_key_mode", getattr(module, "OPERATOR_KEY_MODE", "learned")))
    probe_names = ("rw_key_read_probe", "rw_key_write_probe")
    learned_names = (
        "attn_qk_op_key", "attn_v_op_key", "rst_op_key")
    probe_count = int(sum(
        int(np.prod(pool[name].shape)) for name in probe_names
        if name in pool))
    learned_count = int(sum(
        int(np.prod(pool[name].shape)) for name in learned_names
        if name in pool))
    generalized = mode == getattr(
        module, "OPERATOR_KEY_MODE_GENERALIZED_BILINEAR",
        "generalized_bilinear_rw")
    return {
        "status": "ready",
        "operator_key_mode": mode,
        "operator_key_source": (
            "live_rw_plus_shared_probes" if generalized
            else "learned_operator_key_tables"),
        "learned_operator_key_tables": bool(learned_count),
        "shared_probe_matrices": bool(probe_count),
        "probe_scope": "qk_v_rst_global" if probe_count else None,
        "operator_keys_shared_across_layers": True,
        "operator_rw_shared_across_layers": True,
        "operator_key_probe_parameter_count": probe_count,
        "learned_operator_key_parameter_count": learned_count,
    }


def _unit_rows(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"operator vectors must be a matrix, got {value.shape}")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("operator vectors contain zero or non-finite rows")
    return value / norms


def rw_function_similarity(read: np.ndarray, write: np.ndarray) -> np.ndarray:
    """Normalized Frobenius similarity of rank-one ``write ⊗ read`` maps."""
    read_unit = _unit_rows(read)
    write_unit = _unit_rows(write)
    return (read_unit @ read_unit.T) * (write_unit @ write_unit.T)


def discover_functional_families(
        read: np.ndarray, write: np.ndarray, *, neighbor_k: int,
        similarity_quantile: float) -> dict[str, Any]:
    """Discover reciprocal local families without looking at address vectors."""
    similarity = rw_function_similarity(read, write)
    n_operators = similarity.shape[0]
    if n_operators < 2:
        raise ValueError("functional-family discovery requires >= 2 operators")
    neighbor_k = min(max(1, int(neighbor_k)), n_operators - 1)
    off_diagonal = similarity[~np.eye(n_operators, dtype=bool)]
    threshold = float(np.quantile(off_diagonal, similarity_quantile))
    ranked = np.argsort(-similarity, axis=1, kind="mergesort")
    neighbor_sets = [
        {int(index) for index in row if int(index) != operator_id}
        for operator_id, row in enumerate(ranked[:, :neighbor_k + 1])
    ]
    adjacency: list[set[int]] = [set() for _ in range(n_operators)]
    for left in range(n_operators):
        for right in neighbor_sets[left]:
            if left not in neighbor_sets[right]:
                continue
            if float(similarity[left, right]) < threshold:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
    # Keep seed-local reciprocal neighborhoods.  Connected-component closure
    # would turn chains of merely local similarity into a spurious large unit.
    unique_families: set[tuple[int, ...]] = set()
    for seed in range(n_operators):
        if adjacency[seed]:
            unique_families.add(tuple(sorted({seed, *adjacency[seed]})))
    families = [list(family) for family in unique_families]
    families.sort(key=lambda family: (-len(family), family[0]))
    covered = set().union(*(set(family) for family in families)) if families else set()
    within = [
        float(similarity[left, right])
        for family in families for index, left in enumerate(family)
        for right in family[index + 1:]
    ]
    return {
        "families": families,
        "family_count": len(families),
        "covered_operator_count": len(covered),
        "singleton_count": int(n_operators - len(covered)),
        "neighbor_k": neighbor_k,
        "similarity_quantile": float(similarity_quantile),
        "similarity_threshold": threshold,
        "within_family_function_similarity_mean": (
            float(np.mean(within)) if within else None),
        "discovery_features": ["read_direction", "write_direction"],
        "address_used_for_discovery": False,
        "families_may_overlap": True,
        "transitive_components_treated_as_causal_units": False,
    }


def address_confirmation(families: Sequence[Sequence[int]],
                         address: np.ndarray, *, seed: int) -> dict[str, Any]:
    """Measure address compactness only after functional families are frozen."""
    unit = _unit_rows(address)
    rng = np.random.default_rng(int(seed))
    pairs = sorted({
        (min(int(left), int(right)), max(int(left), int(right)))
        for family in families
        for index, left in enumerate(family)
        for right in family[index + 1:]
    })
    within: list[float] = [
        float(unit[left] @ unit[right]) for left, right in pairs]
    null: list[float] = []
    all_ids = np.arange(unit.shape[0])
    for _ in pairs:
        draws = rng.choice(all_ids, size=2, replace=False)
        null.append(float(unit[draws[0]] @ unit[draws[1]]))
    if not within:
        return {
            "status": "insufficient_family_pairs",
            "address_used_for_discovery": False,
        }
    return {
        "status": "ready",
        "within_family_address_similarity_mean": float(np.mean(within)),
        "pair_count_matched_random_similarity_mean": float(np.mean(null)),
        "address_compactness_effect": float(np.mean(within) - np.mean(null)),
        "pair_count": len(within),
        "address_used_for_discovery": False,
        "confirmation_phase_only": True,
    }
