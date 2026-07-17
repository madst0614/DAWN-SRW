"""Held-out multilayer route trajectories and rerouting/recovery summaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.operator_interpretability.statistics import (
    bootstrap_mean_ci,
    paired_permutation_test,
)


def weighted_jaccard(ids_a: Sequence[int], weights_a: Sequence[float],
                     ids_b: Sequence[int], weights_b: Sequence[float]) -> float:
    left = {int(key): abs(float(value)) for key, value in zip(ids_a, weights_a)}
    right = {int(key): abs(float(value)) for key, value in zip(ids_b, weights_b)}
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return float(numerator / denominator) if denominator > 0.0 else 1.0


def trajectory_similarity(base_rows: Sequence[Mapping[str, Any]],
                          comparison_rows: Sequence[Mapping[str, Any]], *,
                          capture_threshold: float) -> dict[str, Any]:
    if len(base_rows) != len(comparison_rows):
        raise ValueError("trajectory rows must align by layer and route")
    similarities: list[float] = []
    excluded = 0
    for base, comparison in zip(base_rows, comparison_rows):
        identity_a = (int(base["layer"]), str(base["route"]))
        identity_b = (int(comparison["layer"]), str(comparison["route"]))
        if identity_a != identity_b:
            raise ValueError("trajectory layer/route alignment mismatch")
        if min(float(base["captured_mass"]),
               float(comparison["captured_mass"])) < capture_threshold:
            excluded += 1
            continue
        similarities.append(weighted_jaccard(
            base["operator_ids"], base["weights"],
            comparison["operator_ids"], comparison["weights"]))
    return {
        "qualified_layer_route_count": len(similarities),
        "excluded_low_capture_count": excluded,
        "mean_weighted_jaccard": (
            float(np.mean(similarities)) if similarities else None),
        "minimum_weighted_jaccard": (
            float(np.min(similarities)) if similarities else None),
        "capture_threshold": float(capture_threshold),
    }


def held_out_trajectory_confirmation(
        rows: Sequence[Mapping[str, Any]], *, phase: str,
        capture_threshold: float, bootstrap_samples: int,
        permutation_samples: int, alpha: float, seed: int) -> dict[str, Any]:
    """Compare same-variable paths with disjoint cross-variable controls."""
    if phase not in {"validation", "test"}:
        raise ValueError("trajectory confirmation must use a held-out phase")
    grouped_rows: dict[
        str, dict[tuple[int, str], Mapping[str, Any]]] = {}
    variables_by_group: dict[str, str] = {}
    collapsed_duplicates = 0
    for row in rows:
        if row.get("phase") != phase:
            raise ValueError("trajectory rows contain the wrong phase")
        group_id = str(row.get("pair_group_id") or row["example_id"])
        variable = str(row["causal_variable"])
        if (group_id in variables_by_group
                and variables_by_group[group_id] != variable):
            raise ValueError("trajectory pair group changed causal variable")
        variables_by_group[group_id] = variable
        key = (int(row["layer"]), str(row["route"]))
        group = grouped_rows.setdefault(group_id, {})
        if key in group:
            prior = group[key]
            fields = ("captured_mass", "operator_ids", "weights")
            if any(prior[field] != row[field] for field in fields):
                raise ValueError(
                    "cause/isolation rows for one RAVEL base have different "
                    "captured trajectories")
            collapsed_duplicates += 1
            continue
        group[key] = row
    grouped = {
        group_id: [group[key] for key in sorted(group)]
        for group_id, group in grouped_rows.items()
    }
    variable_groups: dict[str, list[str]] = {}
    for group_id, variable in variables_by_group.items():
        variable_groups.setdefault(variable, []).append(group_id)
    variables = sorted(variable_groups)
    if len(variables) < 2:
        return {
            "status": "insufficient_held_out_variables",
            "phase": phase,
            "causal_variable_count": len(variables),
            "address_used_for_discovery": False,
        }

    roles: dict[str, dict[str, list[str]]] = {}
    unused_group_count = 0
    for variable in variables:
        ordered = sorted(
            variable_groups[variable],
            key=lambda group_id: (
                canonical_hash({"seed": int(seed), "group_id": group_id}),
                group_id),
        )
        width = len(ordered) // 3
        roles[variable] = {
            "anchor": ordered[:width],
            "same": ordered[width:2 * width],
            "control": ordered[2 * width:3 * width],
        }
        unused_group_count += len(ordered) - 3 * width
    same_scores: list[float] = []
    cross_scores: list[float] = []
    triplet_ids: list[dict[str, str]] = []
    for variable_index, variable in enumerate(variables):
        control_variable = variables[(variable_index + 1) % len(variables)]
        anchor_ids = roles[variable]["anchor"]
        same_ids = roles[variable]["same"]
        control_ids = roles[control_variable]["control"]
        width = min(len(anchor_ids), len(same_ids), len(control_ids))
        for index in range(width):
            base_id = anchor_ids[index]
            same = same_ids[index]
            cross = control_ids[index]
            same_result = trajectory_similarity(
                grouped[base_id], grouped[same],
                capture_threshold=capture_threshold)
            cross_result = trajectory_similarity(
                grouped[base_id], grouped[cross],
                capture_threshold=capture_threshold)
            same_value = same_result["mean_weighted_jaccard"]
            cross_value = cross_result["mean_weighted_jaccard"]
            if same_value is None or cross_value is None:
                continue
            same_scores.append(float(same_value))
            cross_scores.append(float(cross_value))
            triplet_ids.append({
                "anchor_group_id": base_id,
                "same_variable_group_id": same,
                "cross_variable_group_id": cross,
                "anchor_variable": variable,
                "control_variable": control_variable,
            })
    if len(same_scores) < 2:
        return {
            "status": "insufficient_held_out_pairs",
            "phase": phase,
            "paired_example_count": len(same_scores),
            "address_used_for_discovery": False,
        }
    differences = np.asarray(same_scores) - np.asarray(cross_scores)
    return {
        "status": "ready",
        "phase": phase,
        "paired_example_count": len(same_scores),
        "same_variable_similarity_mean": float(np.mean(same_scores)),
        "cross_variable_similarity_mean": float(np.mean(cross_scores)),
        "same_minus_cross_mean": float(np.mean(differences)),
        "effect_ci": bootstrap_mean_ci(
            differences, samples=bootstrap_samples, alpha=alpha, seed=seed),
        "paired_null": paired_permutation_test(
            same_scores, cross_scores, samples=permutation_samples,
            seed=seed + 1),
        "triplet_group_ids": triplet_ids,
        "same_variable_pairing": "deterministic_disjoint_ravel_base_groups",
        "cross_variable_control": "next_variable_disjoint_control_partition",
        "statistical_unit": "official_ravel_base_row",
        "cause_isolation_duplicate_path_rows_collapsed": collapsed_duplicates,
        "unused_group_count": unused_group_count,
        "capture_threshold": float(capture_threshold),
        "held_out": True,
        "address_used_for_discovery": False,
    }
