"""RAVEL-style causal mediation scores for DAWN route contributions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


INTERCHANGE_SEMANTICS = (
    "base_route = base_route - selected_family(base) "
    "+ selected_family(source)"
)


def normalized_mediation_effect(base_source_margin: float,
                                source_source_margin: float,
                                patched_source_margin: float) -> float | None:
    denominator = float(source_source_margin) - float(base_source_margin)
    if not np.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None
    return (
        float(patched_source_margin) - float(base_source_margin)
    ) / denominator


def score_interchange_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("interchange evaluation contains no rows")
    cause = [row for row in rows if row.get("pair_type") == "cause"]
    isolation = [row for row in rows if row.get("pair_type") == "isolation"]
    if not cause or not isolation:
        raise ValueError("RAVEL evaluation requires both cause and isolation pairs")

    cause_success = np.asarray([
        float(row["patched_source_margin"]) > float(row["base_source_margin"])
        for row in cause
    ], dtype=np.float64)
    mediation = np.asarray([
        value for value in (
            normalized_mediation_effect(
                row["base_source_margin"], row["source_source_margin"],
                row["patched_source_margin"])
            for row in cause
        ) if value is not None and np.isfinite(value)
    ], dtype=np.float64)
    isolation_effect = np.asarray([
        abs(float(row["patched_base_margin"]) - float(row["base_base_margin"]))
        for row in isolation
    ], dtype=np.float64)
    return {
        "status": "ready",
        "intervention": INTERCHANGE_SEMANTICS,
        "cause_pair_count": len(cause),
        "isolation_pair_count": len(isolation),
        "cause_success_fraction": float(cause_success.mean()),
        "normalized_mediation_mean": (
            float(mediation.mean()) if mediation.size else None),
        "normalized_mediation_median": (
            float(np.median(mediation)) if mediation.size else None),
        "isolation_absolute_effect_mean": float(isolation_effect.mean()),
        "isolation_absolute_effect_max": float(isolation_effect.max()),
        "unit_of_intervention": "post_denominator_pool_scaled_operator_family_contribution",
        "official_ravel_featurizer_equivalence_claimed": False,
    }
