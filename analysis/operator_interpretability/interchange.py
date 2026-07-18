"""RAVEL-style causal mediation scores for DAWN route contributions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


INTERCHANGE_SEMANTICS = (
    "base_route = base_route - selected_family(base) "
    "+ selected_family(source)"
)


def score_interchange_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("interchange evaluation contains no rows")
    cause = [row for row in rows if row.get("pair_type") == "cause"]
    isolation = [row for row in rows if row.get("pair_type") == "isolation"]
    if not cause or not isolation:
        raise ValueError("RAVEL evaluation requires both cause and isolation pairs")

    cause_effect = np.asarray([
        float(row["patched_intervention_margin"])
        - float(row["base_intervention_margin"])
        for row in cause
    ], dtype=np.float64)
    cause_success = np.asarray([
        float(row["patched_intervention_margin"]) > 0.0
        for row in cause
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
        "cause_effect_mean": float(cause_effect.mean()),
        "cause_effect_median": float(np.median(cause_effect)),
        "cause_effect_definition": (
            "patched_intervention_margin_minus_unpatched_intervention_margin"),
        "cause_success_definition": (
            "patched_intervention_target_margin_greater_than_zero"),
        "isolation_absolute_effect_mean": float(isolation_effect.mean()),
        "isolation_absolute_effect_max": float(isolation_effect.max()),
        "unit_of_intervention": "post_denominator_pool_scaled_operator_family_contribution",
        "official_ravel_featurizer_equivalence_claimed": False,
    }
