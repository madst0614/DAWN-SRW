"""MIB-adapted operator-site circuit scoring and selection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.operator_interpretability.protocol import CIRCUIT_FRACTIONS


def normalized_faithfulness(circuit_score: float, baseline_score: float,
                            corrupted_score: float) -> float | None:
    denominator = float(baseline_score) - float(corrupted_score)
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None
    return (float(circuit_score) - float(corrupted_score)) / denominator


def bootstrap_faithfulness_ci(
        circuit_margin: Sequence[float], baseline_margin: Sequence[float],
        corrupted_margin: Sequence[float], *, samples: int, alpha: float,
        seed: int) -> dict[str, Any]:
    circuit = np.asarray(circuit_margin, dtype=np.float64)
    baseline = np.asarray(baseline_margin, dtype=np.float64)
    corrupted = np.asarray(corrupted_margin, dtype=np.float64)
    if not (circuit.shape == baseline.shape == corrupted.shape) or circuit.ndim != 1:
        raise ValueError("faithfulness bootstrap vectors must align")
    if circuit.size < 2 or samples < 100:
        raise ValueError("faithfulness bootstrap requires >=2 rows and >=100 samples")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, circuit.size, size=(int(samples), circuit.size))
    circuit_mean = circuit[indices].mean(axis=1)
    baseline_mean = baseline[indices].mean(axis=1)
    corrupted_mean = corrupted[indices].mean(axis=1)
    denominator = baseline_mean - corrupted_mean
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12)
    estimates = (
        circuit_mean[valid] - corrupted_mean[valid]) / denominator[valid]
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size < max(100, int(0.9 * samples)):
        raise ValueError("too many undefined faithfulness bootstrap draws")
    low, high = np.quantile(
        estimates, [float(alpha) / 2.0, 1.0 - float(alpha) / 2.0])
    return {
        "samples": int(samples),
        "valid_samples": int(estimates.size),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def faithfulness_curve(
        rows: Sequence[Mapping[str, Any]], *, log_scale: bool = False) -> dict[str, Any]:
    if tuple(float(row["fraction"]) for row in rows) != CIRCUIT_FRACTIONS:
        raise ValueError("circuit curve must use the complete registered MIB grid")
    faithfulness = [row.get("faithfulness") for row in rows]
    if any(value is None or not math.isfinite(float(value)) for value in faithfulness):
        raise ValueError("circuit curve contains undefined faithfulness")
    x = np.asarray(CIRCUIT_FRACTIONS, dtype=np.float64)
    if log_scale:
        x = np.log(x)
    y = np.asarray(faithfulness, dtype=np.float64)
    area_under = float(np.trapezoid(y, x))
    area_from_one = float(np.trapezoid(np.abs(1.0 - y), x))
    return {
        "fractions": list(CIRCUIT_FRACTIONS),
        "faithfulness": y.tolist(),
        "area_under_curve": area_under,
        "area_from_one": area_from_one,
        "mean_faithfulness": float(y.mean()),
        "log_scale": bool(log_scale),
    }


def necessity_effect(baseline_margin: Sequence[float],
                     suppressed_margin: Sequence[float]) -> dict[str, Any]:
    baseline = np.asarray(baseline_margin, dtype=np.float64)
    suppressed = np.asarray(suppressed_margin, dtype=np.float64)
    if baseline.shape != suppressed.shape:
        raise ValueError("necessity score vectors differ in shape")
    mask = np.isfinite(baseline) & np.isfinite(suppressed)
    drop = baseline[mask] - suppressed[mask]
    return {
        "n": int(drop.size),
        "mean_margin_drop": float(drop.mean()) if drop.size else None,
        "median_margin_drop": float(np.median(drop)) if drop.size else None,
        "positive_drop_fraction": float(np.mean(drop > 0.0)) if drop.size else None,
    }


def select_on_validation(
        validation_rows: Sequence[Mapping[str, Any]], *,
        minimum_faithfulness: float) -> dict[str, Any]:
    """Choose the smallest passing circuit without inspecting test results."""
    candidates = sorted(validation_rows, key=lambda row: float(row["fraction"]))
    passing = [
        row for row in candidates
        if row.get("faithfulness") is not None
        and (row.get("faithfulness_ci") or {}).get("ci_low") is not None
        and float(row["faithfulness_ci"]["ci_low"])
        >= float(minimum_faithfulness)
    ]
    if not passing:
        return {
            "status": "no_passing_validation_circuit",
            "selected_fraction": None,
            "selection_phase": "validation",
            "test_consulted": False,
        }
    selected = passing[0]
    return {
        "status": "selected",
        "selected_fraction": float(selected["fraction"]),
        "selected_site_count": int(selected["site_count"]),
        "validation_faithfulness": float(selected["faithfulness"]),
        "validation_faithfulness_ci": dict(selected["faithfulness_ci"]),
        "selection_threshold_applied_to": "bootstrap_ci_low",
        "selection_phase": "validation",
        "test_consulted": False,
    }
