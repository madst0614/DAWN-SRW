"""Deterministic uncertainty, null tests, and multiple-comparison control."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def bootstrap_mean_ci(values: Iterable[float], *, samples: int, alpha: float,
                      seed: int) -> dict[str, float | int | None]:
    array = _finite(values)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    if samples < 100:
        raise ValueError("bootstrap samples must be >= 100")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_permutation_test(a: Iterable[float], b: Iterable[float], *,
                            samples: int, seed: int) -> dict[str, float | int]:
    left = np.asarray(list(a), dtype=np.float64)
    right = np.asarray(list(b), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired permutation inputs must be equal-length vectors")
    valid = np.isfinite(left) & np.isfinite(right)
    differences = left[valid] - right[valid]
    if differences.size < 2:
        raise ValueError("paired permutation test requires at least two pairs")
    observed = float(np.mean(differences))
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(
        np.asarray((-1.0, 1.0)), size=(int(samples), differences.size))
    null = np.mean(signs * differences[None, :], axis=1)
    p_value = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (
        float(samples) + 1.0)
    return {
        "n": int(differences.size),
        "mean_difference": observed,
        "p_value_two_sided": float(p_value),
        "permutation_samples": int(samples),
    }


def spearman_rank(a: Iterable[float], b: Iterable[float]) -> float | None:
    left = np.asarray(list(a), dtype=np.float64)
    right = np.asarray(list(b), dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        return None
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if left.size < 2:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result

    left_rank, right_rank = ranks(left), ranks(right)
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def benjamini_hochberg(p_values: Sequence[float], alpha: float) -> dict[str, list]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a vector in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.minimum.accumulate(
        (values[order] * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return {
        "adjusted_p_values": adjusted.tolist(),
        "reject": (adjusted <= float(alpha)).tolist(),
    }
