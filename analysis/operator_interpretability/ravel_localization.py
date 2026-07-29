"""Preregistered discovery-only RAVEL localization and circuit freezing."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from analysis.operator_interpretability.benchmark_schema import (
    RAVEL_SOURCE_COLUMNS,
    RAVEL_VARIABLES,
    canonical_hash,
)
from analysis.operator_interpretability.units import ROUTES


RAVEL_DISCOVERY_SPEC_RELATIVE_PATH = (
    "configs/paper_ravel_discovery_localization_v4172_400m.yaml"
)
RAVEL_DISCOVERY_SPEC_CONTENT_HASH = (
    "812ea210137a7fc4b2d5956f59975a5c8efbb8881569110455add15a67aa10df"
)
RAVEL_DISCOVERY_ROW_COUNT = 512
RAVEL_DISCOVERY_PAIRED_CORRECT_COUNT = 142


def _expect(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(
            f"RAVEL discovery specification drift at {name}: "
            f"expected={expected!r} actual={value!r}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RAVEL discovery {name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"RAVEL discovery {name} must be an integer")
    if value <= 0:
        raise ValueError(f"RAVEL discovery {name} must be positive")
    return int(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_spec_path(path: str | Path | None) -> Path:
    value = Path(path or RAVEL_DISCOVERY_SPEC_RELATIVE_PATH)
    return value if value.is_absolute() else _repo_root() / value


def _validate_semantics(payload: Mapping[str, Any]) -> None:
    _expect(payload.get("schema_version"), 1, "schema_version")
    _expect(
        payload.get("status"),
        "preregistered_before_discovery_localization",
        "status",
    )
    target = _mapping(payload.get("target"), "target")
    _expect(target.get("target_id"), "v4172_400M", "target.target_id")
    _expect(
        target.get("model_version"),
        "spatial-r1-v4.1.7.2",
        "target.model_version",
    )
    _expect(target.get("checkpoint_step"), 76293, "target.checkpoint_step")

    discovery = _mapping(payload.get("discovery"), "discovery")
    _expect(discovery.get("benchmark_id"), "ravel", "discovery.benchmark_id")
    _expect(discovery.get("phase"), "discovery", "discovery.phase")
    _expect(
        discovery.get("runtime_selected_row_count"),
        RAVEL_DISCOVERY_ROW_COUNT,
        "discovery.runtime_selected_row_count",
    )
    _expect(
        discovery.get("expected_paired_correct_independent_units"),
        RAVEL_DISCOVERY_PAIRED_CORRECT_COUNT,
        "discovery.expected_paired_correct_independent_units",
    )
    _expect(
        tuple(discovery.get("variables") or ()),
        RAVEL_VARIABLES,
        "discovery.variables",
    )
    _expect(
        tuple(discovery.get("official_counterfactual_columns") or ()),
        RAVEL_SOURCE_COLUMNS,
        "discovery.official_counterfactual_columns",
    )
    for name in (
            "selection_uses_validation",
            "selection_uses_test",
            "validation_data_accessor_allowed",
            "test_data_accessor_allowed"):
        _expect(discovery.get(name), False, f"discovery.{name}")

    capture = _mapping(payload.get("capture"), "capture")
    _expect(capture.get("production_precision"), True, "capture.production_precision")
    _expect(capture.get("capture_threshold"), 0.95, "capture.capture_threshold")
    _expect(
        _mapping(capture.get("initial_widths"), "capture.initial_widths"),
        {"qk": 512, "v": 2048, "rst": 4096},
        "capture.initial_widths",
    )
    _expect(
        _mapping(capture.get("maximum_widths"), "capture.maximum_widths"),
        {"qk": 2048, "v": 8192, "rst": 8192},
        "capture.maximum_widths",
    )

    ranking = _mapping(payload.get("ranking"), "ranking")
    _expect(ranking.get("per_variable"), True, "ranking.per_variable")
    _expect(
        ranking.get("score"),
        "absolute_discovery_mean_contribution_importance",
        "ranking.score",
    )
    _expect(tuple(ranking.get("routes") or ()), ROUTES, "ranking.routes")
    stability = _mapping(
        ranking.get("rank_stability"), "ranking.rank_stability")
    _expect(
        stability.get("split_rule"),
        "seeded_balanced_pair_group_split_within_causal_variable_and_"
        "official_counterfactual_column",
        "ranking.rank_stability.split_rule",
    )
    _expect(
        stability.get("threshold_minimum"),
        0.80,
        "ranking.rank_stability.threshold_minimum",
    )
    _expect(
        stability.get("every_variable_must_pass"),
        True,
        "ranking.rank_stability.every_variable_must_pass",
    )
    overlap = _mapping(ranking.get("topk_overlap"), "ranking.topk_overlap")
    _expect(
        tuple(overlap.get("prefix_counts") or ()),
        (32, 64, 128, 256, 512, 1024, 2048, 4096),
        "ranking.topk_overlap.prefix_counts",
    )

    freeze = _mapping(payload.get("circuit_freeze"), "circuit_freeze")
    _expect(freeze.get("per_variable"), True, "circuit_freeze.per_variable")
    _expect(
        freeze.get("rule"),
        "smallest_audited_prefix_passing_both_discovery_gates",
        "circuit_freeze.rule",
    )
    gates = _mapping(
        freeze.get("discovery_gates"), "circuit_freeze.discovery_gates")
    _expect(
        gates.get("cumulative_absolute_importance_minimum"),
        0.70,
        "circuit_freeze.discovery_gates."
        "cumulative_absolute_importance_minimum",
    )
    _expect(
        gates.get("split_topk_overlap_minimum"),
        0.95,
        "circuit_freeze.discovery_gates.split_topk_overlap_minimum",
    )
    for name in (
            "route_presence_is_a_discovery_result",
            "result_dependent_route_inclusion_forbidden",
            "result_dependent_route_exclusion_forbidden"):
        _expect(freeze.get(name), True, f"circuit_freeze.{name}")
    for name in (
            "validation_may_change_circuit",
            "test_may_change_circuit",
            "operator_reselection_after_discovery"):
        _expect(freeze.get(name), False, f"circuit_freeze.{name}")

    storage = _mapping(payload.get("storage"), "storage")
    for name in (
            "preserve_raw_per_example_behavior_vectors",
            "preserve_raw_per_example_operator_vectors",
            "preserve_raw_capture_rows"):
        _expect(storage.get(name), False, f"storage.{name}")


@dataclass(frozen=True)
class RavelDiscoverySpec:
    path: str
    content_hash: str
    payload: dict[str, Any]

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.payload["discovery"]["variables"])

    @property
    def prefix_counts(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.payload["ranking"]["topk_overlap"][
                "prefix_counts"])

    @property
    def rank_stability_minimum(self) -> float:
        return float(
            self.payload["ranking"]["rank_stability"]["threshold_minimum"])

    @property
    def cumulative_importance_minimum(self) -> float:
        return float(
            self.payload["circuit_freeze"]["discovery_gates"][
                "cumulative_absolute_importance_minimum"])

    @property
    def split_overlap_minimum(self) -> float:
        return float(
            self.payload["circuit_freeze"]["discovery_gates"][
                "split_topk_overlap_minimum"])

    def validate_runtime(
            self, *, target_id: str, model_version: str,
            checkpoint_step: int, checkpoint_identity: str,
            checkpoint_config_hash: str, model_config_hash: str,
            benchmark_build_id: str, benchmark_manifest_hash: str,
            seed: int, ravel_max_examples_per_phase: int,
            capture_threshold: float, capture_widths: tuple[int, ...],
            rank_stability_minimum: float) -> None:
        target = self.payload["target"]
        discovery = self.payload["discovery"]
        capture = self.payload["capture"]
        _expect(target_id, target["target_id"], "runtime.target_id")
        _expect(model_version, target["model_version"], "runtime.model_version")
        _expect(
            int(checkpoint_step),
            int(target["checkpoint_step"]),
            "runtime.checkpoint_step",
        )
        _expect(
            checkpoint_identity,
            target["checkpoint_identity"],
            "runtime.checkpoint_identity",
        )
        _expect(
            checkpoint_config_hash,
            target["checkpoint_config_hash"],
            "runtime.checkpoint_config_hash",
        )
        _expect(
            model_config_hash,
            target["model_config_hash"],
            "runtime.model_config_hash",
        )
        _expect(
            benchmark_build_id,
            discovery["benchmark_build_id"],
            "runtime.benchmark_build_id",
        )
        _expect(
            benchmark_manifest_hash,
            discovery["benchmark_manifest_hash"],
            "runtime.benchmark_manifest_hash",
        )
        _expect(int(seed), int(discovery["seed"]), "runtime.seed")
        _expect(
            int(ravel_max_examples_per_phase),
            int(discovery["runtime_selected_row_count"]),
            "runtime.ravel_max_examples_per_phase",
        )
        _expect(
            float(capture_threshold),
            float(capture["capture_threshold"]),
            "runtime.capture_threshold",
        )
        expected_widths = (
            int(capture["initial_widths"]["qk"]),
            int(capture["initial_widths"]["v"]),
            int(capture["initial_widths"]["rst"]),
            int(capture["maximum_widths"]["qk"]),
            int(capture["maximum_widths"]["v"]),
            int(capture["maximum_widths"]["rst"]),
        )
        _expect(
            tuple(int(value) for value in capture_widths),
            expected_widths,
            "runtime.capture_widths",
        )
        _expect(
            float(rank_stability_minimum),
            self.rank_stability_minimum,
            "runtime.rank_stability_minimum",
        )


def load_ravel_discovery_spec(
        path: str | Path | None = None) -> RavelDiscoverySpec:
    resolved = _resolve_spec_path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("RAVEL discovery specification must be a mapping")
    normalized = dict(payload)
    content_hash = canonical_hash(normalized)
    _expect(
        content_hash,
        RAVEL_DISCOVERY_SPEC_CONTENT_HASH,
        "content_hash",
    )
    _validate_semantics(normalized)
    return RavelDiscoverySpec(
        path=str(resolved),
        content_hash=content_hash,
        payload=normalized,
    )


def _site_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["layer"]),
        str(row["route"]),
        int(row["operator_id"]),
    )


def _site_identity(key: tuple[int, str, int]) -> dict[str, Any]:
    return {
        "layer": int(key[0]),
        "route": str(key[1]),
        "operator_id": int(key[2]),
    }


def _topk_overlap(
        split_totals: tuple[Mapping[tuple[int, str, int], float],
                            Mapping[tuple[int, str, int], float]],
        split_denominators: tuple[Mapping[tuple[int, str], int],
                                  Mapping[tuple[int, str], int]],
        prefix_counts: Sequence[int],
) -> list[dict[str, Any]]:
    rankings: list[list[tuple[int, str, int]]] = []
    for split in (0, 1):
        keys = set(split_totals[split])
        scored = [
            (
                float(split_totals[split][key])
                / max(int(split_denominators[split].get(
                    (key[0], key[1]), 0)), 1),
                key,
            )
            for key in keys
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        rankings.append([key for _, key in scored])
    rows: list[dict[str, Any]] = []
    for requested in prefix_counts:
        count = min(int(requested), len(rankings[0]), len(rankings[1]))
        if count <= 0:
            continue
        split_sets = (set(rankings[0][:count]), set(rankings[1][:count]))
        shared = split_sets[0] & split_sets[1]
        union = split_sets[0] | split_sets[1]
        rows.append({
            "requested_prefix_count": int(requested),
            "evaluated_prefix_count": int(count),
            "shared_site_count": len(shared),
            "overlap_fraction_of_each_prefix": len(shared) / count,
            "jaccard": len(shared) / max(len(union), 1),
            "shared_site_count_by_route": {
                route: sum(key[1] == route for key in shared)
                for route in ROUTES
            },
        })
    return rows


def build_ravel_variable_localization(
        rows: Sequence[Mapping[str, Any]], *,
        spec: RavelDiscoverySpec,
        rank_stability_by_variable: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate transient discovery rows into frozen variable circuits."""
    variables = spec.variables
    denominators = {
        variable: defaultdict(int) for variable in variables}
    mass_totals = {
        variable: defaultdict(float) for variable in variables}
    absolute_totals = {
        variable: defaultdict(float) for variable in variables}
    counts = {
        variable: defaultdict(int) for variable in variables}
    split_denominators = {
        variable: (defaultdict(int), defaultdict(int))
        for variable in variables
    }
    split_totals = {
        variable: (defaultdict(float), defaultdict(float))
        for variable in variables
    }
    qualified_rows = {variable: 0 for variable in variables}

    for row in rows:
        variable = str(row.get("causal_variable"))
        if variable not in variables:
            raise ValueError(
                f"unexpected RAVEL discovery variable={variable!r}")
        if str(row.get("pair_type")) != "cause":
            raise ValueError("RAVEL localization received a non-cause row")
        if not bool(row.get("qualified")):
            continue
        layer = int(row["layer"])
        route = str(row["route"])
        split = int(row["discovery_split"])
        if route not in ROUTES or split not in (0, 1):
            raise ValueError("RAVEL discovery row identity is invalid")
        denominator_key = (layer, route)
        denominators[variable][denominator_key] += 1
        mass_totals[variable][denominator_key] += float(row["captured_mass"])
        split_denominators[variable][split][denominator_key] += 1
        qualified_rows[variable] += 1
        operator_ids = list(row.get("operator_ids") or ())
        weights = list(row.get("weights") or ())
        if len(operator_ids) != len(weights):
            raise ValueError("RAVEL discovery operator arrays are misaligned")
        for operator_id, weight in zip(operator_ids, weights):
            key = (layer, route, int(operator_id))
            value = abs(float(weight))
            if not math.isfinite(value):
                raise ValueError("RAVEL discovery contribution is nonfinite")
            absolute_totals[variable][key] += value
            split_totals[variable][split][key] += value
            counts[variable][key] += 1

    output: dict[str, dict[str, Any]] = {}
    for variable in variables:
        ranked = []
        for key, total in absolute_totals[variable].items():
            denominator = denominators[variable][(key[0], key[1])]
            ranked.append({
                **_site_identity(key),
                "importance": float(total) / max(int(denominator), 1),
                "discovery_count": int(counts[variable][key]),
                "captured_mass_mean": (
                    float(mass_totals[variable][(key[0], key[1])])
                    / max(int(denominator), 1)),
            })
        ranked.sort(key=lambda row: (-float(row["importance"]), _site_key(row)))
        total_importance = float(sum(
            float(row["importance"]) for row in ranked))
        if not ranked or not math.isfinite(total_importance) or (
                total_importance <= 0.0):
            output[variable] = {
                "status": "no_qualified_discovery_site",
                "ranked_site_count": len(ranked),
                "qualified_row_count": qualified_rows[variable],
            }
            continue

        overlap_rows = _topk_overlap(
            split_totals[variable],
            split_denominators[variable],
            spec.prefix_counts,
        )
        overlap_by_prefix = {
            int(row["requested_prefix_count"]): row
            for row in overlap_rows
        }
        cumulative = 0.0
        cumulative_by_prefix: dict[int, float] = {}
        prefix_set = set(spec.prefix_counts)
        for rank, row in enumerate(ranked, start=1):
            cumulative += float(row["importance"])
            if rank in prefix_set:
                cumulative_by_prefix[rank] = cumulative / total_importance
        prefix_audit = []
        selected_prefix: int | None = None
        for prefix in spec.prefix_counts:
            overlap = overlap_by_prefix.get(prefix)
            cumulative_fraction = cumulative_by_prefix.get(prefix)
            evaluated_exact_prefix = (
                overlap is not None
                and int(overlap["evaluated_prefix_count"]) == int(prefix)
                and cumulative_fraction is not None)
            passed = bool(
                evaluated_exact_prefix
                and float(cumulative_fraction)
                >= spec.cumulative_importance_minimum
                and float(overlap["overlap_fraction_of_each_prefix"])
                >= spec.split_overlap_minimum)
            prefix_audit.append({
                "prefix_count": int(prefix),
                "cumulative_absolute_importance_fraction": (
                    float(cumulative_fraction)
                    if cumulative_fraction is not None else None),
                "split_topk_overlap": (
                    float(overlap["overlap_fraction_of_each_prefix"])
                    if overlap is not None else None),
                "split_topk_jaccard": (
                    float(overlap["jaccard"])
                    if overlap is not None else None),
                "passed": passed,
            })
            if selected_prefix is None and passed:
                selected_prefix = int(prefix)

        stability = dict(rank_stability_by_variable.get(variable) or {})
        stability_value = stability.get("rank_stability")
        stability_passed = bool(
            stability.get("status") == "ready"
            and stability_value is not None
            and float(stability_value) >= spec.rank_stability_minimum)
        if not stability_passed:
            status = "unstable_localization"
            selected_prefix = None
        elif selected_prefix is None:
            status = "no_preregistered_prefix"
        else:
            status = "ready"

        importance_by_route = {
            route: float(sum(
                float(row["importance"]) for row in ranked
                if row["route"] == route))
            for route in ROUTES
        }
        importance_by_layer = {
            str(layer): float(sum(
                float(row["importance"]) for row in ranked
                if int(row["layer"]) == layer))
            for layer in sorted({int(row["layer"]) for row in ranked})
        }
        first_rank_by_route = {}
        for route in ROUTES:
            first = next((
                rank for rank, row in enumerate(ranked, start=1)
                if row["route"] == route), None)
            first_rank_by_route[route] = first

        circuit: dict[str, Any]
        if selected_prefix is None:
            circuit = {
                "status": "not_frozen",
                "reason": status,
                "selected_k": 0,
                "sites": [],
            }
        else:
            selected_rows = ranked[:selected_prefix]
            identities = [
                {
                    "layer": int(row["layer"]),
                    "route": str(row["route"]),
                    "operator_id": int(row["operator_id"]),
                }
                for row in selected_rows
            ]
            layer_route_counts = {
                str(layer): {
                    route: sum(
                        int(row["layer"]) == layer
                        and row["route"] == route
                        for row in selected_rows)
                    for route in ROUTES
                }
                for layer in sorted({
                    int(row["layer"]) for row in selected_rows})
            }
            selected_audit = next(
                row for row in prefix_audit
                if int(row["prefix_count"]) == selected_prefix)
            circuit = {
                "status": "frozen_from_discovery",
                "variable": variable,
                "selected_k": selected_prefix,
                "selected_layers": sorted({
                    int(row["layer"]) for row in selected_rows}),
                "selected_route_counts": {
                    route: sum(
                        row["route"] == route for row in selected_rows)
                    for route in ROUTES
                },
                "selected_layer_route_counts": layer_route_counts,
                "cumulative_absolute_importance_fraction": (
                    selected_audit[
                        "cumulative_absolute_importance_fraction"]),
                "discovery_split_topk_overlap": (
                    selected_audit["split_topk_overlap"]),
                "discovery_split_topk_jaccard": (
                    selected_audit["split_topk_jaccard"]),
                "top_importance": float(selected_rows[0]["importance"]),
                "boundary_importance": float(
                    selected_rows[-1]["importance"]),
                "selected_ranked_rows_hash": canonical_hash(selected_rows),
                "selected_site_identity_hash": canonical_hash(identities),
                "circuit_hash": canonical_hash({
                    "benchmark_id": "ravel",
                    "variable": variable,
                    "sites": identities,
                }),
                "sites": identities,
            }

        output[variable] = {
            "status": status,
            "variable": variable,
            "ranking_phase": "discovery",
            "ranking_score": (
                "absolute_discovery_mean_contribution_importance"),
            "ranked_site_count": len(ranked),
            "ranked_sites_content_hash": canonical_hash(ranked),
            "ranked_sites": ranked,
            "qualified_row_count": int(qualified_rows[variable]),
            "total_absolute_importance": total_importance,
            "absolute_importance_by_route": importance_by_route,
            "absolute_importance_fraction_by_route": {
                route: importance_by_route[route] / total_importance
                for route in ROUTES
            },
            "absolute_importance_by_layer": importance_by_layer,
            "first_rank_by_route": first_rank_by_route,
            "rank_stability": stability,
            "rank_stability_gate_threshold": (
                spec.rank_stability_minimum),
            "rank_stability_gate_passed": stability_passed,
            "discovery_split_topk_overlap": overlap_rows,
            "circuit_prefix_audit": prefix_audit,
            "circuit_freeze_rule": (
                "smallest_audited_prefix_passing_both_discovery_gates"),
            "circuit": circuit,
        }
    return output
