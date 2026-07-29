"""Preregistered discovery-only ARC localization and circuit freezing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.operator_interpretability.units import ROUTES


ARC_DISCOVERY_SPEC_RELATIVE_PATH = (
    "configs/paper_arc_discovery_localization_v4172_400m.yaml"
)
ARC_DISCOVERY_SPEC_CONTENT_HASH = (
    "70ff8bcab6b1ad0448c29c582f0fc204e834a4b7c9800e925591d32f82cc6b7b"
)
ARC_DISCOVERY_ROW_COUNT = 128
ARC_DISCOVERY_PAIRED_CORRECT_COUNT = 50


def _expect(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(
            f"ARC discovery specification drift at {name}: "
            f"expected={expected!r} actual={value!r}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"ARC discovery {name} must be a mapping")
    return dict(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_spec_path(path: str | Path | None) -> Path:
    value = Path(path or ARC_DISCOVERY_SPEC_RELATIVE_PATH)
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
    _expect(discovery.get("benchmark_id"), "mib_arc", "discovery.benchmark_id")
    _expect(discovery.get("phase"), "discovery", "discovery.phase")
    _expect(
        discovery.get("runtime_selected_row_count"),
        ARC_DISCOVERY_ROW_COUNT,
        "discovery.runtime_selected_row_count",
    )
    _expect(
        discovery.get("expected_paired_correct_independent_units"),
        ARC_DISCOVERY_PAIRED_CORRECT_COUNT,
        "discovery.expected_paired_correct_independent_units",
    )
    _expect(
        discovery.get("independent_unit"),
        "example_id",
        "discovery.independent_unit",
    )
    _expect(
        discovery.get("localization_pair_type"),
        "symbol_counterfactual",
        "discovery.localization_pair_type",
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
        "seeded_balanced_example_id_split",
        "ranking.rank_stability.split_rule",
    )
    _expect(
        tuple(stability.get("expected_split_independent_unit_counts") or ()),
        (25, 25),
        "ranking.rank_stability.expected_split_independent_unit_counts",
    )
    _expect(
        stability.get("threshold_minimum"),
        0.80,
        "ranking.rank_stability.threshold_minimum",
    )
    overlap = _mapping(ranking.get("topk_overlap"), "ranking.topk_overlap")
    _expect(
        tuple(overlap.get("prefix_counts") or ()),
        (32, 64, 128, 256, 512, 1024, 2048, 4096),
        "ranking.topk_overlap.prefix_counts",
    )

    freeze = _mapping(payload.get("circuit_freeze"), "circuit_freeze")
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

    confirmatory = _mapping(
        payload.get("confirmatory_protocol_if_frozen"),
        "confirmatory_protocol_if_frozen",
    )
    validation = _mapping(confirmatory.get("validation"), "confirmatory.validation")
    _expect(
        validation.get("expected_paired_correct_independent_units"),
        51,
        "confirmatory.validation.expected_paired_correct_independent_units",
    )
    _expect(
        tuple(validation.get("conditions") or ()),
        (
            "intact",
            "frozen_circuit_suppression",
            "matched_random_control_x100",
            "exact_restoration",
        ),
        "confirmatory.validation.conditions",
    )
    suppression = _mapping(
        confirmatory.get("suppression"), "confirmatory.suppression")
    _expect(
        suppression.get("mode"),
        "circuit_wide_execution_numerator_suppression",
        "confirmatory.suppression.mode",
    )
    _expect(
        suppression.get("admission_denominator"),
        "full_production_denominator",
        "confirmatory.suppression.admission_denominator",
    )
    control = _mapping(
        confirmatory.get("matched_random_control"),
        "confirmatory.matched_random_control",
    )
    _expect(control.get("replicate_count"), 100, "confirmatory.control.replicates")
    _expect(control.get("seed"), 24172, "confirmatory.control.seed")
    _expect(
        tuple(control.get("match_fields") or ()),
        ("layer", "route"),
        "confirmatory.control.match_fields",
    )
    _expect(
        control.get("sampling"),
        "uniform_without_replacement_within_layer_route_cell",
        "confirmatory.control.sampling",
    )
    for name in (
            "match_exact_frozen_count_per_cell",
            "frozen_sites_excluded",
            "duplicate_site_within_replicate_forbidden",
            "control_site_set_hashes_must_be_unique"):
        _expect(control.get(name), True, f"confirmatory.control.{name}")
    restoration = _mapping(
        confirmatory.get("restoration"), "confirmatory.restoration")
    _expect(
        restoration.get("mode"),
        "exact_selected_numerator_restore_after_suppression",
        "confirmatory.restoration.mode",
    )
    _expect(
        restoration.get("admission_denominator"),
        "full_production_denominator",
        "confirmatory.restoration.admission_denominator",
    )
    statistics = _mapping(
        confirmatory.get("statistics"), "confirmatory.statistics")
    _expect(statistics.get("bootstrap_samples"), 2000, "statistics.bootstrap")
    _expect(
        statistics.get("permutation_samples"), 2000, "statistics.permutation")
    _expect(statistics.get("alpha"), 0.05, "statistics.alpha")
    confirmatory_gates = _mapping(
        confirmatory.get("validation_gates"), "confirmatory.validation_gates")
    _expect(
        confirmatory_gates.get(
            "suppression_margin_drop_ci_low_above_zero"),
        True,
        "confirmatory.validation_gates.suppression",
    )
    _expect(
        confirmatory_gates.get(
            "frozen_minus_matched_random_margin_drop_ci_low_above_zero"),
        True,
        "confirmatory.validation_gates.control",
    )
    _expect(
        confirmatory_gates.get("restoration_recovery_ci_low_minimum"),
        0.80,
        "confirmatory.validation_gates.restoration",
    )
    held_out = _mapping(
        confirmatory.get("held_out_test"), "confirmatory.held_out_test")
    _expect(
        held_out.get("expected_paired_correct_independent_units"),
        57,
        "confirmatory.held_out_test.expected_paired_correct_independent_units",
    )
    _expect(
        held_out.get("opened_only_after_validation_record_is_final"),
        True,
        "confirmatory.held_out_test.opened_only_after_validation_record_is_final",
    )

    storage = _mapping(payload.get("storage"), "storage")
    for name in (
            "preserve_raw_per_example_behavior_vectors",
            "preserve_raw_per_example_operator_vectors",
            "preserve_raw_capture_rows"):
        _expect(storage.get(name), False, f"storage.{name}")


@dataclass(frozen=True)
class ArcDiscoverySpec:
    path: str
    content_hash: str
    payload: dict[str, Any]

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
            seed: int, max_examples_per_phase: int,
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
            int(max_examples_per_phase),
            int(discovery["runtime_selected_row_count"]),
            "runtime.max_examples_per_phase",
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


def load_arc_discovery_spec(
        path: str | Path | None = None) -> ArcDiscoverySpec:
    resolved = _resolve_spec_path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ARC discovery specification must be a mapping")
    normalized = dict(payload)
    content_hash = canonical_hash(normalized)
    _expect(
        content_hash,
        ARC_DISCOVERY_SPEC_CONTENT_HASH,
        "content_hash",
    )
    _validate_semantics(normalized)
    return ArcDiscoverySpec(
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


def build_arc_localization(
        ranked_sites: Sequence[Mapping[str, Any]], *,
        capture: Mapping[str, Any],
        spec: ArcDiscoverySpec,
        benchmark_id: str = "mib_arc",
) -> dict[str, Any]:
    """Apply preregistered discovery gates to one aggregate MIB ranking."""
    ranked = []
    seen: set[tuple[int, str, int]] = set()
    previous_sort_key: tuple[float, tuple[int, str, int]] | None = None
    for source in ranked_sites:
        row = dict(source)
        key = _site_key(row)
        if key in seen:
            raise ValueError("ARC discovery ranking contains a duplicate site")
        seen.add(key)
        if key[1] not in ROUTES:
            raise ValueError("ARC discovery ranking contains an invalid route")
        importance = float(row["importance"])
        if not math.isfinite(importance) or importance < 0.0:
            raise ValueError("ARC discovery importance must be finite and nonnegative")
        sort_key = (-importance, key)
        if previous_sort_key is not None and sort_key < previous_sort_key:
            raise ValueError("ARC discovery ranking is not canonically sorted")
        previous_sort_key = sort_key
        ranked.append({
            "layer": key[0],
            "route": key[1],
            "operator_id": key[2],
            "importance": importance,
            "discovery_count": int(row["discovery_count"]),
            "captured_mass_mean": float(row["captured_mass_mean"]),
        })

    total_importance = float(sum(row["importance"] for row in ranked))
    if not ranked or not math.isfinite(total_importance) or total_importance <= 0.0:
        return {
            "status": "no_qualified_discovery_site",
            "ranking_phase": "discovery",
            "ranked_site_count": len(ranked),
        }

    split_rule = str(capture.get("rank_stability_split_rule") or "")
    _expect(
        split_rule,
        spec.payload["ranking"]["rank_stability"]["split_rule"],
        "capture.rank_stability_split_rule",
    )
    split_counts = tuple(
        int(value)
        for value in capture.get(
            "rank_stability_split_independent_group_counts", ()))
    stability_spec = spec.payload["ranking"]["rank_stability"]
    expected_split_counts = stability_spec.get(
        "expected_split_independent_unit_counts")
    if expected_split_counts is not None:
        _expect(
            split_counts,
            tuple(int(value) for value in expected_split_counts),
            "capture.rank_stability_split_independent_group_counts",
        )
    else:
        minimum_per_split = int(
            stability_spec["minimum_independent_units_per_split"])
        if len(split_counts) != 2 or min(split_counts) < minimum_per_split:
            raise ValueError(
                "discovery rank-stability split is smaller than the "
                "preregistered minimum")
    stability_value = capture.get("rank_stability")
    stability_passed = bool(
        capture.get("status") == "ready"
        and stability_value is not None
        and math.isfinite(float(stability_value))
        and float(stability_value) >= spec.rank_stability_minimum)

    overlap_by_prefix = {
        int(row["requested_prefix_count"]): dict(row)
        for row in capture.get("discovery_split_topk_overlap", ())
        if isinstance(row, Mapping)
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
        evaluated_exact_prefix = bool(
            overlap is not None
            and int(overlap.get("evaluated_prefix_count", 0)) == int(prefix)
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

    if not stability_passed:
        status = "unstable_localization"
        selected_prefix = None
    elif selected_prefix is None:
        status = "no_preregistered_prefix"
    else:
        status = "ready"

    importance_by_route = {
        route: float(sum(
            row["importance"] for row in ranked if row["route"] == route))
        for route in ROUTES
    }
    importance_by_layer = {
        str(layer): float(sum(
            row["importance"] for row in ranked
            if int(row["layer"]) == layer))
        for layer in sorted({int(row["layer"]) for row in ranked})
    }
    first_rank_by_route = {
        route: next((
            rank for rank, row in enumerate(ranked, start=1)
            if row["route"] == route), None)
        for route in ROUTES
    }

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
                    int(row["layer"]) == layer and row["route"] == route
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
            "selected_k": selected_prefix,
            "selected_layers": sorted({
                int(row["layer"]) for row in selected_rows}),
            "selected_route_counts": {
                route: sum(row["route"] == route for row in selected_rows)
                for route in ROUTES
            },
            "selected_layer_route_counts": layer_route_counts,
            "cumulative_absolute_importance_fraction": selected_audit[
                "cumulative_absolute_importance_fraction"],
            "discovery_split_topk_overlap": selected_audit[
                "split_topk_overlap"],
            "discovery_split_topk_jaccard": selected_audit[
                "split_topk_jaccard"],
            "top_importance": float(selected_rows[0]["importance"]),
            "boundary_importance": float(selected_rows[-1]["importance"]),
            "selected_ranked_rows_hash": canonical_hash(selected_rows),
            "selected_site_identity_hash": canonical_hash(identities),
            "circuit_hash": canonical_hash({
                "benchmark_id": benchmark_id,
                "sites": identities,
            }),
            "sites": identities,
        }

    return {
        "status": status,
        "benchmark": benchmark_id,
        "ranking_phase": "discovery",
        "ranking_score": "absolute_discovery_mean_contribution_importance",
        "ranked_site_count": len(ranked),
        "ranked_sites_content_hash": canonical_hash(ranked),
        "ranked_sites": ranked,
        "qualified_row_count": int(capture["qualified_row_count"]),
        "total_row_count": int(capture["total_row_count"]),
        "total_absolute_importance": total_importance,
        "absolute_importance_by_route": importance_by_route,
        "absolute_importance_fraction_by_route": {
            route: importance_by_route[route] / total_importance
            for route in ROUTES
        },
        "absolute_importance_by_layer": importance_by_layer,
        "first_rank_by_route": first_rank_by_route,
        "rank_stability": {
            "status": (
                "ready" if stability_value is not None else "not_estimable"),
            "rank_stability": (
                float(stability_value)
                if stability_value is not None else None),
            "common_site_count": int(
                capture.get("rank_stability_common_site_count") or 0),
            "split_rule": split_rule,
            "split_independent_unit_counts": list(split_counts),
        },
        "rank_stability_gate_threshold": spec.rank_stability_minimum,
        "rank_stability_gate_passed": stability_passed,
        "discovery_split_topk_overlap": list(
            capture.get("discovery_split_topk_overlap") or ()),
        "circuit_prefix_audit": prefix_audit,
        "circuit_freeze_rule": (
            "smallest_audited_prefix_passing_both_discovery_gates"),
        "circuit": circuit,
    }
