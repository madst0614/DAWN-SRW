"""Preregistered discovery-only IOI localization for the 1.3B replication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from analysis.operator_interpretability.arc_localization import (
    build_arc_localization,
)
from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.operator_interpretability.units import ROUTES


IOI_SCALE_DISCOVERY_SPEC_RELATIVE_PATH = (
    "configs/paper_ioi_scale_discovery_localization_v4172_1p3b.yaml"
)
IOI_SCALE_DISCOVERY_SPEC_CONTENT_HASH = (
    "0fa2462b68fb1c08c6666f48bb63e80bfeec3be2ad3155bce268f4d7a6805348"
)
IOI_SCALE_DISCOVERY_ROW_COUNT = 128
IOI_SCALE_DISCOVERY_MINIMUM_PAIRED_CORRECT_COUNT = 32


def _expect(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(
            f"IOI scale-discovery specification drift at {name}: "
            f"expected={expected!r} actual={value!r}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"IOI scale-discovery {name} must be a mapping")
    return dict(value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_spec_path(path: str | Path | None) -> Path:
    value = Path(path or IOI_SCALE_DISCOVERY_SPEC_RELATIVE_PATH)
    return value if value.is_absolute() else _repo_root() / value


def _validate_semantics(payload: Mapping[str, Any]) -> None:
    _expect(payload.get("schema_version"), 1, "schema_version")
    _expect(
        payload.get("status"),
        "preregistered_before_discovery_localization",
        "status",
    )
    target = _mapping(payload.get("target"), "target")
    _expect(target.get("target_id"), "v4172_1B", "target.target_id")
    _expect(
        target.get("model_version"),
        "spatial-r1-v4.1.7.2",
        "target.model_version",
    )
    _expect(target.get("checkpoint_step"), 87193, "target.checkpoint_step")

    discovery = _mapping(payload.get("discovery"), "discovery")
    _expect(discovery.get("benchmark_id"), "mib_ioi", "discovery.benchmark_id")
    _expect(discovery.get("phase"), "discovery", "discovery.phase")
    _expect(
        discovery.get("runtime_selected_row_count"),
        IOI_SCALE_DISCOVERY_ROW_COUNT,
        "discovery.runtime_selected_row_count",
    )
    _expect(
        discovery.get("independent_unit"),
        "example_id",
        "discovery.independent_unit",
    )
    _expect(
        discovery.get("paired_correct_rule"),
        "all_base_and_source_correct_discovery_rows",
        "discovery.paired_correct_rule",
    )
    _expect(
        discovery.get("minimum_paired_correct_independent_units"),
        IOI_SCALE_DISCOVERY_MINIMUM_PAIRED_CORRECT_COUNT,
        "discovery.minimum_paired_correct_independent_units",
    )
    _expect(
        discovery.get("localization_pair_type"),
        "s2_io_flip_counterfactual",
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
        capture.get("width_scaling_rule"),
        "nearest_power_of_two_preserving_400m_pool_fraction",
        "capture.width_scaling_rule",
    )
    _expect(
        _mapping(capture.get("reference_400m_pool_sizes"), "capture.reference"),
        {"qk": 4600, "v": 14500, "rst": 29300},
        "capture.reference_400m_pool_sizes",
    )
    _expect(
        _mapping(capture.get("target_1p3b_pool_sizes"), "capture.target"),
        {"qk": 8534, "v": 26904, "rst": 54352},
        "capture.target_1p3b_pool_sizes",
    )
    _expect(
        _mapping(capture.get("initial_widths"), "capture.initial_widths"),
        {"qk": 1024, "v": 4096, "rst": 8192},
        "capture.initial_widths",
    )
    _expect(
        _mapping(capture.get("maximum_widths"), "capture.maximum_widths"),
        {"qk": 4096, "v": 16384, "rst": 16384},
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
        "seeded_hash_of_benchmark_and_example_id",
        "ranking.rank_stability.split_rule",
    )
    _expect(
        stability.get("minimum_independent_units_per_split"),
        16,
        "ranking.rank_stability.minimum_independent_units_per_split",
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
        "circuit_freeze.discovery_gates.cumulative_importance",
    )
    _expect(
        gates.get("split_topk_overlap_minimum"),
        0.95,
        "circuit_freeze.discovery_gates.split_overlap",
    )
    for name in (
            "route_presence_is_a_discovery_result",
            "result_dependent_route_inclusion_forbidden",
            "result_dependent_route_exclusion_forbidden",
            "literal_400m_operator_id_transfer_forbidden"):
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
        tuple(validation.get("conditions") or ()),
        (
            "intact",
            "frozen_circuit_suppression",
            "equal_count_random_x100",
            "same_layer_random_x100",
            "activation_matched_x100",
            "route_frequency_matched_x100",
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
    controls = _mapping(confirmatory.get("controls"), "confirmatory.controls")
    _expect(
        controls.get("replicate_count_per_control"),
        100,
        "confirmatory.controls.replicate_count_per_control",
    )
    for name, seed in (
            ("equal_count_random", 14172),
            ("same_layer_random", 14173),
            ("activation_matched", 14174),
            ("route_frequency_matched", 14175)):
        control = _mapping(controls.get(name), f"confirmatory.controls.{name}")
        _expect(control.get("seed"), seed, f"confirmatory.controls.{name}.seed")
    restoration = _mapping(
        confirmatory.get("restoration"), "confirmatory.restoration")
    _expect(
        restoration.get("mode"),
        "exact_selected_numerator_restore_after_suppression",
        "confirmatory.restoration.mode",
    )
    statistics = _mapping(
        confirmatory.get("statistics"), "confirmatory.statistics")
    _expect(statistics.get("bootstrap_samples"), 2000, "statistics.bootstrap")
    _expect(
        statistics.get("permutation_samples"), 2000, "statistics.permutation")
    _expect(statistics.get("alpha"), 0.05, "statistics.alpha")
    validation_gates = _mapping(
        confirmatory.get("validation_gates"),
        "confirmatory.validation_gates",
    )
    _expect(
        validation_gates.get("suppression_margin_drop_ci_low_above_zero"),
        True,
        "confirmatory.validation_gates.suppression",
    )
    _expect(
        validation_gates.get("discovered_drop_exceeds_each_control"),
        True,
        "confirmatory.validation_gates.controls",
    )
    _expect(
        validation_gates.get("restoration_recovery_ci_low_minimum"),
        0.80,
        "confirmatory.validation_gates.restoration",
    )
    held_out = _mapping(
        confirmatory.get("held_out_test"), "confirmatory.held_out_test")
    _expect(
        held_out.get("opened_only_after_validation_record_is_final"),
        True,
        "confirmatory.held_out_test.opened_only_after_validation_record_is_final",
    )
    _expect(
        held_out.get("validation_may_change_specification"),
        False,
        "confirmatory.held_out_test.validation_may_change_specification",
    )

    storage = _mapping(payload.get("storage"), "storage")
    for name in (
            "preserve_raw_per_example_behavior_vectors",
            "preserve_raw_per_example_operator_vectors",
            "preserve_raw_capture_rows"):
        _expect(storage.get(name), False, f"storage.{name}")


@dataclass(frozen=True)
class IOIScaleDiscoverySpec:
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

    @property
    def minimum_paired_correct(self) -> int:
        return int(
            self.payload["discovery"][
                "minimum_paired_correct_independent_units"])

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


def load_ioi_scale_discovery_spec(
        path: str | Path | None = None) -> IOIScaleDiscoverySpec:
    resolved = _resolve_spec_path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("IOI scale-discovery specification must be a mapping")
    normalized = dict(payload)
    content_hash = canonical_hash(normalized)
    _expect(
        content_hash,
        IOI_SCALE_DISCOVERY_SPEC_CONTENT_HASH,
        "content_hash",
    )
    _validate_semantics(normalized)
    return IOIScaleDiscoverySpec(
        path=str(resolved),
        content_hash=content_hash,
        payload=normalized,
    )


def build_ioi_scale_localization(
        ranked_sites: list[Mapping[str, Any]], *,
        capture: Mapping[str, Any],
        spec: IOIScaleDiscoverySpec,
) -> dict[str, Any]:
    return build_arc_localization(
        ranked_sites,
        capture=capture,
        spec=spec,
        benchmark_id="mib_ioi",
    )
