"""Frozen ARC circuit loading and exact layer-route matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from analysis.dawn_analysis_storage import join_path, read_json
from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.operator_interpretability.units import (
    ROUTES,
    OperatorCircuit,
    OperatorSite,
    OperatorSpaceShape,
)


FROZEN_ARC_SPEC_RELATIVE_PATH = (
    "configs/paper_arc_frozen_circuit_v4172_400m.yaml"
)
FROZEN_ARC_SPEC_CONTENT_HASH = (
    "49f3a7dc83d3212f786666d74f77d62c7ccb749fb3df6388ac524c9209728600"
)
FROZEN_ARC_VALIDATION_ROW_COUNT = 128
FROZEN_ARC_VALIDATION_PAIRED_CORRECT_COUNT = 51
FROZEN_ARC_CONTROL_NAME = "layer_route_matched_random"
FROZEN_ARC_CONTROL_ALGORITHM_VERSION = (
    "frozen_arc_layer_route_cell_uniform_complement_v1"
)

_REPORT_FIELDS = (
    "intact_mean_margin",
    "intact_accuracy",
    "suppressed_mean_margin",
    "suppressed_accuracy",
    "mean_margin_drop",
    "margin_drop_bootstrap_ci",
    "prediction_flip_fraction",
    "source_direction_change",
    "matched_random_margin_drop_distribution",
    "frozen_minus_matched_random_paired_effect",
    "unrelated_output_damage",
    "restored_mean_margin",
    "restoration_recovery_fraction",
    "restoration_recovery_bootstrap_ci",
)


def _expect(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(
            f"frozen ARC specification drift at {name}: "
            f"expected={expected!r} actual={value!r}")


def _expect_close(
        value: Any, expected: Any, name: str, *,
        tolerance: float = 1.0e-15) -> None:
    if not np.isclose(
            float(value), float(expected), rtol=0.0, atol=tolerance):
        raise ValueError(
            f"frozen ARC specification drift at {name}: "
            f"expected={expected!r} actual={value!r}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen ARC {name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"frozen ARC {name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"frozen ARC {name} must be positive")
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_spec_path(path: str | Path | None) -> Path:
    value = Path(path or FROZEN_ARC_SPEC_RELATIVE_PATH)
    return value if value.is_absolute() else _repo_root() / value


def _normalized_layer_route_counts(
        value: Any, *, name: str) -> dict[int, dict[str, int]]:
    rows = _mapping(value, name)
    normalized: dict[int, dict[str, int]] = {}
    for layer_value, route_value in rows.items():
        layer = int(layer_value)
        route_counts = _mapping(
            route_value, f"{name}.{layer}")
        normalized[layer] = {
            route: int(route_counts.get(route, 0))
            for route in ROUTES
        }
    return normalized


def _validate_frozen_semantics(spec: Mapping[str, Any]) -> None:
    _expect(spec.get("schema_version"), 1, "schema_version")
    _expect(
        spec.get("status"),
        "frozen_after_discovery_before_validation",
        "status",
    )
    target = _mapping(spec.get("target"), "target")
    _expect(target.get("target_id"), "v4172_400M", "target.target_id")
    _expect(
        target.get("model_version"),
        "spatial-r1-v4.1.7.2",
        "target.model_version",
    )
    _expect(target.get("checkpoint_step"), 76293, "target.checkpoint_step")

    discovery = _mapping(spec.get("discovery"), "discovery")
    _expect(discovery.get("benchmark_id"), "mib_arc", "discovery.benchmark_id")
    _expect(discovery.get("phase"), "discovery", "discovery.phase")
    _expect(
        discovery.get("selection_uses_validation"),
        False,
        "discovery.selection_uses_validation",
    )
    _expect(
        discovery.get("selection_uses_test"),
        False,
        "discovery.selection_uses_test",
    )
    _expect(
        discovery.get("runtime_selected_row_count"),
        128,
        "discovery.runtime_selected_row_count",
    )
    _expect(
        discovery.get("independent_example_count"),
        50,
        "discovery.independent_example_count",
    )
    _expect(
        discovery.get("raw_per_example_vectors_persisted"),
        False,
        "discovery.raw_per_example_vectors_persisted",
    )
    _positive_int(
        discovery.get("ranked_site_count"),
        "discovery.ranked_site_count",
    )

    selection = _mapping(spec.get("selection"), "selection")
    _expect(
        selection.get("ranking_score"),
        "absolute_discovery_mean_contribution_importance",
        "selection.ranking_score",
    )
    _expect(
        selection.get("rule"),
        "smallest_audited_prefix_passing_both_discovery_gates",
        "selection.rule",
    )
    _expect(
        selection.get("site_resolution"),
        "first_selected_k_rows_of_exact_ranked_sites_content_hash",
        "selection.site_resolution",
    )
    _expect(
        selection.get("zero_selected_v_sites_is_a_discovery_result"),
        True,
        "selection.zero_selected_v_sites_is_a_discovery_result",
    )
    _expect(
        selection.get("zero_selected_rst_sites_is_a_discovery_result"),
        True,
        "selection.zero_selected_rst_sites_is_a_discovery_result",
    )
    selected_k = _positive_int(
        selection.get("selected_k"), "selection.selected_k")
    _expect(selected_k, 4096, "selection.selected_k")
    route_counts = _mapping(
        selection.get("selected_route_counts"),
        "selection.selected_route_counts",
    )
    _expect(
        {route: int(route_counts.get(route, 0)) for route in ROUTES},
        {"q": 1770, "k": 2326, "v": 0, "rst": 0},
        "selection.selected_route_counts",
    )
    layer_route_counts = _normalized_layer_route_counts(
        selection.get("selected_layer_route_counts"),
        name="selection.selected_layer_route_counts",
    )
    if sum(
            count
            for counts in layer_route_counts.values()
            for count in counts.values()) != selected_k:
        raise ValueError(
            "frozen ARC selected layer-route counts do not sum to selected_k")

    intervention = _mapping(spec.get("intervention"), "intervention")
    _expect(
        intervention.get("selection_must_not_be_recomputed"),
        True,
        "intervention.selection_must_not_be_recomputed",
    )
    _expect(
        intervention.get("validation_may_change_specification"),
        False,
        "intervention.validation_may_change_specification",
    )
    _expect(
        intervention.get("test_may_change_specification"),
        False,
        "intervention.test_may_change_specification",
    )
    _expect(
        intervention.get("candidate_score"),
        "correct_minus_source_sum_log_probability",
        "intervention.candidate_score",
    )
    _expect(
        intervention.get("primary_effect"),
        "intact_margin_minus_suppressed_margin",
        "intervention.primary_effect",
    )
    suppression = _mapping(
        intervention.get("suppression"), "intervention.suppression")
    _expect(
        suppression.get("mode"),
        "circuit_wide_execution_numerator_suppression",
        "intervention.suppression.mode",
    )
    _expect(
        suppression.get("admission_denominator"),
        "full_production_denominator",
        "intervention.suppression.admission_denominator",
    )
    restoration = _mapping(
        intervention.get("restoration"), "intervention.restoration")
    _expect(
        restoration.get("mode"),
        "exact_selected_numerator_restore_after_suppression",
        "intervention.restoration.mode",
    )
    _expect(
        restoration.get("admission_denominator"),
        "full_production_denominator",
        "intervention.restoration.admission_denominator",
    )
    _expect(
        restoration.get("restored_values_source"),
        "same_example_intact_execution",
        "intervention.restoration.restored_values_source",
    )
    controls = _mapping(
        intervention.get("controls"), "intervention.controls")
    _expect(controls.get("replicate_count"), 100, "controls.replicate_count")
    _expect(controls.get("seed"), 24172, "controls.seed")
    _expect(
        tuple(controls.get("match_fields") or ()),
        ("layer", "route"),
        "controls.match_fields",
    )
    _expect(
        controls.get("sampling"),
        "uniform_without_replacement_within_layer_route_cell",
        "controls.sampling",
    )
    _expect(
        controls.get("sampling_population"),
        "frozen_site_complement",
        "controls.sampling_population",
    )
    for field in (
            "match_exact_frozen_count_per_cell",
            "frozen_sites_excluded",
            "duplicate_site_within_replicate_forbidden",
            "control_site_set_hashes_must_be_unique"):
        _expect(controls.get(field), True, f"controls.{field}")

    evaluation = _mapping(spec.get("evaluation"), "evaluation")
    _expect(evaluation.get("validation_first"), True, "evaluation.validation_first")
    _expect(
        evaluation.get("validation_runtime_selected_row_count"),
        FROZEN_ARC_VALIDATION_ROW_COUNT,
        "evaluation.validation_runtime_selected_row_count",
    )
    _expect(
        evaluation.get("validation_independent_example_count"),
        FROZEN_ARC_VALIDATION_PAIRED_CORRECT_COUNT,
        "evaluation.validation_independent_example_count",
    )
    _expect(
        evaluation.get("held_out_test_opened_only_after_validation_record_is_final"),
        True,
        "evaluation.held_out_test_opened_only_after_validation_record_is_final",
    )
    _expect(evaluation.get("bootstrap_samples"), 2000, "evaluation.bootstrap_samples")
    _expect(
        evaluation.get("permutation_samples"),
        2000,
        "evaluation.permutation_samples",
    )
    _expect_close(evaluation.get("alpha"), 0.05, "evaluation.alpha", tolerance=0.0)
    _expect(
        evaluation.get("report_even_when_gate_fails"),
        True,
        "evaluation.report_even_when_gate_fails",
    )
    _expect(
        tuple(evaluation.get("report") or ()),
        _REPORT_FIELDS,
        "evaluation.report",
    )
    gates = _mapping(
        evaluation.get("validation_gates"),
        "evaluation.validation_gates",
    )
    _expect(
        gates.get("suppression_margin_drop_ci_low_above_zero"),
        True,
        "evaluation.validation_gates.suppression",
    )
    _expect(
        gates.get(
            "frozen_minus_matched_random_margin_drop_ci_low_above_zero"),
        True,
        "evaluation.validation_gates.control",
    )
    _expect_close(
        gates.get("restoration_recovery_ci_low_minimum"),
        0.80,
        "evaluation.validation_gates.restoration",
        tolerance=0.0,
    )
    unrelated = _mapping(
        evaluation.get("unrelated_output_damage"),
        "evaluation.unrelated_output_damage",
    )
    _expect(
        unrelated.get("role"),
        "selectivity_audit_only",
        "evaluation.unrelated_output_damage.role",
    )
    _expect(
        unrelated.get("threshold"),
        None,
        "evaluation.unrelated_output_damage.threshold",
    )

    storage = _mapping(spec.get("storage"), "storage")
    for field, expected in (
            ("preserve_aggregate_statistics_and_hashes", True),
            ("preserve_frozen_site_identity_hash", True),
            ("preserve_raw_per_example_logits", False),
            ("preserve_raw_per_example_margins", False),
            ("preserve_raw_per_example_operator_vectors", False),
            ("semantic_audit_after_run", True)):
        _expect(storage.get(field), expected, f"storage.{field}")


@dataclass(frozen=True)
class FrozenARCCircuit:
    spec_path: str
    spec: Mapping[str, Any]
    spec_content_hash: str
    localization_path: str
    localization_protocol_hash: str
    selected_rows: tuple[Mapping[str, Any], ...]
    circuit: OperatorCircuit

    @property
    def selected_k(self) -> int:
        return int(self.spec["selection"]["selected_k"])

    @property
    def selected_site_identity_hash(self) -> str:
        return str(self.spec["selection"]["selected_site_identity_hash"])

    @property
    def controls(self) -> Mapping[str, Any]:
        return self.spec["intervention"]["controls"]

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.spec["evaluation"]

    @property
    def selected_layer_route_counts(self) -> dict[int, dict[str, int]]:
        return _normalized_layer_route_counts(
            self.spec["selection"]["selected_layer_route_counts"],
            name="selection.selected_layer_route_counts",
        )

    def validate_runtime(
            self, *, shape: OperatorSpaceShape, target_id: str,
            model_version: str, checkpoint_step: int,
            checkpoint_identity: str, checkpoint_config_hash: str,
            model_config_hash: str,
            benchmark_build_id: str,
            benchmark_manifest_hash: str) -> None:
        target = self.spec["target"]
        discovery = self.spec["discovery"]
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
        self.circuit.validate(shape)


def load_frozen_arc_circuit(
        shape: OperatorSpaceShape, *,
        spec_path: str | Path | None = None,
        localization_record: Mapping[str, Any] | None = None,
        localization_path: str | None = None) -> FrozenARCCircuit:
    """Load the hash-bound ARC discovery prefix without re-localization."""
    resolved_spec_path = _resolve_spec_path(spec_path)
    value = yaml.safe_load(resolved_spec_path.read_text(encoding="utf-8"))
    spec = _mapping(value, "root")
    spec_hash = canonical_hash(spec)
    _expect(
        spec_hash,
        FROZEN_ARC_SPEC_CONTENT_HASH,
        "frozen_spec_content_hash",
    )
    _validate_frozen_semantics(spec)

    discovery = spec["discovery"]
    artifact_path = str(localization_path or join_path(
        discovery["result_root"],
        *str(discovery["localization_item"]).replace("\\", "/").split("/"),
    ))
    record = (
        dict(localization_record)
        if localization_record is not None
        else read_json(artifact_path, None)
    )
    if not isinstance(record, Mapping):
        raise FileNotFoundError(
            f"frozen ARC localization artifact is unavailable: {artifact_path}")
    _expect(
        record.get("protocol_hash"),
        discovery["protocol_hash"],
        "localization.protocol_hash",
    )
    protocol = _mapping(record.get("protocol"), "localization.protocol")
    for field, expected in (
            ("checkpoint_identity", spec["target"]["checkpoint_identity"]),
            ("model_config_hash", spec["target"]["model_config_hash"]),
            ("benchmark_manifest_hash", discovery["benchmark_manifest_hash"]),
            ("model_version", spec["target"]["model_version"])):
        _expect(
            protocol.get(field),
            expected,
            f"localization.protocol.{field}",
        )
    analysis_code = _mapping(
        protocol.get("analysis_code"), "localization.protocol.analysis_code")
    _expect(
        analysis_code.get("git_commit"),
        discovery["analysis_commit"],
        "localization.protocol.analysis_code.git_commit",
    )
    _expect(
        analysis_code.get("git_dirty"),
        discovery["analysis_worktree_dirty"],
        "localization.protocol.analysis_code.git_dirty",
    )

    payload = _mapping(record.get("payload"), "localization.payload")
    _expect(
        payload.get("item_id"),
        "mib_arc.discovery_operator_localization",
        "localization.payload.item_id",
    )
    _expect(payload.get("status"), "ready", "localization.payload.status")
    result = _mapping(payload.get("result"), "localization.result")
    _expect(result.get("status"), "ready", "localization.result.status")
    _expect(result.get("phase"), "discovery", "localization.result.phase")
    _expect(result.get("benchmark"), "mib_arc", "localization.result.benchmark")
    _expect(
        result.get("confirmatory_eligible"),
        True,
        "localization.result.confirmatory_eligible",
    )
    localization = _mapping(
        result.get("localization"), "localization.result.localization")
    _expect(localization.get("status"), "ready", "localization.status")
    for field in ("qualified_row_count", "total_row_count"):
        _expect(
            localization.get(field),
            discovery[field],
            f"localization.{field}",
        )
    rank_stability = _mapping(
        localization.get("rank_stability"),
        "localization.rank_stability",
    )
    _expect_close(
        rank_stability.get("rank_stability"),
        discovery["rank_stability"],
        "localization.rank_stability",
    )
    capture = _mapping(result.get("capture"), "localization.result.capture")
    _expect(
        capture.get("raw_capture_digest"),
        discovery["raw_capture_digest"],
        "localization.capture.raw_capture_digest",
    )
    _expect_close(
        capture.get("capture_threshold"),
        discovery["capture_threshold"],
        "localization.capture.capture_threshold",
        tolerance=0.0,
    )
    _expect(
        tuple(int(value) for value in capture.get("final_widths") or ()),
        tuple(
            int(discovery["capture_final_widths"][route])
            for route in ("qk", "v", "rst")),
        "localization.capture.final_widths",
    )

    rows_value = localization.get("ranked_sites")
    if not isinstance(rows_value, Sequence) or isinstance(
            rows_value, (str, bytes)):
        raise ValueError("frozen ARC localization lacks aggregate ranked sites")
    _expect(
        len(rows_value),
        int(discovery["ranked_site_count"]),
        "localization.ranked_site_count",
    )
    ranked_hash = canonical_hash(rows_value)
    _expect(
        ranked_hash,
        discovery["ranked_sites_content_hash"],
        "localization.ranked_sites_content_hash",
    )
    _expect(
        localization.get("ranked_sites_content_hash"),
        ranked_hash,
        "localization.result.ranked_sites_content_hash",
    )

    selection = spec["selection"]
    selected_k = int(selection["selected_k"])
    selected_rows = tuple(
        _mapping(row, "localization ranked-site row")
        for row in rows_value[:selected_k]
    )
    _expect(
        canonical_hash(selected_rows),
        selection["selected_ranked_rows_hash"],
        "selection.selected_ranked_rows_hash",
    )
    identities = [
        {
            "layer": int(row["layer"]),
            "route": str(row["route"]),
            "operator_id": int(row["operator_id"]),
        }
        for row in selected_rows
    ]
    _expect(
        canonical_hash(identities),
        selection["selected_site_identity_hash"],
        "selection.selected_site_identity_hash",
    )
    _expect(
        canonical_hash({"benchmark_id": "mib_arc", "sites": identities}),
        selection["circuit_hash"],
        "selection.circuit_hash",
    )
    sites = tuple(
        OperatorSite(
            layer=identity["layer"],
            route=identity["route"],
            operator_id=identity["operator_id"],
        ).validate(shape)
        for identity in identities
    )
    if len(set(sites)) != selected_k:
        raise ValueError("frozen ARC selected prefix contains duplicate sites")
    route_counts = {
        route: sum(site.route == route for site in sites)
        for route in ROUTES
    }
    _expect(
        route_counts,
        {
            route: int(selection["selected_route_counts"][route])
            for route in ROUTES
        },
        "selection.selected_route_counts",
    )
    layer_route_counts = {
        layer: {
            route: sum(
                site.layer == layer and site.route == route
                for site in sites)
            for route in ROUTES
        }
        for layer in range(shape.n_layers)
    }
    _expect(
        layer_route_counts,
        _normalized_layer_route_counts(
            selection["selected_layer_route_counts"],
            name="selection.selected_layer_route_counts",
        ),
        "selection.selected_layer_route_counts",
    )
    _expect(
        sorted({site.layer for site in sites}),
        [int(layer) for layer in selection["selected_layers"]],
        "selection.selected_layers",
    )
    _expect_close(
        selected_rows[0]["importance"],
        selection["top_importance"],
        "selection.top_importance",
        tolerance=1.0e-12,
    )
    _expect_close(
        selected_rows[-1]["importance"],
        selection["boundary_importance"],
        "selection.boundary_importance",
        tolerance=1.0e-12,
    )
    artifact_circuit = _mapping(
        localization.get("circuit"), "localization.circuit")
    for field in (
            "status", "selected_k", "selected_layers",
            "selected_route_counts", "selected_ranked_rows_hash",
            "selected_site_identity_hash", "circuit_hash"):
        expected = (
            "frozen_from_discovery"
            if field == "status"
            else selection[field]
        )
        _expect(
            artifact_circuit.get(field),
            expected,
            f"localization.circuit.{field}",
        )
    _expect(
        _normalized_layer_route_counts(
            artifact_circuit.get("selected_layer_route_counts"),
            name="localization.circuit.selected_layer_route_counts",
        ),
        layer_route_counts,
        "localization.circuit.selected_layer_route_counts",
    )
    artifact_sites = artifact_circuit.get("sites")
    if not isinstance(artifact_sites, Sequence) or isinstance(
            artifact_sites, (str, bytes)):
        raise ValueError("frozen ARC artifact circuit sites are unavailable")
    _expect(
        canonical_hash(artifact_sites),
        selection["selected_site_identity_hash"],
        "localization.circuit.sites",
    )

    circuit = OperatorCircuit(
        sites=sites,
        discovery_benchmark="mib_arc",
        discovery_phase="discovery",
        metadata={
            "selection": selection["site_resolution"],
            "selected_ranked_rows_hash": selection[
                "selected_ranked_rows_hash"],
            "selected_site_identity_hash": selection[
                "selected_site_identity_hash"],
            "frozen_spec_content_hash": spec_hash,
        },
    ).validate(shape)
    return FrozenARCCircuit(
        spec_path=str(resolved_spec_path),
        spec=spec,
        spec_content_hash=spec_hash,
        localization_path=artifact_path,
        localization_protocol_hash=str(record["protocol_hash"]),
        selected_rows=selected_rows,
        circuit=circuit,
    )


class FrozenARCControlSampler:
    """Generate only the preregistered exact layer-route matched control."""

    def __init__(
            self, frozen: FrozenARCCircuit,
            shape: OperatorSpaceShape) -> None:
        self.frozen = frozen
        self.shape = shape
        self.selected = set(frozen.circuit.sites)
        self.expected_counts = frozen.selected_layer_route_counts
        self.pools: dict[tuple[int, str], np.ndarray] = {}
        for layer in range(shape.n_layers):
            for route in ROUTES:
                count = int(self.expected_counts[layer][route])
                if count == 0:
                    continue
                candidates = np.asarray([
                    operator_id
                    for operator_id in range(shape.pool_size(route))
                    if OperatorSite(
                        layer, route, operator_id) not in self.selected
                ], dtype=np.int32)
                if count > int(candidates.size):
                    raise ValueError(
                        "frozen ARC control cell is smaller than its "
                        f"required match: layer={layer} route={route}")
                self.pools[(layer, route)] = candidates

    @staticmethod
    def _rng(seed: int, replicate_index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([
            int(seed), int(replicate_index),
        ]))

    def generate(
            self, replicate_index: int
    ) -> tuple[OperatorCircuit, dict[str, Any]]:
        controls = self.frozen.controls
        replicate_count = int(controls["replicate_count"])
        if not 0 <= int(replicate_index) < replicate_count:
            raise ValueError("frozen ARC control replicate index is out of range")
        seed = int(controls["seed"])
        rng = self._rng(seed, int(replicate_index))
        selected: list[OperatorSite] = []
        for layer in range(self.shape.n_layers):
            for route in ROUTES:
                pool = self.pools.get((layer, route))
                if pool is None:
                    continue
                count = int(self.expected_counts[layer][route])
                operator_ids = rng.choice(pool, size=count, replace=False)
                selected.extend(
                    OperatorSite(layer, route, int(operator_id))
                    for operator_id in operator_ids
                )
        route_order = {route: index for index, route in enumerate(ROUTES)}
        sites = tuple(sorted(
            selected,
            key=lambda site: (
                site.layer, route_order[site.route], site.operator_id),
        ))
        if len(sites) != self.frozen.selected_k or len(set(sites)) != len(sites):
            raise RuntimeError("frozen ARC control size or uniqueness drift")
        if set(sites) & self.selected:
            raise RuntimeError("frozen ARC control sampled a frozen circuit site")
        layer_route_counts = {
            layer: {
                route: sum(
                    site.layer == layer and site.route == route
                    for site in sites)
                for route in ROUTES
            }
            for layer in range(self.shape.n_layers)
        }
        if layer_route_counts != self.expected_counts:
            raise RuntimeError("frozen ARC control layer-route match drift")
        identities = [
            {
                "layer": site.layer,
                "route": site.route,
                "operator_id": site.operator_id,
            }
            for site in sites
        ]
        identity_hash = canonical_hash(identities)
        circuit = OperatorCircuit(
            sites=sites,
            discovery_benchmark="mib_arc",
            discovery_phase="discovery",
            metadata={
                "control_name": FROZEN_ARC_CONTROL_NAME,
                "replicate_index": int(replicate_index),
                "base_seed": seed,
                "rng_derivation": (
                    "numpy_seed_sequence_base_seed_and_replicate"),
                "algorithm_version": FROZEN_ARC_CONTROL_ALGORITHM_VERSION,
                "site_identity_hash": identity_hash,
            },
        ).validate(self.shape)
        return circuit, {
            "replicate_index": int(replicate_index),
            "site_count": len(sites),
            "site_identity_hash": identity_hash,
            "frozen_circuit_overlap_count": 0,
            "layer_route_counts_match": True,
            "route_counts": {
                route: sum(site.route == route for site in sites)
                for route in ROUTES
            },
            "layer_counts": {
                str(layer): sum(site.layer == layer for site in sites)
                for layer in range(self.shape.n_layers)
            },
        }
