"""Frozen IOI circuit loading, matched controls, and confirmatory statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from analysis.dawn_analysis_storage import join_path, read_json
from analysis.operator_interpretability.benchmark_schema import canonical_hash
from analysis.operator_interpretability.statistics import (
    bootstrap_mean_ci,
    paired_permutation_test,
)
from analysis.operator_interpretability.units import (
    ROUTES,
    OperatorCircuit,
    OperatorSite,
    OperatorSpaceShape,
)


FROZEN_IOI_SPEC_RELATIVE_PATH = (
    "configs/paper_ioi_frozen_circuit_v4172_400m.yaml"
)
FROZEN_IOI_SPEC_CONTENT_HASH = (
    "6ec2661823bb52aa3bc336fb69fed1eef317f9cced02f836eaaade8c116a418c"
)
FROZEN_IOI_VALIDATION_ROW_COUNT = 128
FROZEN_IOI_VALIDATION_PAIRED_CORRECT_COUNT = 123
FROZEN_IOI_CONTROL_ORDER = (
    "equal_count_random",
    "same_layer_random",
    "activation_matched",
    "route_frequency_matched",
)
FROZEN_IOI_CONTROL_ALGORITHM_VERSION = (
    "frozen_ioi_controls_v1_frozen_count_quantile_bins_without_replacement"
)

_REPORT_FIELDS = (
    "intact_mean_margin",
    "intact_accuracy",
    "suppressed_mean_margin",
    "suppressed_accuracy",
    "mean_margin_drop",
    "margin_drop_bootstrap_ci",
    "prediction_flip_fraction",
    "control_margin_drop_distributions",
    "discovered_minus_control_paired_effects",
    "restored_mean_margin",
    "restoration_recovery_fraction",
    "restoration_recovery_bootstrap_ci",
)


def _expect(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(
            f"frozen IOI specification drift at {name}: "
            f"expected={expected!r} actual={value!r}")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen IOI {name} must be a mapping")
    return dict(value)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"frozen IOI {name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"frozen IOI {name} must be positive")
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_local_spec_path(path: str | Path | None) -> Path:
    value = Path(path or FROZEN_IOI_SPEC_RELATIVE_PATH)
    return value if value.is_absolute() else _repo_root() / value


def _validate_frozen_semantics(spec: Mapping[str, Any]) -> None:
    _expect(spec.get("schema_version"), 1, "schema_version")
    _expect(
        spec.get("status"),
        "frozen_after_discovery_before_validation",
        "status",
    )
    target = _require_mapping(spec.get("target"), "target")
    _expect(target.get("target_id"), "v4172_400M", "target.target_id")
    _expect(
        target.get("model_version"),
        "spatial-r1-v4.1.7.2",
        "target.model_version",
    )
    _require_positive_int(target.get("checkpoint_step"), "target.checkpoint_step")

    discovery = _require_mapping(spec.get("discovery"), "discovery")
    _expect(discovery.get("benchmark_id"), "mib_ioi", "discovery.benchmark_id")
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
        discovery.get("raw_per_example_vectors_persisted"),
        False,
        "discovery.raw_per_example_vectors_persisted",
    )
    _require_positive_int(
        discovery.get("ranked_site_count"), "discovery.ranked_site_count")

    selection = _require_mapping(spec.get("selection"), "selection")
    _expect(
        selection.get("ranking_score"),
        "absolute_discovery_mean_contribution_importance",
        "selection.ranking_score",
    )
    _expect(
        selection.get("site_resolution"),
        "first_selected_k_rows_of_exact_ranked_sites_content_hash",
        "selection.site_resolution",
    )
    _expect(
        selection.get("zero_selected_rst_sites_is_a_discovery_result"),
        True,
        "selection.zero_selected_rst_sites_is_a_discovery_result",
    )
    _require_positive_int(selection.get("selected_k"), "selection.selected_k")

    intervention = _require_mapping(spec.get("intervention"), "intervention")
    for field in (
            "selection_must_not_be_recomputed",
            "validation_may_change_specification",
            "test_may_change_specification"):
        expected = field == "selection_must_not_be_recomputed"
        _expect(intervention.get(field), expected, f"intervention.{field}")
    _expect(
        intervention.get("candidate_score"),
        "positive_minus_negative_sum_log_probability",
        "intervention.candidate_score",
    )
    _expect(
        intervention.get("primary_effect"),
        "intact_margin_minus_suppressed_margin",
        "intervention.primary_effect",
    )
    suppression = _require_mapping(
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
    restoration = _require_mapping(
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

    controls = _require_mapping(
        intervention.get("controls"), "intervention.controls")
    _expect(
        controls.get("circuit_sites_excluded_from_sampling"),
        True,
        "intervention.controls.circuit_sites_excluded_from_sampling",
    )
    _require_positive_int(
        controls.get("replicate_count_per_control"),
        "intervention.controls.replicate_count_per_control",
    )
    _expect(
        controls.get("equal_count_random", {}).get("sampling"),
        "uniform_without_replacement",
        "intervention.controls.equal_count_random.sampling",
    )
    _expect(
        controls.get("same_layer_random", {}).get("sampling"),
        "uniform_without_replacement_within_layer",
        "intervention.controls.same_layer_random.sampling",
    )
    _expect(
        controls.get("same_layer_random", {}).get(
            "match_selected_count_per_layer"),
        True,
        "intervention.controls.same_layer_random.match_selected_count_per_layer",
    )
    activation = _require_mapping(
        controls.get("activation_matched"),
        "intervention.controls.activation_matched",
    )
    _expect(
        activation.get("match_fields"),
        [
            "layer",
            "route",
            "discovery_mean_pre_scale_operator_output_norm_quantile",
        ],
        "intervention.controls.activation_matched.match_fields",
    )
    _expect(
        activation.get("sampling"),
        "nearest_bin_without_replacement",
        "intervention.controls.activation_matched.sampling",
    )
    route = _require_mapping(
        controls.get("route_frequency_matched"),
        "intervention.controls.route_frequency_matched",
    )
    _expect(
        route.get("match_selected_count_per_route"),
        True,
        "intervention.controls.route_frequency_matched."
        "match_selected_count_per_route",
    )
    _expect(
        route.get("sampling"),
        "uniform_without_replacement_within_route",
        "intervention.controls.route_frequency_matched.sampling",
    )

    evaluation = _require_mapping(spec.get("evaluation"), "evaluation")
    _expect(evaluation.get("validation_first"), True, "evaluation.validation_first")
    _expect(
        evaluation.get("held_out_test_opened_only_after_validation_record_is_final"),
        True,
        "evaluation.held_out_test_opened_only_after_validation_record_is_final",
    )
    _expect(
        evaluation.get("report_even_when_gate_fails"),
        True,
        "evaluation.report_even_when_gate_fails",
    )
    _expect(tuple(evaluation.get("report") or ()), _REPORT_FIELDS, "evaluation.report")
    _require_positive_int(
        evaluation.get("bootstrap_samples"), "evaluation.bootstrap_samples")
    _require_positive_int(
        evaluation.get("permutation_samples"), "evaluation.permutation_samples")
    alpha = float(evaluation.get("alpha"))
    if not 0.0 < alpha < 0.5:
        raise ValueError("frozen IOI evaluation.alpha must be in (0, 0.5)")
    gates = _require_mapping(
        evaluation.get("validation_gates"), "evaluation.validation_gates")
    _expect(
        gates.get("suppression_margin_drop_ci_low_above_zero"),
        True,
        "evaluation.validation_gates."
        "suppression_margin_drop_ci_low_above_zero",
    )
    _expect(
        gates.get("discovered_drop_exceeds_each_control"),
        True,
        "evaluation.validation_gates.discovered_drop_exceeds_each_control",
    )
    recovery_minimum = float(
        gates.get("restoration_recovery_ci_low_minimum"))
    if not 0.0 <= recovery_minimum <= 1.0:
        raise ValueError(
            "frozen IOI restoration recovery minimum must be in [0, 1]")

    storage = _require_mapping(spec.get("storage"), "storage")
    for field, expected in (
            ("preserve_aggregate_statistics_and_hashes", True),
            ("preserve_frozen_site_identity_hash", True),
            ("preserve_raw_per_example_logits", False),
            ("preserve_raw_per_example_margins", False),
            ("preserve_raw_per_example_operator_vectors", False),
            ("semantic_audit_after_run", True)):
        _expect(storage.get(field), expected, f"storage.{field}")


@dataclass(frozen=True)
class FrozenIOICircuit:
    spec_path: str
    spec: Mapping[str, Any]
    spec_content_hash: str
    localization_path: str
    localization_protocol_hash: str
    ranked_sites: tuple[Mapping[str, Any], ...]
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


def load_frozen_ioi_circuit(
        shape: OperatorSpaceShape, *,
        spec_path: str | Path | None = None,
        localization_record: Mapping[str, Any] | None = None,
        localization_path: str | None = None) -> FrozenIOICircuit:
    """Load the frozen YAML and resolve exactly its hash-bound discovery prefix."""
    resolved_spec_path = _resolve_local_spec_path(spec_path)
    with resolved_spec_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    spec = _require_mapping(value, "root")
    _expect(
        canonical_hash(spec),
        FROZEN_IOI_SPEC_CONTENT_HASH,
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
            f"frozen IOI localization artifact is unavailable: {artifact_path}")
    _expect(
        record.get("protocol_hash"),
        discovery["protocol_hash"],
        "localization.protocol_hash",
    )
    protocol = _require_mapping(record.get("protocol"), "localization.protocol")
    for field in (
            "checkpoint_identity", "model_config_hash",
            "benchmark_manifest_hash", "model_version"):
        expected = (
            spec["target"].get(field)
            if field in spec["target"]
            else discovery.get(field)
        )
        _expect(protocol.get(field), expected, f"localization.protocol.{field}")
    analysis_code = _require_mapping(
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

    payload = _require_mapping(record.get("payload"), "localization.payload")
    _expect(
        payload.get("item_id"),
        "mib_ioi.operator_localization",
        "localization.payload.item_id",
    )
    _expect(payload.get("status"), "ready", "localization.payload.status")
    result = _require_mapping(payload.get("result"), "localization.result")
    _expect(result.get("status"), "ready", "localization.result.status")
    _expect(result.get("phase"), "discovery", "localization.result.phase")
    for field in (
            "qualified_row_count", "total_row_count", "raw_capture_digest"):
        _expect(
            result.get(field),
            discovery[field],
            f"localization.result.{field}",
        )
    if not np.isclose(
            float(result.get("capture_threshold")),
            float(discovery["capture_threshold"]),
            rtol=0.0, atol=0.0):
        raise ValueError("frozen IOI localization capture threshold drift")
    if not np.isclose(
            float(result.get("rank_stability")),
            float(discovery["rank_stability"]),
            rtol=0.0, atol=1.0e-15):
        raise ValueError("frozen IOI localization rank stability drift")

    rows_value = result.get("ranked_sites")
    if not isinstance(rows_value, Sequence) or isinstance(
            rows_value, (str, bytes)):
        raise ValueError("frozen IOI localization lacks aggregate ranked sites")
    ranked_sites = tuple(
        _require_mapping(row, "localization ranked-site row")
        for row in rows_value
    )
    _expect(
        len(ranked_sites),
        int(discovery["ranked_site_count"]),
        "localization.ranked_site_count",
    )
    ranked_hash = canonical_hash(ranked_sites)
    _expect(
        ranked_hash,
        discovery["ranked_sites_content_hash"],
        "localization.ranked_sites_content_hash",
    )
    _expect(
        result.get("ranked_sites_content_hash"),
        ranked_hash,
        "localization.result.ranked_sites_content_hash",
    )

    selection = spec["selection"]
    selected_k = int(selection["selected_k"])
    selected_rows = ranked_sites[:selected_k]
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
    sites = tuple(
        OperatorSite(
            layer=int(row["layer"]),
            route=str(row["route"]),
            operator_id=int(row["operator_id"]),
        ).validate(shape)
        for row in selected_rows
    )
    if len(set(sites)) != selected_k:
        raise ValueError("frozen IOI selected prefix contains duplicate sites")
    route_counts = {
        route: sum(site.route == route for site in sites)
        for route in ROUTES
    }
    _expect(
        route_counts,
        {route: int(selection["selected_route_counts"][route])
         for route in ROUTES},
        "selection.selected_route_counts",
    )
    _expect(
        sorted({site.layer for site in sites}),
        [int(layer) for layer in selection["selected_layers"]],
        "selection.selected_layers",
    )
    if not np.isclose(
            float(selected_rows[0]["importance"]),
            float(selection["top_importance"]),
            rtol=0.0, atol=1.0e-12):
        raise ValueError("frozen IOI top importance drift")
    if not np.isclose(
            float(selected_rows[-1]["importance"]),
            float(selection["boundary_importance"]),
            rtol=0.0, atol=1.0e-12):
        raise ValueError("frozen IOI boundary importance drift")

    circuit = OperatorCircuit(
        sites=sites,
        discovery_benchmark="mib_ioi",
        discovery_phase="discovery",
        metadata={
            "selection": selection["site_resolution"],
            "selected_ranked_rows_hash": selection[
                "selected_ranked_rows_hash"],
            "selected_site_identity_hash": selection[
                "selected_site_identity_hash"],
            "frozen_spec_content_hash": canonical_hash(spec),
        },
    ).validate(shape)
    return FrozenIOICircuit(
        spec_path=str(resolved_spec_path),
        spec=spec,
        spec_content_hash=canonical_hash(spec),
        localization_path=artifact_path,
        localization_protocol_hash=str(record["protocol_hash"]),
        ranked_sites=ranked_sites,
        selected_rows=selected_rows,
        circuit=circuit,
    )


def _midrank_quantiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("activation quantiles require a finite vector")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty((values.size,), dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / float(values.size - 1)


class FrozenIOIControlSampler:
    """Generate only the four hash-bound, circuit-disjoint frozen controls."""

    def __init__(
            self, frozen: FrozenIOICircuit,
            shape: OperatorSpaceShape) -> None:
        self.frozen = frozen
        self.shape = shape
        self.selected = set(frozen.circuit.sites)
        self.selected_by_layer = {
            layer: sum(site.layer == layer for site in self.selected)
            for layer in range(shape.n_layers)
        }
        self.selected_by_route = {
            route: sum(site.route == route for site in self.selected)
            for route in ROUTES
        }
        self._stride = 2 * shape.n_qk + shape.n_v + shape.n_rst
        self._route_offsets = {
            "q": 0,
            "k": shape.n_qk,
            "v": 2 * shape.n_qk,
            "rst": 2 * shape.n_qk + shape.n_v,
        }
        frozen_flat = np.asarray(sorted(
            self._encode(site) for site in self.selected), dtype=np.int64)
        eligible = np.ones((shape.total_sites,), dtype=np.bool_)
        eligible[frozen_flat] = False
        self._eligible = eligible
        self._equal_pool = np.flatnonzero(eligible).astype(np.int64)
        self._layer_pools = {
            layer: np.flatnonzero(eligible[
                layer * self._stride:(layer + 1) * self._stride]).astype(
                    np.int64) + layer * self._stride
            for layer in range(shape.n_layers)
        }
        self._route_pools = {
            route: self._route_pool(route)
            for route in ROUTES
        }
        self._activation_groups = self._build_activation_groups()

    def _encode(self, site: OperatorSite) -> int:
        return (
            int(site.layer) * self._stride
            + self._route_offsets[site.route]
            + int(site.operator_id)
        )

    def _decode(self, flat: int) -> OperatorSite:
        value = int(flat)
        layer, local = divmod(value, self._stride)
        if local < self.shape.n_qk:
            route, operator_id = "q", local
        elif local < 2 * self.shape.n_qk:
            route, operator_id = "k", local - self.shape.n_qk
        elif local < 2 * self.shape.n_qk + self.shape.n_v:
            route = "v"
            operator_id = local - 2 * self.shape.n_qk
        else:
            route = "rst"
            operator_id = local - 2 * self.shape.n_qk - self.shape.n_v
        return OperatorSite(layer, route, operator_id).validate(self.shape)

    def _route_pool(self, route: str) -> np.ndarray:
        size = self.shape.pool_size(route)
        values = np.concatenate([
            (
                layer * self._stride
                + self._route_offsets[route]
                + np.arange(size, dtype=np.int64)
            )
            for layer in range(self.shape.n_layers)
        ])
        return values[self._eligible[values]]

    def _build_activation_groups(self) -> dict[
            tuple[int, str], dict[str, np.ndarray]]:
        importance = {
            (layer, route): np.zeros(
                (self.shape.pool_size(route),), dtype=np.float64)
            for layer in range(self.shape.n_layers)
            for route in ROUTES
        }
        observed: set[OperatorSite] = set()
        for row in self.frozen.ranked_sites:
            site = OperatorSite(
                int(row["layer"]), str(row["route"]),
                int(row["operator_id"])).validate(self.shape)
            if site in observed:
                raise ValueError(
                    "frozen localization ranking contains duplicate sites")
            observed.add(site)
            value = float(row["importance"])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "frozen localization importance must be finite and "
                    "nonnegative")
            importance[(site.layer, site.route)][site.operator_id] = value

        groups: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        for key, values in importance.items():
            selected_ids = np.asarray(sorted(
                site.operator_id for site in self.selected
                if (site.layer, site.route) == key), dtype=np.int32)
            if not selected_ids.size:
                continue
            quantiles = _midrank_quantiles(values)
            candidate_ids = np.asarray([
                operator_id for operator_id in range(values.size)
                if OperatorSite(
                    key[0], key[1], operator_id) not in self.selected
            ], dtype=np.int32)
            candidate_order = np.lexsort((
                candidate_ids,
                quantiles[candidate_ids],
            ))
            candidate_ids = candidate_ids[candidate_order]
            groups[key] = {
                "target_ids": selected_ids,
                "target_quantiles": quantiles[selected_ids],
                "candidate_ids": candidate_ids,
                "candidate_quantiles": quantiles[candidate_ids],
            }
        if sum(
                int(group["target_ids"].size)
                for group in groups.values()) != self.frozen.selected_k:
            raise ValueError(
                "activation-matched groups do not cover the frozen circuit")
        return groups

    @staticmethod
    def _rng(seed: int, replicate_index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([
            int(seed), int(replicate_index),
        ]))

    def _random_sites(
            self, pools: Iterable[tuple[np.ndarray, int]],
            rng: np.random.Generator) -> tuple[OperatorSite, ...]:
        selected: list[OperatorSite] = []
        for pool, count in pools:
            count = int(count)
            if count == 0:
                continue
            if count > int(pool.size):
                raise ValueError("frozen control pool is smaller than its match")
            flat = rng.choice(pool, size=count, replace=False)
            selected.extend(self._decode(value) for value in flat)
        return tuple(sorted(selected))

    def _activation_sites(
            self, rng: np.random.Generator, *,
            quantile_bin_count: int) -> tuple[
                tuple[OperatorSite, ...], dict[str, Any]]:
        if quantile_bin_count <= 1:
            raise ValueError(
                "activation matching requires at least two quantile bins")
        selected: list[OperatorSite] = []
        distances: list[float] = []
        bin_distances: list[int] = []
        for (layer, route), group in sorted(self._activation_groups.items()):
            candidate_ids = group["candidate_ids"]
            candidate_quantiles = group["candidate_quantiles"]
            target_quantiles = group["target_quantiles"]
            candidate_bins = np.minimum(
                (candidate_quantiles * quantile_bin_count).astype(np.int32),
                quantile_bin_count - 1,
            )
            target_bins = np.minimum(
                (target_quantiles * quantile_bin_count).astype(np.int32),
                quantile_bin_count - 1,
            )
            active_by_bin = [
                np.flatnonzero(candidate_bins == bin_index).astype(
                    np.int32).tolist()
                for bin_index in range(quantile_bin_count)
            ]
            for target_index in rng.permutation(target_quantiles.size):
                target = float(target_quantiles[int(target_index)])
                target_bin = int(target_bins[int(target_index)])
                nonempty_bins = [
                    bin_index for bin_index, values in enumerate(active_by_bin)
                    if values
                ]
                if not nonempty_bins:
                    raise RuntimeError(
                        "activation-matched candidate active set exhausted")
                minimum_bin_distance = min(
                    abs(bin_index - target_bin)
                    for bin_index in nonempty_bins)
                nearest_bins = [
                    bin_index for bin_index in nonempty_bins
                    if abs(bin_index - target_bin) == minimum_bin_distance
                ]
                chosen_bin = int(nearest_bins[
                    int(rng.integers(0, len(nearest_bins)))])
                bucket = active_by_bin[chosen_bin]
                bucket_index = int(rng.integers(0, len(bucket)))
                chosen = int(bucket[bucket_index])
                bucket[bucket_index] = bucket[-1]
                bucket.pop()
                selected.append(OperatorSite(
                    layer, route, int(candidate_ids[chosen])))
                distances.append(
                    abs(float(candidate_quantiles[chosen]) - target))
                bin_distances.append(minimum_bin_distance)
        return tuple(sorted(selected)), {
            "empirical_quantile_bin_count": int(quantile_bin_count),
            "mean_absolute_empirical_quantile_distance": float(
                np.mean(distances)),
            "maximum_absolute_empirical_quantile_distance": float(
                np.max(distances)),
            "mean_absolute_empirical_quantile_bin_distance": float(
                np.mean(bin_distances)),
            "maximum_absolute_empirical_quantile_bin_distance": int(
                np.max(bin_distances)),
        }

    def generate(
            self, control_name: str,
            replicate_index: int) -> tuple[OperatorCircuit, dict[str, Any]]:
        if control_name not in FROZEN_IOI_CONTROL_ORDER:
            raise ValueError(f"unknown frozen IOI control: {control_name}")
        controls = self.frozen.controls
        replicate_count = int(controls["replicate_count_per_control"])
        if not 0 <= int(replicate_index) < replicate_count:
            raise ValueError("frozen IOI control replicate index is out of range")
        config = controls[control_name]
        seed = int(config["seed"])
        rng = self._rng(seed, int(replicate_index))
        match_audit: dict[str, Any] = {}
        if control_name == "equal_count_random":
            sites = self._random_sites(
                ((self._equal_pool, self.frozen.selected_k),), rng)
        elif control_name == "same_layer_random":
            sites = self._random_sites((
                (self._layer_pools[layer], self.selected_by_layer[layer])
                for layer in range(self.shape.n_layers)
            ), rng)
        elif control_name == "route_frequency_matched":
            sites = self._random_sites((
                (self._route_pools[route], self.selected_by_route[route])
                for route in ROUTES
            ), rng)
        else:
            sites, match_audit = self._activation_sites(
                rng, quantile_bin_count=replicate_count)
        if len(sites) != self.frozen.selected_k or len(set(sites)) != len(sites):
            raise RuntimeError("frozen IOI control size or uniqueness drift")
        overlap = set(sites) & self.selected
        if overlap:
            raise RuntimeError(
                "frozen IOI control sampled a frozen circuit site")
        route_counts = {
            route: sum(site.route == route for site in sites)
            for route in ROUTES
        }
        layer_counts = {
            str(layer): sum(site.layer == layer for site in sites)
            for layer in range(self.shape.n_layers)
        }
        if (control_name == "same_layer_random"
                and layer_counts != {
                    str(layer): self.selected_by_layer[layer]
                    for layer in range(self.shape.n_layers)}):
            raise RuntimeError("same-layer control match drift")
        if (control_name == "route_frequency_matched"
                and route_counts != self.selected_by_route):
            raise RuntimeError("route-frequency control match drift")
        if control_name == "activation_matched":
            if layer_counts != {
                    str(layer): self.selected_by_layer[layer]
                    for layer in range(self.shape.n_layers)}:
                raise RuntimeError("activation control layer match drift")
            if route_counts != self.selected_by_route:
                raise RuntimeError("activation control route match drift")
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
            discovery_benchmark="mib_ioi",
            discovery_phase="discovery",
            metadata={
                "control_name": control_name,
                "replicate_index": int(replicate_index),
                "base_seed": seed,
                "rng_derivation": "numpy_seed_sequence_base_seed_and_replicate",
                "algorithm_version": FROZEN_IOI_CONTROL_ALGORITHM_VERSION,
                "site_identity_hash": identity_hash,
            },
        ).validate(self.shape)
        audit = {
            "replicate_index": int(replicate_index),
            "site_count": len(sites),
            "site_identity_hash": identity_hash,
            "frozen_circuit_overlap_count": 0,
            "route_counts": route_counts,
            "layer_counts": layer_counts,
            **match_audit,
        }
        return circuit, audit


def _score_vector(
        scores: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(scores[name], dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
        raise ValueError(f"frozen IOI score vector {name} is invalid")
    return value


def condition_effect_vectors(
        intact: Mapping[str, Any],
        condition: Mapping[str, Any]) -> dict[str, np.ndarray]:
    intact_positive = _score_vector(intact, "positive_log_probability")
    intact_negative = _score_vector(intact, "negative_log_probability")
    intact_unrelated = _score_vector(
        intact, "unrelated_mean_log_probability")
    positive = _score_vector(condition, "positive_log_probability")
    negative = _score_vector(condition, "negative_log_probability")
    unrelated = _score_vector(
        condition, "unrelated_mean_log_probability")
    shapes = {
        value.shape for value in (
            intact_positive, intact_negative, intact_unrelated,
            positive, negative, unrelated)
    }
    if len(shapes) != 1:
        raise ValueError("frozen IOI condition score vectors are misaligned")
    intact_margin = intact_positive - intact_negative
    margin = positive - negative
    return {
        "intact_margin": intact_margin,
        "margin": margin,
        "margin_drop": intact_margin - margin,
        "correct_log_probability_change": positive - intact_positive,
        "source_log_probability_change": negative - intact_negative,
        "source_minus_correct_margin_change": (
            (negative - positive) - (intact_negative - intact_positive)),
        "unrelated_log_probability_change": unrelated - intact_unrelated,
        "unrelated_log_probability_damage": intact_unrelated - unrelated,
    }


def summarize_condition(
        intact: Mapping[str, Any], condition: Mapping[str, Any], *,
        bootstrap_samples: int, alpha: float, seed: int) -> dict[str, Any]:
    effects = condition_effect_vectors(intact, condition)
    intact_margin = effects["intact_margin"]
    margin = effects["margin"]
    intact_correct = intact_margin > 0.0
    correct_change = effects["correct_log_probability_change"]
    source_change = effects["source_log_probability_change"]
    source_direction = effects["source_minus_correct_margin_change"]
    unrelated_damage = effects["unrelated_log_probability_damage"]
    return {
        "mean_margin": float(np.mean(margin)),
        "intact_exact_accuracy": float(np.mean(intact_correct)),
        "exact_accuracy": float(np.mean(margin > 0.0)),
        "mean_margin_drop": float(np.mean(effects["margin_drop"])),
        "margin_drop_bootstrap_ci": bootstrap_mean_ci(
            effects["margin_drop"],
            samples=bootstrap_samples,
            alpha=alpha,
            seed=seed,
        ),
        "prediction_flip_fraction": float(np.mean(
            intact_correct != (margin > 0.0))),
        "mean_correct_log_probability_change": float(np.mean(correct_change)),
        "correct_log_probability_change_bootstrap_ci": bootstrap_mean_ci(
            correct_change,
            samples=bootstrap_samples,
            alpha=alpha,
            seed=seed + 1,
        ),
        "mean_source_log_probability_change": float(np.mean(source_change)),
        "mean_source_minus_correct_margin_change": float(
            np.mean(source_direction)),
        "source_direction_bootstrap_ci": bootstrap_mean_ci(
            source_direction,
            samples=bootstrap_samples,
            alpha=alpha,
            seed=seed + 2,
        ),
        "mean_unrelated_log_probability_damage": float(
            np.mean(unrelated_damage)),
        "unrelated_log_probability_damage_bootstrap_ci": bootstrap_mean_ci(
            unrelated_damage,
            samples=bootstrap_samples,
            alpha=alpha,
            seed=seed + 3,
        ),
        "unrelated_behavior_definition": (
            "mean_next_token_log_probability_on_pre_S2_base_prompt_tokens_"
            "excluding_all_subject_and_indirect_object_name_spans"),
    }


def bootstrap_restoration_recovery(
        intact_margin: Sequence[float],
        suppressed_margin: Sequence[float],
        restored_margin: Sequence[float], *,
        samples: int, alpha: float, seed: int) -> dict[str, Any]:
    intact = np.asarray(intact_margin, dtype=np.float64)
    suppressed = np.asarray(suppressed_margin, dtype=np.float64)
    restored = np.asarray(restored_margin, dtype=np.float64)
    if (intact.shape != suppressed.shape
            or intact.shape != restored.shape
            or intact.ndim != 1
            or intact.size < 2
            or not np.all(np.isfinite(
                np.concatenate((intact, suppressed, restored))))):
        raise ValueError("restoration recovery vectors are invalid")

    def ratio(indices: np.ndarray) -> float:
        denominator = float(np.mean(intact[indices] - suppressed[indices]))
        if abs(denominator) <= 1.0e-12:
            return float("nan")
        numerator = float(np.mean(restored[indices] - suppressed[indices]))
        return numerator / denominator

    full_indices = np.arange(intact.size)
    estimate = ratio(full_indices)
    rng = np.random.default_rng(int(seed))
    values = np.empty((int(samples),), dtype=np.float64)
    for index in range(int(samples)):
        sampled = rng.integers(0, intact.size, size=intact.size)
        values[index] = ratio(sampled)
    finite = values[np.isfinite(values)]
    if not np.isfinite(estimate) or finite.size < max(100, samples // 2):
        return {
            "n": int(intact.size),
            "recovery_fraction": None,
            "ci_low": None,
            "ci_high": None,
            "finite_bootstrap_samples": int(finite.size),
        }
    low, high = np.quantile(
        finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": int(intact.size),
        "recovery_fraction": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
        "finite_bootstrap_samples": int(finite.size),
    }


def compare_frozen_to_controls(
        frozen_margin_drop: Sequence[float],
        control_margin_drop: np.ndarray, *,
        bootstrap_samples: int, permutation_samples: int,
        alpha: float, seed: int) -> dict[str, Any]:
    frozen = np.asarray(frozen_margin_drop, dtype=np.float64)
    controls = np.asarray(control_margin_drop, dtype=np.float64)
    if (frozen.ndim != 1 or controls.ndim != 2
            or controls.shape[1] != frozen.size
            or controls.shape[0] < 2
            or not np.all(np.isfinite(frozen))
            or not np.all(np.isfinite(controls))):
        raise ValueError("frozen/control effect arrays are invalid")
    control_per_example = controls.mean(axis=0)
    observed = float(frozen.mean() - controls.mean())
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty((int(bootstrap_samples),), dtype=np.float64)
    for index in range(int(bootstrap_samples)):
        example_indices = rng.integers(
            0, frozen.size, size=frozen.size)
        replicate_indices = rng.integers(
            0, controls.shape[0], size=controls.shape[0])
        bootstrap[index] = (
            float(np.mean(frozen[example_indices]))
            - float(np.mean(
                controls[replicate_indices][:, example_indices]))
        )
    low, high = np.quantile(
        bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    paired_null = paired_permutation_test(
        frozen,
        control_per_example,
        samples=permutation_samples,
        seed=seed + 1,
    )
    replicate_means = controls.mean(axis=1)
    return {
        "mean_frozen_minus_control": observed,
        "bootstrap_ci": {
            "n_examples": int(frozen.size),
            "n_control_replicates": int(controls.shape[0]),
            "mean": observed,
            "ci_low": float(low),
            "ci_high": float(high),
        },
        "paired_permutation_against_mean_control": paired_null,
        "frozen_exceeds_control_mean": bool(observed > 0.0),
        "frozen_exceeds_every_control_replicate": bool(
            np.all(frozen.mean() > replicate_means)),
        "fraction_control_replicates_below_frozen": float(
            np.mean(replicate_means < frozen.mean())),
        "minimum_frozen_minus_replicate_mean": float(
            frozen.mean() - np.max(replicate_means)),
    }
