"""End-to-end, protocol-bound operator interpretability execution."""

from __future__ import annotations

import gc
import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

import jax
import numpy as np
from transformers import AutoTokenizer

from analysis.dawn_analysis_common import (
    analysis_model_module,
    git_info,
    materialize_global_array,
)
from analysis.operator_interpretability.arc_localization import (
    ARC_DISCOVERY_PAIRED_CORRECT_COUNT,
    ARC_DISCOVERY_ROW_COUNT,
    build_arc_localization,
    load_arc_discovery_spec,
)
from analysis.operator_interpretability.arc_frozen_circuit import (
    FROZEN_ARC_CONTROL_ALGORITHM_VERSION,
    FROZEN_ARC_CONTROL_NAME,
    FROZEN_ARC_VALIDATION_PAIRED_CORRECT_COUNT,
    FROZEN_ARC_VALIDATION_ROW_COUNT,
    FrozenARCControlSampler,
    load_frozen_arc_circuit,
)
from analysis.operator_interpretability.artifacts import (
    load_benchmark_examples,
    resolve_benchmark_build,
    write_protocol_bound_artifact,
)
from analysis.operator_interpretability.benchmark_registry import (
    PRIMARY_BENCHMARK_IDS,
    assert_benchmark_support,
    benchmark_spec,
)
from analysis.operator_interpretability.benchmark_schema import (
    BENCHMARK_SCHEMA,
    BENCHMARK_SCHEMA_VERSION,
    canonical_hash,
)
from analysis.operator_interpretability.capture import (
    capture_discovery_candidates,
    capture_held_out_paths,
    capture_program_paths,
    ranked_site_objects,
)
from analysis.operator_interpretability.circuit import (
    bootstrap_faithfulness_ci,
    faithfulness_curve,
    necessity_effect,
    normalized_faithfulness,
    select_on_validation,
)
from analysis.operator_interpretability.claim_gate import evaluate_claims
from analysis.operator_interpretability.eligibility import tokenizer_vocab_hash
from analysis.operator_interpretability.frozen_circuit import (
    FROZEN_IOI_CONTROL_ALGORITHM_VERSION,
    FROZEN_IOI_CONTROL_ORDER,
    FROZEN_IOI_TEST_PAIRED_CORRECT_COUNT,
    FROZEN_IOI_TEST_ROW_COUNT,
    FROZEN_IOI_VALIDATION_ANALYSIS_COMMIT,
    FROZEN_IOI_VALIDATION_PAIRED_CORRECT_COUNT,
    FROZEN_IOI_VALIDATION_RECORD_COMMIT,
    FROZEN_IOI_VALIDATION_RESULT_ROOT,
    FROZEN_IOI_VALIDATION_RESULT_SHA256,
    FROZEN_IOI_VALIDATION_ROW_COUNT,
    FrozenIOIControlSampler,
    bootstrap_restoration_recovery,
    compare_frozen_to_controls,
    condition_effect_vectors,
    load_frozen_ioi_circuit,
    summarize_condition,
)
from analysis.operator_interpretability.interchange import score_interchange_rows
from analysis.operator_interpretability.intervention import (
    all_ones_retention_parity,
    evaluate_behavior,
    evaluate_circuit_necessity,
    evaluate_circuit_retention,
    evaluate_frozen_circuit_condition,
    evaluate_native_operator_program_causal_diagnostics,
    evaluate_native_operator_program_phase_baselines,
    evaluate_native_operator_program_selection_candidate,
    evaluate_operator_interchange,
    prepare_arc_frozen_circuit_evaluation,
    prepare_frozen_circuit_evaluation,
)
from analysis.operator_interpretability.ioi_scale_localization import (
    IOI_SCALE_DISCOVERY_MINIMUM_PAIRED_CORRECT_COUNT,
    IOI_SCALE_DISCOVERY_ROW_COUNT,
    build_ioi_scale_localization,
    load_ioi_scale_discovery_spec,
)
from analysis.operator_interpretability.program import (
    PROGRAM_ALGORITHM_VERSION,
    build_program_schedule,
    capture_schedule_widths,
    compactness_metrics,
    deterministic_mismatch_mapping,
    evaluate_native_program_claims,
    native_program_diagnostic_checks,
    reindex_program_schedule,
    select_validation_program,
)
from analysis.operator_interpretability.paired_trajectory import (
    build_divergence_atlas,
    capture_candidate_site_values,
    capture_full_active_trajectory,
    capture_production_atlas,
    deduplicate_residual_candidates,
    deterministic_deep_selection,
    deterministic_mismatch_mapping as trajectory_mismatch_mapping,
    divergence_extrema,
    evaluate_coarse_site_patches,
    evaluate_cumulative_path,
    evaluate_operator_group_patches,
    freeze_chronological_path,
    freeze_operator_followup_sites,
    ioi_semantic_record,
    merge_divergence_atlases,
    merge_staged_coarse_patch_results,
    merge_trajectory_batch_summaries,
    operator_parameter_provenance,
    select_discovery_candidates,
    write_atlas_metric_artifact,
    write_causal_vector_artifact,
    write_deep_trace_shards,
    write_trajectory_graph,
    write_trajectory_manifest,
)
from analysis.operator_interpretability.protocol import (
    CIRCUIT_FRACTIONS,
    ProtocolConfig,
    protocol_record,
    validate_model_version,
)
from analysis.operator_interpretability.ravel_localization import (
    RAVEL_DISCOVERY_PAIRED_CORRECT_COUNT,
    RAVEL_DISCOVERY_ROW_COUNT,
    build_ravel_variable_localization,
    load_ravel_discovery_spec,
)
from analysis.operator_interpretability.space import (
    address_confirmation,
    discover_functional_families,
    operator_pool_provenance,
)
from analysis.operator_interpretability.statistics import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    paired_permutation_test,
)
from analysis.operator_interpretability.trajectory import (
    held_out_trajectory_confirmation,
)
from analysis.operator_interpretability.units import (
    OperatorCircuit,
    OperatorSpaceShape,
    nested_circuits,
)
from analysis.train_analysis_pool_items import (
    dependency_closure,
    item_definition,
)
from analysis.train_analysis_pool_reporting import TrainAnalysisPoolTextReporter


MIB_CIRCUIT_BENCHMARKS = (
    "mib_ioi", "mib_mcqa", "mib_arithmetic", "mib_arc",
)


def _mean(values: Sequence[float]) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _parameter_schema_record(params: Any) -> list[dict[str, Any]]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        rows.append({
            "path": jax.tree_util.keystr(path),
            "shape": [int(value) for value in leaf.shape],
            "dtype": str(leaf.dtype),
        })
    return rows


class OperatorInterpretabilityRunner:
    """One canonical scientific path; no legacy artifacts or prompt aliases."""

    def __init__(
            self, ctx: Any, *, benchmark_root: str,
            benchmark_ids: Sequence[str], protocol_config: ProtocolConfig,
            text_reporter: TrainAnalysisPoolTextReporter | None = None) -> None:
        self.ctx = ctx
        self.store = ctx.store
        self.config = protocol_config.validate()
        self.text_reporter = text_reporter
        self.model_version = validate_model_version(
            str(ctx.model_cfg["model_version"]))
        self.shape = OperatorSpaceShape.from_model_cfg(ctx.model_cfg)
        self.build = resolve_benchmark_build(benchmark_root)
        self.benchmark_ids = tuple(dict.fromkeys(
            str(value).strip().lower() for value in benchmark_ids))
        if not self.benchmark_ids:
            raise ValueError("at least one benchmark id is required")
        for benchmark_id in self.benchmark_ids:
            assert_benchmark_support(benchmark_id, self.model_version)
            if benchmark_id not in self.build.manifest["benchmarks"]:
                raise FileNotFoundError(
                    f"benchmark build lacks requested id={benchmark_id}")

        tokenizer_record = dict(self.build.manifest["tokenizer"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_record["name"]),
            revision=str(tokenizer_record["resolved_revision"]),
            use_fast=True,
        )
        actual_vocab_hash = tokenizer_vocab_hash(self.tokenizer)
        if actual_vocab_hash != tokenizer_record["vocab_hash"]:
            raise ValueError("runtime tokenizer vocabulary hash mismatch")
        if int(self.tokenizer.pad_token_id) != int(tokenizer_record["pad_token_id"]):
            raise ValueError("runtime tokenizer pad id mismatch")
        logical_vocab_size = int(ctx.model_cfg.get(
            "logical_vocab_size", ctx.model_cfg["vocab_size"]))
        if logical_vocab_size != int(tokenizer_record["vocab_size"]):
            raise ValueError(
                "checkpoint logical vocabulary and benchmark tokenizer differ: "
                f"checkpoint={logical_vocab_size} "
                f"benchmark={tokenizer_record['vocab_size']}")
        configured_tokenizer = ctx.config.get("tokenizer")
        if isinstance(configured_tokenizer, str) and (
                configured_tokenizer != tokenizer_record["name"]):
            raise ValueError("checkpoint tokenizer name and benchmark tokenizer differ")

        parameter_schema_hash = canonical_hash(
            _parameter_schema_record(ctx.params))
        checkpoint_identity_record = {
            "path": str(ctx.checkpoint_path),
            "step": int(ctx.checkpoint_step),
            "run_id": (
                str(ctx.checkpoint_metadata["run_id"])
                if ctx.checkpoint_metadata.get("run_id") is not None
                else None),
            "training_git_commit": (
                str(ctx.checkpoint_metadata.get("git_commit")
                    or ctx.checkpoint_metadata.get("train_script_git_commit"))
                if (ctx.checkpoint_metadata.get("git_commit")
                    or ctx.checkpoint_metadata.get("train_script_git_commit"))
                else None),
            "parameter_schema_hash": parameter_schema_hash,
            "identity_algorithm": (
                "resolved_path_step_run_metadata_and_parameter_schema"),
            "parameter_content_hash_included": False,
        }
        checkpoint_identity = canonical_hash(checkpoint_identity_record)
        model_config_hash = canonical_hash(dict(ctx.model_cfg))
        self.protocol = protocol_record(
            self.config,
            model_version=self.model_version,
            benchmark_manifest_hash=self.build.manifest_hash,
            checkpoint_identity=checkpoint_identity,
            model_config_hash=model_config_hash,
        )
        self.protocol["execution"] = {
            "target_id": ctx.model_info.get("target_id"),
            "runtime_id": ctx.model_info.get("runtime_id"),
            "accelerator_type": ctx.model_info.get("accelerator_type"),
            "checkpoint_mesh": ctx.model_info.get("checkpoint_mesh"),
            "effective_mesh": ctx.model_info.get("mesh"),
        }
        self.protocol["analysis_code"] = git_info()
        self.contract = {
            "status": "ready",
            "model_version": self.model_version,
            "supported_model_versions": list(
                benchmark_spec(self.benchmark_ids[0]).supported_model_versions),
            "checkpoint_path": str(ctx.checkpoint_path),
            "checkpoint_step": int(ctx.checkpoint_step),
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_identity_record": checkpoint_identity_record,
            "parameter_schema_hash": parameter_schema_hash,
            "checkpoint_parameter_content_hash_included": False,
            "model_config_hash": model_config_hash,
            "benchmark_build_id": self.build.build_id,
            "benchmark_schema": BENCHMARK_SCHEMA,
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_manifest_path": self.build.manifest_path,
            "benchmark_manifest_hash": self.build.manifest_hash,
            "benchmark_ids": list(self.benchmark_ids),
            "execution": dict(self.protocol["execution"]),
            "tokenizer": tokenizer_record,
            "protocol": self.protocol,
        }
        self.results: dict[str, dict[str, Any]] = {}
        self.concrete_results: dict[str, dict[str, Any]] = {}
        self._kind_items: dict[str, list[str]] = {}
        self._kind_benchmark_ids: dict[str, tuple[str, ...]] = {}
        self._active_kind: str | None = None
        self._examples: dict[str, dict[str, list[Any]]] = {}
        self._pool_host: dict[str, np.ndarray] | None = None
        self._requested_items: tuple[str, ...] = ()
        self._paired_trajectory_test_isolated = False
        self._frozen_circuit_phase_isolated = False
        self._ioi_scale_discovery_isolated = False
        self._arc_discovery_isolated = False
        self._arc_frozen_validation_isolated = False
        self._ravel_discovery_isolated = False

    def _print(self, message: str) -> None:
        if self.ctx.is_primary:
            print(message, flush=True)

    @staticmethod
    def _item_relative_path(item_id: str) -> str:
        parts = str(item_id).split(".")
        return "/".join(("items", *parts[:-1], f"{parts[-1]}.json"))

    def _artifact_path(self, item_id: str) -> str:
        return self.store.path(*self._item_relative_path(item_id).split("/"))

    def _scope(self, kind: str | None = None) -> tuple[str, ...]:
        key = str(kind or self._active_kind or "")
        scopes = getattr(self, "_kind_benchmark_ids", {})
        return scopes.get(key, tuple(getattr(self, "benchmark_ids", ())))

    def _concrete_payload(
            self, item_id: str, kind: str,
            result: Mapping[str, Any]) -> dict[str, Any]:
        definition = item_definition(item_id)
        benchmark_id = definition.get("benchmark_id")
        if benchmark_id is not None and kind != "input_contract" and (
                isinstance(result.get("benchmarks"), Mapping)):
            benchmark_result = result["benchmarks"].get(benchmark_id)
            if not isinstance(benchmark_result, Mapping):
                raise ValueError(
                    f"analysis kind {kind} omitted requested benchmark "
                    f"{benchmark_id}")
            item_result = dict(benchmark_result)
        elif kind == "input_contract":
            item_result = dict(result)
            item_result["benchmark_ids"] = [str(benchmark_id)]
            item_result["benchmark_id"] = str(benchmark_id)
        else:
            item_result = dict(result)
        payload = {
            "item_id": item_id,
            "backend": definition["backend"],
            "analysis_kind": kind,
            "benchmark_id": benchmark_id,
            "claim_role": definition["claim_role"],
            "scientific_role": definition.get("scientific_role"),
            "status": item_result.get("status"),
            "result": item_result,
        }
        if "test_used" in definition:
            payload["test_used"] = bool(definition["test_used"])
        artifact_warnings = item_result.get("artifact_warnings")
        if isinstance(artifact_warnings, Sequence) and not isinstance(
                artifact_warnings, (str, bytes)):
            payload["artifact_warnings"] = [
                dict(row) for row in artifact_warnings
                if isinstance(row, Mapping)
            ]
        return payload

    @staticmethod
    def _strip_nested_rows(value: Any) -> Any:
        if isinstance(value, Mapping):
            output = {}
            for key, child in value.items():
                if key == "rows" and isinstance(child, Sequence) and not (
                        isinstance(child, (str, bytes))):
                    output["raw_row_count"] = len(child)
                    output["raw_rows_persisted"] = False
                else:
                    output[str(key)] = (
                        OperatorInterpretabilityRunner._strip_nested_rows(
                            child))
            return output
        if isinstance(value, list):
            return [
                OperatorInterpretabilityRunner._strip_nested_rows(child)
                for child in value
            ]
        if isinstance(value, tuple):
            return [
                OperatorInterpretabilityRunner._strip_nested_rows(child)
                for child in value
            ]
        return value

    @staticmethod
    def _compact_selected_circuit(result: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(result)
        selected = output.get("selected_circuit")
        if not isinstance(selected, Mapping):
            return output
        selected_output = dict(selected)
        sites = selected_output.pop("sites", ())
        selected_output.update({
            "circuit_hash": str(
                selected_output.get("circuit_hash")
                or canonical_hash(dict(selected))),
            "explicit_site_count": len(sites),
            "sites_persisted": False,
            "site_definition": (
                "validation_selected_fraction_of_localization_ranking"),
        })
        output["selected_circuit"] = selected_output
        return output

    def _compact_scientific_claims(
            self, item_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        claims = {}
        for name, row in dict(result.get("claims") or {}).items():
            claims[str(name)] = {
                "passed": bool(row.get("passed", False)),
                "unmet_prerequisites": list(
                    row.get("unmet_prerequisites") or ()),
            }
        upstream = dependency_closure([item_id])[:-1]
        return {
            "status": result.get("status"),
            "claims": claims,
            "strongest_supported_claim": result.get(
                "strongest_supported_claim"),
            "checkpoint_scope": result.get("checkpoint_scope"),
            "cross_checkpoint_claim": result.get("cross_checkpoint_claim"),
            "suppression_interpreted_as": result.get(
                "suppression_interpreted_as"),
            "benchmark_scope": list(result.get("benchmark_scope") or ()),
            "checkpoint_identity": result.get("checkpoint_identity"),
            "single_checkpoint_only": result.get("single_checkpoint_only"),
            "official_transformerlens_edge_equivalence_claimed": result.get(
                "official_transformerlens_edge_equivalence_claimed"),
            "official_ravel_featurizer_equivalence_claimed": result.get(
                "official_ravel_featurizer_equivalence_claimed"),
            "evidence_embedded": False,
            "evidence_contract": (
                "protocol_bound_upstream_item_artifact_references"),
            "upstream_item_artifacts": {
                upstream_item: self._artifact_path(upstream_item)
                for upstream_item in upstream
            },
        }

    def _payload_for_storage(
            self, item_id: str, kind: str,
            payload: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(payload)
        result = dict(payload.get("result") or {})
        if kind == "behavioral_eligibility":
            result = self._compact_behavioral_eligibility(result)
        elif kind == "operator_localization" and isinstance(
                result.get("ranked_sites"), Sequence):
            ranked_sites = [
                dict(row) for row in result.get("ranked_sites", ())
                if isinstance(row, Mapping)
            ]
            profiles = dict(
                result.get("causal_variable_control_profiles", {}) or {})
            result.update({
                "ranked_site_count": len(ranked_sites),
                "ranked_site_preview": [
                    dict(row) for row in ranked_sites[:16]],
                "ranked_sites_content_hash": canonical_hash(ranked_sites),
                "ranked_sites_persisted_in_item_json": True,
                "ranked_sites_are_aggregate_discovery_statistics": True,
                "ranked_sites_are_per_example_vectors": False,
                "causal_variable_profile_summary": {
                    str(variable): {
                        "layer": int(profile["layer"]),
                        "route": str(profile["route"]),
                        "qualified_row_denominator": int(
                            profile["qualified_row_denominator"]),
                        "operator_count": len(
                            profile.get("operator_ids") or ()),
                    }
                    for variable, profile in profiles.items()
                },
                "causal_variable_profiles_persisted_in_item_json": bool(
                    profiles),
                "causal_variable_profiles_are_aggregate_statistics": True,
                "raw_parameters_persisted": False,
                "dense_capture_persisted": False,
            })
        elif kind in {
                "conditional_circuit_sufficiency",
                "autonomous_circuit_sufficiency"}:
            result = self._compact_selected_circuit(result)
        elif kind == "circuit_necessity":
            result = self._strip_nested_rows(result)
            intervention = result.get("intervention")
            if isinstance(intervention, Mapping) and isinstance(
                    intervention.get("margin"), Sequence):
                intervention_output = dict(intervention)
                margin = intervention_output.pop("margin")
                intervention_output["margin_count"] = len(margin)
                intervention_output["margin_persisted"] = False
                result["intervention"] = intervention_output
        elif kind in {"ravel_causal_mediation", "multilayer_trajectory"}:
            result = self._strip_nested_rows(result)
        elif kind == "native_operator_program":
            result = self._strip_nested_rows(result)
        elif kind == "paired_operator_trajectory":
            result = self._trajectory_without_private(result)
        elif kind == "scientific_claims":
            result = self._compact_scientific_claims(item_id, result)
        output["result"] = result
        return output

    _BEHAVIOR_RAW_VECTOR_FIELDS = (
        "example_ids",
        "base_positive_logp",
        "base_negative_logp",
        "base_margin",
        "corrupted_margin",
        "source_own_margin",
        "source_behavior_scored",
        "base_known_correct",
        "source_known_correct",
        "known_correct",
    )

    @staticmethod
    def _raw_vector_summary(values: Sequence[Any]) -> dict[str, Any]:
        sequence = list(values)
        non_null = [value for value in sequence if value is not None]
        output: dict[str, Any] = {
            "count": len(sequence),
            "null_count": len(sequence) - len(non_null),
        }
        if all(isinstance(value, (bool, np.bool_)) for value in non_null):
            true_count = sum(bool(value) for value in non_null)
            output.update({
                "value_type": "boolean",
                "true_count": int(true_count),
                "false_count": int(len(non_null) - true_count),
            })
        elif all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in non_null):
            numeric = np.asarray(non_null, dtype=np.float64)
            finite = numeric[np.isfinite(numeric)]
            output.update({
                "value_type": "numeric",
                "finite_count": int(finite.size),
                "non_finite_count": int(numeric.size - finite.size),
                "minimum": float(np.min(finite)) if finite.size else None,
                "maximum": float(np.max(finite)) if finite.size else None,
                "mean": float(np.mean(finite)) if finite.size else None,
            })
        elif all(isinstance(value, str) for value in non_null):
            output.update({
                "value_type": "string",
                "unique_count": len(set(non_null)),
            })
        else:
            output["value_type"] = "mixed"
        return output

    @classmethod
    def _compact_behavioral_eligibility(
            cls, result: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(result)
        phases = output.get("phases")
        if not isinstance(phases, Mapping):
            return output
        compact_phases = {}
        for phase, phase_result in phases.items():
            if not isinstance(phase_result, Mapping):
                compact_phases[str(phase)] = phase_result
                continue
            compact = dict(phase_result)
            raw_vectors = {}
            for field in cls._BEHAVIOR_RAW_VECTOR_FIELDS:
                value = compact.pop(field, None)
                if isinstance(value, Sequence) and not isinstance(
                        value, (str, bytes)):
                    raw_vectors[field] = list(value)
            if raw_vectors:
                compact.update({
                    "raw_vector_fields": sorted(raw_vectors),
                    "raw_vector_field_count": len(raw_vectors),
                    "raw_vector_row_count": max(
                        len(values) for values in raw_vectors.values()),
                    "raw_vector_payload_hash": canonical_hash(raw_vectors),
                    "raw_vector_summaries": {
                        field: cls._raw_vector_summary(values)
                        for field, values in sorted(raw_vectors.items())
                    },
                    "raw_vectors_persisted": False,
                    "raw_vector_storage_policy": (
                        "aggregate_statistics_and_content_hash_only"),
                })
            compact_phases[str(phase)] = compact
        output["phases"] = compact_phases
        return output

    def _ensure_kind(
            self, kind: str, item_ids: Sequence[str]) -> dict[str, Any]:
        if kind in self.results:
            return self.results[kind]
        self._active_kind = kind
        for item_id in item_ids:
            self._print(f"TRAIN_ANALYSIS_POOL item={item_id} status=running")
        method = getattr(self, f"_run_{kind}")
        result = dict(method())
        self.results[kind] = result
        for item_id in item_ids:
            runtime_payload = self._concrete_payload(item_id, kind, result)
            payload = (
                self._payload_for_storage(item_id, kind, runtime_payload)
                if self.ctx.is_primary else runtime_payload)
            self.concrete_results[item_id] = payload
            if self.ctx.is_primary:
                write_protocol_bound_artifact(
                    self.store, self._item_relative_path(item_id), payload,
                    protocol=self.protocol)
            self._print(
                f"TRAIN_ANALYSIS_POOL item={item_id} "
                f"status={payload.get('status')}")
            if self.text_reporter is not None:
                self.text_reporter.emit(
                    payload, artifact_path=self._artifact_path(item_id),
                    event="completed")
        return result

    def run(self, items: Sequence[str]) -> dict[str, Any]:
        self._requested_items = tuple(str(item) for item in items)
        requested_set = set(self._requested_items)
        frozen_validation_item = "mib_ioi.frozen_circuit_validation"
        frozen_test_item = "mib_ioi.frozen_circuit_test"
        frozen_validation_requested = (
            frozen_validation_item in requested_set)
        frozen_test_requested = frozen_test_item in requested_set
        if frozen_validation_requested and frozen_test_requested:
            raise ValueError(
                "frozen IOI validation and held-out test must run as "
                "separate preset invocations")
        frozen_circuit_item = (
            frozen_validation_item if frozen_validation_requested
            else frozen_test_item if frozen_test_requested
            else None)
        frozen_circuit_requested = frozen_circuit_item is not None
        frozen_circuit_only_items = {
            "mib_ioi.input_contract",
            *({frozen_circuit_item} if frozen_circuit_item is not None else ()),
        }
        if (frozen_circuit_requested
                and not requested_set <= frozen_circuit_only_items):
            raise ValueError(
                f"{frozen_circuit_item} must be requested only with its "
                "dedicated input-contract item so split access remains "
                "isolated")
        self._frozen_circuit_phase_isolated = frozen_circuit_requested
        ioi_scale_discovery_item = (
            "mib_ioi.scale_discovery_operator_localization")
        ioi_scale_discovery_requested = (
            ioi_scale_discovery_item in requested_set)
        ioi_scale_discovery_only_items = {
            "mib_ioi.input_contract",
            ioi_scale_discovery_item,
        }
        if (ioi_scale_discovery_requested
                and not requested_set <= ioi_scale_discovery_only_items):
            raise ValueError(
                "mib_ioi.scale_discovery_operator_localization must run only "
                "with its input contract so validation and test access remain "
                "forbidden")
        self._ioi_scale_discovery_isolated = (
            ioi_scale_discovery_requested)
        arc_frozen_validation_item = (
            "mib_arc.frozen_circuit_validation")
        arc_frozen_validation_requested = (
            arc_frozen_validation_item in requested_set)
        arc_frozen_validation_only_items = {
            "mib_arc.input_contract",
            arc_frozen_validation_item,
        }
        if (arc_frozen_validation_requested
                and not requested_set <= arc_frozen_validation_only_items):
            raise ValueError(
                "mib_arc.frozen_circuit_validation must run only with its "
                "input contract so discovery and held-out test access remain "
                "isolated")
        self._arc_frozen_validation_isolated = (
            arc_frozen_validation_requested)
        arc_discovery_item = "mib_arc.discovery_operator_localization"
        arc_discovery_requested = arc_discovery_item in requested_set
        arc_discovery_only_items = {
            "mib_arc.input_contract",
            arc_discovery_item,
        }
        if (arc_discovery_requested
                and not requested_set <= arc_discovery_only_items):
            raise ValueError(
                "mib_arc.discovery_operator_localization must run only with "
                "its input contract so validation and test access remain "
                "forbidden")
        self._arc_discovery_isolated = arc_discovery_requested
        ravel_discovery_item = "ravel.discovery_operator_localization"
        ravel_discovery_requested = ravel_discovery_item in requested_set
        ravel_discovery_only_items = {
            "ravel.input_contract",
            ravel_discovery_item,
        }
        if (ravel_discovery_requested
                and not requested_set <= ravel_discovery_only_items):
            raise ValueError(
                "ravel.discovery_operator_localization must run only with "
                "its input contract so validation and test access remain "
                "forbidden")
        self._ravel_discovery_isolated = ravel_discovery_requested
        trajectory_only_items = {
            "mib_ioi.input_contract",
            "mib_ioi.behavioral_eligibility",
            "mib_ioi.paired_operator_trajectory",
        }
        trajectory_requested = (
            "mib_ioi.paired_operator_trajectory" in requested_set)
        if trajectory_requested and not requested_set <= trajectory_only_items:
            raise ValueError(
                "mib_ioi.paired_operator_trajectory must be requested only "
                "with its dedicated input-contract and behavioral-eligibility "
                "items so held-out test access remains isolated")
        self._paired_trajectory_test_isolated = trajectory_requested
        executed = dependency_closure(items)
        kind_order: list[str] = []
        for item_id in executed:
            definition = item_definition(item_id)
            if definition["backend"] != "operator_interpretability":
                raise ValueError(
                    f"OperatorInterpretabilityRunner cannot execute backend="
                    f"{definition['backend']} item={item_id}")
            kind = str(definition["analysis_kind"])
            if kind not in self._kind_items:
                self._kind_items[kind] = []
                kind_order.append(kind)
            self._kind_items[kind].append(item_id)
        for kind, item_ids in self._kind_items.items():
            benchmark_ids = [
                str(item_definition(item_id)["benchmark_id"])
                for item_id in item_ids
                if item_definition(item_id).get("benchmark_id") is not None
            ]
            self._kind_benchmark_ids[kind] = tuple(dict.fromkeys(benchmark_ids))
        for kind in kind_order:
            self._ensure_kind(kind, self._kind_items[kind])
        artifact_warnings = []
        for item_id in executed:
            for warning in self.concrete_results[item_id].get(
                    "artifact_warnings", ()):
                if isinstance(warning, Mapping):
                    artifact_warnings.append({
                        "item_id": item_id,
                        **dict(warning),
                    })
        summary = {
            "status": "complete",
            "requested_items": list(items),
            "executed_items": executed,
            "model_version": self.model_version,
            "checkpoint_step": int(self.ctx.checkpoint_step),
            "benchmark_build_id": self.build.build_id,
            "protocol_hash": canonical_hash(self.protocol),
            "item_status": {
                item: self.concrete_results[item].get("status")
                for item in executed
            },
            "artifact_warnings": artifact_warnings,
            "strongest_supported_claim": (
                self.results.get("scientific_claims", {}).get(
                    "strongest_supported_claim")
                or self.results.get("frozen_circuit_test", {}).get(
                    "strongest_supported_claim")
                or self.results.get("frozen_circuit_validation", {}).get(
                    "strongest_supported_claim")
                or self.results.get(
                    "ioi_scale_discovery_localization", {}).get(
                        "strongest_supported_claim")
                or self.results.get("arc_frozen_validation", {}).get(
                    "strongest_supported_claim")
                or self.results.get("arc_discovery_localization", {}).get(
                    "strongest_supported_claim")
                or self.results.get("ravel_discovery_localization", {}).get(
                    "strongest_supported_claim")
                or self.results.get("native_operator_program", {}).get(
                    "strongest_supported_claim")),
        }
        if self.ctx.is_primary:
            write_protocol_bound_artifact(
                self.store, "backends/operator_interpretability/summary.json",
                summary, protocol=self.protocol)
        return summary

    def _load_phase_examples(
            self, benchmark_id: str, phase: str) -> list[Any]:
        if phase not in {"discovery", "validation", "test"}:
            raise ValueError(f"unknown benchmark phase={phase}")
        phases = self._examples.setdefault(benchmark_id, {})
        if phase in phases:
            return phases[phase]
        values = load_benchmark_examples(
            self.build, benchmark_id, phase=phase)
        phase_cap = self.config.max_examples_for(benchmark_id)
        if benchmark_id == "ravel":
            spec = benchmark_spec("ravel")
            grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
            for example in values:
                source_column = str(example.metadata.get(
                    "official_counterfactual_column") or "")
                key = (
                    str(example.causal_variable),
                    str(example.pair_type),
                    source_column,
                )
                grouped[key].append(example)
            expected_strata = [
                (variable, pair_type, source_column)
                for variable in spec.causal_variables
                for pair_type in ("cause", "isolation")
                for source_column in spec.counterfactual_columns
            ]
            missing = [key for key in expected_strata if not grouped[key]]
            if missing:
                raise ValueError(
                    "RAVEL phase lacks an official causal stratum: "
                    + ",".join("/".join(key) for key in missing))
            if phase_cap < len(expected_strata):
                raise ValueError(
                    "RAVEL phase cap cannot represent every RAVEL stratum; "
                    f"minimum={len(expected_strata)}")
            for group in grouped.values():
                group.sort(key=lambda example: (
                    canonical_hash(example.example_id), example.example_id))
            selected = []
            cursors = {key: 0 for key in expected_strata}
            used_group_ids = {"cause": set(), "isolation": set()}
            while len(selected) < phase_cap:
                added = False
                for key in expected_strata:
                    group = grouped[key]
                    pair_type = key[1]
                    while cursors[key] < len(group):
                        candidate = group[cursors[key]]
                        cursors[key] += 1
                        group_id = str(candidate.metadata["pair_group_id"])
                        if group_id in used_group_ids[pair_type]:
                            continue
                        used_group_ids[pair_type].add(group_id)
                        selected.append(candidate)
                        added = True
                        break
                    if len(selected) == phase_cap:
                        break
                if not added:
                    break
            if len(selected) != phase_cap:
                raise ValueError(
                    "RAVEL prepared phase cannot satisfy the pre-registered "
                    f"runtime cap: phase={phase} requested={phase_cap} "
                    f"available={len(selected)}; publish a non-truncated "
                    "benchmark build")
            phases[phase] = selected
        else:
            values.sort(key=lambda example: (
                canonical_hash(example.example_id), example.example_id))
            phases[phase] = values[:phase_cap]
        return phases[phase]

    def _load_examples(self, benchmark_id: str) -> dict[str, list[Any]]:
        return {
            phase: self._load_phase_examples(benchmark_id, phase)
            for phase in ("discovery", "validation", "test")
        }

    @staticmethod
    def _independent_capture_examples(
            benchmark_id: str, examples: Sequence[Any]) -> list[Any]:
        if benchmark_id != "ravel":
            return list(examples)
        selected: dict[str, Any] = {}
        for example in examples:
            if example.pair_type != "cause":
                continue
            group_id = str(example.metadata["pair_group_id"])
            current = selected.get(group_id)
            if current is None or example.example_id < current.example_id:
                selected[group_id] = example
        return [selected[key] for key in sorted(selected)]

    def _known_correct(self, benchmark_id: str, phase: str) -> list[Any]:
        behavior = self.results["behavioral_eligibility"]["benchmarks"][
            benchmark_id]["phases"][phase]
        mask = list(behavior["known_correct"])
        examples = self._load_phase_examples(benchmark_id, phase)
        if len(mask) != len(examples):
            raise ValueError("behavior mask and benchmark examples are misaligned")
        return [example for example, keep in zip(examples, mask) if keep]

    def _behavior_margins(
            self, benchmark_id: str, phase: str,
            examples: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        result = self.results["behavioral_eligibility"]["benchmarks"][
            benchmark_id]["phases"][phase]
        index = {
            example_id: row for row, example_id in enumerate(result["example_ids"])
        }
        rows = [index[example.example_id] for example in examples]
        return (
            np.asarray(result["base_margin"], dtype=np.float64)[rows],
            np.asarray(result["corrupted_margin"], dtype=np.float64)[rows],
        )

    def _run_input_contract(self) -> dict[str, Any]:
        result = dict(self.contract)
        result["benchmark_ids"] = list(self._scope("input_contract"))
        return result

    def _run_behavioral_eligibility(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for benchmark_id in self._scope("behavioral_eligibility"):
            phase_results = {}
            isolated_trajectory = (
                benchmark_id == "mib_ioi"
                and self._paired_trajectory_test_isolated)
            phases = (
                ("discovery", "validation")
                if isolated_trajectory
                else ("discovery", "validation", "test"))
            for phase in phases:
                examples = self._load_phase_examples(benchmark_id, phase)
                result = evaluate_behavior(
                    self.ctx, examples,
                    pad_token_id=int(self.tokenizer.pad_token_id))
                known_correct_examples = [
                    example for example, keep in zip(
                        examples, result["known_correct"])
                    if keep
                ]
                independent_count = len(self._independent_capture_examples(
                    benchmark_id, known_correct_examples))
                result["known_correct_independent_unit_count"] = (
                    independent_count)
                result["eligible_for_mechanistic_claims"] = (
                    independent_count
                    >= self.config.minimum_known_correct)
                result["minimum_known_correct"] = (
                    self.config.minimum_known_correct)
                result["runtime_phase_cap"] = self.config.max_examples_for(
                    benchmark_id)
                result["runtime_selected_row_count"] = len(examples)
                phase_results[phase] = result
            if isolated_trajectory:
                phase_results["test"] = {
                    "status": "not_evaluated",
                    "phase": "test",
                    "known_correct": [],
                    "known_correct_count": 0,
                    "eligible_for_mechanistic_claims": False,
                    "test_evaluated": False,
                    "test_evaluation_count": 0,
                    "test_data_accessor_called": False,
                    "reason": (
                        "paired_operator_trajectory_v1_forbids_test_access"),
                }
            output[benchmark_id] = {
                "status": "ready",
                "track": benchmark_spec(benchmark_id).track,
                "phases": phase_results,
            }
        return {"status": "ready", "benchmarks": output}

    def _run_frozen_circuit_phase(self, phase: str) -> dict[str, Any]:
        """Run one isolated phase of the frozen 4,096-site IOI protocol."""
        if phase not in {"validation", "test"}:
            raise ValueError(f"unsupported frozen IOI evaluation phase={phase}")
        kind = f"frozen_circuit_{phase}"
        if self._scope(kind) != ("mib_ioi",):
            raise ValueError(
                f"frozen circuit {phase} is registered only for mib_ioi")
        if not self._frozen_circuit_phase_isolated:
            raise RuntimeError(
                f"frozen IOI {phase} did not enter split-isolated execution")

        row_count = (
            FROZEN_IOI_VALIDATION_ROW_COUNT
            if phase == "validation"
            else FROZEN_IOI_TEST_ROW_COUNT)
        paired_correct_count = (
            FROZEN_IOI_VALIDATION_PAIRED_CORRECT_COUNT
            if phase == "validation"
            else FROZEN_IOI_TEST_PAIRED_CORRECT_COUNT)

        frozen = load_frozen_ioi_circuit(self.shape)
        frozen.validate_runtime(
            shape=self.shape,
            target_id=str(self.ctx.model_info.get("target_id") or ""),
            model_version=self.model_version,
            checkpoint_step=int(self.ctx.checkpoint_step),
            checkpoint_identity=str(self.contract["checkpoint_identity"]),
            checkpoint_config_hash=str(
                self.ctx.model_info.get("checkpoint_config_hash") or ""),
            model_config_hash=str(self.contract["model_config_hash"]),
            benchmark_build_id=self.build.build_id,
            benchmark_manifest_hash=self.build.manifest_hash,
        )
        evaluation = frozen.evaluation
        bootstrap_samples = int(evaluation["bootstrap_samples"])
        permutation_samples = int(evaluation["permutation_samples"])
        alpha = float(evaluation["alpha"])
        if self.config.max_examples_for("mib_ioi") != row_count:
            raise ValueError(
                f"frozen IOI {phase} requires the exact "
                f"{row_count}-row phase")
        if (self.config.bootstrap_samples != bootstrap_samples
                or self.config.permutation_samples != permutation_samples
                or not np.isclose(
                    self.config.alpha, alpha, rtol=0.0, atol=0.0)):
            raise ValueError(
                "runtime resampling settings differ from the frozen IOI spec")

        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            "stage=behavioral_eligibility status=running")
        phase_rows = self._load_phase_examples("mib_ioi", phase)
        if len(phase_rows) != row_count:
            raise ValueError(
                f"frozen IOI canonical {phase} selection count drift: "
                f"expected={row_count} actual={len(phase_rows)}")
        behavior = evaluate_behavior(
            self.ctx, phase_rows,
            pad_token_id=int(self.tokenizer.pad_token_id))
        known_correct = [
            example for example, keep in zip(
                phase_rows, behavior["known_correct"])
            if keep
        ]
        if len(known_correct) != paired_correct_count:
            raise ValueError(
                f"frozen IOI paired-correct {phase} count drift: "
                f"expected={paired_correct_count} "
                f"actual={len(known_correct)}")
        eligible_example_ids_hash = canonical_hash([
            example.example_id for example in known_correct
        ])
        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            f"stage=behavioral_eligibility status=ready "
            f"paired_correct={len(known_correct)}")

        batch = prepare_frozen_circuit_evaluation(
            self.ctx, known_correct,
            tokenizer=self.tokenizer,
            pad_token_id=int(self.tokenizer.pad_token_id))
        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            "condition=intact status=running")
        intact = evaluate_frozen_circuit_condition(
            self.ctx, batch, shape=self.shape, condition="intact")
        intact_positive = np.asarray(
            intact["positive_log_probability"], dtype=np.float64)
        intact_negative = np.asarray(
            intact["negative_log_probability"], dtype=np.float64)
        intact_margin = intact_positive - intact_negative
        known_correct_mask = np.asarray(
            behavior["known_correct"], dtype=np.bool_)
        reference_positive = np.asarray(
            behavior["base_positive_logp"],
            dtype=np.float64)[known_correct_mask]
        reference_negative = np.asarray(
            behavior["base_negative_logp"],
            dtype=np.float64)[known_correct_mask]
        reference_margin = reference_positive - reference_negative
        log_probability_errors = np.concatenate((
            intact_positive - reference_positive,
            intact_negative - reference_negative,
        ))
        margin_errors = intact_margin - reference_margin
        prediction_agreement = (
            (intact_margin > 0.0) == (reference_margin > 0.0))
        intact_reference_prediction_agreement = bool(
            np.all(prediction_agreement))
        intact_summary = {
            "mean_margin": float(np.mean(intact_margin)),
            "mean_correct_log_probability": float(np.mean(intact_positive)),
            "mean_source_log_probability": float(np.mean(intact_negative)),
            "exact_accuracy": float(np.mean(intact_margin > 0.0)),
            "mean_unrelated_log_probability": float(np.mean(
                np.asarray(
                    intact["unrelated_mean_log_probability"],
                    dtype=np.float64))),
            "unrelated_token_count_minimum": int(
                intact["unrelated_token_count_minimum"]),
            "unrelated_token_count_maximum": int(
                intact["unrelated_token_count_maximum"]),
            "production_diagnostics_reference": {
                "mean_margin": float(np.mean(reference_margin)),
                "exact_accuracy": float(np.mean(reference_margin > 0.0)),
            },
            "production_reference_numeric_audit": {
                "prediction_sign_agreement_passed": (
                    intact_reference_prediction_agreement),
                "prediction_sign_agreement_fraction": float(
                    np.mean(prediction_agreement)),
                "max_absolute_log_probability_error": (
                    float(np.max(np.abs(log_probability_errors)))),
                "mean_absolute_log_probability_error": float(
                    np.mean(np.abs(log_probability_errors))),
                "max_absolute_margin_error": float(
                    np.max(np.abs(margin_errors))),
                "mean_absolute_margin_error": float(
                    np.mean(np.abs(margin_errors))),
                "formal_numeric_tolerance_in_frozen_spec": False,
                "used_as_statistical_threshold": False,
            },
        }
        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            f"condition=intact status=ready "
            f"margin={intact_summary['mean_margin']:.8f} "
            f"accuracy={intact_summary['exact_accuracy']:.8f} "
            "reference_sign_agreement="
            f"{float(np.mean(prediction_agreement)):.8f} "
            "reference_max_abs_logp_error="
            f"{float(np.max(np.abs(log_probability_errors))):.8f}")

        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            "condition=frozen_suppression status=running")
        suppressed = evaluate_frozen_circuit_condition(
            self.ctx, batch, shape=self.shape, condition="suppression",
            circuit=frozen.circuit)
        suppression_summary = summarize_condition(
            intact, suppressed,
            bootstrap_samples=bootstrap_samples,
            alpha=alpha,
            seed=self.config.seed + 10_000,
        )
        frozen_effects = condition_effect_vectors(intact, suppressed)
        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            f"condition=frozen_suppression status=ready "
            f"margin_drop={suppression_summary['mean_margin_drop']:.8f} "
            f"accuracy={suppression_summary['exact_accuracy']:.8f}")

        sampler = FrozenIOIControlSampler(frozen, self.shape)
        controls_output: dict[str, Any] = {}
        replicate_count = int(
            frozen.controls["replicate_count_per_control"])
        for family_index, control_name in enumerate(FROZEN_IOI_CONTROL_ORDER):
            self._print(
                f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
                f"control={control_name} replicate=0/{replicate_count} "
                "status=running")
            margin_drop_rows: list[np.ndarray] = []
            mean_margin_drop: list[float] = []
            exact_accuracy: list[float] = []
            prediction_flip: list[float] = []
            correct_logp_change: list[float] = []
            source_direction_change: list[float] = []
            unrelated_damage: list[float] = []
            identity_hashes: list[str] = []
            activation_mean_distance: list[float] = []
            activation_max_distance: list[float] = []
            activation_mean_bin_distance: list[float] = []
            activation_max_bin_distance: list[int] = []
            for replicate_index in range(replicate_count):
                control_circuit, audit = sampler.generate(
                    control_name, replicate_index)
                control_scores = evaluate_frozen_circuit_condition(
                    self.ctx, batch, shape=self.shape,
                    condition="suppression", circuit=control_circuit)
                effects = condition_effect_vectors(intact, control_scores)
                margin_drop_rows.append(effects["margin_drop"])
                mean_margin_drop.append(float(np.mean(
                    effects["margin_drop"])))
                exact_accuracy.append(float(np.mean(
                    effects["margin"] > 0.0)))
                prediction_flip.append(float(np.mean(
                    (effects["intact_margin"] > 0.0)
                    != (effects["margin"] > 0.0))))
                correct_logp_change.append(float(np.mean(
                    effects["correct_log_probability_change"])))
                source_direction_change.append(float(np.mean(
                    effects["source_minus_correct_margin_change"])))
                unrelated_damage.append(float(np.mean(
                    effects["unrelated_log_probability_damage"])))
                identity_hashes.append(str(audit["site_identity_hash"]))
                if control_name == "activation_matched":
                    activation_mean_distance.append(float(audit[
                        "mean_absolute_empirical_quantile_distance"]))
                    activation_max_distance.append(float(audit[
                        "maximum_absolute_empirical_quantile_distance"]))
                    activation_mean_bin_distance.append(float(audit[
                        "mean_absolute_empirical_quantile_bin_distance"]))
                    activation_max_bin_distance.append(int(audit[
                        "maximum_absolute_empirical_quantile_bin_distance"]))
                if ((replicate_index + 1) % 10 == 0
                        or replicate_index + 1 == replicate_count):
                    self._print(
                        f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
                        f"control={control_name} "
                        f"replicate={replicate_index + 1}/"
                        f"{replicate_count} status=running")
                del control_scores, control_circuit
            control_matrix = np.stack(margin_drop_rows, axis=0)
            separation = compare_frozen_to_controls(
                frozen_effects["margin_drop"],
                control_matrix,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                alpha=alpha,
                seed=self.config.seed + 20_000 + family_index * 100,
            )
            config = dict(frozen.controls[control_name])
            controls_output[control_name] = {
                "replicate_count": replicate_count,
                "base_seed": int(config["seed"]),
                "sampling": config["sampling"],
                "control_algorithm_version": (
                    FROZEN_IOI_CONTROL_ALGORITHM_VERSION),
                "site_count_per_replicate": frozen.selected_k,
                "frozen_circuit_sites_excluded": True,
                "site_identity_hashes": identity_hashes,
                "mean_margin_drop_distribution": mean_margin_drop,
                "exact_accuracy_distribution": exact_accuracy,
                "prediction_flip_fraction_distribution": prediction_flip,
                "mean_correct_log_probability_change_distribution": (
                    correct_logp_change),
                "mean_source_direction_change_distribution": (
                    source_direction_change),
                "mean_unrelated_log_probability_damage_distribution": (
                    unrelated_damage),
                "frozen_minus_control": separation,
                **({
                    "mean_absolute_empirical_quantile_distance_distribution": (
                        activation_mean_distance),
                    "maximum_absolute_empirical_quantile_distance_distribution": (
                        activation_max_distance),
                    "mean_absolute_empirical_quantile_bin_distance_distribution": (
                        activation_mean_bin_distance),
                    "maximum_absolute_empirical_quantile_bin_distance_distribution": (
                        activation_max_bin_distance),
                    "empirical_quantile_bin_count": int(
                        frozen.controls["replicate_count_per_control"]),
                    "activation_match_definition": (
                        "within_exact_layer_and_route_nearest_equal_width_"
                        "empirical_midrank_quantile_bin_of_discovery_mean_"
                        "pre_scale_operator_output_norm_without_replacement;"
                        "bin_count_equals_frozen_control_replicate_count"),
                } if control_name == "activation_matched" else {}),
            }
            self._print(
                f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
                f"control={control_name} status=ready "
                f"mean_drop={float(np.mean(mean_margin_drop)):.8f} "
                "frozen_minus_control="
                f"{separation['mean_frozen_minus_control']:.8f}")
            del margin_drop_rows, control_matrix
            gc.collect()

        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            "condition=exact_restoration status=running")
        restored = evaluate_frozen_circuit_condition(
            self.ctx, batch, shape=self.shape, condition="restoration",
            circuit=frozen.circuit)
        restoration_summary = summarize_condition(
            intact, restored,
            bootstrap_samples=bootstrap_samples,
            alpha=alpha,
            seed=self.config.seed + 30_000,
        )
        restored_effects = condition_effect_vectors(intact, restored)
        recovery = bootstrap_restoration_recovery(
            intact_margin,
            frozen_effects["margin"],
            restored_effects["margin"],
            samples=bootstrap_samples,
            alpha=alpha,
            seed=self.config.seed + 31_000,
        )
        restoration_summary.update({
            "restoration_recovery_fraction": recovery[
                "recovery_fraction"],
            "restoration_recovery_bootstrap_ci": recovery,
            "restoration_mode": (
                "exact_selected_numerator_restore_after_suppression"),
            "restored_values_source": "same_example_intact_execution",
        })
        self._print(
            f"TRAIN_ANALYSIS_POOL frozen_ioi phase={phase} "
            f"condition=exact_restoration status=ready "
            "recovery="
            f"{restoration_summary['restoration_recovery_fraction']}")

        suppression_ci_low = suppression_summary[
            "margin_drop_bootstrap_ci"]["ci_low"]
        suppression_gate = (
            suppression_ci_low is not None
            and float(suppression_ci_low) > 0.0)
        control_gates = {
            control_name: (
                row["frozen_minus_control"]["bootstrap_ci"]["ci_low"]
                is not None
                and float(row["frozen_minus_control"][
                    "bootstrap_ci"]["ci_low"]) > 0.0
            )
            for control_name, row in controls_output.items()
        }
        controls_gate = all(control_gates.values())
        recovery_ci_low = recovery["ci_low"]
        recovery_minimum = float(evaluation[
            "validation_gates"][
                "restoration_recovery_ci_low_minimum"])
        restoration_gate = (
            recovery_ci_low is not None
            and float(recovery_ci_low) >= recovery_minimum)
        if (suppression_gate and controls_gate and restoration_gate
                and intact_reference_prediction_agreement):
            decision = "strong_success"
            strongest_claim = (
                "validation_level_causal_ioi_qk_centered_rw_operator_circuit"
                if phase == "validation"
                else "held_out_test_confirmed_causal_ioi_qk_centered_"
                "rw_operator_circuit")
        elif (suppression_gate and not controls_gate
                and intact_reference_prediction_agreement):
            decision = (
                "partial_success_general_attention_damage_not_excluded")
            strongest_claim = (
                f"{phase}_level_frozen_suppression_without_control_separation")
        elif (not suppression_gate
                and not controls_gate
                and not restoration_gate):
            decision = f"failure_not_a_{phase}_causal_circuit"
            strongest_claim = None
        else:
            decision = "mixed_inconclusive"
            strongest_claim = None

        correct_logp_damage = -float(
            suppression_summary["mean_correct_log_probability_change"])
        unrelated_damage = float(
            suppression_summary[
                "mean_unrelated_log_probability_damage"])
        unrelated_damage_ratio = (
            abs(unrelated_damage) / max(abs(correct_logp_damage), 1.0e-12))
        result = {
            "status": "ready",
            "phase": phase,
            "decision": decision,
            "strongest_supported_claim": strongest_claim,
            "record_status": "complete",
            f"{phase}_record_status": "complete",
            **({
                "prior_validation_record": {
                    "status": "final_before_test_access",
                    "analysis_commit": (
                        FROZEN_IOI_VALIDATION_ANALYSIS_COMMIT),
                    "canonical_record_commit": (
                        FROZEN_IOI_VALIDATION_RECORD_COMMIT),
                    "result_root": FROZEN_IOI_VALIDATION_RESULT_ROOT,
                    "validation_artifact_sha256": (
                        FROZEN_IOI_VALIDATION_RESULT_SHA256),
                    "configuration_changed_after_validation": False,
                },
            } if phase == "test" else {}),
            "frozen_specification": {
                "path": frozen.spec_path,
                "content_hash": frozen.spec_content_hash,
                "status": frozen.spec["status"],
                "selected_k": frozen.selected_k,
                "selected_route_counts": dict(
                    frozen.spec["selection"]["selected_route_counts"]),
                "selected_site_identity_hash": (
                    frozen.selected_site_identity_hash),
                "selected_ranked_rows_hash": frozen.spec[
                    "selection"]["selected_ranked_rows_hash"],
                "ranked_sites_content_hash": frozen.spec[
                    "discovery"]["ranked_sites_content_hash"],
                "localization_artifact": frozen.localization_path,
                "localization_protocol_hash": (
                    frozen.localization_protocol_hash),
                "selection_recomputed": False,
                "operator_ids_changed": False,
                "layers_or_routes_changed": False,
            },
            f"{phase}_cohort": {
                "runtime_selected_row_count": len(phase_rows),
                "selection_rule": (
                    "canonical_hash_example_id_then_example_id_first_128"),
                "paired_correct_independent_count": len(known_correct),
                "paired_correct_example_ids_hash": (
                    eligible_example_ids_hash),
                "base_accuracy_all_rows": behavior["accuracy"],
                "source_accuracy_all_rows": behavior["source_accuracy"],
                "pair_accuracy_all_rows": behavior["pair_accuracy"],
                "candidate_score": (
                    "positive_minus_negative_sum_log_probability"),
            },
            "primary_metrics": {
                "intact_mean_margin": intact_summary["mean_margin"],
                "intact_accuracy": intact_summary["exact_accuracy"],
                "production_reference_mean_margin": intact_summary[
                    "production_diagnostics_reference"]["mean_margin"],
                "production_reference_accuracy": intact_summary[
                    "production_diagnostics_reference"]["exact_accuracy"],
                "suppressed_mean_margin": suppression_summary["mean_margin"],
                "suppressed_accuracy": suppression_summary["exact_accuracy"],
                "mean_margin_drop": suppression_summary[
                    "mean_margin_drop"],
                "margin_drop_bootstrap_ci": suppression_summary[
                    "margin_drop_bootstrap_ci"],
                "mean_correct_log_probability_change": suppression_summary[
                    "mean_correct_log_probability_change"],
                "prediction_flip_fraction": suppression_summary[
                    "prediction_flip_fraction"],
                "mean_source_minus_correct_margin_change": (
                    suppression_summary[
                        "mean_source_minus_correct_margin_change"]),
                "mean_unrelated_log_probability_damage": (
                    suppression_summary[
                        "mean_unrelated_log_probability_damage"]),
                "control_margin_drop_distributions": {
                    name: row["mean_margin_drop_distribution"]
                    for name, row in controls_output.items()
                },
                "discovered_minus_control_paired_effects": {
                    name: row["frozen_minus_control"]
                    for name, row in controls_output.items()
                },
                "restored_mean_margin": restoration_summary["mean_margin"],
                "restored_accuracy": restoration_summary["exact_accuracy"],
                "restoration_recovery_fraction": recovery[
                    "recovery_fraction"],
                "restoration_recovery_bootstrap_ci": recovery,
            },
            "conditions": {
                "intact": intact_summary,
                "frozen_circuit_suppression": suppression_summary,
                "matched_controls": controls_output,
                "frozen_circuit_exact_restoration": restoration_summary,
            },
            f"{phase}_gates": {
                "suppression_margin_drop_ci_low_above_zero": {
                    "passed": suppression_gate,
                    "observed_ci_low": suppression_ci_low,
                    "threshold": 0.0,
                },
                "discovered_drop_exceeds_each_control": {
                    "passed": controls_gate,
                    "per_control": control_gates,
                    "rule": (
                        "each_frozen_minus_control_bootstrap_ci_low_above_zero"),
                },
                "restoration_recovery_ci_low_minimum": {
                    "passed": restoration_gate,
                    "observed_ci_low": recovery_ci_low,
                    "threshold": recovery_minimum,
                },
                "all_preregistered_gates_passed": bool(
                    suppression_gate and controls_gate and restoration_gate),
                "execution_reference_prediction_sign_agreement": {
                    "passed": intact_reference_prediction_agreement,
                    "rule": (
                        f"all_{paired_correct_count}_intervention_graph_"
                        "intact_margin_signs_match_production_diagnostics"),
                    "preregistered_statistical_gate": False,
                },
                "threshold_source": (
                    "frozen_spec.evaluation.validation_gates"),
            },
            "unrelated_behavior_audit": {
                "mean_log_probability_damage": unrelated_damage,
                "correct_log_probability_damage": correct_logp_damage,
                "absolute_damage_ratio": unrelated_damage_ratio,
                "formal_gate_defined_in_frozen_spec": False,
                "used_for_preregistered_gate": False,
            },
            "split_isolation": {
                "selection_phase": "discovery",
                "evaluation_phase": phase,
                "validation_used_to_change_specification": False,
                "configuration_changed_after_validation": False,
                "test_evaluated": phase == "test",
                "test_evaluation_count": (
                    len(known_correct) if phase == "test" else 0),
                "test_data_accessor_called": phase == "test",
                "held_out_test_opened": phase == "test",
                "held_out_test_allowed_after_this_record": (
                    phase == "validation"),
            },
            "storage_audit": {
                "status": "passed",
                "aggregate_statistics_and_hashes_preserved": True,
                "raw_per_example_logits_persisted": False,
                "raw_per_example_margins_persisted": False,
                "raw_per_example_operator_vectors_persisted": False,
                "control_site_ids_persisted": False,
                "control_site_identity_hashes_persisted": True,
            },
            "statistical_protocol": {
                "bootstrap_samples": bootstrap_samples,
                "permutation_samples": permutation_samples,
                "alpha": alpha,
                "seed": self.config.seed,
            },
        }
        return {
            "status": "ready",
            "benchmarks": {"mib_ioi": result},
            "strongest_supported_claim": strongest_claim,
            "test_evaluated": phase == "test",
            "test_data_accessor_called": phase == "test",
        }

    def _run_frozen_circuit_validation(self) -> dict[str, Any]:
        return self._run_frozen_circuit_phase("validation")

    def _run_frozen_circuit_test(self) -> dict[str, Any]:
        return self._run_frozen_circuit_phase("test")

    def _run_arc_frozen_validation(self) -> dict[str, Any]:
        """Run only the discovery-frozen ARC validation protocol."""
        if self._scope("arc_frozen_validation") != ("mib_arc",):
            raise ValueError(
                "ARC frozen validation is registered only for mib_arc")
        if not self._arc_frozen_validation_isolated:
            raise RuntimeError(
                "ARC frozen validation did not enter split-isolated execution")

        frozen = load_frozen_arc_circuit(self.shape)
        frozen.validate_runtime(
            shape=self.shape,
            target_id=str(self.ctx.model_info.get("target_id") or ""),
            model_version=self.model_version,
            checkpoint_step=int(self.ctx.checkpoint_step),
            checkpoint_identity=str(self.contract["checkpoint_identity"]),
            checkpoint_config_hash=str(
                self.ctx.model_info.get("checkpoint_config_hash") or ""),
            model_config_hash=str(self.contract["model_config_hash"]),
            benchmark_build_id=self.build.build_id,
            benchmark_manifest_hash=self.build.manifest_hash,
        )
        evaluation = frozen.evaluation
        bootstrap_samples = int(evaluation["bootstrap_samples"])
        permutation_samples = int(evaluation["permutation_samples"])
        alpha = float(evaluation["alpha"])
        if self.config.max_examples_for(
                "mib_arc") != FROZEN_ARC_VALIDATION_ROW_COUNT:
            raise ValueError(
                "frozen ARC validation requires the exact "
                f"{FROZEN_ARC_VALIDATION_ROW_COUNT}-row phase")
        if (self.config.bootstrap_samples != bootstrap_samples
                or self.config.permutation_samples != permutation_samples
                or not np.isclose(
                    self.config.alpha, alpha, rtol=0.0, atol=0.0)):
            raise ValueError(
                "runtime resampling settings differ from the frozen ARC spec")

        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "stage=behavioral_eligibility status=running")
        phase_rows = self._load_phase_examples("mib_arc", "validation")
        if len(phase_rows) != FROZEN_ARC_VALIDATION_ROW_COUNT:
            raise ValueError(
                "frozen ARC canonical validation selection count drift: "
                f"expected={FROZEN_ARC_VALIDATION_ROW_COUNT} "
                f"actual={len(phase_rows)}")
        behavior = evaluate_behavior(
            self.ctx, phase_rows,
            pad_token_id=int(self.tokenizer.pad_token_id))
        known_correct = [
            example for example, keep in zip(
                phase_rows, behavior["known_correct"])
            if keep
        ]
        if len(known_correct) != FROZEN_ARC_VALIDATION_PAIRED_CORRECT_COUNT:
            raise ValueError(
                "frozen ARC paired-correct validation count drift: "
                f"expected={FROZEN_ARC_VALIDATION_PAIRED_CORRECT_COUNT} "
                f"actual={len(known_correct)}")
        eligible_example_ids_hash = canonical_hash([
            example.example_id for example in known_correct
        ])
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "stage=behavioral_eligibility status=ready "
            f"paired_correct={len(known_correct)}")

        batch = prepare_arc_frozen_circuit_evaluation(
            self.ctx, known_correct,
            pad_token_id=int(self.tokenizer.pad_token_id))
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "condition=intact status=running")
        intact = evaluate_frozen_circuit_condition(
            self.ctx, batch, shape=self.shape, condition="intact")
        intact_positive = np.asarray(
            intact["positive_log_probability"], dtype=np.float64)
        intact_negative = np.asarray(
            intact["negative_log_probability"], dtype=np.float64)
        intact_margin = intact_positive - intact_negative
        known_correct_mask = np.asarray(
            behavior["known_correct"], dtype=np.bool_)
        reference_positive = np.asarray(
            behavior["base_positive_logp"],
            dtype=np.float64)[known_correct_mask]
        reference_negative = np.asarray(
            behavior["base_negative_logp"],
            dtype=np.float64)[known_correct_mask]
        reference_margin = reference_positive - reference_negative
        log_probability_errors = np.concatenate((
            intact_positive - reference_positive,
            intact_negative - reference_negative,
        ))
        margin_errors = intact_margin - reference_margin
        prediction_agreement = (
            (intact_margin > 0.0) == (reference_margin > 0.0))
        intact_reference_prediction_agreement = bool(
            np.all(prediction_agreement))
        intact_summary = {
            "mean_margin": float(np.mean(intact_margin)),
            "mean_correct_log_probability": float(np.mean(intact_positive)),
            "mean_source_log_probability": float(np.mean(intact_negative)),
            "exact_accuracy": float(np.mean(intact_margin > 0.0)),
            "mean_unrelated_log_probability": float(np.mean(
                np.asarray(
                    intact["unrelated_mean_log_probability"],
                    dtype=np.float64))),
            "unrelated_token_count_minimum": int(
                intact["unrelated_token_count_minimum"]),
            "unrelated_token_count_maximum": int(
                intact["unrelated_token_count_maximum"]),
            "unrelated_behavior_definition": (
                "mean_next_token_log_probability_on_entire_base_prompt_"
                "excluding_first_token"),
            "production_diagnostics_reference": {
                "mean_margin": float(np.mean(reference_margin)),
                "exact_accuracy": float(np.mean(reference_margin > 0.0)),
            },
            "production_reference_numeric_audit": {
                "prediction_sign_agreement_passed": (
                    intact_reference_prediction_agreement),
                "prediction_sign_agreement_fraction": float(
                    np.mean(prediction_agreement)),
                "max_absolute_log_probability_error": float(
                    np.max(np.abs(log_probability_errors))),
                "mean_absolute_log_probability_error": float(
                    np.mean(np.abs(log_probability_errors))),
                "max_absolute_margin_error": float(
                    np.max(np.abs(margin_errors))),
                "mean_absolute_margin_error": float(
                    np.mean(np.abs(margin_errors))),
                "formal_numeric_tolerance_in_frozen_spec": False,
                "used_as_statistical_threshold": False,
            },
        }
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "condition=intact status=ready "
            f"margin={intact_summary['mean_margin']:.8f} "
            f"accuracy={intact_summary['exact_accuracy']:.8f} "
            "reference_sign_agreement="
            f"{float(np.mean(prediction_agreement)):.8f} "
            "reference_max_abs_logp_error="
            f"{float(np.max(np.abs(log_probability_errors))):.8f}")

        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "condition=frozen_suppression status=running")
        suppressed = evaluate_frozen_circuit_condition(
            self.ctx, batch, shape=self.shape, condition="suppression",
            circuit=frozen.circuit)
        suppression_summary = summarize_condition(
            intact, suppressed,
            bootstrap_samples=bootstrap_samples,
            alpha=alpha,
            seed=self.config.seed + 40_000,
        )
        suppression_summary["unrelated_behavior_definition"] = (
            "mean_next_token_log_probability_on_entire_base_prompt_"
            "excluding_first_token")
        frozen_effects = condition_effect_vectors(intact, suppressed)
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "condition=frozen_suppression status=ready "
            f"margin_drop={suppression_summary['mean_margin_drop']:.8f} "
            f"accuracy={suppression_summary['exact_accuracy']:.8f}")

        sampler = FrozenARCControlSampler(frozen, self.shape)
        replicate_count = int(frozen.controls["replicate_count"])
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            f"control={FROZEN_ARC_CONTROL_NAME} "
            f"replicate=0/{replicate_count} status=running")
        margin_drop_rows: list[np.ndarray] = []
        mean_margin_drop: list[float] = []
        exact_accuracy: list[float] = []
        prediction_flip: list[float] = []
        correct_logp_change: list[float] = []
        source_direction_change: list[float] = []
        unrelated_damage: list[float] = []
        identity_hashes: list[str] = []
        for replicate_index in range(replicate_count):
            control_circuit, audit = sampler.generate(replicate_index)
            control_scores = evaluate_frozen_circuit_condition(
                self.ctx, batch, shape=self.shape,
                condition="suppression", circuit=control_circuit)
            effects = condition_effect_vectors(intact, control_scores)
            margin_drop_rows.append(effects["margin_drop"])
            mean_margin_drop.append(float(np.mean(
                effects["margin_drop"])))
            exact_accuracy.append(float(np.mean(
                effects["margin"] > 0.0)))
            prediction_flip.append(float(np.mean(
                (effects["intact_margin"] > 0.0)
                != (effects["margin"] > 0.0))))
            correct_logp_change.append(float(np.mean(
                effects["correct_log_probability_change"])))
            source_direction_change.append(float(np.mean(
                effects["source_minus_correct_margin_change"])))
            unrelated_damage.append(float(np.mean(
                effects["unrelated_log_probability_damage"])))
            identity_hashes.append(str(audit["site_identity_hash"]))
            if ((replicate_index + 1) % 10 == 0
                    or replicate_index + 1 == replicate_count):
                self._print(
                    "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
                    f"control={FROZEN_ARC_CONTROL_NAME} "
                    f"replicate={replicate_index + 1}/"
                    f"{replicate_count} status=running")
            del control_scores, control_circuit
        if len(set(identity_hashes)) != replicate_count:
            raise RuntimeError(
                "frozen ARC control site-set hashes are not unique")
        control_matrix = np.stack(margin_drop_rows, axis=0)
        separation = compare_frozen_to_controls(
            frozen_effects["margin_drop"],
            control_matrix,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            alpha=alpha,
            seed=self.config.seed + 50_000,
        )
        control_output = {
            "replicate_count": replicate_count,
            "base_seed": int(frozen.controls["seed"]),
            "match_fields": list(frozen.controls["match_fields"]),
            "sampling": frozen.controls["sampling"],
            "sampling_population": frozen.controls["sampling_population"],
            "control_algorithm_version": (
                FROZEN_ARC_CONTROL_ALGORITHM_VERSION),
            "rng_derivation": (
                "numpy_seed_sequence_base_seed_and_replicate"),
            "site_count_per_replicate": frozen.selected_k,
            "exact_frozen_count_matched_per_layer_route_cell": True,
            "frozen_circuit_sites_excluded": True,
            "duplicate_site_within_replicate_forbidden": True,
            "all_site_identity_hashes_unique": True,
            "site_identity_hashes": identity_hashes,
            "mean_margin_drop_distribution": mean_margin_drop,
            "exact_accuracy_distribution": exact_accuracy,
            "prediction_flip_fraction_distribution": prediction_flip,
            "mean_correct_log_probability_change_distribution": (
                correct_logp_change),
            "mean_source_direction_change_distribution": (
                source_direction_change),
            "mean_unrelated_log_probability_damage_distribution": (
                unrelated_damage),
            "frozen_minus_control": separation,
        }
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            f"control={FROZEN_ARC_CONTROL_NAME} status=ready "
            f"mean_drop={float(np.mean(mean_margin_drop)):.8f} "
            "frozen_minus_control="
            f"{separation['mean_frozen_minus_control']:.8f}")
        del margin_drop_rows, control_matrix
        gc.collect()

        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "condition=exact_restoration status=running")
        restored = evaluate_frozen_circuit_condition(
            self.ctx, batch, shape=self.shape, condition="restoration",
            circuit=frozen.circuit)
        restoration_summary = summarize_condition(
            intact, restored,
            bootstrap_samples=bootstrap_samples,
            alpha=alpha,
            seed=self.config.seed + 60_000,
        )
        restoration_summary["unrelated_behavior_definition"] = (
            "mean_next_token_log_probability_on_entire_base_prompt_"
            "excluding_first_token")
        restored_effects = condition_effect_vectors(intact, restored)
        recovery = bootstrap_restoration_recovery(
            intact_margin,
            frozen_effects["margin"],
            restored_effects["margin"],
            samples=bootstrap_samples,
            alpha=alpha,
            seed=self.config.seed + 61_000,
        )
        restoration_summary.update({
            "restoration_recovery_fraction": recovery[
                "recovery_fraction"],
            "restoration_recovery_bootstrap_ci": recovery,
            "restoration_mode": (
                "exact_selected_numerator_restore_after_suppression"),
            "restored_values_source": "same_example_intact_execution",
        })
        self._print(
            "TRAIN_ANALYSIS_POOL frozen_arc phase=validation "
            "condition=exact_restoration status=ready "
            "recovery="
            f"{restoration_summary['restoration_recovery_fraction']}")

        suppression_ci_low = suppression_summary[
            "margin_drop_bootstrap_ci"]["ci_low"]
        suppression_gate = (
            suppression_ci_low is not None
            and float(suppression_ci_low) > 0.0)
        control_ci_low = separation["bootstrap_ci"]["ci_low"]
        control_gate = (
            control_ci_low is not None
            and float(control_ci_low) > 0.0)
        recovery_ci_low = recovery["ci_low"]
        recovery_minimum = float(evaluation[
            "validation_gates"][
                "restoration_recovery_ci_low_minimum"])
        restoration_gate = (
            recovery_ci_low is not None
            and float(recovery_ci_low) >= recovery_minimum)
        if (suppression_gate and control_gate and restoration_gate
                and intact_reference_prediction_agreement):
            decision = "strong_success"
            strongest_claim = (
                "validation_level_auxiliary_causal_arc_rw_operator_circuit")
        elif (suppression_gate and not control_gate
                and intact_reference_prediction_agreement):
            decision = (
                "partial_success_general_operator_damage_not_excluded")
            strongest_claim = (
                "validation_level_arc_frozen_suppression_without_"
                "matched_random_separation")
        elif (not suppression_gate
                and not control_gate
                and not restoration_gate):
            decision = "failure_not_a_validation_causal_arc_circuit"
            strongest_claim = None
        else:
            decision = "mixed_inconclusive"
            strongest_claim = None

        correct_logp_damage = -float(
            suppression_summary["mean_correct_log_probability_change"])
        unrelated_logp_damage = float(
            suppression_summary[
                "mean_unrelated_log_probability_damage"])
        unrelated_damage_ratio = (
            abs(unrelated_logp_damage)
            / max(abs(correct_logp_damage), 1.0e-12))
        result = {
            "status": "ready",
            "benchmark": "mib_arc",
            "phase": "validation",
            "decision": decision,
            "strongest_supported_claim": strongest_claim,
            "record_status": "complete",
            "validation_record_status": "complete",
            "frozen_specification": {
                "path": frozen.spec_path,
                "content_hash": frozen.spec_content_hash,
                "status": frozen.spec["status"],
                "selected_k": frozen.selected_k,
                "selected_route_counts": dict(
                    frozen.spec["selection"]["selected_route_counts"]),
                "selected_layer_route_counts": {
                    str(layer): dict(counts)
                    for layer, counts in (
                        frozen.selected_layer_route_counts.items())
                },
                "selected_site_identity_hash": (
                    frozen.selected_site_identity_hash),
                "selected_ranked_rows_hash": frozen.spec[
                    "selection"]["selected_ranked_rows_hash"],
                "ranked_sites_content_hash": frozen.spec[
                    "discovery"]["ranked_sites_content_hash"],
                "frozen_selection_circuit_hash": frozen.spec[
                    "selection"]["circuit_hash"],
                "runtime_operator_circuit_hash": (
                    frozen.circuit.circuit_hash),
                "localization_artifact": frozen.localization_path,
                "localization_protocol_hash": (
                    frozen.localization_protocol_hash),
                "selection_recomputed": False,
                "operator_ids_changed": False,
                "layers_or_routes_changed": False,
            },
            "validation_cohort": {
                "runtime_selected_row_count": len(phase_rows),
                "selection_rule": (
                    "canonical_hash_example_id_then_example_id_first_128"),
                "paired_correct_independent_count": len(known_correct),
                "independent_unit": "example_id",
                "paired_correct_example_ids_hash": (
                    eligible_example_ids_hash),
                "base_accuracy_all_rows": behavior["accuracy"],
                "source_accuracy_all_rows": behavior["source_accuracy"],
                "pair_accuracy_all_rows": behavior["pair_accuracy"],
                "candidate_score": (
                    "correct_minus_source_sum_log_probability"),
            },
            "primary_metrics": {
                "intact_mean_margin": intact_summary["mean_margin"],
                "intact_accuracy": intact_summary["exact_accuracy"],
                "production_reference_mean_margin": intact_summary[
                    "production_diagnostics_reference"]["mean_margin"],
                "production_reference_accuracy": intact_summary[
                    "production_diagnostics_reference"]["exact_accuracy"],
                "suppressed_mean_margin": suppression_summary["mean_margin"],
                "suppressed_accuracy": suppression_summary["exact_accuracy"],
                "mean_margin_drop": suppression_summary[
                    "mean_margin_drop"],
                "margin_drop_bootstrap_ci": suppression_summary[
                    "margin_drop_bootstrap_ci"],
                "mean_correct_log_probability_change": suppression_summary[
                    "mean_correct_log_probability_change"],
                "prediction_flip_fraction": suppression_summary[
                    "prediction_flip_fraction"],
                "mean_source_minus_correct_margin_change": (
                    suppression_summary[
                        "mean_source_minus_correct_margin_change"]),
                "mean_unrelated_log_probability_damage": (
                    suppression_summary[
                        "mean_unrelated_log_probability_damage"]),
                "matched_random_margin_drop_distribution": (
                    control_output["mean_margin_drop_distribution"]),
                "frozen_minus_matched_random_paired_effect": (
                    control_output["frozen_minus_control"]),
                "restored_mean_margin": restoration_summary["mean_margin"],
                "restored_accuracy": restoration_summary["exact_accuracy"],
                "restoration_recovery_fraction": recovery[
                    "recovery_fraction"],
                "restoration_recovery_bootstrap_ci": recovery,
            },
            "conditions": {
                "intact": intact_summary,
                "frozen_circuit_suppression": suppression_summary,
                FROZEN_ARC_CONTROL_NAME: control_output,
                "frozen_circuit_exact_restoration": restoration_summary,
            },
            "validation_gates": {
                "suppression_margin_drop_ci_low_above_zero": {
                    "passed": suppression_gate,
                    "observed_ci_low": suppression_ci_low,
                    "threshold": 0.0,
                },
                "frozen_minus_matched_random_margin_drop_ci_low_above_zero": {
                    "passed": control_gate,
                    "observed_ci_low": control_ci_low,
                    "threshold": 0.0,
                },
                "restoration_recovery_ci_low_minimum": {
                    "passed": restoration_gate,
                    "observed_ci_low": recovery_ci_low,
                    "threshold": recovery_minimum,
                },
                "all_preregistered_gates_passed": bool(
                    suppression_gate and control_gate and restoration_gate),
                "execution_reference_prediction_sign_agreement": {
                    "passed": intact_reference_prediction_agreement,
                    "rule": (
                        f"all_{FROZEN_ARC_VALIDATION_PAIRED_CORRECT_COUNT}_"
                        "intervention_graph_intact_margin_signs_match_"
                        "production_diagnostics"),
                    "preregistered_statistical_gate": False,
                },
                "threshold_source": (
                    "frozen_spec.evaluation.validation_gates"),
            },
            "unrelated_behavior_audit": {
                "definition": (
                    "mean_next_token_log_probability_on_entire_base_prompt_"
                    "excluding_first_token"),
                "mean_log_probability_damage": unrelated_logp_damage,
                "correct_log_probability_damage": correct_logp_damage,
                "absolute_damage_ratio": unrelated_damage_ratio,
                "formal_gate_defined_in_frozen_spec": False,
                "used_for_preregistered_gate": False,
            },
            "split_isolation": {
                "selection_phase": "discovery",
                "evaluation_phase": "validation",
                "validation_used_to_change_specification": False,
                "configuration_changed_after_discovery": False,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "test_data_accessor_called": False,
                "held_out_test_opened": False,
                "held_out_test_allowed_after_this_record_is_final": True,
            },
            "storage_audit": {
                "status": "passed",
                "aggregate_statistics_and_hashes_preserved": True,
                "raw_per_example_logits_persisted": False,
                "raw_per_example_margins_persisted": False,
                "raw_per_example_operator_vectors_persisted": False,
                "control_site_ids_persisted": False,
                "control_site_identity_hashes_persisted": True,
            },
            "statistical_protocol": {
                "bootstrap_samples": bootstrap_samples,
                "permutation_samples": permutation_samples,
                "alpha": alpha,
                "runtime_seed": self.config.seed,
                "control_seed": int(frozen.controls["seed"]),
            },
            "test_evaluated": False,
            "test_evaluation_count": 0,
            "test_data_accessor_called": False,
            "test_used": False,
        }
        return {
            "status": "ready",
            "benchmarks": {"mib_arc": result},
            "strongest_supported_claim": strongest_claim,
            "test_evaluated": False,
            "test_data_accessor_called": False,
        }

    def _capture_kwargs(self, benchmark_id: str) -> dict[str, Any]:
        return {
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "topk_qk": self.config.capture_topk_qk,
            "topk_v": self.config.capture_topk_v,
            "topk_rst": self.config.capture_topk_rst,
            "max_topk_qk": self.config.capture_max_topk_qk,
            "max_topk_v": self.config.capture_max_topk_v,
            "max_topk_rst": self.config.capture_max_topk_rst,
            "capture_threshold": self.config.capture_threshold,
            "max_examples": self.config.max_examples_for(benchmark_id),
        }

    def _program_capture_kwargs(self) -> dict[str, Any]:
        values = self._capture_kwargs("mib_ioi")
        values["capture_threshold"] = max(
            float(value) for value in self.config.program_mass_candidates)
        return values

    @staticmethod
    def _compact_program_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(capture)
        rows = list(output.pop("rows", ()))
        ranked_sites = list(output.pop("ranked_sites", ()))
        output.pop("candidate_count", None)
        output["raw_capture_row_count"] = len(rows)
        output["raw_capture_rows_persisted_in_item_json"] = False
        output["derived_ranked_site_count"] = int(
            output.get("derived_ranked_site_count", len(ranked_sites)))
        output["derived_ranked_sites_persisted_in_item_json"] = False
        output["raw_capture_retention"] = (
            "aggregate_statistics_and_capture_digest_only")
        return output

    def _write_program_artifacts(
            self, *, phase: str, program_mass: float,
            schedules: Mapping[str, Any]) -> dict[str, Any]:
        records = {}
        for name, schedule in schedules.items():
            schedule.validate(self.shape)
            if not np.isclose(
                    float(schedule.program_mass), float(program_mass),
                    rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    "operator program schedule mass does not match the "
                    "artifact record")
            if self.ctx.is_primary:
                records[str(name)] = {
                    "program_algorithm_version": PROGRAM_ALGORITHM_VERSION,
                    "schedule_hash": schedule.schedule_hash,
                    "program_mass": float(schedule.program_mass),
                    "prompt_side": str(schedule.prompt_side),
                    "example_count": int(schedule.batch_size),
                    "widths": dict(schedule.widths),
                    "phase": str(phase),
                    "artifact_persisted": False,
                    "raw_ids_embedded_in_item_json": False,
                    "raw_ids_persisted": False,
                    "storage_policy": (
                        "aggregate_metadata_and_schedule_hash_only"),
                }
        return records

    def _write_program_effect_artifact(
            self, *, phase: str, program_mass: float,
            vectors: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.ctx.is_primary:
            return None
        digest = hashlib.sha256()
        digest.update(canonical_hash(self.protocol).encode("ascii"))
        digest.update(PROGRAM_ALGORITHM_VERSION.encode("ascii"))
        digest.update(str(phase).encode("utf-8"))
        digest.update(np.asarray([program_mass], dtype="<f8").tobytes())
        vector_metadata = {}
        for source_key in sorted(vectors, key=lambda value: str(value)):
            key = str(source_key)
            array = np.asarray(vectors[source_key])
            encoded_key = key.encode("utf-8")
            digest.update(len(encoded_key).to_bytes(4, "little"))
            digest.update(encoded_key)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            if array.dtype.hasobject or array.dtype.kind in {"U", "S"}:
                digest.update(canonical_hash(array.tolist()).encode("ascii"))
            else:
                digest.update(np.ascontiguousarray(array).tobytes())
            finite_count = None
            if np.issubdtype(array.dtype, np.number):
                finite_count = int(np.isfinite(array).sum())
            vector_metadata[key] = {
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "element_count": int(array.size),
                "finite_count": finite_count,
            }
        return {
            "vector_digest": digest.hexdigest(),
            "program_mass": float(program_mass),
            "phase": str(phase),
            "vector_names": sorted(str(key) for key in vectors),
            "vector_metadata": vector_metadata,
            "artifact_persisted": False,
            "per_example_primary_effects_persisted": False,
            "primary_effects_embedded_in_item_json": False,
            "storage_policy": (
                "aggregate_metadata_and_in_memory_content_digest_only"),
        }

    def _capture_program_phase(
            self, examples: Sequence[Any], *, phase: str,
            seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
        base = capture_program_paths(
            self.ctx, examples, phase=phase, prompt_side="base",
            seed=seed, **self._program_capture_kwargs())
        source = capture_program_paths(
            self.ctx, examples, phase=phase, prompt_side="source",
            seed=seed + 1, **self._program_capture_kwargs())
        widths = capture_schedule_widths((base, source))
        for capture in (base, source):
            ranked_sites = capture.pop("ranked_sites", ())
            capture["derived_ranked_site_count"] = int(
                capture.pop("candidate_count", len(ranked_sites)))
            capture["derived_ranked_sites_persisted_in_item_json"] = False
        return base, source, widths

    def _run_native_operator_program(self) -> dict[str, Any]:
        if self._scope("native_operator_program") != ("mib_ioi",):
            raise ValueError(
                "native operator program is registered only for mib_ioi")
        candidate_masses = tuple(
            float(value) for value in self.config.program_mass_candidates)
        minimum = int(self.config.minimum_known_correct)

        discovery_examples = self._known_correct("mib_ioi", "discovery")
        validation_examples = self._known_correct("mib_ioi", "validation")
        if len(discovery_examples) < minimum or len(validation_examples) < minimum:
            return {
                "status": "insufficient_behavior",
                "passed": False,
                "strongest_supported_claim": None,
                "discovery_known_correct": len(discovery_examples),
                "validation_known_correct": len(validation_examples),
                "minimum_known_correct": minimum,
                "test_evaluated": False,
            }

        discovery_base_capture, discovery_source_capture, discovery_widths = (
            self._capture_program_phase(
                discovery_examples, phase="discovery",
                seed=self.config.seed + 10001))
        discovery_curve = []
        discovery_artifacts = {}
        for mass_index, program_mass in enumerate(candidate_masses):
            base_schedule = build_program_schedule(
                discovery_base_capture, discovery_examples,
                shape=self.shape, program_mass=program_mass,
                prompt_side="base", widths=discovery_widths)
            source_schedule = build_program_schedule(
                discovery_source_capture, discovery_examples,
                shape=self.shape, program_mass=program_mass,
                prompt_side="source", widths=discovery_widths)
            mismatch_base_mapping = deterministic_mismatch_mapping(
                discovery_examples, base_schedule,
                seed=self.config.seed + 11003 + mass_index)
            mismatch_base_schedule = reindex_program_schedule(
                base_schedule, mismatch_base_mapping["donor_indices"],
                recipient_example_ids=[
                    example.example_id for example in discovery_examples],
                prompt_side="mismatched_base", shape=self.shape)
            compactness = compactness_metrics(
                base_schedule, shape=self.shape,
                paired_schedule=source_schedule,
                mismatched_schedule=mismatch_base_schedule)
            artifacts = self._write_program_artifacts(
                phase="discovery", program_mass=program_mass,
                schedules={
                    "base": base_schedule,
                    "source": source_schedule,
                })
            discovery_artifacts[str(program_mass)] = artifacts
            discovery_curve.append({
                "program_mass": program_mass,
                "median_decision_position_site_fraction": compactness[
                    "median_decision_position_site_fraction"],
                "mean_decision_position_site_fraction": compactness[
                    "mean_decision_position_site_fraction"],
                "per_route_decision_position_site_fraction": compactness[
                    "per_route_decision_position_site_fraction"],
                "same_pair_route_overlap": compactness[
                    "same_pair_route_overlap"],
                "mismatched_route_overlap": compactness[
                    "mismatched_route_overlap"],
            })

        validation_base_capture, validation_source_capture, validation_widths = (
            self._capture_program_phase(
                validation_examples, phase="validation",
                seed=self.config.seed + 20011))
        validation_schedules = {}
        validation_artifacts = {}
        validation_seeds = {}
        for mass_index, program_mass in enumerate(candidate_masses):
            base_schedule = build_program_schedule(
                validation_base_capture, validation_examples,
                shape=self.shape, program_mass=program_mass,
                prompt_side="base", widths=validation_widths)
            source_schedule = build_program_schedule(
                validation_source_capture, validation_examples,
                shape=self.shape, program_mass=program_mass,
                prompt_side="source", widths=validation_widths)
            validation_schedules[program_mass] = (
                base_schedule, source_schedule)
            validation_seeds[program_mass] = (
                self.config.seed + 21013 + mass_index * 101)
            validation_artifacts[str(program_mass)] = (
                self._write_program_artifacts(
                    phase="validation", program_mass=program_mass,
                    schedules={
                        "base": base_schedule,
                        "source": source_schedule,
                    }))

        baseline_base, baseline_source = validation_schedules[
            candidate_masses[0]]
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=validation "
            "stage=phase_baselines status=running")
        validation_baselines = evaluate_native_operator_program_phase_baselines(
            self.ctx, validation_examples,
            base_schedule=baseline_base,
            source_schedule=baseline_source,
            shape=self.shape,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=validation "
            "stage=phase_baselines status=ready")

        validation_selection_candidates = []
        validation_replay_margins = {}
        for program_mass in candidate_masses:
            base_schedule, source_schedule = validation_schedules[program_mass]
            self._print(
                "TRAIN_ANALYSIS_POOL native_program phase=validation "
                f"mass={program_mass:.2f} stage=replay status=running")
            candidate, replay_margin = (
                evaluate_native_operator_program_selection_candidate(
                    self.ctx, validation_examples,
                    base_schedule=base_schedule,
                    source_schedule=source_schedule,
                    baselines=validation_baselines,
                    shape=self.shape,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    config=self.config,
                    seed=validation_seeds[program_mass],
                ))
            candidate["schedule_artifacts"] = validation_artifacts[
                str(program_mass)]
            validation_selection_candidates.append(candidate)
            validation_replay_margins[program_mass] = replay_margin
            self._print(
                "TRAIN_ANALYSIS_POOL native_program phase=validation "
                f"mass={program_mass:.2f} stage=replay status=ready")

        selection = select_validation_program(
            validation_selection_candidates, config=self.config)
        selected_validation_diagnostics = None
        if selection["status"] == "selected":
            selected_mass = float(selection["selected_program_mass"])
            frozen_selection_hash = canonical_hash(selection)
            self._print(
                "TRAIN_ANALYSIS_POOL native_program phase=validation "
                f"mass={selected_mass:.2f} stage=selection status=frozen")
            selected_base, selected_source = validation_schedules[selected_mass]
            selected_candidate = next(
                candidate for candidate in validation_selection_candidates
                if float(candidate["program_mass"]) == selected_mass)
            self._print(
                "TRAIN_ANALYSIS_POOL native_program phase=validation "
                f"mass={selected_mass:.2f} stage=causal_diagnostics "
                "status=running")
            selected_validation_diagnostics, selected_controls = (
                evaluate_native_operator_program_causal_diagnostics(
                    self.ctx, validation_examples,
                    base_schedule=selected_base,
                    source_schedule=selected_source,
                    baselines=validation_baselines,
                    selection_candidate=selected_candidate,
                    replay_margin=validation_replay_margins[selected_mass],
                    shape=self.shape,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    config=self.config,
                    seed=validation_seeds[selected_mass],
                ))
            if canonical_hash(selection) != frozen_selection_hash:
                raise RuntimeError(
                    "validation causal diagnostics changed frozen selection")
            effect_vectors = selected_validation_diagnostics.pop(
                "_effect_vectors")
            effect_artifact = self._write_program_effect_artifact(
                phase="validation", program_mass=selected_mass,
                vectors=effect_vectors)
            if effect_artifact is not None:
                selected_validation_diagnostics[
                    "effect_artifact"] = effect_artifact
            selected_validation_diagnostics["control_schedule_artifacts"] = (
                self._write_program_artifacts(
                    phase="validation", program_mass=selected_mass,
                    schedules=selected_controls))
            diagnostic_checks = native_program_diagnostic_checks(
                selected_validation_diagnostics, config=self.config)
            selected_validation_diagnostics[
                "validation_diagnostic_checks"] = diagnostic_checks
            selected_validation_diagnostics[
                "selected_validation_diagnostics_hash"] = canonical_hash({
                    "program_mass": selected_mass,
                    "checks": diagnostic_checks,
                    "ablation": selected_validation_diagnostics["ablation"],
                    "source_id_replay": selected_validation_diagnostics[
                        "source_id_replay"],
                    "transplant": selected_validation_diagnostics[
                        "transplant"],
                })
            selected_validation_diagnostics["selection_record_hash"] = (
                selection["selection_record_hash"])
            self._print(
                "TRAIN_ANALYSIS_POOL native_program phase=validation "
                f"mass={selected_mass:.2f} stage=causal_diagnostics "
                "status=ready")
        else:
            self._print(
                "TRAIN_ANALYSIS_POOL native_program phase=validation "
                "stage=selection status=no_compact_program")
        common = {
            "program_algorithm_version": PROGRAM_ALGORITHM_VERSION,
            "program_mass_candidates": list(candidate_masses),
            "decision_scope": "final_ioi_decision_at_answer_position",
            "program_position_scope": self.config.program_position_scope,
            "program_routes": list(self.config.program_routes),
            "program_denominator_policy": (
                self.config.program_denominator_policy),
            "program_mismatch_matching": (
                self.config.program_mismatch_matching),
            "program_random_sampling": self.config.program_random_sampling,
            "execution_plan": {
                "phase_baselines_computed_once_per_phase": True,
                "validation_selection_evaluator": "replay_only",
                "validation_full_diagnostic_evaluations": (
                    1 if selection["status"] == "selected" else 0),
                "test_full_diagnostic_evaluation_limit": 1,
                "positive_negative_margin_fusion": False,
                "mass_replay_batching": False,
                "contribution_capture_batching": False,
            },
            "discovery": {
                "status": "ready",
                "example_count": len(discovery_examples),
                "mass_curve": discovery_curve,
                "base_capture": self._compact_program_capture(
                    discovery_base_capture),
                "source_capture": self._compact_program_capture(
                    discovery_source_capture),
                "schedule_artifacts": discovery_artifacts,
            },
            "validation": {
                "status": "ready",
                "example_count": len(validation_examples),
                "selection_candidates": validation_selection_candidates,
                "selection": selection,
                "selected_diagnostics": selected_validation_diagnostics,
                "base_capture": self._compact_program_capture(
                    validation_base_capture),
                "source_capture": self._compact_program_capture(
                    validation_source_capture),
            },
            "test_used_for_selection": False,
            "test_program_executable_called_before_selection": False,
        }
        if selection["status"] != "selected":
            return {
                **common,
                "status": "no_compact_validation_program",
                "ready": True,
                "passed": False,
                "selected_program_mass": None,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "strongest_supported_claim": "descriptive_decision_program",
                "checkpoint_specific_claim": True,
                "scientific_claims_primary_modified": False,
            }

        selected_mass = float(selection["selected_program_mass"])
        if selected_validation_diagnostics is None:
            raise RuntimeError(
                "selected validation mass has no causal diagnostics")
        # Test access stays unreachable until selection is frozen and the
        # selected validation diagnostics have completed.
        test_examples = self._known_correct("mib_ioi", "test")
        if len(test_examples) < minimum:
            return {
                **common,
                "status": "insufficient_test_behavior",
                "ready": True,
                "passed": False,
                "selected_program_mass": selected_mass,
                "test_known_correct": len(test_examples),
                "minimum_known_correct": minimum,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "strongest_supported_claim": "descriptive_decision_program",
                "checkpoint_specific_claim": True,
                "scientific_claims_primary_modified": False,
            }
        test_base_capture, test_source_capture, test_widths = (
            self._capture_program_phase(
                test_examples, phase="test",
                seed=self.config.seed + 30029))
        test_base_schedule = build_program_schedule(
            test_base_capture, test_examples,
            shape=self.shape, program_mass=selected_mass,
            prompt_side="base", widths=test_widths)
        test_source_schedule = build_program_schedule(
            test_source_capture, test_examples,
            shape=self.shape, program_mass=selected_mass,
            prompt_side="source", widths=test_widths)
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=test "
            f"mass={selected_mass:.2f} stage=phase_baselines status=running")
        test_baselines = evaluate_native_operator_program_phase_baselines(
            self.ctx, test_examples,
            base_schedule=test_base_schedule,
            source_schedule=test_source_schedule,
            shape=self.shape,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=test "
            f"mass={selected_mass:.2f} stage=phase_baselines status=ready")
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=test "
            f"mass={selected_mass:.2f} stage=replay status=running")
        test_selection_candidate, test_replay_margin = (
            evaluate_native_operator_program_selection_candidate(
                self.ctx, test_examples,
                base_schedule=test_base_schedule,
                source_schedule=test_source_schedule,
                baselines=test_baselines,
                shape=self.shape,
                pad_token_id=int(self.tokenizer.pad_token_id),
                config=self.config,
                seed=self.config.seed + 31033,
            ))
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=test "
            f"mass={selected_mass:.2f} stage=replay status=ready")
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=test "
            f"mass={selected_mass:.2f} stage=causal_diagnostics "
            "status=running")
        test_result, test_controls = (
            evaluate_native_operator_program_causal_diagnostics(
                self.ctx, test_examples,
                base_schedule=test_base_schedule,
                source_schedule=test_source_schedule,
                baselines=test_baselines,
                selection_candidate=test_selection_candidate,
                replay_margin=test_replay_margin,
                shape=self.shape,
                pad_token_id=int(self.tokenizer.pad_token_id),
                config=self.config,
                seed=self.config.seed + 31033,
            ))
        self._print(
            "TRAIN_ANALYSIS_POOL native_program phase=test "
            f"mass={selected_mass:.2f} stage=causal_diagnostics status=ready")
        test_effect_vectors = test_result.pop("_effect_vectors")
        test_effect_artifact = self._write_program_effect_artifact(
            phase="test", program_mass=selected_mass,
            vectors=test_effect_vectors)
        if test_effect_artifact is not None:
            test_result["effect_artifact"] = test_effect_artifact
        test_artifacts = self._write_program_artifacts(
            phase="test", program_mass=selected_mass,
            schedules={
                "base": test_base_schedule,
                "source": test_source_schedule,
                **test_controls,
            })
        claims = evaluate_native_program_claims(
            test_result, config=self.config)
        human_summary = {
            "selected_program_mass": selected_mass,
            "validation_selection_rule": selection["selection_rule"],
            "validation_causal_diagnostics_used_for_selection": selection[
                "causal_diagnostics_used_for_selection"],
            "decision_scope": "final_ioi_decision_at_answer_position",
            "median_decision_position_site_fraction": test_result[
                "compactness"]["median_decision_position_site_fraction"],
            "mean_decision_position_site_fraction": test_result[
                "compactness"]["mean_decision_position_site_fraction"],
            "per_route_decision_position_site_fraction": test_result[
                "compactness"][
                    "per_route_decision_position_site_fraction"],
            "replay_faithfulness": test_result["replay"][
                "normalized_faithfulness"],
            "replay_faithfulness_ci": test_result["replay"][
                "faithfulness_ci"],
            "replay_agreement": test_result["replay"][
                "answer_agreement_with_full"],
            "ablation_margin_drop": test_result["ablation"][
                "own_program"]["mean_margin_drop"],
            "ablation_margin_drop_ci": test_result["ablation"][
                "own_program"]["margin_drop_ci"],
            "ablation_permutation_p": test_result["ablation"][
                "own_program"]["permutation"]["p_value_two_sided"],
            "own_vs_mismatched_ablation": test_result["ablation"][
                "specificity"]["own_vs_mismatched"],
            "own_vs_random_ablation": test_result["ablation"][
                "specificity"]["own_vs_random"],
            "source_id_replay_paired_vs_mismatch": test_result[
                "source_id_replay"]["paired_vs_mismatch"],
            "source_id_replay_paired_vs_random": test_result[
                "source_id_replay"]["paired_vs_random"],
            "source_id_replay_bidirectional_flip": test_result[
                "source_id_replay"][
                    "bidirectional_answer_flip_fraction"],
            "source_contribution_transplant_flip": test_result[
                "transplant"]["paired"]["base_to_source"][
                    "answer_flip_fraction"],
            "mismatched_flip": test_result["transplant"][
                "mismatched"]["base_to_source"]["answer_flip_fraction"],
            "random_flip": test_result["transplant"][
                "random"]["base_to_source"]["answer_flip_fraction"],
            "paired_vs_mismatch_effect": test_result["transplant"][
                "paired_vs_mismatch"]["mean_effect"],
            "paired_vs_mismatch_effect_ci": test_result["transplant"][
                "paired_vs_mismatch"]["effect_ci"],
            "paired_vs_mismatch_permutation_p": test_result["transplant"][
                "paired_vs_mismatch"]["permutation"][
                    "p_value_two_sided"],
            "paired_vs_random_effect": test_result["transplant"][
                "paired_vs_random"]["mean_effect"],
            "paired_vs_random_effect_ci": test_result["transplant"][
                "paired_vs_random"]["effect_ci"],
            "paired_vs_random_permutation_p": test_result["transplant"][
                "paired_vs_random"]["permutation"][
                    "p_value_two_sided"],
            "bidirectional_success": test_result["transplant"][
                "bidirectional_answer_flip_fraction"],
            "strongest_supported_native_program_claim": claims[
                "strongest_supported_claim"],
        }
        return {
            **common,
            "status": "ready",
            "ready": True,
            "passed": claims["passed"],
            "selected_program_mass": selected_mass,
            "frozen_selection": selection,
            "test": {
                **test_result,
                "base_capture": self._compact_program_capture(
                    test_base_capture),
                "source_capture": self._compact_program_capture(
                    test_source_capture),
                "schedule_artifacts": test_artifacts,
            },
            "test_evaluated": True,
            "test_evaluation_count": 1,
            "claims": claims["claims"],
            "strongest_supported_claim": claims[
                "strongest_supported_claim"],
            "checkpoint_specific_claim": True,
            "scientific_claims_primary_modified": False,
            "human_summary": human_summary,
        }

    @staticmethod
    def _trajectory_without_private(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): OperatorInterpretabilityRunner.
                _trajectory_without_private(child)
                for key, child in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [
                OperatorInterpretabilityRunner._trajectory_without_private(
                    child) for child in value
            ]
        return value

    def _trajectory_progress(self, message: str) -> None:
        self._print(f"TRAIN_ANALYSIS_POOL paired_trajectory {message}")

    def _run_paired_operator_trajectory(self) -> dict[str, Any]:
        if self._scope("paired_operator_trajectory") != ("mib_ioi",):
            raise ValueError(
                "paired operator trajectory is registered only for mib_ioi")
        if self.config.trajectory_test_enabled:
            raise ValueError("paired trajectory v1 forbids test evaluation")
        protocol_hash = canonical_hash(self.protocol)
        pad_token_id = int(self.tokenizer.pad_token_id)
        discovery_known = self._known_correct("mib_ioi", "discovery")
        validation_known = self._known_correct("mib_ioi", "validation")
        if len(discovery_known) < 2 or len(validation_known) < 2:
            return {
                "status": "insufficient_behavior",
                "algorithm_version": "paired_s2_operator_trajectory_v1",
                "discovery_paired_correct_count": len(discovery_known),
                "validation_paired_correct_count": len(validation_known),
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "test_data_accessor_called": False,
                "test_used": False,
            }

        discovery = deterministic_deep_selection(
            discovery_known,
            limit=self.config.trajectory_discovery_examples,
            seed=self.config.trajectory_seed)
        validation = deterministic_deep_selection(
            validation_known,
            limit=self.config.trajectory_validation_examples,
            seed=self.config.trajectory_seed + 1)
        deep = deterministic_deep_selection(
            discovery,
            limit=self.config.trajectory_deep_examples,
            seed=self.config.trajectory_seed + 2)
        discovery_semantic = {
            example.example_id: ioi_semantic_record(example, self.tokenizer)
            for example in discovery
        }
        validation_semantic = {
            example.example_id: ioi_semantic_record(example, self.tokenizer)
            for example in validation
        }
        deep_semantic = {
            example.example_id: discovery_semantic[example.example_id]
            for example in deep
        }
        maximum_trace_positions = max(
            len(record.positions) for record in deep_semantic.values())
        maximum_trace_sequence_length = max(
            len(prompt) + len(answer)
            for example in deep
            for prompt, answer in (
                (example.input_ids_base, example.positive_ids),
                (example.input_ids_base, example.negative_ids),
                (example.input_ids_source, example.source_positive_ids),
                (example.input_ids_source, example.source_negative_ids),
            )
        )
        n_rst = int(
            self.ctx.model_cfg["n_rst"]
            if "n_rst" in self.ctx.model_cfg
            else self.ctx.model_cfg["n_know"])
        initial_widths = {
            "qk": min(
                self.config.trajectory_capture_topk_qk,
                int(self.ctx.model_cfg["n_qk"])),
            "v": min(
                self.config.trajectory_capture_topk_v,
                int(self.ctx.model_cfg["n_v"])),
            "rst": min(
                self.config.trajectory_capture_topk_rst, n_rst),
        }
        layer_count = int(self.ctx.model_cfg["n_layers"])
        model_width = int(self.ctx.model_cfg["d_model"])
        data_multiple = max(1, int(self.ctx.mesh.shape["data"]))
        streamed_batch_size = (
            (4 + data_multiple - 1) // data_multiple) * data_multiple
        trace_row_count = (
            layer_count * streamed_batch_size * maximum_trace_positions)
        operator_width_sum = (
            2 * initial_widths["qk"]
            + initial_widths["v"] + initial_widths["rst"])
        estimated_capture_trace_output_bytes = int(
            trace_row_count * (
                92 * model_width + 84 + 25 * operator_width_sum)
            + 5 * streamed_batch_size * maximum_trace_positions
            + 16 * layer_count)
        estimated_compact_replay_output_bytes = int(
            trace_row_count * 64 * model_width)
        estimated_score_residual_output_bytes_per_pass = int(
            streamed_batch_size
            * (maximum_trace_sequence_length * model_width + 1) * 4)
        requested_trace_shape = {
            "layers": layer_count,
            "deep_example_count": len(deep),
            "streamed_fused_candidate_rows_per_batch": 4,
            "streamed_padded_rows_per_batch": streamed_batch_size,
            "sequence_length": maximum_trace_sequence_length,
            "trace_positions": maximum_trace_positions,
            "initial_widths": initial_widths,
            "estimated_capture_trace_output_bytes_initial_width": (
                estimated_capture_trace_output_bytes),
            "estimated_compact_replay_output_bytes": (
                estimated_compact_replay_output_bytes),
            "estimated_score_residual_output_bytes_per_pass": (
                estimated_score_residual_output_bytes_per_pass),
            "estimated_peak_trace_outputs_bytes_initial_width": (
                estimated_capture_trace_output_bytes
                + estimated_compact_replay_output_bytes),
            "estimated_peak_materialized_output_bytes_initial_width": (
                estimated_capture_trace_output_bytes
                + estimated_compact_replay_output_bytes
                + 2 * estimated_score_residual_output_bytes_per_pass),
            "estimate_basis": (
                "materialized_array_shape_and_dtype_sum_for_semantic_"
                "position_states_full_active_operator_fields_and_compact_"
                "replay_route_outputs"),
        }
        stage = "production_atlas_discovery"
        trace_output_bytes = 0
        last_successful_stage = "pair_preparation"
        try:
            discovery_atlas = capture_production_atlas(
                self.ctx, discovery, discovery_semantic,
                pad_token_id=pad_token_id, config=self.config,
                phase="discovery", progress=self._trajectory_progress)
            discovery_atlas_artifact = write_atlas_metric_artifact(
                self.store, discovery_atlas, phase="discovery",
                protocol_hash=protocol_hash)
            last_successful_stage = stage
            stage = "production_atlas_validation"
            validation_atlas = capture_production_atlas(
                self.ctx, validation, validation_semantic,
                pad_token_id=pad_token_id, config=self.config,
                phase="validation", progress=self._trajectory_progress)
            validation_atlas_artifact = write_atlas_metric_artifact(
                self.store, validation_atlas, phase="validation",
                protocol_hash=protocol_hash)
            last_successful_stage = stage
            if (discovery_atlas["status"] != "ready"
                    or validation_atlas["status"] != "ready"):
                return {
                    "status": "s2_prefix_identity_failed",
                    "algorithm_version": (
                        "paired_s2_operator_trajectory_v1"),
                    "discovery": self._trajectory_without_private(
                        discovery_atlas),
                    "validation": self._trajectory_without_private(
                        validation_atlas),
                    "atlas_artifacts": {
                        "discovery": discovery_atlas_artifact,
                        "validation": validation_atlas_artifact,
                    },
                    "causal_intervention_executed": False,
                    "test_evaluated": False,
                    "test_evaluation_count": 0,
                    "test_data_accessor_called": False,
                    "test_used": False,
                }

            stage = "streamed_full_active_trace"
            operator_provenance, operator_keys = (
                operator_parameter_provenance(self.ctx))
            operator_provenance["checkpoint_identity"] = self.protocol[
                "checkpoint_identity"]
            trajectory_batches = []
            divergence = None
            deep_shards = []
            running_widths = dict(initial_widths)
            for deep_index, example in enumerate(deep):
                example_semantic = {
                    example.example_id: deep_semantic[example.example_id]}
                trajectory_batch = capture_full_active_trajectory(
                    self.ctx, [example], example_semantic,
                    pad_token_id=pad_token_id, config=self.config,
                    initial_widths=running_widths,
                    fixed_sequence_length=maximum_trace_sequence_length,
                    fixed_trace_width=maximum_trace_positions,
                    progress=self._trajectory_progress)
                trace_output_bytes = max(
                    trace_output_bytes,
                    int(trajectory_batch.get("trace_output_bytes", 0)))
                if trajectory_batch["status"] != "ready":
                    return {
                        "status": "full_active_replay_parity_failed",
                        "algorithm_version": (
                            "paired_s2_operator_trajectory_v1"),
                        "failed_example_id": example.example_id,
                        "requested_trace_shape": requested_trace_shape,
                        "final_capture_widths": trajectory_batch["widths"],
                        "capture_retries": trajectory_batch["retries"],
                        "trace_completeness": trajectory_batch[
                            "completeness"],
                        "full_active_replay": trajectory_batch["closure"],
                        "causal_intervention_executed": False,
                        "test_evaluated": False,
                        "test_evaluation_count": 0,
                        "test_data_accessor_called": False,
                        "test_used": False,
                    }
                running_widths = {
                    pool: max(
                        int(running_widths[pool]),
                        int(trajectory_batch["widths"][pool]))
                    for pool in ("qk", "v", "rst")
                }
                persisted = write_deep_trace_shards(
                    self.store, trajectory_batch, [example],
                    example_semantic, protocol_hash=protocol_hash,
                    shard_index_offset=deep_index,
                    write_intermediate_manifest=False)
                deep_shards.extend(persisted.get("shards", ()))
                example_divergence = build_divergence_atlas(
                    trajectory_batch, [example], example_semantic,
                    operator_keys=operator_keys,
                    epsilon=self.config.trajectory_divergence_epsilon)
                divergence = (
                    example_divergence if divergence is None
                    else merge_divergence_atlases(
                        (divergence, example_divergence)))
                trajectory_batches.append({
                    "example_id": example.example_id,
                    "initial_widths": trajectory_batch["initial_widths"],
                    "widths": trajectory_batch["widths"],
                    "retries": trajectory_batch["retries"],
                    "completeness": trajectory_batch["completeness"],
                    "closure": trajectory_batch["closure"],
                    "trace_output_bytes": trajectory_batch[
                        "trace_output_bytes"],
                    "replay_trace_output_bytes": trajectory_batch[
                        "replay_trace_output_bytes"],
                    "capture_score_residual_bytes": trajectory_batch[
                        "capture_score_residual_bytes"],
                    "replay_score_residual_bytes": trajectory_batch[
                        "replay_score_residual_bytes"],
                    "peak_materialized_output_bytes": trajectory_batch[
                        "peak_materialized_output_bytes"],
                    "forward_call_count": trajectory_batch[
                        "forward_call_count"],
                })
                del trajectory_batch, persisted, example_divergence
                gc.collect()
            trajectory = merge_trajectory_batch_summaries(
                trajectory_batches)
            if divergence is None:
                raise RuntimeError("streamed divergence atlas is empty")
            deep_trace_artifacts = {
                "status": "complete",
                "shards": deep_shards,
                "shard_count": 2 * len(deep),
                "streamed_trace_batch_count": len(deep),
            }
            extrema = divergence_extrema(divergence)
            last_successful_stage = stage
            del operator_keys, trajectory_batches
            gc.collect()

            stage = "discovery_candidate_freeze"
            selection = select_discovery_candidates(
                divergence, config=self.config)
            candidates = list(selection["candidates"])
            selection_hash_before_causal = str(
                selection["selection_record_hash"])
            last_successful_stage = stage

            stage = "discovery_candidate_value_capture"
            discovery_site_values = capture_candidate_site_values(
                self.ctx, discovery, discovery_semantic, candidates,
                pad_token_id=pad_token_id, phase="discovery",
                progress=self._trajectory_progress)
            discovery_mismatch = trajectory_mismatch_mapping(
                discovery, seed=self.config.trajectory_seed + 101)
            last_successful_stage = stage

            stage = "discovery_route_patch"
            discovery_route = evaluate_coarse_site_patches(
                self.ctx, discovery, discovery_semantic, candidates,
                discovery_site_values, discovery_mismatch,
                production_atlas=discovery_atlas,
                pad_token_id=pad_token_id, config=self.config,
                phase="discovery", patch_kinds=("route",),
                progress=self._trajectory_progress)
            if selection["selection_record_hash"] != (
                    selection_hash_before_causal):
                raise RuntimeError(
                    "causal patching changed the frozen candidate record")
            last_successful_stage = stage

            stage = "route_followup_and_path_freeze"
            operator_followup = freeze_operator_followup_sites(
                discovery_route, candidates, config=self.config)
            path_record = freeze_chronological_path(
                discovery_route, candidates, config=self.config)
            path_hash_before_validation = str(
                path_record["path_record_hash"])
            last_successful_stage = stage

            stage = "discovery_residual_patch"
            residual_candidates = deduplicate_residual_candidates(candidates)
            discovery_residual = evaluate_coarse_site_patches(
                self.ctx, discovery, discovery_semantic, residual_candidates,
                discovery_site_values, discovery_mismatch,
                production_atlas=discovery_atlas,
                pad_token_id=pad_token_id, config=self.config,
                phase="discovery", patch_kinds=("residual",),
                progress=self._trajectory_progress)
            discovery_coarse = merge_staged_coarse_patch_results(
                discovery_route, discovery_residual)
            discovery_site_artifact = write_causal_vector_artifact(
                self.store, "discovery_site_patches.npz",
                discovery_coarse, protocol_hash=protocol_hash)
            last_successful_stage = stage

            stage = "operator_group_preparation"
            discovery_index = {
                example.example_id: index
                for index, example in enumerate(discovery)
            }
            deep_indices = np.asarray([
                discovery_index[example.example_id] for example in deep
            ], dtype=np.int32)
            deep_production_atlas = {
                "_base_margin": np.asarray(
                    discovery_atlas["_base_margin"])[deep_indices],
                "_source_margin": np.asarray(
                    discovery_atlas["_source_margin"])[deep_indices],
            }
            deep_mismatch = trajectory_mismatch_mapping(
                deep, seed=self.config.trajectory_seed + 103)
            last_successful_stage = stage

            stage = "operator_group_patch"
            operator_groups = evaluate_operator_group_patches(
                self.ctx, deep, deep_semantic, divergence,
                operator_followup, deep_mismatch,
                production_atlas=deep_production_atlas,
                sequence_length=int(discovery_site_values[
                    "sequence_length"]),
                pad_token_id=pad_token_id, config=self.config,
                progress=self._trajectory_progress)
            operator_group_artifact = write_causal_vector_artifact(
                self.store, "discovery_operator_group_patches.npz",
                operator_groups, protocol_hash=protocol_hash)
            last_successful_stage = stage

            stage = "discovery_cumulative_path"
            discovery_path = evaluate_cumulative_path(
                self.ctx, discovery, path_record,
                discovery_site_values, discovery_mismatch,
                production_atlas=discovery_atlas,
                pad_token_id=pad_token_id, config=self.config,
                phase="discovery", evaluate_prefix_curve=True,
                progress=self._trajectory_progress)
            discovery_path_artifact = write_causal_vector_artifact(
                self.store, "discovery_cumulative_path.npz",
                discovery_path, protocol_hash=protocol_hash)
            last_successful_stage = stage

            path_indices = {
                int(row["candidate_index"])
                for row in path_record["sites"]
            }
            path_candidates = [
                candidate for candidate in candidates
                if int(candidate["candidate_index"]) in path_indices
            ]
            if path_candidates:
                stage = "validation_frozen_value_capture"
                validation_site_values = capture_candidate_site_values(
                    self.ctx, validation, validation_semantic, candidates,
                    pad_token_id=pad_token_id, phase="validation",
                    progress=self._trajectory_progress)
                validation_mismatch = trajectory_mismatch_mapping(
                    validation, seed=self.config.trajectory_seed + 107)
                validation_route = evaluate_coarse_site_patches(
                    self.ctx, validation, validation_semantic,
                    path_candidates, validation_site_values,
                    validation_mismatch,
                    production_atlas=validation_atlas,
                    pad_token_id=pad_token_id, config=self.config,
                    phase="validation", patch_kinds=("route",),
                    progress=self._trajectory_progress)
                validation_residual_candidates = (
                    deduplicate_residual_candidates(path_candidates))
                validation_residual = evaluate_coarse_site_patches(
                    self.ctx, validation, validation_semantic,
                    validation_residual_candidates, validation_site_values,
                    validation_mismatch,
                    production_atlas=validation_atlas,
                    pad_token_id=pad_token_id, config=self.config,
                    phase="validation", patch_kinds=("residual",),
                    progress=self._trajectory_progress)
                validation_single = merge_staged_coarse_patch_results(
                    validation_route, validation_residual)
                last_successful_stage = stage

                stage = "validation_frozen_path"
                validation_path = evaluate_cumulative_path(
                    self.ctx, validation, path_record,
                    validation_site_values, validation_mismatch,
                    production_atlas=validation_atlas,
                    pad_token_id=pad_token_id, config=self.config,
                    phase="validation", evaluate_prefix_curve=False,
                    progress=self._trajectory_progress)
            else:
                stage = "validation_skipped_no_causal_path"
                validation_site_values = {
                    "status": "not_evaluated_no_causal_path",
                    "sequence_length": int(
                        discovery_site_values["sequence_length"]),
                    "forward_call_count": 0,
                }
                validation_single = {
                    "status": "no_causal_path",
                    "phase": "validation",
                    "candidate_count": 0,
                    "evaluated_patch_count": 0,
                    "patch_kinds_evaluated": [],
                    "site_summaries": [],
                    "_vectors": [],
                    "forward_call_count": 0,
                    "initial_intervention_batch_size": 0,
                    "effective_intervention_batch_size": 0,
                    "resource_retry_count": 0,
                    "resource_retries": [],
                    "validation_path_evaluated": False,
                }
                validation_path = evaluate_cumulative_path(
                    self.ctx, validation, path_record, {}, {},
                    production_atlas=validation_atlas,
                    pad_token_id=pad_token_id, config=self.config,
                    phase="validation", evaluate_prefix_curve=False,
                    progress=self._trajectory_progress)
            if path_record["path_record_hash"] != path_hash_before_validation:
                raise RuntimeError(
                    "validation changed the frozen discovery path")
            validation_vectors = [
                {**record, "evaluation_kind": "frozen_single_site"}
                for record in validation_single.get("_vectors", ())
            ] + [
                {**record, "evaluation_kind": "frozen_cumulative_path"}
                for record in validation_path.get("_vectors", ())
            ]
            validation_path_artifact = write_causal_vector_artifact(
                self.store, "validation_frozen_path.npz",
                {"_vectors": validation_vectors},
                protocol_hash=protocol_hash)
            last_successful_stage = stage

            stage = "trajectory_graph"
            graph_artifact = write_trajectory_graph(
                self.store, candidates, path_record,
                operator_followup, protocol_hash=protocol_hash)
            artifacts = {
                "atlas_discovery": discovery_atlas_artifact,
                "atlas_validation": validation_atlas_artifact,
                "discovery_site_patches": discovery_site_artifact,
                "discovery_operator_group_patches": (
                    operator_group_artifact),
                "discovery_cumulative_path": discovery_path_artifact,
                "validation_frozen_path": validation_path_artifact,
                "trajectory_graph": graph_artifact,
            }
            manifest_artifact = write_trajectory_manifest(
                self.store, protocol_hash=protocol_hash,
                deep_trace=deep_trace_artifacts,
                artifacts=artifacts,
                trace_output_bytes=trace_output_bytes,
                replay_trace_output_bytes=int(
                    trajectory["replay_trace_output_bytes"]),
                peak_materialized_output_bytes=int(
                    trajectory["peak_materialized_output_bytes"]),
                operator_provenance=operator_provenance)
            artifacts["manifest"] = manifest_artifact
            last_successful_stage = stage
        except Exception as exc:
            message = str(exc).lower()
            resource_limit = isinstance(exc, MemoryError) or any(
                value in message for value in (
                    "resource_exhausted", "out of memory", "oom",
                    "allocation failed"))
            if not resource_limit:
                raise
            return {
                "status": "resource_limit",
                "algorithm_version": "paired_s2_operator_trajectory_v1",
                "failed_stage": stage,
                "last_successful_stage": last_successful_stage,
                "requested_shape": requested_trace_shape,
                "estimated_output_bytes": (
                    trace_output_bytes + estimated_compact_replay_output_bytes
                    + 2 * estimated_score_residual_output_bytes_per_pass
                    if trace_output_bytes else (
                        estimated_capture_trace_output_bytes
                        + estimated_compact_replay_output_bytes
                        + 2
                        * estimated_score_residual_output_bytes_per_pass)),
                "resource_error_type": type(exc).__name__,
                "resource_error": str(exc),
                "active_operator_truncation_applied": False,
                "scientific_invariant_disabled": False,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "test_data_accessor_called": False,
                "test_used": False,
            }

        behavior = self.results["behavioral_eligibility"]["benchmarks"][
            "mib_ioi"]["phases"]
        trace_completeness = {}
        for route, row in trajectory["completeness"].items():
            pool = "qk" if route in {"q", "k"} else route
            retry_count = sum(
                pool in retry["affected_routes"]
                for retry in trajectory["retries"])
            trace_completeness[route] = {
                "initial_width": initial_widths[pool],
                "final_width": int(trajectory["widths"][pool]),
                "max_active_count": int(
                    row["numerator_active_count_max"]),
                "omitted_active_count": int(
                    row["omitted_active_count"]),
                "retry_count": int(retry_count),
                "all_active_replay_error": float(
                    trajectory["closure"]["routes"][route][
                        "canonical_selected_replay_max_abs"]),
            }
        compact_discovery_atlas = self._trajectory_without_private(
            discovery_atlas)
        compact_validation_atlas = self._trajectory_without_private(
            validation_atlas)
        compact_discovery_atlas.pop("metric_rows", None)
        compact_validation_atlas.pop("metric_rows", None)
        compact_divergence = self._trajectory_without_private(divergence)
        compact_divergence.pop("site_rows", None)
        compact_divergence.pop("state_rows", None)
        compact_divergence["extrema"] = extrema
        compact_coarse = self._trajectory_without_private(discovery_coarse)
        compact_groups = self._trajectory_without_private(operator_groups)
        compact_discovery_path = self._trajectory_without_private(
            discovery_path)
        compact_validation_single = self._trajectory_without_private(
            validation_single)
        compact_validation_path = self._trajectory_without_private(
            validation_path)

        causal_path_supported = bool(path_record["sites"])
        first = extrema.get("first_divergence") or {}
        human_lines = [
            "The base/source trajectories satisfied the frozen S2-prefix "
            "identity tolerance before the official S2 span.",
        ]
        if first:
            human_lines.append(
                "The first measured operator-space divergence appeared at "
                f"layer {first['layer']}, role {first['semantic_role']}, "
                f"route {first['route']}.")
        if path_record["sites"]:
            first_path = path_record["sites"][0]
            human_lines.append(
                "The discovery-frozen chronological path begins at "
                f"layer {first_path['layer']} {first_path['semantic_role']} "
                f"{first_path['route']} and contains "
                f"{len(path_record['sites'])} complete-route sites.")
        else:
            human_lines.append(
                "No route site passed the paired and pair-specific gates in "
                "each direction, so no causal path was claimed and held-out "
                "path validation was not run.")
        group_rows = compact_groups.get("site_summaries") or []
        if group_rows:
            best_group = max(
                group_rows,
                key=lambda row: float(
                    row["bidirectional_specific_effect_mean"]))
            human_lines.append(
                "Within the followed-up route sites, the largest observed "
                "group-specific shift was the "
                f"{best_group['group_kind']} intervention at layer "
                f"{best_group['layer']} {best_group['semantic_role']} "
                f"{best_group['route']}; this is checkpoint-specific "
                "exploratory evidence, not an additive operator claim.")
        final_validation = (
            compact_validation_path.get("prefixes") or [{}])[-1]
        validation_uncertainty = compact_validation_path.get(
            "final_frozen_path_uncertainty") or {}
        validation_bidirectional_path_supported = bool(
            validation_uncertainty.get(
                "validation_bidirectional_path_supported", False))
        if validation_uncertainty:
            human_lines.append(
                "The final discovery-frozen path was evaluated on validation "
                "with separate bootstrap intervals and paired sign-flip "
                "permutation tests in each intervention direction, plus a "
                "direction-averaged aggregate statistic.")
            if validation_bidirectional_path_supported:
                human_lines.append(
                    "Both validation directions independently supported the "
                    "paired and paired-minus-mismatched path effects.")
            else:
                human_lines.append(
                    "At least one validation direction failed its paired or "
                    "paired-minus-mismatched gate, so bidirectional path "
                    "support was not claimed.")
        human_summary = {
            "narrative": human_lines,
            "first_divergence": first or None,
            "frozen_path_hash": path_record["path_record_hash"],
            "frozen_path_length": int(path_record["path_length"]),
            "discovery_causal_path_selected": causal_path_supported,
            "validation_paired_effect": final_validation.get(
                "bidirectional_paired_effect_mean"),
            "validation_mismatched_effect": final_validation.get(
                "bidirectional_mismatched_effect_mean"),
            "validation_flip_fraction": final_validation.get(
                "bidirectional_flip_fraction"),
            "validation_paired_effect_ci": validation_uncertainty.get(
                "paired_effect_ci"),
            "validation_specific_effect_ci": validation_uncertainty.get(
                "paired_minus_mismatched_effect_ci"),
            "validation_per_direction": validation_uncertainty.get(
                "per_direction"),
            "validation_direction_averaged_supported": (
                validation_uncertainty.get(
                    "direction_averaged_causal_pair_specific_validation_"
                    "passed")),
            "validation_bidirectional_path_supported": (
                validation_bidirectional_path_supported),
            "validation_causal_pair_specific_passed": (
                validation_uncertainty.get(
                    "causal_pair_specific_validation_passed")),
            "test_consulted": False,
        }
        artifact_warnings = []
        if (isinstance(graph_artifact, Mapping)
                and graph_artifact.get("advisory_threshold_exceeded")):
            artifact_warnings.append({
                "code": "trajectory_graph_json_advisory_threshold_exceeded",
                "severity": "warning",
                "path": graph_artifact.get("path"),
                "encoded_bytes": int(graph_artifact["encoded_bytes"]),
                "warning_threshold_bytes": int(
                    graph_artifact["warning_threshold_bytes"]),
                "write_continued": True,
                "artifact_discarded": False,
                "message": (
                    "Trajectory graph JSON exceeded the advisory size "
                    "threshold; artifact writing continued."),
            })
        return {
            "status": (
                "ready" if causal_path_supported else "no_causal_path"),
            "ready": True,
            "passed": None,
            "descriptive_trace_ready": True,
            "single_site_results_ready": True,
            "discovery_causal_path_selected": causal_path_supported,
            "causal_path_supported": causal_path_supported,
            "causal_path_supported_scope": "discovery_selection_only",
            "validation_path_evaluated": bool(
                validation_path.get("validation_path_evaluated", False)),
            "validation_bidirectional_path_supported": (
                validation_bidirectional_path_supported),
            "analysis_kind": "paired_operator_trajectory",
            "interpretation_scope": (
                "causal_site_trajectory_not_complete_token_to_token_"
                "attention_flow_graph"),
            "attention_edge_weights_captured": False,
            "algorithm_version": "paired_s2_operator_trajectory_v1",
            "scientific_role": "exploratory",
            "claim_role": "checkpoint_specific",
            "checkpoint_specific_claim": True,
            "existing_native_operator_program_modified": False,
            "behavioral_baseline": {
                phase: {
                    "behavior_rows": int(
                        behavior[phase]["runtime_selected_row_count"]),
                    "paired_correct_available": int(
                        behavior[phase]["known_correct_count"]),
                    "paired_correct_used": len(
                        discovery if phase == "discovery" else validation),
                    "base_accuracy": behavior[phase]["accuracy"],
                    "source_accuracy": behavior[phase]["source_accuracy"],
                    "base_margin_mean": behavior[phase]["mean_margin"],
                    "source_own_margin_mean": behavior[phase][
                        "mean_source_own_margin"],
                }
                for phase in ("discovery", "validation")
            },
            "cohorts": {
                "selection_algorithm": (
                    "seeded_hash_order_with_answer_disjoint_same_template_"
                    "mismatch_closure"),
                "seed": self.config.trajectory_seed,
                "discovery_count": len(discovery),
                "validation_count": len(validation),
                "deep_count": len(deep),
                "deep_example_ids": [
                    example.example_id for example in deep],
                "deep_cohort_hash": canonical_hash([
                    example.example_id for example in deep]),
            },
            "production_active_definition": {
                "numerator_active": (
                    "canonical production valid mask after "
                    "execution_prune_eps with execution_weight_nonzero"),
                "denominator_active": (
                    "canonical production admission contribution_nonzero"),
                "execution_prune_eps": float(
                    self.ctx.model_cfg.get("execution_prune_eps", 0.0)),
                "mass_prefix_or_compactness_selection_used": False,
                "production_kernel_is_authoritative": True,
            },
            "production_atlas": {
                "discovery": compact_discovery_atlas,
                "validation": compact_validation_atlas,
                "broad_full_state_vectors_persisted": False,
                "runtime_metric_dtype": "float32",
                "aggregate_dtype": "float64",
                "answer_projection_interpretation": (
                    "descriptive_logit_lens_not_intermediate_causal_effect"),
            },
            "full_active_trace": {
                "requested_shape": requested_trace_shape,
                "trace_completeness": trace_completeness,
                "capture_retries": trajectory["retries"],
                "initial_widths_by_deep_example": trajectory[
                    "initial_widths_by_batch"],
                "widths_carried_forward_between_deep_examples": trajectory[
                    "widths_carried_forward_between_deep_examples"],
                "width_inheritance_monotonic": trajectory[
                    "width_inheritance_monotonic"],
                "closure": trajectory["closure"],
                "trace_output_bytes": trace_output_bytes,
                "compact_replay_trace_output_bytes": int(
                    trajectory["replay_trace_output_bytes"]),
                "capture_score_residual_output_bytes": int(
                    trajectory["capture_score_residual_bytes"]),
                "replay_score_residual_output_bytes": int(
                    trajectory["replay_score_residual_bytes"]),
                "total_streamed_trace_output_bytes": int(
                    trajectory["total_streamed_trace_output_bytes"]),
                "estimated_peak_capture_plus_replay_output_bytes": (
                    int(trajectory["peak_materialized_output_bytes"])),
                "deep_trace_streamed_one_example_at_a_time": True,
                "canonical_minimal_kernel_reused": True,
                "precast_closure_is_descriptive": True,
                "canonical_replay_is_authoritative": True,
                "per_operator_scalar_interpretation": (
                    "nonlinear_reroute_execution_trace_not_bitwise_"
                    "additive_route_or_logit_contribution"),
                "exact_causal_units": (
                    "canonical_full_active_group_replay_and_complete_"
                    "route_or_state_patches"),
                "full_vocab_logits_materialized": False,
                "final_logits_parity_basis": (
                    "shared_output_projection_of_parity_checked_final_"
                    "residual_plus_candidate_log_probability_parity"),
            },
            "operator_parameter_provenance": operator_provenance,
            "divergence_atlas": compact_divergence,
            "candidate_selection": selection,
            "discovery_causal_patch": compact_coarse,
            "operator_followup_selection": operator_followup,
            "operator_decomposition": compact_groups,
            "frozen_path": path_record,
            "discovery_cumulative_path": compact_discovery_path,
            "validation_frozen_single_sites": compact_validation_single,
            "validation_frozen_path": compact_validation_path,
            "tpu_execution": {
                "successful_forward_call_count": sum((
                    int(discovery_atlas["forward_call_count"]),
                    int(validation_atlas["forward_call_count"]),
                    int(trajectory["forward_call_count"]),
                    int(discovery_site_values["forward_call_count"]),
                    int(validation_site_values["forward_call_count"]),
                    int(discovery_coarse["forward_call_count"]),
                    int(operator_groups["forward_call_count"]),
                    int(discovery_path["forward_call_count"]),
                    int(validation_single["forward_call_count"]),
                    int(validation_path["forward_call_count"]),
                )),
                "stage_forward_call_count": {
                    "production_atlas_discovery": int(
                        discovery_atlas["forward_call_count"]),
                    "production_atlas_validation": int(
                        validation_atlas["forward_call_count"]),
                    "full_active_capture_and_replay": int(
                        trajectory["forward_call_count"]),
                    "candidate_value_capture_discovery": int(
                        discovery_site_values["forward_call_count"]),
                    "candidate_value_capture_validation": int(
                        validation_site_values["forward_call_count"]),
                    "coarse_patch_discovery": int(
                        discovery_coarse["forward_call_count"]),
                    "operator_group_patch_discovery": int(
                        operator_groups["forward_call_count"]),
                    "cumulative_path_discovery": int(
                        discovery_path["forward_call_count"]),
                    "frozen_single_site_validation": int(
                        validation_single["forward_call_count"]),
                    "frozen_path_validation": int(
                        validation_path["forward_call_count"]),
                },
                "production_scoring_rows_per_example": 4,
                "candidate_state_rows_per_example": 2,
                "deep_trace_candidate_rows_per_example": 4,
                "coarse_patch_variants_per_site_direction": 4,
                "operator_group_variants_per_site_direction": 5,
                "answer_candidates_fused_per_variant": 2,
                "candidate_intervention_batch_size_initial": (
                    self.config.trajectory_intervention_batch_size),
                "candidate_intervention_batch_size_effective_discovery": (
                    discovery_coarse[
                        "effective_intervention_batch_size"]),
                "candidate_intervention_batch_size_effective_validation": (
                    validation_single[
                        "effective_intervention_batch_size"]),
                "full_active_capture_retry_count": len(
                    trajectory["retries"]),
                "coarse_resource_retry_count": int(
                    discovery_coarse["resource_retry_count"])
                    + int(validation_single["resource_retry_count"]),
                "path_prefix_batch_size_initial": int(
                    discovery_path["initial_prefix_batch_size"]),
                "path_prefix_batch_size_effective": int(
                    discovery_path["effective_prefix_batch_size"]),
                "path_prefix_resource_retry_count": int(
                    discovery_path["resource_retry_count"]),
            },
            "memory_protection": {
                "parameters_replicated_per_variant": False,
                "full_pool_activation_tensor_materialized": False,
                "static_operator_key_tables_materialized_once": True,
                "static_operator_vectors_repeated_per_occurrence": False,
                "individual_operator_python_forward_loop": False,
                "candidate_specific_recompilation": False,
                "candidate_answer_variants_fused": True,
                "cumulative_path_prefixes_fused": bool(
                    discovery_path.get("prefix_batch_fusion_enabled", False)),
                "cumulative_path_prefix_batch_size_configured": (
                    self.config.trajectory_path_prefix_batch_size),
                "cumulative_path_prefix_oom_fallback": "halve_to_one",
                "fixed_intervention_slots": (
                    self.config.trajectory_max_patch_sites_per_variant),
                "intervention_variants": (
                    self.config.trajectory_intervention_batch_size),
                "candidate_interventions_fused_per_forward": (
                    self.config.trajectory_intervention_batch_size),
                "raw_trace_streaming": (
                    "one_example_side_sparse_shard_then_release"),
                "semantic_position_state_gather_only": True,
                "compact_replay_output_only": True,
                "fixed_deep_trace_shapes_reused": True,
                "deep_trace_widths_carried_forward_monotonically": bool(
                    trajectory["width_inheritance_monotonic"]),
                "active_operator_omission_on_resource_pressure": False,
            },
            "artifacts": {
                **artifacts,
                "deep_trace": {
                    "shard_count": deep_trace_artifacts["shard_count"],
                    "shards": deep_trace_artifacts.get("shards", []),
                },
            },
            "item_json_contract": {
                "target_bytes": 2 * 1024 * 1024,
                "raw_state_arrays_embedded": False,
                "raw_operator_arrays_embedded": False,
                "raw_effect_vectors_embedded": False,
                "overflow_policy": "warning_and_write_continues",
            },
            "split_isolation": {
                "candidate_selection_phase": "discovery",
                "validation_used_for_selection": False,
                "test_used_for_selection": False,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "test_data_accessor_called": False,
            },
            "test_evaluated": False,
            "test_evaluation_count": 0,
            "test_data_accessor_called": False,
            "test_used": False,
            "artifact_warnings": artifact_warnings,
            "human_summary": human_summary,
        }

    def _run_ioi_scale_discovery_localization(self) -> dict[str, Any]:
        """Localize a new 1.3B IOI circuit without validation/test access."""
        if self._scope("ioi_scale_discovery_localization") != ("mib_ioi",):
            raise ValueError(
                "IOI scale discovery localization is registered only for "
                "mib_ioi")
        if not self._ioi_scale_discovery_isolated:
            raise RuntimeError(
                "IOI scale discovery localization did not enter "
                "split-isolated execution")

        spec = load_ioi_scale_discovery_spec()
        spec.validate_runtime(
            target_id=str(self.ctx.model_info.get("target_id") or ""),
            model_version=self.model_version,
            checkpoint_step=int(self.ctx.checkpoint_step),
            checkpoint_identity=str(self.contract["checkpoint_identity"]),
            checkpoint_config_hash=str(
                self.ctx.model_info.get("checkpoint_config_hash") or ""),
            model_config_hash=str(self.contract["model_config_hash"]),
            benchmark_build_id=self.build.build_id,
            benchmark_manifest_hash=self.build.manifest_hash,
            seed=self.config.seed,
            max_examples_per_phase=self.config.max_examples_per_phase,
            capture_threshold=self.config.capture_threshold,
            capture_widths=(
                self.config.capture_topk_qk,
                self.config.capture_topk_v,
                self.config.capture_topk_rst,
                self.config.capture_max_topk_qk,
                self.config.capture_max_topk_v,
                self.config.capture_max_topk_rst,
            ),
            rank_stability_minimum=self.config.rank_stability_min,
        )

        self._print(
            "TRAIN_ANALYSIS_POOL ioi_scale_discovery "
            "stage=behavioral_eligibility status=running")
        phase_rows = self._load_phase_examples("mib_ioi", "discovery")
        if len(phase_rows) != IOI_SCALE_DISCOVERY_ROW_COUNT:
            raise ValueError(
                "IOI scale discovery canonical row count drift: "
                f"expected={IOI_SCALE_DISCOVERY_ROW_COUNT} "
                f"actual={len(phase_rows)}")
        behavior = evaluate_behavior(
            self.ctx, phase_rows,
            pad_token_id=int(self.tokenizer.pad_token_id))
        known_correct_mask = [
            bool(value) for value in behavior["known_correct"]]
        known_correct = [
            example for example, keep in zip(
                phase_rows, known_correct_mask)
            if keep
        ]
        independent = self._independent_capture_examples(
            "mib_ioi", known_correct)
        independent_ids = [str(example.example_id) for example in independent]
        if len(set(independent_ids)) != len(independent_ids):
            raise ValueError(
                "IOI paired-correct discovery units must have unique "
                "example_id values")
        if any(
                str(example.pair_type) != "s2_io_flip_counterfactual"
                for example in independent):
            raise ValueError(
                "IOI scale discovery received an unexpected pair type")

        behavior_vector_fields = (
            "example_ids",
            "base_positive_logp",
            "base_negative_logp",
            "base_margin",
            "corrupted_margin",
            "source_own_margin",
            "source_behavior_scored",
            "base_known_correct",
            "source_known_correct",
            "known_correct",
        )
        behavior_vector_payload = {
            field: behavior[field] for field in behavior_vector_fields}
        behavior_summary = {
            "status": "ready",
            "phase": "discovery",
            "runtime_selected_row_count": len(phase_rows),
            "known_correct_row_count": int(
                behavior["known_correct_count"]),
            "paired_correct_independent_unit_count": len(independent),
            "paired_correct_minimum": spec.minimum_paired_correct,
            "paired_correct_gate_passed": (
                len(independent) >= spec.minimum_paired_correct),
            "independent_unit": "example_id",
            "base_accuracy": float(behavior["accuracy"]),
            "source_accuracy": behavior["source_accuracy"],
            "pair_accuracy": float(behavior["pair_accuracy"]),
            "mean_margin": float(behavior["mean_margin"]),
            "mean_corrupted_margin": float(
                behavior["mean_corrupted_margin"]),
            "mean_source_own_margin": behavior["mean_source_own_margin"],
            "runtime_selected_example_ids_hash": canonical_hash([
                example.example_id for example in phase_rows]),
            "known_correct_row_ids_hash": canonical_hash([
                example.example_id for example in known_correct]),
            "paired_correct_independent_unit_ids_hash": canonical_hash(
                independent_ids),
            "raw_behavior_vector_payload_hash": canonical_hash(
                behavior_vector_payload),
            "raw_behavior_vectors_persisted": False,
        }
        self._print(
            "TRAIN_ANALYSIS_POOL ioi_scale_discovery "
            "stage=behavioral_eligibility "
            f"status={'ready' if behavior_summary['paired_correct_gate_passed'] else 'insufficient_behavior'} "
            f"paired_correct={len(independent)}")

        split_isolation = {
            "ranking_phase": "discovery",
            "selection_phase": "discovery",
            "validation_evaluated": False,
            "validation_evaluation_count": 0,
            "validation_data_accessor_called": False,
            "test_evaluated": False,
            "test_evaluation_count": 0,
            "test_data_accessor_called": False,
            "validation_used_for_selection": False,
            "test_used_for_selection": False,
            "validation_may_change_circuit": False,
            "test_may_change_circuit": False,
        }
        storage_audit = {
            "aggregate_ranking_persisted": True,
            "selected_prefix_and_identity_hashes_persisted": True,
            "raw_per_example_behavior_vectors_persisted": False,
            "raw_per_example_operator_vectors_persisted": False,
            "raw_capture_rows_persisted": False,
            "raw_parameters_persisted": False,
            "dense_capture_persisted": False,
        }
        preregistered = {
            "path": spec.path,
            "content_hash": spec.content_hash,
            "status": spec.payload["status"],
            "rank_stability_minimum": spec.rank_stability_minimum,
            "rank_stability_split_rule": (
                "seeded_hash_of_benchmark_and_example_id"),
            "rank_stability_minimum_independent_units_per_split": 16,
            "audited_prefix_counts": list(spec.prefix_counts),
            "cumulative_absolute_importance_minimum": (
                spec.cumulative_importance_minimum),
            "split_topk_overlap_minimum": spec.split_overlap_minimum,
            "circuit_freeze_rule": (
                "smallest_audited_prefix_passing_both_discovery_gates"),
            "literal_400m_operator_id_transfer_forbidden": True,
            "result_dependent_route_changes_forbidden": True,
            "confirmatory_conditions_if_frozen": list(
                spec.payload["confirmatory_protocol_if_frozen"][
                    "validation"]["conditions"]),
            "suppression_if_frozen": (
                "circuit_wide_execution_numerator_suppression_with_"
                "full_production_denominator"),
            "control_seeds_if_frozen": {
                name: int(row["seed"])
                for name, row in spec.payload[
                    "confirmatory_protocol_if_frozen"]["controls"].items()
                if isinstance(row, Mapping) and "seed" in row
            },
            "restoration_if_frozen": (
                "exact_selected_numerator_restore_from_same_example_"
                "intact_execution"),
        }
        if len(independent) < IOI_SCALE_DISCOVERY_MINIMUM_PAIRED_CORRECT_COUNT:
            return {
                "status": "insufficient_behavior",
                "decision": (
                    "no_localization_due_to_insufficient_paired_correct_ioi_"
                    "scale_discovery_behavior"),
                "benchmark": "mib_ioi",
                "phase": "discovery",
                "preregistered_specification": preregistered,
                "behavioral_eligibility": behavior_summary,
                "capture": {
                    "status": "not_run_insufficient_behavior",
                    "raw_capture_rows_persisted": False,
                },
                "localization": {
                    "status": "not_run_insufficient_behavior",
                    "circuit": {
                        "status": "not_frozen",
                        "selected_k": 0,
                        "sites": [],
                    },
                },
                "confirmatory_eligible": False,
                "split_isolation": split_isolation,
                "storage_audit": storage_audit,
                "strongest_supported_claim": None,
            }

        self._print(
            "TRAIN_ANALYSIS_POOL ioi_scale_discovery "
            "stage=operator_capture status=running")
        capture = capture_discovery_candidates(
            self.ctx, independent,
            seed=self.config.seed,
            retain_rows=False,
            **self._capture_kwargs("mib_ioi"))
        ranked = [
            dict(row) for row in capture.pop("ranked_sites", ())
            if isinstance(row, Mapping)
        ]
        localization = build_ioi_scale_localization(
            ranked,
            capture=capture,
            spec=spec,
        )
        raw_rows_materialized = bool(capture.pop(
            "raw_rows_materialized_for_runtime", False))
        capture.update({
            "discovery_independent_example_count": len(independent),
            "runtime_phase_cap": self.config.max_examples_for("mib_ioi"),
            "aggregate_ranked_site_count": len(ranked),
            "aggregate_ranked_sites_content_hash": canonical_hash(ranked),
            "aggregate_ranked_site_preview": ranked[:16],
            "raw_capture_row_count": int(capture["total_row_count"]),
            "raw_capture_rows_persisted": False,
            "raw_capture_rows_used_transiently": raw_rows_materialized,
            "raw_capture_retention": (
                "aggregate_ranking_selected_prefix_and_identity_hashes"),
        })

        status = str(localization["status"])
        circuit = dict(localization.get("circuit") or {})
        if status == "ready":
            decision = "ioi_1p3b_discovery_circuit_frozen"
            strongest_claim = "ioi_1p3b_discovery_operator_circuit_frozen"
        elif status == "unstable_localization":
            decision = (
                "no_scale_confirmation_due_to_unstable_ioi_localization")
            strongest_claim = None
        elif status == "no_preregistered_prefix":
            decision = (
                "no_scale_confirmation_due_to_ioi_discovery_freeze_rule_"
                "failure")
            strongest_claim = None
        else:
            decision = (
                "no_scale_confirmation_due_to_ioi_discovery_capture_failure")
            strongest_claim = None
        self._print(
            "TRAIN_ANALYSIS_POOL ioi_scale_discovery "
            f"stage=operator_capture status={status} "
            f"rank_stability={localization.get('rank_stability', {}).get('rank_stability')} "
            f"selected_k={circuit.get('selected_k', 0)} "
            f"route_counts={circuit.get('selected_route_counts')}")

        del ranked
        gc.collect()
        return {
            "status": status,
            "decision": decision,
            "benchmark": "mib_ioi",
            "phase": "discovery",
            "preregistered_specification": preregistered,
            "behavioral_eligibility": behavior_summary,
            "capture": capture,
            "localization": localization,
            "confirmatory_eligible": status == "ready",
            "split_isolation": split_isolation,
            "storage_audit": storage_audit,
            "strongest_supported_claim": strongest_claim,
        }

    def _run_arc_discovery_localization(self) -> dict[str, Any]:
        """Localize and freeze one ARC circuit without held-out access."""
        if self._scope("arc_discovery_localization") != ("mib_arc",):
            raise ValueError(
                "ARC discovery localization is registered only for mib_arc")
        if not self._arc_discovery_isolated:
            raise RuntimeError(
                "ARC discovery localization did not enter split-isolated "
                "execution")

        spec = load_arc_discovery_spec()
        spec.validate_runtime(
            target_id=str(self.ctx.model_info.get("target_id") or ""),
            model_version=self.model_version,
            checkpoint_step=int(self.ctx.checkpoint_step),
            checkpoint_identity=str(self.contract["checkpoint_identity"]),
            checkpoint_config_hash=str(
                self.ctx.model_info.get("checkpoint_config_hash") or ""),
            model_config_hash=str(self.contract["model_config_hash"]),
            benchmark_build_id=self.build.build_id,
            benchmark_manifest_hash=self.build.manifest_hash,
            seed=self.config.seed,
            max_examples_per_phase=self.config.max_examples_per_phase,
            capture_threshold=self.config.capture_threshold,
            capture_widths=(
                self.config.capture_topk_qk,
                self.config.capture_topk_v,
                self.config.capture_topk_rst,
                self.config.capture_max_topk_qk,
                self.config.capture_max_topk_v,
                self.config.capture_max_topk_rst,
            ),
            rank_stability_minimum=self.config.rank_stability_min,
        )

        self._print(
            "TRAIN_ANALYSIS_POOL arc_discovery "
            "stage=behavioral_eligibility status=running")
        phase_rows = self._load_phase_examples("mib_arc", "discovery")
        if len(phase_rows) != ARC_DISCOVERY_ROW_COUNT:
            raise ValueError(
                "ARC discovery canonical row count drift: "
                f"expected={ARC_DISCOVERY_ROW_COUNT} "
                f"actual={len(phase_rows)}")
        behavior = evaluate_behavior(
            self.ctx, phase_rows,
            pad_token_id=int(self.tokenizer.pad_token_id))
        known_correct_mask = [
            bool(value) for value in behavior["known_correct"]]
        known_correct = [
            example for example, keep in zip(
                phase_rows, known_correct_mask)
            if keep
        ]
        independent = self._independent_capture_examples(
            "mib_arc", known_correct)
        if len(independent) != ARC_DISCOVERY_PAIRED_CORRECT_COUNT:
            raise ValueError(
                "ARC discovery paired-correct independent-unit drift: "
                f"expected={ARC_DISCOVERY_PAIRED_CORRECT_COUNT} "
                f"actual={len(independent)}")
        independent_ids = [str(example.example_id) for example in independent]
        if len(set(independent_ids)) != len(independent_ids):
            raise ValueError(
                "ARC paired-correct discovery units must have unique "
                "example_id values")
        if any(
                str(example.pair_type) != "symbol_counterfactual"
                for example in independent):
            raise ValueError(
                "ARC discovery localization received an unexpected pair type")

        behavior_vector_fields = (
            "example_ids",
            "base_positive_logp",
            "base_negative_logp",
            "base_margin",
            "corrupted_margin",
            "source_own_margin",
            "source_behavior_scored",
            "base_known_correct",
            "source_known_correct",
            "known_correct",
        )
        behavior_vector_payload = {
            field: behavior[field] for field in behavior_vector_fields}
        behavior_summary = {
            "status": "ready",
            "phase": "discovery",
            "runtime_selected_row_count": len(phase_rows),
            "known_correct_row_count": int(
                behavior["known_correct_count"]),
            "paired_correct_independent_unit_count": len(independent),
            "independent_unit": "example_id",
            "base_accuracy": float(behavior["accuracy"]),
            "source_accuracy": behavior["source_accuracy"],
            "pair_accuracy": float(behavior["pair_accuracy"]),
            "mean_margin": float(behavior["mean_margin"]),
            "mean_corrupted_margin": float(
                behavior["mean_corrupted_margin"]),
            "mean_source_own_margin": behavior["mean_source_own_margin"],
            "runtime_selected_example_ids_hash": canonical_hash([
                example.example_id for example in phase_rows]),
            "known_correct_row_ids_hash": canonical_hash([
                example.example_id for example in known_correct]),
            "paired_correct_independent_unit_ids_hash": canonical_hash(
                independent_ids),
            "raw_behavior_vector_payload_hash": canonical_hash(
                behavior_vector_payload),
            "raw_behavior_vectors_persisted": False,
        }
        self._print(
            "TRAIN_ANALYSIS_POOL arc_discovery "
            "stage=behavioral_eligibility status=ready "
            f"paired_correct={len(independent)}")

        self._print(
            "TRAIN_ANALYSIS_POOL arc_discovery "
            "stage=operator_capture status=running")
        capture = capture_discovery_candidates(
            self.ctx, independent,
            seed=self.config.seed,
            retain_rows=False,
            **self._capture_kwargs("mib_arc"))
        ranked = [
            dict(row) for row in capture.pop("ranked_sites", ())
            if isinstance(row, Mapping)
        ]
        localization = build_arc_localization(
            ranked,
            capture=capture,
            spec=spec,
        )
        raw_rows_materialized = bool(capture.pop(
            "raw_rows_materialized_for_runtime", False))
        capture.update({
            "discovery_independent_example_count": len(independent),
            "runtime_phase_cap": self.config.max_examples_for("mib_arc"),
            "aggregate_ranked_site_count": len(ranked),
            "aggregate_ranked_sites_content_hash": canonical_hash(ranked),
            "aggregate_ranked_site_preview": ranked[:16],
            "raw_capture_row_count": int(capture["total_row_count"]),
            "raw_capture_rows_persisted": False,
            "raw_capture_rows_used_transiently": raw_rows_materialized,
            "raw_capture_retention": (
                "aggregate_ranking_selected_prefix_and_identity_hashes"),
        })

        status = str(localization["status"])
        circuit = dict(localization.get("circuit") or {})
        if status == "ready":
            decision = "arc_discovery_circuit_frozen"
            strongest_claim = "arc_discovery_operator_circuit_frozen"
        elif status == "unstable_localization":
            decision = "no_validation_due_to_unstable_arc_localization"
            strongest_claim = None
        elif status == "no_preregistered_prefix":
            decision = "no_validation_due_to_arc_discovery_freeze_rule_failure"
            strongest_claim = None
        else:
            decision = "no_validation_due_to_arc_discovery_capture_failure"
            strongest_claim = None
        self._print(
            "TRAIN_ANALYSIS_POOL arc_discovery "
            f"stage=operator_capture status={status} "
            f"rank_stability={localization.get('rank_stability', {}).get('rank_stability')} "
            f"selected_k={circuit.get('selected_k', 0)} "
            f"route_counts={circuit.get('selected_route_counts')}")

        del ranked
        gc.collect()
        return {
            "status": status,
            "decision": decision,
            "benchmark": "mib_arc",
            "phase": "discovery",
            "preregistered_specification": {
                "path": spec.path,
                "content_hash": spec.content_hash,
                "status": spec.payload["status"],
                "rank_stability_minimum": (
                    spec.rank_stability_minimum),
                "rank_stability_split_rule": (
                    "seeded_balanced_example_id_split"),
                "rank_stability_split_independent_unit_counts": [25, 25],
                "audited_prefix_counts": list(spec.prefix_counts),
                "cumulative_absolute_importance_minimum": (
                    spec.cumulative_importance_minimum),
                "split_topk_overlap_minimum": (
                    spec.split_overlap_minimum),
                "circuit_freeze_rule": (
                    "smallest_audited_prefix_passing_both_discovery_gates"),
                "result_dependent_route_changes_forbidden": True,
                "matched_random_if_frozen": {
                    "replicate_count": 100,
                    "seed": 24172,
                    "match_fields": ["layer", "route"],
                    "sampling": (
                        "uniform_without_replacement_within_layer_route_cell"),
                    "sampling_population": "frozen_site_complement",
                },
                "suppression_if_frozen": (
                    "circuit_wide_execution_numerator_suppression_with_"
                    "full_production_denominator"),
                "restoration_if_frozen": (
                    "exact_selected_numerator_restore_from_same_example_"
                    "intact_execution"),
            },
            "behavioral_eligibility": behavior_summary,
            "capture": capture,
            "localization": localization,
            "confirmatory_eligible": status == "ready",
            "split_isolation": {
                "ranking_phase": "discovery",
                "selection_phase": "discovery",
                "validation_evaluated": False,
                "validation_evaluation_count": 0,
                "validation_data_accessor_called": False,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "test_data_accessor_called": False,
                "validation_used_for_selection": False,
                "test_used_for_selection": False,
                "validation_may_change_circuit": False,
                "test_may_change_circuit": False,
            },
            "storage_audit": {
                "aggregate_ranking_persisted": True,
                "selected_prefix_and_identity_hashes_persisted": True,
                "raw_per_example_behavior_vectors_persisted": False,
                "raw_per_example_operator_vectors_persisted": False,
                "raw_capture_rows_persisted": False,
                "raw_parameters_persisted": False,
                "dense_capture_persisted": False,
            },
            "strongest_supported_claim": strongest_claim,
        }

    def _run_ravel_discovery_localization(self) -> dict[str, Any]:
        """Localize and freeze variable circuits without held-out access."""
        if self._scope("ravel_discovery_localization") != ("ravel",):
            raise ValueError(
                "RAVEL discovery localization is registered only for ravel")
        if not self._ravel_discovery_isolated:
            raise RuntimeError(
                "RAVEL discovery localization did not enter split-isolated "
                "execution")

        spec = load_ravel_discovery_spec()
        spec.validate_runtime(
            target_id=str(self.ctx.model_info.get("target_id") or ""),
            model_version=self.model_version,
            checkpoint_step=int(self.ctx.checkpoint_step),
            checkpoint_identity=str(self.contract["checkpoint_identity"]),
            checkpoint_config_hash=str(
                self.ctx.model_info.get("checkpoint_config_hash") or ""),
            model_config_hash=str(self.contract["model_config_hash"]),
            benchmark_build_id=self.build.build_id,
            benchmark_manifest_hash=self.build.manifest_hash,
            seed=self.config.seed,
            ravel_max_examples_per_phase=(
                self.config.ravel_max_examples_per_phase),
            capture_threshold=self.config.capture_threshold,
            capture_widths=(
                self.config.capture_topk_qk,
                self.config.capture_topk_v,
                self.config.capture_topk_rst,
                self.config.capture_max_topk_qk,
                self.config.capture_max_topk_v,
                self.config.capture_max_topk_rst,
            ),
            rank_stability_minimum=self.config.rank_stability_min,
        )

        self._print(
            "TRAIN_ANALYSIS_POOL ravel_discovery "
            "stage=behavioral_eligibility status=running")
        phase_rows = self._load_phase_examples("ravel", "discovery")
        if len(phase_rows) != RAVEL_DISCOVERY_ROW_COUNT:
            raise ValueError(
                "RAVEL discovery canonical row count drift: "
                f"expected={RAVEL_DISCOVERY_ROW_COUNT} "
                f"actual={len(phase_rows)}")
        behavior = evaluate_behavior(
            self.ctx, phase_rows,
            pad_token_id=int(self.tokenizer.pad_token_id))
        known_correct_mask = [
            bool(value) for value in behavior["known_correct"]]
        known_correct = [
            example for example, keep in zip(
                phase_rows, known_correct_mask)
            if keep
        ]
        independent = self._independent_capture_examples(
            "ravel", known_correct)
        if len(independent) != RAVEL_DISCOVERY_PAIRED_CORRECT_COUNT:
            raise ValueError(
                "RAVEL discovery paired-correct independent-unit drift: "
                f"expected={RAVEL_DISCOVERY_PAIRED_CORRECT_COUNT} "
                f"actual={len(independent)}")
        variable_unit_counts = {
            variable: sum(
                str(example.causal_variable) == variable
                for example in independent)
            for variable in benchmark_spec("ravel").causal_variables
        }
        source_unit_counts = {
            variable: {
                source_column: sum(
                    str(example.causal_variable) == variable
                    and str(example.metadata.get(
                        "official_counterfactual_column")) == source_column
                    for example in independent)
                for source_column in benchmark_spec(
                    "ravel").counterfactual_columns
            }
            for variable in benchmark_spec("ravel").causal_variables
        }
        behavior_vector_fields = (
            "example_ids",
            "base_positive_logp",
            "base_negative_logp",
            "base_margin",
            "corrupted_margin",
            "source_own_margin",
            "source_behavior_scored",
            "base_known_correct",
            "source_known_correct",
            "known_correct",
        )
        behavior_vector_payload = {
            field: behavior[field] for field in behavior_vector_fields}
        behavior_summary = {
            "status": "ready",
            "phase": "discovery",
            "runtime_selected_row_count": len(phase_rows),
            "known_correct_row_count": int(
                behavior["known_correct_count"]),
            "paired_correct_independent_unit_count": len(independent),
            "paired_correct_independent_unit_count_by_variable": (
                variable_unit_counts),
            "paired_correct_independent_unit_count_by_variable_and_source": (
                source_unit_counts),
            "base_accuracy": float(behavior["accuracy"]),
            "source_accuracy": behavior["source_accuracy"],
            "pair_accuracy": float(behavior["pair_accuracy"]),
            "mean_margin": float(behavior["mean_margin"]),
            "mean_corrupted_margin": float(
                behavior["mean_corrupted_margin"]),
            "mean_source_own_margin": behavior["mean_source_own_margin"],
            "runtime_selected_example_ids_hash": canonical_hash([
                example.example_id for example in phase_rows]),
            "known_correct_row_ids_hash": canonical_hash([
                example.example_id for example in known_correct]),
            "paired_correct_independent_unit_ids_hash": canonical_hash([
                str(example.metadata["pair_group_id"])
                for example in independent]),
            "raw_behavior_vector_payload_hash": canonical_hash(
                behavior_vector_payload),
            "raw_behavior_vectors_persisted": False,
        }
        self._print(
            "TRAIN_ANALYSIS_POOL ravel_discovery "
            "stage=behavioral_eligibility status=ready "
            f"paired_correct={len(independent)}")

        self._print(
            "TRAIN_ANALYSIS_POOL ravel_discovery "
            "stage=operator_capture status=running")
        capture = capture_discovery_candidates(
            self.ctx, independent,
            seed=self.config.seed,
            retain_rows=True,
            **self._capture_kwargs("ravel"))
        transient_rows = list(capture.get("rows") or ())
        variable_results = build_ravel_variable_localization(
            transient_rows,
            spec=spec,
            rank_stability_by_variable=dict(
                capture.get("rank_stability_by_causal_variable") or {}),
        )
        pooled_ranked = [
            dict(row) for row in capture.pop("ranked_sites", ())
            if isinstance(row, Mapping)
        ]
        capture.pop("rows", None)
        raw_rows_materialized = bool(capture.pop(
            "raw_rows_materialized_for_runtime", False))
        capture.update({
            "discovery_independent_example_count": len(independent),
            "runtime_phase_cap": self.config.max_examples_for("ravel"),
            "pooled_ranked_site_count": len(pooled_ranked),
            "pooled_ranked_sites_content_hash": canonical_hash(pooled_ranked),
            "pooled_ranked_site_preview": pooled_ranked[:16],
            "pooled_ranking_used_for_circuit_freeze": False,
            "raw_capture_row_count": int(capture["total_row_count"]),
            "raw_capture_rows_persisted": False,
            "raw_capture_rows_used_transiently": raw_rows_materialized,
            "raw_capture_retention": (
                "aggregate_variable_rankings_selected_prefixes_and_hashes"),
        })

        variable_statuses = {
            variable: str(variable_results[variable]["status"])
            for variable in spec.variables
        }
        all_variables_ready = all(
            status == "ready" for status in variable_statuses.values())
        if all_variables_ready:
            status = "ready"
            decision = "all_variable_circuits_frozen"
            strongest_claim = (
                "ravel_discovery_variable_operator_circuits_frozen")
        elif any(
                value == "unstable_localization"
                for value in variable_statuses.values()):
            status = "unstable_localization"
            decision = "no_validation_due_to_unstable_variable_localization"
            strongest_claim = None
        else:
            status = "no_preregistered_prefix"
            decision = "no_validation_due_to_discovery_freeze_rule_failure"
            strongest_claim = None
        for variable in spec.variables:
            result = variable_results[variable]
            circuit = dict(result.get("circuit") or {})
            self._print(
                "TRAIN_ANALYSIS_POOL ravel_discovery "
                f"variable={variable} status={result['status']} "
                f"rank_stability={result.get('rank_stability', {}).get('rank_stability')} "
                f"selected_k={circuit.get('selected_k', 0)} "
                f"route_counts={circuit.get('selected_route_counts')}")
        self._print(
            "TRAIN_ANALYSIS_POOL ravel_discovery "
            f"stage=operator_capture status={status}")

        del transient_rows
        del pooled_ranked
        gc.collect()
        return {
            "status": status,
            "decision": decision,
            "benchmark": "ravel",
            "phase": "discovery",
            "preregistered_specification": {
                "path": spec.path,
                "content_hash": spec.content_hash,
                "status": spec.payload["status"],
                "variables": list(spec.variables),
                "rank_stability_minimum": (
                    spec.rank_stability_minimum),
                "audited_prefix_counts": list(spec.prefix_counts),
                "cumulative_absolute_importance_minimum": (
                    spec.cumulative_importance_minimum),
                "split_topk_overlap_minimum": (
                    spec.split_overlap_minimum),
                "circuit_freeze_rule": (
                    "smallest_audited_prefix_passing_both_discovery_gates"),
                "result_dependent_route_changes_forbidden": True,
            },
            "behavioral_eligibility": behavior_summary,
            "capture": capture,
            "variable_localization": variable_results,
            "variable_statuses": variable_statuses,
            "all_preregistered_variables_ready": all_variables_ready,
            "split_isolation": {
                "ranking_phase": "discovery",
                "selection_phase": "discovery",
                "validation_evaluated": False,
                "validation_evaluation_count": 0,
                "validation_data_accessor_called": False,
                "test_evaluated": False,
                "test_evaluation_count": 0,
                "test_data_accessor_called": False,
                "validation_used_for_selection": False,
                "test_used_for_selection": False,
                "validation_may_change_circuit": False,
                "test_may_change_circuit": False,
            },
            "storage_audit": {
                "aggregate_variable_rankings_persisted": True,
                "selected_prefixes_and_identity_hashes_persisted": True,
                "raw_per_example_behavior_vectors_persisted": False,
                "raw_per_example_operator_vectors_persisted": False,
                "raw_capture_rows_persisted": False,
                "raw_parameters_persisted": False,
                "dense_capture_persisted": False,
            },
            "strongest_supported_claim": strongest_claim,
        }

    def _reduce_ravel_capture(
            self, capture: Mapping[str, Any]) -> tuple[
                dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        rows = list(capture.get("rows") or ())
        seeds: dict[str, dict[str, Any]] = {}
        control_profiles: dict[str, dict[str, Any]] = {}
        for variable in benchmark_spec("ravel").causal_variables:
            denominators: dict[tuple[int, str], int] = defaultdict(int)
            absolute_totals: dict[tuple[int, str, int], float] = defaultdict(
                float)
            signed_totals: dict[tuple[int, str, int], float] = defaultdict(
                float)
            counts: dict[tuple[int, str, int], int] = defaultdict(int)
            for row in rows:
                if (not row["qualified"]
                        or row["causal_variable"] != variable):
                    continue
                layer, route = int(row["layer"]), str(row["route"])
                denominators[(layer, route)] += 1
                for operator_id, weight in zip(
                        row["operator_ids"], row["weights"]):
                    key = (layer, route, int(operator_id))
                    value = float(weight)
                    absolute_totals[key] += abs(value)
                    signed_totals[key] += value
                    counts[key] += 1
            if not absolute_totals:
                seeds[variable] = {
                    "status": "no_qualified_discovery_site"}
                continue
            ranked = sorted([
                (
                    total / denominators[(key[0], key[1])], key,
                    counts[key],
                )
                for key, total in absolute_totals.items()
            ], key=lambda value: (-value[0], value[1]))
            importance, (layer, route, operator_id), discovery_count = ranked[0]
            seeds[variable] = {
                "status": "ready",
                "layer": layer,
                "route": route,
                "operator_id": operator_id,
                "importance": importance,
                "discovery_count": discovery_count,
                "selection_phase": "discovery",
            }
            denominator = denominators[(layer, route)]
            profile_ids = sorted(
                key[2] for key in signed_totals
                if key[0] == layer and key[1] == route)
            control_profiles[variable] = {
                "layer": layer,
                "route": route,
                "qualified_row_denominator": denominator,
                "operator_ids": profile_ids,
                "importance": [
                    signed_totals[(layer, route, value)] / denominator
                    for value in profile_ids
                ],
            }
        return seeds, control_profiles

    def _run_operator_localization(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        causal_ids = [
            benchmark_id for benchmark_id in self._scope("operator_localization")
            if benchmark_id in PRIMARY_BENCHMARK_IDS
        ]
        for offset, benchmark_id in enumerate(causal_ids):
            examples = self._independent_capture_examples(
                benchmark_id,
                self._known_correct(benchmark_id, "discovery"))
            if len(examples) < self.config.minimum_known_correct:
                output[benchmark_id] = {
                    "status": "insufficient_behavior",
                    "known_correct_count": len(examples),
                    "minimum_known_correct": self.config.minimum_known_correct,
                }
                continue
            capture = capture_discovery_candidates(
                self.ctx, examples,
                seed=self.config.seed + offset * 1009,
                retain_rows=(benchmark_id == "ravel"),
                **self._capture_kwargs(benchmark_id))
            capture["discovery_independent_example_count"] = len(examples)
            capture["runtime_phase_cap"] = self.config.max_examples_for(
                benchmark_id)
            if benchmark_id == "ravel":
                seeds, profiles = self._reduce_ravel_capture(capture)
                capture["causal_variable_seeds"] = seeds
                capture["causal_variable_control_profiles"] = profiles
                variable_stability = dict(
                    capture.get("rank_stability_by_causal_variable") or {})
                expected_variables = benchmark_spec("ravel").causal_variables
                stability_gate_passed = (
                    set(variable_stability) == set(expected_variables)
                    and all(
                        variable_stability[variable].get("status") == "ready"
                        and float(variable_stability[variable][
                            "rank_stability"])
                        >= self.config.rank_stability_min
                        for variable in expected_variables
                    )
                )
                capture.update({
                    "rank_stability_gate_threshold": (
                        self.config.rank_stability_min),
                    "rank_stability_gate_rule": (
                        "every_preregistered_causal_variable_at_or_above_"
                        "threshold"),
                    "rank_stability_gate_passed": stability_gate_passed,
                })
                if not stability_gate_passed:
                    capture["status"] = "unstable_localization"
            capture.pop("rows", None)
            rows_used_transiently = bool(capture.pop(
                "raw_rows_materialized_for_runtime", False))
            capture.update({
                "raw_capture_row_count": int(capture["total_row_count"]),
                "raw_capture_rows_persisted": False,
                "raw_capture_rows_used_transiently": rows_used_transiently,
                "raw_capture_retention": (
                    "aggregate_ranked_sites_and_preregistered_ravel_profiles"),
            })
            output[benchmark_id] = capture
        ready = [row for row in output.values() if row.get("status") == "ready"]
        attempted = [
            row for row in output.values()
            if row.get("status") in {"ready", "unstable_localization"}
        ]
        complete = bool(causal_ids) and len(ready) == len(causal_ids)
        rank_values = [
            float(row["rank_stability"]) for row in attempted
            if row.get("rank_stability") is not None
        ]
        if complete:
            status = "ready"
        elif any(
                row.get("status") == "insufficient_behavior"
                for row in output.values()):
            status = "insufficient_behavior"
        elif any(
                row.get("status") == "unstable_localization"
                for row in output.values()):
            status = "unstable_localization"
        else:
            status = "incomplete"
        return {
            "status": status,
            "benchmarks": output,
            "qualified_fraction": (
                min(float(row["qualified_fraction"]) for row in attempted)
                if attempted else 0.0),
            "rank_stability": min(rank_values) if rank_values else None,
            "all_primary_benchmarks_ready": complete,
            "ranking_phase": "discovery",
        }

    def _circuits(self, benchmark_id: str):
        capture = self.results["operator_localization"]["benchmarks"][benchmark_id]
        if capture.get("status") != "ready":
            raise ValueError(f"localization is not ready for {benchmark_id}")
        return nested_circuits(
            ranked_site_objects(capture), shape=self.shape,
            benchmark_id=benchmark_id, fractions=CIRCUIT_FRACTIONS)

    def _circuit_curve(
            self, benchmark_id: str, *, mode: str) -> dict[str, Any]:
        validation = self._known_correct(benchmark_id, "validation")
        if len(validation) < self.config.minimum_known_correct:
            return {
                "status": "insufficient_validation_behavior",
                "validation_known_correct": len(validation),
                "test_evaluated": False,
            }
        validation_base, validation_corrupt = self._behavior_margins(
            benchmark_id, "validation", validation)
        if abs(float(validation_base.mean() - validation_corrupt.mean())) <= 1e-12:
            return {
                "status": "undefined_validation_behavior_contrast",
                "test_evaluated": False,
            }
        circuits = self._circuits(benchmark_id)
        parity = all_ones_retention_parity(
            self.ctx, validation[:min(2, len(validation))],
            shape=self.shape,
            pad_token_id=int(self.tokenizer.pad_token_id), mode=mode)

        def evaluate_one(examples, baseline, corrupted, *, fraction,
                         circuit, seed):
            retained = evaluate_circuit_retention(
                self.ctx, examples, circuit, shape=self.shape, mode=mode,
                pad_token_id=int(self.tokenizer.pad_token_id))
            return {
                "fraction": fraction,
                "site_count": circuit.site_count,
                "circuit_hash": circuit.circuit_hash,
                "mean_margin": retained["mean_margin"],
                "accuracy": retained["accuracy"],
                "faithfulness": normalized_faithfulness(
                    retained["mean_margin"], float(baseline.mean()),
                    float(corrupted.mean())),
                "faithfulness_ci": bootstrap_faithfulness_ci(
                    retained["margin"], baseline, corrupted,
                    samples=self.config.bootstrap_samples,
                    alpha=self.config.alpha, seed=seed),
            }

        validation_rows = [
            evaluate_one(
                validation, validation_base, validation_corrupt,
                fraction=fraction, circuit=circuit,
                seed=self.config.seed + 2000 + index)
            for index, (fraction, circuit) in enumerate(circuits)
        ]
        selection = select_on_validation(
            validation_rows,
            minimum_faithfulness=self.config.circuit_faithfulness_min)
        result = {
            "status": selection["status"],
            "mode": mode,
            "all_ones_parity": parity,
            "validation": {
                "phase": "validation",
                "baseline_mean": float(validation_base.mean()),
                "corrupted_mean": float(validation_corrupt.mean()),
                "rows": validation_rows,
                "curve": faithfulness_curve(validation_rows),
            },
            "selection": selection,
            "test": {
                "phase": "test",
                "status": "not_evaluated_validation_rejected",
                "selection_frozen_before_evaluation": True,
                "test_used_for_selection": False,
                "rows": [],
            },
            "selected_test_faithfulness": None,
            "selected_test_faithfulness_ci": None,
            "selected_circuit": None,
        }
        if selection["status"] != "selected":
            return result

        selected_circuit = next((
            circuit for fraction, circuit in circuits
            if fraction == selection.get("selected_fraction")), None)
        if selected_circuit is None:
            raise RuntimeError("validation selected an unknown circuit fraction")
        selected_validation = next(
            row for row in validation_rows
            if row["fraction"] == selection.get("selected_fraction"))
        selected_circuit_record = selected_circuit.to_dict()
        selected_circuit_record["circuit_hash"] = selected_validation[
            "circuit_hash"]
        result["selected_circuit"] = selected_circuit_record

        # The test phase is unopened until the validation choice is frozen,
        # and only that one frozen circuit is evaluated on test.
        test = self._known_correct(benchmark_id, "test")
        if len(test) < self.config.minimum_known_correct:
            result["status"] = "insufficient_test_behavior"
            result["test"].update({
                "status": "insufficient_behavior",
                "known_correct": len(test),
            })
            return result
        test_base, test_corrupt = self._behavior_margins(
            benchmark_id, "test", test)
        if abs(float(test_base.mean() - test_corrupt.mean())) <= 1e-12:
            result["status"] = "undefined_test_behavior_contrast"
            result["test"]["status"] = "undefined_behavior_contrast"
            return result
        selected_test = evaluate_one(
            test, test_base, test_corrupt,
            fraction=float(selection["selected_fraction"]),
            circuit=selected_circuit,
            seed=self.config.seed + 3000)
        result.update({
            "status": "ready",
            "test": {
                "phase": "test",
                "status": "ready",
                "selection_frozen_before_evaluation": True,
                "test_used_for_selection": False,
                "evaluation_scope": "validation_selected_circuit_only",
                "baseline_mean": float(test_base.mean()),
                "corrupted_mean": float(test_corrupt.mean()),
                "rows": [selected_test],
            },
            "selected_test_faithfulness": selected_test["faithfulness"],
            "selected_test_faithfulness_ci": selected_test[
                "faithfulness_ci"],
        })
        return result

    def _run_sufficiency(self, mode: str) -> dict[str, Any]:
        output = {}
        for benchmark_id in self._scope():
            if benchmark_id not in MIB_CIRCUIT_BENCHMARKS:
                continue
            capture = self.results["operator_localization"]["benchmarks"].get(
                benchmark_id, {})
            if capture.get("status") != "ready":
                output[benchmark_id] = {"status": "localization_not_ready"}
            else:
                output[benchmark_id] = self._circuit_curve(
                    benchmark_id, mode=mode)
        complete = bool(output) and all(
            row.get("status") == "ready" for row in output.values())
        return {
            "status": "ready" if complete else "incomplete",
            "mode": mode,
            "benchmarks": output,
            "selection_phase": "validation",
            "evaluation_phase": "test",
            "test_used_for_selection": False,
        }

    def _run_conditional_circuit_sufficiency(self) -> dict[str, Any]:
        return self._run_sufficiency("conditional_execution_sufficiency")

    def _run_autonomous_circuit_sufficiency(self) -> dict[str, Any]:
        return self._run_sufficiency("autonomous_subcircuit_sufficiency")

    def _selected_conditional_circuit(
            self, benchmark_id: str) -> OperatorCircuit | None:
        result = self.results["conditional_circuit_sufficiency"]["benchmarks"][
            benchmark_id]
        if result.get("status") != "ready":
            return None
        fraction = result.get("selection", {}).get("selected_fraction")
        if fraction is None:
            return None
        return next(
            circuit for value, circuit in self._circuits(benchmark_id)
            if value == fraction)

    def _run_circuit_necessity(self) -> dict[str, Any]:
        output = {}
        for benchmark_id, sufficiency in self.results[
                "conditional_circuit_sufficiency"]["benchmarks"].items():
            if sufficiency.get("status") != "ready":
                output[benchmark_id] = {
                    "status": "conditional_sufficiency_not_ready"}
                continue
            circuit = self._selected_conditional_circuit(benchmark_id)
            if circuit is None:
                output[benchmark_id] = {
                    "status": "no_validation_selected_circuit"}
                continue
            examples = self._known_correct(benchmark_id, "test")
            baseline, _ = self._behavior_margins(
                benchmark_id, "test", examples)
            intervention = evaluate_circuit_necessity(
                self.ctx, examples, circuit, shape=self.shape,
                pad_token_id=int(self.tokenizer.pad_token_id))
            effect = necessity_effect(baseline, intervention["margin"])
            margin_drop = (
                baseline - np.asarray(intervention["margin"], dtype=np.float64))
            effect_ci = bootstrap_mean_ci(
                margin_drop, samples=self.config.bootstrap_samples,
                alpha=self.config.alpha, seed=self.config.seed + 4001)
            null = paired_permutation_test(
                baseline, intervention["margin"],
                samples=self.config.permutation_samples,
                seed=self.config.seed + 4003)
            output[benchmark_id] = {
                "status": "ready",
                "selection_phase": "validation",
                "evaluation_phase": "test",
                "test_used_for_selection": False,
                "selected_fraction": sufficiency["selection"][
                    "selected_fraction"],
                "circuit_hash": circuit.circuit_hash,
                "intervention": intervention,
                "effect": effect,
                "effect_ci": effect_ci,
                "paired_null": null,
            }
        ready = [row for row in output.values() if row.get("status") == "ready"]
        if ready:
            correction = benjamini_hochberg(
                [row["paired_null"]["p_value_two_sided"] for row in ready],
                self.config.alpha)
            for row, adjusted, reject in zip(
                    ready, correction["adjusted_p_values"],
                    correction["reject"]):
                row["paired_null"]["adjusted_p_value"] = adjusted
                row["paired_null"]["reject_after_bh"] = reject
        else:
            correction = {"adjusted_p_values": [], "reject": []}
        return {
            "status": (
                "ready" if output and len(ready) == len(output) else "incomplete"),
            "benchmarks": output,
            "mean_margin_drop": _mean([
                row["effect"]["mean_margin_drop"] for row in ready]),
            "all_significant_after_bh": bool(ready) and all(
                row["paired_null"]["reject_after_bh"]
                and row["effect_ci"]["ci_low"] > 0.0
                for row in ready),
            "multiple_comparison_correction": "benjamini_hochberg",
            "bh": correction,
        }

    def _materialize_pool(self) -> dict[str, np.ndarray]:
        if self._pool_host is None:
            module = analysis_model_module(self.ctx.model_cfg)
            params = module._squeeze_params(self.ctx.params)
            pool = module._pool_params_with_operator_keys(
                params["neuron_pool"],
                self.ctx.model_cfg.get("operator_key_mode"))
            names = (
                "attn_qk_read", "attn_qk_write", "attn_qk_op_key",
                "attn_v_read", "attn_v_write", "attn_v_op_key",
                "rst_read", "rst_write", "rst_op_key",
            )
            self._pool_host = {
                name: np.asarray(
                    materialize_global_array(pool[name]), dtype=np.float64)
                for name in names
            }
        return self._pool_host

    def _candidate_ids(self, route: str) -> list[int]:
        pool_size = self.shape.pool_size(route)
        ranked = []
        for result in self.results["operator_localization"]["benchmarks"].values():
            if result.get("status") != "ready":
                continue
            ranked.extend(
                row for row in result["ranked_sites"]
                if row["route"] == route)
        ranked.sort(key=lambda row: (
            -abs(float(row["importance"])), int(row["operator_id"])))
        selected: list[int] = []
        if "ravel" in self.benchmark_ids:
            ravel_capture = self.results["operator_localization"][
                "benchmarks"].get("ravel", {})
            if ravel_capture.get("status") == "ready":
                selected.extend(
                    seed["operator_id"]
                    for seed in self._ravel_variable_seeds().values()
                    if (seed.get("status") == "ready"
                        and seed.get("route") == route))
        selected = list(dict.fromkeys(selected))
        if len(selected) > self.config.space_max_operators:
            raise ValueError(
                "space_max_operators cannot contain all preselected "
                f"RAVEL variable seeds for route={route}")
        for row in ranked:
            operator_id = int(row["operator_id"])
            if operator_id not in selected:
                selected.append(operator_id)
            if len(selected) >= self.config.space_max_operators:
                return selected
        width = min(pool_size, self.config.space_max_operators)
        for operator_id in np.linspace(
                0, pool_size - 1, num=width, dtype=np.int64):
            value = int(operator_id)
            if value not in selected:
                selected.append(value)
            if len(selected) >= width:
                break
        if len(selected) < 2:
            raise ValueError("operator-space candidate set has fewer than two ids")
        return selected

    def _run_operator_space_structure(self) -> dict[str, Any]:
        if "ravel" in self._scope("operator_space_structure"):
            localization = self.results["operator_localization"][
                "benchmarks"].get("ravel", {})
            if (localization.get("status") != "ready"
                    or localization.get(
                        "rank_stability_gate_passed") is not True):
                return {
                    "status": "localization_stability_not_met",
                    "benchmark": "ravel",
                    "rank_stability": localization.get("rank_stability"),
                    "rank_stability_gate_threshold": (
                        self.config.rank_stability_min),
                    "rank_stability_by_causal_variable": localization.get(
                        "rank_stability_by_causal_variable"),
                    "family_discovery_executed": False,
                }
        pool = self._materialize_pool()
        key_prefix = {"q": "attn_qk", "k": "attn_qk",
                      "v": "attn_v", "rst": "rst"}
        output = {}
        for offset, route in enumerate(("q", "k", "v", "rst")):
            ids = self._candidate_ids(route)
            prefix = key_prefix[route]
            read = pool[f"{prefix}_read"][ids]
            write = pool[f"{prefix}_write"][ids]
            address = pool[f"{prefix}_op_key"][ids]
            discovered = discover_functional_families(
                read, write,
                neighbor_k=self.config.family_neighbor_k,
                similarity_quantile=self.config.family_similarity_quantile)
            local_families = discovered.pop("families")
            global_families = [
                [ids[index] for index in family] for family in local_families
            ]
            output[route] = {
                "status": "ready",
                **discovered,
                "candidate_operator_ids": ids,
                "candidate_selection": (
                    "discovery_contribution_then_preregistered_even_pool_coverage"),
                "candidate_funnel_limited": len(ids) < self.shape.pool_size(route),
                "families_global_operator_ids": global_families,
                "address_confirmation": address_confirmation(
                    local_families, address,
                    seed=self.config.seed + offset * 37),
            }
        return {
            "status": "ready",
            "routes": output,
            "family_count": int(sum(
                row["family_count"] for row in output.values())),
            "address_used_for_discovery": False,
            "unit": "reciprocal_local_rw_function_family",
            "pool_provenance": operator_pool_provenance(self.ctx),
            "full_pool_exhaustive": all(
                not row["candidate_funnel_limited"] for row in output.values()),
        }

    def _ravel_variable_seeds(self) -> dict[str, dict[str, Any]]:
        capture = self.results["operator_localization"]["benchmarks"]["ravel"]
        if (capture.get("status") != "ready"
                or capture.get("rank_stability_gate_passed") is not True):
            raise ValueError(
                "RAVEL variable seeds are ineligible under localization "
                "stability gate")
        seeds = capture.get("causal_variable_seeds")
        if not isinstance(seeds, Mapping):
            raise ValueError(
                "RAVEL localization lacks preregistered variable seeds")
        expected = set(benchmark_spec("ravel").causal_variables)
        if set(seeds) != expected:
            raise ValueError("RAVEL localization variable seed set mismatch")
        return {str(key): dict(value) for key, value in seeds.items()}

    def _family_for_seed(
            self, seed: Mapping[str, Any]) -> tuple[int, str, list[int]]:
        if seed.get("status") != "ready":
            raise ValueError("RAVEL variable has no discovery seed")
        layer, route, operator_id = (
            int(seed["layer"]), str(seed["route"]), int(seed["operator_id"]))
        families = self.results["operator_space_structure"]["routes"][route][
            "families_global_operator_ids"]
        containing = [family for family in families if operator_id in family]
        family = min(containing, key=lambda value: (len(value), value)) if containing else [operator_id]
        return layer, route, family

    def _matched_nonfamily_control(
            self, variable: str, *, layer: int, route: str,
            family: Sequence[int]) -> dict[str, Any]:
        capture = self.results["operator_localization"]["benchmarks"]["ravel"]
        profiles = capture.get("causal_variable_control_profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError(
                "RAVEL localization lacks contribution-matching profiles")
        profile = profiles.get(variable)
        if not isinstance(profile, Mapping):
            raise ValueError(
                f"no RAVEL control profile for variable={variable}")
        if (int(profile["layer"]) != int(layer)
                or str(profile["route"]) != str(route)):
            raise ValueError(
                "RAVEL control profile does not match the selected seed site")
        operator_ids = list(profile.get("operator_ids") or ())
        importance_values = list(profile.get("importance") or ())
        if len(operator_ids) != len(importance_values):
            raise ValueError("RAVEL control profile arrays are misaligned")
        importance = {
            int(operator_id): float(value)
            for operator_id, value in zip(operator_ids, importance_values)
        }
        family_ids = [int(value) for value in family]
        available = set(range(self.shape.pool_size(route))) - set(family_ids)
        if len(available) < len(family_ids):
            raise ValueError("operator pool cannot provide a disjoint control")
        matched = []
        absolute_errors = []
        for operator_id in sorted(
                family_ids,
                key=lambda value: (-importance.get(value, 0.0), value)):
            target = importance.get(operator_id, 0.0)
            selected = min(
                available,
                key=lambda value: (
                    abs(importance.get(value, 0.0) - target), value))
            available.remove(selected)
            matched.append(selected)
            absolute_errors.append(
                abs(importance.get(selected, 0.0) - target))
        return {
            "operator_ids": matched,
            "matching_feature": (
                "variable_specific_discovery_mean_absolute_contribution"),
            "matching_without_replacement": True,
            "family_excluded": True,
            "mean_absolute_importance_match_error": float(np.mean(
                absolute_errors)),
        }

    def _paired_interchange_advantage(
            self, candidate: Mapping[str, Any],
            comparator: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
        candidate_rows = {
            str(row["example_id"]): row for row in candidate["rows"]
            if row["pair_type"] == "cause"
        }
        comparator_rows = {
            str(row["example_id"]): row for row in comparator["rows"]
            if row["pair_type"] == "cause"
        }
        if set(candidate_rows) != set(comparator_rows):
            raise ValueError("interchange control rows are not paired")
        example_ids = sorted(candidate_rows)
        candidate_effect = np.asarray([
            float(candidate_rows[example_id]["patched_intervention_margin"])
            - float(candidate_rows[example_id]["base_intervention_margin"])
            for example_id in example_ids
        ], dtype=np.float64)
        comparator_effect = np.asarray([
            float(comparator_rows[example_id]["patched_intervention_margin"])
            - float(comparator_rows[example_id]["base_intervention_margin"])
            for example_id in example_ids
        ], dtype=np.float64)
        difference = candidate_effect - comparator_effect
        return {
            "status": "ready",
            "n": len(example_ids),
            "effect": "cause_margin_improvement_difference",
            "candidate_mean": float(candidate_effect.mean()),
            "comparator_mean": float(comparator_effect.mean()),
            "difference_ci": bootstrap_mean_ci(
                difference, samples=self.config.bootstrap_samples,
                alpha=self.config.alpha, seed=seed),
            "paired_null": paired_permutation_test(
                candidate_effect, comparator_effect,
                samples=self.config.permutation_samples, seed=seed + 1),
        }

    def _interchange_phase(
            self, examples: Sequence[Any], *, phase: str,
            layer: int, route: str, family: Sequence[int],
            seed_offset: int = 0) -> dict[str, Any]:
        rows = evaluate_operator_interchange(
            self.ctx, examples, layer=layer, route=route,
            operator_ids=family,
            pad_token_id=int(self.tokenizer.pad_token_id))
        score = score_interchange_rows(rows)
        variable_results = {}
        p_value_variables = []
        for offset, variable in enumerate(sorted({
                str(row["causal_variable"]) for row in rows})):
            cause_rows = [
                row for row in rows
                if row["pair_type"] == "cause"
                and row["causal_variable"] == variable]
            isolation_rows = [
                row for row in rows
                if row["pair_type"] == "isolation"
                and row["causal_variable"] == variable]
            if min(len(cause_rows), len(isolation_rows)) < 2:
                variable_results[variable] = {
                    "status": "insufficient_pairs",
                    "cause_pair_count": len(cause_rows),
                    "isolation_pair_count": len(isolation_rows),
                }
                continue
            cause_base = [
                row["base_intervention_margin"] for row in cause_rows]
            cause_patched = [
                row["patched_intervention_margin"] for row in cause_rows]
            cause_delta = np.asarray(cause_patched) - np.asarray(cause_base)
            isolation_delta = [
                abs(float(row["patched_base_margin"])
                    - float(row["base_base_margin"]))
                for row in isolation_rows
            ]
            null = paired_permutation_test(
                cause_patched, cause_base,
                samples=self.config.permutation_samples,
                seed=self.config.seed + seed_offset + 5000 + offset)
            variable_results[variable] = {
                "status": "ready",
                "cause_pair_count": len(cause_rows),
                "isolation_pair_count": len(isolation_rows),
                "cause_effect_ci": bootstrap_mean_ci(
                    cause_delta, samples=self.config.bootstrap_samples,
                    alpha=self.config.alpha,
                    seed=self.config.seed + seed_offset + 5100 + offset),
                "cause_paired_null": null,
                "isolation_effect_ci": bootstrap_mean_ci(
                    isolation_delta, samples=self.config.bootstrap_samples,
                    alpha=self.config.alpha,
                    seed=self.config.seed + seed_offset + 5200 + offset),
            }
            p_value_variables.append(variable)
        if p_value_variables:
            correction = benjamini_hochberg([
                variable_results[variable]["cause_paired_null"][
                    "p_value_two_sided"]
                for variable in p_value_variables
            ], self.config.alpha)
            for variable, adjusted, reject in zip(
                    p_value_variables, correction["adjusted_p_values"],
                    correction["reject"]):
                variable_results[variable]["cause_paired_null"].update({
                    "adjusted_p_value": adjusted,
                    "reject_after_bh": reject,
                })
        ready_variables = [
            value for value in variable_results.values()
            if value.get("status") == "ready"
        ]
        score.update({
            "phase": phase,
            "rows": rows,
            "cause_effect_ci": bootstrap_mean_ci(
                [
                    float(row["patched_intervention_margin"])
                    - float(row["base_intervention_margin"])
                    for row in rows if row["pair_type"] == "cause"
                ],
                samples=self.config.bootstrap_samples,
                alpha=self.config.alpha,
                seed=self.config.seed + seed_offset + 101),
            "isolation_effect_ci": bootstrap_mean_ci(
                [
                    abs(float(row["patched_base_margin"])
                        - float(row["base_base_margin"]))
                    for row in rows if row["pair_type"] == "isolation"
                ],
                samples=self.config.bootstrap_samples,
                alpha=self.config.alpha,
                seed=self.config.seed + seed_offset + 103),
            "causal_variables": variable_results,
            "all_variables_causal_after_bh": bool(ready_variables) and all(
                value["cause_paired_null"]["reject_after_bh"]
                and value["cause_effect_ci"]["ci_low"] > 0.0
                for value in ready_variables)
                and len(ready_variables) == len(variable_results),
            "all_variables_isolated": bool(ready_variables) and all(
                value["isolation_effect_ci"]["ci_high"]
                <= self.config.isolation_max_absolute_effect
                for value in ready_variables)
                and len(ready_variables) == len(variable_results),
            "multiple_comparison_correction": "benjamini_hochberg",
        })
        return score

    def _aggregate_ravel_phase(
            self, per_variable: Mapping[str, Mapping[str, Any]], *,
            phase: str) -> dict[str, Any]:
        expected = benchmark_spec("ravel").causal_variables
        ready = {
            variable: dict(per_variable.get(variable) or {})
            for variable in expected
            if (per_variable.get(variable) or {}).get("status") == "ready"
        }
        if not ready:
            return {
                "status": "insufficient_variable_evidence",
                "phase": phase,
                "per_variable": dict(per_variable),
            }
        rows = [
            row for result in ready.values() for row in result["rows"]
        ]
        aggregate = score_interchange_rows(rows)
        p_values = [
            ready[variable]["causal_variables"][variable][
                "cause_paired_null"]["p_value_two_sided"]
            for variable in expected if variable in ready
        ]
        correction = benjamini_hochberg(p_values, self.config.alpha)
        variable_evidence = {}
        for index, variable in enumerate(
                item for item in expected if item in ready):
            evidence = dict(ready[variable]["causal_variables"][variable])
            evidence["cause_paired_null"] = dict(
                evidence["cause_paired_null"])
            evidence["cause_paired_null"].update({
                "adjusted_p_value": correction["adjusted_p_values"][index],
                "reject_after_bh": correction["reject"][index],
            })
            variable_evidence[variable] = evidence
        complete = len(ready) == len(expected)
        advantage_evidence = {
            variable: {
                "family_size": int(ready[variable]["family_advantage"][
                    "family_size"]),
                "vs_seed": {
                    **ready[variable]["family_advantage"]["vs_seed"],
                    "paired_null": dict(ready[variable]["family_advantage"][
                        "vs_seed"]["paired_null"]),
                },
                "vs_matched_nonfamily": {
                    **ready[variable]["family_advantage"][
                        "vs_matched_nonfamily"],
                    "paired_null": dict(ready[variable]["family_advantage"][
                        "vs_matched_nonfamily"]["paired_null"]),
                },
            }
            for variable in expected if variable in ready
        }
        advantage_tests = [
            (variable, comparison)
            for variable in expected if variable in advantage_evidence
            for comparison in ("vs_seed", "vs_matched_nonfamily")
        ]
        advantage_correction = benjamini_hochberg([
            advantage_evidence[variable][comparison]["paired_null"][
                "p_value_two_sided"]
            for variable, comparison in advantage_tests
        ], self.config.alpha)
        for index, (variable, comparison) in enumerate(advantage_tests):
            advantage_evidence[variable][comparison]["paired_null"].update({
                "adjusted_p_value": advantage_correction[
                    "adjusted_p_values"][index],
                "reject_after_bh": advantage_correction["reject"][index],
            })
        all_family_advantages = complete and all(
            evidence["family_size"] > 1
            and all(
                evidence[comparison]["paired_null"]["reject_after_bh"]
                and evidence[comparison]["difference_ci"]["ci_low"] > 0.0
                for comparison in ("vs_seed", "vs_matched_nonfamily"))
            for evidence in advantage_evidence.values())
        aggregate.update({
            "status": "ready" if complete else "incomplete",
            "phase": phase,
            "per_variable": dict(per_variable),
            "causal_variables": variable_evidence,
            "cause_effect_ci": bootstrap_mean_ci(
                [
                    float(row["patched_intervention_margin"])
                    - float(row["base_intervention_margin"])
                    for row in rows if row["pair_type"] == "cause"
                ],
                samples=self.config.bootstrap_samples,
                alpha=self.config.alpha, seed=self.config.seed + 6001),
            "isolation_effect_ci": bootstrap_mean_ci([
                abs(float(row["patched_base_margin"])
                    - float(row["base_base_margin"]))
                for row in rows if row["pair_type"] == "isolation"
            ], samples=self.config.bootstrap_samples,
                alpha=self.config.alpha, seed=self.config.seed + 6003),
            "all_variables_causal_after_bh": complete and all(
                evidence["cause_paired_null"]["reject_after_bh"]
                and evidence["cause_effect_ci"]["ci_low"] > 0.0
                for evidence in variable_evidence.values()),
            "all_variables_isolated": complete and all(
                evidence["isolation_effect_ci"]["ci_high"]
                <= self.config.isolation_max_absolute_effect
                for evidence in variable_evidence.values()),
            "family_advantage_by_variable": advantage_evidence,
            "all_variables_family_advantage_after_bh": (
                all_family_advantages),
            "family_advantage_multiple_comparison_correction": (
                "benjamini_hochberg_across_variables_and_controls"),
            "family_advantage_bh": advantage_correction,
            "multiple_comparison_correction": "benjamini_hochberg",
            "rows": rows,
        })
        return aggregate

    def _run_ravel_causal_mediation(self) -> dict[str, Any]:
        if "ravel" not in self._scope("ravel_causal_mediation"):
            return {"status": "not_requested", "benchmark": "ravel"}
        behavior = self.results["behavioral_eligibility"]["benchmarks"][
            "ravel"]["phases"]
        ineligible_phases = [
            phase for phase in ("validation", "test")
            if not behavior[phase].get(
                "eligible_for_mechanistic_claims", False)
        ]
        if ineligible_phases:
            return {
                "status": "behavior_not_eligible",
                "benchmark": "ravel",
                "ineligible_phases": ineligible_phases,
            }
        capture = self.results["operator_localization"]["benchmarks"].get(
            "ravel", {})
        if (capture.get("status") != "ready"
                or capture.get("rank_stability_gate_passed") is not True):
            return {
                "status": "localization_stability_not_met",
                "benchmark": "ravel",
                "causal_intervention_executed": False,
            }
        space = self.results.get("operator_space_structure", {})
        if space.get("status") != "ready":
            return {
                "status": "operator_space_not_ready",
                "benchmark": "ravel",
                "causal_intervention_executed": False,
            }
        seeds = self._ravel_variable_seeds()
        expected_variables = benchmark_spec("ravel").causal_variables
        selected_units = {}
        for variable in expected_variables:
            seed = seeds.get(variable, {})
            if seed.get("status") != "ready":
                selected_units[variable] = dict(seed)
                continue
            layer, route, family = self._family_for_seed(seed)
            matched_control = self._matched_nonfamily_control(
                variable, layer=layer, route=route, family=family)
            selected_units[variable] = {
                **dict(seed),
                "operator_ids": family,
                "family_size": len(family),
                "seed_only_operator_ids": [int(seed["operator_id"])],
                "matched_nonfamily_control": matched_control,
                "unit_selection_rule": (
                    "smallest_reciprocal_local_rw_family_containing_"
                    "variable_specific_top_discovery_site"),
            }
        phases = {}
        for phase_index, phase in enumerate(("validation", "test")):
            examples = self._known_correct("ravel", phase)
            per_variable = {}
            for variable_index, variable in enumerate(expected_variables):
                unit = selected_units.get(variable, {})
                selected = [
                    example for example in examples
                    if example.causal_variable == variable]
                counts = {
                    pair_type: sum(
                        example.pair_type == pair_type for example in selected)
                    for pair_type in ("cause", "isolation")
                }
                if (unit.get("status") != "ready"
                        or min(counts.values())
                        < self.config.minimum_pairs_per_causal_variable):
                    per_variable[variable] = {
                        "status": "insufficient_behavior_or_discovery_unit",
                        "known_correct_pair_counts": counts,
                        "minimum_pairs_per_causal_variable": (
                            self.config.minimum_pairs_per_causal_variable),
                    }
                    continue
                base_offset = phase_index * 10_000 + variable_index * 1_000
                family_result = self._interchange_phase(
                    selected, phase=phase,
                    layer=int(unit["layer"]), route=str(unit["route"]),
                    family=unit["operator_ids"],
                    seed_offset=base_offset)
                seed_result = self._interchange_phase(
                    selected, phase=phase,
                    layer=int(unit["layer"]), route=str(unit["route"]),
                    family=unit["seed_only_operator_ids"],
                    seed_offset=base_offset + 100)
                matched_result = self._interchange_phase(
                    selected, phase=phase,
                    layer=int(unit["layer"]), route=str(unit["route"]),
                    family=unit["matched_nonfamily_control"]["operator_ids"],
                    seed_offset=base_offset + 200)
                family_result.update({
                    "intervention_unit": {
                        "layer": int(unit["layer"]),
                        "route": str(unit["route"]),
                        "operator_ids": list(unit["operator_ids"]),
                    },
                    "controls": {
                        "seed_only": seed_result,
                        "matched_nonfamily": matched_result,
                    },
                    "family_advantage": {
                        "family_size": int(unit["family_size"]),
                        "vs_seed": self._paired_interchange_advantage(
                            family_result, seed_result,
                            seed=self.config.seed + base_offset + 700),
                        "vs_matched_nonfamily": (
                            self._paired_interchange_advantage(
                                family_result, matched_result,
                                seed=self.config.seed + base_offset + 800)),
                    },
                })
                per_variable[variable] = family_result
            phases[phase] = self._aggregate_ravel_phase(
                per_variable, phase=phase)
        complete = all(
            phases.get(phase, {}).get("status") == "ready"
            for phase in ("validation", "test"))
        return {
            "status": "ready" if complete else "incomplete",
            "benchmark": "ravel",
            "selection_phase": "discovery",
            "selected_units_by_causal_variable": selected_units,
            "validation": phases["validation"],
            "test": phases["test"],
            "official_ravel_featurizer_equivalence_claimed": False,
        }

    def _run_multilayer_trajectory(self) -> dict[str, Any]:
        if "ravel" not in self._scope("multilayer_trajectory"):
            return {"status": "not_requested", "benchmark": "ravel"}
        behavior = self.results["behavioral_eligibility"]["benchmarks"][
            "ravel"]["phases"]["test"]
        if not behavior.get("eligible_for_mechanistic_claims", False):
            return {
                "status": "behavior_not_eligible",
                "benchmark": "ravel",
                "known_correct_independent_unit_count": behavior.get(
                    "known_correct_independent_unit_count", 0),
            }
        localization = self.results["operator_localization"]["benchmarks"].get(
            "ravel", {})
        if (localization.get("status") != "ready"
                or localization.get(
                    "rank_stability_gate_passed") is not True):
            return {
                "status": "localization_stability_not_met",
                "benchmark": "ravel",
                "trajectory_executed": False,
            }
        examples = self._known_correct("ravel", "test")
        examples = self._independent_capture_examples("ravel", examples)
        if len(examples) < self.config.minimum_known_correct:
            return {
                "status": "insufficient_behavior",
                "known_correct_count": len(examples),
            }
        capture = capture_held_out_paths(
            self.ctx, examples, phase="test",
            seed=self.config.seed + 7919,
            **self._capture_kwargs("ravel"))
        confirmation = held_out_trajectory_confirmation(
            capture["rows"], phase="test",
            capture_threshold=self.config.capture_threshold,
            bootstrap_samples=self.config.bootstrap_samples,
            permutation_samples=self.config.permutation_samples,
            alpha=self.config.alpha, seed=self.config.seed + 7927)
        raw_rows = capture.pop("rows")
        rows_used_transiently = bool(capture.pop(
            "raw_rows_materialized_for_runtime", False))
        capture.update({
            "raw_capture_row_count": len(raw_rows),
            "raw_capture_rows_persisted": False,
            "raw_capture_rows_used_transiently": rows_used_transiently,
            "raw_capture_retention": "trajectory_statistics_and_digest_only",
        })
        return {
            "status": confirmation["status"],
            "benchmark": "ravel",
            "capture": capture,
            "confirmation": confirmation,
            "candidate_ranking_performed_on_test": False,
        }

    def _run_scientific_claims(self) -> dict[str, Any]:
        conditional = self.results["conditional_circuit_sufficiency"]
        autonomous = self.results["autonomous_circuit_sufficiency"]
        necessity = self.results["circuit_necessity"]
        circuit_ids = list(conditional["benchmarks"])

        def sufficiency_evidence(result):
            values = [
                result["benchmarks"][benchmark_id].get(
                    "selected_test_faithfulness")
                for benchmark_id in circuit_ids
            ]
            funnel_limited = {
                benchmark_id: bool(
                    (result["benchmarks"][benchmark_id].get(
                        "selected_circuit") or {}).get(
                            "metadata", {}).get("candidate_funnel_limited", True))
                for benchmark_id in circuit_ids
            }
            confidence_intervals = {
                benchmark_id: result["benchmarks"][benchmark_id].get(
                    "selected_test_faithfulness_ci")
                for benchmark_id in circuit_ids
            }
            ready = (
                all(value is not None for value in values)
                and bool(values) and not any(funnel_limited.values())
                and all(
                    interval is not None
                    and interval["ci_low"]
                    >= self.config.circuit_faithfulness_min
                    for interval in confidence_intervals.values()))
            return {
                "status": "ready" if ready else "incomplete",
                "test_faithfulness": min(values) if ready else None,
                "per_benchmark": {
                    benchmark_id: result["benchmarks"][benchmark_id].get(
                        "selected_test_faithfulness")
                    for benchmark_id in circuit_ids
                },
                "candidate_funnel_limited": funnel_limited,
                "faithfulness_confidence_intervals": confidence_intervals,
                "selection_phase": "validation",
                "evaluation_phase": "test",
            }

        necessity_ready = [
            row for row in necessity["benchmarks"].values()
            if row.get("status") == "ready"
        ]
        necessity_evidence = {
            "status": (
                "ready" if necessity_ready
                and len(necessity_ready) == len(necessity["benchmarks"])
                else "incomplete"),
            "mean_margin_drop": necessity.get("mean_margin_drop"),
            "all_significant_after_bh": necessity.get(
                "all_significant_after_bh", False),
            "per_benchmark": necessity["benchmarks"],
        }
        ravel = self.results["ravel_causal_mediation"]
        interchange = dict(ravel.get("test") or {})
        trajectory = self.results["multilayer_trajectory"]
        trajectory_evidence = dict(trajectory.get("confirmation") or {})
        held_out = {
            "status": (
                "ready" if conditional.get("status") == "ready"
                and necessity.get("status") == "ready" else "incomplete"),
            "selection_phase": "validation",
            "evaluation_phase": "test",
            "test_used_for_selection": False,
        }
        evidence = {
            "capture": self.results["operator_localization"],
            "localization": {
                "status": self.results["operator_localization"]["status"],
                "unit": "layer_route_operator_site",
            },
            "necessity": necessity_evidence,
            "conditional_sufficiency": sufficiency_evidence(conditional),
            "autonomous_sufficiency": sufficiency_evidence(autonomous),
            "interchange": interchange,
            "held_out": held_out,
            "spatial_confirmation": self.results["operator_space_structure"],
            "trajectory_confirmation": trajectory_evidence,
        }
        result = evaluate_claims(evidence, self.config)
        result.update({
            "evidence": evidence,
            "benchmark_scope": list(self.benchmark_ids),
            "checkpoint_identity": self.protocol["checkpoint_identity"],
            "single_checkpoint_only": True,
            "official_transformerlens_edge_equivalence_claimed": False,
            "official_ravel_featurizer_equivalence_claimed": False,
        })
        return result
