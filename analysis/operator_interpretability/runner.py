"""End-to-end, protocol-bound operator interpretability execution."""

from __future__ import annotations

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
from analysis.dawn_analysis_storage import write_npz_atomic
from analysis.operator_interpretability.artifacts import (
    load_benchmark_examples,
    resolve_benchmark_build,
    sha256_path,
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
from analysis.operator_interpretability.interchange import score_interchange_rows
from analysis.operator_interpretability.intervention import (
    all_ones_retention_parity,
    evaluate_behavior,
    evaluate_circuit_necessity,
    evaluate_circuit_retention,
    evaluate_native_operator_program_candidate,
    evaluate_operator_interchange,
)
from analysis.operator_interpretability.program import (
    PROGRAM_ALGORITHM_VERSION,
    build_program_schedule,
    capture_schedule_widths,
    compactness_metrics,
    deterministic_mismatch_mapping,
    evaluate_native_program_claims,
    reindex_program_schedule,
    select_validation_program,
    write_program_schedule_artifact,
)
from analysis.operator_interpretability.protocol import (
    CIRCUIT_FRACTIONS,
    ProtocolConfig,
    protocol_record,
    validate_model_version,
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
        return {
            "item_id": item_id,
            "backend": definition["backend"],
            "analysis_kind": kind,
            "benchmark_id": benchmark_id,
            "claim_role": definition["claim_role"],
            "status": item_result.get("status"),
            "result": item_result,
        }

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
        if kind == "operator_localization" and isinstance(
                result.get("ranked_sites"), Sequence):
            ranked_sites = list(result.pop("ranked_sites", ()))
            profiles = dict(result.pop(
                "causal_variable_control_profiles", {}) or {})
            result.update({
                "ranked_site_count": len(ranked_sites),
                "ranked_site_preview": [
                    dict(row) for row in ranked_sites[:5]],
                "ranked_sites_persisted_in_item_json": False,
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
                "causal_variable_profiles_persisted_in_item_json": False,
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
        elif kind == "scientific_claims":
            result = self._compact_scientific_claims(item_id, result)
        output["result"] = result
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
            "strongest_supported_claim": (
                self.results.get("scientific_claims", {}).get(
                    "strongest_supported_claim")
                or self.results.get("native_operator_program", {}).get(
                    "strongest_supported_claim")),
        }
        if self.ctx.is_primary:
            write_protocol_bound_artifact(
                self.store, "backends/operator_interpretability/summary.json",
                summary, protocol=self.protocol)
        return summary

    def _load_examples(self, benchmark_id: str) -> dict[str, list[Any]]:
        if benchmark_id not in self._examples:
            phases = {}
            phase_cap = self.config.max_examples_for(benchmark_id)
            for phase in ("discovery", "validation", "test"):
                values = load_benchmark_examples(
                    self.build, benchmark_id, phase=phase)
                if benchmark_id == "ravel":
                    spec = benchmark_spec("ravel")
                    grouped: dict[tuple[str, str, str], list[Any]] = (
                        defaultdict(list))
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
                            "RAVEL phase cap cannot represent every "
                            f"RAVEL stratum; minimum={len(expected_strata)}")
                    for group in grouped.values():
                        group.sort(key=lambda example: (
                            canonical_hash(example.example_id),
                            example.example_id))
                    selected = []
                    cursors = {key: 0 for key in expected_strata}
                    used_group_ids = {
                        "cause": set(), "isolation": set()}
                    while len(selected) < phase_cap:
                        added = False
                        for key in expected_strata:
                            group = grouped[key]
                            pair_type = key[1]
                            while cursors[key] < len(group):
                                candidate = group[cursors[key]]
                                cursors[key] += 1
                                group_id = str(
                                    candidate.metadata["pair_group_id"])
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
                            "RAVEL prepared phase cannot satisfy the "
                            f"pre-registered runtime cap: phase={phase} "
                            f"requested={phase_cap} available={len(selected)}; "
                            "publish a non-truncated benchmark build")
                    phases[phase] = selected
                else:
                    values.sort(key=lambda example: (
                        canonical_hash(example.example_id), example.example_id))
                    phases[phase] = values[:phase_cap]
            self._examples[benchmark_id] = phases
        return self._examples[benchmark_id]

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
        examples = self._load_examples(benchmark_id)[phase]
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
            for phase, examples in self._load_examples(benchmark_id).items():
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
            output[benchmark_id] = {
                "status": "ready",
                "track": benchmark_spec(benchmark_id).track,
                "phases": phase_results,
            }
        return {"status": "ready", "benchmarks": output}

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
        output["raw_capture_row_count"] = len(rows)
        output["raw_capture_rows_persisted_in_item_json"] = False
        output["raw_capture_retention"] = (
            "program_schedule_binary_artifact_and_capture_digest")
        return output

    @staticmethod
    def _program_mass_label(program_mass: float) -> str:
        return f"{float(program_mass):.2f}".replace(".", "p")

    def _write_program_artifacts(
            self, *, phase: str, program_mass: float,
            schedules: Mapping[str, Any]) -> dict[str, Any]:
        records = {}
        mass_label = self._program_mass_label(program_mass)
        for name, schedule in schedules.items():
            record = write_program_schedule_artifact(
                self.store,
                f"programs/{phase}/mass_{mass_label}/{name}.npz",
                schedule,
                shape=self.shape,
                protocol=self.protocol,
            )
            if record is not None:
                records[str(name)] = record
        return records

    def _write_program_effect_artifact(
            self, *, phase: str, program_mass: float,
            vectors: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.ctx.is_primary:
            return None
        mass_label = self._program_mass_label(program_mass)
        path = self.store.path(
            "programs", phase, f"mass_{mass_label}", "effects.npz")
        arrays = {
            "protocol_hash": np.asarray(canonical_hash(self.protocol)),
            "program_algorithm_version": np.asarray(
                PROGRAM_ALGORITHM_VERSION),
            "program_mass": np.asarray(program_mass, dtype=np.float64),
            **{str(key): np.asarray(value)
               for key, value in vectors.items()},
        }
        write_npz_atomic(path, **arrays)
        return {
            "path": path,
            "sha256": sha256_path(path),
            "program_mass": float(program_mass),
            "phase": str(phase),
            "vector_names": sorted(str(key) for key in vectors),
            "per_example_primary_effects_persisted": True,
            "primary_effects_embedded_in_item_json": False,
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
            mismatch = deterministic_mismatch_mapping(
                discovery_examples, source_schedule,
                seed=self.config.seed + 11003 + mass_index)
            mismatch_schedule = reindex_program_schedule(
                source_schedule, mismatch["donor_indices"],
                recipient_example_ids=[
                    example.example_id for example in discovery_examples],
                prompt_side="mismatched_source", shape=self.shape)
            compactness = compactness_metrics(
                base_schedule, shape=self.shape,
                paired_schedule=source_schedule,
                mismatched_schedule=mismatch_schedule)
            artifacts = self._write_program_artifacts(
                phase="discovery", program_mass=program_mass,
                schedules={
                    "base": base_schedule,
                    "source": source_schedule,
                })
            discovery_artifacts[str(program_mass)] = artifacts
            discovery_curve.append({
                "program_mass": program_mass,
                "median_site_fraction": compactness[
                    "median_site_fraction"],
                "mean_site_fraction": compactness[
                    "mean_site_fraction"],
                "per_route_site_fraction": compactness[
                    "per_route_site_fraction"],
                "same_pair_route_overlap": compactness[
                    "same_pair_route_overlap"],
                "mismatched_route_overlap": compactness[
                    "mismatched_route_overlap"],
            })

        validation_base_capture, validation_source_capture, validation_widths = (
            self._capture_program_phase(
                validation_examples, phase="validation",
                seed=self.config.seed + 20011))
        validation_candidates = []
        validation_artifacts = {}
        for mass_index, program_mass in enumerate(candidate_masses):
            base_schedule = build_program_schedule(
                validation_base_capture, validation_examples,
                shape=self.shape, program_mass=program_mass,
                prompt_side="base", widths=validation_widths)
            source_schedule = build_program_schedule(
                validation_source_capture, validation_examples,
                shape=self.shape, program_mass=program_mass,
                prompt_side="source", widths=validation_widths)
            candidate, controls = evaluate_native_operator_program_candidate(
                self.ctx, validation_examples,
                base_schedule=base_schedule,
                source_schedule=source_schedule,
                shape=self.shape,
                pad_token_id=int(self.tokenizer.pad_token_id),
                config=self.config,
                seed=self.config.seed + 21013 + mass_index * 101,
            )
            effect_vectors = candidate.pop("_effect_vectors")
            effect_artifact = self._write_program_effect_artifact(
                phase="validation", program_mass=program_mass,
                vectors=effect_vectors)
            if effect_artifact is not None:
                candidate["effect_artifact"] = effect_artifact
            validation_candidates.append(candidate)
            validation_artifacts[str(program_mass)] = (
                self._write_program_artifacts(
                    phase="validation", program_mass=program_mass,
                    schedules={
                        "base": base_schedule,
                        "source": source_schedule,
                        **controls,
                    }))

        selection = select_validation_program(
            validation_candidates, config=self.config)
        common = {
            "program_algorithm_version": PROGRAM_ALGORITHM_VERSION,
            "program_mass_candidates": list(candidate_masses),
            "program_position_scope": self.config.program_position_scope,
            "program_routes": list(self.config.program_routes),
            "program_denominator_policy": (
                self.config.program_denominator_policy),
            "program_mismatch_matching": (
                self.config.program_mismatch_matching),
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
                "candidates": validation_candidates,
                "selection": selection,
                "base_capture": self._compact_program_capture(
                    validation_base_capture),
                "source_capture": self._compact_program_capture(
                    validation_source_capture),
                "schedule_artifacts": validation_artifacts,
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
                "strongest_supported_claim": "descriptive_program",
                "checkpoint_specific_claim": True,
                "scientific_claims_primary_modified": False,
            }

        selected_mass = float(selection["selected_program_mass"])
        # The test program path is deliberately unreachable until the
        # validation selection record above has been frozen.
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
                "strongest_supported_claim": "descriptive_program",
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
        test_result, test_controls = evaluate_native_operator_program_candidate(
            self.ctx, test_examples,
            base_schedule=test_base_schedule,
            source_schedule=test_source_schedule,
            shape=self.shape,
            pad_token_id=int(self.tokenizer.pad_token_id),
            config=self.config,
            seed=self.config.seed + 31033,
        )
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
            "median_site_fraction": test_result["compactness"][
                "median_site_fraction"],
            "mean_site_fraction": test_result["compactness"][
                "mean_site_fraction"],
            "per_route_site_fraction": test_result["compactness"][
                "per_route_site_fraction"],
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
            "source_id_replay_flip": test_result["source_id_replay"][
                "paired"]["base_to_source"]["answer_flip_fraction"],
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
