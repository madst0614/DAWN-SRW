"""Immutable prepared form for the current interpretability data contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from analysis.operator_interpretability.protocol import PHASES


BENCHMARK_SCHEMA = "dawn_interpretability_source_contract"
BENCHMARK_SCHEMA_VERSION = 1
RAVEL_VARIABLES = ("Continent", "Country", "Language")
RAVEL_SOURCE_COLUMNS = (
    "attribute_counterfactual", "wikipedia_counterfactual")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkExample:
    benchmark_id: str
    example_id: str
    phase: str
    base_prompt: str
    source_prompt: str
    positive_answer: str
    negative_answer: str
    intervention_positive_answer: str
    intervention_negative_answer: str
    causal_variable: str
    pair_type: str
    source_behavior_required: bool
    trace_position_base: int
    trace_position_source: int
    input_ids_base: tuple[int, ...]
    input_ids_source: tuple[int, ...]
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
    intervention_positive_ids: tuple[int, ...]
    intervention_negative_ids: tuple[int, ...]
    source_positive_ids: tuple[int, ...] = ()
    source_negative_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "BenchmarkExample":
        if self.phase not in PHASES:
            raise ValueError(f"invalid phase for {self.example_id}: {self.phase!r}")
        required_text = {
            "benchmark_id": self.benchmark_id,
            "example_id": self.example_id,
            "base_prompt": self.base_prompt,
            "source_prompt": self.source_prompt,
            "positive_answer": self.positive_answer,
            "negative_answer": self.negative_answer,
            "intervention_positive_answer": self.intervention_positive_answer,
            "intervention_negative_answer": self.intervention_negative_answer,
            "causal_variable": self.causal_variable,
            "pair_type": self.pair_type,
        }
        missing = [name for name, value in required_text.items() if not str(value)]
        if missing:
            raise ValueError(
                f"benchmark example {self.example_id} has empty fields: {missing}")
        required_ids = (
            ("input_ids_base", self.input_ids_base),
            ("input_ids_source", self.input_ids_source),
            ("positive_ids", self.positive_ids),
            ("negative_ids", self.negative_ids),
            ("intervention_positive_ids", self.intervention_positive_ids),
            ("intervention_negative_ids", self.intervention_negative_ids),
        )
        for name, ids in required_ids:
            if not ids or any(int(token) < 0 for token in ids):
                raise ValueError(f"{self.example_id}: invalid {name}")
        source_ids = (self.source_positive_ids, self.source_negative_ids)
        if self.source_behavior_required and any(not ids for ids in source_ids):
            raise ValueError(
                f"{self.example_id}: source behavior answers are required")
        if not self.source_behavior_required and any(source_ids):
            raise ValueError(
                f"{self.example_id}: unscored source behavior must not carry labels")
        if not 0 <= self.trace_position_base < len(self.input_ids_base):
            raise ValueError(f"{self.example_id}: base trace position is out of bounds")
        if not 0 <= self.trace_position_source < len(self.input_ids_source):
            raise ValueError(f"{self.example_id}: source trace position is out of bounds")
        if self.benchmark_id == "ravel":
            if self.pair_type not in {"cause", "isolation"}:
                raise ValueError(f"{self.example_id}: invalid RAVEL pair type")
            if self.causal_variable not in RAVEL_VARIABLES:
                raise ValueError(
                    f"{self.example_id}: invalid RAVEL causal variable")
            if not str(self.metadata.get("pair_group_id") or ""):
                raise ValueError(
                    f"{self.example_id}: RAVEL pair_group_id is required")
            source_column = str(self.metadata.get(
                "official_counterfactual_column") or "")
            if source_column not in RAVEL_SOURCE_COLUMNS:
                raise ValueError(
                    f"{self.example_id}: invalid RAVEL source counterfactual")
            base_query = str(self.metadata.get("base_query_attribute") or "")
            expected_pair_type = (
                "cause" if base_query == self.causal_variable else "isolation")
            if self.pair_type != expected_pair_type:
                raise ValueError(
                    f"{self.example_id}: RAVEL cause/isolation assignment drift")
            if self.metadata.get("candidate_score_normalization") != (
                    "mean_log_probability_per_token"):
                raise ValueError(
                    f"{self.example_id}: invalid RAVEL candidate normalization")
            if self.source_behavior_required != (
                    source_column == "attribute_counterfactual"):
                raise ValueError(
                    f"{self.example_id}: RAVEL source behavior contract drift")
            if self.pair_type == "cause":
                expected_answers = (
                    self.negative_answer, self.positive_answer)
            else:
                expected_answers = (
                    self.positive_answer, self.negative_answer)
            if (self.intervention_positive_answer,
                    self.intervention_negative_answer) != expected_answers:
                raise ValueError(
                    f"{self.example_id}: RAVEL intervention target drift")
        elif self.benchmark_id.startswith("mib_"):
            if not self.source_behavior_required:
                raise ValueError(
                    f"{self.example_id}: MIB circuit source must be scored")
            if (self.intervention_positive_ids != self.source_positive_ids
                    or self.intervention_negative_ids
                    != self.source_negative_ids):
                raise ValueError(
                    f"{self.example_id}: MIB intervention/source labels differ")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkExample":
        tuple_fields = (
            "input_ids_base", "input_ids_source", "positive_ids",
            "negative_ids", "source_positive_ids", "source_negative_ids",
            "intervention_positive_ids", "intervention_negative_ids",
        )
        kwargs = dict(value)
        for name in tuple_fields:
            kwargs[name] = tuple(int(item) for item in kwargs.get(name, ()))
        kwargs["metadata"] = dict(kwargs.get("metadata") or {})
        kwargs["source_behavior_required"] = bool(
            kwargs.get("source_behavior_required", True))
        return cls(**kwargs).validate()


def validate_examples(examples: Sequence[BenchmarkExample]) -> None:
    if not examples:
        raise ValueError("benchmark build contains no examples")
    seen: set[str] = set()
    phase_counts = {phase: 0 for phase in PHASES}
    ravel_strata: dict[tuple[str, str, str, str], int] = {}
    for example in examples:
        example.validate()
        if example.example_id in seen:
            raise ValueError(f"duplicate benchmark example id: {example.example_id}")
        seen.add(example.example_id)
        phase_counts[example.phase] += 1
        if example.benchmark_id == "ravel":
            key = (
                example.phase,
                example.causal_variable,
                example.pair_type,
                str(example.metadata["official_counterfactual_column"]),
            )
            ravel_strata[key] = ravel_strata.get(key, 0) + 1
    missing = [phase for phase, count in phase_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "benchmark build must physically separate discovery/validation/test; "
            f"empty={','.join(missing)}")
    if ravel_strata:
        missing_strata = [
            f"{phase}/{variable}/{pair_type}/{source_column}"
            for phase in PHASES
            for variable in RAVEL_VARIABLES
            for pair_type in ("cause", "isolation")
            for source_column in RAVEL_SOURCE_COLUMNS
            if ravel_strata.get(
                (phase, variable, pair_type, source_column), 0) == 0
        ]
        if missing_strata:
            raise ValueError(
                "RAVEL source contract lacks cause/isolation evidence: "
                + ",".join(missing_strata))


def examples_hash(examples: Iterable[BenchmarkExample]) -> str:
    return canonical_hash([example.to_dict() for example in examples])


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    if value.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError(f"unsupported benchmark schema: {value.get('schema')!r}")
    if int(value.get("schema_version", -1)) != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark schema version: {value.get('schema_version')!r}")
    if value.get("status") != "complete":
        raise ValueError("benchmark manifest is not complete")
    for field_name in (
            "build_id", "protocol_id", "tokenizer", "sources", "benchmarks",
            "registry"):
        if not value.get(field_name):
            raise ValueError(f"benchmark manifest is missing {field_name}")
    tokenizer = value["tokenizer"]
    if not tokenizer.get("name") or not tokenizer.get("vocab_hash"):
        raise ValueError("benchmark tokenizer identity/hash is incomplete")
    if tokenizer.get("add_special_tokens") is not False:
        raise ValueError("benchmark tokenizer must use add_special_tokens=False")
    for benchmark_id, entry in value["benchmarks"].items():
        source = value["sources"].get(benchmark_id)
        registry = value["registry"].get(benchmark_id)
        if not isinstance(source, Mapping) or not isinstance(registry, Mapping):
            raise ValueError(
                f"manifest source/registry entry is missing: {benchmark_id}")
        if dict(source.get("phase_splits") or {}) != dict(
                registry.get("phase_splits") or {}):
            raise ValueError(
                f"manifest source split mapping drift: {benchmark_id}")
        pinned_revision = registry.get("source_revision")
        resolved_revision = source.get("resolved_commit") or source.get(
            "revision")
        if pinned_revision and resolved_revision != pinned_revision:
            raise ValueError(
                f"manifest source revision drift: {benchmark_id}")
        identity_audit = source.get("identity_audit")
        if not isinstance(identity_audit, Mapping):
            raise ValueError(
                f"manifest source identity audit is missing: {benchmark_id}")
        if identity_audit.get(
                "audited_before_selection_and_tokenization") is not True:
            raise ValueError(
                f"manifest source identity audit order is invalid: {benchmark_id}")
        if (int(identity_audit.get("within_split_duplicates", -1)) != 0
                or int(identity_audit.get("cross_split_overlaps", -1)) != 0):
            raise ValueError(
                f"manifest source identity audit found leakage: {benchmark_id}")
        expected_source_rows = dict(registry.get("expected_split_rows") or {})
        audited_source_rows = dict(
            identity_audit.get("split_row_counts") or {})
        if audited_source_rows != expected_source_rows:
            raise ValueError(
                f"manifest source identity counts drift: {benchmark_id}")
        phase_entries = entry.get("phases") or {}
        if set(phase_entries) != set(PHASES):
            raise ValueError(
                f"manifest benchmark phases are incomplete: {benchmark_id}")
        total = 0
        for phase in PHASES:
            phase_entry = phase_entries[phase]
            if (not phase_entry.get("path") or not phase_entry.get("sha256")
                    or int(phase_entry.get("row_count", 0)) <= 0):
                raise ValueError(
                    f"manifest phase entry is incomplete: {benchmark_id}/{phase}")
            total += int(phase_entry["row_count"])
            if int((entry.get("phase_counts") or {}).get(phase, -1)) != int(
                    phase_entry["row_count"]):
                raise ValueError(
                    f"manifest phase count is inconsistent: "
                    f"{benchmark_id}/{phase}")
        if total != int(entry.get("row_count", -1)):
            raise ValueError(
                f"manifest benchmark row total is inconsistent: {benchmark_id}")
    return value
