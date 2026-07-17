"""Strict immutable dataset schema for interpretability benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from analysis.operator_interpretability.protocol import PHASES


BENCHMARK_SCHEMA = "dawn_interpretability_benchmark"
BENCHMARK_SCHEMA_VERSION = 2


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
    causal_variable: str
    pair_type: str
    trace_position_base: int
    trace_position_source: int
    input_ids_base: tuple[int, ...]
    input_ids_source: tuple[int, ...]
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
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
            "causal_variable": self.causal_variable,
            "pair_type": self.pair_type,
        }
        missing = [name for name, value in required_text.items() if not str(value)]
        if missing:
            raise ValueError(
                f"benchmark example {self.example_id} has empty fields: {missing}")
        for name, ids in (
            ("input_ids_base", self.input_ids_base),
            ("input_ids_source", self.input_ids_source),
            ("positive_ids", self.positive_ids),
            ("negative_ids", self.negative_ids),
            ("source_positive_ids", self.source_positive_ids),
            ("source_negative_ids", self.source_negative_ids),
        ):
            if not ids or any(int(token) < 0 for token in ids):
                raise ValueError(f"{self.example_id}: invalid {name}")
        if not 0 <= self.trace_position_base < len(self.input_ids_base):
            raise ValueError(f"{self.example_id}: base trace position is out of bounds")
        if not 0 <= self.trace_position_source < len(self.input_ids_source):
            raise ValueError(f"{self.example_id}: source trace position is out of bounds")
        if self.benchmark_id == "ravel":
            if self.pair_type not in {"cause", "isolation"}:
                raise ValueError(
                    f"{self.example_id}: invalid RAVEL pair type")
            if not str(self.metadata.get("pair_group_id") or ""):
                raise ValueError(
                    f"{self.example_id}: RAVEL pair_group_id is required")
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
        )
        kwargs = dict(value)
        for name in tuple_fields:
            kwargs[name] = tuple(int(item) for item in kwargs.get(name, ()))
        kwargs["metadata"] = dict(kwargs.get("metadata") or {})
        return cls(**kwargs).validate()


def validate_examples(examples: Sequence[BenchmarkExample]) -> None:
    if not examples:
        raise ValueError("benchmark build contains no examples")
    seen: set[str] = set()
    phase_counts = {phase: 0 for phase in PHASES}
    for example in examples:
        example.validate()
        if example.example_id in seen:
            raise ValueError(f"duplicate benchmark example id: {example.example_id}")
        seen.add(example.example_id)
        phase_counts[example.phase] += 1
    missing = [phase for phase, count in phase_counts.items() if count == 0]
    if missing:
        raise ValueError(
            "benchmark build must physically separate discovery/validation/test; "
            f"empty={','.join(missing)}")
    ravel_groups: dict[tuple[str, str], list[BenchmarkExample]] = {}
    for example in examples:
        if example.benchmark_id != "ravel":
            continue
        key = (example.phase, str(example.metadata["pair_group_id"]))
        ravel_groups.setdefault(key, []).append(example)
    for (phase, group_id), group in ravel_groups.items():
        if (len(group) != 2
                or {example.pair_type for example in group}
                != {"cause", "isolation"}
                or len({example.causal_variable for example in group}) != 1):
            raise ValueError(
                "RAVEL phase group must contain one cause and one isolation "
                "row for one causal variable: "
                f"phase={phase} group={group_id}")


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
            "build_id", "protocol_id", "tokenizer", "sources", "benchmarks"):
        if not value.get(field_name):
            raise ValueError(f"benchmark manifest is missing {field_name}")
    tokenizer = value["tokenizer"]
    if not tokenizer.get("name") or not tokenizer.get("vocab_hash"):
        raise ValueError("benchmark tokenizer identity/hash is incomplete")
    if tokenizer.get("add_special_tokens") is not False:
        raise ValueError("benchmark tokenizer must use add_special_tokens=False")
    for benchmark_id, entry in value["benchmarks"].items():
        if not entry.get("sha256") or not entry.get("row_count"):
            raise ValueError(f"manifest entry is incomplete: {benchmark_id}")
        counts = entry.get("phase_counts") or {}
        if any(int(counts.get(phase, 0)) <= 0 for phase in PHASES):
            raise ValueError(
                f"manifest benchmark lacks a physical phase: {benchmark_id}")
    return value
