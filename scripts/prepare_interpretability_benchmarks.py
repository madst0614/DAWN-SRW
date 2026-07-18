#!/usr/bin/env python3
"""Prepare immutable official benchmarks for DAWN operator interpretability."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from analysis.dawn_analysis_storage import (
    exists,
    join_path,
    open_path,
    read_json,
    write_json_atomic,
)
from analysis.operator_interpretability.benchmark_registry import (
    BENCHMARK_SPECS,
    PRIMARY_BENCHMARK_IDS,
    benchmark_spec,
    registry_record,
)
from analysis.operator_interpretability.benchmark_schema import (
    BENCHMARK_SCHEMA,
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkExample,
    canonical_hash,
    validate_examples,
    validate_manifest,
)
from analysis.operator_interpretability.benchmarks.common import AdapterOutput
from analysis.operator_interpretability.eligibility import (
    BenchmarkEligibilityError,
    shared_split_phase,
    tokenize_adapted_pair,
    tokenizer_vocab_hash,
)
from analysis.operator_interpretability.protocol import PHASES, PROTOCOL_ID


DEFAULT_OUTPUT_ROOT = "gs://dawn-tpu-data-c4/dataset/operator_interpretability"
BLIMP_ARCHIVE = "https://github.com/alexwarstadt/blimp/archive/{revision}.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare only published MIB/RAVEL/BLiMP/CounterFact inputs.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument(
        "--benchmarks", default="primary",
        help="primary, all, or comma-separated canonical benchmark ids")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-rows-per-phase", type=int, default=None)
    parser.add_argument("--publish-latest", action="store_true")
    return parser.parse_args()


def _selected(value: str) -> tuple[str, ...]:
    key = str(value).strip().lower()
    if key == "primary":
        return PRIMARY_BENCHMARK_IDS
    if key == "all":
        return tuple(BENCHMARK_SPECS)
    result = tuple(part.strip().lower() for part in key.split(",") if part.strip())
    unknown = [item for item in result if item not in BENCHMARK_SPECS]
    if unknown:
        raise ValueError(f"unknown benchmarks: {','.join(unknown)}")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open_path(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity_audit(
        benchmark_id: str,
        split_rows: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split, rows in split_rows.items():
        local: set[str] = set()
        count = 0
        for row in rows:
            digest = canonical_hash(dict(row))
            if digest in local:
                raise ValueError(
                    f"duplicate official source row within "
                    f"{benchmark_id}/{split}: {digest}")
            previous_split = seen.get(digest)
            if previous_split is not None:
                raise ValueError(
                    "official source split leakage: "
                    f"benchmark={benchmark_id} first={previous_split} "
                    f"second={split} row_hash={digest}")
            local.add(digest)
            seen[digest] = split
            count += 1
        split_counts[split] = count
    return {
        "algorithm": "sha256_canonical_full_source_row",
        "split_row_counts": split_counts,
        "unique_row_count": len(seen),
        "within_split_duplicates": 0,
        "cross_split_overlaps": 0,
        "audited_before_selection_and_tokenization": True,
    }


def _python_feature_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    field_types: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for key, value in row.items():
            field_types[str(key)].add(type(value).__name__)
    return {
        key: sorted(values) for key, values in sorted(field_types.items())
    }


def _hf_source_rows(spec: Any) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    if not spec.source_revision:
        raise ValueError(f"{spec.benchmark_id} lacks a pinned source revision")
    info = HfApi().dataset_info(
        spec.source_dataset, revision=spec.source_revision)
    revision = str(info.sha)
    if revision != spec.source_revision:
        raise ValueError(
            f"resolved source revision drift for {spec.benchmark_id}: "
            f"expected={spec.source_revision} actual={revision}")
    rows: dict[str, list[Mapping[str, Any]]] = {}
    official_datasets: dict[str, Any] = {}
    split_audit: dict[str, Any] = {}
    for split in sorted(set(spec.phase_splits.values())):
        kwargs = {
            "path": spec.source_dataset,
            "split": split,
            "revision": revision,
        }
        if spec.source_config:
            kwargs["name"] = spec.source_config
        dataset = load_dataset(**kwargs)
        official_datasets[split] = dataset
        expected_rows = int(spec.expected_split_rows[split])
        if len(dataset) != expected_rows:
            raise ValueError(
                f"official split row-count drift for {spec.benchmark_id}/{split}: "
                f"expected={expected_rows} actual={len(dataset)}")
        missing_columns = sorted(
            set(spec.required_columns) - set(dataset.column_names))
        if missing_columns:
            raise ValueError(
                f"official source schema drift for {spec.benchmark_id}/{split}: "
                f"missing={','.join(missing_columns)}")
        selected = dataset
        if spec.source_row_limit is not None and len(selected) > int(
                spec.source_row_limit):
            if spec.source_shuffle_seed is None:
                raise ValueError(
                    f"{spec.benchmark_id} source limit lacks a shuffle seed")
            selected = selected.shuffle(
                seed=int(spec.source_shuffle_seed)).select(
                    range(int(spec.source_row_limit)))
        rows[split] = [dict(row) for row in selected]
        split_audit[split] = {
            "official_rows": len(dataset),
            "selected_rows": len(selected),
            "columns": list(dataset.column_names),
            "feature_schema_hash": canonical_hash(dataset.features.to_dict()),
        }
    return rows, {
        "kind": "huggingface_dataset",
        "repository": spec.source_dataset,
        "config": spec.source_config,
        "revision": revision,
        "resolved_commit": revision,
        "phase_splits": dict(spec.phase_splits),
        "counterfactual_columns": list(spec.counterfactual_columns),
        "split_audit": split_audit,
        "identity_audit": _source_identity_audit(
            spec.benchmark_id, official_datasets),
        "selection": {
            "row_limit": spec.source_row_limit,
            "shuffle_seed": spec.source_shuffle_seed,
            "official_splits_preserved": True,
        },
        "reference": {
            "repository": spec.reference_repository,
            "revision": spec.reference_revision,
            "path": spec.reference_path,
        },
    }


def _blimp_source_rows(
        spec: Any) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    revision = str(spec.source_revision or "")
    if not revision:
        raise ValueError("BLiMP source revision is not pinned")
    archive_response = requests.get(
        BLIMP_ARCHIVE.format(revision=revision), timeout=180)
    archive_response.raise_for_status()
    payload = archive_response.content
    rows: list[Mapping[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = sorted(
            name for name in archive.namelist()
            if "/data/" in name and name.endswith(".jsonl"))
        if len(names) != 67:
            raise ValueError(f"official BLiMP archive expected 67 files, got {len(names)}")
        for name in names:
            phenomenon = Path(name).stem
            with archive.open(name, "r") as handle:
                file_rows = [
                    json.loads(line.decode("utf-8"))
                    for line in handle if line.strip()
                ]
            if len(file_rows) != 1000:
                raise ValueError(f"BLiMP {name} expected 1000 rows")
            for row in file_rows:
                row.setdefault("phenomenon", phenomenon)
                rows.append(row)
    if len(rows) != 67_000:
        raise ValueError(f"official BLiMP expected 67000 rows, got {len(rows)}")
    spec = benchmark_spec("blimp")
    missing_columns = sorted(
        set(spec.required_columns) - set(rows[0]))
    if missing_columns:
        raise ValueError(
            "official BLiMP schema drift: "
            f"missing={','.join(missing_columns)}")
    feature_schema = _python_feature_schema(rows)
    return {"train": rows}, {
        "kind": "github_archive",
        "repository": "alexwarstadt/blimp",
        "revision": revision,
        "archive_sha256": _sha256_bytes(payload),
        "phase_splits": dict(benchmark_spec("blimp").phase_splits),
        "split_audit": {
            "train": {
                "official_rows": len(rows), "selected_rows": len(rows),
                "columns": sorted({key for row in rows for key in row}),
                "feature_schema_hash": canonical_hash(feature_schema),
            },
        },
        "identity_audit": _source_identity_audit(
            "blimp", {"train": rows}),
        "reference": {
            "repository": spec.reference_repository,
            "revision": spec.reference_revision,
            "path": spec.reference_path,
        },
    }


def _counterfact_source_rows(spec: Any) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    response = requests.get(spec.source_dataset, timeout=180)
    response.raise_for_status()
    payload = response.content
    content_sha256 = _sha256_bytes(payload)
    if content_sha256 != spec.source_revision:
        raise ValueError(
            "official CounterFact content drift: "
            f"expected={spec.source_revision} actual={content_sha256}")
    value = response.json()
    if not isinstance(value, list) or not value:
        raise ValueError("official CounterFact payload is not a non-empty list")
    expected_rows = int(spec.expected_split_rows["train"])
    if len(value) != expected_rows:
        raise ValueError(
            "official CounterFact row-count drift: "
            f"expected={expected_rows} actual={len(value)}")
    missing_columns = sorted(
        set(spec.required_columns) - set(value[0]))
    if missing_columns:
        raise ValueError(
            "official CounterFact schema drift: "
            f"missing={','.join(missing_columns)}")
    rows = [dict(row) for row in value]
    feature_schema = _python_feature_schema(rows)
    return {"train": rows}, {
        "kind": "published_json",
        "url": spec.source_dataset,
        "content_sha256": content_sha256,
        "revision": content_sha256,
        "phase_splits": dict(spec.phase_splits),
        "split_audit": {
            "train": {
                "official_rows": len(value), "selected_rows": len(value),
                "columns": sorted(value[0]),
                "feature_schema_hash": canonical_hash(feature_schema),
            },
        },
        "identity_audit": _source_identity_audit(
            "counterfact", {"train": rows}),
        "reference": {
            "repository": spec.reference_repository,
            "revision": spec.reference_revision,
            "path": spec.reference_path,
        },
    }


def _source_rows(benchmark_id: str):
    spec = benchmark_spec(benchmark_id)
    if benchmark_id == "blimp":
        return _blimp_source_rows(spec)
    if benchmark_id == "counterfact":
        return _counterfact_source_rows(spec)
    return _hf_source_rows(spec)


def _logical_phase(example_id: str, logical_phases: Sequence[str]) -> str:
    phases = tuple(sorted(logical_phases))
    if len(phases) == 1:
        return phases[0]
    if set(phases) == {"discovery", "validation"}:
        return shared_split_phase(example_id, include_test=False)
    if set(phases) == {"discovery", "validation", "test"}:
        return shared_split_phase(example_id, include_test=True)
    raise ValueError(f"unsupported shared split mapping: {phases}")


def _adapted_rows(
        adapter: Any, row: Mapping[str, Any],
        excluded: Counter) -> list[Mapping[str, Any]]:
    value = adapter(row)
    if isinstance(value, AdapterOutput):
        excluded.update(value.excluded)
        return [dict(item) for item in value.rows]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and all(isinstance(item, Mapping) for item in value):
        return list(value)
    raise ValueError("benchmark adapter must return a mapping or sequence of mappings")


def _exclusion_key(stage: str, exc: Exception) -> str:
    """Keep manifests countable without embedding per-example text or ids."""
    message = str(exc).lower()
    categories = (
        ("max_seq_len", "sequence_too_long"),
        ("no shared token prefix", "minimal_pair_no_shared_prefix"),
        ("lacks divergent continuations", (
            "minimal_pair_no_divergent_continuation")),
        ("one model token", "label_not_single_token"),
        ("token lengths differ", "answer_length_mismatch"),
        ("tokenize identically", "identical_candidate_tokens"),
        ("trace anchor", "trace_anchor_invalid"),
        ("counterfactual", "counterfactual_contract_invalid"),
        ("answerkey", "answer_key_invalid"),
        ("missing", "official_field_missing"),
    )
    category = next(
        (name for fragment, name in categories if fragment in message),
        "contract_invalid",
    )
    return f"{stage}:{type(exc).__name__}:{category}"


def prepare_benchmark(benchmark_id: str, tokenizer: Any, *,
                      max_seq_len: int,
                      max_rows_per_phase: int | None):
    spec = benchmark_spec(benchmark_id)
    source_rows, source_record = _source_rows(benchmark_id)
    module = importlib.import_module(spec.adapter_module)
    adapter = getattr(module, "adapt_rows")
    source_to_phases: dict[str, list[str]] = defaultdict(list)
    for phase, source_split in spec.phase_splits.items():
        source_to_phases[source_split].append(phase)
    examples: list[BenchmarkExample] = []
    excluded = Counter()
    phase_counts = Counter()
    for source_split, rows in source_rows.items():
        logical_phases = source_to_phases[source_split]
        for row in rows:
            adapted_values = _adapted_rows(adapter, row, excluded)
            if not adapted_values:
                continue
            grouped: list[tuple[Mapping[str, Any], str]] = []
            for adapted in adapted_values:
                phase_key = str(adapted.get(
                    "phase_group_id", adapted["example_id"]))
                grouped.append((
                    adapted, _logical_phase(phase_key, logical_phases)))
            group_phase_counts = Counter(phase for _, phase in grouped)
            if max_rows_per_phase and any(
                    phase_counts[phase] + count > max_rows_per_phase
                    for phase, count in group_phase_counts.items()):
                for phase, count in group_phase_counts.items():
                    excluded[f"cap:{phase}"] += count
                continue
            prepared: list[BenchmarkExample] = []
            for adapted, phase in grouped:
                adapted = dict(adapted)
                adapted.pop("phase_group_id", None)
                try:
                    prepared.append(tokenize_adapted_pair(
                        tokenizer, benchmark_id, phase, adapted,
                        max_seq_len=max_seq_len))
                except BenchmarkEligibilityError as exc:
                    excluded[_exclusion_key("eligibility", exc)] += 1
            examples.extend(prepared)
            phase_counts.update(example.phase for example in prepared)
    examples.sort(key=lambda item: (item.phase, item.example_id))
    validate_examples(examples)
    return examples, source_record, {
        "accepted": len(examples),
        "phase_counts": dict(phase_counts),
        "excluded": dict(sorted(excluded.items())),
        "source_rows": {key: len(value) for key, value in source_rows.items()},
    }


def _write_jsonl(path: Path, examples: Sequence[BenchmarkExample]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for example in examples:
            line = (
                json.dumps(example.to_dict(), sort_keys=True, ensure_ascii=False)
                + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def _copy_to(path: Path, destination: str) -> None:
    with path.open("rb") as source, open_path(destination, "wb") as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)


def _publish_immutable_file(path: Path, destination: str,
                            expected_sha256: str) -> None:
    if exists(destination):
        actual_sha256 = _sha256_path(destination)
        if actual_sha256 != expected_sha256:
            raise FileExistsError(
                "immutable benchmark artifact collision: "
                f"path={destination} expected={expected_sha256} "
                f"actual={actual_sha256}")
        print(f"PREPARE REUSE path={destination}", flush=True)
        return
    _copy_to(path, destination)
    actual_sha256 = _sha256_path(destination)
    if actual_sha256 != expected_sha256:
        raise IOError(
            "published benchmark artifact hash mismatch: "
            f"path={destination} expected={expected_sha256} "
            f"actual={actual_sha256}")


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("created_at", None)
    return value


def main() -> int:
    args = parse_args()
    benchmark_ids = _selected(args.benchmarks)
    if args.max_seq_len <= 1:
        raise ValueError("--max-seq-len must be > 1")
    tokenizer_info = HfApi().model_info(
        args.tokenizer, revision=args.tokenizer_revision)
    tokenizer_revision = str(tokenizer_info.sha)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=tokenizer_revision, use_fast=True)
    work_root = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="dawn-interpretability-"))
    work_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Path]] = {}
    entries: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for benchmark_id in benchmark_ids:
        print(f"PREPARE benchmark={benchmark_id}", flush=True)
        examples, source, eligibility = prepare_benchmark(
            benchmark_id, tokenizer, max_seq_len=int(args.max_seq_len),
            max_rows_per_phase=args.max_rows_per_phase)
        phase_files: dict[str, Path] = {}
        phase_entries: dict[str, Any] = {}
        for phase in PHASES:
            phase_examples = [
                example for example in examples if example.phase == phase]
            if not phase_examples:
                raise ValueError(
                    f"prepared benchmark has no {phase} rows: {benchmark_id}")
            relative_path = f"{benchmark_id}/{phase}.jsonl"
            path = work_root / benchmark_id / f"{phase}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            sha256 = _write_jsonl(path, phase_examples)
            phase_files[phase] = path
            phase_entries[phase] = {
                "path": relative_path,
                "sha256": sha256,
                "row_count": len(phase_examples),
            }
        files[benchmark_id] = phase_files
        sources[benchmark_id] = source
        entries[benchmark_id] = {
            "row_count": len(examples),
            "phase_counts": dict(Counter(
                example.phase for example in examples)),
            "phases": phase_entries,
            "eligibility": eligibility,
            "track": benchmark_spec(benchmark_id).track,
            "metric": benchmark_spec(benchmark_id).metric,
            "supported_model_versions": list(
                benchmark_spec(benchmark_id).supported_model_versions),
        }
    tokenizer_record = {
        "name": args.tokenizer,
        "requested_revision": args.tokenizer_revision,
        "resolved_revision": tokenizer_revision,
        "vocab_hash": tokenizer_vocab_hash(tokenizer),
        "vocab_size": len(tokenizer.get_vocab()),
        "pad_token_id": int(tokenizer.pad_token_id),
        "add_special_tokens": False,
        "is_fast": bool(tokenizer.is_fast),
    }
    registry = registry_record()
    build_material = {
        "schema": BENCHMARK_SCHEMA,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "benchmarks": entries,
        "sources": sources,
        "registry": registry,
        "tokenizer": tokenizer_record,
        "max_seq_len": int(args.max_seq_len),
    }
    build_id = canonical_hash(build_material)[:24]
    manifest: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "tokenizer": tokenizer_record,
        "sources": sources,
        "benchmarks": entries,
        "registry": registry,
        "max_seq_len": int(args.max_seq_len),
    }
    manifest = validate_manifest(manifest)
    build_root = join_path(
        str(args.output_root).rstrip("/\\"), "builds", build_id)
    manifest_destination = join_path(build_root, "manifest.json")
    if exists(manifest_destination):
        existing = read_json(manifest_destination, None)
        if not isinstance(existing, Mapping):
            raise ValueError(
                f"invalid existing benchmark manifest: {manifest_destination}")
        existing_manifest = validate_manifest(existing)
        existing_identity_hash = canonical_hash(
            _manifest_identity(existing_manifest))
        requested_identity_hash = canonical_hash(_manifest_identity(manifest))
        if existing_identity_hash != requested_identity_hash:
            raise FileExistsError(
                "immutable benchmark build identity collision: "
                f"build_id={build_id} manifest={manifest_destination}")
        manifest = existing_manifest
        print(
            f"PREPARE REUSE build_id={build_id} manifest={manifest_destination}",
            flush=True)
    else:
        for benchmark_id, phase_files in files.items():
            for phase, path in phase_files.items():
                phase_entry = entries[benchmark_id]["phases"][phase]
                _publish_immutable_file(
                    path,
                    join_path(
                        build_root,
                        *str(phase_entry["path"]).replace(
                            "\\", "/").split("/"),
                    ),
                    str(phase_entry["sha256"]),
                )
        write_json_atomic(manifest_destination, manifest)

    for benchmark_id, entry in manifest["benchmarks"].items():
        for phase in PHASES:
            phase_entry = entry["phases"][phase]
            shard_path = join_path(
                build_root,
                *str(phase_entry["path"]).replace("\\", "/").split("/"),
            )
            if not exists(shard_path):
                raise FileNotFoundError(
                    "benchmark manifest references a missing phase shard: "
                    f"{shard_path}")
            if _sha256_path(shard_path) != str(phase_entry["sha256"]):
                raise ValueError(
                    f"benchmark phase shard hash mismatch: {benchmark_id}/{phase}")

    manifest_hash = canonical_hash(manifest)
    if args.publish_latest:
        pointer = {
            "build_id": build_id,
            "build_path": f"builds/{build_id}",
            "manifest_hash": manifest_hash,
        }
        write_json_atomic(
            join_path(str(args.output_root).rstrip("/\\"), "LATEST.json"),
            pointer,
        )
    print(
        "PREPARE COMPLETE "
        f"build_id={build_id} manifest_hash={manifest_hash} root={build_root}",
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
