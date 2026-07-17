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

from analysis.dawn_analysis_storage import exists, join_path, open_path
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
)
from analysis.operator_interpretability.eligibility import (
    shared_split_phase,
    tokenize_adapted_pair,
    tokenizer_vocab_hash,
)
from analysis.operator_interpretability.protocol import PROTOCOL_ID


DEFAULT_OUTPUT_ROOT = "gs://dawn-tpu-data-c4/dataset/operator_interpretability"
BLIMP_COMMIT_API = "https://api.github.com/repos/alexwarstadt/blimp/commits/master"
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


def _hf_source_rows(spec: Any) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    info = HfApi().dataset_info(spec.source_dataset, revision="main")
    revision = str(info.sha)
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for split in sorted(set(spec.split_map.values())):
        kwargs = {
            "path": spec.source_dataset,
            "split": split,
            "revision": revision,
        }
        if spec.source_config:
            kwargs["name"] = spec.source_config
        dataset = load_dataset(**kwargs)
        rows[split] = [dict(row) for row in dataset]
    return rows, {
        "kind": "huggingface_dataset",
        "repository": spec.source_dataset,
        "revision": revision,
        "resolved_commit": revision,
    }


def _blimp_source_rows() -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    response = requests.get(BLIMP_COMMIT_API, timeout=60)
    response.raise_for_status()
    revision = str(response.json()["sha"])
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
    return {"train": rows}, {
        "kind": "github_archive",
        "repository": "alexwarstadt/blimp",
        "revision": revision,
        "archive_sha256": _sha256_bytes(payload),
    }


def _counterfact_source_rows(spec: Any) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    response = requests.get(spec.source_dataset, timeout=180)
    response.raise_for_status()
    payload = response.content
    value = response.json()
    if not isinstance(value, list) or not value:
        raise ValueError("official CounterFact payload is not a non-empty list")
    return {"train": [dict(row) for row in value]}, {
        "kind": "published_json",
        "url": spec.source_dataset,
        "content_sha256": _sha256_bytes(payload),
        "revision": _sha256_bytes(payload),
    }


def _source_rows(benchmark_id: str):
    spec = benchmark_spec(benchmark_id)
    if benchmark_id == "blimp":
        return _blimp_source_rows()
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


def _adapted_rows(adapter: Any, row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = adapter(row)
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
        ("one model token", "label_not_single_token"),
        ("token lengths differ", "answer_length_mismatch"),
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
    adapter = getattr(module, spec.adapter_name)
    source_to_phases: dict[str, list[str]] = defaultdict(list)
    for phase, source_split in spec.split_map.items():
        source_to_phases[source_split].append(phase)
    examples: list[BenchmarkExample] = []
    excluded = Counter()
    phase_counts = Counter()
    for source_split, rows in source_rows.items():
        logical_phases = source_to_phases[source_split]
        for row in rows:
            try:
                adapted_values = _adapted_rows(adapter, row)
            except Exception as exc:
                excluded[_exclusion_key("adapter", exc)] += 1
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
            try:
                for adapted, phase in grouped:
                    adapted = dict(adapted)
                    adapted.pop("phase_group_id", None)
                    adapted["example_id"] = (
                        f"{phase}:{adapted['example_id']}")
                    prepared.append(tokenize_adapted_pair(
                        tokenizer, benchmark_id, phase, adapted,
                        max_seq_len=max_seq_len))
            except Exception as exc:
                excluded[_exclusion_key("eligibility", exc)] += len(grouped)
                continue
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
    files: dict[str, Path] = {}
    entries: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for benchmark_id in benchmark_ids:
        print(f"PREPARE benchmark={benchmark_id}", flush=True)
        examples, source, eligibility = prepare_benchmark(
            benchmark_id, tokenizer, max_seq_len=int(args.max_seq_len),
            max_rows_per_phase=args.max_rows_per_phase)
        path = work_root / f"{benchmark_id}.jsonl"
        sha256 = _write_jsonl(path, examples)
        files[benchmark_id] = path
        sources[benchmark_id] = source
        entries[benchmark_id] = {
            "path": path.name,
            "sha256": sha256,
            "row_count": len(examples),
            "phase_counts": dict(Counter(
                example.phase for example in examples)),
            "eligibility": eligibility,
            "track": benchmark_spec(benchmark_id).track,
            "metric": benchmark_spec(benchmark_id).metric,
            "supported_model_versions": list(
                benchmark_spec(benchmark_id).supported_model_versions),
        }
    build_material = {
        "benchmarks": entries,
        "sources": sources,
        "tokenizer_revision": tokenizer_revision,
        "max_seq_len": int(args.max_seq_len),
    }
    build_id = canonical_hash(build_material)[:24]
    manifest = {
        "schema": BENCHMARK_SCHEMA,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "tokenizer": {
            "name": args.tokenizer,
            "requested_revision": args.tokenizer_revision,
            "resolved_revision": tokenizer_revision,
            "vocab_hash": tokenizer_vocab_hash(tokenizer),
            "vocab_size": len(tokenizer.get_vocab()),
            "pad_token_id": int(tokenizer.pad_token_id),
            "add_special_tokens": False,
            "is_fast": bool(tokenizer.is_fast),
        },
        "sources": sources,
        "benchmarks": entries,
        "registry": registry_record(),
        "max_seq_len": int(args.max_seq_len),
    }
    manifest_path = work_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    manifest_hash = canonical_hash(manifest)
    build_root = join_path(
        str(args.output_root).rstrip("/\\"), "builds", build_id)
    for benchmark_id, path in files.items():
        destination = join_path(build_root, path.name)
        if exists(destination):
            raise FileExistsError(
                f"immutable benchmark artifact already exists: {destination}")
        _copy_to(path, destination)
    _copy_to(manifest_path, join_path(build_root, "manifest.json"))
    if args.publish_latest:
        pointer = {
            "build_id": build_id,
            "build_path": f"builds/{build_id}",
            "manifest_hash": manifest_hash,
        }
        latest_local = work_root / "LATEST.json"
        latest_local.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        _copy_to(
            latest_local,
            join_path(str(args.output_root).rstrip("/\\"), "LATEST.json"))
    print(
        "PREPARE COMPLETE "
        f"build_id={build_id} manifest_hash={manifest_hash} root={build_root}",
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
