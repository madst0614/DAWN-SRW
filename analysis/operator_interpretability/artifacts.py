"""Immutable benchmark inputs and protocol-bound analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from analysis.dawn_analysis_storage import (
    exists,
    json_safe,
    join_path,
    open_path,
    read_json,
    write_json_atomic,
)
from analysis.operator_interpretability.benchmark_schema import (
    BenchmarkExample,
    canonical_hash,
    validate_examples,
    validate_manifest,
)
from analysis.operator_interpretability.protocol import PHASES, require_protocol_match


DEFAULT_BENCHMARK_ROOT = (
    "gs://dawn-tpu-data-c4/dataset/operator_interpretability"
)

# Item artifacts are aggregate audit records, never raw capture containers.
# Keeping this fail-loud ceiling small prevents a regression back to multi-GB
# JSON even when a new analysis kind accidentally embeds dense evidence.
MAX_PROTOCOL_BOUND_JSON_BYTES = 2 * 1024 * 1024


def sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open_path(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkBuild:
    root: str
    build_id: str
    build_root: str
    manifest_path: str
    manifest: Mapping[str, Any]
    manifest_hash: str


def resolve_benchmark_build(root: str | None = None) -> BenchmarkBuild:
    root = str(root or DEFAULT_BENCHMARK_ROOT).rstrip("/\\")
    latest_path = join_path(root, "LATEST.json")
    if exists(latest_path):
        latest = read_json(latest_path, None)
        if not isinstance(latest, Mapping):
            raise ValueError(f"invalid benchmark pointer: {latest_path}")
        build_id = str(latest.get("build_id") or "")
        build_path = str(latest.get("build_path") or f"builds/{build_id}")
        if not build_id:
            raise ValueError("benchmark LATEST pointer lacks build_id")
        build_root = (
            build_path if build_path.startswith("gs://")
            else join_path(root, *build_path.replace("\\", "/").split("/")))
        manifest_path = join_path(build_root, "manifest.json")
        if not exists(manifest_path):
            raise FileNotFoundError(
                "benchmark LATEST pointer references a missing manifest: "
                f"pointer={latest_path} manifest={manifest_path}")
        manifest_value = read_json(manifest_path, None)
        if not isinstance(manifest_value, Mapping):
            raise ValueError(f"invalid benchmark manifest: {manifest_path}")
        manifest = validate_manifest(manifest_value)
        manifest_hash = canonical_hash(manifest)
        if manifest_hash != latest.get("manifest_hash"):
            raise ValueError("benchmark LATEST manifest hash mismatch")
    else:
        manifest_path = join_path(root, "manifest.json")
        if not exists(manifest_path):
            raise FileNotFoundError(
                "no immutable interpretability benchmark build is published; "
                f"expected pointer={latest_path} or manifest={manifest_path}. "
                "Prepare and publish one with "
                "scripts/prepare_interpretability_benchmarks.py "
                f"--output-root {root} --benchmarks primary --publish-latest")
        manifest_value = read_json(manifest_path, None)
        if not isinstance(manifest_value, Mapping):
            raise ValueError(f"invalid benchmark manifest: {manifest_path}")
        manifest = validate_manifest(manifest_value)
        build_id = str(manifest["build_id"])
        build_root = root
        manifest_hash = canonical_hash(manifest)
    if str(manifest["build_id"]) != build_id:
        raise ValueError("benchmark pointer/build manifest identity mismatch")
    return BenchmarkBuild(
        root=root,
        build_id=build_id,
        build_root=build_root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_hash=manifest_hash,
    )


def load_benchmark_examples(build: BenchmarkBuild, benchmark_id: str, *,
                            phase: str | None = None) -> list[BenchmarkExample]:
    entry = (build.manifest.get("benchmarks") or {}).get(benchmark_id)
    if not isinstance(entry, Mapping):
        raise FileNotFoundError(
            f"benchmark {benchmark_id!r} is absent from build {build.build_id}")
    if phase is not None and phase not in PHASES:
        raise ValueError(f"unknown benchmark phase: {phase!r}")
    phase_entries = entry.get("phases")
    if not isinstance(phase_entries, Mapping):
        raise ValueError(
            f"benchmark manifest lacks physical phase shards: {benchmark_id}")
    selected_phases = (phase,) if phase is not None else PHASES
    examples: list[BenchmarkExample] = []
    for selected_phase in selected_phases:
        phase_entry = phase_entries.get(selected_phase)
        if not isinstance(phase_entry, Mapping):
            raise ValueError(
                f"benchmark manifest lacks phase={selected_phase}: {benchmark_id}")
        relative = str(phase_entry.get("path") or "")
        path = join_path(
            build.build_root, *relative.replace("\\", "/").split("/"))
        if not exists(path):
            raise FileNotFoundError(path)
        if sha256_path(path) != str(phase_entry.get("sha256") or ""):
            raise ValueError(
                f"benchmark phase shard hash mismatch: "
                f"{benchmark_id}/{selected_phase}")
        phase_count = 0
        with open_path(path, "rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    line = raw_line.decode("utf-8")
                    example = BenchmarkExample.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(
                        f"invalid {benchmark_id} row at "
                        f"{path}:{line_number}") from exc
                if example.benchmark_id != benchmark_id:
                    raise ValueError(
                        f"benchmark id mismatch at {path}:{line_number}")
                if example.phase != selected_phase:
                    raise ValueError(
                        "physical phase shard contains a foreign phase: "
                        f"{path}:{line_number} expected={selected_phase} "
                        f"actual={example.phase}")
                examples.append(example)
                phase_count += 1
        expected_count = int(phase_entry.get("row_count", -1))
        if phase_count != expected_count:
            raise ValueError(
                f"benchmark phase row count mismatch: "
                f"{benchmark_id}/{selected_phase} expected={expected_count} "
                f"actual={phase_count}")
    if phase is None:
        validate_examples(examples)
    elif not examples:
        raise ValueError(f"{benchmark_id} has no rows for phase={phase}")
    return examples


def write_protocol_bound_artifact(store: Any, relative_path: str,
                                  payload: Mapping[str, Any], *,
                                  protocol: Mapping[str, Any]) -> str:
    path = store.path(*relative_path.replace("\\", "/").split("/"))
    record = {
        "protocol": dict(protocol),
        "protocol_hash": canonical_hash(protocol),
        "payload": dict(payload),
    }
    safe_record = json_safe(record)
    encoded_size = len((json.dumps(
        safe_record, indent=2, sort_keys=True,
        ensure_ascii=False) + "\n").encode("utf-8"))
    if encoded_size > MAX_PROTOCOL_BOUND_JSON_BYTES:
        raise RuntimeError(
            "protocol-bound item JSON exceeds the compact-artifact contract: "
            f"path={path} bytes={encoded_size} "
            f"limit={MAX_PROTOCOL_BOUND_JSON_BYTES}; retain only aggregate "
            "metrics, uncertainty, decisions, counts, and digests")
    write_json_atomic(path, safe_record)
    return path


def load_protocol_bound_artifact(path: str, *,
                                 protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    if not exists(path):
        return None
    record = read_json(path, None)
    if not isinstance(record, Mapping):
        raise ValueError(f"invalid analysis artifact: {path}")
    stored_protocol = record.get("protocol")
    if not isinstance(stored_protocol, Mapping):
        raise ValueError(f"analysis artifact lacks protocol: {path}")
    require_protocol_match(stored_protocol, protocol)
    if record.get("protocol_hash") != canonical_hash(stored_protocol):
        raise ValueError(f"analysis artifact protocol hash mismatch: {path}")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"analysis artifact payload is invalid: {path}")
    return dict(payload)
