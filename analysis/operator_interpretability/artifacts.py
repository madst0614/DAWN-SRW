"""Immutable benchmark inputs and protocol-bound analysis artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from analysis.dawn_analysis_storage import (
    exists,
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
from analysis.operator_interpretability.protocol import require_protocol_match


DEFAULT_BENCHMARK_ROOT = (
    "gs://dawn-tpu-data-c4/dataset/operator_interpretability"
)


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
        manifest = validate_manifest(read_json(manifest_path, None) or {})
        manifest_hash = canonical_hash(manifest)
        if manifest_hash != latest.get("manifest_hash"):
            raise ValueError("benchmark LATEST manifest hash mismatch")
    else:
        manifest_path = join_path(root, "manifest.json")
        manifest = validate_manifest(read_json(manifest_path, None) or {})
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
    relative = str(entry.get("path") or f"{benchmark_id}.jsonl")
    path = join_path(build.build_root, *relative.replace("\\", "/").split("/"))
    if not exists(path):
        raise FileNotFoundError(path)
    if sha256_path(path) != str(entry.get("sha256") or ""):
        raise ValueError(f"benchmark shard hash mismatch: {benchmark_id}")
    examples: list[BenchmarkExample] = []
    with open_path(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = BenchmarkExample.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid {benchmark_id} row at {path}:{line_number}") from exc
            if phase is None or example.phase == phase:
                examples.append(example)
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
    write_json_atomic(path, record)
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
