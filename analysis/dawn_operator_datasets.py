"""Immutable prepared-dataset contract for v417x operator analysis.

This module deliberately contains no model evaluation.  It resolves a root
pointer to an immutable build, validates the ``dawn_operator_pair_v2``
contract, and downloads only the shards requested by an analysis host.  The
historical v4171 bucket name is a data-build identity, not a model-version
restriction; v4172 consumes the same tokenized examples after vocabulary
validation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from analysis.dawn_analysis_storage import exists, join_path, open_path, read_json


OPERATOR_SCHEMA = "dawn_operator_pair_v2"
OPERATOR_SCHEMA_VERSION = 2
PROBE_TOKENIZER_NAME = "bert-base-uncased"
PROBE_TOKENIZER_VOCAB_HASH = "4d340ba6df0d188a0bc2fdae2f8f4279427ea92d4c2457be7ae8fa149027d0b4"
DEFAULT_OPERATOR_DATASET_ROOT = os.environ.get(
    "DAWN_OPERATOR_DATASET_ROOT",
    "gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2",
)
DEFAULT_OPERATOR_CACHE_DIR = "/tmp/dawn_operator_analysis_cache"
DATASET_IDS = ("ioi", "blimp", "ravel", "synthetic", "lama", "counterfact")
SUBSET_IDS = ("smoke", "monitor", "trace", "causal")

ARRAY_FIELDS = (
    "context_ids_a", "context_len_a", "context_ids_b", "context_len_b",
    "positive_ids_a", "positive_len_a", "negative_ids_a", "negative_len_a",
    "positive_ids_b", "positive_len_b", "negative_ids_b", "negative_len_b",
    "trace_position_a", "trace_position_b",
    "target_span_start_a", "target_span_end_a",
    "target_span_start_b", "target_span_end_b", "row_index",
)

OPERATOR_DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "ioi": {"title": "IOI operator circuit", "item": "ioi_operator_circuit"},
    "blimp": {"title": "BLiMP grammar operators", "item": "blimp_operator_grammar"},
    "ravel": {"title": "RAVEL operator disentanglement", "item": "ravel_operator_disentanglement"},
    "synthetic": {"title": "Synthetic binding sanity", "item": "synthetic_binding_sanity"},
    "lama": {"title": "LAMA factual recall", "item": "lama_counterfact_factual_recall"},
    "counterfact": {"title": "CounterFact factual recall", "item": "lama_counterfact_factual_recall"},
}

OPERATOR_ANALYSIS_ITEM_IDS = (
    "operator_dataset_manifest", "operator_behavior_eval",
    "ravel_operator_disentanglement", "ioi_operator_circuit",
    "blimp_operator_grammar", "lama_counterfact_factual_recall",
    "synthetic_binding_sanity", "operator_function_reuse",
    "operator_route_specificity", "operator_causal_specificity",
    "operator_analysis_summary",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def operator_dataset_root(root: Optional[str] = None) -> str:
    return str(root or DEFAULT_OPERATOR_DATASET_ROOT).rstrip("/\\")


def join_dataset_path(base: str, *parts: str) -> str:
    return join_path(base, *parts)


@dataclass(frozen=True)
class OperatorDatasetBuild:
    root: str
    build_id: str
    build_root: str
    manifest_path: str
    manifest_hash: str
    manifest: Dict[str, Any]


def _read_required_json(path: str) -> Dict[str, Any]:
    if not exists(path):
        raise FileNotFoundError(path)
    value = read_json(path, None)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def resolve_operator_dataset_build(
    root: Optional[str] = None, *, expected_tokenizer_hash: Optional[str] = None,
) -> OperatorDatasetBuild:
    """Resolve ``LATEST.json`` and verify it names an immutable complete build."""
    root_s = operator_dataset_root(root)
    latest_path = join_path(root_s, "LATEST.json")
    if exists(latest_path):
        latest = _read_required_json(latest_path)
        build_id = str(latest.get("build_id") or "")
        build_rel = str(latest.get("build_path") or f"builds/{build_id}")
        if not build_id or not build_rel:
            raise ValueError(f"LATEST pointer is incomplete: {latest_path}")
        if build_rel.startswith("gs://") or os.path.isabs(build_rel):
            build_root = build_rel
        else:
            build_root = join_path(root_s, *build_rel.replace("\\", "/").split("/"))
        manifest_path = join_path(build_root, "manifest.json")
        manifest = _read_required_json(manifest_path)
        manifest_hash = canonical_hash(manifest)
        expected_hash = str(latest.get("manifest_hash") or "")
        if not expected_hash or manifest_hash != expected_hash:
            raise ValueError(
                f"LATEST manifest hash mismatch: expected={expected_hash!r} actual={manifest_hash}")
    else:
        # An immutable build directory may be passed directly for offline smoke.
        manifest_path = join_path(root_s, "manifest.json")
        manifest = _read_required_json(manifest_path)
        build_id = str(manifest.get("build_id") or "")
        if not build_id:
            raise FileNotFoundError(
                f"No LATEST.json and root is not an immutable build: {root_s}")
        build_root = root_s
        manifest_hash = canonical_hash(manifest)
    if manifest.get("status") != "complete":
        raise ValueError(f"Operator dataset build is not complete: {manifest_path}")
    if manifest.get("schema") != OPERATOR_SCHEMA:
        raise ValueError(f"Unsupported operator schema: {manifest.get('schema')!r}")
    if int(manifest.get("schema_version", -1)) != OPERATOR_SCHEMA_VERSION:
        raise ValueError(f"Unsupported operator schema version: {manifest.get('schema_version')!r}")
    if str(manifest.get("build_id")) != build_id:
        raise ValueError("LATEST build_id disagrees with immutable manifest")
    tokenizer = manifest.get("tokenizer") or {}
    tokenizer_hash = str(tokenizer.get("hash") or "")
    if not tokenizer_hash:
        raise ValueError("Immutable manifest is missing tokenizer hash")
    if expected_tokenizer_hash and tokenizer_hash != expected_tokenizer_hash:
        raise ValueError(
            f"Prepared tokenizer hash mismatch: expected={expected_tokenizer_hash} actual={tokenizer_hash}")
    if tokenizer.get("tokenizer_name") != PROBE_TOKENIZER_NAME:
        raise ValueError(
            f"Prepared tokenizer identity violates probe contract: {tokenizer.get('tokenizer_name')!r}")
    if tokenizer.get("vocab_hash") != PROBE_TOKENIZER_VOCAB_HASH:
        raise ValueError(
            "Prepared tokenizer vocabulary hash violates probe contract: "
            f"{tokenizer.get('vocab_hash')!r}")
    if tokenizer.get("add_special_tokens") is not False:
        raise ValueError("Prepared tokenizer must use add_special_tokens=False")
    return OperatorDatasetBuild(
        root=root_s, build_id=build_id, build_root=build_root,
        manifest_path=manifest_path, manifest_hash=manifest_hash, manifest=manifest,
    )


def operator_dataset_manifest_path(root: Optional[str] = None) -> str:
    return resolve_operator_dataset_build(root).manifest_path


def load_operator_dataset_manifest(root: Optional[str] = None) -> Dict[str, Any]:
    return resolve_operator_dataset_build(root).manifest


def load_dataset_manifest(build: OperatorDatasetBuild, dataset_id: str) -> tuple[str, Dict[str, Any], str]:
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"Unknown operator dataset: {dataset_id}")
    entry = (build.manifest.get("datasets") or {}).get(dataset_id)
    if not isinstance(entry, Mapping):
        raise FileNotFoundError(f"Dataset {dataset_id} is absent from build {build.build_id}")
    rel = str(entry.get("manifest") or f"{dataset_id}/manifest.json")
    path = join_path(build.build_root, *rel.replace("\\", "/").split("/"))
    manifest = _read_required_json(path)
    if manifest.get("schema") != OPERATOR_SCHEMA or int(manifest.get("schema_version", -1)) != OPERATOR_SCHEMA_VERSION:
        raise ValueError(f"Dataset manifest contract mismatch: {path}")
    if manifest.get("dataset") != dataset_id:
        raise ValueError(f"Dataset manifest id mismatch: {path}")
    manifest_hash = canonical_hash(manifest)
    expected_hash = str(entry.get("manifest_hash") or "")
    if not expected_hash or manifest_hash != expected_hash:
        raise ValueError(
            "Dataset manifest hash mismatch: "
            f"{path}: expected={expected_hash!r} actual={manifest_hash}")
    return path, manifest, manifest_hash


def _remote_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open_path(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_subset(
    path: str, *, shard_rows: Mapping[str, int], example_ids: Optional[set[str]] = None,
) -> int:
    payload = _read_required_json(path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Subset entries must be a list: {path}")
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid subset locator: {path}")
        required = {"example_id", "shard", "row_offset", "group", "selection_reason", "seed"}
        if required - set(row):
            raise ValueError(f"Subset locator missing fields {sorted(required - set(row))}: {path}")
        shard = str(row["shard"])
        offset = int(row["row_offset"])
        if shard not in shard_rows or offset < 0 or offset >= shard_rows[shard]:
            raise ValueError(f"Subset locator is out of bounds: {path}: {row}")
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"Duplicate subset example id: {path}: {example_id}")
        if example_ids is not None and example_id not in example_ids:
            raise ValueError(f"Subset example id is absent from shard metadata: {example_id}")
        seen.add(example_id)
    return len(entries)


def _load_npz_and_metadata(arrays_path: str | Path, metadata_path: str | Path) -> tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle if line.strip()]
    missing = set(ARRAY_FIELDS) - set(arrays)
    if missing:
        raise ValueError(f"Prepared shard is missing arrays: {sorted(missing)}")
    n = int(arrays["row_index"].shape[0])
    if len(metadata) != n or any(value.shape[0] != n for value in arrays.values()):
        raise ValueError("Prepared arrays/metadata row alignment mismatch")
    return arrays, metadata


def cache_operator_shard(
    build: OperatorDatasetBuild, dataset_id: str, dataset_manifest: Mapping[str, Any],
    dataset_manifest_hash: str, shard: Mapping[str, Any],
    *, cache_dir: str = DEFAULT_OPERATOR_CACHE_DIR,
) -> tuple[Path, Path]:
    """Download a single checksummed shard atomically, or reuse its local cache."""
    shard_name = Path(str(shard["arrays"])).parent.name
    shard_key = str(shard["sha256"])
    cache_root = (
        Path(cache_dir) / build.build_id / dataset_id /
        f"{dataset_manifest_hash[:12]}-{shard_key[:12]}" / shard_name
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for rel_key, sha_key, filename in (
        ("arrays", "arrays_sha256", "arrays.npz"),
        ("metadata", "metadata_sha256", "rows.jsonl.gz"),
    ):
        local = cache_root / filename
        expected = str(shard[sha_key])
        if not local.exists() or sha256_file(local) != expected:
            remote = join_path(build.build_root, dataset_id, *str(shard[rel_key]).replace("\\", "/").split("/"))
            if not exists(remote):
                raise FileNotFoundError(remote)
            fd, tmp_name = tempfile.mkstemp(prefix=filename + ".", suffix=".part", dir=cache_root)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                with open_path(remote, "rb") as source, tmp.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                actual = sha256_file(tmp)
                if actual != expected:
                    raise ValueError(
                        f"Shard checksum mismatch: {remote}: expected={expected} actual={actual}")
                os.replace(tmp, local)
            finally:
                if tmp.exists():
                    tmp.unlink()
        outputs.append(local)
    return outputs[0], outputs[1]


def validate_operator_dataset_build(
    root: Optional[str] = None, *, required_datasets: Optional[Sequence[str]] = None,
    cache_dir: str = DEFAULT_OPERATOR_CACHE_DIR, checksum_scope: str = "all",
    expected_tokenizer_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Fail-loud contract validation used by ``operator_dataset_manifest``."""
    if checksum_scope not in {"all", "first"}:
        raise ValueError("checksum_scope must be 'all' or 'first'")
    build = resolve_operator_dataset_build(root, expected_tokenizer_hash=expected_tokenizer_hash)
    selected = list(required_datasets or build.manifest.get("datasets_selected") or DATASET_IDS)
    unknown = sorted(set(selected) - set(DATASET_IDS))
    if unknown:
        raise ValueError(f"Unknown required operator datasets: {unknown}")
    results: Dict[str, Any] = {}
    for dataset_id in selected:
        path, manifest, manifest_hash = load_dataset_manifest(build, dataset_id)
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Dataset has no shards: {path}")
        rows = sum(int(shard.get("rows", 0)) for shard in shards)
        if rows != int(manifest.get("prepared_rows", -1)):
            raise ValueError(f"Dataset shard row total mismatch: {path}")
        shard_rows = {Path(str(s["arrays"])).parent.name: int(s["rows"]) for s in shards}
        checked = shards if checksum_scope == "all" else shards[:1]
        first_ids: set[str] = set()
        for index, shard in enumerate(checked):
            arrays_path, metadata_path = cache_operator_shard(
                build, dataset_id, manifest, manifest_hash, shard, cache_dir=cache_dir)
            arrays, metadata = _load_npz_and_metadata(arrays_path, metadata_path)
            if len(metadata) != int(shard["rows"]):
                raise ValueError(f"Shard manifest row mismatch: {arrays_path}")
            if metadata and (
                metadata[0].get("example_id") != shard.get("first_example_id") or
                metadata[-1].get("example_id") != shard.get("last_example_id")
            ):
                raise ValueError(f"Shard example alignment mismatch: {metadata_path}")
            if index == 0:
                first_ids.update(str(row["example_id"]) for row in metadata)
            for name in ("context_len_a", "context_len_b"):
                width = arrays[name.replace("len", "ids")].shape[1]
                if np.any(arrays[name] < 0) or np.any(arrays[name] > width):
                    raise ValueError(f"Context token length out of bounds: {arrays_path}")
            for prefix in ("positive", "negative"):
                for suffix in ("a", "b"):
                    lengths = arrays[f"{prefix}_len_{suffix}"]
                    width = arrays[f"{prefix}_ids_{suffix}"].shape[1]
                    if np.any(lengths < 0) or np.any(lengths > width):
                        raise ValueError(f"Candidate token length out of bounds: {arrays_path}")
            for suffix in ("a", "b"):
                starts = arrays[f"target_span_start_{suffix}"]
                ends = arrays[f"target_span_end_{suffix}"]
                lengths = arrays[f"context_len_{suffix}"]
                valid_spans = ((starts == -1) & (ends == -1)) | (
                    (starts >= 0) & (ends > starts) & (ends <= lengths))
                if not np.all(valid_spans):
                    raise ValueError(f"Target span out of context bounds: {arrays_path}")
        subset_counts: Dict[str, int] = {}
        subsets = manifest.get("subsets") or {}
        for subset_id in SUBSET_IDS:
            rel = subsets.get(subset_id)
            if not rel:
                raise ValueError(f"Dataset manifest missing {subset_id} subset: {path}")
            subset_path = join_path(build.build_root, dataset_id, *str(rel).replace("\\", "/").split("/"))
            # All locators are checked for bounds.  Example-id identity is also checked
            # where the locator points into the opened first shard.
            subset_counts[subset_id] = _validate_subset(
                subset_path, shard_rows=shard_rows, example_ids=None)
        results[dataset_id] = {
            "status": "ready", "manifest": path, "manifest_hash": manifest_hash,
            "rows": rows, "shards": len(shards), "checksummed_shards": len(checked),
            "subsets": subset_counts,
        }
    return {
        "status": "ready", "root": build.root, "build_id": build.build_id,
        "build_root": build.build_root, "manifest": build.manifest_path,
        "manifest_hash": build.manifest_hash,
        "schema": OPERATOR_SCHEMA, "schema_version": OPERATOR_SCHEMA_VERSION,
        "tokenizer": build.manifest["tokenizer"], "datasets": results,
    }


def operator_dataset_preflight(
    root: Optional[str] = None, *, required_datasets: Optional[List[str]] = None,
    verify_artifacts: bool = True, cache_dir: str = DEFAULT_OPERATOR_CACHE_DIR,
) -> Dict[str, Any]:
    return validate_operator_dataset_build(
        root, required_datasets=required_datasets, cache_dir=cache_dir,
        checksum_scope="first" if verify_artifacts else "first")


def load_subset(build: OperatorDatasetBuild, dataset_id: str, subset_id: str) -> List[Dict[str, Any]]:
    if subset_id not in SUBSET_IDS:
        raise ValueError(f"Unknown operator subset: {subset_id}")
    _, manifest, _ = load_dataset_manifest(build, dataset_id)
    rel = (manifest.get("subsets") or {}).get(subset_id)
    if not rel:
        raise FileNotFoundError(f"Dataset {dataset_id} has no {subset_id} subset")
    path = join_path(build.build_root, dataset_id, *str(rel).replace("\\", "/").split("/"))
    payload = _read_required_json(path)
    return list(payload.get("entries") or [])


def iter_operator_rows(
    build: OperatorDatasetBuild, dataset_id: str, *, subset: Optional[str] = None,
    max_examples: Optional[int] = None, cache_dir: str = DEFAULT_OPERATOR_CACHE_DIR,
) -> Iterator[Dict[str, Any]]:
    """Yield aligned prepared rows while caching only referenced shards."""
    if max_examples is not None and int(max_examples) <= 0:
        return
    _, manifest, manifest_hash = load_dataset_manifest(build, dataset_id)
    shards = list(manifest["shards"])
    selected: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if subset:
        selected = {}
        for locator in load_subset(build, dataset_id, subset):
            selected.setdefault(str(locator["shard"]), []).append(locator)
    emitted = 0
    for shard in shards:
        shard_name = Path(str(shard["arrays"])).parent.name
        if selected is not None and shard_name not in selected:
            continue
        arrays_path, metadata_path = cache_operator_shard(
            build, dataset_id, manifest, manifest_hash, shard, cache_dir=cache_dir)
        arrays, metadata = _load_npz_and_metadata(arrays_path, metadata_path)
        locators = selected.get(shard_name, []) if selected is not None else [
            {"row_offset": i} for i in range(len(metadata))]
        for locator in locators:
            offset = int(locator["row_offset"])
            row = {
                "arrays": {key: value[offset] for key, value in arrays.items()},
                "metadata": metadata[offset], "locator": locator,
            }
            if subset and str(row["metadata"].get("example_id")) != str(locator.get("example_id")):
                raise ValueError(f"Subset/shard example alignment mismatch: {locator}")
            yield row
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return


def iter_operator_batches(
    build: OperatorDatasetBuild, dataset_id: str, *, batch_size: int,
    subset: Optional[str] = None, max_examples: Optional[int] = None,
    cache_dir: str = DEFAULT_OPERATOR_CACHE_DIR,
) -> Iterator[Dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: List[Dict[str, Any]] = []
    for row in iter_operator_rows(
        build, dataset_id, subset=subset, max_examples=max_examples,
        cache_dir=cache_dir,
    ):
        batch.append(row)
        if len(batch) == batch_size:
            yield {
                "arrays": {key: np.stack([r["arrays"][key] for r in batch]) for key in ARRAY_FIELDS},
                "metadata": [r["metadata"] for r in batch],
            }
            batch = []
    if batch:
        yield {
            "arrays": {key: np.stack([r["arrays"][key] for r in batch]) for key in ARRAY_FIELDS},
            "metadata": [r["metadata"] for r in batch],
        }


def operator_dataset_paths(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    build = resolve_operator_dataset_build(root)
    out: Dict[str, Dict[str, Any]] = {}
    for dataset_id in build.manifest.get("datasets_selected", []):
        path, manifest, manifest_hash = load_dataset_manifest(build, dataset_id)
        out[dataset_id] = {
            "id": dataset_id, "root": join_path(build.build_root, dataset_id),
            "manifest": path, "manifest_hash": manifest_hash,
            "prepared_rows": manifest.get("prepared_rows"),
            "shards": len(manifest.get("shards") or []),
        }
    return out


def operator_dataset_summary(root: Optional[str] = None) -> Dict[str, Any]:
    build = resolve_operator_dataset_build(root)
    return {
        "status": "resolved", "root": build.root, "build_id": build.build_id,
        "build_root": build.build_root, "manifest": build.manifest_path,
        "manifest_hash": build.manifest_hash, "schema": OPERATOR_SCHEMA,
        "schema_version": OPERATOR_SCHEMA_VERSION,
        "tokenizer": build.manifest.get("tokenizer"),
        "datasets": operator_dataset_paths(build.root),
        "prepare_command": "python3 -u scripts/prepare_operator_analysis_datasets.py",
    }
