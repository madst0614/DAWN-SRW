#!/usr/bin/env python3
"""Probe public/generated operator-analysis sources and freeze adapter contracts.

This script deliberately prepares only a few human-readable rows.  It never
publishes GCS manifests, processes complete corpora, restores a checkpoint, or
runs model analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".generated" / "operator_probe_deps"
for candidate in (PROJECT_ROOT, LOCAL_DEPS):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

RAVEL_TGZ_URL = "https://raw.githubusercontent.com/explanare/ravel/main/data.tgz"
RAVEL_HF_TREE = "https://huggingface.co/api/datasets/hij/ravel/tree/main?recursive=1"
RAVEL_HF_RESOLVE = "https://huggingface.co/datasets/hij/ravel/resolve/main"
BLIMP_HF_TREE = "https://huggingface.co/api/datasets/nyu-mll/blimp/tree/main?recursive=1"
BLIMP_HF_RESOLVE = "https://huggingface.co/datasets/nyu-mll/blimp/resolve/main"
LAMA_ZIP_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"

IOI_TEMPLATES = [
    "{name_a} and {name_b} went to the {place}. {name_a} gave a {object} to",
    "After {name_a} and {name_b} visited the {place}, {name_a} handed the {object} to",
    "{name_a} saw {name_b} at the {place}. {name_a} passed the {object} to",
    "At the {place}, {name_a} and {name_b} talked. {name_a} gave the {object} to",
    "{name_a} brought {name_b} to the {place}. {name_a} offered a {object} to",
    "{name_a} and {name_b} were near the {place}. {name_a} sent the {object} to",
    "Before leaving the {place}, {name_a} gave {name_b} the {object}. Then {name_a} waved to",
    "{name_a} met {name_b} by the {place}. {name_a} threw the {object} to",
    "{name_a} and {name_b} entered the {place}. {name_a} showed the {object} to",
    "While {name_a} and {name_b} waited at the {place}, {name_a} gave a {object} to",
    "{name_a} found {name_b} in the {place}. {name_a} delivered the {object} to",
    "{name_a} and {name_b} walked past the {place}. {name_a} loaned the {object} to",
    "Near the {place}, {name_a} and {name_b} stopped. {name_a} returned the {object} to",
    "{name_a} invited {name_b} into the {place}. {name_a} gave the {object} to",
    "{name_a} and {name_b} worked at the {place}. {name_a} handed a {object} to",
]
IOI_NAMES = [
    "John", "Mary", "Alice", "Bob", "Sarah", "Tom", "Emma", "James",
    "Laura", "Michael", "Linda", "Robert", "Emily", "David", "Julia", "Kevin",
]
IOI_PLACES = ["store", "garden", "school", "office", "library", "park", "museum", "station"]
IOI_OBJECTS = ["book", "drink", "bag", "letter", "key", "ball", "ticket", "phone"]
SYNTHETIC_COLORS = ["red", "blue", "green", "yellow", "purple", "white", "black", "orange"]
SYNTHETIC_KEYS = ["key_01", "key_02", "key_03", "key_04", "key_05", "key_06", "key_07", "key_08"]
SYNTHETIC_ENTITIES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]


def generate_ioi_rows(n_examples: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(4166)
    for idx in range(n_examples):
        name_a, name_b = rng.sample(IOI_NAMES, 2)
        place = rng.choice(IOI_PLACES)
        obj = rng.choice(IOI_OBJECTS)
        template_id = idx % len(IOI_TEMPLATES)
        template = IOI_TEMPLATES[template_id]
        yield {
            "id": f"ioi-{idx:06d}", "template_id": template_id,
            "clean_prompt": template.format(name_a=name_a, name_b=name_b, place=place, object=obj),
            "corrupt_prompt": template.format(name_a=name_b, name_b=name_a, place=place, object=obj),
            "correct": name_b, "distractor": name_a,
            "task_variable": "indirect_object", "target_position": "final_prediction_token",
        }


def generate_synthetic_rows(n_examples: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(4167)
    for idx in range(n_examples):
        entities = rng.sample(SYNTHETIC_ENTITIES, 3)
        colors = rng.sample(SYNTHETIC_COLORS, 3)
        keys = rng.sample(SYNTHETIC_KEYS, 3)
        target_idx = idx % 3
        target = entities[target_idx]
        if idx % 2 == 0:
            facts = " ".join(f"{ent} is {color}." for ent, color in zip(entities, colors))
            prompt, answer, attribute = f"{facts} The color of {target} is", colors[target_idx], "color"
        else:
            facts = " ".join(f"{ent} has {key}." for ent, key in zip(entities, keys))
            prompt, answer, attribute = f"{facts} {target}'s key is", keys[target_idx], "key"
        yield {
            "id": f"synthetic-{idx:06d}", "prompt": prompt, "answer": answer,
            "entity": target, "attribute": attribute,
            "task_variable": "entity_attribute_binding", "target_position": "final_prediction_token",
        }

DATASETS = ("ravel", "blimp", "lama", "counterfact", "ioi", "synthetic")
SCHEMA_NAME = "dawn_operator_pair_v2_candidate"
SCHEMA_VERSION = 2
ARRAY_FIELDS = (
    "context_ids_a", "context_len_a", "context_ids_b", "context_len_b",
    "positive_ids_a", "positive_len_a", "negative_ids_a", "negative_len_a",
    "positive_ids_b", "positive_len_b", "negative_ids_b", "negative_len_b",
    "trace_position_a", "trace_position_b",
    "target_span_start_a", "target_span_end_a",
    "target_span_start_b", "target_span_end_b", "row_index",
)
METADATA_FIELDS = (
    "example_id", "pair_id", "dataset", "split", "phenomenon", "relation",
    "group_id", "source_id", "score_mode", "trace_semantics", "text_a",
    "text_b", "positive_text_a", "negative_text_a", "positive_text_b",
    "negative_text_b",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "as_py"):
        return json_safe(value.as_py())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def nested_key_paths(value: Any, prefix: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.append(path)
            out.extend(nested_key_paths(child, path))
    elif isinstance(value, list) and value:
        out.extend(nested_key_paths(value[0], f"{prefix}[]"))
    return sorted(set(out))


def download(url: str, path: Path, *, reuse: bool) -> Dict[str, Any]:
    if reuse and path.exists() and path.stat().st_size:
        return {
            "path": str(path), "url": url, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "reused": True,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DAWN-SRW-operator-probe/2"})
    with urllib.request.urlopen(request, timeout=300) as response, temp.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    os.replace(temp, path)
    return {
        "path": str(path), "url": url, "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "reused": False,
    }


def read_url_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "DAWN-SRW-operator-probe/2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_datasets(value: str) -> List[str]:
    raw = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not raw or raw == ["all"]:
        return list(DATASETS)
    unknown = sorted(set(raw) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {','.join(unknown)}")
    return list(dict.fromkeys(raw))


def load_tokenizer(name: str, cache_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for an actual tokenizer probe; install requirements.txt"
        ) from exc
    return AutoTokenizer.from_pretrained(name, cache_dir=str(cache_dir), use_fast=True)


def tokenizer_manifest(tokenizer, name: str, max_seq_len: int,
                       max_candidate_tokens: int) -> Dict[str, Any]:
    vocab = tokenizer.get_vocab()
    canonical = json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "tokenizer_name": name,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(len(vocab)),
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "add_special_tokens": False,
        "max_seq_len": int(max_seq_len),
        "max_candidate_tokens": int(max_candidate_tokens),
        "vocab_hash": hashlib.sha256(canonical).hexdigest(),
    }


def encode(tokenizer, text: str) -> List[int]:
    return [int(v) for v in tokenizer.encode(str(text), add_special_tokens=False)]


def lcp_length(a: Sequence[int], b: Sequence[int]) -> int:
    count = 0
    for left, right in zip(a, b):
        if int(left) != int(right):
            break
        count += 1
    return count


def trace_for_pair(ids_a: Sequence[int], ids_b: Sequence[int]) -> Tuple[int, Dict[str, Any]]:
    lcp = lcp_length(ids_a, ids_b)
    if lcp <= 0:
        return 0, {
            "longest_common_prefix_length": lcp,
            "first_divergence_index": 0,
            "trace_position_fallback": True,
            "trace_issue": "no_common_prefix; trace position 0 predicts token 1, not divergence token 0",
        }
    if lcp >= min(len(ids_a), len(ids_b)):
        return max(0, min(len(ids_a), len(ids_b)) - 1), {
            "longest_common_prefix_length": lcp,
            "first_divergence_index": None,
            "trace_position_fallback": True,
            "trace_issue": "one sequence is a prefix or sequences are identical",
        }
    return lcp - 1, {
        "longest_common_prefix_length": lcp,
        "first_divergence_index": lcp,
        "trace_position_fallback": False,
    }


def candidate_validation(
    tokenizer, context: str, positive: str, negative: str, *,
    max_seq_len: int, max_candidate_tokens: int, allow_empty: bool = False,
) -> Dict[str, Any]:
    context_ids = encode(tokenizer, context)
    positive_ids = encode(tokenizer, positive) if positive else []
    negative_ids = encode(tokenizer, negative) if negative else []
    errors: List[str] = []
    warnings: List[str] = []
    if len(context_ids) > max_seq_len:
        errors.append("context_too_long")
    if not allow_empty and not positive_ids:
        errors.append("empty_positive_candidate")
    if not allow_empty and not negative_ids:
        errors.append("empty_negative_candidate")
    if positive_ids and positive_ids == negative_ids:
        errors.append("identical_candidate_token_sequence")
    if len(positive_ids) > max_candidate_tokens:
        errors.append("positive_candidate_too_long")
    if len(negative_ids) > max_candidate_tokens:
        errors.append("negative_candidate_too_long")
    unk = tokenizer.unk_token_id
    if unk is not None and unk in context_ids:
        warnings.append("context_contains_unk")
    if unk is not None and unk in positive_ids:
        warnings.append("positive_contains_unk")
    if unk is not None and unk in negative_ids:
        warnings.append("negative_contains_unk")
    decoded_context = tokenizer.decode(context_ids, clean_up_tokenization_spaces=False)
    decoded_positive = tokenizer.decode(positive_ids, clean_up_tokenization_spaces=False)
    decoded_negative = tokenizer.decode(negative_ids, clean_up_tokenization_spaces=False)
    if context and not decoded_context:
        warnings.append("context_decode_roundtrip_empty")
    return {
        "context_text": context,
        "context_token_ids": context_ids,
        "context_decoded": decoded_context,
        "positive_text": positive,
        "positive_token_ids": positive_ids,
        "positive_decoded": decoded_positive,
        "negative_text": negative,
        "negative_token_ids": negative_ids,
        "negative_decoded": decoded_negative,
        "context_length": len(context_ids),
        "positive_length": len(positive_ids),
        "negative_length": len(negative_ids),
        "validation_errors": errors,
        "validation_warnings": warnings,
        "multi_token_positive": len(positive_ids) > 1,
        "multi_token_negative": len(negative_ids) > 1,
    }


def _pad(values: Sequence[int], size: int) -> List[int]:
    return [int(v) for v in values[:size]] + [0] * max(0, size - len(values))


def prepared_row(
    tokenizer,
    *,
    example_id: str,
    pair_id: str,
    dataset: str,
    split: str,
    phenomenon: str,
    relation: str,
    group_id: str,
    source_id: str,
    score_mode: str,
    trace_semantics: str,
    text_a: str,
    text_b: str = "",
    positive_a: str = "",
    negative_a: str = "",
    positive_b: str = "",
    negative_b: str = "",
    row_index: int = 0,
    max_seq_len: int = 512,
    max_candidate_tokens: int = 16,
    extension: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ids_a = encode(tokenizer, text_a)
    ids_b = encode(tokenizer, text_b) if text_b else []
    pos_a = encode(tokenizer, positive_a) if positive_a else []
    neg_a = encode(tokenizer, negative_a) if negative_a else []
    pos_b = encode(tokenizer, positive_b) if positive_b else []
    neg_b = encode(tokenizer, negative_b) if negative_b else []
    errors: List[str] = []
    for label, ids, limit in (
        ("context_a", ids_a, max_seq_len), ("context_b", ids_b, max_seq_len),
        ("positive_a", pos_a, max_candidate_tokens),
        ("negative_a", neg_a, max_candidate_tokens),
        ("positive_b", pos_b, max_candidate_tokens),
        ("negative_b", neg_b, max_candidate_tokens),
    ):
        if len(ids) > limit:
            errors.append(f"{label}_too_long")
    if score_mode == "paired_sequence_logprob":
        trace_a, trace_meta = trace_for_pair(ids_a, ids_b)
        trace_b = trace_a
        divergence = trace_meta.get("first_divergence_index")
        if divergence is None:
            errors.append("paired_sequence_has_no_stable_divergence_span")
        target_start_a = (
            int(divergence)
            if divergence is not None and int(divergence) < len(ids_a) else -1)
        target_start_b = (
            int(divergence)
            if divergence is not None and int(divergence) < len(ids_b) else -1)
    else:
        trace_a = len(ids_a) - 1
        trace_b = len(ids_b) - 1 if ids_b else -1
        trace_meta = {}
        target_start_a = target_start_b = -1
        for side, positive, negative in (("a", pos_a, neg_a), ("b", pos_b, neg_b)):
            if (positive or negative) and (not positive or not negative):
                errors.append(f"empty_{side}_candidate")
            if positive and positive == negative:
                errors.append(f"identical_{side}_candidates")
    if trace_a < 0 or trace_a >= max(len(ids_a), 1):
        errors.append("trace_position_a_out_of_context")
    if ids_b and (trace_b < 0 or trace_b >= len(ids_b)):
        errors.append("trace_position_b_out_of_context")
    arrays = {
        "context_ids_a": _pad(ids_a, max_seq_len),
        "context_len_a": len(ids_a),
        "context_ids_b": _pad(ids_b, max_seq_len),
        "context_len_b": len(ids_b),
        "positive_ids_a": _pad(pos_a, max_candidate_tokens),
        "positive_len_a": len(pos_a),
        "negative_ids_a": _pad(neg_a, max_candidate_tokens),
        "negative_len_a": len(neg_a),
        "positive_ids_b": _pad(pos_b, max_candidate_tokens),
        "positive_len_b": len(pos_b),
        "negative_ids_b": _pad(neg_b, max_candidate_tokens),
        "negative_len_b": len(neg_b),
        "trace_position_a": trace_a,
        "trace_position_b": trace_b,
        # target_span indexes the stored context/full-sequence arrays.  It is
        # not the candidate span in a transient teacher-forced concatenation.
        "target_span_start_a": target_start_a,
        "target_span_end_a": target_start_a + 1 if target_start_a >= 0 else -1,
        "target_span_start_b": target_start_b,
        "target_span_end_b": target_start_b + 1 if target_start_b >= 0 else -1,
        "row_index": int(row_index),
    }
    metadata = {
        "example_id": example_id,
        "pair_id": pair_id,
        "dataset": dataset,
        "split": split,
        "phenomenon": phenomenon,
        "relation": relation,
        "group_id": group_id,
        "source_id": source_id,
        "score_mode": score_mode,
        "trace_semantics": trace_semantics,
        "text_a": text_a,
        "text_b": text_b,
        "positive_text_a": positive_a,
        "negative_text_a": negative_a,
        "positive_text_b": positive_b,
        "negative_text_b": negative_b,
    }
    human = {
        **metadata,
        "context_a_decoded": tokenizer.decode(ids_a, clean_up_tokenization_spaces=False),
        "context_a_tokens": ids_a,
        "context_b_decoded": tokenizer.decode(ids_b, clean_up_tokenization_spaces=False),
        "context_b_tokens": ids_b,
        "positive_a_tokens": pos_a,
        "negative_a_tokens": neg_a,
        "positive_b_tokens": pos_b,
        "negative_b_tokens": neg_b,
        "trace_token_a": (
            tokenizer.convert_ids_to_tokens(ids_a[trace_a])
            if 0 <= trace_a < len(ids_a) else None),
        "trace_token_b": (
            tokenizer.convert_ids_to_tokens(ids_b[trace_b])
            if 0 <= trace_b < len(ids_b) else None),
        "validation_errors": errors,
        "trace_details": trace_meta,
        "teacher_forced_candidate_span_a": [len(ids_a), len(ids_a) + len(pos_a)],
        "teacher_forced_candidate_span_b": (
            [len(ids_b), len(ids_b) + len(pos_b)] if ids_b else None),
        "extension": json_safe(extension or {}),
    }
    return {"arrays": arrays, "metadata": metadata, "human": human}


def base_probe(dataset: str, source: Any) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "status": "partial",
        "source": source,
        "downloaded_files": [],
        "observed_files": [],
        "observed_splits": [],
        "observed_schema": {},
        "observed_columns": [],
        "nested_key_paths": [],
        "row_count": None,
        "raw_samples": [],
        "tokenization_samples": [],
        "proposed_pair_types": [],
        "proposed_score_mode": None,
        "proposed_trace_semantics": None,
        "source_issues": [],
        "adapter_risks": [],
        "recommended_mapping": {},
    }


def inspect_parquet(path: Path, sample_rows: int) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    table = parquet.read_row_group(0).slice(0, sample_rows)
    return {
        "path": str(path),
        "row_count": int(parquet.metadata.num_rows),
        "columns": list(parquet.schema_arrow.names),
        "schema": {field.name: str(field.type) for field in parquet.schema_arrow},
        "samples": json_safe(table.to_pylist()),
    }


def read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    return json_safe(pq.read_table(path).to_pylist())


def hf_parquet_tree(url: str) -> List[str]:
    rows = read_url_json(url)
    return sorted(
        str(row["path"]) for row in rows
        if row.get("type") == "file" and str(row.get("path", "")).endswith(".parquet")
    )


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(value).lower(): str(value) for value in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def probe_ravel(args, tokenizer, cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    probe = base_probe("ravel", {"hf_tree": RAVEL_HF_TREE, "raw_archive": RAVEL_TGZ_URL})
    paths = hf_parquet_tree(RAVEL_HF_TREE)
    selected = [
        path for path in paths
        if path.startswith("city_entity/") or path.startswith("city_prompt/")
    ]
    if not selected:
        raise RuntimeError("RAVEL HF tree has no city_entity/city_prompt parquet files")
    observed = []
    all_samples: List[Dict[str, Any]] = []
    for rel in selected:
        local = cache / "ravel" / "hf" / rel
        file_row = download(f"{RAVEL_HF_RESOLVE}/{rel}", local, reuse=args.reuse_downloads)
        probe["downloaded_files"].append(file_row)
        info = inspect_parquet(local, args.sample_rows)
        info["source_path"] = rel
        info["split"] = Path(rel).stem.split("-")[0]
        info["kind"] = rel.split("/", 1)[0]
        observed.append(info)
        for row in info["samples"]:
            all_samples.append({"source_path": rel, "split": info["split"], **row})
    raw_local = cache / "ravel" / "raw" / "data.tgz"
    raw_info = download(RAVEL_TGZ_URL, raw_local, reuse=args.reuse_downloads)
    probe["downloaded_files"].append(raw_info)
    with tarfile.open(raw_local, "r:gz") as archive:
        names = archive.getnames()
    probe["observed_files"] = [row["source_path"] for row in observed] + names
    probe["observed_splits"] = sorted(set(row["split"] for row in observed))
    probe["observed_schema"] = {
        row["source_path"]: row["schema"] for row in observed
    }
    probe["observed_columns"] = sorted(set(
        column for row in observed for column in row["columns"]))
    probe["row_count"] = {
        row["source_path"]: row["row_count"] for row in observed
    }
    probe["raw_samples"] = all_samples[: max(args.sample_rows, 5)]
    probe["nested_key_paths"] = sorted(set(
        path for row in probe["raw_samples"] for path in nested_key_paths(row)))
    train_entity_path = cache / "ravel" / "hf" / next(
        path for path in selected if path.startswith("city_entity/train"))
    train_prompt_path = cache / "ravel" / "hf" / next(
        path for path in selected if path.startswith("city_prompt/train"))
    entity_rows = read_parquet_rows(train_entity_path)
    prompt_rows = read_parquet_rows(train_prompt_path)
    probe["raw_samples"] = [
        {"source_path": "city_entity/train-00000-of-00001.parquet", "split": "train", **row}
        for row in entity_rows[: args.sample_rows]
    ] + [
        {"source_path": "city_prompt/train-00000-of-00001.parquet", "split": "train", **row}
        for row in prompt_rows[: args.sample_rows]
    ]
    probe["nested_key_paths"] = sorted(set(
        path for row in probe["raw_samples"] for path in nested_key_paths(row)))
    prepared: List[Dict[str, Any]] = []
    required_entity = {"ID", "City"}
    required_prompt = {"Template", "Attribute", "Source", "Entity"}
    if not entity_rows or not prompt_rows:
        raise RuntimeError("RAVEL train entity/prompt parquet is empty")
    missing_entity = required_entity - set(entity_rows[0])
    missing_prompt = required_prompt - set(prompt_rows[0])
    if missing_entity or missing_prompt:
        raise RuntimeError(
            f"RAVEL join schema missing entity={sorted(missing_entity)} prompt={sorted(missing_prompt)}")
    usable_prompts = [
        row for row in prompt_rows
        if str(row.get("Attribute")) in entity_rows[0]
        and str(row.get("Template", "")).count("%s") == 1
    ]
    prompts_by_attribute: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in usable_prompts:
        prompts_by_attribute[str(row["Attribute"])].append(row)
    attributes = [
        attribute for attribute, rows in prompts_by_attribute.items()
        if rows and any(str(entity.get(attribute, "")) for entity in entity_rows)
    ]
    if len(attributes) < 2:
        raise RuntimeError(
            f"RAVEL needs at least two joinable attributes, observed={attributes}")

    def different_entity(base: Mapping[str, Any], attribute: str) -> Dict[str, Any]:
        return next(
            entity for entity in entity_rows
            if str(entity.get("ID")) != str(base.get("ID"))
            and str(entity.get(attribute, ""))
            and str(entity.get(attribute)) != str(base.get(attribute)))

    def condition(entity: Mapping[str, Any], prompt: Mapping[str, Any]) -> Tuple[str, str]:
        template = str(prompt["Template"])
        return template % str(entity["City"]), str(entity[str(prompt["Attribute"])])

    e1 = entity_rows[0]
    attr1, attr2 = attributes[0], attributes[1]
    e2 = different_entity(e1, attr1)
    e3 = different_entity(e1, attr2)
    p1 = prompts_by_attribute[attr1][0]
    p1_alt = prompts_by_attribute[attr1][1]
    p2 = prompts_by_attribute[attr2][0]
    pair_specs = [
        ("same_attribute_different_entity", e1, p1, e2, p1),
        ("same_entity_different_attribute", e1, p1, e1, p2),
        ("same_attribute_different_prompt", e1, p1, e1, p1_alt),
        ("cross_attribute_control", e2, p1, e3, p2),
        ("same_attribute_different_entity", e2, p1_alt, e3, p1_alt),
    ]
    for index, (family, entity_a, prompt_a, entity_b, prompt_b) in enumerate(pair_specs):
        context_a, positive_a = condition(entity_a, prompt_a)
        context_b, positive_b = condition(entity_b, prompt_b)
        control_a = different_entity(entity_a, str(prompt_a["Attribute"]))
        control_b = different_entity(entity_b, str(prompt_b["Attribute"]))
        negative_a = str(control_a[str(prompt_a["Attribute"])])
        negative_b = str(control_b[str(prompt_b["Attribute"])])
        pair_id = f"ravel-city-{family}-{index}"
        prepared.append(prepared_row(
            tokenizer, example_id=pair_id, pair_id=pair_id, dataset="ravel",
            split="train", phenomenon=family,
            relation=f"{prompt_a['Attribute']}->{prompt_b['Attribute']}",
            group_id=family, source_id=f"{entity_a['ID']}:{entity_b['ID']}",
            score_mode="continuation_margin",
            trace_semantics="last_context_token_prediction_state",
            text_a=context_a, text_b=context_b,
            positive_a=positive_a, negative_a=negative_a,
            positive_b=positive_b, negative_b=negative_b,
            row_index=index, max_seq_len=args.max_seq_len,
            max_candidate_tokens=args.max_candidate_tokens,
            extension={
                "family": family,
                "entity_a": entity_a["City"], "entity_b": entity_b["City"],
                "entity_id_a": entity_a["ID"], "entity_id_b": entity_b["ID"],
                "attribute_a": prompt_a["Attribute"], "attribute_b": prompt_b["Attribute"],
                "template_a": prompt_a["Template"], "template_b": prompt_b["Template"],
                "prompt_source_a": prompt_a["Source"], "prompt_source_b": prompt_b["Source"],
            },
        ))
    probe["status"] = "ready_for_adapter"
    probe["source_issues"].append(
        "RAVEL is normalized: city_entity supplies City and attribute values; city_prompt supplies Attribute and one-%s Template. Adapter must join them, not expect input/label columns.")
    probe["proposed_pair_types"] = [
        "same_attribute_different_entity", "same_entity_different_attribute",
        "same_attribute_different_prompt", "cross_attribute_control",
    ]
    probe["proposed_score_mode"] = "continuation_margin"
    probe["proposed_trace_semantics"] = "last_context_token_prediction_state"
    probe["recommended_mapping"] = {
        "context_ids_a/b": "city_prompt.Template % city_entity.City (Template must contain exactly one %s)",
        "positive_ids_a/b": "city_entity[row][city_prompt.Attribute]",
        "negative_ids_a/b": "same Attribute column from a different entity with a different value",
        "entity_extension": "city_entity.City and ID",
        "template_extension": "city_prompt.Template, Source, Entity",
        "attribute/group_id": "city_prompt.Attribute / controlled pair family",
    }
    return probe, prepared


def _choose_blimp_paths(paths: Sequence[str], count: int = 3) -> List[str]:
    groups = (
        ("subject_verb", "distractor_agreement", "agreement"),
        ("anaphor", "principle_a", "binding"),
        ("npi", "negative", "determiner"),
    )
    selected: List[str] = []
    for fragments in groups:
        match = next((path for path in paths if any(f in path.lower() for f in fragments)), None)
        if match and match not in selected:
            selected.append(match)
    for path in paths:
        if len(selected) >= count:
            break
        if path not in selected:
            selected.append(path)
    return selected[:count]


def probe_blimp(args, tokenizer, cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    probe = base_probe("blimp", {"hf_tree": BLIMP_HF_TREE})
    paths = hf_parquet_tree(BLIMP_HF_TREE)
    selected = _choose_blimp_paths(paths)
    observed = []
    raw_samples = []
    prepared = []
    schemas = []
    for rel in selected:
        local = cache / "blimp" / "hf" / rel
        file_row = download(f"{BLIMP_HF_RESOLVE}/{rel}", local, reuse=args.reuse_downloads)
        probe["downloaded_files"].append(file_row)
        info = inspect_parquet(local, args.sample_rows)
        info["source_path"] = rel
        phenomenon = rel.split("/", 1)[0]
        info["phenomenon"] = phenomenon
        observed.append(info)
        schemas.append(info["schema"])
        good_col = _find_col(info["columns"], ("sentence_good", "good_sentence", "good"))
        bad_col = _find_col(info["columns"], ("sentence_bad", "bad_sentence", "bad"))
        if not good_col or not bad_col:
            probe["source_issues"].append(f"{rel}: good/bad sentence columns not found")
            continue
        for row_index, row in enumerate(info["samples"]):
            raw = {"source_path": rel, "phenomenon": phenomenon, **row}
            raw_samples.append(raw)
            good = str(row[good_col])
            bad = str(row[bad_col])
            good_ids = encode(tokenizer, good)
            bad_ids = encode(tokenizer, bad)
            trace_position, trace_meta = trace_for_pair(good_ids, bad_ids)
            raw["tokenizer_probe"] = {
                "good_token_ids": good_ids,
                "bad_token_ids": bad_ids,
                **trace_meta,
                "trace_position": trace_position,
                "trace_token": (
                    tokenizer.convert_ids_to_tokens(good_ids[trace_position])
                    if good_ids else None),
            }
            if len(prepared) < args.sample_rows:
                uid = row.get("UID") or row.get("uid") or f"{phenomenon}-{row_index}"
                prepared.append(prepared_row(
                    tokenizer, example_id=f"blimp-{uid}", pair_id=f"blimp-{uid}",
                    dataset="blimp", split="train", phenomenon=phenomenon,
                    relation="grammatical_minimal_pair", group_id=phenomenon,
                    source_id=str(uid), score_mode="paired_sequence_logprob",
                    trace_semantics="pre_divergence_prediction_state",
                    text_a=good, text_b=bad, row_index=len(prepared),
                    max_seq_len=args.max_seq_len,
                    max_candidate_tokens=args.max_candidate_tokens,
                    extension={key: value for key, value in row.items()
                               if key not in (good_col, bad_col)},
                ))
    probe["observed_files"] = paths
    probe["observed_splits"] = ["train"]
    probe["observed_schema"] = {
        row["source_path"]: row["schema"] for row in observed
    }
    probe["observed_columns"] = sorted(set(
        column for row in observed for column in row["columns"]))
    probe["row_count"] = {row["source_path"]: row["row_count"] for row in observed}
    # Preserve samples from every inspected phenomenon rather than allowing
    # the first parquet to hide cross-file schema or tokenization differences.
    probe["raw_samples"] = raw_samples
    probe["nested_key_paths"] = sorted(set(
        path for row in probe["raw_samples"] for path in nested_key_paths(row)))
    probe["tokenization_samples"] = [row["tokenizer_probe"] for row in raw_samples]
    probe["status"] = (
        "ready_for_adapter" if prepared and all(schema == schemas[0] for schema in schemas)
        else "schema_inconsistent")
    probe["proposed_pair_types"] = ["good_bad_minimal_pair"]
    probe["proposed_score_mode"] = "paired_sequence_logprob"
    probe["proposed_trace_semantics"] = "pre_divergence_prediction_state"
    probe["recommended_mapping"] = {
        "context_ids_a": "sentence_good",
        "context_ids_b": "sentence_bad",
        "phenomenon/group_id": "parquet parent directory",
        "source_id": "UID when present; otherwise phenomenon + row index",
        "trace_position_a/b": "longest common prefix length - 1",
        "candidate_arrays": "unused (length 0)",
    }
    return probe, prepared


def _jsonl_from_zip(archive: zipfile.ZipFile, name: str, limit: int) -> List[Dict[str, Any]]:
    rows = []
    with archive.open(name) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line.decode("utf-8")))
                if len(rows) >= limit:
                    break
    return rows


def _jsonl_count_from_zip(archive: zipfile.ZipFile, name: str) -> int:
    count = 0
    with archive.open(name) as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _lama_prefix(masked: str) -> Tuple[Optional[str], Optional[str]]:
    if masked.count("[MASK]") != 1:
        return None, "masked sentence does not contain exactly one [MASK]"
    before, after = masked.split("[MASK]", 1)
    if after.strip() not in ("", ".", "?", "!", ","):
        return None, f"nontrivial suffix after [MASK]: {after!r}"
    return before.rstrip(), None


def probe_lama(args, tokenizer, cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    probe = base_probe("lama", {"archive": LAMA_ZIP_URL})
    local = cache / "lama" / "data.zip"
    probe["downloaded_files"].append(download(LAMA_ZIP_URL, local, reuse=args.reuse_downloads))
    prepared = []
    raw_samples = []
    schemas: Dict[str, Any] = {}
    row_counts: Dict[str, int] = {}
    with zipfile.ZipFile(local) as archive:
        names = archive.namelist()
        relation_templates: Dict[str, str] = {}
        relation_file = next((name for name in names if name.endswith("relations.jsonl")), None)
        if relation_file:
            for row in _jsonl_from_zip(archive, relation_file, 10000):
                relation_id = str(row.get("relation") or row.get("predicate_id") or "")
                if relation_id and row.get("template"):
                    relation_templates[relation_id] = str(row["template"])
        categories = ("TREx", "Google_RE", "ConceptNet", "Squad")
        selected: List[str] = []
        for category in categories:
            match = next((name for name in names
                          if category.lower() in name.lower() and name.endswith(".jsonl")
                          and not name.endswith("relations.jsonl")), None)
            if match:
                selected.append(match)
        for name in names:
            if len(selected) >= 6:
                break
            if name.endswith(".jsonl") and name != relation_file and name not in selected:
                selected.append(name)
        for name in selected:
            rows = _jsonl_from_zip(archive, name, max(100, args.sample_rows))
            if not rows:
                continue
            category = next((cat for cat in categories if cat.lower() in name.lower()), "other")
            schemas[name] = {
                key: type(value).__name__ for key, value in rows[0].items()
            }
            row_counts[name] = _jsonl_count_from_zip(archive, name)
            for row in rows[: args.sample_rows]:
                raw_samples.append({"source_file": name, "source_category": category, **row})
            relation_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                relation_id = str(
                    row.get("predicate_id") or row.get("relation") or Path(name).stem)
                relation_groups[relation_id].append(row)
            for relation_id, group in relation_groups.items():
                if len(prepared) >= args.sample_rows:
                    break
                for row_index, row in enumerate(group):
                    subject = row.get("sub_label") or row.get("subject")
                    obj = row.get("obj_label") or row.get("object")
                    masked_values = row.get("masked_sentences") or row.get("masked_sentence")
                    masked = masked_values[0] if isinstance(masked_values, list) and masked_values else masked_values
                    template = relation_templates.get(relation_id)
                    if not masked and template and subject:
                        masked = template.replace("[X]", str(subject)).replace("[Y]", "[MASK]")
                    if not subject or not obj or not masked:
                        continue
                    context, risk = _lama_prefix(str(masked))
                    if risk:
                        probe["adapter_risks"].append({"source_file": name, "risk": risk, "row": row})
                        continue
                    distractor = next((
                        other.get("obj_label") or other.get("object") for other in group
                        if (other.get("obj_label") or other.get("object"))
                        and str(other.get("obj_label") or other.get("object")) != str(obj)
                    ), None)
                    if distractor is None:
                        continue
                    prepared.append(prepared_row(
                        tokenizer, example_id=f"lama-{relation_id}-{row_index}",
                        pair_id=f"lama-{relation_id}-{row_index}", dataset="lama",
                        split=category, phenomenon="factual_recall", relation=relation_id,
                        group_id=relation_id, source_id=f"{name}:{row_index}",
                        score_mode="continuation_margin",
                        trace_semantics="last_context_token_prediction_state",
                        text_a=str(context), positive_a=str(obj), negative_a=str(distractor),
                        row_index=len(prepared), max_seq_len=args.max_seq_len,
                        max_candidate_tokens=args.max_candidate_tokens,
                        extension={"subject": subject, "source_file": name,
                                   "source_category": category, "masked_sentence": masked,
                                   "template": template},
                    ))
                    break
        probe["observed_files"] = names
    probe["observed_splits"] = sorted(set(
        row["source_category"] for row in raw_samples))
    probe["observed_schema"] = schemas
    probe["observed_columns"] = sorted(set(
        key for row in raw_samples for key in row.keys()))
    probe["nested_key_paths"] = sorted(set(
        path for row in raw_samples for path in nested_key_paths(row)))
    probe["row_count"] = {
        "archive_entries": len(probe["observed_files"]),
        "sampled_file_rows_read": row_counts,
        "note": "probe intentionally stops after enough rows; these are not full file counts",
    }
    probe["raw_samples"] = raw_samples
    probe["status"] = "ready_for_adapter" if prepared else "schema_inconsistent"
    probe["proposed_pair_types"] = ["same_relation_different_subject"]
    probe["proposed_score_mode"] = "continuation_margin"
    probe["proposed_trace_semantics"] = "last_context_token_prediction_state"
    probe["recommended_mapping"] = {
        "relation": "predicate_id, relation, or relation file stem",
        "subject": "sub_label or subject",
        "positive_ids_a": "obj_label or object",
        "negative_ids_a": "different object from the same relation",
        "context_ids_a": "prefix before the single terminal [MASK] in masked_sentences[0]",
        "source_extension": "source category/file, original masked sentence, object id",
    }
    return probe, prepared


def probe_counterfact(args, tokenizer, cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    probe = base_probe("counterfact", {"json": COUNTERFACT_URL})
    local = cache / "counterfact" / "counterfact.json"
    probe["downloaded_files"].append(download(COUNTERFACT_URL, local, reuse=args.reuse_downloads))
    rows = json.loads(local.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("CounterFact top-level JSON is not a list")
    raw_samples = rows[: max(args.sample_rows, 5)]
    prepared = []
    for index, row in enumerate(raw_samples[: args.sample_rows]):
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            probe["source_issues"].append(f"case {row.get('case_id')}: requested_rewrite missing")
            continue
        prompt = rewrite.get("prompt")
        subject = rewrite.get("subject")
        true_value = rewrite.get("target_true")
        new_value = rewrite.get("target_new")
        true_text = true_value.get("str") if isinstance(true_value, Mapping) else true_value
        new_text = new_value.get("str") if isinstance(new_value, Mapping) else new_value
        if not all(value is not None for value in (prompt, subject, true_text, new_text)):
            probe["source_issues"].append(f"case {row.get('case_id')}: incomplete primary fields")
            continue
        prompt = str(prompt)
        if "{}" in prompt:
            if prompt.count("{}") != 1:
                probe["adapter_risks"].append(
                    f"case {row.get('case_id')}: prompt has {prompt.count('{}')} format slots")
                continue
            context = prompt.format(subject)
            prompt_form = "single_positional_format_slot"
        else:
            context = prompt
            prompt_form = "already_materialized_or_subject_absent"
            if str(subject) not in context:
                probe["adapter_risks"].append(
                    f"case {row.get('case_id')}: prompt has no format slot and does not contain subject")
        case_id = row.get("case_id", index)
        prepared.append(prepared_row(
            tokenizer, example_id=f"counterfact-{case_id}", pair_id=f"counterfact-{case_id}",
            dataset="counterfact", split="counterfact", phenomenon="factual_recall",
            relation=str(rewrite.get("relation_id") or "unknown_relation"),
            group_id=str(rewrite.get("relation_id") or subject), source_id=str(case_id),
            score_mode="true_new_object_margin",
            trace_semantics="last_context_token_prediction_state",
            text_a=context, positive_a=str(true_text), negative_a=str(new_text),
            row_index=index, max_seq_len=args.max_seq_len,
            max_candidate_tokens=args.max_candidate_tokens,
            extension={
                "case_id": case_id, "subject": subject,
                "relation_id": rewrite.get("relation_id"), "prompt_form": prompt_form,
                "paraphrase_prompts": row.get("paraphrase_prompts"),
                "neighborhood_prompts": row.get("neighborhood_prompts"),
                "generation_prompts": row.get("generation_prompts"),
                "target_true": true_value, "target_new": new_value,
            },
        ))
    probe["observed_files"] = [str(local)]
    probe["observed_splits"] = ["counterfact"]
    probe["observed_schema"] = {
        "top_level": "list[object]",
        "first_row_types": {key: type(value).__name__ for key, value in rows[0].items()},
    }
    probe["observed_columns"] = sorted(rows[0].keys())
    probe["nested_key_paths"] = sorted(set(
        path for row in raw_samples for path in nested_key_paths(row)))
    probe["row_count"] = len(rows)
    probe["raw_samples"] = raw_samples
    probe["status"] = "ready_for_adapter" if prepared else "schema_inconsistent"
    probe["proposed_pair_types"] = ["true_object_vs_rewrite_object"]
    probe["proposed_score_mode"] = "true_new_object_margin"
    probe["proposed_trace_semantics"] = "last_context_token_prediction_state"
    probe["recommended_mapping"] = {
        "context_ids_a": "requested_rewrite.prompt.format(requested_rewrite.subject) when exactly one {} exists",
        "positive_ids_a": "requested_rewrite.target_true.str",
        "negative_ids_a": "requested_rewrite.target_new.str",
        "relation": "requested_rewrite.relation_id",
        "source_id": "case_id",
        "extension": "paraphrase_prompts, neighborhood_prompts, generation_prompts and target ids",
    }
    return probe, prepared


def probe_ioi(args, tokenizer, _cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    probe = base_probe("ioi", {"generator": "scripts.prepare_v4166_operator_datasets.generate_ioi_rows"})
    rows = list(generate_ioi_rows(10))
    raw_samples = []
    prepared = []
    for index, row in enumerate(rows):
        # Existing correct/distractor describe the clean prompt only.  Swapping
        # names in the corrupt prompt reverses the semantic labels.
        clean_correct = str(row["correct"])
        clean_distractor = str(row["distractor"])
        corrupt_correct = clean_distractor
        corrupt_distractor = clean_correct
        template_id = int(row["template_id"])
        clean_pattern = "".join(
            match.upper() for match in re.findall(
                r"\{name_([ab])\}", IOI_TEMPLATES[template_id]))
        corrupt_pattern = clean_pattern.translate(str.maketrans({"A": "B", "B": "A"}))
        pattern_type = f"{clean_pattern}_to_{corrupt_pattern}_name_order_swap"
        enriched = {
            **row,
            "clean_correct": clean_correct,
            "clean_distractor": clean_distractor,
            "corrupt_correct": corrupt_correct,
            "corrupt_distractor": corrupt_distractor,
            "pattern_type": pattern_type,
            "common_correct_is_wrong_for_corrupt": clean_correct != corrupt_correct,
            "clean_candidate_tokens": {
                clean_correct: encode(tokenizer, clean_correct),
                clean_distractor: encode(tokenizer, clean_distractor),
            },
            "corrupt_candidate_tokens": {
                corrupt_correct: encode(tokenizer, corrupt_correct),
                corrupt_distractor: encode(tokenizer, corrupt_distractor),
            },
        }
        raw_samples.append(enriched)
        if index < args.sample_rows:
            prepared.append(prepared_row(
                tokenizer, example_id=str(row["id"]), pair_id=str(row["id"]),
                dataset="ioi", split="generated", phenomenon="ioi",
                relation="clean_corrupt_name_binding",
                group_id=f"template_{template_id}:{pattern_type}", source_id=str(row["id"]),
                score_mode="clean_corrupt_continuation_margin",
                trace_semantics="last_context_token_prediction_state",
                text_a=str(row["clean_prompt"]), text_b=str(row["corrupt_prompt"]),
                positive_a=clean_correct, negative_a=clean_distractor,
                positive_b=corrupt_correct, negative_b=corrupt_distractor,
                row_index=index, max_seq_len=args.max_seq_len,
                max_candidate_tokens=args.max_candidate_tokens,
                extension={"template_id": template_id, "pattern_type": pattern_type},
            ))
    probe.update({
        "status": "ready_for_adapter",
        "observed_files": ["generated in memory"],
        "observed_splits": ["generated"],
        "observed_schema": {
            "current": {key: type(value).__name__ for key, value in rows[0].items()},
            "recommended_additions": {
                "clean_correct": "str", "clean_distractor": "str",
                "corrupt_correct": "str", "corrupt_distractor": "str",
                "pattern_type": "str",
            },
        },
        "observed_columns": sorted(rows[0].keys()),
        "nested_key_paths": nested_key_paths(raw_samples[0]),
        "row_count": len(rows),
        "raw_samples": raw_samples,
        "tokenization_samples": [
            {"id": row["id"], "clean": row["clean_candidate_tokens"],
             "corrupt": row["corrupt_candidate_tokens"]}
            for row in raw_samples
        ],
        "proposed_pair_types": ["clean_corrupt_name_binding"],
        "proposed_score_mode": "clean_corrupt_continuation_margin",
        "proposed_trace_semantics": "last_context_token_prediction_state",
        "source_issues": [
            "Current generator stores one correct/distractor pair, but corrupt name order reverses the semantic labels."
        ],
        "recommended_mapping": {
            "context_ids_a": "clean_prompt", "positive_ids_a": "clean_correct",
            "negative_ids_a": "clean_distractor", "context_ids_b": "corrupt_prompt",
            "positive_ids_b": "corrupt_correct", "negative_ids_b": "corrupt_distractor",
            "group_id": "template_id + pattern_type",
        },
    })
    return probe, prepared


def synthetic_pair_examples(seed: int = 4171) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    families = (
        "same_operation_different_entity", "same_entity_different_attribute",
        "attribute_swap", "entity_swap", "distractor_swap", "order_permutation",
        "irrelevant_fact_addition",
    )
    examples = []
    for index, family in enumerate(families):
        entity, paired_entity, other = rng.sample(SYNTHETIC_ENTITIES, 3)
        color, distractor, paired_color = rng.sample(SYNTHETIC_COLORS, 3)
        key, key_distractor = rng.sample(SYNTHETIC_KEYS, 2)
        base = f"{entity} is {color}. {paired_entity} is {distractor}. The color of {entity} is"
        paired = f"{other} is {color}. {paired_entity} is {distractor}. The color of {other} is"
        positive_b, negative_b = color, distractor
        attribute = "color"
        controlled_change = family
        if family == "same_entity_different_attribute":
            paired = f"{entity} has {key}. {paired_entity} has {key_distractor}. {entity}'s key is"
            positive_b, negative_b, attribute = key, key_distractor, "color_to_key"
        elif family == "attribute_swap":
            paired = f"{entity} is {paired_color}. {paired_entity} is {distractor}. The color of {entity} is"
            positive_b = paired_color
        elif family == "entity_swap":
            paired = f"{entity} is {color}. {paired_entity} is {distractor}. The color of {paired_entity} is"
            positive_b, negative_b = distractor, color
        elif family == "distractor_swap":
            paired = f"{entity} is {color}. {paired_entity} is {paired_color}. The color of {entity} is"
            negative_b = paired_color
        elif family == "order_permutation":
            paired = f"{paired_entity} is {distractor}. {entity} is {color}. The color of {entity} is"
        elif family == "irrelevant_fact_addition":
            paired = f"{entity} is {color}. {paired_entity} is {distractor}. {other} is {paired_color}. The color of {entity} is"
        examples.append({
            "pair_id": f"synthetic-pair-{index:02d}", "family": family,
            "context_a": base, "positive_a": color, "negative_a": distractor,
            "context_b": paired, "positive_b": positive_b, "negative_b": negative_b,
            "entity": entity, "paired_entity": paired_entity,
            "attribute": attribute, "correct_value": color,
            "distractor_value": distractor, "controlled_change": controlled_change,
        })
    return examples


def probe_synthetic(args, tokenizer, _cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    probe = base_probe("synthetic", {"generator": "scripts.prepare_v4166_operator_datasets.generate_synthetic_rows"})
    current = list(generate_synthetic_rows(10))
    pair_rows = synthetic_pair_examples()
    raw_samples = []
    for index, row in enumerate(current):
        domain = SYNTHETIC_COLORS if row["attribute"] == "color" else SYNTHETIC_KEYS
        distractor = next(value for value in domain if value != row["answer"])
        raw_samples.append({
            **row,
            "distractor_probe": distractor,
            "answer_token_ids": encode(tokenizer, row["answer"]),
            "distractor_token_ids": encode(tokenizer, distractor),
        })
    prepared = []
    for index, row in enumerate(pair_rows[: args.sample_rows]):
        prepared.append(prepared_row(
            tokenizer, example_id=f"synthetic-{index:05d}", pair_id=row["pair_id"],
            dataset="synthetic", split="generated", phenomenon=row["family"],
            relation=row["family"], group_id=row["family"], source_id=row["pair_id"],
            score_mode="continuation_margin",
            trace_semantics="last_context_token_prediction_state",
            text_a=row["context_a"], text_b=row["context_b"],
            positive_a=row["positive_a"], negative_a=row["negative_a"],
            positive_b=row["positive_b"], negative_b=row["negative_b"],
            row_index=index, max_seq_len=args.max_seq_len,
            max_candidate_tokens=args.max_candidate_tokens,
            extension={key: value for key, value in row.items()
                       if key not in ("context_a", "context_b", "positive_a", "negative_a",
                                      "positive_b", "negative_b")},
        ))
    probe.update({
        "status": "ready_for_adapter",
        "observed_files": ["generated in memory"],
        "observed_splits": ["generated"],
        "observed_schema": {
            "current": {key: type(value).__name__ for key, value in current[0].items()},
            "pair_candidate": {key: type(value).__name__ for key, value in pair_rows[0].items()},
        },
        "observed_columns": sorted(current[0].keys()),
        "nested_key_paths": nested_key_paths(pair_rows[0]),
        "row_count": {"current_generated_probe": len(current), "pair_family_probe": len(pair_rows)},
        "raw_samples": raw_samples,
        "tokenization_samples": [{
            "value": value, "token_ids": encode(tokenizer, value),
            "tokens": tokenizer.convert_ids_to_tokens(encode(tokenizer, value)),
        } for value in (*SYNTHETIC_KEYS, *SYNTHETIC_COLORS)],
        "proposed_pair_types": [row["family"] for row in pair_rows],
        "proposed_score_mode": "continuation_margin",
        "proposed_trace_semantics": "last_context_token_prediction_state",
        "source_issues": [
            "Current generator has independent rows and no distractor, pair_id, family, or controlled_change."
        ],
        "recommended_mapping": {
            "context_ids_a/b": "controlled pair prompts",
            "positive_ids_a/b": "condition-specific correct value",
            "negative_ids_a/b": "same-attribute-domain matched distractor",
            "group_id/phenomenon": "pair family",
            "extension": "entity, paired_entity, attribute, controlled_change",
        },
        "pair_family_examples": pair_rows,
    })
    return probe, prepared


PROBE_FUNCS = {
    "ravel": probe_ravel,
    "blimp": probe_blimp,
    "lama": probe_lama,
    "counterfact": probe_counterfact,
    "ioi": probe_ioi,
    "synthetic": probe_synthetic,
}


def dataset_mapping(probes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "array_fields": list(ARRAY_FIELDS),
        "metadata_fields": list(METADATA_FIELDS),
        "datasets": {
            dataset: {
                "status": probe.get("status"),
                "score_mode": probe.get("proposed_score_mode"),
                "trace_semantics": probe.get("proposed_trace_semantics"),
                "pair_types": probe.get("proposed_pair_types"),
                "mapping": probe.get("recommended_mapping"),
                "source_specific_extension": "metadata.extension",
            }
            for dataset, probe in probes.items()
        },
    }


def schema_markdown(probes: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# Operator dataset schema recommendation", "",
        f"Generated: {utc_now()}", "",
        "## Common contract", "",
        f"Schema candidate: `{SCHEMA_NAME}` version `{SCHEMA_VERSION}`.", "",
        "A/B is retained for every row. Datasets without a second condition set B lengths to 0; "
        "BLiMP uses A/B as good/bad full sequences with all candidate lengths 0. Continuation "
        "datasets store complete multi-token candidates and are scored teacher-forced by summing "
        "all candidate token log probabilities.", "",
        "`target_span_start/end` must index a span that is actually stored in context/full-sequence "
        "arrays. For BLiMP it is the first divergence token span. Continuation candidate spans are "
        "derived at scoring time as `[context_len, context_len + candidate_len)` and must not be put "
        "into context span fields; use `-1/-1` when no source-grounded context span is defined.", "",
        "| Dataset | Status | Score mode | Trace definition | Negative rule |", "|---|---|---|---|---|",
    ]
    for dataset, probe in probes.items():
        mapping = probe.get("recommended_mapping") or {}
        negative = mapping.get("negative_ids_a", mapping.get("candidate_arrays", "n/a"))
        lines.append(
            f"| {dataset} | {probe.get('status')} | {probe.get('proposed_score_mode')} | "
            f"{probe.get('proposed_trace_semantics')} | {negative} |")
    lines.extend([
        "", "## Dataset mappings", "",
    ])
    for dataset, probe in probes.items():
        lines.extend([f"### {dataset}", "", "| Common field | Source rule |", "|---|---|"])
        for field, source in (probe.get("recommended_mapping") or {}).items():
            lines.append(f"| `{field}` | {json.dumps(source, ensure_ascii=False)} |")
        if probe.get("source_issues") or probe.get("adapter_risks"):
            lines.extend(["", "Observed issues/risks:", ""])
            for issue in probe.get("source_issues") or []:
                lines.append(f"- {json.dumps(issue, ensure_ascii=False)}")
            for issue in probe.get("adapter_risks") or []:
                lines.append(f"- {json.dumps(issue, ensure_ascii=False)}")
        lines.append("")
    lines.extend([
        "## Drop conditions", "",
        "Rows are dropped only by the full preparation adapter, never by this probe. Required reasons: "
        "context longer than max_seq_len; candidate longer than max_candidate_tokens; empty required "
        "candidate; identical positive/negative token sequence; target span lost; unstable/out-of-range "
        "trace position; ambiguous source schema; or a LAMA mask with a nontrivial suffix that cannot be "
        "converted to a decoder prefix without changing the task.", "",
        "## Source-specific extensions", "",
        "Keep RAVEL entities/attributes/templates, BLiMP UID and linguistic fields, LAMA original masks "
        "and object ids, CounterFact auxiliary prompts/target ids, IOI template/pattern/name fields, and "
        "Synthetic controlled-change fields under metadata extension objects. Do not widen fixed arrays.", "",
        "## Undecided / fail-loud", "",
        "Any dataset marked partial, source_unavailable, schema_inconsistent, or failed is not an adapter "
        "contract. In particular, no RAVEL join is inferred unless the observed parquet/raw keys support it, "
        "and no LAMA masked sentence with meaningful text after [MASK] is silently rewritten.", "",
    ])
    return "\n".join(lines)


def source_summary_markdown(probes: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["# Operator dataset source probe", "", f"Generated: {utc_now()}", ""]
    for dataset, probe in probes.items():
        lines.extend([
            f"## {dataset}", "",
            f"- Status: `{probe.get('status')}`",
            f"- Splits/categories: `{probe.get('observed_splits')}`",
            f"- Observed columns: `{probe.get('observed_columns')}`",
            f"- Row count: `{probe.get('row_count')}`",
            f"- Prepared sample count: `{probe.get('prepared_sample_count', 0)}`",
            f"- Pair types: `{probe.get('proposed_pair_types')}`",
            "",
        ])
        if probe.get("source_issues"):
            lines.append("Source issues:")
            lines.append("")
            lines.extend(f"- {json.dumps(v, ensure_ascii=False)}" for v in probe["source_issues"])
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe operator-analysis source schemas and tokenization.")
    parser.add_argument("--datasets", default="all", help="Comma-separated ids or all.")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-candidate-tokens", type=int, default=16)
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--work-dir", default="/tmp/dawn_operator_dataset_probe")
    parser.add_argument(
        "--reuse-downloads", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse nonempty cached downloads (default: true).")
    parser.add_argument("--output-dir", default="runs/operator_dataset_probe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rows < 5:
        raise ValueError("--sample-rows must be >= 5 so each source records five raw samples")
    if args.max_seq_len <= 0 or args.max_candidate_tokens <= 0:
        raise ValueError("token limits must be positive")
    selected = parse_datasets(args.datasets)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    cache = work_dir / "downloads"
    tokenizer = load_tokenizer(args.tokenizer, work_dir / "tokenizer_cache")
    tok_manifest = tokenizer_manifest(
        tokenizer, args.tokenizer, args.max_seq_len, args.max_candidate_tokens)
    probes: Dict[str, Dict[str, Any]] = {}
    prepared_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[Dict[str, Any]] = []
    print("OPERATOR DATASET PROBE START", flush=True)
    print(f"  datasets={','.join(selected)}", flush=True)
    print(f"  tokenizer={args.tokenizer} vocab={tok_manifest['vocab_size']}", flush=True)
    for dataset in selected:
        print(f"PROBE {dataset} START", flush=True)
        try:
            probe, prepared = PROBE_FUNCS[dataset](args, tokenizer, cache)
        except Exception as exc:
            probe = base_probe(dataset, "probe failed before source contract was established")
            probe["status"] = "failed"
            probe["source_issues"].append(f"{type(exc).__name__}: {exc}")
            prepared = []
            errors.append({
                "dataset": dataset, "stage": "probe", "error_type": type(exc).__name__,
                "error": str(exc), "recorded_at": utc_now(),
            })
        probe["prepared_sample_count"] = len(prepared)
        probe["tokenization_samples"].extend(
            row["human"] for row in prepared[: args.sample_rows])
        probes[dataset] = probe
        prepared_by_dataset[dataset] = prepared
        write_json(output_dir / "source_probe" / f"{dataset}.json", probe)
        write_json(output_dir / "prepared_probe" / f"{dataset}.json", {
            "dataset": dataset,
            "status": probe["status"],
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "array_fields": list(ARRAY_FIELDS),
            "metadata_fields": list(METADATA_FIELDS),
            "samples": prepared,
        })
        write_jsonl(
            output_dir / "prepared_samples" / f"{dataset}.jsonl",
            [row["human"] for row in prepared],
        )
        print(
            f"PROBE {dataset} DONE status={probe['status']} "
            f"raw={len(probe.get('raw_samples') or [])} prepared={len(prepared)}",
            flush=True,
        )
    mapping = dataset_mapping(probes)
    write_json(output_dir / "dataset_mapping.json", mapping)
    (output_dir / "schema_recommendation.md").write_text(
        schema_markdown(probes), encoding="utf-8")
    (output_dir / "source_probe_summary.md").write_text(
        source_summary_markdown(probes), encoding="utf-8")
    write_jsonl(output_dir / "errors.jsonl", errors)
    manifest = {
        "created_at": utc_now(),
        "script": "scripts/probe_operator_analysis_datasets.py",
        "selected_datasets": selected,
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "tokenizer": tok_manifest,
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dataset_status": {dataset: probe["status"] for dataset, probe in probes.items()},
        "raw_sample_counts": {
            dataset: len(probe.get("raw_samples") or []) for dataset, probe in probes.items()},
        "prepared_sample_counts": {
            dataset: len(rows) for dataset, rows in prepared_by_dataset.items()},
        "error_count": len(errors),
        "artifacts": {
            "source_summary": str(output_dir / "source_probe_summary.md"),
            "schema_recommendation": str(output_dir / "schema_recommendation.md"),
            "dataset_mapping": str(output_dir / "dataset_mapping.json"),
            "errors": str(output_dir / "errors.jsonl"),
        },
    }
    write_json(output_dir / "probe_manifest.json", manifest)
    print("OPERATOR DATASET PROBE DONE", flush=True)
    for dataset in selected:
        print(
            f"  {dataset:<12} status={manifest['dataset_status'][dataset]:<20} "
            f"raw={manifest['raw_sample_counts'][dataset]} "
            f"prepared={manifest['prepared_sample_counts'][dataset]}", flush=True)
    print(f"  output={output_dir}", flush=True)
    print(f"  errors={len(errors)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
