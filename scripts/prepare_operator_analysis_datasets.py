#!/usr/bin/env python3
"""Build immutable, analysis-ready DAWN operator dataset shards.

The adapters in this module are pinned to the observed source contract in
``runs/operator_dataset_probe``.  No root pointer is published until every
selected dataset has been tokenized, sharded, and verified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".generated" / "operator_probe_deps"
for candidate in (PROJECT_ROOT, LOCAL_DEPS):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

PREPARATION_DEPENDENCIES = (
    "numpy==1.26.4",
    "pyarrow==20.0.0",
    "transformers==4.40.2",
    "fsspec==2024.3.1",
    "gcsfs==2024.3.1",
)
PREPARATION_MODULES = ("numpy", "pyarrow", "transformers", "gcsfs")


def bootstrap_preparation_dependencies() -> None:
    """Install dataset-only dependencies into an ignored repository cache."""
    missing = [
        module for module in PREPARATION_MODULES
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    install_command = [
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "--upgrade", "--target",
        str(LOCAL_DEPS), *PREPARATION_DEPENDENCIES,
    ]
    enabled = str(os.environ.get(
        "DAWN_OPERATOR_DATASET_AUTO_INSTALL_DEPS", "1"
    )).strip().lower() not in {"0", "false", "no"}
    if not enabled:
        raise RuntimeError(
            "Missing operator dataset dependencies: " + ", ".join(missing)
            + "\nRun: " + " ".join(install_command))
    LOCAL_DEPS.mkdir(parents=True, exist_ok=True)
    if str(LOCAL_DEPS) not in sys.path:
        sys.path.insert(0, str(LOCAL_DEPS))
    print(
        "BOOTSTRAP OPERATOR DATASET DEPS: " + ",".join(missing)
        + f" -> {LOCAL_DEPS}",
        flush=True,
    )
    try:
        subprocess.check_call(install_command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Failed to install operator dataset dependencies. Run manually: "
            + " ".join(install_command)) from exc
    importlib.invalidate_caches()
    unresolved = [
        module for module in PREPARATION_MODULES
        if importlib.util.find_spec(module) is None
    ]
    if unresolved:
        raise RuntimeError(
            "Operator dataset dependency bootstrap completed but imports are "
            "still missing: " + ", ".join(unresolved))


bootstrap_preparation_dependencies()

import numpy as np  # noqa: E402

from analysis.dawn_analysis_storage import (  # noqa: E402
    exists,
    is_gcs_path,
    join_path,
    open_path,
    read_json,
    write_json_atomic,
)
from scripts import probe_operator_analysis_datasets as probe  # noqa: E402

SCHEMA = "dawn_operator_pair_v2"
SCHEMA_VERSION = 2
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "DAWN_OPERATOR_DATASET_ROOT",
    "gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2",
)
DEFAULT_WORK_DIR = "/tmp/dawn_operator_analysis_v2"
DEFAULT_IOI_EXAMPLES = 4096
DEFAULT_SYNTHETIC_EXAMPLES = 4096
DEFAULT_SHARD_SIZE = 1024
SUBSET_SEED = 4171002


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(
        probe.json_safe(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    probe.write_json(path, value)


def safe_remove_build(path: Path, work_dir: Path) -> None:
    resolved = path.resolve()
    allowed = (work_dir / "builds").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside staging builds: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def safe_remove_staged_dataset(path: Path, build_root: Path) -> None:
    resolved = path.resolve()
    allowed = build_root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(
            f"Refusing to remove path outside the active staging build: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_dataset_ids(value: str) -> List[str]:
    return probe.parse_datasets(value)


def tokenizer_info(tokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    info = probe.tokenizer_manifest(
        tokenizer, args.tokenizer, args.max_seq_len, args.max_candidate_tokens)
    info["hash"] = canonical_hash(info)
    info["revision"] = getattr(tokenizer, "init_kwargs", {}).get("revision")
    return info


def contract_info() -> Dict[str, Any]:
    root = PROJECT_ROOT / "runs" / "operator_dataset_probe"
    out: Dict[str, Any] = {"root": str(root), "available": root.exists()}
    for name in ("probe_manifest.json", "dataset_mapping.json", "schema_recommendation.md"):
        path = root / name
        if path.exists():
            out[name] = {"path": str(path), "sha256": sha256_file(path)}
    return out


def stage_fixed_contract(build_root: Path) -> Dict[str, Any]:
    source_root = PROJECT_ROOT / "runs" / "operator_dataset_probe"
    target_root = build_root / "contract"
    files: Dict[str, Any] = {}
    for name in (
        "probe_manifest.json", "dataset_mapping.json",
        "schema_recommendation.md", "source_probe_summary.md",
    ):
        source = source_root / name
        if not source.exists():
            raise FileNotFoundError(f"Fixed probe contract artifact is missing: {source}")
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = {
            "path": target.relative_to(build_root).as_posix(),
            "sha256": sha256_file(target),
        }
    return {"source": str(source_root), "files": files}


def download_sources(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    cache = Path(args.work_dir) / "downloads"
    rows: List[Dict[str, Any]] = []
    if dataset == "ravel":
        paths = probe.hf_parquet_tree(probe.RAVEL_HF_TREE)
        selected = [p for p in paths if p.startswith(("city_entity/", "city_prompt/"))]
        if len(selected) != 6:
            raise RuntimeError(f"RAVEL contract expected 6 city parquet files, got {selected}")
        for rel in selected:
            rows.append(probe.download(
                f"{probe.RAVEL_HF_RESOLVE}/{rel}", cache / "ravel" / "hf" / rel,
                reuse=args.reuse_downloads))
        rows.append(probe.download(
            probe.RAVEL_TGZ_URL, cache / "ravel" / "raw" / "data.tgz",
            reuse=args.reuse_downloads))
    elif dataset == "blimp":
        paths = probe.hf_parquet_tree(probe.BLIMP_HF_TREE)
        if len(paths) != 67:
            raise RuntimeError(f"BLiMP contract expected 67 parquet files, got {len(paths)}")
        for rel in paths:
            rows.append(probe.download(
                f"{probe.BLIMP_HF_RESOLVE}/{rel}", cache / "blimp" / "hf" / rel,
                reuse=args.reuse_downloads))
    elif dataset == "lama":
        rows.append(probe.download(
            probe.LAMA_ZIP_URL, cache / "lama" / "data.zip",
            reuse=args.reuse_downloads))
    elif dataset == "counterfact":
        rows.append(probe.download(
            probe.COUNTERFACT_URL, cache / "counterfact" / "counterfact.json",
            reuse=args.reuse_downloads))
    return {"dataset": dataset, "files": rows}


def _row(
    tokenizer, args: argparse.Namespace, *, index: int, extension: Mapping[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    result = probe.prepared_row(
        tokenizer, row_index=index, max_seq_len=args.max_seq_len,
        max_candidate_tokens=args.max_candidate_tokens,
        extension=extension, **kwargs)
    score_mode = str(result["metadata"]["score_mode"])
    if score_mode != "paired_sequence_logprob":
        for side in ("a", "b"):
            context_len = int(result["arrays"][f"context_len_{side}"])
            if context_len == 0:
                continue
            for prefix in ("positive", "negative"):
                candidate_len = int(result["arrays"][f"{prefix}_len_{side}"])
                if candidate_len and context_len + candidate_len > args.max_seq_len:
                    result["human"]["validation_errors"].append(
                        f"teacher_forced_{prefix}_{side}_too_long")
    pad_token_id = int(tokenizer.pad_token_id or 0)
    for prefix in ("context", "positive", "negative"):
        for side in ("a", "b"):
            ids_key = f"{prefix}_ids_{side}"
            len_key = f"{prefix}_len_{side}"
            length = int(result["arrays"][len_key])
            values = result["arrays"][ids_key]
            values[length:] = [pad_token_id] * max(0, len(values) - length)
    result["extension"] = dict(extension)
    return result


def iter_ioi(tokenizer, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    for index, source in enumerate(probe.generate_ioi_rows(args.ioi_examples)):
        template_id = int(source["template_id"])
        clean_correct = str(source["correct"])
        clean_distractor = str(source["distractor"])
        clean_pattern = "".join(
            match.upper() for match in re.findall(
                r"\{name_([ab])\}", probe.IOI_TEMPLATES[template_id]))
        corrupt_pattern = clean_pattern.translate(str.maketrans({"A": "B", "B": "A"}))
        pattern_type = f"{clean_pattern}_to_{corrupt_pattern}_name_order_swap"
        name_a, name_b = clean_distractor, clean_correct
        place_object = next((
            (place, obj)
            for place in probe.IOI_PLACES for obj in probe.IOI_OBJECTS
            if probe.IOI_TEMPLATES[template_id].format(
                name_a=name_a, name_b=name_b, place=place, object=obj)
            == str(source["clean_prompt"])
        ), (None, None))
        yield _row(
            tokenizer, args, index=index, example_id=str(source["id"]),
            pair_id=str(source["id"]), dataset="ioi", split="generated",
            phenomenon="ioi", relation="clean_corrupt_name_binding",
            group_id=f"template_{template_id}:{pattern_type}", source_id=str(source["id"]),
            score_mode="clean_corrupt_continuation_margin",
            trace_semantics="last_context_token_prediction_state",
            text_a=str(source["clean_prompt"]), text_b=str(source["corrupt_prompt"]),
            positive_a=clean_correct, negative_a=clean_distractor,
            positive_b=clean_distractor, negative_b=clean_correct,
            extension={
                **source,
                "clean_correct": clean_correct,
                "clean_distractor": clean_distractor,
                "corrupt_correct": clean_distractor,
                "corrupt_distractor": clean_correct,
                "pattern_type": pattern_type,
                "name_a": name_a, "name_b": name_b,
                "place": place_object[0], "object": place_object[1],
            },
        )


def _synthetic_pair(index: int, rng: random.Random) -> Dict[str, Any]:
    families = (
        "same_operation_different_entity", "same_entity_different_attribute",
        "attribute_swap", "entity_swap", "distractor_swap", "order_permutation",
        "irrelevant_fact_addition",
    )
    family = families[index % len(families)]
    entity, paired_entity, other = rng.sample(probe.SYNTHETIC_ENTITIES, 3)
    color, distractor, paired_color = rng.sample(probe.SYNTHETIC_COLORS, 3)
    key, key_distractor = rng.sample(probe.SYNTHETIC_KEYS, 2)
    context_a = f"{entity} is {color}. {paired_entity} is {distractor}. The color of {entity} is"
    context_b = f"{other} is {color}. {paired_entity} is {distractor}. The color of {other} is"
    positive_b, negative_b = color, distractor
    entity_b = other
    attribute_a = attribute_b = "color"
    if family == "same_entity_different_attribute":
        context_b = f"{entity} has {key}. {paired_entity} has {key_distractor}. {entity}'s key is"
        positive_b, negative_b, attribute_b = key, key_distractor, "key"
        entity_b = entity
    elif family == "attribute_swap":
        context_b = f"{entity} is {paired_color}. {paired_entity} is {distractor}. The color of {entity} is"
        positive_b = paired_color
        entity_b = entity
    elif family == "entity_swap":
        context_b = f"{entity} is {color}. {paired_entity} is {distractor}. The color of {paired_entity} is"
        positive_b, negative_b = distractor, color
        entity_b = paired_entity
    elif family == "distractor_swap":
        context_b = f"{entity} is {color}. {paired_entity} is {paired_color}. The color of {entity} is"
        negative_b = paired_color
        entity_b = entity
    elif family == "order_permutation":
        context_b = f"{paired_entity} is {distractor}. {entity} is {color}. The color of {entity} is"
        entity_b = entity
    elif family == "irrelevant_fact_addition":
        context_b = (
            f"{entity} is {color}. {paired_entity} is {distractor}. "
            f"{other} is {paired_color}. The color of {entity} is")
        entity_b = entity
    return {
        "pair_id": f"synthetic-{index:06d}", "family": family,
        "context_a": context_a, "context_b": context_b,
        "positive_a": color, "negative_a": distractor,
        "positive_b": positive_b, "negative_b": negative_b,
        "entity": entity, "entity_a": entity, "entity_b": entity_b,
        "paired_entity": paired_entity,
        "attribute_a": attribute_a, "attribute_b": attribute_b,
        "attribute": attribute_a, "paired_attribute": attribute_b,
        "correct_value": color, "distractor_value": distractor,
        "controlled_change": family,
    }


def iter_synthetic(tokenizer, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    rng = random.Random(4171)
    for index in range(args.synthetic_examples):
        source = _synthetic_pair(index, rng)
        yield _row(
            tokenizer, args, index=index, example_id=source["pair_id"],
            pair_id=source["pair_id"], dataset="synthetic", split="generated",
            phenomenon=source["family"], relation=source["family"],
            group_id=source["family"], source_id=source["pair_id"],
            score_mode="continuation_margin",
            trace_semantics="last_context_token_prediction_state",
            text_a=source["context_a"], text_b=source["context_b"],
            positive_a=source["positive_a"], negative_a=source["negative_a"],
            positive_b=source["positive_b"], negative_b=source["negative_b"],
            extension=source,
        )


def iter_blimp(tokenizer, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    root = Path(args.work_dir) / "downloads" / "blimp" / "hf"
    paths = probe.hf_parquet_tree(probe.BLIMP_HF_TREE)
    index = 0
    expected_schema = {
        "UID", "field", "lexically_identical", "linguistics_term",
        "one_prefix_method", "pair_id", "sentence_bad", "sentence_good",
        "simple_LM_method", "two_prefix_method",
    }
    for rel in paths:
        rows = probe.read_parquet_rows(root / rel)
        if rows and set(rows[0]) != expected_schema:
            raise RuntimeError(f"BLiMP schema drift in {rel}: {sorted(rows[0])}")
        phenomenon = rel.split("/", 1)[0]
        for source in rows:
            uid = str(source.get("UID") or phenomenon)
            pair_index = source.get("pair_id", index)
            yield _row(
                tokenizer, args, index=index,
                example_id=f"blimp-{phenomenon}-{pair_index}",
                pair_id=f"blimp-{phenomenon}-{pair_index}", dataset="blimp",
                split="train", phenomenon=phenomenon,
                relation="grammatical_minimal_pair", group_id=phenomenon,
                source_id=f"{uid}:{pair_index}", score_mode="paired_sequence_logprob",
                trace_semantics="pre_divergence_prediction_state",
                text_a=str(source["sentence_good"]), text_b=str(source["sentence_bad"]),
                extension={"source_path": rel, **source},
            )
            index += 1


def iter_ravel(tokenizer, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    root = Path(args.work_dir) / "downloads" / "ravel" / "hf"
    index = 0
    families = (
        "same_attribute_different_entity", "same_entity_different_attribute",
        "same_attribute_different_prompt", "cross_attribute_control",
    )
    for split in ("train", "val", "test"):
        entities = probe.read_parquet_rows(root / "city_entity" / f"{split}-00000-of-00001.parquet")
        prompts = probe.read_parquet_rows(root / "city_prompt" / f"{split}-00000-of-00001.parquet")
        if not entities or set(entities[0]) != {
            "ID", "City", "Continent", "Country", "Language", "Latitude",
            "Longitude", "Timezone", "URL",
        }:
            raise RuntimeError(f"RAVEL city_entity schema drift in {split}")
        if not prompts or set(prompts[0]) != {"Template", "Attribute", "Source", "Entity"}:
            raise RuntimeError(f"RAVEL city_prompt schema drift in {split}")
        by_attr: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in prompts:
            attribute = str(row["Attribute"])
            if attribute in entities[0] and str(row["Template"]).count("%s") == 1:
                by_attr[attribute].append(row)
        attributes = sorted(attr for attr, values in by_attr.items() if values)
        if len(attributes) < 2:
            raise RuntimeError(f"RAVEL has fewer than two joinable attributes in {split}")
        for local_index, entity_a in enumerate(entities):
            family = families[local_index % len(families)]
            attr_a = attributes[local_index % len(attributes)]
            attr_b = attr_a
            entity_b = entities[(local_index + 1) % len(entities)]
            prompt_a = by_attr[attr_a][local_index % len(by_attr[attr_a])]
            prompt_b = prompt_a
            if family in ("same_entity_different_attribute", "cross_attribute_control"):
                attr_b = attributes[(attributes.index(attr_a) + 1) % len(attributes)]
                prompt_b = by_attr[attr_b][local_index % len(by_attr[attr_b])]
            if family == "same_entity_different_attribute":
                entity_b = entity_a
            if family == "same_attribute_different_prompt":
                entity_b = entity_a
                prompt_b = by_attr[attr_a][(local_index + 1) % len(by_attr[attr_a])]

            def matched_negative(entity: Mapping[str, Any], attribute: str) -> str:
                for offset in range(1, len(entities)):
                    other = entities[(local_index + offset) % len(entities)]
                    value = str(other.get(attribute, ""))
                    if value and value != str(entity.get(attribute, "")):
                        return value
                raise RuntimeError(f"RAVEL lacks matched negative for {attribute}")

            context_a = str(prompt_a["Template"]) % str(entity_a["City"])
            context_b = str(prompt_b["Template"]) % str(entity_b["City"])
            yield _row(
                tokenizer, args, index=index, example_id=f"ravel-{split}-{local_index:05d}",
                pair_id=f"ravel-{split}-{local_index:05d}", dataset="ravel", split=split,
                phenomenon=family, relation=f"{attr_a}->{attr_b}",
                group_id=f"{family}:{attr_a}",
                source_id=f"{entity_a['ID']}:{entity_b['ID']}",
                score_mode="continuation_margin",
                trace_semantics="last_context_token_prediction_state",
                text_a=context_a, text_b=context_b,
                positive_a=str(entity_a[attr_a]), negative_a=matched_negative(entity_a, attr_a),
                positive_b=str(entity_b[attr_b]), negative_b=matched_negative(entity_b, attr_b),
                extension={
                    "family": family, "entity_a": entity_a, "entity_b": entity_b,
                    "attribute_a": attr_a, "attribute_b": attr_b,
                    "template_a": prompt_a, "template_b": prompt_b,
                },
            )
            index += 1


def _zip_jsonl_rows(archive: zipfile.ZipFile, name: str) -> List[Dict[str, Any]]:
    with archive.open(name) as handle:
        return [json.loads(line.decode("utf-8")) for line in handle if line.strip()]


def iter_lama(tokenizer, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    path = Path(args.work_dir) / "downloads" / "lama" / "data.zip"
    index = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        relation_file = next((name for name in names if name.endswith("relations.jsonl")), None)
        relations: Dict[str, Dict[str, Any]] = {}
        if relation_file:
            for row in _zip_jsonl_rows(archive, relation_file):
                relation_id = str(row.get("relation") or row.get("predicate_id") or "")
                if relation_id:
                    relations[relation_id] = row
        data_files = [
            name for name in names
            if name.endswith(".jsonl") and name != relation_file
        ]
        for name in data_files:
            rows = _zip_jsonl_rows(archive, name)
            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for source in rows:
                relation_id = str(
                    source.get("predicate_id") or source.get("pred") or
                    relations.get(Path(name).stem, {}).get("relation") or Path(name).stem)
                groups[relation_id].append(source)
            category = next((
                value for value in ("TREx", "Google_RE", "ConceptNet", "Squad")
                if value.lower() in name.lower()), "other")
            for relation_id, group in groups.items():
                objects = [
                    str(row.get("obj_label") or row.get("obj") or "")
                    for row in group
                ]
                for local_index, source in enumerate(group):
                    subject = source.get("sub_label") or source.get("sub")
                    positive = source.get("obj_label") or source.get("obj")
                    if not subject or not positive:
                        continue
                    negative = next((value for value in objects if value and value != str(positive)), None)
                    if not negative:
                        continue
                    masked_values = source.get("masked_sentences") or source.get("masked_sentence")
                    masked = masked_values[0] if isinstance(masked_values, list) and masked_values else masked_values
                    relation_meta = relations.get(relation_id) or relations.get(Path(name).stem) or {}
                    template = relation_meta.get("template")
                    if not masked and template:
                        masked = str(template).replace("[X]", str(subject)).replace("[Y]", "[MASK]")
                    if not masked:
                        continue
                    context, risk = probe._lama_prefix(str(masked))
                    if risk:
                        continue
                    yield _row(
                        tokenizer, args, index=index,
                        example_id=f"lama-{category}-{relation_id}-{local_index}",
                        pair_id=f"lama-{category}-{relation_id}-{local_index}",
                        dataset="lama", split=category, phenomenon="factual_recall",
                        relation=relation_id, group_id=relation_id,
                        source_id=f"{name}:{local_index}", score_mode="continuation_margin",
                        trace_semantics="last_context_token_prediction_state",
                        text_a=str(context), positive_a=str(positive), negative_a=str(negative),
                        extension={
                            "source_file": name, "source_category": category,
                            "subject": subject, "masked_sentence": masked,
                            "relation_template": template, **source,
                        },
                    )
                    index += 1


def iter_counterfact(tokenizer, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    path = Path(args.work_dir) / "downloads" / "counterfact" / "counterfact.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("CounterFact top-level value is not a list")
    for index, source in enumerate(rows):
        rewrite = source.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            continue
        prompt_text = rewrite.get("prompt")
        subject = rewrite.get("subject")
        target_true = rewrite.get("target_true")
        target_new = rewrite.get("target_new")
        positive = target_true.get("str") if isinstance(target_true, Mapping) else target_true
        negative = target_new.get("str") if isinstance(target_new, Mapping) else target_new
        if not all(v is not None for v in (prompt_text, subject, positive, negative)):
            continue
        prompt_text = str(prompt_text)
        if prompt_text.count("{}") != 1:
            continue
        context = prompt_text.format(subject)
        case_id = source.get("case_id", index)
        relation_id = str(rewrite.get("relation_id") or "unknown_relation")
        yield _row(
            tokenizer, args, index=index, example_id=f"counterfact-{case_id}",
            pair_id=f"counterfact-{case_id}", dataset="counterfact",
            split="counterfact", phenomenon="factual_recall", relation=relation_id,
            group_id=relation_id, source_id=str(case_id),
            score_mode="true_new_object_margin",
            trace_semantics="last_context_token_prediction_state",
            text_a=context, positive_a=str(positive), negative_a=str(negative),
            extension=source,
        )


ITERATORS = {
    "ioi": iter_ioi, "synthetic": iter_synthetic, "blimp": iter_blimp,
    "ravel": iter_ravel, "lama": iter_lama, "counterfact": iter_counterfact,
}


def stage_source_contract(
    dataset_root: Path, dataset_id: str, source_downloads: Mapping[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    """Place the exact probed source contract inside the immutable build."""
    raw_root = dataset_root / "raw"
    staged: List[str] = []
    download_root = Path(args.work_dir) / "downloads" / dataset_id
    for row in source_downloads.get("files", []):
        source = Path(str(row["path"]))
        try:
            relative = source.relative_to(download_root)
        except ValueError:
            relative = Path(source.name)
        target = raw_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256_file(target) != str(row["sha256"]):
            shutil.copy2(source, target)
        staged.append(target.relative_to(dataset_root).as_posix())
    observed = PROJECT_ROOT / "runs" / "operator_dataset_probe" / "source_probe" / f"{dataset_id}.json"
    if observed.exists():
        target = raw_root / "observed_probe.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(observed, target)
        staged.append(target.relative_to(dataset_root).as_posix())
    if not source_downloads.get("files"):
        target = raw_root / "generator_config.json"
        write_json(target, {
            "dataset": dataset_id, "contract": "runs/operator_dataset_probe",
            "seed": 4171,
            "examples": args.ioi_examples if dataset_id == "ioi" else args.synthetic_examples,
        })
        staged.append(target.relative_to(dataset_root).as_posix())
        rows_path = raw_root / "generated_rows.jsonl"
        generated: Iterable[Mapping[str, Any]]
        if dataset_id == "ioi":
            generated = probe.generate_ioi_rows(args.ioi_examples)
        else:
            rng = random.Random(4171)
            generated = (
                _synthetic_pair(index, rng)
                for index in range(args.synthetic_examples))
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in generated:
                handle.write(json.dumps(
                    probe.json_safe(row), sort_keys=True,
                    ensure_ascii=False) + "\n")
        staged.append(rows_path.relative_to(dataset_root).as_posix())
    return staged


def count_contract_source_rows(dataset_id: str, args: argparse.Namespace) -> int:
    if dataset_id == "ioi":
        return int(args.ioi_examples)
    if dataset_id == "synthetic":
        return int(args.synthetic_examples)
    root = Path(args.work_dir) / "downloads" / dataset_id
    if dataset_id == "blimp":
        return sum(len(probe.read_parquet_rows(root / "hf" / rel))
                   for rel in probe.hf_parquet_tree(probe.BLIMP_HF_TREE))
    if dataset_id == "ravel":
        return sum(len(probe.read_parquet_rows(
            root / "hf" / "city_entity" / f"{split}-00000-of-00001.parquet"))
            for split in ("train", "val", "test"))
    if dataset_id == "counterfact":
        return len(json.loads((root / "counterfact.json").read_text(encoding="utf-8")))
    if dataset_id == "lama":
        with zipfile.ZipFile(root / "data.zip") as archive:
            return sum(
                len(_zip_jsonl_rows(archive, name))
                for name in archive.namelist()
                if name.endswith(".jsonl") and not name.endswith("relations.jsonl"))
    raise ValueError(dataset_id)


def write_shard(
    dataset_root: Path, dataset_id: str, shard_index: int,
    rows: Sequence[Dict[str, Any]], locators: List[Dict[str, Any]],
) -> Dict[str, Any]:
    shard_name = f"shard-{shard_index:05d}"
    root = dataset_root / "prepared" / shard_name
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / "arrays.npz"
    metadata_path = root / "rows.jsonl.gz"
    arrays: Dict[str, np.ndarray] = {}
    for field in probe.ARRAY_FIELDS:
        values = [row["arrays"][field] for row in rows]
        dtype = np.int64 if field == "row_index" else np.int32
        arrays[field] = np.asarray(values, dtype=dtype)
    np.savez_compressed(arrays_path, **arrays)
    with gzip.open(metadata_path, "wt", encoding="utf-8") as handle:
        for offset, row in enumerate(rows):
            extension = dict(row.get("extension", {}))
            metadata = {**row["metadata"], **{
                key: value for key, value in row["human"].items()
                if key in ("trace_details", "teacher_forced_candidate_span_a",
                           "teacher_forced_candidate_span_b")
            }, "metadata": extension}
            for key in (
                "attribute", "entity", "subject", "relation_id",
                "template_id", "control_type", "family", "name_a",
                "name_b", "place", "object", "case_id",
            ):
                if extension.get(key) is not None:
                    metadata[key] = extension[key]
            if dataset_id == "ravel":
                metadata["attribute"] = extension.get("attribute_a")
                metadata["entity"] = (extension.get("entity_a") or {}).get("City")
                metadata["prompt_template"] = (extension.get("template_a") or {}).get("Template")
                metadata["control_type"] = extension.get("family")
            elif dataset_id == "lama":
                metadata["relation_id"] = metadata["relation"]
            elif dataset_id == "counterfact":
                rewrite = extension.get("requested_rewrite") or {}
                metadata["subject"] = rewrite.get("subject")
                metadata["relation_id"] = rewrite.get("relation_id")
                metadata["case_id"] = extension.get("case_id")
            elif dataset_id == "synthetic":
                metadata["control_type"] = extension.get("controlled_change")
            handle.write(json.dumps(probe.json_safe(metadata), sort_keys=True, ensure_ascii=False) + "\n")
            locators.append({
                "example_id": metadata["example_id"], "shard": shard_name,
                "row_offset": offset, "group": metadata["group_id"],
                "phenomenon": metadata["phenomenon"], "relation": metadata["relation"],
            })
    arrays_hash = sha256_file(arrays_path)
    metadata_hash = sha256_file(metadata_path)
    return {
        "rows": len(rows),
        "arrays": f"prepared/{shard_name}/arrays.npz",
        "metadata": f"prepared/{shard_name}/rows.jsonl.gz",
        "arrays_bytes": arrays_path.stat().st_size,
        "metadata_bytes": metadata_path.stat().st_size,
        "arrays_sha256": arrays_hash,
        "metadata_sha256": metadata_hash,
        "sha256": sha256_bytes(f"{arrays_hash}:{metadata_hash}".encode()),
        "first_example_id": rows[0]["metadata"]["example_id"],
        "last_example_id": rows[-1]["metadata"]["example_id"],
    }


def stratified_subset(
    locators: Sequence[Dict[str, Any]], limit: int, *, seed: int,
    max_per_group: Optional[int], reason: str,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in locators:
        groups[str(row["group"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: List[Dict[str, Any]] = []
    level = 0
    while len(selected) < limit:
        added = False
        for group in sorted(groups):
            values = groups[group]
            if level < len(values) and (max_per_group is None or level < max_per_group):
                selected.append({
                    **{key: values[level][key] for key in (
                        "example_id", "shard", "row_offset", "group")},
                    "selection_reason": reason, "seed": seed,
                })
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        level += 1
    return {"seed": seed, "selection_reason": reason, "entries": selected}


def write_subsets(dataset_root: Path, dataset_id: str,
                  locators: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    monitor_limits = {
        "ioi": (256, None), "blimp": (16 * 67, 16), "ravel": (512, None),
        "synthetic": (512, None), "lama": (512, None), "counterfact": (512, None),
    }
    specs = {
        "smoke": (32, None, "deterministic_smoke"),
        "monitor": (*monitor_limits[dataset_id], "stratified_monitor"),
        "trace": (128, None, "stratified_trace"),
        "causal": (48, None, "stratified_causal_behavior_priority_at_runtime"),
    }
    paths: Dict[str, str] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    for offset, (name, (limit, max_per_group, reason)) in enumerate(specs.items()):
        source_locators = (
            payloads["monitor"]["entries"]
            if name in {"trace", "causal"} and "monitor" in payloads
            else locators)
        payload = stratified_subset(
            source_locators, min(int(limit), len(source_locators)), seed=SUBSET_SEED + offset,
            max_per_group=max_per_group, reason=reason)
        path = dataset_root / "subsets" / f"{name}.json"
        write_json(path, payload)
        paths[name] = f"subsets/{name}.json"
        payloads[name] = payload
    return paths


def validate_local_shard(dataset_root: Path, shard: Mapping[str, Any],
                         args: argparse.Namespace) -> Dict[str, Any]:
    arrays_path = dataset_root / str(shard["arrays"])
    metadata_path = dataset_root / str(shard["metadata"])
    if sha256_file(arrays_path) != shard["arrays_sha256"]:
        raise RuntimeError(f"arrays checksum mismatch: {arrays_path}")
    if sha256_file(metadata_path) != shard["metadata_sha256"]:
        raise RuntimeError(f"metadata checksum mismatch: {metadata_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        shapes = {key: list(arrays[key].shape) for key in arrays.files}
        n = int(arrays["row_index"].shape[0])
        if set(probe.ARRAY_FIELDS) - set(arrays.files):
            raise RuntimeError(f"missing arrays: {arrays_path}")
        if arrays["context_ids_a"].shape != (n, args.max_seq_len):
            raise RuntimeError(f"context shape mismatch: {arrays_path}")
        if arrays["positive_ids_a"].shape != (n, args.max_candidate_tokens):
            raise RuntimeError(f"candidate shape mismatch: {arrays_path}")
        if np.any(arrays["context_len_a"] > args.max_seq_len):
            raise RuntimeError(f"context bounds mismatch: {arrays_path}")
        if np.any(arrays["positive_len_a"] > args.max_candidate_tokens):
            raise RuntimeError(f"candidate bounds mismatch: {arrays_path}")
        for side in ("a", "b"):
            starts = arrays[f"target_span_start_{side}"]
            ends = arrays[f"target_span_end_{side}"]
            lengths = arrays[f"context_len_{side}"]
            valid = ((starts == -1) & (ends == -1)) | (
                (starts >= 0) & (ends > starts) & (ends <= lengths))
            if not np.all(valid):
                raise RuntimeError(f"target span bounds mismatch: {arrays_path}:{side}")
    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle if line.strip()]
    if len(metadata) != n:
        raise RuntimeError(f"metadata alignment mismatch: {metadata_path}")
    if metadata and metadata[0]["example_id"] != shard["first_example_id"]:
        raise RuntimeError(f"example alignment mismatch: {metadata_path}")
    return {"rows": n, "shapes": shapes, "metadata_rows": len(metadata)}


def prepare_one_dataset(
    dataset_id: str, build_root: Path, tokenizer, args: argparse.Namespace,
    source_downloads: Mapping[str, Any],
) -> Dict[str, Any]:
    dataset_root = build_root / dataset_id
    locators: List[Dict[str, Any]] = []
    shards: List[Dict[str, Any]] = []
    drops: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    phenomenon_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    adapter_rows = 0
    dropped_rows = 0
    prepared_rows = 0
    batch: List[Dict[str, Any]] = []
    limit = int(args.limit_per_dataset or 0)
    for adapter_rows, row in enumerate(ITERATORS[dataset_id](tokenizer, args), start=1):
        errors = list(row["human"].get("validation_errors") or [])
        if errors:
            dropped_rows += 1
            for error in errors:
                drops[str(error)] += 1
            continue
        batch.append(row)
        meta = row["metadata"]
        group_counts[str(meta["group_id"])] += 1
        split_counts[str(meta["split"])] += 1
        phenomenon_counts[str(meta["phenomenon"])] += 1
        relation_counts[str(meta["relation"])] += 1
        prepared_rows += 1
        if len(batch) >= args.shard_size:
            shards.append(write_shard(dataset_root, dataset_id, len(shards), batch, locators))
            batch = []
        if limit and prepared_rows >= limit:
            break
    if batch:
        shards.append(write_shard(dataset_root, dataset_id, len(shards), batch, locators))
    if not prepared_rows:
        raise RuntimeError(f"No prepared rows for {dataset_id}; drops={dict(drops)}")
    staged_sources = stage_source_contract(
        dataset_root, dataset_id, source_downloads, args)
    source_rows = count_contract_source_rows(dataset_id, args)
    unadapted_rows = 0
    omitted_by_limit = 0
    if limit:
        omitted_by_limit = max(0, source_rows - adapter_rows)
    else:
        unadapted_rows = max(0, source_rows - adapter_rows)
        if unadapted_rows:
            drops["source_not_decoder_safe_or_missing_pair"] += unadapted_rows
            dropped_rows += unadapted_rows
    subsets = write_subsets(dataset_root, dataset_id, locators)
    manifest = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "dataset": dataset_id, "source": source_downloads,
        "source_version": "observed-probe-contract-2026-07-14",
        "source_files": staged_sources,
        "rows_source": source_rows, "rows_prepared": prepared_rows,
        "rows_adapter_seen": adapter_rows,
        "rows_unadapted": unadapted_rows,
        "rows_omitted_by_limit": omitted_by_limit,
        "rows_dropped": dropped_rows, "prepared_rows": prepared_rows,
        "shards": shards, "group_counts": dict(group_counts),
        "split_counts": dict(split_counts), "phenomenon_counts": dict(phenomenon_counts),
        "relation_counts": dict(relation_counts), "drop_reason_counts": dict(drops),
        "subsets": subsets, "subset_seed": SUBSET_SEED,
    }
    manifest["first_shard_validation"] = validate_local_shard(dataset_root, shards[0], args)
    write_json(dataset_root / "manifest.json", manifest)
    return manifest


def upload_file(local: Path, remote: str) -> None:
    with local.open("rb") as source, open_path(remote, "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def publish_build(local_build: Path, output_root: str, build_id: str,
                  args: argparse.Namespace) -> str:
    remote_build = join_path(output_root, "builds", build_id)
    if is_gcs_path(output_root):
        if args.skip_upload:
            return remote_build
        remote_manifest = join_path(remote_build, "manifest.json")
        if exists(remote_manifest):
            current = read_json(remote_manifest, {})
            local = json.loads((local_build / "manifest.json").read_text(encoding="utf-8"))
            if canonical_hash(current) == canonical_hash(local):
                return remote_build
            raise FileExistsError(f"Immutable build id already exists with different content: {remote_build}")
        immutable_manifest = local_build / "manifest.json"
        for path in sorted(local_build.rglob("*")):
            if path == immutable_manifest:
                continue
            if path.is_file():
                upload_file(path, join_path(remote_build, path.relative_to(local_build).as_posix()))
        # The immutable manifest is the build completion marker.  Publish it
        # only after every source, shard, subset, and dataset manifest exists.
        upload_file(immutable_manifest, join_path(remote_build, "manifest.json"))
    else:
        target = Path(remote_build)
        if target.exists():
            current_path = target / "manifest.json"
            if current_path.exists():
                current = json.loads(current_path.read_text(encoding="utf-8"))
                local = json.loads((local_build / "manifest.json").read_text(encoding="utf-8"))
                if canonical_hash(current) == canonical_hash(local):
                    return str(target)
            raise FileExistsError(f"Immutable build id already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_build, target)
        remote_build = str(target)
    return remote_build


def validate_published_first_shards(
    remote_build: str, datasets: Sequence[str], args: argparse.Namespace,
) -> Dict[str, Any]:
    """Re-read each published first shard before exposing a root pointer."""
    build_manifest = read_json(join_path(remote_build, "manifest.json"), None)
    if not isinstance(build_manifest, Mapping) or build_manifest.get("status") != "complete":
        raise RuntimeError(f"Published immutable manifest is incomplete: {remote_build}")
    verified: Dict[str, Any] = {}
    for dataset_id in datasets:
        manifest_path = join_path(remote_build, dataset_id, "manifest.json")
        manifest = read_json(manifest_path, None)
        if not isinstance(manifest, Mapping) or not manifest.get("shards"):
            raise RuntimeError(f"Published dataset manifest is incomplete: {manifest_path}")
        dataset_entry = (build_manifest.get("datasets") or {}).get(dataset_id) or {}
        expected_manifest_hash = str(dataset_entry.get("manifest_hash") or "")
        actual_manifest_hash = canonical_hash(manifest)
        if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
            raise RuntimeError(
                "Published dataset manifest hash mismatch: "
                f"{dataset_id}: expected={expected_manifest_hash!r} "
                f"actual={actual_manifest_hash}")
        shard = manifest["shards"][0]
        arrays_path = join_path(remote_build, dataset_id, *str(shard["arrays"]).split("/"))
        metadata_path = join_path(remote_build, dataset_id, *str(shard["metadata"]).split("/"))
        with open_path(arrays_path, "rb") as handle:
            arrays_payload = handle.read()
        with open_path(metadata_path, "rb") as handle:
            metadata_payload = handle.read()
        if sha256_bytes(arrays_payload) != str(shard["arrays_sha256"]):
            raise RuntimeError(f"Published arrays checksum mismatch: {arrays_path}")
        if sha256_bytes(metadata_payload) != str(shard["metadata_sha256"]):
            raise RuntimeError(f"Published metadata checksum mismatch: {metadata_path}")
        with np.load(io.BytesIO(arrays_payload), allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        with gzip.GzipFile(fileobj=io.BytesIO(metadata_payload), mode="rb") as handle:
            metadata = [
                json.loads(line.decode("utf-8"))
                for line in handle if line.strip()
            ]
        missing = set(probe.ARRAY_FIELDS) - set(arrays)
        n = int(arrays["row_index"].shape[0]) if not missing else -1
        if missing or n != int(shard["rows"]) or len(metadata) != n:
            raise RuntimeError(f"Published first-shard shape/alignment mismatch: {dataset_id}")
        if any(value.shape[0] != n for value in arrays.values()):
            raise RuntimeError(f"Published first-shard array row mismatch: {dataset_id}")
        if arrays["context_ids_a"].shape != (n, args.max_seq_len):
            raise RuntimeError(f"Published context shape mismatch: {dataset_id}")
        if arrays["positive_ids_a"].shape != (n, args.max_candidate_tokens):
            raise RuntimeError(f"Published candidate shape mismatch: {dataset_id}")
        for side in ("a", "b"):
            if np.any(arrays[f"context_len_{side}"] > args.max_seq_len):
                raise RuntimeError(f"Published context length out of bounds: {dataset_id}")
            for polarity in ("positive", "negative"):
                if np.any(arrays[f"{polarity}_len_{side}"] > args.max_candidate_tokens):
                    raise RuntimeError(f"Published candidate length out of bounds: {dataset_id}")
            starts = arrays[f"target_span_start_{side}"]
            ends = arrays[f"target_span_end_{side}"]
            lengths = arrays[f"context_len_{side}"]
            valid_spans = ((starts == -1) & (ends == -1)) | (
                (starts >= 0) & (ends > starts) & (ends <= lengths))
            if not np.all(valid_spans):
                raise RuntimeError(f"Published target span out of bounds: {dataset_id}")
        if metadata and (
            metadata[0].get("example_id") != shard.get("first_example_id")
            or metadata[-1].get("example_id") != shard.get("last_example_id")
        ):
            raise RuntimeError(f"Published example-id alignment mismatch: {dataset_id}")
        verified[dataset_id] = {
            "rows": n, "arrays": arrays_path, "metadata": metadata_path,
        }
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DAWN operator analysis dataset build v2.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-candidate-tokens", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--ioi-examples", type=int, default=DEFAULT_IOI_EXAMPLES)
    parser.add_argument("--synthetic-examples", type=int, default=DEFAULT_SYNTHETIC_EXAMPLES)
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--publish-latest", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--overwrite-local", action="store_true")
    parser.add_argument(
        "--reuse-downloads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = parse_dataset_ids(args.datasets)
    output_root = str(args.output_root).rstrip("/\\")
    work_dir = Path(args.work_dir)
    if args.shard_size <= 0 or args.max_seq_len <= 0 or args.max_candidate_tokens <= 0:
        raise ValueError("sequence/candidate/shard sizes must be positive")
    if args.dry_run:
        print(json.dumps({
            "output_root": output_root, "work_dir": str(work_dir),
            "datasets": selected, "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        }, indent=2))
        return 0
    tokenizer = probe.load_tokenizer(args.tokenizer, work_dir / "tokenizer_cache")
    token_info = tokenizer_info(tokenizer, args)
    fixed_probe_manifest = PROJECT_ROOT / "runs" / "operator_dataset_probe" / "probe_manifest.json"
    if not fixed_probe_manifest.exists():
        raise FileNotFoundError(
            "Fixed source contract is missing runs/operator_dataset_probe/probe_manifest.json")
    fixed_tokenizer = json.loads(
        fixed_probe_manifest.read_text(encoding="utf-8")).get("tokenizer") or {}
    if (
        token_info.get("tokenizer_name") != fixed_tokenizer.get("tokenizer_name")
        or token_info.get("vocab_hash") != fixed_tokenizer.get("vocab_hash")
        or token_info.get("add_special_tokens") is not False
    ):
        raise ValueError(
            "Tokenizer conflicts with fixed operator_dataset_probe contract: "
            f"requested={token_info.get('tokenizer_name')} "
            f"vocab_hash={token_info.get('vocab_hash')}")
    code_hash = canonical_hash({
        "prepare": sha256_file(Path(__file__)),
        "probe": sha256_file(Path(probe.__file__)),
    })
    source_config = {
        "datasets": selected, "ioi_examples": args.ioi_examples,
        "synthetic_examples": args.synthetic_examples,
        "limit_per_dataset": args.limit_per_dataset,
        "max_seq_len": args.max_seq_len,
        "max_candidate_tokens": args.max_candidate_tokens,
        "shard_size": args.shard_size,
        "subset_seed": SUBSET_SEED,
        "urls": {
            "ravel": [probe.RAVEL_HF_TREE, probe.RAVEL_TGZ_URL],
            "blimp": probe.BLIMP_HF_TREE, "lama": probe.LAMA_ZIP_URL,
            "counterfact": probe.COUNTERFACT_URL,
        },
    }
    source_hash = canonical_hash(source_config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build_id = args.build_id
    if build_id is None:
        candidates = sorted((work_dir / "builds").glob("*/manifest.json"), reverse=True)
        for candidate in candidates:
            try:
                previous = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                previous.get("status") == "complete"
                and previous.get("datasets_selected") == selected
                and (previous.get("tokenizer") or {}).get("hash") == token_info["hash"]
                and previous.get("preparation_code_hash") == code_hash
                and previous.get("source_configuration_hash") == source_hash
            ):
                build_id = str(previous["build_id"])
                print(f"BUILD AUTO-RESUME candidate build_id={build_id}", flush=True)
                break
    build_id = build_id or (
        f"{timestamp}-s{SCHEMA_VERSION}-{token_info['hash'][:8]}-"
        f"{code_hash[:8]}-{source_hash[:8]}")
    build_root = work_dir / "builds" / build_id
    if build_root.exists() and args.overwrite_local:
        safe_remove_build(build_root, work_dir)
    manifest_path = build_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite_local:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("datasets_selected") == selected:
            print(f"BUILD RESUME complete build_id={build_id}")
            remote_build = publish_build(build_root, output_root, build_id, args)
            if is_gcs_path(output_root) and not args.skip_upload:
                validate_published_first_shards(remote_build, selected, args)
            if args.publish_latest and not (is_gcs_path(output_root) and args.skip_upload):
                manifest_hash = canonical_hash(existing)
                write_json_atomic(join_path(output_root, "manifest.json"), existing)
                write_json_atomic(join_path(output_root, "LATEST.json"), {
                    "build_id": build_id, "build_path": f"builds/{build_id}",
                    "manifest_hash": manifest_hash, "published_at": utc_now(),
                })
            return 0
    build_root.mkdir(parents=True, exist_ok=True)
    staged_contract = stage_fixed_contract(build_root)
    t0 = time.time()
    source_manifest = {
        "created_at": utc_now(), "contract": {
            **contract_info(), "staged": staged_contract},
        "source_configuration": source_config, "source_configuration_hash": source_hash,
        "datasets": {},
    }
    dataset_manifests: Dict[str, Any] = {}
    for dataset_id in selected:
        # A build without a complete root manifest is never resumable by
        # pointer. Regenerate each selected dataset inside this build so stale
        # or extra shards from an interrupted attempt cannot be published.
        safe_remove_staged_dataset(build_root / dataset_id, build_root)
        print(f"DATASET {dataset_id} DOWNLOAD", flush=True)
        downloads = download_sources(args, dataset_id)
        source_manifest["datasets"][dataset_id] = downloads
        print(f"DATASET {dataset_id} PREPARE", flush=True)
        dataset_manifests[dataset_id] = prepare_one_dataset(
            dataset_id, build_root, tokenizer, args, downloads)
        print(
            f"DATASET {dataset_id} DONE rows_source={dataset_manifests[dataset_id]['rows_source']} "
            f"prepared={dataset_manifests[dataset_id]['prepared_rows']} "
            f"dropped={dataset_manifests[dataset_id]['rows_dropped']} "
            f"shards={len(dataset_manifests[dataset_id]['shards'])}", flush=True)
    write_json(build_root / "tokenizer.json", token_info)
    write_json(build_root / "source_manifest.json", source_manifest)
    manifest = {
        "status": "complete", "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "build_id": build_id, "created_at": utc_now(), "datasets_selected": selected,
        "tokenizer": token_info, "preparation_code_hash": code_hash,
        "source_configuration_hash": source_hash, "contract": source_manifest["contract"],
        "datasets": {
            dataset_id: {
                "manifest": f"{dataset_id}/manifest.json",
                "manifest_hash": canonical_hash(row),
                "prepared_rows": row["prepared_rows"], "shards": len(row["shards"]),
                "rows_dropped": row["rows_dropped"],
            }
            for dataset_id, row in dataset_manifests.items()
        },
        "elapsed_sec": time.time() - t0,
    }
    write_json(manifest_path, manifest)
    remote_build = publish_build(build_root, output_root, build_id, args)
    if is_gcs_path(output_root) and not args.skip_upload:
        published_validation = validate_published_first_shards(
            remote_build, selected, args)
        print(
            "PUBLISHED FIRST-SHARD VALIDATION: "
            + ", ".join(
                f"{dataset_id}={row['rows']}"
                for dataset_id, row in published_validation.items()),
            flush=True,
        )
    if args.publish_latest:
        if is_gcs_path(output_root) and args.skip_upload:
            print("PUBLISH LATEST SKIPPED because --skip-upload targets GCS", flush=True)
        else:
            manifest_hash = canonical_hash(manifest)
            write_json_atomic(join_path(output_root, "manifest.json"), manifest)
            # LATEST is the final visibility switch after the convenience root
            # manifest and immutable build have both been verified.
            write_json_atomic(join_path(output_root, "LATEST.json"), {
                "build_id": build_id, "build_path": f"builds/{build_id}",
                "manifest_hash": manifest_hash, "published_at": utc_now(),
            })
    print("OPERATOR DATASET BUILD:")
    print(f"  root={output_root}")
    print(f"  build_id={build_id}")
    print(f"  build_path={remote_build}")
    print(f"  schema={SCHEMA} version={SCHEMA_VERSION}")
    print(f"  tokenizer={args.tokenizer} hash={token_info['hash']}")
    for dataset_id, row in dataset_manifests.items():
        print(
            f"  {dataset_id}: source={row['rows_source']} prepared={row['prepared_rows']} "
            f"dropped={row['rows_dropped']} shards={len(row['shards'])}")
    if not args.keep_work_dir and not args.skip_upload and is_gcs_path(output_root):
        print(f"  staging retained for resumability: {build_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
