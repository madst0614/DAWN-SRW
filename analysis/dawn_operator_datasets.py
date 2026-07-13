"""Operator-analysis dataset registry for DAWN-SRW v4166.

The datasets here are the public/task-generated inputs used by the v4166
operator-family analysis items.  The registry is intentionally path-centric:
the preparation script writes this exact layout, and train-analysis summaries
report the same paths so TPU-side experiment code has one stable source of
truth.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from analysis.dawn_analysis_storage import exists, open_path


DEFAULT_OPERATOR_DATASET_ROOT = os.environ.get(
    "DAWN_OPERATOR_DATASET_ROOT",
    "gs://dawn-tpu-data-c4/dataset/v4166_operator_analysis",
)


def join_dataset_path(base: str, *parts: str) -> str:
    base = str(base).rstrip("/\\")
    if base.startswith("gs://"):
        out = base
        for part in parts:
            part_s = str(part).strip("/\\")
            if part_s:
                out += "/" + part_s
        return out
    return "/".join([base.replace("\\", "/"), *[str(p).strip("/\\") for p in parts if str(p).strip("/\\")]])


OPERATOR_DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "ravel": {
        "title": "RAVEL attribute operator disentanglement",
        "task_variable": "attribute",
        "behavior_metric": "answer logit margin over attribute-value distractors",
        "operator_question": "same-attribute operator overlap and cross-attribute causal drop",
        "path": "ravel",
        "artifacts": {
            "raw_tgz": "ravel/raw/data.tgz",
            "city_entity_train": "ravel/hf/city_entity/train-00000-of-00001.parquet",
            "city_entity_val": "ravel/hf/city_entity/val-00000-of-00001.parquet",
            "city_entity_test": "ravel/hf/city_entity/test-00000-of-00001.parquet",
            "city_prompt_train": "ravel/hf/city_prompt/train-00000-of-00001.parquet",
            "city_prompt_val": "ravel/hf/city_prompt/val-00000-of-00001.parquet",
            "city_prompt_test": "ravel/hf/city_prompt/test-00000-of-00001.parquet",
        },
        "expected": {
            "hf_parquet_files": 6,
            "hf_rows_observed": 5348,
            "raw_json_files_observed": 25,
        },
    },
    "ioi": {
        "title": "IOI operator-level circuit",
        "task_variable": "indirect object name",
        "behavior_metric": "logit(correct IO name) - logit(distractor subject name)",
        "operator_question": "clean/corrupt gate delta by QK/V/RST pool",
        "path": "ioi",
        "artifacts": {
            "generated_examples": "ioi/ioi_examples.jsonl",
            "templates": "ioi/ioi_templates.json",
        },
        "expected": {
            "generated_examples": 4096,
            "templates": 15,
        },
    },
    "blimp": {
        "title": "BLiMP phenomenon-specific grammatical operators",
        "task_variable": "minimal-pair linguistic phenomenon",
        "behavior_metric": "log P(good sentence) - log P(bad sentence)",
        "operator_question": "same-phenomenon operator overlap and phenomenon-specific ablation",
        "path": "blimp",
        "artifacts": {
            "hf_root": "blimp/hf",
        },
        "expected": {
            "hf_parquet_files": 67,
            "hf_rows_observed": 67000,
        },
    },
    "lama": {
        "title": "LAMA factual relation recall",
        "task_variable": "factual relation",
        "behavior_metric": "correct-object logit margin on known-correct subset",
        "operator_question": "relation-specific RST/QK/V causal drop",
        "path": "lama",
        "artifacts": {
            "raw_zip": "lama/raw/data.zip",
        },
        "expected": {
            "zip_entries_observed": 52,
            "jsonl_files_observed": 47,
            "jsonl_lines_observed": 69692,
        },
    },
    "counterfact": {
        "title": "CounterFact factual rewrite probes",
        "task_variable": "subject-relation-object association",
        "behavior_metric": "true/new object logit margin",
        "operator_question": "relation and rewrite-target operator specificity",
        "path": "counterfact",
        "artifacts": {
            "json": "counterfact/counterfact.json",
        },
        "expected": {
            "rows_observed": 21919,
        },
    },
    "synthetic": {
        "title": "Synthetic binding-retrieval sanity set",
        "task_variable": "controlled entity attribute binding",
        "behavior_metric": "target attribute logit margin",
        "operator_question": "entity binding, attribute query, and residual write decomposition",
        "path": "synthetic",
        "artifacts": {
            "binding_examples": "synthetic/binding_examples.jsonl",
            "spec": "synthetic/synthetic_spec.json",
        },
        "expected": {
            "generated_examples": 4096,
        },
    },
}


OPERATOR_ANALYSIS_ITEM_IDS = (
    "operator_dataset_manifest",
    "ravel_operator_disentanglement",
    "ioi_operator_circuit",
    "blimp_operator_grammar",
    "lama_counterfact_factual_recall",
    "synthetic_binding_sanity",
)


def operator_dataset_root(root: Optional[str] = None) -> str:
    return str(root or DEFAULT_OPERATOR_DATASET_ROOT).rstrip("/\\")


def operator_dataset_manifest_path(root: Optional[str] = None) -> str:
    return join_dataset_path(operator_dataset_root(root), "manifest.json")


def operator_dataset_paths(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    base = operator_dataset_root(root)
    out: Dict[str, Dict[str, Any]] = {}
    for dataset_id, spec in OPERATOR_DATASET_SPECS.items():
        artifacts = {
            key: join_dataset_path(base, value)
            for key, value in spec.get("artifacts", {}).items()
        }
        out[dataset_id] = {
            "id": dataset_id,
            "title": spec["title"],
            "root": join_dataset_path(base, spec["path"]),
            "artifacts": artifacts,
            "expected": dict(spec.get("expected", {})),
            "task_variable": spec["task_variable"],
            "behavior_metric": spec["behavior_metric"],
            "operator_question": spec["operator_question"],
        }
    return out


def operator_dataset_summary(root: Optional[str] = None) -> Dict[str, Any]:
    base = operator_dataset_root(root)
    return {
        "root": base,
        "manifest": operator_dataset_manifest_path(base),
        "prepare_command": "python3 -u scripts/prepare_v4166_operator_datasets.py",
        "datasets": operator_dataset_paths(base),
    }


def _operator_dataset_exists(path: str) -> bool:
    try:
        return bool(exists(path))
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Operator dataset preflight cannot access GCS because neither "
            "gcsfs nor TensorFlow gfile is available in this environment: "
            f"{path}"
        ) from exc


def load_operator_dataset_manifest(root: Optional[str] = None) -> Dict[str, Any]:
    """Read and validate the prepared manifest through the shared local/GCS I/O layer."""
    base = operator_dataset_root(root)
    path = operator_dataset_manifest_path(base)
    if not _operator_dataset_exists(path):
        raise FileNotFoundError(f"Operator dataset manifest not found: {path}")
    with open_path(path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Operator dataset manifest must be a mapping: {path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Operator dataset manifest missing datasets mapping: {path}")
    unknown = sorted(set(datasets) - set(OPERATOR_DATASET_SPECS))
    if unknown:
        raise ValueError(
            f"Operator dataset manifest has unknown dataset ids: {','.join(unknown)}")
    return payload


def operator_dataset_preflight(
    root: Optional[str] = None,
    *,
    required_datasets: Optional[list[str]] = None,
    verify_artifacts: bool = True,
) -> Dict[str, Any]:
    """Fail-loud preflight for prepared datasets actually consumed by an item."""
    base = operator_dataset_root(root)
    manifest = load_operator_dataset_manifest(base)
    selected = list(required_datasets or manifest.get("datasets", {}).keys())
    bad = [dataset_id for dataset_id in selected if dataset_id not in OPERATOR_DATASET_SPECS]
    if bad:
        raise ValueError(f"Unknown required operator datasets: {','.join(bad)}")
    configured = operator_dataset_paths(base)
    missing = []
    checked = []
    if verify_artifacts:
        for dataset_id in selected:
            for artifact, path in configured[dataset_id]["artifacts"].items():
                checked.append(path)
                if not _operator_dataset_exists(path):
                    missing.append({
                        "dataset": dataset_id,
                        "artifact": artifact,
                        "path": path,
                    })
    if missing:
        raise FileNotFoundError(
            "Prepared operator dataset artifacts are missing: "
            + "; ".join(row["path"] for row in missing))
    return {
        "status": "ready",
        "root": base,
        "manifest": operator_dataset_manifest_path(base),
        "datasets": selected,
        "checked_artifacts": len(checked),
    }
