#!/usr/bin/env python3
"""Prepare public/operator datasets for DAWN-SRW v4166 analysis.

Default behavior mirrors the analysis datasets into:

  gs://dawn-tpu-data-c4/dataset/v4166_operator_analysis

The script uses only standard-library download/parsing plus GCS write support
through gcsfs or TensorFlow.  If ``runs/dataset_probe`` exists, it is used as a
local source cache before downloading again.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.dawn_operator_datasets import (  # noqa: E402
    DEFAULT_OPERATOR_DATASET_ROOT,
    OPERATOR_DATASET_SPECS,
    join_dataset_path,
    operator_dataset_manifest_path,
    operator_dataset_paths,
)


RAVEL_TGZ_URL = "https://raw.githubusercontent.com/explanare/ravel/main/data.tgz"
RAVEL_HF_TREE = "https://huggingface.co/api/datasets/hij/ravel/tree/main?recursive=1"
RAVEL_HF_RESOLVE = "https://huggingface.co/datasets/hij/ravel/resolve/main"
BLIMP_HF_TREE = "https://huggingface.co/api/datasets/nyu-mll/blimp/tree/main?recursive=1"
BLIMP_HF_RESOLVE = "https://huggingface.co/datasets/nyu-mll/blimp/resolve/main"
LAMA_ZIP_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"

DEFAULT_WORK_DIR = "runs/v4166_operator_datasets_staging"
DEFAULT_IOI_EXAMPLES = 4096
DEFAULT_SYNTHETIC_EXAMPLES = 4096


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
IOI_PLACES = [
    "store", "garden", "school", "office", "library", "park", "museum", "station",
]
IOI_OBJECTS = [
    "book", "drink", "bag", "letter", "key", "ball", "ticket", "phone",
]


SYNTHETIC_COLORS = [
    "red", "blue", "green", "yellow", "purple", "white", "black", "orange",
]
SYNTHETIC_KEYS = [
    "key_01", "key_02", "key_03", "key_04", "key_05", "key_06", "key_07", "key_08",
]
SYNTHETIC_ENTITIES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
]


def is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_gcs_fs():
    try:
        import gcsfs

        return gcsfs.GCSFileSystem()
    except ImportError:
        return None


def tf_gfile():
    import tensorflow as tf

    return tf.io.gfile


def open_uri(path: str, mode: str = "rb"):
    if is_gcs_path(path):
        fs = get_gcs_fs()
        if fs is not None:
            return fs.open(path, mode)
        return tf_gfile().GFile(path, mode)
    p = Path(path)
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode)


def makedirs_uri(path: str) -> None:
    if is_gcs_path(path):
        fs = get_gcs_fs()
        if fs is not None:
            fs.mkdirs(path, exist_ok=True)
            return
        tf_gfile().makedirs(path)
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json_uri(path: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    with open_uri(path, "wb") as f:
        f.write(data)
        f.write(b"\n")


def read_json_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def download_url(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    total = 0
    with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    os.replace(tmp, dest)
    return total


def copy_file(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest.stat().st_size


def upload_file(src: Path, dst_uri: str) -> int:
    size = src.stat().st_size
    with open(src, "rb") as inp, open_uri(dst_uri, "wb") as out:
        shutil.copyfileobj(inp, out, length=1024 * 1024)
    return size


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def is_parquet_file(path: Path) -> bool:
    if path.stat().st_size < 8:
        return False
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        tail = f.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def parquet_row_count(path: Path) -> Optional[int]:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def local_source_path(local_source: Optional[Path], *parts: str) -> Optional[Path]:
    if local_source is None:
        return None
    candidate = local_source.joinpath(*parts)
    return candidate if candidate.exists() else None


def hf_parquet_paths(tree_url: str) -> List[str]:
    rows = read_json_url(tree_url)
    paths = [
        str(row["path"])
        for row in rows
        if row.get("type") == "file" and str(row.get("path", "")).endswith(".parquet")
    ]
    if not paths:
        raise RuntimeError(f"No parquet files found in HF tree: {tree_url}")
    return sorted(paths)


def parquet_stats(root: Path) -> Dict[str, Any]:
    files = sorted(root.rglob("*.parquet"))
    bad = [str(p.relative_to(root)).replace("\\", "/") for p in files if not is_parquet_file(p)]
    rows_known = 0
    rows_unknown = 0
    for path in files:
        count = parquet_row_count(path)
        if count is None:
            rows_unknown += 1
        else:
            rows_known += count
    stats: Dict[str, Any] = {
        "parquet_files": len(files),
        "parquet_bytes": sum(p.stat().st_size for p in files),
        "bad_parquet_magic": bad,
    }
    if rows_known:
        stats["parquet_rows"] = rows_known
    if rows_unknown:
        stats["parquet_rows_unknown_files"] = rows_unknown
    return stats


def prepare_ravel(work_root: Path, local_source: Optional[Path]) -> Dict[str, Any]:
    out = work_root / "ravel"
    raw_tgz = out / "raw" / "data.tgz"
    source_tgz = local_source_path(local_source, "ravel", "data.tgz")
    if source_tgz is not None:
        copy_file(source_tgz, raw_tgz)
    else:
        download_url(RAVEL_TGZ_URL, raw_tgz)

    hf_root = out / "hf"
    source_ravel = local_source_path(local_source, "ravel")
    copied = 0
    if source_ravel is not None:
        for src in sorted(source_ravel.rglob("*.parquet")):
            rel = src.relative_to(source_ravel)
            copy_file(src, hf_root / rel)
            copied += 1
    if copied == 0:
        for rel in hf_parquet_paths(RAVEL_HF_TREE):
            download_url(f"{RAVEL_HF_RESOLVE}/{rel}", hf_root / rel)

    with tarfile.open(raw_tgz, "r:gz") as tar:
        names = tar.getnames()
    stats = {
        "raw_tgz_bytes": raw_tgz.stat().st_size,
        "raw_tar_entries": len(names),
        "raw_json_files": sum(1 for name in names if name.endswith(".json")),
    }
    stats.update(parquet_stats(hf_root))
    return stats


def prepare_blimp(work_root: Path, local_source: Optional[Path]) -> Dict[str, Any]:
    hf_root = work_root / "blimp" / "hf"
    source_blimp = local_source_path(local_source, "blimp")
    copied = 0
    if source_blimp is not None:
        for src in sorted(source_blimp.rglob("*.parquet")):
            rel = src.relative_to(source_blimp)
            copy_file(src, hf_root / rel)
            copied += 1
    if copied == 0:
        for rel in hf_parquet_paths(BLIMP_HF_TREE):
            download_url(f"{BLIMP_HF_RESOLVE}/{rel}", hf_root / rel)
    return parquet_stats(hf_root)


def prepare_lama(work_root: Path, local_source: Optional[Path]) -> Dict[str, Any]:
    out = work_root / "lama" / "raw" / "data.zip"
    source_zip = local_source_path(local_source, "lama", "data.zip")
    if source_zip is not None:
        copy_file(source_zip, out)
    else:
        download_url(LAMA_ZIP_URL, out)

    jsonl_files = 0
    jsonl_lines = 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            jsonl_files += 1
            jsonl_lines += len(zf.read(name).decode("utf-8").splitlines())
    return {
        "zip_bytes": out.stat().st_size,
        "zip_entries": len(names),
        "jsonl_files": jsonl_files,
        "jsonl_lines": jsonl_lines,
    }


def prepare_counterfact(work_root: Path, local_source: Optional[Path]) -> Dict[str, Any]:
    out = work_root / "counterfact" / "counterfact.json"
    source_json = (
        local_source_path(local_source, "counterfact", "counterfact.json")
        or local_source_path(local_source, "probe_5.txt")
    )
    if source_json is not None:
        copy_file(source_json, out)
    else:
        download_url(COUNTERFACT_URL, out)

    rows = json.loads(out.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("CounterFact JSON is not a list.")
    return {
        "json_bytes": out.stat().st_size,
        "rows": len(rows),
    }


def generate_ioi_rows(n_examples: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(4166)
    for idx in range(n_examples):
        name_a, name_b = rng.sample(IOI_NAMES, 2)
        place = rng.choice(IOI_PLACES)
        obj = rng.choice(IOI_OBJECTS)
        template_id = idx % len(IOI_TEMPLATES)
        template = IOI_TEMPLATES[template_id]
        clean = template.format(name_a=name_a, name_b=name_b, place=place, object=obj)
        corrupt = template.format(name_a=name_b, name_b=name_a, place=place, object=obj)
        yield {
            "id": f"ioi-{idx:06d}",
            "template_id": template_id,
            "clean_prompt": clean,
            "corrupt_prompt": corrupt,
            "correct": name_b,
            "distractor": name_a,
            "task_variable": "indirect_object",
            "target_position": "final_prediction_token",
        }


def prepare_ioi(work_root: Path, n_examples: int) -> Dict[str, Any]:
    out = work_root / "ioi"
    templates_path = out / "ioi_templates.json"
    examples_path = out / "ioi_examples.jsonl"
    templates = [
        {"template_id": idx, "template": template}
        for idx, template in enumerate(IOI_TEMPLATES)
    ]
    write_text(templates_path, json.dumps(templates, indent=2, sort_keys=True) + "\n")
    count = write_jsonl(examples_path, generate_ioi_rows(n_examples))
    return {
        "templates": len(templates),
        "generated_examples": count,
        "jsonl_bytes": examples_path.stat().st_size,
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
            prompt = f"{facts} The color of {target} is"
            answer = colors[target_idx]
            attr = "color"
        else:
            facts = " ".join(f"{ent} has {key}." for ent, key in zip(entities, keys))
            prompt = f"{facts} {target}'s key is"
            answer = keys[target_idx]
            attr = "key"
        yield {
            "id": f"synthetic-{idx:06d}",
            "prompt": prompt,
            "answer": answer,
            "entity": target,
            "attribute": attr,
            "task_variable": "entity_attribute_binding",
            "target_position": "final_prediction_token",
        }


def prepare_synthetic(work_root: Path, n_examples: int) -> Dict[str, Any]:
    out = work_root / "synthetic"
    spec_path = out / "synthetic_spec.json"
    examples_path = out / "binding_examples.jsonl"
    spec = {
        "entities": SYNTHETIC_ENTITIES,
        "colors": SYNTHETIC_COLORS,
        "keys": SYNTHETIC_KEYS,
        "target_position": "final_prediction_token",
    }
    write_text(spec_path, json.dumps(spec, indent=2, sort_keys=True) + "\n")
    count = write_jsonl(examples_path, generate_synthetic_rows(n_examples))
    return {
        "generated_examples": count,
        "jsonl_bytes": examples_path.stat().st_size,
    }


PREPARE_FUNCS = {
    "ravel": prepare_ravel,
    "blimp": prepare_blimp,
    "lama": prepare_lama,
    "counterfact": prepare_counterfact,
}


def parse_dataset_ids(value: str) -> List[str]:
    raw = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not raw or raw == ["all"]:
        return list(OPERATOR_DATASET_SPECS.keys())
    bad = [item for item in raw if item not in OPERATOR_DATASET_SPECS]
    if bad:
        raise ValueError(f"Unknown dataset id(s): {','.join(bad)}")
    dedup: List[str] = []
    for item in raw:
        if item not in dedup:
            dedup.append(item)
    return dedup


def upload_tree(work_root: Path, output_root: str, selected: Sequence[str]) -> Dict[str, Any]:
    uploaded = []
    for dataset_id in selected:
        src_root = work_root / dataset_id
        if not src_root.exists():
            continue
        for src in sorted(src_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(work_root).as_posix()
            dst = join_dataset_path(output_root, rel)
            size = upload_file(src, dst)
            uploaded.append({"path": dst, "bytes": size})
    return {
        "files": len(uploaded),
        "bytes": sum(row["bytes"] for row in uploaded),
        "uploaded": uploaded,
    }


def write_dataset_manifests(work_root: Path, output_root: str, stats: Dict[str, Any],
                            selected: Sequence[str]) -> Dict[str, Any]:
    manifest = {
        "created_at": utc_now(),
        "dataset_root": output_root,
        "manifest_path": operator_dataset_manifest_path(output_root),
        "selected": list(selected),
        "datasets": {},
    }
    path_summary = operator_dataset_paths(output_root)
    for dataset_id in selected:
        dataset_manifest = {
            "id": dataset_id,
            "spec": OPERATOR_DATASET_SPECS[dataset_id],
            "paths": path_summary[dataset_id],
            "stats": stats.get(dataset_id, {}),
        }
        manifest["datasets"][dataset_id] = dataset_manifest
        write_text(
            work_root / dataset_id / "manifest.json",
            json.dumps(dataset_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
    write_text(
        work_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare DAWN-SRW v4166 operator-analysis datasets.")
    p.add_argument("--output-root", default=DEFAULT_OPERATOR_DATASET_ROOT,
                   help="Destination root, usually gs://.../dataset/v4166_operator_analysis.")
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                   help="Local staging directory.")
    p.add_argument("--local-source", default=None,
                   help="Optional local source cache, e.g. runs/dataset_probe. Default downloads from source URLs.")
    p.add_argument("--datasets", default="all",
                   help="Comma-separated dataset ids or all.")
    p.add_argument("--ioi-examples", type=int, default=DEFAULT_IOI_EXAMPLES)
    p.add_argument("--synthetic-examples", type=int, default=DEFAULT_SYNTHETIC_EXAMPLES)
    p.add_argument("--skip-upload", action="store_true",
                   help="Prepare local staging files and manifests only.")
    p.add_argument("--clean-work-dir", action="store_true",
                   help="Remove the local staging directory after a successful upload.")
    p.add_argument("--keep-work-dir", action="store_true",
                   help="Keep staging files even after a successful GCS upload.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the selected layout without downloading or uploading.")
    args = p.parse_args()

    output_root = str(args.output_root).rstrip("/\\")
    selected = parse_dataset_ids(args.datasets)
    work_root = Path(args.work_dir)
    local_source = Path(args.local_source) if args.local_source else None

    print("DATASETS START")
    print(f"  output_root={output_root}")
    print(f"  manifest={operator_dataset_manifest_path(output_root)}")
    print(f"  work_dir={work_root}")
    print(f"  local_source={local_source or 'none'}")
    print(f"  selected={','.join(selected)}")
    if args.dry_run:
        for dataset_id, row in operator_dataset_paths(output_root).items():
            if dataset_id in selected:
                print(f"  {dataset_id}: root={row['root']}")
        return 0

    t0 = time.time()
    stats: Dict[str, Any] = {}
    for dataset_id in selected:
        print(f"DATASET {dataset_id} START", flush=True)
        if dataset_id in PREPARE_FUNCS:
            stats[dataset_id] = PREPARE_FUNCS[dataset_id](work_root, local_source)
        elif dataset_id == "ioi":
            stats[dataset_id] = prepare_ioi(work_root, int(args.ioi_examples))
        elif dataset_id == "synthetic":
            stats[dataset_id] = prepare_synthetic(work_root, int(args.synthetic_examples))
        else:
            raise AssertionError(dataset_id)
        write_text(
            work_root / dataset_id / "stats.json",
            json.dumps(stats[dataset_id], indent=2, sort_keys=True) + "\n",
        )
        print(
            "DATASET "
            f"{dataset_id} DONE "
            + " ".join(f"{k}={v}" for k, v in stats[dataset_id].items() if not isinstance(v, list)),
            flush=True,
        )

    manifest = write_dataset_manifests(work_root, output_root, stats, selected)
    upload = {"files": 0, "bytes": 0, "uploaded": []}
    if not args.skip_upload:
        makedirs_uri(output_root)
        upload = upload_tree(work_root, output_root, selected)
    manifest["upload"] = {k: v for k, v in upload.items() if k != "uploaded"}
    write_text(
        work_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    if not args.skip_upload:
        upload_file(work_root / "manifest.json", operator_dataset_manifest_path(output_root))

    clean_work_dir = (
        bool(args.clean_work_dir)
        or (is_gcs_path(output_root) and not args.skip_upload and not args.keep_work_dir)
    )
    if clean_work_dir and not args.skip_upload:
        shutil.rmtree(work_root)

    elapsed = time.time() - t0
    print(
        "DATASETS DONE "
        f"datasets={len(selected)} files={upload.get('files', 0)} "
        f"bytes={upload.get('bytes', 0)} elapsed_sec={elapsed:.1f}"
    )
    print(f"  manifest={operator_dataset_manifest_path(output_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
