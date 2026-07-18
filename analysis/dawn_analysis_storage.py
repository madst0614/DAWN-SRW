"""Storage, manifest, and logging helpers for DAWN-SRW analysis.

The analysis pipeline writes both human-readable stdout progress and durable
artifacts.  This module keeps all filesystem/GCS handling in one place so the
stage code can stay focused on metrics.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import time
import traceback as traceback_lib
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - report-only mode may not import JAX.
    jax = None
    jnp = None


STAGE_NAMES = (
    "eval",
    "prune",
    "geometry",
    "usage",
    "trace",
    "ablation",
    "patching",
    "steering",
    "report",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_gcs_path(path: str | os.PathLike[str]) -> bool:
    return str(path).startswith("gs://")


def join_path(base: str | os.PathLike[str], *parts: str | os.PathLike[str]) -> str:
    base_s = str(base)
    if is_gcs_path(base_s):
        out = base_s.rstrip("/")
        for part in parts:
            part_s = str(part).strip("/\\")
            if part_s:
                out += "/" + part_s
        return out
    return str(Path(base_s, *map(str, parts)))


def basename(path: str | os.PathLike[str]) -> str:
    return str(path).rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def dirname(path: str | os.PathLike[str]) -> str:
    path_s = str(path).rstrip("/\\").replace("\\", "/")
    if is_gcs_path(path_s):
        return path_s.rsplit("/", 1)[0] if "/" in path_s[5:] else path_s
    return str(Path(path_s).parent)


_GCS_FS = None


def _get_gcs_fs():
    global _GCS_FS
    if _GCS_FS is not None:
        return _GCS_FS
    try:
        import gcsfs

        _GCS_FS = gcsfs.GCSFileSystem()
        return _GCS_FS
    except ImportError:
        return None


def _tf_gfile():
    import tensorflow as tf

    return tf.io.gfile


def open_path(path: str | os.PathLike[str], mode: str = "r"):
    path_s = str(path)
    if is_gcs_path(path_s):
        fs = _get_gcs_fs()
        if fs is not None:
            return fs.open(path_s, mode)
        return _tf_gfile().GFile(path_s, mode)
    p = Path(path_s)
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, mode)


def exists(path: str | os.PathLike[str]) -> bool:
    path_s = str(path)
    if is_gcs_path(path_s):
        fs = _get_gcs_fs()
        if fs is not None:
            return fs.exists(path_s)
        return bool(_tf_gfile().exists(path_s))
    return Path(path_s).exists()


def makedirs(path: str | os.PathLike[str]) -> None:
    path_s = str(path)
    if is_gcs_path(path_s):
        try:
            fs = _get_gcs_fs()
            if fs is not None:
                fs.mkdirs(path_s, exist_ok=True)
                return
        except Exception:
            pass
        try:
            _tf_gfile().makedirs(path_s)
        except Exception:
            pass
        return
    Path(path_s).mkdir(parents=True, exist_ok=True)


def list_paths(path: str | os.PathLike[str], pattern: str = "*") -> List[str]:
    path_s = str(path).rstrip("/")
    if is_gcs_path(path_s):
        fs = _get_gcs_fs()
        glob_pat = path_s + "/" + pattern
        if fs is not None:
            out = fs.glob(glob_pat)
            return sorted("gs://" + p if not p.startswith("gs://") else p for p in out)
        return sorted(_tf_gfile().glob(glob_pat))
    return sorted(str(p) for p in Path(path_s).glob(pattern))


def remove_path(path: str | os.PathLike[str]) -> None:
    path_s = str(path)
    if not exists(path_s):
        return
    if is_gcs_path(path_s):
        fs = _get_gcs_fs()
        if fs is not None:
            fs.rm(path_s)
            return
        _tf_gfile().remove(path_s)
        return
    Path(path_s).unlink()


def _gcs_copy(src: str, dst: str) -> None:
    fs = _get_gcs_fs()
    if fs is not None:
        fs.copy(src, dst)
        return
    _tf_gfile().copy(src, dst, overwrite=True)


def _gcs_bucket_blob(path: str) -> tuple[str, str]:
    rest = str(path)[5:]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def _set_gcs_object_metadata(path: str | os.PathLike[str],
                             content_type: Optional[str] = None,
                             content_disposition: Optional[str] = None) -> bool:
    path_s = str(path)
    if not is_gcs_path(path_s) or (not content_type and not content_disposition):
        return True
    attrs = {}
    if content_type:
        attrs["content_type"] = content_type
    if content_disposition:
        attrs["content_disposition"] = content_disposition
    try:
        fs = _get_gcs_fs()
        if fs is not None and hasattr(fs, "setxattrs"):
            fs.setxattrs(path_s, **attrs)
            return True
    except Exception:
        pass
    try:
        from google.cloud import storage

        bucket_name, blob_name = _gcs_bucket_blob(path_s)
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        if content_type:
            blob.content_type = content_type
        if content_disposition:
            blob.content_disposition = content_disposition
        blob.patch()
        return True
    except Exception:
        return False


def _text_content_type_for_path(path: str | os.PathLike[str]) -> str:
    suffix = Path(str(path).rstrip("/\\")).suffix.lower()
    return {
        ".csv": "text/csv; charset=utf-8",
        ".htm": "text/html; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".jsonl": "application/x-ndjson; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }.get(suffix, "text/plain; charset=utf-8")


def _json_default(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return _json_default(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if jnp is not None and isinstance(obj, getattr(jnp, "ndarray", ())):
        return np.asarray(obj).tolist()
    if jax is not None:
        try:
            jax_array_type = getattr(jax, "Array", None)
            if jax_array_type is not None and isinstance(obj, jax_array_type):
                return np.asarray(jax.device_get(obj)).tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return _json_default(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return _json_default(obj)


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    if not exists(path):
        return deepcopy(default)
    with open_path(path, "r") as f:
        return json.load(f)


def write_bytes_atomic(path: str | os.PathLike[str], payload: bytes,
                       content_type: Optional[str] = None,
                       content_disposition: Optional[str] = None,
                       require_metadata: bool = False) -> str:
    path_s = str(path)
    tmp = f"{path_s}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    parent = dirname(path_s)
    makedirs(parent)
    if is_gcs_path(path_s):
        try:
            with open_path(tmp, "wb") as f:
                f.write(payload)
            _gcs_copy(tmp, path_s)
            metadata_set = _set_gcs_object_metadata(
                path_s, content_type, content_disposition)
            if require_metadata and not metadata_set:
                raise RuntimeError(
                    "failed to establish required GCS object metadata: "
                    f"path={path_s} content_type={content_type!r} "
                    f"content_disposition={content_disposition!r}")
        finally:
            try:
                remove_path(tmp)
            except Exception:
                pass
        return path_s

    tmp_path = Path(tmp)
    with open(tmp_path, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path_s)
    return path_s


def write_text_atomic(path: str | os.PathLike[str], text: str,
                      content_type: Optional[str] = None,
                      content_disposition: str = "inline",
                      require_metadata: bool = False) -> str:
    return write_bytes_atomic(
        path,
        text.encode("utf-8"),
        content_type=content_type or _text_content_type_for_path(path),
        content_disposition=content_disposition,
        require_metadata=require_metadata,
    )


def read_text(path: str | os.PathLike[str], default: str = "") -> str:
    if not exists(path):
        return default
    with open_path(path, "r") as f:
        return f.read()


def append_text(path: str | os.PathLike[str], text: str) -> str:
    """Append text to local paths, or read+rewrite for GCS paths."""
    path_s = str(path)
    if is_gcs_path(path_s):
        return write_text_atomic(path_s, read_text(path_s) + text)
    p = Path(path_s)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    return path_s


def write_json_atomic(path: str | os.PathLike[str], obj: Any) -> str:
    payload = json.dumps(
        json_safe(obj),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )
    return write_text_atomic(path, payload + "\n")


def write_jsonl_atomic(path: str | os.PathLike[str], rows: Iterable[Dict[str, Any]]) -> str:
    lines = [
        json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False, default=_json_default)
        for row in rows
    ]
    text = "\n".join(lines)
    if text:
        text += "\n"
    return write_text_atomic(path, text)


def append_jsonl(path: str | os.PathLike[str], row: Dict[str, Any]) -> str:
    """Append one JSONL row.

    Local paths use append mode.  GCS has no general append primitive, so this
    falls back to read+atomic rewrite.  Stage code uses per-batch or small
    per-part files, so this remains acceptable for analysis logs.
    """
    path_s = str(path)
    if is_gcs_path(path_s):
        rows = read_jsonl(path_s)
        rows.append(row)
        return write_jsonl_atomic(path_s, rows)
    p = Path(path_s)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False, default=_json_default)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path_s


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    if not exists(path):
        return []
    rows = []
    with open_path(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_npz_atomic(path: str | os.PathLike[str], **arrays: Any) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return write_bytes_atomic(path, buf.getvalue())


def read_npz(path: str | os.PathLike[str]) -> Dict[str, np.ndarray]:
    with open_path(path, "rb") as f:
        data = f.read()
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def write_csv_atomic(path: str | os.PathLike[str], rows: Sequence[Dict[str, Any]],
                     fieldnames: Optional[Sequence[str]] = None) -> str:
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json_safe(v) for k, v in row.items()})
    return write_text_atomic(path, buf.getvalue())


def should_skip_job(path: str | os.PathLike[str], required_keys: Optional[Sequence[str]] = None) -> bool:
    """Return true when an artifact exists and validates enough for resume."""
    path_s = str(path)
    if not exists(path_s):
        return False
    required = list(required_keys or [])
    try:
        if path_s.endswith(".json"):
            obj = read_json(path_s)
            if required:
                return isinstance(obj, dict) and all(k in obj for k in required)
            return obj is not None
        if path_s.endswith(".jsonl"):
            rows = read_jsonl(path_s)
            if not rows:
                return False
            if required:
                return all(isinstance(r, dict) and all(k in r for k in required) for r in rows)
            return True
        if path_s.endswith(".npz"):
            obj = read_npz(path_s)
            return all(k in obj for k in required) if required else bool(obj)
        if path_s.endswith(".csv") or path_s.endswith(".md") or path_s.endswith(".txt"):
            with open_path(path_s, "r") as f:
                return bool(f.read(1))
        return True
    except Exception:
        return False


def _empty_stage_state() -> Dict[str, Any]:
    return {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "running_jobs": [],
        "artifacts": {},
        "summaries": {},
        "cursor": {},
        "errors": {},
    }


class AnalysisStore:
    """Persistent output store for one analysis run."""

    def __init__(self, output_dir: str, *, is_primary: bool = True,
                 analysis_version: str = "unknown") -> None:
        self.output_dir = str(output_dir).rstrip("/")
        self.is_primary = bool(is_primary)
        self.analysis_version = analysis_version
        self.manifest_path = join_path(self.output_dir, "manifest.json")
        self.run_state_path = join_path(self.output_dir, "run_state.json")
        self.log_path = join_path(self.output_dir, "logs", "analysis_log.jsonl")
        self.error_dir = join_path(self.output_dir, "logs", "errors")

    def path(self, *parts: str | os.PathLike[str]) -> str:
        return join_path(self.output_dir, *parts)

    def ensure_layout(self) -> None:
        for rel in (
            "",
            "logs",
            "logs/errors",
            "eval",
            "prune",
            "geometry",
            "usage",
            "usage/top_contexts",
            "usage/usage_parts",
            "trace",
            "ablation",
            "ablation/jobs",
            "patching",
            "steering",
            "report",
        ):
            makedirs(self.path(rel))

    def load_manifest(self) -> Dict[str, Any]:
        manifest = read_json(self.manifest_path, default=None)
        if not isinstance(manifest, dict):
            manifest = {
                "analysis_version": self.analysis_version,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "stages": {name: _empty_stage_state() for name in STAGE_NAMES},
            }
        manifest.setdefault("analysis_version", self.analysis_version)
        manifest.setdefault("created_at", utc_now())
        manifest.setdefault("stages", {})
        for name in STAGE_NAMES:
            manifest["stages"].setdefault(name, _empty_stage_state())
            for key, value in _empty_stage_state().items():
                manifest["stages"][name].setdefault(key, deepcopy(value))
        return manifest

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        if not self.is_primary:
            return
        manifest["updated_at"] = utc_now()
        write_json_atomic(self.manifest_path, manifest)
        write_json_atomic(self.run_state_path, {
            "updated_at": manifest["updated_at"],
            "stage_status": {
                stage: state.get("status")
                for stage, state in manifest.get("stages", {}).items()
            },
        })

    def update_manifest(self, **fields: Any) -> Dict[str, Any]:
        manifest = self.load_manifest()
        manifest.update(json_safe(fields))
        self.save_manifest(manifest)
        return manifest

    def set_stage_status(self, stage: str, status: str) -> None:
        manifest = self.load_manifest()
        st = manifest["stages"].setdefault(stage, _empty_stage_state())
        st["status"] = status
        if status == "running":
            st["started_at"] = st.get("started_at") or utc_now()
        elif status == "complete":
            st["completed_at"] = utc_now()
        elif status == "failed":
            st["failed_at"] = utc_now()
        self.save_manifest(manifest)

    def mark_job_started(self, stage: str, job_id: str) -> None:
        if not self.is_primary:
            return
        manifest = self.load_manifest()
        st = manifest["stages"].setdefault(stage, _empty_stage_state())
        st["status"] = "running"
        st["started_at"] = st.get("started_at") or utc_now()
        if job_id not in st["running_jobs"]:
            st["running_jobs"].append(job_id)
        self.save_manifest(manifest)

    def mark_job_complete(self, stage: str, job_id: str,
                          artifact_path: Optional[str] = None,
                          summary: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_primary:
            return
        manifest = self.load_manifest()
        st = manifest["stages"].setdefault(stage, _empty_stage_state())
        if job_id in st["running_jobs"]:
            st["running_jobs"].remove(job_id)
        if job_id not in st["completed_jobs"]:
            st["completed_jobs"].append(job_id)
        if job_id in st["failed_jobs"]:
            st["failed_jobs"].remove(job_id)
        if artifact_path is not None:
            st["artifacts"][job_id] = artifact_path
        if summary is not None:
            st["summaries"][job_id] = json_safe(summary)
        self.save_manifest(manifest)

    def mark_job_failed(self, stage: str, job_id: str, error: str,
                        traceback_text: Optional[str] = None) -> str:
        if traceback_text is None:
            traceback_text = traceback_lib.format_exc()
        err_path = join_path(self.error_dir, f"{stage}-{job_id}.txt")
        if self.is_primary:
            write_text_atomic(err_path, traceback_text)
            manifest = self.load_manifest()
            st = manifest["stages"].setdefault(stage, _empty_stage_state())
            if job_id in st["running_jobs"]:
                st["running_jobs"].remove(job_id)
            if job_id not in st["failed_jobs"]:
                st["failed_jobs"].append(job_id)
            st["errors"][job_id] = {"error": str(error), "traceback": err_path}
            st["status"] = "failed"
            st["failed_at"] = utc_now()
            self.save_manifest(manifest)
        return err_path

    def log_event(self, stage: str, event: str, *,
                  message: Optional[str] = None, print_stdout: bool = True,
                  **fields: Any) -> None:
        row = {
            "time": utc_now(),
            "stage": stage,
            "event": event,
            **json_safe(fields),
        }
        if message:
            row["message"] = message
        if self.is_primary:
            append_jsonl(self.log_path, row)
        if print_stdout:
            if message is None:
                details = " ".join(f"{k}={v}" for k, v in fields.items())
                message = f"{stage} {event} {details}".strip()
            print(message, flush=True)


_DEFAULT_STORE: Optional[AnalysisStore] = None


def set_default_store(store: AnalysisStore) -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


def _store_or_default(store: Optional[AnalysisStore]) -> AnalysisStore:
    if store is not None:
        return store
    if _DEFAULT_STORE is None:
        raise RuntimeError("No AnalysisStore has been configured.")
    return _DEFAULT_STORE


def mark_job_started(stage: str, job_id: str, store: Optional[AnalysisStore] = None) -> None:
    _store_or_default(store).mark_job_started(stage, job_id)


def mark_job_complete(stage: str, job_id: str, artifact_path: Optional[str] = None,
                      summary: Optional[Dict[str, Any]] = None,
                      store: Optional[AnalysisStore] = None) -> None:
    _store_or_default(store).mark_job_complete(stage, job_id, artifact_path, summary)


def mark_job_failed(stage: str, job_id: str, error: str,
                    traceback: Optional[str] = None,
                    store: Optional[AnalysisStore] = None) -> str:
    return _store_or_default(store).mark_job_failed(stage, job_id, error, traceback)


def temp_local_path(suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(prefix="dawn-analysis-", suffix=suffix)
    os.close(fd)
    return path

