#!/usr/bin/env python3
"""Probe public/generated operator-analysis sources and freeze adapter contracts.

This script deliberately prepares only a few human-readable rows.  It never
publishes GCS manifests, processes complete corpora, restores a checkpoint, or
runs model analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".generated" / "operator_probe_deps"
for candidate in (PROJECT_ROOT, LOCAL_DEPS):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

RAVEL_TGZ_URL = "https://raw.githubusercontent.com/explanare/ravel/main/data.tgz"
RAVEL_HF_TREE = "https://huggingface.co/api/datasets/hij/ravel/tree/main?recursive=1"
RAVEL_HF_REPO_ID = "hij/ravel"
BLIMP_ZIP_URL = "https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip"
BLIMP_HF_TREE = "https://huggingface.co/api/datasets/nyu-mll/blimp/tree/main?recursive=1"
BLIMP_HF_REPO_ID = "nyu-mll/blimp"
LAMA_ZIP_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"

RAVEL_SOURCE_CONTRACT = "ravel_official_github_archive"
BLIMP_SOURCE_CONTRACT = "blimp_official_github_jsonl"
SOURCE_NORMALIZATION_DESCRIPTION = (
    "The official GitHub records are normalized to the already probed "
    "dawn_operator_pair_v2 semantic contract. The transport/source container "
    "changed; the prepared schema and analysis semantics did not."
)
RAVEL_CITY_BASENAMES = (
    "ravel_city_entity_attributes.json",
    "ravel_city_attribute_to_prompts.json",
    "ravel_city_entity_to_split.json",
    "ravel_city_prompt_to_split.json",
)
RAVEL_ENTITY_ATTRIBUTES = (
    "Continent", "Country", "Language", "Latitude", "Longitude", "Timezone",
)
BLIMP_CORE_FIELDS = {
    "sentence_good", "sentence_bad", "UID", "pairID", "field",
    "linguistics_term", "lexically_identical", "simple_LM_method",
    "one_prefix_method", "two_prefix_method",
}
BLIMP_EXPECTED_FILES = 67
BLIMP_EXPECTED_ROWS_PER_FILE = 1000
BLIMP_EXPECTED_TOTAL_ROWS = 67000

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
DOWNLOAD_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/octet-stream,application/json,text/plain,*/*",
}
RETRYABLE_HTTP_STATUSES = {403, 408, 429, 500, 502, 503, 504}
MAX_DOWNLOAD_ATTEMPTS = 5
CURL_FALLBACK_SOURCE_TYPES = {
    "lama_zip", "counterfact_json", RAVEL_SOURCE_CONTRACT,
    BLIMP_SOURCE_CONTRACT,
}

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


class DownloadError(RuntimeError):
    """A source download failed after its retry/fallback policy was exhausted."""


class DownloadValidationError(RuntimeError):
    """A transfer completed but did not produce the expected source artifact."""


def _temp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".part")


def _checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(4096).lstrip().lower()
    html_tokens = (b"<html", b"<head", b"<body", b"<title")
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(html_tokens)
        or (head.startswith(b"<") and any(token in head for token in html_tokens))
    )


def validate_download(path: Path, source_type: str) -> None:
    """Reject empty/error responses and verify the source's container format."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise DownloadValidationError("downloaded file is empty")
    if _looks_like_html(path):
        raise DownloadValidationError("downloaded file is an HTML response, not a dataset")
    if source_type == "huggingface_parquet":
        if path.stat().st_size < 8:
            raise DownloadValidationError("parquet file is shorter than its magic bytes")
        with path.open("rb") as handle:
            leading_magic = handle.read(4)
            handle.seek(-4, os.SEEK_END)
            trailing_magic = handle.read(4)
        if leading_magic != b"PAR1" or trailing_magic != b"PAR1":
            raise DownloadValidationError("parquet magic bytes are missing")
    elif source_type in {"lama_zip", BLIMP_SOURCE_CONTRACT}:
        try:
            with zipfile.ZipFile(path) as archive:
                if not archive.namelist():
                    raise DownloadValidationError(
                        f"{source_type} zip archive has no entries")
                bad_entry = archive.testzip()
                if bad_entry is not None:
                    raise DownloadValidationError(
                        f"{source_type} zip CRC check failed for {bad_entry}")
        except zipfile.BadZipFile as exc:
            raise DownloadValidationError(
                f"{source_type} zip is unreadable: {exc}") from exc
    elif source_type == "counterfact_json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DownloadValidationError(f"CounterFact JSON is unreadable: {exc}") from exc
    elif source_type == RAVEL_SOURCE_CONTRACT:
        try:
            with tarfile.open(path, "r:*") as archive:
                if not archive.getmembers():
                    raise DownloadValidationError(
                        "RAVEL official archive has no entries")
        except tarfile.TarError as exc:
            raise DownloadValidationError(
                f"RAVEL official archive is unreadable: {exc}") from exc


def _stored_checksum(path: Path) -> Optional[str]:
    checksum_path = _checksum_path(path)
    try:
        checksum = checksum_path.read_text(encoding="ascii").strip().split()[0].lower()
    except (FileNotFoundError, IndexError, OSError, UnicodeError):
        return None
    if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        return None
    return checksum


def _write_checksum(path: Path, checksum: str) -> None:
    checksum_path = _checksum_path(path)
    checksum_temp = checksum_path.with_suffix(checksum_path.suffix + ".part")
    checksum_temp.write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    os.replace(checksum_temp, checksum_path)


def _reusable_download(path: Path, source_type: str) -> Optional[str]:
    expected = _stored_checksum(path)
    if expected is None or not path.is_file():
        return None
    try:
        validate_download(path, source_type)
        actual = sha256_file(path)
    except (OSError, DownloadValidationError):
        return None
    return actual if actual == expected else None


def _download_row(
    path: Path, url: str, checksum: str, *, reused: bool,
    source_type: str, final_url: Optional[str] = None,
    http_status: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "path": str(path), "url": url, "final_url": final_url or url,
        "http_status": http_status, "source_type": source_type,
        "bytes": path.stat().st_size, "sha256": checksum, "reused": reused,
    }


def _failure(
    *, dataset_id: str, source_type: str, original_url: str,
    final_url: Optional[str], http_status: Optional[int], detail: str,
) -> DownloadError:
    message = (
        f"download failed dataset={dataset_id} source_type={source_type} "
        f"original_url={original_url} final_url={final_url or original_url} "
        f"http_status={http_status if http_status is not None else 'unknown'}: {detail}"
    )
    print(f"[download-failed] {message}", file=sys.stderr, flush=True)
    return DownloadError(message)


def _exception_http_details(
    exc: BaseException, original_url: str,
    current_status: Optional[int] = None,
) -> Tuple[Optional[int], str]:
    status = getattr(exc, "code", current_status)
    final_url = original_url
    if isinstance(exc, urllib.error.HTTPError):
        final_url = exc.geturl() or original_url
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", status)
        final_url = getattr(response, "url", None) or final_url
    return int(status) if status is not None else None, str(final_url)


def _retry_delay(attempt: int) -> float:
    return float(min(2 ** (attempt - 1), 30))


def _should_retry(exc: BaseException, status: Optional[int]) -> bool:
    if status in RETRYABLE_HTTP_STATUSES:
        return True
    if isinstance(exc, DownloadValidationError):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (
        urllib.error.URLError,
        http.client.IncompleteRead,
        ConnectionError,
        TimeoutError,
    )):
        return True
    return status is None


def _log_retry(
    *, dataset_id: str, source_type: str, attempt: int,
    status: Optional[int], delay: float, exc: BaseException,
) -> None:
    print(
        f"[download-retry] dataset={dataset_id} source_type={source_type} "
        f"attempt={attempt}/{MAX_DOWNLOAD_ATTEMPTS} "
        f"http_status={status if status is not None else 'unknown'} "
        f"backoff_seconds={delay:g} error={type(exc).__name__}: {exc}",
        file=sys.stderr, flush=True,
    )


def _curl_download(
    url: str, temp: Path, *, dataset_id: str, source_type: str,
) -> Tuple[str, int]:
    curl = shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("curl is not installed")
    _unlink_if_exists(temp)
    marker_status = "__DAWN_CURL_STATUS__="
    marker_url = "__DAWN_CURL_URL__="
    command = [
        curl,
        "--fail",
        "--location",
        "--retry", "5",
        "--retry-delay", "2",
        "--retry-all-errors",
        "--connect-timeout", "30",
        "--max-time", "1800",
        "--user-agent", BROWSER_USER_AGENT,
        "--header", f"Accept: {DOWNLOAD_HEADERS['Accept']}",
        "--output", str(temp),
        "--write-out", f"\n{marker_status}%{{http_code}}\n{marker_url}%{{url_effective}}\n",
        url,
    ]
    print(
        f"[download-fallback] dataset={dataset_id} source_type={source_type} tool=curl",
        file=sys.stderr, flush=True,
    )
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=1830)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _unlink_if_exists(temp)
        raise RuntimeError(f"curl execution failed: {exc}") from exc
    status_match = re.search(rf"(?m)^{re.escape(marker_status)}(\d+)$", result.stdout)
    url_match = re.search(rf"(?m)^{re.escape(marker_url)}(.+)$", result.stdout)
    status = int(status_match.group(1)) if status_match else 0
    final_url = url_match.group(1).strip() if url_match else url
    if result.returncode != 0:
        _unlink_if_exists(temp)
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"curl exited with code {result.returncode}: {stderr or 'no stderr'} "
            f"(final_url={final_url}, http_status={status or 'unknown'})")
    return final_url, status


def download(
    url: str, path: Path, *, reuse: bool, dataset_id: str,
    source_type: str, timeout: float = 300.0,
) -> Dict[str, Any]:
    """Download and validate a regular HTTP source with retries and curl-on-403."""
    print(
        f"[download] dataset={dataset_id} url={url} destination={path}",
        flush=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path(path)
    _unlink_if_exists(temp)
    if reuse:
        checksum = _reusable_download(path, source_type)
        if checksum is not None:
            return _download_row(
                path, url, checksum, reused=True, source_type=source_type)

    last_exc: BaseException = RuntimeError("download did not start")
    last_status: Optional[int] = None
    last_final_url = url
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                last_final_url = response.geturl() or url
                last_status = int(response.getcode())
                with temp.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            validate_download(temp, source_type)
            checksum = sha256_file(temp)
            os.replace(temp, path)
            _write_checksum(path, checksum)
            return _download_row(
                path, url, checksum, reused=False, source_type=source_type,
                final_url=last_final_url, http_status=last_status)
        except Exception as exc:
            last_exc = exc
            last_status, last_final_url = _exception_http_details(
                exc, last_final_url, last_status)
            _unlink_if_exists(temp)
            retryable = _should_retry(exc, last_status)
            if not retryable or attempt >= MAX_DOWNLOAD_ATTEMPTS:
                break
            delay = _retry_delay(attempt)
            _log_retry(
                dataset_id=dataset_id, source_type=source_type, attempt=attempt,
                status=last_status, delay=delay, exc=exc)
            time.sleep(delay)

    if last_status == 403 and source_type in CURL_FALLBACK_SOURCE_TYPES:
        curl_final_url: Optional[str] = None
        curl_status: Optional[int] = None
        try:
            curl_final_url, curl_status = _curl_download(
                url, temp, dataset_id=dataset_id, source_type=source_type)
            validate_download(temp, source_type)
            checksum = sha256_file(temp)
            os.replace(temp, path)
            _write_checksum(path, checksum)
            return _download_row(
                path, url, checksum, reused=False, source_type=source_type,
                final_url=curl_final_url, http_status=curl_status)
        except Exception as curl_exc:
            _unlink_if_exists(temp)
            detail = f"urllib error: {last_exc}; curl fallback error: {curl_exc}"
            raise _failure(
                dataset_id=dataset_id, source_type=source_type,
                original_url=url, final_url=curl_final_url or last_final_url,
                http_status=curl_status or last_status, detail=detail) from curl_exc

    raise _failure(
        dataset_id=dataset_id, source_type=source_type, original_url=url,
        final_url=last_final_url, http_status=last_status,
        detail=f"{type(last_exc).__name__}: {last_exc}") from last_exc


def download_hf_dataset_file(
    repo_id: str, filename: str, path: Path, *, reuse: bool,
    dataset_id: str, timeout: float = 300.0,
) -> Dict[str, Any]:
    """Download a tree-discovered dataset file through huggingface_hub."""
    source_type = "huggingface_parquet"
    try:
        from huggingface_hub import constants as hf_constants
        from huggingface_hub import hf_hub_download, hf_hub_url
    except ImportError as exc:
        source_url = f"https://huggingface.co/datasets/{repo_id}"
        print(
            f"[download] dataset={dataset_id} url={source_url} destination={path}",
            flush=True,
        )
        raise _failure(
            dataset_id=dataset_id, source_type=source_type,
            original_url=source_url, final_url=source_url, http_status=None,
            detail="huggingface_hub is required for Hugging Face parquet downloads",
        ) from exc

    source_url = hf_hub_url(
        repo_id=repo_id, filename=filename, repo_type="dataset")
    print(
        f"[download] dataset={dataset_id} url={source_url} destination={path}",
        flush=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_path(path)
    _unlink_if_exists(temp)
    if reuse:
        checksum = _reusable_download(path, source_type)
        if checksum is not None:
            return _download_row(
                path, source_url, checksum, reused=True, source_type=source_type)

    last_exc: BaseException = RuntimeError("Hugging Face download did not start")
    last_status: Optional[int] = None
    last_final_url = source_url
    previous_timeout = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
    hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = timeout
    try:
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                cached_path = Path(hf_hub_download(
                    repo_type="dataset",
                    repo_id=repo_id,
                    filename=filename,
                    etag_timeout=min(float(timeout), 30.0),
                ))
                try:
                    os.link(cached_path, temp)
                except OSError:
                    shutil.copy2(cached_path, temp)
                validate_download(temp, source_type)
                checksum = sha256_file(temp)
                os.replace(temp, path)
                _write_checksum(path, checksum)
                return _download_row(
                    path, source_url, checksum, reused=False,
                    source_type=source_type, final_url=source_url,
                    http_status=200)
            except Exception as exc:
                last_exc = exc
                last_status, last_final_url = _exception_http_details(
                    exc, source_url, last_status)
                _unlink_if_exists(temp)
                retryable = _should_retry(exc, last_status)
                if not retryable or attempt >= MAX_DOWNLOAD_ATTEMPTS:
                    break
                delay = _retry_delay(attempt)
                _log_retry(
                    dataset_id=dataset_id, source_type=source_type,
                    attempt=attempt, status=last_status, delay=delay, exc=exc)
                time.sleep(delay)
    finally:
        hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = previous_timeout

    raise _failure(
        dataset_id=dataset_id, source_type=source_type,
        original_url=source_url, final_url=last_final_url,
        http_status=last_status,
        detail=f"{type(last_exc).__name__}: {last_exc}") from last_exc


def read_url_json(
    url: str, *, dataset_id: str = "unknown", source_type: str = "json_api",
    timeout: float = 120.0,
) -> Any:
    last_exc: BaseException = RuntimeError("request did not start")
    last_status: Optional[int] = None
    last_final_url = url
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                last_final_url = response.geturl() or url
                last_status = int(response.getcode())
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            last_status, last_final_url = _exception_http_details(
                exc, last_final_url, last_status)
            retryable = _should_retry(exc, last_status)
            if not retryable or attempt >= MAX_DOWNLOAD_ATTEMPTS:
                break
            delay = _retry_delay(attempt)
            _log_retry(
                dataset_id=dataset_id, source_type=source_type,
                attempt=attempt, status=last_status, delay=delay, exc=exc)
            time.sleep(delay)
    raise _failure(
        dataset_id=dataset_id, source_type=source_type,
        original_url=url, final_url=last_final_url, http_status=last_status,
        detail=f"{type(last_exc).__name__}: {last_exc}") from last_exc


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


def hf_parquet_tree(url: str, *, dataset_id: Optional[str] = None) -> List[str]:
    if dataset_id is None:
        if "/hij/ravel/" in url:
            dataset_id = "ravel"
        elif "/nyu-mll/blimp/" in url:
            dataset_id = "blimp"
        else:
            dataset_id = "unknown"
    rows = read_url_json(
        url, dataset_id=dataset_id, source_type="huggingface_tree_api")
    return sorted(
        str(row["path"]) for row in rows
        if row.get("type") == "file" and str(row.get("path", "")).endswith(".parquet")
    )


def ravel_archive_inventory(
    path: Path, *, announce: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """Return the full archive inventory and unique required city members."""
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    if announce:
        print("RAVEL archive contents:", flush=True)
        for name in names:
            print(f"  {name}", flush=True)
    selected: Dict[str, str] = {}
    for basename in RAVEL_CITY_BASENAMES:
        matches = [
            name for name in names
            if PurePosixPath(name).name == basename
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "RAVEL official archive requires exactly one member with "
                f"basename {basename!r}; observed={matches}")
        selected[basename] = matches[0]
    return names, selected


def read_ravel_city_records(
    path: Path, *, announce: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Normalize the four official city JSON files to the probed HF semantics."""
    names, selected = ravel_archive_inventory(path, announce=announce)
    values: Dict[str, Any] = {}
    with tarfile.open(path, "r:gz") as archive:
        for basename, member in selected.items():
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"RAVEL archive member is not a file: {member}")
            values[basename] = json.load(handle)

    entity_attributes = values["ravel_city_entity_attributes.json"]
    attribute_prompts = values["ravel_city_attribute_to_prompts.json"]
    entity_splits = values["ravel_city_entity_to_split.json"]
    prompt_splits = values["ravel_city_prompt_to_split.json"]
    for label, value in (
        ("entity attributes", entity_attributes),
        ("attribute prompts", attribute_prompts),
        ("entity splits", entity_splits),
        ("prompt splits", prompt_splits),
    ):
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"RAVEL {label} JSON must be an object, got {type(value).__name__}")

    entities: List[Dict[str, Any]] = []
    for index, (city_value, attributes_value) in enumerate(entity_attributes.items()):
        city = str(city_value)
        if not isinstance(attributes_value, Mapping):
            raise RuntimeError(f"RAVEL attributes for {city!r} are not an object")
        if city not in entity_splits:
            raise RuntimeError(f"RAVEL entity split is missing for {city!r}")
        split = str(entity_splits[city])
        if split not in {"train", "val", "test"}:
            raise RuntimeError(f"RAVEL entity {city!r} has invalid split {split!r}")
        row: Dict[str, Any] = {
            "ID": f"{index}-0",
            "City": city,
            **{attribute: attributes_value.get(attribute)
               for attribute in RAVEL_ENTITY_ATTRIBUTES},
            "split": split,
        }
        url = attributes_value.get("URL")
        if url not in (None, ""):
            row["URL"] = url
        entities.append(row)
    extra_entity_splits = set(map(str, entity_splits)) - {
        str(city) for city in entity_attributes
    }
    if extra_entity_splits:
        raise RuntimeError(
            "RAVEL entity split JSON contains unknown entities: "
            f"{sorted(extra_entity_splits)[:10]}")

    prompts: List[Dict[str, Any]] = []
    normalized_templates: List[str] = []
    for attribute_value, templates_value in attribute_prompts.items():
        attribute = str(attribute_value)
        if not isinstance(templates_value, list):
            raise RuntimeError(
                f"RAVEL prompt list for {attribute!r} is not an array")
        for template_value in templates_value:
            template = str(template_value)
            if template not in prompt_splits:
                raise RuntimeError(
                    f"RAVEL prompt split is missing for template {template!r}")
            split = str(prompt_splits[template])
            if split not in {"train", "val", "test"}:
                raise RuntimeError(
                    f"RAVEL prompt {template!r} has invalid split {split!r}")
            prompts.append({
                "Template": template,
                "Attribute": attribute,
                "Source": "RAVEL",
                "Entity": "",
                "split": split,
            })
            normalized_templates.append(template)
    extra_prompt_splits = set(map(str, prompt_splits)) - set(normalized_templates)
    if extra_prompt_splits:
        raise RuntimeError(
            "RAVEL prompt split JSON contains unknown templates: "
            f"{sorted(extra_prompt_splits)[:10]}")
    return entities, prompts, names


def normalize_blimp_row(
    source: Mapping[str, Any], *, member: str, line_number: int,
) -> Dict[str, Any]:
    missing = BLIMP_CORE_FIELDS - set(source)
    if missing:
        raise RuntimeError(
            f"BLiMP schema drift in {member}:{line_number}; missing={sorted(missing)} "
            f"observed={sorted(source)}")
    row = dict(source)
    pair_id = row.pop("pairID")
    if isinstance(pair_id, str) and re.fullmatch(r"[0-9]+", pair_id):
        pair_id = int(pair_id)
    row["pair_id"] = pair_id
    return row


def read_blimp_jsonl_rows(
    archive: zipfile.ZipFile, member: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with archive.open(member) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"BLiMP JSONL is unreadable at {member}:{line_number}: {exc}") from exc
            if not isinstance(source, Mapping):
                raise RuntimeError(
                    f"BLiMP row is not an object at {member}:{line_number}")
            rows.append(normalize_blimp_row(
                source, member=member, line_number=line_number))
    return rows


def blimp_archive_inventory(
    path: Path,
) -> Tuple[List[str], Dict[str, int], Dict[str, Dict[str, str]]]:
    """Validate the complete official BLiMP data/*.jsonl contract."""
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if (
                len(PurePosixPath(name).parts) >= 2
                and PurePosixPath(name).parts[-2] == "data"
                and PurePosixPath(name).suffix == ".jsonl"
            )
        )
        if len(members) != BLIMP_EXPECTED_FILES:
            raise RuntimeError(
                f"BLiMP official archive expected {BLIMP_EXPECTED_FILES} data/*.jsonl "
                f"files, got {len(members)}: {members}")
        row_counts: Dict[str, int] = {}
        schemas: Dict[str, Dict[str, str]] = {}
        for member in members:
            rows = read_blimp_jsonl_rows(archive, member)
            row_counts[member] = len(rows)
            schemas[member] = (
                {key: type(value).__name__ for key, value in rows[0].items()}
                if rows else {}
            )
            if len(rows) != BLIMP_EXPECTED_ROWS_PER_FILE:
                raise RuntimeError(
                    f"BLiMP file {member} expected {BLIMP_EXPECTED_ROWS_PER_FILE} "
                    f"rows, got {len(rows)}")
    total = sum(row_counts.values())
    if total != BLIMP_EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"BLiMP official archive expected {BLIMP_EXPECTED_TOTAL_ROWS} total "
            f"rows, got {total}")
    return members, row_counts, schemas


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(value).lower(): str(value) for value in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def probe_ravel(args, tokenizer, cache: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    source = {
        "contract": RAVEL_SOURCE_CONTRACT,
        "canonical_url": RAVEL_TGZ_URL,
        "normalization": SOURCE_NORMALIZATION_DESCRIPTION,
    }
    probe = base_probe("ravel", source)
    archive_path = cache / "ravel" / "official" / "data.tgz"
    probe["downloaded_files"].append(download(
        RAVEL_TGZ_URL, archive_path, reuse=args.reuse_downloads,
        dataset_id="ravel", source_type=RAVEL_SOURCE_CONTRACT))
    entity_rows_all, prompt_rows_all, names = read_ravel_city_records(
        archive_path, announce=True)
    probe["observed_files"] = names
    probe["observed_splits"] = sorted({
        str(row["split"]) for row in entity_rows_all + prompt_rows_all
    })
    observed_rows = {
        f"normalized/city_entity/{split}": [
            row for row in entity_rows_all if row["split"] == split
        ]
        for split in ("train", "val", "test")
    }
    observed_rows.update({
        f"normalized/city_prompt/{split}": [
            row for row in prompt_rows_all if row["split"] == split
        ]
        for split in ("train", "val", "test")
    })
    probe["observed_schema"] = {
        name: {key: type(value).__name__ for key, value in rows[0].items()}
        for name, rows in observed_rows.items() if rows
    }
    probe["observed_columns"] = sorted({
        key for rows in observed_rows.values() for row in rows[:1] for key in row
    })
    probe["row_count"] = {
        name: len(rows) for name, rows in observed_rows.items()
    }
    entity_rows = observed_rows["normalized/city_entity/train"]
    prompt_rows = observed_rows["normalized/city_prompt/train"]
    probe["raw_samples"] = [
        {"source_path": "normalized/city_entity/train", **row}
        for row in entity_rows[: args.sample_rows]
    ] + [
        {"source_path": "normalized/city_prompt/train", **row}
        for row in prompt_rows[: args.sample_rows]
    ]
    probe["nested_key_paths"] = sorted(set(
        path for row in probe["raw_samples"] for path in nested_key_paths(row)))
    prepared: List[Dict[str, Any]] = []
    required_entity = {"ID", "City"}
    required_prompt = {"Template", "Attribute", "Source", "Entity"}
    if not entity_rows or not prompt_rows:
        raise RuntimeError("RAVEL normalized train entity/prompt records are empty")
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
    attributes = sorted(
        attribute for attribute, rows in prompts_by_attribute.items()
        if rows and len({
            str(entity.get(attribute)) for entity in entity_rows
            if entity.get(attribute) not in (None, "")
        }) >= 2
    )
    if len(attributes) < 2:
        raise RuntimeError(
            f"RAVEL needs at least two joinable attributes, observed={attributes}")

    def different_entity(
        base: Mapping[str, Any], attribute: str,
        required_attributes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        match = next((
            entity for entity in entity_rows
            if str(entity.get("ID")) != str(base.get("ID"))
            and entity.get(attribute) not in (None, "")
            and str(entity.get(attribute)) != str(base.get(attribute))
            and all(entity.get(value) not in (None, "")
                    for value in required_attributes)
        ), None)
        if match is None:
            raise RuntimeError(
                f"RAVEL lacks a same-domain negative for {attribute}")
        return match

    def condition(entity: Mapping[str, Any], prompt: Mapping[str, Any]) -> Tuple[str, str]:
        template = str(prompt["Template"])
        return template % str(entity["City"]), str(entity[str(prompt["Attribute"])])

    attr1 = next((
        attribute for attribute in attributes
        if len(prompts_by_attribute[attribute]) >= 2
    ), None)
    if attr1 is None:
        raise RuntimeError("RAVEL needs an attribute with at least two train prompts")
    attr2 = next(attribute for attribute in attributes if attribute != attr1)
    e1 = next((
        entity for entity in entity_rows
        if entity.get(attr1) not in (None, "")
        and entity.get(attr2) not in (None, "")
    ), None)
    if e1 is None:
        raise RuntimeError(f"RAVEL has no entity with both {attr1} and {attr2}")
    e2 = different_entity(e1, attr1, (attr2,))
    e3 = different_entity(e1, attr2, (attr1,))
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
    probe["source_issues"].append(SOURCE_NORMALIZATION_DESCRIPTION)
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
    source = {
        "contract": BLIMP_SOURCE_CONTRACT,
        "canonical_url": BLIMP_ZIP_URL,
        "normalization": SOURCE_NORMALIZATION_DESCRIPTION,
    }
    probe = base_probe("blimp", source)
    archive_path = cache / "blimp" / "official" / "blimp-master.zip"
    probe["downloaded_files"].append(download(
        BLIMP_ZIP_URL, archive_path, reuse=args.reuse_downloads,
        dataset_id="blimp", source_type=BLIMP_SOURCE_CONTRACT))
    paths, row_counts, archive_schemas = blimp_archive_inventory(archive_path)
    selected = _choose_blimp_paths(paths)
    observed = []
    raw_samples = []
    prepared = []
    with zipfile.ZipFile(archive_path) as archive:
        for rel in selected:
            rows = read_blimp_jsonl_rows(archive, rel)
            phenomenon = PurePosixPath(rel).stem
            samples = rows[: args.sample_rows]
            info = {
                "source_path": rel,
                "phenomenon": phenomenon,
                "row_count": len(rows),
                "columns": sorted(rows[0]) if rows else [],
                "schema": archive_schemas[rel],
                "samples": samples,
            }
            observed.append(info)
            for row_index, row in enumerate(samples):
                raw = {"source_path": rel, "phenomenon": phenomenon, **row}
                raw_samples.append(raw)
                good = str(row["sentence_good"])
                bad = str(row["sentence_bad"])
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
                    uid = row.get("UID") or f"{phenomenon}-{row_index}"
                    pair_index = row.get("pair_id", row_index)
                    prepared.append(prepared_row(
                        tokenizer,
                        example_id=f"blimp-{phenomenon}-{pair_index}",
                        pair_id=f"blimp-{phenomenon}-{pair_index}",
                        dataset="blimp", split="train", phenomenon=phenomenon,
                        relation="grammatical_minimal_pair", group_id=phenomenon,
                        source_id=f"{uid}:{pair_index}",
                        score_mode="paired_sequence_logprob",
                        trace_semantics="pre_divergence_prediction_state",
                        text_a=good, text_b=bad, row_index=len(prepared),
                        max_seq_len=args.max_seq_len,
                        max_candidate_tokens=args.max_candidate_tokens,
                        extension={key: value for key, value in row.items()
                                   if key not in ("sentence_good", "sentence_bad")},
                    ))
    probe["observed_files"] = paths
    probe["observed_splits"] = ["train"]
    probe["observed_schema"] = archive_schemas
    probe["observed_columns"] = sorted({
        column for schema in archive_schemas.values() for column in schema
    })
    probe["row_count"] = row_counts
    # Preserve samples from every inspected phenomenon rather than allowing
    # the first JSONL file to hide cross-file schema or tokenization differences.
    probe["raw_samples"] = raw_samples
    probe["nested_key_paths"] = sorted(set(
        path for row in probe["raw_samples"] for path in nested_key_paths(row)))
    probe["tokenization_samples"] = [row["tokenizer_probe"] for row in raw_samples]
    probe["status"] = "ready_for_adapter" if prepared else "empty"
    probe["source_issues"].append(SOURCE_NORMALIZATION_DESCRIPTION)
    probe["proposed_pair_types"] = ["good_bad_minimal_pair"]
    probe["proposed_score_mode"] = "paired_sequence_logprob"
    probe["proposed_trace_semantics"] = "pre_divergence_prediction_state"
    probe["recommended_mapping"] = {
        "context_ids_a": "sentence_good",
        "context_ids_b": "sentence_bad",
        "phenomenon/group_id": "official data/*.jsonl file stem",
        "source_id": "UID + normalized pair_id",
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
    probe["downloaded_files"].append(download(
        LAMA_ZIP_URL, local, reuse=args.reuse_downloads,
        dataset_id="lama", source_type="lama_zip"))
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
    probe["downloaded_files"].append(download(
        COUNTERFACT_URL, local, reuse=args.reuse_downloads,
        dataset_id="counterfact", source_type="counterfact_json"))
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
        help="Reuse cached downloads only after format and recorded SHA-256 validation.")
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
