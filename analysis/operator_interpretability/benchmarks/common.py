"""Strict source-form helpers shared by official benchmark adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AdapterOutput:
    rows: tuple[Mapping[str, Any], ...]
    excluded: Mapping[str, int] = field(default_factory=dict)


def require_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"official row is missing mapping {key!r}")
    return value


def require_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or not str(value):
        raise ValueError(f"official row is missing text {key!r}")
    return str(value)


def official_choice_answer(row: Mapping[str, Any]) -> str:
    choices = row.get("choices")
    if isinstance(choices, Mapping):
        labels = list(choices.get("label") or ())
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        labels = list(choices)
    else:
        raise ValueError("official row is missing choices")
    if not labels:
        raise ValueError("official row has no choice labels")
    answer_key = row.get("answerKey")
    if isinstance(answer_key, bool):
        raise ValueError("official answerKey cannot be bool")
    if isinstance(answer_key, int):
        if not 0 <= answer_key < len(labels):
            raise ValueError(f"answerKey index {answer_key} is out of range")
        return str(labels[answer_key])
    answer_key_text = str(answer_key)
    label_text = [str(label) for label in labels]
    if answer_key_text not in label_text:
        raise ValueError(
            f"answerKey {answer_key_text!r} is absent from official choices")
    return str(labels[label_text.index(answer_key_text)])


def stable_row_id(row: Mapping[str, Any], prefix: str) -> str:
    payload = json.dumps(
        dict(row), sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    content_hash = hashlib.sha256(payload).hexdigest()[:24]
    for key in ("id", "example_id", "idx", "case_id", "arc_id"):
        if row.get(key) is not None:
            return f"{prefix}:{key}={row[key]}:{content_hash}"
    return f"{prefix}:{content_hash}"


def leading_space(text: Any) -> str:
    value = str(text)
    return value if value.startswith(" ") else " " + value


def one_row(value: Mapping[str, Any]) -> AdapterOutput:
    return AdapterOutput(rows=(dict(value),))
