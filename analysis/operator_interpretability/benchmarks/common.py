"""Small strict helpers shared by benchmark adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


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


def choice_answer(row: Mapping[str, Any]) -> str:
    choices = require_mapping(row, "choices")
    labels = list(choices.get("label") or ())
    answer_key = row.get("answerKey")
    if isinstance(answer_key, int):
        if not 0 <= answer_key < len(labels):
            raise ValueError(f"answerKey index {answer_key} is out of range")
        return str(labels[answer_key])
    answer_key = str(answer_key)
    if answer_key not in [str(label) for label in labels]:
        raise ValueError(f"answerKey {answer_key!r} is absent from choices")
    return str(labels[[str(label) for label in labels].index(answer_key)])


def stable_row_id(row: Mapping[str, Any], prefix: str) -> str:
    for key in ("id", "example_id", "idx", "case_id"):
        if row.get(key) is not None:
            return f"{prefix}:{row[key]}"
    payload = json.dumps(
        dict(row), sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:24]}"


def leading_space(text: str) -> str:
    value = str(text)
    return value if value.startswith(" ") else " " + value
