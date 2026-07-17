"""Adapter for the official RAVEL cause/isolation counterfactual columns."""

from __future__ import annotations

from typing import Any, Mapping

from .common import leading_space, require_text, stable_row_id


ATTRIBUTES = ("Continent", "Country", "Language")


def _atomic(row: Mapping[str, Any]) -> dict[str, Any]:
    entity = require_text(row, "entity")
    attribute = require_text(row, "attribute")
    if attribute not in ATTRIBUTES:
        raise ValueError(f"unsupported RAVEL city attribute: {attribute!r}")
    answer = row.get("answer")
    if answer is None:
        answer = row.get(attribute)
    if answer is None and isinstance(row.get("metadata"), Mapping):
        answer = row["metadata"].get(attribute)
    if answer is None or not str(answer):
        raise ValueError("RAVEL row lacks its official attribute answer")
    return {
        "example_id": stable_row_id(row, "ravel"),
        "prompt": require_text(row, "prompt"),
        "entity": entity,
        "attribute": attribute,
        "answer": leading_space(str(answer)),
        "metadata": {},
    }


def adapt_pairs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use RAVEL's published columns; never synthesize or rematch rows."""
    base = _atomic(row)
    cause = _atomic(row.get("prompt_template_counterfactual") or {})
    isolate = _atomic(row.get("attribute_counterfactual") or {})
    if cause["attribute"] != base["attribute"]:
        raise ValueError("RAVEL prompt-template counterfactual changed attribute")
    if isolate["attribute"] == base["attribute"]:
        raise ValueError("RAVEL attribute counterfactual did not change attribute")
    return [
        _pair(base, cause, base["attribute"], "cause",
              "prompt_template_counterfactual"),
        _pair(base, isolate, base["attribute"], "isolation",
              "attribute_counterfactual"),
    ]


def _pair(base: Mapping[str, Any], source: Mapping[str, Any],
          variable: str, pair_type: str, official_column: str) -> dict[str, Any]:
    return {
        "example_id": f"{base['example_id']}::{source['example_id']}::{pair_type}",
        "phase_group_id": base["example_id"],
        "base_prompt": base["prompt"],
        "source_prompt": source["prompt"],
        "positive_answer": base["answer"],
        "negative_answer": source["answer"],
        "source_positive_answer": source["answer"],
        "source_negative_answer": base["answer"],
        "causal_variable": variable,
        "pair_type": pair_type,
        "position_kind": "entity_last_token",
        "trace_anchor_base": base["entity"],
        "trace_anchor_source": source["entity"],
        "metadata": {
            "pair_group_id": base["example_id"],
            "base_entity": base["entity"],
            "source_entity": source["entity"],
            "base_attribute": base["attribute"],
            "source_attribute": source["attribute"],
            "ravel_pair_semantics": pair_type,
            "official_counterfactual_column": official_column,
        },
    }
