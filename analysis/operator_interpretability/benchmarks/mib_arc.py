"""Adapter for official MIB ARC Easy symbol counterfactuals."""

from __future__ import annotations

from typing import Any, Mapping

from .common import choice_answer, require_mapping, require_text, stable_row_id


def adapt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    choices = require_mapping(row, "choices")
    if len(list(choices.get("label") or ())) != 4:
        raise ValueError("MIB ARC adapter requires exactly four choices")
    counterfactual = require_mapping(row, "symbol_counterfactual")
    return {
        "example_id": stable_row_id(row, "mib_arc"),
        "base_prompt": require_text(row, "prompt"),
        "source_prompt": require_text(counterfactual, "prompt"),
        "positive_answer": choice_answer(row),
        "negative_answer": choice_answer(counterfactual),
        "source_positive_answer": choice_answer(counterfactual),
        "source_negative_answer": choice_answer(row),
        "causal_variable": "answer_pointer",
        "pair_type": "symbol_counterfactual",
        "position_kind": "last_token",
        "metadata": {
            "answer_key": row.get("answerKey"),
            "official_counterfactual_column": "symbol_counterfactual",
        },
    }

