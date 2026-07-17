"""Adapter for official MIB two-digit addition rows."""

from __future__ import annotations

from typing import Any, Mapping

from .common import require_mapping, require_text, stable_row_id


def adapt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if int(row.get("num_digit", -1)) != 2:
        raise ValueError("MIB arithmetic adapter accepts only num_digit=2")
    if str(row.get("operator")) != "+":
        raise ValueError("MIB arithmetic adapter accepts only addition")
    counterfactual = require_mapping(row, "random_counterfactual")
    return {
        "example_id": stable_row_id(row, "mib_arithmetic"),
        "base_prompt": require_text(row, "prompt"),
        "source_prompt": require_text(counterfactual, "prompt"),
        "positive_answer": str(row["label"]),
        "negative_answer": str(counterfactual["label"]),
        "source_positive_answer": str(counterfactual["label"]),
        "source_negative_answer": str(row["label"]),
        "causal_variable": "two_digit_sum",
        "pair_type": "random_counterfactual",
        "position_kind": "last_token",
        "metadata": {
            "operator": "+",
            "num_digit": 2,
            "official_counterfactual_column": "random_counterfactual",
        },
    }

