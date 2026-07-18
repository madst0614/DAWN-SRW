"""Official MIB circuit-pair adapter for arithmetic addition."""

from __future__ import annotations

from typing import Any, Mapping

from .common import one_row, require_mapping, require_text, stable_row_id


def adapt_rows(row: Mapping[str, Any]):
    if str(row.get("operator")) != "+":
        raise ValueError("official arithmetic_addition row has a non-addition operator")
    source = require_mapping(row, "random_counterfactual")
    if row.get("label") is None or source.get("label") is None:
        raise ValueError("official arithmetic pair is missing a label")
    base_answer = str(row["label"])
    source_answer = str(source["label"])
    return one_row({
        "example_id": stable_row_id(row, "mib_arithmetic"),
        "base_prompt": require_text(row, "prompt"),
        "source_prompt": require_text(source, "prompt"),
        "positive_answer": base_answer,
        "negative_answer": source_answer,
        "source_positive_answer": source_answer,
        "source_negative_answer": base_answer,
        "source_behavior_required": True,
        "intervention_positive_answer": source_answer,
        "intervention_negative_answer": base_answer,
        "causal_variable": "sum_output",
        "pair_type": "random_counterfactual",
        "position_kind": "last_token",
        "metadata": {
            "official_counterfactual_column": "random_counterfactual",
            "official_task": "arithmetic_addition",
            "token_contract": "mib_single_token_candidates",
            "operator": row.get("operator"),
            "num_digit": row.get("num_digit"),
            "operand1": row.get("operand1"),
            "operand2": row.get("operand2"),
            "template": row.get("template"),
        },
    })
