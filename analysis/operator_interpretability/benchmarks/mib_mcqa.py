"""Official MIB circuit-pair adapter for four-choice CopyColors MCQA."""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    official_choice_answer,
    one_row,
    require_mapping,
    require_text,
    stable_row_id,
)


def adapt_rows(row: Mapping[str, Any]):
    source = require_mapping(row, "symbol_counterfactual")
    base_answer = official_choice_answer(row)
    source_answer = official_choice_answer(source)
    return one_row({
        "example_id": stable_row_id(row, "mib_mcqa"),
        "base_prompt": require_text(row, "prompt"),
        "source_prompt": require_text(source, "prompt"),
        "positive_answer": base_answer,
        "negative_answer": source_answer,
        "source_positive_answer": source_answer,
        "source_negative_answer": base_answer,
        "source_behavior_required": True,
        "intervention_positive_answer": source_answer,
        "intervention_negative_answer": base_answer,
        "causal_variable": "answer_symbol",
        "pair_type": "symbol_counterfactual",
        "position_kind": "last_token",
        "metadata": {
            "official_counterfactual_column": "symbol_counterfactual",
            "official_task": "mcqa",
            "token_contract": "mib_equal_candidate_lengths",
            "dataset_specific_id": row.get("dataset_specific_id"),
            "answer_key": row.get("answerKey"),
        },
    })
