"""Official MIB circuit-pair adapter for IOI."""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    leading_space,
    official_choice_answer,
    one_row,
    require_mapping,
    require_text,
    stable_row_id,
)


def adapt_rows(row: Mapping[str, Any]):
    metadata = require_mapping(row, "metadata")
    source = require_mapping(row, "s2_io_flip_counterfactual")
    base_answer = official_choice_answer(row)
    source_answer = official_choice_answer(source)
    indirect_object = require_text(metadata, "indirect_object")
    subject = require_text(metadata, "subject")
    if base_answer != indirect_object or source_answer != subject:
        raise ValueError(
            "official IOI choices disagree with metadata under the s2 flip")
    return one_row({
        "example_id": stable_row_id(row, "mib_ioi"),
        "base_prompt": require_text(row, "prompt"),
        "source_prompt": require_text(source, "prompt"),
        "positive_answer": leading_space(base_answer),
        "negative_answer": leading_space(source_answer),
        "source_positive_answer": leading_space(source_answer),
        "source_negative_answer": leading_space(base_answer),
        "source_behavior_required": True,
        "intervention_positive_answer": leading_space(source_answer),
        "intervention_negative_answer": leading_space(base_answer),
        "causal_variable": "ioi_task_output",
        "pair_type": "s2_io_flip_counterfactual",
        "position_kind": "last_token",
        "metadata": {
            "official_counterfactual_column": "s2_io_flip_counterfactual",
            "official_task": "ioi",
            "token_contract": "mib_equal_candidate_lengths",
            "subject": subject,
            "indirect_object": indirect_object,
            "template": row.get("template"),
        },
    })
