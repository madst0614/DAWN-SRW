"""Source-faithful MIB RAVEL causal-variable pairs.

The official RAVEL baseline evaluates ``attribute_counterfactual`` and
``wikipedia_counterfactual``.  Cause versus isolation is determined by whether
the base prompt queries the intervened variable; it is not a property of the
counterfactual column.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .common import (
    AdapterOutput,
    leading_space,
    require_mapping,
    require_text,
    stable_row_id,
)


ATTRIBUTES = ("Continent", "Country", "Language")
SOURCE_COLUMNS = ("attribute_counterfactual", "wikipedia_counterfactual")


def _attribute_value(row: Mapping[str, Any], attribute: str) -> str | None:
    value = row.get(attribute)
    if value is None or not str(value):
        return None
    return str(value)


def adapt_rows(row: Mapping[str, Any]) -> AdapterOutput:
    base_id = stable_row_id(row, "ravel")
    base_prompt = require_text(row, "prompt")
    base_entity = require_text(row, "entity")
    base_query = require_text(row, "attribute")
    if base_query not in ATTRIBUTES:
        raise ValueError(f"unsupported official RAVEL base attribute: {base_query!r}")
    base_output = _attribute_value(row, base_query)
    if base_output is None:
        return AdapterOutput(rows=(), excluded={"missing_base_query_value": 1})

    rows: list[Mapping[str, Any]] = []
    excluded = Counter()
    for source_column in SOURCE_COLUMNS:
        source = require_mapping(row, source_column)
        source_prompt = require_text(source, "prompt")
        source_entity = require_text(source, "entity")
        source_query = require_text(source, "attribute")
        if (source_column == "attribute_counterfactual"
                and source_query not in ATTRIBUTES):
            raise ValueError(
                "RAVEL attribute_counterfactual has an unsupported query")
        if (source_column == "wikipedia_counterfactual"
                and source_query != "wikipedia"):
            raise ValueError(
                "RAVEL wikipedia_counterfactual changed its query contract")

        source_value_for_base_query = _attribute_value(source, base_query)
        if source_value_for_base_query is None:
            excluded["missing_source_base_query_value"] += len(ATTRIBUTES)
            continue
        if source_value_for_base_query == base_output:
            excluded["indistinguishable_base_entity_contrast"] += len(ATTRIBUTES)
            continue

        source_behavior_required = source_query in ATTRIBUTES
        source_answer = ""
        source_contrast = ""
        if source_behavior_required:
            source_answer_value = _attribute_value(source, source_query)
            source_contrast_value = _attribute_value(row, source_query)
            if source_answer_value is None or source_contrast_value is None:
                excluded["missing_source_behavior_value"] += len(ATTRIBUTES)
                continue
            if source_answer_value == source_contrast_value:
                excluded["indistinguishable_source_behavior_contrast"] += len(
                    ATTRIBUTES)
                continue
            source_answer = leading_space(source_answer_value)
            source_contrast = leading_space(source_contrast_value)

        for variable in ATTRIBUTES:
            base_variable_value = _attribute_value(row, variable)
            source_variable_value = _attribute_value(source, variable)
            if base_variable_value is None or source_variable_value is None:
                excluded["missing_intervention_variable_value"] += 1
                continue
            pair_type = "cause" if base_query == variable else "isolation"
            if pair_type == "cause" and base_variable_value == source_variable_value:
                excluded["indistinguishable_cause"] += 1
                continue
            intervention_answer = (
                source_value_for_base_query if pair_type == "cause"
                else base_output)
            intervention_contrast = (
                base_output if pair_type == "cause"
                else source_value_for_base_query)
            pair_group_id = f"{base_id}::{variable}"
            rows.append({
                "example_id": f"{pair_group_id}::{source_column}",
                "phase_group_id": pair_group_id,
                "base_prompt": base_prompt,
                "source_prompt": source_prompt,
                "positive_answer": leading_space(base_output),
                "negative_answer": leading_space(source_value_for_base_query),
                "source_positive_answer": source_answer,
                "source_negative_answer": source_contrast,
                "source_behavior_required": source_behavior_required,
                "intervention_positive_answer": leading_space(
                    intervention_answer),
                "intervention_negative_answer": leading_space(
                    intervention_contrast),
                "causal_variable": variable,
                "pair_type": pair_type,
                "position_kind": "entity_last_token",
                "trace_anchor_base": base_entity,
                "trace_anchor_source": source_entity,
                "metadata": {
                    "pair_group_id": pair_group_id,
                    "official_counterfactual_column": source_column,
                    "official_task": "ravel",
                    "token_contract": "variable_length_candidate_contrasts",
                    "candidate_score_normalization": (
                        "mean_log_probability_per_token"),
                    "base_entity": base_entity,
                    "source_entity": source_entity,
                    "base_query_attribute": base_query,
                    "source_query_attribute": source_query,
                    "intervened_variable": variable,
                    "expected_intervention_rule": (
                        "source_value_when_queried_else_base_output"),
                    "source_behavior_required": source_behavior_required,
                },
            })
    return AdapterOutput(rows=tuple(rows), excluded=dict(excluded))
