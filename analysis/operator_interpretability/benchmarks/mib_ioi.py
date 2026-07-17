"""Adapter for the official ``mib-bench/ioi`` rows."""

from __future__ import annotations

from typing import Any, Mapping

from .common import leading_space, require_mapping, require_text, stable_row_id


def adapt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = require_mapping(row, "metadata")
    counterfactual = require_mapping(row, "s2_io_flip_counterfactual")
    subject = require_text(metadata, "subject")
    indirect_object = require_text(metadata, "indirect_object")
    source_metadata = counterfactual.get("metadata")
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}
    return {
        "example_id": stable_row_id(row, "mib_ioi"),
        "base_prompt": require_text(row, "prompt"),
        "source_prompt": require_text(counterfactual, "prompt"),
        "positive_answer": leading_space(indirect_object),
        "negative_answer": leading_space(subject),
        "source_positive_answer": leading_space(str(
            source_metadata.get("indirect_object", subject))),
        "source_negative_answer": leading_space(str(
            source_metadata.get("subject", indirect_object))),
        "causal_variable": "output_token",
        "pair_type": "s2_io_flip_counterfactual",
        "position_kind": "last_token",
        "metadata": {
            "subject": subject,
            "indirect_object": indirect_object,
            "template": row.get("template"),
            "official_counterfactual_column": "s2_io_flip_counterfactual",
        },
    }

