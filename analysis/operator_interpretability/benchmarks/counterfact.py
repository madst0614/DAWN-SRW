"""Secondary CounterFact adapter using the published rewrite request."""

from __future__ import annotations

from typing import Any, Mapping

from .common import leading_space, require_mapping, require_text, stable_row_id


def adapt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    rewrite = require_mapping(row, "requested_rewrite")
    subject = require_text(rewrite, "subject")
    template = require_text(rewrite, "prompt")
    prompt = template.format(subject)
    target_true = require_mapping(rewrite, "target_true")
    target_new = require_mapping(rewrite, "target_new")
    true_text = require_text(target_true, "str")
    new_text = require_text(target_new, "str")
    return {
        "example_id": stable_row_id(row, "counterfact"),
        "base_prompt": prompt,
        "source_prompt": prompt,
        "positive_answer": leading_space(true_text),
        "negative_answer": leading_space(new_text),
        "source_positive_answer": leading_space(true_text),
        "source_negative_answer": leading_space(new_text),
        "causal_variable": str(rewrite.get("relation_id") or "relation"),
        "pair_type": "true_vs_counterfactual_object",
        "position_kind": "last_token",
        "metadata": {
            "subject": subject,
            "relation_id": rewrite.get("relation_id"),
            "secondary_only": True,
        },
    }
