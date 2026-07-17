"""Secondary BLiMP adapter; prompts are never generated locally."""

from __future__ import annotations

from typing import Any, Mapping

from .common import require_text, stable_row_id


def adapt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    good = require_text(row, "sentence_good")
    bad = require_text(row, "sentence_bad")
    phenomenon = str(row.get("phenomenon") or row.get("UID") or "unknown")
    return {
        "example_id": stable_row_id(row, "blimp"),
        "base_prompt": good,
        "source_prompt": bad,
        "positive_answer": good,
        "negative_answer": bad,
        "source_positive_answer": bad,
        "source_negative_answer": good,
        "causal_variable": phenomenon,
        "pair_type": "official_minimal_pair",
        "position_kind": "last_token",
        "metadata": {
            "phenomenon": phenomenon,
            "secondary_only": True,
            "full_sequence_minimal_pair": True,
        },
    }
