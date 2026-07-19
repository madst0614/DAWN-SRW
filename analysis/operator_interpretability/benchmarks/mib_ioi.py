"""Official MIB circuit-pair adapter for IOI."""

from __future__ import annotations

from string import Formatter
from typing import Any, Mapping

from .common import (
    leading_space,
    official_choice_answer,
    one_row,
    require_mapping,
    require_text,
    stable_row_id,
)


def _render_semantic_prompt(
        template: str, metadata: Mapping[str, Any], *,
        s2_name: str) -> tuple[str, dict[str, list[int]]]:
    """Render official IOI placeholders while retaining semantic spans.

    Positions are derived from the benchmark template contract, never by
    searching the rendered prompt for a person name.  This remains unambiguous
    when a name occurs more than once or tokenizes into multiple subtokens.
    """
    values = {
        "name_A": require_text(metadata, "subject"),
        "name_B": require_text(metadata, "indirect_object"),
        "name_C": str(s2_name),
        "place": require_text(metadata, "place"),
        "object": require_text(metadata, "object"),
    }
    pieces: list[str] = []
    spans: dict[str, list[int]] = {}
    cursor = 0
    for literal, field_name, format_spec, conversion in Formatter().parse(
            str(template)):
        pieces.append(literal)
        cursor += len(literal)
        if field_name is None:
            continue
        if format_spec or conversion:
            raise ValueError(
                "official IOI template uses unsupported formatting syntax")
        if field_name not in values:
            raise ValueError(
                f"official IOI template has unknown placeholder {field_name!r}")
        value = values[field_name]
        start = cursor
        pieces.append(value)
        cursor += len(value)
        if field_name in {"name_A", "name_B", "name_C"}:
            if field_name in spans:
                raise ValueError(
                    f"official IOI template repeats semantic field {field_name}")
            spans[field_name] = [start, cursor]
    required = {"name_A", "name_B", "name_C"}
    if set(spans) != required:
        raise ValueError(
            "official IOI template lacks required name_A/name_B/name_C spans")
    return "".join(pieces), spans


def _semantic_spans_from_rendered_prompt(
        template: str, prompt: str,
        known_fields: Mapping[str, str]) -> dict[str, list[int]]:
    """Recover placeholder spans from template structure in older builds."""
    parsed = list(Formatter().parse(str(template)))
    spans: dict[str, list[int]] = {}
    cursor = 0
    for index, (literal, field_name, format_spec, conversion) in enumerate(
            parsed):
        if not prompt.startswith(literal, cursor):
            raise ValueError(
                "official IOI prompt literal disagrees with its template")
        cursor += len(literal)
        if field_name is None:
            continue
        if format_spec or conversion:
            raise ValueError(
                "official IOI template uses unsupported formatting syntax")
        known = known_fields.get(field_name)
        if known is not None:
            value = str(known)
            if not prompt.startswith(value, cursor):
                raise ValueError(
                    f"official IOI prompt disagrees at {field_name!r}")
            end = cursor + len(value)
        else:
            next_literal = next((
                item[0] for item in parsed[index + 1:] if item[0]), "")
            if not next_literal:
                end = len(prompt)
            else:
                end = prompt.find(next_literal, cursor)
                if end < cursor:
                    raise ValueError(
                        "official IOI template delimiter is absent")
        if field_name in {"name_A", "name_B", "name_C"}:
            if field_name in spans:
                raise ValueError(
                    f"official IOI template repeats semantic field {field_name}")
            spans[field_name] = [cursor, end]
        cursor = end
    if cursor != len(prompt):
        raise ValueError("official IOI prompt has unparsed template suffix")
    if set(spans) != {"name_A", "name_B", "name_C"}:
        raise ValueError(
            "official IOI template lacks required name_A/name_B/name_C spans")
    return spans


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
    template = require_text(row, "template")
    base_prompt = require_text(row, "prompt")
    source_prompt = require_text(source, "prompt")
    rendered_base, base_spans = _render_semantic_prompt(
        template, metadata, s2_name=subject)
    rendered_source, source_spans = _render_semantic_prompt(
        template, metadata, s2_name=indirect_object)
    if rendered_base != base_prompt or rendered_source != source_prompt:
        raise ValueError(
            "official IOI prompt disagrees with template/metadata rendering")
    return one_row({
        "example_id": stable_row_id(row, "mib_ioi"),
        "base_prompt": base_prompt,
        "source_prompt": source_prompt,
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
            "place": require_text(metadata, "place"),
            "object": require_text(metadata, "object"),
            "template": template,
            "semantic_char_spans": {
                "base": {
                    "first_name_a": base_spans["name_A"],
                    "first_name_b": base_spans["name_B"],
                    "s2_counterfactual": base_spans["name_C"],
                },
                "source": {
                    "first_name_a": source_spans["name_A"],
                    "first_name_b": source_spans["name_B"],
                    "s2_counterfactual": source_spans["name_C"],
                },
            },
            "semantic_span_source": (
                "official_ioi_template_placeholder_render"),
        },
    })
