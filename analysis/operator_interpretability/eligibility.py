"""Tokenizer eligibility and trace-position audit for the source contract."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from analysis.operator_interpretability.benchmark_schema import BenchmarkExample


class BenchmarkEligibilityError(ValueError):
    """A valid official row is unusable under the selected tokenizer."""


def tokenizer_vocab_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    ordered = sorted((str(token), int(index)) for token, index in vocab.items())
    payload = "\n".join(f"{index}\t{token}" for token, index in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_bucket(example_id: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def shared_split_phase(example_id: str, *, include_test: bool) -> str:
    """Create disjoint phases only for a source with no official split."""
    bucket = stable_bucket(example_id)
    if include_test:
        if bucket < 7000:
            return "discovery"
        if bucket < 8500:
            return "validation"
        return "test"
    return "discovery" if bucket < 8000 else "validation"


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tokenizer(str(text), add_special_tokens=False).input_ids
    return tuple(int(token) for token in ids)


def _anchor_last_token(tokenizer: Any, prompt: str, anchor: str) -> int:
    # Match the official RAVEL token-position function: first entity occurrence.
    start = prompt.find(anchor)
    if start < 0:
        raise BenchmarkEligibilityError(
            f"trace anchor {anchor!r} is absent from official prompt")
    end = start + len(anchor)
    try:
        encoded = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        candidates = [
            index for index, (left, right) in enumerate(offsets)
            if int(left) < end and int(right) > start
        ]
        if not candidates:
            raise BenchmarkEligibilityError(
                "tokenizer offsets do not cover trace anchor")
        return int(candidates[-1])
    except (KeyError, TypeError, NotImplementedError):
        prefix_ids = _encode(tokenizer, prompt[:end])
        if not prefix_ids:
            raise BenchmarkEligibilityError("trace anchor tokenization is empty")
        return len(prefix_ids) - 1


def resolve_trace_position(tokenizer: Any, prompt: str, *, kind: str,
                           anchor: str | None = None) -> int:
    ids = _encode(tokenizer, prompt)
    if not ids:
        raise BenchmarkEligibilityError(
            "official benchmark prompt tokenizes to zero tokens")
    if kind == "last_token":
        return len(ids) - 1
    if kind == "entity_last_token":
        if not anchor:
            raise ValueError(
                "entity_last_token source contract requires an entity anchor")
        return _anchor_last_token(tokenizer, prompt, anchor)
    raise ValueError(f"unsupported benchmark position kind: {kind!r}")


def _equal_length(name: str, left: tuple[int, ...],
                  right: tuple[int, ...]) -> None:
    if len(left) != len(right):
        raise BenchmarkEligibilityError(
            f"{name} candidate token lengths differ: {len(left)} != {len(right)}")


def _different(name: str, left: tuple[int, ...],
               right: tuple[int, ...]) -> None:
    if left == right:
        raise BenchmarkEligibilityError(f"{name} candidates tokenize identically")


def tokenize_adapted_pair(
        tokenizer: Any, benchmark_id: str, phase: str,
        adapted: Mapping[str, Any], *, max_seq_len: int) -> BenchmarkExample:
    required_text_fields = (
        "example_id", "base_prompt", "source_prompt",
        "positive_answer", "negative_answer",
        "intervention_positive_answer", "intervention_negative_answer",
        "causal_variable", "pair_type",
    )
    missing_fields = [
        name for name in required_text_fields
        if name not in adapted or not str(adapted[name])
    ]
    if missing_fields:
        raise ValueError(
            "adapted benchmark row is missing required fields: "
            + ",".join(missing_fields))
    metadata = dict(adapted.get("metadata") or {})
    if metadata.get("full_sequence_minimal_pair"):
        good_ids = _encode(tokenizer, str(adapted["positive_answer"]))
        bad_ids = _encode(tokenizer, str(adapted["negative_answer"]))
        prefix_length = 0
        for good, bad in zip(good_ids, bad_ids):
            if good != bad:
                break
            prefix_length += 1
        if prefix_length <= 0:
            raise BenchmarkEligibilityError(
                "official minimal pair has no shared token prefix")
        if prefix_length >= min(len(good_ids), len(bad_ids)):
            raise BenchmarkEligibilityError(
                "official minimal pair lacks divergent continuations")
        base_ids = tuple(good_ids[:prefix_length])
        positive_ids = tuple(good_ids[prefix_length:])
        negative_ids = tuple(bad_ids[prefix_length:])
        if len(base_ids) + max(len(positive_ids), len(negative_ids)) > max_seq_len:
            raise BenchmarkEligibilityError(
                "official minimal pair exceeds max_seq_len")
        base_prompt = tokenizer.decode(
            base_ids, clean_up_tokenization_spaces=False)
        metadata.update({
            "shared_prefix_token_count": prefix_length,
            "token_position_audited": True,
        })
        positive_answer = str(adapted["positive_answer"])
        negative_answer = str(adapted["negative_answer"])

        def continuation(answer: Any, field_name: str) -> tuple[int, ...]:
            value = str(answer)
            if value == positive_answer:
                return positive_ids
            if value == negative_answer:
                return negative_ids
            raise ValueError(
                f"{adapted['example_id']}: {field_name} is not one of the "
                "official minimal-pair sentences")

        source_behavior_required = bool(adapted.get(
            "source_behavior_required", False))
        source_positive_ids: tuple[int, ...] = ()
        source_negative_ids: tuple[int, ...] = ()
        if source_behavior_required:
            source_positive_ids = continuation(
                adapted.get("source_positive_answer"),
                "source_positive_answer")
            source_negative_ids = continuation(
                adapted.get("source_negative_answer"),
                "source_negative_answer")
        intervention_positive_ids = continuation(
            adapted["intervention_positive_answer"],
            "intervention_positive_answer")
        intervention_negative_ids = continuation(
            adapted["intervention_negative_answer"],
            "intervention_negative_answer")
        return BenchmarkExample(
            benchmark_id=benchmark_id,
            example_id=str(adapted["example_id"]),
            phase=phase,
            base_prompt=base_prompt,
            source_prompt=base_prompt,
            positive_answer=positive_answer,
            negative_answer=negative_answer,
            intervention_positive_answer=str(
                adapted["intervention_positive_answer"]),
            intervention_negative_answer=str(
                adapted["intervention_negative_answer"]),
            causal_variable=str(adapted["causal_variable"]),
            pair_type=str(adapted["pair_type"]),
            source_behavior_required=source_behavior_required,
            trace_position_base=prefix_length - 1,
            trace_position_source=prefix_length - 1,
            input_ids_base=base_ids,
            input_ids_source=base_ids,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            source_positive_ids=source_positive_ids,
            source_negative_ids=source_negative_ids,
            intervention_positive_ids=intervention_positive_ids,
            intervention_negative_ids=intervention_negative_ids,
            metadata=metadata,
        ).validate()

    base_prompt = str(adapted["base_prompt"])
    source_prompt = str(adapted["source_prompt"])
    base_ids = _encode(tokenizer, base_prompt)
    source_ids = _encode(tokenizer, source_prompt)
    positive_ids = _encode(tokenizer, str(adapted["positive_answer"]))
    negative_ids = _encode(tokenizer, str(adapted["negative_answer"]))
    intervention_positive_ids = _encode(
        tokenizer, str(adapted["intervention_positive_answer"]))
    intervention_negative_ids = _encode(
        tokenizer, str(adapted["intervention_negative_answer"]))
    source_behavior_required = bool(adapted.get(
        "source_behavior_required", True))
    source_positive_ids: tuple[int, ...] = ()
    source_negative_ids: tuple[int, ...] = ()
    if source_behavior_required:
        for field_name in ("source_positive_answer", "source_negative_answer"):
            if not str(adapted.get(field_name) or ""):
                raise ValueError(
                    f"{adapted['example_id']}: missing required {field_name}")
        source_positive_ids = _encode(
            tokenizer, str(adapted["source_positive_answer"]))
        source_negative_ids = _encode(
            tokenizer, str(adapted["source_negative_answer"]))

    required_candidates = (
        positive_ids, negative_ids,
        intervention_positive_ids, intervention_negative_ids,
    )
    if not all(required_candidates):
        raise BenchmarkEligibilityError(
            f"{adapted['example_id']}: answer tokenization is empty")
    if source_behavior_required and not all(
            (source_positive_ids, source_negative_ids)):
        raise BenchmarkEligibilityError(
            f"{adapted['example_id']}: source answer tokenization is empty")
    _different("base", positive_ids, negative_ids)
    _different(
        "intervention", intervention_positive_ids,
        intervention_negative_ids)
    if source_behavior_required:
        _different("source", source_positive_ids, source_negative_ids)

    token_contract = str(metadata.get("token_contract") or "")
    if token_contract in {
            "mib_equal_candidate_lengths",
            "equal_length_candidate_contrasts"}:
        _equal_length("base", positive_ids, negative_ids)
        _equal_length(
            "intervention", intervention_positive_ids,
            intervention_negative_ids)
        if source_behavior_required:
            _equal_length("source", source_positive_ids, source_negative_ids)
    elif token_contract == "mib_single_token_candidates":
        candidates = [
            positive_ids, negative_ids,
            intervention_positive_ids, intervention_negative_ids,
        ]
        if source_behavior_required:
            candidates.extend((source_positive_ids, source_negative_ids))
        if any(len(ids) != 1 for ids in candidates):
            raise BenchmarkEligibilityError(
                "MIB arithmetic candidates must each be one model token")
    elif token_contract == "variable_length_candidate_contrasts":
        if metadata.get("candidate_score_normalization") != (
                "mean_log_probability_per_token"):
            raise ValueError(
                f"{adapted['example_id']}: variable-length candidates require "
                "mean-log-probability normalization")
    else:
        raise ValueError(
            f"{adapted['example_id']}: unknown token contract {token_contract!r}")

    candidate_lengths = [
        len(positive_ids), len(negative_ids),
        len(intervention_positive_ids), len(intervention_negative_ids),
    ]
    if source_behavior_required:
        candidate_lengths.extend((
            len(source_positive_ids), len(source_negative_ids)))
    if max(len(base_ids), len(source_ids)) + max(candidate_lengths) > max_seq_len:
        raise BenchmarkEligibilityError(
            f"{adapted['example_id']}: sequence exceeds max_seq_len={max_seq_len}")
    position_kind = str(adapted.get("position_kind") or "last_token")
    base_position = resolve_trace_position(
        tokenizer, base_prompt, kind=position_kind,
        anchor=adapted.get("trace_anchor_base"))
    source_position = resolve_trace_position(
        tokenizer, source_prompt, kind=position_kind,
        anchor=adapted.get("trace_anchor_source"))
    metadata.update({
        "position_kind": position_kind,
        "base_prompt_token_count": len(base_ids),
        "source_prompt_token_count": len(source_ids),
        "positive_token_count": len(positive_ids),
        "negative_token_count": len(negative_ids),
        "intervention_positive_token_count": len(intervention_positive_ids),
        "intervention_negative_token_count": len(intervention_negative_ids),
        "token_position_audited": True,
    })
    return BenchmarkExample(
        benchmark_id=benchmark_id,
        example_id=str(adapted["example_id"]),
        phase=phase,
        base_prompt=base_prompt,
        source_prompt=source_prompt,
        positive_answer=str(adapted["positive_answer"]),
        negative_answer=str(adapted["negative_answer"]),
        intervention_positive_answer=str(
            adapted["intervention_positive_answer"]),
        intervention_negative_answer=str(
            adapted["intervention_negative_answer"]),
        causal_variable=str(adapted["causal_variable"]),
        pair_type=str(adapted["pair_type"]),
        source_behavior_required=source_behavior_required,
        trace_position_base=base_position,
        trace_position_source=source_position,
        input_ids_base=base_ids,
        input_ids_source=source_ids,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        source_positive_ids=source_positive_ids,
        source_negative_ids=source_negative_ids,
        intervention_positive_ids=intervention_positive_ids,
        intervention_negative_ids=intervention_negative_ids,
        metadata=metadata,
    ).validate()
