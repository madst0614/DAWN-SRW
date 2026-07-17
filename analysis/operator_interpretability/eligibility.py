"""Tokenizer-level eligibility and benchmark-specific position auditing."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from analysis.operator_interpretability.benchmark_schema import BenchmarkExample


def tokenizer_vocab_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    ordered = sorted((str(token), int(index)) for token, index in vocab.items())
    payload = "\n".join(f"{index}\t{token}" for token, index in ordered)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_bucket(example_id: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def shared_split_phase(example_id: str, *, include_test: bool) -> str:
    """Create immutable disjoint phases when an official dataset lacks one."""
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
    # RAVEL prompts can contain demonstrations before the queried entity.
    # The target entity is the final official occurrence in the prompt.
    start = prompt.rfind(anchor)
    if start < 0:
        raise ValueError(f"trace anchor {anchor!r} is absent from official prompt")
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
            raise ValueError("tokenizer offsets do not cover trace anchor")
        return int(candidates[-1])
    except (KeyError, TypeError, NotImplementedError):
        prefix_ids = _encode(tokenizer, prompt[:end])
        if not prefix_ids:
            raise ValueError("trace anchor tokenization is empty")
        return len(prefix_ids) - 1


def resolve_trace_position(tokenizer: Any, prompt: str, *, kind: str,
                           anchor: str | None = None) -> int:
    ids = _encode(tokenizer, prompt)
    if not ids:
        raise ValueError("official benchmark prompt tokenizes to zero tokens")
    if kind == "last_token":
        return len(ids) - 1
    if kind == "entity_last_token":
        if not anchor:
            raise ValueError("entity_last_token requires an explicit entity anchor")
        return _anchor_last_token(tokenizer, prompt, anchor)
    raise ValueError(f"unsupported benchmark position kind: {kind!r}")


def tokenize_adapted_pair(
        tokenizer: Any, benchmark_id: str, phase: str,
        adapted: Mapping[str, Any], *, max_seq_len: int) -> BenchmarkExample:
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
            raise ValueError("official minimal pair has no shared token prefix")
        if prefix_length >= min(len(good_ids), len(bad_ids)):
            raise ValueError("official minimal pair lacks divergent continuations")
        base_ids = tuple(good_ids[:prefix_length])
        source_ids = base_ids
        positive_ids = tuple(good_ids[prefix_length:])
        negative_ids = tuple(bad_ids[prefix_length:])
        base_prompt = tokenizer.decode(base_ids, clean_up_tokenization_spaces=False)
        source_prompt = base_prompt
        if len(base_ids) + max(len(positive_ids), len(negative_ids)) > max_seq_len:
            raise ValueError("official minimal pair exceeds max_seq_len")
        metadata.update({
            "shared_prefix_token_count": prefix_length,
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
            causal_variable=str(adapted["causal_variable"]),
            pair_type=str(adapted["pair_type"]),
            trace_position_base=prefix_length - 1,
            trace_position_source=prefix_length - 1,
            input_ids_base=base_ids,
            input_ids_source=source_ids,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            source_positive_ids=positive_ids,
            source_negative_ids=negative_ids,
            metadata=metadata,
        ).validate()

    base_prompt = str(adapted["base_prompt"])
    source_prompt = str(adapted["source_prompt"])
    base_ids = _encode(tokenizer, base_prompt)
    source_ids = _encode(tokenizer, source_prompt)
    positive_ids = _encode(tokenizer, str(adapted["positive_answer"]))
    negative_ids = _encode(tokenizer, str(adapted["negative_answer"]))
    source_positive_ids = _encode(
        tokenizer, str(adapted.get(
            "source_positive_answer", adapted["positive_answer"])))
    source_negative_ids = _encode(
        tokenizer, str(adapted.get(
            "source_negative_answer", adapted["negative_answer"])))
    if not all((positive_ids, negative_ids, source_positive_ids,
                source_negative_ids)):
        raise ValueError(f"{adapted['example_id']}: answer tokenization is empty")
    longest_base = len(base_ids) + max(len(positive_ids), len(negative_ids))
    longest_source = len(source_ids) + max(
        len(source_positive_ids), len(source_negative_ids))
    if max(longest_base, longest_source) > max_seq_len:
        raise ValueError(
            f"{adapted['example_id']}: sequence exceeds max_seq_len={max_seq_len}")
    primary_mib = benchmark_id.startswith("mib_")
    if primary_mib and (len(positive_ids) != 1 or len(negative_ids) != 1):
        raise ValueError(
            f"{adapted['example_id']}: MIB labels must each be one model token")
    if benchmark_id == "ravel" and (
            len(source_positive_ids) != len(positive_ids)):
        raise ValueError(
            f"{adapted['example_id']}: RAVEL source/base answer token lengths differ")
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
        causal_variable=str(adapted["causal_variable"]),
        pair_type=str(adapted["pair_type"]),
        trace_position_base=base_position,
        trace_position_source=source_position,
        input_ids_base=base_ids,
        input_ids_source=source_ids,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        source_positive_ids=source_positive_ids,
        source_negative_ids=source_negative_ids,
        metadata=metadata,
    ).validate()
