import math

import pytest

from dawn.eval.zero_shot_protocol import (
    PRIMARY_TASKS,
    build_results_summary,
    encode_pair_harness_compatible,
    normalize_buckets,
    prepare_causal_request,
    reduce_token_scores,
    request_sequence_hash,
)


class TableTokenizer:
    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text not in self.table:
            raise KeyError(text)
        return list(self.table[text])


def make_request(context_ids, continuation_ids, max_length=8):
    return prepare_causal_request(
        ordinal=0,
        task="synthetic",
        doc_id=3,
        choice_index=0,
        context="ctx",
        continuation=" cont",
        context_ids=context_ids,
        continuation_ids=continuation_ids,
        max_length=max_length,
        buckets=normalize_buckets((4, 8), max_length),
    )


def test_one_token_continuation_shift_and_padding():
    request = make_request([10, 11], [20])
    input_ids, labels, attention = request.materialize(pad_token_id=0)
    assert input_ids[:3] == [10, 11, 0]
    assert labels[:3] == [-100, -100, 20]
    assert attention[:3] == [1, 1, 0]
    assert labels[1:][1] == 20  # logits at context position 1 score token 20
    assert len(input_ids) == request.bucket == 4


def test_multi_token_continuation_excludes_final_target_from_input():
    request = make_request([10, 11], [20, 21, 22], max_length=8)
    input_ids, labels, _ = request.materialize(pad_token_id=0)
    assert input_ids[:5] == [10, 11, 20, 21, 0]
    assert labels[:5] == [-100, -100, 20, 21, 22]
    assert 22 not in input_ids[:5]


def test_empty_context_requires_explicit_eot():
    tokenizer = TableTokenizer({" answer": [5, 6]})
    with pytest.raises(RuntimeError, match="explicit source EOT/BOS"):
        encode_pair_harness_compatible(tokenizer, "", " answer")
    context, continuation = encode_pair_harness_compatible(
        tokenizer, "", " answer", eot_token_id=2)
    assert context == [2]
    assert continuation == [5, 6]


def test_trailing_context_space_moves_into_continuation_like_harness():
    tokenizer = TableTokenizer({"hello": [1], "hello world": [1, 2]})
    context, continuation = encode_pair_harness_compatible(
        tokenizer, "hello ", "world")
    assert context == [1]
    assert continuation == [2]


def test_wordpiece_boundary_uses_combined_encoding_suffix():
    tokenizer = TableTokenizer({"play": [7], "playing": [8, 9]})
    context, continuation = encode_pair_harness_compatible(
        tokenizer, "play", "ing")
    assert context == [7]
    assert continuation == [9]


def test_left_truncation_preserves_entire_continuation():
    request = make_request([1, 2, 3, 4, 5, 6], [7, 8, 9], max_length=8)
    assert request.left_truncated is True
    assert request.context_ids == (2, 3, 4, 5, 6)
    assert request.continuation_ids == (7, 8, 9)
    assert request.encoded_length == 8


def test_continuation_too_long_fails_loudly():
    with pytest.raises(ValueError, match="continuation is too long"):
        make_request([1], list(range(8)), max_length=8)


def test_padding_is_never_scored_and_greedy_is_all_tokens():
    true_result = reduce_token_scores(
        [0.1, 0.2, 999.0], [True, True, False], [True, True, False])
    false_result = reduce_token_scores(
        [0.1, 0.2, 999.0], [True, False, True], [True, True, False])
    assert math.isclose(true_result.loglikelihood, -0.3)
    assert true_result.scored_tokens == 2
    assert true_result.is_greedy is True
    assert false_result.is_greedy is False


def test_adapter_nll_to_validation_ce_identity():
    rows = [
        reduce_token_scores([0.2, 0.4], [True, True], [True, True]),
        reduce_token_scores([0.1, 0.3], [True, True], [True, True]),
    ]
    adapter_ce = -sum(row.loglikelihood for row in rows) / sum(
        row.scored_tokens for row in rows)
    assert adapter_ce == pytest.approx((0.2 + 0.4 + 0.1 + 0.3) / 4)


def test_request_hash_detects_order_changes_for_multihost_consensus():
    first = make_request([1], [2])
    second = prepare_causal_request(
        ordinal=1,
        task="synthetic",
        doc_id=4,
        choice_index=0,
        context="other",
        continuation=" answer",
        context_ids=[3],
        continuation_ids=[4],
        max_length=8,
        buckets=(4, 8),
    )
    assert request_sequence_hash([first, second]) == request_sequence_hash(
        [first, second])
    assert request_sequence_hash([first, second]) != request_sequence_hash(
        [second, first])


def test_primary_metric_mapping_and_mean_accuracy_preserve_stderr():
    raw = {
        "results": {
            "lambada_openai": {
                "perplexity,none": 5.0,
                "perplexity_stderr,none": 0.2,
                "acc,none": 0.5,
                "acc_stderr,none": 0.01,
            },
            "hellaswag": {"acc,none": 0.4, "acc_norm,none": 0.6},
            "piqa": {"acc,none": 0.5, "acc_norm,none": 0.7},
            "arc_easy": {"acc,none": 0.4, "acc_norm,none": 0.8},
            "arc_challenge": {"acc,none": 0.3, "acc_norm,none": 0.4},
            "winogrande": {"acc,none": 0.6},
        }
    }
    summary = build_results_summary(
        raw, model="synthetic", step=12, validation_loss=1.2,
        comparable=True)
    assert summary["table"]["LAMBADA PPL"] == 5.0
    assert summary["table"]["HellaSwag"] == 0.6
    assert summary["table"]["Mean ACC"] == pytest.approx(
        (0.5 + 0.6 + 0.7 + 0.8 + 0.4 + 0.6) / 6)
    assert summary["metrics"]["lambada_openai"][
        "perplexity_stderr,none"] == 0.2


def test_missing_primary_metric_never_falls_back_to_acc():
    raw = {"results": {task: {} for task in PRIMARY_TASKS}}
    raw["results"]["lambada_openai"] = {"perplexity,none": 5, "acc,none": .5}
    raw["results"]["hellaswag"] = {"acc,none": .5}
    raw["results"]["piqa"] = {"acc_norm,none": .5}
    raw["results"]["arc_easy"] = {"acc_norm,none": .5}
    raw["results"]["arc_challenge"] = {"acc_norm,none": .5}
    raw["results"]["winogrande"] = {"acc,none": .5}
    with pytest.raises(KeyError, match="acc_norm"):
        build_results_summary(
            raw, model="synthetic", step=1, validation_loss=None,
            comparable=False)
