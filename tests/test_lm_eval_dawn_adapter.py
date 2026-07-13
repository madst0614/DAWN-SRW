from types import SimpleNamespace

import pytest

pytest.importorskip("lm_eval", reason="requires pinned lm-eval==0.4.2")

from dawn.eval.lm_eval_dawn_adapter import DawnJaxLM
from dawn.eval.zero_shot_protocol import ScoredRequest


class TinyTokenizer:
    vocab_size = 32
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [3 + (ord(char) % 20) for char in text]

    def decode(self, tokens, **kwargs):
        del kwargs
        return "".join(chr(97 + (int(token) % 26)) for token in tokens)


class MockScorer:
    global_batch_size = 4

    def __init__(self):
        self.digests = []

    def assert_request_consensus(self, digest):
        self.digests.append(digest)

    def score_requests(self, requests, pad_token_id):
        del pad_token_id
        return [
            ScoredRequest(
                loglikelihood=-0.25 * request.scored_tokens,
                is_greedy=all(token % 2 == 0 for token in request.continuation_ids),
                scored_tokens=request.scored_tokens,
            )
            for request in requests
        ]

    def generate_one(self, **kwargs):
        assert kwargs["max_gen_toks"] > 0
        return "generated"


class PartialBatchScorer(MockScorer):
    def consume_runtime_delta(self):
        return {
            "synthetic": {"dummy_examples": 1, "wall_time": 0.25},
        }


def request(context, continuation, idx=0):
    return SimpleNamespace(
        args=(context, continuation), task_name="synthetic", doc_id=9, idx=idx)


def test_adapter_returns_sum_natural_loglikelihood_and_exact_greedy():
    scorer = MockScorer()
    model = DawnJaxLM(
        tokenizer=TinyTokenizer(), scorer=scorer, max_length=64,
        length_buckets=(64,), pad_token_id=0, eot_token_id=2)
    answers = model.loglikelihood([
        request("a", "b"),
        request("a", "d"),
    ])
    assert len(answers) == 2
    assert answers[0][0] < 0
    assert isinstance(answers[0][1], bool)
    assert len(scorer.digests) == 1
    traces = model.sample_traces()
    assert traces[0]["encoded_length"] >= 2
    assert traces[0]["summed_loglikelihood"] == answers[0][0]


def test_adapter_generate_until_is_real_runtime_delegation():
    model = DawnJaxLM(
        tokenizer=TinyTokenizer(), scorer=MockScorer(), max_length=64,
        length_buckets=(64,), pad_token_id=0, eot_token_id=2)
    req = SimpleNamespace(args=("prompt", {"until": ["stop"],
                                           "max_gen_toks": 4}))
    assert model.generate_until([req]) == ["generated"]


def test_non_greedy_generation_fails_loudly():
    model = DawnJaxLM(
        tokenizer=TinyTokenizer(), scorer=MockScorer(), max_length=64,
        length_buckets=(64,), pad_token_id=0, eot_token_id=2)
    req = SimpleNamespace(args=("prompt", {"do_sample": True}))
    with pytest.raises(ValueError, match="deterministic greedy"):
        model.generate_until([req])


def test_final_partial_batch_excludes_dummy_from_scores_and_token_count():
    model = DawnJaxLM(
        tokenizer=TinyTokenizer(), scorer=PartialBatchScorer(), max_length=64,
        length_buckets=(64,), pad_token_id=0, eot_token_id=2)
    answers = model.loglikelihood([
        request("a", "b", idx=0),
        request("a", "c", idx=1),
        request("a", "d", idx=2),
    ])
    assert len(answers) == 3
    stats = model.runtime_stats()["synthetic"]
    assert stats["real_examples"] == 3
    assert stats["dummy_examples"] == 1
    assert stats["scored_tokens"] == 3
