"""lm-evaluation-harness 0.4.2 adapter for a frozen DAWN JAX runtime."""

from __future__ import annotations

import collections
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import transformers
from lm_eval import utils as lm_eval_utils
from lm_eval.api.model import TemplateLM

from .zero_shot_protocol import (
    PreparedRequest,
    ScoredRequest,
    TaskRuntimeStats,
    encode_pair_harness_compatible,
    json_safe,
    normalize_buckets,
    prepare_causal_request,
    request_sequence_hash,
    sha256_json,
    task_stats_dict,
)


class DawnJaxLM(TemplateLM):
    """Stock lm-eval interface backed only by DAWN's JAX causal forward.

    ``scorer`` is a small runtime object supplied by ``zero_shot_eval_jax``.
    It must implement ``score_requests`` and ``generate_one``.  Keeping model
    restore and device orchestration out of this class makes the tokenizer and
    causal-shift contract independently testable.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        *,
        tokenizer: Any,
        scorer: Any,
        max_length: int,
        length_buckets: Sequence[int],
        pad_token_id: int,
        eot_token_id: Optional[int],
        max_gen_toks: int = 256,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.scorer = scorer
        self._max_length = int(max_length)
        self._length_buckets = normalize_buckets(
            length_buckets, self._max_length)
        self._pad_token_id = int(pad_token_id)
        self._eot_token_id = (
            None if eot_token_id is None else int(eot_token_id))
        self._max_gen_toks = int(max_gen_toks)
        if self._max_gen_toks <= 0:
            raise ValueError("max_gen_toks must be positive")
        self._task_stats: Dict[str, TaskRuntimeStats] = collections.defaultdict(
            TaskRuntimeStats)
        self._sample_traces: List[Dict[str, Any]] = []

    @property
    def eot_token_id(self) -> int:
        if self._eot_token_id is None:
            raise RuntimeError(
                "EOT/BOS is undefined by the source checkpoint tokenizer "
                "policy. Pass an explicitly verified --eot-token-id before "
                "using empty-context or rolling-loglikelihood requests.")
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return int(getattr(self.scorer, "global_batch_size", 1))

    @property
    def device(self) -> str:
        return "jax-tpu"

    def tok_encode(self, string: str, **kwargs: Any) -> List[int]:
        if kwargs.get("add_special_tokens") not in (None, False):
            raise ValueError(
                "automatic special-token insertion is disabled by protocol")
        return list(self.tokenizer.encode(string, add_special_tokens=False))

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(tokens), skip_special_tokens=False,
            clean_up_tokenization_spaces=False)

    @staticmethod
    def _metadata(req: Any, ordinal: int) -> Tuple[str, Any, int]:
        task = getattr(req, "task_name", None)
        doc_id = getattr(req, "doc_id", None)
        choice_index = getattr(req, "idx", 0)
        metadata = getattr(req, "metadata", None)
        if metadata is not None and isinstance(metadata, (list, tuple)):
            if task is None and len(metadata) > 0:
                task = metadata[0]
            if doc_id is None and len(metadata) > 1:
                doc_id = metadata[1]
        return str(task or "<unknown>"), (
            ordinal if doc_id is None else doc_id), int(choice_index or 0)

    def _prepare_text_requests(self, requests: Sequence[Any]) -> List[PreparedRequest]:
        prepared = []
        for ordinal, req in enumerate(requests):
            context, continuation = req.args
            task, doc_id, choice_index = self._metadata(req, ordinal)
            context_ids, continuation_ids = encode_pair_harness_compatible(
                self.tokenizer,
                context,
                continuation,
                eot_token_id=self._eot_token_id,
            )
            prepared.append(prepare_causal_request(
                ordinal=ordinal,
                task=task,
                doc_id=doc_id,
                choice_index=choice_index,
                context=context,
                continuation=continuation,
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                max_length=self._max_length,
                buckets=self._length_buckets,
            ))
        return prepared

    def _apply_runtime_delta(self) -> None:
        consume = getattr(self.scorer, "consume_runtime_delta", None)
        if consume is None:
            return
        delta = consume()
        for task, values in (delta or {}).items():
            stats = self._task_stats[str(task)]
            stats.dummy_examples += int(values.get("dummy_examples", 0))
            stats.wall_time += float(values.get("wall_time", 0.0))

    def _score_prepared(
        self, prepared: Sequence[PreparedRequest]
    ) -> List[ScoredRequest]:
        if not prepared:
            return []
        digest = request_sequence_hash(prepared)
        consensus = getattr(self.scorer, "assert_request_consensus", None)
        if consensus is not None:
            consensus(digest)
        for request in prepared:
            self._task_stats[request.task].observe_request(request)
        start = time.monotonic()
        scored = list(self.scorer.score_requests(
            list(prepared), pad_token_id=self._pad_token_id))
        elapsed = time.monotonic() - start
        if len(scored) != len(prepared):
            raise RuntimeError(
                "JAX scorer returned a different number of responses: "
                f"requests={len(prepared)} responses={len(scored)}")
        self._apply_runtime_delta()
        # Mock/single-process scorers do not necessarily provide per-task timing.
        if not hasattr(self.scorer, "consume_runtime_delta"):
            tasks = sorted({request.task for request in prepared})
            share = elapsed / max(len(tasks), 1)
            for task in tasks:
                self._task_stats[task].wall_time += share
        for request, result in zip(prepared, scored):
            if not isinstance(result, ScoredRequest):
                result = ScoredRequest(
                    loglikelihood=float(result[0]),
                    is_greedy=bool(result[1]),
                    scored_tokens=request.scored_tokens,
                )
            if result.scored_tokens != request.scored_tokens:
                raise RuntimeError(
                    "continuation target alignment mismatch: "
                    f"task={request.task} doc_id={request.doc_id} "
                    f"expected={request.scored_tokens} "
                    f"actual={result.scored_tokens}")
            self._sample_traces.append({
                "task": request.task,
                "document_id": json_safe(request.doc_id),
                "choice_index": request.choice_index,
                "formatted_context": request.context,
                "continuation": request.continuation,
                "context_tokens": len(request.context_ids),
                "continuation_tokens": len(request.continuation_ids),
                "encoded_length": request.encoded_length,
                "original_encoded_length": request.original_encoded_length,
                "bucket": request.bucket,
                "summed_loglikelihood": result.loglikelihood,
                "is_greedy": result.is_greedy,
                "truncated": request.left_truncated,
            })
        return scored

    def loglikelihood(
        self, requests: Sequence[Any], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        del disable_tqdm
        prepared = self._prepare_text_requests(requests)
        scored = self._score_prepared(prepared)
        answers = [
            (float(result.loglikelihood), bool(result.is_greedy))
            for result in scored]
        for req, answer in zip(requests, answers):
            self.cache_hook.add_partial("loglikelihood", req.args, answer)
        return answers

    def _loglikelihood_tokens(
        self, requests: Sequence[Any], **kwargs: Any
    ) -> List[Tuple[float, bool]]:
        del kwargs
        prepared = []
        for ordinal, request in enumerate(requests):
            if len(request) == 3:
                cache_key, context_ids, continuation_ids = request
            elif len(request) == 2:
                cache_key, (context_ids, continuation_ids) = None, request
            else:
                raise ValueError("unexpected lm-eval token request shape")
            context = "" if cache_key is None else str(cache_key[0])
            continuation = "" if cache_key is None else str(cache_key[1])
            prepared.append(prepare_causal_request(
                ordinal=ordinal,
                task="<token_request>",
                doc_id=ordinal,
                choice_index=0,
                context=context,
                continuation=continuation,
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                max_length=self._max_length,
                buckets=self._length_buckets,
            ))
        return [
            (result.loglikelihood, result.is_greedy)
            for result in self._score_prepared(prepared)]

    def loglikelihood_rolling(
        self, requests: Sequence[Any], disable_tqdm: bool = False
    ) -> List[float]:
        del disable_tqdm
        prefix = self.eot_token_id
        answers: List[float] = []
        all_prepared: List[PreparedRequest] = []
        window_counts: List[int] = []
        for request_ordinal, req in enumerate(requests):
            (text,) = req.args
            task, doc_id, _ = self._metadata(req, request_ordinal)
            tokens = self.tok_encode(text)
            if not tokens:
                raise ValueError("rolling loglikelihood text encoded to zero tokens")
            windows = list(map(
                lm_eval_utils.make_disjoint_window,
                lm_eval_utils.get_rolling_token_windows(
                    token_list=tokens,
                    prefix_token=prefix,
                    max_seq_len=self._max_length,
                    context_len=1,
                ),
            ))
            window_counts.append(len(windows))
            for window_index, (context_ids, continuation_ids) in enumerate(windows):
                all_prepared.append(prepare_causal_request(
                    ordinal=len(all_prepared),
                    task=task,
                    doc_id=f"{doc_id}:window:{window_index}",
                    choice_index=0,
                    context=text,
                    continuation="",
                    context_ids=context_ids,
                    continuation_ids=continuation_ids,
                    max_length=self._max_length,
                    buckets=self._length_buckets,
                ))
        scored = self._score_prepared(all_prepared)
        offset = 0
        for req, count in zip(requests, window_counts):
            total = float(sum(
                item.loglikelihood for item in scored[offset:offset + count]))
            if not math.isfinite(total):
                raise FloatingPointError("rolling loglikelihood is NaN/Inf")
            answers.append(total)
            self.cache_hook.add_partial(
                "loglikelihood_rolling", req.args, total)
            offset += count
        return answers

    def generate_until(
        self, requests: Sequence[Any], disable_tqdm: bool = False
    ) -> List[str]:
        del disable_tqdm
        request_payload = []
        for req in requests:
            context, generation_kwargs = req.args
            request_payload.append((context, json_safe(generation_kwargs)))
        consensus = getattr(self.scorer, "assert_request_consensus", None)
        if consensus is not None:
            consensus(sha256_json(request_payload))

        results = []
        for req in requests:
            context, generation_kwargs = req.args
            if not isinstance(generation_kwargs, Mapping):
                raise TypeError("generate_until kwargs must be a mapping")
            do_sample = bool(generation_kwargs.get("do_sample", False))
            temperature = float(generation_kwargs.get("temperature", 0.0))
            if do_sample or temperature not in (0.0,):
                raise ValueError(
                    "DAWN zero-shot adapter supports deterministic greedy "
                    "generation only")
            unsupported = set(generation_kwargs) - {
                "until", "max_gen_toks", "do_sample", "temperature"}
            if unsupported:
                raise ValueError(
                    "unsupported generate_until arguments: "
                    + ", ".join(sorted(unsupported)))
            until = generation_kwargs.get("until", [])
            if until is None:
                until = []
            elif isinstance(until, str):
                until = [until]
            else:
                until = [str(item) for item in until]
            max_gen_toks = int(generation_kwargs.get(
                "max_gen_toks", self._max_gen_toks))
            if max_gen_toks <= 0:
                raise ValueError("max_gen_toks must be positive")
            if context == "":
                context_ids = [self.eot_token_id]
            else:
                context_ids = self.tok_encode(context)
            if not context_ids:
                raise ValueError("generation context encoded to zero tokens")
            generated = self.scorer.generate_one(
                context_ids=context_ids,
                max_gen_toks=max_gen_toks,
                until=until,
                tokenizer=self.tokenizer,
                pad_token_id=self._pad_token_id,
                length_buckets=self._length_buckets,
            )
            if not isinstance(generated, str):
                raise TypeError("JAX generate_one must return decoded text")
            results.append(generated)
            self.cache_hook.add_partial("generate_until", req.args, generated)
        return results

    def runtime_stats(self) -> Dict[str, Dict[str, Any]]:
        return task_stats_dict(self._task_stats)

    def sample_traces(self) -> List[Dict[str, Any]]:
        return list(self._sample_traces)
