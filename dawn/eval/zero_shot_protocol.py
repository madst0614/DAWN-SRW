"""Frozen protocol and pure helpers for DAWN zero-shot evaluation.

This module intentionally has no JAX, Transformers, datasets, or lm-eval
imports.  The causal alignment and result summarization logic can therefore be
unit-tested on CPU without downloading models or benchmark datasets.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROTOCOL_NAME = "mamba_table3_zero_shot_v1"
PROTOCOL_VERSION = 1
LM_EVAL_VERSION = "0.4.2"
NUM_FEWSHOT = 0
PRIMARY_TASKS: Tuple[str, ...] = (
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
)
DEFAULT_LENGTH_BUCKETS: Tuple[int, ...] = (64, 128, 256, 512)

PRIMARY_METRICS: Mapping[str, Tuple[str, ...]] = {
    "lambada_openai": ("perplexity", "acc"),
    "hellaswag": ("acc_norm",),
    "piqa": ("acc_norm",),
    "arc_easy": ("acc_norm",),
    "arc_challenge": ("acc_norm",),
    "winogrande": ("acc",),
}


def normalize_tasks(tasks: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    values = tuple(PRIMARY_TASKS if tasks is None else tasks)
    normalized = tuple(dict.fromkeys(str(task).strip() for task in values))
    if not normalized or any(not task for task in normalized):
        raise ValueError("at least one non-empty zero-shot task is required")
    unknown = [task for task in normalized if task not in PRIMARY_TASKS]
    if unknown:
        raise ValueError(
            "unknown stock zero-shot tasks: " + ",".join(unknown))
    return normalized


@dataclass(frozen=True)
class PreparedRequest:
    """One causal request after boundary tokenization and left truncation."""

    ordinal: int
    task: str
    doc_id: Any
    choice_index: int
    context: str
    continuation: str
    context_ids: Tuple[int, ...]
    continuation_ids: Tuple[int, ...]
    max_length: int
    bucket: int
    original_encoded_length: int
    left_truncated: bool

    @property
    def encoded_length(self) -> int:
        return len(self.context_ids) + len(self.continuation_ids)

    @property
    def scored_tokens(self) -> int:
        return len(self.continuation_ids)

    def materialize(self, pad_token_id: int) -> Tuple[List[int], List[int], List[int]]:
        """Build fixed-shape model input, labels, and attention mask.

        DAWN's canonical forward shifts ``labels`` internally.  The final
        continuation token is therefore excluded from the input and replaced
        by a pad placeholder at the same label position.
        """
        context = list(self.context_ids)
        continuation = list(self.continuation_ids)
        used = len(context) + len(continuation)
        if used > self.bucket:
            raise AssertionError(
                f"encoded request length {used} exceeds bucket {self.bucket}")
        input_ids = context + continuation[:-1] + [int(pad_token_id)]
        labels = [-100] * len(context) + continuation
        # The placeholder is not a real input token.  It is also causally after
        # every position used for scoring, so models that ignore this mask are
        # still correct for the continuation logits.
        attention_mask = [1] * (used - 1) + [0]
        padding = self.bucket - used
        input_ids.extend([int(pad_token_id)] * padding)
        labels.extend([-100] * padding)
        attention_mask.extend([0] * padding)
        if not (
            len(input_ids) == len(labels) == len(attention_mask) == self.bucket
        ):
            raise AssertionError("materialized causal request has invalid shape")
        return input_ids, labels, attention_mask


@dataclass(frozen=True)
class ScoredRequest:
    loglikelihood: float
    is_greedy: bool
    scored_tokens: int
    token_loglikelihoods: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.loglikelihood)):
            raise FloatingPointError(
                f"non-finite request loglikelihood: {self.loglikelihood}")
        if self.scored_tokens <= 0:
            raise ValueError("valid target token count must be positive")


@dataclass
class TaskRuntimeStats:
    total_requests: int = 0
    left_truncated_requests: int = 0
    continuation_too_long_requests: int = 0
    encoded_length_sum: int = 0
    max_encoded_length: int = 0
    real_examples: int = 0
    dummy_examples: int = 0
    scored_tokens: int = 0
    wall_time: float = 0.0
    buckets_seen: set = field(default_factory=set)
    bucket_request_counts: Dict[str, int] = field(default_factory=dict)

    def observe_request(self, request: PreparedRequest) -> None:
        self.total_requests += 1
        self.left_truncated_requests += int(request.left_truncated)
        self.encoded_length_sum += request.encoded_length
        self.max_encoded_length = max(
            self.max_encoded_length, request.encoded_length)
        self.real_examples += 1
        self.scored_tokens += request.scored_tokens
        self.buckets_seen.add(int(request.bucket))
        key = str(int(request.bucket))
        self.bucket_request_counts[key] = self.bucket_request_counts.get(key, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        mean_length = (
            self.encoded_length_sum / self.total_requests
            if self.total_requests else 0.0)
        return {
            "total_requests": self.total_requests,
            "left_truncated_requests": self.left_truncated_requests,
            "continuation_too_long_requests": (
                self.continuation_too_long_requests),
            "max_encoded_length": self.max_encoded_length,
            "mean_encoded_length": mean_length,
            "compile_count": len(self.buckets_seen),
            "bucket_request_counts": dict(
                sorted(self.bucket_request_counts.items(), key=lambda x: int(x[0]))),
            "real_examples": self.real_examples,
            "dummy_examples": self.dummy_examples,
            "scored_tokens": self.scored_tokens,
            "wall_time": self.wall_time,
            "examples_per_sec": (
                self.real_examples / self.wall_time if self.wall_time > 0 else 0.0),
            "tokens_per_sec": (
                self.scored_tokens / self.wall_time if self.wall_time > 0 else 0.0),
        }


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def normalize_buckets(
    buckets: Iterable[int], max_length: int
) -> Tuple[int, ...]:
    max_length = int(max_length)
    if max_length < 2:
        raise ValueError("max_length must be at least 2 for causal scoring")
    values = sorted({int(x) for x in buckets if 2 <= int(x) <= max_length})
    if not values or values[-1] != max_length:
        values.append(max_length)
    return tuple(values)


def select_bucket(length: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if length <= int(bucket):
            return int(bucket)
    raise ValueError(
        f"encoded length {length} exceeds largest length bucket {max(buckets)}")


def encode_pair_harness_compatible(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    eot_token_id: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Match lm-eval 0.4.2 ``TemplateLM._encode_pair`` exactly.

    Trailing spaces are moved from context to continuation before tokenizing.
    For causal models, the combined string and context are tokenized separately
    and the continuation is the suffix after the encoded context length.
    """
    if not isinstance(context, str) or not isinstance(continuation, str):
        raise TypeError("context and continuation must be strings")
    if continuation == "":
        raise ValueError("continuation must encode at least one target token")
    if context == "":
        if eot_token_id is None:
            raise RuntimeError(
                "empty-context scoring requires an explicit source EOT/BOS token; "
                "the tokenizer/checkpoint policy did not define one")
        context_ids = [int(eot_token_id)]
        continuation_ids = list(tokenizer.encode(
            continuation, add_special_tokens=False))
    else:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_ids = list(tokenizer.encode(
            context + continuation, add_special_tokens=False))
        context_ids = list(tokenizer.encode(
            context, add_special_tokens=False))
        continuation_ids = whole_ids[len(context_ids):]
    if not context_ids:
        raise ValueError(
            "context tokenization produced zero tokens; no causal prefix is "
            "available and automatic special-token insertion is disabled")
    if not continuation_ids:
        raise ValueError("continuation tokenization produced zero target tokens")
    return context_ids, continuation_ids


def prepare_causal_request(
    *,
    ordinal: int,
    task: str,
    doc_id: Any,
    choice_index: int,
    context: str,
    continuation: str,
    context_ids: Sequence[int],
    continuation_ids: Sequence[int],
    max_length: int,
    buckets: Sequence[int],
) -> PreparedRequest:
    context_ids = [int(x) for x in context_ids]
    continuation_ids = [int(x) for x in continuation_ids]
    max_length = int(max_length)
    if not context_ids:
        raise ValueError("causal scoring requires at least one context token")
    if not continuation_ids:
        raise ValueError("valid target token count is zero")
    # One causal prefix position is required to score the first continuation
    # token.  We preserve every continuation token and only truncate context.
    if len(continuation_ids) + 1 > max_length:
        raise ValueError(
            "continuation is too long for explicit preserve-continuation "
            f"policy: continuation_tokens={len(continuation_ids)} "
            f"max_length={max_length}")
    original_length = len(context_ids) + len(continuation_ids)
    keep_context = max_length - len(continuation_ids)
    left_truncated = len(context_ids) > keep_context
    if left_truncated:
        context_ids = context_ids[-keep_context:]
    encoded_length = len(context_ids) + len(continuation_ids)
    bucket = select_bucket(encoded_length, buckets)
    return PreparedRequest(
        ordinal=int(ordinal),
        task=str(task),
        doc_id=doc_id,
        choice_index=int(choice_index),
        context=context,
        continuation=continuation,
        context_ids=tuple(context_ids),
        continuation_ids=tuple(continuation_ids),
        max_length=max_length,
        bucket=bucket,
        original_encoded_length=original_length,
        left_truncated=left_truncated,
    )


def reduce_token_scores(
    token_nll: Sequence[float],
    token_correct: Sequence[bool],
    target_mask: Sequence[bool],
) -> ScoredRequest:
    if not (len(token_nll) == len(token_correct) == len(target_mask)):
        raise ValueError("token score arrays must have identical lengths")
    selected_nll = [
        float(nll) for nll, keep in zip(token_nll, target_mask) if keep]
    selected_correct = [
        bool(value) for value, keep in zip(token_correct, target_mask) if keep]
    if not selected_nll:
        raise ValueError("valid target token count is zero")
    if not all(math.isfinite(value) for value in selected_nll):
        raise FloatingPointError("NaN/Inf token NLL detected")
    token_ll = tuple(-value for value in selected_nll)
    return ScoredRequest(
        loglikelihood=float(sum(token_ll)),
        is_greedy=all(selected_correct),
        scored_tokens=len(selected_nll),
        token_loglikelihoods=token_ll,
    )


def _metric_candidates(result: Mapping[str, Any], metric: str) -> List[str]:
    return [
        key for key in result
        if key == metric or key.startswith(metric + ",")]


def require_metric(result: Mapping[str, Any], task: str, metric: str) -> float:
    keys = _metric_candidates(result, metric)
    if len(keys) != 1:
        raise KeyError(
            f"task {task!r} expected exactly one stock metric {metric!r}; "
            f"found {keys or '<none>'}; available={sorted(result)}")
    value = float(result[keys[0]])
    if not math.isfinite(value):
        raise FloatingPointError(
            f"task {task!r} metric {keys[0]!r} is non-finite: {value}")
    return value


def build_results_summary(
    raw_results: Mapping[str, Any],
    *,
    model: str,
    step: int,
    validation_loss: Optional[float],
    comparable: bool,
    task_runtime: Optional[Mapping[str, Any]] = None,
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    selected_tasks = normalize_tasks(tasks)
    results = raw_results.get("results")
    if not isinstance(results, Mapping):
        raise KeyError("lm-eval result is missing the top-level 'results' map")
    missing = [task for task in selected_tasks if task not in results]
    if missing:
        raise KeyError("final task result missing: " + ", ".join(missing))
    for task in selected_tasks:
        metrics = PRIMARY_METRICS[task]
        for metric in metrics:
            require_metric(results[task], task, metric)

    task_accuracy_metric = {
        "lambada_openai": "acc",
        "hellaswag": "acc_norm",
        "piqa": "acc_norm",
        "arc_easy": "acc_norm",
        "arc_challenge": "acc_norm",
        "winogrande": "acc",
    }
    accuracies = {
        task: require_metric(results[task], task, task_accuracy_metric[task])
        for task in selected_tasks
    }
    mean_acc = sum(accuracies.values()) / len(accuracies)
    row = {
        "Model": str(model),
        "Step": int(step),
        "Val loss": None if validation_loss is None else float(validation_loss),
        "LAMBADA PPL": (
            require_metric(results["lambada_openai"], "lambada_openai", "perplexity")
            if "lambada_openai" in selected_tasks else None),
        "LAMBADA ACC": accuracies.get("lambada_openai"),
        "HellaSwag": accuracies.get("hellaswag"),
        "PIQA": accuracies.get("piqa"),
        "ARC-E": accuracies.get("arc_easy"),
        "ARC-C": accuracies.get("arc_challenge"),
        "WinoGrande": accuracies.get("winogrande"),
        "Mean ACC": mean_acc,
    }
    full_suite = selected_tasks == PRIMARY_TASKS
    return {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "num_fewshot": NUM_FEWSHOT,
        "tasks": list(selected_tasks),
        "complete_primary_suite": full_suite,
        "comparable": bool(comparable and full_suite),
        "smoke_test_only": not bool(comparable and full_suite),
        "table": row,
        "metrics": json_safe(results),
        "task_runtime": json_safe(task_runtime or {}),
    }


def csv_header_and_row(summary: Mapping[str, Any]) -> Tuple[List[str], List[Any]]:
    table = summary["table"]
    header = [
        "Model", "Step", "Val loss", "LAMBADA PPL", "LAMBADA ACC",
        "HellaSwag", "PIQA", "ARC-E", "ARC-C", "WinoGrande", "Mean ACC",
    ]
    return header, [table.get(key) for key in header]


def request_sequence_hash(requests: Sequence[PreparedRequest]) -> str:
    payload = [
        {
            "ordinal": request.ordinal,
            "task": request.task,
            "doc_id": json_safe(request.doc_id),
            "choice_index": request.choice_index,
            "context": request.context,
            "continuation": request.continuation,
            "context_ids": request.context_ids,
            "continuation_ids": request.continuation_ids,
        }
        for request in requests
    ]
    return sha256_json(payload)


def task_stats_dict(
    stats: Mapping[str, TaskRuntimeStats]
) -> Dict[str, Dict[str, Any]]:
    return {task: value.as_dict() for task, value in sorted(stats.items())}


def prepared_request_dict(request: PreparedRequest) -> Dict[str, Any]:
    return json_safe(asdict(request))
