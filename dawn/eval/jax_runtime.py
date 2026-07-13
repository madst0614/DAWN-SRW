"""JAX/SPMD execution runtime used by the DAWN lm-eval adapter."""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding, PartitionSpec as P

from models.vocab_parallel import (
    make_unsharded_argmax,
    make_unsharded_eval_stats,
    make_vocab_parallel_argmax,
    make_vocab_parallel_eval_stats,
)
from scripts import train_jax as canonical

from .zero_shot_protocol import (
    PreparedRequest,
    ScoredRequest,
    reduce_token_scores,
    select_bucket,
)


class JaxDawnScorer:
    """Frozen-parameter, fixed-shape JAX scorer for DAWN and dense baseline."""

    def __init__(
        self,
        *,
        model: Any,
        params: Any,
        mesh: Any,
        config: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
        base_sharded_fns: Any,
        global_batch_size: int,
        token_chunk_size: int = 32768,
    ) -> None:
        self.model = model
        self.params = params
        self.mesh = mesh
        self.config = config
        self.runtime_state = dict(runtime_state)
        self.global_batch_size = int(global_batch_size)
        self.process_count = int(jax.process_count())
        self.process_index = int(jax.process_index())
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.global_batch_size % self.process_count != 0:
            raise ValueError(
                "global batch must divide evenly across hosts: "
                f"batch={self.global_batch_size} hosts={self.process_count}")
        mesh_data = int(mesh.shape["data"])
        if self.global_batch_size % mesh_data != 0:
            raise ValueError(
                "global batch must divide the checkpoint data mesh: "
                f"batch={self.global_batch_size} mesh_data={mesh_data}")
        self.local_batch_size = self.global_batch_size // self.process_count
        self.data_sharding = NamedSharding(mesh, P("data", None))

        model_cfg = config["model"]
        logical_vocab = int(model_cfg.get(
            "logical_vocab_size", model_cfg["vocab_size"]))
        padded_vocab = int(model_cfg.get("vocab_size_padded", logical_vocab))
        mesh_model = int(mesh.shape["model"])
        if mesh_model > 1:
            eval_stats = make_vocab_parallel_eval_stats(
                mesh, logical_vocab, padded_vocab, token_chunk_size)
            vocab_argmax = make_vocab_parallel_argmax(
                mesh, logical_vocab, padded_vocab, token_chunk_size)
        else:
            eval_stats = make_unsharded_eval_stats(
                logical_vocab, token_chunk_size)
            vocab_argmax = make_unsharded_argmax(
                logical_vocab, token_chunk_size)
        self.sharded_fns = dict(base_sharded_fns or {})
        self.sharded_fns["vocab_eval_stats"] = eval_stats
        self.sharded_fns["vocab_argmax"] = vocab_argmax
        self._forward_kwargs = canonical._fixed_runtime_forward_kwargs(
            model, self.sharded_fns, self.runtime_state,
            compute_accuracy=True)
        self._runtime_delta: Dict[str, Dict[str, float]] = {}

        @jax.jit
        def score_step(params, input_ids, labels, attention_mask):
            result = model.apply(
                {"params": params},
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                deterministic=True,
                rngs={"dropout": jax.random.PRNGKey(0)},
                **self._forward_kwargs,
            )
            if "per_token_ce" not in result:
                raise RuntimeError(
                    "evaluation forward did not return per_token_ce")
            if "per_token_correct" not in result:
                raise RuntimeError(
                    "evaluation forward did not return per_token_correct")
            return result["per_token_ce"], result["per_token_correct"]

        @jax.jit
        def argmax_step(params, input_ids, attention_mask):
            result = model.apply(
                {"params": params},
                input_ids,
                labels=None,
                attention_mask=attention_mask,
                deterministic=True,
                rngs={"dropout": jax.random.PRNGKey(0)},
                **self._forward_kwargs,
            )
            if "argmax_token_ids" not in result:
                raise RuntimeError(
                    "evaluation generation forward did not return "
                    "argmax_token_ids")
            return result["argmax_token_ids"]

        self._score_step = score_step
        self._argmax_step = argmax_step

    def assert_request_consensus(self, digest: str) -> None:
        raw = bytes.fromhex(digest)
        local = np.frombuffer(raw, dtype=np.uint8)
        gathered = np.asarray(process_allgather(local)).reshape(
            self.process_count, -1)
        if any(bytes(row.tolist()).hex() != digest for row in gathered):
            values = [bytes(row.tolist()).hex() for row in gathered]
            raise RuntimeError(
                "request ordering differs across hosts: " + repr(values))

    def _global_array(self, local: np.ndarray, global_shape: Sequence[int]):
        return canonical.shard_to_mesh(local, self.data_sharding, tuple(global_shape))

    @staticmethod
    def _slice_start(index: Any) -> int:
        first = index[0]
        if isinstance(first, slice):
            return int(first.start or 0)
        return int(first)

    def _host_local_rows(self, value: Any) -> np.ndarray:
        """Materialize only this host's small score shards, deduping model replicas."""
        by_index = {}
        for shard in value.addressable_shards:
            key = tuple(
                (item.start, item.stop, item.step)
                if isinstance(item, slice) else int(item)
                for item in shard.index)
            if key not in by_index:
                by_index[key] = np.asarray(jax.device_get(shard.data))
        ordered = sorted(
            by_index.items(), key=lambda item: self._slice_start(
                tuple(slice(*part) if isinstance(part, tuple) else part
                      for part in item[0])))
        if not ordered:
            raise RuntimeError("JAX result has no addressable shards on this host")
        return np.concatenate([array for _, array in ordered], axis=0)

    def _gather_rows(self, value: Any) -> np.ndarray:
        local = self._host_local_rows(value)
        if local.shape[0] != self.local_batch_size:
            raise RuntimeError(
                "unexpected host-local result rows: "
                f"actual={local.shape[0]} expected={self.local_batch_size}")
        if self.process_count == 1:
            return local
        gathered = np.asarray(process_allgather(local))
        return gathered.reshape(
            self.global_batch_size, *local.shape[1:])

    def consume_runtime_delta(self) -> Dict[str, Dict[str, float]]:
        value = self._runtime_delta
        self._runtime_delta = {}
        return value

    def _record_batch_runtime(
        self,
        real_requests: Sequence[PreparedRequest],
        dummy_count: int,
        elapsed: float,
    ) -> None:
        counts = collections.Counter(request.task for request in real_requests)
        total = max(sum(counts.values()), 1)
        for task, count in counts.items():
            target = self._runtime_delta.setdefault(
                task, {"dummy_examples": 0, "wall_time": 0.0})
            target["wall_time"] += float(elapsed) * count / total
        if dummy_count and real_requests:
            # Dummy rows have no benchmark identity. Attribute them to the last
            # real task solely so per-task diagnostics sum to the actual batches.
            task = real_requests[-1].task
            target = self._runtime_delta.setdefault(
                task, {"dummy_examples": 0, "wall_time": 0.0})
            target["dummy_examples"] += int(dummy_count)

    def score_requests(
        self,
        requests: List[PreparedRequest],
        *,
        pad_token_id: int,
    ) -> List[ScoredRequest]:
        if not requests:
            return []
        output: List[Any] = [None] * len(requests)
        by_bucket: Dict[int, List[Any]] = collections.defaultdict(list)
        for position, request in enumerate(requests):
            by_bucket[int(request.bucket)].append((position, request))

        for bucket in sorted(by_bucket):
            entries = by_bucket[bucket]
            for offset in range(0, len(entries), self.global_batch_size):
                real = entries[offset:offset + self.global_batch_size]
                dummy_count = self.global_batch_size - len(real)
                input_rows = []
                label_rows = []
                attention_rows = []
                for _, request in real:
                    input_ids, labels, attention = request.materialize(
                        pad_token_id)
                    input_rows.append(input_ids)
                    label_rows.append(labels)
                    attention_rows.append(attention)
                for _ in range(dummy_count):
                    input_rows.append([int(pad_token_id)] * bucket)
                    label_rows.append([-100] * bucket)
                    attention_rows.append([0] * bucket)

                input_array = np.asarray(input_rows, dtype=np.int32)
                label_array = np.asarray(label_rows, dtype=np.int32)
                attention_array = np.asarray(attention_rows, dtype=np.int32)
                local_start = self.process_index * self.local_batch_size
                local_stop = local_start + self.local_batch_size
                global_shape = (self.global_batch_size, bucket)
                input_global = self._global_array(
                    input_array[local_start:local_stop], global_shape)
                label_global = self._global_array(
                    label_array[local_start:local_stop], global_shape)
                attention_global = self._global_array(
                    attention_array[local_start:local_stop], global_shape)

                start = time.monotonic()
                token_nll, token_correct = self._score_step(
                    self.params, input_global, label_global, attention_global)
                jax.block_until_ready((token_nll, token_correct))
                nll_rows = self._gather_rows(token_nll)
                correct_rows = self._gather_rows(token_correct)
                elapsed = time.monotonic() - start
                self._record_batch_runtime(
                    [request for _, request in real], dummy_count, elapsed)

                for row, (position, request) in enumerate(real):
                    target_mask = label_array[row, 1:] != -100
                    result = reduce_token_scores(
                        nll_rows[row].tolist(),
                        correct_rows[row].tolist(),
                        target_mask.tolist(),
                    )
                    if result.scored_tokens != request.scored_tokens:
                        raise RuntimeError(
                            "continuation target alignment mismatch after JAX "
                            f"forward: expected={request.scored_tokens} "
                            f"actual={result.scored_tokens}")
                    output[position] = result
        if any(item is None for item in output):
            raise RuntimeError("one or more requests were not scored")
        return output

    def _next_token(
        self,
        context_ids: Sequence[int],
        *,
        pad_token_id: int,
        length_buckets: Sequence[int],
    ) -> int:
        context = [int(x) for x in context_ids][-int(
            self.config["model"]["max_seq_len"]):]
        bucket = select_bucket(len(context), length_buckets)
        input_rows = []
        attention_rows = []
        base_input = context + [int(pad_token_id)] * (bucket - len(context))
        base_attention = [1] * len(context) + [0] * (bucket - len(context))
        input_rows.append(base_input)
        attention_rows.append(base_attention)
        for _ in range(self.global_batch_size - 1):
            input_rows.append([int(pad_token_id)] * bucket)
            attention_rows.append([0] * bucket)
        inputs = np.asarray(input_rows, dtype=np.int32)
        attention = np.asarray(attention_rows, dtype=np.int32)
        local_start = self.process_index * self.local_batch_size
        local_stop = local_start + self.local_batch_size
        shape = (self.global_batch_size, bucket)
        input_global = self._global_array(inputs[local_start:local_stop], shape)
        attention_global = self._global_array(
            attention[local_start:local_stop], shape)
        predictions = self._argmax_step(
            self.params, input_global, attention_global)
        jax.block_until_ready(predictions)
        rows = self._gather_rows(predictions)
        return int(rows[0, len(context) - 1])

    def generate_one(
        self,
        *,
        context_ids: Sequence[int],
        max_gen_toks: int,
        until: Sequence[str],
        tokenizer: Any,
        pad_token_id: int,
        length_buckets: Sequence[int],
    ) -> str:
        generated: List[int] = []
        running = [int(x) for x in context_ids]
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(int(max_gen_toks)):
            next_id = self._next_token(
                running,
                pad_token_id=pad_token_id,
                length_buckets=length_buckets,
            )
            if eos_id is not None and next_id == int(eos_id):
                break
            generated.append(next_id)
            running.append(next_id)
            text = tokenizer.decode(
                generated,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            stops = [text.find(marker) for marker in until if marker]
            stops = [position for position in stops if position >= 0]
            if stops:
                return text[:min(stops)]
        text = tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        stops = [text.find(marker) for marker in until if marker]
        stops = [position for position in stops if position >= 0]
        return text[:min(stops)] if stops else text
