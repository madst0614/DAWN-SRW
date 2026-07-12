#!/usr/bin/env python3
"""Prompt-task transfer frontend for the canonical DAWN JAX training engine.

This module owns downstream data, answer-only labels, candidate evaluation,
and experiment records. Model construction, sharding, optimizer/update policy,
train steps, runtime kwargs, and Orbax transfer restore are owned by
scripts.train_jax.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding, PartitionSpec as P

import scripts.train_jax as tj

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def is_host0() -> bool:
    return jax.process_index() == 0


def log(message: str) -> None:
    if is_host0():
        print(message, flush=True)


def join_path(base: str, name: str) -> str:
    return str(base).rstrip('/\\') + '/' + str(name).lstrip('/\\')


def load_yaml(path: str) -> Dict[str, Any]:
    with tj._open_file(path, 'r') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config root must be a mapping: {path}')
    return data


def write_text(path: str, text: str) -> None:
    if not str(path).startswith('gs://'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with tj._open_file(path, 'w') as handle:
        handle.write(text)


def append_text(path: str, text: str) -> None:
    previous = ''
    if tj._file_exists(path):
        with tj._open_file(path, 'r') as handle:
            previous = handle.read()
    write_text(path, previous + text)


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    candidates: Tuple[Tuple[str, int], ...]
    gold_index: int


def _gold(candidates: Sequence[Tuple[str, int]], label: int) -> int:
    for index, (_, candidate_label) in enumerate(candidates):
        if int(candidate_label) == int(label):
            return index
    raise ValueError(f'label={label} not present in candidates={candidates}')


def build_prompt(task: str, example: Dict[str, Any]) -> PromptExample:
    task = task.lower()
    if task == 'sst2':
        candidates = ((" negative", 0), (" positive", 1))
        prompt = f"Sentence: {example['sentence']}\nSentiment:"
    elif task == 'rte':
        candidates = ((" yes", 0), (" no", 1))
        prompt = (
            f"Premise: {example['sentence1']}\n"
            f"Hypothesis: {example['sentence2']}\n"
            "Does the premise entail the hypothesis? Answer:")
    elif task == 'mnli':
        candidates = ((" yes", 0), (" maybe", 1), (" no", 2))
        prompt = (
            f"Premise: {example['premise']}\n"
            f"Hypothesis: {example['hypothesis']}\n"
            "Does the premise entail the hypothesis? Answer:")
    elif task == 'qqp':
        candidates = ((" no", 0), (" yes", 1))
        prompt = (
            f"Question 1: {example['question1']}\n"
            f"Question 2: {example['question2']}\n"
            "Are these questions duplicates? Answer:")
    elif task == 'mrpc':
        candidates = ((" no", 0), (" yes", 1))
        prompt = (
            f"Sentence 1: {example['sentence1']}\n"
            f"Sentence 2: {example['sentence2']}\n"
            "Are these sentences paraphrases? Answer:")
    elif task == 'boolq':
        candidates = ((" no", 0), (" yes", 1))
        prompt = (
            f"Passage: {example['passage']}\n"
            f"Question: {example['question']}\nAnswer:")
    elif task == 'wic':
        candidates = ((" no", 0), (" yes", 1))
        prompt = (
            f"Word: {example['word']}\n"
            f"Sentence 1: {example['sentence1']}\n"
            f"Sentence 2: {example['sentence2']}\n"
            "Does the word have the same meaning in both sentences? Answer:")
    else:
        raise ValueError(f'Unsupported task: {task}')
    label = int(bool(example['label'])) if task in ('boolq', 'wic') else int(
        example['label'])
    return PromptExample(prompt, candidates, _gold(candidates, label))


def hf_spec(task: str) -> Tuple[str, Optional[str], str, str]:
    task = task.lower()
    if task in ('sst2', 'rte', 'mnli', 'qqp', 'mrpc'):
        return (
            'glue', task, 'train',
            'validation_matched' if task == 'mnli' else 'validation')
    if task in ('boolq', 'wic'):
        return 'super_glue', task, 'train', 'validation'
    raise ValueError(f'No Hugging Face dataset mapping for task={task}')


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with tj._open_file(path, 'r') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tsv(path: str) -> List[Dict[str, Any]]:
    with tj._open_file(path, 'r') as handle:
        return list(csv.DictReader(handle.read().splitlines(), delimiter='\t'))


def load_raw_splits(data_cfg: Dict[str, Any], task: str):
    source = str(data_cfg.get('source', 'hf')).lower()
    if source == 'jsonl':
        return (
            load_jsonl(data_cfg['train_path']),
            load_jsonl(data_cfg['eval_path']))
    if source == 'tsv':
        return (
            load_tsv(data_cfg['train_path']),
            load_tsv(data_cfg['eval_path']))
    if source != 'hf':
        raise ValueError(f'Unknown downstream source: {source}')
    if load_dataset is None:
        raise RuntimeError(
            'datasets is not installed; install it or use jsonl/tsv input')
    default_name, default_subset, train_split, eval_split = hf_spec(task)
    name = data_cfg.get('hf_name', default_name)
    subset = data_cfg.get('hf_config', default_subset)
    train_split = data_cfg.get('train_split', train_split)
    eval_split = data_cfg.get('eval_split', eval_split)
    candidates = [name]
    if 'hf_name' not in data_cfg:
        aliases = {'glue': 'nyu-mll/glue', 'super_glue': 'aps/super_glue'}
        if name in aliases:
            candidates.insert(0, aliases[name])
    errors = []
    for candidate in candidates:
        try:
            log(
                f'[data] load_dataset({candidate!r}, {subset!r}) '
                f'train={train_split} eval={eval_split}')
            return (
                load_dataset(candidate, subset, split=train_split),
                load_dataset(candidate, subset, split=eval_split))
        except Exception as exc:
            errors.append(f'{candidate}: {type(exc).__name__}: {exc}')
    raise RuntimeError(
        f'Failed to load dataset for task={task}:\n' + '\n'.join(errors))


def tokenizer_name(tokenizer_config: Any) -> str:
    if isinstance(tokenizer_config, str) and tokenizer_config.strip():
        return tokenizer_config.strip()
    if isinstance(tokenizer_config, dict):
        for key in ('name', 'name_or_path', 'pretrained_model_name_or_path'):
            value = tokenizer_config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise RuntimeError(
        'Tokenizer config must provide a name; silent tokenizer fallback is disabled')


def validate_tokenizer(tokenizer, model_cfg: Dict[str, Any],
                       downstream_cfg: Dict[str, Any]) -> int:
    logical_vocab = model_cfg.get('logical_vocab_size')
    if logical_vocab is None:
        logical_vocab = model_cfg.get('vocab_size')
    if logical_vocab is None:
        raise RuntimeError(
            'Checkpoint model config is missing logical_vocab_size/vocab_size')
    logical_vocab = int(logical_vocab)
    if len(tokenizer) != logical_vocab:
        raise ValueError(
            f'Tokenizer vocab size {len(tokenizer)} does not match '
            f'model logical_vocab_size={logical_vocab}')

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        configured_pad = downstream_cfg.get('pad_token_id')
        if configured_pad is not None:
            pad_id = int(configured_pad)
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = int(tokenizer.eos_token_id)
        else:
            raise RuntimeError(
                'Tokenizer has no pad token; configure an existing token id '
                'or use its existing eos token')
    for name, token_id in (
            ('pad', pad_id),
            ('eos', tokenizer.eos_token_id),
            ('unk', tokenizer.unk_token_id)):
        if token_id is not None and not (0 <= int(token_id) < logical_vocab):
            raise ValueError(
                f'{name}_token_id={token_id} is outside logical vocab '
                f'[0, {logical_vocab})')
    return int(pad_id)


def encode_prompt_answer(tokenizer, prompt: str, answer: str,
                         max_seq_len: int, *, add_eos=False):
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False)['input_ids']
    answer_ids = tokenizer(
        answer, add_special_tokens=False)['input_ids']
    if add_eos:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('downstream.add_eos requires an existing eos token')
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    if not answer_ids:
        return None
    if len(prompt_ids) + len(answer_ids) > max_seq_len:
        keep_prompt = max_seq_len - len(answer_ids)
        if keep_prompt <= 0:
            return None
        prompt_ids = prompt_ids[-keep_prompt:]
    input_ids = prompt_ids + answer_ids
    return {
        'input_ids': np.asarray(input_ids, dtype=np.int32),
        'labels': np.asarray(
            [-100] * len(prompt_ids) + answer_ids, dtype=np.int32),
    }


def _valid_labeled_example(example: Dict[str, Any]) -> bool:
    if 'label' not in example:
        return True
    try:
        return int(example['label']) >= 0
    except (TypeError, ValueError):
        return True


def make_train_rows(raw, task: str, tokenizer, max_seq_len: int,
                    max_samples: Optional[int], seed: int, *, add_eos=False):
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    if max_samples is not None:
        indices = indices[:int(max_samples)]
    rows = []
    for index in indices:
        example = raw[index]
        if not _valid_labeled_example(example):
            continue
        prompt = build_prompt(task, example)
        answer = prompt.candidates[prompt.gold_index][0]
        row = encode_prompt_answer(
            tokenizer, prompt.prompt, answer, max_seq_len,
            add_eos=add_eos)
        if row is not None:
            row['example_id'] = int(index)
            rows.append(row)
    return rows


def make_eval_rows(raw, task: str, tokenizer, max_seq_len: int,
                   max_samples: Optional[int], *, add_eos=False):
    limit = len(raw) if max_samples is None else min(
        int(max_samples), len(raw))
    rows = []
    for index in range(limit):
        example = raw[index]
        if not _valid_labeled_example(example):
            continue
        prompt = build_prompt(task, example)
        candidates = []
        for answer, _ in prompt.candidates:
            row = encode_prompt_answer(
                tokenizer, prompt.prompt, answer, max_seq_len,
                add_eos=add_eos)
            if row is None:
                candidates = []
                break
            candidates.append(row)
        if candidates:
            rows.append({
                'example_id': int(index),
                'candidates': candidates,
                'gold_index': int(prompt.gold_index),
            })
    return rows


def pad_fixed(rows: Sequence[np.ndarray], max_seq_len: int,
              pad_value: int) -> np.ndarray:
    output = np.full(
        (len(rows), max_seq_len), pad_value, dtype=np.int32)
    for index, row in enumerate(rows):
        output[index, :len(row)] = row[:max_seq_len]
    return output


def build_train_batch(rows, indices, max_seq_len, pad_token_id):
    input_rows = []
    label_rows = []
    attention_rows = []
    example_valid = []
    for index in indices:
        if index is None:
            input_rows.append(np.asarray([], dtype=np.int32))
            label_rows.append(np.asarray([], dtype=np.int32))
            attention_rows.append(np.asarray([], dtype=np.int32))
            example_valid.append(False)
        else:
            row = rows[index]
            input_rows.append(row['input_ids'])
            label_rows.append(row['labels'])
            attention_rows.append(np.ones_like(row['input_ids']))
            example_valid.append(True)
    return (
        pad_fixed(input_rows, max_seq_len, pad_token_id),
        pad_fixed(label_rows, max_seq_len, -100),
        pad_fixed(attention_rows, max_seq_len, 0),
        np.asarray(example_valid, dtype=np.bool_),
    )


def build_candidate_batch(rows, max_seq_len, pad_token_id):
    input_rows = [row['input_ids'] for row in rows]
    label_rows = [row['labels'] for row in rows]
    attention_rows = [np.ones_like(row['input_ids']) for row in rows]
    return (
        pad_fixed(input_rows, max_seq_len, pad_token_id),
        pad_fixed(label_rows, max_seq_len, -100),
        pad_fixed(attention_rows, max_seq_len, 0),
    )


def local_slice(global_array: np.ndarray) -> np.ndarray:
    process_count = jax.process_count()
    if global_array.shape[0] % process_count != 0:
        raise ValueError(
            f'Global batch {global_array.shape[0]} is not divisible by '
            f'process_count={process_count}')
    per_process = global_array.shape[0] // process_count
    start = jax.process_index() * per_process
    return global_array[start:start + per_process]


def shard_batch(array, sharding, global_shape):
    return tj.shard_to_mesh(local_slice(array), sharding, global_shape)


def flatten_eval(eval_rows):
    flat = []
    metadata = []
    for example_index, row in enumerate(eval_rows):
        for candidate_index, candidate in enumerate(row['candidates']):
            flat.append(candidate)
            metadata.append((
                example_index, candidate_index, row['gold_index']))
    return flat, metadata


def evaluate(params, score_step, eval_rows, batch_size, max_seq_len,
             pad_token_id, data_sharding):
    flat, metadata = flatten_eval(eval_rows)
    n_batches = math.ceil(len(flat) / batch_size)
    scores_by_example = {}
    gold_by_example = {}
    for batch_index in range(n_batches):
        start = batch_index * batch_size
        valid_rows = flat[start:min(start + batch_size, len(flat))]
        valid_count = len(valid_rows)
        dummy = {
            'input_ids': np.asarray([], dtype=np.int32),
            'labels': np.asarray([], dtype=np.int32),
        }
        padded_rows = valid_rows + [dummy] * (batch_size - valid_count)
        global_ids, global_labels, global_attention = build_candidate_batch(
            padded_rows, max_seq_len, pad_token_id)
        global_shape = (batch_size, max_seq_len)
        ids = shard_batch(global_ids, data_sharding, global_shape)
        labels = shard_batch(global_labels, data_sharding, global_shape)
        attention = shard_batch(global_attention, data_sharding, global_shape)
        batch_nll = score_step(params, ids, labels, attention)
        gathered = np.asarray(
            process_allgather(batch_nll, tiled=True)).reshape(-1)[:valid_count]
        if is_host0():
            for offset, nll in enumerate(gathered):
                example_index, candidate_index, gold = metadata[start + offset]
                scores_by_example.setdefault(example_index, []).append(
                    (candidate_index, float(nll)))
                gold_by_example[example_index] = gold
    if is_host0():
        correct = sum(
            int(min(scores, key=lambda item: item[1])[0]
                == gold_by_example[example_index])
            for example_index, scores in scores_by_example.items())
        total = len(scores_by_example)
        local_result = np.asarray(
            [correct / max(total, 1), total], dtype=np.float64)
    else:
        local_result = np.zeros((2,), dtype=np.float64)
    if jax.process_count() > 1:
        results = np.asarray(process_allgather(local_result)).reshape(-1, 2)
        local_result = results[0]
    return {'accuracy': float(local_result[0]), 'total': int(local_result[1])}


def create_run_dir(output_root: str, run_name: str, task: str) -> str:
    if is_host0():
        suffix = time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
        value = join_path(
            output_root,
            f"run_{run_name}_{task}_{suffix}")
    else:
        value = ''
    return tj._broadcast_str_from_host0(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--init-from', '--transfer-from', dest='init_from')
    args = parser.parse_args()

    requested_cfg = load_yaml(args.config)
    requested_downstream = requested_cfg.get('downstream', {})
    if (requested_cfg.get('resume_from') is not None
            or requested_downstream.get('resume_from') is not None):
        raise ValueError(
            'Downstream resume has been removed; use a committed pretraining '
            'checkpoint as init_from for a fresh transfer phase')
    task = requested_downstream.get('task') or requested_cfg.get('task')
    if not task:
        raise ValueError('Config must set downstream.task')
    transfer_from = (
        args.init_from
        or requested_cfg.get('init_from')
        or requested_downstream.get('init_from'))
    if not transfer_from:
        raise ValueError(
            'Downstream is transfer-only and requires --init-from or config init_from')

    tj.initialize_distributed_runtime(requested_cfg)
    source = tj.resolve_transfer_checkpoint(transfer_from)
    cfg = tj.build_effective_transfer_config(source, requested_cfg)
    downstream_cfg = cfg.get('downstream', {})
    training_cfg = cfg.get('training', {})
    model_cfg = cfg['model']
    task = downstream_cfg.get('task') or cfg.get('task')

    seed = int(cfg.get('seed', 1))
    random.seed(seed)
    np.random.seed(seed)
    if AutoTokenizer is None:
        raise RuntimeError('transformers is not installed')
    tok_name = tokenizer_name(cfg.get('tokenizer'))
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    pad_token_id = validate_tokenizer(tokenizer, model_cfg, downstream_cfg)

    source_max_seq_len = int(model_cfg['max_seq_len'])
    max_seq_len = int(downstream_cfg.get(
        'max_seq_len', source_max_seq_len))
    if max_seq_len > source_max_seq_len:
        raise ValueError(
            f'downstream.max_seq_len={max_seq_len} exceeds source model '
            f'max_seq_len={source_max_seq_len}')
    if max_seq_len <= 1:
        raise ValueError('downstream.max_seq_len must be > 1')

    raw_train, raw_eval = load_raw_splits(downstream_cfg, task)
    add_eos = bool(downstream_cfg.get('add_eos', False))
    train_rows = make_train_rows(
        raw_train, task, tokenizer, max_seq_len,
        downstream_cfg.get('max_train_samples'), seed, add_eos=add_eos)
    eval_rows = make_eval_rows(
        raw_eval, task, tokenizer, max_seq_len,
        downstream_cfg.get('max_eval_samples'), add_eos=add_eos)
    if not train_rows or not eval_rows:
        raise RuntimeError(
            f'Empty downstream rows: train={len(train_rows)} '
            f'eval={len(eval_rows)}')

    total_devices = jax.device_count()
    batch_size = int(training_cfg.get('batch_size', 64))
    eval_batch_size = int(training_cfg.get('eval_batch_size', batch_size))
    for name, value in (
            ('batch_size', batch_size),
            ('eval_batch_size', eval_batch_size)):
        if value <= 0 or value % total_devices != 0:
            raise ValueError(
                f'training.{name}={value} must be positive and divisible by '
                f'device_count={total_devices}')
    mesh_model = int(training_cfg.get('mesh_model', 1))
    mesh_data = int(training_cfg.get('mesh_data', 0)) or (
        total_devices // mesh_model)
    if mesh_data * mesh_model != total_devices:
        raise ValueError(
            f'mesh_data({mesh_data}) * mesh_model({mesh_model}) must equal '
            f'device_count={total_devices}')

    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    num_epochs = int(training_cfg.get('num_epochs', 3))
    max_steps = training_cfg.get('max_steps')
    total_steps = int(max_steps) if max_steps is not None else (
        num_epochs * steps_per_epoch)
    if total_steps <= 0:
        raise ValueError('Downstream total_steps must be > 0')
    eval_interval = int(training_cfg.get(
        'eval_interval', training_cfg.get('val_interval', 200)))
    log_interval = int(training_cfg.get('log_interval', 20))

    output_root = requested_cfg.get('checkpoint_dir')
    if not output_root:
        raise ValueError('Config must set checkpoint_dir as experiment output root')
    run_name = str(requested_cfg.get('run_name') or model_cfg['model_version'])
    run_dir = create_run_dir(output_root, run_name, task)
    train_log_path = join_path(run_dir, 'train.log')
    metrics_path = join_path(run_dir, 'metrics.jsonl')
    effective_config_path = join_path(run_dir, 'effective_config.yaml')

    def record(message: str) -> None:
        if is_host0():
            print(message, flush=True)
            append_text(train_log_path, message + '\n')

    def metric_record(payload: Dict[str, Any]) -> None:
        if is_host0():
            append_text(
                metrics_path,
                json.dumps(payload, sort_keys=True) + '\n')

    if is_host0():
        write_text(
            effective_config_path,
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    model = tj.build_model_from_config(cfg)
    init_key = jax.random.PRNGKey(seed)
    init_len = min(max_seq_len, 32)
    dummy_ids = jnp.ones((1, init_len), dtype=jnp.int32)
    dummy_labels = jnp.ones_like(dummy_ids)
    variables = model.init(
        {'params': init_key, 'dropout': init_key},
        dummy_ids,
        labels=dummy_labels,
        attention_mask=jnp.ones_like(dummy_ids),
        deterministic=True)
    params = variables['params']

    mesh = tj.create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    param_shardings = tj.get_param_shardings(
        params,
        mesh,
        model_cfg['model_version'],
        vocab_size_padded=model_cfg.get('vocab_size_padded'))
    params = tj.shard_params_to_mesh(params, param_shardings)
    sharded_train = tj.build_canonical_sharded_fns(cfg, mesh)
    sharded_eval = tj.build_canonical_sharded_fns(
        cfg, mesh, for_eval=True)

    grad_accum = int(training_cfg.get('gradient_accumulation_steps', 1))
    total_optimizer_steps = math.ceil(total_steps / max(grad_accum, 1))
    optimizer_bundle = tj.create_canonical_optimizer(
        params,
        training_cfg,
        total_optimizer_steps,
        log_groups=is_host0())
    initialized = tj.initialize_training_state(
        'transfer',
        params,
        optimizer_bundle.optimizer,
        mesh,
        jax.random.PRNGKey(seed + 1000),
        source_checkpoint=source)
    params = initialized.params
    opt_state = initialized.opt_state
    rng = initialized.rng
    phase_step = initialized.phase_step
    if phase_step != 0:
        raise AssertionError('Transfer phase_step must start at 0')

    runtime_state = dict(source.runtime_state)
    runtime_state['ce_token_chunk_size'] = int(training_cfg.get(
        'ce_token_chunk_size', 32768))
    train_step = tj.create_canonical_train_step(
        model,
        optimizer_bundle.optimizer,
        cfg,
        sharded_train,
        mesh,
        total_steps,
        runtime_state=runtime_state)
    score_step = tj.create_candidate_score_step(
        model, sharded_eval, runtime_state)
    previous_op_key_snapshot = {
        'attn_qk_op_key': jnp.zeros((), dtype=jnp.float32),
        'attn_v_op_key': jnp.zeros((), dtype=jnp.float32),
        'rst_op_key': jnp.zeros((), dtype=jnp.float32),
    }

    header = {
        'phase_type': 'transfer',
        'source_checkpoint': source.checkpoint_path,
        'source_checkpoint_step': source.step,
        'model_config_source': 'checkpoint.full_config.model',
        'params_source': 'source_checkpoint.params',
        'optimizer_policy': 'fresh',
        'phase_step': 0,
        'runtime_policy': 'source_final_constant',
        'checkpoint_write': 'disabled',
        'checkpoint_resume': 'disabled',
        'task': task,
        'train_rows': len(train_rows),
        'eval_rows': len(eval_rows),
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'max_seq_len': max_seq_len,
        'tokenizer': tok_name,
        'tau_lr_mult': runtime_state['tau_lr_mult'],
        'soft_gate_T_qk': runtime_state['soft_gate_T_qk'],
        'soft_gate_T_v': runtime_state['soft_gate_T_v'],
        'soft_gate_T_rst': runtime_state['soft_gate_T_rst'],
        'boundary_power': runtime_state['soft_gate_boundary_power'],
    }
    record(' '.join(f'{key}={value}' for key, value in header.items()))
    metric_record({'type': 'run_start', **header, 'run_dir': run_dir})

    initial_eval = evaluate(
        params, score_step, eval_rows, eval_batch_size, max_seq_len,
        pad_token_id, data_sharding)
    best_seen_acc = initial_eval['accuracy']
    final_eval = initial_eval
    record(
        f"[eval/initial] phase_step=0 acc={initial_eval['accuracy']:.6f} "
        f"total={initial_eval['total']} best_seen_acc={best_seen_acc:.6f}")
    metric_record({
        'type': 'eval_initial', 'phase_step': 0,
        'accuracy': initial_eval['accuracy'],
        'total': initial_eval['total'],
        'best_seen_acc': best_seen_acc,
    })

    start_time = time.time()
    while phase_step < total_steps:
        epoch = phase_step // steps_per_epoch
        step_in_epoch = phase_step % steps_per_epoch
        order = list(range(len(train_rows)))
        random.Random(seed + epoch).shuffle(order)
        start = step_in_epoch * batch_size
        indices = order[start:min(start + batch_size, len(order))]
        indices.extend([None] * (batch_size - len(indices)))
        global_ids, global_labels, global_attention, example_valid = (
            build_train_batch(
                train_rows, indices, max_seq_len, pad_token_id))
        if int(example_valid.sum()) != min(
                batch_size, len(train_rows) - start):
            raise AssertionError('Invalid real/dummy example accounting')
        global_shape = (batch_size, max_seq_len)
        ids = shard_batch(global_ids, data_sharding, global_shape)
        labels = shard_batch(global_labels, data_sharding, global_shape)
        attention = shard_batch(
            global_attention, data_sharding, global_shape)
        rng, dropout_key = jax.random.split(rng)
        dropout_key = jax.random.fold_in(
            dropout_key, jax.process_index())
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            ids,
            labels,
            attention,
            dropout_key,
            previous_op_key_snapshot,
            jnp.asarray(phase_step, dtype=jnp.int32))
        phase_step += 1
        tj.require_finite_metrics(
            {'total_loss': metrics['total_loss']}, phase_step=phase_step)

        if phase_step % log_interval == 0 or phase_step == 1:
            reduced = tj.reduce_scalar_metrics({
                'loss': metrics['total_loss'],
                'valid_count': metrics['valid_count'],
            })
            loss = reduced['loss']
            valid_count = int(round(reduced['valid_count']))
            elapsed = time.time() - start_time
            record(
                f'[train] phase_step={phase_step}/{total_steps} '
                f'epoch={epoch} loss={loss:.6f} valid_tokens={valid_count} '
                f'real_examples={int(example_valid.sum())} '
                f'elapsed_sec={elapsed:.1f}')
            metric_record({
                'type': 'train', 'phase_step': phase_step, 'epoch': epoch,
                'loss': loss, 'valid_tokens': valid_count,
                'real_examples': int(example_valid.sum()),
                'elapsed_sec': elapsed,
            })

        if eval_interval > 0 and phase_step % eval_interval == 0:
            final_eval = evaluate(
                params, score_step, eval_rows, eval_batch_size, max_seq_len,
                pad_token_id, data_sharding)
            best_seen_acc = max(best_seen_acc, final_eval['accuracy'])
            record(
                f"[eval/interval] phase_step={phase_step} "
                f"acc={final_eval['accuracy']:.6f} "
                f"total={final_eval['total']} "
                f"best_seen_acc={best_seen_acc:.6f}")
            metric_record({
                'type': 'eval_interval', 'phase_step': phase_step,
                'accuracy': final_eval['accuracy'],
                'total': final_eval['total'],
                'best_seen_acc': best_seen_acc,
            })

    final_eval = evaluate(
        params, score_step, eval_rows, eval_batch_size, max_seq_len,
        pad_token_id, data_sharding)
    best_seen_acc = max(best_seen_acc, final_eval['accuracy'])
    record(
        f"[eval/final] phase_step={phase_step} "
        f"final_acc={final_eval['accuracy']:.6f} "
        f"best_seen_acc={best_seen_acc:.6f} total={final_eval['total']}")
    metric_record({
        'type': 'eval_final', 'phase_step': phase_step,
        'final_acc': final_eval['accuracy'],
        'best_seen_acc': best_seen_acc,
        'total': final_eval['total'],
        'checkpoint_write': 'disabled',
    })

    if jax.process_count() > 1:
        completed = np.asarray([phase_step], dtype=np.int64)
        all_completed = np.asarray(process_allgather(completed)).reshape(-1)
        if not np.all(all_completed == phase_step):
            raise RuntimeError(
                f'End-of-task phase_step mismatch: {all_completed.tolist()}')


if __name__ == '__main__':
    main()
