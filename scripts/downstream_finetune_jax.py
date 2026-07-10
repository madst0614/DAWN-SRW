#!/usr/bin/env python3
"""Downstream prompt-style fine-tuning/eval for DAWN/Transformer causal LMs on TPU pods.

Does not modify or depend on C4 loaders. Uses task examples -> prompt + answer-token labels.

Modes:
  --init-from  : start each task from pretrained params only; optimizer fresh; step=0
  --resume-from: optionally read an existing downstream checkpoint; no checkpoint is written

Path resolution for both init/resume:
  pretrain init-from -> .flax as above, or latest committed Orbax run/checkpoints step
  downstream resume-from -> latest committed Orbax best_checkpoints/checkpoints step, read-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import inspect
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

# Reuse the verified model registry, GCS I/O, sharding and checkpoint helpers.
import scripts.train_jax as tj

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# -----------------------------
# Small utilities
# -----------------------------

def is_host0() -> bool:
    return jax.process_index() == 0


def log(msg: str):
    if is_host0():
        print(msg, flush=True)


def broadcast_str_from_host0(value: Optional[str], max_len: int = 1024) -> str:
    """Broadcast a short UTF-8 string from process 0 to every host."""
    if jax.process_count() <= 1:
        return '' if value is None else str(value)
    if is_host0():
        encoded = ('' if value is None else str(value)).encode('utf-8')
        if len(encoded) > max_len:
            raise ValueError(f'Broadcast string too long: {len(encoded)} > {max_len}')
        buf = np.zeros((max_len,), dtype=np.uint8)
        buf[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    else:
        buf = np.zeros((max_len,), dtype=np.uint8)
    gathered = np.asarray(process_allgather(buf))
    if gathered.ndim == 1 and gathered.size >= max_len * jax.process_count():
        host0_buf = gathered[:max_len]
    else:
        host0_buf = gathered[0]
    return bytes(np.asarray(host0_buf, dtype=np.uint8)).rstrip(b'\x00').decode('utf-8')


def join_path(base: str, name: str) -> str:
    return base.rstrip('/') + '/' + name


def path_basename(path: str) -> str:
    return path.rstrip('/').rsplit('/', 1)[-1]


def step_num(path: str) -> int:
    m = re.search(r'checkpoint_step(\d+)\.flax', path_basename(path))
    return int(m.group(1)) if m else -1


@dataclass(frozen=True)
class CheckpointRef:
    kind: str
    path: str
    run_folder: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    step: Optional[int] = None


def resolve_flax_checkpoint_path(path: Optional[str]) -> Optional[str]:
    """Resolve a legacy Flax file/dir checkpoint."""
    if not path:
        return None
    p = str(path)
    if p.endswith('.flax'):
        if not tj._file_exists(p):
            raise FileNotFoundError(f'Checkpoint file not found: {p}')
        return p

    best = join_path(p, 'best_model.flax')
    if tj._file_exists(best):
        return best

    step_files = tj._list_files(p, 'checkpoint_step*.flax')
    if step_files:
        step_files = sorted(step_files, key=step_num)
        return step_files[-1]

    any_files = tj._list_files(p, '*.flax')
    if any_files:
        return any_files[-1]

    raise FileNotFoundError(f'No .flax checkpoint found in: {p}')


def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    return resolve_flax_checkpoint_path(path)


def resolve_init_checkpoint_ref(path: Optional[str]) -> Optional[CheckpointRef]:
    if not path:
        return None
    try:
        flax_path = resolve_flax_checkpoint_path(path)
        if flax_path:
            return CheckpointRef(kind='flax', path=flax_path)
    except FileNotFoundError:
        if str(path).endswith('.flax'):
            raise
        pass

    run_folder, step, found = tj._resolve_orbax_resume_from(path)
    if found and step is not None:
        checkpoint_dir = tj._join_path(run_folder, 'checkpoints')
        return CheckpointRef(
            kind='orbax',
            path=tj._join_path(checkpoint_dir, str(int(step))),
            run_folder=run_folder,
            checkpoint_dir=checkpoint_dir,
            step=int(step),
        )
    raise FileNotFoundError(
        f'No downstream .flax checkpoint or committed Orbax checkpoint found in: {path}')


def resolve_downstream_orbax_resume_ref(path: Optional[str]) -> Optional[CheckpointRef]:
    if not path:
        return None
    p = str(path).rstrip('/\\')
    candidates = []
    name = path_basename(p)
    if name in ('best_checkpoints', 'checkpoints'):
        candidates.append(p)
    else:
        candidates.extend([join_path(p, 'best_checkpoints'), join_path(p, 'checkpoints')])
    for checkpoint_dir in candidates:
        run_folder = checkpoint_dir.rsplit('/', 1)[0]
        step = None
        try:
            paths = tj._list_files(checkpoint_dir, '*')
            steps = [
                tj._orbax_step_from_path_name(path_basename(x))
                for x in paths
            ]
            steps = sorted(
                s for s in steps
                if s is not None and tj._orbax_step_is_committed(join_path(checkpoint_dir, str(int(s))))
            )
            if steps:
                step = int(steps[-1])
        except Exception:
            step = None
        if step is not None:
            return CheckpointRef(
                kind='orbax',
                path=join_path(checkpoint_dir, str(int(step))),
                run_folder=run_folder,
                checkpoint_dir=checkpoint_dir,
                step=int(step),
            )
    return None


def load_orbax_checkpoint_model_config(ref: Optional[CheckpointRef]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    if ref is None or ref.kind != 'orbax' or ref.checkpoint_dir is None or ref.step is None:
        return None, None, 'current_yaml'
    metadata = tj._restore_orbax_metadata(ref.checkpoint_dir, int(ref.step))
    full_config = metadata.get('full_config')
    if not isinstance(full_config, dict):
        full_config = {}
    model_config = (
        full_config.get('model')
        if isinstance(full_config.get('model'), dict)
        else metadata.get('model_config')
    )
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(
            f'Orbax checkpoint is missing model config metadata: {ref.path}')
    source = (
        'checkpoint full_config.model'
        if isinstance(full_config.get('model'), dict)
        else 'checkpoint metadata.model_config')
    return deepcopy(model_config), deepcopy(full_config), source


CHECKPOINT_OPTIONAL_RUNTIME_TRAINING_KEYS = (
    'opspace_gate_den_power',
)

CHECKPOINT_STRICT_RUNTIME_MODEL_VERSIONS = (
    tj.V4166_MODEL_VERSION,
    tj.V4168_MODEL_VERSION,
)

CHECKPOINT_POOL_FINAL_KEYS = {
    'qk': 'soft_gate_t_qk_final',
    'v': 'soft_gate_t_v_final',
    'rst': 'soft_gate_t_rst_final',
}

CHECKPOINT_REQUIRED_RUNTIME_TRAINING_KEYS = (
    'soft_gate_t_final',
    'soft_gate_boundary_power_final',
    'admission_den_power',
    'admission_den_grad_scale',
    'soft_gate_effective_active_eps',
    'tau_lr_mult',
)


def _require_checkpoint_training_value(training: Dict[str, Any],
                                       key: str,
                                       ref_path: str):
    if key not in training:
        raise ValueError(
            f'Checkpoint {ref_path} is missing full_config.training.{key}; '
            'downstream refuses to use YAML/code fallback for calibrated '
            'runtime values.')
    return deepcopy(training[key])


def _require_positive_float(value, key: str, ref_path: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(
            f'Checkpoint {ref_path} has invalid full_config.training.{key}={value!r}; '
            'expected a finite positive value.')
    return out


def _require_nonnegative_float(value, key: str, ref_path: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(
            f'Checkpoint {ref_path} has invalid full_config.training.{key}={value!r}; '
            'expected a finite nonnegative value.')
    return out


def checkpoint_runtime_training_config(full_config: Optional[Dict[str, Any]],
                                       model_config: Optional[Dict[str, Any]],
                                       ref_path: str) -> Dict[str, Any]:
    if not isinstance(full_config, dict):
        full_config = {}
    training = full_config.get('training', {})
    version = str((model_config or {}).get('model_version', ''))
    strict_runtime = version in CHECKPOINT_STRICT_RUNTIME_MODEL_VERSIONS
    if not isinstance(training, dict):
        if strict_runtime:
            raise ValueError(
                f'Checkpoint {ref_path} is missing full_config.training; '
                'cannot restore calibrated downstream runtime config.')
        return {}
    if not strict_runtime:
        return {
            key: deepcopy(training[key])
            for key in CHECKPOINT_OPTIONAL_RUNTIME_TRAINING_KEYS
            if key in training
        }

    runtime: Dict[str, Any] = {}
    for pool, source_key in CHECKPOINT_POOL_FINAL_KEYS.items():
        value = _require_checkpoint_training_value(training, source_key, ref_path)
        runtime[source_key] = deepcopy(value)
        runtime[f'soft_gate_T_{pool}'] = _require_positive_float(
            value, source_key, ref_path)
    runtime['soft_gate_temperature'] = runtime['soft_gate_T_qk']

    for key in CHECKPOINT_REQUIRED_RUNTIME_TRAINING_KEYS:
        runtime[key] = _require_checkpoint_training_value(training, key, ref_path)

    boundary_final = _require_positive_float(
        runtime['soft_gate_boundary_power_final'],
        'soft_gate_boundary_power_final',
        ref_path)
    runtime['soft_gate_boundary_power_final'] = boundary_final
    runtime['soft_gate_boundary_power'] = boundary_final

    runtime['admission_den_power'] = _require_nonnegative_float(
        runtime['admission_den_power'], 'admission_den_power', ref_path)
    runtime['admission_den_grad_scale'] = _require_nonnegative_float(
        runtime['admission_den_grad_scale'], 'admission_den_grad_scale', ref_path)
    runtime['soft_gate_effective_active_eps'] = _require_positive_float(
        runtime['soft_gate_effective_active_eps'],
        'soft_gate_effective_active_eps',
        ref_path)
    runtime['tau_lr_mult'] = _require_nonnegative_float(
        runtime['tau_lr_mult'], 'tau_lr_mult', ref_path)

    for key in CHECKPOINT_OPTIONAL_RUNTIME_TRAINING_KEYS:
        if key in training:
            runtime[key] = deepcopy(training[key])
    for key, value in training.items():
        if (key.startswith('selection_calibration')
                or key.startswith('soft_gate_t_')
                or key.startswith('soft_gate_boundary_power_')):
            runtime.setdefault(key, deepcopy(value))
    return runtime


def apply_init_checkpoint_model_config(cfg: Dict[str, Any],
                                       init_ref: Optional[CheckpointRef]) -> str:
    model_config, full_config, source = load_orbax_checkpoint_model_config(init_ref)
    if model_config is None:
        if not isinstance(cfg.get('model'), dict) or not cfg.get('model'):
            raise ValueError(
                'Downstream config has no model section and init_from is not '
                'an Orbax checkpoint with model metadata.')
        return source
    cfg['model'] = model_config
    runtime_config = checkpoint_runtime_training_config(
        full_config, model_config, init_ref.path if init_ref is not None else source)
    if runtime_config:
        training = cfg.setdefault('training', {})
        for key, value in runtime_config.items():
            training[key] = value
    if 'tokenizer' not in cfg and isinstance(full_config, dict) and 'tokenizer' in full_config:
        cfg['tokenizer'] = deepcopy(full_config['tokenizer'])
    return source


def _truthy_env_or_cfg(value) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if text in ('0', 'false', 'no', 'n', 'off'):
        return False
    if text in ('auto', ''):
        return None
    raise ValueError(f'Invalid boolean/auto value: {value!r}')


def cfg_bool(value, default: bool) -> bool:
    parsed = _truthy_env_or_cfg(value)
    return bool(default) if parsed is None else bool(parsed)


def maybe_initialize_jax_distributed(cfg: Dict[str, Any]) -> None:
    tcfg = cfg.get('training', {})
    raw = tcfg.get(
        'initialize_jax_distributed',
        os.environ.get('DOWNSTREAM_JAX_DISTRIBUTED', 'auto'))
    decision = _truthy_env_or_cfg(raw)
    if decision is None:
        tpu_env_keys = (
            'TPU_NAME',
            'CLOUD_TPU_TASK_ID',
            'TPU_WORKER_ID',
            'TPU_PROCESS_BOUNDS',
            'TPU_CHIPS_PER_HOST_BOUNDS',
        )
        decision = any(os.environ.get(k) for k in tpu_env_keys)
    if decision:
        tj._maybe_initialize_jax_distributed()


def load_yaml(path: str) -> Dict[str, Any]:
    with tj._open_file(path, 'r') as f:
        return yaml.safe_load(f) or {}


def write_text(path: str, text: str):
    with tj._open_file(path, 'w') as f:
        f.write(text)
    set_text_gcs_metadata(path)


def _gcs_bucket_blob(path: str) -> Tuple[str, str]:
    rest = str(path)[5:]
    bucket, _, blob = rest.partition('/')
    return bucket, blob


def set_text_gcs_metadata(path: str):
    path_s = str(path)
    if not path_s.startswith('gs://'):
        return
    content_type = 'text/plain; charset=utf-8'
    content_disposition = 'inline'
    try:
        fs = tj._get_gcs_fs()
        if fs is not None and hasattr(fs, 'setxattrs'):
            fs.setxattrs(
                path_s,
                content_type=content_type,
                content_disposition=content_disposition,
            )
            return
    except Exception:
        pass
    try:
        from google.cloud import storage

        bucket_name, blob_name = _gcs_bucket_blob(path_s)
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        blob.content_type = content_type
        blob.content_disposition = content_disposition
        blob.patch()
    except Exception:
        pass


def append_text(path: str, text: str):
    # GCS does not support normal append reliably; rewrite small log file.
    old = ''
    if tj._file_exists(path):
        with tj._open_file(path, 'r') as f:
            old = f.read()
    write_text(path, old + text)


# -----------------------------
# Task prompts
# -----------------------------

@dataclass(frozen=True)
class PromptExample:
    prompt: str
    candidates: Tuple[Tuple[str, int], ...]
    gold_index: int


def _gold(cands: Sequence[Tuple[str, int]], label: int) -> int:
    for i, (_, y) in enumerate(cands):
        if int(y) == int(label):
            return i
    raise ValueError(f'label={label} not in {cands}')


def build_prompt(task: str, ex: Dict[str, Any]) -> PromptExample:
    task = task.lower()
    if task == 'sst2':
        c = ((" negative", 0), (" positive", 1))
        return PromptExample(f"Sentence: {ex['sentence']}\nSentiment:", c, _gold(c, ex['label']))
    if task == 'rte':
        c = ((" yes", 0), (" no", 1))  # GLUE RTE: 0 entailment, 1 not_entailment
        return PromptExample(
            f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}\nDoes the premise entail the hypothesis? Answer:",
            c, _gold(c, ex['label']))
    if task == 'mnli':
        c = ((" yes", 0), (" maybe", 1), (" no", 2))
        return PromptExample(
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nDoes the premise entail the hypothesis? Answer:",
            c, _gold(c, ex['label']))
    if task == 'qqp':
        c = ((" no", 0), (" yes", 1))
        return PromptExample(
            f"Question 1: {ex['question1']}\nQuestion 2: {ex['question2']}\nAre these questions duplicates? Answer:",
            c, _gold(c, ex['label']))
    if task == 'mrpc':
        c = ((" no", 0), (" yes", 1))
        return PromptExample(
            f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nAre these sentences paraphrases? Answer:",
            c, _gold(c, ex['label']))
    if task == 'boolq':
        c = ((" no", 0), (" yes", 1))
        lab = int(bool(ex['label']))
        return PromptExample(f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:", c, _gold(c, lab))
    if task == 'wic':
        c = ((" no", 0), (" yes", 1))
        lab = int(bool(ex['label']))
        return PromptExample(
            f"Word: {ex['word']}\nSentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nDoes the word have the same meaning in both sentences? Answer:",
            c, _gold(c, lab))
    raise ValueError(f'Unsupported task: {task}')


def hf_spec(task: str) -> Tuple[str, Optional[str], str, str]:
    task = task.lower()
    if task in ('sst2', 'rte', 'mnli', 'qqp', 'mrpc'):
        eval_split = 'validation_matched' if task == 'mnli' else 'validation'
        return 'glue', task, 'train', eval_split
    if task in ('boolq', 'wic'):
        return 'super_glue', task, 'train', 'validation'
    raise ValueError(f'No HF spec for task={task}')


def hf_name_candidates(name: str) -> List[str]:
    """Return compatible HF dataset ids for old and new hub versions."""
    aliases = {
        'glue': 'nyu-mll/glue',
        'super_glue': 'aps/super_glue',
    }
    alias = aliases.get(name)
    if alias:
        return [alias, name]
    return [name]


# -----------------------------
# Data loading
# -----------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with tj._open_file(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tsv(path: str) -> List[Dict[str, Any]]:
    with tj._open_file(path, 'r') as f:
        text = f.read().splitlines()
    return list(csv.DictReader(text, delimiter='\t'))


def load_raw_splits(data_cfg: Dict[str, Any], task: str):
    source = data_cfg.get('source', 'hf')
    if source == 'hf':
        if load_dataset is None:
            raise RuntimeError('datasets is not installed. Install datasets or use data.source=jsonl/tsv.')
        name, subset, train_split, eval_split = hf_spec(task)
        name = data_cfg.get('hf_name', name)
        subset = data_cfg.get('hf_config', subset)
        train_split = data_cfg.get('train_split', train_split)
        eval_split = data_cfg.get('eval_split', eval_split)
        explicit_name = 'hf_name' in data_cfg
        names = [name] if explicit_name else hf_name_candidates(name)
        errors = []
        for candidate in names:
            try:
                log(f'[data] HF load_dataset({candidate!r}, {subset!r}) train={train_split} eval={eval_split}')
                train = load_dataset(candidate, subset, split=train_split)
                evals = load_dataset(candidate, subset, split=eval_split)
                return train, evals
            except Exception as e:
                errors.append(f'{candidate}: {type(e).__name__}: {e}')
                if candidate == names[-1]:
                    break
                log(f'[data] HF load_dataset failed for {candidate!r}; retrying with {names[-1]!r}')
        raise RuntimeError(
            f'Failed to load HF dataset for task={task}. Attempts:\n'
            + '\n'.join(errors))
    if source == 'jsonl':
        return load_jsonl(data_cfg['train_path']), load_jsonl(data_cfg['eval_path'])
    if source == 'tsv':
        return load_tsv(data_cfg['train_path']), load_tsv(data_cfg['eval_path'])
    raise ValueError(f'Unknown data.source: {source}')


def ensure_pad_token(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return int(tokenizer.pad_token_id)


def encode_prompt_answer(tokenizer, prompt: str, answer: str, max_seq_len: int, add_eos=False) -> Optional[Dict[str, np.ndarray]]:
    pids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    aids = tokenizer(answer, add_special_tokens=False)['input_ids']
    if add_eos and tokenizer.eos_token_id is not None:
        aids = aids + [tokenizer.eos_token_id]
    if not aids:
        return None
    if len(pids) + len(aids) > max_seq_len:
        keep = max_seq_len - len(aids)
        if keep <= 0:
            return None
        pids = pids[-keep:]
    ids = pids + aids
    labels = [-100] * len(pids) + aids
    score_mask = [0] * len(pids) + [1] * len(aids)
    return {
        'input_ids': np.asarray(ids, dtype=np.int32),
        'labels': np.asarray(labels, dtype=np.int32),
        'score_mask': np.asarray(score_mask, dtype=np.int32),
    }


def make_train_rows(raw, task: str, tokenizer, max_seq_len: int, max_samples: Optional[int], seed: int, add_eos=False):
    idxs = list(range(len(raw)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if max_samples is not None:
        idxs = idxs[:int(max_samples)]
    rows = []
    for i in idxs:
        ex = raw[i]
        if 'label' in ex:
            try:
                if int(ex['label']) < 0:
                    continue
            except Exception:
                pass
        pe = build_prompt(task, ex)
        answer = pe.candidates[pe.gold_index][0]
        row = encode_prompt_answer(tokenizer, pe.prompt, answer, max_seq_len, add_eos=add_eos)
        if row is not None:
            rows.append(row)
    return rows


def make_eval_rows(raw, task: str, tokenizer, max_seq_len: int, max_samples: Optional[int], add_eos=False):
    lim = len(raw) if max_samples is None else min(int(max_samples), len(raw))
    rows = []
    for i in range(lim):
        ex = raw[i]
        if 'label' in ex:
            try:
                if int(ex['label']) < 0:
                    continue
            except Exception:
                pass
        pe = build_prompt(task, ex)
        cands = []
        for txt, _ in pe.candidates:
            row = encode_prompt_answer(tokenizer, pe.prompt, txt, max_seq_len, add_eos=add_eos)
            if row is None:
                cands = []
                break
            cands.append(row)
        if cands:
            rows.append({'candidates': cands, 'gold_index': pe.gold_index})
    return rows


def pad_fixed(rows: Sequence[np.ndarray], max_seq_len: int, pad_value: int, dtype=np.int32) -> np.ndarray:
    out = np.full((len(rows), max_seq_len), pad_value, dtype=dtype)
    for i, x in enumerate(rows):
        n = min(len(x), max_seq_len)
        out[i, :n] = x[:n]
    return out


def build_global_train_batch(rows: List[Dict[str, np.ndarray]], indices: List[int], max_seq_len: int, pad_token_id: int):
    br = [rows[i] for i in indices]
    input_ids = pad_fixed([r['input_ids'] for r in br], max_seq_len, pad_token_id)
    labels = pad_fixed([r['labels'] for r in br], max_seq_len, -100)
    attention_mask = (input_ids != pad_token_id).astype(np.int32)
    return input_ids, labels, attention_mask


def build_candidate_global_batch(flat_rows: List[Dict[str, np.ndarray]], indices: List[int], max_seq_len: int, pad_token_id: int):
    br = [flat_rows[i] for i in indices]
    input_ids = pad_fixed([r['input_ids'] for r in br], max_seq_len, pad_token_id)
    score_mask = pad_fixed([r['score_mask'] for r in br], max_seq_len, 0)
    attention_mask = (input_ids != pad_token_id).astype(np.int32)
    return input_ids, score_mask, attention_mask


def local_slice(global_arr: np.ndarray) -> np.ndarray:
    n_hosts = jax.process_count()
    host = jax.process_index()
    assert global_arr.shape[0] % n_hosts == 0, (global_arr.shape, n_hosts)
    per_host = global_arr.shape[0] // n_hosts
    return global_arr[host * per_host:(host + 1) * per_host]


# -----------------------------
# Checkpoints
# -----------------------------

def _copy_dict_tree(x):
    if isinstance(x, dict):
        return {k: _copy_dict_tree(v) for k, v in x.items()}
    return x


def _adapt_checkpoint_params_to_target(raw_params, target_params):
    """Adapt known DAWN checkpoint naming variants to the instantiated target.

    Important: do NOT blindly call train_jax.migrate_legacy_v4155_params().
    That migration is for loading legacy params into the *new* dawn_srw model.
    For v3.9.4 downstream the target still expects qk/v/know keys, so applying
    the migration first turns the checkpoint into attn_qk/attn_v/rst and breaks
    restore with missing qk_* keys.
    """
    if not isinstance(raw_params, dict) or not isinstance(target_params, dict):
        return raw_params

    target_pool = target_params.get('neuron_pool', {})
    raw_pool = raw_params.get('neuron_pool', {})
    if not isinstance(target_pool, dict) or not isinstance(raw_pool, dict):
        return raw_params

    target_keys = set(target_pool.keys())
    raw_keys = set(raw_pool.keys())

    legacy_pool_keys = {'qk_emb', 'qk_read', 'qk_write',
                        'v_emb', 'v_read', 'v_write',
                        'know_emb', 'know_read', 'know_write'}
    modern_pool_keys = {'attn_qk_emb', 'attn_qk_read', 'attn_qk_write',
                        'attn_v_emb', 'attn_v_read', 'attn_v_write',
                        'rst_emb', 'rst_read', 'rst_write'}

    # Modern/SRW checkpoint -> legacy v3.9.4 target.
    if legacy_pool_keys.issubset(target_keys) and not legacy_pool_keys.issubset(raw_keys) and modern_pool_keys.intersection(raw_keys):
        rp = _copy_dict_tree(raw_params)
        pool = rp.setdefault('neuron_pool', {})
        pool_map = {
            'attn_qk_emb': 'qk_emb', 'attn_qk_read': 'qk_read', 'attn_qk_write': 'qk_write',
            'attn_v_emb': 'v_emb', 'attn_v_read': 'v_read', 'attn_v_write': 'v_write',
            'rst_emb': 'know_emb', 'rst_read': 'know_read', 'rst_write': 'know_write',
        }
        for src, dst in pool_map.items():
            if src in pool and dst not in pool:
                pool[dst] = pool[src]

        router = rp.get('router', {})
        if isinstance(router, dict):
            if 'proj_rst' in router and 'proj_know' not in router:
                router['proj_know'] = router['proj_rst']
            if 'tau_rst' in router and 'tau_know' not in router:
                router['tau_know'] = router['tau_rst']
        return rp

    # Legacy checkpoint -> modern dawn_srw target. Only then use train_jax migration.
    if modern_pool_keys.issubset(target_keys) and not modern_pool_keys.issubset(raw_keys) and legacy_pool_keys.intersection(raw_keys):
        if hasattr(tj, 'migrate_legacy_v4155_params'):
            migrated = tj.migrate_legacy_v4155_params({'params': raw_params})
            if isinstance(migrated, dict) and 'params' in migrated:
                return migrated['params']

    return raw_params


def _summarize_param_key_mismatch(raw_params, target_params) -> str:
    lines = []
    for section in ('neuron_pool', 'router'):
        raw_s = raw_params.get(section, {}) if isinstance(raw_params, dict) else {}
        tgt_s = target_params.get(section, {}) if isinstance(target_params, dict) else {}
        if isinstance(raw_s, dict) and isinstance(tgt_s, dict):
            lines.append(f'{section}: target={sorted(tgt_s.keys())[:30]} raw={sorted(raw_s.keys())[:30]}')
    return '\n'.join(lines)


def restore_orbax_params_only(ref: CheckpointRef, params, opt_state, cfg: Dict[str, Any], mesh, rng):
    if ref.kind != 'orbax' or ref.checkpoint_dir is None or ref.step is None:
        raise ValueError(f'Invalid Orbax checkpoint ref: {ref}')
    tj._require_orbax_checkpoint_compat()
    manager = tj._create_orbax_checkpoint_manager(
        ref.checkpoint_dir,
        create=False,
        read_only=True,
    )
    try:
        restored = manager.restore(
            int(ref.step),
            args=tj.ocp.args.Composite(
                state=tj.ocp.args.StandardRestore(),
                metadata=tj.ocp.args.JsonRestore(),
            ),
        )
        restored_state = tj._composite_item(restored, 'state')
    finally:
        manager.close()
    if not isinstance(restored_state, dict) or 'params' not in restored_state:
        raise ValueError(f'Orbax checkpoint did not restore params: {ref.path}')
    raw_params = _adapt_checkpoint_params_to_target(restored_state['params'], params)
    try:
        restored_params = tj._match_tree_to_template_on_mesh(
            raw_params, params, mesh, name='params')
    except ValueError as e:
        detail = _summarize_param_key_mismatch(raw_params, params)
        raise ValueError(
            f'Failed to restore Orbax params from {ref.path}. '
            f'Key summary after adapter:\n{detail}') from e
    log(f'[ckpt] Orbax params-only loaded: {ref.path}')
    return restored_params


def restore_flax_params_only(path: str, params):
    import flax.serialization as serialization
    with tj._open_file(path, 'rb') as f:
        data = f.read()
    raw = serialization.msgpack_restore(data)
    if not isinstance(raw, dict) or 'params' not in raw:
        raise ValueError(f'Checkpoint does not contain params: {path}')

    raw_params = _adapt_checkpoint_params_to_target(raw['params'], params)
    try:
        restored = serialization.from_state_dict({'params': params}, {'params': raw_params})
    except ValueError as e:
        detail = _summarize_param_key_mismatch(raw_params, params)
        raise ValueError(f'Failed to restore params from {path}. Key summary after adapter:\n{detail}') from e

    log(f'[ckpt] params-only loaded: {path}')
    return restored['params']


def restore_params_only(ref_or_path, params, opt_state=None, cfg=None, mesh=None, rng=None):
    if isinstance(ref_or_path, CheckpointRef):
        if ref_or_path.kind == 'orbax':
            if opt_state is None or cfg is None or mesh is None or rng is None:
                raise ValueError('Orbax params restore requires opt_state, cfg, mesh, and rng.')
            return restore_orbax_params_only(ref_or_path, params, opt_state, cfg, mesh, rng)
        return restore_flax_params_only(ref_or_path.path, params)
    return restore_flax_params_only(str(ref_or_path), params)


def downstream_training_config(cfg: Dict[str, Any], extra=None) -> Dict[str, Any]:
    train_cfg = dict(cfg.get('training', {}))
    train_cfg['downstream'] = cfg.get('downstream', {})
    if extra:
        train_cfg['extra'] = extra
    return train_cfg


def restore_downstream_orbax_checkpoint(ref: CheckpointRef, target_params,
                                        target_opt_state, cfg: Dict[str, Any],
                                        mesh, rng):
    if ref.kind != 'orbax' or ref.checkpoint_dir is None or ref.step is None:
        raise ValueError(f'Invalid downstream Orbax checkpoint ref: {ref}')
    target_state = tj._build_orbax_state(
        target_params,
        target_opt_state,
        rng,
        epoch=0,
        global_step=0,
        step_in_epoch=0,
        steps_per_epoch=0,
        best_val_loss=float('inf'),
        training_config=downstream_training_config(cfg),
        full_config=cfg,
        model_config=cfg.get('model', {}),
    )
    manager = tj._create_orbax_checkpoint_manager(
        ref.checkpoint_dir,
        create=False,
        read_only=True,
    )
    try:
        restored_state, _ = tj._restore_orbax_state(
            manager, int(ref.step), target_state)
    finally:
        manager.close()
    if not isinstance(restored_state, dict) or 'params' not in restored_state:
        raise ValueError(f'Orbax checkpoint did not restore state.params: {ref.path}')
    params = tj._match_tree_to_template_on_mesh(
        restored_state['params'], target_params, mesh, name='params')
    opt_state = tj._match_tree_to_template_on_mesh(
        restored_state['opt_state'], target_opt_state, mesh, name='opt_state')
    global_step = int(np.asarray(jax.device_get(
        restored_state.get('global_step', restored_state.get('step', 0)))).reshape(()))
    best_val_loss = float(np.asarray(jax.device_get(
        restored_state.get('best_val_loss', 1.0))).reshape(()))
    restored_rng = np.asarray(restored_state.get('rng', np.asarray(jax.device_get(rng))), dtype=np.uint32).reshape((2,))
    return params, opt_state, jnp.asarray(restored_rng, dtype=jnp.uint32), global_step, -best_val_loss


# -----------------------------
# Model/optimizer/mesh
# -----------------------------

def make_optimizer(cfg: Dict[str, Any], total_steps: int):
    t = cfg.get('training', {})
    lr = float(t.get('lr', t.get('learning_rate', 2e-5)))
    warmup_steps = int(t.get('warmup_steps', max(1, int(total_steps * float(t.get('warmup_ratio', 0.06))))))
    wd = float(t.get('weight_decay', 0.01))
    end_ratio = float(t.get('min_lr_ratio', 0.1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps), end_value=lr * end_ratio)
    return optax.chain(
        optax.clip_by_global_norm(float(t.get('max_grad_norm', 1.0))),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=wd),
    )


def _tree_path_to_str(path) -> str:
    return '/'.join(str(p.key if hasattr(p, 'key') else p) for p in path)


def _tree_path_has_part(path, *names: str) -> bool:
    parts = tuple(part for part in _tree_path_to_str(path).split('/') if part)
    return any(name in parts for name in names)


def _is_tau_update_path(path) -> bool:
    return _tree_path_has_part(
        path,
        'raw_tau',
        'raw_tau_attn',
        'raw_tau_attn_qk',
        'raw_tau_attn_v',
        'raw_tau_qk',
        'raw_tau_v',
        'raw_tau_rst',
        'tau_attn',
        'tau_rst',
    )


def _factory_kwargs(factory, kwargs):
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _chunk_size_from_count(name: str, local_count: int, n_chunks: int) -> int:
    n_chunks = max(1, int(n_chunks))
    local_count = int(local_count)
    if local_count <= 0:
        raise ValueError(f'{name} local count must be > 0, got {local_count}')
    return max(1, int(math.ceil(local_count / n_chunks)))


def _strict_checkpoint_runtime_required(version: str) -> bool:
    return str(version) in CHECKPOINT_STRICT_RUNTIME_MODEL_VERSIONS


def _require_training_runtime_value(cfg: Dict[str, Any], key: str, context: str):
    training = cfg.get('training', {})
    if not isinstance(training, dict) or key not in training:
        raise ValueError(
            f'training.{key} is required for {context}; downstream refuses '
            'to use YAML/code fallback for checkpoint-calibrated runtime.')
    return training[key]


def _require_positive_training_float(cfg: Dict[str, Any],
                                     key: str,
                                     context: str) -> float:
    value = _require_training_runtime_value(cfg, key, context)
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(
            f'training.{key} must be finite and > 0 for {context}, got {value!r}')
    return out


def _require_nonnegative_training_float(cfg: Dict[str, Any],
                                        key: str,
                                        context: str) -> float:
    value = _require_training_runtime_value(cfg, key, context)
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(
            f'training.{key} must be finite and >= 0 for {context}, got {value!r}')
    return out


def build_sharded_fns_if_needed(cfg: Dict[str, Any], mesh):
    version = cfg.get('model', {}).get('model_version', '')
    mesh_model = int(cfg.get('training', {}).get('mesh_model', 1))
    if tj._is_baseline_version(version):
        if mesh_model <= 1:
            return None
        from models.baseline_transformer_jax import create_baseline_sharded_fns
        return create_baseline_sharded_fns(mesh, cfg)

    if version == 'spatial-r1-v3.9.4' and mesh_model > 1:
        import models.legacy.dawn_spatial_v394_exp as dawn394
        m = cfg['model']; tr = cfg.get('training', {})
        n_know = int(m.get('n_know', 25200)) // mesh_model
        n_qk = int(m.get('n_qk', 1580)) // mesh_model
        n_v = int(m.get('n_v', 2600)) // mesh_model
        ck = max(1, math.ceil(n_know / max(1, int(tr.get('n_chunks_know', 1)))))
        cq = max(1, math.ceil(n_qk / max(1, int(tr.get('n_chunks_qk', 1)))))
        cv = max(1, math.ceil(n_v / max(1, int(tr.get('n_chunks_v', 1)))))
        single_chunk = max(ck, cv)
        paired_chunk = cq
        return (dawn394.make_sharded_srw(mesh, max_chunk_size=single_chunk),
                dawn394.make_sharded_srw_paired(mesh, max_chunk_size=paired_chunk))

    if not tj._is_active_srw_version(version):
        return None

    entry = tj._model_registry_entry(version)
    srw_module = __import__(entry['module'], fromlist=['make_sharded_srw'])
    make_sharded_srw = srw_module.make_sharded_srw
    make_sharded_srw_minimal = getattr(srw_module, 'make_sharded_srw_minimal', None)
    make_sharded_srw_paired = getattr(srw_module, 'make_sharded_srw_paired', None)
    make_sharded_srw_paired_minimal = getattr(
        srw_module, 'make_sharded_srw_paired_minimal', None)
    if make_sharded_srw_paired is None:
        raise RuntimeError(f'{version} module is missing make_sharded_srw_paired.')

    m = cfg['model']; tr = cfg.get('training', {})
    for name in ('n_qk', 'n_v'):
        if int(m[name]) % mesh_model != 0:
            raise ValueError(
                f'model.{name}={m[name]} must be divisible by mesh_model={mesh_model}.')
    n_rst = int(m.get('n_rst', m.get('n_know', 25200)))
    if n_rst % mesh_model != 0:
        raise ValueError(
            f'model.n_rst={n_rst} must be divisible by mesh_model={mesh_model}.')

    nqk_local = int(m['n_qk']) // mesh_model
    nv_local = int(m['n_v']) // mesh_model
    nrst_local = n_rst // mesh_model
    qk_chunk = _chunk_size_from_count(
        'attn_qk', nqk_local, tr.get('n_chunks_qk', 1))
    v_chunk = _chunk_size_from_count(
        'attn_v', nv_local, tr.get('n_chunks_v', 1))
    rst_chunk = _chunk_size_from_count(
        'rst', nrst_local, tr.get('n_chunks_rst', tr.get('n_chunks_know', 1)))

    base_kwargs = {'mesh': mesh}
    if _strict_checkpoint_runtime_required(version):
        _require_nonnegative_training_float(
            cfg, 'admission_den_power', f'{version} sharded SRW runtime')
        _require_nonnegative_training_float(
            cfg, 'admission_den_grad_scale', f'{version} sharded SRW runtime')
        _require_positive_training_float(
            cfg, 'soft_gate_effective_active_eps',
            f'{version} sharded SRW runtime')
    base_kwargs.update(tj._v4164_sharded_kwargs(cfg))
    single_v = make_sharded_srw(
        max_chunk_size=v_chunk,
        **_factory_kwargs(make_sharded_srw, base_kwargs))
    single_rst = make_sharded_srw(
        max_chunk_size=rst_chunk,
        **_factory_kwargs(make_sharded_srw, base_kwargs))
    paired_qk = make_sharded_srw_paired(
        max_chunk_size=qk_chunk,
        **_factory_kwargs(make_sharded_srw_paired, base_kwargs))

    sharded_fns = {
        'single': single_v,
        'attn_v_single': single_v,
        'rst_single': single_rst,
        'paired': paired_qk,
        'attn_qk_paired': paired_qk,
    }
    if make_sharded_srw_minimal is not None:
        single_qk_min = make_sharded_srw_minimal(
            max_chunk_size=qk_chunk,
            **_factory_kwargs(make_sharded_srw_minimal, base_kwargs))
        single_v_min = make_sharded_srw_minimal(
            max_chunk_size=v_chunk,
            **_factory_kwargs(make_sharded_srw_minimal, base_kwargs))
        single_rst_min = make_sharded_srw_minimal(
            max_chunk_size=rst_chunk,
            **_factory_kwargs(make_sharded_srw_minimal, base_kwargs))
        sharded_fns.update({
            'attn_qk_single_minimal': single_qk_min,
            'attn_v_single_minimal': single_v_min,
            'rst_single_minimal': single_rst_min,
        })
    if make_sharded_srw_paired_minimal is not None:
        paired_min = make_sharded_srw_paired_minimal(
            max_chunk_size=qk_chunk,
            **_factory_kwargs(make_sharded_srw_paired_minimal, base_kwargs))
        sharded_fns['attn_qk_paired_minimal'] = paired_min

    use_vocab_parallel = cfg_bool(tr.get('use_vocab_parallel'), True)
    if (use_vocab_parallel
            and version in (tj.V4166_MODEL_VERSION, tj.V4168_MODEL_VERSION)
            and mesh_model > 1
            and m.get('logical_vocab_size') is not None
            and m.get('vocab_size_padded') is not None):
        from models.vocab_parallel import (
            make_vocab_parallel_ce,
            make_vocab_parallel_embedding,
        )
        logical_vocab = int(m['logical_vocab_size'])
        padded_vocab = int(m['vocab_size_padded'])
        if padded_vocab % mesh_model != 0:
            raise ValueError(
                f'model.vocab_size_padded={padded_vocab} must be divisible by mesh_model={mesh_model}.')
        ce_chunk = int(tr.get('ce_token_chunk_size', 32768))
        sharded_fns['vocab_parallel_embedding'] = make_vocab_parallel_embedding(
            mesh, logical_vocab, padded_vocab)
        sharded_fns['vocab_ce'] = make_vocab_parallel_ce(
            mesh,
            logical_vocab_size=logical_vocab,
            vocab_size_padded=padded_vocab,
            token_chunk_size=ce_chunk,
            compute_accuracy=bool(tr.get('train_compute_accuracy', True)),
            compute_logit_stats=False,
        )
    log(
        f'[shard] {version} shard_map enabled mesh_model={mesh_model} '
        f'chunks qk/v/rst={qk_chunk}/{v_chunk}/{rst_chunk}')
    return sharded_fns


def _call_extra_kwargs(model, cfg, sharded_fns, deterministic: bool,
                       compute_accuracy: bool):
    kwargs = {}
    if sharded_fns is not None:
        kwargs['sharded_fns'] = sharded_fns
    if tj._model_accepts_analysis(model):
        kwargs['analysis'] = False
    model_version = str(getattr(
        model, '__version__', getattr(type(model), '__version__', '')))
    tr = cfg.get('training', {})
    strict_runtime = _strict_checkpoint_runtime_required(model_version)
    use_minimal_train = cfg_bool(
        tr.get('use_minimal_train_path', tr.get('use_minimal_train')), True)
    if (model_version in (tj.V4166_MODEL_VERSION, tj.V4168_MODEL_VERSION)
            and tj._model_accepts_minimal_train(model)
            and use_minimal_train):
        kwargs['minimal_train'] = True
    if tj._model_accepts_soft_gate_schedule(model):
        if strict_runtime:
            soft_t = _require_positive_training_float(
                cfg, 'soft_gate_temperature', f'{model_version} forward')
            kwargs['soft_gate_temperature'] = soft_t
            kwargs['soft_gate_T_qk'] = _require_positive_training_float(
                cfg, 'soft_gate_T_qk', f'{model_version} forward')
            kwargs['soft_gate_T_v'] = _require_positive_training_float(
                cfg, 'soft_gate_T_v', f'{model_version} forward')
            kwargs['soft_gate_T_rst'] = _require_positive_training_float(
                cfg, 'soft_gate_T_rst', f'{model_version} forward')
        else:
            soft_t = float(tr.get(
                'soft_gate_temperature',
                tr.get('soft_gate_t_final', 0.07)))
            kwargs['soft_gate_temperature'] = soft_t
            kwargs['soft_gate_T_qk'] = float(tr.get('soft_gate_T_qk', soft_t))
            kwargs['soft_gate_T_v'] = float(tr.get('soft_gate_T_v', soft_t))
            kwargs['soft_gate_T_rst'] = float(tr.get('soft_gate_T_rst', soft_t))
    if tj._model_accepts_soft_gate_t_final(model):
        if strict_runtime:
            kwargs['soft_gate_t_final'] = _require_positive_training_float(
                cfg, 'soft_gate_t_final', f'{model_version} forward')
        else:
            kwargs['soft_gate_t_final'] = float(tr.get('soft_gate_t_final', 0.07))
    if tj._model_accepts_soft_gate_boundary_power(model):
        if strict_runtime:
            kwargs['soft_gate_boundary_power'] = _require_positive_training_float(
                cfg, 'soft_gate_boundary_power', f'{model_version} forward')
            kwargs['soft_gate_boundary_power_final'] = (
                _require_positive_training_float(
                    cfg, 'soft_gate_boundary_power_final',
                    f'{model_version} forward'))
        else:
            boundary = float(tr.get(
                'soft_gate_boundary_power',
                tr.get('soft_gate_boundary_power_final', 4.0)))
            kwargs['soft_gate_boundary_power'] = boundary
            kwargs['soft_gate_boundary_power_final'] = float(
                tr.get('soft_gate_boundary_power_final', boundary))
    if tj._model_accepts_admission_den_power(model):
        if strict_runtime:
            kwargs['admission_den_power'] = _require_nonnegative_training_float(
                cfg, 'admission_den_power', f'{model_version} forward')
        else:
            kwargs['admission_den_power'] = float(tr.get('admission_den_power', 1.0))
    if tj._model_accepts_execution_prune_eps(model):
        kwargs['execution_prune_eps'] = 0.0
    if tj._model_accepts_ce_token_chunk_size(model):
        if strict_runtime:
            chunk_size = int(_require_training_runtime_value(
                cfg, 'ce_token_chunk_size', f'{model_version} forward'))
            if chunk_size <= 0:
                raise ValueError(
                    f'training.ce_token_chunk_size must be > 0, got {chunk_size}')
            kwargs['ce_token_chunk_size'] = chunk_size
        else:
            kwargs['ce_token_chunk_size'] = int(tr.get('ce_token_chunk_size', 32768))
    if tj._model_accepts_compute_accuracy(model):
        kwargs['compute_accuracy'] = bool(compute_accuracy)
    return kwargs


def model_apply_train(model, cfg, params, input_ids, labels, attention_mask,
                      dropout_key, sharded_fns, deterministic=False,
                      compute_accuracy=True):
    kwargs = _call_extra_kwargs(
        model, cfg, sharded_fns, deterministic, compute_accuracy)
    return model.apply({'params': params}, input_ids, labels=labels,
                       attention_mask=attention_mask, deterministic=deterministic,
                       rngs={'dropout': dropout_key}, **kwargs)


def model_apply_logits(model, cfg, params, input_ids, attention_mask, sharded_fns):
    kwargs = _call_extra_kwargs(
        model, cfg, sharded_fns, deterministic=True, compute_accuracy=False)
    return model.apply({'params': params}, input_ids, labels=None,
                       attention_mask=attention_mask, deterministic=True,
                       rngs={'dropout': jax.random.PRNGKey(0)}, **kwargs)['logits']


def make_train_step(model, cfg, optimizer, sharded_fns, aux_weight: float,
                    tau_weight: float, tau_lr_mult: float = 1.0):
    tau_lr_mult = float(tau_lr_mult)
    if not math.isfinite(tau_lr_mult) or tau_lr_mult < 0.0:
        raise ValueError(f'training.tau_lr_mult must be finite and >= 0, got {tau_lr_mult}')

    @jax.jit
    def train_step(params, opt_state, input_ids, labels, attention_mask, dropout_key):
        def loss_fn(p):
            out = model_apply_train(
                model, cfg, p, input_ids, labels, attention_mask,
                dropout_key, sharded_fns, deterministic=False,
                compute_accuracy=True)
            lm_loss = out['loss']
            aux_loss = out.get('aux_loss', jnp.float32(0.0))
            tau_reg = out.get('tau_reg', jnp.float32(0.0))
            total = lm_loss + aux_weight * aux_loss + tau_weight * tau_reg
            acc = out['correct'] / jnp.maximum(out['valid_count'], 1)
            return total, {'loss': total, 'lm_loss': lm_loss, 'aux_loss': aux_loss, 'tau_reg': tau_reg, 'acc': acc,
                           'valid_count': out['valid_count']}
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        if tau_lr_mult != 1.0:
            tau_mult = jnp.float32(tau_lr_mult)

            def scale_tau_update(path, update):
                if _is_tau_update_path(path):
                    return update * tau_mult.astype(update.dtype)
                return update

            updates = jax.tree.map_with_path(scale_tau_update, updates)
        new_params = optax.apply_updates(params, updates)
        metrics['grad_norm'] = optax.global_norm(grads)
        metrics['tau_lr_mult'] = jnp.float32(tau_lr_mult)
        return new_params, new_opt, metrics
    return train_step


def make_score_step(model, cfg, sharded_fns, use_ce_scoring: bool):
    @jax.jit
    def score_step(params, input_ids, score_mask, attention_mask):
        if use_ce_scoring:
            labels = jnp.where(score_mask == 1, input_ids, -100)
            out = model_apply_train(
                model, cfg, params, input_ids, labels, attention_mask,
                jax.random.PRNGKey(0), sharded_fns, deterministic=True,
                compute_accuracy=False)
            per_token_ce = out['per_token_ce']
            mask = score_mask[:, 1:].astype(jnp.float32)
            summed = (per_token_ce * mask).sum(axis=-1)
            denom = jnp.maximum(mask.sum(axis=-1), 1.0)
            return -summed / denom
        logits = model_apply_logits(
            model, cfg, params, input_ids, attention_mask, sharded_fns)
        logits = logits[:, :-1, :]
        target = input_ids[:, 1:]
        mask = score_mask[:, 1:].astype(jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        tok = jnp.take_along_axis(logp, target[..., None], axis=-1).squeeze(-1)
        summed = (tok * mask).sum(axis=-1)
        denom = jnp.maximum(mask.sum(axis=-1), 1.0)
        return summed / denom
    return score_step


# -----------------------------
# Evaluation
# -----------------------------

def flatten_eval(eval_rows):
    flat, meta = [], []
    for ex_id, row in enumerate(eval_rows):
        for cand_id, cand in enumerate(row['candidates']):
            flat.append(cand)
            meta.append((ex_id, cand_id, row['gold_index']))
    return flat, meta


def evaluate(params, score_step, eval_rows, batch_size: int, max_seq_len: int, pad_token_id: int, data_sharding):
    flat, meta = flatten_eval(eval_rows)
    # Pad flat candidates to full global batches for static shape.
    n = len(flat)
    n_batches = math.ceil(n / batch_size)
    scores_by_ex: Dict[int, List[Tuple[int, float]]] = {}
    gold_by_ex: Dict[int, int] = {}
    for b in range(n_batches):
        idxs = list(range(b * batch_size, min((b + 1) * batch_size, n)))
        valid_n = len(idxs)
        if valid_n < batch_size:
            idxs = idxs + [idxs[-1]] * (batch_size - valid_n)
        g_ids, g_mask, g_attn = build_candidate_global_batch(flat, idxs, max_seq_len, pad_token_id)
        l_ids, l_mask, l_attn = local_slice(g_ids), local_slice(g_mask), local_slice(g_attn)
        gs = (batch_size, max_seq_len)
        ids = tj.shard_to_mesh(l_ids, data_sharding, gs)
        sm = tj.shard_to_mesh(l_mask, data_sharding, gs)
        am = tj.shard_to_mesh(l_attn, data_sharding, gs)
        sc_arr = score_step(params, ids, sm, am)
        # Multi-host SPMD arrays can span non-addressable devices; do not
        # jax.device_get() them directly. Gather the global candidate scores
        # across hosts, then only host 0 materializes/records metrics.
        sc_global = np.asarray(process_allgather(sc_arr, tiled=True)).reshape(-1)[:valid_n]
        if is_host0():
            for s, mi in zip(sc_global, idxs[:valid_n]):
                ex_id, cand_id, gold = meta[mi]
                scores_by_ex.setdefault(ex_id, []).append((cand_id, float(s)))
                gold_by_ex[ex_id] = gold
    if not is_host0():
        return {'accuracy': 0.0, 'total': 0}
    correct = 0
    total = 0
    for ex_id, xs in scores_by_ex.items():
        pred = max(xs, key=lambda x: x[1])[0]
        gold = gold_by_ex[ex_id]
        correct += int(pred == gold)
        total += 1
    return {'accuracy': correct / max(total, 1), 'total': total}


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--init-from', default=None)
    ap.add_argument('--resume-from', default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ds_cfg = cfg.get('downstream', cfg.get('data', {}))
    task = ds_cfg.get('task') or cfg.get('task')
    if not task:
        raise ValueError('Config must set downstream.task')

    if args.init_from and args.resume_from:
        raise ValueError('Use only one of --init-from or --resume-from')

    maybe_initialize_jax_distributed(cfg)

    cfg_resume = cfg.get('resume_from') or ds_cfg.get('resume_from')
    resume_from = args.resume_from or cfg_resume
    init_from = None if resume_from else (args.init_from or cfg.get('init_from') or ds_cfg.get('init_from'))

    init_ref = resolve_init_checkpoint_ref(init_from) if init_from else None
    resume_ref = resolve_downstream_orbax_resume_ref(resume_from) if resume_from else None
    model_config_source = apply_init_checkpoint_model_config(
        cfg, resume_ref if resume_ref is not None else init_ref)
    ds_cfg = cfg.get('downstream', cfg.get('data', {}))

    seed = int(cfg.get('seed', 1))
    random.seed(seed + jax.process_index())
    np.random.seed(seed + jax.process_index())

    if AutoTokenizer is None:
        raise RuntimeError('transformers is not installed. Install transformers.')
    tok_name = cfg.get('tokenizer', ds_cfg.get('tokenizer', 'bert-base-uncased'))
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    pad_id = ensure_pad_token(tokenizer)

    use_vocab_parallel = cfg_bool(
        cfg.get('training', {}).get('use_vocab_parallel'), True)
    if use_vocab_parallel:
        tj._maybe_materialize_vocab_parallel_config(cfg)

    # Build model from the existing train_jax registry. No train_jax/model code is modified here.
    model = tj.build_model_from_config(cfg)
    model_version = cfg.get('model', {}).get('model_version', 'baseline')
    use_ce_scoring = (
        model_version in (tj.V4166_MODEL_VERSION, tj.V4168_MODEL_VERSION)
        and int(cfg.get('training', {}).get('mesh_model', 1)) > 1
        and use_vocab_parallel)

    mcfg, tcfg = cfg.get('model', {}), cfg.get('training', {})
    max_seq_len = int(ds_cfg.get('max_seq_len', mcfg.get('max_seq_len', 512)))
    batch_size = int(tcfg.get('batch_size', 64))
    eval_batch_size = int(tcfg.get('eval_batch_size', batch_size))

    total_devices = jax.device_count()
    n_hosts = jax.process_count()
    host = jax.process_index()
    if batch_size % total_devices != 0:
        raise ValueError(f'training.batch_size={batch_size} must be divisible by device_count={total_devices}')
    if eval_batch_size % total_devices != 0:
        raise ValueError(f'eval_batch_size={eval_batch_size} must be divisible by device_count={total_devices}')
    mesh_model = int(tcfg.get('mesh_model', 1))
    mesh_data = int(tcfg.get('mesh_data', 0)) or (total_devices // mesh_model)
    if mesh_data * mesh_model != total_devices:
        raise ValueError(
            f'training.mesh_data({mesh_data}) * mesh_model({mesh_model}) '
            f'must equal device_count={total_devices}')
    if n_hosts > 1:
        tj._assert_multihost_same_startup_context({
            'trainer_script': 'scripts/downstream_finetune_jax.py',
            'config_path': str(args.config),
            'model_version': model_version,
            'model_config_source': model_config_source,
            'task': task,
            'init_from': init_from,
            'resume_from': resume_from,
            'process_count': n_hosts,
            'mesh_data': mesh_data,
            'mesh_model': mesh_model,
            'batch_size': batch_size,
            'eval_batch_size': eval_batch_size,
        })

    # Load downstream data, not C4.
    raw_train, raw_eval = load_raw_splits(ds_cfg, task)
    train_rows = make_train_rows(raw_train, task, tokenizer, max_seq_len, ds_cfg.get('max_train_samples'), seed, bool(ds_cfg.get('add_eos', False)))
    eval_rows = make_eval_rows(raw_eval, task, tokenizer, max_seq_len, ds_cfg.get('max_eval_samples'), bool(ds_cfg.get('add_eos', False)))
    if not train_rows or not eval_rows:
        raise RuntimeError(f'Empty downstream rows: train={len(train_rows)}, eval={len(eval_rows)}')

    num_epochs = int(tcfg.get('num_epochs', 3))
    max_steps_cfg = tcfg.get('max_steps')
    steps_per_epoch = max(1, len(train_rows) // batch_size)
    total_steps = int(max_steps_cfg) if max_steps_cfg else steps_per_epoch * num_epochs
    log_every = int(tcfg.get('log_interval', 20))
    eval_every = int(tcfg.get('eval_interval', 200))

    # Output/run dir.
    ckpt_root = cfg.get('checkpoint_dir') or ds_cfg.get('checkpoint_dir')
    if not ckpt_root:
        raise ValueError('Config must set checkpoint_dir')
    # train_jax-style run directory: never write directly into a fixed run_name.
    # Host 0 creates the unique folder name and broadcasts it; otherwise each
    # worker would include its own PID and enter a different Orbax barrier.
    legacy_resume_path = None
    if resume_from:
        if resume_ref is not None:
            run_dir = str(resume_ref.run_folder).rstrip('/')
        else:
            legacy_resume_path = resolve_flax_checkpoint_path(resume_from)
            run_dir = str(resume_from).rstrip('/')
        run_name = run_dir.rstrip('/').rsplit('/', 1)[-1]
    else:
        if is_host0():
            run_prefix = cfg.get('run_name') or f"{model_version}_{task}"
            if not str(run_prefix).startswith('run_'):
                host0_run_name = f"run_v{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            else:
                host0_run_name = f"{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            host0_run_dir = join_path(ckpt_root, host0_run_name)
        else:
            host0_run_dir = None
        run_dir = broadcast_str_from_host0(host0_run_dir)
        if not run_dir:
            raise RuntimeError('Failed to broadcast downstream run_dir from host 0.')
        run_name = run_dir.rstrip('/').rsplit('/', 1)[-1]
    train_log_path = join_path(run_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    def record(msg: str):
        if is_host0():
            print(msg, flush=True)
            append_text(train_log_path, msg + '\n')

    if is_host0():
        record('=' * 60)
        record(f'Downstream fine-tune: model={model_version} task={task}')
        record(f'Config: {args.config}')
        record(f'model_config_source={model_config_source}')
        record(f'Hosts={n_hosts} devices={total_devices} local_devices={jax.local_device_count()} host_id={host}')
        record(f'mesh_data={mesh_data} mesh_model={mesh_model}')
        record(f'batch={batch_size} eval_batch={eval_batch_size} max_seq_len={max_seq_len}')
        record(f'lr={tcfg.get("lr")} warmup_ratio={tcfg.get("warmup_ratio")} eval_interval={eval_every} log_interval={log_every}')
        record(
            f'use_vocab_parallel={str(use_vocab_parallel).lower()} '
            f'use_minimal_train_path={str(cfg_bool(tcfg.get("use_minimal_train_path", tcfg.get("use_minimal_train")), True)).lower()} '
            f'use_ce_scoring={str(use_ce_scoring).lower()} '
            f'tau_lr_mult={float(tcfg.get("tau_lr_mult", 1.0)):.3f}')
        record(f'downstream_source={ds_cfg.get("source", "hf")} hf_name={ds_cfg.get("hf_name", "<default>")} hf_config={ds_cfg.get("hf_config", "<default>")}')
        record(f'train_rows={len(train_rows)} eval_rows={len(eval_rows)} total_steps={total_steps}')
        record(f'init_from={init_from or "<none>"}')
        record('init_policy=params_only_fresh_optimizer_step0')
        record(f'resume_from={resume_from or "<none>"}')
        record(f'run_name={run_name}')
        record(f'run_dir={run_dir}')
        record(f'training_log={train_log_path}')
        record('checkpoint_write=disabled')
        record('=' * 60)

    # Initialize params.
    key = jax.random.PRNGKey(seed)
    dummy_ids = jnp.ones((1, min(max_seq_len, 32)), dtype=jnp.int32)
    dummy_labels = jnp.ones_like(dummy_ids)
    variables = model.init({'params': key, 'dropout': key}, dummy_ids, labels=dummy_labels, deterministic=True)
    params = variables['params']

    # Mesh/shard params.
    mesh = tj.create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P('data', None))
    param_shardings = tj.get_param_shardings(
        params,
        mesh,
        model_version,
        vocab_size_padded=(
            cfg.get('model', {}).get('vocab_size_padded', None)
            if use_vocab_parallel else None),
    )
    params = tj.shard_params_to_mesh(params, param_shardings)

    sharded_fns = build_sharded_fns_if_needed(cfg, mesh)

    optimizer = make_optimizer(cfg, total_steps)
    opt_state = optimizer.init(params)
    opt_state = tj._replicate_optimizer_state_scalars_to_mesh(opt_state, mesh)
    rng = jax.random.PRNGKey(seed + 1000)

    # Load params-only or resume full state.
    global_step = 0
    best_acc = -1.0
    if resume_from:
        if resume_ref is not None:
            params, opt_state, rng, global_step, best_acc = restore_downstream_orbax_checkpoint(
                resume_ref, params, opt_state, cfg, mesh, rng)
            log(f'[resume] loaded downstream Orbax state from: {resume_ref.path} step={global_step}')
        else:
            ckpt = tj.load_checkpoint(legacy_resume_path, params, opt_state)
            params = ckpt['params']; opt_state = ckpt['opt_state']
            global_step = int(ckpt.get('step', 0))
            best_acc = float(-ckpt.get('best_val_loss', 1.0))
            log(f'[resume] loaded legacy Flax params+optimizer+step from: {legacy_resume_path} step={global_step}')
    elif init_from:
        ip = init_ref
        if ip is None:
            raise RuntimeError('init_from is set but checkpoint ref was not resolved.')
        params = restore_params_only(ip, params, opt_state=opt_state, cfg=cfg, mesh=mesh, rng=key)
        opt_state = optimizer.init(params)
        opt_state = tj._replicate_optimizer_state_scalars_to_mesh(opt_state, mesh)
        global_step = 0
        best_acc = -1.0
        log(f'[init] loaded params only from: {ip.path}; optimizer=fresh; step=0')
    else:
        log('[init] no init-from: random-init downstream control')

    # Verify step consistency.
    if n_hosts > 1:
        local = np.array([global_step], dtype=np.int32)
        all_steps = np.asarray(process_allgather(local)).flatten()
        if not np.all(all_steps == global_step):
            raise RuntimeError(f'global_step mismatch across hosts: {all_steps.tolist()}')
        log(f'[verified] global_step={global_step} consistent across {n_hosts} hosts')

    train_step = make_train_step(
        model, cfg, optimizer, sharded_fns,
        float(tcfg.get('aux_weight', 0.0)),
        float(tcfg.get('tau_weight', 0.0)),
        float(tcfg.get('tau_lr_mult', 1.0)))
    score_step = make_score_step(model, cfg, sharded_fns, use_ce_scoring)

    def _sync_eval_decision(ev: Dict[str, Any], current_best: float):
        """Broadcast host0 eval accuracy and new-best decision to all hosts.

        evaluate() materializes metrics only on host0, so every host enters
        the same gather and then host0 writes the log record.
        """
        if is_host0():
            acc = float(ev.get('accuracy', 0.0))
            total = float(ev.get('total', 0))
            flag = 1.0 if acc > current_best else 0.0
            local = np.array([flag, acc, total], dtype=np.float32)
        else:
            local = np.zeros((3,), dtype=np.float32)
        if jax.process_count() > 1:
            gathered = np.asarray(process_allgather(local, tiled=True)).reshape(-1, 3)
            # process_index 0 is the logging/writing leader.
            flag, acc, total = gathered[0].tolist()
        else:
            flag, acc, total = local.tolist()
        return bool(flag > 0.5), float(acc), int(total)

    # Initial eval.
    ev = evaluate(params, score_step, eval_rows, eval_batch_size, max_seq_len, pad_id, data_sharding)
    new_best, ev_acc, ev_total = _sync_eval_decision(ev, best_acc)
    if new_best:
        best_acc = ev_acc
    if is_host0():
        record(
            f"[eval] step={global_step} acc={ev_acc:.4f} "
            f"total={ev_total} best_acc={best_acc:.4f} "
            f"new_best={str(new_best).lower()}")

    # Training.
    t0 = time.time()
    epoch = global_step // max(steps_per_epoch, 1)
    while global_step < total_steps:
        # Deterministic shuffle per epoch; all hosts compute same global batch ids.
        epoch = global_step // max(steps_per_epoch, 1)
        rng_py = random.Random(seed + epoch)
        order = list(range(len(train_rows)))
        rng_py.shuffle(order)
        pos = (global_step % max(steps_per_epoch, 1)) * batch_size
        if pos + batch_size <= len(order):
            idxs = order[pos:pos + batch_size]
        else:
            idxs = (order[pos:] + order[:batch_size - (len(order) - pos)])
        if len(idxs) < batch_size:
            idxs = (idxs * ((batch_size // max(len(idxs), 1)) + 1))[:batch_size]

        g_ids, g_labels, g_attn = build_global_train_batch(train_rows, idxs, max_seq_len, pad_id)
        l_ids, l_labels, l_attn = local_slice(g_ids), local_slice(g_labels), local_slice(g_attn)
        gs = (batch_size, max_seq_len)
        ids = tj.shard_to_mesh(l_ids, data_sharding, gs)
        labels = tj.shard_to_mesh(l_labels, data_sharding, gs)
        attn = tj.shard_to_mesh(l_attn, data_sharding, gs)
        rng, step_rng = jax.random.split(rng)
        params, opt_state, metrics = train_step(params, opt_state, ids, labels, attn, step_rng)
        global_step += 1

        if global_step % log_every == 0 or global_step == 1:
            # Metrics may be global sharded arrays in multi-host mode. Gather
            # each scalar and print only on host 0.
            m = jax.tree.map(lambda x: np.asarray(process_allgather(x)).reshape(-1)[0], metrics)
            if is_host0():
                elapsed = time.time() - t0
                tok = global_step * batch_size * max_seq_len
                record(
                    f"[train] step={global_step}/{total_steps} "
                    f"loss={float(m['loss']):.4f} "
                    f"lm_loss={float(m['lm_loss']):.4f} "
                    f"acc={float(m['acc']):.4f} "
                    f"grad_norm={float(m['grad_norm']):.3f} "
                    f"tokens={tok} best_acc={best_acc:.4f} "
                    f"elapsed_sec={elapsed:.1f}")

        if global_step % eval_every == 0 or global_step == total_steps:
            ev = evaluate(params, score_step, eval_rows, eval_batch_size, max_seq_len, pad_id, data_sharding)
            new_best, ev_acc, ev_total = _sync_eval_decision(ev, best_acc)
            if new_best:
                best_acc = ev_acc
            if is_host0():
                record(
                    f"[eval] step={global_step} acc={ev_acc:.4f} "
                    f"total={ev_total} best_acc={best_acc:.4f} "
                    f"new_best={str(new_best).lower()} "
                    f"elapsed_sec={time.time() - t0:.1f}")

    if is_host0():
        final_msg = (
            f'[summary] task={task} best_acc={best_acc:.4f} '
            f'step={global_step} run_dir={run_dir} '
            f'training_log={train_log_path}')
        record(final_msg)

    # Explicit end-of-task barrier.  The outer sequence script starts the next
    # Python process independently on each worker, so all hosts must leave this
    # task together.  Without this, fast non-host0 workers can start the next
    # config while host0 is still writing the final log line, causing hangs.
    if n_hosts > 1:
        done = np.array([global_step], dtype=np.int32)
        gathered_done = np.asarray(process_allgather(done)).reshape(-1)
        if not np.all(gathered_done == global_step):
            raise RuntimeError(f'end-of-task step mismatch across hosts: {gathered_done.tolist()}')


if __name__ == '__main__':
    main()
