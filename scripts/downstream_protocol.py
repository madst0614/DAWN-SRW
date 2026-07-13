#!/usr/bin/env python3
"""Downstream-only source pinning, schedule, and result policies.

This module intentionally has no JAX imports at module load time so its policy
logic can be unit-tested on a CPU-only machine.  The ``pin-source`` CLI imports
the canonical training engine lazily only when a real checkpoint must be
resolved.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TaskDefaults:
    epochs: int
    eval_interval: int


# Fair benchmark defaults.  Keep this as the single policy table rather than
# duplicating task overrides across suite YAML files.
TASK_DEFAULTS: Dict[str, TaskDefaults] = {
    'sst2': TaskDefaults(epochs=3, eval_interval=100),
    'rte': TaskDefaults(epochs=5, eval_interval=10),
    'wic': TaskDefaults(epochs=5, eval_interval=20),
    'boolq': TaskDefaults(epochs=5, eval_interval=100),
    'mnli': TaskDefaults(epochs=3, eval_interval=250),
}


@dataclass(frozen=True)
class DownstreamSchedule:
    task: str
    train_rows: int
    batch_size: int
    steps_per_epoch: int
    requested_epochs: Optional[int]
    total_steps: int
    expected_examples_seen: int
    effective_epochs: float
    schedule_source: str
    eval_interval: int

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values['calculated_total_steps'] = values['total_steps']
        values['global_batch_size'] = values['batch_size']
        return values


@dataclass(frozen=True)
class PinnedSource:
    requested: str
    resolved: str
    step: int


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be an integer, got {value!r}') from exc
    if parsed <= 0:
        raise ValueError(f'{name} must be > 0, got {parsed}')
    return parsed


def _configured_epochs(training: Mapping[str, Any]) -> Optional[int]:
    present = [key for key in ('epochs', 'num_epochs') if key in training]
    if len(present) > 1:
        raise ValueError('Set only one of training.epochs and training.num_epochs')
    if not present:
        return None
    return _positive_int(training[present[0]], f'training.{present[0]}')


def _examples_seen(train_rows: int, batch_size: int, steps_per_epoch: int,
                   total_steps: int) -> int:
    full_epochs, remaining_steps = divmod(total_steps, steps_per_epoch)
    return (
        full_epochs * train_rows
        + min(remaining_steps * batch_size, train_rows)
    )


def validate_schedule_config(task: str, training: Mapping[str, Any]) -> None:
    task = str(task).lower().strip()
    explicit_total = 'total_steps' in training
    legacy_total = 'max_steps' in training
    epochs = _configured_epochs(training)
    if explicit_total and legacy_total:
        raise ValueError(
            'Set only one of training.total_steps and legacy training.max_steps')
    if (explicit_total or legacy_total) and epochs is not None:
        raise ValueError(
            'training.total_steps/max_steps and epochs/num_epochs are '
            'mutually exclusive')
    if not explicit_total and not legacy_total and epochs is None:
        if task not in TASK_DEFAULTS:
            raise ValueError(
                f'No task default schedule for {task!r}; set training.epochs '
                'or training.total_steps explicitly')


def calculate_schedule(task: str, train_rows: int, batch_size: int,
                       training: Mapping[str, Any]) -> DownstreamSchedule:
    """Materialize the task schedule from actual rows and global batch size.

    ``total_steps`` is the explicit modern override.  ``max_steps`` remains an
    unambiguous legacy compatibility field and is reported as
    ``recovered_legacy_protocol``.  A step cap combined with an epoch count is
    rejected instead of silently choosing one.
    """
    task = str(task).lower().strip()
    train_rows = _positive_int(train_rows, 'train_rows')
    batch_size = _positive_int(batch_size, 'global batch size')
    steps_per_epoch = math.ceil(train_rows / batch_size)
    if steps_per_epoch < 1:
        raise ValueError('calculated steps_per_epoch must be >= 1')

    validate_schedule_config(task, training)
    explicit_total = 'total_steps' in training
    legacy_total = 'max_steps' in training
    epochs = _configured_epochs(training)

    defaults = TASK_DEFAULTS.get(task)
    if explicit_total:
        total_steps = _positive_int(
            training['total_steps'], 'training.total_steps')
        requested_epochs = None
        schedule_source = 'explicit_total_steps'
    elif legacy_total:
        total_steps = _positive_int(training['max_steps'], 'training.max_steps')
        requested_epochs = None
        schedule_source = 'recovered_legacy_protocol'
    elif epochs is not None:
        requested_epochs = epochs
        total_steps = requested_epochs * steps_per_epoch
        schedule_source = 'explicit_epochs'
    else:
        if defaults is None:
            raise ValueError(
                f'No task default schedule for {task!r}; set training.epochs '
                'or training.total_steps explicitly')
        requested_epochs = defaults.epochs
        total_steps = requested_epochs * steps_per_epoch
        schedule_source = 'task_default_epochs'

    if total_steps < 1:
        raise ValueError('calculated total_steps must be >= 1')

    explicit_eval = training.get('eval_interval')
    if explicit_eval is None and 'val_interval' in training:
        explicit_eval = training['val_interval']
    if explicit_eval is None:
        if defaults is None:
            raise ValueError(
                f'No task default eval interval for {task!r}; set '
                'training.eval_interval explicitly')
        raw_eval_interval = defaults.eval_interval
    else:
        try:
            raw_eval_interval = int(explicit_eval)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'training.eval_interval must be an integer, got '
                f'{explicit_eval!r}') from exc
        if raw_eval_interval < 0:
            raise ValueError('training.eval_interval must be >= 0')
    eval_interval = (
        min(raw_eval_interval, total_steps) if raw_eval_interval > 0 else 0)

    return DownstreamSchedule(
        task=task,
        train_rows=train_rows,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        requested_epochs=requested_epochs,
        total_steps=total_steps,
        expected_examples_seen=_examples_seen(
            train_rows, batch_size, steps_per_epoch, total_steps),
        effective_epochs=total_steps / steps_per_epoch,
        schedule_source=schedule_source,
        eval_interval=eval_interval,
    )


def evaluation_reasons(step: int, schedule: DownstreamSchedule) -> List[str]:
    """Return unique reasons to evaluate after a completed optimizer step."""
    step = int(step)
    if step < 1 or step > schedule.total_steps:
        return []
    reasons = []
    if (schedule.eval_interval > 0
            and step % schedule.eval_interval == 0):
        reasons.append('interval')
    if step % schedule.steps_per_epoch == 0:
        reasons.append('epoch_boundary')
    if step == schedule.total_steps:
        reasons.append('final')
    return reasons


def evaluation_steps(schedule: DownstreamSchedule) -> List[int]:
    return [
        step for step in range(1, schedule.total_steps + 1)
        if evaluation_reasons(step, schedule)
    ]


def normalize_checkpoint_path(path: str) -> str:
    value = str(path).strip().rstrip('/\\')
    if not value:
        raise ValueError('checkpoint path must not be empty')
    return value


def pin_source_once(requested: str, resolver: Callable[[str], Any]) -> PinnedSource:
    requested = normalize_checkpoint_path(requested)
    resolved_source = resolver(requested)
    if isinstance(resolved_source, Mapping):
        resolved = resolved_source.get('checkpoint_path')
        step = resolved_source.get('step')
    else:
        resolved = getattr(resolved_source, 'checkpoint_path', None)
        step = getattr(resolved_source, 'step', None)
    if resolved is None or step is None:
        raise RuntimeError('checkpoint resolver did not return checkpoint_path and step')
    return PinnedSource(
        requested=requested,
        resolved=normalize_checkpoint_path(str(resolved)),
        step=_positive_int(step, 'source checkpoint step'),
    )


def pinned_sources_for_tasks(tasks: Iterable[str],
                             source: PinnedSource) -> Dict[str, PinnedSource]:
    return {str(task).lower(): source for task in tasks}


def verify_task_source(actual_path: str, actual_step: int,
                       expected_path: str, expected_step: int) -> None:
    actual_path = normalize_checkpoint_path(actual_path)
    expected_path = normalize_checkpoint_path(expected_path)
    actual_step = _positive_int(actual_step, 'actual source step')
    expected_step = _positive_int(expected_step, 'expected source step')
    if actual_path != expected_path or actual_step != expected_step:
        raise RuntimeError(
            'Task source checkpoint does not match pinned suite source: '
            f'actual={actual_path}@{actual_step} '
            f'expected={expected_path}@{expected_step}')


def source_request_from_configs(config_paths: Sequence[str],
                                override: Optional[str] = None) -> str:
    if override is not None and str(override).strip():
        return normalize_checkpoint_path(str(override))
    requests = []
    import yaml

    for config_path in config_paths:
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle) or {}
        downstream = config.get('downstream', {}) or {}
        source = config.get('init_from') or downstream.get('init_from')
        if not source:
            raise ValueError(
                f'Downstream config has no init_from: {config_path}')
        requests.append(normalize_checkpoint_path(str(source)))
    unique = sorted(set(requests))
    if len(unique) != 1:
        raise RuntimeError(
            'Sequence configs request different source checkpoints: '
            + ', '.join(unique))
    if not unique:
        raise ValueError('No downstream configs were provided')
    return unique[0]


def build_result(*, task: str, source: PinnedSource,
                 schedule: DownstreamSchedule, initial_acc: float,
                 best_seen_acc: float, best_seen_step: int,
                 final_acc: float, final_step: int,
                 eval_total: int) -> Dict[str, Any]:
    values = (initial_acc, best_seen_acc, final_acc)
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError('initial/best/final accuracy must all be finite')
    best_seen_step = int(best_seen_step)
    final_step = int(final_step)
    if not (0 <= best_seen_step <= schedule.total_steps):
        raise RuntimeError('best_seen_step is outside the training schedule')
    if final_step != schedule.total_steps:
        raise RuntimeError(
            'Final evaluation was not performed at calculated total_steps: '
            f'final_step={final_step} total_steps={schedule.total_steps}')
    if int(eval_total) <= 0:
        raise RuntimeError('final evaluation did not produce any examples')

    result = schedule.to_dict()
    result.update({
        'task': str(task).lower(),
        'source_checkpoint_requested': source.requested,
        'source_checkpoint_resolved': source.resolved,
        'source_checkpoint_step': source.step,
        'source_checkpoint_resolved_once': True,
        'task_source_policy': 'pinned_same_checkpoint',
        'initial_acc': float(initial_acc),
        'best_seen_acc': float(best_seen_acc),
        'best_seen_step': best_seen_step,
        'best_seen_epoch': best_seen_step / schedule.steps_per_epoch,
        'final_acc': float(final_acc),
        'final_step': final_step,
        'final_epoch': final_step / schedule.steps_per_epoch,
        'reported_acc': float(best_seen_acc),
        'eval_total': int(eval_total),
    })
    return result


def format_summary(results: Sequence[Mapping[str, Any]]) -> str:
    required = (
        'task', 'source_checkpoint_step', 'calculated_total_steps',
        'effective_epochs', 'initial_acc', 'best_seen_acc',
        'best_seen_step', 'final_acc', 'reported_acc')
    lines = [
        'Task | Source step | Steps | Epochs | Initial | Best | Best step | Final',
        '--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:',
    ]
    for result in results:
        missing = [key for key in required if key not in result]
        if missing:
            raise RuntimeError(
                'Downstream result is missing fields: ' + ', '.join(missing))
        if float(result['reported_acc']) != float(result['best_seen_acc']):
            raise RuntimeError('reported_acc must equal best_seen_acc')
        lines.append(
            f"{result['task']} | {int(result['source_checkpoint_step'])} | "
            f"{int(result['calculated_total_steps'])} | "
            f"{float(result['effective_epochs']):.4f} | "
            f"{float(result['initial_acc']):.6f} | "
            f"{float(result['best_seen_acc']):.6f} | "
            f"{int(result['best_seen_step'])} | "
            f"{float(result['final_acc']):.6f}")
    return '\n'.join(lines)


def _pin_source_cli(args: argparse.Namespace) -> None:
    requested = source_request_from_configs(args.config, args.source)
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    import scripts.train_jax as train_jax

    # Keep stdout machine-readable for the sequence shell.  Canonical resolver
    # diagnostics still go to stderr and therefore remain visible in logs.
    with contextlib.redirect_stdout(sys.stderr):
        pinned = pin_source_once(requested, train_jax.resolve_transfer_checkpoint)
    print(f'{pinned.requested}\t{pinned.resolved}\t{pinned.step}')


def _summary_cli(args: argparse.Namespace) -> None:
    results = []
    for path in args.result_json:
        with open(path, 'r', encoding='utf-8') as handle:
            results.append(json.load(handle))
    print(format_summary(results))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    pin_parser = subparsers.add_parser('pin-source')
    pin_parser.add_argument('--source')
    pin_parser.add_argument('--config', action='append', default=[])
    pin_parser.set_defaults(func=_pin_source_cli)

    summary_parser = subparsers.add_parser('summary')
    summary_parser.add_argument('--result-json', action='append', required=True)
    summary_parser.set_defaults(func=_summary_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
