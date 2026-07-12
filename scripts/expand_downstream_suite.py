#!/usr/bin/env python3
"""Expand a downstream suite YAML into per-task downstream configs.

Existing downstream training consumes one YAML per task.  This helper lets a
launcher accept a single suite YAML that names tasks, the pretrain checkpoint,
and shared model/training settings, then materializes task-specific YAML files
for scripts/downstream_finetune_jax.py.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import yaml


def resolve_local_config(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.suffix == "":
        yp = Path(f"{path}.yaml")
        if yp.is_file():
            return yp
        ymlp = Path(f"{path}.yml")
        if ymlp.is_file():
            return ymlp
    raise FileNotFoundError(f"Config not found: {path}")


def resolve_base_config(suite_path: Path, path: str) -> Path:
    p = Path(path)
    candidates: List[Path]
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [p, suite_path.parent / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.suffix == "":
            for suffix in (".yaml", ".yml"):
                with_suffix = Path(f"{candidate}{suffix}")
                if with_suffix.is_file():
                    return with_suffix
    raise FileNotFoundError(f"Base config not found for {suite_path}: {path}")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, Mapping)
            and isinstance(out.get(key), dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("._-") or "downstream_suite"


def task_name_and_override(item: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(item, str):
        return item, {}
    if isinstance(item, Mapping):
        if "task" in item:
            name = str(item["task"])
            override = {k: copy.deepcopy(v) for k, v in item.items() if k != "task"}
            return name, override
        if len(item) == 1:
            name, override = next(iter(item.items()))
            if override is None:
                override = {}
            if not isinstance(override, Mapping):
                raise ValueError(f"Task override for {name!r} must be a mapping")
            return str(name), dict(override)
    raise ValueError(f"Invalid task entry: {item!r}")


def render_template(template: str, values: Mapping[str, str]) -> str:
    return str(template).format(**values)


def build_task_config(
    suite_path: Path,
    suite: Mapping[str, Any],
    task: str,
    task_override: Mapping[str, Any],
) -> Dict[str, Any]:
    base_config = suite.get("base_config")
    if base_config:
        if str(base_config).startswith("gs://"):
            raise ValueError("downstream_suite.base_config must be a local repo path")
        base_path = resolve_base_config(suite_path, str(base_config))
        task_cfg = load_yaml(base_path)
    else:
        task_cfg = {}

    task_cfg = copy.deepcopy(task_cfg)
    task_cfg.pop("data", None)
    task_cfg.pop("log_dir", None)
    if not bool(suite.get("inherit_base_training", False)):
        task_cfg.pop("training", None)

    for key in ("seed", "tokenizer"):
        if key in suite:
            task_cfg[key] = copy.deepcopy(suite[key])

    for key in ("model", "training", "downstream"):
        if key in suite:
            task_cfg[key] = deep_merge(task_cfg.get(key, {}), suite[key])
        if key in task_override:
            task_cfg[key] = deep_merge(task_cfg.get(key, {}), task_override[key])

    downstream = dict(task_cfg.get("downstream", {}))
    downstream["task"] = task
    task_cfg["downstream"] = downstream

    init_from = (
        task_override.get("init_from")
        or suite.get("init_from")
        or suite.get("checkpoint")
    )
    resume_from = task_override.get("resume_from") or suite.get("resume_from")
    if resume_from:
        raise ValueError(
            "Downstream resume_from has been removed; every task must transfer "
            "independently from one committed pretraining checkpoint")
    if init_from:
        task_cfg["init_from"] = str(init_from)

    suite_name = str(suite.get("name") or suite_path.stem)
    values = {
        "name": slugify(suite_name),
        "task": slugify(task),
        "raw_task": task,
    }

    if "checkpoint_dir" in task_override:
        checkpoint_dir = str(task_override["checkpoint_dir"])
    elif "checkpoint_dir_template" in suite:
        checkpoint_dir = render_template(str(suite["checkpoint_dir_template"]), values)
    else:
        output_root = suite.get("output_root") or suite.get("checkpoint_root")
        if not output_root:
            raise ValueError("downstream_suite must set output_root, checkpoint_root, or checkpoint_dir_template")
        checkpoint_dir = str(output_root).rstrip("/")
    task_cfg["checkpoint_dir"] = checkpoint_dir

    if "run_name" in task_override:
        run_name = str(task_override["run_name"])
    else:
        run_name_template = str(suite.get("run_name_template", "{name}"))
        run_name = render_template(run_name_template, values)
    task_cfg["run_name"] = run_name

    metadata = dict(task_cfg.get("suite", {}))
    metadata.update({
        "source": str(suite_path),
        "name": suite_name,
        "task": task,
    })
    task_cfg["suite"] = metadata
    return task_cfg


def expand_config(path: str, output_dir: Path) -> List[Path]:
    config_path = resolve_local_config(path)
    cfg = load_yaml(config_path)
    suite = cfg.get("downstream_suite")
    if suite is None:
        return [config_path]
    if not isinstance(suite, Mapping):
        raise ValueError("downstream_suite must be a mapping")

    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("downstream_suite.tasks must be a non-empty list")

    suite_name = str(suite.get("name") or config_path.stem)
    target_dir = output_dir / slugify(suite_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []
    task_overrides = suite.get("task_overrides", {}) or {}
    if not isinstance(task_overrides, Mapping):
        raise ValueError("downstream_suite.task_overrides must be a mapping when set")

    for item in tasks:
        task, inline_override = task_name_and_override(item)
        named_override = task_overrides.get(task, {}) or {}
        if not isinstance(named_override, Mapping):
            raise ValueError(f"task_overrides.{task} must be a mapping")
        override = deep_merge(dict(named_override), inline_override)
        task_cfg = build_task_config(config_path, suite, task, override)
        out_path = target_dir / f"{slugify(task)}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(task_cfg, f, sort_keys=False, allow_unicode=False)
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--output-dir", default=".generated/downstream_suites")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    paths: List[Path] = []
    for config in args.config:
        paths.extend(expand_config(config, output_dir))

    for path in paths:
        print(path.as_posix())


if __name__ == "__main__":
    main()
