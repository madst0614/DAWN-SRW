#!/usr/bin/env python3
"""External DAWN-SRW TPU benchmark wrapper.

This script deliberately keeps benchmark logic out of scripts/train_jax.py.
It creates a short-lived config, runs train_jax as a normal subprocess, then
summarizes the ordinary training logs in the console.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml


SUPPORTED_MODEL_VERSIONS = (
    "spatial-r1-v4.1.6.6",
    "spatial-r1-v4.1.6.8",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run external DAWN-SRW TPU benchmark configs.")
    parser.add_argument("--config", nargs="+", action="append", required=True,
                        help="One or more training config YAML files.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Measured training steps.")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps excluded from the summary.")
    parser.add_argument("--model-version", default=None,
                        help="Optional expected model version.")
    parser.add_argument("--allow-model-version-override", action="store_true",
                        help="Rewrite model.model_version in the temp config.")
    parser.add_argument("--xla-dump-dir", default=None,
                        help="Optional XLA dump directory.")
    parser.add_argument("--dummy-data", action="store_true",
                        help="Use generated local uint16 data instead of config data.")
    parser.add_argument("--work-dir", default=None,
                        help="Temp working directory. Default: system temp.")
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="Keep temp configs, logs, checkpoints, and dummy data.")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be >= 0")
    if args.model_version and args.model_version not in SUPPORTED_MODEL_VERSIONS:
        raise SystemExit(
            f"--model-version must be one of {SUPPORTED_MODEL_VERSIONS}")
    configs = [item for group in args.config for item in group]

    summaries = []
    for index, config_arg in enumerate(configs):
        xla_dump_dir = _format_optional_path(
            args.xla_dump_dir, index, config_arg, len(configs))
        print("\n" + "=" * 72, flush=True)
        print(
            f"SRW external benchmark {index + 1}/{len(configs)}: "
            f"{config_arg}",
            flush=True)
        print("=" * 72, flush=True)
        summary = _run_one_config(config_arg, index, args, xla_dump_dir)
        if summary:
            summaries.append(summary)

    if len(summaries) > 1:
        _print_comparison(summaries)


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _resolve_config(path_text):
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    candidate = _repo_root() / path_text
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Config file not found: {path_text}")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _stem(path_text):
    return Path(path_text.replace("\\", "/")).name.rsplit(".", 1)[0]


def _format_optional_path(template, index, config_arg, config_count):
    if not template:
        return None
    if config_count <= 1:
        return template
    return template.format(index=index, config=_stem(config_arg))


def _run_one_config(config_arg, index, args, xla_dump_dir):
    config_path = _resolve_config(config_arg)
    cfg = _load_yaml(config_path)
    model_version = str(cfg.get("model", {}).get("model_version", ""))
    if args.model_version:
        if model_version != args.model_version:
            if not args.allow_model_version_override:
                raise SystemExit(
                    "Config model.model_version disagrees with "
                    f"--model-version: {model_version!r} != "
                    f"{args.model_version!r}")
            cfg.setdefault("model", {})["model_version"] = args.model_version
            model_version = args.model_version
    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise SystemExit(
            f"Unsupported benchmark model_version={model_version!r}; "
            f"supported={SUPPORTED_MODEL_VERSIONS}")

    total_steps = int(args.warmup_steps) + int(args.steps)
    train_cfg = cfg.setdefault("training", {})
    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    batch_size = int(train_cfg["batch_size"])
    seq_len = int(model_cfg.get("max_seq_len", 512))
    run_tokens = int(total_steps * batch_size * seq_len)

    work_root = Path(args.work_dir or tempfile.gettempdir())
    worker = os.environ.get("TPU_WORKER_INDEX", "local")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = work_root / "srw_external_benchmark" / (
        f"{stamp}_{index}_{_stem(config_arg)}_worker_{worker}")
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = work_dir / "checkpoints"

    _prepare_short_config(
        cfg, run_tokens=run_tokens, batch_size=batch_size, seq_len=seq_len,
        total_steps=total_steps, checkpoint_dir=checkpoint_dir,
        dummy_data=bool(args.dummy_data), work_dir=work_dir)
    temp_config = work_dir / "benchmark_config.yaml"
    _write_yaml(temp_config, cfg)

    print("Benchmark wrapper:", flush=True)
    print(f"  train_jax entry: scripts/train_jax.py", flush=True)
    print(f"  temp_config: {temp_config}", flush=True)
    print(f"  model_version: {model_version}", flush=True)
    print(f"  warmup_steps: {args.warmup_steps}", flush=True)
    print(f"  measure_steps: {args.steps}", flush=True)
    print(f"  total_train_steps: {total_steps}", flush=True)
    print(f"  batch_size: {batch_size}", flush=True)
    print(f"  seq_len: {seq_len}", flush=True)
    print(f"  checkpoint_dir: {checkpoint_dir}", flush=True)
    if xla_dump_dir:
        print(f"  xla_dump_dir: {xla_dump_dir}", flush=True)

    argv = [
        sys.executable,
        str(_repo_root() / "scripts" / "train_jax.py"),
        "--config",
        str(temp_config),
        "--from-scratch",
    ]
    env = os.environ.copy()
    if xla_dump_dir:
        _enable_xla_dump(env, xla_dump_dir)

    wall_start = time.perf_counter()
    rc = _run_and_stream(argv, env)
    wall_seconds = time.perf_counter() - wall_start
    if rc != 0:
        raise SystemExit(rc)

    summary = _summarize_training_run(
        config_arg=config_arg,
        temp_config=str(temp_config),
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        warmup_steps=int(args.warmup_steps),
        measure_steps=int(args.steps),
        wall_seconds=wall_seconds,
        xla_dump_dir=xla_dump_dir,
    )
    _print_summary(summary)

    if not args.keep_artifacts:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"  Kept benchmark artifacts: {work_dir}", flush=True)
    return summary


def _prepare_short_config(cfg, *, run_tokens, batch_size, seq_len, total_steps,
                          checkpoint_dir, dummy_data, work_dir):
    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})
    cfg["checkpoint_dir"] = str(checkpoint_dir)

    data_cfg["max_train_tokens"] = int(run_tokens)
    data_cfg["max_val_tokens"] = int(batch_size * seq_len)

    train_cfg["num_epochs"] = 1
    train_cfg["log_interval"] = 1
    train_cfg["val_interval"] = int(total_steps + 1000000)
    train_cfg["checkpoint_interval"] = int(total_steps + 1000000)
    train_cfg["log_analysis_multiplier"] = int(total_steps + 1000000)
    train_cfg["heavy_geometry_multiplier"] = int(total_steps + 1000000)
    train_cfg["oom_check"] = False
    train_cfg["speed_check"] = False
    train_cfg["eval_effective_prune_enabled"] = False
    train_cfg["training_log_append_on_resume"] = False

    if dummy_data:
        vocab_size = int(cfg.get("model", {}).get("vocab_size", 30522))
        data_dir = work_dir / "dummy_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_path = data_dir / "train.bin"
        val_path = data_dir / "val.bin"
        _write_dummy_bin(train_path, max(run_tokens, batch_size * seq_len),
                         vocab_size)
        _write_dummy_bin(val_path, batch_size * seq_len, vocab_size)
        data_cfg["bin_train"] = str(train_path)
        data_cfg["bin_val"] = str(val_path)
        data_cfg.pop("local_cache_dir", None)
        data_cfg.pop("evict_train_shard_cache", None)


def _write_dummy_bin(path, token_count, vocab_size):
    token_count = int(token_count)
    vocab_size = max(2, min(int(vocab_size), 65535))
    values = (np.arange(token_count, dtype=np.uint32) % vocab_size).astype(
        np.uint16)
    values.tofile(path)


def _enable_xla_dump(env, dump_dir):
    env["XLA_DUMP_DIR"] = dump_dir
    flags = env.get("XLA_FLAGS", "")
    dump_flag = f"--xla_dump_to={dump_dir}"
    if dump_flag not in flags:
        flags = (flags + " " + dump_flag).strip()
    if "--xla_dump_hlo_as_text" not in flags:
        flags = (flags + " --xla_dump_hlo_as_text").strip()
    env["XLA_FLAGS"] = flags


def _run_and_stream(argv, env):
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_repo_root()),
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def _summarize_training_run(*, config_arg, temp_config, cfg, checkpoint_dir,
                            warmup_steps, measure_steps, wall_seconds,
                            xla_dump_dir):
    records = _read_train_records(checkpoint_dir)
    train_records = [r for r in records if r.get("type") == "train"]
    train_records.sort(key=lambda r: int(r.get("step", 0)))
    measure_records = [
        r for r in train_records
        if int(r.get("step", 0)) > warmup_steps
        and int(r.get("step", 0)) <= warmup_steps + measure_steps
    ]
    if not measure_records and len(train_records) > warmup_steps:
        measure_records = train_records[warmup_steps:warmup_steps + measure_steps]

    step_seconds = [_step_seconds(r) for r in measure_records]
    step_seconds = [v for v in step_seconds if v is not None and v > 0]
    batch_size = int(cfg.get("training", {}).get("batch_size", 0))
    seq_len = int(cfg.get("model", {}).get("max_seq_len", 0))
    tokens_per_second = [
        batch_size * seq_len / sec for sec in step_seconds if sec > 0]
    first_record = train_records[0] if train_records else None
    xla_report = collect_xla_memory_report(xla_dump_dir) if xla_dump_dir else {}

    return {
        "config_path": config_arg,
        "temp_config": temp_config,
        "model_version": str(cfg.get("model", {}).get("model_version", "")),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "logged_train_steps": len(train_records),
        "wall_seconds": wall_seconds,
        "first_step_seconds": _step_seconds(first_record) if first_record else None,
        "mean_step_seconds": _mean(step_seconds),
        "median_step_seconds": _median(step_seconds),
        "p90_step_seconds": _percentile(step_seconds, 90),
        "min_step_seconds": min(step_seconds) if step_seconds else None,
        "max_step_seconds": max(step_seconds) if step_seconds else None,
        "mean_tokens_per_second": _mean(tokens_per_second),
        "last_loss": _last_float(measure_records, "total_loss"),
        "last_grad_norm": _last_float(measure_records, "grad_norm"),
        "xla_total_hbm_usage": xla_report.get("total_hbm_usage"),
        "xla_program_hbm_requirement": xla_report.get(
            "program_hbm_requirement"),
        "xla_source_file": xla_report.get("source_file"),
    }


def _read_train_records(checkpoint_dir):
    paths = sorted(
        Path(checkpoint_dir).glob("run_*/logs/metrics_*.jsonl"),
        key=lambda p: p.stat().st_mtime)
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _step_seconds(record):
    if not record:
        return None
    raw = _num(record.get("raw_step_time_window"))
    if raw and raw > 0:
        return raw
    sps = _num(record.get("steps_per_sec"))
    if sps and sps > 0:
        return 1.0 / sps
    return None


def _last_float(records, key):
    for record in reversed(records):
        value = _num(record.get(key))
        if value is not None:
            return value
    return None


def collect_xla_memory_report(dump_dir):
    if not dump_dir:
        return {}
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return {"warning": f"XLA dump dir not found: {dump_dir}"}
    patterns = ("*memory*", "*buffer*", "*after_optimizations.txt", "*.txt")
    files = []
    seen = set()
    for pattern in patterns:
        for path in dump_path.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    needles = (
        "Total hbm usage",
        "Program hbm requirement",
        "Largest program allocations",
        "Allocation type: HLO temp",
        "Size:",
        "source_file=",
        "Shape:",
    )
    for path in files[:20]:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if not any(n in text for n in needles[:3]):
            continue
        report = {"source_file": str(path)}
        excerpt = []
        for line in text.splitlines():
            line_s = line.strip()
            if "Total hbm usage" in line_s:
                report["total_hbm_usage"] = line_s
            if "Program hbm requirement" in line_s:
                report["program_hbm_requirement"] = line_s
            if any(n in line_s for n in needles) or "Operator:" in line_s:
                excerpt.append(line_s[:240])
            if len(excerpt) >= 36:
                break
        report["excerpt"] = excerpt
        return report
    return {"warning": f"No XLA memory report found under {dump_dir}"}


def _print_summary(summary):
    print("\n=== SRW External Benchmark Summary ===", flush=True)
    print(f"config={summary.get('config_path')}", flush=True)
    print(f"model={summary.get('model_version')}", flush=True)
    print(
        "steps "
        f"warmup={summary.get('warmup_steps')} "
        f"measure={summary.get('measure_steps')} "
        f"logged={summary.get('logged_train_steps')}",
        flush=True)
    print(
        "step_s "
        f"first={_fmt(summary.get('first_step_seconds'))} "
        f"mean={_fmt(summary.get('mean_step_seconds'))} "
        f"median={_fmt(summary.get('median_step_seconds'))} "
        f"p90={_fmt(summary.get('p90_step_seconds'))} "
        f"min={_fmt(summary.get('min_step_seconds'))} "
        f"max={_fmt(summary.get('max_step_seconds'))}",
        flush=True)
    print(
        "throughput "
        f"mean_tokens_per_second="
        f"{_fmt(summary.get('mean_tokens_per_second'), 1)}",
        flush=True)
    print(
        "last "
        f"loss={_fmt(summary.get('last_loss'), 4)} "
        f"grad_norm={_fmt(summary.get('last_grad_norm'), 3)}",
        flush=True)
    if summary.get("xla_total_hbm_usage") or summary.get(
            "xla_program_hbm_requirement"):
        print(
            "xla_memory "
            f"{summary.get('xla_total_hbm_usage') or ''} "
            f"{summary.get('xla_program_hbm_requirement') or ''}",
            flush=True)
        print(f"xla_source={summary.get('xla_source_file')}", flush=True)
    print(f"wall_seconds={_fmt(summary.get('wall_seconds'))}", flush=True)
    print(
        "SRW_BENCHMARK_SUMMARY_JSON "
        + json.dumps(summary, sort_keys=True),
        flush=True)


def _print_comparison(summaries):
    print("\n" + "=" * 72, flush=True)
    print("SRW Benchmark Comparison", flush=True)
    print("=" * 72, flush=True)
    header = (
        f"{'#':>2}  {'model':24s}  {'mean_s':>10s}  {'tok/s':>12s}  "
        f"{'first_s':>10s}  {'wall_s':>10s}  config")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, summary in enumerate(summaries, 1):
        print(
            f"{i:>2}  {summary.get('model_version', ''):24s}  "
            f"{_fmt(summary.get('mean_step_seconds')):>10s}  "
            f"{_fmt(summary.get('mean_tokens_per_second'), 1):>12s}  "
            f"{_fmt(summary.get('first_step_seconds')):>10s}  "
            f"{_fmt(summary.get('wall_seconds')):>10s}  "
            f"{summary.get('config_path', '')}",
            flush=True)
    base = summaries[0]
    print("\nRatios vs first config:", flush=True)
    for i, summary in enumerate(summaries[1:], 2):
        print(
            f"  #{i}: step_time="
            f"{_ratio(summary.get('mean_step_seconds'), base.get('mean_step_seconds'))} "
            f"tokens="
            f"{_ratio(summary.get('mean_tokens_per_second'), base.get('mean_tokens_per_second'))} "
            f"wall={_ratio(summary.get('wall_seconds'), base.get('wall_seconds'))}",
            flush=True)


def _num(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def _fmt(value, digits=4):
    value = _num(value)
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _ratio(a, b):
    a = _num(a)
    b = _num(b)
    if a is None or b in (None, 0.0):
        return "n/a"
    return f"{a / b:.3f}x"


if __name__ == "__main__":
    main()
