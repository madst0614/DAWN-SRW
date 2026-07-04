#!/usr/bin/env python3
"""Standalone TPU benchmark harness for DAWN-SRW v4166/v4168.

Runs model modules directly and reports benchmark details to stdout.
"""

import argparse
from copy import deepcopy
import importlib
import inspect
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

optax = None

V4166_MODEL_VERSION = "spatial-r1-v4.1.6.6"
V4168_MODEL_VERSION = "spatial-r1-v4.1.6.8"
SUPPORTED_MODEL_VERSIONS = (V4166_MODEL_VERSION, V4168_MODEL_VERSION)
MODEL_MODULES = {
    V4166_MODEL_VERSION: ("models.dawn_srw_v4166", "DAWN_SRW_V4166"),
    V4168_MODEL_VERSION: ("models.dawn_srw_v4168", "DAWN_SRW_V4168"),
}
SECTOR_RUNTIME_METRIC_NAMES = (
    "bucket_fill_mean",
    "bucket_fill_p50",
    "bucket_fill_p90",
    "bucket_fill_p95",
    "bucket_fill_p99",
    "bucket_fill_max",
    "overflow_count",
    "overflow_frac",
    "bucket_capacity",
    "bucket_capacity_util_p50",
    "bucket_capacity_util_p95",
    "bucket_capacity_util_p99",
    "bucket_capacity_util_max",
    "expected_selected_pair_count",
    "executed_pair_count",
    "executed_selected_pair_frac",
    "batch_union_selected_sector_frac",
    "batch_union_effective_operator_frac",
    "per_token_selected_sector_count",
    "per_token_selected_ops",
    "per_token_effective_operator_frac",
    "semantic_work_frac_vs_dense",
    "padded_work_frac_vs_dense",
    "hot_sector_skew_p99_over_mean",
    "attempted_fill_mean",
    "attempted_fill_p50",
    "attempted_fill_p95",
    "attempted_fill_p99",
    "attempted_fill_max",
    "required_capacity_no_overflow",
    "current_capacity",
    "capacity_shortfall",
    "sector_fill_mean",
    "sector_fill_max",
    "sector_overflow_count",
    "selected_sector_frac",
    "effective_operator_frac",
)


def require_optax():
    global optax
    if optax is None:
        try:
            import optax as optax_module
        except ImportError as exc:
            raise SystemExit(
                "optax is required to run the benchmark. "
                "Install dependencies or run through the TPU launcher."
            ) from exc
        optax = optax_module
    return optax


def main():
    parser = argparse.ArgumentParser(
        description="Run standalone DAWN-SRW TPU benchmark configs.")
    parser.add_argument("--config", nargs="+", action="append", required=True,
                        help="One or more training config YAML files.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Measured benchmark steps.")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps excluded from summary.")
    parser.add_argument("--model-version", default=None,
                        help="Optional expected model version.")
    parser.add_argument("--allow-model-version-override", action="store_true",
                        help="Rewrite model.model_version in memory.")
    parser.add_argument("--xla-dump-dir", default=None,
                        help="Optional XLA dump directory.")
    parser.add_argument("--dummy-data", action="store_true",
                        help="Use generated data instead of utils.data_jax.")
    parser.add_argument("--metrics-jsonl",
                        default="benchmark_srw_tpu_metrics.jsonl",
                        help=("Optional JSONL path for sector_runtime, "
                              "vq_layout, and summary records. Supports "
                              "{index} and {config}. Pass an empty string to "
                              "disable."))
    parser.add_argument("--benchmark-force-vq-repack-before-measure",
                        action="store_true",
                        help=("After compile/warmup, force one v4168 V/RST "
                              "hardware repack before measured steps."))
    parser.add_argument("--post-repack-warmup-steps", type=int, default=2,
                        help=("Warmup steps to run after forced VQ repack "
                              "before measured steps. Excluded from summary."))
    parser.add_argument("--override-hardware-repack-max-move-frac", type=float,
                        default=None,
                        help="Runtime override for benchmark VQ max_move_frac.")
    parser.add_argument("--override-hardware-repack-vq-iterations", type=int,
                        default=None,
                        help="Runtime override for benchmark VQ iterations.")
    parser.add_argument("--override-sector-bucket-capacity-mult", type=float,
                        default=None,
                        help="Runtime override for v4168 sector bucket capacity multiplier.")
    parser.add_argument("--sweep-sector-capacity-mult", default=None,
                        help=("Comma-separated sector bucket capacity "
                              "multipliers, e.g. 2,4,8,12,16. "
                              "Use 'high' for 2,4,8,12,16."))
    parser.add_argument("--sweep-hardware-repack-max-move-frac", default=None,
                        help="Comma-separated VQ max_move_frac values.")
    parser.add_argument("--sweep-hardware-repack-vq-iterations", default=None,
                        help="Comma-separated VQ iteration counts.")
    parser.add_argument("--sweep-v-top-blocks", default=None,
                        help="Comma-separated v_top_blocks values.")
    parser.add_argument("--sweep-rst-top-blocks", default=None,
                        help="Comma-separated rst_top_blocks values.")
    parser.add_argument("--sweep-v-block-size", default=None,
                        help="Comma-separated v_block_size values.")
    parser.add_argument("--sweep-rst-block-size", default=None,
                        help="Comma-separated rst_block_size values.")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be >= 0")
    if args.post_repack_warmup_steps < 0:
        raise SystemExit("--post-repack-warmup-steps must be >= 0")
    if args.model_version and args.model_version not in SUPPORTED_MODEL_VERSIONS:
        raise SystemExit(
            f"--model-version must be one of {SUPPORTED_MODEL_VERSIONS}")
    configs = [item for group in args.config for item in group]
    run_specs = expand_run_specs(configs, args)
    xla_dump_dir = _normalize_xla_dump_dir(args.xla_dump_dir, len(run_specs))
    if xla_dump_dir:
        _enable_xla_dump(xla_dump_dir)

    _maybe_initialize_jax_distributed()
    summaries = []
    for run_index, (config_arg, overrides, label) in enumerate(run_specs, 1):
        _log("\n" + "=" * 72)
        _log(
            f"SRW standalone benchmark "
            f"{run_index}/{len(run_specs)}: {config_arg}")
        if label:
            _log(f"variant: {label}")
        _log("=" * 72)
        summary = run_one_config(
            config_arg, args, xla_dump_dir,
            overrides=overrides, variant_label=label,
            run_index=run_index, run_count=len(run_specs))
        summaries.append(summary)

    if len(summaries) > 1 and _is_host0():
        print_comparison(summaries)


def parse_csv_values(text, cast, name, aliases=None):
    if text is None:
        return None
    alias_key = str(text).strip().lower()
    if aliases and alias_key in aliases:
        return [cast(value) for value in aliases[alias_key]]
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(cast(raw))
        except ValueError as exc:
            raise SystemExit(f"{name} contains invalid value {raw!r}") from exc
    if not values:
        raise SystemExit(f"{name} must contain at least one value")
    return values


def expand_run_specs(configs, args):
    base = {}
    if args.override_sector_bucket_capacity_mult is not None:
        base["sector_capacity_mult"] = float(
            args.override_sector_bucket_capacity_mult)
    if args.override_hardware_repack_max_move_frac is not None:
        base["hardware_repack_max_move_frac"] = float(
            args.override_hardware_repack_max_move_frac)
    if args.override_hardware_repack_vq_iterations is not None:
        base["hardware_repack_vq_iterations"] = int(
            args.override_hardware_repack_vq_iterations)

    sweep_items = []
    sweep_defs = (
        ("sector_capacity_mult", args.sweep_sector_capacity_mult, float,
         "--sweep-sector-capacity-mult",
         {"high": (2, 4, 8, 12, 16)}),
        ("hardware_repack_max_move_frac",
         args.sweep_hardware_repack_max_move_frac, float,
         "--sweep-hardware-repack-max-move-frac", None),
        ("hardware_repack_vq_iterations",
         args.sweep_hardware_repack_vq_iterations, int,
         "--sweep-hardware-repack-vq-iterations", None),
        ("v_top_blocks", args.sweep_v_top_blocks, int,
         "--sweep-v-top-blocks", None),
        ("rst_top_blocks", args.sweep_rst_top_blocks, int,
         "--sweep-rst-top-blocks", None),
        ("v_block_size", args.sweep_v_block_size, int,
         "--sweep-v-block-size", None),
        ("rst_block_size", args.sweep_rst_block_size, int,
         "--sweep-rst-block-size", None),
    )
    for key, text, cast, name, aliases in sweep_defs:
        values = parse_csv_values(text, cast, name, aliases=aliases)
        if values is not None:
            sweep_items.append((key, values))

    specs = []
    for config_arg in configs:
        if not sweep_items:
            label = format_variant_label(base)
            specs.append((config_arg, dict(base), label))
            continue
        keys = [item[0] for item in sweep_items]
        value_lists = [item[1] for item in sweep_items]
        for combo in itertools.product(*value_lists):
            overrides = dict(base)
            overrides.update(dict(zip(keys, combo)))
            specs.append((config_arg, overrides, format_variant_label(overrides)))
    return specs


def format_variant_label(overrides):
    if not overrides:
        return ""
    order = (
        "sector_capacity_mult",
        "hardware_repack_max_move_frac",
        "hardware_repack_vq_iterations",
        "v_top_blocks",
        "rst_top_blocks",
        "v_block_size",
        "rst_block_size",
    )
    parts = []
    for key in order:
        if key in overrides:
            parts.append(f"{key}={overrides[key]}")
    return ",".join(parts)


def _is_host0():
    return jax.process_index() == 0


_LIVE_STATUS_ACTIVE = False
_LIVE_STATUS_LEN = 0


def _console_width():
    try:
        return max(40, int(shutil.get_terminal_size((120, 20)).columns))
    except Exception:
        return 120


def _clip_status_line(message):
    message = str(message)
    width = _console_width()
    if len(message) < width:
        return message
    if width <= 4:
        return message[:width]
    return message[:width - 4] + " ..."


def _finish_status_line():
    global _LIVE_STATUS_ACTIVE, _LIVE_STATUS_LEN
    if not _is_host0():
        return
    if _LIVE_STATUS_ACTIVE:
        print("", flush=True)
        _LIVE_STATUS_ACTIVE = False
        _LIVE_STATUS_LEN = 0


def _status(message, persist=True):
    global _LIVE_STATUS_ACTIVE, _LIVE_STATUS_LEN
    del persist
    if not _is_host0():
        return
    line = _clip_status_line(message)
    pad = max(0, _LIVE_STATUS_LEN - len(line))
    print("\r" + line + (" " * pad), end="", flush=True)
    _LIVE_STATUS_ACTIVE = True
    _LIVE_STATUS_LEN = len(line)


def _log(message):
    if _is_host0():
        _finish_status_line()
        print(message, flush=True)


def _maybe_initialize_jax_distributed():
    try:
        initialized = bool(
            getattr(jax.distributed, "is_initialized", lambda: False)())
    except Exception:
        initialized = False
    if initialized:
        return
    try:
        print("Initializing jax.distributed with Cloud TPU auto-detection.",
              flush=True)
        jax.distributed.initialize()
        print(
            "Initialized jax.distributed: "
            f"process_index={jax.process_index()} "
            f"process_count={jax.process_count()}",
            flush=True)
    except RuntimeError as exc:
        if "already initialized" not in str(exc).lower():
            raise


def _enable_xla_dump(dump_dir):
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    os.environ["XLA_DUMP_DIR"] = dump_dir
    flags = os.environ.get("XLA_FLAGS", "")
    dump_flag = f"--xla_dump_to={dump_dir}"
    if dump_flag not in flags:
        flags = (flags + " " + dump_flag).strip()
    if "--xla_dump_hlo_as_text" not in flags:
        flags = (flags + " --xla_dump_hlo_as_text").strip()
    os.environ["XLA_FLAGS"] = flags


def _normalize_xla_dump_dir(template, config_count):
    if not template:
        return None
    if config_count > 1 and ("{index}" in template or "{config}" in template):
        return template.format(index="all", config="combined")
    return template


def _resolve_config(path_text):
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    candidate = PROJECT_ROOT / path_text
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Config file not found: {path_text}")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_benchmark_overrides(cfg, overrides):
    cfg = deepcopy(cfg)
    if not overrides:
        return cfg
    model_cfg = cfg.setdefault("model", {})
    train_cfg = cfg.setdefault("training", {})
    if "sector_capacity_mult" in overrides:
        train_cfg["benchmark_sector_bucket_capacity_mult"] = float(
            overrides["sector_capacity_mult"])
    if "hardware_repack_max_move_frac" in overrides:
        train_cfg["hardware_repack_max_move_frac"] = float(
            overrides["hardware_repack_max_move_frac"])
    if "hardware_repack_vq_iterations" in overrides:
        train_cfg["hardware_repack_vq_iterations"] = int(
            overrides["hardware_repack_vq_iterations"])
    for key in ("v_top_blocks", "rst_top_blocks",
                "v_block_size", "rst_block_size"):
        if key in overrides:
            model_cfg[key] = int(overrides[key])
    return cfg


def format_output_template(template, run_index, config_path):
    if not template:
        return None
    safe_config = Path(config_path).stem
    return str(template).format(index=run_index, config=safe_config)


def tree_map_with_path(fn, tree):
    mapper = getattr(jax.tree, "map_with_path", None)
    if mapper is not None:
        return mapper(fn, tree)
    return jax.tree_util.tree_map_with_path(fn, tree)


def run_one_config(config_arg, args, xla_dump_dir, overrides=None,
                   variant_label="", run_index=1, run_count=1):
    config_path = _resolve_config(config_arg)
    cfg = _load_yaml(config_path)
    cfg = apply_benchmark_overrides(cfg, overrides or {})
    model_version = str(cfg.get("model", {}).get("model_version", ""))
    if args.model_version and model_version != args.model_version:
        if not args.allow_model_version_override:
            raise SystemExit(
                "Config model.model_version disagrees with --model-version: "
                f"{model_version!r} != {args.model_version!r}")
        cfg.setdefault("model", {})["model_version"] = args.model_version
        model_version = args.model_version
    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise SystemExit(
            f"Unsupported model_version={model_version!r}; "
            f"supported={SUPPORTED_MODEL_VERSIONS}")

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    batch_size = int(train_cfg["batch_size"])
    seq_len = int(model_cfg.get("max_seq_len", 512))
    mesh_model = int(train_cfg.get("mesh_model", 1))
    mesh_data = int(train_cfg.get("mesh_data", 0))
    if mesh_data == 0:
        mesh_data = jax.device_count() // mesh_model
    mesh = create_mesh(mesh_data, mesh_model)
    data_sharding = NamedSharding(mesh, P("data", None))
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    if batch_size % n_hosts != 0:
        raise ValueError(
            f"training.batch_size ({batch_size}) must be divisible by "
            f"jax.process_count() ({n_hosts})")
    per_host_batch = batch_size // n_hosts

    _log("Benchmark target:")
    _log(f"  script: scripts/benchmark_srw_tpu.py")
    _log(f"  config: {config_path}")
    _log(f"  model_version: {model_version}")
    _log(f"  global_devices: {jax.device_count()}")
    _log(f"  local_devices: {jax.local_device_count()}")
    _log(f"  mesh_data: {mesh_data}")
    _log(f"  mesh_model: {mesh_model}")
    _log(f"  batch_size: {batch_size}")
    _log(f"  per_host_batch: {per_host_batch}")
    _log(f"  seq_len: {seq_len}")
    _log(f"  warmup_steps: {args.warmup_steps}")
    _log(f"  measure_steps: {args.steps}")
    if variant_label:
        _log(f"  variant: {variant_label}")
    if overrides:
        for key, value in overrides.items():
            _log(f"  override/{key}: {value}")
    if xla_dump_dir:
        _log(f"  xla_dump_dir: {xla_dump_dir}")
    metrics_jsonl = format_output_template(
        args.metrics_jsonl, run_index, config_path)
    if metrics_jsonl:
        _log(f"  metrics_jsonl: {metrics_jsonl}")

    model = build_model(cfg)
    sharded_fns = build_sharded_fns(cfg, mesh)
    train_loader = build_loader(
        cfg, batch_size, seq_len, n_hosts, host_id,
        total_steps=(
            args.steps + args.warmup_steps
            + int(args.post_repack_warmup_steps) + 2),
        dummy_data=args.dummy_data)
    iterator = iter(train_loader)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy = jnp.ones((1, seq_len), dtype=jnp.int32)
    _log("Initializing model params...")
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy,
        labels=dummy,
        attention_mask=jnp.ones_like(dummy),
        deterministic=True,
        sharded_fns=sharded_fns,
        minimal_train=True,
    )
    params = variables["params"]
    params = shard_params_to_mesh(params, get_param_shardings(params, mesh))

    optax_mod = require_optax()
    optimizer = optax_mod.adamw(
        learning_rate=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    opt_state = optimizer.init(params)

    train_step = create_benchmark_train_step(model, optimizer, sharded_fns, cfg)
    prev_metrics = None
    _log("Compiling benchmark step...")
    batch, iterator = next_batch(iterator, train_loader)
    ids, mask = shard_batch_to_mesh(batch, data_sharding, batch_size, seq_len)
    rng, step_rng = jax.random.split(rng)
    compile_t0 = time.perf_counter()
    params, opt_state, prev_metrics = train_step(
        params, opt_state, ids, mask, step_rng, jnp.asarray(0, jnp.int32))
    block_until_ready(prev_metrics)
    compile_seconds = time.perf_counter() - compile_t0
    hbm_after = collect_hbm_stats()
    _log(
        "[bench] compile "
        f"{compile_seconds:.3f}s "
        f"HBM={fmt(hbm_after.get('hbm_used_gb'))}G "
        f"peak={fmt(hbm_after.get('hbm_peak_gb'))}G")

    warmup_times = []
    measure_times = []
    measure_tokens = []
    measure_sector_records = []
    vq_layout_records = []
    forced_repack_done = False
    repack_metrics_host = {}
    repack_seconds = None
    post_repack_warmup_times = []
    peak_hbm = hbm_after.get("hbm_peak_gb")
    execution_step_no = 0

    def run_benchmark_step(phase, phase_step, phase_total, step_no):
        nonlocal params, opt_state, iterator, rng, peak_hbm, execution_step_no
        del step_no
        execution_step_no += 1
        batch, iterator = next_batch(iterator, train_loader)
        ids, mask = shard_batch_to_mesh(
            batch, data_sharding, batch_size, seq_len)
        rng, step_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        params, opt_state, metrics = train_step(
            params, opt_state, ids, mask, step_rng,
            jnp.asarray(execution_step_no, jnp.int32))
        block_until_ready(metrics)
        seconds = time.perf_counter() - t0
        hbm = collect_hbm_stats()
        if hbm.get("hbm_peak_gb") is not None:
            peak_hbm = (
                hbm["hbm_peak_gb"] if peak_hbm is None
                else max(peak_hbm, hbm["hbm_peak_gb"]))
        tokens_per_second = float(batch_size) * float(seq_len) / seconds
        metrics_host = jax.device_get(metrics)
        write_jsonl_record(metrics_jsonl, benchmark_step_record(
            run_index, variant_label, execution_step_no, phase, phase_step,
            seconds, tokens_per_second, metrics_host, hbm))
        _status(format_step_status(
            phase, phase_step, phase_total, seconds, tokens_per_second,
            metrics_host, hbm), persist=True)
        step_sector_records = []
        if model_version == V4168_MODEL_VERSION:
            step_sector_records = sector_runtime_records(
                metrics_host, cfg, execution_step_no, phase, run_index,
                variant_label, seconds, hbm)
            write_jsonl_records(metrics_jsonl, step_sector_records)
        return seconds, tokens_per_second, step_sector_records

    total_to_run = int(args.warmup_steps) + int(args.steps)
    for i in range(total_to_run):
        phase = "warmup" if i < int(args.warmup_steps) else "measure"
        step_no = i + 1
        if (not forced_repack_done
                and phase == "measure"
                and model_version == V4168_MODEL_VERSION
                and args.benchmark_force_vq_repack_before_measure):
            _status(
                "[vq] forcing v4168 V/RST repack before measured steps...",
                persist=True)
            (params, opt_state, new_layout_records, repack_metrics_host,
             repack_seconds) = (
                force_vq_repack_before_measure(
                    params, opt_state, cfg, model_version, mesh, step_no,
                    run_index, variant_label, metrics_jsonl))
            vq_layout_records.extend(new_layout_records)
            forced_repack_done = True
            for post_i in range(int(args.post_repack_warmup_steps)):
                post_seconds, _post_tokens, _post_sector_records = (
                    run_benchmark_step(
                        "post_repack_warmup", post_i + 1,
                        int(args.post_repack_warmup_steps),
                        step_no + post_i))
                post_repack_warmup_times.append(post_seconds)
        phase_step = step_no if phase == "warmup" else len(measure_times) + 1
        phase_total = int(args.warmup_steps) if phase == "warmup" else int(args.steps)
        seconds, tokens_per_second, step_sector_records = run_benchmark_step(
            phase, phase_step, phase_total, step_no)
        if phase == "warmup":
            warmup_times.append(seconds)
        else:
            measure_times.append(seconds)
            measure_tokens.append(tokens_per_second)
            if model_version == V4168_MODEL_VERSION:
                measure_sector_records.extend(step_sector_records)

    xla_report = collect_xla_memory_report(xla_dump_dir) if xla_dump_dir else {}
    sector_summary = summarize_sector_records(measure_sector_records)
    compute_summary = summarize_compute_records(measure_sector_records)
    benchmark_valid = all(
        record.get("benchmark_valid", True)
        for record in measure_sector_records)
    invalid_reason = (
        "sector_bucket_overflow"
        if measure_sector_records and not benchmark_valid else None)
    summary = {
        "config_path": config_arg,
        "variant": variant_label,
        "overrides": dict(overrides or {}),
        "model_version": model_version,
        "compile_seconds": compile_seconds,
        "warmup_steps": int(args.warmup_steps),
        "post_repack_warmup_steps": int(
            args.post_repack_warmup_steps
            if forced_repack_done else 0),
        "measure_steps": int(args.steps),
        "repack_seconds": repack_seconds,
        "post_repack_warmup_mean_step_seconds": mean(
            post_repack_warmup_times),
        "mean_step_seconds": mean(measure_times),
        "mean_step_seconds_without_repack": mean(measure_times),
        "mean_step_seconds_plus_amortized_repack_100": (
            mean(measure_times) + (float(repack_seconds) / 100.0)
            if measure_times and repack_seconds is not None
            else mean(measure_times)),
        "median_step_seconds": median(measure_times),
        "p90_step_seconds": percentile(measure_times, 90),
        "min_step_seconds": min(measure_times) if measure_times else None,
        "max_step_seconds": max(measure_times) if measure_times else None,
        "mean_tokens_per_second": mean(measure_tokens),
        "peak_hbm_gb": peak_hbm,
        "hbm_limit_gb": hbm_after.get("hbm_limit_gb"),
        "xla_total_hbm_usage": xla_report.get("total_hbm_usage"),
        "xla_program_hbm_requirement": xla_report.get(
            "program_hbm_requirement"),
        "xla_source_file": xla_report.get("source_file"),
        "benchmark_valid": benchmark_valid,
        "invalid_reason": invalid_reason,
        "forced_repack_before_measure": bool(
            args.benchmark_force_vq_repack_before_measure
            and model_version == V4168_MODEL_VERSION),
        "sector_summary": sector_summary,
        "compute_summary": compute_summary,
        "vq_layout_records": vq_layout_records,
        "repack_metrics": {
            key: float(value)
            for key, value in (repack_metrics_host or {}).items()
        },
    }
    write_jsonl_record(metrics_jsonl, {
        "type": "benchmark_summary",
        "run_index": int(run_index),
        "variant": variant_label,
        **summary,
    })
    print_summary(summary)
    return summary


def build_model(cfg):
    version = str(cfg["model"].get("model_version", ""))
    module_name, class_name = MODEL_MODULES[version]
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = model_kwargs(cfg)
    _log(f"route dims: d_route={kwargs['d_route']}")
    return cls(**kwargs)


def model_kwargs(cfg):
    m = cfg["model"]
    t = cfg["training"]
    version = str(m.get("model_version", ""))
    n_rst = m.get("n_rst", None)
    n_know = m.get("n_know", None)
    if n_rst is None and n_know is not None:
        n_rst = n_know
    kw = {
        "vocab_size": int(m.get("vocab_size", 30522)),
        "d_model": int(m["d_model"]),
        "n_layers": int(m["n_layers"]),
        "n_heads": int(m["n_heads"]),
        "max_seq_len": int(m.get("max_seq_len", 512)),
        "dropout_rate": float(m.get("dropout", m.get("dropout_rate", 0.0))),
        "gradient_checkpointing": bool(m.get("gradient_checkpointing", False)),
        "d_route": int(m.get("d_route", 256)),
        "n_qk": int(m.get("n_qk", m.get("n_q", 0))),
        "n_v": int(m.get("n_v", 0)),
        "n_rst": int(n_rst) if n_rst is not None else None,
        "n_know": int(n_know) if n_know is not None else None,
        "router_dropout": float(m.get("router_dropout", 0.0)),
        "n_chunks_qk": int(t.get("n_chunks_qk", 1)),
        "n_chunks_v": int(t.get("n_chunks_v", 1)),
        "n_chunks_rst": int(t.get("n_chunks_rst", t.get("n_chunks_know", 1))),
        "tau_init_attn_qk": float(t.get("tau_init_attn_qk", m.get("tau_init_attn_qk", 0.0)) or 0.0),
        "tau_init_attn_v": float(t.get("tau_init_attn_v", m.get("tau_init_attn_v", 0.0)) or 0.0),
        "tau_init_rst": float(t.get("tau_init_rst", m.get("tau_init_rst", 0.0)) or 0.0),
    }
    if version == V4168_MODEL_VERSION:
        kw.update({
            "qk_block_size": int(m.get("qk_block_size", 256)),
            "v_block_size": int(m.get("v_block_size", 256)),
            "rst_block_size": int(m.get("rst_block_size", 256)),
            "qk_top_blocks": int(m.get("qk_top_blocks", 2)),
            "v_top_blocks": int(m.get("v_top_blocks", 2)),
            "rst_top_blocks": int(m.get("rst_top_blocks", 2)),
            "block_margin": float(m.get("block_margin", 0.0)),
        })
    return kw


def build_sharded_fns(cfg, mesh):
    version = str(cfg["model"].get("model_version", ""))
    module_name, _class_name = MODEL_MODULES[version]
    module = importlib.import_module(module_name)
    capacity_mult = cfg["training"].get(
        "benchmark_sector_bucket_capacity_mult", None)
    if version == V4168_MODEL_VERSION and capacity_mult is not None:
        setattr(module, "_V4168_SECTOR_BUCKET_CAPACITY_MULT",
                float(capacity_mult))
        _log(f"v4168 bucket_capacity_mult override: {float(capacity_mult)}")
    make_single = getattr(module, "make_sharded_srw_minimal", None)
    make_paired = getattr(module, "make_sharded_srw_paired_minimal", None)
    if make_single is None or make_paired is None:
        raise RuntimeError(f"{module_name} does not expose minimal SRW factories.")
    t = cfg["training"]
    m = cfg["model"]
    max_chunk = t.get("max_chunk_size", None)
    qk_chunk = int(t.get("attn_qk_max_chunk_size", max_chunk or 2048))
    v_chunk = int(t.get("attn_v_max_chunk_size", max_chunk or 2048))
    rst_chunk = int(t.get("rst_max_chunk_size", max_chunk or 2048))
    base_kwargs = {
        "mesh": mesh,
        "dead_exposure_target": 0.0,
        "soft_gate_effective_active_eps": float(
            t.get("soft_gate_effective_active_eps", 1.0e-6)),
        "admission_den_power": float(t.get("admission_den_power", 1.0)),
        "admission_den_grad_scale": float(
            t.get("admission_den_grad_scale", 1.0)),
    }

    def filtered(factory, kwargs):
        sig = inspect.signature(factory)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    def pool_kwargs(pool):
        kwargs = dict(base_kwargs)
        if version == V4168_MODEL_VERSION:
            kwargs.update({
                "block_size": int(m.get(f"{pool}_block_size", 256)),
                "top_blocks": int(m.get(f"{pool}_top_blocks", 2)),
                "block_margin": float(m.get("block_margin", 0.0)),
                "hardware_sector_execution_enabled": bool(
                    t.get("hardware_sector_execution_enabled",
                          t.get("hardware_repack_enabled", False))),
                "hardware_sector_debug_token_gather_fallback": bool(
                    t.get("hardware_sector_debug_token_gather_fallback", False)),
                "benchmark_runtime_metrics": True,
            })
        return kwargs

    attn_v_single = make_single(
        max_chunk_size=v_chunk,
        **filtered(make_single, pool_kwargs("v")))
    rst_single = make_single(
        max_chunk_size=rst_chunk,
        **filtered(make_single, pool_kwargs("rst")))
    attn_qk_single = None
    if version != V4168_MODEL_VERSION:
        attn_qk_single = make_single(
            max_chunk_size=qk_chunk,
            **filtered(make_single, pool_kwargs("qk")))
    attn_qk_paired = make_paired(
        max_chunk_size=qk_chunk,
        **filtered(make_paired, pool_kwargs("qk")))
    out = {
        "single": attn_v_single,
        "attn_v_single": attn_v_single,
        "rst_single": rst_single,
        "paired": attn_qk_paired,
        "attn_qk_paired": attn_qk_paired,
        "attn_v_single_minimal": attn_v_single,
        "rst_single_minimal": rst_single,
        "attn_qk_paired_minimal": attn_qk_paired,
    }
    if attn_qk_single is not None:
        out["attn_qk_single_minimal"] = attn_qk_single
    _log("sharded SRW kernels: enabled")
    return out


def create_benchmark_train_step(model, optimizer, sharded_fns, cfg):
    optax_mod = require_optax()
    t = cfg["training"]
    version = str(cfg["model"].get("model_version", ""))
    soft_gate_t = float(t.get("soft_gate_temperature", 0.07))
    boundary_power = float(t.get("soft_gate_boundary_power_final",
                                 t.get("soft_gate_boundary_power_mid",
                                       t.get("soft_gate_boundary_power_start", 4.0))))
    admission_den_power = float(t.get("admission_den_power", 1.0))

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, rng, step):
        del step

        def loss_fn(p):
            apply_kwargs = {
                "labels": input_ids,
                "attention_mask": attention_mask,
                "deterministic": True,
                "rngs": {"dropout": rng},
                "sharded_fns": sharded_fns,
                "minimal_train": True,
                "soft_gate_temperature": soft_gate_t,
                "soft_gate_t_final": soft_gate_t,
                "soft_gate_T_qk": soft_gate_t,
                "soft_gate_T_v": soft_gate_t,
                "soft_gate_T_rst": soft_gate_t,
                "soft_gate_boundary_power": boundary_power,
                "soft_gate_boundary_power_final": boundary_power,
                "admission_den_power": admission_den_power,
                "execution_prune_eps": 0.0,
            }
            if version == V4168_MODEL_VERSION:
                apply_kwargs["benchmark_runtime_metrics"] = True
            result = model.apply({"params": p}, input_ids, **apply_kwargs)
            loss = result["loss"] + result.get("aux_loss", jnp.float32(0.0))
            return loss, result

        (loss, result), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax_mod.apply_updates(params, updates)
        metrics = {
            "loss": loss,
            "ce_loss": result["loss"],
            "aux_loss": result.get("aux_loss", jnp.float32(0.0)),
            "grad_norm": optax_mod.global_norm(grads),
            "valid_count": result.get("valid_count", jnp.float32(0.0)),
        }
        for pool in ("attn_v", "rst"):
            for name in SECTOR_RUNTIME_METRIC_NAMES:
                key = f"sector/{pool}/{name}"
                if key in result:
                    metrics[key] = result[key]
        return params, opt_state, metrics

    return train_step


def create_mesh(mesh_data, mesh_model):
    devices = jax.devices()
    if len(devices) != mesh_data * mesh_model:
        raise ValueError(
            f"mesh_data({mesh_data}) * mesh_model({mesh_model}) != "
            f"device_count({len(devices)})")
    return Mesh(np.array(devices).reshape(mesh_data, mesh_model),
                ("data", "model"))


def get_param_shardings(params, mesh):
    replicated = NamedSharding(mesh, P())
    n_sharded = NamedSharding(mesh, P("model", None))
    n_sharded_3d = NamedSharding(mesh, P("model", None, None))

    def choose(path, value):
        path_str = "/".join(
            str(p.key if hasattr(p, "key") else p) for p in path)
        if "neuron_pool" in path_str:
            if value.ndim == 2:
                return n_sharded
            if value.ndim == 3:
                return n_sharded_3d
        return replicated

    return tree_map_with_path(lambda path, x: choose(path, x), params)


def shard_params_to_mesh(params, shardings):
    return jax.tree.map(lambda p, s: jax.device_put(p, s), params, shardings)


def shard_batch_to_mesh(batch, sharding, batch_size, seq_len):
    ids, mask = batch
    return (
        shard_to_mesh(ids, sharding, (batch_size, seq_len)),
        shard_to_mesh(mask, sharding, (batch_size, seq_len)),
    )


def shard_to_mesh(data, sharding, global_shape):
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    per_host = data.shape[0]

    def callback(index):
        batch_slice = index[0]
        start = batch_slice.start or 0
        stop = batch_slice.stop or global_shape[0]
        local_start = start - host_id * per_host
        local_stop = stop - host_id * per_host
        if 0 <= local_start < per_host:
            return np.array(data[local_start:local_stop])
        raise RuntimeError(
            f"shard_to_mesh request [{start}, {stop}) outside host "
            f"{host_id} local range [0, {per_host}) across {n_hosts} hosts")

    return jax.make_array_from_callback(global_shape, sharding, callback)


class DummyLoader:
    def __init__(self, batch_size, seq_len, vocab_size, n_hosts, host_id, steps):
        self.global_batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.vocab_size = max(2, int(vocab_size))
        self.n_hosts = int(n_hosts)
        self.host_id = int(host_id)
        self.steps = max(1, int(steps))
        if self.global_batch_size % self.n_hosts != 0:
            raise ValueError(
                f"batch_size ({self.global_batch_size}) must be divisible by "
                f"n_hosts ({self.n_hosts})")
        self.per_host_batch = self.global_batch_size // self.n_hosts

    def __len__(self):
        return self.steps

    def reset(self, start_step=0):
        del start_step

    def __iter__(self):
        base = np.arange(
            self.per_host_batch * self.seq_len, dtype=np.int32).reshape(
                self.per_host_batch, self.seq_len)
        for step in range(self.steps):
            ids = (base + step + self.host_id * 997) % self.vocab_size
            yield jnp.asarray(ids, jnp.int32), jnp.ones_like(
                jnp.asarray(ids, jnp.int32))


def build_loader(cfg, batch_size, seq_len, n_hosts, host_id, total_steps,
                 dummy_data=False):
    if dummy_data:
        return DummyLoader(
            batch_size, seq_len, cfg["model"].get("vocab_size", 30522),
            n_hosts, host_id, total_steps)
    from utils.data_jax import load_data
    train_loader, _val_loader, _vocab_size = load_data(
        cfg["data"],
        max_length=seq_len,
        batch_size=batch_size,
        n_devices=1,
        n_hosts=n_hosts,
        host_id=host_id,
    )
    return train_loader


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        if hasattr(loader, "reset"):
            loader.reset()
        iterator = iter(loader)
        return next(iterator), iterator


def block_until_ready(metrics):
    if isinstance(metrics, dict) and "loss" in metrics:
        jax.block_until_ready(metrics["loss"])
    else:
        jax.block_until_ready(jax.tree.leaves(metrics))


def collect_hbm_stats():
    try:
        mem = jax.local_devices()[0].memory_stats()
    except Exception as exc:
        return {"warning": f"{type(exc).__name__}: {exc}"}
    if not mem:
        return {}
    return {
        "hbm_used_gb": (
            float(mem["bytes_in_use"]) / 1e9
            if "bytes_in_use" in mem else None),
        "hbm_peak_gb": (
            float(mem["peak_bytes_in_use"]) / 1e9
            if "peak_bytes_in_use" in mem else None),
        "hbm_limit_gb": (
            float(mem["bytes_limit"]) / 1e9
            if "bytes_limit" in mem else None),
    }


def write_jsonl_record(path, record):
    if not path or not _is_host0():
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_jsonl_records(path, records):
    for record in records:
        write_jsonl_record(path, record)


def json_float(value, default=0.0):
    value = num(value)
    if value is None:
        return default
    return float(value)


def pool_static_info(cfg, pool):
    m = cfg["model"]
    t = cfg["training"]
    mesh_model = int(t.get("mesh_model", 1))
    if pool == "attn_v":
        n_key = "n_v"
        block_key = "v_block_size"
        top_key = "v_top_blocks"
    elif pool == "rst":
        n_key = "n_rst"
        block_key = "rst_block_size"
        top_key = "rst_top_blocks"
    else:
        n_key = "n_qk"
        block_key = "qk_block_size"
        top_key = "qk_top_blocks"
    n_ops = int(m.get(n_key, m.get("n_know", 0)))
    block_size = int(m.get(block_key, 256))
    top_blocks = int(m.get(top_key, 2))
    n_local_ops = max(1, (n_ops + mesh_model - 1) // mesh_model)
    n_local_sectors = max(1, (n_local_ops + block_size - 1) // block_size)
    num_global_sectors = n_local_sectors * mesh_model
    return {
        "block_size": block_size,
        "top_blocks": min(max(1, top_blocks), num_global_sectors),
        "num_global_sectors": num_global_sectors,
        "num_local_sectors": n_local_sectors,
    }


def sector_runtime_records(metrics, cfg, step, phase, run_index,
                           variant_label, step_seconds, hbm):
    records = []
    for pool in ("attn_v", "rst"):
        prefix = f"sector/{pool}/"
        if prefix + "overflow_count" not in metrics:
            continue
        info = pool_static_info(cfg, pool)
        expected = json_float(
            metrics.get(prefix + "expected_selected_pair_count"))
        top_blocks = max(1.0, float(info["top_blocks"]))
        record = {
            "type": "sector_runtime",
            "run_index": int(run_index),
            "variant": variant_label,
            "step": int(step),
            "phase": phase,
            "pool": pool,
            "block_size": int(info["block_size"]),
            "top_blocks": int(info["top_blocks"]),
            "num_global_sectors": int(info["num_global_sectors"]),
            "num_local_sectors": int(info["num_local_sectors"]),
            "pair_span": expected / top_blocks if expected else 0.0,
            "step_seconds": float(step_seconds),
            "hbm_used_gb": json_float(hbm.get("hbm_used_gb"), None),
            "hbm_peak_gb": json_float(hbm.get("hbm_peak_gb"), None),
        }
        for name in SECTOR_RUNTIME_METRIC_NAMES:
            key = prefix + name
            if key in metrics:
                record[name] = json_float(metrics[key])
        record["benchmark_valid"] = record.get("overflow_count", 0.0) <= 0.0
        if not record["benchmark_valid"]:
            record["invalid_reason"] = "sector_bucket_overflow"
        records.append(record)
    return records


def hbm_inline(hbm):
    used = hbm.get("hbm_used_gb")
    peak = hbm.get("hbm_peak_gb")
    limit = hbm.get("hbm_limit_gb")
    if used is None:
        return "HBM=n/a"
    if limit is not None:
        return f"HBM={fmt(used)}G/{fmt(limit)}G peak={fmt(peak)}G"
    return f"HBM={fmt(used)}G peak={fmt(peak)}G"


def sector_brief(metrics, pool):
    prefix = f"sector/{pool}/"
    if prefix + "overflow_frac" not in metrics:
        return ""
    label = "v" if pool == "attn_v" else pool
    return (
        f"{label}[ovf={fmt(metrics.get(prefix + 'overflow_frac'), 3)} "
        f"util99={fmt(metrics.get(prefix + 'bucket_capacity_util_p99'), 3)} "
        f"short={fmt(metrics.get(prefix + 'capacity_shortfall'), 2)} "
        f"pad={fmt(metrics.get(prefix + 'padded_work_frac_vs_dense'), 3)}]"
    )


def benchmark_step_record(run_index, variant_label, step, phase, phase_step,
                          step_seconds, tokens_per_second, metrics, hbm):
    return {
        "type": "benchmark_step",
        "run_index": int(run_index),
        "variant": variant_label,
        "step": int(step),
        "phase": phase,
        "phase_step": int(phase_step),
        "step_seconds": float(step_seconds),
        "tokens_per_second": float(tokens_per_second),
        "loss": json_float(metrics.get("loss"), None),
        "grad_norm": json_float(metrics.get("grad_norm"), None),
        "hbm_used_gb": json_float(hbm.get("hbm_used_gb"), None),
        "hbm_peak_gb": json_float(hbm.get("hbm_peak_gb"), None),
        "hbm_limit_gb": json_float(hbm.get("hbm_limit_gb"), None),
    }


def format_step_status(phase, phase_step, phase_total, step_seconds,
                       tokens_per_second, metrics, hbm):
    sector_parts = [
        part for part in (
            sector_brief(metrics, "attn_v"),
            sector_brief(metrics, "rst"),
        )
        if part
    ]
    sector_text = " | " + " ".join(sector_parts) if sector_parts else ""
    return (
        f"[bench] {phase} {int(phase_step)}/{int(phase_total)} "
        f"step_s={step_seconds:.4f} tok/s={tokens_per_second:.1f} "
        f"loss={fmt(metrics.get('loss'), 4)} "
        f"grad={fmt(metrics.get('grad_norm'), 3)} | "
        f"{hbm_inline(hbm)}{sector_text}"
    )


def summarize_sector_records(records):
    by_pool = {}
    for record in records:
        by_pool.setdefault(record["pool"], []).append(record)
    summaries = {}
    for pool, items in by_pool.items():
        summaries[pool] = {
            "overflow_frac": max(
                item.get("overflow_frac", 0.0) for item in items),
            "executed_pair_frac": min(
                item.get("executed_selected_pair_frac", 0.0)
                for item in items),
            "bucket_util_p99": max(
                item.get("bucket_capacity_util_p99", 0.0)
                for item in items),
            "bucket_fill_max": max(
                item.get("bucket_fill_max", 0.0) for item in items),
            "capacity": max(item.get("bucket_capacity", 0.0)
                            for item in items),
            "attempted_fill_p99": max(
                item.get("attempted_fill_p99", 0.0) for item in items),
            "attempted_fill_max": max(
                item.get("attempted_fill_max", 0.0) for item in items),
            "required_capacity_no_overflow": max(
                item.get("required_capacity_no_overflow", 0.0)
                for item in items),
            "capacity_shortfall": max(
                item.get("capacity_shortfall", 0.0) for item in items),
        }
    return summaries


def summarize_compute_records(records):
    by_pool = {}
    for record in records:
        by_pool.setdefault(record["pool"], []).append(record)
    summaries = {}
    for pool, items in by_pool.items():
        summaries[pool] = {
            "semantic_frac": mean([
                item.get("semantic_work_frac_vs_dense", 0.0)
                for item in items]),
            "padded_frac": mean([
                item.get("padded_work_frac_vs_dense", 0.0)
                for item in items]),
            "batch_union_frac": mean([
                item.get("batch_union_effective_operator_frac", 0.0)
                for item in items]),
            "per_token_frac": mean([
                item.get("per_token_effective_operator_frac", 0.0)
                for item in items]),
        }
    return summaries


def benchmark_repack_config(cfg):
    t = cfg["training"]
    return {
        "hardware_repack_enabled": True,
        "hardware_sector_execution_enabled": bool(
            t.get("hardware_sector_execution_enabled",
                  t.get("hardware_repack_enabled", True))),
        "hardware_repack_strategy": str(
            t.get("hardware_repack_strategy", "balanced_vq")).lower(),
        "hardware_repack_farthest_per_sector": int(
            t.get("hardware_repack_farthest_per_sector", 10)),
        "hardware_repack_gain_eps": float(
            t.get("hardware_repack_gain_eps", 1.0e-3)),
        "hardware_repack_max_move_frac": float(
            t.get("hardware_repack_max_move_frac", 0.08)),
        "hardware_repack_vq_iterations": int(
            t.get("hardware_repack_vq_iterations", 4)),
    }


def collect_vq_layout_records(params, cfg, model_version, phase, run_index,
                              variant_label, repack_metrics=None,
                              repack_seconds=None):
    if model_version != V4168_MODEL_VERSION:
        return []
    module_name, _class_name = MODEL_MODULES[model_version]
    module = importlib.import_module(module_name)
    records = []
    pool_params = params["neuron_pool"]
    op_keys = module._pool_operator_keys(pool_params)
    for pool, op_key_name, block_key in (
            ("attn_v", "attn_v_op_key", "v_block_size"),
            ("rst", "rst_op_key", "rst_block_size")):
        block_size = int(cfg["model"].get(block_key, 256))
        key_device = module._forward_unit_direction(
            op_keys[op_key_name].astype(jnp.float32))
        key_host = module._global_jax_array_to_host_np(
            key_device, dtype=np.float32)
        quality = module._sector_layout_quality_np(key_host, block_size)
        prefix = f"repack/{pool}/"
        moved_frac = json_float(
            (repack_metrics or {}).get(prefix + "moved_frac"), 0.0)
        record = {
            "type": "vq_layout",
            "run_index": int(run_index),
            "variant": variant_label,
            "phase": phase,
            "pool": pool,
            "block_size": block_size,
            "moved_frac": moved_frac,
            "applied_moved_frac": moved_frac,
            "full_vq_moved_frac": json_float(
                (repack_metrics or {}).get(prefix + "full_vq_moved_frac"),
                moved_frac),
            "max_move_frac": json_float(
                (repack_metrics or {}).get(
                    prefix + "max_move_frac",
                    (repack_metrics or {}).get("repack/max_move_frac")),
                0.0),
            "vq_iterations": json_float(
                (repack_metrics or {}).get(
                    prefix + "vq_iterations",
                    (repack_metrics or {}).get("repack/vq_iterations")),
                0.0),
            "mean_compactness_cos": float(
                quality.get("mean_compactness_cos", 0.0)),
            "mean_sector_radius": float(
                quality.get("mean_sector_radius", 0.0)),
            "max_sector_radius": float(
                quality.get("max_sector_radius", 0.0)),
        }
        if repack_seconds is not None:
            record["repack_seconds"] = float(repack_seconds)
        records.append(record)
    return records


def _records_by_pool(records):
    return {record.get("pool"): record for record in records}


def print_vq_repack_lines(before_records, after_records, repack_metrics,
                          repack_seconds):
    if not _is_host0():
        return
    _finish_status_line()
    before_by_pool = _records_by_pool(before_records)
    after_by_pool = _records_by_pool(after_records)
    for pool in ("attn_v", "rst"):
        before = before_by_pool.get(pool, {})
        after = after_by_pool.get(pool, {})
        prefix = f"repack/{pool}/"
        applied = json_float(
            (repack_metrics or {}).get(prefix + "moved_frac"),
            after.get("applied_moved_frac", 0.0))
        full = json_float(
            (repack_metrics or {}).get(prefix + "full_vq_moved_frac"),
            after.get("full_vq_moved_frac", applied))
        max_move = json_float(
            (repack_metrics or {}).get(
                prefix + "max_move_frac",
                (repack_metrics or {}).get("repack/max_move_frac")),
            after.get("max_move_frac", 0.0))
        iterations = json_float(
            (repack_metrics or {}).get(
                prefix + "vq_iterations",
                (repack_metrics or {}).get("repack/vq_iterations")),
            after.get("vq_iterations", 0.0))
        print(
            "[vq] "
            f"pool={pool} "
            f"before_compact={fmt(before.get('mean_compactness_cos'), 6)} "
            f"after_compact={fmt(after.get('mean_compactness_cos'), 6)} "
            f"before_mean_radius={fmt(before.get('mean_sector_radius'), 6)} "
            f"after_mean_radius={fmt(after.get('mean_sector_radius'), 6)} "
            f"before_max_radius={fmt(before.get('max_sector_radius'), 6)} "
            f"after_max_radius={fmt(after.get('max_sector_radius'), 6)} "
            f"full_vq_moved_frac={fmt(full, 6)} "
            f"applied_moved_frac={fmt(applied, 6)} "
            f"max_move_frac={fmt(max_move, 3)} "
            f"vq_iterations={fmt(iterations, 0)} "
            f"repack_s={fmt(repack_seconds, 3)}",
            flush=True)


def force_vq_repack_before_measure(params, opt_state, cfg, model_version, mesh,
                                   step, run_index, variant_label,
                                   metrics_jsonl):
    if model_version != V4168_MODEL_VERSION:
        return params, opt_state, [], {}, None
    module_name, _class_name = MODEL_MODULES[model_version]
    module = importlib.import_module(module_name)
    before_records = collect_vq_layout_records(
        params, cfg, model_version, "before_repack", run_index,
        variant_label)
    write_jsonl_records(metrics_jsonl, before_records)
    repack_cfg = benchmark_repack_config(cfg)
    t0 = time.perf_counter()
    params, opt_state, repack_metrics = module.maybe_hardware_repack(
        params, opt_state, cfg["model"], mesh, step, repack_cfg)
    repack_seconds = time.perf_counter() - t0
    after_records = collect_vq_layout_records(
        params, cfg, model_version, "after_repack", run_index,
        variant_label, repack_metrics=repack_metrics,
        repack_seconds=repack_seconds)
    write_jsonl_records(metrics_jsonl, after_records)
    print_vq_repack_lines(
        before_records, after_records, repack_metrics, repack_seconds)
    return (
        params, opt_state, before_records + after_records, repack_metrics,
        repack_seconds)


def collect_xla_memory_report(dump_dir):
    if not dump_dir:
        return {}
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return {}
    files = []
    seen = set()
    for pattern in ("*memory*", "*buffer*", "*after_optimizations.txt", "*.txt"):
        for path in dump_path.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    needles = ("Total hbm usage", "Program hbm requirement",
               "Largest program allocations")
    for path in files[:20]:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if not any(n in text for n in needles):
            continue
        report = {"source_file": str(path)}
        for line in text.splitlines():
            line = line.strip()
            if "Total hbm usage" in line:
                report["total_hbm_usage"] = line
            elif "Program hbm requirement" in line:
                report["program_hbm_requirement"] = line
        return report
    return {}


def print_sector_metrics(metrics):
    if not _is_host0():
        return
    for pool in ("attn_v", "rst"):
        prefix = f"sector/{pool}/"
        if prefix + "overflow_frac" not in metrics:
            continue
        print(
            "[sector-validity] "
            f"{pool} "
            f"overflow_frac={fmt(metrics.get(prefix + 'overflow_frac'), 6)} "
            f"executed_pair_frac="
            f"{fmt(metrics.get(prefix + 'executed_selected_pair_frac'), 6)} "
            f"bucket_util_p99="
            f"{fmt(metrics.get(prefix + 'bucket_capacity_util_p99'), 4)} "
            f"bucket_fill_max={fmt(metrics.get(prefix + 'bucket_fill_max'), 1)} "
            f"capacity={fmt(metrics.get(prefix + 'bucket_capacity'), 1)}",
            flush=True)
        print(
            "[sector-work] "
            f"{pool} "
            f"per_token_effective_operator_frac="
            f"{fmt(metrics.get(prefix + 'per_token_effective_operator_frac'), 6)} "
            f"padded_work_frac_vs_dense="
            f"{fmt(metrics.get(prefix + 'padded_work_frac_vs_dense'), 6)} "
            f"semantic_work_frac_vs_dense="
            f"{fmt(metrics.get(prefix + 'semantic_work_frac_vs_dense'), 6)} "
            f"batch_union_effective_operator_frac="
            f"{fmt(metrics.get(prefix + 'batch_union_effective_operator_frac'), 6)}",
            flush=True)


def print_summary(summary):
    if not _is_host0():
        return
    _finish_status_line()
    print("\n=== SRW Standalone Benchmark Summary ===", flush=True)
    print(f"config={summary['config_path']}", flush=True)
    if summary.get("variant"):
        print(f"variant={summary['variant']}", flush=True)
    print(f"model={summary['model_version']}", flush=True)
    print(
        "valid "
        f"benchmark_valid={str(summary.get('benchmark_valid', True)).lower()} "
        f"reason={summary.get('invalid_reason') or 'none'}",
        flush=True)
    print(
        "step_s "
        f"mean={fmt(summary.get('mean_step_seconds'), 4)} "
        f"median={fmt(summary.get('median_step_seconds'), 4)} "
        f"p90={fmt(summary.get('p90_step_seconds'), 4)} "
        f"min={fmt(summary.get('min_step_seconds'), 4)} "
        f"max={fmt(summary.get('max_step_seconds'), 4)}",
        flush=True)
    if summary.get("repack_seconds") is not None:
        print(
            "repack "
            f"repack_s={fmt(summary.get('repack_seconds'), 3)} "
            f"post_repack_warmup_steps="
            f"{int(summary.get('post_repack_warmup_steps') or 0)} "
            f"post_repack_warmup_mean_step_s="
            f"{fmt(summary.get('post_repack_warmup_mean_step_seconds'), 4)} "
            f"mean_step_s_without_repack="
            f"{fmt(summary.get('mean_step_seconds_without_repack'), 4)} "
            f"mean_step_s_plus_amortized_repack_100="
            f"{fmt(summary.get('mean_step_seconds_plus_amortized_repack_100'), 4)}",
            flush=True)
    print(
        "throughput "
        f"mean_tokens_per_second="
        f"{fmt(summary.get('mean_tokens_per_second'), 1)}",
        flush=True)
    print(
        "memory "
        f"peak_hbm_gb={fmt(summary.get('peak_hbm_gb'))} "
        f"limit_hbm_gb={fmt(summary.get('hbm_limit_gb'))}",
        flush=True)
    if summary.get("xla_total_hbm_usage"):
        print(
            "xla_memory "
            f"{summary.get('xla_total_hbm_usage')} "
            f"{summary.get('xla_program_hbm_requirement') or ''}",
            flush=True)
    if summary.get("sector_summary"):
        print("\n[sector]", flush=True)
        compute = summary.get("compute_summary") or {}
        for pool, row in summary["sector_summary"].items():
            work = compute.get(pool, {})
            print(
                "[sector] "
                f"pool={pool} "
                f"step_s={fmt(summary.get('mean_step_seconds'), 4)} "
                f"overflow_frac={fmt(row.get('overflow_frac'), 6)} "
                f"executed_pair_frac={fmt(row.get('executed_pair_frac'), 6)} "
                f"bucket_util_p99={fmt(row.get('bucket_util_p99'), 4)} "
                f"attempted_fill_p99={fmt(row.get('attempted_fill_p99'), 1)} "
                f"attempted_fill_max={fmt(row.get('attempted_fill_max'), 1)} "
                f"required_capacity_no_overflow="
                f"{fmt(row.get('required_capacity_no_overflow'), 1)} "
                f"current_capacity={fmt(row.get('capacity'), 1)} "
                f"capacity_shortfall={fmt(row.get('capacity_shortfall'), 3)} "
                f"per_token_effective_operator_frac="
                f"{fmt(work.get('per_token_frac'), 6)} "
                f"padded_work_frac_vs_dense={fmt(work.get('padded_frac'), 6)} "
                f"semantic_work_frac_vs_dense={fmt(work.get('semantic_frac'), 6)}",
                flush=True)
    if summary.get("model_version") == V4168_MODEL_VERSION:
        print(
            "\n[vq] "
            f"forced_repack_before_measure="
            f"{str(summary.get('forced_repack_before_measure', False)).lower()}",
            flush=True)
    if summary.get("vq_layout_records"):
        by_phase = {}
        for record in summary["vq_layout_records"]:
            by_phase.setdefault(record.get("phase"), {})[
                record.get("pool")] = record
        before = by_phase.get("before_repack", {})
        after = by_phase.get("after_repack", {})
        for pool in ("attn_v", "rst"):
            if pool not in before and pool not in after:
                continue
            b = before.get(pool, {})
            a = after.get(pool, b)
            print(
                "[vq-summary] "
                f"pool={pool} "
                f"before_compact={fmt(b.get('mean_compactness_cos'), 6)} "
                f"after_compact={fmt(a.get('mean_compactness_cos'), 6)} "
                f"before_mean_radius={fmt(b.get('mean_sector_radius'), 6)} "
                f"after_mean_radius={fmt(a.get('mean_sector_radius'), 6)} "
                f"before_max_radius={fmt(b.get('max_sector_radius'), 6)} "
                f"after_max_radius={fmt(a.get('max_sector_radius'), 6)} "
                f"full_vq_moved_frac={fmt(a.get('full_vq_moved_frac'), 6)} "
                f"applied_moved_frac={fmt(a.get('applied_moved_frac'), 6)} "
                f"max_move_frac={fmt(a.get('max_move_frac'), 3)} "
                f"vq_iterations={fmt(a.get('vq_iterations'), 0)} "
                f"repack_s={fmt(a.get('repack_seconds'), 3)}",
                flush=True)
    print("\n[op-breakdown]", flush=True)
    print(
        "op | mean_ms | hbm_peak_gb | memory_note",
        flush=True)
    print(
        "full_train_step | "
        f"{fmt(num(summary.get('mean_step_seconds')) * 1000.0 if summary.get('mean_step_seconds') is not None else None, 3)} | "
        f"{fmt(summary.get('peak_hbm_gb'))} | "
        "process allocator stats, not exact op-local allocation",
        flush=True)


def print_comparison(summaries):
    _finish_status_line()
    print("\n" + "=" * 72, flush=True)
    print("SRW Benchmark Comparison", flush=True)
    print("=" * 72, flush=True)
    header = (
        f"{'#':>2}  {'model':24s}  {'mean_s':>10s}  "
        f"{'tok/s':>12s}  {'peak_hbm':>10s}  {'valid':>5s}  "
        f"{'variant':28s}  config")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, summary in enumerate(summaries, 1):
        print(
            f"{i:>2}  {summary.get('model_version', ''):24s}  "
            f"{fmt(summary.get('mean_step_seconds'), 4):>10s}  "
            f"{fmt(summary.get('mean_tokens_per_second'), 1):>12s}  "
            f"{fmt(summary.get('peak_hbm_gb')):>10s}  "
            f"{str(summary.get('benchmark_valid', True)).lower():>5s}  "
            f"{summary.get('variant', '')[:28]:28s}  "
            f"{summary.get('config_path', '')}",
            flush=True)
    base = summaries[0]
    print("\nRatios vs first config:", flush=True)
    for i, summary in enumerate(summaries[1:], 2):
        valid_pair = (
            bool(base.get("benchmark_valid", True))
            and bool(summary.get("benchmark_valid", True)))
        raw_mean_ratio = ratio(
            summary.get('mean_step_seconds'), base.get('mean_step_seconds'))
        raw_median_ratio = ratio(
            summary.get('median_step_seconds'),
            base.get('median_step_seconds'))
        step_time_ratio = (
            raw_mean_ratio if valid_pair
            else f"invalid_overflow raw_mean={raw_mean_ratio} raw_median={raw_median_ratio}")
        token_ratio = (
            ratio(summary.get('mean_tokens_per_second'),
                  base.get('mean_tokens_per_second'))
            if valid_pair else "invalid_overflow")
        print(
            f"  #{i}: step_time="
            f"{step_time_ratio} "
            f"tokens="
            f"{token_ratio} "
            f"peak_hbm="
            f"{ratio(summary.get('peak_hbm_gb'), base.get('peak_hbm_gb'))}",
            flush=True)
    if len(summaries) >= 2:
        print("\n[compare]", flush=True)
        base_step = base.get("mean_step_seconds")
        base_median = base.get("median_step_seconds")
        for i, summary in enumerate(summaries[1:], 2):
            valid_pair = (
                bool(base.get("benchmark_valid", True))
                and bool(summary.get("benchmark_valid", True)))
            speed_ratio = (
                ratio(base_step, summary.get("mean_step_seconds"))
                if valid_pair else "invalid_overflow")
            print(
                f"base_mean_step_s={fmt(base_step, 4)} "
                f"run_{i}_mean_step_s={fmt(summary.get('mean_step_seconds'), 4)} "
                f"speed_ratio={speed_ratio} "
                f"raw_median_ratio="
                f"{ratio(summary.get('median_step_seconds'), base_median)} "
                f"valid={str(summary.get('benchmark_valid', True)).lower()}",
                flush=True)


def num(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    return float(np.mean(values)) if values else None


def median(values):
    return float(np.median(values)) if values else None


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def fmt(value, digits=3):
    value = num(value)
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def ratio(a, b):
    a = num(a)
    b = num(b)
    if a is None or b in (None, 0.0):
        return "n/a"
    return f"{a / b:.3f}x"


if __name__ == "__main__":
    main()
