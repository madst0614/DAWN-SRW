#!/usr/bin/env python3
"""Standalone TPU benchmark harness for DAWN-SRW runtime comparisons.

Runs model modules directly and reports benchmark details to stdout.
"""

import argparse
from copy import deepcopy
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

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
V4172_MODEL_VERSION = "spatial-r1-v4.1.7.2"
V4174_MODEL_VERSION = "spatial-r1-v4.1.7.4"
V417X_MODEL_VERSIONS = (V4172_MODEL_VERSION, V4174_MODEL_VERSION)
SUPPORTED_MODEL_VERSIONS = (
    V4166_MODEL_VERSION,
    V4168_MODEL_VERSION,
    *V417X_MODEL_VERSIONS,
)
MODEL_MODULES = {
    V4166_MODEL_VERSION: ("models.dawn_srw_v4166", "DAWN_SRW_V4166"),
    V4168_MODEL_VERSION: ("models.dawn_srw_v4168", "DAWN_SRW_V4168"),
    V4172_MODEL_VERSION: ("models.dawn_srw_v4172", "DAWN_SRW_V4172"),
    V4174_MODEL_VERSION: ("models.dawn_srw_v4174", "DAWN_SRW_V4174"),
}
RUNTIME_POOLS = ("attn_qk", "attn_v", "rst")
RUNTIME_POOL_LABELS = {
    "attn_qk": "qk",
    "attn_v": "v",
    "rst": "rst",
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
OPSPACE_RUNTIME_METRIC_NAMES = (
    "enabled",
    "k_exec",
    "exec_slots",
    "valid_exec_slots_mean",
    "bucket_capacity",
    "bucket_fill_mean",
    "selected_lane_top1_frac",
    "primary_accept_frac",
    "bucket_fill_skew",
    "reroute_frac",
    "relu_gate_count_mean",
    "gate_denominator_mean",
    "gate_mass_mean",
    "effective_ops",
    "padded_exec_slots_mean",
    "overflow_frac",
    "no_nan",
    "owner_model_axis",
    "selected_requests",
    "processed_requests",
    "all_processed",
    "factorized_tile_layout_ok",
)
OPSPACE_FINAL_RUNTIME_METRIC_NAMES = (
    "assignment_score_best_mean",
    "assignment_score_chosen_mean",
    "assignment_score_loss_mean",
    "assignment_score_loss_p95",
    "assignment_score_loss_max",
    "assignment_regret_mean",
    "assignment_regret_p95",
    "assignment_regret_max",
    "low_regret_spill_frac",
    "high_regret_spill_frac",
    "unresolved_after_rank_pass_count",
    "assignment_collision_count",
    "overflow_slowpath_frac",
    "semantic_drop_frac",
    "bucket_fill_p50",
    "bucket_fill_p95",
    "bucket_fill_p99",
    "bucket_fill_max",
    "bucket_capacity_util_p50",
    "bucket_capacity_util_p95",
    "bucket_capacity_util_p99",
    "bucket_capacity_util_max",
    "pallas_logical_executed_chunks",
    "pallas_padding_frac",
    "pallas_valid_token_frac",
    "pallas_backend_available",
    "pallas_logical_empty_chunks",
    "pallas_actual_launched_chunks",
    "pallas_actual_skipped_chunks",
    "actual_vs_logical_chunk_ratio",
    "output_tile_capacity",
    "output_tile_fill_p50",
    "output_tile_fill_p95",
    "output_tile_fill_p99",
    "output_tile_fill_max",
    "output_tile_overflow_frac",
    "output_tile_padding_frac",
    "d_tile_size",
    "output_tile_size",
    "output_tiled_backend_active",
)


def refresh_v4168_runtime_metric_names(module):
    global OPSPACE_RUNTIME_METRIC_NAMES
    global OPSPACE_FINAL_RUNTIME_METRIC_NAMES
    runtime_names = getattr(module, "OPSPACE_RUNTIME_DIAG_NAMES", None)
    final_names = getattr(module, "OPSPACE_FINAL_RUNTIME_DIAG_NAMES", None)
    if runtime_names is not None:
        OPSPACE_RUNTIME_METRIC_NAMES = tuple(str(name) for name in runtime_names)
    if final_names is not None:
        OPSPACE_FINAL_RUNTIME_METRIC_NAMES = tuple(
            str(name) for name in final_names)


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
    parser.add_argument("--forward-profile-steps", type=int, default=1,
                        help=("Measured forward-only steps after the train "
                              "benchmark. Set 0 to disable."))
    parser.add_argument("--module-profile-steps", type=int, default=1,
                        help=("Measured split-module forward passes "
                              "after the train benchmark. Set 0 to disable."))
    parser.add_argument("--fast", "--fast-only", dest="fast_only",
                        action="store_true",
                        help=("Run a quick real-data forward bottleneck "
                              "diagnosis; skip train-step benchmark."))
    parser.add_argument(
        "--backward-profile", action="store_true",
        help=("With --fast, also measure v4174 QKV and RST "
              "forward+backward kernels."))
    parser.add_argument(
        "--detailed-profile-layers", type=int, default=0,
        help=("Limit v4174 detailed profiling to the first N layers. "
              "Use 0 for all layers."))
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
                              "benchmark_profile, and summary records. "
                              "Supports {index} and {config}. Pass an empty "
                              "string to disable."))
    parser.add_argument("--override-sector-bucket-capacity-mult", type=float,
                        default=None,
                        help="Runtime override for v4168 sector bucket capacity multiplier.")
    parser.add_argument("--sweep-sector-capacity-mult", default=None,
                        help=("Comma-separated sector bucket capacity "
                              "multipliers, e.g. 2,4,8,12,16. "
                              "Use 'high' for 2,4,8,12,16."))
    parser.add_argument("--sweep-v-top-blocks", default=None,
                        help="Comma-separated v_top_blocks values.")
    parser.add_argument("--sweep-rst-top-blocks", default=None,
                        help="Comma-separated rst_top_blocks values.")
    parser.add_argument("--sweep-v-block-size", default=None,
                        help="Comma-separated v_block_size values.")
    parser.add_argument("--sweep-rst-block-size", default=None,
                        help="Comma-separated rst_block_size values.")
    args = parser.parse_args()

    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    if args.steps == 0 and not args.fast_only:
        raise SystemExit("--steps must be > 0 unless --fast is set")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be >= 0")
    if args.forward_profile_steps < 0:
        raise SystemExit("--forward-profile-steps must be >= 0")
    if args.module_profile_steps < 0:
        raise SystemExit("--module-profile-steps must be >= 0")
    if args.backward_profile and not args.fast_only:
        raise SystemExit("--backward-profile requires --fast")
    if args.detailed_profile_layers < 0:
        raise SystemExit("--detailed-profile-layers must be >= 0")
    if args.model_version and args.model_version not in SUPPORTED_MODEL_VERSIONS:
        raise SystemExit(
            f"--model-version must be one of {SUPPORTED_MODEL_VERSIONS}")
    fast_detailed_profile = bool(args.fast_only)
    if fast_detailed_profile:
        args.forward_profile_steps = 1
        args.module_profile_steps = 0
        args.steps = 0
        args.warmup_steps = 0
    args.fast_detailed_profile = fast_detailed_profile
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

    if _is_host0():
        if len(summaries) > 1:
            print_comparison(summaries)
        print_copy_summary(summaries, args)


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

    sweep_items = []
    sweep_defs = (
        ("sector_capacity_mult", args.sweep_sector_capacity_mult, float,
         "--sweep-sector-capacity-mult",
         {"high": (2, 4, 8, 12, 16)}),
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
    _log(f"  fast_only: {str(bool(args.fast_only)).lower()}")
    _log(f"  warmup_steps: {0 if args.fast_only else args.warmup_steps}")
    _log(f"  measure_steps: {0 if args.fast_only else args.steps}")
    _log(f"  forward_profile_steps: {args.forward_profile_steps}")
    _log(f"  module_profile_steps: {args.module_profile_steps}")
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
    data_source = "dummy" if args.dummy_data else "real_config_data"
    if args.fast_only:
        _log("fast_only=true")
        _log("fast_mode=quick_detailed_forward_diagnosis")
        _log(f"data_source={data_source}")
        _log("train_step=skipped")
        _log("forward_profile_steps=1")
        _log("detailed_profile_steps=1")
        _log("module_profile_steps=0")

    model = build_model(cfg)
    sharded_fns = build_sharded_fns(cfg, mesh)
    train_step_budget = (
        0 if args.fast_only
        else int(args.steps) + int(args.warmup_steps) + 2)
    profile_step_budget = (
        (int(args.forward_profile_steps) + 1
         if int(args.forward_profile_steps) > 0 else 0)
        + (int(args.module_profile_steps) + 1
           if int(args.module_profile_steps) > 0 else 0)
        + (2 if bool(getattr(args, "fast_detailed_profile", False)) else 0))
    train_loader = build_loader(
        cfg, batch_size, seq_len, n_hosts, host_id,
        total_steps=train_step_budget + profile_step_budget,
        dummy_data=args.dummy_data)
    iterator = iter(train_loader)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy = jnp.ones((1, seq_len), dtype=jnp.int32)
    _log("Initializing model params...")
    if model_version in V417X_MODEL_VERSIONS:
        variables = model.init(
            {"params": init_rng, "dropout": dropout_rng},
            dummy,
            deterministic=True,
        )
    else:
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
    if model_version in V417X_MODEL_VERSIONS:
        from scripts import train_jax
        param_shardings = train_jax.get_param_shardings(
            params, mesh, model_version=model_version)
    else:
        param_shardings = get_param_shardings(params, mesh)
    params = shard_params_to_mesh(params, param_shardings)

    hbm_after = collect_hbm_stats()
    compile_seconds = None
    warmup_times = []
    measure_times = []
    measure_tokens = []
    measure_sector_records = []
    peak_hbm = hbm_after.get("hbm_peak_gb")
    execution_step_no = 0

    if args.fast_only:
        _log(
            "[bench] full train step skipped (--fast) "
            f"HBM={fmt(hbm_after.get('hbm_used_gb'))}G "
            f"peak={fmt(hbm_after.get('hbm_peak_gb'))}G")
    else:
        optax_mod = require_optax()
        optimizer = optax_mod.adamw(
            learning_rate=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)))
        opt_state = optimizer.init(params)

        train_step = create_benchmark_train_step(
            model, optimizer, sharded_fns, cfg)
        prev_metrics = None
        _log("Compiling benchmark step...")
        batch, iterator = next_batch(iterator, train_loader)
        ids, mask = shard_batch_to_mesh(
            batch, data_sharding, batch_size, seq_len)
        rng, step_rng = jax.random.split(rng)
        compile_t0 = time.perf_counter()
        params, opt_state, prev_metrics = train_step(
            params, opt_state, ids, mask, step_rng, jnp.asarray(0, jnp.int32))
        block_until_ready(prev_metrics)
        compile_seconds = time.perf_counter() - compile_t0
        hbm_after = collect_hbm_stats()
        peak_hbm = hbm_after.get("hbm_peak_gb")
        _log(
            "[bench] compile "
            f"{compile_seconds:.3f}s "
            f"HBM={fmt(hbm_after.get('hbm_used_gb'))}G "
            f"peak={fmt(hbm_after.get('hbm_peak_gb'))}G")

        def run_benchmark_step(phase, phase_step, phase_total, step_no):
            nonlocal params, opt_state, iterator, rng
            nonlocal peak_hbm, execution_step_no
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
            phase_step = step_no if phase == "warmup" else len(measure_times) + 1
            phase_total = (
                int(args.warmup_steps)
                if phase == "warmup" else int(args.steps))
            seconds, tokens_per_second, step_sector_records = (
                run_benchmark_step(phase, phase_step, phase_total, step_no))
            if phase == "warmup":
                warmup_times.append(seconds)
            else:
                measure_times.append(seconds)
                measure_tokens.append(tokens_per_second)
                if model_version == V4168_MODEL_VERSION:
                    measure_sector_records.extend(step_sector_records)

    train_peak_hbm = None if args.fast_only else peak_hbm
    train_mean_step_seconds = mean(measure_times)
    profile_summary = {}
    profile_records = []
    if (int(args.forward_profile_steps) > 0
            or int(args.module_profile_steps) > 0):
        (profile_summary, profile_records, iterator, rng,
         execution_step_no) = run_forward_profile(
            model, sharded_fns, params, cfg, args, iterator, train_loader,
            data_sharding, batch_size, seq_len, rng, run_index,
            variant_label, execution_step_no, train_mean_step_seconds)
        write_jsonl_records(metrics_jsonl, profile_records)
        peak_hbm = update_peak_hbm(
            peak_hbm, {"hbm_peak_gb": profile_summary.get(
                "profile_peak_hbm_gb")})

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
        "run_index": int(run_index),
        "config_path": config_arg,
        "variant": variant_label,
        "overrides": dict(overrides or {}),
        "model_version": model_version,
        "data_source": data_source,
        "fast_detailed_profile": bool(
            getattr(args, "fast_detailed_profile", False)),
        "train_benchmark_enabled": not bool(args.fast_only),
        "compile_seconds": compile_seconds,
        "warmup_steps": 0 if args.fast_only else int(args.warmup_steps),
        "measure_steps": 0 if args.fast_only else int(args.steps),
        "configured_warmup_steps": int(args.warmup_steps),
        "configured_measure_steps": int(args.steps),
        "mean_step_seconds": train_mean_step_seconds,
        "median_step_seconds": median(measure_times),
        "p90_step_seconds": percentile(measure_times, 90),
        "min_step_seconds": min(measure_times) if measure_times else None,
        "max_step_seconds": max(measure_times) if measure_times else None,
        "mean_tokens_per_second": mean(measure_tokens),
        "train_peak_hbm_gb": train_peak_hbm,
        "peak_hbm_gb": peak_hbm,
        "hbm_limit_gb": hbm_after.get("hbm_limit_gb"),
        "xla_total_hbm_usage": xla_report.get("total_hbm_usage"),
        "xla_program_hbm_requirement": xla_report.get(
            "program_hbm_requirement"),
        "xla_source_file": xla_report.get("source_file"),
        "benchmark_valid": benchmark_valid,
        "invalid_reason": invalid_reason,
        "sector_summary": sector_summary,
        "compute_summary": compute_summary,
    }
    summary.update(profile_summary)
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
    if version in V417X_MODEL_VERSIONS:
        from scripts import train_jax
        train_jax._maybe_materialize_vocab_parallel_config(cfg)
        return train_jax.build_model_from_config(cfg)
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


def _ceil_to_multiple(value, multiple):
    value = int(value)
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def _v4168_uses_execution_backend(pools):
    if not isinstance(pools, dict):
        return False
    for pool in ("qk", "v", "rst"):
        pool_cfg = pools.get(pool, {})
        if isinstance(pool_cfg, dict) and "execution_backend" in pool_cfg:
            return True
    return False


def v4168_operation_space_backend_layouts(cfg):
    """Parse the v4168 clean Region-Block atlas config used by train_jax."""
    t = cfg["training"]
    m = cfg["model"]
    opspace = t.get("operation_space", {})
    pools = opspace.get("pools", {})
    if pools is None:
        pools = {}
    if not isinstance(pools, dict):
        raise ValueError("training.operation_space.pools must be a mapping.")

    extra_pools = sorted(set(pools) - {"qk", "v", "rst"})
    if extra_pools:
        raise ValueError(
            "training.operation_space.pools only supports qk, v, rst; "
            f"remove: {', '.join(extra_pools)}")

    removed_pool_keys = {
        "mode",
        "routing_mode",
        "execution_mode",
        "execution_backend_mode",
        "lanes",
        "local_lanes",
        "k_exec",
        "tiles_per_lane",
        "tile_size",
        "exec_tiles_per_block",
        "blocks_per_lane",
        "block_size",
        "local_padded_ops",
        "local_operator_slots",
        "exec_slots",
        "region_block_pallas",
        "factorized_lane_mean",
        "lane_output_mode",
        "visible_regions_start",
        "visible_regions_mid",
        "visible_regions_final",
        "candidate_region_count",
        "block_mixing_enabled",
    }
    common_pool_keys = {
        "execution_backend",
        "num_regions",
        "blocks_per_region",
        "operators_per_block",
        "visible_regions",
        "visible_blocks_per_region",
    }
    allowed_rst_keys = {
        *common_pool_keys,
        "region_score_pooling",
        "region_score_temperature",
        "region_capacity_factor",
        "block_capacity_factor",
        "bucket_capacity_factor",
        "high_regret_threshold",
    }
    defaults = {
        "qk": {"n_key": "n_qk", "execution_backend": "dense"},
        "v": {"n_key": "n_v", "execution_backend": "dense"},
        "rst": {"n_key": "n_rst", "execution_backend": "sparse_region_block"},
    }
    mesh_model = max(1, int(t.get("mesh_model", 1)))
    layouts = {}

    def _bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    load_smoothing = t.get("operation_space_load_smoothing", {})
    if load_smoothing is None:
        load_smoothing = {}
    if not isinstance(load_smoothing, dict):
        raise ValueError(
            "training.operation_space_load_smoothing must be a mapping.")
    load_smoothing_defaults = {
        "load_smoothing_enabled": _bool(load_smoothing.get("enabled", True)),
        "load_smoothing_rst_region_weight": float(
            load_smoothing.get("rst_region_weight", 3.0e-4)),
        "load_smoothing_rst_block_weight": float(
            load_smoothing.get("rst_block_weight", 0.0)),
        "load_smoothing_load_temperature": float(
            load_smoothing.get("load_temperature", 0.7)),
        "load_smoothing_space_temperature": float(
            load_smoothing.get("space_temperature", 0.12)),
        "load_smoothing_alpha": float(load_smoothing.get("alpha", 1.25)),
        "load_smoothing_warmup_tokens": float(
            load_smoothing.get("warmup_tokens", 2.0e8)),
        "load_smoothing_peak_tokens": float(
            load_smoothing.get("peak_tokens", 1.0e9)),
        "load_smoothing_final_weight_frac": float(
            load_smoothing.get("final_weight_frac", 1.0)),
    }

    for label, default in defaults.items():
        pool = pools.get(label, {})
        if not isinstance(pool, dict) or not pool:
            raise ValueError(
                f"training.operation_space.pools.{label} is required when "
                "operation_space.enabled=true.")
        if "routing_mode" in pool:
            raise ValueError(
                f"operation_space.pools.{label}.routing_mode has been "
                "removed. Use execution_backend only.")
        present_removed = sorted(set(pool) & removed_pool_keys)
        if present_removed:
            raise ValueError(
                "training.operation_space.pools."
                f"{label} contains removed clean-config keys: "
                + ", ".join(present_removed))
        if label == "rst":
            if "bucket_chunk_size" in pool:
                raise ValueError(
                    "training.operation_space.pools.rst.bucket_chunk_size "
                    "was removed from the sparse Region-Block backend.")
            if "assignment_policy" in pool:
                raise ValueError(
                    "training.operation_space.pools.rst.assignment_policy "
                    "was removed from the sparse Region-Block backend.")
        allowed_pool_keys = allowed_rst_keys if label == "rst" else common_pool_keys
        extra = sorted(set(pool) - allowed_pool_keys)
        if extra:
            raise ValueError(
                f"training.operation_space.pools.{label} only supports "
                "the clean operation-space fields for that pool; remove: "
                f"{', '.join(extra)}")

        execution_backend = str(pool.get(
            "execution_backend", default["execution_backend"])).strip().lower()
        expected_backend = default["execution_backend"]
        if execution_backend != expected_backend:
            raise ValueError(
                f"training.operation_space.pools.{label}.execution_backend "
                f"must be {expected_backend!r}, got {execution_backend!r}.")

        n_key = default["n_key"]
        n_ops = int(m.get(n_key, m.get("n_know", 0) if label == "rst" else 0))
        if n_ops <= 0:
            raise ValueError(f"model.{n_key} must be > 0 for operation_space.")

        required_ints = {
            "num_regions": pool.get("num_regions"),
            "blocks_per_region": pool.get("blocks_per_region"),
            "operators_per_block": pool.get("operators_per_block"),
            "visible_regions": pool.get("visible_regions"),
            "visible_blocks_per_region": pool.get("visible_blocks_per_region"),
        }
        missing = [name for name, value in required_ints.items() if value is None]
        if missing:
            raise ValueError(
                f"training.operation_space.pools.{label} missing required "
                f"field(s): {', '.join(missing)}")
        num_regions = int(required_ints["num_regions"])
        blocks_per_region = int(required_ints["blocks_per_region"])
        operators_per_block = int(required_ints["operators_per_block"])
        visible_regions = int(required_ints["visible_regions"])
        visible_blocks_per_region = int(
            required_ints["visible_blocks_per_region"])
        for field_name, value in (
                ("num_regions", num_regions),
                ("blocks_per_region", blocks_per_region),
                ("operators_per_block", operators_per_block),
                ("visible_regions", visible_regions),
                ("visible_blocks_per_region", visible_blocks_per_region)):
            if value < 1:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{label}.{field_name} must be >= 1, got {value}.")
        if visible_regions > num_regions:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.visible_regions={visible_regions} must be <= "
                f"num_regions={num_regions}.")
        if visible_blocks_per_region > blocks_per_region:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.visible_blocks_per_region="
                f"{visible_blocks_per_region} must be <= "
                f"blocks_per_region={blocks_per_region}.")
        if num_regions % mesh_model != 0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.num_regions={num_regions} must be divisible "
                f"by mesh_model={mesh_model}.")

        total_capacity = (
            num_regions * blocks_per_region * operators_per_block)
        if n_ops > total_capacity:
            raise ValueError(
                "training.operation_space.pools."
                f"{label} capacity {total_capacity} is smaller than "
                f"model.{n_key}={n_ops}.")

        region_score_pooling = str(pool.get(
            "region_score_pooling", "smoothmax")).strip().lower()
        if label == "rst" and region_score_pooling != "smoothmax":
            raise ValueError(
                "training.operation_space.pools.rst."
                "region_score_pooling must be 'smoothmax'.")
        region_score_temperature = float(pool.get(
            "region_score_temperature", 0.25))
        if region_score_temperature <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.region_score_temperature must be > 0, got "
                f"{region_score_temperature}.")
        region_capacity_factor = float(pool.get(
            "region_capacity_factor", 1.25))
        block_capacity_factor = float(pool.get(
            "block_capacity_factor", 1.25))
        if region_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.region_capacity_factor must be > 0, got "
                f"{region_capacity_factor}.")
        if block_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.block_capacity_factor must be > 0, got "
                f"{block_capacity_factor}.")
        bucket_capacity_factor = float(pool.get(
            "bucket_capacity_factor", block_capacity_factor))
        if bucket_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.bucket_capacity_factor must be > 0, got "
                f"{bucket_capacity_factor}.")
        high_regret_threshold = float(pool.get(
            "high_regret_threshold", 0.05))
        if high_regret_threshold < 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{label}.high_regret_threshold must be >= 0, got "
                f"{high_regret_threshold}.")

        semantic_visible_ops_per_token = (
            visible_regions * visible_blocks_per_region
            * operators_per_block)
        if execution_backend == "sparse_region_block":
            physical_visible_ops_per_token = (
                visible_regions * blocks_per_region * operators_per_block)
        else:
            physical_visible_ops_per_token = total_capacity // mesh_model
        semantic_compute_frac_vs_dense = (
            float(semantic_visible_ops_per_token) / float(total_capacity))
        physical_compute_frac_vs_dense = (
            float(physical_visible_ops_per_token) / float(total_capacity))
        raw_tiles = (n_ops + operators_per_block - 1) // operators_per_block
        total_tiles = num_regions * blocks_per_region
        layouts[label] = {
            "routing_mode": "region_block",
            "execution_mode": execution_backend,
            "execution_backend": execution_backend,
            "lanes": num_regions,
            "k_exec": visible_regions,
            "num_regions": num_regions,
            "regions_per_owner": num_regions // mesh_model,
            "blocks_per_region": blocks_per_region,
            "operators_per_block": operators_per_block,
            "operators_per_region": blocks_per_region * operators_per_block,
            "visible_regions": visible_regions,
            "visible_blocks_per_region": visible_blocks_per_region,
            "visible_ops_per_token": semantic_visible_ops_per_token,
            "semantic_visible_ops_per_token": semantic_visible_ops_per_token,
            "physical_visible_ops_per_token": physical_visible_ops_per_token,
            "semantic_compute_frac_vs_dense": semantic_compute_frac_vs_dense,
            "physical_compute_frac_vs_dense": physical_compute_frac_vs_dense,
            "region_score_pooling": region_score_pooling,
            "region_score_temperature": region_score_temperature,
            "region_capacity_factor": region_capacity_factor,
            "block_capacity_factor": block_capacity_factor,
            "bucket_capacity_factor": bucket_capacity_factor,
            "tile_size": operators_per_block,
            "raw_tiles": raw_tiles,
            "total_tiles": total_tiles,
            "tiles_per_lane": blocks_per_region,
            "exec_tiles_per_block": 1,
            "blocks_per_lane": blocks_per_region,
            "block_size": operators_per_block,
            "padded_ops": total_capacity,
            "invalid_padded_ops": total_capacity - n_ops,
            "bucket_chunk_size": 256,
            "assignment_policy": "region_block",
            "high_regret_threshold": high_regret_threshold,
            "load_smoothing_enabled": (
                load_smoothing_defaults["load_smoothing_enabled"]
                if label == "rst" else False),
            "load_smoothing_rst_region_weight": (
                load_smoothing_defaults[
                    "load_smoothing_rst_region_weight"]
                if label == "rst" else 0.0),
            "load_smoothing_rst_block_weight": (
                load_smoothing_defaults["load_smoothing_rst_block_weight"]
                if label == "rst" else 0.0),
            "load_smoothing_load_temperature": (
                load_smoothing_defaults[
                    "load_smoothing_load_temperature"]),
            "load_smoothing_space_temperature": (
                load_smoothing_defaults[
                    "load_smoothing_space_temperature"]),
            "load_smoothing_alpha": (
                load_smoothing_defaults["load_smoothing_alpha"]),
            "load_smoothing_warmup_tokens": (
                load_smoothing_defaults[
                    "load_smoothing_warmup_tokens"]),
            "load_smoothing_peak_tokens": (
                load_smoothing_defaults["load_smoothing_peak_tokens"]),
            "load_smoothing_final_weight_frac": (
                load_smoothing_defaults[
                    "load_smoothing_final_weight_frac"]),
            "global_operator_capacity": total_capacity,
            "local_operator_capacity": total_capacity // mesh_model,
            "operator_capacity": total_capacity,
            "invalid_operator_capacity": total_capacity - n_ops,
            "num_devices": mesh_model,
        }
    return layouts


def v4168_operation_space_enabled(cfg):
    version = str(cfg["model"].get("model_version", ""))
    opspace = cfg["training"].get("operation_space", {})
    return (
        version == V4168_MODEL_VERSION
        and isinstance(opspace, dict)
        and bool(opspace.get("enabled", False))
    )


def v4168_operation_space_layouts(cfg):
    if not v4168_operation_space_enabled(cfg):
        return {}
    t = cfg["training"]
    m = cfg["model"]
    opspace = t.get("operation_space", {})
    if not isinstance(opspace, dict):
        raise ValueError("training.operation_space must be a mapping.")
    tile_size = int(opspace.get("tile_size", 128))
    if tile_size <= 0:
        raise ValueError(
            f"training.operation_space.tile_size must be > 0, got {tile_size}.")
    pools = opspace.get("pools", {})
    if pools is None:
        pools = {}
    if not isinstance(pools, dict):
        raise ValueError("training.operation_space.pools must be a mapping.")
    if _v4168_uses_execution_backend(pools):
        return v4168_operation_space_backend_layouts(cfg)
    mesh_model = max(1, int(t.get("mesh_model", 1)))
    defaults = {
        "qk": {
            "n_key": "n_qk",
            "routing_mode": "factorized_lane_mean",
            "execution_mode": "dense_masked",
            "lanes": 8,
            "k_exec": 4,
        },
        "v": {
            "n_key": "n_v",
            "routing_mode": "factorized_lane_mean",
            "execution_mode": "dense_masked",
            "lanes": 8,
            "k_exec": 5,
        },
        "rst": {
            "n_key": "n_rst",
            "routing_mode": "factorized_lane_mean",
            "execution_mode": "block_bucketed_dense",
            "lanes": 32,
            "k_exec": 32,
        },
    }
    layouts = {}

    def _bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    load_smoothing = t.get("operation_space_load_smoothing", {})
    if load_smoothing is None:
        load_smoothing = {}
    if not isinstance(load_smoothing, dict):
        raise ValueError(
            "training.operation_space_load_smoothing must be a mapping.")
    load_smoothing_defaults = {
        "load_smoothing_enabled": _bool(load_smoothing.get("enabled", True)),
        "load_smoothing_rst_region_weight": float(
            load_smoothing.get("rst_region_weight", 3.0e-4)),
        "load_smoothing_rst_block_weight": float(
            load_smoothing.get("rst_block_weight", 0.0)),
        "load_smoothing_load_temperature": float(
            load_smoothing.get("load_temperature", 0.7)),
        "load_smoothing_space_temperature": float(
            load_smoothing.get("space_temperature", 0.12)),
        "load_smoothing_alpha": float(load_smoothing.get("alpha", 1.25)),
        "load_smoothing_warmup_tokens": float(
            load_smoothing.get("warmup_tokens", 2.0e8)),
        "load_smoothing_peak_tokens": float(
            load_smoothing.get("peak_tokens", 1.0e9)),
        "load_smoothing_final_weight_frac": float(
            load_smoothing.get("final_weight_frac", 1.0)),
    }

    removed_pool_keys = {
        "mode",
        "lanes",
        "k_exec",
        "exec_tiles_per_block",
        "blocks_per_lane",
        "block_size",
        "lane_output_mode",
        "visible_regions_start",
        "visible_regions_mid",
        "visible_regions_final",
        "candidate_region_count",
        "block_mixing_enabled",
    }
    for pool, default in defaults.items():
        pool_cfg = pools.get(pool, {})
        if pool_cfg is None:
            pool_cfg = {}
        if not isinstance(pool_cfg, dict):
            raise ValueError(
                f"training.operation_space.pools.{pool} must be a mapping.")
        present_removed = sorted(set(pool_cfg) & removed_pool_keys)
        if present_removed:
            raise ValueError(
                "training.operation_space.pools."
                f"{pool} contains removed clean-config keys: "
                + ", ".join(present_removed))
        execution_mode = str(pool_cfg.get(
            "execution_mode", default["execution_mode"])).strip().lower()
        valid_execution_modes = (
            "dense_masked",
            "block_bucketed_dense",
            "block_bucketed_pallas_final",
            "region_block_pallas",
        )
        if execution_mode not in valid_execution_modes:
            raise ValueError(
                f"training.operation_space.pools.{pool}.execution_mode must be "
                "'dense_masked', 'block_bucketed_dense', or "
                "'block_bucketed_pallas_final'/'region_block_pallas', got "
                f"{execution_mode!r}.")
        if (execution_mode in (
                "block_bucketed_pallas_final",
                "region_block_pallas") and pool != "rst"):
            raise ValueError(
                "training.operation_space.pools."
                f"{pool}.execution_mode={execution_mode} is only supported "
                "for the RST single-route backend in v4168 benchmark mode.")
        n_key = default["n_key"]
        n_ops = int(m.get(n_key, m.get("n_know", 0) if pool == "rst" else 0))
        if n_ops <= 0:
            raise ValueError(f"model.{n_key} must be > 0 for operation_space.")

        if execution_mode == "region_block_pallas":
            routing_mode = "region_block"
            num_regions = int(pool_cfg.get("num_regions", default["lanes"]))
            if num_regions <= 0:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool}.num_regions must be > 0, got {num_regions}.")
            if num_regions % mesh_model != 0:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool}.num_regions={num_regions} must be divisible "
                    f"by mesh_model={mesh_model}.")
            blocks_per_region = int(pool_cfg.get("blocks_per_region", 2))
            operators_per_block = int(pool_cfg.get(
                "operators_per_block", 512))
            if blocks_per_region <= 0:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool}.blocks_per_region must be > 0, got "
                    f"{blocks_per_region}.")
            if operators_per_block <= 0:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool}.operators_per_block must be > 0, got "
                    f"{operators_per_block}.")
            if operators_per_block % tile_size != 0:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool}.operators_per_block={operators_per_block} must "
                    f"be divisible by tile_size={tile_size}.")
            total_capacity = (
                num_regions * blocks_per_region * operators_per_block)
            if n_ops > total_capacity:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool} capacity {total_capacity} is smaller than "
                    f"model.{n_key}={n_ops}.")
            visible_regions = int(pool_cfg.get("visible_regions", 4))
            if not (1 <= visible_regions <= num_regions):
                raise ValueError(
                    "training.operation_space.pools.rst.visible_regions "
                    f"must be in [1, num_regions={num_regions}], got "
                    f"{visible_regions}.")
            visible_blocks_per_region = int(pool_cfg.get(
                "visible_blocks_per_region", 1))
            if not (1 <= visible_blocks_per_region <= blocks_per_region):
                raise ValueError(
                    "training.operation_space.pools.rst."
                    "visible_blocks_per_region must be in "
                    f"[1, blocks_per_region={blocks_per_region}], got "
                    f"{visible_blocks_per_region}.")
            region_score_pooling = str(pool_cfg.get(
                "region_score_pooling", "smoothmax")).strip().lower()
            if region_score_pooling != "smoothmax":
                raise ValueError(
                    "training.operation_space.pools.rst."
                    "region_score_pooling must be 'smoothmax'.")
            region_score_temperature = float(pool_cfg.get(
                "region_score_temperature", 0.25))
            if region_score_temperature <= 0.0:
                raise ValueError(
                    "training.operation_space.pools.rst."
                    "region_score_temperature must be > 0, got "
                    f"{region_score_temperature}.")
            region_capacity_factor = float(pool_cfg.get(
                "region_capacity_factor", 1.25))
            if region_capacity_factor <= 0.0:
                raise ValueError(
                    "training.operation_space.pools.rst."
                    "region_capacity_factor must be > 0, got "
                    f"{region_capacity_factor}.")
            block_capacity_factor = float(pool_cfg.get(
                "block_capacity_factor", 1.25))
            if block_capacity_factor <= 0.0:
                raise ValueError(
                    "training.operation_space.pools.rst."
                    "block_capacity_factor must be > 0, got "
                    f"{block_capacity_factor}.")
            k_exec = visible_regions
            lanes = num_regions
            exec_tiles_per_block = operators_per_block // tile_size
            tiles_per_lane = blocks_per_region * exec_tiles_per_block
            raw_tiles = (n_ops + tile_size - 1) // tile_size
            total_tiles = num_regions * tiles_per_lane
            blocks_per_lane = blocks_per_region
            block_size = operators_per_block
            bucket_capacity_factor = block_capacity_factor
        else:
            routing_mode = str(pool_cfg.get(
                "routing_mode",
                default["routing_mode"])).strip().lower()
            if routing_mode != "factorized_lane_mean":
                raise ValueError(
                    f"training.operation_space.pools.{pool}.routing_mode "
                    "must be 'factorized_lane_mean', got "
                    f"{routing_mode!r}.")
            lanes = int(default["lanes"])
            k_exec = int(default["k_exec"])
            num_regions = lanes
            blocks_per_region = None
            operators_per_block = None
            visible_regions = k_exec
            visible_blocks_per_region = 1
            region_score_pooling = "smoothmax"
            region_score_temperature = 0.25
            region_capacity_factor = 1.25
            block_capacity_factor = 1.25
            raw_tiles = (n_ops + tile_size - 1) // tile_size
            total_tiles = _ceil_to_multiple(
                raw_tiles, math.lcm(lanes, mesh_model))
            tiles_per_lane = total_tiles // lanes
            exec_tiles_per_block = (
                2 if execution_mode in (
                    "block_bucketed_dense",
                    "block_bucketed_pallas_final") else 1)
            if tiles_per_lane % exec_tiles_per_block != 0:
                raise ValueError(
                    "training.operation_space.pools."
                    f"{pool}.exec_tiles_per_block={exec_tiles_per_block} "
                    f"must divide tiles_per_lane={tiles_per_lane}.")
            blocks_per_lane = tiles_per_lane // exec_tiles_per_block
            block_size = tile_size * exec_tiles_per_block
            blocks_per_region = blocks_per_lane
            operators_per_block = block_size
            bucket_capacity_factor = float(pool_cfg.get(
                "bucket_capacity_factor",
                1.5 if execution_mode == "block_bucketed_pallas_final"
                else 1.25))
        bucket_capacity_factor = float(pool_cfg.get(
            "bucket_capacity_factor", bucket_capacity_factor))
        if bucket_capacity_factor <= 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{pool}.bucket_capacity_factor must be > 0, got "
                f"{bucket_capacity_factor}.")
        bucket_chunk_size = int(pool_cfg.get(
            "bucket_chunk_size",
            128 if execution_mode in (
                "block_bucketed_pallas_final",
                "region_block_pallas") else 1024))
        if bucket_chunk_size <= 0:
            raise ValueError(
                "training.operation_space.pools."
                f"{pool}.bucket_chunk_size must be > 0, got "
                f"{bucket_chunk_size}.")
        assignment_policy = str(pool_cfg.get(
            "assignment_policy",
            "low_regret"
            if execution_mode in (
                "block_bucketed_pallas_final",
                "region_block_pallas")
            else "token_order")).strip().lower()
        if (execution_mode in (
                "block_bucketed_pallas_final",
                "region_block_pallas")
                and assignment_policy != "low_regret"):
            raise ValueError(
                "training.operation_space.pools."
                f"{pool}.assignment_policy must be 'low_regret' for "
                f"{execution_mode}.")
        high_regret_threshold = float(pool_cfg.get(
            "high_regret_threshold", 0.05))
        if high_regret_threshold < 0.0:
            raise ValueError(
                "training.operation_space.pools."
                f"{pool}.high_regret_threshold must be >= 0, got "
                f"{high_regret_threshold}.")
        layouts[pool] = {
            "routing_mode": routing_mode,
            "execution_mode": execution_mode,
            "lanes": lanes,
            "k_exec": k_exec,
            "num_regions": num_regions,
            "blocks_per_region": blocks_per_region,
            "operators_per_block": operators_per_block,
            "visible_regions": visible_regions,
            "visible_blocks_per_region": visible_blocks_per_region,
            "region_score_pooling": region_score_pooling,
            "region_score_temperature": region_score_temperature,
            "region_capacity_factor": region_capacity_factor,
            "block_capacity_factor": block_capacity_factor,
            "tile_size": tile_size,
            "raw_tiles": raw_tiles,
            "total_tiles": total_tiles,
            "tiles_per_lane": tiles_per_lane,
            "exec_tiles_per_block": exec_tiles_per_block,
            "blocks_per_lane": blocks_per_lane,
            "block_size": block_size,
            "padded_ops": total_tiles * tile_size,
            "invalid_padded_ops": total_tiles * tile_size - n_ops,
            "bucket_capacity_factor": bucket_capacity_factor,
            "bucket_chunk_size": bucket_chunk_size,
            "assignment_policy": assignment_policy,
            "high_regret_threshold": high_regret_threshold,
            "load_smoothing_enabled": (
                load_smoothing_defaults["load_smoothing_enabled"]
                if pool == "rst" else False),
            "load_smoothing_rst_region_weight": (
                load_smoothing_defaults[
                    "load_smoothing_rst_region_weight"]
                if pool == "rst" else 0.0),
            "load_smoothing_rst_block_weight": (
                load_smoothing_defaults["load_smoothing_rst_block_weight"]
                if pool == "rst" else 0.0),
            "load_smoothing_load_temperature": (
                load_smoothing_defaults[
                    "load_smoothing_load_temperature"]),
            "load_smoothing_space_temperature": (
                load_smoothing_defaults[
                    "load_smoothing_space_temperature"]),
            "load_smoothing_alpha": (
                load_smoothing_defaults["load_smoothing_alpha"]),
            "load_smoothing_warmup_tokens": (
                load_smoothing_defaults[
                    "load_smoothing_warmup_tokens"]),
            "load_smoothing_peak_tokens": (
                load_smoothing_defaults["load_smoothing_peak_tokens"]),
            "load_smoothing_final_weight_frac": (
                load_smoothing_defaults[
                    "load_smoothing_final_weight_frac"]),
        }
    return layouts


def v4168_operation_space_pool_kwargs(layout):
    kwargs = {
        "operation_space_mode": str(layout["routing_mode"]).lower(),
        "operation_space_routing_mode": str(layout["routing_mode"]).lower(),
        "operation_space_execution_mode": str(layout["execution_mode"]).lower(),
        "opspace_lanes": int(layout["lanes"]),
        "opspace_tiles_per_lane": int(layout["tiles_per_lane"]),
        "opspace_exec_tiles_per_block": int(layout["exec_tiles_per_block"]),
        "opspace_blocks_per_lane": int(layout["blocks_per_lane"]),
        "opspace_block_size": int(layout["block_size"]),
        "opspace_padded_ops": int(layout["padded_ops"]),
        "opspace_tile_size": int(layout["tile_size"]),
        "opspace_k_exec": int(layout["k_exec"]),
        "opspace_num_regions": int(layout["num_regions"]),
        "opspace_blocks_per_region": int(layout["blocks_per_region"]),
        "opspace_operators_per_block": int(layout["operators_per_block"]),
        "opspace_visible_regions": int(layout["visible_regions"]),
        "opspace_visible_blocks_per_region": int(
            layout["visible_blocks_per_region"]),
        "opspace_region_score_pooling": str(
            layout["region_score_pooling"]).lower(),
        "opspace_region_score_temperature": float(
            layout["region_score_temperature"]),
        "opspace_region_capacity_factor": float(
            layout["region_capacity_factor"]),
        "opspace_block_capacity_factor": float(
            layout["block_capacity_factor"]),
        "opspace_bucket_capacity_factor": float(
            layout["bucket_capacity_factor"]),
        "opspace_bucket_chunk_size": int(layout["bucket_chunk_size"]),
        "opspace_assignment_policy": str(
            layout["assignment_policy"]).lower(),
        "opspace_high_regret_threshold": float(
            layout["high_regret_threshold"]),
        "opspace_load_smoothing_enabled": bool(
            layout.get("load_smoothing_enabled", False)),
        "opspace_load_smoothing_rst_region_weight": float(
            layout.get("load_smoothing_rst_region_weight", 3.0e-4)),
        "opspace_load_smoothing_rst_block_weight": float(
            layout.get("load_smoothing_rst_block_weight", 0.0)),
        "opspace_load_smoothing_load_temperature": float(
            layout.get("load_smoothing_load_temperature", 0.7)),
        "opspace_load_smoothing_space_temperature": float(
            layout.get("load_smoothing_space_temperature", 0.12)),
        "opspace_load_smoothing_alpha": float(
            layout.get("load_smoothing_alpha", 1.25)),
        "opspace_load_smoothing_warmup_tokens": float(
            layout.get("load_smoothing_warmup_tokens", 2.0e8)),
        "opspace_load_smoothing_peak_tokens": float(
            layout.get("load_smoothing_peak_tokens", 1.0e9)),
        "opspace_load_smoothing_final_weight_frac": float(
            layout.get("load_smoothing_final_weight_frac", 1.0)),
    }
    if "execution_backend" in layout:
        kwargs["operation_space_execution_backend"] = str(
            layout["execution_backend"]).lower()
    return kwargs


def v4168_operation_space_backend_label(layout):
    return str(layout.get(
        "execution_backend", layout.get("execution_mode", "unknown"))).lower()


def build_sharded_fns(cfg, mesh):
    version = str(cfg["model"].get("model_version", ""))
    if version in V417X_MODEL_VERSIONS:
        from scripts import train_jax
        sharded = train_jax.build_canonical_sharded_fns(
            cfg, mesh, kernel_profile="production")
        _log(
            "sharded SRW kernels: canonical production "
            f"model_version={version}")
        return sharded
    module_name, _class_name = MODEL_MODULES[version]
    module = importlib.import_module(module_name)
    if version == V4168_MODEL_VERSION:
        refresh_v4168_runtime_metric_names(module)
    capacity_mult = cfg["training"].get(
        "benchmark_sector_bucket_capacity_mult", None)
    if version == V4168_MODEL_VERSION and capacity_mult is not None:
        setattr(module, "_V4168_SECTOR_BUCKET_CAPACITY_MULT",
                float(capacity_mult))
        _log(f"v4168 bucket_capacity_mult override: {float(capacity_mult)}")
    make_single = getattr(module, "make_sharded_srw_minimal", None)
    make_paired = getattr(module, "make_sharded_srw_paired_minimal", None)
    make_paired_dense = getattr(
        module, "make_sharded_srw_paired_dense_minimal", None)
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

    opspace_enabled = v4168_operation_space_enabled(cfg)
    opspace_layouts = v4168_operation_space_layouts(cfg)

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
            if opspace_enabled:
                kwargs.update(v4168_operation_space_pool_kwargs(
                    opspace_layouts[pool]))
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
    if opspace_enabled:
        if make_paired_dense is None:
            raise RuntimeError(
                "operation_space enabled but v4168 paired dense minimal "
                "QK factory is missing.")
        attn_qk_paired = make_paired_dense(
            max_chunk_size=qk_chunk,
            **filtered(make_paired_dense, pool_kwargs("qk")))
    else:
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
    if opspace_enabled:
        out.update({
            "operation_space_tau_free": True,
            "qk_backend": v4168_operation_space_backend_label(
                opspace_layouts["qk"]),
            "v_backend": v4168_operation_space_backend_label(
                opspace_layouts["v"]),
            "rst_backend": v4168_operation_space_backend_label(
                opspace_layouts["rst"]),
        })
        _log(
            "sharded SRW kernels: operation_space "
            f"qk/v/rst={out['qk_backend']}/{out['v_backend']}/"
            f"{out['rst_backend']}")
    else:
        _log("sharded SRW kernels: enabled")
    return out


def benchmark_apply_kwargs(cfg, version, sharded_fns, attention_mask, rng, step):
    t = cfg["training"]
    m = cfg["model"]
    soft_gate_t = float(t.get("soft_gate_temperature", 0.07))
    boundary_power = float(t.get(
        "soft_gate_boundary_power_final",
        t.get("soft_gate_boundary_power_mid",
              t.get("soft_gate_boundary_power_start", 4.0))))
    is_v4172 = version == V4172_MODEL_VERSION
    den_power_key = "gate_den_power" if is_v4172 else "admission_den_power"
    den_power = float(
        m.get(den_power_key, t.get(den_power_key, 1.0)))
    tokens_per_step = (
        int(t["batch_size"])
        * int(cfg["model"].get("max_seq_len", 512)))
    out = {
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
        den_power_key: den_power,
    }
    if not is_v4172:
        out["execution_prune_eps"] = 0.0
    if version in V417X_MODEL_VERSIONS:
        out.update({
            "minimal_runtime_profile": "training",
            "ce_token_chunk_size": int(t.get("ce_token_chunk_size", 32768)),
            "compute_accuracy": False,
        })
        if version == V4174_MODEL_VERSION:
            out["collect_train_metrics"] = False
    if version == V4168_MODEL_VERSION:
        training_tokens = (
            jnp.asarray(step, dtype=jnp.float32)
            * jnp.asarray(float(tokens_per_step), dtype=jnp.float32))
        out["training_tokens"] = training_tokens
        out["benchmark_runtime_metrics"] = True
    return out


def copy_runtime_metrics(result, metrics):
    for pool in RUNTIME_POOLS:
        for name in SECTOR_RUNTIME_METRIC_NAMES:
            key = f"sector/{pool}/{name}"
            if key in result:
                metrics[key] = result[key]
        for name in OPSPACE_RUNTIME_METRIC_NAMES:
            key = f"opspace/{pool}/{name}"
            if key in result:
                metrics[key] = result[key]
        for name in OPSPACE_FINAL_RUNTIME_METRIC_NAMES:
            key = f"opspace/{pool}/final/{name}"
            if key in result:
                metrics[key] = result[key]
    return metrics


def create_benchmark_forward_step(model, sharded_fns, cfg):
    version = str(cfg["model"].get("model_version", ""))

    @jax.jit
    def forward_step(params, input_ids, attention_mask, rng, step):
        apply_kwargs = benchmark_apply_kwargs(
            cfg, version, sharded_fns, attention_mask, rng, step)
        result = model.apply(
            {"params": params}, input_ids, labels=input_ids, **apply_kwargs)
        loss = result["loss"] + result.get("aux_loss", jnp.float32(0.0))
        metrics = {
            "loss": loss,
            "ce_loss": result["loss"],
            "aux_loss": result.get("aux_loss", jnp.float32(0.0)),
            "valid_count": result.get("valid_count", jnp.float32(0.0)),
        }
        return copy_runtime_metrics(result, metrics)

    return forward_step


def create_benchmark_train_step(model, optimizer, sharded_fns, cfg):
    optax_mod = require_optax()
    version = str(cfg["model"].get("model_version", ""))

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, rng, step):
        def loss_fn(p):
            apply_kwargs = benchmark_apply_kwargs(
                cfg, version, sharded_fns, attention_mask, rng, step)
            result = model.apply(
                {"params": p}, input_ids, labels=input_ids, **apply_kwargs)
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
        return params, opt_state, copy_runtime_metrics(result, metrics)

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


def block_value(value):
    jax.block_until_ready(jax.tree.leaves(value))
    return value


def hbm_peak_value(hbm):
    return num(hbm.get("hbm_peak_gb"))


def hbm_used_value(hbm):
    return num(hbm.get("hbm_used_gb"))


def update_peak_hbm(current_peak, hbm):
    peak = hbm_peak_value(hbm)
    if peak is None:
        return current_peak
    return peak if current_peak is None else max(current_peak, peak)


def benchmark_profile_record(run_index, variant_label, phase, step, op, group,
                             seconds, batch_size, seq_len, hbm,
                             hbm_before=None, layer=None, note=None):
    used = hbm_used_value(hbm)
    peak = hbm_peak_value(hbm)
    before_used = hbm_used_value(hbm_before or {})
    before_peak = hbm_peak_value(hbm_before or {})
    tokens_per_second = (
        float(batch_size) * float(seq_len) / float(seconds)
        if seconds and seconds > 0.0 else None)
    record = {
        "type": "benchmark_profile",
        "run_index": int(run_index),
        "variant": variant_label,
        "phase": phase,
        "step": int(step),
        "op": op,
        "group": group,
        "seconds": float(seconds),
        "tokens_per_second": json_float(tokens_per_second, None),
        "hbm_used_gb": json_float(used, None),
        "hbm_peak_gb": json_float(peak, None),
        "hbm_used_delta_gb": json_float(
            used - before_used
            if used is not None and before_used is not None else None,
            None),
        "hbm_peak_delta_gb": json_float(
            peak - before_peak
            if peak is not None and before_peak is not None else None,
            None),
    }
    if layer is not None:
        record["layer"] = int(layer)
    if note:
        record["note"] = note
    return record


def short_exception(exc, limit=260):
    text = f"{type(exc).__name__}: {exc}"
    text = " ".join(str(text).split())
    if len(text) > limit:
        return text[:limit - 3] + "..."
    return text


def traceback_text(limit=18):
    return "".join(traceback.format_exc(limit=limit)).strip()


def benchmark_profile_error_record(run_index, variant_label, phase, error):
    return {
        "type": "benchmark_profile_error",
        "run_index": int(run_index),
        "variant": variant_label,
        "phase": phase,
        "error": short_exception(error),
        "traceback": traceback_text(),
    }


def log_profile_error(phase, error):
    error_text = short_exception(error)
    _log(f"[profile-error] {phase} failed: {error_text}")
    trace = traceback_text(limit=14)
    if trace:
        for line in trace.splitlines()[-12:]:
            _log(f"[profile-trace] {line}")
    return error_text


def profile_timed_call(fn, *args):
    t0 = time.perf_counter()
    value = fn(*args)
    block_value(value)
    return value, time.perf_counter() - t0


def create_module_profile_fns(cfg, sharded_fns):
    version = str(cfg["model"].get("model_version", ""))
    module = importlib.import_module(MODEL_MODULES[version][0])
    t = cfg["training"]
    m = cfg["model"]
    n_layers = int(m["n_layers"])
    n_heads = int(m["n_heads"])
    d_model = int(m["d_model"])
    n_qk = int(m.get("n_qk", m.get("n_q", 0)))
    n_v = int(m.get("n_v", 0))
    dropout_rate = float(m.get("dropout", m.get("dropout_rate", 0.0)))
    router_dropout = float(m.get("router_dropout", 0.0))
    soft_gate_t = float(t.get("soft_gate_temperature", 0.07))
    boundary_power = float(t.get(
        "soft_gate_boundary_power_final",
        t.get("soft_gate_boundary_power_mid",
              t.get("soft_gate_boundary_power_start", 4.0))))
    is_v4172 = version == V4172_MODEL_VERSION
    den_power_key = "gate_den_power" if is_v4172 else "admission_den_power"
    den_power = float(
        m.get(den_power_key, t.get(den_power_key, 1.0)))

    def execution_kwargs():
        kwargs = {den_power_key: den_power}
        if not is_v4172:
            kwargs["execution_prune_eps"] = 0.0
        return kwargs
    tokens_per_step = (
        int(t["batch_size"])
        * int(m.get("max_seq_len", 512)))

    def training_tokens(step):
        return (
            jnp.asarray(step, dtype=jnp.float32)
            * jnp.asarray(float(tokens_per_step), dtype=jnp.float32))

    @jax.jit
    def embed_step(token_embedding, pos_embedding, input_ids):
        positions = jnp.arange(input_ids.shape[1])[jnp.newaxis, :]
        return token_embedding[input_ids] + pos_embedding[positions]

    @jax.jit
    def attn_step(pool_params, router_params, block_params, x, rng, step):
        normed = module._layer_norm(
            x, block_params["norm1"]["scale"], block_params["norm1"]["bias"])
        if version == V4168_MODEL_VERSION:
            attn_out, sector_diag, opspace_diag, opspace_aux = (
                module._attn_forward_minimal(
                    normed, pool_params, router_params,
                    block_params["attn"]["expand_O"]["kernel"], rng,
                    n_qk, n_v, n_heads, d_model, n_layers,
                    router_dropout, dropout_rate, True,
                    sharded_fns=sharded_fns,
                    soft_gate_temperature=soft_gate_t,
                    soft_gate_t_final=soft_gate_t,
                    soft_gate_T_qk=soft_gate_t,
                    soft_gate_T_v=soft_gate_t,
                    soft_gate_boundary_power=boundary_power,
                    soft_gate_boundary_power_final=boundary_power,
                    **execution_kwargs(),
                    training_tokens=training_tokens(step)))
            diag_guard = (
                jnp.sum(sector_diag)
                + jnp.sum(opspace_diag)
                + module._opspace_aux_loss(opspace_aux))
        else:
            attn_out = module._attn_forward_minimal(
                normed, pool_params, router_params,
                block_params["attn"]["expand_O"]["kernel"], rng,
                n_qk, n_v, n_heads, d_model, n_layers,
                router_dropout, dropout_rate, True,
                sharded_fns=sharded_fns,
                soft_gate_temperature=soft_gate_t,
                soft_gate_t_final=soft_gate_t,
                soft_gate_T_qk=soft_gate_t,
                soft_gate_T_v=soft_gate_t,
                soft_gate_boundary_power=boundary_power,
                soft_gate_boundary_power_final=boundary_power,
                **execution_kwargs())
            diag_guard = jnp.mean(attn_out)
        return x + attn_out, diag_guard

    @jax.jit
    def rst_step(pool_params, router_params, block_params, x, rng, step):
        normed = module._layer_norm(
            x, block_params["norm2"]["scale"], block_params["norm2"]["bias"])
        if version == V4168_MODEL_VERSION:
            rst_out, sector_diag, opspace_diag, opspace_aux = (
                module._rst_forward_minimal(
                    normed, pool_params, router_params, rng,
                    router_dropout, dropout_rate, True,
                    sharded_fns=sharded_fns,
                    d_model=d_model,
                    n_layers=n_layers,
                    soft_gate_temperature=soft_gate_t,
                    soft_gate_t_final=soft_gate_t,
                    soft_gate_T_rst=soft_gate_t,
                    soft_gate_boundary_power=boundary_power,
                    soft_gate_boundary_power_final=boundary_power,
                    **execution_kwargs(),
                    training_tokens=training_tokens(step)))
            diag_guard = (
                jnp.sum(sector_diag)
                + jnp.sum(opspace_diag)
                + module._opspace_aux_loss(opspace_aux))
        else:
            rst_out = module._rst_forward_minimal(
                normed, pool_params, router_params, rng,
                router_dropout, dropout_rate, True,
                sharded_fns=sharded_fns,
                d_model=d_model,
                n_layers=n_layers,
                soft_gate_temperature=soft_gate_t,
                soft_gate_t_final=soft_gate_t,
                soft_gate_T_rst=soft_gate_t,
                soft_gate_boundary_power=boundary_power,
                soft_gate_boundary_power_final=boundary_power,
                **execution_kwargs())
            diag_guard = jnp.mean(rst_out)
        return x + rst_out, diag_guard

    @jax.jit
    def final_loss_step(norm_scale, norm_bias, embedding_matrix, x, labels):
        x = module._layer_norm(x, norm_scale, norm_bias)
        shift_x = x[:, :-1, :]
        shift_labels = labels[:, 1:].astype(jnp.int32)
        valid_mask = shift_labels != -100
        loss, correct, valid_count = module._chunked_ce_loss_and_acc(
            shift_x, embedding_matrix, shift_labels, valid_mask)
        return {
            "loss": loss,
            "correct": correct,
            "valid_count": valid_count,
        }

    return {
        "module": module,
        "model_version": version,
        "embed_step": embed_step,
        "attn_step": attn_step,
        "rst_step": rst_step,
        "final_loss_step": final_loss_step,
        "n_layers": n_layers,
    }


def run_module_profile_pass(profile_fns, params, input_ids, rng, step,
                            run_index, variant_label, batch_size,
                            seq_len, phase, record=True):
    module = profile_fns["module"]
    records = []
    hbm_before = collect_hbm_stats()
    t0 = time.perf_counter()
    pool_params = module._pool_params_with_operator_keys(params["neuron_pool"])
    block_value(pool_params)
    seconds = time.perf_counter() - t0
    hbm_after = collect_hbm_stats()
    if record:
        records.append(benchmark_profile_record(
            run_index, variant_label, phase, step, "op_key_setup", "setup",
            seconds, batch_size, seq_len, hbm_after,
            hbm_before=hbm_before,
            note="shared op-key materialization"))

    hbm_before = hbm_after
    (x, seconds) = profile_timed_call(
        profile_fns["embed_step"],
        params["token_emb"]["embedding"],
        params["pos_emb"]["embedding"],
        input_ids)
    hbm_after = collect_hbm_stats()
    if record:
        records.append(benchmark_profile_record(
            run_index, variant_label, phase, step, "embedding", "embedding",
            seconds, batch_size, seq_len, hbm_after,
            hbm_before=hbm_before))

    router_params = params["router"]
    layer_rngs = jax.random.split(rng, int(profile_fns["n_layers"]))
    step_array = jnp.asarray(step, dtype=jnp.int32)
    for layer_idx in range(int(profile_fns["n_layers"])):
        layer_rng = layer_rngs[layer_idx]
        layer_rng, rng_attn, rng_rst = jax.random.split(layer_rng, 3)
        block_params = params[f"block_{layer_idx}"]

        hbm_before = hbm_after
        (attn_result, seconds) = profile_timed_call(
            profile_fns["attn_step"],
            pool_params, router_params, block_params, x, rng_attn, step_array)
        x = attn_result[0]
        hbm_after = collect_hbm_stats()
        if record:
            records.append(benchmark_profile_record(
                run_index, variant_label, phase, step,
                f"layer_{layer_idx:02d}.attn", "attn",
                seconds, batch_size, seq_len, hbm_after,
                hbm_before=hbm_before, layer=layer_idx))

        hbm_before = hbm_after
        (rst_result, seconds) = profile_timed_call(
            profile_fns["rst_step"],
            pool_params, router_params, block_params, x, rng_rst, step_array)
        x = rst_result[0]
        hbm_after = collect_hbm_stats()
        if record:
            records.append(benchmark_profile_record(
                run_index, variant_label, phase, step,
                f"layer_{layer_idx:02d}.rst", "rst",
                seconds, batch_size, seq_len, hbm_after,
                hbm_before=hbm_before, layer=layer_idx))

    hbm_before = hbm_after
    (_loss_metrics, seconds) = profile_timed_call(
        profile_fns["final_loss_step"],
        params["norm"]["scale"],
        params["norm"]["bias"],
        params["token_emb"]["embedding"],
        x,
        input_ids)
    hbm_after = collect_hbm_stats()
    if record:
        records.append(benchmark_profile_record(
            run_index, variant_label, phase, step, "final_norm_ce_loss",
            "loss", seconds, batch_size, seq_len, hbm_after,
            hbm_before=hbm_before,
            note="final layer norm plus chunked CE"))
    return records


def create_v4174_detailed_profile_fns(
        cfg, sharded_fns, *, include_backward=False):
    """Split the fused bundle path at its material execution boundaries."""
    module = importlib.import_module(MODEL_MODULES[V4174_MODEL_VERSION][0])
    m = cfg["model"]
    t = cfg["training"]
    n_layers = int(m["n_layers"])
    n_heads = int(m["n_heads"])
    d_model = int(m["d_model"])
    dropout_rate = float(m.get("dropout", m.get("dropout_rate", 0.0)))
    soft_gate_t = float(t.get("soft_gate_temperature", 0.07))
    boundary_power = float(t.get(
        "soft_gate_boundary_power_final",
        t.get("soft_gate_boundary_power_mid",
              t.get("soft_gate_boundary_power_start", 4.0))))
    attention_dense = sharded_fns.get("attention_space_dense")
    rst_dense = sharded_fns.get("rst_space_dense")
    vocab_embed = sharded_fns.get("vocab_parallel_embedding")
    vocab_ce_loss = sharded_fns.get("vocab_ce_loss")
    if attention_dense is None or rst_dense is None:
        raise RuntimeError(
            "v4174 detailed profile requires fused attention and RST "
            "production executors.")
    if vocab_embed is None or vocab_ce_loss is None:
        raise RuntimeError(
            "v4174 detailed profile requires canonical vocab-parallel "
            "embedding and CE loss.")
    qk_scale, v_scale, rst_scale = module._shared_pool_output_scales(
        d_model, n_layers)
    collect_metrics = jnp.asarray(False, dtype=jnp.bool_)

    @jax.jit
    def embed_step(token_embedding, pos_embedding, input_ids):
        positions = jnp.arange(input_ids.shape[1])[jnp.newaxis, :]
        return (
            vocab_embed(input_ids, token_embedding)
            + pos_embedding[positions])

    @jax.jit
    def attn_norm_step(block_params, x):
        return module._shared_layer_norm(
            x, block_params["norm1"]["scale"], block_params["norm1"]["bias"])

    def qkv_bundle_srw(
            operator_bank_params, operation_space_params, normed):
        space_selector_params = operation_space_params["space_selector"]
        space_interface_params = operation_space_params["space_interface"]
        operator_controller_params = operation_space_params[
            "operator_controller"]
        flat_state = normed.reshape((-1, d_model))
        q, k, v, _metrics = attention_dense(
            flat_state,
            space_selector_params["space_query_proj"]["kernel"],
            space_selector_params["space_route_keys"],
            space_interface_params["space_read_proj"],
            space_interface_params["space_write_proj"],
            operator_controller_params["q_tau_kernel"],
            operator_controller_params["q_tau_bias"],
            operator_controller_params["k_tau_kernel"],
            operator_controller_params["k_tau_bias"],
            operator_controller_params["v_tau_kernel"],
            operator_controller_params["v_tau_bias"],
            operator_bank_params["q_read_vectors"],
            operator_bank_params["q_write_vectors"],
            operator_bank_params["k_read_vectors"],
            operator_bank_params["k_write_vectors"],
            operator_bank_params["v_read_vectors"],
            operator_bank_params["v_write_vectors"],
            jnp.float32(soft_gate_t),
            jnp.float32(soft_gate_t),
            jnp.float32(boundary_power),
            jnp.float32(0.0),
            qk_scale,
            v_scale,
            collect_metrics)
        return (
            q.reshape(normed.shape),
            k.reshape(normed.shape),
            v.reshape(normed.shape))

    qkv_bundle_srw_step = jax.jit(qkv_bundle_srw)

    def gradient_checksum(grads):
        leaves = jax.tree.leaves(grads)
        return sum(
            jnp.sum(jnp.square(leaf.astype(jnp.float32)))
            for leaf in leaves)

    if include_backward:
        @jax.jit
        def qkv_bundle_srw_fwd_bwd_step(
                operator_bank_params, operation_space_params, normed):
            def objective(
                    exact_operator_bank,
                    exact_operation_space_params,
                    exact_normed):
                q, k, v = qkv_bundle_srw(
                    exact_operator_bank,
                    exact_operation_space_params,
                    exact_normed)
                return (
                    jnp.mean(q)
                    + jnp.float32(0.5) * jnp.mean(k)
                    + jnp.float32(0.25) * jnp.mean(v))

            value, grads = jax.value_and_grad(
                objective, argnums=(0, 1, 2))(
                    operator_bank_params, operation_space_params, normed)
            return value, gradient_checksum(grads)
    else:
        qkv_bundle_srw_fwd_bwd_step = None

    @jax.jit
    def attn_core_step(q, k, v, rng):
        batch_size, sequence_length = q.shape[:2]
        d_head = d_model // n_heads

        def heads(value):
            return value.reshape(
                batch_size, sequence_length, n_heads, d_head
            ).transpose(0, 2, 1, 3)

        context = module._causal_attention_core(
            heads(q), heads(k), heads(v),
            dropout_rate, True, rng, throughput_bf16=True)
        return context.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, d_model)

    @jax.jit
    def attn_out_proj_step(block_params, x, context):
        update = module._throughput_linear_bf16_f32(
            block_params["attn"]["expand_O"], context)
        return x + update

    @jax.jit
    def rst_norm_step(block_params, x):
        return module._shared_layer_norm(
            x, block_params["norm2"]["scale"], block_params["norm2"]["bias"])

    def rst_bundle_srw(
            operator_bank_params, operation_space_params, normed, x):
        space_selector_params = operation_space_params["space_selector"]
        space_interface_params = operation_space_params["space_interface"]
        operator_controller_params = operation_space_params[
            "operator_controller"]
        flat_state = normed.reshape((-1, d_model))
        update, _metrics = rst_dense(
            flat_state,
            space_selector_params["space_query_proj"]["kernel"],
            space_selector_params["space_route_keys"],
            space_interface_params["space_read_proj"],
            space_interface_params["space_write_proj"],
            operator_controller_params["rst_tau_kernel"],
            operator_controller_params["rst_tau_bias"],
            operator_bank_params["rst_read_vectors"],
            operator_bank_params["rst_write_vectors"],
            jnp.float32(soft_gate_t),
            jnp.float32(boundary_power),
            jnp.float32(0.0),
            rst_scale,
            collect_metrics)
        return x + update.reshape(normed.shape)

    rst_bundle_srw_step = jax.jit(rst_bundle_srw)

    if include_backward:
        @jax.jit
        def rst_bundle_srw_fwd_bwd_step(
                operator_bank_params, operation_space_params, normed, x):
            def objective(
                    exact_operator_bank,
                    exact_operation_space_params,
                    exact_normed):
                update = rst_bundle_srw(
                    exact_operator_bank,
                    exact_operation_space_params,
                    exact_normed, x)
                return jnp.mean(update)

            value, grads = jax.value_and_grad(
                objective, argnums=(0, 1, 2))(
                    operator_bank_params, operation_space_params, normed)
            return value, gradient_checksum(grads)
    else:
        rst_bundle_srw_fwd_bwd_step = None

    @jax.jit
    def final_loss_step(norm_scale, norm_bias, embedding_matrix, x, labels):
        x = module._shared_layer_norm(x, norm_scale, norm_bias)
        shift_x = x[:, :-1, :]
        shift_labels = labels[:, 1:].astype(jnp.int32)
        valid = shift_labels != -100
        return {
            "loss": vocab_ce_loss(
                shift_x, embedding_matrix, shift_labels, valid),
            "valid_count": valid.sum().astype(jnp.int32),
        }

    return {
        "profile_kind": "v4174_bundle",
        "module": module,
        "model_version": V4174_MODEL_VERSION,
        "embed_step": embed_step,
        "attn_norm_step": attn_norm_step,
        "qkv_bundle_srw_step": qkv_bundle_srw_step,
        "qkv_bundle_srw_fwd_bwd_step": qkv_bundle_srw_fwd_bwd_step,
        "attn_core_step": attn_core_step,
        "attn_out_proj_step": attn_out_proj_step,
        "rst_norm_step": rst_norm_step,
        "rst_bundle_srw_step": rst_bundle_srw_step,
        "rst_bundle_srw_fwd_bwd_step": rst_bundle_srw_fwd_bwd_step,
        "final_loss_step": final_loss_step,
        "n_layers": n_layers,
        "include_backward_profile": bool(include_backward),
    }


def create_detailed_profile_fns(
        cfg, sharded_fns, *, include_backward=False):
    version = str(cfg["model"].get("model_version", ""))
    if version == V4174_MODEL_VERSION:
        return create_v4174_detailed_profile_fns(
            cfg, sharded_fns, include_backward=include_backward)
    if include_backward:
        raise RuntimeError(
            "Detailed backward profile is currently defined for v4174 only.")
    module = importlib.import_module(MODEL_MODULES[version][0])
    t = cfg["training"]
    m = cfg["model"]
    n_layers = int(m["n_layers"])
    n_heads = int(m["n_heads"])
    d_model = int(m["d_model"])
    dropout_rate = float(m.get("dropout", m.get("dropout_rate", 0.0)))
    router_dropout = float(m.get("router_dropout", 0.0))
    del dropout_rate, router_dropout
    soft_gate_t = float(t.get("soft_gate_temperature", 0.07))
    boundary_power = float(t.get(
        "soft_gate_boundary_power_final",
        t.get("soft_gate_boundary_power_mid",
              t.get("soft_gate_boundary_power_start", 4.0))))
    tokens_per_step = (
        int(t["batch_size"])
        * int(m.get("max_seq_len", 512)))
    opspace_tau_free = (
        version == V4168_MODEL_VERSION
        and isinstance(sharded_fns, dict)
        and bool(sharded_fns.get("operation_space_tau_free", False)))
    direct_state_query = (
        str(m.get("operator_query_mode", ""))
        == "direct_state_projection")
    vocab_embed = (
        sharded_fns.get("vocab_parallel_embedding")
        if isinstance(sharded_fns, dict) else None)
    vocab_ce_loss = (
        sharded_fns.get("vocab_ce_loss")
        if isinstance(sharded_fns, dict) else None)

    if isinstance(sharded_fns, dict):
        fused_paired_qk = sharded_fns.get(
            "attn_qk_paired_minimal",
            sharded_fns.get(
                "attn_qk_paired", sharded_fns.get("paired", None)))
        fused_single_v = sharded_fns.get(
            "attn_v_single_minimal",
            sharded_fns.get("attn_v_single", sharded_fns.get("single", None)))
        fused_single_rst = sharded_fns.get(
            "rst_single_minimal",
            sharded_fns.get("rst_single", sharded_fns.get("single", None)))
    else:
        fused_single_v, fused_paired_qk = sharded_fns
        fused_single_rst, _unused_paired = sharded_fns
    if fused_paired_qk is None:
        raise RuntimeError("Detailed profile requires a QK paired SRW executor.")
    if fused_single_v is None:
        raise RuntimeError("Detailed profile requires a V SRW executor.")
    if fused_single_rst is None:
        raise RuntimeError("Detailed profile requires an RST SRW executor.")

    call_minimal_srw_fn = getattr(module, "_call_minimal_srw", None)
    split_minimal_srw_return_fn = getattr(
        module, "_split_minimal_srw_return", None)

    def training_tokens(step):
        return (
            jnp.asarray(step, dtype=jnp.float32)
            * jnp.asarray(float(tokens_per_step), dtype=jnp.float32))

    def call_minimal_srw(fused, x, h, op_key, raw_tau, read, write,
                         step):
        if call_minimal_srw_fn is not None:
            return call_minimal_srw_fn(
                fused, x, h, op_key, raw_tau, read, write,
                soft_gate_t, soft_gate_t, boundary_power, boundary_power,
                0.0, training_tokens(step))
        return fused(
            x, h, op_key, raw_tau, read, write,
            soft_gate_t, soft_gate_t, boundary_power, boundary_power, 0.0)

    def split_minimal_srw_return(ret):
        if split_minimal_srw_return_fn is not None:
            return split_minimal_srw_return_fn(ret)
        out = ret[0] if isinstance(ret, tuple) else ret
        zero = jnp.float32(0.0)
        return out, zero, zero, zero

    def first_scalar(value):
        arr = jnp.asarray(value)
        if arr.size == 0:
            return jnp.float32(0.0)
        return jnp.ravel(arr.astype(jnp.float32))[0]

    def guard_from_values(*values):
        total = jnp.float32(0.0)
        leaves = []
        for value in values:
            leaves.extend(jax.tree.leaves(value))
        for leaf in leaves[:8]:
            total = total + first_scalar(leaf)
        return total

    @jax.jit
    def embed_step(token_embedding, pos_embedding, input_ids):
        positions = jnp.arange(input_ids.shape[1])[jnp.newaxis, :]
        token_state = (
            vocab_embed(input_ids, token_embedding)
            if vocab_embed is not None else token_embedding[input_ids])
        return token_state + pos_embedding[positions]

    @jax.jit
    def attn_norm_step(block_params, x):
        return module._layer_norm(
            x, block_params["norm1"]["scale"], block_params["norm1"]["bias"])

    @jax.jit
    def attn_route_step(router_params, normed):
        B, S = normed.shape[:2]
        attn_read_query = (
            normed @ router_params["proj_attn"]["kernel"]
            + router_params["proj_attn"]["bias"])
        q_read_query, k_read_query, v_read_query = jnp.split(
            attn_read_query, 3, axis=-1)
        if direct_state_query:
            h_q, h_k, h_v = q_read_query, k_read_query, v_read_query
        else:
            h_q = module._read_write_operator_query(
                q_read_query, normed,
                router_params["q_op_write_query_proj"])
            h_k = module._read_write_operator_query(
                k_read_query, normed,
                router_params["k_op_write_query_proj"])
            h_v = module._read_write_operator_query(
                v_read_query, normed,
                router_params["v_op_write_query_proj"])
        if opspace_tau_free:
            raw_tau_qk = jnp.zeros((B, S, 2, 1), dtype=normed.dtype)
            raw_tau_v = jnp.zeros((B, S, 1), dtype=normed.dtype)
        else:
            raw_tau_all = (
                normed @ router_params["raw_tau_attn"]["kernel"]
                + router_params["raw_tau_attn"]["bias"])
            raw_tau_qk = jnp.stack(
                [raw_tau_all[:, :, 0:1], raw_tau_all[:, :, 1:2]], axis=2)
            raw_tau_v = raw_tau_all[:, :, 2:3]
        return h_q, h_k, h_v, raw_tau_qk, raw_tau_v, guard_from_values(
            h_q, h_k, h_v, raw_tau_qk, raw_tau_v)

    @jax.jit
    def qk_srw_step(pool_params, normed, h_q, h_k, raw_tau_qk, step):
        h_qk = jnp.stack([h_q, h_k], axis=2)
        qk_ret = call_minimal_srw(
            fused_paired_qk,
            normed,
            h_qk,
            pool_params["attn_qk_op_key"],
            raw_tau_qk,
            pool_params["attn_qk_read"],
            pool_params["attn_qk_write"],
            step)
        qk_out = qk_ret[0] if isinstance(qk_ret, tuple) else qk_ret
        qk_scale, _v_scale, _rst_scale = module._effective_pool_output_scales(
            pool_params, d_model, n_layers)
        q = qk_out[:, :, 0, :] * qk_scale
        k = qk_out[:, :, 1, :] * qk_scale
        return q, k, guard_from_values(qk_ret)

    @jax.jit
    def v_srw_step(pool_params, normed, h_v, raw_tau_v, step):
        v_ret = call_minimal_srw(
            fused_single_v,
            normed,
            h_v,
            pool_params["attn_v_op_key"],
            raw_tau_v,
            pool_params["attn_v_read"],
            pool_params["attn_v_write"],
            step)
        v_out, v_sector_diag, v_opspace_diag, v_aux = (
            split_minimal_srw_return(v_ret))
        _qk_scale, v_scale, _rst_scale = module._effective_pool_output_scales(
            pool_params, d_model, n_layers)
        v_out = v_out * v_scale
        return v_out, guard_from_values(
            v_sector_diag, v_opspace_diag, v_aux)

    @jax.jit
    def attn_core_step(q, k, v):
        B, S, D = q.shape
        d_head = D // n_heads
        q = q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = (
            jnp.einsum("bhsd,bhtd->bhst", q, k)
            / jnp.sqrt(jnp.float32(d_head)))
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(
            causal[None, None, :, :],
            scores,
            jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhst,bhtd->bhsd", weights, v)
        return context.transpose(0, 2, 1, 3).reshape(B, S, D)

    @jax.jit
    def attn_out_proj_step(block_params, x, context):
        attn_out = context @ block_params["attn"]["expand_O"]["kernel"]
        return x + attn_out

    @jax.jit
    def rst_norm_step(block_params, x):
        return module._layer_norm(
            x, block_params["norm2"]["scale"], block_params["norm2"]["bias"])

    @jax.jit
    def rst_route_step(router_params, normed):
        B, S = normed.shape[:2]
        rst_read_query = (
            normed @ router_params["proj_rst"]["kernel"]
            + router_params["proj_rst"]["bias"])
        h_rst = (
            rst_read_query
            if direct_state_query else module._read_write_operator_query(
                rst_read_query, normed,
                router_params["rst_op_write_query_proj"]))
        if opspace_tau_free:
            raw_tau_rst = jnp.zeros((B, S, 1), dtype=normed.dtype)
        else:
            raw_tau_rst = (
                normed @ router_params["raw_tau_rst"]["kernel"]
                + router_params["raw_tau_rst"]["bias"])
        return h_rst, raw_tau_rst, guard_from_values(h_rst, raw_tau_rst)

    @jax.jit
    def rst_srw_step(pool_params, normed, h_rst, raw_tau_rst, x, step):
        rst_ret = call_minimal_srw(
            fused_single_rst,
            normed,
            h_rst,
            pool_params["rst_op_key"],
            raw_tau_rst,
            pool_params["rst_read"],
            pool_params["rst_write"],
            step)
        rst_out, rst_sector_diag, rst_opspace_diag, rst_aux = (
            split_minimal_srw_return(rst_ret))
        _qk_scale, _v_scale, rst_scale = module._effective_pool_output_scales(
            pool_params, d_model, n_layers)
        rst_out = rst_out * rst_scale
        return x + rst_out, guard_from_values(
            rst_sector_diag, rst_opspace_diag, rst_aux)

    @jax.jit
    def final_loss_step(norm_scale, norm_bias, embedding_matrix, x, labels):
        x = module._layer_norm(x, norm_scale, norm_bias)
        shift_x = x[:, :-1, :]
        shift_labels = labels[:, 1:].astype(jnp.int32)
        valid_mask = shift_labels != -100
        if vocab_ce_loss is not None:
            loss = vocab_ce_loss(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            correct = jnp.int32(0)
            valid_count = valid_mask.sum().astype(jnp.int32)
        else:
            loss, correct, valid_count = module._chunked_ce_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
        return {
            "loss": loss,
            "correct": correct,
            "valid_count": valid_count,
        }

    return {
        "module": module,
        "model_version": version,
        "embed_step": embed_step,
        "attn_norm_step": attn_norm_step,
        "attn_route_step": attn_route_step,
        "qk_srw_step": qk_srw_step,
        "v_srw_step": v_srw_step,
        "attn_core_step": attn_core_step,
        "attn_out_proj_step": attn_out_proj_step,
        "rst_norm_step": rst_norm_step,
        "rst_route_step": rst_route_step,
        "rst_srw_step": rst_srw_step,
        "final_loss_step": final_loss_step,
        "n_layers": n_layers,
    }


def run_v4174_detailed_profile_pass(
        profile_fns, params, input_ids, step, run_index, variant_label,
        batch_size, seq_len, phase, record=True):
    records = []
    total_seconds = 0.0

    def add_record(op, group, seconds, hbm_after, hbm_before,
                   layer=None, note=None):
        nonlocal total_seconds
        total_seconds += float(seconds)
        if record:
            records.append(benchmark_profile_record(
                run_index, variant_label, phase, step, op, group,
                seconds, batch_size, seq_len, hbm_after,
                hbm_before=hbm_before, layer=layer, note=note))

    def timed_call(label, fn, *args):
        if not record:
            _log(f"[profile-compile] start {label}")
        value, seconds = profile_timed_call(fn, *args)
        if not record:
            _log(f"[profile-compile] done {label} {seconds:.3f}s")
        return value, seconds

    hbm_before = collect_hbm_stats()
    (x, seconds) = timed_call(
        "embedding",
        profile_fns["embed_step"],
        params["token_emb"]["embedding"],
        params["pos_emb"]["embedding"],
        input_ids)
    hbm_after = collect_hbm_stats()
    add_record("embedding", "embedding", seconds, hbm_after, hbm_before)

    operator_bank_params = params["operator_bank"]
    operation_space_params = {
        "space_selector": params["space_selector"],
        "space_interface": params["space_interface"],
        "operator_controller": params["operator_controller"],
    }
    layer_rngs = jax.random.split(
        jax.random.PRNGKey(0), int(profile_fns["n_layers"]))
    layer_count = int(profile_fns["n_layers"])
    profile_layer_limit = int(
        profile_fns.get("profile_layer_limit", 0))
    if profile_layer_limit > 0:
        layer_count = min(layer_count, profile_layer_limit)
    for layer_idx in range(layer_count):
        block_params = params[f"block_{layer_idx}"]

        hbm_before = hbm_after
        (normed, seconds) = timed_call(
            f"layer_{layer_idx:02d}.attn_norm",
            profile_fns["attn_norm_step"], block_params, x)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_norm", "attn_norm",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (qkv, seconds) = timed_call(
            f"layer_{layer_idx:02d}.qkv_bundle_srw",
            profile_fns["qkv_bundle_srw_step"],
            operator_bank_params, operation_space_params, normed)
        q, k, v = qkv
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.qkv_bundle_srw", "qkv_bundle_srw",
            seconds, hbm_after, hbm_before, layer=layer_idx,
            note="routing + prefix-count bundle pack + Q/K/V SRW + writeback")
        if profile_fns.get("include_backward_profile"):
            hbm_before = hbm_after
            (_qkv_grad_checksum, seconds) = timed_call(
                f"layer_{layer_idx:02d}.qkv_bundle_srw_fwd_bwd",
                profile_fns["qkv_bundle_srw_fwd_bwd_step"],
                operator_bank_params, operation_space_params, normed)
            hbm_after = collect_hbm_stats()
            add_record(
                f"layer_{layer_idx:02d}.qkv_bundle_srw_fwd_bwd",
                "qkv_bundle_srw_fwd_bwd",
                seconds, hbm_after, hbm_before, layer=layer_idx,
                note=("QKV forward plus exact custom-VJP backward; "
                      "all gradients reduced to a device checksum"))

        hbm_before = hbm_after
        (context, seconds) = timed_call(
            f"layer_{layer_idx:02d}.attn_core",
            profile_fns["attn_core_step"],
            q, k, v, layer_rngs[layer_idx])
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_core", "attn_core",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (x, seconds) = timed_call(
            f"layer_{layer_idx:02d}.attn_out_proj",
            profile_fns["attn_out_proj_step"], block_params, x, context)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_out_proj", "attn_out_proj",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (normed, seconds) = timed_call(
            f"layer_{layer_idx:02d}.rst_norm",
            profile_fns["rst_norm_step"], block_params, x)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.rst_norm", "rst_norm",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        rst_input = x
        hbm_before = hbm_after
        (x, seconds) = timed_call(
            f"layer_{layer_idx:02d}.rst_bundle_srw",
            profile_fns["rst_bundle_srw_step"],
            operator_bank_params, operation_space_params,
            normed, rst_input)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.rst_bundle_srw", "rst_bundle_srw",
            seconds, hbm_after, hbm_before, layer=layer_idx,
            note="routing + prefix-count bundle pack + RST SRW + writeback")
        if profile_fns.get("include_backward_profile"):
            hbm_before = hbm_after
            (_rst_grad_checksum, seconds) = timed_call(
                f"layer_{layer_idx:02d}.rst_bundle_srw_fwd_bwd",
                profile_fns["rst_bundle_srw_fwd_bwd_step"],
                operator_bank_params, operation_space_params,
                normed, rst_input)
            hbm_after = collect_hbm_stats()
            add_record(
                f"layer_{layer_idx:02d}.rst_bundle_srw_fwd_bwd",
                "rst_bundle_srw_fwd_bwd",
                seconds, hbm_after, hbm_before, layer=layer_idx,
                note=("RST forward plus exact custom-VJP backward; "
                      "all gradients reduced to a device checksum"))

    hbm_before = hbm_after
    (_loss_metrics, seconds) = timed_call(
        "final_norm_ce_loss",
        profile_fns["final_loss_step"],
        params["norm"]["scale"],
        params["norm"]["bias"],
        params["token_emb"]["embedding"],
        x,
        input_ids)
    hbm_after = collect_hbm_stats()
    add_record(
        "final_norm_ce_loss", "loss", seconds, hbm_after, hbm_before,
        note="final layer norm plus vocab-parallel CE")
    return records, total_seconds


def run_detailed_profile_pass(profile_fns, params, input_ids, step,
                              run_index, variant_label, batch_size,
                              seq_len, phase, record=True):
    if profile_fns.get("profile_kind") == "v4174_bundle":
        return run_v4174_detailed_profile_pass(
            profile_fns, params, input_ids, step,
            run_index, variant_label, batch_size, seq_len, phase,
            record=record)
    module = profile_fns["module"]
    records = []
    total_seconds = 0.0

    def add_record(op, group, seconds, hbm_after, hbm_before,
                   layer=None, note=None):
        nonlocal total_seconds
        total_seconds += float(seconds)
        if not record:
            return
        records.append(benchmark_profile_record(
            run_index, variant_label, phase, step, op, group,
            seconds, batch_size, seq_len, hbm_after,
            hbm_before=hbm_before, layer=layer, note=note))

    hbm_before = collect_hbm_stats()
    t0 = time.perf_counter()
    pool_params = module._pool_params_with_operator_keys(params["neuron_pool"])
    block_value(pool_params)
    seconds = time.perf_counter() - t0
    hbm_after = collect_hbm_stats()
    add_record(
        "op_key_setup", "setup", seconds, hbm_after, hbm_before,
        note="shared op-key materialization")

    hbm_before = hbm_after
    (x, seconds) = profile_timed_call(
        profile_fns["embed_step"],
        params["token_emb"]["embedding"],
        params["pos_emb"]["embedding"],
        input_ids)
    hbm_after = collect_hbm_stats()
    add_record("embedding", "embedding", seconds, hbm_after, hbm_before)

    router_params = params["router"]
    step_array = jnp.asarray(step, dtype=jnp.int32)
    for layer_idx in range(int(profile_fns["n_layers"])):
        block_params = params[f"block_{layer_idx}"]

        hbm_before = hbm_after
        (normed, seconds) = profile_timed_call(
            profile_fns["attn_norm_step"], block_params, x)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_norm", "attn_norm",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (route_result, seconds) = profile_timed_call(
            profile_fns["attn_route_step"], router_params, normed)
        h_q, h_k, h_v, raw_tau_qk, raw_tau_v, _guard = route_result
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_route", "attn_route",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (qk_result, seconds) = profile_timed_call(
            profile_fns["qk_srw_step"],
            pool_params, normed, h_q, h_k, raw_tau_qk, step_array)
        q, k, _guard = qk_result
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.qk_srw", "qk_srw",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (v_result, seconds) = profile_timed_call(
            profile_fns["v_srw_step"],
            pool_params, normed, h_v, raw_tau_v, step_array)
        v, _guard = v_result
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.v_srw", "v_srw",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (context, seconds) = profile_timed_call(
            profile_fns["attn_core_step"], q, k, v)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_core", "attn_core",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (x, seconds) = profile_timed_call(
            profile_fns["attn_out_proj_step"], block_params, x, context)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.attn_out_proj", "attn_out_proj",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (normed, seconds) = profile_timed_call(
            profile_fns["rst_norm_step"], block_params, x)
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.rst_norm", "rst_norm",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (route_result, seconds) = profile_timed_call(
            profile_fns["rst_route_step"], router_params, normed)
        h_rst, raw_tau_rst, _guard = route_result
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.rst_route", "rst_route",
            seconds, hbm_after, hbm_before, layer=layer_idx)

        hbm_before = hbm_after
        (rst_result, seconds) = profile_timed_call(
            profile_fns["rst_srw_step"],
            pool_params, normed, h_rst, raw_tau_rst, x, step_array)
        x, _guard = rst_result
        hbm_after = collect_hbm_stats()
        add_record(
            f"layer_{layer_idx:02d}.rst_srw", "rst_srw",
            seconds, hbm_after, hbm_before, layer=layer_idx)

    hbm_before = hbm_after
    (_loss_metrics, seconds) = profile_timed_call(
        profile_fns["final_loss_step"],
        params["norm"]["scale"],
        params["norm"]["bias"],
        params["token_emb"]["embedding"],
        x,
        input_ids)
    hbm_after = collect_hbm_stats()
    add_record(
        "final_norm_ce_loss", "loss", seconds, hbm_after, hbm_before,
        note="final layer norm plus chunked CE")
    return records, total_seconds


def summarize_profile_records(records):
    measured = [
        r for r in records
        if r.get("phase") in ("fast_forward_measure", "module_measure")]
    groups = {}
    layer_rows = {}
    module_records = [
        r for r in measured if r.get("phase") == "module_measure"]
    module_total_s = sum(float(r.get("seconds", 0.0)) for r in module_records)
    for record in measured:
        group = record.get("group", "unknown")
        row = groups.setdefault(group, {
            "calls": 0,
            "total_seconds": 0.0,
            "max_hbm_peak_gb": None,
            "max_hbm_used_delta_gb": None,
        })
        row["calls"] += 1
        row["total_seconds"] += float(record.get("seconds", 0.0))
        peak = num(record.get("hbm_peak_gb"))
        delta = num(record.get("hbm_used_delta_gb"))
        if peak is not None:
            row["max_hbm_peak_gb"] = (
                peak if row["max_hbm_peak_gb"] is None
                else max(row["max_hbm_peak_gb"], peak))
        if delta is not None:
            row["max_hbm_used_delta_gb"] = (
                delta if row["max_hbm_used_delta_gb"] is None
                else max(row["max_hbm_used_delta_gb"], delta))
        if record.get("layer") is not None:
            layer = int(record["layer"])
            layer_row = layer_rows.setdefault(layer, {})
            layer_row[record["group"]] = record

    aggregates = []
    for group, row in groups.items():
        calls = max(1, int(row["calls"]))
        total_s = float(row["total_seconds"])
        pct_split = (
            total_s / module_total_s * 100.0
            if module_total_s and group != "fast_forward" else None)
        aggregates.append({
            "group": group,
            "calls": int(row["calls"]),
            "total_seconds": total_s,
            "mean_seconds": total_s / calls,
            "pct_module_split": pct_split,
            "max_hbm_peak_gb": row["max_hbm_peak_gb"],
            "max_hbm_used_delta_gb": row["max_hbm_used_delta_gb"],
        })
    order = {
        "fast_forward": 0,
        "setup": 1,
        "embedding": 2,
        "attn": 3,
        "rst": 4,
        "loss": 5,
    }
    aggregates.sort(key=lambda r: order.get(r["group"], 99))
    layer_summary = []
    for layer, row in sorted(layer_rows.items()):
        attn_s = num((row.get("attn") or {}).get("seconds"))
        rst_s = num((row.get("rst") or {}).get("seconds"))
        layer_summary.append({
            "layer": int(layer),
            "attn_seconds": attn_s,
            "rst_seconds": rst_s,
            "attn_hbm_peak_gb": num(
                (row.get("attn") or {}).get("hbm_peak_gb")),
            "rst_hbm_peak_gb": num(
                (row.get("rst") or {}).get("hbm_peak_gb")),
            "rst_over_attn": ratio_float(rst_s, attn_s),
        })
    return aggregates, layer_summary


def summarize_detailed_profile_records(records):
    measured = [
        r for r in records
        if r.get("phase") == "detailed_measure"]
    groups = {}
    layer_rows = {}
    detailed_total_s = sum(float(r.get("seconds", 0.0)) for r in measured)
    for record in measured:
        group = record.get("group", "unknown")
        row = groups.setdefault(group, {
            "calls": 0,
            "total_seconds": 0.0,
            "max_hbm_peak_gb": None,
            "max_hbm_used_delta_gb": None,
        })
        row["calls"] += 1
        row["total_seconds"] += float(record.get("seconds", 0.0))
        peak = num(record.get("hbm_peak_gb"))
        delta = num(record.get("hbm_used_delta_gb"))
        if peak is not None:
            row["max_hbm_peak_gb"] = (
                peak if row["max_hbm_peak_gb"] is None
                else max(row["max_hbm_peak_gb"], peak))
        if delta is not None:
            row["max_hbm_used_delta_gb"] = (
                delta if row["max_hbm_used_delta_gb"] is None
                else max(row["max_hbm_used_delta_gb"], delta))
        if record.get("layer") is not None:
            layer = int(record["layer"])
            layer_rows.setdefault(layer, {})[group] = record

    aggregates = []
    for group, row in groups.items():
        calls = max(1, int(row["calls"]))
        total_s = float(row["total_seconds"])
        pct_split = (
            total_s / detailed_total_s * 100.0
            if detailed_total_s else None)
        aggregates.append({
            "group": group,
            "calls": int(row["calls"]),
            "total_seconds": total_s,
            "mean_seconds": total_s / calls,
            "pct_detailed_split": pct_split,
            "max_hbm_peak_gb": row["max_hbm_peak_gb"],
            "max_hbm_used_delta_gb": row["max_hbm_used_delta_gb"],
        })
    order = {
        "setup": 1,
        "embedding": 2,
        "attn_norm": 3,
        "attn_route": 4,
        "qk_srw": 5,
        "v_srw": 6,
        "qkv_bundle_srw": 6,
        "attn_core": 7,
        "attn_out_proj": 8,
        "rst_norm": 9,
        "rst_route": 10,
        "rst_srw": 11,
        "rst_bundle_srw": 11,
        "loss": 12,
    }
    aggregates.sort(key=lambda r: order.get(r["group"], 99))

    def layer_seconds(row, group):
        return num((row.get(group) or {}).get("seconds"))

    def layer_peak(row, group):
        return num((row.get(group) or {}).get("hbm_peak_gb"))

    layer_summary = []
    for layer, row in sorted(layer_rows.items()):
        qk_s = layer_seconds(row, "qk_srw")
        v_s = layer_seconds(row, "v_srw")
        qkv_bundle_s = layer_seconds(row, "qkv_bundle_srw")
        attn_core_s = layer_seconds(row, "attn_core")
        rst_s = (
            layer_seconds(row, "rst_srw")
            or layer_seconds(row, "rst_bundle_srw"))
        qkv_s = (
            qkv_bundle_s
            if qkv_bundle_s is not None
            else (qk_s or 0.0) + (v_s or 0.0))
        layer_summary.append({
            "layer": int(layer),
            "attn_norm_seconds": layer_seconds(row, "attn_norm"),
            "attn_route_seconds": layer_seconds(row, "attn_route"),
            "qk_srw_seconds": qk_s,
            "v_srw_seconds": v_s,
            "qkv_bundle_srw_seconds": qkv_bundle_s,
            "attn_core_seconds": attn_core_s,
            "attn_out_proj_seconds": layer_seconds(row, "attn_out_proj"),
            "rst_norm_seconds": layer_seconds(row, "rst_norm"),
            "rst_route_seconds": layer_seconds(row, "rst_route"),
            "rst_srw_seconds": rst_s,
            "rst_bundle_srw_seconds": layer_seconds(
                row, "rst_bundle_srw"),
            "qkv_srw_seconds": qkv_s,
            "rst_over_qkv": ratio_float(rst_s, max(qkv_s, 1.0e-12)),
            "v_over_qk": ratio_float(v_s, qk_s),
            "rst_over_v": ratio_float(rst_s, v_s),
            "qk_srw_hbm_peak_gb": layer_peak(row, "qk_srw"),
            "v_srw_hbm_peak_gb": layer_peak(row, "v_srw"),
            "attn_core_hbm_peak_gb": layer_peak(row, "attn_core"),
            "rst_srw_hbm_peak_gb": layer_peak(row, "rst_srw"),
        })

    totals = {row["group"]: float(row["total_seconds"]) for row in aggregates}
    attn_srw_total = (
        totals.get("qkv_bundle_srw", 0.0)
        + totals.get("qk_srw", 0.0)
        + totals.get("v_srw", 0.0))
    attn_non_srw_total = (
        totals.get("attn_norm", 0.0)
        + totals.get("attn_route", 0.0)
        + totals.get("attn_core", 0.0)
        + totals.get("attn_out_proj", 0.0))
    rst_total = (
        totals.get("rst_norm", 0.0)
        + totals.get("rst_route", 0.0)
        + totals.get("rst_srw", 0.0)
        + totals.get("rst_bundle_srw", 0.0))
    derived = {
        "attn_srw_seconds": attn_srw_total,
        "attn_non_srw_seconds": attn_non_srw_total,
        "rst_total_seconds": rst_total,
        "rst_srw_over_qkv": ratio_float(
            totals.get("rst_srw", 0.0)
            + totals.get("rst_bundle_srw", 0.0),
            max(attn_srw_total, 1.0e-12)),
        "v_over_qk": ratio_float(
            totals.get("v_srw", 0.0), max(totals.get("qk_srw", 0.0),
                                          1.0e-12)),
    }
    return aggregates, layer_summary, derived


def summarize_runtime_metric_snapshots(snapshots):
    if not snapshots:
        return {}
    summary = {}
    for pool in RUNTIME_POOLS:
        row = {}
        for name in OPSPACE_RUNTIME_METRIC_NAMES:
            collect_runtime_metric(
                row, snapshots, f"opspace/{pool}/{name}", name)
        for name in OPSPACE_FINAL_RUNTIME_METRIC_NAMES:
            collect_runtime_metric(
                row, snapshots, f"opspace/{pool}/final/{name}", name)
        if row:
            summary[pool] = row
    return summary


def collect_runtime_metric(row, snapshots, key, name):
    values = []
    for metrics in snapshots:
        if key not in metrics:
            continue
        value = num(metrics.get(key))
        if value is not None:
            values.append(value)
    if values:
        row[name] = {
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }


def runtime_stat(summary, pool, name, stat="max"):
    row = (summary.get("runtime_metric_summary") or {}).get(pool, {})
    value = row.get(name, {})
    if isinstance(value, dict):
        return value.get(stat)
    return None


def profile_bottleneck_rows(summary):
    rows = []
    for row in summary.get("profile_aggregates", []) or []:
        group = row.get("group")
        if group == "fast_forward":
            continue
        total_s = num(row.get("total_seconds"))
        if total_s is None:
            continue
        rows.append({
            "group": group,
            "total_seconds": total_s,
            "mean_seconds": num(row.get("mean_seconds")),
            "pct_module_split": num(row.get("pct_module_split")),
            "max_hbm_peak_gb": num(row.get("max_hbm_peak_gb")),
        })
    rows.sort(key=lambda row: row["total_seconds"], reverse=True)
    return rows


def detailed_bottleneck_rows(summary):
    rows = []
    for row in summary.get("detailed_profile_aggregates", []) or []:
        total_s = num(row.get("total_seconds"))
        if total_s is None:
            continue
        rows.append({
            "group": row.get("group"),
            "total_seconds": total_s,
            "mean_seconds": num(row.get("mean_seconds")),
            "pct_detailed_split": num(row.get("pct_detailed_split")),
            "max_hbm_peak_gb": num(row.get("max_hbm_peak_gb")),
        })
    rows.sort(key=lambda row: row["total_seconds"], reverse=True)
    return rows


def top_layer_bottlenecks(summary):
    rows = summary.get("profile_layer_rows", []) or []
    attn_rows = [
        row for row in rows
        if row.get("attn_seconds") is not None
    ]
    rst_rows = [
        row for row in rows
        if row.get("rst_seconds") is not None
    ]
    top_attn = max(
        attn_rows, key=lambda row: num(row.get("attn_seconds")) or -1.0,
        default=None)
    top_rst = max(
        rst_rows, key=lambda row: num(row.get("rst_seconds")) or -1.0,
        default=None)
    top_ratio = max(
        rows, key=lambda row: num(row.get("rst_over_attn")) or -1.0,
        default=None)
    return top_attn, top_rst, top_ratio


def ratio_float(a, b):
    a = num(a)
    b = num(b)
    if a is None or b is None or abs(float(b)) < 1.0e-12:
        return None
    return float(a) / float(b)


def run_forward_profile(model, sharded_fns, params, cfg, args, iterator, loader,
                        data_sharding, batch_size, seq_len, rng, run_index,
                        variant_label, start_step, train_mean_seconds):
    records = []
    summary = {}
    next_step = int(start_step)
    peak_hbm = None
    fast_metric_snapshots = []
    fast_detailed_profile = bool(
        getattr(args, "fast_detailed_profile", False))

    def next_profile_batch():
        nonlocal iterator, rng, next_step
        next_step += 1
        batch, iterator = next_batch(iterator, loader)
        ids, mask = shard_batch_to_mesh(
            batch, data_sharding, batch_size, seq_len)
        rng, step_rng = jax.random.split(rng)
        return batch, ids, mask, step_rng, next_step

    fast_times = []
    fast_tokens = []
    if int(args.forward_profile_steps) > 0:
        try:
            forward_step = create_benchmark_forward_step(
                model, sharded_fns, cfg)
            _log("Compiling forward-only benchmark step...")
            batch, ids, mask, step_rng, step_no = next_profile_batch()
            hbm_before = collect_hbm_stats()
            t0 = time.perf_counter()
            metrics = forward_step(
                params, ids, mask, step_rng, jnp.asarray(step_no, jnp.int32))
            block_value(metrics)
            compile_seconds = time.perf_counter() - t0
            hbm = collect_hbm_stats()
            peak_hbm = update_peak_hbm(peak_hbm, hbm)
            records.append(benchmark_profile_record(
                run_index, variant_label, "fast_forward_compile", step_no,
                "fast_forward_loss", "fast_forward", compile_seconds,
                batch_size, seq_len, hbm, hbm_before=hbm_before,
                note="compile plus first forward-only execution"))
            _log(
                "[forward] compile "
                f"{compile_seconds:.3f}s {hbm_inline(hbm)}")

            for profile_i in range(int(args.forward_profile_steps)):
                batch, ids, mask, step_rng, step_no = next_profile_batch()
                hbm_before = collect_hbm_stats()
                t0 = time.perf_counter()
                metrics = forward_step(
                    params, ids, mask, step_rng,
                    jnp.asarray(step_no, jnp.int32))
                block_value(metrics)
                seconds = time.perf_counter() - t0
                hbm = collect_hbm_stats()
                peak_hbm = update_peak_hbm(peak_hbm, hbm)
                tok_s = float(batch_size) * float(seq_len) / seconds
                fast_times.append(seconds)
                fast_tokens.append(tok_s)
                metrics_host = jax.device_get(metrics)
                fast_metric_snapshots.append(metrics_host)
                records.append(benchmark_profile_record(
                    run_index, variant_label, "fast_forward_measure", step_no,
                    "fast_forward_loss", "fast_forward", seconds,
                    batch_size, seq_len, hbm, hbm_before=hbm_before,
                    note="forward plus loss; no backward or optimizer"))
                _status(
                    "[forward] "
                    f"{profile_i + 1}/{int(args.forward_profile_steps)} "
                    f"step_s={seconds:.4f} tok/s={tok_s:.1f} "
                    f"loss={fmt(metrics_host.get('loss'), 4)} | "
                    f"{hbm_inline(hbm)}",
                    persist=True)

            summary.update({
                "fast_forward_compile_seconds": compile_seconds,
                "fast_forward_steps": int(args.forward_profile_steps),
                "fast_forward_mean_seconds": mean(fast_times),
                "fast_forward_tokens_per_second": mean(fast_tokens),
                "fast_forward_speedup_vs_train_step": ratio_float(
                    train_mean_seconds, mean(fast_times)),
                "runtime_metric_summary": summarize_runtime_metric_snapshots(
                    fast_metric_snapshots),
            })
        except Exception as exc:
            _finish_status_line()
            error_text = log_profile_error("fast_forward", exc)
            records.append(benchmark_profile_error_record(
                run_index, variant_label, "fast_forward", exc))
            summary.update({
                "fast_forward_error": error_text,
                "fast_forward_steps": 0,
            })

    if fast_detailed_profile:
        try:
            detailed_fns = create_detailed_profile_fns(
                cfg, sharded_fns,
                include_backward=bool(args.backward_profile))
            detailed_fns["profile_layer_limit"] = int(
                args.detailed_profile_layers)
            profile_kind = (
                "forward/backward"
                if args.backward_profile else "forward")
            _log(f"Compiling detailed split {profile_kind} profile...")
            _batch, ids, _mask, _step_rng, step_no = next_profile_batch()
            (_compile_records, detailed_compile_seconds) = (
                run_detailed_profile_pass(
                    detailed_fns, params, ids, step_no,
                    run_index, variant_label, batch_size, seq_len,
                    "detailed_compile", record=False))
            _log(
                "[profile] detailed compile "
                f"{detailed_compile_seconds:.3f}s")

            _batch, ids, _mask, _step_rng, step_no = next_profile_batch()
            step_records, detailed_total_seconds = run_detailed_profile_pass(
                detailed_fns, params, ids, step_no,
                run_index, variant_label, batch_size, seq_len,
                "detailed_measure", record=True)
            records.extend(step_records)
            for record in step_records:
                peak_hbm = update_peak_hbm(
                    peak_hbm, {"hbm_peak_gb": record.get("hbm_peak_gb")})
            tok_s = (
                float(batch_size) * float(seq_len) / detailed_total_seconds
                if detailed_total_seconds else None)
            _status(
                "[profile] detailed "
                f"split_total_s={detailed_total_seconds:.4f}",
                persist=True)
            summary.update({
                "fast_detailed_profile": True,
                "detailed_profile_compile_seconds": (
                    detailed_compile_seconds),
                "detailed_profile_steps": 1,
                "detailed_profile_mean_split_seconds": (
                    detailed_total_seconds),
                "detailed_profile_tokens_per_second": tok_s,
            })
        except Exception as exc:
            _finish_status_line()
            error_text = log_profile_error("detailed_profile", exc)
            records.append(benchmark_profile_error_record(
                run_index, variant_label, "detailed_profile", exc))
            summary.update({
                "fast_detailed_profile": True,
                "detailed_profile_error": error_text,
                "detailed_profile_steps": 0,
            })

    if int(args.module_profile_steps) > 0:
        try:
            profile_fns = create_module_profile_fns(cfg, sharded_fns)
            _log("Compiling split-module forward profile...")
            batch, ids, _mask, step_rng, step_no = next_profile_batch()
            t0 = time.perf_counter()
            run_module_profile_pass(
                profile_fns, params, ids, step_rng, step_no,
                run_index, variant_label, batch_size, seq_len,
                "module_compile", record=False)
            module_compile_seconds = time.perf_counter() - t0
            _log(f"[profile] module compile {module_compile_seconds:.3f}s")
            module_total_times = []
            for profile_i in range(int(args.module_profile_steps)):
                batch, ids, _mask, step_rng, step_no = next_profile_batch()
                t0 = time.perf_counter()
                step_records = run_module_profile_pass(
                    profile_fns, params, ids, step_rng, step_no,
                    run_index, variant_label, batch_size, seq_len,
                    "module_measure", record=True)
                total_seconds = time.perf_counter() - t0
                records.extend(step_records)
                module_total_times.append(total_seconds)
                for record in step_records:
                    peak_hbm = update_peak_hbm(
                        peak_hbm, {"hbm_peak_gb": record.get("hbm_peak_gb")})
                _status(
                    "[profile] module "
                    f"{profile_i + 1}/{int(args.module_profile_steps)} "
                    f"split_total_s={total_seconds:.4f}",
                    persist=True)
            summary.update({
                "module_profile_compile_seconds": module_compile_seconds,
                "module_profile_steps": int(args.module_profile_steps),
                "module_profile_mean_split_seconds": mean(module_total_times),
                "module_profile_tokens_per_second": (
                    float(batch_size) * float(seq_len) / mean(
                        module_total_times)
                    if module_total_times and mean(module_total_times)
                    else None),
            })
        except Exception as exc:
            _finish_status_line()
            error_text = log_profile_error("module_profile", exc)
            records.append(benchmark_profile_error_record(
                run_index, variant_label, "module_profile", exc))
            summary.update({
                "module_profile_error": error_text,
                "module_profile_steps": 0,
            })

    aggregates, layer_rows = summarize_profile_records(records)
    (detailed_aggregates, detailed_layer_rows,
     detailed_derived) = summarize_detailed_profile_records(records)
    summary.update({
        "profile_records": records,
        "profile_aggregates": aggregates,
        "profile_layer_rows": layer_rows,
        "detailed_profile_aggregates": detailed_aggregates,
        "detailed_profile_layer_rows": detailed_layer_rows,
        "detailed_profile_derived": detailed_derived,
        "profile_peak_hbm_gb": peak_hbm,
    })
    return summary, records, iterator, rng, next_step


def json_float(value, default=0.0):
    value = num(value)
    if value is None:
        return default
    return float(value)


def runtime_pool_label(pool):
    return RUNTIME_POOL_LABELS.get(pool, str(pool))


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
    for pool in RUNTIME_POOLS:
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
    label = runtime_pool_label(pool)
    return (
        f"{label}[ovf={fmt(metrics.get(prefix + 'overflow_frac'), 3)} "
        f"util99={fmt(metrics.get(prefix + 'bucket_capacity_util_p99'), 3)} "
        f"short={fmt(metrics.get(prefix + 'capacity_shortfall'), 2)} "
        f"pad={fmt(metrics.get(prefix + 'padded_work_frac_vs_dense'), 3)}]"
    )


def opspace_brief(metrics, pool):
    prefix = f"opspace/{pool}/"
    if prefix + "enabled" not in metrics:
        return ""
    label = runtime_pool_label(pool)
    final_prefix = prefix + "final/"
    final_text = ""
    if final_prefix + "semantic_drop_frac" in metrics:
        skipped = metrics.get(
            final_prefix + "pallas_actual_skipped_chunks",
            metrics.get(final_prefix + "pallas_skipped_chunks"))
        launched = metrics.get(
            final_prefix + "pallas_actual_launched_chunks",
            metrics.get(final_prefix + "pallas_executed_chunks"))
        final_text = (
            f" loss95={fmt(metrics.get(final_prefix + 'assignment_score_loss_p95'), 4)}"
            f" drop={fmt(metrics.get(final_prefix + 'semantic_drop_frac'), 6)}"
            f" launch={fmt(launched, 1)}"
            f" skip={fmt(skipped, 1)}"
            f" pad={fmt(metrics.get(final_prefix + 'pallas_padding_frac'), 3)}")
    return (
        f"op-{label}[ovf={fmt(metrics.get(prefix + 'overflow_frac'), 6)} "
        f"reroute={fmt(metrics.get(prefix + 'reroute_frac'), 4)}"
        f"{final_text}]"
    )


def benchmark_step_record(run_index, variant_label, step, phase, phase_step,
                          step_seconds, tokens_per_second, metrics, hbm):
    record = {
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
    for pool in RUNTIME_POOLS:
        for name in OPSPACE_RUNTIME_METRIC_NAMES:
            key = f"opspace/{pool}/{name}"
            if key in metrics:
                record[key.replace("/", "_")] = json_float(metrics[key])
        for name in OPSPACE_FINAL_RUNTIME_METRIC_NAMES:
            key = f"opspace/{pool}/final/{name}"
            if key in metrics:
                record[key.replace("/", "_")] = json_float(metrics[key])
    return record


def format_step_status(phase, phase_step, phase_total, step_seconds,
                       tokens_per_second, metrics, hbm):
    sector_parts = [
        part for part in
        (sector_brief(metrics, pool) for pool in RUNTIME_POOLS) if part
    ]
    sector_text = " | " + " ".join(sector_parts) if sector_parts else ""
    opspace_parts = [
        part for part in
        (opspace_brief(metrics, pool) for pool in RUNTIME_POOLS) if part
    ]
    opspace_text = " | " + " ".join(opspace_parts) if opspace_parts else ""
    return (
        f"[bench] {phase} {int(phase_step)}/{int(phase_total)} "
        f"step_s={step_seconds:.4f} tok/s={tokens_per_second:.1f} "
        f"loss={fmt(metrics.get('loss'), 4)} "
        f"grad={fmt(metrics.get('grad_norm'), 3)} | "
        f"{hbm_inline(hbm)}{sector_text}{opspace_text}"
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
    for pool in RUNTIME_POOLS:
        prefix = f"sector/{pool}/"
        if prefix + "overflow_frac" not in metrics:
            continue
        print(
            "[sector-validity] "
            f"{runtime_pool_label(pool)} "
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
            f"{runtime_pool_label(pool)} "
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
    print(
        "throughput "
        f"mean_tokens_per_second="
        f"{fmt(summary.get('mean_tokens_per_second'), 1)}",
        flush=True)
    print(
        "memory "
        f"train_peak_hbm_gb={fmt(summary.get('train_peak_hbm_gb'))} "
        f"profile_peak_hbm_gb={fmt(summary.get('profile_peak_hbm_gb'))} "
        f"peak_hbm_gb={fmt(summary.get('peak_hbm_gb'))} "
        f"limit_hbm_gb={fmt(summary.get('hbm_limit_gb'))}",
        flush=True)
    if summary.get("fast_forward_mean_seconds") is not None:
        print(
            "fast_forward "
            f"steps={int(summary.get('fast_forward_steps') or 0)} "
            f"compile_s={fmt(summary.get('fast_forward_compile_seconds'), 3)} "
            f"mean_s={fmt(summary.get('fast_forward_mean_seconds'), 4)} "
            f"tok/s={fmt(summary.get('fast_forward_tokens_per_second'), 1)} "
            f"speedup_vs_train_step="
            f"{fmt(summary.get('fast_forward_speedup_vs_train_step'), 3)}",
            flush=True)
    if summary.get("fast_forward_error"):
        print(
            "fast_forward_error "
            f"error={summary.get('fast_forward_error')}",
            flush=True)
    if summary.get("module_profile_mean_split_seconds") is not None:
        print(
            "module_profile "
            f"steps={int(summary.get('module_profile_steps') or 0)} "
            f"compile_s={fmt(summary.get('module_profile_compile_seconds'), 3)} "
            f"split_mean_s="
            f"{fmt(summary.get('module_profile_mean_split_seconds'), 4)} "
            f"tok/s={fmt(summary.get('module_profile_tokens_per_second'), 1)} "
            "note=diagnostic_split_launches_excluding_compile",
            flush=True)
    if summary.get("module_profile_error"):
        print(
            "module_profile_error "
            f"error={summary.get('module_profile_error')}",
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
                f"pool={runtime_pool_label(pool)} "
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
    print_runtime_bottleneck(summary)
    print_profile_bottleneck(summary)
    print("\n[op-breakdown]", flush=True)
    print(
        "op | calls | total_ms | mean_ms | pct_split | "
        "hbm_peak_gb | hbm_delta_gb | note",
        flush=True)
    train_ms = (
        num(summary.get("mean_step_seconds")) * 1000.0
        if summary.get("mean_step_seconds") is not None else None)
    train_note = (
        "forward+backward+optimizer"
        if summary.get("train_benchmark_enabled", True)
        else "skipped (--fast)")
    print(
        "full_train_step | "
        f"{int(summary.get('measure_steps') or 0)} | "
        f"{fmt(train_ms, 3)} | {fmt(train_ms, 3)} | n/a | "
        f"{fmt(summary.get('train_peak_hbm_gb'))} | n/a | "
        f"{train_note}",
        flush=True)
    note_by_group = {
        "fast_forward": "forward+loss only, no backward/optimizer",
        "setup": "shared op-key materialization",
        "embedding": "token+position embedding",
        "attn": "sum over layer attention modules",
        "rst": "sum over layer RST modules",
        "loss": "final norm plus chunked CE",
    }
    for row in summary.get("profile_aggregates", []) or []:
        total_ms = num(row.get("total_seconds"))
        mean_ms = num(row.get("mean_seconds"))
        total_ms = total_ms * 1000.0 if total_ms is not None else None
        mean_ms = mean_ms * 1000.0 if mean_ms is not None else None
        print(
            f"{row.get('group', 'unknown')} | "
            f"{int(row.get('calls') or 0)} | "
            f"{fmt(total_ms, 3)} | "
            f"{fmt(mean_ms, 3)} | "
            f"{fmt(row.get('pct_module_split'), 1)} | "
            f"{fmt(row.get('max_hbm_peak_gb'))} | "
            f"{fmt(row.get('max_hbm_used_delta_gb'))} | "
            f"{note_by_group.get(row.get('group'), 'profile record')}",
            flush=True)
    layer_rows = summary.get("profile_layer_rows", []) or []
    if layer_rows:
        print("\n[op-breakdown/layers]", flush=True)
        print(
            "layer | attn_ms | rst_ms | rst/attn | "
            "attn_hbm_peak_gb | rst_hbm_peak_gb",
            flush=True)
        for row in layer_rows:
            attn_ms = (
                num(row.get("attn_seconds")) * 1000.0
                if row.get("attn_seconds") is not None else None)
            rst_ms = (
                num(row.get("rst_seconds")) * 1000.0
                if row.get("rst_seconds") is not None else None)
            print(
                f"{int(row.get('layer')):02d} | "
                f"{fmt(attn_ms, 3)} | "
                f"{fmt(rst_ms, 3)} | "
                f"{fmt(row.get('rst_over_attn'), 3)} | "
                f"{fmt(row.get('attn_hbm_peak_gb'))} | "
                f"{fmt(row.get('rst_hbm_peak_gb'))}",
                flush=True)
    print_detailed_profile_summary(summary)


def print_runtime_bottleneck(summary):
    runtime = summary.get("runtime_metric_summary") or {}
    if not runtime:
        return
    print("\n[runtime-bottleneck]", flush=True)
    print(
        "pool | overflow_max | semantic_drop_max | all_processed_min | "
        "pallas_launch_max | pallas_skip_max | pallas_pad_max | "
        "output_overflow_max | output_pad_max | backend_active",
        flush=True)
    for pool in RUNTIME_POOLS:
        if pool not in runtime:
            continue
        print(
            f"{runtime_pool_label(pool)} | "
            f"{fmt(runtime_stat(summary, pool, 'overflow_frac'))} | "
            f"{fmt(runtime_stat(summary, pool, 'semantic_drop_frac'))} | "
            f"{fmt(runtime_stat(summary, pool, 'all_processed', 'min'))} | "
            f"{fmt(runtime_stat(summary, pool, 'pallas_actual_launched_chunks'))} | "
            f"{fmt(runtime_stat(summary, pool, 'pallas_actual_skipped_chunks'))} | "
            f"{fmt(runtime_stat(summary, pool, 'pallas_padding_frac'))} | "
            f"{fmt(runtime_stat(summary, pool, 'output_tile_overflow_frac'))} | "
            f"{fmt(runtime_stat(summary, pool, 'output_tile_padding_frac'))} | "
            f"{fmt(runtime_stat(summary, pool, 'output_tiled_backend_active'))}",
            flush=True)


def print_profile_bottleneck(summary):
    rows = profile_bottleneck_rows(summary)
    if not rows:
        return
    print("\n[bottleneck]", flush=True)
    print(
        "rank | module | total_ms | mean_ms | pct_split | hbm_peak_gb",
        flush=True)
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank} | "
            f"{row.get('group')} | "
            f"{fmt_ms(row.get('total_seconds'))} | "
            f"{fmt_ms(row.get('mean_seconds'))} | "
            f"{fmt(row.get('pct_module_split'), 1)} | "
            f"{fmt(row.get('max_hbm_peak_gb'))}",
            flush=True)
    top_attn, top_rst, top_ratio = top_layer_bottlenecks(summary)
    if top_attn or top_rst or top_ratio:
        print(
            "[bottleneck-layer] "
            f"top_attn_layer={layer_label(top_attn)} "
            f"top_attn_ms={fmt_ms((top_attn or {}).get('attn_seconds'))} "
            f"top_rst_layer={layer_label(top_rst)} "
            f"top_rst_ms={fmt_ms((top_rst or {}).get('rst_seconds'))} "
            f"max_rst_over_attn_layer={layer_label(top_ratio)} "
            f"max_rst_over_attn={fmt((top_ratio or {}).get('rst_over_attn'))}",
            flush=True)


def print_detailed_profile_summary(summary):
    if not summary.get("fast_detailed_profile"):
        return
    print("\n=== SRW FAST BOTTLENECK SUMMARY ===", flush=True)
    print(f"config={summary.get('config_path', '')}", flush=True)
    print(
        f"data_source={summary.get('data_source', 'real_config_data')}",
        flush=True)
    print("train_step=skipped", flush=True)
    print(
        f"forward_s={fmt(summary.get('fast_forward_mean_seconds'), 4)}",
        flush=True)
    print(
        "detailed_split_s="
        f"{fmt(summary.get('detailed_profile_mean_split_seconds'), 4)}",
        flush=True)
    if summary.get("detailed_profile_error"):
        print(
            "detailed_profile_error "
            f"error={summary.get('detailed_profile_error')}",
            flush=True)
        return
    rows = detailed_bottleneck_rows(summary)
    if rows:
        print("\n[detailed-bottleneck]", flush=True)
        print("rank | group | total_ms | pct_split | hbm_peak_gb",
              flush=True)
        for rank, row in enumerate(rows, 1):
            print(
                f"{rank} | "
                f"{row.get('group')} | "
                f"{fmt_ms(row.get('total_seconds'))} | "
                f"{fmt(row.get('pct_detailed_split'), 1)} | "
                f"{fmt(row.get('max_hbm_peak_gb'))}",
                flush=True)
    print_detailed_copy_lines(int(summary.get("run_index") or 1), summary)


def print_detailed_copy_lines(run_index, summary):
    rows = {
        row.get("group"): row
        for row in summary.get("detailed_profile_aggregates", []) or []
    }
    if not rows:
        return

    def group_ms(group):
        return fmt_ms((rows.get(group) or {}).get("total_seconds"))

    module_groups = (
        "qk_srw",
        "v_srw",
        "qkv_bundle_srw",
        "attn_core",
        "attn_out_proj",
        "rst_srw",
        "rst_bundle_srw",
        "attn_route",
        "rst_route",
        "loss",
    )
    print(
        f"run={run_index} detailed_module_summary "
        + " ".join(f"{group}_ms={group_ms(group)}"
                   for group in module_groups),
        flush=True)

    rank_parts = []
    for rank, row in enumerate(detailed_bottleneck_rows(summary), 1):
        rank_parts.append(
            f"{rank}:{row.get('group')}"
            f"_ms={fmt_ms(row.get('total_seconds'))}"
            f"_pct={fmt(row.get('pct_detailed_split'), 1)}"
            f"_hbm={fmt(row.get('max_hbm_peak_gb'))}")
    if rank_parts:
        print(
            f"run={run_index} detailed_bottleneck_rank "
            + " ".join(rank_parts),
            flush=True)

    layer_rows = summary.get("detailed_profile_layer_rows", []) or []
    if layer_rows:
        def layer_line(label, key, formatter):
            parts = []
            for row in layer_rows:
                layer = int(row.get("layer"))
                parts.append(f"{layer:02d}:{formatter(row.get(key))}")
            print(f"run={run_index} {label} " + ",".join(parts),
                  flush=True)

        layer_line("layer_qk_srw_ms", "qk_srw_seconds", fmt_ms)
        layer_line("layer_v_srw_ms", "v_srw_seconds", fmt_ms)
        layer_line(
            "layer_qkv_bundle_srw_ms", "qkv_bundle_srw_seconds", fmt_ms)
        layer_line("layer_attn_core_ms", "attn_core_seconds", fmt_ms)
        layer_line("layer_rst_srw_ms", "rst_srw_seconds", fmt_ms)
        layer_line("layer_rst_over_qkv", "rst_over_qkv",
                   lambda value: fmt(value, 3))

    derived = summary.get("detailed_profile_derived") or {}
    if derived:
        print(
            f"run={run_index} derived_summary "
            f"attn_srw_ms={fmt_ms(derived.get('attn_srw_seconds'))} "
            f"attn_non_srw_ms="
            f"{fmt_ms(derived.get('attn_non_srw_seconds'))} "
            f"rst_total_ms={fmt_ms(derived.get('rst_total_seconds'))} "
            f"rst_srw_over_qkv="
            f"{fmt(derived.get('rst_srw_over_qkv'), 3)} "
            f"v_over_qk={fmt(derived.get('v_over_qk'), 3)}",
            flush=True)


def layer_label(row):
    if not row:
        return "n/a"
    return f"{int(row.get('layer')):02d}"


def print_comparison(summaries):
    _finish_status_line()
    print("\n" + "=" * 72, flush=True)
    print("SRW Benchmark Comparison", flush=True)
    print("=" * 72, flush=True)
    header = (
        f"{'#':>2}  {'model':24s}  {'train_s':>9s}  "
        f"{'fast_s':>9s}  {'fast_x':>7s}  {'split_s':>9s}  "
        f"{'tok/s':>12s}  {'peak_hbm':>10s}  {'valid':>5s}  "
        f"{'variant':28s}  config")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, summary in enumerate(summaries, 1):
        display_tokens = (
            summary.get("mean_tokens_per_second")
            if summary.get("mean_tokens_per_second") is not None
            else summary.get("fast_forward_tokens_per_second"))
        print(
            f"{i:>2}  {summary.get('model_version', ''):24s}  "
            f"{fmt(summary.get('mean_step_seconds'), 4):>9s}  "
            f"{fmt(summary.get('fast_forward_mean_seconds'), 4):>9s}  "
            f"{fmt(summary.get('fast_forward_speedup_vs_train_step'), 2):>7s}  "
            f"{fmt(summary.get('module_profile_mean_split_seconds'), 4):>9s}  "
            f"{fmt(display_tokens, 1):>12s}  "
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
        fast_time_ratio = ratio(
            summary.get('fast_forward_mean_seconds'),
            base.get('fast_forward_mean_seconds'))
        fast_speed_ratio = ratio(
            base.get('fast_forward_mean_seconds'),
            summary.get('fast_forward_mean_seconds'))
        split_time_ratio = ratio(
            summary.get('module_profile_mean_split_seconds'),
            base.get('module_profile_mean_split_seconds'))
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
            f"fast_time={fast_time_ratio} "
            f"fast_speed={fast_speed_ratio} "
            f"split_time={split_time_ratio} "
            f"peak_hbm="
            f"{ratio(summary.get('peak_hbm_gb'), base.get('peak_hbm_gb'))}",
            flush=True)
    if len(summaries) >= 2:
        print("\n[compare]", flush=True)
        base_step = base.get("mean_step_seconds")
        base_median = base.get("median_step_seconds")
        base_fast = base.get("fast_forward_mean_seconds")
        base_split = base.get("module_profile_mean_split_seconds")
        for i, summary in enumerate(summaries[1:], 2):
            valid_pair = (
                bool(base.get("benchmark_valid", True))
                and bool(summary.get("benchmark_valid", True)))
            speed_ratio = (
                ratio(base_step, summary.get("mean_step_seconds"))
                if valid_pair else "invalid_overflow")
            fast_speed_ratio = ratio(
                base_fast, summary.get("fast_forward_mean_seconds"))
            split_speed_ratio = ratio(
                base_split, summary.get("module_profile_mean_split_seconds"))
            print(
                f"base_mean_step_s={fmt(base_step, 4)} "
                f"run_{i}_mean_step_s={fmt(summary.get('mean_step_seconds'), 4)} "
                f"speed_ratio={speed_ratio} "
                f"base_fast_s={fmt(base_fast, 4)} "
                f"run_{i}_fast_s="
                f"{fmt(summary.get('fast_forward_mean_seconds'), 4)} "
                f"fast_speed_ratio={fast_speed_ratio} "
                f"base_split_s={fmt(base_split, 4)} "
                f"run_{i}_split_s="
                f"{fmt(summary.get('module_profile_mean_split_seconds'), 4)} "
                f"split_speed_ratio={split_speed_ratio} "
                f"raw_median_ratio="
                f"{ratio(summary.get('median_step_seconds'), base_median)} "
                f"valid={str(summary.get('benchmark_valid', True)).lower()}",
                flush=True)


def copy_safe(value):
    if value is None or value == "":
        return "none"
    return "_".join(str(value).replace("\\", "/").split())


def print_copy_summary(summaries, args):
    _finish_status_line()
    mode = "fast" if args.fast_only else "train+profile"
    train_state = "skipped" if args.fast_only else "enabled"
    measure_steps = 0 if args.fast_only else int(args.steps)
    warmup_steps = 0 if args.fast_only else int(args.warmup_steps)
    print("\n=== SRW BENCHMARK COPY SUMMARY BEGIN ===", flush=True)
    print(
        "run_config "
        f"mode={mode} "
        f"train_step={train_state} "
        f"configs={len(summaries)} "
        f"warmup_steps={warmup_steps} "
        f"measure_steps={measure_steps} "
        f"forward_profile_steps={int(args.forward_profile_steps)} "
        f"detailed_profile_steps={1 if args.fast_only else 0} "
        f"module_profile_steps={int(args.module_profile_steps)} "
        f"metrics_jsonl={args.metrics_jsonl or 'disabled'}",
        flush=True)
    print(
        "columns "
        "run model valid train_s fast_s fast_tok_s split_s "
        "peak_hbm_gb profile_peak_hbm_gb fast_error module_error "
        "config variant",
        flush=True)
    for i, summary in enumerate(summaries, 1):
        variant = summary.get("variant") or "none"
        print(
            f"run={i} "
            f"model={summary.get('model_version', 'unknown')} "
            f"valid={str(summary.get('benchmark_valid', True)).lower()} "
            f"reason={summary.get('invalid_reason') or 'none'} "
            f"train_s={fmt(summary.get('mean_step_seconds'), 4)} "
            f"fast_s={fmt(summary.get('fast_forward_mean_seconds'), 4)} "
            f"fast_tok_s={fmt(summary.get('fast_forward_tokens_per_second'), 1)} "
            f"split_s={fmt(summary.get('module_profile_mean_split_seconds'), 4)} "
            f"peak_hbm_gb={fmt(summary.get('peak_hbm_gb'))} "
            f"profile_peak_hbm_gb={fmt(summary.get('profile_peak_hbm_gb'))} "
            f"fast_error={copy_safe(summary.get('fast_forward_error'))} "
            f"module_error={copy_safe(summary.get('module_profile_error'))} "
            f"config={summary.get('config_path', '')} "
            f"variant={variant}",
            flush=True)
        print_copy_bottleneck_summary(i, summary)
        print_copy_runtime_summary(i, summary)
        print_copy_module_summary(i, summary)
        print_copy_layer_summary(i, summary)
        print_detailed_copy_lines(i, summary)

    if len(summaries) >= 2:
        base = summaries[0]
        base_step = base.get("mean_step_seconds")
        base_fast = base.get("fast_forward_mean_seconds")
        base_split = base.get("module_profile_mean_split_seconds")
        base_fast_tok = base.get("fast_forward_tokens_per_second")
        print(
            "compare_note "
            "base=run_1 "
            "speed_ratio_above_1_means_candidate_is_faster",
            flush=True)
        for i, summary in enumerate(summaries[1:], 2):
            valid_pair = (
                bool(base.get("benchmark_valid", True))
                and bool(summary.get("benchmark_valid", True)))
            train_speed = (
                ratio(base_step, summary.get("mean_step_seconds"))
                if valid_pair else "invalid_overflow")
            print(
                f"compare base=1 run={i} "
                f"valid_pair={str(valid_pair).lower()} "
                f"train_speed_vs_base={train_speed} "
                f"train_time_vs_base="
                f"{ratio(summary.get('mean_step_seconds'), base_step)} "
                f"fast_speed_vs_base="
                f"{ratio(base_fast, summary.get('fast_forward_mean_seconds'))} "
                f"fast_time_vs_base="
                f"{ratio(summary.get('fast_forward_mean_seconds'), base_fast)} "
                f"fast_tok_vs_base="
                f"{ratio(summary.get('fast_forward_tokens_per_second'), base_fast_tok)} "
                f"split_speed_vs_base="
                f"{ratio(base_split, summary.get('module_profile_mean_split_seconds'))} "
                f"split_time_vs_base="
                f"{ratio(summary.get('module_profile_mean_split_seconds'), base_split)} "
                f"peak_hbm_vs_base="
                f"{ratio(summary.get('peak_hbm_gb'), base.get('peak_hbm_gb'))}",
                flush=True)
    print("=== SRW BENCHMARK COPY SUMMARY END ===", flush=True)


def print_copy_module_summary(run_index, summary):
    rows = {
        row.get("group"): row
        for row in summary.get("profile_aggregates", []) or []
    }
    if not rows:
        return
    parts = []
    for group in ("fast_forward", "setup", "embedding", "attn", "rst", "loss"):
        row = rows.get(group)
        if not row:
            continue
        parts.extend([
            f"{group}_calls={int(row.get('calls') or 0)}",
            f"{group}_total_ms={fmt_ms(row.get('total_seconds'))}",
            f"{group}_mean_ms={fmt_ms(row.get('mean_seconds'))}",
            f"{group}_pct_split={fmt(row.get('pct_module_split'), 1)}",
            f"{group}_hbm_peak_gb={fmt(row.get('max_hbm_peak_gb'))}",
        ])
    if parts:
        print(f"run={run_index} module_summary " + " ".join(parts),
              flush=True)


def print_copy_bottleneck_summary(run_index, summary):
    rows = profile_bottleneck_rows(summary)
    if rows:
        parts = []
        for rank, row in enumerate(rows, 1):
            parts.append(
                f"{rank}:{row.get('group')}"
                f"_total_ms={fmt_ms(row.get('total_seconds'))}"
                f"_pct={fmt(row.get('pct_module_split'), 1)}"
                f"_hbm={fmt(row.get('max_hbm_peak_gb'))}")
        print(f"run={run_index} bottleneck_rank " + " ".join(parts),
              flush=True)
    elif summary.get("fast_forward_error") or summary.get(
            "module_profile_error"):
        print(
            f"run={run_index} bottleneck_rank unavailable "
            "reason=profile_error "
            f"fast_error={copy_safe(summary.get('fast_forward_error'))} "
            f"module_error={copy_safe(summary.get('module_profile_error'))}",
            flush=True)
    top_attn, top_rst, top_ratio = top_layer_bottlenecks(summary)
    if top_attn or top_rst or top_ratio:
        print(
            f"run={run_index} layer_bottleneck "
            f"top_attn={layer_label(top_attn)}:"
            f"{fmt_ms((top_attn or {}).get('attn_seconds'))}ms "
            f"top_rst={layer_label(top_rst)}:"
            f"{fmt_ms((top_rst or {}).get('rst_seconds'))}ms "
            f"max_rst_over_attn={layer_label(top_ratio)}:"
            f"{fmt((top_ratio or {}).get('rst_over_attn'), 3)}",
            flush=True)


def print_copy_runtime_summary(run_index, summary):
    runtime = summary.get("runtime_metric_summary") or {}
    if not runtime:
        return
    for pool in RUNTIME_POOLS:
        if pool not in runtime:
            continue
        print(
            f"run={run_index} runtime_summary "
            f"pool={runtime_pool_label(pool)} "
            f"overflow_max={fmt(runtime_stat(summary, pool, 'overflow_frac'))} "
            f"semantic_drop_max="
            f"{fmt(runtime_stat(summary, pool, 'semantic_drop_frac'))} "
            f"all_processed_min="
            f"{fmt(runtime_stat(summary, pool, 'all_processed', 'min'))} "
            f"pallas_launch_max="
            f"{fmt(runtime_stat(summary, pool, 'pallas_actual_launched_chunks'))} "
            f"pallas_skip_max="
            f"{fmt(runtime_stat(summary, pool, 'pallas_actual_skipped_chunks'))} "
            f"pallas_pad_max="
            f"{fmt(runtime_stat(summary, pool, 'pallas_padding_frac'))} "
            f"output_overflow_max="
            f"{fmt(runtime_stat(summary, pool, 'output_tile_overflow_frac'))} "
            f"output_pad_max="
            f"{fmt(runtime_stat(summary, pool, 'output_tile_padding_frac'))} "
            f"backend_active="
            f"{fmt(runtime_stat(summary, pool, 'output_tiled_backend_active'))}",
            flush=True)


def print_copy_layer_summary(run_index, summary):
    layer_rows = summary.get("profile_layer_rows", []) or []
    if not layer_rows:
        return
    attn = []
    rst = []
    ratio_parts = []
    for row in layer_rows:
        layer = int(row.get("layer"))
        attn.append(f"{layer:02d}:{fmt_ms(row.get('attn_seconds'))}")
        rst.append(f"{layer:02d}:{fmt_ms(row.get('rst_seconds'))}")
        ratio_parts.append(f"{layer:02d}:{fmt(row.get('rst_over_attn'), 3)}")
    print(f"run={run_index} layer_attn_ms " + ",".join(attn), flush=True)
    print(f"run={run_index} layer_rst_ms " + ",".join(rst), flush=True)
    print(f"run={run_index} layer_rst_over_attn " + ",".join(ratio_parts),
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


def fmt_ms(seconds, digits=3):
    value = num(seconds)
    return fmt(value * 1000.0 if value is not None else None, digits)


def ratio(a, b):
    a = num(a)
    b = num(b)
    if a is None or b in (None, 0.0):
        return "n/a"
    return f"{a / b:.3f}x"


if __name__ == "__main__":
    main()
