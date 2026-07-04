#!/usr/bin/env python3
"""Standalone TPU benchmark harness for DAWN-SRW v4166/v4168.

Runs model modules directly and reports benchmark details to stdout.
"""

import argparse
import importlib
import inspect
import os
from pathlib import Path
import random
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
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be >= 0")
    if args.model_version and args.model_version not in SUPPORTED_MODEL_VERSIONS:
        raise SystemExit(
            f"--model-version must be one of {SUPPORTED_MODEL_VERSIONS}")
    configs = [item for group in args.config for item in group]
    xla_dump_dir = _normalize_xla_dump_dir(args.xla_dump_dir, len(configs))
    if xla_dump_dir:
        _enable_xla_dump(xla_dump_dir)

    _maybe_initialize_jax_distributed()
    summaries = []
    for config_arg in configs:
        _log("\n" + "=" * 72)
        _log(
            f"SRW standalone benchmark "
            f"{len(summaries) + 1}/{len(configs)}: {config_arg}")
        _log("=" * 72)
        summary = run_one_config(config_arg, args, xla_dump_dir)
        summaries.append(summary)

    if len(summaries) > 1 and _is_host0():
        print_comparison(summaries)


def _is_host0():
    return jax.process_index() == 0


def _log(message):
    if _is_host0():
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


def tree_map_with_path(fn, tree):
    mapper = getattr(jax.tree, "map_with_path", None)
    if mapper is not None:
        return mapper(fn, tree)
    return jax.tree_util.tree_map_with_path(fn, tree)


def run_one_config(config_arg, args, xla_dump_dir):
    config_path = _resolve_config(config_arg)
    cfg = _load_yaml(config_path)
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
    if xla_dump_dir:
        _log(f"  xla_dump_dir: {xla_dump_dir}")

    model = build_model(cfg)
    sharded_fns = build_sharded_fns(cfg, mesh)
    train_loader = build_loader(
        cfg, batch_size, seq_len, n_hosts, host_id,
        total_steps=args.steps + args.warmup_steps + 2,
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
    peak_hbm = hbm_after.get("hbm_peak_gb")
    total_to_run = int(args.warmup_steps) + int(args.steps)
    for i in range(total_to_run):
        phase = "warmup" if i < int(args.warmup_steps) else "measure"
        step_no = i + 1
        batch, iterator = next_batch(iterator, train_loader)
        ids, mask = shard_batch_to_mesh(batch, data_sharding, batch_size, seq_len)
        rng, step_rng = jax.random.split(rng)
        t0 = time.perf_counter()
        params, opt_state, metrics = train_step(
            params, opt_state, ids, mask, step_rng,
            jnp.asarray(step_no, jnp.int32))
        block_until_ready(metrics)
        seconds = time.perf_counter() - t0
        hbm = collect_hbm_stats()
        if hbm.get("hbm_peak_gb") is not None:
            peak_hbm = (
                hbm["hbm_peak_gb"] if peak_hbm is None
                else max(peak_hbm, hbm["hbm_peak_gb"]))
        tokens_per_second = float(batch_size) * float(seq_len) / seconds
        metrics_host = jax.device_get(metrics)
        if phase == "warmup":
            warmup_times.append(seconds)
        else:
            measure_times.append(seconds)
            measure_tokens.append(tokens_per_second)
        _log(
            "[bench] "
            f"{phase} step={step_no if phase == 'warmup' else len(measure_times)} "
            f"step_s={seconds:.4f} "
            f"tok/s={tokens_per_second:.1f} "
            f"loss={fmt(metrics_host.get('loss'), 4)} "
            f"grad={fmt(metrics_host.get('grad_norm'), 3)} "
            f"HBM={fmt(hbm.get('hbm_used_gb'))}G "
            f"peak={fmt(hbm.get('hbm_peak_gb'))}G")
        if model_version == V4168_MODEL_VERSION and phase == "measure":
            print_sector_metrics(metrics_host)

    xla_report = collect_xla_memory_report(xla_dump_dir) if xla_dump_dir else {}
    summary = {
        "config_path": config_arg,
        "model_version": model_version,
        "compile_seconds": compile_seconds,
        "warmup_steps": int(args.warmup_steps),
        "measure_steps": int(args.steps),
        "mean_step_seconds": mean(measure_times),
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
    }
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
    soft_gate_t = float(t.get("soft_gate_temperature", 0.07))
    boundary_power = float(t.get("soft_gate_boundary_power_final",
                                 t.get("soft_gate_boundary_power_mid",
                                       t.get("soft_gate_boundary_power_start", 4.0))))
    admission_den_power = float(t.get("admission_den_power", 1.0))

    @jax.jit
    def train_step(params, opt_state, input_ids, attention_mask, rng, step):
        del step

        def loss_fn(p):
            result = model.apply(
                {"params": p},
                input_ids,
                labels=input_ids,
                attention_mask=attention_mask,
                deterministic=True,
                rngs={"dropout": rng},
                sharded_fns=sharded_fns,
                minimal_train=True,
                soft_gate_temperature=soft_gate_t,
                soft_gate_t_final=soft_gate_t,
                soft_gate_T_qk=soft_gate_t,
                soft_gate_T_v=soft_gate_t,
                soft_gate_T_rst=soft_gate_t,
                soft_gate_boundary_power=boundary_power,
                soft_gate_boundary_power_final=boundary_power,
                admission_den_power=admission_den_power,
                execution_prune_eps=0.0,
            )
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
        for key in (
                "sector/attn_v/sector_fill_mean",
                "sector/attn_v/sector_fill_max",
                "sector/attn_v/sector_overflow_count",
                "sector/attn_v/selected_sector_frac",
                "sector/attn_v/effective_operator_frac",
                "sector/rst/sector_fill_mean",
                "sector/rst/sector_fill_max",
                "sector/rst/sector_overflow_count",
                "sector/rst/selected_sector_frac",
                "sector/rst/effective_operator_frac"):
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
    keys = (
        "sector/attn_v/sector_fill_mean",
        "sector/attn_v/sector_fill_max",
        "sector/attn_v/sector_overflow_count",
        "sector/attn_v/selected_sector_frac",
        "sector/attn_v/effective_operator_frac",
        "sector/rst/sector_fill_mean",
        "sector/rst/sector_fill_max",
        "sector/rst/sector_overflow_count",
        "sector/rst/selected_sector_frac",
        "sector/rst/effective_operator_frac",
    )
    parts = [f"{k}={fmt(metrics[k], 4)}" for k in keys if k in metrics]
    if parts:
        print("[bench-sector] " + " ".join(parts), flush=True)


def print_summary(summary):
    if not _is_host0():
        return
    print("\n=== SRW Standalone Benchmark Summary ===", flush=True)
    print(f"config={summary['config_path']}", flush=True)
    print(f"model={summary['model_version']}", flush=True)
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
        f"peak_hbm_gb={fmt(summary.get('peak_hbm_gb'))} "
        f"limit_hbm_gb={fmt(summary.get('hbm_limit_gb'))}",
        flush=True)
    if summary.get("xla_total_hbm_usage"):
        print(
            "xla_memory "
            f"{summary.get('xla_total_hbm_usage')} "
            f"{summary.get('xla_program_hbm_requirement') or ''}",
            flush=True)


def print_comparison(summaries):
    print("\n" + "=" * 72, flush=True)
    print("SRW Benchmark Comparison", flush=True)
    print("=" * 72, flush=True)
    header = (
        f"{'#':>2}  {'model':24s}  {'mean_s':>10s}  "
        f"{'tok/s':>12s}  {'peak_hbm':>10s}  config")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for i, summary in enumerate(summaries, 1):
        print(
            f"{i:>2}  {summary.get('model_version', ''):24s}  "
            f"{fmt(summary.get('mean_step_seconds'), 4):>10s}  "
            f"{fmt(summary.get('mean_tokens_per_second'), 1):>12s}  "
            f"{fmt(summary.get('peak_hbm_gb')):>10s}  "
            f"{summary.get('config_path', '')}",
            flush=True)
    base = summaries[0]
    print("\nRatios vs first config:", flush=True)
    for i, summary in enumerate(summaries[1:], 2):
        print(
            f"  #{i}: step_time="
            f"{ratio(summary.get('mean_step_seconds'), base.get('mean_step_seconds'))} "
            f"tokens="
            f"{ratio(summary.get('mean_tokens_per_second'), base.get('mean_tokens_per_second'))} "
            f"peak_hbm="
            f"{ratio(summary.get('peak_hbm_gb'), base.get('peak_hbm_gb'))}",
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
