"""Operator geometry and neuron-health analysis."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import AnalysisContext
from analysis.dawn_analysis_storage import should_skip_job, write_json_atomic, write_npz_atomic

from models import dawn_srw_v4166 as v4166


POOL_SPECS = {
    "qk": ("Attention-QK", "attn_qk_op_key", "attn_qk_read", "attn_qk_write"),
    "v": ("Attention-V", "attn_v_op_key", "attn_v_read", "attn_v_write"),
    "rst": ("RST", "rst_op_key", "rst_read", "rst_write"),
}


def _to_host_tree(tree):
    return jax.device_get(tree)


def _jsonify_stats(obj):
    if isinstance(obj, dict):
        return {k: _jsonify_stats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_stats(v) for v in obj]
    if hasattr(obj, "shape"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return float(arr)
        return arr.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _norm_arrays(params):
    pool = v4166._pool_params_with_operator_keys(params["neuron_pool"])
    out = {}
    for key, (_, op_key_key, read_key, write_key) in POOL_SPECS.items():
        out[f"{key}_op_key_norm"] = jnp.linalg.norm(pool[op_key_key], axis=-1)
        out[f"{key}_read_norm"] = jnp.linalg.norm(pool[read_key], axis=-1)
        out[f"{key}_write_norm"] = jnp.linalg.norm(pool[write_key], axis=-1)
    return out


def _sample_cosines(params, max_sample: int):
    pool = v4166._pool_params_with_operator_keys(params["neuron_pool"])
    out = {}
    for key, (_, op_key_key, _, _) in POOL_SPECS.items():
        x = pool[op_key_key].astype(jnp.float32)
        n = x.shape[0]
        if n > max_sample:
            idx = jnp.linspace(0, n - 1, max_sample, dtype=jnp.int32)
            x = x[idx]
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        sim = jnp.abs(x @ x.T)
        mask = 1.0 - jnp.eye(sim.shape[0], dtype=jnp.float32)
        vals = (sim * mask).reshape(-1)
        vals = vals[vals > 0]
        out[f"{key}_cosine_sample"] = vals
    return out


def _hist(arr: np.ndarray, bins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((bins,), dtype=np.int64), np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    return np.histogram(arr, bins=bins)


def _build_histograms(norms: Dict[str, np.ndarray], cosines: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    arrays = {}
    for name, arr in norms.items():
        h, e = _hist(arr)
        arrays[f"{name}_hist"] = h
        arrays[f"{name}_edges"] = e
    for name, arr in cosines.items():
        h, e = _hist(arr)
        arrays[f"{name}_hist"] = h
        arrays[f"{name}_edges"] = e
    return arrays


def run_geometry_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "geometry"
    summary_path = store.path("geometry", "operator_geometry_summary.json")
    if args.resume and should_skip_job(summary_path, ["pools"]):
        summary = store.load_manifest().get("stages", {}).get(stage, {}).get("summaries", {}).get("geometry")
        if not summary:
            summary = {}
        store.log_event(stage, "skip", message="GEOMETRY SKIP complete", **summary)
        return summary

    store.set_stage_status(stage, "running")
    store.mark_job_started(stage, "geometry")
    max_sample = int(args.geometry_max_sample)
    store.log_event(stage, "start", message=f"GEOMETRY START max_sample={max_sample}", max_sample=max_sample)

    health = _jsonify_stats(_to_host_tree(v4166.vectorized_neuron_health(ctx.params)))
    weights = _jsonify_stats(_to_host_tree(v4166.vectorized_weight_analysis(ctx.params, max_sample=max_sample)))
    norm_arrays_host = {
        k: np.asarray(v)
        for k, v in _to_host_tree(_norm_arrays(ctx.params)).items()
    }
    cosine_host = {
        k: np.asarray(v)
        for k, v in _to_host_tree(_sample_cosines(ctx.params, max_sample=max_sample)).items()
    }
    hist_arrays = _build_histograms(norm_arrays_host, cosine_host)

    pools = {}
    for short, (display, _, _, _) in POOL_SPECS.items():
        h = health.get(display, {})
        w = weights.get(display, {})
        pools[short] = {
            "display": display,
            "N": int(h.get("N", w.get("N", 0))),
            "op_key_norm_mean": float(h.get("op_key_mean", 0.0)),
            "op_key_norm_std": float(h.get("op_key_std", 0.0)),
            "op_key_dead": int(h.get("op_key_dead", 0)),
            "read_norm_mean": float(h.get("read_mean", 0.0)),
            "read_norm_std": float(h.get("read_std", 0.0)),
            "read_dead": int(h.get("read_dead", 0)),
            "write_norm_mean": float(h.get("write_mean", 0.0)),
            "write_norm_std": float(h.get("write_std", 0.0)),
            "write_dead": int(h.get("write_dead", 0)),
            "effective_rank": float(w.get("effective_rank", 0.0)),
            "mean_cosine_similarity": float(w.get("mean_cosine_sim", 0.0)),
            "max_cosine_similarity": float(w.get("max_cosine_sim", 0.0)),
            "top_singular_values": w.get("top5_sv", []),
        }

    summary = {
        "checkpoint_step": ctx.checkpoint_step,
        "max_sample": max_sample,
        "pools": pools,
    }
    if ctx.is_primary:
        write_json_atomic(store.path("geometry", "neuron_health.json"), health)
        write_json_atomic(store.path("geometry", "weight_analysis.json"), weights)
        write_npz_atomic(store.path("geometry", "operator_norm_histograms.npz"), **hist_arrays)
        write_json_atomic(summary_path, summary)
        store.mark_job_complete(stage, "geometry", summary_path, summary)
        store.set_stage_status(stage, "complete")
        for pool, rec in pools.items():
            store.log_event(
                stage,
                "pool_summary",
                message=(
                    f"GEOMETRY {pool} N={rec['N']:,} "
                    f"rank={rec['effective_rank']:.2f} "
                    f"mean_cos={rec['mean_cosine_similarity']:.5f} "
                    f"max_cos={rec['max_cosine_similarity']:.5f} "
                    f"dead(op/read/write)="
                    f"{rec['op_key_dead']}/{rec['read_dead']}/{rec['write_dead']}"
                ),
                **rec,
            )
        store.log_event(stage, "summary", message="GEOMETRY SUMMARY complete", **summary)
    return summary if ctx.is_primary else {}

