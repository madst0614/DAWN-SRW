"""Compact token/layer trace stage and reusable top-k forward."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import (
    AnalysisContext,
    format_duration,
    load_eval_data,
    maybe_load_tokenizer,
)
from analysis.dawn_analysis_storage import (
    read_jsonl,
    should_skip_job,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)

from models import dawn_srw_v4166 as v4166


TRACE_POOLS = ("q", "k", "v", "rst")


def _srw_with_topk(x, h, op_key, raw_tau, read, write, *,
                   topk: int,
                   execution_kwargs: Dict[str, Any],
                   admission_den_power: float):
    kwargs = dict(execution_kwargs)
    kwargs.pop("admission_den_power", None)
    _, admission, _, execution_weight, active_mask = v4166._angular_execution(
        h,
        op_key,
        raw_tau,
        None,
        **kwargs,
    )
    k = min(int(topk), int(execution_weight.shape[-1]))
    top_val, top_idx = jax.lax.top_k(execution_weight.astype(jnp.float32), k)
    mass = execution_weight.sum(axis=-1)
    active_count = active_mask.astype(jnp.int32).sum(axis=-1)
    top1_frac = top_val[..., 0] / jnp.maximum(mass, 1.0e-8)

    r_n = v4166._forward_unit_direction(read.astype(jnp.float32))
    w_n = v4166._forward_unit_direction(write.astype(jnp.float32))
    xr = x.astype(jnp.float32) @ r_n.T
    out = (execution_weight * xr) @ w_n
    den = jnp.power(
        jnp.maximum(admission.sum(axis=-1, keepdims=True), 1.0),
        jnp.asarray(admission_den_power, dtype=jnp.float32),
    )
    out = (out.astype(jnp.float32) / den).astype(jnp.float32)
    stats = {
        "top_idx": top_idx.astype(jnp.int32),
        "top_val": top_val.astype(jnp.float32),
        "active_count": active_count.astype(jnp.int32),
        "mass": mass.astype(jnp.float32),
        "top1_frac": top1_frac.astype(jnp.float32),
    }
    return out, stats


def topk_trace_forward(params, model_cfg: Dict[str, Any], input_ids, *,
                       topk: int = 8,
                       execution_prune_eps: Optional[float] = None):
    """Return compact top-k operator traces for a small fixed-shape batch."""
    params = v4166._squeeze_params(params)
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    bsz, seq_len = input_ids.shape
    d_model = int(model_cfg["d_model"])
    n_layers = int(model_cfg["n_layers"])
    n_heads = int(model_cfg["n_heads"])
    d_head = d_model // n_heads
    execution_kwargs = v4166._angular_execution_kwargs_from_model_cfg(model_cfg)
    if execution_prune_eps is not None:
        execution_kwargs["execution_prune_eps"] = float(execution_prune_eps)
    admission_den_power = float(execution_kwargs.get("admission_den_power", 1.0))

    pool = v4166._pool_params_with_operator_keys(params["neuron_pool"])
    router = params["router"]
    qk_scale, v_scale, rst_scale = v4166._effective_pool_output_scales(
        pool,
        d_model,
        n_layers,
    )
    positions = jnp.arange(seq_len)[jnp.newaxis, :]
    x = (
        params["token_emb"]["embedding"][input_ids]
        + params["pos_emb"]["embedding"][positions]
    )

    traces = {f"{pool_name}_{field}": [] for pool_name in TRACE_POOLS
              for field in ("top_idx", "top_val", "active_count", "mass", "top1_frac")}
    attn_out_norm = []
    rst_out_norm = []

    for layer_idx in range(n_layers):
        bp = params[f"block_{layer_idx}"]
        normed = v4166._layer_norm(x, bp["norm1"]["scale"], bp["norm1"]["bias"])
        h_all = normed @ router["proj_attn"]["kernel"] + router["proj_attn"]["bias"]
        h_q, h_k, h_v = jnp.split(h_all, 3, axis=-1)
        h_q, h_k, h_v = v4166._read_write_attn_operator_queries(
            router,
            normed,
            h_q,
            h_k,
            h_v,
        )
        tau_all = normed @ router["raw_tau_attn"]["kernel"] + router["raw_tau_attn"]["bias"]
        q, q_stats = _srw_with_topk(
            normed,
            h_q,
            pool["attn_qk_op_key"],
            tau_all[:, :, 0:1],
            pool["attn_qk_read"],
            pool["attn_qk_write"],
            topk=topk,
            execution_kwargs=execution_kwargs,
            admission_den_power=admission_den_power,
        )
        k, k_stats = _srw_with_topk(
            normed,
            h_k,
            pool["attn_qk_op_key"],
            tau_all[:, :, 1:2],
            pool["attn_qk_read"],
            pool["attn_qk_write"],
            topk=topk,
            execution_kwargs=execution_kwargs,
            admission_den_power=admission_den_power,
        )
        v, v_stats = _srw_with_topk(
            normed,
            h_v,
            pool["attn_v_op_key"],
            tau_all[:, :, 2:3],
            pool["attn_v_read"],
            pool["attn_v_write"],
            topk=topk,
            execution_kwargs=execution_kwargs,
            admission_den_power=admission_den_power,
        )
        q = q * qk_scale
        k = k * qk_scale
        v = v * v_scale
        qr = q.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        kr = k.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        vr = v.reshape(bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhsd,bhtd->bhst", qr, kr) / jnp.sqrt(jnp.float32(d_head))
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum("bhst,bhtd->bhsd", attn_w, vr)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
        attn_out = attn_out @ bp["attn"]["expand_O"]["kernel"]
        attn_out_norm.append(jnp.linalg.norm(attn_out, axis=-1).mean())
        x = x + attn_out

        normed = v4166._layer_norm(x, bp["norm2"]["scale"], bp["norm2"]["bias"])
        h_rst = normed @ router["proj_rst"]["kernel"] + router["proj_rst"]["bias"]
        h_rst = v4166._read_write_rst_operator_query(router, normed, h_rst)
        tau_rst = normed @ router["raw_tau_rst"]["kernel"] + router["raw_tau_rst"]["bias"]
        rst, rst_stats = _srw_with_topk(
            normed,
            h_rst,
            pool["rst_op_key"],
            tau_rst,
            pool["rst_read"],
            pool["rst_write"],
            topk=topk,
            execution_kwargs=execution_kwargs,
            admission_den_power=admission_den_power,
        )
        rst = rst * rst_scale
        rst_out_norm.append(jnp.linalg.norm(rst, axis=-1).mean())
        x = x + rst

        for prefix, stats in (
            ("q", q_stats),
            ("k", k_stats),
            ("v", v_stats),
            ("rst", rst_stats),
        ):
            for field, value in stats.items():
                traces[f"{prefix}_{field}"].append(value)

    out = {key: jnp.stack(values, axis=0) for key, values in traces.items()}
    out["attn_out_norm"] = jnp.stack(attn_out_norm)
    out["rst_out_norm"] = jnp.stack(rst_out_norm)
    return out


def _pad_or_trim(ids: Sequence[int], seq_len: int) -> tuple[np.ndarray, int]:
    ids = [int(x) for x in ids]
    actual = min(len(ids), seq_len)
    arr = np.zeros((seq_len,), dtype=np.int32)
    if actual:
        arr[:actual] = np.asarray(ids[:actual], dtype=np.int32)
    return arr, actual


def _load_prompt_file(path: str, tokenizer, seq_len: int) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    prompts = []
    for idx, row in enumerate(rows):
        if "token_ids" in row:
            ids = [int(x) for x in row["token_ids"]]
        elif "text" in row:
            if tokenizer is None:
                raise RuntimeError("Trace prompt text requires a locally available tokenizer.")
            ids = tokenizer.encode(str(row["text"]), add_special_tokens=False)
        else:
            continue
        arr, actual = _pad_or_trim(ids, seq_len)
        prompts.append({
            "prompt_id": row.get("id", f"prompt-{idx:06d}"),
            "text": row.get("text"),
            "token_ids": ids[:actual],
            "input_array": arr,
            "length": actual,
        })
    return prompts


def _default_prompts(ctx: AnalysisContext, seq_len: int, max_prompts: int) -> List[Dict[str, Any]]:
    loader = load_eval_data(
        ctx.config,
        max_length=seq_len,
        batch_size=max(1, ctx.n_hosts),
        host_id=ctx.host_id,
        n_hosts=ctx.n_hosts,
        max_tokens=seq_len * max(max_prompts, 1) * max(ctx.n_hosts, 1),
    )
    prompts = []
    for input_ids, _ in loader:
        host_rows = np.asarray(input_ids)
        for row in host_rows:
            arr, actual = _pad_or_trim(row.tolist(), seq_len)
            prompts.append({
                "prompt_id": f"val-{len(prompts):06d}",
                "text": None,
                "token_ids": [int(x) for x in row[:actual]],
                "input_array": arr,
                "length": actual,
            })
            if len(prompts) >= max_prompts:
                return prompts
    return prompts


def _heatmap_rows(trace: Dict[str, np.ndarray], actual_len: int) -> List[Dict[str, Any]]:
    rows = []
    for pool in TRACE_POOLS:
        active = trace[f"{pool}_active_count"][:, 0, :actual_len]
        mass = trace[f"{pool}_mass"][:, 0, :actual_len]
        top1 = trace[f"{pool}_top1_frac"][:, 0, :actual_len]
        top_idx = trace[f"{pool}_top_idx"][:, 0, :actual_len, 0]
        for layer in range(active.shape[0]):
            for pos in range(active.shape[1]):
                rows.append({
                    "layer": layer,
                    "position": pos,
                    "pool": pool,
                    "active_count": int(active[layer, pos]),
                    "gate_mass": float(mass[layer, pos]),
                    "top1_frac": float(top1[layer, pos]),
                    "top_operator_id": int(top_idx[layer, pos]),
                })
    return rows


def _prompt_summary(trace: Dict[str, np.ndarray], actual_len: int) -> Dict[str, Any]:
    summary = {}
    for pool in TRACE_POOLS:
        active = trace[f"{pool}_active_count"][:, 0, :actual_len]
        mass = trace[f"{pool}_mass"][:, 0, :actual_len]
        top1 = trace[f"{pool}_top1_frac"][:, 0, :actual_len]
        summary[pool] = {
            "active_mean": float(np.mean(active)) if active.size else 0.0,
            "active_max": int(np.max(active)) if active.size else 0,
            "mass_mean": float(np.mean(mass)) if mass.size else 0.0,
            "top1_frac_mean": float(np.mean(top1)) if top1.size else 0.0,
        }
    return summary


def run_trace_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "trace"
    store.set_stage_status(stage, "running")
    seq_len = int(args.trace_seq_len)
    topk = int(args.trace_topk or args.usage_topk)
    max_prompts = int(args.trace_max_prompts)
    tokenizer = maybe_load_tokenizer(local_only=True)

    if args.trace_prompts:
        prompts = _load_prompt_file(args.trace_prompts, tokenizer, seq_len)
    else:
        prompts = _default_prompts(ctx, seq_len, max_prompts)
    prompts = prompts[:max_prompts]
    if args.max_jobs_per_stage is not None:
        prompts = prompts[: int(args.max_jobs_per_stage)]

    store.log_event(
        stage,
        "start",
        message=f"TRACE START prompts={len(prompts)} seq_len={seq_len} topk={topk}",
        prompts=len(prompts),
        seq_len=seq_len,
        topk=topk,
    )

    summaries = []
    trace_fn = jax.jit(lambda p, x: topk_trace_forward(p, ctx.model_cfg, x, topk=topk))
    trace_t0 = time.time()
    for i, prompt in enumerate(prompts):
        json_path = store.path("trace", f"prompt-{i:06d}.json")
        npz_path = store.path("trace", f"prompt-{i:06d}_topk.npz")
        csv_path = store.path("trace", f"prompt-{i:06d}_heatmap.csv")
        if args.resume and should_skip_job(json_path, ["prompt_id", "summary"]) and should_skip_job(npz_path):
            meta = json.loads(json.dumps(prompt, default=str))
            summaries.append({"prompt_idx": i, "prompt_id": prompt["prompt_id"], "skipped": True})
            done = i + 1
            elapsed = time.time() - trace_t0
            eta = (elapsed / done) * max(0, len(prompts) - done) if done else None
            store.log_event(
                stage,
                "prompt_skip",
                message=(
                    f"TRACE prompt {done}/{len(prompts)} SKIP id={prompt['prompt_id']} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                ),
            )
            continue
        job_id = f"prompt-{i:06d}"
        store.mark_job_started(stage, job_id)
        prompt_t0 = time.time()
        input_arr = np.asarray(prompt["input_array"], dtype=np.int32)[None, :]
        trace_host = jax.device_get(trace_fn(ctx.params, jnp.asarray(input_arr)))
        prompt_sec = time.time() - prompt_t0
        done = i + 1
        elapsed = time.time() - trace_t0
        eta = (elapsed / done) * max(0, len(prompts) - done) if done else None
        trace_np = {k: np.asarray(v) for k, v in trace_host.items()}
        actual_len = int(prompt["length"])
        summary = _prompt_summary(trace_np, actual_len)
        meta = {
            "prompt_idx": i,
            "prompt_id": prompt["prompt_id"],
            "text": prompt.get("text"),
            "length": actual_len,
            "seq_len": seq_len,
            "topk": topk,
            "token_ids": prompt["token_ids"],
            "summary": summary,
        }
        if ctx.is_primary:
            write_json_atomic(json_path, meta)
            write_npz_atomic(npz_path, **trace_np)
            write_csv_atomic(csv_path, _heatmap_rows(trace_np, actual_len))
            store.mark_job_complete(stage, job_id, json_path, meta)
            store.log_event(
                stage,
                "prompt",
                message=(
                    f"TRACE prompt {i + 1}/{len(prompts)} id={prompt['prompt_id']} "
                    f"len={actual_len} "
                    f"prompt_sec={prompt_sec:.1f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                    f"q_active={summary['q']['active_mean']:.1f} "
                    f"k_active={summary['k']['active_mean']:.1f} "
                    f"v_active={summary['v']['active_mean']:.1f} "
                    f"rst_active={summary['rst']['active_mean']:.1f}"
                ),
                **meta,
            )
        summaries.append(meta)

    final = {"num_prompts": len(prompts), "prompts": summaries}
    if ctx.is_primary:
        write_json_atomic(store.path("trace", "trace_summary.json"), final)
        store.mark_job_complete(stage, "trace", store.path("trace", "trace_summary.json"), final)
        store.set_stage_status(stage, "complete")
        store.log_event(stage, "summary", message=f"TRACE SUMMARY prompts={len(prompts)}", **final)
    return final if ctx.is_primary else {}
