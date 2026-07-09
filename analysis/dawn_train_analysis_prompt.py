"""v4166 prompt, trace, and generation helpers for train analysis.

The train-analysis entry point owns orchestration. This module owns the
v4166-specific prompt-side probes so the item registry can stay declarative and
the main analyzer does not grow another block of prompt/generation code.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_common import AnalysisContext, maybe_load_tokenizer
from analysis.dawn_analysis_trace import (
    TRACE_POOLS,
    _default_prompts,
    _load_prompt_file,
    _pad_or_trim,
    topk_trace_forward,
)
from models import dawn_srw_v4166 as v4166


DEFAULT_TRACE_TEXT_PROMPTS = (
    "The capital of France is",
    "The largest planet in the solar system is",
    "I deposited money in the bank",
    "He sat by the river bank",
    "The phone needs a charge",
    "The police filed a charge",
)

DEFAULT_GENERATION_PROMPTS = (
    "The capital of France is",
    "Once upon a time",
    "In machine learning, gradient descent is",
)


def _env_list(name: str) -> Optional[List[str]]:
    raw = os.environ.get(name)
    if not raw:
        return None
    parts = [part.strip() for part in raw.split("|||")]
    return [part for part in parts if part]


def _mean(value: Any) -> Optional[float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    val = float(np.mean(arr))
    return val if math.isfinite(val) else None


def _max(value: Any) -> Optional[float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return None
    val = float(np.max(arr))
    return val if math.isfinite(val) else None


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def _pool_sizes_from_config(cfg: Dict[str, Any]) -> Dict[str, Optional[int]]:
    mcfg = cfg.get("model", {})
    n_rst = mcfg.get("n_rst", mcfg.get("n_know"))
    return {
        "q": int(mcfg["n_qk"]) if mcfg.get("n_qk") is not None else None,
        "k": int(mcfg["n_qk"]) if mcfg.get("n_qk") is not None else None,
        "v": int(mcfg["n_v"]) if mcfg.get("n_v") is not None else None,
        "rst": int(n_rst) if n_rst is not None else None,
    }


def _log_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def _text_prompts(tokenizer: Any, seq_len: int, max_prompts: int) -> List[Dict[str, Any]]:
    texts = _env_list("DAWN_TRAIN_ANALYSIS_PROMPTS") or list(DEFAULT_TRACE_TEXT_PROMPTS)
    prompts = []
    for idx, text in enumerate(texts[:max_prompts]):
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        arr, actual = _pad_or_trim(ids, seq_len)
        prompts.append({
            "prompt_id": f"text-{idx:06d}",
            "text": str(text),
            "token_ids": [int(x) for x in ids[:actual]],
            "input_array": arr,
            "length": actual,
        })
    return prompts


def _load_trace_prompts(ctx: AnalysisContext, tokenizer: Any, seq_len: int,
                        max_prompts: int) -> List[Dict[str, Any]]:
    args = ctx.args
    if getattr(args, "trace_prompts", None):
        prompts = _load_prompt_file(str(args.trace_prompts), tokenizer, seq_len)
    elif tokenizer is not None:
        prompts = _text_prompts(tokenizer, seq_len, max_prompts)
    else:
        prompts = _default_prompts(ctx, seq_len, max_prompts)
    prompts = prompts[:max_prompts]
    if getattr(args, "max_jobs_per_stage", None) is not None:
        prompts = prompts[: int(args.max_jobs_per_stage)]
    return prompts


def _summarize_trace(trace_np: Dict[str, np.ndarray], actual_len: int,
                     pool_sizes: Dict[str, Optional[int]]) -> Dict[str, Any]:
    pools: Dict[str, Dict[str, Any]] = {}
    for pool in TRACE_POOLS:
        active = np.asarray(trace_np[f"{pool}_active_count"])[:, 0, :actual_len]
        mass = np.asarray(trace_np[f"{pool}_mass"])[:, 0, :actual_len]
        top1 = np.asarray(trace_np[f"{pool}_top1_frac"])[:, 0, :actual_len]
        layer_active = active.mean(axis=1) if active.size else np.asarray([])
        layer_top1 = top1.mean(axis=1) if top1.size else np.asarray([])
        active_mean = _mean(active)
        pool_size = pool_sizes.get(pool)
        pools[pool] = {
            "pool_size": pool_size,
            "active_mean": active_mean,
            "active_frac_mean": _safe_ratio(active_mean, float(pool_size)) if pool_size else None,
            "active_max": int(np.max(active)) if active.size else None,
            "mass_mean": _mean(mass),
            "top1_mean": _mean(top1),
            "top1_max": _max(top1),
            "weakest_layer": int(np.argmin(layer_active)) if layer_active.size else None,
            "most_concentrated_layer": int(np.argmax(layer_top1)) if layer_top1.size else None,
        }
    attn_norm = np.asarray(trace_np.get("attn_out_norm", []), dtype=np.float64)
    rst_norm = np.asarray(trace_np.get("rst_out_norm", []), dtype=np.float64)
    attn_mean = _mean(attn_norm)
    rst_mean = _mean(rst_norm)
    return {
        "pools": pools,
        "attn_out_norm_mean": attn_mean,
        "rst_out_norm_mean": rst_mean,
        "rst_attn_norm_ratio": _safe_ratio(rst_mean, attn_mean),
    }


def run_train_prompt_trace(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    seq_len = int(getattr(args, "trace_seq_len", 128) or 128)
    max_prompts = int(getattr(args, "trace_max_prompts", 6) or 6)
    topk = int(getattr(args, "trace_topk", None) or getattr(args, "usage_topk", 8) or 8)
    tokenizer = maybe_load_tokenizer(local_only=True)
    try:
        prompts = _load_trace_prompts(ctx, tokenizer, seq_len, max_prompts)
    except Exception as exc:
        if ctx.is_primary:
            print(
                "TRAIN_ANALYSIS PROMPT_TRACE SKIP "
                f"reason={type(exc).__name__}: {exc}",
                flush=True,
            )
            return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "prompts": []}
        return {}

    if ctx.is_primary:
        print(
            "TRAIN_ANALYSIS PROMPT_TRACE START "
            f"prompts={len(prompts)} seq_len={seq_len} topk={topk}",
            flush=True,
        )
    if not prompts:
        return {"status": "empty", "prompts": []} if ctx.is_primary else {}

    trace_fn = jax.jit(lambda p, x: topk_trace_forward(p, ctx.model_cfg, x, topk=topk))
    pool_sizes = _pool_sizes_from_config(ctx.config)
    rows = []
    started = time.time()
    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        input_arr = np.asarray(prompt["input_array"], dtype=np.int32)[None, :]
        trace_host = jax.device_get(trace_fn(ctx.params, jnp.asarray(input_arr)))
        trace_np = {key: np.asarray(value) for key, value in trace_host.items()}
        prompt_sec = time.time() - t0
        actual_len = int(prompt.get("length") or 0)
        summary = _summarize_trace(trace_np, actual_len, pool_sizes)
        row = {
            "prompt_idx": idx,
            "prompt_id": prompt.get("prompt_id", f"prompt-{idx:06d}"),
            "text": prompt.get("text"),
            "length": actual_len,
            "topk": topk,
            "prompt_sec": prompt_sec,
            "token_ids_head": [int(x) for x in prompt.get("token_ids", [])[:32]],
            "summary": summary,
        }
        rows.append(row)
        if ctx.is_primary:
            pools = summary.get("pools", {})
            print(
                "TRAIN_ANALYSIS PROMPT_TRACE "
                f"prompt={idx + 1:03d}/{len(prompts):03d} "
                f"id={row['prompt_id']} len={actual_len} "
                f"q={_log_float(pools.get('q', {}).get('active_mean')):.1f} "
                f"k={_log_float(pools.get('k', {}).get('active_mean')):.1f} "
                f"v={_log_float(pools.get('v', {}).get('active_mean')):.1f} "
                f"rst={_log_float(pools.get('rst', {}).get('active_mean')):.1f} "
                f"sec={prompt_sec:.2f}",
                flush=True,
            )

    if not ctx.is_primary:
        return {}
    return {
        "status": "ready",
        "num_prompts": len(rows),
        "seq_len": seq_len,
        "topk": topk,
        "pool_sizes": pool_sizes,
        "inference_model_cfg": {
            "soft_gate_boundary_power": ctx.model_cfg.get("soft_gate_boundary_power"),
            "soft_gate_temperature": ctx.model_cfg.get("soft_gate_temperature"),
            "admission_den_power": ctx.model_cfg.get("admission_den_power"),
            "soft_gate_effective_active_eps": ctx.model_cfg.get("soft_gate_effective_active_eps"),
        },
        "sec": time.time() - started,
        "prompts": rows,
    }


def build_train_prompt_decision(prompt_trace: Dict[str, Any]) -> Dict[str, Any]:
    if not prompt_trace or prompt_trace.get("status") not in ("ready", "empty"):
        return {
            "status": "skipped",
            "reason": prompt_trace.get("reason", "prompt_trace unavailable") if prompt_trace else "prompt_trace unavailable",
            "rows": [],
        }
    rows = []
    watch_count = 0
    for prompt in prompt_trace.get("prompts", []):
        summary = prompt.get("summary", {})
        pools = summary.get("pools", {})
        q_frac = pools.get("q", {}).get("active_frac_mean")
        k_frac = pools.get("k", {}).get("active_frac_mean")
        v_frac = pools.get("v", {}).get("active_frac_mean")
        rst_frac = pools.get("rst", {}).get("active_frac_mean")
        qk_frac = None
        if q_frac is not None and k_frac is not None:
            qk_frac = (float(q_frac) + float(k_frac)) * 0.5
        rst_top1 = pools.get("rst", {}).get("top1_max")
        q_top1 = pools.get("q", {}).get("top1_max")
        k_top1 = pools.get("k", {}).get("top1_max")
        ratio = summary.get("rst_attn_norm_ratio")
        flags = []
        if qk_frac is not None and v_frac is not None and qk_frac < float(v_frac) * 0.50:
            flags.append("qk_route_thin_vs_v")
        if rst_top1 is not None and float(rst_top1) > 0.25:
            flags.append("rst_concentrated")
        qk_top1 = max(float(q_top1 or 0.0), float(k_top1 or 0.0))
        if qk_top1 > 0.25:
            flags.append("qk_concentrated")
        if ratio is not None and float(ratio) < 0.15:
            flags.append("rst_output_low")
        if ratio is not None and float(ratio) > 4.0:
            flags.append("rst_output_high")
        status = "watch" if flags else "ok"
        if status == "watch":
            watch_count += 1
        rows.append({
            "prompt_id": prompt.get("prompt_id"),
            "length": prompt.get("length"),
            "status": status,
            "qk_active_frac": qk_frac,
            "v_active_frac": v_frac,
            "rst_active_frac": rst_frac,
            "rst_top1_max": rst_top1,
            "rst_attn_norm_ratio": ratio,
            "reason": ",".join(flags) if flags else "balanced",
        })
    return {
        "status": "ready",
        "num_prompts": len(rows),
        "watch_count": watch_count,
        "rows": rows,
    }


def _generation_prompts() -> List[str]:
    return _env_list("DAWN_TRAIN_ANALYSIS_GENERATION_PROMPTS") or list(DEFAULT_GENERATION_PROMPTS)


def _decode_ids(tokenizer: Any, ids: Sequence[int]) -> str:
    ids = [int(x) for x in ids]
    if tokenizer is not None:
        try:
            return tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    return "ids:" + " ".join(str(x) for x in ids)


def _token_text(tokenizer: Any, token_id: int) -> str:
    token_id = int(token_id)
    if tokenizer is not None:
        try:
            return str(tokenizer.convert_ids_to_tokens([token_id])[0])
        except Exception:
            pass
        try:
            return tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            pass
    return str(token_id)


def _top_token_snapshot(logits: np.ndarray, tokenizer: Any, top_n: int = 8) -> List[Dict[str, Any]]:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return []
    top_n = min(int(top_n), int(logits.shape[-1]))
    top_idx = np.argpartition(logits, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(logits[top_idx])[::-1]]
    shifted = logits - np.nanmax(logits)
    probs = np.exp(shifted)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return [
        {
            "id": int(idx),
            "token": _token_text(tokenizer, int(idx)),
            "logit": float(logits[idx]),
            "prob": float(probs[idx]),
        }
        for idx in top_idx
    ]


def _dominant_token_summary(ids: Sequence[int], tokenizer: Any) -> Dict[str, Any]:
    ids = [int(x) for x in ids]
    if not ids:
        return {"id": None, "token": None, "count": 0, "frac": None, "unique": 0}
    counts: Dict[int, int] = {}
    for token_id in ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    token_id, count = max(counts.items(), key=lambda item: item[1])
    return {
        "id": int(token_id),
        "token": _token_text(tokenizer, token_id),
        "count": int(count),
        "frac": float(count) / float(len(ids)),
        "unique": len(counts),
    }


def _sample_token(logits: np.ndarray, temperature: float, top_k: int,
                  rng: np.random.Generator) -> int:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        return int(np.argmax(logits))
    logits = logits / max(float(temperature), 1e-6)
    if top_k and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        k = int(top_k)
        keep = np.argpartition(logits, -k)[-k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits = logits - np.nanmax(logits)
    probs = np.exp(logits)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return int(rng.choice(np.arange(probs.shape[-1]), p=probs))


def run_train_generation_samples(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    tokenizer = maybe_load_tokenizer(local_only=True)
    max_new = int(getattr(args, "train_analysis_generation_max_tokens", 64) or 64)
    temperature = float(getattr(args, "train_analysis_generation_temperature", 0.8) or 0.0)
    top_k = int(getattr(args, "train_analysis_generation_top_k", 50) or 50)
    max_prompts = int(getattr(args, "train_analysis_generation_max_prompts", 3) or 3)
    max_seq = int(ctx.model_cfg.get("max_seq_len", 512))
    if tokenizer is not None:
        prompts: List[Any] = _generation_prompts()[:max_prompts]
    else:
        prompts = _default_prompts(
            ctx,
            seq_len=min(128, max_seq),
            max_prompts=max_prompts,
        )
    if getattr(args, "max_jobs_per_stage", None) is not None:
        prompts = prompts[: int(args.max_jobs_per_stage)]

    if ctx.is_primary:
        print(
            "TRAIN_ANALYSIS GENERATION START "
            f"prompts={len(prompts)} max_new={max_new} temp={temperature:g} top_k={top_k} "
            f"tokenizer={'yes' if tokenizer is not None else 'no'}",
            flush=True,
        )
    if not prompts:
        return {"status": "empty", "samples": []} if ctx.is_primary else {}

    stop_ids = {
        int(x)
        for x in (getattr(tokenizer, "sep_token_id", None), getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None))
        if x is not None
    }
    jit_prefill = jax.jit(lambda p, ids: v4166.prefill(p, ctx.model_cfg, ids))
    jit_decode = jax.jit(lambda p, tok, c_k, c_v, c_len: v4166.decode_step(p, ctx.model_cfg, tok, c_k, c_v, c_len))
    rng = np.random.default_rng(123)
    samples = []
    started = time.time()
    for idx, prompt in enumerate(prompts):
        if isinstance(prompt, dict):
            prompt_text = prompt.get("text")
            ids = [int(x) for x in prompt.get("token_ids", [])]
            prompt_id = str(prompt.get("prompt_id", f"gen-{idx:06d}"))
        else:
            prompt_text = str(prompt)
            ids = tokenizer.encode(str(prompt), add_special_tokens=False)
            prompt_id = f"gen-{idx:06d}"
        ids = [int(x) for x in ids[: max(1, max_seq - 1)]]
        if not ids:
            continue
        if prompt_text is None:
            prompt_text = _decode_ids(tokenizer, ids)
        t0 = time.time()
        logits, c_k, c_v, c_len = jit_prefill(ctx.params, jnp.asarray([ids], dtype=jnp.int32))
        jax.block_until_ready(logits)
        last_logits = np.asarray(jax.device_get(logits[0, -1, :]))
        first_step_top_tokens = _top_token_snapshot(last_logits, tokenizer)
        generated: List[int] = []
        for _ in range(min(max_new, max_seq - len(ids))):
            next_id = _sample_token(last_logits, temperature, top_k, rng)
            generated.append(next_id)
            if next_id in stop_ids or int(c_len) >= max_seq:
                break
            logits_d, c_k, c_v, c_len = jit_decode(
                ctx.params,
                jnp.asarray([next_id], dtype=jnp.int32),
                c_k,
                c_v,
                c_len,
            )
            jax.block_until_ready(logits_d)
            last_logits = np.asarray(jax.device_get(logits_d[0, :]))
        sec = time.time() - t0
        full_ids = ids + generated
        full_text = _decode_ids(tokenizer, full_ids)
        continuation = _decode_ids(tokenizer, generated)
        dominant = _dominant_token_summary(generated, tokenizer)
        row = {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "prompt_token_ids_head": ids[:32],
            "continuation": continuation,
            "continuation_token_ids": generated,
            "continuation_token_texts": [_token_text(tokenizer, token_id) for token_id in generated[:128]],
            "full_text": full_text,
            "prompt_tokens": len(ids),
            "new_tokens": len(generated),
            "sec": sec,
            "tokens_per_sec": (len(generated) / sec) if sec > 0 else None,
            "temperature": temperature,
            "top_k": top_k,
            "first_step_top_tokens": first_step_top_tokens,
            "dominant_generated_token": dominant,
        }
        samples.append(row)
        if ctx.is_primary:
            dom = dominant.get("token")
            dom_frac = dominant.get("frac")
            dom_text = f" dom={dom}:{dom_frac:.2f}" if dom is not None and dom_frac is not None else ""
            print(
                "TRAIN_ANALYSIS GENERATION "
                f"sample={idx + 1:03d}/{len(prompts):03d} "
                f"new_tokens={len(generated)} sec={sec:.2f}{dom_text}",
                flush=True,
            )

    if not ctx.is_primary:
        return {}
    return {
        "status": "ready",
        "num_samples": len(samples),
        "tokenizer": "bert-base-uncased" if tokenizer is not None else None,
        "decode_mode": "text" if tokenizer is not None else "token_ids",
        "max_new_tokens": max_new,
        "temperature": temperature,
        "top_k": top_k,
        "sampling_seed": 123,
        "inference_model_cfg": {
            "soft_gate_boundary_power": ctx.model_cfg.get("soft_gate_boundary_power"),
            "soft_gate_temperature": ctx.model_cfg.get("soft_gate_temperature"),
            "admission_den_power": ctx.model_cfg.get("admission_den_power"),
            "soft_gate_effective_active_eps": ctx.model_cfg.get("soft_gate_effective_active_eps"),
        },
        "sec": time.time() - started,
        "samples": samples,
    }
