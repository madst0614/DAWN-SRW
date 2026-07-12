"""Small-budget causal operator suppression experiments."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from analysis.dawn_analysis_common import (
    AnalysisContext,
    analysis_model_module,
    format_duration,
    host_aligned_batch_size,
    load_eval_data,
)
from analysis.dawn_analysis_storage import (
    list_paths,
    read_json,
    read_npz,
    should_skip_job,
    write_csv_atomic,
    write_json_atomic,
)
ABLATION_ANALYSIS_IMPL = "dynamic_mask_forward_v2"


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> List[int]:
    if not value:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strs(value: str | None, default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    allowed = {"top", "random", "low"}
    out = []
    for item in value.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(f"Unknown ablation strategy {name!r}; expected one of {sorted(allowed)}")
        out.append(name)
    return out or list(default)


def _parse_pools(value: str | None) -> List[str]:
    if not value:
        return ["qk", "v", "rst"]
    return [x.strip() for x in value.split(",") if x.strip()]


def _pool_size(ctx: AnalysisContext, pool: str) -> int:
    if pool == "qk":
        return int(ctx.model_cfg["n_qk"])
    if pool == "v":
        return int(ctx.model_cfg["n_v"])
    return int(ctx.model_cfg["n_rst"])


def _loss_from_logits(logits, input_ids):
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    preds = jnp.argmax(logits, axis=-1)
    valid = jnp.ones_like(labels, dtype=jnp.bool_)
    return (
        token_loss.sum(),
        ((preds == labels) & valid).astype(jnp.int32).sum(),
        valid.astype(jnp.int32).sum(),
    )


def _build_dynamic_suppressed_forward(params, model_cfg):
    model_module = analysis_model_module(model_cfg)
    params = model_module._squeeze_params(params)
    params = jax.tree.map(jnp.asarray, params)
    angular_execution_kwargs = (
        model_module._angular_execution_kwargs_from_model_cfg(model_cfg))

    def _srw_sup(x, h, op_key, tau_off, raw_scan_offset, w_read, w_write, mult):
        r_n = model_module._forward_unit_direction(w_read.astype(jnp.float32))
        w_n = model_module._forward_unit_direction(w_write.astype(jnp.float32))
        execution_kwargs, admission_den_power = model_module._split_admission_den_kwargs(
            angular_execution_kwargs
        )
        _, admission, _, execution_weight, _ = model_module._angular_execution(
            h, op_key, tau_off, raw_scan_offset, **execution_kwargs
        )
        mult = jnp.asarray(mult, dtype=jnp.float32)
        execution_weight = execution_weight * mult[None, None, :]
        admission = admission * mult[None, None, :]
        xr = x.astype(jnp.float32) @ r_n.T
        out = (execution_weight * xr) @ w_n
        admission_mass = admission.sum(axis=-1, keepdims=True)
        composition_den = getattr(model_module, "_composition_den", None)
        if composition_den is None:
            admission_den = jnp.power(
                jnp.maximum(admission_mass, 1.0), admission_den_power)
        elif hasattr(model_module, "DEFAULT_SRW_COMPOSITION_MODE"):
            admission_den = composition_den(
                admission_mass,
                admission_den_power,
                execution_kwargs.get(
                    "srw_composition_mode",
                    model_module.DEFAULT_SRW_COMPOSITION_MODE),
            )
        else:
            admission_den = composition_den(
                admission_mass, admission_den_power)
        return (out.astype(jnp.float32) / admission_den).astype(jnp.float32)

    def forward_fn(input_ids, qk_mult, v_mult, rst_mult):
        input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
        qk_mult = jnp.asarray(qk_mult, dtype=jnp.float32)
        v_mult = jnp.asarray(v_mult, dtype=jnp.float32)
        rst_mult = jnp.asarray(rst_mult, dtype=jnp.float32)
        bsz, seq_len = input_ids.shape
        d_model = int(model_cfg["d_model"])
        n_layers = int(model_cfg["n_layers"])
        n_heads = int(model_cfg["n_heads"])
        d_head = d_model // n_heads
        pp = model_module._pool_params_with_operator_keys(params["neuron_pool"])
        rp = params["router"]
        qk_scale_eff, v_scale_eff, rst_scale_eff = model_module._effective_pool_output_scales(
            pp,
            d_model,
            n_layers,
        )

        positions = jnp.arange(seq_len)[jnp.newaxis, :]
        x = params["token_emb"]["embedding"][input_ids] + params["pos_emb"]["embedding"][positions]
        qk_n = pp["attn_qk_op_key"]
        v_n = pp["attn_v_op_key"]
        rst_n = pp["rst_op_key"]

        for i in range(n_layers):
            bp = params[f"block_{i}"]
            normed = model_module._layer_norm(
                x, bp["norm1"]["scale"], bp["norm1"]["bias"])
            h_all = normed @ rp["proj_attn"]["kernel"] + rp["proj_attn"]["bias"]
            h_q, h_k, h_v = jnp.split(h_all, 3, axis=-1)
            query_adapter = getattr(
                model_module, "_read_write_attn_operator_queries", None)
            if query_adapter is not None:
                h_q, h_k, h_v = query_adapter(
                    rp, normed, h_q, h_k, h_v)
            tau_all = normed @ rp["raw_tau_attn"]["kernel"] + rp["raw_tau_attn"]["bias"]
            raw_scan_offset_all = jnp.zeros_like(tau_all)

            q = _srw_sup(
                normed,
                h_q,
                qk_n,
                tau_all[:, :, 0:1],
                raw_scan_offset_all[:, :, 0:1],
                pp["attn_qk_read"],
                pp["attn_qk_write"],
                qk_mult,
            )
            k = _srw_sup(
                normed,
                h_k,
                qk_n,
                tau_all[:, :, 1:2],
                raw_scan_offset_all[:, :, 1:2],
                pp["attn_qk_read"],
                pp["attn_qk_write"],
                qk_mult,
            )
            v = _srw_sup(
                normed,
                h_v,
                v_n,
                tau_all[:, :, 2:3],
                raw_scan_offset_all[:, :, 2:3],
                pp["attn_v_read"],
                pp["attn_v_write"],
                v_mult,
            )
            q = q * qk_scale_eff
            k = k * qk_scale_eff
            v = v * v_scale_eff

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
            x = x + attn_out

            normed = model_module._layer_norm(
                x, bp["norm2"]["scale"], bp["norm2"]["bias"])
            h_rst = normed @ rp["proj_rst"]["kernel"] + rp["proj_rst"]["bias"]
            rst_query_adapter = getattr(
                model_module, "_read_write_rst_operator_query", None)
            if rst_query_adapter is not None:
                h_rst = rst_query_adapter(rp, normed, h_rst)
            tau_rst = normed @ rp["raw_tau_rst"]["kernel"] + rp["raw_tau_rst"]["bias"]
            raw_scan_offset_rst = jnp.zeros_like(tau_rst)
            rst = _srw_sup(
                normed,
                h_rst,
                rst_n,
                tau_rst,
                raw_scan_offset_rst,
                pp["rst_read"],
                pp["rst_write"],
                rst_mult,
            )
            x = x + rst * rst_scale_eff

        norm_p = params["norm"]
        x = model_module._layer_norm(
            x, norm_p["scale"], norm_p["bias"])
        return x @ params["token_emb"]["embedding"].T

    return forward_fn


def _eval_forward(forward_fn, batches: List[np.ndarray],
                  mults: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Dict[str, Any]:
    loss_sum = jnp.float32(0.0)
    correct = jnp.int32(0)
    valid = jnp.int32(0)
    qk_mult, v_mult, rst_mult = mults
    for batch in batches:
        ids = jnp.asarray(batch, dtype=jnp.int32)
        logits = forward_fn(ids, qk_mult, v_mult, rst_mult)
        lsum, corr, val = _loss_from_logits(logits, ids)
        loss_sum = loss_sum + lsum
        correct = correct + corr
        valid = valid + val
    local = jnp.asarray([loss_sum, correct.astype(jnp.float32), valid.astype(jnp.float32)], dtype=jnp.float32)
    if int(jax.process_count()) > 1:
        gathered = np.asarray(process_allgather(local)).reshape(-1, 3)
        totals = gathered.sum(axis=0)
    else:
        totals = np.asarray(jax.device_get(local), dtype=np.float32)
    loss_sum_h, correct_h, valid_h = float(totals[0]), int(totals[1]), int(totals[2])
    loss = float(loss_sum_h) / max(1, int(valid_h))
    return {
        "loss_sum": float(loss_sum_h),
        "loss": loss,
        "accuracy": int(correct_h) / max(1, int(valid_h)),
        "correct": int(correct_h),
        "valid_count": int(valid_h),
    }


def _operator_lists(ctx: AnalysisContext) -> Dict[str, Dict[str, List[int]]]:
    usage_npz = ctx.store.path("usage", "operator_usage_by_pool.npz")
    out = {}
    if should_skip_job(usage_npz):
        arrays = read_npz(usage_npz)
    else:
        arrays = {}
    rng = np.random.default_rng(0)
    for pool in _parse_pools(ctx.args.ablation_pools):
        n = _pool_size(ctx, pool)
        mass = np.asarray(arrays.get(f"{pool}_mass_sum", np.zeros((n,), dtype=np.float64)))
        count = np.asarray(arrays.get(f"{pool}_usage_count", np.zeros((n,), dtype=np.int64)))
        top = np.argsort(-mass).astype(np.int64).tolist()
        low = np.argsort(count + (mass > 0).astype(np.int64) * 1_000_000_000).astype(np.int64).tolist()
        random_ids = rng.permutation(n).astype(np.int64).tolist()
        out[pool] = {
            "top": top,
            "low": low,
            "random": random_ids,
        }
    return out


def _job_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _make_mask(ctx: AnalysisContext, pool: str, ids: Sequence[int]) -> Dict[str, Any]:
    n = _pool_size(ctx, pool)
    arr = np.zeros((n,), dtype=np.bool_)
    ids = [int(i) for i in ids if 0 <= int(i) < n]
    if ids:
        arr[np.asarray(ids, dtype=np.int64)] = True
    key = "rst" if pool == "rst" else pool
    return {key: jnp.asarray(arr)}


def _make_multipliers(ctx: AnalysisContext, pool: str, ids: Sequence[int]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    qk = np.ones((_pool_size(ctx, "qk"),), dtype=np.float32)
    v = np.ones((_pool_size(ctx, "v"),), dtype=np.float32)
    rst = np.ones((_pool_size(ctx, "rst"),), dtype=np.float32)
    target = {"qk": qk, "v": v, "rst": rst}[pool]
    valid_ids = [int(i) for i in ids if 0 <= int(i) < target.shape[0]]
    if valid_ids:
        target[np.asarray(valid_ids, dtype=np.int64)] = 0.0
    return jnp.asarray(qk), jnp.asarray(v), jnp.asarray(rst)


def _base_multipliers(ctx: AnalysisContext) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.ones((_pool_size(ctx, "qk"),), dtype=jnp.float32),
        jnp.ones((_pool_size(ctx, "v"),), dtype=jnp.float32),
        jnp.ones((_pool_size(ctx, "rst"),), dtype=jnp.float32),
    )


def _job_expected_params(ctx: AnalysisContext, payload: Dict[str, Any],
                         batch_size: int, seq_len: int, max_sequences: int,
                         num_batches: int) -> Dict[str, Any]:
    out = {
        "analysis_impl": ABLATION_ANALYSIS_IMPL,
        "pool": payload["pool"],
        "strategy": payload["strategy"],
        "k": int(payload["k"]),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "max_sequences": int(max_sequences),
        "num_batches": int(num_batches),
        "n_hosts": int(ctx.n_hosts),
    }
    if ctx.checkpoint_step is not None:
        out["checkpoint_step"] = int(ctx.checkpoint_step)
    return out


def _job_params_match(rec: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    actual = rec.get("analysis_params", {})
    for key, expected_value in expected.items():
        if isinstance(expected_value, str):
            if actual.get(key) != expected_value:
                return False
            continue
        try:
            if int(actual.get(key)) != int(expected_value):
                return False
        except Exception:
            return False
    return True


def _all_hosts_true(value: bool) -> bool:
    if int(jax.process_count()) <= 1:
        return bool(value)
    local = jnp.asarray([1 if value else 0], dtype=jnp.int32)
    gathered = np.asarray(process_allgather(local)).reshape(-1)
    return bool(np.all(gathered == 1))


def _load_batches(ctx: AnalysisContext) -> List[np.ndarray]:
    args = ctx.args
    seq_len = int(args.ablation_seq_len)
    batch_size = host_aligned_batch_size(int(args.ablation_batch_size), ctx.n_hosts)
    max_sequences = int(args.ablation_max_sequences)
    max_tokens = seq_len * max_sequences
    loader = load_eval_data(ctx.config, seq_len, batch_size, ctx.host_id, ctx.n_hosts, max_tokens)
    max_batches = len(loader)
    if args.max_jobs_per_stage is not None:
        max_batches = min(max_batches, int(args.max_jobs_per_stage))
    batches = []
    for i, (input_ids, _) in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(np.asarray(input_ids, dtype=np.int32))
    return batches


def run_ablation_stage(ctx: AnalysisContext) -> Dict[str, Any]:
    args = ctx.args
    store = ctx.store
    stage = "ablation"
    store.set_stage_status(stage, "running")
    k_list = _parse_csv_ints(args.ablation_k_list, [1, 16, 64])
    pools = _parse_pools(args.ablation_pools)
    strategies = _parse_csv_strs(getattr(args, "ablation_strategies", None), ["top"])
    operator_lists = _operator_lists(ctx)
    batches = _load_batches(ctx)
    requested_batch_size = int(args.ablation_batch_size)
    batch_size = host_aligned_batch_size(requested_batch_size, ctx.n_hosts)
    seq_len = int(args.ablation_seq_len)
    max_sequences = int(args.ablation_max_sequences)
    total_jobs = len(pools) * len(k_list) * len(strategies)

    store.log_event(
        stage,
        "start",
        message=(
            f"ABLATION START jobs={total_jobs} "
            f"batches={len(batches)} batch_size={batch_size} "
            f"requested_batch_size={requested_batch_size} "
            f"k={k_list} pools={','.join(pools)} "
            f"strategies={','.join(strategies)} host={ctx.host_id}/{ctx.n_hosts}"
        ),
        pools=pools,
        k_list=k_list,
        strategies=strategies,
        batches=len(batches),
        batch_size=batch_size,
        requested_batch_size=requested_batch_size,
    )

    store.log_event(
        stage,
        "base_start",
        message=(
            f"ABLATION BASE START dynamic_mask_jit_compile_and_eval "
            f"batches={len(batches)} batch_size={batch_size} "
            f"host={ctx.host_id}/{ctx.n_hosts} primary={ctx.is_primary}"
        ),
    )
    forward = jax.jit(_build_dynamic_suppressed_forward(ctx.params, ctx.model_cfg))
    base_t0 = time.time()
    base = _eval_forward(forward, batches, _base_multipliers(ctx))
    base_sec = time.time() - base_t0
    if not ctx.is_primary:
        store.log_event(
            stage,
            "base_host_done",
            message=(
                f"ABLATION BASE HOST DONE loss={base['loss']:.6f} "
                f"acc={base['accuracy']:.4f} tokens={base['valid_count']:,} "
                f"base_sec={base_sec:.1f} host={ctx.host_id}/{ctx.n_hosts} "
                "primary host=0 writes summary"
            ),
            **base,
        )
    if ctx.is_primary:
        store.log_event(
            stage,
            "base",
            message=(
                f"ABLATION BASE loss={base['loss']:.6f} acc={base['accuracy']:.4f} "
                f"tokens={base['valid_count']:,} base_sec={base_sec:.1f} hosts={ctx.n_hosts}"
            ),
            **base,
        )

    records = []
    completed_jobs = 0
    jobs_t0 = time.time()
    for pool in pools:
        for strategy in strategies:
            candidates = operator_lists[pool][strategy]
            for k in k_list:
                op_ids = [int(x) for x in candidates[: min(int(k), len(candidates))]]
                payload = {
                    "checkpoint_step": ctx.checkpoint_step,
                    "pool": pool,
                    "strategy": strategy,
                    "k": int(k),
                    "operator_ids": op_ids,
                }
                jid = _job_id(payload)
                path = store.path("ablation", "jobs", f"job-{jid}.json")
                expected_params = _job_expected_params(
                    ctx,
                    payload,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    max_sequences=max_sequences,
                    num_batches=len(batches),
                )
                rec = {}
                local_skip = False
                if args.resume and should_skip_job(path, ["delta_loss", "analysis_params"]):
                    rec = read_json(path)
                    local_skip = _job_params_match(rec, expected_params)
                if _all_hosts_true(local_skip):
                    records.append(rec)
                    completed_jobs += 1
                    elapsed = time.time() - jobs_t0
                    eta = (elapsed / completed_jobs) * max(0, total_jobs - completed_jobs)
                    if not ctx.is_primary:
                        store.log_event(
                            stage,
                            "job_host_skip",
                            message=(
                                f"ABLATION job {completed_jobs:03d}/{total_jobs:03d} "
                                f"{pool}/{strategy}/k={k} SKIP "
                                f"host={ctx.host_id}/{ctx.n_hosts} primary host=0 writes summary"
                            ),
                        )
                    else:
                        store.log_event(
                            stage,
                            "job_skip",
                            message=(
                                f"ABLATION job {completed_jobs:03d}/{total_jobs:03d} "
                                f"{pool}/{strategy}/k={k} SKIP "
                                f"delta_loss={float(rec.get('delta_loss', 0.0)):.6f} "
                                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                            ),
                            **rec,
                        )
                    continue
                store.mark_job_started(stage, jid)
                try:
                    job_t0 = time.time()
                    job_no = completed_jobs + 1
                    store.log_event(
                        stage,
                        "job_start",
                        message=(
                            f"ABLATION job {job_no:03d}/{total_jobs:03d} START "
                            f"{pool}/{strategy}/k={k} ops={len(op_ids)} "
                            f"host={ctx.host_id}/{ctx.n_hosts} primary={ctx.is_primary}"
                        ),
                    )
                    ablated = _eval_forward(forward, batches, _make_multipliers(ctx, pool, op_ids))
                    job_sec = time.time() - job_t0
                    completed_jobs += 1
                    elapsed = time.time() - jobs_t0
                    eta = (elapsed / completed_jobs) * max(0, total_jobs - completed_jobs)
                    rec = {
                        **payload,
                        "job_id": jid,
                        "analysis_params": expected_params,
                        "base_loss": base["loss"],
                        "ablated_loss": ablated["loss"],
                        "delta_loss": ablated["loss"] - base["loss"],
                        "base_acc": base["accuracy"],
                        "ablated_acc": ablated["accuracy"],
                        "delta_acc": ablated["accuracy"] - base["accuracy"],
                        "valid_tokens": ablated["valid_count"],
                        "job_sec": job_sec,
                    }
                    if not ctx.is_primary:
                        store.log_event(
                            stage,
                            "job_host_done",
                            message=(
                                f"ABLATION job {completed_jobs:03d}/{total_jobs:03d} HOST DONE "
                                f"{pool}/{strategy}/k={k} job_sec={job_sec:.1f} "
                                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                                f"host={ctx.host_id}/{ctx.n_hosts} primary host=0 writes result"
                            ),
                            **rec,
                        )
                    if ctx.is_primary:
                        write_json_atomic(path, rec)
                        store.mark_job_complete(stage, jid, path, rec)
                        store.log_event(
                            stage,
                            "job",
                            message=(
                                f"ABLATION job {completed_jobs:03d}/{total_jobs:03d} "
                                f"{pool}/{strategy}/k={k} "
                                f"delta_loss={rec['delta_loss']:.6f} "
                                f"delta_acc={rec['delta_acc']:.4f} "
                                f"ablated_loss={rec['ablated_loss']:.6f} "
                                f"job_sec={job_sec:.1f} "
                                f"elapsed={format_duration(elapsed)} "
                                f"eta={format_duration(eta)}"
                            ),
                            **rec,
                        )
                    records.append(rec)
                except Exception as exc:
                    err = traceback.format_exc()
                    store.mark_job_failed(stage, jid, str(exc), err)
                    store.log_event(
                        stage,
                        "job_failed",
                        message=f"ABLATION {pool}/{strategy}/k={k} FAILED {type(exc).__name__}: {exc}",
                        error=str(exc),
                        job_id=jid,
                    )
                    if args.fail_fast:
                        raise

    if ctx.is_primary:
        all_records = list(records)
        csv_rows = [
            {
                "pool": r["pool"],
                "strategy": r["strategy"],
                "k": r["k"],
                "base_loss": r["base_loss"],
                "ablated_loss": r["ablated_loss"],
                "delta_loss": r["delta_loss"],
                "base_acc": r["base_acc"],
                "ablated_acc": r["ablated_acc"],
                "delta_acc": r["delta_acc"],
                "valid_tokens": r["valid_tokens"],
            }
            for r in all_records
        ]
        summary = {
            "base": base,
            "jobs": all_records,
            "num_jobs": len(all_records),
        }
        write_json_atomic(store.path("ablation", "summary.json"), summary)
        write_csv_atomic(store.path("ablation", "ablation_curve.csv"), csv_rows)
        store.mark_job_complete(stage, "summary", store.path("ablation", "summary.json"), summary)
        store.set_stage_status(stage, "complete")
        top_records = sorted(all_records, key=lambda r: -float(r.get("delta_loss", 0.0)))[:5]
        store.log_event(
            stage,
            "summary",
            message=(
                f"ABLATION SUMMARY jobs={len(all_records)} "
                + " ".join(
                    f"{r['pool']}/{r['strategy']}/k={r['k']} dL={float(r['delta_loss']):.4f}"
                    for r in top_records
                )
            ),
            **summary,
        )
        return summary
    store.log_event(
        stage,
        "host_done",
        message=(
            f"ABLATION HOST DONE host={ctx.host_id}/{ctx.n_hosts} "
            "summary is emitted by primary host=0"
        ),
    )
    return {}
