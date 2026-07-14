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
    analysis_model_module,
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

TRACE_POOLS = ("q", "k", "v", "rst")
TRANSITION_CANDIDATE_STRATEGIES = (
    "top_contribution",
    "top_gate",
    "inactive_random",
    "active_random",
    "matched_active",
)
TRANSITION_CANDIDATE_FIELDS = (
    "candidate_ids",
    "candidate_valid",
    "candidate_execution",
    "candidate_admission",
    "candidate_abs_coefficient",
    "candidate_coefficient",
)
TRACE_FIELDS = (
    "top_idx",
    "top_val",
    "top_admission",
    "top_rho",
    "top_read",
    "top_coefficient",
    "active_count",
    "effective_count",
    "mass",
    "top1_frac",
    "captured_mass",
    "tau",
    "query_norm",
    "update_norm",
)


def _srw_with_topk(state, operator_query, operator_keys, raw_tau,
                   read_vectors, write_vectors, *,
                   model_module,
                   topk: int,
                   execution_kwargs: Dict[str, Any],
                   admission_den_power: float,
                   target_positions=None,
                   candidate_seed: int = 0):
    # This is an observational side-car.  Its coefficient estimates may rank
    # candidates, but causal execution must use the production-core kernel.
    kwargs = dict(execution_kwargs)
    kwargs.pop("admission_den_power", None)
    production_bfloat16 = (
        str(getattr(model_module, "MODEL_VERSION", ""))
        in ("spatial-r1-v4.1.7.1", "spatial-r1-v4.1.7.2"))
    if production_bfloat16:
        operator_query_direction = model_module._forward_unit_direction(
            operator_query.astype(jnp.bfloat16).astype(jnp.float32)).astype(
                jnp.bfloat16)
        operator_key_directions = model_module._forward_unit_direction(
            operator_keys.astype(jnp.bfloat16).astype(jnp.float32)).astype(
                jnp.bfloat16)
        rho = (operator_query_direction @ operator_key_directions.T).astype(
            jnp.float32)
        tau = model_module._tau_from_param(raw_tau).astype(jnp.float32)
        drive_kwargs = dict(kwargs)
        temperature = drive_kwargs.pop("soft_gate_temperature")
        boundary_power = drive_kwargs.pop("soft_gate_boundary_power")
        effective_active_eps = drive_kwargs.pop(
            "soft_gate_effective_active_eps")
        (selection_margin, admission, _, execution_weight,
         active_mask) = model_module._compute_admission_drive(
            rho,
            tau,
            temperature,
            boundary_power=boundary_power,
            effective_active_eps=effective_active_eps,
            **drive_kwargs,
        )
    else:
        selection_margin, admission, _, execution_weight, active_mask = (
            model_module._angular_execution(
                operator_query,
                operator_keys,
                raw_tau,
                None,
                **kwargs,
            ))
        tau = model_module._tau_from_param(raw_tau).astype(jnp.float32)
    admission_mass = admission.sum(axis=-1, keepdims=True)
    if production_bfloat16:
        read_directions = model_module._forward_unit_direction(
            read_vectors.astype(jnp.bfloat16).astype(jnp.float32)).astype(
                jnp.bfloat16)
        write_directions = model_module._forward_unit_direction(
            write_vectors.astype(jnp.bfloat16).astype(jnp.float32)).astype(
                jnp.bfloat16)
        read_activations = state.astype(jnp.bfloat16) @ read_directions.T
    else:
        read_directions = model_module._forward_unit_direction(
            read_vectors.astype(jnp.float32))
        write_directions = model_module._forward_unit_direction(
            write_vectors.astype(jnp.float32))
        read_activations = state.astype(jnp.float32) @ read_directions.T
    coefficient = execution_weight * read_activations
    state_transition = (
        (coefficient.astype(jnp.bfloat16) @ write_directions).astype(jnp.float32)
        if production_bfloat16 else
        coefficient @ write_directions)
    composition_den = getattr(model_module, "_composition_den", None)
    if composition_den is None:
        den = jnp.power(
            jnp.maximum(admission_mass, 1.0),
            jnp.asarray(admission_den_power, dtype=jnp.float32),
        )
    elif hasattr(model_module, "DEFAULT_SRW_COMPOSITION_MODE"):
        den = composition_den(
            admission_mass,
            admission_den_power,
            kwargs.get(
                "srw_composition_mode",
                model_module.DEFAULT_SRW_COMPOSITION_MODE),
        )
    else:
        den = composition_den(admission_mass, admission_den_power)
    state_transition = (
        state_transition.astype(jnp.float32) / den).astype(jnp.float32)
    if production_bfloat16:
        state_transition = state_transition.astype(jnp.bfloat16).astype(
            jnp.float32)
    rho = (selection_margin + tau).astype(jnp.float32)
    coefficient = (
        coefficient / jnp.maximum(den, jnp.float32(1.0e-8))).astype(jnp.float32)

    def target_value(value):
        return value[
            jnp.arange(value.shape[0], dtype=jnp.int32),
            jnp.asarray(target_positions, dtype=jnp.int32),
        ]

    if target_positions is not None:
        execution_stats = target_value(execution_weight.astype(jnp.float32))
        admission_stats = target_value(admission.astype(jnp.float32))
        active_stats = target_value(active_mask)
        rho_stats = target_value(rho)
        read_stats = target_value(read_activations.astype(jnp.float32))
        coefficient_stats = target_value(coefficient)
        tau_stats = target_value(tau)
        query_stats = target_value(operator_query.astype(jnp.float32))
        update_stats = target_value(state_transition)
    else:
        execution_stats = execution_weight.astype(jnp.float32)
        admission_stats = admission.astype(jnp.float32)
        active_stats = active_mask
        rho_stats = rho
        read_stats = read_activations.astype(jnp.float32)
        coefficient_stats = coefficient
        tau_stats = tau
        query_stats = operator_query.astype(jnp.float32)
        update_stats = state_transition

    k = min(int(topk), int(execution_stats.shape[-1]))
    top_val, top_idx = jax.lax.top_k(execution_stats, k)
    mass = execution_stats.sum(axis=-1)
    active_count = active_stats.astype(jnp.int32).sum(axis=-1)
    top1_frac = top_val[..., 0] / jnp.maximum(mass, 1.0e-8)
    captured_mass = top_val.sum(axis=-1) / jnp.maximum(mass, 1.0e-8)
    execution_prob = execution_stats / jnp.maximum(
        mass[..., None], jnp.float32(1.0e-8))
    effective_count = jnp.where(
        mass > 0.0,
        1.0 / jnp.maximum(
            jnp.sum(execution_prob * execution_prob, axis=-1),
            jnp.float32(1.0e-8)),
        0.0,
    )
    stats = {
        "top_idx": top_idx.astype(jnp.int32),
        "top_val": top_val.astype(jnp.float32),
        "top_admission": jnp.take_along_axis(
            admission_stats, top_idx, axis=-1),
        "top_rho": jnp.take_along_axis(rho_stats, top_idx, axis=-1),
        "top_read": jnp.take_along_axis(
            read_stats, top_idx, axis=-1),
        "top_coefficient": jnp.take_along_axis(
            coefficient_stats, top_idx, axis=-1),
        "active_count": active_count.astype(jnp.int32),
        "effective_count": effective_count.astype(jnp.float32),
        "mass": mass.astype(jnp.float32),
        "top1_frac": top1_frac.astype(jnp.float32),
        "captured_mass": captured_mass.astype(jnp.float32),
        "tau": jnp.squeeze(tau_stats, axis=-1),
        "query_norm": jnp.linalg.norm(query_stats, axis=-1),
        "update_norm": jnp.linalg.norm(update_stats, axis=-1),
    }
    if target_positions is not None:
        abs_coefficient = jnp.abs(coefficient_stats)
        selected_id = jnp.argmax(abs_coefficient, axis=-1).astype(jnp.int32)
        top_gate_id = jnp.argmax(execution_stats, axis=-1).astype(jnp.int32)
        operator_ids = jnp.arange(
            execution_stats.shape[-1], dtype=jnp.uint32)
        priority = (
            operator_ids * jnp.uint32(1664525)
            + jnp.uint32(int(candidate_seed) & 0xFFFFFFFF) * jnp.uint32(1013904223)
        )
        selected_mask = operator_ids[None, :] != selected_id[:, None].astype(jnp.uint32)
        active_eligible = (execution_stats > 0.0) & selected_mask
        inactive_eligible = (
            ((admission_stats == 0.0) | (execution_stats == 0.0))
            & selected_mask)

        def prioritized_candidate(eligible):
            valid = jnp.any(eligible, axis=-1)
            chosen = jnp.argmin(
                jnp.where(eligible, priority[None, :], jnp.uint32(0xFFFFFFFF)),
                axis=-1,
            ).astype(jnp.int32)
            return jnp.where(valid, chosen, jnp.int32(-1)), valid

        inactive_id, inactive_valid = prioritized_candidate(inactive_eligible)
        active_id, active_valid = prioritized_candidate(active_eligible)
        selected_abs = jnp.take_along_axis(
            abs_coefficient, selected_id[:, None], axis=-1)
        matched_distance = jnp.abs(abs_coefficient - selected_abs)
        matched_valid = jnp.any(active_eligible, axis=-1)
        matched_id = jnp.argmin(
            jnp.where(active_eligible, matched_distance, jnp.float32(jnp.inf)),
            axis=-1,
        ).astype(jnp.int32)
        matched_id = jnp.where(matched_valid, matched_id, jnp.int32(-1))
        candidate_ids = jnp.stack(
            (selected_id, top_gate_id, inactive_id, active_id, matched_id), axis=-1)
        candidate_valid = jnp.stack(
            (
                jnp.ones_like(inactive_valid),
                jnp.ones_like(inactive_valid),
                inactive_valid,
                active_valid,
                matched_valid,
            ),
            axis=-1,
        )
        safe_ids = jnp.maximum(candidate_ids, 0)
        stats.update({
            "candidate_ids": candidate_ids,
            "candidate_valid": candidate_valid,
            "candidate_execution": jnp.take_along_axis(
                execution_stats, safe_ids, axis=-1),
            "candidate_admission": jnp.take_along_axis(
                admission_stats, safe_ids, axis=-1),
            "candidate_abs_coefficient": jnp.take_along_axis(
                abs_coefficient, safe_ids, axis=-1),
            "candidate_coefficient": jnp.take_along_axis(
                coefficient_stats, safe_ids, axis=-1),
        })
    return state_transition, stats


def topk_trace_forward(params, model_cfg: Dict[str, Any], input_ids, *,
                       topk: int = 8,
                       topk_qk: Optional[int] = None,
                       topk_v: Optional[int] = None,
                       topk_rst: Optional[int] = None,
                       execution_prune_eps: Optional[float] = None,
                       target_positions=None,
                       candidate_seed: int = 0,
                       production_srw_fns=None):
    """Return sparse side-car traces while advancing via production kernels.

    For v4171, outputs reconstructed by :func:`_srw_with_topk` are
    observation-only and are never used as model state.
    """
    model_module = analysis_model_module(model_cfg)
    params = model_module._squeeze_params(params)
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    bsz, seq_len = input_ids.shape
    if target_positions is not None:
        target_positions = jnp.asarray(target_positions, dtype=jnp.int32)
        if target_positions.shape != (bsz,):
            raise ValueError(
                f"target_positions must have shape ({bsz},), got {target_positions.shape}")

    def target_value(value):
        if target_positions is None:
            return None
        return value[jnp.arange(bsz, dtype=jnp.int32), target_positions]
    d_model = int(model_cfg["d_model"])
    n_layers = int(model_cfg["n_layers"])
    n_heads = int(model_cfg["n_heads"])
    d_head = d_model // n_heads
    execution_kwargs = model_module._angular_execution_kwargs_from_model_cfg(model_cfg)
    if execution_prune_eps is not None:
        execution_kwargs["execution_prune_eps"] = float(execution_prune_eps)
    admission_den_power = float(execution_kwargs.get("admission_den_power", 1.0))

    def execution_for(temperature_key: str) -> Dict[str, Any]:
        out = dict(execution_kwargs)
        value = model_cfg.get(temperature_key)
        if value is not None:
            out["soft_gate_temperature"] = float(value)
        return out

    execution_qk = execution_for("soft_gate_T_qk")
    execution_v = execution_for("soft_gate_T_v")
    execution_rst = execution_for("soft_gate_T_rst")
    topk_qk = int(topk if topk_qk is None else topk_qk)
    topk_v = int(topk if topk_v is None else topk_v)
    topk_rst = int(topk if topk_rst is None else topk_rst)
    is_v417x = str(model_cfg.get("model_version", "")) in (
        "spatial-r1-v4.1.7.1",
        "spatial-r1-v4.1.7.2",
    )
    if is_v417x:
        required = {
            "attn_qk_paired_minimal", "attn_v_single_minimal",
            "rst_single_minimal",
        }
        if not isinstance(production_srw_fns, dict) or required - set(production_srw_fns):
            raise ValueError(
                "v4171 production-only trace requires canonical minimal kernels: "
                + ",".join(sorted(required)))

    pool = model_module._pool_params_with_operator_keys(params["neuron_pool"])
    router = params["router"]
    qk_scale, v_scale, rst_scale = model_module._effective_pool_output_scales(
        pool,
        d_model,
        n_layers,
    )
    positions = jnp.arange(seq_len)[jnp.newaxis, :]
    residual_state = (
        params["token_emb"]["embedding"][input_ids]
        + params["pos_emb"]["embedding"][positions]
    )

    traces = {
        f"{pool_name}_{field}": []
        for pool_name in TRACE_POOLS
        for field in TRACE_FIELDS
    }
    attn_out_norm = []
    rst_out_norm = []
    residual_before_norm = []
    residual_after_attn_norm = []
    residual_after_rst_norm = []
    target_vectors = {
        name: []
        for name in (
            "residual_before_router",
            "router_input_attn",
            "query_q",
            "query_k",
            "query_v",
            "srw_feature_q",
            "srw_feature_k",
            "srw_feature_v",
            "delta_attention",
            "residual_after_attention",
            "router_input_rst",
            "query_rst",
            "delta_rst",
            "residual_after_update",
        )
    } if target_positions is not None else None

    for layer_idx in range(n_layers):
        bp = params[f"block_{layer_idx}"]
        residual_before_norm.append(
            jnp.linalg.norm(residual_state.astype(jnp.float32), axis=-1))
        if target_vectors is not None:
            target_vectors["residual_before_router"].append(
                target_value(residual_state.astype(jnp.float32)))
        normed = model_module._layer_norm(
            residual_state, bp["norm1"]["scale"], bp["norm1"]["bias"])
        attn_operator_queries = (
            normed @ router["proj_attn"]["kernel"]
            + router["proj_attn"]["bias"])
        q_operator_query, k_operator_query, v_operator_query = jnp.split(
            attn_operator_queries, 3, axis=-1)
        if target_vectors is not None:
            target_vectors["router_input_attn"].append(
                target_value(normed.astype(jnp.float32)))
        query_adapter = getattr(
            model_module, "_read_write_attn_operator_queries", None)
        if query_adapter is not None:
            (q_operator_query, k_operator_query, v_operator_query) = (
                query_adapter(
                    router, normed,
                    q_operator_query, k_operator_query, v_operator_query))
        if target_vectors is not None:
            target_vectors["query_q"].append(
                target_value(q_operator_query.astype(jnp.float32)))
            target_vectors["query_k"].append(
                target_value(k_operator_query.astype(jnp.float32)))
            target_vectors["query_v"].append(
                target_value(v_operator_query.astype(jnp.float32)))
        tau_all = normed @ router["raw_tau_attn"]["kernel"] + router["raw_tau_attn"]["bias"]
        q_state_transition, q_stats = _srw_with_topk(
            normed,
            q_operator_query,
            pool["attn_qk_op_key"],
            tau_all[:, :, 0:1],
            pool["attn_qk_read"],
            pool["attn_qk_write"],
            model_module=model_module,
            topk=topk_qk,
            execution_kwargs=execution_qk,
            admission_den_power=admission_den_power,
            target_positions=target_positions,
            candidate_seed=candidate_seed + layer_idx * 17,
        )
        k_state_transition, k_stats = _srw_with_topk(
            normed,
            k_operator_query,
            pool["attn_qk_op_key"],
            tau_all[:, :, 1:2],
            pool["attn_qk_read"],
            pool["attn_qk_write"],
            model_module=model_module,
            topk=topk_qk,
            execution_kwargs=execution_qk,
            admission_den_power=admission_den_power,
            target_positions=target_positions,
            candidate_seed=candidate_seed + layer_idx * 17 + 1,
        )
        v_state_transition, v_stats = _srw_with_topk(
            normed,
            v_operator_query,
            pool["attn_v_op_key"],
            tau_all[:, :, 2:3],
            pool["attn_v_read"],
            pool["attn_v_write"],
            model_module=model_module,
            topk=topk_v,
            execution_kwargs=execution_v,
            admission_den_power=admission_den_power,
            target_positions=target_positions,
            candidate_seed=candidate_seed + layer_idx * 17 + 2,
        )
        if is_v417x:
            qk_operator_queries = jnp.stack(
                (q_operator_query, k_operator_query), axis=2)
            tau_qk = jnp.stack(
                (tau_all[:, :, 0:1], tau_all[:, :, 1:2]), axis=2)
            qk_state_transitions = production_srw_fns[
                "attn_qk_paired_minimal"](
                normed, qk_operator_queries, pool["attn_qk_op_key"], tau_qk,
                pool["attn_qk_read"], pool["attn_qk_write"],
                execution_qk["soft_gate_temperature"],
                model_cfg.get("soft_gate_t_final", execution_qk["soft_gate_temperature"]),
                execution_qk["soft_gate_boundary_power"],
                model_cfg.get("soft_gate_boundary_power_final",
                              execution_qk["soft_gate_boundary_power"]),
                execution_qk.get("execution_prune_eps", 0.0),
            )[0]
            q_state_transition = qk_state_transitions[:, :, 0, :]
            k_state_transition = qk_state_transitions[:, :, 1, :]
            v_state_transition = production_srw_fns[
                "attn_v_single_minimal"](
                normed, v_operator_query, pool["attn_v_op_key"],
                tau_all[:, :, 2:3],
                pool["attn_v_read"], pool["attn_v_write"],
                execution_v["soft_gate_temperature"],
                model_cfg.get("soft_gate_t_final", execution_v["soft_gate_temperature"]),
                execution_v["soft_gate_boundary_power"],
                model_cfg.get("soft_gate_boundary_power_final",
                              execution_v["soft_gate_boundary_power"]),
                execution_v.get("execution_prune_eps", 0.0),
            )[0]
        attention_q = q_state_transition * qk_scale
        attention_k = k_state_transition * qk_scale
        attention_v = v_state_transition * v_scale
        for stats, scale in (
            (q_stats, qk_scale),
            (k_stats, qk_scale),
            (v_stats, v_scale),
        ):
            stats["top_coefficient"] = stats["top_coefficient"] * scale
            if "candidate_abs_coefficient" in stats:
                stats["candidate_abs_coefficient"] = (
                    stats["candidate_abs_coefficient"] * jnp.abs(scale))
            if "candidate_coefficient" in stats:
                stats["candidate_coefficient"] = (
                    stats["candidate_coefficient"] * scale)
            stats["update_norm"] = stats["update_norm"] * jnp.abs(scale)
        if target_vectors is not None:
            target_vectors["srw_feature_q"].append(target_value(attention_q))
            target_vectors["srw_feature_k"].append(target_value(attention_k))
            target_vectors["srw_feature_v"].append(target_value(attention_v))
        attention_q_heads = attention_q.reshape(
            bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_k_heads = attention_k.reshape(
            bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        attention_v_heads = attention_v.reshape(
            bsz, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = jnp.einsum(
            "bhsd,bhtd->bhst", attention_q_heads,
            attention_k_heads) / jnp.sqrt(jnp.float32(d_head))
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum(
            "bhst,bhtd->bhsd", attn_w, attention_v_heads)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
        attn_out = attn_out @ bp["attn"]["expand_O"]["kernel"]
        attn_out_norm.append(jnp.linalg.norm(attn_out, axis=-1).mean())
        if target_vectors is not None:
            target_vectors["delta_attention"].append(target_value(attn_out))
        residual_state = residual_state + attn_out
        if target_vectors is not None:
            target_vectors["residual_after_attention"].append(
                target_value(residual_state.astype(jnp.float32)))
        residual_after_attn_norm.append(
            jnp.linalg.norm(residual_state.astype(jnp.float32), axis=-1))

        normed = model_module._layer_norm(
            residual_state, bp["norm2"]["scale"], bp["norm2"]["bias"])
        rst_operator_query = (
            normed @ router["proj_rst"]["kernel"]
            + router["proj_rst"]["bias"])
        if target_vectors is not None:
            target_vectors["router_input_rst"].append(
                target_value(normed.astype(jnp.float32)))
        rst_query_adapter = getattr(
            model_module, "_read_write_rst_operator_query", None)
        if rst_query_adapter is not None:
            rst_operator_query = rst_query_adapter(
                router, normed, rst_operator_query)
        if target_vectors is not None:
            target_vectors["query_rst"].append(
                target_value(rst_operator_query.astype(jnp.float32)))
        tau_rst = normed @ router["raw_tau_rst"]["kernel"] + router["raw_tau_rst"]["bias"]
        rst_state_transition, rst_stats = _srw_with_topk(
            normed,
            rst_operator_query,
            pool["rst_op_key"],
            tau_rst,
            pool["rst_read"],
            pool["rst_write"],
            model_module=model_module,
            topk=topk_rst,
            execution_kwargs=execution_rst,
            admission_den_power=admission_den_power,
            target_positions=target_positions,
            candidate_seed=candidate_seed + layer_idx * 17 + 3,
        )
        if is_v417x:
            rst_state_transition = production_srw_fns[
                "rst_single_minimal"](
                normed, rst_operator_query, pool["rst_op_key"], tau_rst,
                pool["rst_read"], pool["rst_write"],
                execution_rst["soft_gate_temperature"],
                model_cfg.get("soft_gate_t_final", execution_rst["soft_gate_temperature"]),
                execution_rst["soft_gate_boundary_power"],
                model_cfg.get("soft_gate_boundary_power_final",
                              execution_rst["soft_gate_boundary_power"]),
                execution_rst.get("execution_prune_eps", 0.0),
            )[0]
        rst_state_transition = rst_state_transition * rst_scale
        rst_stats["top_coefficient"] = rst_stats["top_coefficient"] * rst_scale
        if "candidate_abs_coefficient" in rst_stats:
            rst_stats["candidate_abs_coefficient"] = (
                rst_stats["candidate_abs_coefficient"] * jnp.abs(rst_scale))
        if "candidate_coefficient" in rst_stats:
            rst_stats["candidate_coefficient"] = (
                rst_stats["candidate_coefficient"] * rst_scale)
        rst_stats["update_norm"] = rst_stats["update_norm"] * jnp.abs(rst_scale)
        rst_out_norm.append(
            jnp.linalg.norm(rst_state_transition, axis=-1).mean())
        if target_vectors is not None:
            target_vectors["delta_rst"].append(
                target_value(rst_state_transition))
        residual_state = residual_state + rst_state_transition
        if target_vectors is not None:
            target_vectors["residual_after_update"].append(
                target_value(residual_state.astype(jnp.float32)))
        residual_after_rst_norm.append(
            jnp.linalg.norm(residual_state.astype(jnp.float32), axis=-1))

        for prefix, stats in (
            ("q", q_stats),
            ("k", k_stats),
            ("v", v_stats),
            ("rst", rst_stats),
        ):
            for field, value in stats.items():
                traces.setdefault(f"{prefix}_{field}", []).append(value)

    out = {key: jnp.stack(values, axis=0) for key, values in traces.items()}
    out["attn_out_norm"] = jnp.stack(attn_out_norm)
    out["rst_out_norm"] = jnp.stack(rst_out_norm)
    if target_positions is None:
        out["residual_before_norm"] = jnp.stack(residual_before_norm)
        out["residual_after_attn_norm"] = jnp.stack(residual_after_attn_norm)
        out["residual_after_rst_norm"] = jnp.stack(residual_after_rst_norm)
    else:
        gather = lambda values: jnp.stack([
            value[jnp.arange(bsz, dtype=jnp.int32), target_positions]
            for value in values
        ], axis=0)
        out["residual_before_norm"] = gather(residual_before_norm)
        out["residual_after_attn_norm"] = gather(residual_after_attn_norm)
        out["residual_after_rst_norm"] = gather(residual_after_rst_norm)
    if target_vectors is not None:
        for key, values in target_vectors.items():
            out[key] = jnp.stack(values, axis=0)
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
    trace_fn = jax.jit(lambda p, x: topk_trace_forward(
        p, ctx.model_cfg, x, topk=topk,
        production_srw_fns=ctx.sharded_fns))
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
