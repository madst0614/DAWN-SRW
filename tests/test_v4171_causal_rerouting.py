"""Focused tiny-model regression coverage for v4171 causal rerouting."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from analysis.dawn_v4171_transition import (
    TRACE_POOLS,
    _adaptive_rerouting_capture,
    _canonical_causal_trace_forward,
    _capture_reliability_summary,
    _rerouting_layer_rows,
    _rerouting_trajectory_metrics,
    _tree_machine_exact,
    classify_path_dependence_judgment,
    classify_rerouting_trajectory,
)
from models.dawn_srw_v4171 import (
    DAWN_SRW_V4171,
    make_sharded_srw_minimal,
    make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_suppression_minimal,
)


def _tiny_context():
    device = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    mesh = Mesh(device, ("data", "model"))
    single = make_sharded_srw_minimal(mesh, max_chunk_size=4)
    paired = make_sharded_srw_paired_minimal(mesh, max_chunk_size=4)
    single_suppression = make_sharded_srw_suppression_minimal(
        mesh, max_chunk_size=4)
    paired_suppression = make_sharded_srw_paired_suppression_minimal(
        mesh, max_chunk_size=4)
    sharded_fns = {
        "single": single,
        "paired": paired,
        "attn_v_single_minimal": single,
        "rst_single_minimal": single,
        "attn_qk_paired_minimal": paired,
        "attn_v_single_suppression_minimal": single_suppression,
        "rst_single_suppression_minimal": single_suppression,
        "attn_qk_paired_suppression_minimal": paired_suppression,
    }
    model = DAWN_SRW_V4171(
        vocab_size=16, d_model=4, d_route=4, n_layers=2, n_heads=1,
        max_seq_len=3, dropout_rate=0.0, router_dropout=0.0,
        n_qk=8, n_v=8, n_rst=8,
        tau_init_attn_qk=-0.5, tau_init_attn_v=-0.5, tau_init_rst=-0.5,
    )
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(101),
         "dropout": jax.random.PRNGKey(102)},
        input_ids, deterministic=True)
    model_cfg = {
        "model_version": "spatial-r1-v4.1.7.1",
        "d_model": 4,
        "n_layers": 2,
        "n_qk": 8,
        "n_v": 8,
        "n_rst": 8,
        "soft_gate_temperature": 0.07,
        "soft_gate_t_final": 0.07,
        "soft_gate_T_qk": 0.07,
        "soft_gate_T_v": 0.07,
        "soft_gate_T_rst": 0.07,
        "soft_gate_boundary_power": 2.0,
        "soft_gate_boundary_power_final": 4.0,
        "soft_gate_effective_active_eps": 1.0e-6,
        "admission_den_power": float(model.admission_den_power),
        "srw_composition_mode": str(model.srw_composition_mode),
        "heat_kernel_beta": float(model.heat_kernel_beta),
        "execution_prune_eps": 0.0,
    }
    args = SimpleNamespace(
        transition_topk_qk=8,
        transition_topk_v=8,
        transition_topk_rst=8,
        transition_capture_threshold=0.95,
        transition_adaptive_capture=False,
        transition_adaptive_final_topk_v=8,
        transition_adaptive_final_topk_rst=8,
    )
    return SimpleNamespace(
        model=model,
        params=variables["params"],
        model_cfg=model_cfg,
        sharded_fns=sharded_fns,
        args=args,
    ), input_ids


def test_inactive_is_exact_and_active_changes_target_only_rerouting() -> None:
    ctx, input_ids = _tiny_context()
    target_positions = jnp.asarray([1], dtype=jnp.int32)
    forward = _canonical_causal_trace_forward(ctx)
    canonical_function_identity = id(forward)
    baseline = jax.device_get(forward(
        ctx.params, input_ids, target_positions,
        jnp.int32(0), jnp.int32(2), jnp.asarray([0], dtype=jnp.int32),
        jnp.bool_(False)))

    capture_fns = {}
    baseline_capture = _adaptive_rerouting_capture(
        ctx, baseline[3], 0, capture_fns)
    v_ids = np.asarray(
        baseline_capture["trace"]["v"]["top_operator_ids"])[0, 0]
    v_weights = np.asarray(
        baseline_capture["trace"]["v"]["top_execution_weights"])[0, 0]
    v_local = np.asarray(
        baseline_capture["trace"]["v"]["top_local_contributions"])[0, 0]
    inactive_indices = np.flatnonzero(v_weights == 0.0)
    active_indices = np.flatnonzero(
        (v_weights > 0.0) & (np.abs(v_local) > 0.0))
    assert inactive_indices.size > 0
    assert active_indices.size > 0
    inactive_id = int(v_ids[inactive_indices[0]])
    active_id = int(v_ids[active_indices[np.argmax(
        np.abs(v_local[active_indices]))]])

    inactive = jax.device_get(forward(
        ctx.params, input_ids, target_positions,
        jnp.int32(0), jnp.int32(2),
        jnp.asarray([inactive_id], dtype=jnp.int32), jnp.bool_(True)))
    inactive_capture = _adaptive_rerouting_capture(
        ctx, inactive[3], 0, capture_fns,
        suppression_route="v", suppression_operator_id=inactive_id)
    assert id(forward) == canonical_function_identity
    assert _tree_machine_exact(baseline, inactive)
    for route in TRACE_POOLS:
        assert np.array_equal(
            baseline_capture["trace"][route]["top_operator_ids"],
            inactive_capture["trace"][route]["top_operator_ids"])
        assert np.array_equal(
            baseline_capture["trace"][route]["top_execution_weights"],
            inactive_capture["trace"][route]["top_execution_weights"])
        assert np.array_equal(
            baseline_capture["trace"][route]["top_local_contributions"],
            inactive_capture["trace"][route]["top_local_contributions"])

    active = jax.device_get(forward(
        ctx.params, input_ids, target_positions,
        jnp.int32(0), jnp.int32(2),
        jnp.asarray([active_id], dtype=jnp.int32), jnp.bool_(True)))
    active_capture = _adaptive_rerouting_capture(
        ctx, active[3], 0, capture_fns,
        suppression_route="v", suppression_operator_id=active_id)
    changed_trace_keys = [
        key for key in baseline[3]
        if not np.array_equal(baseline[3][key], active[3][key])]
    assert changed_trace_keys
    assert (
        not np.array_equal(baseline[2], active[2])
        or any(key.startswith(("query_", "route_transition_"))
               for key in changed_trace_keys))
    assert active_capture["suppression"]["enabled"] is True
    assert active_capture["suppression"]["execution_weights_preserved"] is True
    layer_rows = _rerouting_layer_rows(
        baseline[3], active[3], baseline_capture, active_capture,
        route="v", target_layer=0)
    assert len(layer_rows) == 2
    assert all(row["route"] == "v" for row in layer_rows)
    assert {
        "baseline_route_input_residual", "intervention_route_input_residual",
        "baseline_post_layer_residual", "intervention_post_layer_residual",
        "baseline_route_query", "intervention_route_query",
        "baseline_top_operator_ids", "intervention_top_operator_ids",
        "baseline_top_execution_weights",
        "intervention_top_execution_weights",
        "baseline_top_local_contributions",
        "intervention_top_local_contributions",
        "baseline_route_transition_delta",
        "intervention_route_transition_delta",
    }.issubset(layer_rows[0])
    trajectory = _rerouting_trajectory_metrics(layer_rows, target_layer=0)
    assert trajectory["target_layer_immediate_routing_similarity"] is not None
    assert trajectory["cumulative_routing_divergence_auc"] is not None

    sparse_keys = {
        "top_operator_ids", "top_execution_weights",
        "top_local_contributions", "top_admission_weights",
        "captured_mass", "execution_mass",
    }
    for route in TRACE_POOLS:
        assert set(active_capture["trace"][route]) == sparse_keys
        assert all("full" not in key for key in active_capture["trace"][route])
        assert active_capture["trace"][route][
            "top_operator_ids"].shape[-1] <= 8
    assert all(value.shape[1] == 1 for value in active[3].values())


def test_capture_reliability_reports_retries_recovery_and_exclusions() -> None:
    recovered = {
        route: {
            "before": np.asarray([0.80, 0.96]),
            "after": np.asarray([0.97, 0.96]),
            "retry_count": 1,
        }
        for route in TRACE_POOLS
    }
    remaining = {
        route: {
            "before": np.asarray([0.70]),
            "after": np.asarray([0.90]),
            "retry_count": 1,
        }
        for route in TRACE_POOLS
    }
    summary = _capture_reliability_summary(
        [{"capture": recovered}, {"capture": remaining}], threshold=0.95)
    assert summary["status"] == "partial"
    assert summary["total_observations"] == 12
    assert summary["qualified_observations"] == 8
    assert summary["excluded_observations"] == 4
    for route in TRACE_POOLS:
        row = summary["pools"][route]
        assert row["adaptive_retry_count"] == 2
        assert row["adaptive_recovered_count"] == 1
        assert row["remaining_low_capture_count"] == 1
        assert row["captured_mass_min"] == 0.90


def test_rerouting_trajectory_separates_divergence_and_reconvergence() -> None:
    def rows(similarities):
        return [
            {"layer": layer, "weighted_jaccard": similarity}
            for layer, similarity in enumerate(similarities)
        ]

    no_divergence = classify_rerouting_trajectory(
        rows([0.98, 0.97, 0.98]), 0.90, 0.90)
    assert no_divergence["routing_path_classification"] == (
        "no_meaningful_divergence")
    assert no_divergence["meaningful_divergence"] is False
    assert no_divergence["first_threshold_return_layer"] is None

    reconverged = classify_rerouting_trajectory(
        rows([0.70, 0.60, 0.92]), 0.80, 0.90)
    assert reconverged["routing_path_classification"] == (
        "diverged_then_reconverged")
    assert reconverged["first_threshold_return_layer"] == 2
    assert reconverged["final_qualified_layer"] == 2
    assert reconverged["layers_after_minimum"] == 1

    not_reconverged = classify_rerouting_trajectory(
        rows([0.70, 0.60, 0.72]), 0.80, 0.90)
    assert not_reconverged["routing_path_classification"] == (
        "diverged_not_reconverged")

    transient_return = classify_rerouting_trajectory(
        rows([0.70, 0.60, 0.92, 0.80]), 0.80, 0.90)
    assert transient_return["first_threshold_return_layer"] == 2
    assert transient_return["final_qualified_layer"] == 3
    assert transient_return["routing_path_classification"] == (
        "diverged_not_reconverged")

    final_minimum = classify_rerouting_trajectory(
        rows([0.95, 0.70]), 0.80, None)
    assert final_minimum["routing_path_classification"] == (
        "diverged_not_reconverged")
    assert final_minimum["layers_after_minimum"] == 0

    insufficient = classify_rerouting_trajectory(
        rows([0.70, 0.60, 0.92]), None, None)
    assert insufficient["routing_path_classification"] == "indeterminate"


def test_path_dependence_judgment_has_three_evidence_tiers() -> None:
    supported = classify_path_dependence_judgment(
        inactive_exact_noop=True,
        important_control_classification="strong",
        predictive_classification="strong",
        meaningful_divergence_count=4,
        nonreconvergence_fraction_among_diverged=0.75,
    )
    assert supported == {
        "status": "supported", "supported": True, "suggestive": False}

    suggestive = classify_path_dependence_judgment(
        inactive_exact_noop=True,
        important_control_classification="directional",
        predictive_classification="directional",
        meaningful_divergence_count=2,
        nonreconvergence_fraction_among_diverged=0.5,
    )
    assert suggestive == {
        "status": "suggestive", "supported": False, "suggestive": True}

    not_supported = classify_path_dependence_judgment(
        inactive_exact_noop=False,
        important_control_classification="strong",
        predictive_classification="strong",
        meaningful_divergence_count=8,
        nonreconvergence_fraction_among_diverged=1.0,
    )
    assert not_supported == {
        "status": "not_supported", "supported": False,
        "suggestive": False}
