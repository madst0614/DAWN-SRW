"""Lightweight synthetic tests for v4171 operator-family analysis helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from analysis.dawn_train_analysis_items import (
    TRAIN_ANALYSIS_PRESETS,
    TrainAnalysisFormatters,
    format_train_analysis_item,
)
from analysis.dawn_v4171_transition import (
    _classify_important_intervention_control_evidence,
    _classify_predictive_relation_evidence,
    _classify_recovery_rows,
    _paired_dose_response,
    _paired_group_comparison,
    _recovery_group_summary,
    _resolve_group_member_singles,
    classify_function_address_pairs,
    classify_paired_directional_evidence,
    classify_predictive_correlation_evidence,
    compute_causal_output_metrics,
    compute_causal_recovery_metrics,
    compute_group_additivity_metrics,
    functional_percolation_summary,
    group_operator_membership_mask,
    mutual_neighbor_families,
    pad_group_operator_ids,
    pairwise_win_rate,
    reciprocal_neighbor_edges,
    rw_functional_similarity,
    run_causal_recovery_trace,
    single_effect_key,
    spearman_correlation,
)
from models.dawn_srw_v4171 import (
    DAWN_SRW_V4171,
    analysis_operator_membership,
    make_sharded_srw_minimal,
    make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_suppression_minimal,
)
from scripts.analyze_dawn_srw_v4166 import (
    _format_train_analysis,
    _operator_family_decision,
)


def test_existing_preset_is_unchanged_and_extended_preset_is_opt_in() -> None:
    original = TRAIN_ANALYSIS_PRESETS["v4171_self_organization"]
    assert original == (
        "global_router_audit", "trajectory_trace", "context_divergence",
        "state_transition_decoupling", "causal_intervention")
    extended = TRAIN_ANALYSIS_PRESETS["v4171_operator_family"]
    assert extended[:len(original)] == original
    assert extended[-5:] == (
        "causal_rerouting_trace", "causal_recovery_trace",
        "operator_functional_graph",
        "group_causal_intervention", "causal_ranking_calibration")


def test_causal_metric_names_separate_sequence_and_shifted_target() -> None:
    baseline = np.zeros((1, 3, 4), dtype=np.float64)
    intervention = baseline.copy()
    intervention[0, 0, 1] = -1.0
    intervention[0, 1, 2] = 2.0
    metrics = compute_causal_output_metrics(
        baseline, intervention, [0, 1, 2], 3, target_position=1)
    assert metrics["sequence_ce_delta"] == pytest.approx(
        -metrics["sequence_behavior_delta"])
    assert metrics["sequence_behavior_drop"] == pytest.approx(
        -metrics["sequence_behavior_delta"])
    expected_target = 2.0 - np.log(np.exp(2.0) + 3.0) - (-np.log(4.0))
    assert metrics["target_gold_position"] == 2
    assert metrics["target_gold_token_id"] == 2
    assert metrics["target_next_token_logprob_delta"] == pytest.approx(
        expected_target)
    assert metrics["target_logprob_delta_legacy"] == pytest.approx(
        metrics["sequence_behavior_delta"])
    assert metrics["legacy_target_logprob_metric"] == "sequence_behavior_delta"

    last = compute_causal_output_metrics(
        baseline, intervention, [0, 1, 2], 3, target_position=2)
    assert last["target_next_token_logprob_delta"] is None
    assert last["target_distribution_kl"] is None
    assert last["target_gold_token_id"] is None


def test_single_effect_keys_are_layer_aware_and_size_one_is_exact() -> None:
    layer_zero = single_effect_key("prompt", "v", 0, 7)
    layer_one = single_effect_key("prompt", "v", 1, 7)
    cache = {layer_zero: 0.25, layer_one: -0.5}
    assert len(cache) == 2
    assert cache[layer_zero] != cache[layer_one]

    exact = compute_group_additivity_metrics(
        0.25, [0.25], size_one=True)
    assert exact["magnitude_synergy"] == 0.0
    assert exact["signed_additivity_residual"] == 0.0
    assert exact["group_effect_over_sum_single_abs_effect"] == 1.0
    assert exact["group_effect_over_max_single_abs_effect"] == 1.0
    with pytest.raises(ValueError, match="every member"):
        compute_group_additivity_metrics(0.25, [])

    multi = compute_group_additivity_metrics(0.7, [0.2, -0.1, 0.3, 0.1])
    assert multi["sum_single_abs_effect"] == pytest.approx(0.7)
    assert multi["sum_single_signed_effect"] == pytest.approx(0.5)
    assert multi["magnitude_synergy"] == pytest.approx(0.0)
    assert multi["signed_additivity_residual"] == pytest.approx(0.2)

    member_cache = {
        single_effect_key("prompt", "v", 3, operator_id): {
            "sequence_behavior_delta": float(operator_id) / 100.0}
        for operator_id in range(8)
    }
    for size in (2, 4, 8):
        lookup = _resolve_group_member_singles(
            member_cache, prompt_id="prompt", route="v", layer=3,
            operator_ids=range(size))
        assert lookup["all_member_singles_available"] is True
        assert lookup["missing_member_single_count"] == 0
        assert len(lookup["rows"]) == size
    del member_cache[single_effect_key("prompt", "v", 3, 7)]
    missing = _resolve_group_member_singles(
        member_cache, prompt_id="prompt", route="v", layer=3,
        operator_ids=range(8))
    assert missing["all_member_singles_available"] is False
    assert missing["missing_member_single_count"] == 1


def test_rw_functional_similarity_has_rank1_sign_invariance() -> None:
    read = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    write = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    same = rw_functional_similarity(read, write, read, write)
    flipped = rw_functional_similarity(read, write, -read, -write)
    one_sided = rw_functional_similarity(read, write, -read, write)
    np.testing.assert_allclose(same, np.ones(2))
    np.testing.assert_allclose(flipped, np.ones(2))
    np.testing.assert_allclose(one_sided, -np.ones(2))


def test_family_quantiles_and_mutual_neighbor_grouping() -> None:
    classified = classify_function_address_pairs(
        [0.01, 0.20, 0.70, 0.99], [0.99, 0.20, 0.70, 0.01])
    thresholds = classified["thresholds"]
    assert thresholds["functional_high"] == np.quantile(
        [0.01, 0.20, 0.70, 0.99], 0.90)
    assert thresholds["address_low"] == np.quantile(
        [0.99, 0.20, 0.70, 0.01], 0.50)
    assert classified["labels"][0] == "function_low_address_high"
    assert classified["labels"][-1] == "function_high_address_low"

    families = mutual_neighbor_families(
        {0: [1], 1: [0, 2], 2: [3], 3: [2]},
        qualified_edges=[(1, 2)],
    )
    assert families == [[0, 1, 2, 3]]


def test_percolation_component_is_separate_from_bounded_local_neighbors() -> None:
    neighbors = {
        0: [{"operator_id": 1, "similarity": 0.9}],
        1: [
            {"operator_id": 0, "similarity": 0.9},
            {"operator_id": 2, "similarity": 0.9},
        ],
        2: [
            {"operator_id": 1, "similarity": 0.9},
            {"operator_id": 3, "similarity": 0.9},
        ],
        3: [{"operator_id": 2, "similarity": 0.9}],
    }
    edges = reciprocal_neighbor_edges(neighbors, minimum_similarity=0.8)
    assert edges == [(0, 1), (1, 2), (2, 3)]
    diagnostic = functional_percolation_summary(range(4), edges)
    assert diagnostic["components"] == [[0, 1, 2, 3]]
    assert diagnostic["largest_component_fraction"] == 1.0
    assert diagnostic["percolated"] is True
    seed_one_local = [1] + sorted(
        right if left == 1 else left
        for left, right in edges if 1 in (left, right))
    assert seed_one_local == [1, 0, 2]
    assert seed_one_local != diagnostic["components"][0]


def test_causal_recovery_metrics_use_route_specific_immediate_state() -> None:
    baseline = {
        "post_attention_residual": np.ones((3, 2)),
        "post_layer_residual": np.ones((3, 2)),
        "attention_update": np.zeros((3, 2)),
        "rst_update": np.zeros((3, 2)),
    }
    intervention = {key: value.copy() for key, value in baseline.items()}
    intervention["post_attention_residual"][:, 0] += [1.0, 0.5, 0.25]
    intervention["post_layer_residual"][:, 0] += [0.8, 0.4, 0.2]
    base_logits = np.zeros((1, 3, 4))
    after_logits = base_logits.copy()
    after_logits[0, 1, 0] = 0.5
    metrics = compute_causal_recovery_metrics(
        baseline, intervention, route="q", target_layer=0,
        baseline_final_residual=np.ones(2),
        intervention_final_residual=np.asarray([1.25, 1.0]),
        baseline_logits=base_logits, intervention_logits=after_logits,
        target_position=1,
    )
    assert metrics["immediate_state"] == "post_attention_residual"
    assert metrics["immediate_delta_norm"] == 1.0
    assert metrics["final_delta_norm"] == 0.25
    assert metrics["recovery_ratio_final"] == 0.25
    assert metrics["minimum_remaining_layer"] == 2
    assert metrics["first_half_recovery_layer"] == 1
    assert metrics["final_logit_delta"] == 0.5


def test_relative_recovery_uses_baseline_scale_and_semantic_boundary() -> None:
    baseline = {
        "post_attention_residual": np.asarray([[1.0, 0.0], [2.0, 0.0]]),
        "post_layer_residual": np.asarray([[1.0, 0.0], [100.0, 0.0]]),
    }
    intervention = {
        "post_attention_residual": np.asarray([[2.0, 0.0], [2.0, 0.0]]),
        "post_layer_residual": np.asarray([[2.0, 0.0], [110.0, 0.0]]),
    }
    metrics = compute_causal_recovery_metrics(
        baseline, intervention, route="q", target_layer=0,
        baseline_final_residual=np.asarray([100.0, 0.0]),
        intervention_final_residual=np.asarray([110.0, 0.0]),
        baseline_logits=np.zeros((1, 2, 3)),
        intervention_logits=np.zeros((1, 2, 3)), target_position=0)
    assert metrics["absolute_delta_ratio"] == pytest.approx(10.0)
    assert metrics["immediate_relative_delta"] == pytest.approx(1.0)
    assert metrics["final_relative_delta"] == pytest.approx(0.1)
    assert metrics["relative_delta_ratio"] == pytest.approx(0.1)
    assert metrics["relative_delta_log_ratio"] < 0.0

    no_op = compute_causal_recovery_metrics(
        baseline, baseline, route="q", target_layer=0,
        baseline_final_residual=np.asarray([100.0, 0.0]),
        intervention_final_residual=np.asarray([100.0, 0.0]),
        baseline_logits=np.zeros((1, 2, 3)),
        intervention_logits=np.zeros((1, 2, 3)), target_position=0)
    assert no_op["relative_delta_ratio"] == 1.0
    assert no_op["relative_delta_log_ratio"] == 0.0

    rows = [
        {"status": "ready", "relative_delta_ratio": np.exp(value),
         "relative_delta_log_ratio": value,
         "immediate_delta_norm": 1.0, "final_delta_norm": 1.0,
         "immediate_relative_delta": 1.0,
         "final_relative_delta": np.exp(value),
         "absolute_delta_ratio": 1.0}
        for value in (-0.10, 0.0, 0.10)
    ]
    basis = _classify_recovery_rows(rows, neutral_log_band=0.05)
    assert basis["semantic_boundary"] == "relative_delta_ratio == 1"
    assert basis["quantile_role"] == "descriptive_intensity_only"
    assert [row["recovery_phenomenon"] for row in rows] == [
        "relative_recovery", "approximately_preserved",
        "relative_amplification"]
    summary = _recovery_group_summary(rows, seed=17)
    assert summary["relative_delta_ratio_mean"] is not None
    assert summary["relative_delta_log_ratio_mean"] == pytest.approx(0.0)
    assert "absolute_delta_ratio_mean_diagnostic" in summary

    judgment_rows = []
    for strategy, immediate, final_relative, log_ratio in (
            ("top_gate", 4.0, 0.1, -1.0),
            ("top_contribution", 3.0, 0.2, -0.8),
            ("matched_active", 2.0, 0.3, 0.0),
            ("active_random", 1.0, 0.4, 0.2)):
        judgment_rows.append({
            "status": "ready", "prompt_id": "p", "pool": "q",
            "layer": 0, "strategy": strategy, "phenomenon": "x",
            "immediate_delta_norm": immediate,
            "final_delta_norm": final_relative,
            "immediate_relative_delta": 1.0,
            "final_relative_delta": final_relative,
            "absolute_delta_ratio": final_relative / immediate,
            "relative_delta_ratio": float(np.exp(log_ratio)),
            "relative_delta_log_ratio": log_ratio,
        })
    recovery_ctx = SimpleNamespace(
        args=SimpleNamespace(causal_recovery_neutral_log_band=0.05),
        model_cfg={"n_layers": 2}, config={"seed": 11},
        store=SimpleNamespace(path=lambda name: name), is_primary=False)
    judgment = run_causal_recovery_trace(recovery_ctx, judgment_rows)[
        "downstream_compensation_dominant"]
    assert judgment["supported"] is True
    assert judgment["evidence"][
        "immediate_delta_vs_final_relative_delta_spearman"] == -1.0
    assert judgment["evidence"][
        "spearman_consistent_with_compensation"] is True


def test_group_padding_and_membership_match_single_semantics() -> None:
    padded = pad_group_operator_ids([2, 5], 8)
    np.testing.assert_array_equal(padded[:3], np.asarray([2, 5, -1]))
    mask = group_operator_membership_mask(np.arange(7), padded)
    np.testing.assert_array_equal(
        mask[0], np.asarray([False, False, True, False, False, True, False]))

    single = np.asarray(analysis_operator_membership(
        np.arange(7), np.asarray([2], dtype=np.int32)))
    group = np.asarray(analysis_operator_membership(
        np.arange(7), padded[None, :]))
    assert single.shape == (1, 1, 7)
    assert group.shape == (1, 1, 7)
    assert single[0, 0, 2]
    assert group[0, 0, 2] and group[0, 0, 5]
    assert not group[0, 0, 0]


def test_tiny_model_group_baseline_and_target_only_causal_trace() -> None:
    device = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    mesh = Mesh(device, ("data", "model"))
    single = make_sharded_srw_minimal(mesh, max_chunk_size=2)
    paired = make_sharded_srw_paired_minimal(mesh, max_chunk_size=2)
    single_suppression = make_sharded_srw_suppression_minimal(
        mesh, max_chunk_size=2)
    paired_suppression = make_sharded_srw_paired_suppression_minimal(
        mesh, max_chunk_size=2)
    assert (
        single._v4171_canonical_shard_map_kernel
        is single_suppression._v4171_canonical_shard_map_kernel)
    assert (
        paired._v4171_canonical_shard_map_kernel
        is paired_suppression._v4171_canonical_shard_map_kernel)
    model = DAWN_SRW_V4171(
        vocab_size=16, d_model=4, d_route=4, n_layers=1, n_heads=1,
        max_seq_len=3, dropout_rate=0.0, router_dropout=0.0,
        n_qk=4, n_v=4, n_rst=4,
        tau_init_attn_qk=-0.99, tau_init_attn_v=-0.99,
        tau_init_rst=-0.99,
    )
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(7),
         "dropout": jax.random.PRNGKey(8)},
        input_ids, deterministic=True)
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
    common = {
        "labels": input_ids,
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(9)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "minimal_train": True,
        "analysis_target_layer": jnp.int32(0),
        "analysis_target_positions": jnp.asarray([1], dtype=jnp.int32),
        "analysis_target_route": jnp.int32(2),
        "analysis_return_residual": True,
        "analysis_return_logits": True,
        "analysis_causal_trace": True,
        "compute_accuracy": False,
    }
    disabled = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.asarray([0], dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(False),
        **common)
    group_baseline = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.full((1, 8), -1, dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(True),
        **common)
    assert np.array_equal(
        np.asarray(disabled["logits"]), np.asarray(group_baseline["logits"]))
    assert np.array_equal(
        np.asarray(disabled["final_residual"]),
        np.asarray(group_baseline["final_residual"]))
    trace_keys = {
        "pre_layer_residual", "attention_router_input",
        "query_q", "query_k", "query_v",
        "raw_tau_q", "raw_tau_k", "raw_tau_v",
        "route_transition_q", "route_transition_k", "route_transition_v",
        "post_attention_residual", "attention_update",
        "rst_router_input", "query_rst", "raw_tau_rst",
        "route_transition_rst", "rst_update", "post_layer_residual",
    }
    assert set(group_baseline["causal_trace"]) == trace_keys
    assert all(np.array_equal(
        np.asarray(disabled["causal_trace"][key]),
        np.asarray(group_baseline["causal_trace"][key])) for key in trace_keys)
    for key, value in group_baseline["causal_trace"].items():
        assert value.shape == (
            (1, 1, 1) if key.startswith("raw_tau_") else (1, 1, 4))

    scalar_single = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.asarray([0], dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(True),
        **common)
    group_single = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.asarray(
            [[0, -1, -1, -1, -1, -1, -1, -1]], dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(True),
        **common)
    assert np.array_equal(
        np.asarray(scalar_single["logits"]), np.asarray(group_single["logits"]))
    assert np.array_equal(
        np.asarray(scalar_single["final_residual"]),
        np.asarray(group_single["final_residual"]))
    assert all(np.array_equal(
        np.asarray(scalar_single["causal_trace"][key]),
        np.asarray(group_single["causal_trace"][key])) for key in trace_keys)

    changed = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.asarray(
            [[0, 1, -1, -1, -1, -1, -1, -1]], dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(True),
        **common)
    assert not np.array_equal(
        np.asarray(group_baseline["final_residual"]),
        np.asarray(changed["final_residual"]))


def test_group_inference_is_layer_paired_and_dose_response_is_per_seed() -> None:
    rows = []
    for layer, left, right in ((0, 3.0, 1.0), (1, 1.0, 2.0)):
        common = {
            "status": "ready", "prompt_id": "p", "pool": "v",
            "layer": layer, "seed_operator_id": 9,
            "seed_strategy": "top_gate", "requested_group_size": 4,
        }
        rows.extend([
            {**common, "group_type": "reciprocal_functional_neighbors",
             "abs_sequence_behavior_delta": left},
            {**common, "group_type": "random_active_size_matched",
             "abs_sequence_behavior_delta": right},
        ])
    paired = _paired_group_comparison(
        rows, "reciprocal_functional_neighbors",
        "random_active_size_matched", seed=5)
    assert paired["paired_n"] == 2
    assert paired["paired_mean_difference"] == pytest.approx(0.5)
    assert paired["paired_median_difference"] == pytest.approx(0.5)
    assert paired["paired_sign_win_rate"] == 0.5
    assert paired["sign_flip_two_sided_p"] is not None

    dose_rows = [
        {"status": "ready", "prompt_id": "p", "pool": "q",
         "layer": 2, "seed_operator_id": 4,
         "seed_strategy": "top_contribution",
         "group_type": "reciprocal_functional_neighbors",
         "matched_target_group_type": None,
         "requested_group_size": size,
         "abs_sequence_behavior_delta": effect}
        for size, effect in ((1, 0.1), (2, 0.2), (4, 0.6), (8, 0.5))
    ]
    dose = _paired_dose_response(dose_rows)
    assert dose["paired_seed_count"] == 1
    assert dose["paired_effect_delta_2_minus_1"]["mean"] == pytest.approx(0.1)
    assert dose["paired_effect_delta_4_minus_2"]["mean"] == pytest.approx(0.4)
    assert dose["paired_effect_delta_8_minus_4"]["mean"] == pytest.approx(-0.1)
    assert dose["monotonic_nondecreasing_fraction"] == 0.0
    assert dose["peak_effect_group_size_distribution"] == {"4": 1}


def test_ranking_correlation_and_pairwise_wins() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == 1.0
    rows = [
        {"prompt_id": "p", "pool": "q", "strategy": "top_gate",
         "effect": 3.0},
        {"prompt_id": "p", "pool": "q", "strategy": "active_random",
         "effect": 1.0},
        {"prompt_id": "r", "pool": "v", "strategy": "top_gate",
         "effect": 0.5},
        {"prompt_id": "r", "pool": "v", "strategy": "active_random",
         "effect": 0.7},
    ]
    result = pairwise_win_rate(rows, "top_gate", "active_random", "effect")
    assert result == {"n": 2, "win_rate": 0.5}


def test_rerouting_paired_evidence_requires_sample_and_inference_support() -> None:
    strong = {
        "paired_n": 6,
        "paired_mean_difference": 0.2,
        "bootstrap_ci95": [0.01, 0.4],
        "paired_sign_win_rate": 2.0 / 3.0,
        "sign_flip_two_sided_p": 0.2,
    }
    directional = {
        "paired_n": 4,
        "paired_mean_difference": 0.1,
        "bootstrap_ci95": [-0.1, 0.3],
        "paired_sign_win_rate": 0.75,
        "sign_flip_two_sided_p": 0.2,
    }
    zero = {
        "paired_n": 8,
        "paired_mean_difference": 0.0,
        "bootstrap_ci95": [-0.1, 0.1],
        "paired_sign_win_rate": 0.5,
        "sign_flip_two_sided_p": 1.0,
    }
    insufficient = {
        "paired_n": 3,
        "paired_mean_difference": 0.2,
        "bootstrap_ci95": [0.1, 0.3],
        "paired_sign_win_rate": 1.0,
        "sign_flip_two_sided_p": 0.01,
    }
    assert classify_paired_directional_evidence(strong)[
        "classification"] == "strong_positive"
    assert classify_paired_directional_evidence(directional)[
        "classification"] == "directional_positive"
    assert classify_paired_directional_evidence(zero)[
        "classification"] == "no_positive_evidence"
    assert classify_paired_directional_evidence(insufficient)[
        "classification"] == "insufficient_evidence"

    aggregate = _classify_important_intervention_control_evidence({
        "top_gate_vs_active_random": strong,
        "top_contribution_vs_active_random": zero,
    })
    assert aggregate["aggregate_classification"] == "strong"
    opposed = _classify_important_intervention_control_evidence({
        "top_gate_vs_active_random": strong,
        "top_contribution_vs_active_random": {
            **zero, "paired_mean_difference": -0.1},
    })
    assert opposed["aggregate_classification"] == "not_supported"


def test_rerouting_predictive_evidence_uses_n_rho_and_ci() -> None:
    strong = {"n": 12, "rho": 0.4, "bootstrap_ci95": [0.1, 0.7]}
    directional = {"n": 8, "rho": 0.2, "bootstrap_ci95": [-0.2, 0.5]}
    too_small = {"n": 7, "rho": 0.8, "bootstrap_ci95": [0.4, 0.9]}
    assert classify_predictive_correlation_evidence(strong)[
        "classification"] == "strong_predictive_evidence"
    assert classify_predictive_correlation_evidence(directional)[
        "classification"] == "directional_predictive_evidence"
    assert classify_predictive_correlation_evidence(too_small)[
        "classification"] == "no_predictive_evidence"

    aggregate = _classify_predictive_relation_evidence({
        "routing_divergence_auc_vs_final_relative_residual_delta": strong,
        "routing_divergence_auc_vs_sequence_behavior_effect": directional,
    })
    assert aggregate["aggregate_classification"] == "strong"


def test_operator_family_decision_replaces_unrequested_active_failure() -> None:
    capture = {
        "total_observations": 8, "qualified_observations": 8,
        "excluded_observations": 0, "remaining_low_capture_count": 0,
        "pools": {},
    }
    analysis = {
        "causal_intervention": {
            "zero_suppression_parity": {
                "machine_exact": True, "ce_abs_diff": 0.0,
                "max_logit_abs_diff": 0.0,
                "final_residual_max_abs_diff": 0.0,
            },
        },
        "causal_ranking_calibration": {"judgments": {
            "local_ranking_valid": {"supported": True, "evidence": {
                "spearman": 0.6, "bootstrap_ci95": [0.2, 0.8], "n": 20,
            }},
        }},
        "causal_recovery_trace": {"downstream_compensation_dominant": {
            "supported": True, "evidence": {
                "median_relative_delta_ratio": 0.8,
                "relative_recovery_fraction": 0.7,
                "relative_amplification_fraction": 0.2,
            },
        }},
        "causal_rerouting_trace": {
            "capture_reliability": capture,
            "important_intervention_control_evidence": {
                "aggregate_classification": "strong"},
            "predictive_relation_evidence": {
                "aggregate_classification": "strong"},
            "trajectory_classification": {
                "important_meaningful_divergence_count": 8,
                "nonreconvergence_fraction_among_diverged": 0.8,
            },
            "path_dependence_supported": {
                "status": "supported", "supported": True,
                "suggestive": False, "evidence": {}, "limitations": [],
            },
        },
        "trajectory_trace": {"capture_reliability": capture},
        "group_causal_intervention": {
            "zero_size_group_parity": {
                "machine_exact": True, "num_comparisons": 4},
            "functional_redundancy_supported": {
                "supported": True, "evidence": {
                    "paired_n": 10, "paired_mean_difference": 0.1,
                    "bootstrap_ci95": [0.02, 0.2],
                }},
        },
        "operator_functional_graph": {"pools": {
            "qk": {"address_function_spearman": 0.1,
                   "percolated": True,
                   "largest_component_fraction": 0.9},
            "v": {"address_function_spearman": 0.2,
                  "percolated": False,
                  "largest_component_fraction": 0.3},
        }},
    }
    judgments, lines = _operator_family_decision(analysis)
    assert set(judgments) == {
        "canonical_causal_path_valid", "local_operator_ranking_valid",
        "downstream_compensation_dominant", "path_dependence_supported",
        "functional_redundancy_supported", "address_function_alignment",
        "functional_graph_percolated", "capture_reliability",
    }
    assert judgments["canonical_causal_path_valid"]["supported"] is True
    assert judgments["functional_graph_percolated"]["percolated"] is True
    assert all("Active/tau metrics are unavailable" not in line for line in lines)

    report_summary = {
        "run": {"model_version": "spatial-r1-v4.1.7.1",
                "analysis_batches": 8},
        "progress": {},
        "selection_calibration": {},
        "active_dynamics": {
            "status": "not_requested", "num_batches": 0, "pools": {},
            "reason": "active/tau metrics were not requested by this preset",
        },
        "reference_400m": {"status": "not_applicable"},
        "recent_trend": [], "warnings": [], "decision": lines,
        "analysis_preset": "v4171_operator_family",
        "analysis_items": ["causal_intervention"],
        "analysis_required_sections": ["v4171_transition"],
        "v4171_transition_analysis": analysis,
    }
    report = _format_train_analysis(report_summary)
    assert "status          : not_requested" in report
    assert "active/tau metrics were not requested by this preset" in report
    assert "pool    active_tau" not in report


def test_formatter_names_canonical_causal_baseline() -> None:
    fmt = TrainAnalysisFormatters(
        num=lambda value, _digits: "n/a" if value is None else str(value),
        delta=lambda value, _digits: "n/a" if value is None else str(value),
        pct=lambda value, _digits: "n/a" if value is None else str(value),
        eps=str,
        safe_float=lambda value: None if value is None else float(value),
    )
    summary = {
        "v4171_transition_analysis": {
            "causal_intervention": {
                "status": "ready",
                "causal_baseline": "canonical_suppression_disabled",
                "effect_reference": "canonical_suppression_disabled",
                "zero_suppression_parity": {
                    "machine_exact": True, "ce_abs_diff": 0.0,
                    "mean_logit_abs_diff": 0.0, "max_logit_abs_diff": 0.0,
                    "final_residual_max_abs_diff": 0.0,
                },
                "normal_production_cross_graph_audit": {
                    "machine_exact": False, "blocking": False},
            }
        }
    }
    text = "\n".join(format_train_analysis_item(
        summary, "causal_intervention", fmt))
    assert "baseline        : canonical_suppression_disabled" in text
    assert "effect reference: canonical_suppression_disabled" in text
    assert "parity           : machine_exact=True" in text
    assert "cross graph audit: machine_exact=False blocking=False" in text
    assert "legacy" not in text
