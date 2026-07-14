"""Lightweight synthetic tests for v4171 operator-family analysis helpers."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from analysis.dawn_train_analysis_items import (
    TRAIN_ANALYSIS_PRESETS,
    TrainAnalysisFormatters,
    format_train_analysis_item,
)
from analysis.dawn_v4171_transition import (
    classify_function_address_pairs,
    compute_causal_recovery_metrics,
    group_operator_membership_mask,
    mutual_neighbor_families,
    pad_group_operator_ids,
    pairwise_win_rate,
    rw_functional_similarity,
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


def test_existing_preset_is_unchanged_and_extended_preset_is_opt_in() -> None:
    original = TRAIN_ANALYSIS_PRESETS["v4171_self_organization"]
    assert original == (
        "global_router_audit", "trajectory_trace", "context_divergence",
        "state_transition_decoupling", "causal_intervention")
    extended = TRAIN_ANALYSIS_PRESETS["v4171_operator_family"]
    assert extended[:len(original)] == original
    assert extended[-4:] == (
        "causal_recovery_trace", "operator_functional_graph",
        "group_causal_intervention", "causal_ranking_calibration")


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
    assert set(group_baseline["causal_trace"]) == {
        "post_attention_residual", "post_layer_residual",
        "attention_update", "rst_update"}
    assert all(
        value.shape == (1, 1, 4)
        for value in group_baseline["causal_trace"].values())

    changed = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.asarray(
            [[0, 1, -1, -1, -1, -1, -1, -1]], dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(True),
        **common)
    assert not np.array_equal(
        np.asarray(group_baseline["final_residual"]),
        np.asarray(changed["final_residual"]))


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
