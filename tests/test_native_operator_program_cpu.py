"""CPU contracts for IOI native operator program surgery."""

from __future__ import annotations

import copy
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_storage import AnalysisStore
from analysis.operator_interpretability.intervention import (
    evaluate_native_operator_program_candidate,
)
from analysis.operator_interpretability.program import (
    OperatorProgramSchedule,
    build_program_schedule,
    deterministic_mismatch_mapping,
    evaluate_native_program_claims,
    load_program_schedule_artifact,
    select_validation_program,
    select_validation_then_evaluate_test,
    write_program_schedule_artifact,
)
from analysis.operator_interpretability.protocol import (
    ProtocolConfig,
    protocol_record,
)
from analysis.operator_interpretability.units import OperatorSpaceShape
from test_operator_interpretability_cpu import (
    _assert_exact,
    _build_mesh_and_kernels,
    _example,
    _host,
    _make_model_harness,
    _route_target,
    _view,
)


ROUTES = ("q", "k", "v", "rst")


def _program_arrays(harness, *, all_ids: bool):
    width = harness.dim if all_ids else 2
    ids = {}
    valid = {}
    for route in ROUTES:
        ids[route] = jnp.zeros(
            (harness.layers, harness.batch, width), dtype=jnp.int32)
        valid[route] = jnp.zeros_like(ids[route], dtype=jnp.bool_)
        if all_ids:
            values = jnp.arange(harness.dim, dtype=jnp.int32)
            ids[route] = jnp.broadcast_to(values, ids[route].shape)
            valid[route] = jnp.ones_like(ids[route], dtype=jnp.bool_)
    return ids, valid


def _zero_sources(harness):
    return {
        route: jnp.zeros(
            (harness.layers, harness.batch, harness.dim), dtype=jnp.float32)
        for route in ROUTES
    }


def test_model_program_semantics(harness) -> None:
    positions = jnp.full((harness.batch,), 2, dtype=jnp.int32)
    all_ids, all_valid = _program_arrays(harness, all_ids=True)
    empty_ids, empty_valid = _program_arrays(harness, all_ids=False)
    zero_sources = _zero_sources(harness)

    @jax.jit
    def production(params, input_ids):
        return _view(harness.model.apply(
            params, input_ids, labels=input_ids, minimal_train=True,
            analysis_return_residual=True, **harness.kwargs))

    @jax.jit
    def program(
            params, input_ids, ids_q, ids_k, ids_v, ids_rst,
            valid_q, valid_k, valid_v, valid_rst,
            source_q, source_k, source_v, source_rst, mode):
        result = harness.model.apply(
            params, input_ids,
            selected_ids_q=ids_q,
            selected_ids_k=ids_k,
            selected_ids_v=ids_v,
            selected_ids_rst=ids_rst,
            selected_valid_q=valid_q,
            selected_valid_k=valid_k,
            selected_valid_v=valid_v,
            selected_valid_rst=valid_rst,
            target_positions=positions,
            program_mode=mode,
            source_contribution_q=source_q,
            source_contribution_k=source_k,
            source_contribution_v=source_v,
            source_contribution_rst=source_rst,
            labels=input_ids,
            return_residual=True,
            method=harness.model.analysis_forward_with_operator_program,
            **harness.kwargs)
        return _view(result)

    def run(input_ids, ids, valid, mode, source=zero_sources):
        return program(
            harness.variables, input_ids,
            ids["q"], ids["k"], ids["v"], ids["rst"],
            valid["q"], valid["k"], valid["v"], valid["rst"],
            source["q"], source["k"], source["v"], source["rst"],
            jnp.int32(mode))

    base = production(harness.variables, harness.base_ids)
    mode0 = run(harness.base_ids, all_ids, all_valid, 0)
    _assert_exact(base, mode0, "native program mode-0 production parity")

    all_replay = run(harness.base_ids, all_ids, all_valid, 1)
    _assert_exact(base, all_replay, "all-operator replay parity")

    empty_ablation = run(harness.base_ids, empty_ids, empty_valid, 2)
    _assert_exact(base, empty_ablation, "empty program ablation parity")

    @jax.jit
    def capture(params, input_ids, ids_q, ids_k, ids_v, ids_rst,
                valid_q, valid_k, valid_v, valid_rst):
        result = harness.model.apply(
            params, input_ids,
            selected_ids_q=ids_q,
            selected_ids_k=ids_k,
            selected_ids_v=ids_v,
            selected_ids_rst=ids_rst,
            selected_valid_q=valid_q,
            selected_valid_k=valid_k,
            selected_valid_v=valid_v,
            selected_valid_rst=valid_rst,
            target_positions=positions,
            labels=input_ids,
            method=(
                harness.model.analysis_capture_operator_program_contributions),
            **harness.kwargs)
        return result["operator_program_contributions"]

    base_contribution = capture(
        harness.variables, harness.base_ids,
        all_ids["q"], all_ids["k"], all_ids["v"], all_ids["rst"],
        all_valid["q"], all_valid["k"], all_valid["v"], all_valid["rst"])
    same_transplant = run(
        harness.base_ids, all_ids, all_valid, 4, base_contribution)
    _assert_exact(base, same_transplant, "source-equals-base transplant parity")

    q_ids, q_valid = _program_arrays(harness, all_ids=False)
    q_ids["q"] = q_ids["q"].at[0, :, 0].set(0)
    q_valid["q"] = q_valid["q"].at[0, :, 0].set(True)
    q_ablated = run(harness.base_ids, q_ids, q_valid, 2)
    for sibling in ("k", "v"):
        assert np.array_equal(
            np.asarray(_host(base["parity_debug"][sibling]))[0],
            np.asarray(_host(q_ablated["parity_debug"][sibling]))[0])
    assert not np.array_equal(
        _route_target(base, "q", 0, positions),
        _route_target(q_ablated, "q", 0, positions))
    q_contribution = capture(
        harness.variables, harness.base_ids,
        q_ids["q"], q_ids["k"], q_ids["v"], q_ids["rst"],
        q_valid["q"], q_valid["k"], q_valid["v"], q_valid["rst"])
    reconstructed_q = (
        _route_target(q_ablated, "q", 0, positions).astype(np.float32)
        + np.asarray(_host(q_contribution["q"]))[0].astype(np.float32))
    np.testing.assert_allclose(
        _route_target(base, "q", 0, positions), reconstructed_q,
        rtol=0.0, atol=2.0e-2)

    source_id_replay = run(harness.base_ids, q_ids, q_valid, 3)
    assert not np.array_equal(
        _route_target(base, "q", 0, positions),
        _route_target(source_id_replay, "q", 0, positions))
    source_q_contribution = capture(
        harness.variables, harness.source_ids,
        q_ids["q"], q_ids["k"], q_ids["v"], q_ids["rst"],
        q_valid["q"], q_valid["k"], q_valid["v"], q_valid["rst"])
    source_transplant = run(
        harness.base_ids, q_ids, q_valid, 4, source_q_contribution)
    expected_source_transplant_q = (
        _route_target(q_ablated, "q", 0, positions).astype(np.float32)
        + np.asarray(_host(source_q_contribution["q"]))[0].astype(
            np.float32))
    np.testing.assert_allclose(
        _route_target(source_transplant, "q", 0, positions),
        expected_source_transplant_q, rtol=0.0, atol=2.0e-2)
    assert not np.allclose(
        _route_target(base, "q", 0, positions),
        _route_target(source_transplant, "q", 0, positions),
        rtol=0.0, atol=2.0e-2)
    for route in ROUTES:
        assert np.array_equal(
            np.asarray(_host(base["parity_debug"][route]))[0, :, :2],
            np.asarray(_host(q_ablated["parity_debug"][route]))[0, :, :2])
        assert np.array_equal(
            np.asarray(_host(base["parity_debug"][route]))[0, :, 3:],
            np.asarray(_host(q_ablated["parity_debug"][route]))[0, :, 3:])

    batch_ids, batch_valid = _program_arrays(harness, all_ids=False)
    batch_ids["q"] = batch_ids["q"].at[:, 0, 0].set(0)
    batch_valid["q"] = batch_valid["q"].at[:, 0, 0].set(True)
    batch_result = run(harness.base_ids, batch_ids, batch_valid, 2)
    assert np.array_equal(
        np.asarray(_host(base["final_residual"]))[1],
        np.asarray(_host(batch_result["final_residual"]))[1])
    assert not np.array_equal(
        np.asarray(_host(base["final_residual"]))[0],
        np.asarray(_host(batch_result["final_residual"]))[0])

    padded_ids = dict(q_ids)
    padded_valid = dict(q_valid)
    padded_ids["q"] = padded_ids["q"].at[:, :, 1].set(0)
    padded_valid["q"] = padded_valid["q"].at[:, :, 1].set(False)
    padded = run(harness.base_ids, padded_ids, padded_valid, 2)
    _assert_exact(q_ablated, padded, "invalid padded id contributes zero")
    print("NATIVE_PROGRAM_MODEL_SEMANTICS_OK")


def _capture_rows(examples, shape):
    rows = []
    for route in ROUTES:
        pool = shape.pool_size(route)
        for layer in range(shape.n_layers):
            for example in examples:
                rows.append({
                    "example_id": example.example_id,
                    "layer": layer,
                    "route": route,
                    "captured_mass": 1.0,
                    "operator_ids": list(range(pool)),
                    "weights": [float(pool - index)
                                for index in range(pool)],
                })
    return {"status": "ready", "rows": rows}


def _asymmetric_capture_rows(examples, shape, *, prompt_side: str):
    if any(shape.pool_size(route) != 8 for route in ROUTES):
        raise ValueError("asymmetric capture fixture requires pool size 8")
    weights_by_count = {
        1: [0.55, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04],
        2: [0.30, 0.25, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04],
        3: [0.20, 0.18, 0.14, 0.13, 0.11, 0.09, 0.08, 0.07],
        4: [0.15, 0.14, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11],
    }
    rows = []
    for route in ROUTES:
        pool = shape.pool_size(route)
        for layer in range(shape.n_layers):
            for example_index, example in enumerate(examples):
                selected_count = 1 + (
                    example_index % 4 if prompt_side == "base"
                    else (example_index + 1) % 4)
                operator_ids = (
                    list(range(pool)) if prompt_side == "base"
                    else list(range(pool - 1, -1, -1)))
                rows.append({
                    "example_id": example.example_id,
                    "layer": layer,
                    "route": route,
                    "captured_mass": 1.0,
                    "operator_ids": operator_ids,
                    "weights": weights_by_count[selected_count],
                })
    return {"status": "ready", "rows": rows}


def _validation_candidate(
        mass: float, *, replay_passing: bool, causal_passing: bool):
    sign = 1.0 if causal_passing else -1.0
    p_value = 0.01 if causal_passing else 0.5
    advantage = {
        "effect_ci": {"ci_low": sign},
        "permutation": {"p_value_two_sided": p_value},
    }
    return {
        "program_mass": mass,
        "compactness": {
            "median_decision_position_site_fraction": 0.1},
        "replay": {
            "faithfulness_ci": {
                "ci_low": 0.9 if replay_passing else 0.7},
            "answer_agreement_with_full": (
                1.0 if replay_passing else 0.0),
        },
        "ablation": {
            "own_program": {
                "margin_drop_ci": {"ci_low": sign},
                "permutation": {"p_value_two_sided": p_value},
            },
            "specificity": {
                "own_vs_mismatched": dict(advantage),
                "own_vs_random": dict(advantage),
            },
        },
        "source_id_replay": {
            "paired_vs_mismatch": dict(advantage),
            "paired_vs_random": dict(advantage),
            "bidirectional_answer_flip_fraction": (
                1.0 if causal_passing else 0.0),
        },
        "transplant": {
            "paired_vs_mismatch": {
                "effect_ci": {"ci_low": sign},
                "permutation": {"p_value_two_sided": p_value},
            },
            "paired_vs_random": dict(advantage),
            "bidirectional_answer_flip_fraction": (
                1.0 if causal_passing else 0.0),
        },
    }


def test_program_artifact_and_freeze_contracts() -> None:
    shape = OperatorSpaceShape(n_layers=2, n_qk=4, n_v=4, n_rst=4)
    examples = [
        replace(
            _example("validation", f"program-{index}", benchmark_id="mib_ioi"),
            pair_type="s2_io_flip_counterfactual",
            positive_ids=(1000 + 2 * index,),
            negative_ids=(1001 + 2 * index,),
            source_positive_ids=(1001 + 2 * index,),
            source_negative_ids=(1000 + 2 * index,),
            intervention_positive_ids=(1001 + 2 * index,),
            intervention_negative_ids=(1000 + 2 * index,),
            metadata={"template": index % 2})
        for index in range(4)
    ]
    schedule = build_program_schedule(
        _capture_rows(examples, shape), examples, shape=shape,
        program_mass=0.5, prompt_side="base",
        widths={route: 4 for route in ROUTES})
    mapping_a = deterministic_mismatch_mapping(examples, schedule, seed=4172)
    mapping_b = deterministic_mismatch_mapping(examples, schedule, seed=4172)
    assert mapping_a == mapping_b
    assert all(
        index != donor for index, donor in enumerate(mapping_a["donor_indices"]))
    assert all(row["answer_token_disjoint"] for row in mapping_a["rows"])

    config = ProtocolConfig(
        bootstrap_samples=100, permutation_samples=100)
    candidates = [
        _validation_candidate(
            mass, replay_passing=(mass >= 0.8), causal_passing=False)
        for mass in config.program_mass_candidates
    ]
    selected = select_validation_program(candidates, config=config)
    assert selected["selected_program_mass"] == 0.8
    assert selected["test_consulted"] is False
    assert selected["causal_diagnostics_used_for_selection"] is False
    selected_candidate = next(
        row for row in candidates if row["program_mass"] == 0.8)
    assert set(selected_candidate["validation_selection_checks"]) == {
        "replay_faithfulness_ci", "replay_answer_agreement", "compactness"}
    assert not all(
        selected_candidate["validation_diagnostic_checks"].values())
    frozen = selected["selected_program_mass"]
    assert frozen in config.program_mass_candidates

    protocol = protocol_record(
        config,
        model_version="spatial-r1-v4.1.7.1",
        benchmark_manifest_hash="benchmark",
        checkpoint_identity="checkpoint",
        model_config_hash="model")
    with tempfile.TemporaryDirectory() as tmp:
        store = AnalysisStore(tmp, analysis_version="native-program-test")
        record = write_program_schedule_artifact(
            store, "program.npz", schedule, shape=shape, protocol=protocol)
        assert record is not None
        loaded = load_program_schedule_artifact(
            record["path"], expected_sha256=record["sha256"],
            shape=shape, protocol=protocol)
        assert loaded.schedule_hash == schedule.schedule_hash

        changed_protocol = protocol_record(
            config,
            model_version="spatial-r1-v4.1.7.1",
            benchmark_manifest_hash="benchmark",
            checkpoint_identity="changed-checkpoint",
            model_config_hash="model")
        try:
            load_program_schedule_artifact(
                record["path"], expected_sha256=record["sha256"],
                shape=shape, protocol=changed_protocol)
        except ValueError as exc:
            assert "protocol/config/checkpoint mismatch" in str(exc)
        else:
            raise AssertionError("program protocol mismatch was accepted")

        path = Path(record["path"])
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt")
        try:
            load_program_schedule_artifact(
                str(path), expected_sha256=record["sha256"],
                shape=shape, protocol=protocol)
        except ValueError as exc:
            assert "binary artifact hash mismatch" in str(exc)
        else:
            raise AssertionError("program binary hash mismatch was accepted")
    print("NATIVE_PROGRAM_ARTIFACT_FREEZE_CONTRACTS_OK")


def _ioi_examples(phase: str, count: int):
    return [
        replace(
            _example(phase, f"integration-{index}", benchmark_id="mib_ioi"),
            pair_type="s2_io_flip_counterfactual",
            positive_ids=(2000 + 2 * index,),
            negative_ids=(2001 + 2 * index,),
            source_positive_ids=(2001 + 2 * index,),
            source_negative_ids=(2000 + 2 * index,),
            intervention_positive_ids=(2001 + 2 * index,),
            intervention_negative_ids=(2000 + 2 * index,),
            metadata={"template": index % 2},
        )
        for index in range(count)
    ]


def test_paired_controls_and_claim_gate_integration() -> None:
    shape = OperatorSpaceShape(n_layers=2, n_qk=8, n_v=8, n_rst=8)
    examples = _ioi_examples("test", 16)
    base_capture_rows = _asymmetric_capture_rows(
        examples, shape, prompt_side="base")
    source_capture_rows = _asymmetric_capture_rows(
        examples, shape, prompt_side="source")
    widths = {route: 8 for route in ROUTES}
    base_schedule = build_program_schedule(
        base_capture_rows, examples, shape=shape, program_mass=0.5,
        prompt_side="base", widths=widths)
    source_schedule = build_program_schedule(
        source_capture_rows, examples, shape=shape, program_mass=0.5,
        prompt_side="source", widths=widths)
    assert all(
        base["site_count"] != source["site_count"]
        for base, source in zip(
            base_schedule.records, source_schedule.records))
    assert any(
        not np.array_equal(
            base_schedule.ids[route], source_schedule.ids[route])
        for route in ROUTES)

    ablation_schedule_hashes: list[str] = []

    def fake_capture(
            _ctx, capture_examples, _schedule, *, prompt_side,
            pad_token_id):
        del prompt_side, pad_token_id
        return {
            route: np.zeros(
                (shape.n_layers, len(capture_examples), 4),
                dtype=np.float32)
            for route in ROUTES
        }

    def fake_margin(
            _ctx, margin_examples, schedule, *, prompt_side,
            positive_side, negative_side, pad_token_id, program_mode,
            source=None):
        del prompt_side, negative_side, pad_token_id, source
        count = len(margin_examples)
        if program_mode == 0:
            if positive_side in {"intervention_positive", "source_negative"}:
                value = -1.0
            elif schedule.prompt_side == "source":
                value = -2.0
            else:
                value = 2.0
        elif program_mode == 1:
            value = 2.0
        elif program_mode == 2:
            ablation_schedule_hashes.append(schedule.schedule_hash)
            value = {
                "base": 0.0,
                "mismatched_base": 1.5,
                "random_id": 1.8,
            }[schedule.prompt_side]
        elif program_mode == 3:
            if schedule.prompt_side in {"base", "source"}:
                value = 1.0
            elif schedule.prompt_side.startswith("mismatched_"):
                value = -0.4
            else:
                value = -0.7
        elif program_mode == 4:
            if schedule.prompt_side in {"base", "source"}:
                value = 1.2
            elif schedule.prompt_side.startswith("mismatched_"):
                value = -0.3
            else:
                value = -0.6
        else:
            raise AssertionError(f"unexpected program mode: {program_mode}")
        return np.full((count,), value, dtype=np.float64)

    config = ProtocolConfig(
        bootstrap_samples=100,
        permutation_samples=100,
        program_compact_fraction_max=0.6,
    )
    with patch(
            "analysis.operator_interpretability.intervention."
            "capture_operator_program_contributions",
            side_effect=fake_capture), patch(
                "analysis.operator_interpretability.intervention."
                "_program_margin", side_effect=fake_margin):
        result, controls = evaluate_native_operator_program_candidate(
            object(), examples,
            base_schedule=base_schedule,
            source_schedule=source_schedule,
            shape=shape,
            pad_token_id=0,
            config=config,
            seed=4172,
        )

    for name in ("own_vs_mismatched", "own_vs_random"):
        stats = result["ablation"]["specificity"][name]
        assert stats["effect_ci"]["ci_low"] > 0.0
        assert stats["permutation"]["p_value_two_sided"] < config.alpha
    for name in ("paired_vs_mismatch", "paired_vs_random"):
        stats = result["source_id_replay"][name]
        assert stats["effect_ci"]["ci_low"] > 0.0
        assert stats["permutation"]["p_value_two_sided"] < config.alpha
    assert result["source_id_replay"][
        "bidirectional_answer_flip_fraction"] == 1.0
    assert result["transplant"]["paired_vs_mismatch"][
        "effect_ci"]["ci_low"] > 0.0
    assert result["transplant"]["paired_vs_random"][
        "effect_ci"]["ci_low"] > 0.0
    assert result["transplant"][
        "bidirectional_answer_flip_fraction"] == 1.0
    assert result["random_control"]["source"][
        "max_overlap_fraction"] == 0.0
    assert result["random_control"]["base"][
        "max_overlap_fraction"] == 0.0
    assert result["ablation"]["control_schedule_side"] == "base"
    assert ablation_schedule_hashes == [
        base_schedule.schedule_hash,
        controls["mismatch_base"].schedule_hash,
        controls["random_base"].schedule_hash,
    ]
    assert result["ablation"]["mismatched_schedule_hash"] == (
        controls["mismatch_base"].schedule_hash)
    assert result["ablation"]["random_schedule_hash"] == (
        controls["random_base"].schedule_hash)

    def assert_mapping_matches_schedule(mapping, schedule, control):
        assert all(row["answer_token_disjoint"] for row in mapping["rows"])
        for index, row in enumerate(mapping["rows"]):
            donor_index = int(mapping["donor_indices"][index])
            assert row["recipient_site_count"] == int(
                schedule.records[index]["site_count"])
            assert row["donor_site_count"] == int(
                schedule.records[donor_index]["site_count"])
            assert int(control.records[index]["site_count"]) == int(
                row["donor_site_count"])

    assert_mapping_matches_schedule(
        result["mismatch_source_mapping"], source_schedule,
        controls["mismatch_source"])
    assert_mapping_matches_schedule(
        result["mismatch_base_mapping"], base_schedule,
        controls["mismatch_base"])
    assert result["mismatch_source_mapping"]["seed"] != (
        result["mismatch_base_mapping"]["seed"])

    def assert_random_disjoint(reference, random_control):
        for route in ROUTES:
            for layer in range(shape.n_layers):
                for example_index in range(len(examples)):
                    reference_ids = set(int(value) for value in reference.ids[
                        route][layer, example_index, reference.valid[
                            route][layer, example_index]])
                    random_ids = set(int(value) for value in random_control.ids[
                        route][layer, example_index, random_control.valid[
                            route][layer, example_index]])
                    assert reference_ids.isdisjoint(random_ids)

    assert_random_disjoint(base_schedule, controls["random_base"])
    assert_random_disjoint(source_schedule, controls["random_source"])
    assert set(controls) == {
        "mismatch_source", "mismatch_base", "random_source", "random_base"}
    claims = evaluate_native_program_claims(result, config=config)
    assert claims["passed"] is True
    assert claims["strongest_supported_claim"] == (
        "counterfactual_contribution_transplant")
    specificity_blocked = copy.deepcopy(result)
    specificity_blocked["ablation"]["specificity"]["own_vs_random"][
        "effect_ci"]["ci_low"] = 0.0
    blocked_claims = evaluate_native_program_claims(
        specificity_blocked, config=config)
    assert blocked_claims["strongest_supported_claim"] == (
        "compact_dynamic_sufficiency")
    id_transfer_blocked = copy.deepcopy(result)
    id_transfer_blocked["source_id_replay"]["paired_vs_random"][
        "permutation"]["p_value_two_sided"] = 0.5
    blocked_claims = evaluate_native_program_claims(
        id_transfer_blocked, config=config)
    assert blocked_claims["strongest_supported_claim"] == (
        "specific_causal_decision_program")
    transplant_random_blocked = copy.deepcopy(result)
    transplant_random_blocked["transplant"]["paired_vs_random"][
        "permutation"]["p_value_two_sided"] = 0.5
    blocked_claims = evaluate_native_program_claims(
        transplant_random_blocked, config=config)
    assert blocked_claims["strongest_supported_claim"] == (
        "counterfactual_operator_selection_transfer")
    print("NATIVE_PROGRAM_PAIRED_CONTROL_INTEGRATION_OK")


def test_validation_selection_precedes_single_test_call() -> None:
    config = ProtocolConfig(
        bootstrap_samples=100, permutation_samples=100)
    calls: list[float] = []

    def evaluate_test(selected_mass: float):
        calls.append(selected_mass)
        return {"selected_mass": selected_mass}

    failing = [
        _validation_candidate(
            mass, replay_passing=False, causal_passing=True)
        for mass in config.program_mass_candidates
    ]
    selection, test_result = select_validation_then_evaluate_test(
        failing, config=config, test_evaluator=evaluate_test)
    assert selection["status"] == "no_compact_validation_program"
    assert test_result is None
    assert calls == []

    passing = [
        _validation_candidate(
            mass, replay_passing=(mass >= 0.8), causal_passing=False)
        for mass in config.program_mass_candidates
    ]
    selection, test_result = select_validation_then_evaluate_test(
        passing, config=config, test_evaluator=evaluate_test)
    assert selection["selected_program_mass"] == 0.8
    assert test_result == {"selected_mass": 0.8}
    assert calls == [0.8]
    print("NATIVE_PROGRAM_VALIDATION_TEST_ORDER_OK")


def main() -> None:
    (mesh, production_single, suppression_single,
     production_paired, suppression_paired) = _build_mesh_and_kernels()
    harness = _make_model_harness(
        mesh, production_single, suppression_single,
        production_paired, suppression_paired)
    test_model_program_semantics(harness)
    test_program_artifact_and_freeze_contracts()
    test_paired_controls_and_claim_gate_integration()
    test_validation_selection_precedes_single_test_call()
    print("NATIVE_OPERATOR_PROGRAM_CPU_OK")


if __name__ == "__main__":
    main()
