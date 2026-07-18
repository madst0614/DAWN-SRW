"""CPU contracts for IOI native operator program surgery."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from analysis.dawn_analysis_storage import AnalysisStore
from analysis.operator_interpretability.program import (
    OperatorProgramSchedule,
    build_program_schedule,
    deterministic_mismatch_mapping,
    load_program_schedule_artifact,
    select_validation_program,
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


def _validation_candidate(mass: float, *, passing: bool):
    sign = 1.0 if passing else -1.0
    return {
        "program_mass": mass,
        "compactness": {"median_site_fraction": 0.1},
        "replay": {
            "faithfulness_ci": {"ci_low": 0.9},
            "answer_agreement_with_full": 1.0,
        },
        "ablation": {
            "own_program": {"margin_drop_ci": {"ci_low": sign}}},
        "transplant": {
            "paired_vs_mismatch": {"effect_ci": {"ci_low": sign}}},
    }


def test_program_artifact_and_freeze_contracts() -> None:
    shape = OperatorSpaceShape(n_layers=2, n_qk=4, n_v=4, n_rst=4)
    examples = [
        replace(
            _example("validation", f"program-{index}", benchmark_id="mib_ioi"),
            pair_type="s2_io_flip_counterfactual",
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

    config = ProtocolConfig(
        bootstrap_samples=100, permutation_samples=100)
    candidates = [
        _validation_candidate(mass, passing=(mass >= 0.8))
        for mass in config.program_mass_candidates
    ]
    selected = select_validation_program(candidates, config=config)
    assert selected["selected_program_mass"] == 0.8
    assert selected["test_consulted"] is False
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


def main() -> None:
    (mesh, production_single, suppression_single,
     production_paired, suppression_paired) = _build_mesh_and_kernels()
    harness = _make_model_harness(
        mesh, production_single, suppression_single,
        production_paired, suppression_paired)
    test_model_program_semantics(harness)
    test_program_artifact_and_freeze_contracts()
    print("NATIVE_OPERATOR_PROGRAM_CPU_OK")


if __name__ == "__main__":
    main()
