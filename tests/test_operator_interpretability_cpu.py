"""Deterministic 2x2 CPU-mesh contracts for operator interpretability."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import analysis.operator_interpretability.runner as runner_module
from analysis.operator_interpretability.artifacts import (
    load_protocol_bound_artifact,
    resolve_benchmark_build,
    write_protocol_bound_artifact,
)
from analysis.operator_interpretability.benchmark_schema import (
    BENCHMARK_SCHEMA,
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkExample,
    canonical_hash,
    validate_examples,
)
from analysis.operator_interpretability.circuit import select_on_validation
from analysis.operator_interpretability.claim_gate import evaluate_claims
from analysis.operator_interpretability.eligibility import tokenize_adapted_pair
from analysis.operator_interpretability.intervention import (
    evaluate_operator_interchange,
)
from analysis.operator_interpretability.protocol import (
    CIRCUIT_FRACTIONS,
    ProtocolConfig,
    protocol_record,
)
from analysis.operator_interpretability.runner import (
    OperatorInterpretabilityRunner,
)
from analysis.operator_interpretability.space import (
    address_confirmation,
    discover_functional_families,
)
from analysis.operator_interpretability.units import (
    OperatorCircuit,
    OperatorSite,
    OperatorSpaceShape,
)
from analysis.train_analysis_pool_items import dependency_closure
from models.dawn_srw_v4171 import (
    DAWN_SRW_V4171,
    make_sharded_srw_minimal,
    make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_suppression_minimal,
)


ROUTES = ("q", "k", "v", "rst")


def _host(value):
    return jax.device_get(value)


def _all_exact(left, right) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(_host(left)),
                        jax.tree.leaves(_host(right)))
    )


def _assert_exact(left, right, label: str) -> None:
    if not _all_exact(left, right):
        differences = []
        for index, (a, b) in enumerate(zip(
                jax.tree.leaves(_host(left)),
                jax.tree.leaves(_host(right)))):
            a_np, b_np = np.asarray(a), np.asarray(b)
            if not np.array_equal(a_np, b_np):
                differences.append((
                    index, a_np.shape,
                    float(np.max(np.abs(
                        a_np.astype(np.float64) - b_np.astype(np.float64))))))
        raise AssertionError(f"{label}: non-exact leaves={differences}")


def _expect_raises(error_types, fn, label: str) -> None:
    try:
        value = fn()
        for leaf in jax.tree.leaves(value):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    except error_types:
        return
    raise AssertionError(f"{label}: expected {error_types}")


def _view(result):
    return {
        "per_token_ce": result["per_token_ce"],
        "final_residual": result["final_residual"],
        "parity_debug": result["parity_debug"],
    }


def _single_route_reference(x, query, keys, read, write, retain, *,
                            autonomous: bool) -> np.ndarray:
    """Independent float32 formula; kernel BF16 boundaries allow 2e-2 error."""
    x = np.array(x, dtype=np.float32, copy=True)
    query = np.array(query, dtype=np.float32, copy=True)
    keys = np.array(keys, dtype=np.float32, copy=True)
    read = np.array(read, dtype=np.float32, copy=True)
    write = np.array(write, dtype=np.float32, copy=True)
    query /= np.linalg.norm(query, axis=-1, keepdims=True)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    read /= np.linalg.norm(read, axis=-1, keepdims=True)
    write /= np.linalg.norm(write, axis=-1, keepdims=True)
    rho = np.einsum("bsd,nd->bsn", query, keys)
    # raw_tau=0 maps to tau=0. Linear-angular admission/execution is rho
    # clipped to [0,1]. The production denominator floor is one.
    admission = np.clip(rho, 0.0, 1.0).astype(np.float32)
    xr = np.einsum("bsd,nd->bsn", x, read)
    keep = np.asarray(retain, dtype=np.bool_)[None, None, :]
    numerator = np.einsum(
        "bsn,nd->bsd", np.where(keep, admission * xr, 0.0), write)
    den_mass = np.where(keep, admission, 0.0).sum(
        axis=-1, keepdims=True) if autonomous else admission.sum(
            axis=-1, keepdims=True)
    denominator = np.maximum(den_mass, 1.0)
    return numerator / denominator


def _build_mesh_and_kernels():
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.size != 4:
        raise RuntimeError(
            f"test requires four CPU devices, found {devices.size}; "
            "set --xla_force_host_platform_device_count=4")
    mesh = Mesh(devices.reshape(2, 2), ("data", "model"))
    production_single = make_sharded_srw_minimal(mesh, max_chunk_size=2)
    suppression_single = make_sharded_srw_suppression_minimal(
        mesh, max_chunk_size=2)
    production_paired = make_sharded_srw_paired_minimal(
        mesh, max_chunk_size=2)
    suppression_paired = make_sharded_srw_paired_suppression_minimal(
        mesh, max_chunk_size=2)
    return (
        mesh, production_single, suppression_single,
        production_paired, suppression_paired,
    )


def test_retention_denominator_reference(production_single) -> None:
    batch, seq, dim = 2, 2, 4
    x = jnp.asarray(np.tile(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (batch, seq, 1)))
    query = jnp.asarray(np.tile(
        np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        (batch, seq, 1)))
    keys = jnp.eye(dim, dtype=jnp.float32)
    read = jnp.eye(dim, dtype=jnp.float32)
    write = jnp.eye(dim, dtype=jnp.float32)
    raw_tau = jnp.zeros((batch, seq, 1), dtype=jnp.float32)
    scalar_args = (
        jnp.float32(0.07), jnp.float32(0.07),
        jnp.float32(2.0), jnp.float32(4.0), jnp.float32(0.0),
    )
    positions = jnp.zeros((batch,), dtype=jnp.int32)
    empty_ids = jnp.full((batch,), -1, dtype=jnp.int32)
    all_positions = jnp.ones((batch, seq), dtype=jnp.bool_)
    all_keep = jnp.ones((dim,), dtype=jnp.bool_)
    one_keep = jnp.asarray([True, False, False, False])
    no_keep = jnp.zeros((dim,), dtype=jnp.bool_)
    kernel = production_single._v4171_canonical_shard_map_kernel

    production = production_single(
        x, query, keys, raw_tau, read, write, *scalar_args)

    def retained(mask, mode):
        return kernel(
            x, query, keys, raw_tau, read, write, *scalar_args,
            empty_ids, positions, jnp.bool_(False), mask,
            all_positions, jnp.int32(mode))

    conditional_all = retained(all_keep, 1)
    autonomous_all = retained(all_keep, 2)
    assert np.array_equal(
        np.asarray(production[0]), np.asarray(conditional_all[0]))
    assert np.array_equal(
        np.asarray(production[0]), np.asarray(autonomous_all[0]))

    conditional = retained(one_keep, 1)
    autonomous = retained(one_keep, 2)
    assert not np.array_equal(
        np.asarray(conditional[0]), np.asarray(autonomous[0]))
    conditional_reference = _single_route_reference(
        x, query, keys, read, write, one_keep, autonomous=False)
    autonomous_reference = _single_route_reference(
        x, query, keys, read, write, one_keep, autonomous=True)
    # The reference is deliberately independent float32 math. The kernel
    # casts each model-shard contribution through BF16 before psum.
    np.testing.assert_allclose(
        np.asarray(conditional[0]), conditional_reference,
        rtol=0.0, atol=2.0e-2)
    np.testing.assert_allclose(
        np.asarray(autonomous[0]), autonomous_reference,
        rtol=0.0, atol=2.0e-2)

    for mode in (1, 2):
        empty = retained(no_keep, mode)
        assert np.array_equal(
            np.asarray(empty[0]), np.zeros_like(np.asarray(empty[0])))
        assert np.all(np.isfinite(np.asarray(empty[0])))

    target_only = jnp.asarray([[False, True], [False, True]])
    positioned = kernel(
        x, query, keys, raw_tau, read, write, *scalar_args,
        empty_ids, positions, jnp.bool_(False), one_keep,
        target_only, jnp.int32(1))
    assert np.array_equal(
        np.asarray(positioned[0][:, 0]), np.asarray(production[0][:, 0]))
    print("RETENTION_DENOMINATOR_REFERENCE_OK")


def _make_model_harness(mesh, production_single, suppression_single,
                        production_paired, suppression_paired):
    batch, seq, dim, layers = 2, 5, 8, 2
    model = DAWN_SRW_V4171(
        vocab_size=32, d_model=dim, d_route=dim,
        n_layers=layers, n_heads=2, max_seq_len=seq,
        dropout_rate=0.0, router_dropout=0.0,
        n_qk=dim, n_v=dim, n_rst=dim,
        tau_init_attn_qk=-0.99, tau_init_attn_v=-0.99,
        tau_init_rst=-0.99)
    base_ids = jnp.asarray([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ], dtype=jnp.int32)
    source_ids = jnp.asarray([
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(4172),
         "dropout": jax.random.PRNGKey(4173)},
        base_ids, deterministic=True)
    sharded_fns = {
        "single": production_single,
        "paired": production_paired,
        "attn_v_single_minimal": production_single,
        "rst_single_minimal": production_single,
        "attn_qk_paired_minimal": production_paired,
        "attn_v_single_suppression_minimal": suppression_single,
        "rst_single_suppression_minimal": suppression_single,
        "attn_qk_paired_suppression_minimal": suppression_paired,
    }
    kwargs = {
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(4174)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "compute_accuracy": False,
        "analysis_parity_debug": True,
    }

    @jax.jit
    def production_and_suppression(
            params, ids, selected_group, positions, layer, route, apply):
        production = model.apply(
            params, ids, labels=ids, minimal_train=True,
            analysis_return_residual=True, **kwargs)
        intervened = model.apply(
            params, ids, selected_group, layer, positions, route,
            labels=ids, apply_suppression=apply, return_residual=True,
            method=model.analysis_forward_with_operator_group_suppression,
            **kwargs)
        return _view(production), _view(intervened)

    @partial(jax.jit, static_argnames=("mode",))
    def retention(params, ids, keep_qk, keep_v, keep_rst, position_mask, *,
                  mode):
        result = model.apply(
            params, ids, keep_qk, keep_v, keep_rst,
            mode=mode, position_mask=position_mask, labels=ids,
            return_residual=True,
            method=model.analysis_forward_with_circuit_retention,
            **kwargs)
        return _view(result)

    @jax.jit
    def capture_and_patch(
            params, base, source, selected_group,
            base_positions, source_positions, layer, route, use_zero_source):
        captured = model.apply(
            params, source, selected_group, layer, source_positions, route,
            labels=source, return_residual=False,
            method=model.analysis_capture_operator_group_contribution,
            **kwargs)
        contributions = captured["operator_route_contributions"]
        source_contribution = sum(
            contributions[name][layer] for name in ROUTES)
        source_contribution = jnp.where(
            use_zero_source,
            jnp.zeros_like(source_contribution), source_contribution)
        patched = model.apply(
            params, base, selected_group, layer, base_positions, route,
            source_contribution, labels=base, return_residual=True,
            method=model.analysis_forward_with_operator_interchange,
            **kwargs)
        return contributions, source_contribution, _view(patched)

    return SimpleNamespace(
        model=model,
        variables=variables,
        base_ids=base_ids,
        source_ids=source_ids,
        production_and_suppression=production_and_suppression,
        retention=retention,
        capture_and_patch=capture_and_patch,
        kwargs=kwargs,
        batch=batch,
        seq=seq,
        dim=dim,
        layers=layers,
    )


def _ones_masks(harness):
    return (
        jnp.ones((harness.layers, 2, harness.dim), dtype=jnp.bool_),
        jnp.ones((harness.layers, harness.dim), dtype=jnp.bool_),
        jnp.ones((harness.layers, harness.dim), dtype=jnp.bool_),
    )


def _zeros_masks(harness):
    return tuple(jnp.zeros_like(value) for value in _ones_masks(harness))


def test_model_retention(harness) -> None:
    group = jnp.tile(
        jnp.arange(harness.dim, dtype=jnp.int32)[None, :],
        (harness.batch, 1))
    positions = jnp.full((harness.batch,), 2, dtype=jnp.int32)
    production, disabled = harness.production_and_suppression(
        harness.variables, harness.base_ids, group, positions,
        jnp.int32(0), jnp.int32(0), jnp.bool_(False))
    _assert_exact(production, disabled, "disabled production parity")

    ones = _ones_masks(harness)
    zeros = _zeros_masks(harness)
    all_positions = jnp.ones(
        (harness.batch, harness.seq), dtype=jnp.bool_)
    no_positions = jnp.zeros_like(all_positions)
    target_positions = jnp.zeros_like(all_positions).at[:, 2].set(True)

    conditional_all = harness.retention(
        harness.variables, harness.base_ids, *ones, all_positions,
        mode="conditional_execution_sufficiency")
    autonomous_all = harness.retention(
        harness.variables, harness.base_ids, *ones, all_positions,
        mode="autonomous_subcircuit_sufficiency")
    _assert_exact(production, conditional_all, "conditional all-ones parity")
    _assert_exact(production, autonomous_all, "autonomous all-ones parity")

    disabled_by_position = harness.retention(
        harness.variables, harness.base_ids, *zeros, no_positions,
        mode="conditional_execution_sufficiency")
    _assert_exact(
        production, disabled_by_position, "all-false position parity")

    conditional_empty = harness.retention(
        harness.variables, harness.base_ids, *zeros, all_positions,
        mode="conditional_execution_sufficiency")
    autonomous_empty = harness.retention(
        harness.variables, harness.base_ids, *zeros, all_positions,
        mode="autonomous_subcircuit_sufficiency")
    for result in (conditional_empty, autonomous_empty):
        assert all(
            np.all(np.isfinite(np.asarray(value)))
            for value in jax.tree.leaves(_host(result)))

    target_empty = harness.retention(
        harness.variables, harness.base_ids, *zeros, target_positions,
        mode="conditional_execution_sufficiency")
    assert np.array_equal(
        np.asarray(_host(production["final_residual"]))[:, :2],
        np.asarray(_host(target_empty["final_residual"]))[:, :2])
    assert not np.array_equal(
        np.asarray(_host(production["final_residual"]))[:, 2],
        np.asarray(_host(target_empty["final_residual"]))[:, 2])

    for route in ROUTES:
        keep = [np.asarray(value).copy() for value in ones]
        if route == "q":
            keep[0][0, 0, :] = False
            unaffected = ("k", "v")
        elif route == "k":
            keep[0][0, 1, :] = False
            unaffected = ("q", "v")
        elif route == "v":
            keep[1][0, :] = False
            unaffected = ("q", "k")
        else:
            keep[2][0, :] = False
            unaffected = ("q", "k", "v")
        result = harness.retention(
            harness.variables, harness.base_ids,
            *(jnp.asarray(value) for value in keep), target_positions,
            mode="conditional_execution_sufficiency")
        base_parity = _host(production["parity_debug"])
        changed_parity = _host(result["parity_debug"])
        assert not np.array_equal(
            np.asarray(base_parity[route][0, :, 2]),
            np.asarray(changed_parity[route][0, :, 2]))
        for sibling in unaffected:
            assert np.array_equal(
                np.asarray(base_parity[sibling][0]),
                np.asarray(changed_parity[sibling][0]))

    layer0 = [np.asarray(value).copy() for value in ones]
    layer1 = [np.asarray(value).copy() for value in ones]
    layer0[0][0, 0, :] = False
    layer1[0][1, 0, :] = False
    changed_layer0 = harness.retention(
        harness.variables, harness.base_ids,
        *(jnp.asarray(value) for value in layer0), target_positions,
        mode="conditional_execution_sufficiency")
    changed_layer1 = harness.retention(
        harness.variables, harness.base_ids,
        *(jnp.asarray(value) for value in layer1), target_positions,
        mode="conditional_execution_sufficiency")
    base_q = np.asarray(_host(production["parity_debug"]["q"]))
    assert not np.array_equal(
        base_q[0, :, 2],
        np.asarray(_host(changed_layer0["parity_debug"]["q"]))[0, :, 2])
    assert np.array_equal(
        base_q[0],
        np.asarray(_host(changed_layer1["parity_debug"]["q"]))[0])

    shape = OperatorSpaceShape(
        harness.layers, harness.dim, harness.dim, harness.dim)
    selected_q = OperatorCircuit(
        sites=tuple(
            OperatorSite(0, "q", operator_id)
            for operator_id in range(harness.dim)),
        discovery_benchmark="tiny").validate(shape)
    selected_masks = selected_q.dense_masks(shape)
    assert all(value.dtype == np.bool_ for value in selected_masks.values())
    complement = {key: ~value for key, value in selected_masks.items()}
    complement_result = harness.retention(
        harness.variables, harness.base_ids,
        jnp.asarray(complement["qk"]), jnp.asarray(complement["v"]),
        jnp.asarray(complement["rst"]), target_positions,
        mode="conditional_execution_sufficiency")
    _, suppressed_q = harness.production_and_suppression(
        harness.variables, harness.base_ids, group, positions,
        jnp.int32(0), jnp.int32(0), jnp.bool_(True))
    _assert_exact(
        complement_result, suppressed_q,
        "necessity complement removes only selected q numerator")

    bad_integer_masks = tuple(
        jnp.ones(value.shape, dtype=jnp.int32) for value in ones)
    _expect_raises(
        (TypeError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, *bad_integer_masks,
            mode="conditional_execution_sufficiency",
            position_mask=all_positions, labels=harness.base_ids,
            method=harness.model.analysis_forward_with_circuit_retention,
            **harness.kwargs),
        "integer retention masks")
    print("MODEL_RETENTION_CONTRACTS_OK")


def _route_target(view, route: str, layer: int, positions) -> np.ndarray:
    values = np.asarray(_host(view["parity_debug"][route]))
    return values[
        layer, np.arange(values.shape[1]), np.asarray(positions), :]


def test_model_interchange(harness) -> None:
    group = jnp.tile(
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        (harness.batch, 1))
    empty_group = jnp.full_like(group, -1)
    positions = jnp.full((harness.batch,), 2, dtype=jnp.int32)
    base_prefix = harness.base_ids[:, :3]
    source_prefix = harness.source_ids[:, :3]

    base_by_route = {}
    base_capture_by_route = {}
    for route_index, route in enumerate(ROUTES):
        production, suppressed = harness.production_and_suppression(
            harness.variables, harness.base_ids, group, positions,
            jnp.int32(0), jnp.int32(route_index), jnp.bool_(True))
        base_by_route[route] = production

        contributions, base_contribution, reinserted = (
            harness.capture_and_patch(
                harness.variables, harness.base_ids, base_prefix, group,
                positions, positions, jnp.int32(0),
                jnp.int32(route_index), jnp.bool_(False)))
        base_capture_by_route[route] = np.asarray(_host(base_contribution))
        _assert_exact(
            production, reinserted,
            f"base contribution reinsertion {route}")

        captured_host = _host(contributions)
        assert base_contribution.dtype == jnp.float32
        for captured_route in ROUTES:
            for layer in range(harness.layers):
                value = np.asarray(captured_host[captured_route][layer])
                if captured_route == route and layer == 0:
                    assert np.array_equal(
                        value, base_capture_by_route[route])
                else:
                    assert np.array_equal(value, np.zeros_like(value))

        base_target = _route_target(production, route, 0, positions)
        suppressed_target = _route_target(suppressed, route, 0, positions)
        assert np.array_equal(
            base_target - suppressed_target,
            base_capture_by_route[route])
        assert np.array_equal(
            base_target - base_capture_by_route[route],
            suppressed_target)
        assert not np.array_equal(base_target, suppressed_target)

        _, zero_contribution, zero_patched = harness.capture_and_patch(
            harness.variables, harness.base_ids, base_prefix, group,
            positions, positions, jnp.int32(0), jnp.int32(route_index),
            jnp.bool_(True))
        assert np.array_equal(
            np.asarray(_host(zero_contribution)),
            np.zeros_like(np.asarray(_host(zero_contribution))))
        _assert_exact(
            suppressed, zero_patched,
            f"zero source equals group suppression {route}")

        _, source_contribution, patched = (
            harness.capture_and_patch(
                harness.variables, harness.base_ids, source_prefix, group,
                positions, positions, jnp.int32(0),
                jnp.int32(route_index), jnp.bool_(False)))
        patched_target = _route_target(patched, route, 0, positions)
        expected_target = (
            base_target.astype(np.float32)
            - base_capture_by_route[route].astype(np.float32)
            + np.asarray(_host(source_contribution)).astype(np.float32))
        assert np.array_equal(patched_target, expected_target)
        route_values = np.asarray(_host(patched["parity_debug"][route]))
        base_route_values = np.asarray(
            _host(production["parity_debug"][route]))
        assert np.array_equal(
            route_values[0, :, :2], base_route_values[0, :, :2])
        assert np.array_equal(
            route_values[0, :, 3:], base_route_values[0, :, 3:])
        assert np.array_equal(
            np.asarray(_host(patched["final_residual"]))[:, :2],
            np.asarray(_host(production["final_residual"]))[:, :2])
        assert not np.array_equal(patched_target, base_target)

        unaffected = (
            ("k", "v") if route == "q" else
            ("q", "v") if route == "k" else
            ("q", "k") if route == "v" else
            ("q", "k", "v"))
        for sibling in unaffected:
            assert np.array_equal(
                np.asarray(_host(production["parity_debug"][sibling]))[0],
                np.asarray(_host(patched["parity_debug"][sibling]))[0])

    empty_contributions, empty_source, empty_patched = (
        harness.capture_and_patch(
            harness.variables, harness.base_ids, source_prefix, empty_group,
            positions, positions, jnp.int32(0), jnp.int32(0),
            jnp.bool_(False)))
    assert empty_source.shape == (harness.batch, harness.dim)
    assert empty_source.dtype == jnp.float32
    assert all(
        np.array_equal(
            np.asarray(value), np.zeros_like(np.asarray(value)))
        for value in jax.tree.leaves(_host(empty_contributions)))
    _assert_exact(
        base_by_route["q"], empty_patched, "empty group exact no-op")

    layer1_contributions, _, layer1_patched = harness.capture_and_patch(
        harness.variables, harness.base_ids, source_prefix, group,
        positions, positions, jnp.int32(1), jnp.int32(0), jnp.bool_(False))
    assert np.array_equal(
        np.asarray(_host(layer1_contributions["q"][0])),
        np.zeros((harness.batch, harness.dim), dtype=np.float32))
    for route in ROUTES:
        assert np.array_equal(
            np.asarray(_host(base_by_route["q"]["parity_debug"][route]))[0],
            np.asarray(_host(layer1_patched["parity_debug"][route]))[0])

    # Reverse direction checks each directional formula, not false symmetry.
    _, a_source_for_b, b_from_a = harness.capture_and_patch(
        harness.variables, harness.source_ids, base_prefix, group,
        positions, positions, jnp.int32(0), jnp.int32(2), jnp.bool_(False))
    b_production, b_suppressed = harness.production_and_suppression(
        harness.variables, harness.source_ids, group, positions,
        jnp.int32(0), jnp.int32(2), jnp.bool_(True))
    _, b_base_capture, _ = harness.capture_and_patch(
        harness.variables, harness.source_ids, source_prefix, group,
        positions, positions, jnp.int32(0), jnp.int32(2), jnp.bool_(False))
    b_target = _route_target(b_production, "v", 0, positions)
    b_suppressed_target = _route_target(
        b_suppressed, "v", 0, positions)
    assert np.array_equal(
        b_target - b_suppressed_target,
        np.asarray(_host(b_base_capture)))
    assert np.array_equal(
        b_target - np.asarray(_host(b_base_capture)),
        b_suppressed_target)
    reverse_expected = (
        b_target.astype(np.float32)
        - np.asarray(_host(b_base_capture)).astype(np.float32)
        + np.asarray(_host(a_source_for_b)).astype(np.float32))
    assert np.array_equal(
        _route_target(b_from_a, "v", 0, positions), reverse_expected)

    bad_kwargs = dict(harness.kwargs)
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids,
            jnp.zeros((harness.batch, 1, 1), dtype=jnp.int32),
            jnp.int32(0), positions, jnp.int32(0),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "group rank")
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids,
            jnp.zeros((harness.batch, 0), dtype=jnp.int32),
            jnp.int32(0), positions, jnp.int32(0),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "zero-width empty representation")
    _expect_raises(
        (TypeError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids,
            jnp.zeros((harness.batch, 1), dtype=jnp.float32),
            jnp.int32(0), positions, jnp.int32(0),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "floating-point operator ids")
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(0), jnp.zeros((harness.batch, 1), dtype=jnp.int32),
            jnp.int32(0), labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "position shape")
    _expect_raises(
        (TypeError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(0), positions, jnp.int32(0),
            jnp.zeros((harness.batch, harness.dim), dtype=jnp.bfloat16),
            labels=harness.base_ids,
            method=harness.model.analysis_forward_with_operator_interchange,
            **bad_kwargs),
        "source dtype")
    _expect_raises(
        (TypeError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(0), positions, jnp.int32(0),
            np.zeros((harness.batch, harness.dim), dtype=np.float64),
            labels=harness.base_ids,
            method=harness.model.analysis_forward_with_operator_interchange,
            **bad_kwargs),
        "source float64 dtype")
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(0), positions, jnp.int32(0),
            jnp.zeros((harness.batch + 1, harness.dim), dtype=jnp.float32),
            labels=harness.base_ids,
            method=harness.model.analysis_forward_with_operator_interchange,
            **bad_kwargs),
        "source batch")

    # Concrete production-hook calls reject values before entering shard_map;
    # the jitted evaluator validates the same request on host before device_put.
    invalid_positions = jnp.full(
        (harness.batch,), harness.seq, dtype=jnp.int32)
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(0), invalid_positions, jnp.int32(0),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "production-hook invalid position")
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(harness.model.n_layers), positions, jnp.int32(0),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "production-hook invalid layer")
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, group,
            jnp.int32(0), positions, jnp.int32(4),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "production-hook invalid route")
    invalid_group = jnp.full(
        (harness.batch, 1), harness.model.n_v, dtype=jnp.int32)
    _expect_raises(
        (ValueError,),
        lambda: harness.model.apply(
            harness.variables, harness.base_ids, invalid_group,
            jnp.int32(0), positions, jnp.int32(2),
            labels=harness.base_ids,
            method=harness.model.analysis_capture_operator_group_contribution,
            **bad_kwargs),
        "production-hook invalid operator id")
    print("MODEL_INTERCHANGE_CONTRACTS_OK")


def _example(phase: str, suffix: str, *, benchmark_id: str = "tiny",
             pair_type: str = "pair", group_id: str | None = None,
             causal_variable: str = "variable",
             source_column: str | None = None):
    metadata = {}
    if group_id is not None:
        metadata["pair_group_id"] = group_id
    source_behavior_required = True
    positive_answer = "yes"
    negative_answer = "no"
    intervention_positive_answer = "source-yes"
    intervention_negative_answer = "source-no"
    intervention_positive_ids = (7,)
    intervention_negative_ids = (8,)
    source_positive_ids = (7,)
    source_negative_ids = (8,)
    if benchmark_id == "ravel":
        source_column = str(source_column)
        other_variable = next(
            value for value in ("Continent", "Country", "Language")
            if value != causal_variable)
        metadata.update({
            "official_counterfactual_column": source_column,
            "base_query_attribute": (
                causal_variable if pair_type == "cause" else other_variable),
            "candidate_score_normalization": (
                "mean_log_probability_per_token"),
        })
        source_behavior_required = (
            source_column == "attribute_counterfactual")
        if not source_behavior_required:
            source_positive_ids = ()
            source_negative_ids = ()
        if pair_type == "cause":
            intervention_positive_answer = negative_answer
            intervention_negative_answer = positive_answer
            intervention_positive_ids = (6,)
            intervention_negative_ids = (5,)
        else:
            intervention_positive_answer = positive_answer
            intervention_negative_answer = negative_answer
            intervention_positive_ids = (5,)
            intervention_negative_ids = (6,)
    return BenchmarkExample(
        benchmark_id=benchmark_id,
        example_id=f"{phase}-{suffix}",
        phase=phase,
        base_prompt="base",
        source_prompt="source",
        positive_answer=positive_answer,
        negative_answer=negative_answer,
        intervention_positive_answer=intervention_positive_answer,
        intervention_negative_answer=intervention_negative_answer,
        causal_variable=causal_variable,
        pair_type=pair_type,
        source_behavior_required=source_behavior_required,
        trace_position_base=1,
        trace_position_source=1,
        input_ids_base=(1, 2),
        input_ids_source=(3, 4),
        positive_ids=(5,),
        negative_ids=(6,),
        source_positive_ids=source_positive_ids,
        source_negative_ids=source_negative_ids,
        intervention_positive_ids=intervention_positive_ids,
        intervention_negative_ids=intervention_negative_ids,
        metadata=metadata,
    )


def test_protocol_and_schema_contracts() -> None:
    assert dependency_closure(["mib_ioi.conditional_circuit_sufficiency"]) == [
        "mib_ioi.input_contract", "mib_ioi.behavioral_eligibility",
        "mib_ioi.operator_localization",
        "mib_ioi.conditional_circuit_sufficiency",
    ]
    closure = dependency_closure(["scientific_claims.primary"])
    assert closure[-1] == "scientific_claims.primary"
    assert "mib_ioi.circuit_necessity" in closure
    assert "mib_arc.autonomous_circuit_sufficiency" in closure
    assert "ravel.causal_mediation" in closure
    assert "ravel.multilayer_trajectory" in closure
    assert len(closure) == len(set(closure))

    validation_rows = [
        {
            "fraction": fraction,
            "site_count": index + 1,
            "faithfulness": 1.0,
            "faithfulness_ci": {
                "ci_low": 0.9 if fraction >= 0.1 else 0.7,
                "ci_high": 1.0,
            },
        }
        for index, fraction in enumerate(CIRCUIT_FRACTIONS)
    ]
    selected = select_on_validation(
        validation_rows, minimum_faithfulness=0.8)
    assert selected["selected_fraction"] == 0.1
    assert selected["test_consulted"] is False
    rejected = select_on_validation([
        {**row, "faithfulness_ci": {"ci_low": 0.0, "ci_high": 1.0}}
        for row in validation_rows
    ], minimum_faithfulness=0.8)
    assert rejected["selected_fraction"] is None

    read = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    write = read.copy()
    family_a = discover_functional_families(
        read, write, neighbor_k=1, similarity_quantile=0.5)
    family_b = discover_functional_families(
        read, write, neighbor_k=1, similarity_quantile=0.5)
    assert family_a["families"] == family_b["families"]
    assert family_a["discovery_features"] == [
        "read_direction", "write_direction"]
    assert family_a["address_used_for_discovery"] is False
    address_confirmation(
        family_a["families"], np.eye(4), seed=1)

    config = replace(
        ProtocolConfig(), bootstrap_samples=100, permutation_samples=100)
    protocol = protocol_record(
        config,
        model_version="spatial-r1-v4.1.7.1",
        benchmark_manifest_hash="benchmark",
        checkpoint_identity="checkpoint",
        model_config_hash="model")

    class Store:
        def __init__(self, root):
            self.root = Path(root)

        def path(self, *parts):
            return str(self.root.joinpath(*parts))

    with tempfile.TemporaryDirectory() as temp_dir:
        store = Store(temp_dir)
        artifact_path = write_protocol_bound_artifact(
            store, "items/value.json", {"status": "ready"},
            protocol=protocol)
        assert load_protocol_bound_artifact(
            artifact_path, protocol=protocol) == {"status": "ready"}
        changed_protocols = [
            protocol_record(
                config,
                model_version="spatial-r1-v4.1.7.1",
                benchmark_manifest_hash="changed",
                checkpoint_identity="checkpoint",
                model_config_hash="model"),
            protocol_record(
                config,
                model_version="spatial-r1-v4.1.7.1",
                benchmark_manifest_hash="benchmark",
                checkpoint_identity="changed",
                model_config_hash="model"),
            protocol_record(
                config,
                model_version="spatial-r1-v4.1.7.1",
                benchmark_manifest_hash="benchmark",
                checkpoint_identity="checkpoint",
                model_config_hash="changed"),
            protocol_record(
                replace(config, capture_threshold=0.9),
                model_version="spatial-r1-v4.1.7.1",
                benchmark_manifest_hash="benchmark",
                checkpoint_identity="checkpoint",
                model_config_hash="model"),
            protocol_record(
                replace(config, circuit_faithfulness_min=0.9),
                model_version="spatial-r1-v4.1.7.1",
                benchmark_manifest_hash="benchmark",
                checkpoint_identity="checkpoint",
                model_config_hash="model"),
            protocol_record(
                replace(config, seed=config.seed + 1),
                model_version="spatial-r1-v4.1.7.1",
                benchmark_manifest_hash="benchmark",
                checkpoint_identity="checkpoint",
                model_config_hash="model"),
        ]
        for changed in changed_protocols:
            _expect_raises(
                (ValueError,),
                lambda changed=changed: load_protocol_bound_artifact(
                    artifact_path, protocol=changed),
                "protocol mismatch resume")

        build_root = Path(temp_dir, "benchmarks")
        manifest_root = build_root / "builds" / "build-a"
        manifest_root.mkdir(parents=True)
        manifest = {
            "schema": BENCHMARK_SCHEMA,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "complete",
            "build_id": "build-a",
            "protocol_id": "p",
            "tokenizer": {
                "name": "tiny", "vocab_hash": "hash",
                "add_special_tokens": False,
            },
            "sources": {"tiny": {"revision": "fixed"}},
            "benchmarks": {
                "tiny": {
                    "path": "tiny.jsonl", "sha256": "hash",
                    "row_count": 3,
                    "phase_counts": {
                        "discovery": 1, "validation": 1, "test": 1},
                }
            },
        }
        (manifest_root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        (build_root / "LATEST.json").write_text(json.dumps({
            "build_id": "build-a",
            "build_path": "builds/build-a",
            "manifest_hash": "wrong",
        }), encoding="utf-8")
        _expect_raises(
            (ValueError,),
            lambda: resolve_benchmark_build(str(build_root)),
            "immutable manifest hash")

    ordinary = [_example(phase, "one") for phase in (
        "discovery", "validation", "test")]
    validate_examples(ordinary)
    _expect_raises(
        (ValueError,), lambda: validate_examples(ordinary[:2]),
        "physical phase separation")
    _expect_raises(
        (ValueError,),
        lambda: validate_examples([ordinary[0], ordinary[0], *ordinary[1:]]),
        "duplicate id")
    _expect_raises(
        (ValueError,),
        lambda: replace(ordinary[0], trace_position_base=2).validate(),
        "invalid trace position")
    _expect_raises(
        (ValueError,),
        lambda: replace(ordinary[0], positive_ids=()).validate(),
        "empty answer")

    ravel_rows = []
    for phase in ("discovery", "validation", "test"):
        for variable in ("Continent", "Country", "Language"):
            for pair_type in ("cause", "isolation"):
                for source_column in (
                        "attribute_counterfactual",
                        "wikipedia_counterfactual"):
                    ravel_rows.append(_example(
                        phase,
                        f"{variable}-{pair_type}-{source_column}",
                        benchmark_id="ravel",
                        pair_type=pair_type,
                        group_id=f"{phase}-{variable}-{pair_type}",
                        causal_variable=variable,
                        source_column=source_column,
                    ))
    validate_examples(ravel_rows)
    _expect_raises(
        (ValueError,),
        lambda: validate_examples(ravel_rows[:-1]),
        "RAVEL cause/isolation group")

    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return SimpleNamespace(input_ids=[1] if text else [])

    _expect_raises(
        (ValueError,),
        lambda: tokenize_adapted_pair(
            TinyTokenizer(), "tiny", "discovery", {
                "example_id": "missing-source-answer",
                "base_prompt": "base",
                "source_prompt": "source",
                "positive_answer": "yes",
                "negative_answer": "no",
                "causal_variable": "v",
                "pair_type": "p",
                "position_kind": "last_token",
            }, max_seq_len=8),
        "missing source answers")

    empty_claims = evaluate_claims({}, config)
    assert empty_claims["strongest_supported_claim"] == "descriptive_only"
    evidence = {
        "capture": {
            "status": "ready", "qualified_fraction": 1.0,
            "rank_stability": 1.0},
        "localization": {"status": "ready"},
        "necessity": {
            "status": "ready", "mean_margin_drop": 1.0,
            "all_significant_after_bh": True},
        "conditional_sufficiency": {
            "status": "ready", "test_faithfulness": 1.0},
        "autonomous_sufficiency": {"status": "incomplete"},
        "interchange": {
            "status": "ready", "cause_success_fraction": 1.0,
            "cause_effect_ci": {"ci_low": 0.5},
            "all_variables_causal_after_bh": True,
            "isolation_absolute_effect_mean": 0.0,
            "isolation_effect_ci": {"ci_high": 0.0},
            "all_variables_isolated": True,
            "all_variables_family_advantage_after_bh": True},
        "held_out": {
            "status": "ready", "selection_phase": "validation",
            "evaluation_phase": "test", "test_used_for_selection": False},
        "spatial_confirmation": {
            "status": "ready", "address_used_for_discovery": False,
            "family_count": 1},
        "trajectory_confirmation": {
            "status": "ready", "same_minus_cross_mean": 1.0,
            "effect_ci": {"ci_low": 0.5},
            "paired_null": {"p_value_two_sided": 0.01}},
    }
    gated = evaluate_claims(evidence, config)
    assert gated["strongest_supported_claim"] == "conditional_sufficiency"
    assert gated["claims"]["interchange_causality"]["passed"] is False
    isolation_failed = dict(evidence)
    isolation_failed["autonomous_sufficiency"] = {
        "status": "ready", "test_faithfulness": 1.0}
    isolation_failed["interchange"] = {
        **evidence["interchange"],
        "isolation_absolute_effect_mean": 1.0,
        "isolation_effect_ci": {"ci_high": 1.0},
        "all_variables_isolated": False,
    }
    gated = evaluate_claims(isolation_failed, config)
    assert gated["strongest_supported_claim"] == "interchange_causality"
    assert gated["claims"]["held_out_generalization"]["passed"] is False
    print("PROTOCOL_SCHEMA_CLAIM_GATES_OK")


class _DummyCircuit:
    def __init__(self, fraction):
        self.fraction = float(fraction)
        self.site_count = max(1, int(round(self.fraction * 1000)))
        self.circuit_hash = f"circuit-{self.fraction}"

    def to_dict(self):
        return {
            "sites": [],
            "metadata": {"candidate_funnel_limited": False},
        }


def _fake_curve_runner(*, allow_selection: bool):
    fake = object.__new__(OperatorInterpretabilityRunner)
    fake.config = replace(
        ProtocolConfig(), minimum_known_correct=2,
        bootstrap_samples=100, permutation_samples=100,
        circuit_faithfulness_min=0.8)
    fake.ctx = SimpleNamespace()
    fake.shape = SimpleNamespace()
    fake.tokenizer = SimpleNamespace(pad_token_id=0)
    validation = [("validation", 0), ("validation", 1)]
    test = [("test", 0), ("test", 1)]
    opened_test = {"count": 0}

    def known_correct(self, benchmark_id, phase):
        del self, benchmark_id
        if phase == "validation":
            return validation
        opened_test["count"] += 1
        if not allow_selection:
            raise AssertionError("test phase opened after validation rejection")
        return test

    def behavior_margins(self, benchmark_id, phase, examples):
        del self, benchmark_id, phase, examples
        return np.asarray([2.0, 2.0]), np.asarray([0.0, 0.0])

    def circuits(self, benchmark_id):
        del self, benchmark_id
        return [
            (fraction, _DummyCircuit(fraction))
            for fraction in CIRCUIT_FRACTIONS]

    fake._known_correct = MethodType(known_correct, fake)
    fake._behavior_margins = MethodType(behavior_margins, fake)
    fake._circuits = MethodType(circuits, fake)
    return fake, opened_test


def test_runner_validation_test_isolation() -> None:
    original_parity = runner_module.all_ones_retention_parity
    original_retention = runner_module.evaluate_circuit_retention
    test_evaluations = {"count": 0}
    try:
        runner_module.all_ones_retention_parity = (
            lambda *args, **kwargs: {"status": "exact"})

        def rejected_retention(ctx, examples, circuit, **kwargs):
            del ctx, examples, circuit, kwargs
            return {
                "mean_margin": 0.0,
                "accuracy": 0.0,
                "margin": [0.0, 0.0],
            }

        runner_module.evaluate_circuit_retention = rejected_retention
        fake, opened = _fake_curve_runner(allow_selection=False)
        rejected = OperatorInterpretabilityRunner._circuit_curve(
            fake, "tiny", mode="conditional_execution_sufficiency")
        assert rejected["status"] == "no_passing_validation_circuit"
        assert opened["count"] == 0
        assert rejected["test"]["rows"] == []

        def selected_retention(ctx, examples, circuit, **kwargs):
            del ctx, kwargs
            if examples[0][0] == "test":
                test_evaluations["count"] += 1
            value = 2.0 if circuit.fraction >= 0.1 else 0.0
            return {
                "mean_margin": value,
                "accuracy": float(value > 0.0),
                "margin": [value, value],
            }

        runner_module.evaluate_circuit_retention = selected_retention
        fake, opened = _fake_curve_runner(allow_selection=True)
        selected = OperatorInterpretabilityRunner._circuit_curve(
            fake, "tiny", mode="conditional_execution_sufficiency")
        assert selected["status"] == "ready"
        assert selected["selection"]["selected_fraction"] == 0.1
        assert opened["count"] == 1
        assert test_evaluations["count"] == 1
        assert len(selected["test"]["rows"]) == 1
        assert selected["test"]["test_used_for_selection"] is False
    finally:
        runner_module.all_ones_retention_parity = original_parity
        runner_module.evaluate_circuit_retention = original_retention

    fake = object.__new__(OperatorInterpretabilityRunner)
    fake.benchmark_ids = ("ravel",)
    fake.results = {
        "behavioral_eligibility": {
            "benchmarks": {
                "ravel": {
                    "phases": {
                        "test": {
                            "eligible_for_mechanistic_claims": False,
                            "known_correct_independent_unit_count": 0,
                        }
                    }
                }
            }
        },
        "operator_localization": {"benchmarks": {"ravel": {"status": "ready"}}},
    }
    trajectory = OperatorInterpretabilityRunner._run_multilayer_trajectory(fake)
    assert trajectory["status"] == "behavior_not_eligible"
    print("RUNNER_VALIDATION_TEST_ISOLATION_OK")


def test_high_level_fail_loud_contract() -> None:
    ctx = SimpleNamespace(model_cfg={
        "n_layers": 2, "n_qk": 8, "n_v": 8, "n_rst": 8,
        "d_model": 8,
    })
    example = _example("test", "high-level")
    _expect_raises(
        (ValueError,),
        lambda: evaluate_operator_interchange(
            ctx, [example], layer=0, route="bad",
            operator_ids=[0], pad_token_id=0),
        "invalid high-level route")
    _expect_raises(
        (ValueError,),
        lambda: evaluate_operator_interchange(
            ctx, [example], layer=2, route="q",
            operator_ids=[0], pad_token_id=0),
        "invalid high-level layer")
    _expect_raises(
        (ValueError,),
        lambda: evaluate_operator_interchange(
            ctx, [example], layer=0, route="v",
            operator_ids=[8], pad_token_id=0),
        "invalid high-level operator id")
    _expect_raises(
        (ValueError,),
        lambda: evaluate_operator_interchange(
            ctx, [example], layer=0, route="v",
            operator_ids=[], pad_token_id=0),
        "high-level empty family")
    print("HIGH_LEVEL_FAIL_LOUD_CONTRACT_OK")


def main() -> None:
    (mesh, production_single, suppression_single,
     production_paired, suppression_paired) = _build_mesh_and_kernels()
    test_retention_denominator_reference(production_single)
    harness = _make_model_harness(
        mesh, production_single, suppression_single,
        production_paired, suppression_paired)
    test_model_retention(harness)
    test_model_interchange(harness)
    test_protocol_and_schema_contracts()
    test_runner_validation_test_isolation()
    test_high_level_fail_loud_contract()
    print("OPERATOR_INTERPRETABILITY_CPU_OK")


if __name__ == "__main__":
    main()
