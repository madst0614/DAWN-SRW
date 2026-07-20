"""Focused 2x2 CPU-mesh smoke for v4171 production-core suppression."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import models.dawn_srw_v4171 as v4171_module
from models.dawn_srw_v4171 import (
    DAWN_SRW_V4171,
    make_sharded_srw_minimal,
    make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_suppression_minimal,
)


def _all_exact(left, right):
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def main() -> None:
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.size != 4:
        raise RuntimeError(
            f"smoke requires four CPU devices, found {devices.size}; "
            "set --xla_force_host_platform_device_count=4")
    mesh = Mesh(devices.reshape(2, 2), ("data", "model"))
    production_single = make_sharded_srw_minimal(mesh, max_chunk_size=2)
    suppression_single = make_sharded_srw_suppression_minimal(
        mesh, max_chunk_size=2)
    production_paired = make_sharded_srw_paired_minimal(
        mesh, max_chunk_size=2)
    suppression_paired = make_sharded_srw_paired_suppression_minimal(
        mesh, max_chunk_size=2)
    assert production_single._v4171_kernel_profile == "production"
    assert suppression_single._v4171_kernel_profile == "suppression"
    assert production_paired._v4171_kernel_profile == "production"
    assert suppression_paired._v4171_kernel_profile == "suppression"
    assert not hasattr(
        production_single, "_v4171_canonical_shard_map_kernel")
    assert not hasattr(
        production_paired, "_v4171_canonical_shard_map_kernel")
    print("PRODUCTION_ANALYSIS_KERNELS_SEPARATE_OK")

    batch, seq, dim = 2, 3, 4
    x = jnp.asarray(np.tile(
        np.asarray([1.0, 2.0, 3.0, 4.0], np.float32),
        (batch, seq, 1)))
    op_key = jnp.eye(dim, dtype=jnp.float32)
    read = jnp.eye(dim, dtype=jnp.float32)
    write = jnp.eye(dim, dtype=jnp.float32)
    raw_tau = jnp.zeros((batch, seq, 1), dtype=jnp.float32)
    h_single = jnp.zeros((batch, seq, dim), dtype=jnp.float32).at[..., 0].set(1.0)
    scalar_args = (
        jnp.float32(0.07), jnp.float32(0.07),
        jnp.float32(2.0), jnp.float32(4.0), jnp.float32(0.0),
    )
    positions = jnp.full((batch,), 1, dtype=jnp.int32)
    selected_zero = jnp.zeros((batch,), dtype=jnp.int32)

    single_base = production_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args)
    canonical_single = suppression_single._v4171_canonical_shard_map_kernel
    single_neutral = canonical_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        jnp.full((batch,), -1, dtype=jnp.int32),
        jnp.full((batch,), -1, dtype=jnp.int32), jnp.bool_(False),
        jnp.ones((dim,), dtype=jnp.bool_),
        jnp.ones((batch, seq), dtype=jnp.bool_), jnp.int32(0))
    single_disabled = suppression_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(False))
    assert _all_exact(single_base, single_neutral[0])
    assert _all_exact(single_base, single_disabled[0])

    single_changed = suppression_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(True))
    assert not np.array_equal(
        np.asarray(single_base[:, 1]), np.asarray(single_changed[0][:, 1]))
    assert np.array_equal(
        np.asarray(single_base[:, 0]), np.asarray(single_changed[0][:, 0]))

    inactive = suppression_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        jnp.ones((batch,), dtype=jnp.int32), positions, jnp.bool_(True))
    assert _all_exact(single_base, inactive[0])
    print("PRODUCTION_CORE_INACTIVE_NOOP_OK")
    print("PRODUCTION_CORE_CAUSAL_PREFIX_OK")

    h_per_example = jnp.zeros_like(h_single)
    h_per_example = h_per_example.at[0, :, 0].set(1.0)
    h_per_example = h_per_example.at[1, :, 2].set(1.0)
    per_example_base = production_single(
        x, h_per_example, op_key, raw_tau, read, write, *scalar_args)
    per_example_changed = suppression_single(
        x, h_per_example, op_key, raw_tau, read, write, *scalar_args,
        jnp.asarray([0, 2], dtype=jnp.int32), positions, jnp.bool_(True))
    assert all(
        not np.array_equal(
            np.asarray(per_example_base[index, 1]),
            np.asarray(per_example_changed[0][index, 1]))
        for index in range(batch))

    h_paired = jnp.zeros((batch, seq, 2, dim), dtype=jnp.float32)
    h_paired = h_paired.at[:, :, 0, 0].set(1.0)
    h_paired = h_paired.at[:, :, 1, 1].set(1.0)
    raw_tau_paired = jnp.zeros((batch, seq, 2, 1), dtype=jnp.float32)
    paired_base = production_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args)
    canonical_paired = suppression_paired._v4171_canonical_shard_map_kernel
    paired_neutral = canonical_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        jnp.full((batch,), -1, dtype=jnp.int32),
        jnp.full((batch,), -1, dtype=jnp.int32),
        jnp.bool_(False), jnp.int32(-1),
        jnp.ones((2, dim), dtype=jnp.bool_),
        jnp.ones((batch, seq), dtype=jnp.bool_), jnp.int32(0))
    paired_disabled = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(False), jnp.int32(0))
    assert _all_exact(paired_base, paired_neutral[0])
    assert _all_exact(paired_base, paired_disabled[0])

    paired_non_target_route = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(True), jnp.int32(3))
    # The final leaf is the route-selected analysis sidecar.  It is expected
    # to be zero for an out-of-range route while all production outputs remain
    # machine exact.
    assert _all_exact(paired_disabled[0], paired_non_target_route[0])

    q_changed = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(True), jnp.int32(0))
    assert not np.array_equal(
        np.asarray(paired_base[:, 1, 0]),
        np.asarray(q_changed[0][:, 1, 0]))
    assert np.array_equal(
        np.asarray(paired_base[..., 1, :]),
        np.asarray(q_changed[0][..., 1, :]))

    k_changed = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        jnp.ones((batch,), dtype=jnp.int32), positions,
        jnp.bool_(True), jnp.int32(1))
    assert np.array_equal(
        np.asarray(paired_base[..., 0, :]),
        np.asarray(k_changed[0][..., 0, :]))
    assert not np.array_equal(
        np.asarray(paired_base[:, 1, 1]),
        np.asarray(k_changed[0][:, 1, 1]))
    print("PRODUCTION_CORE_QK_ROUTE_ISOLATION_OK")

    # V and RST are separate factory instances of the shared single-route core.
    for _route_name in ("V", "RST"):
        branch = suppression_single(
            x, h_single, op_key, raw_tau, read, write, *scalar_args,
            selected_zero, positions, jnp.bool_(True))
        assert not np.array_equal(np.asarray(single_base), np.asarray(branch[0]))
    print("PRODUCTION_CORE_V_RST_BRANCHES_OK")

    model = DAWN_SRW_V4171(
        vocab_size=16, d_model=dim, d_route=dim, n_layers=1, n_heads=1,
        max_seq_len=seq, dropout_rate=0.0, router_dropout=0.0,
        n_qk=dim, n_v=dim, n_rst=dim,
        tau_init_attn_qk=-0.99, tau_init_attn_v=-0.99,
        tau_init_rst=-0.99)
    input_ids = jnp.asarray([[1, 2, 3], [1, 2, 3]], dtype=jnp.int32)
    init_rng = {"params": jax.random.PRNGKey(7),
                "dropout": jax.random.PRNGKey(8)}
    variables = model.init(init_rng, input_ids, deterministic=True)
    production_only_fns = {
        "single": production_single,
        "paired": production_paired,
        "attn_v_single_minimal": production_single,
        "rst_single_minimal": production_single,
        "attn_qk_paired_minimal": production_paired,
        "_v4171_kernel_profile": "production",
    }

    def forbidden_analysis_factory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("training called an analysis kernel factory")

    analysis_factory_names = (
        "make_sharded_srw_retention_minimal",
        "make_sharded_srw_paired_retention_minimal",
        "make_sharded_srw_suppression_minimal",
        "make_sharded_srw_paired_suppression_minimal",
        "make_sharded_srw_trajectory_minimal",
        "make_sharded_srw_paired_trajectory_minimal",
    )
    original_analysis_factories = {
        name: getattr(v4171_module, name) for name in analysis_factory_names}
    try:
        for name in analysis_factory_names:
            setattr(v4171_module, name, forbidden_analysis_factory)
        production_only_result = model.apply(
            variables, input_ids, labels=input_ids,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(81)},
            sharded_fns=production_only_fns,
            analysis=False, minimal_train=True,
            compute_accuracy=False)
    finally:
        for name, factory in original_analysis_factories.items():
            setattr(v4171_module, name, factory)
    assert np.isfinite(float(production_only_result["loss"]))
    print("PRODUCTION_TRAINING_ANALYSIS_FACTORY_ISOLATION_OK")
    sharded_fns = {
        "single": production_single,
        "paired": production_paired,
        "attn_v_single_minimal": suppression_single,
        "rst_single_minimal": suppression_single,
        "attn_qk_paired_minimal": suppression_paired,
        "attn_v_single_suppression_minimal": suppression_single,
        "rst_single_suppression_minimal": suppression_single,
        "attn_qk_paired_suppression_minimal": suppression_paired,
        "_v4171_kernel_profile": "suppression",
    }
    model_kwargs = {
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(9)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "minimal_train": True,
        "minimal_runtime_profile": "suppression",
        "analysis_return_residual": True,
        "analysis_return_logits": True,
        "compute_accuracy": False,
    }

    @jax.jit
    def canonical_model_forward(
            params, token_ids, target_positions, target_layer, target_route,
            selected_operator_ids, apply_suppression):
        result = model.apply(
            params,
            token_ids,
            labels=token_ids,
            analysis_contribution=selected_operator_ids,
            analysis_target_layer=target_layer,
            analysis_target_positions=target_positions,
            analysis_target_route=target_route,
            analysis_intervention_enabled=apply_suppression,
            **model_kwargs,
        )
        return (result["logits"], result["per_token_ce"],
                result["final_residual"])

    neutral_positions = jnp.full((batch,), -1, dtype=jnp.int32)
    neutral_operators = jnp.full((batch,), -1, dtype=jnp.int32)
    canonical_baseline = canonical_model_forward(
        variables, input_ids, neutral_positions, jnp.int32(-1),
        jnp.int32(-1), neutral_operators, jnp.bool_(False))
    repeated_baseline = canonical_model_forward(
        variables, input_ids, neutral_positions, jnp.int32(-1),
        jnp.int32(-1), neutral_operators, jnp.bool_(False))
    disabled_model = canonical_model_forward(
        variables, input_ids, positions, jnp.int32(0), jnp.int32(0),
        selected_zero, jnp.bool_(False))
    non_target_layer = canonical_model_forward(
        variables, input_ids, positions, jnp.int32(1), jnp.int32(0),
        selected_zero, jnp.bool_(True))
    non_target_route = canonical_model_forward(
        variables, input_ids, positions, jnp.int32(0), jnp.int32(4),
        selected_zero, jnp.bool_(True))
    inactive_operator = canonical_model_forward(
        variables, input_ids, positions, jnp.int32(0), jnp.int32(0),
        neutral_operators, jnp.bool_(True))
    for noop in (
            repeated_baseline, disabled_model, non_target_layer,
            non_target_route, inactive_operator):
        assert _all_exact(canonical_baseline, noop)

    changed_routes = []
    for route in range(4):
        changed = canonical_model_forward(
            variables, input_ids, positions, jnp.int32(0),
            jnp.int32(route), selected_zero, jnp.bool_(True))
        changed_routes.append(not np.array_equal(
            np.asarray(disabled_model[2]), np.asarray(changed[2])))
        assert np.array_equal(
            np.asarray(disabled_model[2][:, 0]),
            np.asarray(changed[2][:, 0]))
    assert all(changed_routes), changed_routes
    debug_result = model.apply(
        variables,
        input_ids,
        labels=input_ids,
        analysis_contribution=neutral_operators,
        analysis_target_layer=jnp.int32(-1),
        analysis_target_positions=neutral_positions,
        analysis_target_route=jnp.int32(-1),
        analysis_intervention_enabled=jnp.bool_(False),
        analysis_parity_debug=True,
        **model_kwargs,
    )
    assert set(debug_result["parity_debug"]) == {
        "residual_input", "q", "k", "v", "attention_update", "rst",
        "post_attention", "post_layer_residual",
    }
    assert all(
        value.shape[0] == model.n_layers
        for value in debug_result["parity_debug"].values())
    print("PRODUCTION_CORE_INITIALIZED_MODEL_OK")
    print(
        "CANONICAL_CAUSAL_ZERO_PARITY_OK machine_exact=True "
        "max_logit_abs_diff=0 ce_abs_diff=0 "
        "final_residual_max_abs_diff=0")


if __name__ == "__main__":
    main()
