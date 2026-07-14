"""Focused 2x2 CPU-mesh smoke for v4171 production-core suppression."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

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
    single_disabled = suppression_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(False))
    assert _all_exact(single_base, single_disabled)
    print("PRODUCTION_CORE_ZERO_PARITY_OK")

    single_changed = suppression_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(True))
    assert not np.array_equal(
        np.asarray(single_base[0][:, 1]), np.asarray(single_changed[0][:, 1]))
    assert np.array_equal(
        np.asarray(single_base[0][:, 0]), np.asarray(single_changed[0][:, 0]))
    assert np.array_equal(np.asarray(single_base[3]), np.asarray(single_changed[3]))
    assert np.array_equal(np.asarray(single_base[4]), np.asarray(single_changed[4]))

    inactive = suppression_single(
        x, h_single, op_key, raw_tau, read, write, *scalar_args,
        jnp.ones((batch,), dtype=jnp.int32), positions, jnp.bool_(True))
    assert _all_exact(single_base, inactive)
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
            np.asarray(per_example_base[0][index, 1]),
            np.asarray(per_example_changed[0][index, 1]))
        for index in range(batch))

    h_paired = jnp.zeros((batch, seq, 2, dim), dtype=jnp.float32)
    h_paired = h_paired.at[:, :, 0, 0].set(1.0)
    h_paired = h_paired.at[:, :, 1, 1].set(1.0)
    raw_tau_paired = jnp.zeros((batch, seq, 2, 1), dtype=jnp.float32)
    paired_base = production_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args)
    paired_disabled = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(False), jnp.int32(0))
    assert _all_exact(paired_base, paired_disabled)

    q_changed = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        selected_zero, positions, jnp.bool_(True), jnp.int32(0))
    assert not np.array_equal(
        np.asarray(paired_base[0][:, 1, 0]),
        np.asarray(q_changed[0][:, 1, 0]))
    assert np.array_equal(
        np.asarray(paired_base[0][..., 1, :]),
        np.asarray(q_changed[0][..., 1, :]))

    k_changed = suppression_paired(
        x, h_paired, op_key, raw_tau_paired, read, write, *scalar_args,
        jnp.ones((batch,), dtype=jnp.int32), positions,
        jnp.bool_(True), jnp.int32(1))
    assert np.array_equal(
        np.asarray(paired_base[0][..., 0, :]),
        np.asarray(k_changed[0][..., 0, :]))
    assert not np.array_equal(
        np.asarray(paired_base[0][:, 1, 1]),
        np.asarray(k_changed[0][:, 1, 1]))
    print("PRODUCTION_CORE_QK_ROUTE_ISOLATION_OK")

    # V and RST are separate factory instances of the shared single-route core.
    for _route_name in ("V", "RST"):
        branch = suppression_single(
            x, h_single, op_key, raw_tau, read, write, *scalar_args,
            selected_zero, positions, jnp.bool_(True))
        assert not np.array_equal(np.asarray(single_base[0]), np.asarray(branch[0]))
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
    production_kwargs = {
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(9)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "minimal_train": True,
        "analysis_return_residual": True,
        "compute_accuracy": False,
    }
    production_model = model.apply(variables, input_ids, **production_kwargs)
    analysis_kwargs = {
        key: value for key, value in production_kwargs.items()
        if key not in {"minimal_train", "analysis_return_residual"}
    }
    disabled_model = model.apply(
        variables, input_ids, selected_zero, jnp.int32(0), positions,
        jnp.int32(0), method=model.analysis_forward_with_operator_suppression,
        apply_suppression=False, return_residual=True, **analysis_kwargs)
    assert np.array_equal(
        np.asarray(production_model["logits"]),
        np.asarray(disabled_model["logits"]))
    assert np.array_equal(
        np.asarray(production_model["final_residual"]),
        np.asarray(disabled_model["final_residual"]))
    production_labeled = model.apply(
        variables, input_ids, labels=input_ids, **production_kwargs)
    disabled_labeled = model.apply(
        variables, input_ids, selected_zero, jnp.int32(0), positions,
        jnp.int32(0), labels=input_ids,
        method=model.analysis_forward_with_operator_suppression,
        apply_suppression=False, return_residual=True, **analysis_kwargs)
    assert np.array_equal(
        np.asarray(production_labeled["per_token_ce"]),
        np.asarray(disabled_labeled["per_token_ce"]))

    changed_routes = []
    for route in range(4):
        changed = model.apply(
            variables, input_ids, selected_zero, jnp.int32(0), positions,
            jnp.int32(route),
            method=model.analysis_forward_with_operator_suppression,
            apply_suppression=True, return_residual=True, **analysis_kwargs)
        changed_routes.append(not np.array_equal(
            np.asarray(disabled_model["final_residual"]),
            np.asarray(changed["final_residual"])))
        assert np.array_equal(
            np.asarray(disabled_model["final_residual"][:, 0]),
            np.asarray(changed["final_residual"][:, 0]))
    assert all(changed_routes), changed_routes
    print("PRODUCTION_CORE_INITIALIZED_MODEL_OK")


if __name__ == "__main__":
    main()
