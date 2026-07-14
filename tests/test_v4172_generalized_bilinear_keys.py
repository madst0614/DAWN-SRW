from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax.sharding import Mesh

from analysis.dawn_v4171_transition import (
    V4172_MODEL_VERSION,
    _candidate_pool_vectors,
    run_global_router_audit,
)
from models import dawn_srw_v4171 as core
from models.dawn_srw_v4171 import (
    DAWN_SRW_V4171,
    OPERATOR_KEY_MODE_GENERALIZED_BILINEAR,
    OPERATOR_KEY_MODE_LEARNED,
    _pool_operator_keys,
    _srw_inference,
    make_sharded_srw_minimal,
    make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_suppression_minimal,
    materialize_generalized_bilinear_operator_keys,
    symbolic_parameter_count,
)
from models.dawn_srw_v4172 import DAWN_SRW_V4172


def _tiny_kwargs(n_layers: int = 1) -> dict:
    return {
        "vocab_size": 16,
        "d_model": 4,
        "d_route": 4,
        "n_layers": n_layers,
        "n_heads": 1,
        "max_seq_len": 3,
        "dropout_rate": 0.0,
        "router_dropout": 0.0,
        "n_qk": 4,
        "n_v": 4,
        "n_rst": 4,
        "tau_init_attn_qk": -0.99,
        "tau_init_attn_v": -0.99,
        "tau_init_rst": -0.99,
    }


def _runtime_fns(mesh: Mesh) -> dict:
    single = make_sharded_srw_minimal(mesh, max_chunk_size=2)
    paired = make_sharded_srw_paired_minimal(mesh, max_chunk_size=2)
    single_suppression = make_sharded_srw_suppression_minimal(
        mesh, max_chunk_size=2)
    paired_suppression = make_sharded_srw_paired_suppression_minimal(
        mesh, max_chunk_size=2)
    return {
        "single": single,
        "paired": paired,
        "attn_v_single_minimal": single,
        "rst_single_minimal": single,
        "attn_qk_paired_minimal": paired,
        "attn_v_single_suppression_minimal": single_suppression,
        "rst_single_suppression_minimal": single_suppression,
        "attn_qk_paired_suppression_minimal": paired_suppression,
    }


def _one_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("data", "model"))


def _tree_arrays(tree):
    return [np.asarray(value) for value in jax.tree.leaves(tree)]


def test_generalized_key_shape_norm_sign_and_no_outer_tensor() -> None:
    read = jax.random.normal(jax.random.PRNGKey(1), (5, 7))
    write = jax.random.normal(jax.random.PRNGKey(2), (5, 7))
    read_probe = jax.random.normal(jax.random.PRNGKey(3), (7, 3))
    write_probe = jax.random.normal(jax.random.PRNGKey(4), (7, 3))

    keys = materialize_generalized_bilinear_operator_keys(
        read, write, read_probe, write_probe)
    both_flipped = materialize_generalized_bilinear_operator_keys(
        -read, -write, read_probe, write_probe)
    read_flipped = materialize_generalized_bilinear_operator_keys(
        -read, write, read_probe, write_probe)
    assert keys.shape == (5, 3)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(keys), axis=-1), 1.0,
        rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_array_equal(np.asarray(keys), np.asarray(both_flipped))
    np.testing.assert_allclose(
        np.asarray(read_flipped), -np.asarray(keys), rtol=0.0, atol=0.0)

    jaxpr = jax.make_jaxpr(
        materialize_generalized_bilinear_operator_keys)(
            read, write, read_probe, write_probe)
    shapes = {
        tuple(getattr(var.aval, "shape", ()))
        for equation in jaxpr.jaxpr.eqns
        for var in (*equation.invars, *equation.outvars)
        if hasattr(var, "aval")
    }
    assert (5, 3, 3) not in shapes


def test_generalized_key_has_live_gradients_for_rw_and_both_probes() -> None:
    read = jax.random.normal(jax.random.PRNGKey(11), (5, 7))
    write = jax.random.normal(jax.random.PRNGKey(12), (5, 7))
    read_probe = jax.random.normal(jax.random.PRNGKey(13), (7, 3))
    write_probe = jax.random.normal(jax.random.PRNGKey(14), (7, 3))
    weights = jax.random.normal(jax.random.PRNGKey(15), (5, 3))

    def loss(*args):
        return jnp.sum(
            materialize_generalized_bilinear_operator_keys(*args) * weights)

    gradients = jax.grad(loss, argnums=(0, 1, 2, 3))(
        read, write, read_probe, write_probe)
    for gradient in gradients:
        assert np.all(np.isfinite(np.asarray(gradient)))
        assert float(jnp.linalg.norm(gradient)) > 0.0


def test_v4171_and_v4172_parameter_schemas_and_symbolic_tiny_count() -> None:
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    kwargs = _tiny_kwargs()
    learned = DAWN_SRW_V4171(**kwargs)
    generalized = DAWN_SRW_V4172(**kwargs)
    learned_params = learned.init(
        {"params": jax.random.PRNGKey(21),
         "dropout": jax.random.PRNGKey(22)},
        input_ids, deterministic=True)["params"]
    generalized_params = generalized.init(
        {"params": jax.random.PRNGKey(21),
         "dropout": jax.random.PRNGKey(22)},
        input_ids, deterministic=True)["params"]

    learned_pool = learned_params["neuron_pool"]
    generalized_pool = generalized_params["neuron_pool"]
    assert set(learned_pool) == {
        "attn_qk_read", "attn_qk_write", "attn_qk_op_key",
        "attn_v_read", "attn_v_write", "attn_v_op_key",
        "rst_read", "rst_write", "rst_op_key",
    }
    assert set(generalized_pool) == {
        "attn_qk_read", "attn_qk_write",
        "attn_v_read", "attn_v_write",
        "rst_read", "rst_write",
        "rw_key_read_probe", "rw_key_write_probe",
    }
    assert generalized_pool["rw_key_read_probe"].shape == (4, 4)
    assert generalized_pool["rw_key_write_probe"].shape == (4, 4)

    for model, params in (
            (learned, learned_params), (generalized, generalized_params)):
        model_cfg = {
            **kwargs,
            "operator_key_mode": model.operator_key_mode,
        }
        assert symbolic_parameter_count(model_cfg)["total"] == sum(
            value.size for value in jax.tree.leaves(params))


def test_v4171_default_and_explicit_learned_mode_are_machine_exact() -> None:
    mesh = _one_device_mesh()
    sharded_fns = _runtime_fns(mesh)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    default_model = DAWN_SRW_V4171(**_tiny_kwargs())
    explicit_model = DAWN_SRW_V4171(
        **_tiny_kwargs(), operator_key_mode=OPERATOR_KEY_MODE_LEARNED)
    rngs = {
        "params": jax.random.PRNGKey(31),
        "dropout": jax.random.PRNGKey(32),
    }
    default_variables = default_model.init(rngs, input_ids, deterministic=True)
    explicit_variables = explicit_model.init(rngs, input_ids, deterministic=True)
    assert jax.tree.structure(default_variables["params"]) == jax.tree.structure(
        explicit_variables["params"])
    for default, explicit in zip(
            _tree_arrays(default_variables["params"]),
            _tree_arrays(explicit_variables["params"])):
        np.testing.assert_array_equal(default, explicit)

    apply_kwargs = {
        "labels": input_ids,
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(33)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "minimal_train": True,
        "compute_accuracy": False,
    }
    default_out = default_model.apply(
        default_variables, input_ids, **apply_kwargs)
    explicit_out = explicit_model.apply(
        explicit_variables, input_ids, **apply_kwargs)
    assert jax.tree.structure(default_out) == jax.tree.structure(explicit_out)
    for default, explicit in zip(
            _tree_arrays(default_out), _tree_arrays(explicit_out)):
        np.testing.assert_array_equal(default, explicit)


def test_v4172_query_attention_and_key_dimensions_forward_once_and_parity(
        monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _one_device_mesh()
    sharded_fns = _runtime_fns(mesh)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    model = DAWN_SRW_V4172(**_tiny_kwargs(n_layers=2))
    variables = model.init(
        {"params": jax.random.PRNGKey(41),
         "dropout": jax.random.PRNGKey(42)},
        input_ids, deterministic=True)
    pool = variables["params"]["neuron_pool"]
    operator_keys = _pool_operator_keys(
        pool, OPERATOR_KEY_MODE_GENERALIZED_BILINEAR)
    state = jnp.ones((1, 3, 4), dtype=jnp.float32)
    operator_query = state @ variables["params"]["router"][
        "proj_attn"]["kernel"][:, :4]
    attention_q = _srw_inference(
        state, operator_query, operator_keys["attn_qk_op_key"],
        jnp.full((1, 3, 1), -4.0), jnp.zeros((1, 3, 1)),
        pool["attn_qk_read"], pool["attn_qk_write"],
        admission_den_power=1.0,
        soft_gate_temperature=0.07,
        soft_gate_boundary_power=4.0,
        execution_prune_eps=0.0,
        soft_gate_effective_active_eps=1.0e-6,
        srw_composition_mode="linear_angular",
        heat_kernel_beta=2.0,
    )
    assert operator_query.shape[-1] == model.d_route
    assert attention_q.shape[-1] == model.d_model
    assert all(value.shape[-1] == model.d_route
               for value in operator_keys.values())

    call_count = 0
    original = core._pool_operator_keys

    def counted(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_pool_operator_keys", counted)
    common = {
        "labels": input_ids,
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(43)},
        "sharded_fns": sharded_fns,
        "analysis": False,
        "minimal_train": True,
        "analysis_target_layer": jnp.int32(0),
        "analysis_target_positions": jnp.asarray([1], dtype=jnp.int32),
        "analysis_target_route": jnp.int32(2),
        "analysis_return_residual": True,
        "analysis_return_logits": True,
        "compute_accuracy": False,
    }
    disabled = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.asarray([0], dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(False),
        **common)
    assert call_count == 1
    assert np.all(np.isfinite(np.asarray(disabled["logits"])))
    empty_group = model.apply(
        variables, input_ids,
        analysis_contribution=jnp.full((1, 8), -1, dtype=jnp.int32),
        analysis_intervention_enabled=jnp.bool_(True),
        **common)
    np.testing.assert_array_equal(
        np.asarray(disabled["logits"]), np.asarray(empty_group["logits"]))
    np.testing.assert_array_equal(
        np.asarray(disabled["final_residual"]),
        np.asarray(empty_group["final_residual"]))


def test_pool_operator_key_schema_mismatch_fails_loud() -> None:
    read = jnp.ones((2, 4), dtype=jnp.float32)
    probe = jnp.eye(4, dtype=jnp.float32)
    mixed = {
        "attn_qk_read": read, "attn_qk_write": read,
        "attn_v_read": read, "attn_v_write": read,
        "rst_read": read, "rst_write": read,
        "attn_qk_op_key": read, "attn_v_op_key": read,
        "rst_op_key": read,
        "rw_key_read_probe": probe, "rw_key_write_probe": probe,
    }
    with pytest.raises(ValueError, match="mixes learned operator key tables"):
        _pool_operator_keys(mixed)


def test_v4171_v4172_cross_version_resume_fails_loud() -> None:
    from scripts import train_jax

    v4171 = {"model_version": train_jax.V4171_MODEL_VERSION}
    v4172 = {"model_version": train_jax.V4172_MODEL_VERSION}
    with pytest.raises(RuntimeError, match="exact same model version"):
        train_jax._validate_v4171_resume_compatibility(v4172, v4171)
    with pytest.raises(RuntimeError, match="exact same model version"):
        train_jax._validate_v4171_resume_compatibility(v4171, v4172)


def test_400m_symbolic_counts_match_spec() -> None:
    expected = {
        "configs/train_config_v4171_400M_c4_40B_v4_64.yaml": 393_804_804,
        "configs/train_config_v4172_400M_c4_40B_v4_64_same_pool.yaml": 383_183_876,
        "configs/train_config_v4172_400M_c4_40B_v4_64.yaml": 393_800_708,
    }
    for path, count in expected.items():
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert symbolic_parameter_count(config)["total"] == count


def test_v4172_analysis_candidate_address_and_router_provenance() -> None:
    mesh = _one_device_mesh()
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    model = DAWN_SRW_V4172(**_tiny_kwargs())
    params = model.init(
        {"params": jax.random.PRNGKey(51),
         "dropout": jax.random.PRNGKey(52)},
        input_ids, deterministic=True)["params"]
    model_cfg = {
        **_tiny_kwargs(),
        "model_version": V4172_MODEL_VERSION,
        "operator_key_mode": OPERATOR_KEY_MODE_GENERALIZED_BILINEAR,
        "srw_composition_mode": "linear_angular",
        "admission_den_power": 1.0,
    }
    ctx = SimpleNamespace(
        mesh=mesh,
        model_cfg=model_cfg,
        config={"model": model_cfg},
        params=params,
        is_primary=False,
    )
    selected = _candidate_pool_vectors(ctx, "qk", [0, 2])
    pool = params["neuron_pool"]
    selected_ids = jnp.asarray([0, 2], dtype=jnp.int32)
    expected = materialize_generalized_bilinear_operator_keys(
        pool["attn_qk_read"][selected_ids],
        pool["attn_qk_write"][selected_ids],
        pool["rw_key_read_probe"],
        pool["rw_key_write_probe"],
    )
    np.testing.assert_allclose(selected["address"], np.asarray(expected))
    audit = run_global_router_audit(ctx)
    assert audit["operator_key_mode"] == OPERATOR_KEY_MODE_GENERALIZED_BILINEAR
    assert audit["operator_key_source"] == "live_rw_plus_shared_probes"
    assert audit["learned_operator_key_tables"] is False
    assert audit["shared_probe_matrices"] is True
    assert audit["probe_scope"] == "qk_v_rst_global"
    assert audit["operator_keys_shared_across_layers"] is True
    assert audit["operator_rw_shared_across_layers"] is True
    assert audit["operator_key_probe_parameter_count"] == 32
    assert audit["learned_operator_key_parameter_count"] == 0


def test_v4172_probe_sharding_is_replicated_and_rw_axis_is_model_sharded() -> None:
    from scripts import train_jax

    mesh = _one_device_mesh()
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    params = DAWN_SRW_V4172(**_tiny_kwargs()).init(
        {"params": jax.random.PRNGKey(61),
         "dropout": jax.random.PRNGKey(62)},
        input_ids, deterministic=True)["params"]
    shardings = train_jax.get_param_shardings(
        params, mesh, V4172_MODEL_VERSION)
    pool = shardings["neuron_pool"]
    assert str(pool["rw_key_read_probe"].spec) == "PartitionSpec()"
    assert str(pool["rw_key_write_probe"].spec) == "PartitionSpec()"
    assert str(pool["attn_qk_read"].spec) == (
        "PartitionSpec('model', None)")
    assert str(pool["attn_qk_write"].spec) == (
        "PartitionSpec('model', None)")
