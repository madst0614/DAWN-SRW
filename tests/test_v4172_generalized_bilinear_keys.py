from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax.sharding import Mesh, PartitionSpec

from analysis.operator_interpretability.space import (
    candidate_pool_vectors,
    operator_pool_provenance,
)
from models import dawn_srw_v4171 as core
from models.dawn_srw_v4171 import (
    DAWN_SRW_V4171,
    OPERATOR_KEY_MODE_GENERALIZED_BILINEAR,
    OPERATOR_KEY_MODE_LEARNED,
    _composition_den,
    _pool_operator_keys,
    _resolve_v417x_admission_den_powers,
    _split_admission_den_kwargs,
    _srw_inference,
    _validate_v4171_sharded_fns,
    make_sharded_srw,
    make_sharded_srw_minimal,
    make_sharded_srw_paired,
    make_sharded_srw_paired_minimal,
    make_sharded_srw_paired_suppression_minimal,
    make_sharded_srw_suppression_minimal,
    materialize_generalized_bilinear_operator_keys,
    symbolic_parameter_count,
)
from models.dawn_srw_v4172 import (
    DAWN_SRW_V4172,
    MODEL_VERSION as V4172_MODEL_VERSION,
)


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


def _runtime_fns(
        mesh: Mesh, *, qk_power: float = 1.0, v_power: float = 1.0,
        rst_power: float = 1.0, analysis: bool = False,
        kernel_profile: str = "production") -> dict:
    single_v_full = make_sharded_srw(
        mesh, max_chunk_size=2, analysis=analysis,
        admission_den_power=v_power)
    single_rst_full = make_sharded_srw(
        mesh, max_chunk_size=2, analysis=analysis,
        admission_den_power=rst_power)
    paired_full = make_sharded_srw_paired(
        mesh, max_chunk_size=2, analysis=analysis,
        admission_den_power=qk_power)
    if kernel_profile == "production":
        single_min_factory = make_sharded_srw_minimal
        paired_min_factory = make_sharded_srw_paired_minimal
    elif kernel_profile == "suppression":
        single_min_factory = make_sharded_srw_suppression_minimal
        paired_min_factory = make_sharded_srw_paired_suppression_minimal
    else:
        raise ValueError(f"unsupported test kernel profile={kernel_profile}")
    single_qk = single_min_factory(
        mesh, max_chunk_size=2, admission_den_power=qk_power)
    single_v = single_min_factory(
        mesh, max_chunk_size=2, admission_den_power=v_power)
    single_rst = single_min_factory(
        mesh, max_chunk_size=2, admission_den_power=rst_power)
    paired = paired_min_factory(
        mesh, max_chunk_size=2, admission_den_power=qk_power)
    result = {
        "single": single_v_full,
        "attn_v_single": single_v_full,
        "rst_single": single_rst_full,
        "paired": paired_full,
        "attn_qk_paired": paired_full,
        "attn_qk_single_minimal": single_qk,
        "attn_v_single_minimal": single_v,
        "rst_single_minimal": single_rst,
        "attn_qk_paired_minimal": paired,
        "_v4171_kernel_profile": kernel_profile,
    }
    if kernel_profile == "suppression":
        result.update({
            "attn_v_single_suppression_minimal": single_v,
            "rst_single_suppression_minimal": single_rst,
            "attn_qk_paired_suppression_minimal": paired,
        })
    return result


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

    from scripts import train_jax

    diagnostics = train_jax._shared_probe_gradient_diagnostics(
        {
            "rw_key_read_probe": read_probe,
            "rw_key_write_probe": write_probe,
        },
        {
            "rw_key_read_probe": gradients[2],
            "rw_key_write_probe": gradients[3],
        },
        train_jax.V4172_MODEL_VERSION,
    )
    read_grad = float(diagnostics["grad_pool_shared_read_probe"])
    write_grad = float(diagnostics["grad_pool_shared_write_probe"])
    combined_grad = float(diagnostics["grad_pool_shared_key_probe"])
    assert read_grad > 0.0
    assert write_grad > 0.0
    assert combined_grad == pytest.approx(
        np.sqrt(read_grad**2 + write_grad**2))
    aliases = train_jax._canonical_pool_op_key_grad_norms(
        {
            "rw_key_read_probe": gradients[2],
            "rw_key_write_probe": gradients[3],
        },
        train_jax.V4172_MODEL_VERSION,
    )
    assert float(aliases["attn_qk"]) == pytest.approx(combined_grad)
    assert float(aliases["attn_v"]) == 0.0
    assert float(aliases["rst"]) == 0.0

    v4171_shared = train_jax._shared_probe_gradient_diagnostics(
        {}, {}, train_jax.V4171_MODEL_VERSION)
    assert all(float(value) == 0.0 for value in v4171_shared.values())
    learned_gradients = {
        "attn_qk_op_key": gradients[0],
        "attn_v_op_key": gradients[1],
        "rst_op_key": gradients[0],
    }
    learned_aliases = train_jax._canonical_pool_op_key_grad_norms(
        learned_gradients, train_jax.V4171_MODEL_VERSION)
    assert all(float(value) > 0.0 for value in learned_aliases.values())


def test_compact_metrics_prefer_v4172_shared_probe_and_preserve_v4171() -> None:
    from scripts import train_jax

    zero = jnp.float32(0.0)
    result = {
        key: zero
        for key in (
            *train_jax.LINEAR_DIRECT_TAU_REGULAR_REQUIRED_METRIC_NAMES,
            *train_jax.V4171_COMPOSITION_METRIC_NAMES,
        )
    }
    result.update({"correct": zero, "valid_count": jnp.float32(1.0)})
    shared = {
        key: jnp.float32(index + 1)
        for index, key in enumerate(
            train_jax.V417X_SHARED_PROBE_GRADIENT_METRIC_NAMES)
    }
    v4172_metrics = {
        **shared,
        "grad_pool_attn_qk_op_key": shared["grad_pool_shared_key_probe"],
        "grad_pool_attn_v_op_key": zero,
        "grad_pool_rst_op_key": zero,
    }

    def compact(model_version, diagnostics):
        return train_jax._v4170_compact_train_metrics(
            result,
            total_loss=zero,
            ce_loss=zero,
            aux_loss=zero,
            tau_reg=zero,
            orth_loss=zero,
            div_loss=zero,
            grad_norm=zero,
            tau_lr_mult=zero,
            tau_update_qk_max_abs=zero,
            tau_update_v_max_abs=zero,
            tau_update_rst_max_abs=zero,
            model_version=model_version,
            operator_key_gradient_metrics=diagnostics,
        )

    generalized = compact(train_jax.V4172_MODEL_VERSION, v4172_metrics)
    assert generalized["grad_pool_shared_key_probe"] == shared[
        "grad_pool_shared_key_probe"]
    assert generalized["grad_pool_attn_qk_op_key"] == generalized[
        "grad_pool_shared_key_probe"]
    assert generalized["grad_pool_attn_v_op_key"] == 0.0
    assert generalized["grad_pool_rst_op_key"] == 0.0

    learned_shared = {key: zero for key in shared}
    learned = compact(train_jax.V4171_MODEL_VERSION, learned_shared)
    assert all(float(learned[key]) == 0.0 for key in learned_shared)
    assert all(
        key not in learned
        for key in train_jax.V4172_LEGACY_OPERATOR_KEY_ALIAS_METRIC_NAMES)


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
    sharded_fns = _runtime_fns(mesh, kernel_profile="suppression")
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
        "minimal_runtime_profile": "suppression",
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


def test_v4172_400m_parameter_matched_count() -> None:
    expected = {
        "configs/train_config_v4171_400M_c4_40B_v4_64.yaml": 393_804_804,
        "configs/train_config_v4172_400M_c4_40B_v4_64.yaml": 393_800_708,
    }
    counts = {}
    operator_counts = {}
    loaded = {}
    for path, count in expected.items():
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        loaded[path] = config
        counts[path] = symbolic_parameter_count(config)["total"]
        operator_counts[path] = sum(
            int(config["model"][key]) for key in ("n_qk", "n_v", "n_rst"))
        assert counts[path] == count
    v4171_path, v4172_path = tuple(expected)
    assert counts[v4172_path] - counts[v4171_path] == -4096
    assert abs(counts[v4172_path] - counts[v4171_path]) / counts[
        v4171_path] < 2.0e-5
    assert operator_counts[v4171_path] == 46_096
    assert operator_counts[v4172_path] == 48_400
    assert operator_counts[v4172_path] - operator_counts[v4171_path] == 2_304
    model = loaded[v4172_path]["model"]
    assert {
        key: model[key]
        for key in (
            "model_version", "operator_key_mode", "operator_query_mode",
            "d_model", "d_route", "n_layers", "n_heads", "n_qk", "n_v",
            "n_rst", "admission_den_power", "srw_composition_mode",
        )
    } == {
        "model_version": "spatial-r1-v4.1.7.2",
        "operator_key_mode": "generalized_bilinear_rw",
        "operator_query_mode": "direct_state_projection",
        "d_model": 2304,
        "d_route": 256,
        "n_layers": 18,
        "n_heads": 36,
        "n_qk": 3988,
        "n_v": 12414,
        "n_rst": 31998,
        "admission_den_power": 1.0,
        "srw_composition_mode": "linear_angular",
    }
    assert loaded[v4172_path]["checkpoint_dir"] == (
        "gs://dawn-tpu-data-c4/checkpoints/"
        "dawn_srw_v4172_400M_c4_40B_v4_64")


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
    selected = candidate_pool_vectors(ctx, "qk", [0, 2])
    pool = params["neuron_pool"]
    selected_ids = jnp.asarray([0, 2], dtype=jnp.int32)
    expected = core._pool_params_with_operator_keys(
        pool, OPERATOR_KEY_MODE_GENERALIZED_BILINEAR
    )["attn_qk_op_key"][selected_ids]
    np.testing.assert_allclose(selected["address"], np.asarray(expected))
    audit = operator_pool_provenance(ctx)
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
    assert pool["rw_key_read_probe"].spec == PartitionSpec()
    assert pool["rw_key_write_probe"].spec == PartitionSpec()
    assert pool["attn_qk_read"].spec == PartitionSpec("model", None)
    assert pool["attn_qk_write"].spec == PartitionSpec("model", None)


def test_v4172_pool_den_power_fallback_and_validation() -> None:
    from scripts import train_jax

    base = {
        "model_version": V4172_MODEL_VERSION,
        "operator_key_mode": OPERATOR_KEY_MODE_GENERALIZED_BILINEAR,
        "operator_query_mode": "direct_state_projection",
        "d_route": 4,
    }
    cases = (
        ({}, (1.0, 1.0, 1.0, 1.0)),
        ({"admission_den_power": 0.8}, (0.8, 0.8, 0.8, 0.8)),
        ({"admission_den_power": 0.8,
          "admission_den_power_qk": 0.5}, (0.8, 0.5, 0.8, 0.8)),
        ({"admission_den_power": 1.0,
          "admission_den_power_qk": 0.5,
          "admission_den_power_v": 1.0,
          "admission_den_power_rst": 1.2}, (1.0, 0.5, 1.0, 1.2)),
    )
    for overrides, expected in cases:
        model_cfg = {**base, **overrides}
        train_jax._validate_v4171_model_config(model_cfg)
        assert tuple(model_cfg[key] for key in (
            "admission_den_power", "admission_den_power_qk",
            "admission_den_power_v", "admission_den_power_rst",
        )) == expected

    invalid_values = (-0.1, np.nan, np.inf, True, "0.5", jnp.float32(0.5))
    for pool in ("qk", "v", "rst"):
        field = f"admission_den_power_{pool}"
        for value in invalid_values:
            with pytest.raises(ValueError, match=field):
                train_jax._validate_v4171_model_config({
                    **base,
                    "admission_den_power": 1.0,
                    field: value,
                })

    with pytest.raises(ValueError, match="admission_den_power_qk"):
        jax.make_jaxpr(lambda value: _resolve_v417x_admission_den_powers(
            1.0, value, 1.0, 1.0, context="traced config"))(
                jnp.float32(0.5))


def test_v4172_pool_denominator_math_and_qk_pairing() -> None:
    kwargs = {
        "admission_den_power": 0.8,
        "admission_den_power_qk": 0.5,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.2,
        "srw_composition_mode": "linear_angular",
    }
    expected = {
        "q": (9.0, 0.5),
        "k": (16.0, 0.5),
        "v": (3.0, 1.0),
        "rst": (4.0, 1.2),
    }
    for route, (mass, power) in expected.items():
        pool = "qk" if route in ("q", "k") else route
        execution_kwargs, selected_power = _split_admission_den_kwargs(
            kwargs, pool)
        denominator = _composition_den(
            jnp.float32(mass), selected_power,
            execution_kwargs["srw_composition_mode"])
        assert float(selected_power) == pytest.approx(power)
        assert float(denominator) == pytest.approx(
            max(mass, 1.0) ** power, rel=1.0e-6)


def test_v4172_route_specific_factory_metadata_and_suppression_pairing() -> None:
    mesh = _one_device_mesh()
    production_fns = _runtime_fns(
        mesh, qk_power=0.5, v_power=1.0, rst_power=1.2)
    suppression_fns = _runtime_fns(
        mesh, qk_power=0.5, v_power=1.0, rst_power=1.2,
        kernel_profile="suppression")
    expected = {
        "attn_qk_paired": 0.5,
        "attn_qk_paired_minimal": 0.5,
        "attn_v_single": 1.0,
        "attn_v_single_minimal": 1.0,
        "rst_single": 1.2,
        "rst_single_minimal": 1.2,
    }
    for fns in (production_fns, suppression_fns):
        for name, power in expected.items():
            assert getattr(fns[name], "_v4171_admission_den_power") == power
        assert fns["attn_v_single_minimal"] is not fns["rst_single_minimal"]
    for name in (
            "attn_qk_paired_minimal", "attn_v_single_minimal",
            "rst_single_minimal"):
        assert getattr(
            production_fns[name], "_v4171_kernel_profile") == "production"
        assert getattr(
            suppression_fns[name], "_v4171_kernel_profile") == "suppression"
        assert not hasattr(
            production_fns[name], "_v4171_canonical_shard_map_kernel")
        assert hasattr(
            suppression_fns[name], "_v4171_canonical_shard_map_kernel")

    _validate_v4171_sharded_fns(
        production_fns, 1.0, expected_power_qk=0.5,
        expected_power_v=1.0, expected_power_rst=1.2)
    _validate_v4171_sharded_fns(
        suppression_fns, 1.0, expected_power_qk=0.5,
        expected_power_v=1.0, expected_power_rst=1.2)
    wrong = dict(production_fns)
    wrong["rst_single_minimal"] = production_fns[
        "attn_v_single_minimal"]
    with pytest.raises(ValueError, match="rst pool"):
        _validate_v4171_sharded_fns(
            wrong, 1.0, expected_power_qk=0.5,
            expected_power_v=1.0, expected_power_rst=1.2)


def test_v4172_canonical_builder_routes_pool_denominator_powers() -> None:
    from scripts import train_jax

    cfg = {
        "model": {
            "model_version": V4172_MODEL_VERSION,
            "d_route": 4,
            "n_qk": 4,
            "n_v": 4,
            "n_rst": 4,
            "admission_den_power": 1.0,
            "admission_den_power_qk": 0.5,
            "admission_den_power_v": 1.0,
            "admission_den_power_rst": 1.2,
        },
        "training": {
            "mesh_model": 1,
            "max_chunk_size": 2,
        },
    }
    sharded = train_jax.build_canonical_sharded_fns(
        cfg, _one_device_mesh())
    expected = {
        "attn_qk_paired": 0.5,
        "attn_qk_single_minimal": 0.5,
        "attn_qk_paired_minimal": 0.5,
        "attn_v_single": 1.0,
        "attn_v_single_minimal": 1.0,
        "rst_single": 1.2,
        "rst_single_minimal": 1.2,
    }
    for name, power in expected.items():
        assert getattr(
            sharded[name], "_v4171_admission_den_power") == power
    _validate_v4171_sharded_fns(
        sharded, 1.0, expected_power_qk=0.5,
        expected_power_v=1.0, expected_power_rst=1.2)


def test_v4172_candidate_scoring_uses_diagnostics_runtime_profile() -> None:
    from scripts import train_jax

    cfg = {
        "model": {
            "model_version": V4172_MODEL_VERSION,
            "d_route": 4,
            "n_qk": 4,
            "n_v": 4,
            "n_rst": 4,
            "admission_den_power": 1.0,
            "admission_den_power_qk": 0.5,
            "admission_den_power_v": 1.0,
            "admission_den_power_rst": 1.2,
        },
        "training": {
            "mesh_model": 1,
            "max_chunk_size": 2,
            "ce_token_chunk_size": 8,
        },
    }
    mesh = _one_device_mesh()
    sharded = train_jax.build_canonical_sharded_fns(
        cfg, mesh, for_eval=True,
        kernel_profile="production_diagnostics")
    model = DAWN_SRW_V4172(
        **_tiny_kwargs(), admission_den_power=1.0,
        admission_den_power_qk=0.5,
        admission_den_power_v=1.0,
        admission_den_power_rst=1.2)
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    labels = jnp.asarray([[-100, 2, 3]], dtype=jnp.int32)
    params = model.init(
        {"params": jax.random.PRNGKey(81),
         "dropout": jax.random.PRNGKey(82)},
        input_ids, labels=labels, deterministic=True)["params"]
    runtime_state = {
        "soft_gate_temperature": 0.07,
        "soft_gate_T_qk": 0.07,
        "soft_gate_T_v": 0.07,
        "soft_gate_T_rst": 0.07,
        "soft_gate_t_final": 0.07,
        "soft_gate_boundary_power": 2.0,
        "soft_gate_boundary_power_final": 4.0,
        "admission_den_power": 1.0,
        "admission_den_grad_scale": 1.0,
        "srw_composition_mode": "linear_angular",
        "heat_kernel_beta": 2.0,
        "training_tokens": 0.0,
        "ce_token_chunk_size": 8,
    }

    forward_kwargs = train_jax._fixed_runtime_forward_kwargs(
        model, sharded, runtime_state, compute_accuracy=False)
    assert forward_kwargs["minimal_runtime_profile"] == "diagnostics"

    score_step = train_jax.create_candidate_score_step(
        model, sharded, runtime_state)
    scores = score_step(
        params, input_ids, labels, jnp.ones_like(input_ids))
    assert scores.shape == (1,)
    assert np.isfinite(np.asarray(scores)).all()


def test_v4172_pool_powers_preserve_parameter_schema_and_default_forward() -> None:
    from scripts import train_jax

    mesh = _one_device_mesh()
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    legacy_model = DAWN_SRW_V4172(
        **_tiny_kwargs(), admission_den_power=1.0)
    explicit_model = DAWN_SRW_V4172(
        **_tiny_kwargs(), admission_den_power=1.0,
        admission_den_power_qk=1.0,
        admission_den_power_v=1.0,
        admission_den_power_rst=1.0)
    pool_model = DAWN_SRW_V4172(
        **_tiny_kwargs(), admission_den_power=1.0,
        admission_den_power_qk=0.5,
        admission_den_power_v=1.0,
        admission_den_power_rst=1.2)
    rngs = {
        "params": jax.random.PRNGKey(71),
        "dropout": jax.random.PRNGKey(72),
    }
    legacy_vars = legacy_model.init(
        rngs, input_ids, deterministic=True)
    explicit_vars = explicit_model.init(
        rngs, input_ids, deterministic=True)
    pool_vars = pool_model.init(
        rngs, input_ids, deterministic=True)
    assert jax.tree.structure(legacy_vars["params"]) == jax.tree.structure(
        pool_vars["params"])
    for legacy, explicit, pool_specific in zip(
            _tree_arrays(legacy_vars["params"]),
            _tree_arrays(explicit_vars["params"]),
            _tree_arrays(pool_vars["params"])):
        np.testing.assert_array_equal(legacy, explicit)
        np.testing.assert_array_equal(legacy, pool_specific)

    apply_kwargs = {
        "labels": input_ids,
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(73)},
        "sharded_fns": _runtime_fns(mesh),
        "analysis": False,
        "minimal_train": True,
        "compute_accuracy": False,
    }
    legacy_out = legacy_model.apply(legacy_vars, input_ids, **apply_kwargs)
    explicit_out = explicit_model.apply(explicit_vars, input_ids, **apply_kwargs)
    assert jax.tree.structure(legacy_out) == jax.tree.structure(explicit_out)
    for legacy, explicit in zip(
            _tree_arrays(legacy_out), _tree_arrays(explicit_out)):
        np.testing.assert_array_equal(legacy, explicit)

    pool_fns = _runtime_fns(
        mesh, qk_power=0.5, v_power=1.0, rst_power=1.2)
    pool_out = pool_model.apply(
        pool_vars, input_ids, **{**apply_kwargs, "sharded_fns": pool_fns})
    assert np.isfinite(float(pool_out["loss"]))
    assert pool_model.get_config()["admission_den_power_rst"] == 1.2
    suppression_pool_fns = _runtime_fns(
        mesh, qk_power=0.5, v_power=1.0, rst_power=1.2,
        kernel_profile="suppression")
    zero_suppression_out = pool_model.apply(
        pool_vars,
        input_ids,
        **{
            **apply_kwargs,
            "sharded_fns": suppression_pool_fns,
            "minimal_runtime_profile": "suppression",
            "analysis_contribution": jnp.full(
                (1, 8), -1, dtype=jnp.int32),
            "analysis_intervention_enabled": jnp.bool_(True),
        },
    )
    for key in ("loss", "correct", "valid_count"):
        np.testing.assert_array_equal(
            np.asarray(pool_out[key]), np.asarray(zero_suppression_out[key]))

    legacy_checkpoint_cfg = {
        "model_version": V4172_MODEL_VERSION,
        "d_route": 4,
        "operator_key_mode": OPERATOR_KEY_MODE_GENERALIZED_BILINEAR,
        "operator_query_mode": "direct_state_projection",
        "admission_den_power": 1.0,
    }
    explicit_default_cfg = {
        **legacy_checkpoint_cfg,
        "admission_den_power_qk": 1.0,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.0,
    }
    train_jax._validate_v4171_resume_compatibility(
        explicit_default_cfg, legacy_checkpoint_cfg)
    with pytest.raises(RuntimeError, match="admission_den_power_qk"):
        train_jax._validate_v4171_resume_compatibility(
            {**explicit_default_cfg, "admission_den_power_qk": 0.5},
            legacy_checkpoint_cfg)


def test_v4172_pool_power_experiment_config_only_changes_requested_fields() -> None:
    base_path = "configs/train_config_v4172_400M_c4_40B_v4_64_ver1.yaml"
    experiment_path = (
        "configs/train_config_v4172_400M_c4_40B_v4_64_ver1_"
        "den_qk0p5_v1p0_rst1p2.yaml")
    with open(base_path, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with open(experiment_path, "r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    expected_model = dict(base["model"])
    expected_model.update({
        "admission_den_power_qk": 0.5,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.2,
        "tau_init_target_qk_frac": 0.05,
        "tau_init_target_v_frac": 0.03,
        "tau_init_target_rst_frac": 0.01,
    })
    assert experiment["model"] == expected_model
    assert symbolic_parameter_count(experiment)["total"] == (
        symbolic_parameter_count(base)["total"])
    assert experiment["seed"] == base["seed"]
    assert experiment["data"] == base["data"]
    assert experiment["training"] == base["training"]
    assert experiment["checkpoint_dir"] != base["checkpoint_dir"]
    assert experiment["checkpoint_dir"].endswith(
        "dawn_srw_v4172_400M_c4_40B_v4_64_ver1_"
        "den_qk0p5_v1p0_rst1p2")
