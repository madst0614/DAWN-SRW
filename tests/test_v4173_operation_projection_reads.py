from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax.sharding import Mesh

from analysis import dawn_analysis_common
from analysis.operator_interpretability import protocol
from models.dawn_srw_v4171 import DAWN_SRW_V4171
from models.dawn_srw_v4172 import DAWN_SRW_V4172
from models import dawn_srw_v4173 as v4173
from scripts import train_jax


def _one_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("data", "model"))


def _tiny_model_kwargs() -> dict:
    return {
        "vocab_size": 16,
        "d_model": 6,
        "d_route": 3,
        "n_layers": 1,
        "n_heads": 2,
        "max_seq_len": 3,
        "dropout_rate": 0.0,
        "router_dropout": 0.0,
        "n_qk": 2,
        "n_v": 2,
        "n_rst": 2,
        "tau_init_attn_qk": -0.9,
        "tau_init_attn_v": -0.9,
        "tau_init_rst": -0.9,
        "admission_den_power": 1.0,
        "admission_den_power_qk": 0.5,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.2,
    }


def _tiny_trainer_config() -> dict:
    return {
        "model": {
            "model_version": train_jax.V4173_MODEL_VERSION,
            "operator_key_mode": "generalized_bilinear_rw",
            "operator_query_mode": "direct_state_projection",
            "d_route": 3,
            "n_qk": 2,
            "n_v": 2,
            "n_rst": 2,
            "admission_den_power": 1.0,
            "admission_den_power_qk": 0.5,
            "admission_den_power_v": 1.0,
            "admission_den_power_rst": 1.2,
            "srw_composition_mode": "linear_angular",
        },
        "training": {"mesh_model": 1, "max_chunk_size": 2},
    }


def _manual_single(query, keys, raw_tau, read, write, den_power):
    query_unit = v4173._forward_unit_direction(
        query.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    key_unit = v4173._forward_unit_direction(
        keys.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    read_unit = v4173._forward_unit_direction(
        read.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    write_unit = v4173._forward_unit_direction(
        write.astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    rho = (query_unit @ key_unit.T).astype(jnp.float32)
    tau = v4173._tau_from_param(raw_tau)
    _, admission, _, execution_weight, _ = v4173._compute_admission_drive(
        rho, tau, jnp.float32(0.07), boundary_power=jnp.float32(2.0),
        effective_active_eps=jnp.float32(1.0e-6),
        execution_prune_eps=jnp.float32(0.0),
        srw_composition_mode="linear_angular", heat_kernel_beta=2.0)
    read_scalar = v4173._operation_projection_read(query, read_unit)
    raw = ((execution_weight * read_scalar).astype(jnp.bfloat16)
           @ write_unit).astype(jnp.float32)
    den = v4173._composition_den(
        admission.sum(axis=-1, keepdims=True), den_power,
        "linear_angular")
    return (raw / den).astype(jnp.bfloat16).astype(jnp.float32)


def test_v4171_v4172_remain_full_state_read_architectures() -> None:
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    common = _tiny_model_kwargs()
    common.pop("admission_den_power_qk")
    common.pop("admission_den_power_v")
    common.pop("admission_den_power_rst")
    for model_type in (DAWN_SRW_V4171, DAWN_SRW_V4172):
        params = model_type(**common).init(
            {"params": jax.random.PRNGKey(1),
             "dropout": jax.random.PRNGKey(2)},
            input_ids, deterministic=True)["params"]
        pool = params["neuron_pool"]
        assert pool["attn_qk_read"].shape == (2, 6)
        assert pool["attn_v_read"].shape == (2, 6)
        assert pool["rst_read"].shape == (2, 6)

    for path in (
            "models/dawn_srw_v4171.py", "models/dawn_srw_v4172.py"):
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "execution_read_mode" not in source
        assert "operation_projection_read" not in source


def test_v4173_parameter_tree_has_native_local_rw_and_shared_up_projections() -> None:
    model = v4173.DAWN_SRW_V4173(**_tiny_model_kwargs())
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    params = model.init(
        {"params": jax.random.PRNGKey(3),
         "dropout": jax.random.PRNGKey(4)},
        input_ids, deterministic=True)["params"]
    pool = params["neuron_pool"]
    assert pool["attn_qk_read"].shape == (2, 3)
    assert pool["attn_v_read"].shape == (2, 3)
    assert pool["rst_read"].shape == (2, 3)
    assert pool["attn_qk_write"].shape == (2, 3)
    assert pool["attn_v_write"].shape == (2, 3)
    assert pool["rst_write"].shape == (2, 3)
    assert pool["rw_key_read_probe"].shape == (3, 3)
    assert pool["rw_key_write_probe"].shape == (3, 3)
    assert not any(name.endswith("op_key") for name in pool)
    router = params["router"]
    assert set(name for name in router if name.startswith("up_")) == {
        "up_qk", "up_v", "up_rst"}
    for name in ("up_qk", "up_v", "up_rst"):
        kernel = router[name]["kernel"]
        assert kernel.shape == (3, 6)
        np.testing.assert_allclose(
            np.asarray(kernel @ kernel.T), np.eye(3),
            rtol=2.0e-5, atol=2.0e-5)
    assert model.__version__ == "spatial-r1-v4.1.7.3"
    assert v4173.DAWN is v4173.DAWN_SRW_V4173


def test_v4173_generalized_key_is_local_rw_and_live_gradient() -> None:
    read = jax.random.normal(jax.random.PRNGKey(5), (4, 3))
    write = jax.random.normal(jax.random.PRNGKey(6), (4, 3))
    read_probe = jax.random.normal(jax.random.PRNGKey(7), (3, 3))
    write_probe = jax.random.normal(jax.random.PRNGKey(8), (3, 3))
    weights = jax.random.normal(jax.random.PRNGKey(9), (4, 3))

    def loss(*args):
        return jnp.sum(
            v4173.materialize_generalized_bilinear_operator_keys(*args)
            * weights)

    keys = v4173.materialize_generalized_bilinear_operator_keys(
        read, write, read_probe, write_probe)
    assert keys.shape == (4, 3)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(keys), axis=-1), 1.0,
        rtol=1.0e-5, atol=1.0e-5)
    for gradient in jax.grad(loss, argnums=(0, 1, 2, 3))(
            read, write, read_probe, write_probe):
        assert np.all(np.isfinite(np.asarray(gradient)))
        assert float(jnp.linalg.norm(gradient)) > 0.0


def test_v4173_production_qk_v_rst_match_manual_route_formulas() -> None:
    mesh = _one_device_mesh()
    single = v4173.make_sharded_srw_minimal(mesh, max_chunk_size=2)
    paired = v4173.make_sharded_srw_paired_minimal(
        mesh, max_chunk_size=2)
    x = jax.random.normal(jax.random.PRNGKey(10), (1, 2, 6))
    keys = jnp.asarray([[1.0, 0.0, 0.0], [0.2, 0.9, 0.1]])
    read = jnp.asarray([[1.0, 0.5, -0.25], [-0.1, 0.3, 1.0]])
    write = jax.random.normal(jax.random.PRNGKey(11), (2, 3))
    raw_tau = jnp.full((1, 2, 1), -2.0, dtype=jnp.float32)
    q = jnp.asarray([[[0.8, 0.1, 0.4], [0.4, 0.7, -0.2]]])
    k = jnp.asarray([[[0.1, 0.9, 0.3], [-0.5, 0.4, 0.8]]])
    v = jnp.asarray([[[0.6, -0.2, 0.9], [0.7, 0.1, 0.2]]])
    rst = jnp.asarray([[[0.2, 0.3, 1.1], [-0.2, 0.8, 0.6]]])
    scalar_args = (
        jnp.float32(0.07), jnp.float32(0.07),
        jnp.float32(2.0), jnp.float32(4.0), jnp.float32(0.0))

    outputs = {}
    for name, query in (("q", q), ("k", k), ("v", v), ("rst", rst)):
        outputs[name] = single(
            x, query, keys, raw_tau, read, write, *scalar_args)
        manual = _manual_single(query, keys, raw_tau, read, write, 1.0)
        np.testing.assert_array_equal(
            np.asarray(outputs[name]), np.asarray(manual))
        assert outputs[name].shape == (1, 2, 3)

    paired_query = jnp.stack((q, k), axis=2)
    paired_tau = jnp.stack((raw_tau, raw_tau), axis=2)
    paired_output = paired(
        x, paired_query, keys, paired_tau, read, write, *scalar_args)
    np.testing.assert_array_equal(
        np.asarray(paired_output[..., 0, :]), np.asarray(outputs["q"]))
    np.testing.assert_array_equal(
        np.asarray(paired_output[..., 1, :]), np.asarray(outputs["k"]))
    q_read, k_read = np.asarray(v4173._operation_projection_read_paired(
        paired_query, v4173._forward_unit_direction(read)))[0, 0]
    assert not np.array_equal(q_read, k_read)


def test_v4173_raw_query_amplitude_and_no_score_read() -> None:
    state1 = jnp.ones((1, 2, 6), dtype=jnp.float32)
    state2 = jnp.full((1, 2, 6), 19.0, dtype=jnp.float32)
    query = jnp.asarray([[[1.0, 0.5, -0.25], [0.4, 0.8, 0.2]]])
    keys = jnp.asarray([[1.0, 0.0, 0.0], [0.2, 0.9, 0.1]])
    read = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    write = jax.random.normal(jax.random.PRNGKey(12), (2, 3))
    tau = jnp.full((1, 2, 1), -2.0)
    offset = jnp.zeros_like(tau)
    kwargs = {
        "admission_den_power": 1.0,
        "srw_composition_mode": "linear_angular",
    }
    out1 = v4173._srw_inference(
        state1, query, keys, tau, offset, read, write, **kwargs)
    out_state_changed = v4173._srw_inference(
        state2, query, keys, tau, offset, read, write, **kwargs)
    out_scaled = v4173._srw_inference(
        state1, query * 2.0, keys, tau, offset, read, write, **kwargs)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out_state_changed))
    np.testing.assert_allclose(
        np.asarray(out_scaled), np.asarray(out1) * 2.0,
        rtol=2.0e-2, atol=2.0e-2)
    np.testing.assert_allclose(
        np.asarray(v4173._angular_relation(query, keys)),
        np.asarray(v4173._angular_relation(query * 2.0, keys)),
        rtol=1.0e-5, atol=1.0e-5)

    alternate_read = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])
    rho1 = v4173._angular_relation(query, keys)
    rho2 = v4173._angular_relation(query, keys)
    read1 = v4173._operation_projection_read(
        query, v4173._forward_unit_direction(read))
    read2 = v4173._operation_projection_read(
        query, v4173._forward_unit_direction(alternate_read))
    np.testing.assert_array_equal(np.asarray(rho1), np.asarray(rho2))
    assert not np.array_equal(np.asarray(read1), np.asarray(read2))


def test_v4173_model_backward_and_diagnostics_parity() -> None:
    mesh = _one_device_mesh()
    cfg = _tiny_trainer_config()
    production = train_jax.build_canonical_sharded_fns(cfg, mesh)
    diagnostics = train_jax.build_canonical_sharded_fns(
        cfg, mesh, kernel_profile="production_diagnostics")
    assert production["_v4173_kernel_profile"] == "production"
    assert diagnostics["_v4173_kernel_profile"] == "production_diagnostics"

    model = v4173.DAWN_SRW_V4173(**_tiny_model_kwargs())
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(13),
         "dropout": jax.random.PRNGKey(14)},
        input_ids, deterministic=True)
    apply_kwargs = {
        "labels": input_ids,
        "deterministic": True,
        "rngs": {"dropout": jax.random.PRNGKey(15)},
        "analysis": False,
        "minimal_train": True,
        "compute_accuracy": False,
    }

    def loss_fn(params):
        return model.apply(
            {"params": params}, input_ids,
            sharded_fns=production, **apply_kwargs)["loss"]

    loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
    assert np.isfinite(float(loss))
    pool_grads = grads["neuron_pool"]
    for name in (
            "attn_qk_read", "attn_v_read", "rst_read",
            "attn_qk_write", "attn_v_write", "rst_write",
            "rw_key_read_probe", "rw_key_write_probe"):
        assert np.all(np.isfinite(np.asarray(pool_grads[name])))
        assert float(jnp.linalg.norm(pool_grads[name])) > 0.0
    router_grads = grads["router"]
    for name in (
            "proj_attn", "proj_rst", "raw_tau_attn", "raw_tau_rst",
            "up_qk", "up_v", "up_rst"):
        leaves = jax.tree.leaves(router_grads[name])
        assert all(np.all(np.isfinite(np.asarray(value))) for value in leaves)
        assert sum(float(jnp.linalg.norm(value)) for value in leaves) > 0.0

    production_out = model.apply(
        variables, input_ids, sharded_fns=production, **apply_kwargs)
    diagnostic_out = model.apply(
        variables, input_ids, sharded_fns=diagnostics,
        minimal_runtime_profile="diagnostics", **apply_kwargs)
    np.testing.assert_array_equal(
        np.asarray(production_out["loss"]),
        np.asarray(diagnostic_out["loss"]))


def test_v4173_inference_eval_analysis_and_suppression_keep_residual_width() -> None:
    model_kwargs = _tiny_model_kwargs()
    model = v4173.DAWN_SRW_V4173(**model_kwargs)
    prompt = jnp.asarray([[1, 2]], dtype=jnp.int32)
    params = model.init(
        {"params": jax.random.PRNGKey(21),
         "dropout": jax.random.PRNGKey(22)},
        prompt, deterministic=True)["params"]
    model_cfg = {
        **model_kwargs,
        "model_version": train_jax.V4173_MODEL_VERSION,
        "operator_key_mode": "generalized_bilinear_rw",
        "srw_composition_mode": "linear_angular",
    }

    logits, cache_k, cache_v, cache_len = v4173.prefill(
        params, model_cfg, prompt)
    assert logits.shape == (1, 2, 16)
    assert cache_k.shape[-2:] == (3, 3)
    assert cache_v.shape[-2:] == (3, 3)
    decoded, _, _, decoded_len = v4173.decode_step(
        params, model_cfg, jnp.asarray([3]),
        cache_k, cache_v, cache_len)
    assert decoded.shape == (1, 16)
    assert decoded_len == 3

    all_tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    avg_loss, ppl, accuracy, total_valid = v4173.vectorized_eval(
        params, model_cfg, all_tokens, batch_size=1)
    assert all(np.isfinite(float(value)) for value in (
        avg_loss, ppl, accuracy, total_valid))
    analysis_logits, layer_info = v4173.analysis_forward(
        params, model_cfg, all_tokens)
    assert analysis_logits.shape == (1, 3, 16)
    assert layer_info["attn_out_norm"].shape == (1,)
    assert layer_info["rst_out_norm"].shape == (1,)
    suppressed = v4173.build_suppressed_forward(
        params, model_cfg, {
            "qk": jnp.zeros((2,), dtype=jnp.bool_),
            "v": jnp.zeros((2,), dtype=jnp.bool_),
            "rst": jnp.zeros((2,), dtype=jnp.bool_),
        })
    suppressed_logits = suppressed(all_tokens)
    assert suppressed_logits.shape == (1, 3, 16)
    assert np.all(np.isfinite(np.asarray(suppressed_logits)))


def test_v4173_trajectory_outputs_project_only_selected_local_packets() -> None:
    mesh = _one_device_mesh()
    cfg = _tiny_trainer_config()
    trajectory_kernels = train_jax.build_canonical_sharded_fns(
        cfg, mesh, kernel_profile="trajectory")
    model = v4173.DAWN_SRW_V4173(**_tiny_model_kwargs())
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(23),
         "dropout": jax.random.PRNGKey(24)},
        input_ids, deterministic=True)
    route_ids = jnp.zeros((1, 1, 1, 1), dtype=jnp.int32)
    route_valid = jnp.zeros_like(route_ids, dtype=jnp.bool_)
    output = model.apply(
        variables, input_ids, labels=input_ids, deterministic=True,
        rngs={"dropout": jax.random.PRNGKey(25)},
        sharded_fns=trajectory_kernels, analysis=True,
        minimal_train=True, minimal_runtime_profile="trajectory",
        analysis_trajectory_positions=jnp.asarray([[1]], dtype=jnp.int32),
        analysis_trajectory_position_valid=jnp.asarray(
            [[True]], dtype=jnp.bool_),
        analysis_trajectory_ids_q=route_ids,
        analysis_trajectory_ids_k=route_ids,
        analysis_trajectory_ids_v=route_ids,
        analysis_trajectory_ids_rst=route_ids,
        analysis_trajectory_valid_q=route_valid,
        analysis_trajectory_valid_k=route_valid,
        analysis_trajectory_valid_v=route_valid,
        analysis_trajectory_valid_rst=route_valid,
        analysis_trajectory_replay_enabled=False)
    trace = output["operator_trajectory_trace"]
    for route in ("q", "k", "v", "rst"):
        route_trace = trace["routes"][route]
        for field in (
                "production_output", "selected_replay_output",
                "production_precast_output",
                "selected_replay_precast_output"):
            assert route_trace[field].shape[-1] == 6
        assert route_trace["query"].shape[-1] == 3


def test_v4173_config_registry_resume_and_fail_loud_contracts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    path = (
        "configs/train_config_v4173_400M_c4_40B_v4_64_ver1_"
        "den_qk0p5_v1p0_rst1p2.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = cfg["model"]
    assert model_cfg["model_version"] == "spatial-r1-v4.1.7.3"
    assert (model_cfg["n_qk"], model_cfg["n_v"], model_cfg["n_rst"]) == (
        8300, 26160, 52864)
    assert not any(
        field in model_cfg for field in (
            "execution_read_mode", "read_mode", "low_dimensional_read",
            "operation_projection_read"))
    assert v4173.symbolic_parameter_count(cfg)["total"] == 216_204_292
    assert cfg["training"]["n_chunks_qk"] == 2
    assert cfg["training"]["n_chunks_v"] == 2
    assert cfg["training"]["n_chunks_rst"] == 8
    assert train_jax.MODEL_REGISTRY[train_jax.V4173_MODEL_VERSION][
        "class"] is v4173.DAWN_SRW_V4173
    assert dawn_analysis_common.V4173_MODEL_VERSION in (
        dawn_analysis_common.SUPPORTED_ANALYSIS_MODEL_VERSIONS)
    assert train_jax.V4173_MODEL_VERSION in protocol.SUPPORTED_MODEL_VERSIONS

    validated = deepcopy(model_cfg)
    train_jax._validate_v4171_model_config(validated)
    assert "execution_read_mode" not in validated
    with pytest.raises(ValueError, match="execution_read_mode"):
        train_jax._validate_v4171_model_config({
            **model_cfg, "execution_read_mode": "operation_projection"})

    for checkpoint, requested in (
            ({"model_version": train_jax.V4172_MODEL_VERSION},
             {"model_version": train_jax.V4173_MODEL_VERSION}),
            ({"model_version": train_jax.V4173_MODEL_VERSION},
             {"model_version": train_jax.V4172_MODEL_VERSION})):
        with pytest.raises(RuntimeError, match="exact same model version"):
            train_jax._validate_v4171_resume_compatibility(
                checkpoint, requested)

    with pytest.raises(ValueError, match="width mismatch"):
        v4173._operation_projection_read(
            jnp.ones((1, 1, 3)), jnp.ones((2, 6)))
    with pytest.raises(ValueError, match="operator row mismatch"):
        v4173.materialize_generalized_bilinear_operator_keys(
            jnp.ones((2, 3)), jnp.ones((3, 3)),
            jnp.ones((3, 3)), jnp.ones((3, 3)))
    with pytest.raises(ValueError, match="probe shape mismatch"):
        v4173.materialize_generalized_bilinear_operator_keys(
            jnp.ones((2, 3)), jnp.ones((2, 3)),
            jnp.ones((4, 3)), jnp.ones((3, 3)))

    atol, rtol = train_jax._production_diagnostic_parity_tolerances(
        cfg["training"], train_jax.V4173_MODEL_VERSION)
    observed_fast_loss = 10.4026623
    observed_abs_diff = 5.05447388e-5
    assert observed_abs_diff <= atol + rtol * abs(observed_fast_loss)
    assert 1.0e-3 > atol + rtol * abs(observed_fast_loss)
    legacy_atol, legacy_rtol = (
        train_jax._production_diagnostic_parity_tolerances(
            {}, train_jax.V4172_MODEL_VERSION))
    assert (legacy_atol, legacy_rtol) == (1.0e-5, 1.0e-6)
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        train_jax._production_diagnostic_parity_tolerances(
            {"production_diagnostic_parity_rtol": -1.0},
            train_jax.V4173_MODEL_VERSION)

    poc_path = "configs/train_config_v4173_40M_c4_5B.yaml"
    with open(poc_path, "r", encoding="utf-8") as handle:
        poc_cfg = yaml.safe_load(handle)
    assert v4173.symbolic_parameter_count(poc_cfg)["total"] == 27_271_428
    assert poc_cfg["training"]["mesh_data"] == 4
    auto_mesh_cfg = deepcopy(poc_cfg)
    auto_mesh_cfg["training"].pop("mesh_data")
    monkeypatch.setattr(train_jax.jax, "device_count", lambda: 4)
    assert train_jax._materialize_mesh_config(auto_mesh_cfg) == (4, 1)
    assert auto_mesh_cfg["training"]["mesh_data"] == 4
    train_jax._require_resume_materialized_fields(auto_mesh_cfg)

    registry = train_jax.MODEL_REGISTRY[train_jax.V4173_MODEL_VERSION]
    monkeypatch.setitem(registry, "module", "models.dawn_srw_v4171")
    with pytest.raises(RuntimeError, match="wrong sharded factory module"):
        train_jax.build_canonical_sharded_fns(
            _tiny_trainer_config(), _one_device_mesh())


def test_v4173_legacy_full_write_checkpoint_fails_loud() -> None:
    model = v4173.DAWN_SRW_V4173(**_tiny_model_kwargs())
    input_ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    target = model.init(
        {"params": jax.random.PRNGKey(31),
         "dropout": jax.random.PRNGKey(32)},
        input_ids, deterministic=True)["params"]
    legacy = deepcopy(target)
    for name in ("attn_qk_write", "attn_v_write", "rst_write"):
        legacy["neuron_pool"][name] = jnp.ones((2, 6), dtype=jnp.float32)
    legacy["neuron_pool"]["rw_key_write_probe"] = jnp.ones(
        (6, 3), dtype=jnp.float32)
    for name in ("up_qk", "up_v", "up_rst"):
        legacy["router"].pop(name)
    with pytest.raises(
            RuntimeError,
            match="legacy v4173 full-write checkpoint is incompatible with "
                  "native v4173 local-write architecture; start a fresh run"):
        train_jax._validate_v4171_checkpoint_param_schema(
            legacy, target, model_version=train_jax.V4173_MODEL_VERSION)
