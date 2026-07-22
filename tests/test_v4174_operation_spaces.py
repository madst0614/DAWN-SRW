import gc
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
import yaml

from models import dawn_srw_v4174 as v4174
from scripts import train_jax


@pytest.fixture(autouse=True)
def _release_jax_test_caches():
    yield
    gc.collect()


def _model_cfg(spaces=2, top_k=2):
    return {
        "model_version": v4174.MODEL_VERSION,
        "vocab_size": 32,
        "logical_vocab_size": 32,
        "vocab_size_padded": 32,
        "max_seq_len": 8,
        "d_model": 8,
        "d_route": 4,
        "n_layers": 1,
        "n_heads": 2,
        "n_qk": 8,
        "n_v": 8,
        "n_rst": 8,
        "n_operation_spaces": spaces,
        "operation_space_top_k": top_k,
        "operator_key_mode": "generalized_bilinear_rw",
        "operator_query_mode": "direct_state_projection",
        "admission_den_power": 1.0,
        "admission_den_power_qk": 0.5,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.2,
        "srw_composition_mode": "linear_angular",
        "heat_kernel_beta": 1.0,
        "tau_init_attn_qk": 0.0,
        "tau_init_attn_v": 0.0,
        "tau_init_rst": 0.0,
        "dropout": 0.0,
        "router_dropout": 0.0,
        "gradient_checkpointing": False,
    }


def _model(spaces=2, top_k=2):
    return v4174.DAWN_SRW_V4174(**{
        key: value for key, value in _model_cfg(spaces, top_k).items()
        if key not in ("model_version", "dropout", "operator_query_mode")
    }, dropout_rate=0.0)


@lru_cache(maxsize=None)
def _cached_init(spaces=2, top_k=2):
    model = _model(spaces, top_k)
    ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    params = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        ids, deterministic=True)["params"]
    return model, params, ids


def _init(model=None):
    if model is None:
        return _cached_init()
    ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    params = model.init(
        {"params": jax.random.key(0), "dropout": jax.random.key(1)},
        ids, deterministic=True)["params"]
    return model, params, ids


def _tree_norm(value):
    return float(jnp.sqrt(sum(
        jnp.sum(jnp.square(x.astype(jnp.float32)))
        for x in jax.tree.leaves(value))))


def _tree_paths(tree):
    return tuple(sorted(
        '/'.join(str(item.key if hasattr(item, 'key') else item) for item in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]))


def _serialized_name(canonical_name):
    return v4174.V4174_SERIALIZED_SPACE_PARAM_NAMES.get(
        canonical_name, canonical_name)


def test_a_shared_operation_space_keys_and_guard():
    _, params, ids = _init()
    router = params["router"]
    serialized_key = _serialized_name("operation_space_keys")
    matching = [key for key in router if key == serialized_key]
    assert matching == ["operation_address_keys"]
    assert "operation_space_keys" not in router
    assert not any(
        key.startswith(("q_", "k_", "v_", "rst_")) and key.endswith("space_keys")
        for key in router)
    keys = router[serialized_key]
    assert keys.shape == (2, 4)
    np.testing.assert_allclose(jnp.linalg.norm(keys, axis=-1), 1.0, atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(keys)))

    query_params = router[_serialized_name("q_space_query_proj")]
    ids_q, weights_q, scores_q = v4174._select_operation_spaces(
        jnp.ones((1, 4, 8)), query_params, keys, 2)
    with pytest.raises(ValueError, match="near-identical"):
        v4174.operation_space_initialization_diagnostics(
            jnp.stack((keys[0], keys[0])),
            {route: (ids_q, weights_q, scores_q) for route in v4174.ROUTES})
    diagnostics = v4174.initialization_diagnostics_from_params(params, ids, 2)
    assert "operation_space_pair_cosine_max" in diagnostics
    assert "operation address count" not in " ".join(_model().get_model_info()).lower()


def test_b_route_space_selection_is_independent():
    state = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    keys = jnp.eye(2)
    q_params = {"kernel": jnp.eye(2), "bias": jnp.zeros(2)}
    k_params = {
        "kernel": jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "bias": jnp.zeros(2),
    }
    q_ids, _, _ = v4174._select_operation_spaces(state, q_params, keys, 1)
    k_ids, _, _ = v4174._select_operation_spaces(state, k_params, keys, 1)
    np.testing.assert_array_equal(q_ids, [[[0], [1]]])
    np.testing.assert_array_equal(k_ids, [[[1], [0]]])


def test_b_source_level_operation_space_api_is_canonical():
    required = (
        "SpaceDense", "OperationSpaceNeuronPool", "OperationSpaceRouter",
        "_select_operation_spaces", "_dense_space_weights",
        "_space_weighted_state_writeback", "_rw_compose_space_dense",
    )
    assert all(hasattr(v4174, name) for name in required)
    assert not any(hasattr(v4174, name) for name in (
        "AddressDense", "OperationAddressNeuronPool", "OperationAddressRouter"))


def test_c_d_parameter_tree_sharing_and_independence():
    _, params, _ = _init()
    pool = params["neuron_pool"]
    router = params["router"]
    assert {"qk_read_vectors", "qk_write_vectors"} <= set(pool)
    assert not any(name in pool for name in (
        "q_read_vectors", "k_read_vectors", "q_write_vectors", "k_write_vectors"))
    assert {"q_operation_proj", "k_operation_proj",
            "q_operator_tau_proj", "k_operator_tau_proj",
            "q_address_query_proj", "k_address_query_proj",
            "qk_state_writeback"} <= set(router)
    assert {"v_read_vectors", "v_write_vectors",
            "rst_read_vectors", "rst_write_vectors"} <= set(pool)
    assert {"v_state_writeback", "rst_state_writeback"} <= set(router)
    keys = v4174._pool_operator_keys(pool)
    assert keys["qk_operator_keys"].shape == (2, 4, 4)
    assert keys["v_operator_keys"].shape == (2, 4, 4)
    assert keys["rst_operator_keys"].shape == (2, 4, 4)


def test_c_active_m8_native_v4174_serialized_parameter_abi():
    _, params, _ = _cached_init(8, 2)
    router = params["router"]
    expected_serialized = set(
        v4174.V4174_SERIALIZED_SPACE_PARAM_NAMES.values())
    assert expected_serialized <= set(router)
    assert not set(v4174.V4174_SERIALIZED_SPACE_PARAM_NAMES).intersection(router)
    paths = _tree_paths(params)
    for module_name in expected_serialized:
        assert any(path.startswith(f"router/{module_name}/")
                   or path == f"router/{module_name}" for path in paths)


def test_c_sharding_replicates_space_and_shards_operator_rows():
    _, params, _ = _init()
    mesh = train_jax.create_mesh(1, 1)
    shardings = train_jax.get_param_shardings(
        params, mesh, model_version=train_jax.V4174_MODEL_VERSION)
    assert shardings["router"]["operation_address_keys"].spec == jax.sharding.PartitionSpec()
    assert shardings["router"]["q_address_query_proj"]["kernel"].spec == (
        jax.sharding.PartitionSpec())
    assert shardings["neuron_pool"]["qk_read_vectors"].spec == (
        jax.sharding.PartitionSpec(None, "model", None))


@pytest.mark.parametrize("route", ("q", "k", "v", "rst"))
def test_e_space_weighted_writeback_matches_explicit(route):
    del route
    space_results = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10
    weights = jnp.asarray([[0.2, 0.8], [1.0, 0.0], [0.4, 0.6]])
    kernel = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5) / 20
    actual = v4174._space_weighted_state_writeback(
        space_results, weights, kernel, 0.7)
    reference = sum(
        (space_results[m] * weights[:, m, None] * 0.7) @ kernel[m]
        for m in range(2))
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_f_fused_writeback_jaxpr_has_no_space_token_model_tensor():
    m, t, r, d = 2, 3, 4, 5
    graph = jax.make_jaxpr(v4174._space_weighted_state_writeback)(
        jnp.ones((m, t, r)), jnp.ones((t, m)),
        jnp.ones((m, r, d)), jnp.float32(1.0))
    forbidden = (m, t, d)
    shapes = {
        tuple(aval.shape) for equation in graph.jaxpr.eqns
        for aval in equation.outvars
        if hasattr(getattr(aval, "aval", None), "shape")
        for aval in (aval.aval,)
    }
    assert forbidden not in shapes


def test_g_attention_consumes_only_composed_model_width_routes():
    model, params, ids = _init()
    out = model.apply(
        {"params": params}, ids, deterministic=True,
        rngs={"dropout": jax.random.key(2)})
    assert out["logits"].shape == (1, 4, 32)
    assert not any(key.endswith("space_outputs") for key in out)


def test_h_production_diagnostics_logits_and_loss_parity():
    model, params, ids = _init()
    labels = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    kwargs = dict(deterministic=True, rngs={"dropout": jax.random.key(5)})
    production = model.apply({"params": params}, ids, labels=labels, **kwargs)
    diagnostics = model.apply(
        {"params": params}, ids, labels=labels,
        minimal_runtime_profile="diagnostics", **kwargs)
    np.testing.assert_allclose(
        production["loss"], diagnostics["loss"], rtol=1e-6, atol=1e-6)
    assert all(jnp.isfinite(value) for key, value in diagnostics.items()
               if key.endswith(("_mean", "_frac", "_norm")))


def test_h_sharded_route_kernels_have_production_diagnostics_parity():
    mesh = train_jax.create_mesh(1, 1)
    kwargs = dict(
        mesh=mesh, max_chunk_size=2, admission_den_power=1.0,
        srw_composition_mode="linear_angular", heat_kernel_beta=1.0)
    production = v4174.make_sharded_space_dense_minimal(**kwargs)
    diagnostics = v4174.make_sharded_space_dense_diagnostics(**kwargs)
    query = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10
    read = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 20
    write = jnp.flip(read, axis=-1)
    keys = v4174._materialize_space_operator_keys(
        read, write, jnp.eye(4), jnp.eye(4))
    tau = jnp.zeros((2, 3, 1))
    valid = jnp.ones((2, 3), dtype=jnp.bool_)
    scalars = (0.07, 0.07, 2.0, 2.0, 0.0)
    expected = production(query, keys, tau, valid, read, write, *scalars)
    observed = diagnostics(query, keys, tau, valid, read, write, *scalars)[0]
    np.testing.assert_allclose(expected, observed, rtol=1e-6, atol=1e-6)
    paired_production = v4174.make_sharded_qk_space_dense_minimal(**kwargs)
    paired_diagnostics = v4174.make_sharded_qk_space_dense_diagnostics(**kwargs)
    paired_query = jnp.stack((query, query * 0.7), axis=2)
    paired_tau = jnp.stack((tau, tau + 0.1), axis=2)
    paired_expected = paired_production(
        paired_query, keys, paired_tau, valid, read, write, *scalars)
    q_diag, k_diag = paired_diagnostics(
        paired_query, keys, paired_tau, valid, read, write, *scalars)
    np.testing.assert_allclose(
        paired_expected, jnp.stack((q_diag[0], k_diag[0]), axis=2),
        rtol=1e-6, atol=1e-6)


def test_i_per_space_tau_calibration_is_independent():
    base = jnp.linspace(-0.9, 0.9, 40).reshape(2, 4, 5)
    scores = {
        "q": base,
        "k": base + jnp.asarray([0.1, -0.2])[:, None, None],
        "v": base * 0.5,
        "rst": base * 0.25,
    }
    tau = v4174.calibrate_operator_tau_per_space(
        scores, target_qk_frac=0.2, target_v_frac=0.3,
        target_rst_frac=0.4)
    assert all(value.shape == (2,) for value in tau.values())
    assert not np.allclose(tau["q"], tau["k"])
    for route, target in (("q", 0.2), ("k", 0.2),
                          ("v", 0.3), ("rst", 0.4)):
        measured = (scores[route] > tau[route][:, None, None]).mean(axis=(1, 2))
        np.testing.assert_allclose(measured, target, atol=0.06)


def test_j_config_alias_normalization_and_conflict():
    canonical = {"n_operation_spaces": 8, "operation_space_top_k": 2}
    serialized = {
        "n_operation_addresses": 8, "operation_address_top_k": 2}
    assert v4174.resolve_operation_space_config(canonical) == (8, 2)
    assert v4174.resolve_operation_space_config(serialized) == (8, 2)
    both = {**canonical, **serialized}
    assert v4174.resolve_operation_space_config(both) == (8, 2)
    with pytest.raises(ValueError, match="conflicts"):
        v4174.resolve_operation_space_config({
            **both, "n_operation_addresses": 4})
    materialized = v4174.materialize_operation_space_config({
        **serialized, "n_qk": 8, "n_v": 8, "n_rst": 8})
    assert materialized["n_operation_spaces"] == 8
    assert materialized["operation_space_top_k"] == 2
    assert materialized["n_operation_addresses"] == 8


def test_k_native_v4174_exact_version_requirement():
    requested = _model_cfg()
    train_jax._validate_v4171_resume_compatibility(
        dict(requested), dict(requested))
    for checkpoint_version in (train_jax.V4173_MODEL_VERSION, None):
        checkpoint = dict(requested)
        if checkpoint_version is None:
            checkpoint.pop("model_version")
        else:
            checkpoint["model_version"] = checkpoint_version
        with pytest.raises(RuntimeError, match="accepts only native v4174"):
            train_jax._validate_v4171_resume_compatibility(
                dict(requested), checkpoint)
    assert "legacy_converter" not in train_jax.MODEL_REGISTRY[
        train_jax.V4174_MODEL_VERSION]


def test_k_native_m1_v4174_tree_forward_loss_gradient_and_restore():
    model, params, ids = _cached_init(1, 1)
    router = params["router"]
    assert all(name in router for name in (
        "q_operation_proj", "k_operation_proj", "v_operation_proj",
        "rst_operation_proj", "q_operator_tau_proj", "k_operator_tau_proj",
        "v_operator_tau_proj", "rst_operator_tau_proj"))
    assert not any(name in router for name in
                   v4174.V4174_SERIALIZED_SPACE_PARAM_NAMES.values())

    def loss_fn(p):
        return model.apply(
            {"params": p}, ids, labels=ids, deterministic=True,
            rngs={"dropout": jax.random.key(31)})["loss"]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    opt_state = optax.adamw(1e-3).init(params)
    item = {
        "params": params,
        "opt_state": opt_state,
        "model_config": _model_cfg(1, 1),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "m1")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, item)
        restored = checkpointer.restore(path, item=item)
    for expected, actual in zip(
            jax.tree.leaves(params), jax.tree.leaves(restored["params"])):
        np.testing.assert_array_equal(expected, actual)


def test_l_symbolic_count_matches_tree_and_40m_report():
    model, params, _ = _init()
    del model
    actual = sum(value.size for value in jax.tree.leaves(params))
    assert v4174.symbolic_parameter_count(_model_cfg())["total"] == actual
    config_path = Path(__file__).parents[1] / "configs" / (
        "train_config_v4174_40M_c4_5B_address8_top2_parammatch.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]
    assert v4174.symbolic_parameter_count(cfg)["total"] == 40_343_584
    assert all(cfg[name] % 8 == 0 for name in ("n_qk", "n_v", "n_rst"))


def test_m_main_loss_gradients_reach_canonical_space_tree():
    jax.clear_caches()
    gc.collect()
    model = v4174.DAWN_SRW_V4174(
        vocab_size=8, d_model=4, n_layers=1, n_heads=1,
        max_seq_len=4, dropout_rate=0.0, router_dropout=0.0,
        d_route=2, n_qk=4, n_v=4, n_rst=4,
        n_operation_spaces=2, operation_space_top_k=2,
        tau_init_attn_qk=0.0, tau_init_attn_v=0.0, tau_init_rst=0.0)
    ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    params = model.init(
        {"params": jax.random.key(20), "dropout": jax.random.key(21)},
        ids, deterministic=True)["params"]
    labels = ids

    def loss_fn(p):
        return model.apply(
            {"params": p}, ids, labels=labels, deterministic=True,
            rngs={"dropout": jax.random.key(11)})["loss"]

    grads = jax.grad(loss_fn)(params)
    router, pool = grads["router"], grads["neuron_pool"]
    required_router = ["operation_address_keys", "q_address_query_proj",
        "k_address_query_proj", "v_address_query_proj", "rst_address_query_proj",
        "q_operation_proj", "k_operation_proj", "v_operation_proj",
        "rst_operation_proj", "q_operator_tau_proj", "k_operator_tau_proj",
        "v_operator_tau_proj", "rst_operator_tau_proj", "qk_state_writeback",
        "v_state_writeback", "rst_state_writeback"]
    required_pool = ["qk_read_vectors", "qk_write_vectors", "v_read_vectors",
        "v_write_vectors", "rst_read_vectors", "rst_write_vectors",
        "operator_key_read_probe", "operator_key_write_probe"]
    for name in required_router:
        assert np.isfinite(_tree_norm(router[name])) and _tree_norm(router[name]) > 0
    for name in required_pool:
        assert np.isfinite(_tree_norm(pool[name])) and _tree_norm(pool[name]) > 0


def test_n_diagnostics_naming_and_analysis_boundary():
    model, params, ids = _init()
    regular = model.apply(
        {"params": params}, ids, deterministic=True,
        rngs={"dropout": jax.random.key(12)})
    diagnostics = model.apply(
        {"params": params}, ids, deterministic=True,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.key(12)})
    analysis = model.apply(
        {"params": params}, ids, deterministic=True,
        analysis=True, minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.key(12)})
    assert all(key in regular for key in train_jax.V4174_SPACE_METRIC_NAMES)
    assert all(key in regular for key in train_jax.V4174_ADDRESS_METRIC_ALIASES)
    for canonical in train_jax.V4174_SPACE_METRIC_NAMES:
        alias = canonical.replace("_space_", "_address_")
        np.testing.assert_array_equal(regular[canonical], regular[alias])
    assert "rst_route_update_norm" in diagnostics
    np.testing.assert_array_equal(
        diagnostics["rst_route_update_norm"],
        diagnostics["rst_route_output_norm"])
    assert not any("per_space_" in key for key in regular)
    for route in v4174.ROUTES:
        for suffix in ("selection_frac", "mean_selected_weight", "active_frac",
                       "active_count", "tau_mean", "gate_mass", "gate_den",
                       "output_norm"):
            key = f"{route}_per_space_{suffix}"
            assert key in analysis and analysis[key].shape == (2,)
            np.testing.assert_array_equal(
                analysis[key], analysis[key.replace("_per_space_", "_per_address_")])


def test_n_trainer_pool_diagnostics_use_native_v4174_schema():
    model, params, _ = _init()
    grads = jax.tree.map(jnp.ones_like, params)
    param_diag = train_jax._pool_param_diagnostics(
        params, full=True, model=model)
    update_diag = train_jax._pool_update_diagnostics(
        params, grads, model_version=v4174.MODEL_VERSION)
    assert param_diag and update_diag
    for value in (*param_diag.values(), *update_diag.values()):
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(value))))


def test_o_checkpoint_round_trip_preserves_params_optimizer_and_config():
    _, params, _ = _init()
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    config = _model_cfg()
    config["n_operation_addresses"] = config.pop("n_operation_spaces")
    config["operation_address_top_k"] = config.pop("operation_space_top_k")
    item = {
        "params": params,
        "opt_state": opt_state,
        "rng": jax.random.key(91),
        "global_step": jnp.int32(17),
        "epoch": jnp.int32(3),
        "step_in_epoch": jnp.int32(5),
        "consumed_examples": np.asarray(2048, dtype=np.int64),
        "consumed_tokens": np.asarray(1_048_576, dtype=np.int64),
        "best_val_loss": jnp.float32(2.5),
        "model_config": config,
    }
    param_paths = _tree_paths(params)
    opt_paths = _tree_paths(opt_state)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "checkpoint")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, item)
        restored = checkpointer.restore(path, item=item)
    assert restored["model_config"] == config
    assert _tree_paths(restored["params"]) == param_paths
    assert _tree_paths(restored["opt_state"]) == opt_paths
    for expected, actual in zip(
            jax.tree.leaves(params), jax.tree.leaves(restored["params"])):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(
            jax.tree.leaves(opt_state), jax.tree.leaves(restored["opt_state"])):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(
        jax.random.key_data(item["rng"]),
        jax.random.key_data(restored["rng"]))
    for field in (
            "global_step", "epoch", "step_in_epoch",
            "consumed_examples", "consumed_tokens", "best_val_loss"):
        np.testing.assert_array_equal(item[field], restored[field])


@pytest.mark.parametrize("spaces", (8, 32))
def test_p_tau_compact_dynamic_broadcast_payload(spaces):
    tau = {
        route: np.linspace(-0.5, 0.5, spaces).tolist()
        for route in ("q", "k", "qk", "v", "rst")}
    full = {
        "tau_init_quantile_tau": tau,
        "tau_init_target_frac": {"qk": 0.2, "v": 0.12, "rst": 0.04},
        "tau_init_target_local_frac": {"qk": 0.2, "v": 0.12, "rst": 0.04},
        "tau_init_est_active": {key: [0.1] * spaces for key in tau},
        "tau_init_est_active_local": {key: [0.1] * spaces for key in tau},
        "tau_init_est_active_pool": {key: [0.1] * spaces for key in tau},
        "tau_calibration": {"q": [{"rho": "x" * 4096}] * spaces},
        "tau_init_calibration": {"sample": "진단" * 4096},
    }
    payload = train_jax._tau_init_apply_payload(full)
    assert set(payload) == set(train_jax._TAU_INIT_APPLY_PAYLOAD_FIELDS)
    assert "tau_calibration" not in payload
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    received = train_jax.broadcast_str_from_host0(encoded, max_len=None)
    assert json.loads(received) == payload
    for route in tau:
        np.testing.assert_array_equal(
            json.loads(received)["tau_init_quantile_tau"][route], tau[route])
