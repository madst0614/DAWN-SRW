import copy
import gc
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


def _model_cfg(addresses=2, top_k=2):
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
        "n_operation_addresses": addresses,
        "operation_address_top_k": top_k,
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


def _model(addresses=2, top_k=2):
    return v4174.DAWN_SRW_V4174(**{
        key: value for key, value in _model_cfg(addresses, top_k).items()
        if key not in ("model_version", "dropout", "operator_query_mode")
    }, dropout_rate=0.0)


@lru_cache(maxsize=None)
def _cached_init(addresses=2, top_k=2):
    model = _model(addresses, top_k)
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


def test_a_shared_operation_address_keys_and_guard():
    _, params, ids = _init()
    router = params["router"]
    matching = [key for key in router if key == "operation_address_keys"]
    assert matching == ["operation_address_keys"]
    assert not any(
        key.startswith(("q_", "k_", "v_", "rst_")) and key.endswith("address_keys")
        for key in router)
    keys = router["operation_address_keys"]
    assert keys.shape == (2, 4)
    np.testing.assert_allclose(jnp.linalg.norm(keys, axis=-1), 1.0, atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(keys)))

    query_params = router["q_address_query_proj"]
    ids_q, weights_q, scores_q = v4174._select_operation_addresses(
        jnp.ones((1, 4, 8)), query_params, keys, 2)
    with pytest.raises(ValueError, match="near-identical"):
        v4174.operation_address_initialization_diagnostics(
            jnp.stack((keys[0], keys[0])),
            {route: (ids_q, weights_q, scores_q) for route in v4174.ROUTES})
    diagnostics = v4174.initialization_diagnostics_from_params(params, ids, 2)
    assert "operation_address_pair_cosine_max" in diagnostics


def test_b_route_address_selection_is_independent():
    state = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    keys = jnp.eye(2)
    q_params = {"kernel": jnp.eye(2), "bias": jnp.zeros(2)}
    k_params = {
        "kernel": jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "bias": jnp.zeros(2),
    }
    q_ids, _, _ = v4174._select_operation_addresses(state, q_params, keys, 1)
    k_ids, _, _ = v4174._select_operation_addresses(state, k_params, keys, 1)
    np.testing.assert_array_equal(q_ids, [[[0], [1]]])
    np.testing.assert_array_equal(k_ids, [[[1], [0]]])


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


def test_c_sharding_replicates_address_and_shards_operator_rows():
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
def test_e_address_weighted_writeback_matches_explicit(route):
    del route
    address_results = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10
    weights = jnp.asarray([[0.2, 0.8], [1.0, 0.0], [0.4, 0.6]])
    kernel = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5) / 20
    actual = v4174._weighted_state_writeback(address_results, weights, kernel, 0.7)
    reference = sum(
        (address_results[m] * weights[:, m, None] * 0.7) @ kernel[m]
        for m in range(2))
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_f_fused_writeback_jaxpr_has_no_address_token_model_tensor():
    m, t, r, d = 2, 3, 4, 5
    graph = jax.make_jaxpr(v4174._weighted_state_writeback)(
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
    assert not any(key.endswith("address_outputs") for key in out)


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
    production = v4174.make_sharded_address_dense_minimal(**kwargs)
    diagnostics = v4174.make_sharded_address_dense_diagnostics(**kwargs)
    query = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 10
    read = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4) / 20
    write = jnp.flip(read, axis=-1)
    keys = v4174._materialize_address_operator_keys(
        read, write, jnp.eye(4), jnp.eye(4))
    tau = jnp.zeros((2, 3, 1))
    valid = jnp.ones((2, 3), dtype=jnp.bool_)
    scalars = (0.07, 0.07, 2.0, 2.0, 0.0)
    expected = production(query, keys, tau, valid, read, write, *scalars)
    observed = diagnostics(query, keys, tau, valid, read, write, *scalars)[0]
    np.testing.assert_allclose(expected, observed, rtol=1e-6, atol=1e-6)
    paired_production = v4174.make_sharded_qk_address_dense_minimal(**kwargs)
    paired_diagnostics = v4174.make_sharded_qk_address_dense_diagnostics(**kwargs)
    paired_query = jnp.stack((query, query * 0.7), axis=2)
    paired_tau = jnp.stack((tau, tau + 0.1), axis=2)
    paired_expected = paired_production(
        paired_query, keys, paired_tau, valid, read, write, *scalars)
    q_diag, k_diag = paired_diagnostics(
        paired_query, keys, paired_tau, valid, read, write, *scalars)
    np.testing.assert_allclose(
        paired_expected, jnp.stack((q_diag[0], k_diag[0]), axis=2),
        rtol=1e-6, atol=1e-6)


def test_i_per_address_tau_calibration_is_independent():
    base = jnp.linspace(-0.9, 0.9, 40).reshape(2, 4, 5)
    scores = {
        "q": base,
        "k": base + jnp.asarray([0.1, -0.2])[:, None, None],
        "v": base * 0.5,
        "rst": base * 0.25,
    }
    tau = v4174.calibrate_operator_tau_per_address(
        scores, target_qk_frac=0.2, target_v_frac=0.3,
        target_rst_frac=0.4)
    assert all(value.shape == (2,) for value in tau.values())
    assert not np.allclose(tau["q"], tau["k"])
    for route, target in (("q", 0.2), ("k", 0.2),
                          ("v", 0.3), ("rst", 0.4)):
        measured = (scores[route] > tau[route][:, None, None]).mean(axis=(1, 2))
        np.testing.assert_allclose(measured, target, atol=0.06)


def _legacy_cfg():
    cfg = _model_cfg(addresses=1, top_k=1)
    cfg.pop("n_operation_addresses")
    cfg.pop("operation_address_top_k")
    cfg["model_version"] = train_jax.V4173_MODEL_VERSION
    cfg["n_operation_spaces"] = 1
    cfg["operation_space_top_k"] = 1
    return cfg


def test_j_single_address_converter_slices_and_renames():
    legacy_cfg = _legacy_cfg()
    legacy_model = train_jax.DAWN_SRW_V4173(**{
        key: value for key, value in legacy_cfg.items()
        if key not in ("model_version", "dropout", "operator_query_mode")
    }, dropout_rate=0.0)
    ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    legacy = legacy_model.init(
        {"params": jax.random.key(7), "dropout": jax.random.key(8)},
        ids, deterministic=True)["params"]
    converted, metadata = v4174._convert_v4173_single_address_params_to_v4174(
        legacy, source_model_cfg=legacy_cfg, return_metadata=True)
    old_router, new_router = legacy["router"], converted["router"]
    r = legacy_cfg["d_route"]
    np.testing.assert_array_equal(
        new_router["q_operation_proj"]["kernel"],
        old_router["proj_attn"]["kernel"][:, :r])
    np.testing.assert_array_equal(
        new_router["v_operator_tau_proj"]["bias"],
        old_router["raw_tau_attn"]["bias"][2:3])
    assert "operation_address_keys" not in new_router
    assert not any(name.endswith("address_query_proj") for name in new_router)
    assert metadata["converted_source_version"] == train_jax.V4173_MODEL_VERSION
    assert metadata["optimizer_restore_policy"] == "fresh_optimizer"
    _, checkpoint_metadata = (
        train_jax.convert_v4173_single_address_checkpoint_for_v4174(
            legacy, {"full_config": {"model": copy.deepcopy(legacy_cfg)}}))
    assert checkpoint_metadata["full_config"]["model"]["model_version"] == (
        train_jax.V4174_MODEL_VERSION)
    assert checkpoint_metadata["full_config"]["model"]["n_operation_addresses"] == 1
    old_cfg = {
        "model": copy.deepcopy(legacy_cfg),
        "training": {
            "mesh_data": 1, "mesh_model": 1, "n_chunks_qk": 1,
            "n_chunks_v": 1, "n_chunks_rst": 1, "tau_lr_mult": 0.001,
        },
    }
    mesh = train_jax.create_mesh(1, 1)
    old_sharded = train_jax.build_canonical_sharded_fns(
        old_cfg, mesh, kernel_profile="production")
    old_output = legacy_model.apply(
        {"params": legacy}, ids, deterministic=True, minimal_train=True,
        sharded_fns=old_sharded,
        rngs={"dropout": jax.random.key(9)})["logits"]
    new_model = _model(addresses=1, top_k=1)
    new_output = new_model.apply(
        {"params": converted}, ids, deterministic=True,
        rngs={"dropout": jax.random.key(9)})["logits"]
    np.testing.assert_allclose(old_output, new_output, rtol=1e-5, atol=1e-5)


def test_k_legacy_rst_only_multi_address_rejected():
    cfg = _legacy_cfg()
    cfg["n_operation_spaces"] = 2
    with pytest.raises(ValueError, match="RST-only multi-space checkpoints"):
        v4174.resolve_operation_address_config(cfg)
    cfg["n_operation_spaces"] = 1
    params = {"router": {"rst_space_keys": jnp.ones((2, 4))}}
    with pytest.raises(ValueError, match="RST-only multi-space checkpoints"):
        v4174._convert_v4173_single_address_params_to_v4174(
            params, source_model_cfg=cfg)


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
    baseline = yaml.safe_load((Path(__file__).parents[1] / "configs" /
        "train_config_v4173_40M_c4_5B.yaml").read_text(encoding="utf-8"))["model"]
    report = v4174.search_parameter_matched_operator_counts(baseline, cfg)
    assert report["absolute_difference"] == 540
    assert report["relative_difference"] < 0.0005


def test_m_main_loss_gradients_reach_canonical_address_tree():
    jax.clear_caches()
    gc.collect()
    model = v4174.DAWN_SRW_V4174(
        vocab_size=8, d_model=4, n_layers=1, n_heads=1,
        max_seq_len=4, dropout_rate=0.0, router_dropout=0.0,
        d_route=2, n_qk=4, n_v=4, n_rst=4,
        n_operation_addresses=2, operation_address_top_k=2,
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
    assert not any("space_" in key for key in (*regular, *diagnostics, *analysis))
    assert all(key in regular for key in train_jax.V4174_ADDRESS_METRIC_NAMES)
    assert not any("per_address_" in key for key in regular)
    for route in v4174.ROUTES:
        for suffix in ("selection_frac", "mean_selected_weight", "active_frac",
                       "active_count", "tau_mean", "gate_mass", "gate_den",
                       "output_norm"):
            key = f"{route}_per_address_{suffix}"
            assert key in analysis and analysis[key].shape == (2,)


def test_o_checkpoint_round_trip_preserves_params_optimizer_and_config():
    _, params, _ = _init()
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    config = _model_cfg()
    item = {"params": params, "opt_state": opt_state, "model_config": config}
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "checkpoint")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, item)
        restored = checkpointer.restore(path)
    assert restored["model_config"] == config
    for expected, actual in zip(
            jax.tree.leaves(params), jax.tree.leaves(restored["params"])):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(
            jax.tree.leaves(opt_state), jax.tree.leaves(restored["opt_state"])):
        np.testing.assert_array_equal(expected, actual)
