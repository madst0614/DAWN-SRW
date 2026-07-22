import inspect
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml

from models import dawn_srw_v4174 as v4174


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = (
    ROOT / "configs" /
    "train_config_v4174_400M_c4_40B_v4_64_space8_top2_kernel_field.yaml")


def _model_config(**updates):
    config = {
        "model_version": v4174.MODEL_VERSION,
        "vocab_size": 64,
        "d_model": 16,
        "d_route": 4,
        "n_layers": 1,
        "n_heads": 4,
        "max_seq_len": 16,
        "n_operation_spaces": 4,
        "operation_space_top_k": 2,
        "operator_key_mode": v4174.OPERATOR_KEY_MODE,
        "n_qk": 32,
        "n_v": 32,
        "n_rst": 32,
        "space_kernel_beta_qk": 9.0,
        "space_kernel_beta_v": 12.0,
        "space_kernel_beta_rst": 16.0,
        "admission_den_power": 1.0,
        "admission_den_power_qk": 0.5,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.2,
        "srw_composition_mode": "linear_angular",
        "tau_init_attn_qk": 0.0,
        "tau_init_attn_v": 0.0,
        "tau_init_rst": 0.0,
    }
    config.update(updates)
    return config


def _model(**updates):
    config = _model_config(**updates)
    return v4174.DAWN_SRW_V4174(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        d_route=config["d_route"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        dropout_rate=0.0,
        router_dropout=0.0,
        n_qk=config["n_qk"],
        n_v=config["n_v"],
        n_rst=config["n_rst"],
        n_operation_spaces=config["n_operation_spaces"],
        operation_space_top_k=config["operation_space_top_k"],
        space_kernel_beta_qk=config["space_kernel_beta_qk"],
        space_kernel_beta_v=config["space_kernel_beta_v"],
        space_kernel_beta_rst=config["space_kernel_beta_rst"],
        operator_key_mode=config["operator_key_mode"],
        admission_den_power=config["admission_den_power"],
        admission_den_power_qk=config["admission_den_power_qk"],
        admission_den_power_v=config["admission_den_power_v"],
        admission_den_power_rst=config["admission_den_power_rst"],
        srw_composition_mode=config["srw_composition_mode"],
        tau_init_attn_qk=config["tau_init_attn_qk"],
        tau_init_attn_v=config["tau_init_attn_v"],
        tau_init_rst=config["tau_init_rst"],
    )


def _variables(model=None, seed=0):
    model = model or _model()
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    variables = model.init(
        {"params": jax.random.PRNGKey(seed),
         "dropout": jax.random.PRNGKey(seed + 1)},
        tokens, deterministic=True)
    return model, tokens, variables


def _count(tree):
    return sum(int(value.size) for value in jax.tree.leaves(tree))


def test_architecture_has_one_shared_state_basis_and_canonical_tree():
    _, _, variables = _variables()
    router = variables["params"]["router"]
    assert set(router) == {
        "space_state_proj", "space_state_writeback",
        "q_operator_query_proj", "k_operator_query_proj",
        "v_operator_query_proj", "rst_operator_query_proj",
        "q_operator_tau_proj", "k_operator_tau_proj",
        "v_operator_tau_proj", "rst_operator_tau_proj",
    }
    assert router["space_state_proj"].shape == (4, 16, 4)
    assert router["space_state_writeback"].shape == (4, 4, 16)
    np.testing.assert_allclose(
        router["space_state_writeback"],
        jnp.swapaxes(router["space_state_proj"], -1, -2), atol=0, rtol=0)
    pool = variables["params"]["neuron_pool"]
    assert pool["qk_read_vectors"].shape == (4, 8, 4)
    assert pool["v_read_vectors"].shape == (4, 8, 4)
    assert pool["rst_read_vectors"].shape == (4, 8, 4)


def test_stacked_projection_is_full_rank_orthogonal_and_decoder_is_not_tied():
    _, _, variables = _variables()
    router = variables["params"]["router"]
    projection = router["space_state_proj"]
    stacked = jnp.swapaxes(projection, -1, -2).reshape(16, 16)
    singular = np.asarray(jnp.linalg.svd(stacked, compute_uv=False))
    np.testing.assert_allclose(singular, np.ones(16), atol=2e-5, rtol=2e-5)
    assert int(jnp.linalg.matrix_rank(stacked)) == 16
    diagnostics = v4174._space_geometry_diagnostics(
        projection, router["space_state_writeback"])
    assert float(diagnostics["stacked_projection_effective_rank"]) > 15.99
    assert float(diagnostics["encoder_writeback_init_error"]) == 0.0
    changed_encoder = projection.at[0, 0, 0].add(0.25)
    assert not np.allclose(
        np.asarray(router["space_state_writeback"]),
        np.asarray(jnp.swapaxes(changed_encoder, -1, -2)))


def test_all_routes_share_the_single_projected_local_state():
    state = jax.random.normal(jax.random.PRNGKey(3), (7, 16))
    projection = jax.random.normal(jax.random.PRNGKey(4), (4, 16, 4))
    local = v4174._project_space_local_states(state, projection)
    expected = jnp.einsum("td,mdr->mtr", state, projection)
    assert local.shape == (4, 7, 4)
    np.testing.assert_allclose(local, expected)
    source = inspect.getsource(v4174.DAWN_SRW_V4174.__call__)
    assert source.count("_project_space_local_states(") == 1
    assert inspect.getsource(v4174._qk_shared_read_compose).count(
        "_space_read_scalar(") == 1


def test_exact_kernel_field_is_masked_log_mean_exp():
    query = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    keys = jnp.array([
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
    ])
    valid = jnp.array([[True, True, False], [True, False, False]])
    beta = 2.5
    actual = v4174._exact_space_log_field(query, keys, beta, valid)
    cosine = jnp.einsum("tr,mnr->tmn", query, keys)
    expected = []
    for token in range(2):
        row = []
        for space in range(2):
            values = beta * (cosine[token, space, valid[space]] - 1.0)
            row.append(jax.scipy.special.logsumexp(values) - jnp.log(values.size))
        expected.append(row)
    np.testing.assert_allclose(actual, jnp.asarray(expected), atol=1e-6)


def test_positive_features_and_live_space_sketch_contract():
    _, _, variables = _variables()
    pool = variables["params"]["neuron_pool"]
    keys = v4174._pool_operator_keys(pool)
    sketches = v4174._build_space_kernel_sketches(
        keys, {"qk": 9.0, "v": 12.0, "rst": 16.0})
    features = v4174._build_positive_kernel_features(
        keys["qk_operator_keys"], 9.0)
    assert features.shape[-1] == 2 * 4
    assert bool(jnp.all(jnp.isfinite(features)))
    assert bool(jnp.all(features > 0))
    assert sketches["qk_space_kernel_sketch"].shape == (4, 8)
    norms = jnp.linalg.norm(sketches["qk_space_kernel_sketch"], axis=-1)
    assert not np.allclose(np.asarray(norms), np.ones(4))
    assert not any("kernel_sketch" in path
                   for path, _ in jax.tree_util.tree_flatten_with_path(pool)[0])


def test_initialized_exact_vs_sketch_guard_before_and_after_rw_update():
    _, tokens, variables = _variables(_model(n_qk=128, n_v=128, n_rst=128))
    params = variables["params"]
    attention_state, _ = v4174._sampled_layer_states(params, tokens, 16)

    def guard(current_pool):
        keys = v4174._pool_operator_keys(current_pool)
        sketches = v4174._build_space_kernel_sketches(
            keys, {"qk": 9.0, "v": 12.0, "rst": 16.0})
        query = v4174._shared_forward_unit_direction(v4174._linear(
            params["router"]["q_operator_query_proj"], attention_state))
        return v4174._kernel_sketch_reference_metrics(
            query, keys["qk_operator_keys"],
            sketches["qk_space_kernel_sketch"], 9.0, 2)

    initial = guard(params["neuron_pool"])
    assert float(initial["top1_agreement"]) >= 0.95
    assert float(initial["topk_set_agreement"]) >= 0.90
    assert bool(jnp.isfinite(initial["pearson"]))

    def sketch_loss(read_vectors):
        changed = dict(params["neuron_pool"])
        changed["qk_read_vectors"] = read_vectors
        keys = v4174._pool_operator_keys(changed)
        sketch = v4174._build_space_kernel_sketches(
            keys, {"qk": 9.0, "v": 12.0, "rst": 16.0})
        return sketch["qk_space_kernel_sketch"].sum()

    read = params["neuron_pool"]["qk_read_vectors"]
    read = read - 1e-6 * jax.grad(sketch_loss)(read)
    changed_pool = dict(params["neuron_pool"])
    changed_pool["qk_read_vectors"] = read
    updated = guard(changed_pool)
    assert float(updated["top1_agreement"]) >= 0.95
    assert float(updated["topk_set_agreement"]) >= 0.90


def test_hard_topk_has_exact_zero_nonselected_weights_and_no_ste():
    scores = jnp.array([[0.1, 2.0, -1.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    ids, weights = v4174._select_operation_spaces(scores, 2)
    dense = v4174._dense_space_weights(ids, weights, 4)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-7)
    np.testing.assert_allclose(dense.sum(axis=-1), 1.0, atol=1e-7)
    assert int((dense == 0).sum()) == 4
    gradient = jax.grad(lambda value: (
        v4174._select_operation_spaces(value, 2)[1]
        * jnp.array([[1.0, -1.0], [1.0, -1.0]])).sum())(scores)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_rw_read_uses_local_state_and_qk_routes_have_independent_gates():
    key = jax.random.PRNGKey(10)
    local = jax.random.normal(key, (2, 3, 4))
    read = jax.random.normal(jax.random.PRNGKey(11), (2, 4, 4))
    write = jax.random.normal(jax.random.PRNGKey(12), (2, 4, 4))
    operator_keys = jax.random.normal(jax.random.PRNGKey(13), (2, 4, 4))
    query_q = jax.random.normal(jax.random.PRNGKey(14), (3, 4))
    query_k = jax.random.normal(jax.random.PRNGKey(15), (3, 4))
    tau_q = jnp.zeros((2, 3, 1))
    tau_k = jnp.full((2, 3, 1), -0.2)
    kwargs = dict(
        soft_gate_temperature=0.07, soft_gate_boundary_power=2.0,
        admission_den_power=1.0, srw_composition_mode="linear_angular",
        heat_kernel_beta=4.0)
    q_out, k_out = v4174._qk_shared_read_compose(
        query_q, query_k, operator_keys, tau_q, tau_k, local,
        read, write, **kwargs)
    q_changed, _ = v4174._qk_shared_read_compose(
        query_q, query_k, operator_keys, tau_q, tau_k, local + 0.3,
        read, write, **kwargs)
    assert not np.allclose(q_out, k_out)
    assert not np.allclose(q_out, q_changed)

    def both_route_loss(read_vectors):
        q_value, k_value = v4174._qk_shared_read_compose(
            query_q, query_k, operator_keys, tau_q, tau_k, local,
            read_vectors, write, **kwargs)
        return q_value.sum() + 0.7 * k_value.sum()

    gradient = jax.grad(both_route_loss)(read)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_forward_loss_gradients_and_one_optimizer_update_are_finite():
    model, tokens, variables = _variables()
    params = variables["params"]

    def loss_fn(current):
        result = model.apply(
            {"params": current}, tokens, labels=tokens,
            deterministic=True, minimal_runtime_profile="diagnostics",
            rngs={"dropout": jax.random.PRNGKey(21)})
        return result["loss"], result

    (loss, result), gradients = jax.value_and_grad(
        loss_fn, has_aux=True)(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(value)))
               for value in jax.tree.leaves(gradients))
    required_paths = (
        ("router", "space_state_proj"),
        ("router", "space_state_writeback"),
        ("router", "q_operator_query_proj", "kernel"),
        ("neuron_pool", "operator_key_read_probe"),
        ("neuron_pool", "operator_key_write_probe"),
        ("neuron_pool", "qk_read_vectors"),
        ("neuron_pool", "qk_write_vectors"),
        ("neuron_pool", "v_read_vectors"),
        ("neuron_pool", "rst_write_vectors"),
    )
    for path in required_paths:
        value = gradients
        for name in path:
            value = value[name]
        assert float(jnp.linalg.norm(value.astype(jnp.float32))) > 0.0, path
    assert result["q_space_usage_max"] >= result["q_space_usage_min"]
    optimizer = optax.adamw(1e-4)
    state = optimizer.init(params)
    updates, state = optimizer.update(gradients, state, params)
    changed = optax.apply_updates(params, updates)
    assert all(bool(jnp.all(jnp.isfinite(value)))
               for value in jax.tree.leaves(changed))


def test_symbolic_count_matches_initialized_tree_and_canonical_reference():
    _, _, variables = _variables()
    assert v4174.symbolic_parameter_count(_model_config())["total"] == _count(
        variables["params"])
    canonical = yaml.safe_load(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    counts = v4174.symbolic_parameter_count(canonical["model"])
    assert counts["total"] == 216_204_292
    assert counts["learned_key_tables"] == 0
    assert canonical["model"]["n_qk"] == 12_320
    assert canonical["model"]["n_v"] == 38_832
    assert canonical["model"]["n_rst"] == 78_496
    for name in ("n_qk", "n_v", "n_rst"):
        assert canonical["model"][name] % 16 == 0


def test_active_count_matching_and_beta_calibration_formula():
    canonical = yaml.safe_load(CANONICAL_CONFIG.read_text(encoding="utf-8"))[
        "model"]
    expected = {
        "qk": 415.0,
        "v": 784.8,
        "rst": 528.64,
    }
    for pool, target in expected.items():
        count = canonical[f"n_{pool}"]
        fraction = canonical[f"tau_init_target_{pool}_frac"]
        actual = (canonical["operation_space_top_k"]
                  * (count / canonical["n_operation_spaces"]) * fraction)
        assert actual == pytest.approx(target, rel=0, abs=1e-9)

    scores = {
        "q": jnp.linspace(-0.8, 0.9, 400).reshape(2, 10, 20),
        "k": jnp.linspace(-0.7, 0.95, 400).reshape(2, 10, 20),
        "v": jnp.linspace(-0.9, 0.8, 400).reshape(2, 10, 20),
        "rst": jnp.linspace(-0.95, 0.7, 400).reshape(2, 10, 20),
    }
    betas = v4174.calibrate_space_kernel_betas(
        scores, target_qk_frac=0.1, target_v_frac=0.08,
        target_rst_frac=0.04)
    assert set(betas) == {
        "space_kernel_beta_qk", "space_kernel_beta_v",
        "space_kernel_beta_rst"}
    assert all(np.isfinite(value) and value > 0.0 for value in betas.values())


def test_config_validation_and_fresh_checkpoint_round_trip(tmp_path: Path):
    from scripts import train_jax

    config = _model_config()
    assert v4174.resolve_operation_space_config(config) == (4, 2)
    materialized = v4174.materialize_operation_space_config(dict(config))
    assert materialized["n_operation_spaces"] == 4
    with pytest.raises(ValueError):
        v4174.resolve_operation_space_config({"operation_space_top_k": 2})
    with pytest.raises(ValueError):
        v4174.materialize_operation_space_config({
            **config, "d_model": 12})

    _, _, variables = _variables()
    payload = serialization.to_bytes(variables["params"])
    restored = serialization.from_bytes(variables["params"], payload)
    assert jax.tree.structure(restored) == jax.tree.structure(
        variables["params"])
    for expected, actual in zip(
            jax.tree.leaves(variables["params"]), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(expected, actual)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(variables["params"])
    full_config = {
        "model": config,
        "training": {"mesh_data": 1, "mesh_model": 1},
    }
    manager = train_jax._create_orbax_checkpoint_manager(
        str(tmp_path / "checkpoints"), checkpoint_interval=1, keep_last=1)
    try:
        saved = train_jax.save_orbax_checkpoint(
            manager, variables["params"], opt_state,
            jax.random.PRNGKey(41), epoch=0, global_step=1,
            step_in_epoch=1, steps_per_epoch=2, best_val_loss=1.0,
            model_config=config, training_config=full_config["training"],
            full_config=full_config, raw_config=full_config,
            config_path="fresh-v4174.yaml", run_id="v4174-fresh-roundtrip",
            checkpoint_kind="latest", train_loss=1.0, wait=True)
        assert saved
        target = train_jax._build_orbax_state(
            variables["params"], opt_state, jax.random.PRNGKey(41),
            epoch=0, global_step=1, step_in_epoch=1, steps_per_epoch=2,
            best_val_loss=1.0, training_config=full_config["training"],
            full_config=full_config, model_config=config)
        restored_state, metadata = train_jax._restore_orbax_state(
            manager, 1, target)
    finally:
        manager.close()

    assert metadata["model_config"]["model_version"] == v4174.MODEL_VERSION
    assert metadata["model_config"]["space_kernel_beta_qk"] == 9.0
    for expected, actual in zip(
            jax.tree.leaves(variables["params"]),
            jax.tree.leaves(restored_state["params"])):
        np.testing.assert_array_equal(expected, actual)


def test_analysis_geometry_and_operator_field_outputs_are_finite():
    model, tokens, variables = _variables()
    result = model.apply(
        variables, tokens, deterministic=True, analysis=True,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.PRNGKey(31)})
    assert int(result["stacked_projection_rank"]) == 16
    assert result["space_local_state_norm"].shape == (4,)
    assert result["qk_operator_key_covariance_effective_rank"].shape == (4,)
    for route in ("q", "k", "v", "rst"):
        assert bool(jnp.isfinite(result[f"{route}_kernel_sketch_pearson"]))
        assert result[f"{route}_kernel_sketch_top1_agreement"] >= 0.95
        assert result[f"{route}_kernel_sketch_topk_set_agreement"] >= 0.90


def test_model_info_states_canonical_geometry_and_dense_execution():
    info = "\n".join(_model().get_model_info())
    assert "D=16, M=4, R=4" in info
    assert "same global route query" in info
    assert "all-space dense" in info
    assert "shared writeback" in info


def test_trainer_diagnostics_accept_only_canonical_pool_shapes():
    from scripts import train_jax

    model, _, variables = _variables()
    params = variables["params"]
    geometry = train_jax.create_geometry_step(
        max_sample=32, model_version=v4174.MODEL_VERSION)(params)
    pool_diagnostics = train_jax._pool_param_diagnostics(
        {"neuron_pool": params["neuron_pool"]}, full=True, model=model)
    pool_grads = jax.tree.map(jnp.ones_like, params["neuron_pool"])
    key_grad_norms = train_jax._canonical_pool_op_key_grad_norms(
        pool_grads, v4174.MODEL_VERSION)
    probe_grad_norms = train_jax._shared_probe_gradient_diagnostics(
        params["neuron_pool"], pool_grads, v4174.MODEL_VERSION)

    assert geometry
    assert pool_diagnostics
    assert set(key_grad_norms) == {"attn_qk", "attn_v", "rst"}
    assert all(bool(jnp.all(jnp.isfinite(value)))
               for value in (*geometry.values(), *pool_diagnostics.values(),
                             *key_grad_norms.values(),
                             *probe_grad_norms.values()))
