import inspect
from pathlib import Path

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
    "train_config_v4174_400M_c4_40B_v4_64_space8_top2_direct_read.yaml")


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
        "n_q": 32,
        "n_k": 32,
        "n_v": 32,
        "n_rst": 32,
        "admission_den_power": 1.0,
        "admission_den_power_qk": 0.5,
        "admission_den_power_v": 1.0,
        "admission_den_power_rst": 1.2,
        "srw_composition_mode": "linear_angular",
        "tau_init_attn_q": 0.0,
        "tau_init_attn_k": 0.0,
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
        n_q=config["n_q"],
        n_k=config["n_k"],
        n_v=config["n_v"],
        n_rst=config["n_rst"],
        n_operation_spaces=config["n_operation_spaces"],
        operation_space_top_k=config["operation_space_top_k"],
        admission_den_power=config["admission_den_power"],
        admission_den_power_qk=config["admission_den_power_qk"],
        admission_den_power_v=config["admission_den_power_v"],
        admission_den_power_rst=config["admission_den_power_rst"],
        srw_composition_mode=config["srw_composition_mode"],
        tau_init_attn_q=config["tau_init_attn_q"],
        tau_init_attn_k=config["tau_init_attn_k"],
        tau_init_attn_v=config["tau_init_attn_v"],
        tau_init_rst=config["tau_init_rst"],
    )


def _variables(model=None, seed=0):
    model = model or _model()
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    variables = model.init(
        {
            "params": jax.random.PRNGKey(seed),
            "dropout": jax.random.PRNGKey(seed + 1),
        },
        tokens,
        deterministic=True,
    )
    return model, tokens, variables


def _count(tree):
    return sum(int(value.size) for value in jax.tree.leaves(tree))


def _path(tree, *parts):
    value = tree
    for part in parts:
        value = value[part]
    return value


def _execution_kwargs():
    return {
        "soft_gate_temperature": 0.07,
        "soft_gate_boundary_power": 2.0,
        "admission_den_power": 1.0,
        "srw_composition_mode": "linear_angular",
        "heat_kernel_beta": 4.0,
    }


def test_config_validation_rejects_noncanonical_and_removed_schema():
    config = _model_config()
    assert v4174.resolve_operation_space_config(config) == (4, 2)
    assert v4174.materialize_operation_space_config(dict(config))[
        "n_operation_spaces"] == 4
    invalid = (
        {"d_model": 12},
        {"operation_space_top_k": 5},
        {"n_operation_spaces": 5},
        {"n_q": 30},
        {"n_k": 30},
        {"n_v": 30},
        {"n_rst": 30},
        {"n_qk": 64},
        {"space_kernel_beta_qk": 4.0},
        {"router_dropout": 0.0},
    )
    for update in invalid:
        with pytest.raises(ValueError):
            v4174.materialize_operation_space_config({
                **config, **update})


def test_parameter_tree_is_exact_direct_read_schema():
    _, _, variables = _variables()
    params = variables["params"]
    router = params["router"]
    assert set(router) == {
        "space_route_proj",
        "space_read_vectors",
        "space_state_proj",
        "space_state_writeback",
        "q_operator_tau_proj",
        "k_operator_tau_proj",
        "v_operator_tau_proj",
        "rst_operator_tau_proj",
    }
    assert router["space_route_proj"]["kernel"].shape == (16, 4)
    assert router["space_read_vectors"].shape == (4, 4)
    assert router["space_state_proj"].shape == (4, 16, 4)
    assert router["space_state_writeback"].shape == (4, 4, 16)
    np.testing.assert_allclose(
        router["space_state_writeback"],
        jnp.swapaxes(router["space_state_proj"], -1, -2),
        atol=0,
        rtol=0,
    )
    space_read_gram = (
        router["space_read_vectors"] @ router["space_read_vectors"].T)
    np.testing.assert_allclose(
        space_read_gram, np.eye(4), atol=2e-5, rtol=2e-5)

    pool = params["neuron_pool"]
    assert set(pool) == {
        f"{route}_{kind}_vectors"
        for route in ("q", "k", "v", "rst")
        for kind in ("read", "write")
    }
    assert all(value.shape == (4, 8, 4) for value in pool.values())
    forbidden = (
        "qk_", "operator_key_", "operator_query_",
        "space_kernel_beta", "kernel_sketch")
    paths = (
        "/".join(str(part.key) for part in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(params)[0])
    assert not any(
        marker in path for path in paths for marker in forbidden)
    for route in ("q", "k", "v", "rst"):
        np.testing.assert_array_equal(
            router[f"{route}_operator_tau_proj"]["kernel"],
            np.zeros((4, 1), dtype=np.float32))


def test_tau_calibration_bias_attains_each_route_target():
    from scripts import train_jax

    _, tokens, variables = _variables(seed=31)
    params = variables["params"]
    targets = {"qk": 0.25, "v": 0.375, "rst": 0.50}
    cfg = {
        "model": {
            **_model_config(),
            "tau_init_mode": "quantile_frac",
            "tau_init_min": -0.95,
            "tau_init_max": 0.95,
            "tau_init_target_qk_frac": targets["qk"],
            "tau_init_target_v_frac": targets["v"],
            "tau_init_target_rst_frac": targets["rst"],
            "tau_init_calibration_tokens": 16,
        },
        "training": {
            "soft_gate_t_start": 0.07,
            "soft_gate_boundary_power_start": 2.0,
        },
    }
    tau_cfg = train_jax._v4164_tau_init_config(cfg)
    summary = train_jax._compute_srw_quantile_tau_init(
        params, tokens, cfg, tau_cfg)
    calibrated = train_jax._set_srw_quantile_tau_biases(
        params, summary, model_version=v4174.MODEL_VERSION)
    _, _, _, score_kwargs = train_jax._srw_selection_score_setup(
        calibrated, cfg, 16)
    scores = v4174._tau_init_calibration_scores(
        calibrated, tokens, **score_kwargs)
    attention_state, rst_state = v4174._sampled_layer_states(
        calibrated, tokens, **score_kwargs)
    router = calibrated["router"]
    for route, target in (
            ("q", targets["qk"]),
            ("k", targets["qk"]),
            ("v", targets["v"]),
            ("rst", targets["rst"])):
        np.testing.assert_array_equal(
            router[f"{route}_operator_tau_proj"]["kernel"],
            np.zeros((4, 1), dtype=np.float32))
        state = rst_state if route == "rst" else attention_state
        local = v4174._project_space_local_states(
            state, router["space_state_proj"])
        actual_tau = v4174._shared_tau_from_param(v4174._linear(
            router[f"{route}_operator_tau_proj"], local))
        active = float(jnp.mean(scores[route] > actual_tau))
        tolerance = max(0.02, 2.0 / scores[route].size)
        assert abs(active - target) <= tolerance, (
            route, active, target, tolerance)

    embedding_state = (
        calibrated["token_emb"]["embedding"][tokens]
        + calibrated["pos_emb"]["embedding"][
            jnp.arange(tokens.shape[1])[None, :]])
    naive_rst = v4174._shared_layer_norm(
        embedding_state,
        calibrated["block_0"]["norm2"]["scale"],
        calibrated["block_0"]["norm2"]["bias"],
    ).reshape(rst_state.shape)
    assert not np.allclose(rst_state, naive_rst)


def test_projection_and_direct_read_shapes_and_equivalence():
    state = jax.random.normal(jax.random.PRNGKey(1), (7, 16))
    projection = jax.random.normal(
        jax.random.PRNGKey(2), (4, 16, 4))
    local = v4174._project_space_local_states(state, projection)
    assert local.shape == (4, 7, 4)
    reads = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 4))
    read_value, rho_reused, local_norm = v4174._direct_read_match(
        local, reads)
    assert read_value.shape == (4, 7, 8)
    assert rho_reused.shape == (4, 7, 8)
    assert local_norm.shape == (4, 7, 1)
    rho_direct = jnp.einsum(
        "mtr,mnr->mtn",
        v4174.forward_unit_direction(local),
        v4174.forward_unit_direction(reads),
    )
    np.testing.assert_allclose(
        rho_reused, rho_direct, atol=6e-3, rtol=6e-3)


def test_space_gate_is_fixed_nonsoftmax_topk_relu_squared_gate():
    scores = jnp.array([
        [0.1, 2.0, -1.0, 1.0],
        [-4.0, -3.0, -2.0, -1.0],
    ])
    routing = v4174._space_gate_from_scores(scores, 2)
    gate = routing["space_gate"]
    weights = routing["dense_space_weights"]
    np.testing.assert_array_equal(
        np.asarray(gate[0] == 0.0),
        np.asarray([True, False, True, False]))
    assert float(gate[1].sum()) == 0.0
    assert float(weights[1].sum()) == 0.0
    assert not np.isclose(float(weights[0].sum()), 1.0)
    assert bool(jnp.all(jnp.isfinite(weights)))
    source = inspect.getsource(v4174._space_gate_from_scores)
    assert "softmax" not in source
    assert "top_k" in source
    assert "relu" in source
    assert "sqrt" in source

    local_output = jnp.ones((4, 2, 3))
    writeback = jnp.ones((4, 3, 5))
    zero_route = v4174._space_weighted_writeback(
        local_output,
        jnp.zeros((2, 4)),
        writeback,
        1.0,
    )
    np.testing.assert_array_equal(zero_route, np.zeros((2, 5)))


def test_qk_banks_and_execution_are_fully_separate():
    paired_factories = (
        "make_sharded_srw_paired",
        "make_sharded_srw_paired_minimal",
        "make_sharded_srw_paired_diagnostics_minimal",
        "make_sharded_srw_paired_retention_minimal",
        "make_sharded_srw_paired_suppression_minimal",
        "make_sharded_srw_paired_trajectory_minimal",
    )
    assert not any(hasattr(v4174, name) for name in paired_factories)
    _, _, variables = _variables(seed=7)
    params = variables["params"]
    pool = params["neuron_pool"]
    assert "q_read_vectors" in pool and "k_read_vectors" in pool
    assert "q_write_vectors" in pool and "k_write_vectors" in pool
    assert not np.array_equal(
        np.asarray(pool["q_read_vectors"]),
        np.asarray(pool["k_read_vectors"]))
    assert not np.array_equal(
        np.asarray(pool["q_write_vectors"]),
        np.asarray(pool["k_write_vectors"]))
    local = jax.random.normal(jax.random.PRNGKey(8), (4, 6, 4))
    router = params["router"]
    q_tau = v4174._linear(router["q_operator_tau_proj"], local)
    k_tau = v4174._linear(router["k_operator_tau_proj"], local)
    q_output = v4174._rw_compose_space_dense(
        local, pool["q_read_vectors"], pool["q_write_vectors"],
        q_tau, **_execution_kwargs())
    k_output = v4174._rw_compose_space_dense(
        local, pool["k_read_vectors"], pool["k_write_vectors"],
        k_tau, **_execution_kwargs())
    assert q_output.shape == k_output.shape == (4, 6, 4)
    assert not np.allclose(q_output, k_output)
    source = inspect.getsource(v4174.DAWN_SRW_V4174.__call__)
    assert "_qk_shared_read" not in source
    assert 'bank_values[route]' in source


def test_rst_routing_local_tau_and_output_use_fresh_post_attention_state():
    _, _, variables = _variables(seed=9)
    params = variables["params"]
    router = params["router"]
    pool = params["neuron_pool"]
    state_before = jax.random.normal(jax.random.PRNGKey(10), (8, 16))
    state_after = state_before + jnp.linspace(
        -0.4, 0.5, state_before.size).reshape(state_before.shape)
    learned_tau = {
        "kernel": jax.random.normal(
            jax.random.PRNGKey(11),
            router["rst_operator_tau_proj"]["kernel"].shape) * 0.1,
        "bias": router["rst_operator_tau_proj"]["bias"],
    }

    def rst_values(state):
        routing = v4174._compute_space_routing(
            state,
            router["space_route_proj"]["kernel"],
            router["space_read_vectors"],
            2,
        )
        local = v4174._project_space_local_states(
            state, router["space_state_proj"])
        tau = v4174._linear(learned_tau, local)
        output = v4174._rw_compose_space_dense(
            local,
            pool["rst_read_vectors"],
            pool["rst_write_vectors"],
            tau,
            **_execution_kwargs(),
        )
        return routing["space_scores"], local, tau, output

    before = rst_values(state_before)
    after = rst_values(state_after)
    assert all(
        not np.allclose(left, right)
        for left, right in zip(before, after))
    source = inspect.getsource(v4174.DAWN_SRW_V4174.__call__)
    assert "flat_rst_state" in source
    assert "rst_routing, rst_local = route_and_local(flat_rst_state)" in source
    assert "execute(\"rst\", rst_local, rst_tau)" in source


def test_forward_loss_gradient_jit_diagnostics_and_analysis_smoke():
    model, tokens, variables = _variables(seed=12)
    params = variables["params"]
    eager = model.apply(
        {"params": params},
        tokens,
        deterministic=True,
        rngs={"dropout": jax.random.PRNGKey(13)},
    )
    assert eager["logits"].shape == (1, 16, 64)

    def loss_fn(current):
        result = model.apply(
            {"params": current},
            tokens,
            labels=tokens,
            deterministic=True,
            minimal_runtime_profile="diagnostics",
            rngs={"dropout": jax.random.PRNGKey(14)},
        )
        return result["loss"], result

    (loss, diagnostics), gradients = jax.value_and_grad(
        loss_fn, has_aux=True)(params)
    assert bool(jnp.isfinite(loss))
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree.leaves(gradients))
    required_paths = (
        ("router", "space_route_proj", "kernel"),
        ("router", "space_read_vectors"),
        ("router", "space_state_proj"),
        ("router", "space_state_writeback"),
        ("neuron_pool", "q_read_vectors"),
        ("neuron_pool", "q_write_vectors"),
        ("neuron_pool", "k_read_vectors"),
        ("neuron_pool", "k_write_vectors"),
        ("neuron_pool", "v_read_vectors"),
        ("neuron_pool", "v_write_vectors"),
        ("neuron_pool", "rst_read_vectors"),
        ("neuron_pool", "rst_write_vectors"),
        ("router", "q_operator_tau_proj", "kernel"),
        ("router", "k_operator_tau_proj", "kernel"),
        ("router", "v_operator_tau_proj", "kernel"),
        ("router", "rst_operator_tau_proj", "kernel"),
    )
    for path in required_paths:
        gradient = _path(gradients, *path).astype(jnp.float32)
        assert float(jnp.linalg.norm(gradient)) > 0.0, path

    for stage in ("attention", "rst"):
        for suffix in (
                "gate_mass_mean", "gate_den_mean",
                "active_count_mean", "zero_gate_frac",
                "top1_rate", "usage_min", "usage_max", "usage_std"):
            assert bool(jnp.isfinite(
                diagnostics[f"{stage}_space_{suffix}"]))
    for route in ("q", "k", "v", "rst"):
        assert bool(jnp.isfinite(
            diagnostics[f"{route}_operator_gate_mass_mean"]))

    jitted = jax.jit(lambda current, ids: model.apply(
        {"params": current},
        ids,
        deterministic=True,
        rngs={"dropout": jax.random.PRNGKey(15)},
    ))(params, tokens)
    assert jitted["logits"].shape == (1, 16, 64)

    analysis = v4174.analysis_forward(
        params, _model_config(), tokens)
    assert analysis["space_read_norm"].shape == (4,)
    assert analysis["q_read_vector_covariance_effective_rank"].shape == (4,)
    assert bool(jnp.all(jnp.isfinite(
        analysis["space_read_pairwise_cosine"])))


def test_symbolic_count_matches_tree_and_canonical_budget():
    _, _, variables = _variables()
    tiny = v4174.symbolic_parameter_count(_model_config())
    assert tiny["total"] == _count(variables["params"])
    assert tiny["learned_key_tables"] == 0
    assert tiny["bilinear_probe_matrices"] == 0
    assert tiny["operator_query_projections"] == 0

    canonical = yaml.safe_load(
        CANONICAL_CONFIG.read_text(encoding="utf-8"))["model"]
    counts = v4174.symbolic_parameter_count(canonical)
    assert counts["total"] == 214_502_404
    assert canonical["n_q"] == canonical["n_k"] == 6_160
    assert canonical["n_q"] + canonical["n_k"] == 12_320
    for name in ("n_q", "n_k", "n_v", "n_rst"):
        assert canonical[name] % canonical["n_operation_spaces"] == 0


def test_trainer_builds_only_new_v4174_schema_and_direct_diagnostics():
    from scripts import train_jax

    model_cfg = _model_config()
    cfg = {
        "model": model_cfg,
        "training": {
            "n_chunks_q": 1,
            "n_chunks_k": 1,
            "n_chunks_v": 1,
            "n_chunks_rst": 1,
            "tau_lr_mult": 0.001,
        },
    }
    kwargs = train_jax._dawn_srw_kwargs(cfg)
    assert kwargs["n_q"] == 32 and kwargs["n_k"] == 32
    assert "n_qk" not in kwargs
    assert "n_know" not in kwargs
    assert "operator_key_mode" not in kwargs
    assert "router_dropout" not in kwargs
    built = train_jax.build_model_from_config(cfg)
    assert isinstance(built, v4174.DAWN_SRW_V4174)
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "model"),
    )
    sharded = train_jax.build_canonical_sharded_fns(cfg, mesh)
    assert all(
        name in sharded
        for name in (
            "q_space_dense", "k_space_dense",
            "v_space_dense", "rst_space_dense"))
    assert not any("qk" in name for name in sharded)
    assert all(
        getattr(sharded[f"{route}_space_dense"],
                "_v4174_direct_read_matmuls") == 1
        for route in ("q", "k", "v", "rst"))
    sharded_source = inspect.getsource(
        v4174._make_sharded_space_dense_direct)
    assert sharded_source.count('"mtr,mnr->mtn"') == 1

    _, tokens, variables = _variables()
    params = variables["params"]
    local = jax.random.normal(jax.random.PRNGKey(21), (4, 6, 4))
    raw_tau = v4174._linear(
        params["router"]["q_operator_tau_proj"], local)
    direct_sharded = sharded["q_space_dense"](
        local,
        raw_tau,
        jnp.ones(local.shape[:2], dtype=jnp.bool_),
        params["neuron_pool"]["q_read_vectors"],
        params["neuron_pool"]["q_write_vectors"],
        0.07, 0.07, 2.0, 2.0, 0.0,
    )
    direct_reference = v4174._rw_compose_space_dense(
        local,
        params["neuron_pool"]["q_read_vectors"],
        params["neuron_pool"]["q_write_vectors"],
        raw_tau,
        soft_gate_temperature=0.07,
        soft_gate_boundary_power=2.0,
        admission_den_power=0.5,
        srw_composition_mode="linear_angular",
        heat_kernel_beta=v4174.DEFAULT_HEAT_KERNEL_BETA,
    )
    np.testing.assert_allclose(
        direct_sharded, direct_reference, atol=1e-5, rtol=1e-5)
    sharded_forward = built.apply(
        {"params": params},
        tokens,
        deterministic=True,
        sharded_fns=sharded,
        rngs={"dropout": jax.random.PRNGKey(22)},
    )
    assert sharded_forward["logits"].shape == (1, 16, 64)
    diagnostics_fns = train_jax.build_canonical_sharded_fns(
        cfg, mesh, for_eval=True,
        kernel_profile="production_diagnostics")
    sharded_diagnostics = built.apply(
        {"params": params},
        tokens,
        labels=tokens,
        deterministic=True,
        sharded_fns=diagnostics_fns,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.PRNGKey(23)},
    )
    assert bool(jnp.isfinite(
        sharded_diagnostics["q_operator_gate_mass_mean"]))
    assert bool(jnp.isfinite(
        sharded_diagnostics["attention_space_gate_mass_mean"]))
    geometry = train_jax.create_geometry_step(
        max_sample=32,
        model_version=v4174.MODEL_VERSION,
    )(variables["params"])
    pool_diagnostics = train_jax._pool_param_diagnostics(
        {"neuron_pool": variables["params"]["neuron_pool"]},
        full=True,
        model=built,
    )
    assert geometry
    assert pool_diagnostics
    assert any(key.startswith("attn_q_read_") for key in pool_diagnostics)
    assert any(key.startswith("attn_k_read_") for key in pool_diagnostics)


def test_actual_create_train_step_updates_with_complete_compact_schema():
    from scripts import train_jax

    model, tokens, variables = _variables(seed=41)
    params = variables["params"]
    optimizer = optax.adam(1.0e-3)
    opt_state = optimizer.init(params)
    cfg = {
        "model": _model_config(),
        "training": {
            "batch_size": 1,
            "weight_decay": 0.0,
            "pool_weight_decay": 0.0,
            "inactive_aux_enabled": False,
            "soft_gate_t_start": 0.07,
            "soft_gate_t_final": 0.07,
            "soft_gate_boundary_power_start": 2.0,
            "soft_gate_boundary_power_mid": 2.0,
            "soft_gate_boundary_power_final": 2.0,
        },
    }
    train_step = train_jax.create_canonical_train_step(
        model, optimizer, cfg, None, None, total_training_steps=2)
    token_embedding_before = np.array(
        params["token_emb"]["embedding"], copy=True)
    dummy_drift = {
        "attn_qk_op_key": jnp.float32(0.0),
        "attn_v_op_key": jnp.float32(0.0),
        "rst_op_key": jnp.float32(0.0),
    }
    new_params, new_opt_state, metrics = train_step(
        params,
        opt_state,
        tokens,
        tokens,
        jnp.ones_like(tokens, dtype=jnp.bool_),
        jax.random.PRNGKey(42),
        dummy_drift,
        jnp.int32(0),
    )
    jax.block_until_ready(metrics["total_loss"])
    assert bool(jnp.isfinite(metrics["total_loss"]))
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree.leaves(new_params))
    assert jax.tree.leaves(new_opt_state)
    assert not np.array_equal(
        token_embedding_before,
        np.asarray(new_params["token_emb"]["embedding"]))
    assert set(metrics) == set(train_jax.V4174_COMPACT_TRAIN_METRIC_NAMES)
    assert all(
        bool(jnp.isfinite(metrics[name]))
        for name in train_jax.V4174_DIRECT_RW_GRADIENT_METRIC_NAMES)
    assert not any(
        name in metrics
        for name in train_jax.V417X_SHARED_PROBE_GRADIENT_METRIC_NAMES)


def test_model_info_describes_only_canonical_architecture():
    info = "\n".join(_model().get_model_info())
    assert "D=16, M=4, R=4" in info
    assert "explicit space read vectors" in info
    assert "non-softmax" in info
    assert "read vector itself is the operator key" in info
    assert "Q/K/V/RST are fully separate" in info
    assert "RST recomputes routing after attention" in info
    assert "physical all-space dense" in info
    assert "kernel sketch" not in info
    assert "generalized" not in info
