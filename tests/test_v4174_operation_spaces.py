import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from models import dawn_srw_v4174 as v4174


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = (
    ROOT / "configs" /
    "train_config_v4174_400M_c4_40B_v4_64_space24_top2_direct_read.yaml")


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
        gradient_checkpointing=config.get(
            "gradient_checkpointing", False),
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


def _fp32_rw_reference(
        local, read_vectors, write_vectors, raw_tau, *,
        temperature, boundary_power, den_power):
    read_unit = v4174.forward_unit_direction(read_vectors)
    write_unit = v4174.forward_unit_direction(write_vectors)
    read_value = v4174._control_einsum_f32(
        "mtr,mnr->mtn", local, read_unit)
    local_norm = jnp.maximum(
        jnp.linalg.norm(local.astype(jnp.float32), axis=-1, keepdims=True),
        jnp.float32(v4174.RW_FORWARD_NORM_EPS))
    tau = v4174._shared_tau_from_param(raw_tau)
    rho = jnp.clip(read_value / local_norm, -1.0, 1.0)
    margin, gate, depth, execution_weight, _ = (
        v4174._shared_compute_admission_drive(
            rho, tau, jnp.float32(temperature),
            boundary_power=jnp.float32(boundary_power),
            effective_active_eps=jnp.float32(1.0e-6),
            execution_prune_eps=jnp.float32(0.0),
            srw_composition_mode="linear_angular",
            heat_kernel_beta=jnp.float32(
                v4174.DEFAULT_HEAT_KERNEL_BETA)))
    raw_out = v4174._control_einsum_f32(
        "mtn,mnr->mtr", execution_weight * read_value, write_unit)
    gate_mass = gate.sum(axis=-1, keepdims=True)
    gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
    gate_max = gate.max(axis=-1, keepdims=True)
    active_count = (margin > 0.0).sum(
        axis=-1, keepdims=True).astype(jnp.float32)
    depth_sum = depth.sum(axis=-1, keepdims=True)
    gate_den = v4174._shared_composition_den(
        gate_mass, jnp.float32(den_power), "linear_angular")
    return (
        raw_out / gate_den,
        active_count,
        gate_mass,
        gate_sq,
        gate_max,
        depth_sum,
        tau,
        gate_den,
    )


def _fp32_fused_sharded(mesh):
    common = {
        "operation_space_top_k": 2,
        "srw_composition_mode": "linear_angular",
        "heat_kernel_beta": v4174.DEFAULT_HEAT_KERNEL_BETA,
        "soft_gate_effective_active_eps": 1.0e-6,
    }
    return {
        "attention_space_dense":
            v4174.make_sharded_attention_space_dense_fp32_reference(
                mesh,
                max_chunk_size_qk=8,
                max_chunk_size_v=8,
                admission_den_power_qk=0.5,
                admission_den_power_v=1.0,
                **common),
        "rst_space_dense":
            v4174.make_sharded_rst_space_dense_fp32_reference(
                mesh,
                max_chunk_size=8,
                admission_den_power=1.2,
                **common),
        "_v4174_kernel_profile": "production",
    }


def _fp32_separate_sharded(mesh):
    def make_route(den_power):
        def route_core(
                local, raw_tau, token_valid, read_vectors, write_vectors,
                temperature, temperature_final,
                boundary_power, boundary_power_final,
                execution_prune_eps):
            del (
                token_valid, temperature_final,
                boundary_power_final, execution_prune_eps)
            return _fp32_rw_reference(
                local, read_vectors, write_vectors, raw_tau,
                temperature=temperature,
                boundary_power=boundary_power,
                den_power=den_power)[0]

        return shard_map(
            route_core,
            mesh=mesh,
            in_specs=(
                P(None, "data", None),
                P(None, "data", None),
                P(None, "data"),
                P(None, "model", None),
                P(None, "model", None),
                P(), P(), P(), P(), P(),
            ),
            out_specs=P(None, "data", None),
            check_rep=False)

    return {
        f"{route}_space_dense": make_route(den_power)
        for route, den_power in (
            ("q", 0.5),
            ("k", 0.5),
            ("v", 1.0),
            ("rst", 1.2),
        )
    }


def _make_post_rw_collective_probe(mesh, *, u_before_psum):
    def core(
            sharded_raw_out, sharded_gate_mass,
            space_weights, space_state_writeback,
            route_scales, den_powers):
        raw_out = sharded_raw_out[0]
        gate_mass = sharded_gate_mass[0]
        global_gate_mass, gate_den = (
            v4174._global_dense_rw_den_sharded(
                gate_mass, den_powers, "linear_angular"))
        if u_before_psum:
            local_output = v4174._control_einsum_f32(
                "amtr,mrd->atd",
                raw_out / gate_den * space_weights * route_scales,
                space_state_writeback).astype(jnp.float32)
            output = jax.lax.psum(local_output, "model")
        else:
            space_results = jax.lax.psum(
                (raw_out / gate_den).astype(jnp.float32), "model")
            output = v4174._control_einsum_f32(
                "amtr,mrd->atd",
                space_results * space_weights * route_scales,
                space_state_writeback).astype(jnp.float32)
        return global_gate_mass, gate_den, output

    return shard_map(
        core,
        mesh=mesh,
        in_specs=(
            P("model", None, None, None, None),
            P("model", None, None, None, None),
            P(), P(), P(), P(),
        ),
        out_specs=(P(), P(), P()),
        check_rep=False)


def test_config_validation_rejects_noncanonical_and_removed_schema():
    config = _model_config()
    assert v4174.resolve_operation_space_config(config) == (4, 2)
    assert v4174.materialize_operation_space_config(dict(config))[
        "n_operation_spaces"] == 4
    nonfactorized = {
        **config,
        "n_operation_spaces": 3,
        "n_q": 30,
        "n_k": 30,
        "n_v": 30,
        "n_rst": 30,
    }
    assert nonfactorized["d_model"] != (
        nonfactorized["n_operation_spaces"] * nonfactorized["d_route"])
    assert v4174.materialize_operation_space_config(
        dict(nonfactorized))["n_operation_spaces"] == 3
    invalid = (
        {"d_route": 20},
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


def test_nonfactorized_operation_spaces_initialize_independent_coordinates():
    model = _model(
        n_operation_spaces=3,
        n_q=30,
        n_k=30,
        n_v=30,
        n_rst=30,
    )
    _, _, variables = _variables(model=model, seed=5)
    projection = variables["params"]["router"]["space_state_proj"]
    assert projection.shape == (3, 16, 4)
    gram = jnp.einsum("mdr,mds->mrs", projection, projection)
    np.testing.assert_allclose(
        gram,
        np.broadcast_to(np.eye(4), (3, 4, 4)),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert not np.allclose(projection[0], projection[1])
    counts = v4174.symbolic_parameter_count({
        **_model_config(),
        "n_operation_spaces": 3,
        "n_q": 30,
        "n_k": 30,
        "n_v": 30,
        "n_rst": 30,
    })
    assert counts["total"] == _count(variables["params"])


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


def test_qk_banks_are_independent_and_production_execution_is_paired():
    assert hasattr(v4174, "make_sharded_attention_space_dense_minimal")
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
    source = inspect.getsource(
        v4174._make_sharded_attention_space_dense)
    assert "jnp.stack((q_read, k_read), axis=0)" in source
    assert "jnp.stack((q_write, k_write), axis=0)" in source
    assert "qk_raw_tau" in source
    assert "q_tau_kernel" in source and "k_tau_kernel" in source
    assert "throughput_dot_bf16_f32" in source
    assert "_control_einsum_f32" in source
    assert "grouped_output" in source


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
    source = inspect.getsource(v4174._make_sharded_rst_space_dense)
    assert "_compute_space_routing" in source
    assert "throughput_dot_bf16_f32" in source
    assert '"mtr,mrd->td"' in source
    assert "return update, metrics" in source


def test_one_layer_checkpointing_is_numerically_equivalent():
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    plain_model = _model(
        n_layers=1, gradient_checkpointing=False)
    remat_model = _model(
        n_layers=1, gradient_checkpointing=True)
    variables = plain_model.init(
        {
            "params": jax.random.PRNGKey(71),
            "dropout": jax.random.PRNGKey(72),
        },
        tokens,
        deterministic=True,
    )
    params = variables["params"]

    def forward(model, current):
        return model.apply(
            {"params": current},
            tokens,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(73)},
        )["logits"]

    def loss(model, current):
        return model.apply(
            {"params": current},
            tokens,
            labels=tokens,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(74)},
        )["loss"]

    plain_logits = forward(plain_model, params)
    remat_logits = forward(remat_model, params)
    np.testing.assert_allclose(
        remat_logits, plain_logits, atol=0.0, rtol=0.0)
    plain_loss, plain_grads = jax.value_and_grad(
        lambda current: loss(plain_model, current))(params)
    remat_loss, remat_grads = jax.value_and_grad(
        lambda current: loss(remat_model, current))(params)
    np.testing.assert_allclose(
        remat_loss, plain_loss, atol=2.0e-6, rtol=2.0e-6)
    for path in (
            ("block_0", "attn", "expand_O", "kernel"),
            ("router", "space_state_proj"),
            ("neuron_pool", "q_read_vectors"),
            ("neuron_pool", "rst_write_vectors")):
        np.testing.assert_allclose(
            _path(remat_grads, *path),
            _path(plain_grads, *path),
            atol=2.0e-6,
            rtol=2.0e-6,
        )


def test_full_18_layer_checkpointing_changes_ir_and_reduces_temp_memory():
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 64
    plain_model = _model(
        n_layers=18, gradient_checkpointing=False)
    remat_model = _model(
        n_layers=18, gradient_checkpointing=True)
    variables = plain_model.init(
        {
            "params": jax.random.PRNGKey(75),
            "dropout": jax.random.PRNGKey(76),
        },
        tokens,
        deterministic=True,
    )
    params = variables["params"]

    def loss(model, current):
        return model.apply(
            {"params": current},
            tokens,
            labels=tokens,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(77)},
        )["loss"]

    plain_grad = lambda current: jax.grad(
        lambda value: loss(plain_model, value))(current)
    remat_grad = lambda current: jax.grad(
        lambda value: loss(remat_model, value))(current)
    plain_jaxpr = str(jax.make_jaxpr(plain_grad)(params))
    remat_jaxpr = str(jax.make_jaxpr(remat_grad)(params))
    assert "remat2" not in plain_jaxpr
    assert "remat2" in remat_jaxpr
    assert "length=18" in plain_jaxpr
    assert "length=18" in remat_jaxpr
    plain_lowered = jax.jit(plain_grad).lower(params)
    remat_lowered = jax.jit(remat_grad).lower(params)
    plain_hlo = plain_lowered.compiler_ir(
        dialect="hlo").as_hlo_text()
    remat_hlo = remat_lowered.compiler_ir(
        dialect="hlo").as_hlo_text()
    assert plain_hlo != remat_hlo
    assert "while" in plain_hlo.lower()
    assert "while" in remat_hlo.lower()
    plain_memory = plain_lowered.compile().memory_analysis()
    remat_memory = remat_lowered.compile().memory_analysis()
    assert (
        remat_memory.temp_size_in_bytes
        < plain_memory.temp_size_in_bytes)


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

    canonical_config = yaml.safe_load(
        CANONICAL_CONFIG.read_text(encoding="utf-8"))
    canonical = canonical_config["model"]
    training = canonical_config["training"]
    counts = v4174.symbolic_parameter_count(canonical)
    assert counts["total"] == 393_755_652
    assert canonical["d_model"] == 2048
    assert canonical["d_route"] == 256
    assert canonical["n_layers"] == 18
    assert canonical["n_heads"] == 32
    assert canonical["gradient_checkpointing"] is True
    assert canonical["n_operation_spaces"] == 24
    assert canonical["operation_space_top_k"] == 2
    assert canonical["n_q"] == canonical["n_k"] == 21_504
    assert canonical["n_q"] + canonical["n_k"] == 43_008
    assert canonical["n_v"] == 134_016
    assert canonical["n_rst"] == 269_952
    assert (
        canonical["n_q"] // canonical["n_operation_spaces"],
        canonical["n_k"] // canonical["n_operation_spaces"],
        canonical["n_v"] // canonical["n_operation_spaces"],
        canonical["n_rst"] // canonical["n_operation_spaces"],
    ) == (896, 896, 5_584, 11_248)
    assert (
        canonical["n_q"] // canonical["n_operation_spaces"] // 2,
        canonical["n_k"] // canonical["n_operation_spaces"] // 2,
        canonical["n_v"] // canonical["n_operation_spaces"] // 2,
        canonical["n_rst"] // canonical["n_operation_spaces"] // 2,
    ) == (448, 448, 2_792, 5_624)
    assert training["batch_size"] == 1_024
    assert training["mesh_data"] == 16
    assert training["mesh_model"] == 2
    assert training["gradient_accumulation_steps"] == 1
    assert (
        training["n_chunks_q"],
        training["n_chunks_k"],
        training["n_chunks_v"],
        training["n_chunks_rst"],
    ) == (1, 1, 2, 8)
    for name in ("n_q", "n_k", "n_v", "n_rst"):
        assert canonical[name] % canonical["n_operation_spaces"] == 0


def test_trainer_builds_only_new_v4174_schema_and_direct_diagnostics():
    from scripts import train_jax

    model_cfg = _model_config(gradient_checkpointing=True)
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
    assert kwargs["gradient_checkpointing"] is True
    built = train_jax.build_model_from_config(cfg)
    assert isinstance(built, v4174.DAWN_SRW_V4174)
    assert built.gradient_checkpointing is True
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "model"),
    )
    sharded = train_jax.build_canonical_sharded_fns(cfg, mesh)
    assert "attention_space_dense" in sharded
    assert "rst_space_dense" in sharded
    assert not any(
        name in sharded
        for name in (
            "q_space_dense", "k_space_dense",
            "v_space_dense"))
    attention_dense = sharded["attention_space_dense"]
    rst_dense = sharded["rst_space_dense"]
    fp32_sharded = _fp32_fused_sharded(mesh)
    fp32_separate_sharded = _fp32_separate_sharded(mesh)
    fp32_attention_dense = fp32_sharded["attention_space_dense"]
    fp32_rst_dense = fp32_sharded["rst_space_dense"]
    assert attention_dense._v4174_qk_paired is True
    assert attention_dense._v4174_dynamic_metric_flag is True
    assert attention_dense._v4174_dense_grouped_execution == "attention_qkv"
    assert rst_dense._v4174_dense_grouped_execution == "rst_end_to_end"
    assert rst_dense._v4174_dynamic_metric_flag is True
    assert attention_dense._v4174_chunk_remat_policy == "always"
    assert rst_dense._v4174_chunk_remat_policy == "always"
    assert attention_dense._v4174_throughput_precision == (
        "bf16_operands_f32_accum")
    assert rst_dense._v4174_throughput_precision == (
        "bf16_operands_f32_accum")
    assert fp32_attention_dense._v4174_throughput_precision == (
        "fp32_reference")
    assert fp32_rst_dense._v4174_throughput_precision == "fp32_reference"
    assert attention_dense._v4174_output_contract == (
        "q[T,D]", "k[T,D]", "v[T,D]", "scalars")
    assert rst_dense._v4174_output_contract == ("rst[T,D]", "scalars")

    _, tokens, variables = _variables()
    params = variables["params"]
    router = params["router"]
    pool = params["neuron_pool"]
    flat = jax.random.normal(jax.random.PRNGKey(21), (6, 16))
    qk_scale, v_scale, rst_scale = v4174._shared_pool_output_scales(16, 1)
    attention_args = (
        flat,
        router["space_route_proj"]["kernel"],
        router["space_read_vectors"],
        router["space_state_proj"],
        router["space_state_writeback"],
        router["q_operator_tau_proj"]["kernel"],
        router["q_operator_tau_proj"]["bias"],
        router["k_operator_tau_proj"]["kernel"],
        router["k_operator_tau_proj"]["bias"],
        router["v_operator_tau_proj"]["kernel"],
        router["v_operator_tau_proj"]["bias"],
        pool["q_read_vectors"], pool["q_write_vectors"],
        pool["k_read_vectors"], pool["k_write_vectors"],
        pool["v_read_vectors"], pool["v_write_vectors"],
        0.07, 0.07, 2.0, 0.0, qk_scale, v_scale,
        jnp.asarray(True, dtype=jnp.bool_))
    q_output, k_output, v_output, attention_metrics = attention_dense(
        *attention_args)
    (q_output_minimal, k_output_minimal, v_output_minimal,
     attention_metrics_minimal) = attention_dense(
         *attention_args[:-1], jnp.asarray(False, dtype=jnp.bool_))
    (q_output_fp32, k_output_fp32, v_output_fp32,
     attention_metrics_fp32) = fp32_attention_dense(*attention_args)
    for minimal, collected in zip(
            (q_output_minimal, k_output_minimal, v_output_minimal),
            (q_output, k_output, v_output)):
        np.testing.assert_array_equal(minimal, collected)
    def causal_output(q_value, k_value, v_value):
        def heads(value):
            return value.reshape(1, 6, 4, 4).transpose(0, 2, 1, 3)

        return v4174._causal_attention_core(
            heads(q_value), heads(k_value), heads(v_value),
            0.0, True, jax.random.PRNGKey(24),
            throughput_bf16=True)

    np.testing.assert_array_equal(
        causal_output(
            q_output_minimal, k_output_minimal, v_output_minimal),
        causal_output(q_output, k_output, v_output))
    assert all(
        float(value) == 0.0
        for value in attention_metrics_minimal.values())
    assert q_output.shape == k_output.shape == v_output.shape == (6, 16)
    assert all(value.shape == () for value in attention_metrics.values())
    assert all(
        value.shape == () for value in attention_metrics_fp32.values())
    assert all(
        value.dtype == jnp.float32
        for value in (
            q_output, k_output, v_output,
            q_output_fp32, k_output_fp32, v_output_fp32))

    routing = v4174._compute_space_routing(
        flat, router["space_route_proj"]["kernel"],
        router["space_read_vectors"], 2)
    np.testing.assert_allclose(
        attention_metrics["attention_space_gate_mass_mean"],
        routing["space_gate_mass"].mean(),
        atol=1e-5, rtol=1e-5)
    local = v4174._control_einsum_f32(
        "td,mdr->mtr", flat, router["space_state_proj"])
    reference_outputs = {}
    for route, scale, den in (
            ("q", qk_scale, 0.5),
            ("k", qk_scale, 0.5),
            ("v", v_scale, 1.0)):
        raw_tau = v4174._control_linear_f32(
            router[f"{route}_operator_tau_proj"], local)
        direct_details = _fp32_rw_reference(
            local,
            pool[f"{route}_read_vectors"],
            pool[f"{route}_write_vectors"],
            raw_tau,
            temperature=0.07,
            boundary_power=2.0,
            den_power=den)
        local_output = direct_details[0]
        np.testing.assert_allclose(
            attention_metrics_fp32[
                f"{route}_operator_gate_mass_mean"],
            direct_details[2].mean(), atol=1e-6, rtol=1e-6)
        weighted = (
            local_output
            * jnp.swapaxes(
                routing["dense_space_weights"], 0, 1)[..., None]
            * jnp.float32(scale))
        reference_outputs[route] = v4174._control_einsum_f32(
            "mtr,mrd->td", weighted,
            router["space_state_writeback"])
        assert abs(float(
            attention_metrics[
                f"{route}_operator_active_tau_frac"]
            - attention_metrics_fp32[
                f"{route}_operator_active_tau_frac"])) <= 0.001
    for actual, route in zip(
            (q_output_fp32, k_output_fp32, v_output_fp32),
            ("q", "k", "v")):
        np.testing.assert_allclose(
            actual, reference_outputs[route], atol=1e-5, rtol=0.0)
    for mixed, fp32 in zip(
            (q_output, k_output, v_output),
            (q_output_fp32, k_output_fp32, v_output_fp32)):
        mixed_norm = float(jnp.linalg.norm(mixed))
        fp32_norm = float(jnp.linalg.norm(fp32))
        assert 0.98 <= mixed_norm / max(fp32_norm, 1.0e-8) <= 1.02

    def rst_routing_after_attention(q_value, k_value, v_value, mixed):
        def heads(value):
            return value.reshape(1, 6, 4, 4).transpose(0, 2, 1, 3)

        attention = v4174._causal_attention_core(
            heads(q_value), heads(k_value), heads(v_value),
            0.0, True, jax.random.PRNGKey(25),
            throughput_bf16=mixed)
        attention = attention.transpose(0, 2, 1, 3).reshape(1, 6, 16)
        block = params["block_0"]
        attention = (
            v4174._throughput_linear_bf16_f32(
                block["attn"]["expand_O"], attention)
            if mixed else v4174._control_linear_f32(
                block["attn"]["expand_O"], attention))
        post_attention = flat.reshape(1, 6, 16) + attention
        rst_state = v4174._shared_layer_norm(
            post_attention,
            block["norm2"]["scale"],
            block["norm2"]["bias"]).reshape(6, 16)
        return v4174._compute_space_routing(
            rst_state,
            router["space_route_proj"]["kernel"],
            router["space_read_vectors"],
            2)

    mixed_rst_routing = rst_routing_after_attention(
        q_output, k_output, v_output, True)
    fp32_rst_routing = rst_routing_after_attention(
        q_output_fp32, k_output_fp32, v_output_fp32, False)
    top_k_agreement = jnp.mean(
        jnp.all(
            mixed_rst_routing["selected_ids"]
            == fp32_rst_routing["selected_ids"],
            axis=-1).astype(jnp.float32))
    assert float(top_k_agreement) >= 0.995

    rst_args = (
        flat,
        router["space_route_proj"]["kernel"],
        router["space_read_vectors"],
        router["space_state_proj"],
        router["space_state_writeback"],
        router["rst_operator_tau_proj"]["kernel"],
        router["rst_operator_tau_proj"]["bias"],
        pool["rst_read_vectors"], pool["rst_write_vectors"],
        0.07, 2.0, 0.0, rst_scale,
        jnp.asarray(True, dtype=jnp.bool_))
    rst_output, rst_metrics = rst_dense(*rst_args)
    rst_output_minimal, rst_metrics_minimal = rst_dense(
        *rst_args[:-1], jnp.asarray(False, dtype=jnp.bool_))
    rst_output_fp32, rst_metrics_fp32 = fp32_rst_dense(*rst_args)
    np.testing.assert_array_equal(rst_output_minimal, rst_output)
    assert all(
        float(value) == 0.0
        for value in rst_metrics_minimal.values())
    assert rst_output.shape == (6, 16)
    assert all(value.shape == () for value in rst_metrics.values())
    assert rst_output.dtype == rst_output_fp32.dtype == jnp.float32
    rst_tau = v4174._control_linear_f32(
        router["rst_operator_tau_proj"], local)
    rst_reference_details = _fp32_rw_reference(
        local,
        pool["rst_read_vectors"],
        pool["rst_write_vectors"],
        rst_tau,
        temperature=0.07,
        boundary_power=2.0,
        den_power=1.2)
    rst_reference_weighted = (
        rst_reference_details[0]
        * jnp.swapaxes(
            routing["dense_space_weights"], 0, 1)[..., None]
        * jnp.float32(rst_scale))
    rst_reference = v4174._control_einsum_f32(
        "mtr,mrd->td", rst_reference_weighted,
        router["space_state_writeback"])
    np.testing.assert_allclose(
        rst_output_fp32, rst_reference, atol=1e-5, rtol=0.0)
    assert abs(float(
        rst_metrics["rst_operator_active_tau_frac"]
        - rst_metrics_fp32["rst_operator_active_tau_frac"])) <= 0.001
    rst_norm_ratio = float(
        jnp.linalg.norm(rst_output)
        / jnp.maximum(jnp.linalg.norm(rst_output_fp32), 1.0e-8))
    assert 0.98 <= rst_norm_ratio <= 1.02
    sharded_forward = built.apply(
        {"params": params},
        tokens,
        deterministic=True,
        sharded_fns=sharded,
        rngs={"dropout": jax.random.PRNGKey(22)},
    )
    reference_forward = built.apply(
        {"params": params},
        tokens,
        deterministic=True,
        sharded_fns=fp32_sharded,
        rngs={"dropout": jax.random.PRNGKey(22)},
    )
    separate_fp32_forward = built.apply(
        {"params": params},
        tokens,
        deterministic=True,
        sharded_fns=fp32_separate_sharded,
        rngs={"dropout": jax.random.PRNGKey(22)},
    )
    assert sharded_forward["logits"].shape == (1, 16, 64)
    np.testing.assert_allclose(
        reference_forward["logits"], separate_fp32_forward["logits"],
        atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(
        sharded_forward["logits"], reference_forward["logits"],
        atol=2e-2, rtol=2e-2)

    def model_loss(current, sharded_fns):
        return built.apply(
            {"params": current},
            tokens,
            labels=tokens,
            deterministic=True,
            sharded_fns=sharded_fns,
            compute_accuracy=False,
            rngs={"dropout": jax.random.PRNGKey(24)},
        )["loss"]

    reference_loss, reference_grads = jax.jit(jax.value_and_grad(
        lambda current: model_loss(current, fp32_sharded)))(params)
    separate_fp32_loss, separate_fp32_grads = jax.jit(
        jax.value_and_grad(
            lambda current: model_loss(
                current, fp32_separate_sharded)))(params)
    fused_loss, fused_grads = jax.jit(jax.value_and_grad(
        lambda current: model_loss(current, sharded)))(params)
    assert bool(jnp.isfinite(reference_loss))
    assert bool(jnp.isfinite(fused_loss))
    assert abs(float(
        reference_loss - separate_fp32_loss)) <= 1.0e-6
    separate_gradient_dot = sum(
        jnp.sum(
            left.astype(jnp.float32) * right.astype(jnp.float32))
        for left, right in zip(
            jax.tree.leaves(reference_grads),
            jax.tree.leaves(separate_fp32_grads)))
    separate_gradient_sq = sum(
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree.leaves(separate_fp32_grads))
    reference_gradient_sq = sum(
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree.leaves(reference_grads))
    assert float(
        separate_gradient_dot / jnp.sqrt(jnp.maximum(
            separate_gradient_sq * reference_gradient_sq,
            1.0e-12))) >= 0.99999
    fp32_optimizer = optax.sgd(1.0e-3)
    fp32_opt_state = fp32_optimizer.init(params)
    reference_updates, _ = fp32_optimizer.update(
        reference_grads, fp32_opt_state, params)
    separate_updates, _ = fp32_optimizer.update(
        separate_fp32_grads, fp32_opt_state, params)
    reference_next = optax.apply_updates(params, reference_updates)
    separate_next = optax.apply_updates(params, separate_updates)
    for reference_value, separate_value in zip(
            jax.tree.leaves(reference_next),
            jax.tree.leaves(separate_next)):
        np.testing.assert_allclose(
            reference_value, separate_value,
            atol=1.0e-7, rtol=1.0e-6)
    assert float(
        jnp.abs(fused_loss - reference_loss)
        / jnp.maximum(jnp.abs(reference_loss), 1.0e-8)) <= 0.002
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree.leaves(fused_grads))
    assert all(
        not jnp.issubdtype(value.dtype, jnp.inexact)
        or value.dtype == jnp.float32
        for value in jax.tree.leaves(fused_grads))
    grad_diff_sq = sum(
        jnp.sum(jnp.square(
            left.astype(jnp.float32) - right.astype(jnp.float32)))
        for left, right in zip(
            jax.tree.leaves(fused_grads),
            jax.tree.leaves(reference_grads)))
    grad_reference_sq = sum(
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree.leaves(reference_grads))
    gradient_relative_error = jnp.sqrt(
        grad_diff_sq / jnp.maximum(grad_reference_sq, 1.0e-12))
    assert float(gradient_relative_error) < 0.05
    gradient_dot = sum(
        jnp.sum(
            left.astype(jnp.float32) * right.astype(jnp.float32))
        for left, right in zip(
            jax.tree.leaves(fused_grads),
            jax.tree.leaves(reference_grads)))
    fused_gradient_sq = sum(
        jnp.sum(jnp.square(value.astype(jnp.float32)))
        for value in jax.tree.leaves(fused_grads))
    gradient_cosine = gradient_dot / jnp.sqrt(
        jnp.maximum(
            fused_gradient_sq * grad_reference_sq, 1.0e-12))
    assert float(gradient_cosine) >= 0.99
    diagnostics_fns = train_jax.build_canonical_sharded_fns(
        cfg, mesh, for_eval=True,
        kernel_profile="production_diagnostics")
    analysis_fns = train_jax.build_canonical_sharded_fns(
        cfg, mesh, for_eval=True, analysis=True)
    assert all(
        name in analysis_fns
        for name in (
            "q_space_dense", "k_space_dense",
            "v_space_dense", "rst_space_dense"))
    assert all(
        getattr(
            analysis_fns[f"{route}_space_dense"],
            "_v4174_kernel_profile") == "production_diagnostics"
        for route in ("q", "k", "v", "rst"))
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


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires two CPU or accelerator devices")
def test_u_before_psum_fp32_math_gradient_optimizer_and_partition_parity():
    devices = np.asarray(jax.devices()[:2])
    mesh_one = jax.sharding.Mesh(
        devices[:1].reshape((1, 1)), ("data", "model"))
    mesh_two = jax.sharding.Mesh(
        devices.reshape((1, 2)), ("data", "model"))
    legacy_one = _make_post_rw_collective_probe(
        mesh_one, u_before_psum=False)
    changed_one = _make_post_rw_collective_probe(
        mesh_one, u_before_psum=True)
    legacy_two = _make_post_rw_collective_probe(
        mesh_two, u_before_psum=False)
    changed_two = _make_post_rw_collective_probe(
        mesh_two, u_before_psum=True)

    raw_out = jax.random.normal(
        jax.random.PRNGKey(101), (2, 3, 4, 6, 4))
    gate_mass = (
        jnp.abs(jax.random.normal(
            jax.random.PRNGKey(102), (2, 3, 4, 6, 1)))
        + jnp.float32(0.25))
    space_weights = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(103), (4, 6)),
        axis=0)[None, ..., None]
    space_state_writeback = jax.random.normal(
        jax.random.PRNGKey(104), (4, 4, 16))
    route_scales = jnp.asarray(
        (0.5, 0.5, 0.75), dtype=jnp.float32).reshape((3, 1, 1, 1))
    den_powers = jnp.asarray(
        (0.5, 0.5, 1.0), dtype=jnp.float32).reshape((3, 1, 1, 1))
    operands_two = (
        raw_out, gate_mass, space_weights,
        space_state_writeback, route_scales, den_powers)
    legacy_two_result = legacy_two(*operands_two)
    changed_two_result = changed_two(*operands_two)
    np.testing.assert_allclose(
        changed_two_result[0], legacy_two_result[0],
        atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        changed_two_result[1], legacy_two_result[1],
        atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        changed_two_result[2], legacy_two_result[2],
        atol=1.0e-5, rtol=1.0e-5)

    operands_one = (
        raw_out.sum(axis=0, keepdims=True),
        gate_mass.sum(axis=0, keepdims=True),
        space_weights,
        space_state_writeback,
        route_scales,
        den_powers,
    )
    legacy_one_result = legacy_one(*operands_one)
    changed_one_result = changed_one(*operands_one)
    for one_result in (legacy_one_result, changed_one_result):
        np.testing.assert_allclose(
            one_result[0], changed_two_result[0],
            atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            one_result[1], changed_two_result[1],
            atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            one_result[2], changed_two_result[2],
            atol=1.0e-5, rtol=1.0e-5)

    legacy_rst = legacy_two(
        raw_out[:, :1],
        gate_mass[:, :1],
        space_weights,
        space_state_writeback,
        route_scales[:1],
        jnp.float32(1.2).reshape((1, 1, 1, 1)),
    )[2]
    changed_rst = changed_two(
        raw_out[:, :1],
        gate_mass[:, :1],
        space_weights,
        space_state_writeback,
        route_scales[:1],
        jnp.float32(1.2).reshape((1, 1, 1, 1)),
    )[2]
    np.testing.assert_allclose(
        changed_rst, legacy_rst, atol=1.0e-5, rtol=1.0e-5)

    target = jax.random.normal(
        jax.random.PRNGKey(105), changed_two_result[2].shape)

    def loss_fn(probe, raw, gate, weights, writeback, scales):
        output = probe(
            raw, gate, weights, writeback, scales, den_powers)[2]
        return jnp.mean(jnp.square(output - target))

    differentiable_operands = operands_two[:5]
    legacy_loss, legacy_grads = jax.value_and_grad(
        lambda *values: loss_fn(legacy_two, *values),
        argnums=(0, 1, 2, 3, 4))(*differentiable_operands)
    changed_loss, changed_grads = jax.value_and_grad(
        lambda *values: loss_fn(changed_two, *values),
        argnums=(0, 1, 2, 3, 4))(*differentiable_operands)
    assert abs(float(changed_loss - legacy_loss)) <= 1.0e-6
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree.leaves(changed_grads))
    assert all(
        float(jnp.linalg.norm(value.astype(jnp.float32))) > 0.0
        for value in jax.tree.leaves(changed_grads))
    gradient_dot = sum(
        jnp.sum(left * right)
        for left, right in zip(
            jax.tree.leaves(legacy_grads),
            jax.tree.leaves(changed_grads)))
    legacy_gradient_sq = sum(
        jnp.sum(jnp.square(value))
        for value in jax.tree.leaves(legacy_grads))
    changed_gradient_sq = sum(
        jnp.sum(jnp.square(value))
        for value in jax.tree.leaves(changed_grads))
    gradient_cosine = gradient_dot / jnp.sqrt(jnp.maximum(
        legacy_gradient_sq * changed_gradient_sq, 1.0e-12))
    assert float(gradient_cosine) >= 0.99999

    optimizer = optax.sgd(1.0e-3)
    opt_state = optimizer.init(differentiable_operands)
    legacy_updates, _ = optimizer.update(
        legacy_grads, opt_state, differentiable_operands)
    changed_updates, _ = optimizer.update(
        changed_grads, opt_state, differentiable_operands)
    legacy_next = optax.apply_updates(
        differentiable_operands, legacy_updates)
    changed_next = optax.apply_updates(
        differentiable_operands, changed_updates)
    for legacy_value, changed_value in zip(
            jax.tree.leaves(legacy_next),
            jax.tree.leaves(changed_next)):
        np.testing.assert_allclose(
            changed_value, legacy_value, atol=1.0e-7, rtol=1.0e-6)


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires two CPU or accelerator devices")
def test_v4174_production_hlo_collective_shapes_and_kernel_partition_parity():
    devices = np.asarray(jax.devices()[:2])
    mesh_one = jax.sharding.Mesh(
        devices[:1].reshape((1, 1)), ("data", "model"))
    mesh_two = jax.sharding.Mesh(
        devices.reshape((1, 2)), ("data", "model"))
    sharded_one = _fp32_fused_sharded(mesh_one)
    sharded_two = _fp32_fused_sharded(mesh_two)
    _, _, variables = _variables(seed=106)
    params = variables["params"]
    router = params["router"]
    pool = params["neuron_pool"]
    flat = jax.random.normal(jax.random.PRNGKey(107), (6, 16))
    qk_scale, v_scale, rst_scale = v4174._shared_pool_output_scales(16, 1)
    attention_operands = (
        flat,
        router["space_route_proj"]["kernel"],
        router["space_read_vectors"],
        router["space_state_proj"],
        router["space_state_writeback"],
        router["q_operator_tau_proj"]["kernel"],
        router["q_operator_tau_proj"]["bias"],
        router["k_operator_tau_proj"]["kernel"],
        router["k_operator_tau_proj"]["bias"],
        router["v_operator_tau_proj"]["kernel"],
        router["v_operator_tau_proj"]["bias"],
        pool["q_read_vectors"], pool["q_write_vectors"],
        pool["k_read_vectors"], pool["k_write_vectors"],
        pool["v_read_vectors"], pool["v_write_vectors"],
        jnp.float32(0.07), jnp.float32(0.07),
        jnp.float32(2.0), jnp.float32(0.0),
        qk_scale, v_scale,
    )
    collect_metrics = jnp.asarray(False, dtype=jnp.bool_)
    attention_one = sharded_one["attention_space_dense"](
        *attention_operands, collect_metrics)
    attention_two = sharded_two["attention_space_dense"](
        *attention_operands, collect_metrics)
    for output_one, output_two in zip(
            attention_one[:3], attention_two[:3]):
        np.testing.assert_allclose(
            output_two, output_one, atol=1.0e-5, rtol=1.0e-5)

    rst_operands = (
        flat,
        router["space_route_proj"]["kernel"],
        router["space_read_vectors"],
        router["space_state_proj"],
        router["space_state_writeback"],
        router["rst_operator_tau_proj"]["kernel"],
        router["rst_operator_tau_proj"]["bias"],
        pool["rst_read_vectors"], pool["rst_write_vectors"],
        jnp.float32(0.07), jnp.float32(2.0), jnp.float32(0.0),
        rst_scale,
    )
    rst_one = sharded_one["rst_space_dense"](
        *rst_operands, collect_metrics)[0]
    rst_two = sharded_two["rst_space_dense"](
        *rst_operands, collect_metrics)[0]
    np.testing.assert_allclose(
        rst_two, rst_one, atol=1.0e-5, rtol=1.0e-5)

    def attention_outputs(*operands):
        return sharded_two["attention_space_dense"](
            *operands, collect_metrics)[:3]

    def rst_output(*operands):
        return sharded_two["rst_space_dense"](
            *operands, collect_metrics)[0]

    attention_hlo = jax.jit(attention_outputs).lower(
        *attention_operands).compiler_ir(
            dialect="hlo").as_hlo_text().lower()
    rst_hlo = jax.jit(rst_output).lower(
        *rst_operands).compiler_ir(
            dialect="hlo").as_hlo_text().lower()
    attention_collectives = [
        line for line in attention_hlo.splitlines()
        if "all-reduce" in line]
    rst_collectives = [
        line for line in rst_hlo.splitlines()
        if "all-reduce" in line]
    assert any(
        "f32[3,4,6,1]" in line
        for line in attention_collectives)
    assert any(
        "f32[3,6,16]" in line
        for line in attention_collectives)
    assert not any(
        "f32[3,4,6,4]" in line
        for line in attention_collectives)
    assert any(
        "f32[1,4,6,1]" in line
        for line in rst_collectives)
    assert any(
        "f32[6,16]" in line
        for line in rst_collectives)
    assert not any(
        "f32[1,4,6,4]" in line
        for line in rst_collectives)


def test_v4174_precision_partition_and_vocab_ce_contract():
    from models.vocab_parallel import make_vocab_parallel_ce_loss

    control = v4174.control_dot_f32(
        jnp.ones((3, 4), dtype=jnp.bfloat16),
        jnp.ones((4, 2), dtype=jnp.bfloat16),
        dimension_numbers=(((1,), (0,)), ((), ())))
    throughput = v4174.throughput_dot_bf16_f32(
        jnp.ones((3, 4), dtype=jnp.float32),
        jnp.ones((4, 2), dtype=jnp.float32),
        dimension_numbers=(((1,), (0,)), ((), ())))
    assert control.dtype == throughput.dtype == jnp.float32
    throughput_hlo = jax.jit(
        lambda left, right: v4174.throughput_dot_bf16_f32(
            left, right,
            dimension_numbers=(((1,), (0,)), ((), ())))
    ).lower(
        jnp.ones((3, 4), dtype=jnp.float32),
        jnp.ones((4, 2), dtype=jnp.float32),
    ).compiler_ir(dialect="hlo").as_hlo_text().lower()
    assert "bf16" in throughput_hlo
    assert "f32" in throughput_hlo
    assert "jax_default_matmul_precision" not in inspect.getsource(v4174)

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "model"),
    )
    mixed_ce = make_vocab_parallel_ce_loss(
        mesh,
        logical_vocab_size=16,
        vocab_size_padded=16,
        token_chunk_size=8,
        throughput_bf16_f32=True)
    fp32_ce = make_vocab_parallel_ce_loss(
        mesh,
        logical_vocab_size=16,
        vocab_size_padded=16,
        token_chunk_size=8,
        throughput_bf16_f32=False)
    hidden = jax.random.normal(jax.random.PRNGKey(81), (1, 8, 4))
    embedding = jax.random.normal(jax.random.PRNGKey(82), (16, 4))
    labels = jnp.arange(8, dtype=jnp.int32)[None, :] % 16
    valid = jnp.ones_like(labels, dtype=jnp.bool_)
    mixed_loss = mixed_ce(hidden, embedding, labels, valid)
    fp32_loss = fp32_ce(hidden, embedding, labels, valid)
    assert mixed_loss.dtype == fp32_loss.dtype == jnp.float32
    assert bool(jnp.isfinite(mixed_loss))
    assert float(
        jnp.abs(mixed_loss - fp32_loss)
        / jnp.maximum(jnp.abs(fp32_loss), 1.0e-8)) <= 0.002


def test_dynamic_metric_flag_preserves_logits_loss_gradients_and_jit_cache():
    from scripts import train_jax

    model, tokens, variables = _variables(seed=87)
    params = variables["params"]
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "model"),
    )
    sharded = _fp32_fused_sharded(mesh)
    dropout_key = jax.random.PRNGKey(88)

    @jax.jit
    def forward_logits(current, collect_metrics):
        return model.apply(
            {"params": current},
            tokens,
            deterministic=True,
            sharded_fns=sharded,
            collect_train_metrics=collect_metrics,
            rngs={"dropout": dropout_key},
        )["logits"]

    @jax.jit
    def train_math(current, collect_metrics):
        def loss_fn(candidate):
            result = model.apply(
                {"params": candidate},
                tokens,
                labels=tokens,
                deterministic=True,
                sharded_fns=sharded,
                compute_accuracy=False,
                collect_train_metrics=collect_metrics,
                rngs={"dropout": dropout_key},
            )
            return result["loss"], result

        (loss, result), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(current)
        return loss, result, grads

    false_flag = jnp.asarray(False, dtype=jnp.bool_)
    true_flag = jnp.asarray(True, dtype=jnp.bool_)
    false_logits = forward_logits(params, false_flag)
    true_logits = forward_logits(params, true_flag)
    np.testing.assert_allclose(
        false_logits, true_logits, atol=0.0, rtol=0.0)

    current_state = jax.random.normal(
        jax.random.PRNGKey(89), (1, tokens.shape[1], 16))
    block_params = params["block_0"]
    router_params = params["router"]
    pool_params = params["neuron_pool"]
    qk_scale, v_scale, rst_scale = v4174._shared_pool_output_scales(16, 1)

    def layer_intermediates(collect_metrics):
        normalized = v4174._shared_layer_norm(
            current_state,
            block_params["norm1"]["scale"],
            block_params["norm1"]["bias"])
        flat_attention = normalized.reshape((-1, 16))
        q_output, k_output, v_output, _ = (
            sharded["attention_space_dense"](
                flat_attention,
                router_params["space_route_proj"]["kernel"],
                router_params["space_read_vectors"],
                router_params["space_state_proj"],
                router_params["space_state_writeback"],
                router_params["q_operator_tau_proj"]["kernel"],
                router_params["q_operator_tau_proj"]["bias"],
                router_params["k_operator_tau_proj"]["kernel"],
                router_params["k_operator_tau_proj"]["bias"],
                router_params["v_operator_tau_proj"]["kernel"],
                router_params["v_operator_tau_proj"]["bias"],
                pool_params["q_read_vectors"],
                pool_params["q_write_vectors"],
                pool_params["k_read_vectors"],
                pool_params["k_write_vectors"],
                pool_params["v_read_vectors"],
                pool_params["v_write_vectors"],
                jnp.float32(0.07),
                jnp.float32(0.07),
                jnp.float32(2.0),
                jnp.float32(0.0),
                qk_scale,
                v_scale,
                collect_metrics))
        head_width = 4
        query = q_output.reshape(
            1, tokens.shape[1], 4, head_width).transpose(0, 2, 1, 3)
        key = k_output.reshape(
            1, tokens.shape[1], 4, head_width).transpose(0, 2, 1, 3)
        value = v_output.reshape(
            1, tokens.shape[1], 4, head_width).transpose(0, 2, 1, 3)
        attention_output = v4174._causal_attention_core(
            query, key, value, 0.0, True, jax.random.PRNGKey(90),
            throughput_bf16=False)
        attention_output = attention_output.transpose(
            0, 2, 1, 3).reshape(current_state.shape)
        attention_output = v4174._linear(
            block_params["attn"]["expand_O"], attention_output)
        post_attention = current_state + attention_output
        rst_normalized = v4174._shared_layer_norm(
            post_attention,
            block_params["norm2"]["scale"],
            block_params["norm2"]["bias"])
        rst_output, _ = sharded["rst_space_dense"](
            rst_normalized.reshape((-1, 16)),
            router_params["space_route_proj"]["kernel"],
            router_params["space_read_vectors"],
            router_params["space_state_proj"],
            router_params["space_state_writeback"],
            router_params["rst_operator_tau_proj"]["kernel"],
            router_params["rst_operator_tau_proj"]["bias"],
            pool_params["rst_read_vectors"],
            pool_params["rst_write_vectors"],
            jnp.float32(0.07),
            jnp.float32(2.0),
            jnp.float32(0.0),
            rst_scale,
            collect_metrics)
        return (
            q_output,
            k_output,
            v_output,
            attention_output,
            rst_output,
        )

    false_intermediates = layer_intermediates(false_flag)
    true_intermediates = layer_intermediates(true_flag)
    for false_value, true_value in zip(
            false_intermediates, true_intermediates):
        np.testing.assert_allclose(
            false_value, true_value, atol=1.0e-6, rtol=0.0)

    calls = [
        train_math(params, flag)
        for flag in (false_flag, true_flag, false_flag, true_flag)
    ]
    jax.block_until_ready(calls[-1][0])
    false_loss, false_result, false_grads = calls[0]
    true_loss, true_result, true_grads = calls[1]
    assert abs(float(false_loss - true_loss)) <= 1.0e-6

    def tree_cosine(left_tree, right_tree):
        dot = sum(
            jnp.sum(left.astype(jnp.float32) * right.astype(jnp.float32))
            for left, right in zip(
                jax.tree.leaves(left_tree),
                jax.tree.leaves(right_tree)))
        left_sq = sum(
            jnp.sum(jnp.square(value.astype(jnp.float32)))
            for value in jax.tree.leaves(left_tree))
        right_sq = sum(
            jnp.sum(jnp.square(value.astype(jnp.float32)))
            for value in jax.tree.leaves(right_tree))
        return dot / jnp.sqrt(jnp.maximum(left_sq * right_sq, 1.0e-12))

    assert float(tree_cosine(false_grads, true_grads)) >= 0.9999

    optimizer = optax.sgd(1.0e-3)
    opt_state = optimizer.init(params)
    false_updates, _ = optimizer.update(
        false_grads, opt_state, params)
    true_updates, _ = optimizer.update(
        true_grads, opt_state, params)
    false_params = optax.apply_updates(params, false_updates)
    true_params = optax.apply_updates(params, true_updates)
    for false_value, true_value in zip(
            jax.tree.leaves(false_params),
            jax.tree.leaves(true_params)):
        np.testing.assert_allclose(
            false_value, true_value, atol=1.0e-9, rtol=1.0e-7)

    optional_metrics = (
        *train_jax.LINEAR_DIRECT_TAU_REGULAR_REQUIRED_METRIC_NAMES,
        *train_jax.V4174_COMPOSITION_REGULAR_METRIC_NAMES,
        *train_jax.V4174_SELECTOR_METRIC_NAMES,
    )
    assert float(false_result["train_metrics_collected"]) == 0.0
    assert all(float(false_result[name]) == 0.0 for name in optional_metrics)
    assert float(true_result["train_metrics_collected"]) == 1.0
    assert all(
        value.shape == () and bool(jnp.isfinite(value))
        for name in optional_metrics
        for value in (true_result[name],))
    if hasattr(train_math, "_cache_size"):
        assert train_math._cache_size() == 1
    if hasattr(forward_logits, "_cache_size"):
        assert forward_logits._cache_size() == 1
    train_hlo = train_math.lower(
        params, false_flag).compiler_ir(
            dialect="hlo").as_hlo_text().lower()
    assert "conditional" in train_hlo

    mixed_cfg = {
        "model": _model_config(),
        "training": {
            "n_chunks_q": 1,
            "n_chunks_k": 1,
            "n_chunks_v": 1,
            "n_chunks_rst": 1,
            "tau_lr_mult": 0.001,
        },
    }
    mixed_sharded = train_jax.build_canonical_sharded_fns(
        mixed_cfg, mesh)

    @jax.jit
    def mixed_logits(current, collect_metrics):
        return model.apply(
            {"params": current},
            tokens,
            deterministic=True,
            sharded_fns=mixed_sharded,
            collect_train_metrics=collect_metrics,
            rngs={"dropout": dropout_key},
        )["logits"]

    @jax.jit
    def mixed_train_math(current, collect_metrics):
        def loss_fn(candidate):
            result = model.apply(
                {"params": candidate},
                tokens,
                labels=tokens,
                deterministic=True,
                sharded_fns=mixed_sharded,
                compute_accuracy=False,
                collect_train_metrics=collect_metrics,
                rngs={"dropout": dropout_key},
            )
            return result["loss"]

        return jax.value_and_grad(loss_fn)(current)

    mixed_false_logits = mixed_logits(params, false_flag)
    mixed_true_logits = mixed_logits(params, true_flag)
    mixed_output_max_abs_diff = jnp.max(jnp.abs(
        mixed_false_logits.astype(jnp.float32)
        - mixed_true_logits.astype(jnp.float32)))
    assert float(mixed_output_max_abs_diff) <= 1.0e-6
    mixed_false_loss, mixed_false_grads = mixed_train_math(
        params, false_flag)
    mixed_true_loss, mixed_true_grads = mixed_train_math(
        params, true_flag)
    assert abs(float(mixed_false_loss - mixed_true_loss)) <= 1.0e-6
    assert float(tree_cosine(
        mixed_false_grads, mixed_true_grads)) >= 0.9999
    if hasattr(mixed_train_math, "_cache_size"):
        assert mixed_train_math._cache_size() == 1


def test_v4174_mixed_precision_20_step_trajectory_is_stable():
    from scripts import train_jax

    model, tokens, variables = _variables(
        model=_model(gradient_checkpointing=True), seed=91)
    initial_params = variables["params"]
    assert all(
        not jnp.issubdtype(value.dtype, jnp.inexact)
        or value.dtype == jnp.float32
        for value in jax.tree.leaves(initial_params))
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "model"),
    )
    cfg = {
        "model": _model_config(gradient_checkpointing=True),
        "training": {
            "n_chunks_q": 1,
            "n_chunks_k": 1,
            "n_chunks_v": 1,
            "n_chunks_rst": 1,
            "tau_lr_mult": 0.001,
        },
    }
    mixed_sharded = train_jax.build_canonical_sharded_fns(cfg, mesh)
    fp32_sharded = _fp32_fused_sharded(mesh)
    optimizer = optax.adam(2.0e-4)

    def make_step(sharded_fns):
        def step(params, opt_state):
            def loss_fn(current):
                result = model.apply(
                    {"params": current},
                    tokens,
                    labels=tokens,
                    deterministic=True,
                    sharded_fns=sharded_fns,
                    compute_accuracy=False,
                    rngs={"dropout": jax.random.PRNGKey(92)},
                )
                active = jnp.stack([
                    result["attn_q_active_tau_frac"],
                    result["attn_k_active_tau_frac"],
                    result["attn_v_active_tau_frac"],
                    result["rst_active_tau_frac"],
                ])
                return result["loss"], active

            (loss, active), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(params)
            grad_sq = sum(
                jnp.sum(jnp.square(value.astype(jnp.float32)))
                for value in jax.tree.leaves(grads))
            grad_norm = jnp.sqrt(grad_sq)
            updates, new_opt_state = optimizer.update(
                grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, grad_norm, active

        return jax.jit(step)

    mixed_step = make_step(mixed_sharded)
    fp32_step = make_step(fp32_sharded)
    mixed_params = initial_params
    fp32_params = initial_params
    mixed_opt_state = optimizer.init(mixed_params)
    fp32_opt_state = optimizer.init(fp32_params)
    assert all(
        not jnp.issubdtype(value.dtype, jnp.inexact)
        or value.dtype == jnp.float32
        for value in (
            *jax.tree.leaves(mixed_opt_state),
            *jax.tree.leaves(fp32_opt_state)))
    mixed_history = []
    fp32_history = []
    for _ in range(20):
        (mixed_params, mixed_opt_state, mixed_loss,
         mixed_grad, mixed_active) = mixed_step(
             mixed_params, mixed_opt_state)
        (fp32_params, fp32_opt_state, fp32_loss,
         fp32_grad, fp32_active) = fp32_step(
             fp32_params, fp32_opt_state)
        mixed_history.append((
            float(mixed_loss), float(mixed_grad),
            np.asarray(mixed_active)))
        fp32_history.append((
            float(fp32_loss), float(fp32_grad),
            np.asarray(fp32_active)))

    mixed_losses = np.asarray([value[0] for value in mixed_history])
    fp32_losses = np.asarray([value[0] for value in fp32_history])
    mixed_grads = np.asarray([value[1] for value in mixed_history])
    fp32_grads = np.asarray([value[1] for value in fp32_history])
    assert np.all(np.isfinite(mixed_losses))
    assert np.all(np.isfinite(mixed_grads))
    loss_relative = np.abs(
        mixed_losses - fp32_losses) / np.maximum(
            np.abs(fp32_losses), 1.0e-8)
    grad_ratio = mixed_grads / np.maximum(fp32_grads, 1.0e-8)
    assert loss_relative[0] <= 0.002
    assert np.max(loss_relative) <= 0.02
    assert np.all((grad_ratio >= 0.8) & (grad_ratio <= 1.2))
    active_diffs = np.abs(
        np.stack([value[2] for value in mixed_history])
        - np.stack([value[2] for value in fp32_history]))
    assert np.max(active_diffs[0]) <= 0.001
    assert np.max(active_diffs) <= 0.01


def test_v4174_regular_console_maps_direct_operator_diagnostics(monkeypatch):
    from scripts import train_jax

    model, tokens, variables = _variables(seed=35)
    metrics = dict(model.apply(
        {"params": variables["params"]},
        tokens,
        labels=tokens,
        deterministic=True,
        minimal_runtime_profile="diagnostics",
        rngs={"dropout": jax.random.PRNGKey(36)},
    ))
    metrics.update({
        "tau_lr_mult": jnp.float32(0.001),
        **{
            key: jnp.float32(0.0)
            for key in train_jax.V4170_TAU_UPDATE_METRIC_NAMES
        },
    })
    for key in train_jax.V4174_COMPACT_TRAIN_METRIC_NAMES:
        metrics.setdefault(key, jnp.float32(0.0))
    win_avgs = {
        "loss": float(metrics["loss"]),
        "ce": float(metrics["loss"]),
        "aux": 0.0,
        "tau_reg": 0.0,
        "orth": 0.0,
        "div": 0.0,
        "acc": 0.0,
    }
    ctx = {
        "model_version": v4174.MODEL_VERSION,
        "lb_weight": 0.0,
        "tau_reg_weight": 0.0,
        "orth_weight": 0.0,
        "div_weight": 0.0,
        "dead_penalty_weight": 0.0,
        "current_lr": 0.0,
        "steps_per_sec": 1.0,
        "total_elapsed": 1.0,
        "epoch_elapsed": 1.0,
        "eta": 1.0,
        "s_per_it": 1.0,
        "progress": 0.0,
        "d_model_cfg": model.d_model,
        "n_layers_cfg": model.n_layers,
        "train_compute_accuracy": True,
    }
    rec = train_jax._build_regular_record(
        metrics, win_avgs, ctx, global_step=1, epoch=0)
    rec["raw_step_time_window"] = 1.0
    rec["logging_time"] = 0.0
    assert rec["_linear_direct_tau_regular_missing_metrics"] == ()
    np.testing.assert_allclose(
        rec["attn_q_active_tau_frac"],
        metrics["q_operator_active_tau_frac"])
    np.testing.assert_allclose(
        rec["attn_k_active_tau_frac"],
        metrics["k_operator_active_tau_frac"])
    np.testing.assert_allclose(
        rec["attn_qk_active_tau_frac"],
        0.5 * (
            metrics["q_operator_active_tau_frac"]
            + metrics["k_operator_active_tau_frac"]))

    messages = []
    monkeypatch.setattr(train_jax, "log_message", messages.append)
    train_jax._print_linear_direct_tau_regular_block(rec, ctx)
    assert any(message.startswith("  active:") for message in messages)
    compact = train_jax._v4170_compact_regular_jsonl_record(rec, ctx)
    assert tuple(compact) == train_jax.V4174_COMPACT_REGULAR_JSONL_KEYS


def test_actual_create_train_step_updates_with_complete_compact_schema():
    from scripts import train_jax

    model, tokens, variables = _variables(
        model=_model(gradient_checkpointing=True), seed=41)
    params = variables["params"]
    optimizer = optax.adam(1.0e-3)
    opt_state = optimizer.init(params)
    cfg = {
        "model": _model_config(gradient_checkpointing=True),
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
                "n_chunks_q": 1,
                "n_chunks_k": 1,
                "n_chunks_v": 1,
                "n_chunks_rst": 1,
            },
        }
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "model"),
    )
    sharded = train_jax.build_canonical_sharded_fns(cfg, mesh)
    train_step = train_jax.create_canonical_train_step(
        model, optimizer, cfg, sharded, mesh, total_training_steps=2)
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
        jnp.asarray(True, dtype=jnp.bool_),
    )
    jax.block_until_ready(metrics["total_loss"])
    assert all(
        bool(jnp.isfinite(metrics[name]))
        for name in ("total_loss", "ce_loss", "grad_norm"))
    assert all(
        getattr(value, "shape", None) == ()
        for value in metrics.values())
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree.leaves(new_params))
    assert jax.tree.leaves(new_opt_state)
    assert not np.array_equal(
        token_embedding_before,
        np.asarray(new_params["token_emb"]["embedding"]))
    assert set(metrics) == set(train_jax.V4174_COMPACT_TRAIN_METRIC_NAMES)
    assert all(
        name not in metrics
        for name in train_jax.V4174_DIRECT_RW_GRADIENT_METRIC_NAMES)
    assert not any(
        name in metrics
        for name in train_jax.V417X_SHARED_PROBE_GRADIENT_METRIC_NAMES)
    assert not any(
        name in train_jax.V4174_COMPACT_REGULAR_JSONL_REC_KEYS
        for name in train_jax.V4174_DIRECT_RW_GRADIENT_METRIC_NAMES)
    assert not any(
        name in train_jax.V4174_COMPACT_REGULAR_JSONL_REC_KEYS
        for name in train_jax.V417X_SHARED_PROBE_GRADIENT_METRIC_NAMES)
    assert all(
        name in metrics
        for name in train_jax.LINEAR_DIRECT_TAU_REGULAR_REQUIRED_METRIC_NAMES)
    assert float(metrics["train_metrics_collected"]) == 1.0

    stable_train_step = train_jax.create_canonical_train_step(
        model, optimizer, cfg, sharded, mesh, total_training_steps=4)
    newer_params, newer_opt_state, minimal_metrics = stable_train_step(
        new_params,
        new_opt_state,
        tokens,
        tokens,
        jnp.ones_like(tokens, dtype=jnp.bool_),
        jax.random.PRNGKey(43),
        dummy_drift,
        jnp.int32(1),
        jnp.asarray(False, dtype=jnp.bool_),
    )
    jax.block_until_ready(minimal_metrics["total_loss"])
    assert jax.tree.leaves(newer_params)
    assert jax.tree.leaves(newer_opt_state)
    assert float(minimal_metrics["train_metrics_collected"]) == 0.0
    assert all(
        float(minimal_metrics[name]) == 0.0
        for name in (
            *train_jax.LINEAR_DIRECT_TAU_REGULAR_REQUIRED_METRIC_NAMES,
            *train_jax.V4174_COMPOSITION_REGULAR_METRIC_NAMES,
            *train_jax.V4174_SELECTOR_METRIC_NAMES))
    for step_index, collect_metrics in enumerate(
            (True, False, True), start=2):
        newer_params, newer_opt_state, alternating_metrics = (
            stable_train_step(
                newer_params,
                newer_opt_state,
                tokens,
                tokens,
                jnp.ones_like(tokens, dtype=jnp.bool_),
                jax.random.PRNGKey(43 + step_index),
                dummy_drift,
                jnp.int32(step_index),
                jnp.asarray(collect_metrics, dtype=jnp.bool_),
            ))
        jax.block_until_ready(alternating_metrics["total_loss"])
        assert float(
            alternating_metrics["train_metrics_collected"]
        ) == float(collect_metrics)
    if hasattr(stable_train_step, "_cache_size"):
        assert stable_train_step._cache_size() == 1


def test_rare_analysis_step_uses_model_geometry_contract():
    from scripts import train_jax

    model, tokens, variables = _variables(seed=51)
    assert inspect.signature(
        v4174._direct_read_geometry_diagnostics
    ).parameters["max_tokens"].default == 128
    analysis_step = train_jax.create_analysis_step(
        model,
        total_training_steps=2,
        soft_gate_schedule_active=True,
        soft_gate_t_start=0.09,
        soft_gate_t_final=0.07,
        boundary_power_schedule_active=True,
        soft_gate_boundary_power_start=2.0,
        soft_gate_boundary_power_mid=2.0,
        soft_gate_boundary_power_final=2.0,
        admission_den_power=model.admission_den_power,
    )
    result = analysis_step(
        variables["params"],
        tokens,
        jnp.ones_like(tokens, dtype=jnp.bool_),
        jnp.int32(0),
    )
    jax.block_until_ready(result["q_direct_read_score_mean"])
    for route in ("q", "k", "v", "rst"):
        for statistic in ("mean", "std", "max"):
            assert bool(jnp.isfinite(
                result[f"{route}_direct_read_score_{statistic}"]))


def test_model_info_describes_only_canonical_architecture():
    info = "\n".join(_model().get_model_info())
    assert "D=16, M=4, R=4" in info
    assert "explicit space read vectors" in info
    assert "non-softmax" in info
    assert "read vector itself is the operator key" in info
    assert "Q/K/V/RST are fully separate" in info
    assert "RST recomputes routing after attention" in info
    assert "physical all-space dense" in info
    assert "independent D->R coordinates" in info
    assert "D=M*R" not in info
    assert "lax.scan" in info
    assert v4174.ATTENTION_CORE_NAME in info
    assert "kernel sketch" not in info
    assert "generalized" not in info
