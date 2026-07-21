from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from jax.sharding import Mesh

from models import dawn_srw_v4173 as v4173
from models import dawn_srw_v4172 as v4172
from scripts import train_jax


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ('data', 'model'))


def _kwargs(*, spaces: int = 8, top_k: int = 2) -> dict:
    return {
        'vocab_size': 16,
        'd_model': 6,
        'd_route': 3,
        'n_layers': 1,
        'n_heads': 2,
        'max_seq_len': 3,
        'dropout_rate': 0.0,
        'router_dropout': 0.0,
        'n_qk': 2,
        'n_v': 2,
        'n_rst': 16,
        'n_operation_spaces': spaces,
        'operation_space_top_k': top_k,
        'tau_init_attn_qk': -0.9,
        'tau_init_attn_v': -0.9,
        'tau_init_rst': -0.9,
        'admission_den_power': 1.0,
        'admission_den_power_qk': 0.5,
        'admission_den_power_v': 1.0,
        'admission_den_power_rst': 1.2,
    }


def _trainer_cfg(*, spaces: int = 8, top_k: int = 2) -> dict:
    kw = _kwargs(spaces=spaces, top_k=top_k)
    return {
        'model': {
            'model_version': train_jax.V4173_MODEL_VERSION,
            'operator_key_mode': 'generalized_bilinear_rw',
            'operator_query_mode': 'direct_state_projection',
            'vocab_size': kw['vocab_size'],
            'd_model': kw['d_model'],
            'd_route': kw['d_route'],
            'n_layers': kw['n_layers'],
            'n_heads': kw['n_heads'],
            'max_seq_len': kw['max_seq_len'],
            'n_qk': kw['n_qk'],
            'n_v': kw['n_v'],
            'n_rst': kw['n_rst'],
            'n_operation_spaces': spaces,
            'operation_space_top_k': top_k,
            'admission_den_power': 1.0,
            'admission_den_power_qk': 0.5,
            'admission_den_power_v': 1.0,
            'admission_den_power_rst': 1.2,
            'srw_composition_mode': 'linear_angular',
            'heat_kernel_beta': 2.0,
        },
        'training': {'mesh_model': 1, 'max_chunk_size': 2},
    }


def _init(spaces: int = 8, top_k: int = 2):
    model = v4173.DAWN_SRW_V4173(**_kwargs(
        spaces=spaces, top_k=top_k))
    tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    variables = model.init(
        {'params': jax.random.PRNGKey(1),
         'dropout': jax.random.PRNGKey(2)},
        tokens, deterministic=True)
    return model, variables, tokens


def test_single_space_defaults_preserve_tree_and_numerics() -> None:
    implicit = _kwargs(spaces=1, top_k=1)
    implicit.pop('n_operation_spaces')
    implicit.pop('operation_space_top_k')
    explicit = _kwargs(spaces=1, top_k=1)
    tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    rngs = {'params': jax.random.PRNGKey(3),
            'dropout': jax.random.PRNGKey(4)}
    implicit_model = v4173.DAWN_SRW_V4173(**implicit)
    explicit_model = v4173.DAWN_SRW_V4173(**explicit)
    implicit_vars = implicit_model.init(rngs, tokens, deterministic=True)
    explicit_vars = explicit_model.init(rngs, tokens, deterministic=True)
    implicit_shapes = jax.tree.map(lambda x: x.shape, implicit_vars['params'])
    explicit_shapes = jax.tree.map(lambda x: x.shape, explicit_vars['params'])
    assert implicit_shapes == explicit_shapes
    for left, right in zip(
            jax.tree.leaves(implicit_vars['params']),
            jax.tree.leaves(explicit_vars['params'])):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert 'rst_space_query' not in implicit_vars['params']['router']
    assert 'rst_space_keys' not in implicit_vars['params']['router']
    assert implicit_vars['params']['neuron_pool']['rst_read'].shape == (16, 3)
    sharded = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(spaces=1, top_k=1), _mesh())
    implicit_logits = implicit_model.apply(
        implicit_vars, tokens, deterministic=True,
        rngs={'dropout': jax.random.PRNGKey(21)},
        sharded_fns=sharded, minimal_train=True)['logits']
    explicit_logits = explicit_model.apply(
        explicit_vars, tokens, deterministic=True,
        rngs={'dropout': jax.random.PRNGKey(21)},
        sharded_fns=sharded, minimal_train=True)['logits']
    np.testing.assert_array_equal(
        np.asarray(implicit_logits), np.asarray(explicit_logits))


def test_old_checkpoint_config_materializes_single_space() -> None:
    old_model_cfg = _trainer_cfg(spaces=1, top_k=1)['model']
    old_model_cfg.pop('n_operation_spaces')
    old_model_cfg.pop('operation_space_top_k')
    requested = deepcopy(old_model_cfg)
    train_jax._validate_v4171_resume_compatibility(requested, old_model_cfg)
    assert requested['n_operation_spaces'] == 1
    assert requested['operation_space_top_k'] == 1
    assert old_model_cfg['n_operation_spaces'] == 1
    assert old_model_cfg['operation_space_top_k'] == 1


def test_multispace_shapes_selection_dense_parity() -> None:
    model, variables, tokens = _init()
    params = variables['params']
    pool = v4173._pool_params_with_operator_keys(params['neuron_pool'])
    router = params['router']
    assert pool['rst_read'].shape == (8, 2, 3)
    assert pool['rst_write'].shape == (8, 2, 3)
    assert pool['rst_op_key'].shape == (8, 2, 3)
    assert router['proj_rst']['kernel'].shape == (8, 6, 3)
    assert router['proj_rst']['bias'].shape == (8, 3)
    assert router['raw_tau_rst']['kernel'].shape == (8, 6, 1)
    assert router['raw_tau_rst']['bias'].shape == (8, 1)
    assert router['up_rst']['kernel'].shape == (8, 3, 6)
    assert router['rst_space_query']['kernel'].shape == (6, 3)
    assert router['rst_space_keys'].shape == (8, 3)

    sharded = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(), _mesh())
    assert getattr(
        sharded['rst_multispace_dense_minimal'],
        '_v4173_dense_grouped_execution') == 'all_spaces'
    x = jax.random.normal(jax.random.PRNGKey(5), (1, 3, 6))
    scalar = (jnp.float32(0.07), jnp.float32(0.07),
              jnp.float32(2.0), jnp.float32(4.0), jnp.float32(0.0))
    dense_actual, metrics, details = v4173._rst_multispace_dense_forward(
        x, pool, router, jax.random.PRNGKey(6),
        0.0, 0.0, True, sharded['rst_multispace_dense_minimal'],
        8, 2, 6, 1, *scalar, return_details=True)
    assert details['selected_space_ids'].shape == (1, 3, 2)
    assert details['selected_space_weights'].shape == (1, 3, 2)
    assert details['local_outputs'].shape == (1, 3, 2, 3)
    assert 'space_outputs' not in details
    assert 'all_space_outputs' not in details
    assert 'all_space_local_outputs' not in details
    np.testing.assert_allclose(
        np.asarray(details['selected_space_weights'].sum(axis=-1)), 1.0,
        rtol=1.0e-6, atol=1.0e-6)
    assert tuple(metrics) == v4173.RST_SPACE_METRIC_NAMES
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert not np.allclose(
        np.asarray(jnp.linalg.norm(dense_actual, axis=-1)), 1.0)

    flat_x = x.reshape((3, 6))
    dense_query = jnp.einsum(
        'td,mdr->mtr', flat_x, router['proj_rst']['kernel']
    ) + router['proj_rst']['bias'][:, None, :]
    dense_tau = jnp.einsum(
        'td,mdr->mtr', flat_x, router['raw_tau_rst']['kernel']
    ) + router['raw_tau_rst']['bias'][:, None, :]
    dense_local = sharded['rst_multispace_dense_minimal'](
        dense_query, pool['rst_op_key'], dense_tau,
        jnp.ones((8, 3), dtype=jnp.bool_),
        pool['rst_read'], pool['rst_write'], *scalar)
    assert not isinstance(dense_local, tuple)
    assert dense_local.shape == (8, 3, 3)
    _, _, rst_scale = v4173._pool_output_scales(6, 1)
    dense_space_outputs = jnp.einsum(
        'mtr,mrd->mtd', dense_local * rst_scale,
        router['up_rst']['kernel']).transpose((1, 0, 2))
    token_index = jnp.arange(3, dtype=jnp.int32)[:, None]
    selected_dense = dense_space_outputs[
        token_index, details['selected_space_ids'].reshape((3, 2))]
    dense_selected = (
        selected_dense
        * details['selected_space_weights'].reshape((3, 2, 1))
    ).sum(axis=1).reshape((1, 3, 6))
    np.testing.assert_allclose(
        np.asarray(dense_actual), np.asarray(dense_selected),
        rtol=2.0e-6, atol=2.0e-7)

    diagnostics = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(), _mesh(), kernel_profile='production_diagnostics')
    assert 'rst_multispace_dense_minimal' not in diagnostics
    assert getattr(
        diagnostics['rst_multispace_dense_diagnostics'],
        '_v4173_dense_grouped_diagnostics') == 'all_spaces'
    diagnostic_kernel_out = diagnostics[
        'rst_multispace_dense_diagnostics'](
            dense_query, pool['rst_op_key'], dense_tau,
            jnp.ones((8, 3), dtype=jnp.bool_),
            pool['rst_read'], pool['rst_write'], *scalar)
    assert isinstance(diagnostic_kernel_out, tuple)
    assert len(diagnostic_kernel_out) == 8
    assert diagnostic_kernel_out[0].shape == (8, 3, 3)
    assert all(value.shape == (8, 3, 1)
               for value in diagnostic_kernel_out[1:])
    diagnostic_out = model.apply(
        variables, tokens, labels=tokens, deterministic=True,
        rngs={'dropout': jax.random.PRNGKey(16)},
        sharded_fns=diagnostics, minimal_train=True,
        minimal_runtime_profile='diagnostics', analysis=False,
        compute_accuracy=False)
    for name in v4173.RST_SPACE_METRIC_NAMES:
        assert name in diagnostic_out
        assert np.isfinite(float(diagnostic_out[name]))
    for name in (
            'rst_active_tau_frac', 'rst_active_tau_count', 'rst_gate_mass',
            'rst_gate_den', 'rst_tau_mean'):
        assert np.isfinite(float(diagnostic_out[name]))
        assert float(diagnostic_out[name]) != 0.0
    assert not any(
        name.startswith(('per_space_', 'all_space_'))
        for name in diagnostic_out)
    for name in ('operator_gate', 'operator_rho', 'space_outputs'):
        assert name not in diagnostic_out
    for name in (
            'space_selected_requests', 'space_processed_requests',
            'space_all_processed', 'space_overflow_requests'):
        assert name not in diagnostic_out


def test_fused_commit_graph_has_no_full_space_model_width_tensor() -> None:
    n_spaces, token_count, d_route, d_model = 4, 3, 2, 7
    local = jnp.ones((n_spaces, token_count, d_route), dtype=jnp.float32)
    weights = jnp.full(
        (token_count, n_spaces), 1.0 / n_spaces, dtype=jnp.float32)
    up = jnp.ones((n_spaces, d_route, d_model), dtype=jnp.float32)
    closed = jax.make_jaxpr(v4173._rst_multispace_fused_commit)(
        local, weights, up, jnp.float32(1.0))
    intermediate_shapes = {
        tuple(var.aval.shape)
        for equation in closed.jaxpr.eqns
        for var in equation.outvars
        if hasattr(getattr(var, 'aval', None), 'shape')
    }
    assert (n_spaces, token_count, d_model) not in intermediate_shapes
    assert closed.out_avals[0].shape == (token_count, d_model)


def test_multispace_production_diagnostics_update_parity() -> None:
    _, variables, _ = _init()
    pool = v4173._pool_params_with_operator_keys(
        variables['params']['neuron_pool'])
    router = variables['params']['router']
    production = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(), _mesh())
    diagnostics = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(), _mesh(), kernel_profile='production_diagnostics')
    x = jax.random.normal(jax.random.PRNGKey(101), (1, 3, 6))
    kwargs = dict(
        d_model=6, n_layers=1, n_operation_spaces=8,
        operation_space_top_k=2, soft_gate_T_rst=jnp.float32(0.07),
        soft_gate_t_final=jnp.float32(0.07),
        soft_gate_boundary_power=jnp.float32(2.0),
        soft_gate_boundary_power_final=jnp.float32(4.0),
        execution_prune_eps=jnp.float32(0.0))
    production_out, _ = v4173._rst_forward_training_fast(
        x, pool, router, jax.random.PRNGKey(102),
        0.0, 0.0, True, production, **kwargs)
    diagnostics_result = v4173._rst_forward_production_diagnostics(
        x, pool, router, jax.random.PRNGKey(102),
        0.0, 0.0, True, diagnostics, **kwargs)
    np.testing.assert_allclose(
        np.asarray(production_out), np.asarray(diagnostics_result[0]),
        rtol=1.0e-5, atol=1.0e-5)


def test_multispace_diagnostic_aggregate_manual_reference() -> None:
    top_ids = jnp.asarray([[[0, 1], [1, 2]]], dtype=jnp.int32)
    top_weights = jnp.asarray(
        [[[0.25, 0.75], [0.60, 0.40]]], dtype=jnp.float32)

    def aggregate(values) -> jax.Array:
        return jnp.asarray(values, dtype=jnp.float32)[..., None]

    active_count = aggregate([[2, 4], [6, 8], [10, 12]])
    gate_mass = aggregate([[0.5, 2.0], [1.5, 0.25], [3.0, 0.75]])
    gate_sq = aggregate([[0.2, 1.5], [0.7, 0.1], [2.0, 0.3]])
    gate_max = aggregate([[0.3, 1.0], [0.8, 0.2], [1.2, 0.4]])
    angular_sum = aggregate([[1, 2], [3, 4], [5, 6]])
    tau = aggregate([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    gate_den = aggregate([[1.0, 2.0], [1.5, 1.0], [3.0, 1.0]])
    actual = v4173._rst_multispace_diagnostic_aggregates(
        top_ids, top_weights, active_count, gate_mass, gate_sq, gate_max,
        angular_sum, tau, gate_den, n_rst_per_space=20,
        composition_floor_mass=1.0, composition_mode='linear_angular')

    ids = np.asarray(top_ids).reshape((2, 2))
    weights = np.asarray(top_weights).reshape((2, 2))

    def select(values: jax.Array) -> np.ndarray:
        values_np = np.asarray(values)[..., 0]
        return np.stack(
            [values_np[ids[token], token] for token in range(2)])

    active_np = np.asarray(active_count)
    mass_np = np.asarray(gate_mass)
    depth = np.asarray(angular_sum) / np.maximum(active_np, 1.0)
    eff_n = np.square(mass_np) / np.maximum(np.asarray(gate_sq), 1.0e-8)
    top1 = np.asarray(gate_max) / np.maximum(mass_np, 1.0e-8)
    floor = (mass_np < 1.0).astype(np.float32)

    expected = {
        'rst_active_tau_frac': select(active_count / 20.0).mean(),
        'rst_active_tau_count': select(active_count).sum(axis=-1).mean(),
        'rst_gate_mass': (select(gate_mass) * weights).sum(axis=-1).mean(),
        'rst_gate_den': (select(gate_den) * weights).sum(axis=-1).mean(),
        'rst_depth_active': (
            select(jnp.asarray(depth)) * weights).sum(axis=-1).mean(),
        'rst_gate_eff_n': (
            select(jnp.asarray(eff_n)) * weights).sum(axis=-1).mean(),
        'rst_top1_gate_frac': (
            select(jnp.asarray(top1)) * weights).sum(axis=-1).mean(),
        'rst_den_floor_frac': (
            select(jnp.asarray(floor)) * weights).sum(axis=-1).mean(),
        'rst_tau_mean': select(tau).mean(),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            np.asarray(actual[name]), value, rtol=1.0e-6, atol=1.0e-6)
    weighted_tau = (select(tau) * weights).sum(axis=-1).mean()
    assert not np.isclose(float(actual['rst_tau_mean']), weighted_tau)


def test_grouped_diagnostics_m1_matches_single_space_definitions() -> None:
    mesh = _mesh()
    common = dict(
        max_chunk_size=2, admission_den_power=1.2,
        srw_composition_mode='linear_angular')
    single_kernel = v4173.make_sharded_srw_diagnostics_minimal(
        mesh, **common)
    grouped_kernel = v4173.make_sharded_rst_multispace_dense_diagnostics(
        mesh, **common)
    x = jax.random.normal(jax.random.PRNGKey(111), (1, 3, 6))
    query = jax.random.normal(jax.random.PRNGKey(112), (1, 3, 3))
    keys = jax.random.normal(jax.random.PRNGKey(113), (4, 3))
    read = jax.random.normal(jax.random.PRNGKey(114), (4, 3))
    write = jax.random.normal(jax.random.PRNGKey(115), (4, 3))
    raw_tau = jnp.full((1, 3, 1), -0.4, dtype=jnp.float32)
    scalar = (jnp.float32(0.07), jnp.float32(0.07),
              jnp.float32(2.0), jnp.float32(4.0), jnp.float32(0.0))
    single = single_kernel(
        x, query, keys, raw_tau, read, write, *scalar)
    grouped = grouped_kernel(
        query.reshape((1, 3, 3)), keys[None, ...],
        raw_tau.reshape((1, 3, 1)),
        jnp.ones((1, 3), dtype=jnp.bool_),
        read[None, ...], write[None, ...], *scalar)
    np.testing.assert_allclose(
        np.asarray(grouped[0][0]).reshape((1, 3, 3)),
        np.asarray(single[0]), rtol=1.0e-6, atol=1.0e-6)
    aggregate = v4173._rst_multispace_diagnostic_aggregates(
        jnp.zeros((1, 3, 1), dtype=jnp.int32),
        jnp.ones((1, 3, 1), dtype=jnp.float32),
        *grouped[1:7], grouped[7], n_rst_per_space=4,
        composition_floor_mass=getattr(
            grouped_kernel, '_v4173_composition_floor_mass'),
        composition_mode='linear_angular')
    metric_indices = {
        'rst_active_tau_frac': 1,
        'rst_active_tau_count': 2,
        'rst_gate_mass': 3,
        'rst_gate_den': 4,
        'rst_depth_active': 5,
        'rst_gate_eff_n': 6,
        'rst_top1_gate_frac': 7,
        'rst_den_floor_frac': 8,
        'rst_tau_mean': 9,
    }
    for name, index in metric_indices.items():
        np.testing.assert_allclose(
            np.asarray(aggregate[name]), np.asarray(single[index]),
            rtol=1.0e-6, atol=1.0e-6)


def test_no_dead_sparse_path_and_selector_uses_router_control_group() -> None:
    source = Path(v4173.__file__).read_text(encoding='utf-8')
    for removed in (
            '_rst_multispace_sparse_forward', 'request_capacity',
            'overflow round', 'packed_x', 'sorted_request_ids',
            'make_sharded_rst_request_minimal', 'rst_request_minimal'):
        assert removed not in source
    assert train_jax._is_v4173_space_selector_router_path(
        'router/rst_space_query/kernel')
    assert train_jax._is_v4173_space_selector_router_path(
        'router/rst_space_keys')
    assert not train_jax._is_v4173_space_selector_router_path(
        'neuron_pool/rst_op_key')


def test_multispace_main_loss_gradients_and_analysis_ids() -> None:
    model, variables, tokens = _init()
    sharded = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(), _mesh())

    def loss_fn(params):
        return model.apply(
            {'params': params}, tokens, labels=tokens,
            deterministic=True,
            rngs={'dropout': jax.random.PRNGKey(7)},
            sharded_fns=sharded, minimal_train=True,
            analysis=False, compute_accuracy=False)['loss']

    loss, grads = jax.value_and_grad(loss_fn)(variables['params'])
    assert np.isfinite(float(loss))
    for name in ('rst_space_query', 'rst_space_keys', 'proj_rst',
                 'raw_tau_rst', 'up_rst'):
        leaves = jax.tree.leaves(grads['router'][name])
        assert all(np.all(np.isfinite(np.asarray(x))) for x in leaves), name
        assert sum(float(jnp.linalg.norm(x)) for x in leaves) > 0.0
    for name in ('rst_read', 'rst_write'):
        gradient = grads['neuron_pool'][name]
        assert np.all(np.isfinite(np.asarray(gradient)))
        assert float(jnp.linalg.norm(gradient)) > 0.0

    model_cfg = {
        **_kwargs(),
        'model_version': v4173.MODEL_VERSION,
        'operator_key_mode': v4173.OPERATOR_KEY_MODE,
        'srw_composition_mode': 'linear_angular',
    }
    logits, info = v4173.analysis_forward(
        variables['params'], model_cfg, tokens)
    assert logits.shape == (1, 3, 16)
    local = info['local_operator_id']
    space = info['selected_space_id']
    np.testing.assert_array_equal(
        np.asarray(info['global_rst_operator_id']),
        np.asarray(space * 2 + local))
    assert info['space_local_contribution'].shape == (1, 1, 3, 2, 6)
    assert info['weighted_global_contribution'].shape == (1, 1, 3, 2, 6)
    assert info['space_internal_gate_den'].shape == (1, 8)
    assert np.all(np.isfinite(np.asarray(info['space_internal_gate_den'])))
    for name in (
            'per_space_selection_frac', 'per_space_mean_selected_weight',
            'per_space_active_frac', 'per_space_active_count',
            'per_space_tau_mean', 'per_space_gate_mass',
            'per_space_gate_den', 'per_space_output_norm',
            'per_space_weighted_contribution_norm'):
        assert info[name].shape == (1, 8)
        assert np.all(np.isfinite(np.asarray(info[name])))


def test_per_space_tau_calibration_is_independent_and_selector_free(
        monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_cfg = {
        'model': {
            'tau_init_mode': 'quantile_frac',
            'tau_init_target_qk_frac': 0.1,
            'tau_init_target_v_frac': 0.1,
            'tau_init_target_rst_frac': 0.0,
        },
        'training': {},
    }
    with pytest.raises(ValueError, match='must be in'):
        train_jax._v4164_tau_init_config(invalid_cfg)

    n_spaces = 4
    base = np.linspace(-0.9, 0.6, 1000, dtype=np.float32)
    rst_scores = np.stack(
        [base + np.float32(0.08 * index)
         for index in range(n_spaces)])
    q_scores = np.linspace(-0.8, 0.8, 400, dtype=np.float32)
    k_scores = np.linspace(-0.7, 0.9, 400, dtype=np.float32)
    v_scores = np.linspace(-0.6, 0.7, 400, dtype=np.float32)
    scores = {
        'q': q_scores,
        'k': k_scores,
        'qk': np.concatenate((q_scores, k_scores)),
        'v': v_scores,
        'rst': rst_scores,
    }

    def meta(pool_size: int) -> dict:
        return {
            'pages_enabled': False,
            'candidate_valid_count': float(pool_size),
            'candidate_count': float(pool_size),
            'full_pool_size': float(pool_size),
            'candidate_frac': 1.0,
            'sample_count': 1000,
        }

    page_stats = {
        'q': meta(400), 'k': meta(400), 'qk': meta(400),
        'v': meta(400), 'rst': meta(1000),
    }
    sampled = {
        'q': q_scores, 'k': k_scores, 'v': v_scores,
        'rst': rst_scores, 'tokens': np.int32(1000),
    }
    monkeypatch.setattr(
        train_jax, '_sample_srw_selection_scores',
        lambda *args, **kwargs: (scores, sampled, page_stats))
    tau_cfg = {
        'targets': {'qk': 0.1, 'v': 0.1, 'rst': 0.1},
        'tau_min': -0.95,
        'tau_max': 0.95,
        'calibration_tokens': 1000,
    }
    summary = train_jax._compute_srw_quantile_tau_init(
        {}, jnp.zeros((1, 1), dtype=jnp.int32),
        {'model': {'model_version': train_jax.V4173_MODEL_VERSION}},
        tau_cfg)
    tau = np.asarray(summary['tau_init_quantile_tau']['rst'])
    assert tau.shape == (n_spaces,)
    assert np.unique(tau).size == n_spaces
    measured = np.mean(rst_scores > tau[:, None], axis=1)
    np.testing.assert_allclose(measured, 0.1, rtol=0.0, atol=0.002)

    _, variables, tokens = _init(spaces=n_spaces, top_k=2)
    calibrated = train_jax._set_srw_quantile_tau_biases(
        variables['params'], summary, train_jax.V4173_MODEL_VERSION)
    raw_bias = calibrated['router']['raw_tau_rst']['bias']
    assert raw_bias.shape == (n_spaces, 1)
    np.testing.assert_allclose(
        np.asarray(v4173._tau_from_param(raw_bias[:, 0])), tau,
        rtol=1.0e-6, atol=1.0e-6)

    cfg = _trainer_cfg(spaces=n_spaces, top_k=2)
    monkeypatch.undo()
    real_summary = train_jax._compute_srw_quantile_tau_init(
        variables['params'], tokens, cfg,
        {**tau_cfg, 'calibration_tokens': 3})
    assert np.asarray(
        real_summary['tau_init_quantile_tau']['rst']).shape == (n_spaces,)
    _, score_impl, score_params, score_kwargs = (
        train_jax._srw_selection_score_setup(
            variables['params'], cfg, max_tokens=3))
    assert 'rst_space_query' not in score_params['router']
    assert 'rst_space_keys' not in score_params['router']
    sampled_scores = score_impl(score_params, tokens, **score_kwargs)
    assert sampled_scores['rst'].shape == (n_spaces, 3, 4)

    _, single_variables, single_tokens = _init(spaces=1, top_k=1)
    single_cfg = _trainer_cfg(spaces=1, top_k=1)
    _, _, single_score_params, single_score_kwargs = (
        train_jax._srw_selection_score_setup(
            single_variables['params'], single_cfg, max_tokens=3))
    current = v4173._tau_init_calibration_scores(
        single_score_params, single_tokens, **single_score_kwargs)
    legacy = v4172._tau_init_calibration_scores(
        single_score_params, single_tokens, **single_score_kwargs)
    np.testing.assert_array_equal(
        np.asarray(current['rst']), np.asarray(legacy['rst']))


def test_multispace_update_checkpoint_resume_and_forward(tmp_path: Path) -> None:
    model, variables, tokens = _init()
    sharded = train_jax.build_canonical_sharded_fns(
        _trainer_cfg(), _mesh())
    optimizer = optax.adam(1.0e-3)
    opt_state = optimizer.init(variables['params'])

    def loss_fn(params):
        return model.apply(
            {'params': params}, tokens, labels=tokens,
            deterministic=True,
            rngs={'dropout': jax.random.PRNGKey(17)},
            sharded_fns=sharded, minimal_train=True,
            analysis=False, compute_accuracy=False)['loss']

    loss, grads = jax.value_and_grad(loss_fn)(variables['params'])
    updates, updated_opt_state = optimizer.update(
        grads, opt_state, variables['params'])
    updated_params = optax.apply_updates(variables['params'], updates)
    updated_loss = loss_fn(updated_params)
    assert np.isfinite(float(loss))
    assert np.isfinite(float(updated_loss))

    full_config = _trainer_cfg()
    training_config = full_config['training']
    manager = train_jax._create_orbax_checkpoint_manager(
        str(tmp_path / 'checkpoints'), checkpoint_interval=1,
        keep_last=1)
    try:
        saved = train_jax.save_orbax_checkpoint(
            manager, updated_params, updated_opt_state,
            jax.random.PRNGKey(18), epoch=0, global_step=1,
            step_in_epoch=1, steps_per_epoch=2,
            best_val_loss=float(updated_loss),
            model_config=full_config['model'],
            training_config=training_config,
            full_config=full_config, raw_config=full_config,
            config_path='test.yaml', run_id='opspace-smoke',
            checkpoint_kind='latest', train_loss=float(updated_loss),
            wait=True)
        assert saved
        target = train_jax._build_orbax_state(
            updated_params, updated_opt_state, jax.random.PRNGKey(18),
            epoch=0, global_step=1, step_in_epoch=1,
            steps_per_epoch=2, best_val_loss=float(updated_loss),
            training_config=training_config, full_config=full_config,
            model_config=full_config['model'])
        restored, metadata = train_jax._restore_orbax_state(
            manager, 1, target)
    finally:
        manager.close()

    assert metadata['full_config']['model']['n_operation_spaces'] == 8
    assert metadata['full_config']['model']['operation_space_top_k'] == 2
    for expected, actual in zip(
            jax.tree.leaves(updated_params),
            jax.tree.leaves(restored['params'])):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    resumed_loss = loss_fn(restored['params'])
    np.testing.assert_array_equal(
        np.asarray(updated_loss), np.asarray(resumed_loss))


def test_multispace_selected_space_trajectory_uses_global_ids() -> None:
    _, variables, _ = _init()
    params = variables['params']
    pool = v4173._pool_params_with_operator_keys(params['neuron_pool'])
    trajectory_cfg = _trainer_cfg()
    trajectory_cfg['training']['analysis_trajectory_capture_width'] = 2
    sharded = train_jax.build_canonical_sharded_fns(
        trajectory_cfg, _mesh(), kernel_profile='trajectory')
    result = v4173._rst_forward_analysis_minimal(
        jax.random.normal(jax.random.PRNGKey(19), (1, 3, 6)),
        pool, params['router'], jax.random.PRNGKey(20),
        0.0, 0.0, True, sharded, d_model=6, n_layers=1,
        n_operation_spaces=8, operation_space_top_k=2,
        analysis_trajectory_enabled=True,
        analysis_trajectory_positions=jnp.asarray([[1]], dtype=jnp.int32),
        analysis_trajectory_position_valid=jnp.asarray([[True]]),
        analysis_trajectory_ids_rst=jnp.asarray(
            [[[0, 3]]], dtype=jnp.int32),
        analysis_trajectory_valid_rst=jnp.asarray(
            [[[True, True]]]),
        analysis_trajectory_replay_enabled=True)
    trace, scale = result[-1]
    assert len(trace) == len(v4173.TRAJECTORY_TRACE_FIELDS)
    assert trace[0].shape == (1, 1, 6)
    assert trace[1].shape == (1, 1, 6)
    assert trace[10].shape == (1, 1, 2)
    assert np.all((np.asarray(trace[10]) < 16)
                  | (np.asarray(trace[10]) == -1))
    assert float(scale) == 1.0
    assert np.all(np.isfinite(np.asarray(result[0])))


def test_40m_parameter_match_symbolic_and_initialized_shape_tree() -> None:
    baseline_path = Path('configs/train_config_v4173_40M_c4_5B.yaml')
    multi_path = Path(
        'configs/train_config_v4173_40M_c4_5B_'
        'opspace8_top2_parammatch.yaml')
    baseline = yaml.safe_load(baseline_path.read_text(encoding='utf-8'))
    multi = yaml.safe_load(multi_path.read_text(encoding='utf-8'))
    report = v4173.search_parameter_matched_n_rst(
        baseline['model'], multi['model'])
    assert report == {
        'baseline_params': 40_343_044,
        'multi_space_params': 40_342_155,
        'absolute_difference': 889,
        'relative_difference': pytest.approx(889 / 40_343_044),
        'baseline_n_rst': 62_094,
        'n_rst': 59_192,
        'n_rst_per_space': 7_399,
        'n_operation_spaces': 8,
        'operation_space_top_k': 2,
    }
    assert report['relative_difference'] < 0.0005
    assert multi['model']['tau_init_target_rst_frac'] == 0.04

    def initialized_count(m: dict, key: int) -> int:
        model = v4173.DAWN_SRW_V4173(
            vocab_size=m['vocab_size'], d_model=m['d_model'],
            d_route=m['d_route'], n_layers=m['n_layers'],
            n_heads=m['n_heads'], max_seq_len=m['max_seq_len'],
            dropout_rate=m['dropout'], router_dropout=m['router_dropout'],
            gradient_checkpointing=m['gradient_checkpointing'],
            n_qk=m['n_qk'], n_v=m['n_v'], n_rst=m['n_rst'],
            n_operation_spaces=m.get('n_operation_spaces', 1),
            operation_space_top_k=m.get('operation_space_top_k', 1),
            tau_init_attn_qk=0.0, tau_init_attn_v=0.0,
            tau_init_rst=0.0,
            operator_key_mode=m['operator_key_mode'],
            admission_den_power=m['admission_den_power'],
            admission_den_power_qk=m['admission_den_power_qk'],
            admission_den_power_v=m['admission_den_power_v'],
            admission_den_power_rst=m['admission_den_power_rst'],
            srw_composition_mode=m['srw_composition_mode'])
        abstract = jax.eval_shape(
            model.init,
            {'params': jax.random.PRNGKey(key),
             'dropout': jax.random.PRNGKey(key + 1)},
            jnp.ones((1, 1), dtype=jnp.int32),
            deterministic=True)['params']
        return sum(
            int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(abstract))

    assert initialized_count(baseline['model'], 8) == report['baseline_params']
    assert initialized_count(multi['model'], 10) == report['multi_space_params']
