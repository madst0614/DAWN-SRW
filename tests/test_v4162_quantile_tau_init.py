import copy

import pytest

try:
    import jax
    import jax.numpy as jnp
    import flax  # noqa: F401
    import optax  # noqa: F401
    _HAS_JAX_DEPS = True
except ModuleNotFoundError:
    _HAS_JAX_DEPS = False

pytestmark = pytest.mark.skipif(
    not _HAS_JAX_DEPS, reason="JAX/Flax/Optax are not installed")

if _HAS_JAX_DEPS:
    from models.dawn_srw_v4162 import (
        _raw_tau_init_from_cosine_tau,
        _tau_init_calibration_scores,
    )
    from scripts.train_jax import (
        _dawn_srw_kwargs,
        _set_v4162_quantile_tau_biases,
        _v4162_tau_init_config,
    )


def _explicit_cfg():
    return {
        "model": {
            "model_version": "spatial-r1-v4.1.6.2",
            "tau_init_attn_qk": 0.0,
            "tau_init_attn_v": 0.14,
            "tau_init_rst": 0.18,
        },
        "training": {},
    }


def test_explicit_mode_remains_default():
    parsed = _v4162_tau_init_config(_explicit_cfg())

    assert parsed == {
        "mode": "explicit",
        "explicit": {"qk": 0.0, "v": 0.14, "rst": 0.18},
    }


def test_quantile_mode_requires_valid_target_fractions():
    cfg = _explicit_cfg()
    cfg["model"] = {
        "model_version": "spatial-r1-v4.1.6.2",
        "tau_init_mode": "quantile_frac",
        "tau_init_target_qk_frac": 0.08,
        "tau_init_target_v_frac": 0.06,
        "tau_init_target_rst_frac": 0.05,
        "tau_init_min": -0.2,
        "tau_init_max": 0.8,
    }
    parsed = _v4162_tau_init_config(cfg)
    assert parsed["mode"] == "quantile_frac"
    assert parsed["targets"] == {"qk": 0.08, "v": 0.06, "rst": 0.05}

    bad = copy.deepcopy(cfg)
    bad["model"]["tau_init_target_v_frac"] = 1.0
    with pytest.raises(ValueError, match="tau_init_target_v_frac"):
        _v4162_tau_init_config(bad)


def test_quantile_mode_passes_safe_constructor_placeholders():
    cfg = {
        "model": {
            "model_version": "spatial-r1-v4.1.6.2",
            "d_route": 8,
            "d_select": 4,
            "tau_init_mode": "quantile_frac",
            "tau_init_target_qk_frac": 0.08,
            "tau_init_target_v_frac": 0.06,
            "tau_init_target_rst_frac": 0.05,
        },
        "training": {},
    }

    kwargs = _dawn_srw_kwargs(cfg)

    assert kwargs["tau_init_attn_qk"] == 0.0
    assert kwargs["tau_init_attn_v"] == 0.0
    assert kwargs["tau_init_rst"] == 0.0


def test_quantile_tau_overwrites_only_raw_tau_biases():
    params = {
        "router": {
            "raw_tau_attn": {
                "kernel": jnp.zeros((2, 3), dtype=jnp.float32),
                "bias": jnp.zeros((3,), dtype=jnp.float32),
            },
            "raw_tau_rst": {
                "kernel": jnp.zeros((2, 1), dtype=jnp.float32),
                "bias": jnp.zeros((1,), dtype=jnp.float32),
            },
        },
        "untouched": jnp.array([7.0], dtype=jnp.float32),
    }
    summary = {
        "tau_init_quantile_tau": {"qk": 0.1, "v": 0.2, "rst": 0.3},
    }

    updated = _set_v4162_quantile_tau_biases(params, summary)

    assert jnp.allclose(
        updated["router"]["raw_tau_attn"]["bias"],
        jnp.array([
            _raw_tau_init_from_cosine_tau(0.1),
            _raw_tau_init_from_cosine_tau(0.1),
            _raw_tau_init_from_cosine_tau(0.2),
        ]))
    assert jnp.allclose(
        updated["router"]["raw_tau_rst"]["bias"],
        _raw_tau_init_from_cosine_tau(0.3))
    assert jnp.array_equal(updated["untouched"], params["untouched"])


def test_calibration_scores_sample_tokens_and_keep_full_candidate_pools():
    key = jax.random.PRNGKey(0)

    def normal(shape):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape)

    d_model = 4
    d_route = 4
    params = {
        "token_emb": {"embedding": normal((16, d_model))},
        "pos_emb": {"embedding": normal((8, d_model))},
        "block_0": {
            "norm1": {
                "scale": jnp.ones((d_model,)),
                "bias": jnp.zeros((d_model,)),
            },
            "norm2": {
                "scale": jnp.ones((d_model,)),
                "bias": jnp.zeros((d_model,)),
            },
        },
        "router": {
            "proj_attn": {
                "kernel": normal((d_model, d_route * 3)),
                "bias": jnp.zeros((d_route * 3,)),
            },
            "proj_rst": {
                "kernel": normal((d_model, d_route)),
                "bias": jnp.zeros((d_route,)),
            },
        },
        "neuron_pool": {
            "attn_qk_emb": normal((5, d_route)),
            "attn_v_emb": normal((6, d_route)),
            "rst_emb": normal((7, d_route)),
        },
    }
    input_ids = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)

    scores = _tau_init_calibration_scores(
        params, input_ids, d_select=2, max_tokens=3)

    assert scores["q"].shape == (3, 5)
    assert scores["k"].shape == (3, 5)
    assert scores["v"].shape == (3, 6)
    assert scores["rst"].shape == (3, 7)
