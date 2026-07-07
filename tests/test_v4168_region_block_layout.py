from types import SimpleNamespace

import pytest

try:
    import jax  # noqa: F401
    import flax  # noqa: F401
    _HAS_JAX_DEPS = True
except ModuleNotFoundError:
    _HAS_JAX_DEPS = False

pytestmark = pytest.mark.skipif(
    not _HAS_JAX_DEPS, reason="JAX/Flax are not installed")

if _HAS_JAX_DEPS:
    from models.dawn_srw_v4168 import (
        _opspace_region_block_model_axis_layout,
    )


def _mesh(model_axis_size):
    return SimpleNamespace(shape={'model': model_axis_size})


def test_region_block_layout_splits_regions_over_model_axis():
    layout = _opspace_region_block_model_axis_layout(
        _mesh(8),
        num_regions=32,
        blocks_per_region=2,
        operators_per_block=512)

    assert layout == {
        'num_regions': 32,
        'blocks_per_region': 2,
        'operators_per_block': 512,
        'model_axis_size': 8,
        'local_regions': 4,
        'local_block_count': 8,
        'local_operator_capacity': 4096,
        'global_operator_capacity': 32768,
    }


def test_region_block_layout_requires_even_region_ownership():
    with pytest.raises(ValueError, match="num_regions=10.*divisible.*4"):
        _opspace_region_block_model_axis_layout(
            _mesh(4),
            num_regions=10,
            blocks_per_region=2,
            operators_per_block=128)


def test_region_block_layout_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="operators_per_block >= 1"):
        _opspace_region_block_model_axis_layout(
            _mesh(4),
            num_regions=8,
            blocks_per_region=2,
            operators_per_block=0)
