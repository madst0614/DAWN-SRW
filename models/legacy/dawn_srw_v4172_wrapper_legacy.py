"""DAWN-SRW v4.1.7.2 generalized bilinear RW-key wrapper.

The canonical model, SRW kernels, DirectTau path, and analysis hooks live in
``dawn_srw_v4171``.  This module changes only the version identity and fixes
the static operator-key mode to live generalized coordinate-wise bilinear RW
keys.
"""

from models import dawn_srw_v4171 as _core
from models.dawn_srw_v4171 import *  # noqa: F401,F403


MODEL_VERSION = "spatial-r1-v4.1.7.2"
OPERATOR_KEY_MODE = _core.OPERATOR_KEY_MODE_GENERALIZED_BILINEAR


class DAWN_SRW_V4172(_core.DAWN_SRW_V4171):
    """Thin v4172 identity wrapper around the shared canonical v417x core."""

    __version__ = MODEL_VERSION
    operator_key_mode: str = OPERATOR_KEY_MODE

    def setup(self):
        mode = _core._validate_operator_key_mode(
            self.operator_key_mode, context="DAWN_SRW_V4172 constructor")
        if mode != OPERATOR_KEY_MODE:
            raise ValueError(
                "v4172 requires operator_key_mode="
                f"{OPERATOR_KEY_MODE!r}, got {mode!r}")
        super().setup()


DAWN = DAWN_SRW_V4172


def __getattr__(name):
    """Expose private shared-core hooks used by trainer and analysis code."""
    return getattr(_core, name)
