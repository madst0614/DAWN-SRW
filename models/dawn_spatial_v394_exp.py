"""Compatibility entry point for spatial-r1-v3.9.4.

The implementation lives in `models.legacy`.  This module keeps existing
training configs and registry imports active while the source tree keeps old
model bodies out of the active model directory.
"""

from .legacy.dawn_spatial_v394_exp import *  # noqa: F401,F403
