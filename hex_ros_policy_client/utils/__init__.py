"""Helpers shared by the layers above: logging, recording and arithmetic.

The sub packages keep the code apart, one directory per kind of helper, but a
caller does not have to know that split: everything meant to be used from
outside is re-exported here, so ``from <package>.utils import drop`` is enough
and a helper can move between the sub packages without touching a caller.

Names that start with an underscore stay reachable only through the module that
defines them; they are internals, not part of this facade.
"""

from __future__ import annotations

from .common_util.csv_recorder import (
    KIND_CHUNK,
    KIND_STATE,
    KIND_TARGET,
    KIND_WEIGHT,
    RECORDER_KINDS,
    STAGE_LIMITED,
    STAGE_RAW,
    STAGES,
    CsvRecorder,
)
from .common_util.logkit import drop, suppressed
from .math_util.interpolation import (
    EPSILON,
    INTERPOLATORS,
    get_interpolator,
    limit_to_step,
    linear_interpolate,
    pose_interpolate,
    slerp_interpolate,
)
from .math_util.weighting import (
    DEFAULT_DECAY,
    DEFAULT_WEIGHT,
    weight_ema,
    weight_recency,
)

__all__ = [
    "CsvRecorder",
    "DEFAULT_DECAY",
    "DEFAULT_WEIGHT",
    "EPSILON",
    "INTERPOLATORS",
    "KIND_CHUNK",
    "KIND_STATE",
    "KIND_TARGET",
    "KIND_WEIGHT",
    "RECORDER_KINDS",
    "STAGES",
    "STAGE_LIMITED",
    "STAGE_RAW",
    "drop",
    "get_interpolator",
    "limit_to_step",
    "linear_interpolate",
    "pose_interpolate",
    "slerp_interpolate",
    "suppressed",
    "weight_ema",
    "weight_recency",
]
