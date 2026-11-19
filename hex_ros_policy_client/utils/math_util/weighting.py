################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Weighting the candidates of one instant into the target of that instant.

Two ways to weight live here: by recency over the whole queue and one EMA step
between two neighbouring targets. Not weighting is not a weighting, so it has
no entry here: without a strategy the caller dispatches the candidate nearest
the queried instant as it is.

Every candidate and every result is one flat target vector, so no function has
to know how many joints or grippers a device has. The queue is ordered oldest
first, therefore its last entry is the newest candidate.

Which one to use is decided by the caller with an ``if`` on the strategy name;
this module only holds the two weightings.
"""

from __future__ import annotations

from typing import Union

import numpy as np

# Defaults of the reference implementation, so a task calibrated on the old
# package behaves the same here.
DEFAULT_DECAY = 0.0
DEFAULT_WEIGHT = 0.2


def weight_recency(
        samples: Union[np.ndarray, list[list[float]]],
        weight: float = DEFAULT_DECAY) -> np.ndarray:
    """Weight the whole queue by recency on the index.

    The candidate at index ``i``, counted from the oldest, gets the weight
    ``exp(weight * i)`` before normalisation: a positive weight favours the new
    candidates, a negative one favours the old ones and zero weights them
    equally. Every position of the window therefore gets its own weight, which
    is what tells this apart from :func:`weight_ema`, where one coefficient is
    applied again and again. The largest exponent is subtracted first, so the
    largest weight is always one and a large weight cannot overflow.

    Args:
        samples (Union[np.ndarray, list[list[float]]]): Candidates as a nested
            array of shape ``(M, D)``: ``M`` candidates already ordered oldest
            first, ``D`` values each. A nested list and an array are both
            accepted.
        weight (float): Decay per index step; positive favours new, negative
            favours old.

    Returns:
        np.ndarray: Merged target of shape ``(D,)``.

    Raises:
        ValueError: If the queue is not a nested array of shape ``(M, D)``,
            holds a value that is not finite, or the weight is not finite.
    """
    queue = np.asarray(samples, dtype=np.float64)
    if queue.ndim != 2 or 0 in queue.shape:
        raise ValueError(
            f"samples must be a nested array of shape (M, D), "
            f"got {queue.shape}")
    if not np.isfinite(queue).all():
        raise ValueError("samples must be finite")
    if not np.isfinite(weight):
        raise ValueError(f"weight must be finite, got {weight!r}")
    exponents = weight * np.arange(queue.shape[0], dtype=np.float64)
    weights = np.exp(exponents - exponents.max())
    return weights @ queue / weights.sum()


def weight_ema(
        previous: Union[np.ndarray, list[float]],
        current: Union[np.ndarray, list[float]],
        weight: float = DEFAULT_WEIGHT) -> np.ndarray:
    """Weight two candidates as one EMA step, as the reference does.

    The older target keeps ``1 - weight`` and the newer one takes ``weight``,
    the same coefficient every time it is applied. Merging a whole queue means
    folding this function from the oldest entry, so the newest entry ends up
    with ``weight`` and the oldest with ``(1 - weight) ** (n - 1)``.

    Args:
        previous (Union[np.ndarray, list[float]]): Older target, one
            dimensional.
        current (Union[np.ndarray, list[float]]): Newer target, one
            dimensional.
        weight (float): Weight of the newer target, in ``[0, 1]``.

    Returns:
        np.ndarray: Merged target of shape ``(D,)``.

    Raises:
        ValueError: If the weight is out of range, a target is not one
            dimensional, the two shapes differ, or a value is not finite.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be within [0, 1], got {weight!r}")
    older = np.asarray(previous, dtype=np.float64)
    newer = np.asarray(current, dtype=np.float64)
    if older.ndim != 1 or older.shape != newer.shape:
        raise ValueError(
            f"targets must be one dimensional and of equal shape, got "
            f"{older.shape} and {newer.shape}")
    if not np.isfinite(older).all() or not np.isfinite(newer).all():
        raise ValueError("targets must be finite")
    return (1.0 - weight) * older + weight * newer
