################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

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
    """Weight a queue of candidates by index and return their merged target.

    Args:
        samples (Union[np.ndarray, list[list[float]]]): Candidates as a nested
            array of shape ``(M, D)``, oldest first.
        weight (float): Exponent per index step, finite; positive favours the
            newest candidate, negative the oldest, zero weights all equally.

    Returns:
        np.ndarray: Merged target of shape ``(D,)``, a convex combination of
        the candidates.

    Raises:
        ValueError: If ``samples`` is not a nested array of shape ``(M, D)``,
            if a sample or ``weight`` is not finite.
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
    """Blend two targets with one exponential moving average step.

    Args:
        previous (Union[np.ndarray, list[float]]): Older target, one
            dimensional.
        current (Union[np.ndarray, list[float]]): Newer target, same shape as
            ``previous``.
        weight (float): Weight of the newer target, within ``[0, 1]``.

    Returns:
        np.ndarray: Merged target of shape ``(D,)``; ``previous`` when
        ``weight`` is 0, ``current`` when ``weight`` is 1.

    Raises:
        ValueError: If ``weight`` is outside ``[0, 1]``, if a target is not one
            dimensional, if the two shapes differ, or if a value is not finite.
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
