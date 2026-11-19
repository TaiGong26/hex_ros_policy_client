################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the sample weight tables used when chunks overlap."""

from __future__ import annotations

import numpy as np
import pytest

from hex_ros_policy_client.process.action_average import (
    AVERAGERS,
    average,
    weights_of,
)


def samples(values: list[float]) -> list[np.ndarray]:
    """Build one sample per value, newest first.

    Args:
        values (list): Sample values.

    Returns:
        list: Samples of shape ``(1,)``.
    """
    return [np.asarray([value], dtype=np.float64) for value in values]


def test_none_keeps_the_newest_sample() -> None:
    """The no averaging strategy uses the newest sample only."""
    assert average("none", samples([3.0, 1.0, 2.0]))[0] == 3.0


def test_plain_weights_every_sample_equally() -> None:
    """The plain strategy is the arithmetic mean."""
    assert average("plain", samples([3.0, 1.0, 2.0]))[0] == pytest.approx(2.0)


def test_exponential_favours_the_newest_sample() -> None:
    """The exponential strategy decays with recency."""
    value = average("exponential", samples([3.0, 1.0, 2.0]), 0.5)[0]
    weight = (3.0 + 1.0 * 0.5 + 2.0 * 0.25) / (1.0 + 0.5 + 0.25)
    assert value == pytest.approx(weight)


def test_exponential_weights_follow_the_decay() -> None:
    """Weights descend by the decay ratio."""
    assert weights_of("exponential", 3, 0.5).tolist() == [1.0, 0.5, 0.25]


def test_none_weights_zero_every_older_sample() -> None:
    """Only the newest sample carries weight, so a merge is a selection."""
    assert weights_of("none", 3, 0.5).tolist() == [1.0, 0.0, 0.0]


def test_every_strategy_merges_a_single_sample() -> None:
    """One sample averages to itself whatever the strategy is."""
    for strategy in AVERAGERS:
        assert average(strategy, samples([4.0]))[0] == pytest.approx(4.0)


def test_unknown_strategy_is_rejected() -> None:
    """A strategy outside the table raises."""
    with pytest.raises(ValueError):
        average("median", samples([1.0]))


def test_empty_samples_are_rejected() -> None:
    """Merging nothing raises instead of returning a silent zero."""
    with pytest.raises(ValueError):
        average("plain", [])
