################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Pytest bootstrap for the package tests.

Makes the package importable when the tests run from any working directory,
so the suite does not depend on an installed colcon overlay.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture()
def wire(monkeypatch):
    """Replace the Zenoh transport of the middleware by recorders.

    The fixture keeps the rest of the middleware intact, so the frame handling
    and the event path stay under test.

    Args:
        monkeypatch: Fixture used to patch the class.

    Returns:
        dict: Recorded frames per link.
    """
    from hex_ros_policy_client.protocol import policy_middleware as middleware

    record = {"obs": [], "inference": []}
    monkeypatch.setattr(middleware.PolicyMiddleware, "start",
                        lambda self: None)
    monkeypatch.setattr(middleware.PolicyMiddleware, "close",
                        lambda self: None)
    monkeypatch.setattr(
        middleware.PolicyMiddleware, "pub_obs",
        lambda self, frame: record["obs"].append(frame) is None)
    monkeypatch.setattr(
        middleware.PolicyMiddleware, "pub_inference_state",
        lambda self, frame:
        record["inference"].append(frame["inference_enabled"]) is None)
    return record
