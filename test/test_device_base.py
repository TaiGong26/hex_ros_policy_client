################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Tests for the device base contract."""

from __future__ import annotations

import pytest

from hex_ros_policy_client.device.device_base import DeviceBase


class Echo(DeviceBase):
    """Minimal device that satisfies the contract."""

    def __init__(self) -> None:
        """Create a device without any sample."""
        self.sample = None

    def on_states(self, states) -> None:
        """Store a sample.

        Args:
            states: Anything the caller hands over.
        """
        self.sample = states

    def get_obs(self):
        """Return the stored sample.

        Returns:
            Any: Last stored sample.
        """
        return self.sample


def test_base_is_abstract() -> None:
    """The base class cannot be instantiated on its own."""
    with pytest.raises(TypeError):
        DeviceBase()


def test_device_has_no_lifecycle() -> None:
    """A device is a topic, so it exposes no start and no stop."""
    assert hasattr(DeviceBase, "start") is False
    assert hasattr(DeviceBase, "stop") is False
    assert hasattr(Echo(), "start") is False
    assert hasattr(Echo(), "stop") is False


def test_public_surface_is_two_methods() -> None:
    """The public surface stays on_states and get_obs."""
    public = {name for name in dir(DeviceBase) if not name.startswith("_")}
    assert public == {"on_states", "get_obs"}


def test_subclass_works() -> None:
    """A subclass stores and returns the newest sample."""
    device = Echo()
    assert device.get_obs() is None
    device.on_states(42)
    assert device.get_obs() == 42
