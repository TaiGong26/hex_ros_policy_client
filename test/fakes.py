################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Test doubles for the ROS node and the devices.

The suite never starts ROS: a fake node records the endpoints the devices
create and lets a test deliver messages by hand.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

FAKE_QOS_DEPTH = 1

DUAL_TABLE = """
cameras:
  fixed:
    ros_cfg:
      namespace: fixed
      subscribers:
        color_topic: image_color_raw
  left_wrist:
    ros_cfg:
      namespace: left_wrist
      subscribers:
        color_topic: image_color_raw
        depth_topic: image_depth_raw
arms:
  left:
    ros_cfg:
      namespace: left
      subscribers:
        state_topic: manip_state
      publishers:
        ctrl_topic: manip_ctrl
    arm_cfg:
      lim_vel: 0.5
      lim_acc: 50.0
    grip_cfg:
      lim_pos: [0.0, 0.75]
  right:
    ros_cfg:
      namespace: right
      subscribers:
        state_topic: manip_state
      publishers:
        ctrl_topic: manip_ctrl
    arm_cfg:
      lim_vel: 0.5
      lim_acc: 50.0
    grip_cfg:
      lim_pos: [0.0, 0.75]
"""

SINGLE_TABLE = """
cameras:
  main_camera:
    ros_cfg:
      namespace: main_camera
      subscribers:
        color_topic: image_color_raw
arms:
  main:
    ros_cfg:
      namespace: main
      subscribers:
        state_topic: manip_state
      publishers:
        ctrl_topic: manip_ctrl
    arm_cfg: {}
    grip_cfg: {}
"""


def write_table(tmp_path: Any, text: str) -> str:
    """Write a device table and return its path.

    Args:
        tmp_path: Pytest temporary directory.
        text (str): YAML contents.

    Returns:
        str: Path of the written table.
    """
    path = tmp_path / "devices.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


class Capture:
    """Logger stand-in that keeps every emitted line.

    Attributes:
        records (list): Pairs of (level, message).
    """

    def __init__(self) -> None:
        """Create an empty capture."""
        self.records: list[tuple[int, str]] = []

    def log(self, level: int, message: str) -> None:
        """Record one line.

        Args:
            level (int): Log level.
            message (str): Rendered message.
        """
        self.records.append((level, message))

    def info(self, message: str, *args: Any) -> None:
        """Record an informational message.

        Args:
            message (str): Message text.
            *args (Any): Format arguments, ignored by the double.
        """
        self.records.append((20, message.format(*args) if args else message))

    def logd(self, message: str) -> None:
        """Record a debug line.

        Args:
            message (str): Rendered message.
        """
        self.records.append((10, message))

    def logi(self, message: str) -> None:
        """Record an informational line.

        Args:
            message (str): Rendered message.
        """
        self.records.append((20, message))

    def logw(self, message: str) -> None:
        """Record a warning line.

        Args:
            message (str): Rendered message.
        """
        self.records.append((30, message))

    def loge(self, message: str) -> None:
        """Record an error line.

        Args:
            message (str): Rendered message.
        """
        self.records.append((40, message))

    def logf(self, message: str) -> None:
        """Record a fatal line.

        Args:
            message (str): Rendered message.
        """
        self.records.append((50, message))

    def debug(self, message: str, *args: Any) -> None:
        """Record a debug line.

        Args:
            message (str): Message text.
            *args (Any): Format arguments, ignored by the double.
        """
        self.records.append((10, message.format(*args) if args else message))

    def warning(self, message: str, *args: Any) -> None:
        """Record a warning.

        Args:
            message (str): Message text.
            *args (Any): Format arguments, ignored by the double.
        """
        self.records.append((30, message.format(*args) if args else message))

    def messages(self) -> list[str]:
        """Return the recorded messages only.

        Returns:
            list: Message texts.
        """
        return [message for _, message in self.records]


class FakePublisher:
    """Publisher stand-in that records what was sent.

    Attributes:
        topic (str): Topic it was created for.
        sent (list): Messages published.
    """

    def __init__(self, topic: str) -> None:
        """Create a publisher for one topic.

        Args:
            topic (str): Topic name.
        """
        self.topic = topic
        self.sent: list[Any] = []

    def publish(self, message: Any) -> None:
        """Record one message.

        Args:
            message: Message published by the device.
        """
        self.sent.append(message)


class FakeSubscription:
    """Subscription stand-in that keeps its callback.

    Attributes:
        topic (str): Topic it was created for.
        callback (Any): Callback registered by the device.
        destroyed (bool): Whether the endpoint was released.
    """

    def __init__(self, topic: str, callback: Any) -> None:
        """Create a subscription for one topic.

        Args:
            topic (str): Topic name.
            callback (Any): Callback registered by the device.
        """
        self.topic = topic
        self.callback = callback
        self.destroyed = False


class FakeInterface:
    """Interface double: an optional node, a captured logger and a rate.

    Attributes:
        node (Any): Node handed to the collaborators.
        logger (Capture): Logger the five log methods write to.
        rate_param (dict): Rate group, as the real interface exposes it.
        device_path (str): Device table path, as the real interface exposes it.
        shutdown_calls (int): Number of shutdown calls.
    """

    def __init__(self, node: Any = None, rate_hz: float = 1000.0,
                 device_path: str = "") -> None:
        """Build the double.

        Args:
            node (Any): Node to expose; a plain fake node when omitted.
            rate_hz (float): Control rate exposed through the rate group.
            device_path (str): Device table path exposed as a value.
        """
        self.node = node if node is not None else FakeNode()
        self.logger = Capture()
        self.rate_param = {"ros": rate_hz}
        self.device_path = device_path
        self.shutdown_calls = 0

    def ok(self) -> bool:
        """Report a running runtime.

        Returns:
            bool: Always ``True``.
        """
        return True

    def shutdown(self) -> None:
        """Count the call."""
        self.shutdown_calls += 1

    def sleep(self) -> None:
        """Return at once."""

    def getlogger(self) -> Capture:
        """Return the captured logger.

        Returns:
            Capture: Logger of this double.
        """
        return self.logger

    def get_rate_param(self) -> dict:
        """Return the rate group.

        Returns:
            dict: Rate group of this double.
        """
        return self.rate_param

    def create_subscription(self, msg_type: Any, topic: str,
                            callback: Any) -> Any:
        """Create a subscription on the node of this double.

        Args:
            msg_type (Any): Message class; unused by the double.
            topic (str): Topic name.
            callback (Any): Callback registered by the device.

        Returns:
            Any: The created subscription.
        """
        return self.node.create_subscription(msg_type, topic, callback,
                                             FAKE_QOS_DEPTH)

    def create_publisher(self, msg_type: Any, topic: str) -> Any:
        """Create a publisher on the node of this double.

        Args:
            msg_type (Any): Message class; unused by the double.
            topic (str): Topic name.

        Returns:
            Any: The created publisher.
        """
        return self.node.create_publisher(msg_type, topic, FAKE_QOS_DEPTH)

    def logd(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record a debug line.

        Args:
            msg (str): Message text or template.
            *args (Any): Format arguments of the template.
            **kwargs (Any): Ignored by the double.
        """
        self.logger.logd(msg.format(*args) if args else msg)

    def logi(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record an informational line.

        Args:
            msg (str): Message text or template.
            *args (Any): Format arguments of the template.
            **kwargs (Any): Ignored by the double.
        """
        self.logger.logi(msg.format(*args) if args else msg)

    def logw(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record a warning line.

        Args:
            msg (str): Message text or template.
            *args (Any): Format arguments of the template.
            **kwargs (Any): Ignored by the double.
        """
        self.logger.logw(msg.format(*args) if args else msg)

    def loge(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record an error line.

        Args:
            msg (str): Message text or template.
            *args (Any): Format arguments of the template.
            **kwargs (Any): Ignored by the double.
        """
        self.logger.loge(msg.format(*args) if args else msg)

    def logf(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record a fatal line.

        Args:
            msg (str): Message text or template.
            *args (Any): Format arguments of the template.
            **kwargs (Any): Ignored by the double.
        """
        self.logger.logf(msg.format(*args) if args else msg)


class FakeNode:
    """Minimal node used instead of a real ROS node.

    Attributes:
        publishers (dict): Topic to fake publisher.
        subscriptions (list): Every created subscription.
        logger (Capture): Logger returned by :meth:`get_logger`.
        destroyed (list): Endpoints destroyed through the node.
    """

    def __init__(self) -> None:
        """Create a node with no endpoints."""
        self.publishers: dict[str, FakePublisher] = {}
        self.subscriptions: list[FakeSubscription] = []
        self.logger = Capture()
        self.destroyed: list[Any] = []

    def create_publisher(self, message_type: Any,
                         topic: str, depth: int) -> FakePublisher:
        """Create and remember a publisher.

        Args:
            message_type: Message class; unused by the double.
            topic (str): Topic name.
            depth (int): Queue depth; unused by the double.

        Returns:
            FakePublisher: The created publisher.
        """
        publisher = FakePublisher(topic)
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, message_type: Any, topic: str,
                            callback: Any, depth: int) -> FakeSubscription:
        """Create and remember a subscription.

        Args:
            message_type: Message class; unused by the double.
            topic (str): Topic name.
            callback (Any): Callback registered by the device.
            depth (int): Queue depth; unused by the double.

        Returns:
            FakeSubscription: The created subscription.
        """
        subscription = FakeSubscription(topic, callback)
        self.subscriptions.append(subscription)
        return subscription

    def get_logger(self) -> Capture:
        """Return the capture logger.

        Returns:
            Capture: Logger of this node.
        """
        return self.logger

    def destroy_subscription(self, subscription: FakeSubscription) -> None:
        """Mark a subscription as released.

        Args:
            subscription (FakeSubscription): Endpoint to release.
        """
        subscription.destroyed = True
        self.destroyed.append(subscription)

    def destroy_publisher(self, publisher: FakePublisher) -> None:
        """Record a released publisher.

        Args:
            publisher (FakePublisher): Endpoint to release.
        """
        self.destroyed.append(publisher)

    def subscription(self, topic: str) -> FakeSubscription:
        """Look up a subscription by topic.

        Args:
            topic (str): Topic name.

        Returns:
            FakeSubscription: The matching subscription.

        Raises:
            KeyError: If no subscription was created for that topic.
        """
        for subscription in self.subscriptions:
            if subscription.topic == topic:
                return subscription
        raise KeyError(topic)


class FakeClock:
    """Clock stand-in that returns a controllable instant.

    Attributes:
        now_ns (int): Current time in nanoseconds.
    """

    def __init__(self, now_ns: int = 0) -> None:
        """Create a clock at the given instant.

        Args:
            now_ns (int): Initial time.
        """
        self.now_ns = now_ns

    def now(self) -> Any:
        """Return the current instant.

        Returns:
            Any: Object with a ``nanoseconds`` attribute.
        """
        return SimpleNamespace(nanoseconds=self.now_ns)


class FakeParameter:
    """Parameter stand-in.

    Attributes:
        value (Any): Value carried by the parameter.
    """

    def __init__(self, value: Any) -> None:
        """Store the value.

        Args:
            value (Any): Parameter value.
        """
        self.value = value


class FakeRosNode(FakeNode):
    """Node double that also carries parameters and a clock.

    Attributes:
        params (dict): Parameter name to value.
        declared (list): Names declared through the node.
        clock (FakeClock): Clock of the node.
        destroyed_node (bool): Whether the node was destroyed.
    """

    def __init__(self, params: dict | None = None,
                 now_ns: int = 0) -> None:
        """Create a node with the given parameters.

        Args:
            params (dict | None): Parameter table.
            now_ns (int): Initial time.
        """
        super().__init__()
        self.params = dict(params or {})
        self.declared: list[str] = []
        self.clock = FakeClock(now_ns)
        self.destroyed_node = False

    def declare_parameter(self, name: str, value: Any = None,
                          descriptor: Any = None) -> None:
        """Record a declaration and keep a default when one is given.

        Args:
            name (str): Parameter name.
            value (Any): Default value, ``None`` to keep the table value.
            descriptor (Any): Parameter descriptor, unused by the double.
        """
        self.declared.append(name)
        if name not in self.params and value is not None:
            self.params[name] = value

    def get_parameter(self, name: str) -> FakeParameter:
        """Return the parameter of one name.

        Args:
            name (str): Parameter name.

        Returns:
            FakeParameter: Parameter carrying the table value or ``None``.
        """
        return FakeParameter(self.params.get(name))

    def get_clock(self) -> FakeClock:
        """Return the clock of the node.

        Returns:
            FakeClock: Clock stand-in.
        """
        return self.clock

    def destroy_node(self) -> None:
        """Mark the node as destroyed."""
        self.destroyed_node = True
