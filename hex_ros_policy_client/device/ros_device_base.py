################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-24
################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Callable


class RosDeviceBase(ABC):
    """Base of every device reached through ROS topics.

    A subclass only declares what it subscribes to and what it publishes. The
    endpoints are created here from the device table, with the two callables
    the caller injects, so the subclass never sees the node nor the interface
    that owns it.

    Attributes:
        SUBSCRIBER_SPECS (dict): Entry key to the message type and the name of
            the method that receives it.
        PUBLISHER_SPECS (dict): Entry key to the message type.
        REQUIRED_KEYS (tuple): Keys the entry of the device table has to
            provide.
    """

    SUBSCRIBER_SPECS: dict[str, tuple] = {}
    PUBLISHER_SPECS: dict[str, tuple] = {}
    REQUIRED_KEYS: tuple[str, ...] = ("ros_cfg",)

    _SUBSCRIBER_TABLE = "subscribers"
    _PUBLISHER_TABLE = "publishers"

    def __init__(self, name: str, 
                 params: dict,
                 create_subscription: Callable,
                 create_publisher: Callable,
                 logger: Any,
                 now_fn: Callable[[], int] | None = None) -> None:
        """Create the endpoints the device table asks for.

        Args:
            name (str): Device name taken from the device table.
            params (dict): Entry of the device table.
            create_subscription (Callable): ``(msg_type, topic, callback)``
                returning the handle of the new subscription.
            create_publisher (Callable): ``(msg_type, topic)`` returning the
                handle of the new publisher.
            logger (Any): Logger object exposing standard logging methods.
            now_fn (Callable[[], int] | None): Function returning the current
                time in nanoseconds; defaults to ``time.time_ns``.

        Raises:
            ValueError: If a required key is missing, if ``ros_cfg`` is not a
                mapping, if the namespace is not a string, if a table
                declares a key the class does not know, or if the device ends
                up with no endpoint at all.
        """
        ros_cfg, namespace = self.__read_ros_cfg(name=name, params=params)
        
        self._name = name
        self._logger = logger
        self._now_fn: Callable[[], int] = (
            now_fn if now_fn is not None else time.time_ns
        )
        self.__create_subscription = create_subscription
        self.__create_publisher = create_publisher
        self.__subscribers: dict[str, Any] = {}
        self.__publishers: dict[str, Any] = {}
        self.__create_subscribers(ros_cfg=ros_cfg, namespace=namespace)
        self.__create_publishers(ros_cfg=ros_cfg, namespace=namespace)
        if not self.__subscribers and not self.__publishers:
            raise ValueError(
                f"device {name!r} declares neither a subscriber nor a "
                "publisher")


    # @abstractmethod
    # def fetch_states(self, states: Any) -> None:
    #     """Store the newest sample of this device.

    #     Args:
    #         states: Message delivered by the subscription.
    #     """

    @abstractmethod
    def get_obs(self) -> Any:
        """Return the newest sample.

        Returns:
            Any: Latest observation, or ``None`` before the first message.
        """

    
    @property
    def name(self) -> str:
        """Name of this device as declared in the device table.

        Returns:
            str: Device name.
        """
        return self._name

    @property
    def subscribers(self) -> dict:
        """Subscriptions of this device, keyed by table key.

        Returns:
            dict: Entry key to subscription handle.
        """
        return self.__subscribers

    @property
    def publishers(self) -> dict:
        """Publishers of this device, keyed by table key.

        Returns:
            dict: Entry key to publisher handle.
        """
        return self.__publishers


    def __create_subscribers(self, ros_cfg: dict, namespace: str) -> None:
        entries = self._resolve_entries(
            ros_cfg=ros_cfg, table=self._SUBSCRIBER_TABLE,
            specs=self.SUBSCRIBER_SPECS,
            name=self._name
        )
        
        for key, (topic, msg_type, handler) in entries.items():
            self.__subscribers[key] = self.__create_subscription(
                msg_type=msg_type,
                topic=self._resolve_topic(namespace=namespace, topic=topic),
                callback=getattr(self, handler),
            )

    def __create_publishers(self, ros_cfg: dict, namespace: str) -> None:
        entries = self._resolve_entries(
            ros_cfg=ros_cfg, 
            table=self._PUBLISHER_TABLE,
            specs=self.PUBLISHER_SPECS,
            name=self._name
        )
        
        for key, (topic, msg_type) in entries.items():
            self.__publishers[key] = self.__create_publisher(
                msg_type=msg_type,
                topic=self._resolve_topic(namespace=namespace, topic=topic),
            )

    ##################################
    ### cfg_tools
    ##################################

    def __read_ros_cfg(self, name: str, params: dict) -> tuple[dict, str]:
        """Check the entry of the device table and read its ROS part.

        Args:
            name (str): Device name, used in the error messages.
            params (dict): Entry of the device table.

        Returns:
            tuple: ``ros_cfg`` mapping and the namespace, empty when the
            device declares none.

        Raises:
            ValueError: If a required key is missing, if ``ros_cfg`` is not
                a mapping, or if the namespace is not a string.
        """
        missing = [key for key in self.REQUIRED_KEYS if key not in params]
        if missing:
            raise ValueError(
                f"device {name!r} is missing required keys {missing}")
        ros_cfg = params["ros_cfg"]
        if not isinstance(ros_cfg, dict):
            raise ValueError(
                f"ros_cfg of device {name!r} must be a mapping, got "
                f"{type(ros_cfg).__name__}")
        namespace = ros_cfg.get("namespace")
        if namespace is None:
            return ros_cfg, ""
        if not isinstance(namespace, str):
            raise ValueError(
                f"namespace of device {name!r} must be a string, got "
                f"{type(namespace).__name__}")
        return ros_cfg, namespace


    def _resolve_topic(self, namespace: str, topic: str) -> str:
        """Join a namespace and a topic into the name ROS 2 will resolve.

        Args:
            namespace (str): Namespace of the device; may nest as ``ns1/ns2``,
                may be absolute as ``/ns1/ns2``, and may be empty.
            topic (str): Topic name as written in the device table; an absolute
                topic is used exactly as it is.

        Returns:
            str: Full topic name.

        Raises:
            ValueError: If the topic is empty, if a segment between two slashes
                is empty, or if an absolute topic is given together with a
                namespace, because the namespace would then be ignored.
        """
        if not isinstance(topic, str) or not topic:
            raise ValueError(
                f"topic must be a non empty string, got {topic!r}")
        if topic.endswith("/"):
            raise ValueError(f"topic {topic!r} has an empty segment")
        if topic.startswith("/"):
            if namespace:
                raise ValueError(
                    f"absolute topic {topic!r} cannot be combined with the "
                    f"namespace {namespace!r}")
            joined = topic
        elif not namespace:
            joined = topic
        else:
            root = "/" if namespace.startswith("/") else ""
            joined = f"{root}{namespace.strip('/')}/{topic}"
        if "//" in joined:
            raise ValueError(f"topic {joined!r} has an empty segment")
        return joined


    def _resolve_entries(
            self, 
            ros_cfg: dict, 
            table: str, 
            specs: dict,
            name: str
        ) -> dict[str, tuple]:
        
        """Pair the entries a device declares with the specs of its class.

        Args:
            ros_cfg (dict): Contents of ``ros_cfg`` of one device.
            table (str): Table name, ``subscribers`` or ``publishers``.
            specs (dict): Spec table declared by the device class.
            name (str): Device name, used in the error messages.

        Returns:
            dict: Entry key to ``(topic, *spec)``; an entry whose topic is
            ``null`` is left out, which is how an absent stream is declared. A
            spec that is not a tuple is taken as a one element one, so a
            publisher spec can be the message type itself.

        Raises:
            ValueError: If the table is not a mapping, or declares a key the
                class does not know.
        """
        entries = ros_cfg.get(table)
        if entries is None:
            return {}
        if not isinstance(entries, dict):
            raise ValueError(
                f"{table} of device {name!r} must be a mapping, got "
                f"{type(entries).__name__}")
        unknown = sorted(key for key in entries if key not in specs)
        if unknown:
            raise ValueError(
                f"device {name!r} declares unknown {table} keys {unknown}, "
                f"expected a subset of {sorted(specs)}")
        resolved = {}
        for key, topic in entries.items():
            if topic is None:
                continue
            spec = specs[key]
            if not isinstance(spec, tuple):
                spec = (spec,)
            resolved[key] = (topic, *spec)
        return resolved
