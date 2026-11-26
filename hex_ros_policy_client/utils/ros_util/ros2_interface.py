#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################

import threading

import rclpy
import rclpy.node
from builtin_interfaces.msg import Time
from hex_ros_msgs.msg import HexRosTeleopKeyboardStateStamped
from hex_util_msg.dataclass.dataclass_teleop import HexDcTeleopKeyboardState

from .interface_base import InterfaceBase

_LETTERS = [chr(c) for c in range(ord('a'), ord('z') + 1)]


class DataInterface(InterfaceBase):

    _DEFAULT_QOS_DEPTH = 10

    def __init__(self, name: str = "unknown"):
        super(DataInterface, self).__init__(name=name)

        ### ros node
        rclpy.init()
        self.__node = rclpy.node.Node(name)
        self.__logger = self.__node.get_logger()

        ### parameters
        self.__node.declare_parameter('rate_ros', 1000.0)
        self.__node.declare_parameter('observation_rate_hz', 30.0)
        self.__node.declare_parameter('stable_duration_s', 2.0)
        self.__node.declare_parameter('arrive_threshold', 0.05)
        self.__node.declare_parameter('init_timeout_s', 5.0)
        self.__node.declare_parameter('max_episode_s', 120.0)
        self.__node.declare_parameter('data_freshness_ms', 40.0)
        self.__node.declare_parameter('observation_topic',
                                      'policy/observation')
        self.__node.declare_parameter('inference_topic',
                                      'policy/inference')
        self.__node.declare_parameter('action_topic', 'policy/action')
        self.__node.declare_parameter('key_online', 's')
        self.__node.declare_parameter('key_start', 'b')
        self.__node.declare_parameter('key_end', 'e')
        self.__node.declare_parameter('key_exit', 'q')
        self.__node.declare_parameter('recorder_dir', "")
        self.__node.declare_parameter('device_path', "")

        self._client_param = {
            "rate_ros":
            self.__node.get_parameter('rate_ros').value,
            "observation_rate_hz":
            self.__node.get_parameter('observation_rate_hz').value,
            "stable_duration_s":
            self.__node.get_parameter('stable_duration_s').value,
            "arrive_threshold":
            self.__node.get_parameter('arrive_threshold').value,
            "init_timeout_s":
            self.__node.get_parameter('init_timeout_s').value,
            "max_episode_s":
            self.__node.get_parameter('max_episode_s').value,
            "data_freshness_ms":
            self.__node.get_parameter('data_freshness_ms').value,
            "observation_topic":
            self.__node.get_parameter('observation_topic').value,
            "inference_topic":
            self.__node.get_parameter('inference_topic').value,
            "action_topic":
            self.__node.get_parameter('action_topic').value,
            "key_online": self.__node.get_parameter('key_online').value,
            "key_start": self.__node.get_parameter('key_start').value,
            "key_end": self.__node.get_parameter('key_end').value,
            "key_exit": self.__node.get_parameter('key_exit').value,
            "recorder_dir":
            self.__node.get_parameter('recorder_dir').value,
            "device_path":
            self.__node.get_parameter('device_path').value,
        }
        self.__rate = self.__node.create_rate(self._client_param["rate_ros"])

        ### subscriber
        self.__keyboard_sub = self.__node.create_subscription(
            HexRosTeleopKeyboardStateStamped,
            'teleop_keyboard_state',
            self.__keyboard_callback,
            self._DEFAULT_QOS_DEPTH,
        )

        ### spin thread
        self.__shutting_down = False
        self.__spin_thread = threading.Thread(target=self.__spin)
        self.__spin_thread.start()


    def __spin(self):
        rclpy.spin(self.__node)

    def ok(self):
        return rclpy.ok()

    def shutdown(self):
        if self.__shutting_down:
            return
        self.__shutting_down = True
        try:
            self.__node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self.__spin_thread.join()

    def sleep(self):
        try:
            self.__rate.sleep()
        except rclpy.exceptions.ROSInterruptException:
            pass

    ####################
    ### clock
    ####################
    def get_nowstamp(self) -> Time:
        """Return the node clock as a message header stamp.

        Returns:
            Time: Current time as ``builtin_interfaces/Time``.
        """
        return self.__node.get_clock().now().to_msg()

    ####################
    ### endpoints
    ####################
    def create_subscription(self, msg_type, topic, callback):
        """Create one subscription.

        Args:
            msg_type: Message type the subscription receives.
            topic (str): Full topic name.
            callback: Function called with every message.

        Returns:
            Any: Handle of the new subscription.
        """
        return self.__node.create_subscription(msg_type, topic, callback,
                                               self._DEFAULT_QOS_DEPTH)

    def create_publisher(self, msg_type, topic):
        """Create one publisher.

        Args:
            msg_type: Message type the publisher sends.
            topic (str): Full topic name.

        Returns:
            Any: Handle of the new publisher.
        """
        return self.__node.create_publisher(msg_type, topic,
                                            self._DEFAULT_QOS_DEPTH)

    ####################
    ### subscribers
    ####################
    def __keyboard_callback(self, msg: HexRosTeleopKeyboardStateStamped):
        self._keyboard_deque.append(self.__keyboard_msg_to_dc(msg))

    @staticmethod
    def __keyboard_msg_to_dc(
            msg: HexRosTeleopKeyboardStateStamped) -> HexDcTeleopKeyboardState:
        kb = msg.keyboard_state
        kwargs = {
            f"key_{letter}": bool(getattr(kb, f"key_{letter}"))
            for letter in _LETTERS
        }
        return HexDcTeleopKeyboardState(**kwargs)

    ####################
    ### logging
    ####################
    def getlogger(self):
        """Return a logger named like the log methods of this interface.

        Returns:
            _LoggerAdapter: Logger exposing ``logd`` .. ``logf``.
        """
        return _LoggerAdapter(self.__logger)

    def logd(self, msg, *args, **kwargs):
        self.__logger.debug(msg, *args, **kwargs)

    def logi(self, msg, *args, **kwargs):
        self.__logger.info(msg, *args, **kwargs)

    def logw(self, msg, *args, **kwargs):
        self.__logger.warning(msg, *args, **kwargs)

    def loge(self, msg, *args, **kwargs):
        self.__logger.error(msg, *args, **kwargs)

    def logf(self, msg, *args, **kwargs):
        self.__logger.fatal(msg, *args, **kwargs)


class _LoggerAdapter:
    """Forward the interface log levels to an ``rclpy`` logger."""

    def __init__(self, logger):
        self.__logger = logger

    def logd(self, msg, *args, **kwargs):
        self.__logger.debug(msg, *args, **kwargs)

    def logi(self, msg, *args, **kwargs):
        self.__logger.info(msg, *args, **kwargs)

    def logw(self, msg, *args, **kwargs):
        self.__logger.warning(msg, *args, **kwargs)

    def loge(self, msg, *args, **kwargs):
        self.__logger.error(msg, *args, **kwargs)

    def logf(self, msg, *args, **kwargs):
        self.__logger.fatal(msg, *args, **kwargs)
