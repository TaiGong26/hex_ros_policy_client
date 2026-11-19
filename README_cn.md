# hex_ros_policy_client

模仿学习部署用的 ROS 2 策略客户端。设备变成话题：节点订阅 `manip_state` 与
`image_color_raw`，发布 `manip_ctrl`，把观测发给模型侧并执行返回的动作块。

## 分层

| 层 | 目录 | 放什么 | 允许 import |
|---|---|---|---|
| 节点层 `policy_client` | 包根 | ROS 节点、参数校验、事件表与行为表 | 以下全部 |
| 运行时层 `core` | `core/` | 组合机器人、状态机 | `process`、适配器、`utils`、`struct` |
| 处理层 `process` | `process/` | 观测装配、动作块集成 | 适配器、`utils`、`struct` |
| 适配器 `protocol` | `protocol/` | 线上契约、帧编解码、Zenoh 传输 | `utils`、`struct` |
| 适配器 `device` | `device/` | ROS 设备封装、设备表与工厂 | `utils`、`struct` |
| 工具层 `utils` | `utils/` | 日志文法、数学表、CSV 落盘 | `struct` |
| 数据层 `struct` | `struct/` | 数据类与线上帧契约 | 仅标准库与 `numpy` |

`protocol` 与 `device` 是**同层的两个适配器**（一个朝模型侧、一个朝 ROS），
二者**互不 import**，所以更换传输载体不会波及设备层。其下为单向依赖
`policy_client → core → process → {protocol, device} → utils → struct`，由
`test/test_import_direction.py` 守住；`struct` 是纯数据层，由
`test/test_struct_purity.py` 守住。

Zenoh 侧不自建：`PolicyMiddleware` **持有一个** `hex_util_zenoh.HexZenohNode`
（组合，不继承），只额外负责三条话题、构帧调用与动作块队列。该工具包用
`pip install -e <workspace>/pkg/hex_util_zenoh` 安装。

帧是与模型侧的约定而非领域数据，所以**形状**声明在
`protocol/policy_protocol.py`，**转字节**在 `protocol/frame_codec.py`，
**搬字节**在 `protocol/policy_middleware.py`。字段清单不重复：编解码直接读
帧类型的 `__required_keys__`。改线上格式只需动前两个文件。

两处可替换的决策都做成"签名固定 + 一张表"：`process/action_restore.py` 按
`action_mode` 把增量还原成绝对位置，`process/action_average.py` 给重叠块的多路
样本加权（`none` / `exponential` / `plain`）。

## 运行

```bash
ros2 launch hex_ros_policy_client policy_client.launch.py
```

设备侧要先起来：`hex_ros_robot_arm` 提供 `<arm>/manip_state` 并消费
`<arm>/manip_ctrl`，`hex_ros_camera` 提供 `<camera>/image_color_raw`。

## 状态

| 状态 | 含义 | 离开条件 |
|---|---|---|
| `INIT` | 等表内每个设备的第一帧 | `init_success`、`init_timeout` |
| `RETURN_TO_STABLE` | 插值回稳到 STABLE 目标 | `stable_arrived`、`stable_timeout` |
| `WAITING` | 保持位姿，等任务 | `start` |
| `RUNNING` | 发观测、执行动作块 | `end`、`episode_timeout` |
| `EXIT` | 终态，资源已回收 | — |

四条终止路径（模型侧 `exit`、Ctrl+C、启动超时、回稳超时）都收敛到 `EXIT`，
差别只在记录的 `reason` 与进程退出码。

## 退出码

| reason | 退出码 |
|---|---|
| `exit` | 0 |
| `init_timeout` | 1 |
| `stable_timeout` | 1 |
| `keyboard_interrupt` | 130 |
| 其它 | 1 |

## 扩展点

每种改动需要碰哪些文件，见下表；表外文件不需要改。

| 想做的事 | 要改的文件 |
|---|---|
| 增删设备、改话题 / 图像尺寸 / `grip_limit` / `lim_vel` / `lim_acc` / `grav` | 只改 `config/ros2/devices.yaml` |
| 改标量（频率、阈值、话题、插值、录制目录） | 只改 `config/ros2/policy_client.yaml` |
| 支持另一种插值 | `utils/math_util.py` 的 `INTERPOLATORS` 加一项 |
| 换观测装配方式 | 只改 `process/observation_base.py` 与 `process/history_observation.py` |
| 换动作块集成方式 | 只改 `process/action_process_base.py` 与 `process/time_integration_chunk.py` |
| 换帧的线上格式 | 改 `protocol/policy_protocol.py`（形状）与 `protocol/frame_codec.py`（字节） |
| 支持另一种动作编码 | `ACTION_RESTORERS` 加一项（`process/action_restore.py`） |
| 换重叠块的合并方式 | `AVERAGERS` 加一项（`process/action_average.py`） |
| 换传输载体（Zenoh 换别的） | 只改 `protocol/policy_middleware.py` |
| 支持另一种控制模式 | `struct/cmd_struct.py` 加一个常量；`utils/arm_device.py` 映射表加一行 |
| 新增设备类型（新传感器） | 新增 `device/<name>_device.py` 继承 `DeviceBase()` + `core/device_hub.py: DEVICE_BUILDERS` 加一项 |
| 新增另一种拓扑的节点 | 包根新增 `<name>_policy_client.py` 继承 `PolicyClientBase`（覆盖 `_build_components` 与 `check_conditions`，注册自己的表）+ 它的 YAML + `setup.py` 一个入口 |

单臂不需要改任何 `.py`：设备表只声明一条臂，再把 `stable_arm_order` 与
`stable_jnt` 调成一致即可。`test/test_extension.py` 里有这两种情形的可运行示例。

## 测试

```bash
pytest hex_ros2_ws/src/hex_ros_policy_client/test -v
```

单测不启动 ROS、不碰硬件：节点、设备与传输都由假对象替代。

## 与 `hex_policy_bridge` 的关系

旧包 `hex_policy_bridge` 保留并存、继续可用。新部署以本包为唯一入口：设备层走
ROS 话题而不是厂商 SDK，节点按职责拆成多个类。
