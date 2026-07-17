# PRISM 灵巧手数采系统项目规划

## 1. 项目定位

本项目用于灵巧手数据采集、离线后处理，以及后续在真实机器人上的部署。系统整体分为两个阶段：

- 在线采集：同步采集多相机视频、实时显示轨迹和画面、转发 RPi 指令到灵巧手 SDK、记录灵巧手反馈信号，并按任务保存 raw 数据。
- 离线处理：对 raw 数据进行时间戳对齐、轨迹精细重建、数据清洗与格式化，生成可用于训练、分析或部署的数据集。

项目需要兼容两类灵巧手：

- 二代手：记录主机发出的 SDK 命令与时间戳。
- 三代手：记录灵巧手端口返回的角度反馈与时间戳。

## 2. 在线采集流程

在线部分负责完成一个任务下的多条数据采集。操作者运行采集脚本后，可以通过键盘控制每一条数据的开始、停止和整个任务的结束。

### 2.1 启动阶段

1. 操作者运行在线采集脚本，并指定任务名称、采集条数、保存目录、设备配置等参数。
2. 系统初始化以下设备和服务：
   - 4 个高速相机。
   - 1 个 Intel RealSense 相机。
   - 高速相机轨迹实时重建模块。
   - 5 路相机实时画面显示模块。
   - 与 RPi 通信的串口。
   - 与灵巧手 SDK 或灵巧手控制端口通信的接口。
   - 灵巧手反馈记录模块。
3. 系统进入等待状态，等待操作者按键开始采集单条数据。

### 2.2 单条数据采集

当操作者按下指定按键后，系统开始采集一条数据：

1. 生成本条数据的唯一编号和保存目录。
2. 同步启动以下记录：
   - 4 个高速相机带时间戳视频。
   - 1 个 RealSense 带时间戳视频。
   - 4 个高速相机重建得到的实时轨迹。
   - RPi 串口接收到的原始命令与时间戳。
   - 转换后的灵巧手 SDK 命令与时间戳。
   - 灵巧手反馈信号与时间戳。
3. 主机实时显示：
   - 4 个高速相机重建出的实时轨迹。
   - 5 个相机的实时画面。
4. RPi 会通过串口不定时向主机发送命令。主机需要持续监听串口，并完成：
   - 读取 RPi 原始命令。
   - 校验和解析命令。
   - 转换为灵巧手 SDK 命令。
   - 发送到灵巧手控制端口。
   - 记录原始命令、转换后命令、发送结果和时间戳。
5. 灵巧手反馈记录规则：
   - 二代手：记录已发送命令和发送时间戳。
   - 三代手：记录灵巧手端口返回的角度反馈和接收时间戳。

### 2.3 单条数据结束

当操作者按下停止按键，或系统收到停止条件后：

1. 停止 4 个高速相机和 RealSense 的录制。
2. 停止当前条目的轨迹和反馈记录。
3. 写入本条数据的 metadata，包括：
   - 任务名称。
   - 条目编号。
   - 开始和结束时间。
   - 使用的设备配置。
   - 灵巧手型号。
   - 相机序列号或设备 ID。
   - RPi 串口配置。
   - 采集过程中出现的错误或丢帧信息。
4. 将本条 raw 数据保存到指定目录。
5. 系统返回等待状态，准备采集下一条数据。

### 2.4 任务结束

当操作者采集够足够多条数据后，可以通过命令结束本次任务。结束时系统需要：

1. 关闭所有相机、串口和灵巧手通信接口。
2. 汇总任务级 metadata。
3. 检查 raw 数据完整性。
4. 提供后处理选项：
   - 立即后处理：在线采集结束后自动启动离线处理流程。
   - 之后后处理：仅保存 raw 数据，后续通过离线脚本单独处理。

## 3. 离线后处理流程

离线部分读取在线阶段保存的 raw 数据，并生成对齐、清洗和重建后的 processed 数据。

### 3.1 输入

离线处理输入为一个任务目录，包含多条 raw 数据。每条数据至少包括：

- 4 个高速相机视频及其时间戳。
- 1 个 RealSense 视频及其时间戳。
- RPi 原始命令日志。
- 灵巧手 SDK 命令日志。
- 灵巧手反馈日志。
- 在线实时轨迹日志。
- metadata 文件。

### 3.2 处理步骤

1. 加载 raw 数据和 metadata。
2. 校验各传感器数据完整性。
3. 统一时间基准，并进行多源时间戳对齐。
4. 对 4 个高速相机数据进行轨迹精细重建。
5. 对 RealSense 数据进行必要的帧提取、深度处理或外参对齐。
6. 将 RPi 命令、SDK 命令、灵巧手反馈和视觉轨迹对齐到同一时间轴。
7. 输出 processed 数据，并生成处理报告。

### 3.3 输出

离线处理输出建议包括：

- 对齐后的多相机视频索引。
- 精细重建后的轨迹数据。
- 对齐后的灵巧手命令和反馈序列。
- 每条数据的处理状态和质量报告。
- 可用于模型训练、分析或机器人部署的数据格式。

## 4. 建议数据组织

建议所有采集数据放在项目外部或项目根目录的 `data/` 下。若数据量很大，`data/` 应加入 `.gitignore`，只保留样例数据和格式说明。

```text
data/
  raw/
    task_YYYYMMDD_HHMMSS_task-name/
      task_metadata.yaml
      trial_000001/
        metadata.yaml
        cameras/
          highspeed_01.mp4
          highspeed_01_timestamps.csv
          highspeed_02.mp4
          highspeed_02_timestamps.csv
          highspeed_03.mp4
          highspeed_03_timestamps.csv
          highspeed_04.mp4
          highspeed_04_timestamps.csv
          realsense_rgb.mp4
          realsense_depth.mkv
          realsense_timestamps.csv
        hand/
          rpi_commands.csv
          sdk_commands.csv
          hand_feedback.csv
        trajectory/
          realtime_trajectory.csv
        logs/
          collector.log
      trial_000002/
        ...
  processed/
    task_YYYYMMDD_HHMMSS_task-name/
      processing_config.yaml
      processing_report.md
      trial_000001/
        aligned_timeline.csv
        refined_trajectory.csv
        hand_sequence.csv
        camera_index.json
      trial_000002/
        ...
```

## 5. 推荐项目文件结构

```text
PRISM/
  README.md
  pyproject.toml
  .gitignore
  configs/
    devices/
      cameras.yaml
      hand_gen2.yaml
      hand_gen3.yaml
      rpi_serial.yaml
    collection/
      default_online.yaml
    processing/
      default_offline.yaml
    deployment/
      real_robot.yaml
  src/
    prism/
      __init__.py
      cli/
        collect.py
        process.py
        validate_data.py
        deploy.py
      online/
        session_manager.py
        trial_controller.py
        keyboard_controller.py
        display_server.py
      devices/
        cameras/
          highspeed_camera.py
          realsense_camera.py
          camera_manager.py
        hand/
          hand_base.py
          hand_gen2.py
          hand_gen3.py
          sdk_command.py
        rpi/
          serial_client.py
          command_parser.py
      recording/
        video_recorder.py
        timestamp_writer.py
        hand_logger.py
        metadata_writer.py
      reconstruction/
        realtime_reconstruction.py
        offline_reconstruction.py
        calibration.py
      processing/
        align_timestamps.py
        build_dataset.py
        quality_check.py
      deployment/
        robot_adapter.py
        policy_runner.py
        safety_monitor.py
      common/
        timebase.py
        config.py
        logging.py
        types.py
  scripts/
    collect_task.sh
    process_task.sh
    calibrate_cameras.sh
  tools/
    inspect_trial.py
    replay_trial.py
    visualize_trajectory.py
  docs/
    hardware_setup.md
    data_format.md
    collection_protocol.md
    deployment_plan.md
  tests/
    unit/
    integration/
  data/
    raw/
    processed/
```

## 6. 模块职责说明

### 6.1 CLI 层

- `collect.py`：启动在线采集任务。
- `process.py`：启动离线后处理任务。
- `validate_data.py`：检查 raw 或 processed 数据完整性。
- `deploy.py`：后续在真实机器人上运行部署流程。

### 6.2 在线采集层

- `session_manager.py`：管理一个任务级采集会话。
- `trial_controller.py`：管理单条数据的开始、停止、保存和状态切换。
- `keyboard_controller.py`：监听操作者键盘输入。
- `display_server.py`：显示实时轨迹和 5 路相机画面。

### 6.3 设备层

- `camera_manager.py`：统一管理 4 个高速相机和 1 个 RealSense。
- `highspeed_camera.py`：封装高速相机初始化、录制、时间戳读取。
- `realsense_camera.py`：封装 RealSense RGB、depth 和时间戳读取。
- `serial_client.py`：负责读取 RPi 串口命令。
- `command_parser.py`：将 RPi 原始命令解析成内部命令格式。
- `hand_gen2.py`：二代手 SDK 命令发送与日志记录。
- `hand_gen3.py`：三代手角度反馈读取与日志记录。

### 6.4 记录与元数据层

- `video_recorder.py`：统一视频写入。
- `timestamp_writer.py`：统一时间戳写入。
- `hand_logger.py`：记录命令、反馈和发送状态。
- `metadata_writer.py`：写入任务级和单条数据级 metadata。

### 6.5 重建与后处理层

- `realtime_reconstruction.py`：在线显示用的高速相机实时轨迹重建。
- `offline_reconstruction.py`：离线精细轨迹重建。
- `align_timestamps.py`：统一时间轴和多模态数据对齐。
- `quality_check.py`：检查丢帧、时间戳异常、命令缺失等问题。
- `build_dataset.py`：生成训练、分析或部署所需的数据格式。

### 6.6 部署层

- `robot_adapter.py`：封装真实机器人控制接口。
- `policy_runner.py`：加载模型或策略，并生成机器人动作。
- `safety_monitor.py`：运行限位、速度、碰撞、急停等安全检查。

部署层需要和数据采集层共享统一的数据类型、时间基准、灵巧手命令格式和配置系统，避免后续从数据集迁移到真实机器人时重复改接口。

## 7. 关键设计建议

### 7.1 统一时间基准

所有模块都应该记录同一种主机时间戳，例如 `time.monotonic_ns()` 或统一封装后的 `Timebase.now_ns()`。设备自带时间戳也应保留，但不要直接作为唯一对齐依据。

### 7.2 命令链路可追溯

RPi 原始命令、解析后的内部命令、发送到灵巧手 SDK 的命令、灵巧手反馈都应分别记录。这样后续可以排查：

- RPi 是否发错。
- 主机是否解析错。
- SDK 是否发送失败。
- 灵巧手是否执行异常。

### 7.3 二代手和三代手使用统一接口

建议定义统一的 `HandInterface`，二代手和三代手分别实现：

- `connect()`
- `send_command(command)`
- `read_feedback()`
- `start_logging(trial_dir)`
- `stop_logging()`
- `close()`

这样在线采集、离线处理和后续部署不需要关心具体是哪一代手。

### 7.4 raw 数据不可覆盖

在线采集得到的 raw 数据应只追加、不覆盖。离线处理结果写入 `processed/`，并保存处理配置和处理报告，保证实验可复现。

### 7.5 面向真实机器人部署预留接口

后续部署时，系统需要支持：

- 从 processed 数据训练或验证策略。
- 在真实机器人上读取相机和灵巧手状态。
- 将策略输出转换为机器人和灵巧手动作。
- 运行实时安全检查。
- 记录部署过程中的观测、动作、反馈和异常。

因此，采集阶段就应尽量使用清晰的数据结构和统一的动作表示，减少后续迁移成本。

## 8. 第一阶段开发顺序建议

1. 搭建配置系统和日志系统。
2. 实现单设备最小闭环：一个相机录制、一个串口读取、一个灵巧手命令发送日志。
3. 扩展到 4 个高速相机和 1 个 RealSense 同步采集。
4. 实现任务级和单条数据级目录保存规范。
5. 实现键盘控制的开始、停止和结束任务流程。
6. 加入实时画面和实时轨迹显示。
7. 加入完整 raw 数据校验。
8. 实现离线时间戳对齐和轨迹精细重建。
9. 实现 processed 数据导出。
10. 设计并接入真实机器人部署接口。
