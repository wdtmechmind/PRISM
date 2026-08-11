# RPi 脚本启动指南

本文档整理 PRISM 仓库里与 RPi 相关的脚本启动方式，覆盖以下场景：

- RPi 端编码器读取与 UDP 事件发送
- 主机端在线采集接收 RPi 事件并写入 trial 日志
- 主机端实时可视化 RPi 编码器流
- Isaac Sim 中 support_right 手部遥操作
- 三代手实机（MechHand V3 daemon）遥操作

## 1. 相关脚本一览

- RPi 编码器与事件发送：`src/prism/devices/rpi/io_interface.py`
- RPi 编码器流可视化：`tools/plot_rpi_encoder_stream.py`
- Isaac support_right 遥操作：`simulation/scripts/teleop_support_right_from_rpi.py`
- 三代手实机遥操作：`tools/teleop_mechhand_v3_from_rpi.py`
- USB 网卡重连辅助：`scripts/restart_rpi_usb.sh`

## 2. 启动前准备

### 2.1 主机侧先恢复 RPi USB 网络

```bash
cd /mnt/projects-8tb/PRISM
bash scripts/restart_rpi_usb.sh
```

可选参数（按顺序）：

```bash
bash scripts/restart_rpi_usb.sh <IF_NAME> <RPI_IP> <SSH_USER> <CONNECTION_NAME>
```

示例：

```bash
bash scripts/restart_rpi_usb.sh enx3ed0bbcddb23 10.12.194.1 xining rpi-usb-host
```

如果只想恢复网络不自动 ssh：

```bash
AUTO_SSH=0 bash scripts/restart_rpi_usb.sh
```

### 2.2 RPi 端依赖

`io_interface.py` 依赖 `pigpio` 和 `gpiozero`，且要求 `pigpiod` 已启动。

另：当前 RPi（xining@10.12.194.1）实测仓库目录为 `/home/xining/mechhand_project`，
且直接 `python3` 默认无法导入 `prism`（需激活环境或设置 `PYTHONPATH=src`）。

在 RPi 上：

```bash
sudo pigpiod
```

可选：如果你有专用虚拟环境，先激活再启动脚本：

```bash
cd /home/xining/mechhand_project
source .venv/bin/activate
python3 -m prism.devices.rpi.io_interface --help
```

如果没有虚拟环境，使用 `PYTHONPATH=src` 兜底（下节默认用这个方式）。

RPi 一键自检（建议先跑）：

```bash
cd /home/xining/mechhand_project
python3 -c "import prism" || PYTHONPATH=src python3 -c "import prism; print('prism import OK via PYTHONPATH=src')"
```

## 3. 场景 A：RPi 持续发送编码器 UDP 流（推荐基础链路）

用途：给主机持续发送 `prism.rpi_hand_event.v1` 数据包（包含 Enc1..Enc5 角度），供采集/仿真/实机桥接消费。

在 RPi 上执行（先切到 RPi 本机上的仓库目录，不是主机的 `/mnt/projects-8tb/PRISM`）：

```bash
cd <RPI_PRISM_DIR>
PYTHONPATH=src python3 -m prism.devices.rpi.io_interface \
  --disable-hand-trigger \
  --event-host <HOST_IP> \
  --event-port 60701 \
  --event-stream-hz 30
```

示例（已在 RPi 上核对的路径）：

```bash
cd /home/xining/mechhand_project
PYTHONPATH=src python3 -m prism.devices.rpi.io_interface \
  --disable-hand-trigger \
  --event-host 10.12.194.2 \
  --event-port 60701 \
  --event-stream-hz 30
```

说明：

- `--disable-hand-trigger`：仅发编码器事件，不直接发 SDK 手势命令。
- `--event-stream-hz 30`：连续事件频率，建议 20 到 40 Hz。
- `<RPI_PRISM_DIR>` 填 RPi 本机仓库目录。
- `<HOST_IP>` 填主机 IP（USB 直连常见 `10.12.194.2`，以实际网段为准）。

如果你希望 RPi 端继续做边沿触发手势，也可去掉 `--disable-hand-trigger`，但实时跟随场景建议关闭触发，专注连续流。

## 4. 场景 B：主机在线采集并记录 RPi 事件

### 4.1 启动在线采集

```bash
cd /mnt/projects-8tb/PRISM
scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 5 \
  --output-dir data/raw \
  --rpi-event-host 0.0.0.0 \
  --rpi-event-port 60701
```

说明：

- `--rpi-event-host/--rpi-event-port` 控制主机 UDP 监听地址。
- 监听成功时会打印 `RPi hand event UDP logging ...`。

### 4.2 产物位置

录制期间收到的 RPi 事件会写到：

- `trial_xxxxxx/hand/rpi_commands.csv`
- 并同步写入 `trial_xxxxxx/hand/sdk_commands.csv`（按统一手命令时间线格式）

## 5. 场景 C：主机实时查看 RPi 编码器曲线

```bash
cd /mnt/projects-8tb/PRISM
python3 tools/plot_rpi_encoder_stream.py \
  --host 0.0.0.0 \
  --port 60701 \
  --window-sec 20 \
  --refresh-ms 50
```

用途：快速确认 Enc1..Enc5 是否在稳定更新，排查“主机收不到包”或“角度抖动过大”。

## 6. 场景 D：Isaac Sim 支持手遥操作（support_right）

前提：场景 A 正在发 UDP 流。

```bash
cd /mnt/projects-8tb/PRISM
/isaac-sim/python.sh simulation/scripts/teleop_support_right_from_rpi.py \
  --config simulation/configs/support_right_rpi_5ch.yaml
```

可选调试：

```bash
/isaac-sim/python.sh simulation/scripts/teleop_support_right_from_rpi.py \
  --config simulation/configs/support_right_rpi_5ch.yaml \
  --event-host 0.0.0.0 --event-port 60701 --print-dofs
```

## 7. 场景 E：三代手实机遥操作（保持 absolute 模式）

前提：

- 场景 A 正在发 UDP 流。
- MechHandDaemonV3 已在主机可达地址启动，端口与配置一致。

启动命令：

```bash
cd /mnt/projects-8tb/PRISM
python3 tools/teleop_mechhand_v3_from_rpi.py \
  --config configs/deployment/mechhand_v3_teleop.yaml
```

常用选项：

- `--calibrate-sec 10`：启动前做 10 秒输入范围标定（手动开启）。
- `--calibrate-sec 0`：跳过标定（默认）。
- `--dry-run`：只打印，不发 RM 到 daemon。

触发式发命令（推荐实机）：

- 配置项：`control.command_trigger_delta_deg`
- 含义：与上次已发送命令相比，任一关节目标变化达到该阈值才发送新 RM。
- 建议起步值：`6.0`（若仍超时可加到 `8.0` 到 `12.0`；若跟随变钝可降到 `3.0` 到 `5.0`）。

### 7.1 独立标定：把外骨骼角度映射到三代手满量程

用途：独立于采集流程，标定每个关节达到目标满量程时对应的外骨骼角度，
并写回 `configs/deployment/mechhand_v3_teleop.yaml` 的 `mapping` 字段。

这个脚本与 RPi PWM 标定脚本不冲突：

- 本脚本：`tools/calibrate_mechhand_v3_mapping.py`
  - 修改 `mapping.J1..J5.source_min/source_max/invert`
  - 用于“想达到三代手满行程（输出端接近 +/-900）需要的角度”
- 现有脚本：`tools/calibrate_encoder_pwm.py`
  - 修改 `configs/devices/encoder_pwm_calibration*.json`
  - 用于 PWM duty 到编码器角度/归一化位置的底层标定

启动方式（主机执行，监听 RPi UDP 流）：

```bash
cd /mnt/projects-8tb/PRISM
python3 tools/calibrate_mechhand_v3_mapping.py \
  --config configs/deployment/mechhand_v3_teleop.yaml \
  --event-host 0.0.0.0 \
  --event-port 60701
```

交互流程：

1. 逐个关节 J1..J5 提示你先摆到 target_min，再摆到 target_max。
2. 每次按 Enter 抓一段 UDP 角度样本并取中位数。
3. 结束后确认写入配置，默认会先备份原 YAML。

## 8. 常见问题排查

### 8.1 RPi 端报错：pigpio daemon not running

处理：

```bash
sudo pigpiod
```

### 8.2 主机日志提示 waiting fresh encoder stream

说明主机监听到了端口，但没有收到新包。检查：

1. RPi 端 `io_interface.py` 是否正在运行。
2. `--event-host` 是否指向主机实际 IP。
3. 主机端口是否一致（默认 60701）。
4. 防火墙或网段是否阻断 UDP。

### 8.3 端口 bind failed

说明端口被占用。处理：

- 先停掉占用进程，或换一个 `--event-port`。
- RPi 发送端和主机接收端端口必须一致。

### 8.4 `RM failed: ERR:device_timeout`（三代手实机常见）

现象：`teleop_mechhand_v3_from_rpi.py` 连续打印 `device_timeout, backoff ...`，最后触发
`too many consecutive device_timeout errors` 退出。

原因：通常是控制参数过激（发送频率太高、关节步进太大、输入抖动导致频繁改目标），
设备端任务积压或超时。

处理：

1. 先把部署配置降档为稳态参数（仓库当前已更新）：
  - `command_rate_hz: 18.0`
  - `min_send_interval_sec: 0.03`
  - `input_deadband_deg: 2.5`
  - `joint_deadband_deg: 0.45`
  - `max_joint_speed_deg_s: 180.0`
  - `device_timeout_retry_delay_sec: 1.2`
2. 启动时先跳过标定排查链路（避免把变量混在一起）：

```bash
cd /mnt/projects-8tb/PRISM
python3 tools/teleop_mechhand_v3_from_rpi.py \
  --config configs/deployment/mechhand_v3_teleop.yaml \
  --calibrate-sec 0
```

3. 如果仍频繁超时：
  - 继续下调 `command_rate_hz` 到 `12` 到 `15`
  - 或临时加大 `input_deadband_deg` 到 `3.0` 到 `5.0`

### 8.5 restart_rpi_usb.sh 返回 1

常见原因是脚本默认网卡名不存在。处理：

```bash
ip link
bash scripts/restart_rpi_usb.sh <实际网卡名> <RPI_IP> <SSH_USER> <CONNECTION_NAME>
```

## 9. 推荐启动顺序（实操）

1. 主机执行 `bash scripts/restart_rpi_usb.sh`，确认能 ping 通 RPi。
2. RPi 启动 `io_interface.py` 连续发 UDP（建议 `--disable-hand-trigger --event-stream-hz 30`）。
3. 主机先开 `plot_rpi_encoder_stream.py` 看曲线是否实时。
4. 根据目标场景启动：
   - 采集：`scripts/collect_task.sh ... --rpi-event-port 60701`
   - 仿真：`teleop_support_right_from_rpi.py`
   - 实机三代手：`teleop_mechhand_v3_from_rpi.py`
