# PRISM 灵巧手多相机采集

PRISM 当前提供一套 native 在线采集流程和离线严格对齐重建流程，用于 4 路 Hik 高速相机、1 路 Intel RealSense D435、red/yellow/blue/green 四个刚性安装 LED 的轨迹和 dex hand 刚体 6D pose 记录。

当前默认入口已经不再桥接旧的 MVS Recording 在线脚本；旧 fallback 环境变量和旧离线脚本入口也已经从 PRISM 源码、脚本和文档中移除。

## 1. 外部依赖边界

PRISM 代码已经迁入本仓库，但仍依赖厂商/设备运行时：

- MVS SDK: `/opt/MVS`
- MVS Python binding: `/opt/MVS/Samples/64/Python/MvImport/MvCameraControl_class.py`
- MVS runtime setup: `/opt/MVS/bin/set_env_path.sh /opt/MVS`
- Python 环境建议: `/home/daotan/miniforge3/envs/camera/bin/python`

这类 SDK 文件不复制进 PRISM，运行脚本会自动设置需要的 MVS 环境和 `PYTHONPATH`。

## 2. 在线采集

默认启动命令：

```bash
cd /mnt/projects-8tb/PRISM

scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 20 \
  --output-dir data/raw
```

也可以直接覆盖关键参数：

```bash
scripts/collect_task.sh \
  --calib-json configs/devices/charuco_4cam_result.json \
  --rs-calib-json configs/devices/d435_charuco_intrinsics.json \
  --output-dir data/raw \
  --hik-frame-rate 300 \
  --track-every 1 \
  --viz-3d y
```

启动后会列出 USB Hik 相机并提示输入 4 个 Hik 相机索引。直接回车表示选择前 4 台。

摄像头打开后会自动识别并配置 Master/Slave（根据序列号 DA8165486），然后进入正常采集循环。

运行时按键：

- `Space`: 开始/停止当前 trial 录制
- `p`: 暂停轨迹更新
- `r`: 恢复轨迹更新
- 在统一预览窗口右侧 hand 控制面板中直接点击手势按钮 `1..17`（推荐）
- 也可在预览窗口键盘输入手势编号 `1..17`：
  - `2..9` 输入后立即发送
  - `1` 可能是 `1` 或 `10..17`：可继续输入第二位，或按 `Enter` 立即发送 `1`
  - `Backspace` 清除待发送编号
- `q` 或 `ESC`: 结束采集任务

Qt 试验 UI（`--ui-backend qt`）下，当前保持键盘/CLI 控制，不启用点击发送手势。

手势协议规则：`gesture_id = ROG_value + 1`。例如：`gesture_id=1 -> @ROG<0>&`，`gesture_id=17 -> @ROG<16>&`。

手部控制连接参数（同一条 `prism-collect` 命令生效）：

- `--hand-ip`: 机械手控制器 IP
- `--hand-port`: 机械手控制器 TCP 端口
- `--hand-timeout-s`: socket 超时
- `--hand-settle-time-s`: 每次指令后的等待时间
- `--hand-auto-connect y|n`: 是否在采集启动时主动连接；默认 `n`（首次按 1/2/3/4 时懒连接）

## 3. 在线采集能力

当前 native 在线流程包括：

- 4 路 Hik USB 相机**硬件触发同步**采集（Master + Slave 配置）
- 1 路 RealSense 彩色流采集
- 每路视频异步写盘，写盘队列满时丢帧保护采集节奏
- 实时显示 Hik/RealSense fps
- 四个 LED HSV 检测：red、yellow、blue、green
- 最近邻时间对齐轨迹：`trajectory_led_nearest.csv`
- 时间插值轨迹：`trajectory_led_interp.csv`
- Hik/RealSense 时间对齐质量日志：`time_alignment_log.csv`
- 刚体 6D pose：`rigid_pose_6d.csv`
- 单一统一预览窗口：左侧保留原视频/overlay，右侧嵌入原 3D 轨迹视图
- 预览新增 hand 状态区：socket 连接状态、最近动作、最近命令及其时间戳、累计命令数
- open-loop hand SDK 指令日志，时间戳与轨迹 `t_sec` 使用同一时间基准
- **硬件触发帧级同步**：所有 4 路 Hik 相机在硬件级同步（< 1 ms），Master 相机输出 GPIO 脉冲同步 Slave 相机

单个 LED 轨迹重建的必要条件：该 LED 至少被两个 Hik 相机同时检测到。dex hand 6D pose 的必要条件：当前帧至少有 3 个已经进入刚体模型的 LED 被成功三角化；不要求每个 LED 被每台相机看到，也不要求四个 LED 每帧都同时可见。刚体模型会先用第一帧可用的非共线 3 个 LED 初始化，后续如果第四个 LED 出现，会自动加入模型。若条件不足，预览会显示 `rigid6d: need >=3 modeled LEDs`。

## 4. 在线采集默认配置

在线采集默认配置在：

```text
configs/collection/default_online.yaml
```

命令行参数会覆盖 YAML 配置。长期稳定参数建议写进 YAML；单次实验临时变化建议用 CLI 参数覆盖。

常用参数：

- `--calib-json`: 4 Hik charuco 标定 JSON
- `--rs-calib-json`: RealSense 内参 JSON
- `--output-dir`: raw 数据输出目录
- `--hik-exposure-us`: Hik 曝光时间
- `--hik-gain`: Hik 增益
- `--hik-frame-rate`: Hik 目标采集帧率
- `--writer-queue`: 每路写盘队列长度
- `--track-every`: 每 N 次 preview loop 做一次 LED tracking
- `--frame-buffer`: 每路相机时间戳帧缓存长度
- `--preview-target-w`: 统一预览里每个相机子图的渲染宽度（更大更清晰，但更耗 CPU）
- `--preview-window-width`: 统一预览窗口初始宽度
- `--preview-window-height`: 统一预览窗口初始高度
- `--ui-backend opencv|qt`: 预览窗口后端（`qt` 为试验路径，仍保持键盘/CLI 控制）

说明：统一预览会尽量把拼接结果宽度对齐到窗口宽度，减少 OpenCV 二次缩放导致的发糊；若仍偏糊，可优先增大 `--preview-window-width`，其次增大 `--preview-target-w`。
另外，系统会根据窗口高度自动下调子图宽度，避免超过窗口高度后被再次缩放（这会让视频和文字一起变糊）。
- `--viz-3d`: 是否打开 Matplotlib 3D 轨迹窗口
- `--rigid-axis-len`: 3D 窗口里刚体坐标轴长度，单位米

注意：这里的 `red`、`yellow`、`blue`、`green` 是刚体上 LED 的身份标签，不一定等于 Hik 图像里肉眼看到的颜色。当前 Hik 画面里 red LED 偏橙/黄，因此默认 `r_h_low/r_h_high` 使用 orange/amber 区间，`yellow` 区间相应后移以减少重叠。若更换 LED、曝光或白平衡，应重新调 `configs/collection/default_online.yaml` 里的 HSV 阈值。

## 5. 多相机硬件触发同步

⚠️ **重要更新**（2026-07-23）：PRISM 已迁移至硬件触发同步方案，所有 4 路 Hik 相机现已采用以下配置：

- **Master 相机** (DA8165486): 软件触发 + GPIO Line1 输出
- **Slave 相机** (cam1~3): 硬件触发 on GPIO Line0 输入
- **同步精度**: < 1 ms（帧级别）
- **校准周期**: 需要进行 ChArUco 标定（一次性，与相机内参一起保存）

### 硬件触发的优势

相比之前的自由运行 + 事后 AprilTag 时间校准：

| 指标 | 自由运行 | 硬件触发 |
|---|---|---|
| 帧同步精度 | 0 ~ 33ms（随机相位） | < 1 ms（同一脉冲） |
| 时间校准方式 | 每 session AprilTag 检测 | 无需校准（硬件保证） |
| 摄像头工作模式 | 连续采集 | 等待脉冲 |
| 码流稳定性 | 帧率漂移 ±5% | 严格 CFR |

### ChArUco 标定流程（首次设置必要）

执行硬件触发采集前，需要重新标定 4 个相机的内外参：

```bash
# 1. 捕获标定图像（硬件触发专用捕获脚本）
python3 tools/prism_charuco_calibration_capture.py \
  --output-dir ~/mvs_charuco_data \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25

# 2. 运行官方 MVS 标定
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoCalibrate4Cam.py \
  --dataset-root ~/mvs_charuco_data/CharucoCapture_[timestamp] \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --output ~/mvs_charuco_data/charuco_4cam_result.json

# 3. 更新项目配置
cp ~/mvs_charuco_data/charuco_4cam_result.json \
   /mnt/projects-8tb/PRISM/configs/devices/charuco_4cam_result.json
```

详见 [docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md](docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md)。

### GPIO 硬件连接

确保以下连接已正确建立：

- **Master 相机 DA8165486**
  - GPIO Line1 输出 → 连接至所有 Slave 相机的 GPIO Line0
  - 输出：3.3V TTL 脉冲，频率 = 目标帧率

- **Slave 相机** (cam1~3)
  - GPIO Line0 输入 ← 连接至 Master 的 GPIO Line1
  - 输入：3.3V TTL 脉冲

标定完成并更新 `charuco_4cam_result.json` 后，`prism-collect` 会自动识别 Master/Slave 配置。

## 6. 输出结构

一次任务会生成类似结构：

```text
data/raw/
  task_YYYYmmdd_HHMMSS_task-name/
    task_metadata.yaml
    trajectory_led_nearest.csv
    trajectory_led_interp.csv
    rigid_pose_6d.csv
    time_alignment_log.csv
    hand_sdk_commands_timeline.csv
    realsense_intrinsics.json
    trial_000001/
      metadata.yaml
      cameras/
        hik0_<serial>.mp4
        hik0_<serial>_timestamps.csv
        hik1_<serial>.mp4
        hik1_<serial>_timestamps.csv
        hik2_<serial>.mp4
        hik2_<serial>_timestamps.csv
        hik3_<serial>.mp4
        hik3_<serial>_timestamps.csv
        realsense_color.mp4
        realsense_color_timestamps.csv
      hand/
        rpi_commands.csv
        sdk_commands.csv
        hand_feedback.csv
      logs/
        fps_log.csv
```

说明：

- `hand_sdk_commands_timeline.csv` 记录任务级 open-loop 指令，`t_sec` 与轨迹 CSV 对齐。
- `trial_xxxxxx/hand/sdk_commands.csv` 与 `trial_xxxxxx/hand/rpi_commands.csv` 记录 trial 期间指令。
- `hand_feedback.csv` 仍是占位文件（反馈链路未实现）。

## 7. 离线严格对齐重建

PRISM 已迁入离线 CFR 对齐重建，可对一个 trial、`cameras/` 目录或旧式 segment 目录里的 `*_timestamps.csv` 和 `.mp4` 做最近邻重采样，输出统一帧率、统一时间轴的视频。

推荐对 PRISM trial 运行：

```bash
cd /mnt/projects-8tb/PRISM

scripts/rebuild_aligned_segment.sh \
  data/raw/task_YYYYmmdd_HHMMSS_task-name/trial_000001 \
  --time-range overlap \
  --include-rs y
```

也可以直接运行模块：

```bash
PYTHONPATH=src /home/daotan/miniforge3/envs/camera/bin/python \
  -m prism.processing.offline_rebuild \
  data/raw/task_YYYYmmdd_HHMMSS_task-name/trial_000001 \
  --target-fps 0 \
  --time-range overlap \
  --include-rs y \
  --codec mp4v
```

关键参数：

- `--target-fps`: 输出 CFR 帧率，`<=0` 表示自动取 Hik 实测最小 fps
- `--time-range overlap|union`: `overlap` 取所有流公共时间区间，`union` 覆盖全时段并在边缘重复帧
- `--include-rs y|n`: 是否把 RealSense 一起重建
- `--codec`: 输出视频 fourcc，默认 `mp4v`

默认输出：

```text
trial_000001/
  aligned_offline/
    hik*_aligned.mp4
    hik*_aligned_map.csv
    realsense_color_aligned.mp4
    realsense_color_aligned_map.csv
    alignment_summary.csv
```

`alignment_summary.csv` 会记录每路源帧数、输出帧数、实测 fps、平均/最大时间误差。

## 8. 仍未实现的边界

以下部分还不是完整功能：

- RPi 串口命令读取
- 灵巧手 SDK 命令转发
- Gen2/Gen3 hand feedback 实时记录
- 在线结束后的自动后处理入口

**硬件触发同步已实现**。Master 相机 (DA8165486) 输出 GPIO Line1 脉冲，Slave 相机在 GPIO Line0 接收触发。相机内参标定（ChArUco）需要进行一次性标定（首次设置或更换相机时）。完成标定后 `prism-collect` 会自动识别和配置 Master/Slave，后续采集无需手动干预。详见 [docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md](docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md)。

## 9. 手姿态 CLI 控制（RPi 就绪前）

在 RPi 串口桥接完成前，可以直接通过 TCP socket 控制机械手预设姿态。推荐直接在在线采集 CLI 内控制（同一个窗口同时控制采集和手势）。

示例：

```bash
cd /mnt/projects-8tb/PRISM

scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 20 \
  --output-dir data/raw \
  --hand-ip 127.0.0.1 \
  --hand-port 60686
```

运行中直接按 `1/2/3/4` 发送姿态命令，按 `Space` 开始/停止录制。
运行中可直接输入 `1..17` 发送对应手势，右侧 hand 控制面板会显示完整编号说明。

安装/更新项目后可使用：

```bash
cd /mnt/projects-8tb/PRISM
pip install -e .

prism-hand --ip 127.0.0.1 --port 60686
```

进入交互菜单后按键：

- `1..17`: 对应发送手势 1..17（协议中 `ROG = 手势号 - 1`）
- `0`: 退出

也可一条命令发送：

```bash
prism-hand --ip 127.0.0.1 --port 60686 --gesture-id 1
prism-hand --ip 127.0.0.1 --port 60686 --gesture-id 17
prism-hand --ip 127.0.0.1 --port 60686 --pose five_grasp
prism-hand --ip 127.0.0.1 --port 60686 --raw-cmd '@ROG<6>&'
```

## 10. 轨迹分析与诊断

采集完成后，可以用轨迹分析工具生成可视化，对比在线采集和离线重建的轨迹，诊断采集时帧率下降是否由于写队列溢出或 CPU 瓶颈引起。

### 自动分析（采集后立即运行）

```bash
cd /mnt/projects-8tb/PRISM

scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 5 \
  --output-dir data/raw \
  --post-process now
```

采集完成后会自动分析轨迹，生成以下可视化：

- `traj_3d_online.png` — 3D 轨迹视图（四个 LED）
- `traj_2d_online.png` — XY、XZ、YZ 二维投影
- `traj_timeline_online.png` — LED 追踪状态时间线（绿=已测量，黄=预测，红=丢失，灰=暂停）

### 手动分析

如果采集时选择 `--post-process later` 或要重新分析现有数据：

```bash
cd /mnt/projects-8tb/PRISM

# 用脚本
scripts/analyze_trajectory.sh data/raw/task_20260723_120000_grasp-demo

# 或者直接用 CLI
prism-analyze-trajectory data/raw/task_20260723_120000_grasp-demo --output-dir ./traj_plots
```

### 轨迹统计

分析输出会包含每个 LED 的统计信息：

- **总帧数** — 采集期间输出的总轨迹帧数
- **时长** — 采集时间长度
- **已测量比例** — `measured` 帧占比（越高越好）
- **预测帧数** — 当 LED 丢失时 Kalman 预测的帧数
- **丢失帧数** — 无法检测/预测的帧数
- **空间范围** — X、Y、Z 三个轴的运动范围

例如，某个 LED 的 **已测量比例** 从 98% 突然降到 50% 可能表示：
- **写队列溢出** — 相机帧被丢弃，导致 LED 检测间断。查看 `fps_log.csv` 是否有帧率下降。
- **CPU 瓶颈** — 预处理（LED 检测、三角化）跟不上采集速度。尝试：
  - 降低 `--track-every`（减少追踪频率）
  - 降低 `--preview-target-w`（减少预览渲染）
  - 关闭 `--viz-3d`（关闭 3D 窗口）

### 离线对比

完成离线重建后可以对比在线和离线轨迹：

```bash
scripts/rebuild_aligned_segment.sh \
  data/raw/task_20260723_120000_grasp-demo/trial_000001 \
  --time-range overlap

# 然后重新分析（会自动检测离线数据）
prism-analyze-trajectory data/raw/task_20260723_120000_grasp-demo
```

对比结果会显示在线采集和离线重建的 3D 轨迹差异，帮助判断采集质量。

## 11. 开发与验证

常用无硬件验证：
