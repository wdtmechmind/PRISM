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

摄像头打开后会自动进入时间偏差标定流程（见第 5 节）；完成后再进入正常采集循环。

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

- 4 路 Hik USB 相机连续采集
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
- **多相机时间偏差标定**：摄像头打开后自动进行，屏幕闪 AprilTag，检测各相机首次捕获的时刻，计算并保存各相机相对时间偏移

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

## 5. 多相机时间偏差标定

由于 4 路 Hik 相机是**自由运行**（不触发），每次上电相位随机，各相机 `capture_wall_time` 之间可能有 0 ~ 1/fps 的偏差（30fps 下最多约 33ms）。

标定在每次 `prism-collect` 启动后、第一个 trial 录制前**自动运行**：

```
摄像头打开（随机相位启动）
  ↓
终端提示：Aim ALL cameras at this screen, then press Enter
  ↓  ← 把所有摄像头镜头朝向屏幕，按回车
倒计时 3s
  ↓
AprilTag 全屏显示 1.5s（white → tag → black）
  ↓
检测每路视频中 tag 首次出现的时刻，计算偏移
  ↓
保存 camera_delay_calib.json，进入正常采集循环
```

**标定结果文件格式**：

```json
{
  "reference_serial": "00DA001",
  "offsets_s": {
    "00DA001":  0.0,
    "00DA002":  0.018,
    "00DA003": -0.011,
    "00DA004":  0.027,
    "234522...": 0.033
  },
  "note": "aligned_time = capture_wall_time - offsets_s[serial]"
}
```

后处理时对每路相机的时间戳做：`aligned_time = capture_wall_time - offsets_s[serial]`。

**标定相关 CLI 参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--skip-temporal-calib` | off | 跳过标定，直接进入采集 |
| `--temporal-calib-tag-id` | 0 | 使用的 AprilTag ID（tag36h11 family）|
| `--temporal-calib-tag-delay-s` | 3.0 | 按下回车后到 tag 出现的秒数 |
| `--temporal-calib-tag-display-s` | 1.5 | tag 在屏幕上的显示时长 |

**精度**：±1 帧（30fps 下约 ±16ms，300fps 下约 ±1.7ms）。

**如果某个摄像头没检测到 tag**：该摄像头的偏移不写入 JSON，其他摄像头不受影响；采集正常继续。

**独立标定工具**（不录制，仅测量偏差）：

```bash
/home/daotan/miniforge3/envs/camera/bin/python \
  tools/calibrate_temporal_delay.py \
  --hik-serials 00DA001 00DA002 00DA003 00DA004 \
  --rs-serial 234522070717 \
  --fps 30 --exposure-us 8000 \
  --output configs/devices/camera_delay_calib.json
```

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
    camera_delay_calib.json          ← 时间偏差标定结果（每 session 生成一次）
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
- 硬件触发/软件触发配置入口
- 标定偏移自动写入 `_timestamps.csv`（目前需后处理手动应用）

当前采集模式是 Hik 自由运行 + 时间戳对齐 + 每 session AprilTag 时间偏差标定。若需要硬件触发同步，需要继续在 PRISM native 的 MVS adapter 和 session 参数中实现。

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

## 10. 开发与验证

常用无硬件验证：
