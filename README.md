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

运行时按键：

- `Space`: 开始/停止当前 trial 录制
- `p`: 暂停轨迹更新
- `r`: 恢复轨迹更新
- `q` 或 `ESC`: 结束采集任务

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
- OpenCV 预览窗口 overlay 显示 3D 点状态、同步误差、rigid6d xyz/rpy
- Matplotlib 3D 窗口显示四个 LED 点轨迹、dex hand 刚体中心轨迹和刚体坐标轴

单个 LED 轨迹重建的必要条件：该 LED 至少被两个 Hik 相机同时检测到。dex hand 6D pose 的必要条件：当前帧至少有 3 个已经进入刚体模型的 LED 被成功三角化；不要求每个 LED 被每台相机看到，也不要求四个 LED 每帧都同时可见。刚体模型会先用第一帧可用的非共线 3 个 LED 初始化，后续如果第四个 LED 出现，会自动加入模型。若条件不足，预览会显示 `rigid6d: need >=3 modeled LEDs`。

## 4. 默认配置

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
- `--viz-3d`: 是否打开 Matplotlib 3D 轨迹窗口
- `--rigid-axis-len`: 3D 窗口里刚体坐标轴长度，单位米

注意：这里的 `red`、`yellow`、`blue`、`green` 是刚体上 LED 的身份标签，不一定等于 Hik 图像里肉眼看到的颜色。当前 Hik 画面里 red LED 偏橙/黄，因此默认 `r_h_low/r_h_high` 使用 orange/amber 区间，`yellow` 区间相应后移以减少重叠。若更换 LED、曝光或白平衡，应重新调 `configs/collection/default_online.yaml` 里的 HSV 阈值。

## 5. 输出结构

一次任务会生成类似结构：

```text
data/raw/
  task_YYYYmmdd_HHMMSS_task-name/
    task_metadata.yaml
  trajectory_led_nearest.csv
  trajectory_led_interp.csv
    rigid_pose_6d.csv
    time_alignment_log.csv
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

注意：`hand/` 里的三个 CSV 当前是占位文件，真实 RPi/SDK/反馈记录还未实现。

## 6. 离线严格对齐重建

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

## 7. 仍未实现的边界

以下部分还不是完整功能：

- RPi 串口命令读取
- 灵巧手 SDK 命令转发
- Gen2/Gen3 hand feedback 实时记录
- 在线结束后的自动后处理入口
- 硬件触发/软件触发配置入口

当前采集模式是 Hik 自由运行 + 时间戳对齐。若需要硬件触发同步，需要继续在 PRISM native 的 MVS adapter 和 session 参数中实现。

## 8. 开发与验证

常用无硬件验证：

```bash
cd /mnt/projects-8tb/PRISM

scripts/collect_task.sh --help
scripts/rebuild_aligned_segment.sh --help

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:/opt/MVS/Samples/64/Python/MvImport \
/home/daotan/miniforge3/envs/camera/bin/python -m compileall -q src/prism

find src -type d -name __pycache__ -prune -exec rm -rf {} +
```
