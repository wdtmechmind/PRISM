# PRISM Tools Usage

本文件汇总 `tools/` 目录下各工具的用途、典型命令、输入输出约定。

建议从仓库根目录运行：`/mnt/projects-8tb/PRISM`

## 运行环境

- 常规 Python 工具：优先使用项目的相机/视觉环境，例如 `python3` 或你当前可用的 PRISM Python 环境。
- Isaac Sim 相关工具：使用 `/isaac-sim/python.sh` 运行。
- 依赖 Hik MVS SDK 的工具：需要已正确设置 `PRISM_MVIMPORT_DIR`，或本机已安装 `/opt/MVS`。

## 1. 数据采集与标注

### `annotate_led_dataset.py`

用途：打开 4 路 Hik 相机实时预览，冻结一组同步图像并手工标注 LED 框，输出 YOLO 标注数据。

典型命令：

```bash
python3 tools/annotate_led_dataset.py \
  --output-dir data/datasets/led_yolo \
  --camera-indices 0,1,2,3 \
  --auto-from-hsv y \
  --hsv-config configs/collection/default_online.yaml \
  --box-size-px 28
```

输入：4 路 Hik 相机实时画面。

输出：

- `data/datasets/led_yolo/images/cam*/set_xxxxxx.png`
- `data/datasets/led_yolo/labels/cam*/set_xxxxxx.txt`

交互按键：

- `c` / `Space`：冻结当前 4 相机帧
- `1..4`：选择颜色类别 red / yellow / blue / green
- 鼠标左键：添加框
- 鼠标右键：撤销当前相机最后一个框
- `s` / `Enter`：保存当前冻结帧与标签
- `n`：丢弃当前冻结帧
- `q` / `Esc`：退出

### `capture_led_photos.py`

用途：只采集 4 路同步图片，不做标注，适合后续离线标注。

典型命令：

```bash
python3 tools/capture_led_photos.py \
  --output-dir data/datasets/led_yolo_raw \
  --camera-indices 0,1,2,3
```

输出：

- `data/datasets/led_yolo_raw/LedPhoto_<timestamp>/images/<serial>/set_xxxxxx.png`
- `data/datasets/led_yolo_raw/LedPhoto_<timestamp>/meta.json`

按键：

- `Space` / `Enter`：保存一组同步图片
- `q` / `Esc`：退出

### `annotate_led_dataset_offline.py`

用途：对 `capture_led_photos.py` 采集到的同步图片集做离线标注。

典型命令：

```bash
python3 tools/annotate_led_dataset_offline.py \
  --dataset-root data/datasets/led_yolo_raw/LedPhoto_YYYYmmdd_HHMMSS \
  --auto-from-hsv y \
  --hsv-config configs/collection/default_online.yaml \
  --box-size-px 28
```

输入目录约定：

- `images/<serial>/set_000000.png`

输出目录约定：

- `labels/<serial>/set_000000.txt`

### `train_led_yolo.py`

用途：将 PRISM LED 标注数据转换为 Ultralytics YOLO 训练格式并启动训练。

典型命令：

```bash
python3 tools/train_led_yolo.py \
  --dataset-root data/datasets/led_yolo \
  --weights yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

输入：

- `images/cam*/set_xxxxxx.png`
- `labels/cam*/set_xxxxxx.txt`

输出：YOLO 训练数据目录与训练结果目录。

## 2. 在线/离线标定

### `prism_charuco_calibration_capture.py`

用途：采集 4 路硬触发同步的 ChArUco 标定图像，用于相机内参与多相机外参标定。

典型命令：

```bash
python3 tools/prism_charuco_calibration_capture.py \
  --output-dir ~/mvs_charuco_data \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --live y --detect-overlay y
```

输出：标定图像和配套元数据，供后续标定流程使用。

### `calibrate_hsv_led.py`

用途：单相机交互式 HSV 阈值调参工具，通过点击 LED 像素扩展颜色阈值范围。

典型命令：

```bash
python3 tools/calibrate_hsv_led.py \
  --config configs/collection/default_online.yaml \
  --output configs/collection/hsv_tuned.yaml
```

输出：保存调好的 YAML，或将 YAML 片段打印到终端。

交互按键：

- `1..4`：切换当前颜色
- 鼠标左键：采样并扩展当前颜色阈值
- 鼠标右键：撤销最近一次采样
- `c` / `r`：清空当前颜色调整，回退到初始配置
- `w`：保存到 `--output`
- `p`：打印 YAML 片段
- `q` / `Esc`：退出

### `calibrate_temporal_delay.py`

用途：已废弃。

说明：PRISM 当前使用硬件触发同步，不再支持旧的时间偏移标定流程。

## 3. 重建、评估与轨迹分析

### `eval_led_accuracy.py`

用途：评估 LED 三角化和刚体位姿轨迹的自洽精度。

适用场景：单个轨迹 CSV 的手动分析；更偏底层。

典型命令：

```bash
python3 tools/eval_led_accuracy.py \
  data/raw/task_xxx/trajectory_led_nearest.csv \
  --rigid data/raw/task_xxx/rigid_pose_6d.csv
```

可选：

- `--static-t0` / `--static-t1`：手动指定静止时间段
- `--output results.json`：导出评估结果

### `generate_trajectory_quality_reports.py`

用途：对单个 trial 或整个 task 目录批量生成轨迹质量评估报告，等价于之前离线后处理里自动打印的那一步，但现在改为按需手动执行。

典型命令：

```bash
/isaac-sim/python.sh tools/generate_trajectory_quality_reports.py \
  --trial-dir data/raw/task_20260730_103856_ball_picking/trial_000012 \
  --write-json
```

批量处理整个 task：

```bash
/isaac-sim/python.sh tools/generate_trajectory_quality_reports.py \
  --task-dir data/raw/task_20260730_103856_ball_picking \
  --write-json
```

默认输入：

- `trial_xxxxxx/trajectory/trajectory_led.csv`
- `trial_xxxxxx/trajectory/rigid_pose_6d.csv`

默认输出：

- `trial_xxxxxx/trajectory/accuracy_report.md`
- `trial_xxxxxx/trajectory/accuracy_report.json`（当传 `--write-json` 时）

常用参数：

- `--task-dir`：批量处理整个任务目录下所有 `trial_*`
- `--trial-dir`：只处理一个 trial
- `--write-json`：额外输出 JSON 报告
- `--no-md`：不输出 Markdown 报告
- `--static-t0` / `--static-t1`：手动指定静止时间窗

### `plot_corrected_trial_projections.py`

用途：输入单个 trial 目录，读取轨迹 CSV，并将轨迹映射到 corrected 坐标系后绘制 XY / XZ / YZ 平面投影。

典型命令：

```bash
/isaac-sim/python.sh tools/plot_corrected_trial_projections.py \
  --trial-dir data/raw/task_20260730_103856_ball_picking/trial_000012 \
  --calib-json configs/devices/charuco_4cam_result.json
```

默认输入：

- `trial_xxxxxx/trajectory/trajectory_led.csv`
- `trial_xxxxxx/trajectory/rigid_pose_6d.csv`

默认输出：

- `trial_xxxxxx/trajectory/corrected_projections.png`

常用参数：

- `--led-csv`：覆盖默认 LED 轨迹 CSV 路径
- `--rigid-csv`：覆盖默认刚体轨迹 CSV 路径
- `--output`：指定输出 PNG 路径
- `--use-raw`：优先使用原始列 `x_m/y_m/z_m`，不使用平滑列

## 4. Isaac Sim 仿真流水线

用途：将原始 trial 转换到 corrected 坐标系，生成 AUBO i5 + MechHand 的计划关节轨迹，并导出 replay 仿真视频。

典型命令：

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_YYYYmmdd_HHMMSS_task-name/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

常用参数：

- `--out-dir`：指定 corrected/planned/video 输出目录
- `--video-path`：指定 mp4 输出路径
- `--camera-pos 0 -0.2 0`：从对侧俯视相机录制
- `--max-frames` / `--replay-max-frames`：快速 smoke test
- `--dry-run`：只打印底层阶段命令

## 5. 使用建议

- 做真实采集与标定：优先看 `annotate_led_dataset.py`、`capture_led_photos.py`、`prism_charuco_calibration_capture.py`。
- 做离线几何分析：优先看 `eval_led_accuracy.py`、`track_apriltag_trajectory.py`、`plot_corrected_trial_projections.py`。
- 做仿真回放与生成：优先看 `simulation/scripts/run_replay_pipeline.py` 和 `simulation/README.md`。
- 若工具支持 `--trial-dir`，通常默认会从 `trial_xxxxxx/trajectory/` 和 `trial_xxxxxx/hand/` 自动推断输入文件。