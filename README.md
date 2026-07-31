# PRISM 多相机采集与重建

PRISM 用于 4 路 Hik 相机 + 1 路 RealSense D435 的同步采集、LED 三角化重建与 dex hand 刚体 6D 位姿估计。

当前主链路：

- 在线采集（硬件触发同步）
- 在线可视化与轨迹监控
- 采集后离线 per-trial 重建
- 轨迹图与诊断图生成
- 可选 YOLO 检测替代 HSV

## 1. 依赖边界

PRISM 代码在本仓库；厂商 SDK 仍为外部依赖：

- MVS SDK: /opt/MVS
- MVS Python binding: /opt/MVS/Samples/64/Python/MvImport/MvCameraControl_class.py

建议 Python 环境：

- /home/daotan/miniforge3/envs/camera

## 2. 快速开始

在线采集：

```bash
cd /mnt/projects-8tb/PRISM

scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 20 \
  --output-dir data/raw
```

常用按键：

- Space: 开始/停止当前 trial 录制
- p / r: 暂停 / 恢复轨迹更新
- 1..17: 发送手势编号
- q 或 ESC: 结束任务

## 3. 检测后端

支持三种检测后端：

- hsv: 仅 HSV
- yolo: 仅 YOLO
- hybrid: YOLO 优先，HSV 补缺

YOLO 参数（CLI）：

- --detector-backend yolo|hybrid
- --yolo-weights 权重路径
- --yolo-conf 置信度阈值（越高越保守）
- --yolo-iou NMS IoU 阈值
- --yolo-imgsz 推理尺寸

示例：

```bash
scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --detector-backend yolo \
  --yolo-weights /home/daotan/runs/detect/runs/detect/prism_led_yolo/weights/best.pt \
  --yolo-conf 0.25 \
  --yolo-imgsz 960
```

## 4. 后处理与 YOLO 继承规则

当采集命令使用 YOLO 或 hybrid 时，后处理会继承同一 detector 配置，按同一条链路生成结果：

- yolo -> triangulation/extrinsics -> 6D pose

配置来源优先级：

1. 离线重建命令行参数
2. task_metadata.yaml 中 tracking_detector
3. 配置文件默认值

因此：

- 在线采集后立刻后处理，默认会沿用当次 YOLO 参数
- 对历史任务重跑，可显式传 detector 参数覆盖

手动重跑离线重建：

```bash
prism-reconstruct-trials data/raw/task_YYYYmmdd_HHMMSS_task-name \
  --detector-backend yolo \
  --yolo-weights /home/daotan/runs/detect/runs/detect/prism_led_yolo/weights/best.pt \
  --yolo-conf 0.25 \
  --yolo-imgsz 960
```

## 5. 硬件触发与标定

当前使用硬件触发同步：

- Master: DA8165486（Line1 输出）
- Slave: 其余相机（Line0 输入）
- 目标同步精度: 帧级（< 1 ms）

首次部署或更换相机后，需要重新做 ChArUco 标定。详见：

- docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md
- docs/QUICK_START_HARDWARE_TRIGGER.md

## 6. 数据集标注与 YOLO 训练

在线同步标注：

```bash
python3 tools/annotate_led_dataset.py \
  --output-dir data/datasets/led_yolo \
  --camera-indices 0,1,2,3 \
  --auto-from-hsv y \
  --hsv-config configs/collection/default_online.yaml \
  --box-size-px 28
```

离线标注（先采集后标注）：

```bash
python3 tools/capture_led_photos.py \
  --output-dir data/datasets/led_yolo_raw \
  --camera-indices 0,1,2,3

python3 tools/annotate_led_dataset_offline.py \
  --dataset-root data/datasets/led_yolo_raw \
  --auto-from-hsv y \
  --hsv-config configs/collection/default_online.yaml \
  --box-size-px 28
```

训练：

```bash
python3 tools/train_led_yolo.py \
  --dataset-root data/datasets/led_yolo \
  --weights yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

## 7. 输出结构（关键文件）

任务级目录：

- task_metadata.yaml
- trajectory_led_nearest.csv
- trajectory_led_interp.csv
- rigid_pose_6d.csv
- time_alignment_log.csv

每个 trial 后处理目录：

- trial_xxxxxx/trajectory/trajectory_led.csv
- trial_xxxxxx/trajectory/rigid_pose_6d.csv
- trial_xxxxxx/trajectory/rigid_6d_frames.png

说明：

- 在线轨迹用于实时监控
- trial_xxxxxx/trajectory 下的是离线重建正式产物

## 8. 轨迹分析

采集后自动后处理：

```bash
scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 5 \
  --output-dir data/raw \
  --post-process now
```

手动分析：

```bash
scripts/analyze_trajectory.sh data/raw/task_YYYYmmdd_HHMMSS_task-name
# 或
prism-analyze-trajectory data/raw/task_YYYYmmdd_HHMMSS_task-name --output-dir ./traj_plots
```

## 9. Isaac Sim 仿真流水线

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_YYYYmmdd_HHMMSS_task-name/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

更多参数见：

- simulation/README.md

## 10. 已知边界

以下能力尚未完整实现：

- RPi 串口命令读取
- 灵巧手 SDK 转发链路的完整闭环
- 实时 hand feedback 全链路记录

## 11. 常用命令速查

```bash
# 安装本项目 CLI
pip install -e .

# 离线 per-trial 三维重建
prism-reconstruct-trials data/raw/task_YYYYmmdd_HHMMSS_task-name

# 轨迹分析
prism-analyze-trajectory data/raw/task_YYYYmmdd_HHMMSS_task-name
```
