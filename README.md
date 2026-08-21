# PRISM

---

## 0. Hardware Setup

见钉钉内文件

---

## 1. 环境配置

### 1.1 进入仓库

```bash
cd /mnt/projects-8tb/PRISM
```

### 1.2 选择 Python 环境

推荐环境是 `camera`，也可使用团队约定的其他环境。

```bash
# 可选：使用 conda/mamba 时
mamba activate camera
```

### 1.3 安装项目命令入口

```bash
pip install -e .
```

安装后可用命令：

1. `prism-collect`
2. `prism-reconstruct-trials`
3. `prism-analyze-trajectory`
4. `prism-rebuild-aligned`
5. `prism-hand`

### 1.4 配置 MVS SDK 运行环境（Hik 相机）

采集脚本 `scripts/collect_task.sh` 会自动执行：

1. `source /opt/MVS/bin/set_env_path.sh /opt/MVS`
2. 设置 `PRISM_MVIMPORT_DIR`
3. 设置 `PYTHONPATH`

若 SDK 路径不同，请先导出环境变量：

```bash
export MVS_ROOT=/your/mvs/path
export PRISM_PYTHON=/your/python/bin/python
export PRISM_MVIMPORT_DIR=/your/mvs/path/Samples/64/Python/MvImport
```

### 1.5 快速自检

```bash
python3 -m py_compile tools/run_ur3_replay.py tools/replay_ur3_base_trajectory.py tools/transform_cam_trajectory_to_base.py
```

---

## 2. 运行前配置检查（采集前）

主配置文件：`configs/collection/default_online.yaml`

重点先确认这些键：

1. `calib_json`：四相机标定文件路径。
2. `rs_calib_json`：RealSense 内参路径。
3. `detector_backend`：`hsv`、`yolo` 或 `hybrid`。
4. `yolo_weights`：YOLO 权重路径。
5. `hik_exposure_us`、`hik_gain`、`hik_frame_rate`。
6. `output_dir`：输出根目录（默认 `data/raw`）。
7. `post_process`：`ask`、`now`、`later`。

建议第一次先用默认配置，不要大改。

---

## 3. 数据采集（按顺序执行）

## 3.1 启动在线采集

```bash
cd /mnt/projects-8tb/PRISM

scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 3 \
  --output-dir data/raw
```

## 3.2 采集过程按键

1. `Space`：开始/停止当前 trial 录制。
2. `p` / `r`：暂停/恢复轨迹更新。
3. `1..17`：发送手势命令。
4. `q` 或 `Esc`：结束任务。

## 3.3 采集结束后确认产物

检查目录：

```bash
ls -lah data/raw | tail
```

进入最新 task 目录，确认至少有：

1. `task_metadata.yaml`
2. `trial_xxxxxx/cameras/*.mp4`
3. `trial_xxxxxx/cameras/*_timestamps.csv`

---

## 4. 离线重建（采集后立刻做）

假设任务目录是：

`data/raw/task_YYYYmmdd_HHMMSS_grasp-demo`

执行：

```bash
prism-reconstruct-trials data/raw/task_YYYYmmdd_HHMMSS_grasp-demo
```

如果要强制指定 detector（覆盖任务继承）：

```bash
prism-reconstruct-trials data/raw/task_YYYYmmdd_HHMMSS_grasp-demo \
  --detector-backend yolo \
  --yolo-weights /path/to/best.pt \
  --yolo-conf 0.5 \
  --yolo-imgsz 640
```

检查每个 trial 的 `trajectory` 目录，确认有：

1. `trajectory_led.csv`
2. `rigid_pose_6d.csv`
3. `rigid_6d_frames.png`

---

## 5. 轨迹分析与质量报告

## 5.1 快速轨迹分析

```bash
scripts/analyze_trajectory.sh data/raw/task_YYYYmmdd_HHMMSS_grasp-demo
```

或：

```bash
prism-analyze-trajectory data/raw/task_YYYYmmdd_HHMMSS_grasp-demo --output-dir ./traj_plots
```

## 5.2 生成质量报告

整任务批量：

```bash
python3 tools/generate_trajectory_quality_reports.py \
  --task-dir data/raw/task_YYYYmmdd_HHMMSS_grasp-demo \
  --write-json
```

单 trial：

```bash
python3 tools/generate_trajectory_quality_reports.py \
  --trial-dir data/raw/task_YYYYmmdd_HHMMSS_grasp-demo/trial_000001 \
  --write-json
```

---

## 6. 仿真环境搭建与初始化（第一次做）

这一段用于 Isaac Sim 侧资产准备。默认使用 `/isaac-sim/python.sh`。

## 6.1 构建组合 URDF（AUBO + MechHand）

```bash
cd /mnt/projects-8tb/PRISM
python3 simulation/scripts/build_combined_urdf.py
```

默认输出：

`simulation/assets/aubo_i5_mechhand/aubo_i5_mechhand.urdf`

## 6.2 生成 USD 资产

```bash
/isaac-sim/python.sh simulation/scripts/build_robot_usd.py --headless --no-preview
```

默认输出：

`simulation/assets/aubo_i5_mechhand/aubo_i5_mechhand.usd`

## 6.3 可选：重刷材质

```bash
/isaac-sim/python.sh simulation/scripts/apply_robot_materials.py
```

---

## 7. 仿真 replay 轨迹步骤

可选择两条路径。

## 7.1 路径 A：用真实采集数据做仿真 replay

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_YYYYmmdd_HHMMSS_grasp-demo/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

输出通常在：

`data/processed/simulation/<task>/<trial>/`

核心文件：

1. `corrected_trajectory.csv`
2. `planned_motion.csv`
3. `planned_motion_overhead.mp4`

## 7.2 路径 B：先生成纯仿真 trial，再 replay

先生成仿真 trial：

```bash
python3 simulation/scripts/collect_sim_trial.py \
  --task-name sim_collect \
  --num-trials 1 \
  --duration-sec 10
```

再用该 trial 跑 replay pipeline：

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_YYYYmmdd_HHMMSS_sim_collect/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

---

## 8. 真机执行前准备（UR3）

在执行真实运动前，按顺序确认：

1. 已有离线重建产物 `trial_xxxxxx/trajectory/rigid_pose_6d.csv`。
2. 手眼文件存在：`configs/deployment/robot_camera_handeye.json`。
3. wrist 标定存在：`configs/deployment/ur3_wrist_mount.json`。
4. 机器人网络连通（默认 IP `192.168.1.102`）。
5. `localhost:60686` 手部服务可用，或计划 `--skip-hand`。
6. 工作区清空、急停可用、旁站监护到位。

---

## 9. 真机执行命令（必须按顺序）

假设 trial 为：

`data/raw/task_20260820_155002_grasp-demo/trial_000002`

## 9.1 Step 1: Dry-run（不连机器人，不运动）

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002
```

## 9.2 Step 2: Preflight（连机器人，不运动）

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --preflight
```

## 9.3 Step 3: Execute（真实执行）

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --execute
```

## 9.4 常用覆盖参数

```bash
--robot-ip 192.168.1.102
--base-offset 0 0 -0.05
--mount-correction-axis right
--mount-correction-deg -45
--time-scale 15
--skip-hand
--no-wrist-calibration
--hand-ip localhost
--hand-port 60686
```

---

## 10. 怎么改 config 参数

PRISM 的参数覆盖优先级是：

1. 命令行参数（最高优先级）
2. 配置文件参数
3. 代码默认值

## 10.1 在线采集参数（最常改）

编辑：`configs/collection/default_online.yaml`

常改项：

1. 相机采集：`hik_exposure_us`、`hik_gain`、`hik_frame_rate`
2. 检测后端：`detector_backend`
3. YOLO：`yolo_weights`、`yolo_conf`、`yolo_iou`、`yolo_imgsz`
4. 输出：`output_dir`
5. 后处理策略：`post_process`
6. RealSense：`rs_width`、`rs_height`、`rs_fps`、`rs_auto_exposure`

示例：把 YOLO 置信度改为 0.6。

```yaml
detector_backend: yolo
yolo_conf: 0.6
```

## 10.2 不改文件，直接命令行覆盖

```bash
scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --detector-backend yolo \
  --yolo-conf 0.6
```

## 10.3 UR3 部署参数

相关文件：

1. `configs/deployment/robot_camera_handeye.json`
2. `configs/deployment/ur3_wrist_mount.json`

建议：

1. 手眼矩阵只在重标定后更新。
2. wrist 偏差只在安装关系变化后更新。
3. 常规试验优先用命令行改 `--base-offset` 和 `--mount-correction-*`，不要频繁改标定文件。

## 10.4 仿真参数

主配置：`simulation/configs/aubo_i5_mechhand.yaml`

常改项：

1. `world_from_base`：机器人在场景中的放置。
2. `planning.use_orientation`：是否强制考虑姿态。
3. `planning.orientation_weight`、`planning.max_iters`、`planning.tolerance`。
4. `default_arm_q`、`default_hand_pose`。

示例：放宽姿态约束。

```yaml
planning:
  use_orientation: true
  orientation_weight: 0.2
  max_iters: 100
  tolerance: 0.008
```

## 10.5 RPi/手部遥操作参数

文件：`configs/deployment/mechhand_v3_teleop.yaml`

常改项：

1. `rpi_udp.host`、`rpi_udp.port`
2. `control.command_rate_hz`
3. `control.input_mode`（`absolute` 或 `incremental`）
4. `mapping` 下每个关节的 `source_min/source_max/target_min_deg/target_max_deg`

---

## 11. 推荐的第一次完整跑通顺序

按下面顺序做，最稳：

1. 完成第 0 章硬件准备（团队本地 SOP）。
2. 完成第 1 章环境配置。
3. 按第 3 章做一次 1 到 3 个 trial 的小采集。
4. 按第 4 章离线重建。
5. 按第 5 章生成分析图和质量报告。
6. 按第 6 到第 7 章跑一遍仿真 replay。
7. 按第 8 到第 9 章执行真机 dry-run -> preflight -> execute。

---

## 12. 常见问题的最小排查顺序

1. 命令是否在仓库根目录执行。
2. 配置路径是否是绝对路径或相对仓库根目录的正确路径。
3. `trial_xxxxxx/trajectory/rigid_pose_6d.csv` 是否存在。
4. 手眼与 wrist 标定文件是否存在且可读。
5. 真机前是否已经通过 preflight。
