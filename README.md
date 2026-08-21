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

使用 `--detector-backend yolo` 采集时，四路 Hik 实时窗口会在每个被采用的预测中心显示十字标记、颜色类别、confidence 和像素坐标；HSV 与 hybrid 后端不显示该 YOLO 详情叠加。

示例：

```bash
scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --detector-backend yolo \
  --yolo-weights /home/daotan/runs/detect/runs/detect/prism_led_yolo/weights/best.pt \
  --yolo-conf 0.5 \
  --yolo-imgsz 960
```

## 4. 后处理与 YOLO 继承规则

当采集命令使用 YOLO 或 hybrid 时，后处理会继承同一 detector 配置，按同一条链路生成结果：

- yolo -> triangulation/extrinsics -> 6D pose

三角化对两视角和多视角结果统一执行重投影误差门控。6D 刚体拟合对 LED 几何残差执行 `5 mm RMS` 门控；四灯中有一个离群点时尝试用一致的三灯子集恢复，三灯形状本身不一致时拒绝该帧。`rigid_pose_6d.csv` 始终保存相机观测到的 `led_base_link` 绝对位姿，不包含机械臂安装角补偿。

LED 局部语义轴与原点定义：

```text
forward = red -> blue
right   = yellow -> green（先对 forward 正交化）
up      = right x forward
origin  = blue + 5 mm * forward + 0 mm * right - 30 mm * up
```

`origin_up_m=-0.03` 表示沿局部 `up` 的反方向移动 30 mm，即 `down 30 mm`。修改 LED 原点后必须重新生成 `rigid_pose_6d.csv` 和 UR3 bundle；旧 IK cache 会因轨迹指纹变化失效，不应复用。

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
  --yolo-conf 0.5 \
  --yolo-imgsz 960
```

对已录制 trial 单独调节 green 的置信度（其余三色固定为 `--conf`）：

```bash
python3 tools/visualize_yolo_conf.py \
  --image-root data/raw/task_20260819_172319_replay-demo/trial_000001/cameras \
  --yolo-weights /home/daotan/runs/detect/runs/detect/prism_led_yolo/weights/best.pt \
  --conf 0.5 \
  --green-conf 0.5 \
  --start-index 600 \
  --frame-step 30 \
  --imgsz 960 \
  --device 0
```

窗口中的 `green conf x0.01` 滑条只过滤 green；`A/D` 或左右方向键切换视频帧，`S` 保存当前对比图。

## 5. 硬件触发与标定

当前使用硬件触发同步：

- Master: DA8165486（Line1 输出）
- Slave: 其余相机（Line0 输入）
- 目标同步精度: 帧级（< 1 ms）

首次部署或更换相机后，需要重新做相机 ChArUco 标定。标定结果和用途如下：

| 标定项目 | 结果文件 | 用途 |
|---|---|---|
| 四路 HIK 相机内参 + 相机间外参 | `configs/devices/charuco_4cam_result.json` | 去畸变、多视角三角化，以及把 `cam1..cam3` 统一到 `cam0` |
| RealSense D435 内参 | `configs/devices/d435_charuco_intrinsics.json` | RealSense 彩色图像去畸变和内参读取 |
| UR3 + 相机手眼标定 | `configs/deployment/robot_camera_handeye.json` | 把 `cam0` 坐标系中的轨迹转换到 UR3 `robot_base`，供 UR3 replay 使用 |
| UR3 `wrist_3` 安装角标定 | `configs/deployment/ur3_wrist_mount.json` | 修正末端安装角，不属于相机内外参或手眼标定 |

这里要区分两种“外参”：`charuco_4cam_result.json` 保存的是相机之间的几何关系；`robot_camera_handeye.json` 保存的是机器人 Base、相机和末端标定板之间的关系。UR3 replay 的 `--handeye` 使用后者。

完整流程（包括采图、四相机内外参、UR3 联合手眼标定）见：

- [硬件触发与相机/手眼标定指南](docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md)
- [硬件触发快速开始](docs/QUICK_START_HARDWARE_TRIGGER.md)

如果要使用 UR3 手眼标定时使用的 CGB-035 非对称圆点板来标定四路相机，可直接运行：

```bash
python3 tools/capture_circle_grid_calibration.py \
  --output-dir ~/mvs_circle_grid_data \
  --exposure-us 12000 --gain 0 --frame-rate 15
```

窗口会显示四路相机画面及每路 `VALID/INVALID` 检测状态；当至少 3 路识别成功时按 Enter 或 Space 保存一组同步图像，按 `Q` 结束。采集完成后运行：

```bash
python3 tools/calibrate_4cam_circle_grid.py \
  --image-root ~/mvs_circle_grid_data/CircleGridCapture_YYYYMMDD_HHMMSS \
  --output configs/devices/circle_grid_4cam_result.json
```

建议采集 25-50 组，覆盖画面中心、四角、近中远距离以及不同俯仰/偏航/滚转。该流程输出 CGB-035 的四相机内参和 `cam0_to_cam1..3` 外参；若要重新生成 UR3 手眼标定，还需按指南第 11 节同时记录每帧的 UR3 TCP 位姿。

UR3 手眼标定的详细章节是指南第 11 节；四相机 ChArUco 内外参标定是第 4 节。标定结果属于部署配置，重新标定后应检查结果文件中的采集目录、有效帧数和重投影/拟合误差，并重新生成依赖它的 replay bundle。

单独查看 RealSense 实时彩色画面：

```bash
python3 tools/view_realsense_live.py
```

默认使用 `1280x720@30 FPS`、曝光 `260`、增益 `64`，并读取 `configs/devices/d435_charuco_intrinsics.json` 去畸变。按 `S` 保存当前帧，`F` 切换全屏，`Q/Esc` 退出。使用自动曝光或关闭去畸变：

```bash
python3 tools/view_realsense_live.py --auto-exposure --no-undistort
```

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

若没有真实相机，也可先生成“接近真实格式”的仿真采集数据：

```bash
python3 simulation/scripts/collect_sim_trial.py --task-name sim_collect --num-trials 3 --duration-sec 10
```

这会在 `data/raw/task_*/trial_*/trajectory/` 下生成 `trajectory_led.csv` 和
`rigid_pose_6d.csv`，并同步生成 `hand/sdk_commands.csv`、`hand/rpi_commands.csv`。

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_YYYYmmdd_HHMMSS_task-name/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

更多参数见：

- simulation/README.md

## 10. UR3 CB3 真机回放

真机链路分为两个独立阶段：

1. 相机重建生成 `led_base_link` 的原始绝对位姿，不包含机械安装角。
2. 生成 UR3 bundle 时执行手眼变换、Base 平移和机械安装角补偿。

不要在两个阶段重复应用安装补偿。

### 10.0 当前 trial 最简命令

使用统一入口 `tools/run_ur3_replay.py`。只需要给原始 trial 路径；脚本会自动：

- 读取 `trajectory/rigid_pose_6d.csv` 和 `hand/sdk_commands.csv`；
- 应用手眼矩阵、zero Base offset、`right -45°` 安装补偿；
- 使用部署级 wrist 标定 `configs/deployment/ur3_wrist_mount.json`；
- 将 bundle 写入 `data/replay_bundles/ur3/<task>/<trial>/`；
- 加载或生成 IK cache；
- 执行净空、关节速度和 moveJ 路径检查。

进入仓库：

```bash
cd /mnt/projects-8tb/PRISM
```

1. 本地 dry-run：自动生成 bundle，不连接机器人，不运动：

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002
```

2. 连接 UR3，生成/检查 IK cache，但不运动：

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --preflight
```

3. preflight 通过后，真机回放 UR3，并同步向 `localhost:60686` 发送 SDK 指令：

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --execute
```

当前 bundle 目录为：

```text
data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002/
```

其中包含 `base_trajectory.csv`、`sdk_commands.csv`、`manifest.json`，preflight 后还会生成 `ik_plan_cache_wrist.npz`。部署级 wrist_3 标定保存在 `configs/deployment/ur3_wrist_mount.json`，同一套 UR3 与机械手安装可跨 trial 复用。

常用覆盖参数可直接附加在统一命令后，例如：

```text
--robot-ip 192.168.1.102
--time-scale 15
--max-joint-speed 1.2
--base-offset 0 0 0
--skip-hand
--replan-ik
```

查看底层两条命令但不执行：

```bash
python3 tools/run_ur3_replay.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --preflight \
  --print-only
```

若出现 `One of the RTDE input registers are already in use`，说明 UR 控制器的 RTDE 输入寄存器被其他客户端或 EtherNet/IP、PROFINET、MODBUS 占用。先停止其他 RTDE 程序，并在 PolyScope 中禁用占用寄存器的现场总线配置，再重新运行第 2 步；错误发生在运动前。

### 10.1 重建纯 LED 位姿

若 trial 已经完成最新离线重建，可跳过此步。否则运行：

```bash
prism-reconstruct-trials \
  data/raw/task_20260820_155002_grasp-demo \
  --detector-backend yolo \
  --yolo-weights /home/daotan/runs/detect/runs/detect/prism_led_yolo/weights/best.pt \
  --yolo-conf 0.5 \
  --yolo-imgsz 640
```

正式输入文件为：

```text
data/raw/task_20260820_155002_grasp-demo/trial_000002/trajectory/rigid_pose_6d.csv
```

该文件中的 XYZ/RPY 表示相机观测到的 `led_base_link`，不包含 `-45°` 真机安装补偿。

### 10.2 生成 UR3 Base bundle

以下命令显式指定：

- 使用平滑位姿；
- 至少三灯参与刚体估计；
- 不额外平移 UR3 Base 轨迹（`0,0,0`）；
- 只在此阶段绕 LED 语义 `right` 轴补偿 `-45°`。

```bash
python3 tools/transform_cam_trajectory_to_base.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --handeye configs/deployment/robot_camera_handeye.json \
  --output-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --use-smoothed \
  --min-num-leds 3 \
  --base-offset 0 0 0 \
  --mount-correction-axis right \
  --mount-correction-deg -45
```

输出目录包含：

```text
base_trajectory.csv  # t_sec,x,y,z,rx,ry,rz；UR Base 坐标，姿态为 UR rotation vector
sdk_commands.csv     # 原 trial 的手部命令时间线
manifest.json        # 手眼矩阵、Base 偏移、安装补偿及来源记录
```

检查 manifest：

```bash
python3 -m json.tool \
  data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002/manifest.json
```

应看到：

```json
"source_pose_frame": "led_base_link_as_captured",
"mount_correction": {
  "applied_during_rigid_reconstruction": false,
  "axis_in_semantic_frame": "right",
  "angle_deg": -45.0,
  "applied_during_base_conversion": true
}
```

`--base-offset X Y Z` 的单位是米，只改变整段轨迹在 UR3 Base 中的位置，不改变轨迹形状。不同偏移或补偿方向应使用不同的 `--output-dir`。

生成无安装补偿的诊断 bundle：

```bash
python3 tools/transform_cam_trajectory_to_base.py \
  --trial-dir data/raw/task_20260820_155002_grasp-demo/trial_000002 \
  --handeye configs/deployment/robot_camera_handeye.json \
  --output-dir replay_pose_ur3_trial_000002_no_mount \
  --use-smoothed \
  --min-num-leds 3 \
  --base-offset 0 0 0 \
  --no-mount-correction
```

### 10.3 本地 dry-run：不连接机器人

不加 `--execute` 或 `--ik-preflight-only` 时，只读取本地 CSV，检查轨迹步长和速度：

```bash
python3 tools/replay_ur3_base_trajectory.py \
  --bundle-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --robot-ip 192.168.1.102 \
  --time-scale 15 \
  --rate-hz 125 \
  --movej-speed 0.05 \
  --movej-acceleration 0.10 \
  --skip-hand
```

该命令不会连接 UR3，也不会发送手部命令。当前数据预期输出约为：

```text
2749 source -> 25972 servo frames
scaled duration: 207.760 s
max linear step/speed: 0.7 mm / 0.082 m/s
max angular step/speed: 0.41 deg / 51.07 deg/s
DRY RUN: no robot motion and no SDK command sent
```

### 10.4 IK preflight：连接但不运动

本地检查通过后，连接 UR3 并抽样检查整段 IK：

```bash
python3 tools/replay_ur3_base_trajectory.py \
  --bundle-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --robot-ip 192.168.1.102 \
  --time-scale 15 \
  --rate-hz 125 \
  --min-link-clearance 0.01 \
  --link-radius 0.04 \
  --max-source-joint-step-deg 30 \
  --max-joint-speed 1.2 \
  --skip-hand \
  --ik-preflight-only
```

该模式会建立 RTDE Control/Receive 连接并启动 ur_rtde 控制脚本，但在 `moveJ` 和 `servoJ` 之前退出，不会移动机器人。成功标志：

```text
IK/clearance preflight passed; no robot motion has occurred yet.
IK PREFLIGHT ONLY: exiting before moveJ and servoJ.
```

若出现 `Failed to start control script`，检查是否存在其他 RTDE 客户端、机器人是否处于 `RUNNING/NORMAL`、PolyScope 程序是否停止，然后重新连接。

### 10.5 只回放 UR3

确认以下条件后才执行：

- 起始 TCP 和整段工作空间无碰撞；
- `IK preflight passed`；
- 急停可用，人员离开机器人工作区；
- 当前 PolyScope TCP 与轨迹控制参考点一致。

```bash
python3 tools/replay_ur3_base_trajectory.py \
  --bundle-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --robot-ip 192.168.1.102 \
  --time-scale 15 \
  --rate-hz 125 \
  --movej-speed 0.05 \
  --movej-acceleration 0.10 \
  --min-link-clearance 0.01 \
  --link-radius 0.04 \
  --max-joint-speed 1.2 \
  --hand-ip localhost \
  --hand-port 60686 \
  --execute
```

执行顺序为：

1. 用多个 shoulder/elbow/wrist seed 求不同 IK 分支；
2. 用 UR3 CB3 nominal DH 计算各移动连杆相对 Base `z=0` 的最低高度；
3. 扣除 `--link-radius` 后，选择整段最小净空最大的连续分支；
4. 检查当前关节到起始关节的 `moveJ` 插值路径净空；
5. `moveJ` 到选定分支的第一帧并验证完整 6D TCP；
6. 将选中的关节轨迹插值到 125 Hz，以 `servoJ` 回放。

使用 `servoJ` 是必要的：若继续使用 `servoL`，控制器可能在内部重新选择另一套 IK 分支，无法保证沿预检选中的高净空分支运动。

当前 trial 按新的快速配置 `--time-scale 15 --max-joint-speed 1.2` 预期的 preflight 结果：

```text
IK branch minimum clearances: 73.6 mm, 25.2 mm, -92.0 mm
selected IK branch clearance: 73.6 mm
planned max joint speed: ~0.840 rad/s (wrist_3 at source t=6.631 s)
initial moveJ path clearance: 111.9 mm
IK/clearance preflight passed; no robot motion has occurred yet.
```

### 10.6 IK 规划缓存

`--ik-preflight-only` 首次规划成功后会保存：

```text
使用 wrist 标定：<bundle-dir>/ik_plan_cache_wrist.npz
不使用 wrist 标定：<bundle-dir>/ik_plan_cache.npz
```

缓存包含源帧关节轨迹、`base_trajectory.csv` 的 SHA-256、机器人 IP、规划版本、候选分支净空和规划参数。再次运行 preflight 或 `--execute` 时会自动加载，输出：

```text
IK cache seeded-check max joint error: 0.000000 rad
loaded IK cache: .../ik_plan_cache_wrist.npz
```

加载缓存时仍会：

- 抽样调用 UR 控制器 IK，确认目标位姿与缓存关节解一致；
- 按当前参数重新检查关节连续性、连杆净空和关节速度；
- 根据机器人当前关节重新检查到起点的 `moveJ` 路径净空。

因此缓存只省略整段多分支 IK 搜索，不绕过安全门控。`base_trajectory.csv` 内容、机器人 IP 或缓存版本变化时缓存自动失效。

强制重新规划并覆盖缓存：

```bash
python3 tools/replay_ur3_base_trajectory.py \
  --bundle-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --robot-ip 192.168.1.102 \
  --time-scale 15 \
  --rate-hz 125 \
  --min-link-clearance 0.01 \
  --link-radius 0.04 \
  --max-source-joint-step-deg 30 \
  --max-joint-speed 1.2 \
  --skip-hand \
  --ik-preflight-only \
  --replan-ik
```

使用其他缓存路径：`--ik-cache /path/to/plan.npz`。

### 10.7 标定 wrist_3 安装角偏差

标定工具复用已保存的 `ik_plan_cache.npz`。它先回放缓存关节轨迹，你在合适时刻输入 `p` 回车暂停；暂停后只允许低速调整最后一个关节 `wrist_3`，前五个关节保持不变。

先 dry-run：

```bash
python3 tools/calibrate_ur3_wrist_mount.py \
  --bundle-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --robot-ip 192.168.1.102 \
  --time-scale 35 \
  --rate-hz 125
```

确认缓存和参数后执行：

```bash
python3 tools/calibrate_ur3_wrist_mount.py \
  --bundle-dir data/replay_bundles/ur3/task_20260820_155002_grasp-demo/trial_000002 \
  --robot-ip 192.168.1.102 \
  --time-scale 35 \
  --rate-hz 125 \
  --jog-step-deg 1 \
  --jog-speed 0.03 \
  --jog-acceleration 0.08 \
  --execute
```

回放过程中：

```text
p + Enter       暂停
```

暂停后可输入：

```text
+               wrist_3 正向 1°
-               wrist_3 反向 1°
+ 0.2           正向 0.2°
- 0.2           反向 0.2°
set -3.5        相对计划 wrist_3 设置为 -3.5°
show            显示当前偏差
save            保存标定
cancel          取消，不写文件
```

也可以不用人工选择暂停点，直接指定源轨迹时间：

```text
--pause-time 5.0
```

保存后生成：

```text
<bundle-dir>/wrist_mount_calibration.json
<bundle-dir>/ik_plan_cache_wrist.npz
```

`wrist_3` 的末端姿态以 $360°$ 为周期。工具会在回放和保存标定缓存前，对整段 wrist_3 统一加减整数个 $360°$，选择远离 $[-360°,360°]$ 限位的等价表示。当前缓存自动执行 `-1 x 360°`，原始 `203.96° .. 271.21°` 被重映射为 `-156.04° .. -88.79°`，限位余量约 `204°`。因此在源时间约 5 秒执行 `set 100` 时，目标约为 `4.3°`，而不是 `364.3°`。

这类情况通常不需要更换物理安装角。只有整段 wrist_3 跨度本身接近或超过机械可用范围、或其他腕关节也同时接近限位时，才需要重新选择 IK 分支或调整物理安装。交互命令默认拒绝超出 `[-360°,360°]` 的目标；可通过 `--wrist3-lower-deg/--wrist3-upper-deg` 修改软件边界，但不应超过机器人的实际安全限制。

后续 replay 会自动：

1. 将标定角作为末端局部 Z 轴旋转应用到整段 TCP 姿态；
2. 加载 `ik_plan_cache_wrist.npz`；
3. 重新执行 seeded-IK、净空、速度和当前 moveJ 路径检查；
4. 不再执行整段多分支 IK 搜索。

若要暂时忽略标定，使用 `--no-wrist-calibration`。指定其他标定文件使用 `--wrist-calibration PATH`。

这里没有使用 ur_rtde `freedriveMode`，因为其 `free_axes` 是笛卡尔自由度，不能可靠地只释放第 6 关节。标定工具使用低速 `moveJ` 仅调整 `wrist_3`，避免其他关节漂移。

### 10.8 同步回放 MechHand

默认行为是读取 bundle 内的 `sdk_commands.csv`，按 `time-scale` 后的时间线向以下地址发送原始 SDK 命令：

```text
--hand-ip localhost --hand-port 60686
```

连接在任何 `moveJ` 之前建立；如果 `localhost:60686` 连接失败，机械臂不会开始运动。连续重复命令会折叠为一次，但同一命令在其他动作之后再次出现时仍会发送。

当前 trial 去重后有 6 个 SDK 动作切换，源时间范围为 `3.041929 .. 11.573594 s`。只调试机械臂时使用 `--skip-hand` 完全关闭 SDK 连接和发送。

### 10.9 关键参数

- `--time-scale 15`：执行时间放大 15 倍，即以原速度的 $1/15$ 回放；当前轨迹执行约 `207.760 s`，峰值关节速度约 `0.840 rad/s`。
- `--rate-hz 125`：按 UR3 CB3 控制周期重采样；位置线性插值，姿态四元数 SLERP。
- `--movej-speed/--movej-acceleration`：移动到第一帧时使用的关节速度和加速度。
- `--min-link-clearance 0.01`：移动连杆表面必须高于 Base `z=0` 至少 `10 mm`。
- `--link-radius 0.04`：把移动连杆近似为半径 `40 mm` 的胶囊，用于从中心线高度扣除实体尺寸。
- `--max-source-joint-step-deg 30`：源轨迹相邻帧任一关节跳变超过 `30°` 时拒绝该 IK 分支。
- `--max-joint-speed 1.2`：125 Hz 插值后的最大允许关节速度，单位 rad/s。
- `--skip-hand`：完全禁用手部命令发送。
- `--ik-preflight-only`：连接并检查 IK，但在任何运动前退出。
- `--ik-cache PATH`：指定缓存路径；默认根据是否启用 wrist 标定选择 `<bundle-dir>/ik_plan_cache_wrist.npz` 或 `<bundle-dir>/ik_plan_cache.npz`。
- `--replan-ik`：忽略已有缓存并重新执行多分支 IK 规划。
- `--wrist-calibration PATH`：指定 wrist_3 安装角标定文件。
- `--no-wrist-calibration`：忽略 bundle 中已有的 wrist 标定。
- `--execute`：唯一启用真实运动的开关。
- `--mount-correction-deg -45`：只用于 bundle 生成，不应传给 replay 脚本。

Base 平面净空使用 UR3 CB3 nominal DH 与胶囊近似，只检查机器人本体相对 `z=0` 平面，不包含桌面、相机、线缆、MechHand 外形或其他障碍物，也不能替代现场碰撞评估。固定 Base 到 shoulder 的连杆与安装平面相交是正常现象，因此该固定段不参与净空门控。

## 11. 已知边界

以下能力尚未完整实现：

- RPi 串口命令读取
- 灵巧手 SDK 转发链路的完整闭环
- 实时 hand feedback 全链路记录

## 12. 常用命令速查

```bash
# 安装本项目 CLI
pip install -e .

# 离线 per-trial 三维重建
prism-reconstruct-trials data/raw/task_YYYYmmdd_HHMMSS_task-name

# 轨迹分析
prism-analyze-trajectory data/raw/task_YYYYmmdd_HHMMSS_task-name
```
