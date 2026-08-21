# PRISM 4相机硬件触发标定操作指南

本文档记录使用 **硬件触发同步** 对 4 台 HIK USB 工业相机进行 ChArUco 标定的完整流程。

与自由运行模式不同，硬件触发确保所有相机在帧级别严格同步，无需事后时间对齐校准。

---

## 前置条件

### 硬件配置

- **4 台 HIK 相机固定安装**，标定期间不得移动
- **主相机**: DA8165486
  - 使用**软件触发**，通过程序控制触发时刻
  - GPIO Line1 **输出** 触发信号给其他相机
- **从相机** (cam1-3)
  - 使用**硬件触发**，接收主相机的触发信号
  - GPIO Line0 **输入**接收来自主相机 GPIO Line1 的信号

### ChArUco 标定板

本指南使用大凡视觉 CC200-15-11.25 标定板：

| 参数 | 值 |
|------|-----|
| `squares_x` | 12 |
| `squares_y` | 9 |
| `square_length_mm` | 15 |
| `marker_length_mm` | 11.25 |
| `aruco_dict` | DICT_5X5_1000 |

若使用其他板，按实际参数修改。

### 软件依赖

```bash
# PRISM 已包含的依赖
pip install numpy opencv-contrib-python
```

**注意**: 需要 `opencv-contrib-python`（非 `opencv-python`），因为 ChArUco 在 aruco/contrib 模块。

---

## 3. 数据采集流程

### 3.1 启动采集脚本

```bash
python3 /mnt/projects-8tb/PRISM/tools/prism_charuco_calibration_capture.py \
  --output-dir ~/mvs_charuco_data \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --exposure-us 12000 --gain 0 --frame-rate 15
```

**参数说明**:

- `--output-dir`: 输出目录（会自动创建时间戳子目录）
- `--squares-x / --squares-y`: 标定板网格尺寸
- `--square-length-mm / --marker-length-mm`: 标定板尺寸（毫米）
- `--aruco-dict`: ArUco 字典（通常用 DICT_5X5_1000）
- `--exposure-us`: 固定曝光时间（微秒），推荐 12000 μs
- `--gain`: 增益值，推荐 0
- `--frame-rate`: 采集帧率（fps），推荐 15 用于标定

### 3.2 交互操作

脚本启动后：

1. **显示可用相机列表**
   ```
   Found 4 USB cameras:
     [0] model=..., serial=...
     [1] model=..., serial=...
     ...
   ```

2. **选择相机**
   - 输入空行表示选择前 4 个 (0,1,2,3)
   - 或输入指定索引，如 `0,1,2,3`

3. **相机初始化**
   ```
   Initializing hik0: model=... serial=DA8165486
     -> Master camera: software trigger + GPIO Line1 output
   Initializing hik1: model=... serial=...
     -> Slave camera: hardware trigger on GPIO Line0
   ...
   ```

4. **交互式采集**
   ```
   Frame 0: 
   ```
   - 按 ENTER 键采集一组同步图（4 个相机同时拍照）
   - 按 `q` 后回车完成采集

5. **采集建议**
   - 采集 **25-50 组**图像（建议 30-40 组）
   - 标定板姿态要**多样**：近/中/远，俯仰/偏航/滚转
   - 覆盖画面**中心与四角**
   - 保证**无运动模糊、无过曝、角点清晰**

### 3.3 输出目录结构

采集完成后，会生成如下目录结构（例如 `~/mvs_charuco_data/CharucoCapture_20260723_144500/`）：

```
CharucoCapture_20260723_144500/
  calibration_metadata.json      # 本次采集配置和元数据
  cam0_DA8165486/                # 主相机的所有帧
    frame_0000.png
    frame_0001.png
    ...
  cam1_...../                    # 从相机 1 的所有帧
    frame_0000.png
    frame_0001.png
    ...
  cam2_...../                    # 从相机 2
    ...
  cam3_...../                    # 从相机 3
    ...
```

**注意**: 同名文件（同一 `frame_XXXX.png`）来自不同相机但在同一时刻采集（硬件同步）。

---

## 4. 标定流程（内参 + 外参）

### 4.1 执行标定命令

在采集完成后，使用 MVS 官方标定脚本进行处理：

```bash
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoCalibrate4Cam.py \
  --dataset-root ~/mvs_charuco_data/CharucoCapture_20260723_144500 \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --output ~/mvs_charuco_data/charuco_4cam_result.json
```

**关键参数**:

- `--dataset-root`: 指向 **具体的采集会话目录**（含时间戳的 CharucoCapture_xxx），**不是其父目录**
- `--squares-x / --squares-y`: 必须与采集时保持一致
- `--output`: 内/外参标定结果（JSON 格式）

### 4.2 低分辨率或角点检测困难时

若出现 marker 检测但 ChArUco 插值失败的情况，尝试：

```bash
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoCalibrate4Cam.py \
  --dataset-root ~/mvs_charuco_data/CharucoCapture_20260723_144500 \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --upsample 3 \
  --min-markers 2 \
  --min-charuco-corners 3 \
  --min-valid-images 8 \
  --min-pair-samples 4 \
  --output ~/mvs_charuco_data/charuco_4cam_result.json
```

### 4.3 输出结果说明

标定完成后生成 `charuco_4cam_result.json`，包含：

```json
{
  "board": {
    "squares_x": 12,
    "squares_y": 9,
    "square_length_mm": 15,
    "marker_length_mm": 11.25,
    "aruco_dict": "DICT_5X5_1000"
  },
  "intrinsics": {
    "cam0": { "K": [...], "D": [...], "RMS": 0.3 },
    "cam1": { "K": [...], "D": [...], "RMS": 0.25 },
    ...
  },
  "multi_camera": {
    "frames_total": 1200,
    "extrinsics": {
      "cam0_to_cam1": { "R": [...], "t": [...], "link_type": "direct" },
      ...
    }
  }
}
```

### 4.4 使用 CGB-035 手眼标定板采集相机内外参

如果现场使用的是固定在 UR3 末端的 CGB-035 非对称圆点板，也可以直接用它采集四相机内外参，不必另外准备 ChArUco 板。该流程不需要机器人运动：手持或固定圆点板，在四路实时画面中改变位置和姿态，按 Enter 或 Space 保存同步图像。

启动实时采集：

```bash
python3 tools/capture_circle_grid_calibration.py \
  --output-dir ~/mvs_circle_grid_data \
  --exposure-us 12000 --gain 0 --frame-rate 15
```

每路画面会显示 `VALID` 或 `INVALID`。默认至少 3 路识别成功才允许保存；建议采集 25-50 组，并覆盖中心、四角、近中远距离以及不同俯仰/偏航/滚转。输出目录类似：

```text
CircleGridCapture_YYYYMMDD_HHMMSS/
  cam0_<serial>/frame_0000.png
  cam1_<serial>/frame_0000.png
  cam2_<serial>/frame_0000.png
  cam3_<serial>/frame_0000.png
```

使用 CGB-035 的 4 列 x 5 行、35 mm 间距求解内参和相机间外参：

```bash
python3 tools/calibrate_4cam_circle_grid.py \
  --image-root ~/mvs_circle_grid_data/CircleGridCapture_YYYYMMDD_HHMMSS \
  --output configs/devices/circle_grid_4cam_result.json
```

输出 JSON 的结构与 `charuco_4cam_result.json` 兼容，包含每台相机的 `K`、`D` 和 `cam0_to_cam1..3`。如果将它用于现有重建链路，需要把对应配置路径传给重建命令；如果要重新计算 UR3 手眼矩阵，还需使用同一批图像并按第 11 节记录每帧的 TCP 位姿。

---

## 5. 外参可视化

### 5.1 执行可视化命令

```bash
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoVisualize4Cam.py \
  --result-json ~/mvs_charuco_data/charuco_4cam_result.json \
  --save-path ~/mvs_charuco_data/extrinsics_view.png \
  --show y
```

输出：

- **3D 图**: 4 个相机坐标轴与视锥
- **终端输出**: cam0 到 cam1/2/3 的基线长度（m）

---

## 6. 质量验收建议

### 内参质量

- **单相机 RMS**: 建议 < 0.5 px（视镜头和分辨率）
- **重投影误差**: 尽量小

### 外参质量

- **旋转稳定度** (`rot_std_deg`): 越小越好（< 1°）
- **平移稳定度** (`trans_std_m`): 越小越好（< 5 mm）
- **基线长度**: 应与实际安装尺寸一致

### 同步效果

由于使用硬件触发，多个相机的**帧级别同步精度通常优于 1 ms**，远优于自由运行模式。

---

## 7. 常见问题

### 1. 相机无法识别或连接失败

- 检查 USB 连接是否稳定
- 运行 `lsusb` 确认相机被识别
- 确认 MVS SDK 已正确安装

### 2. 采集过程中某相机超时

- 检查 GPIO 接线（Line0 输入、Line1 输出）
- 检查主相机 (DA8165486) 的 GPIO Line1 是否正确输出触发信号
- 尝试降低 `--frame-rate` 以获得更长的触发等待时间

### 3. 检测不到角点

- 增加照明，避免反光
- 提高标定板占图像的比例
- 确保清晰对焦
- 若 marker 能检出但 ChArUco 插值失败，使用 `--upsample 3`

### 4. 外参抖动较大

- **首先检查触发同步**：确认 GPIO Line0/Line1 接线正确
- 检查相机/支架是否有微小位移
- 重新采集，覆盖更丰富的标定板姿态

### 5. 某台相机与其他相机共视不足

- 重新采集时，加强相邻两台相机**同时看到标定板**的样本数
- 用实时检测预览（如 MVS 官方脚本的 `--detect-overlay y`）来指导采集

---

## 8. 一套完整的推荐执行顺序

本指南的相机标定流程与 UR3 replay 流程相互独立。当前 replay 脚本版本说明见项目 [README.md](../README.md) 第 10 节；不要把旧版日志中的 clearance-aware IK、IK cache 或 `servoJ` 描述用于判断当前版本是否完成安全预检。

### 第 1 步：采集

```bash
python3 /mnt/projects-8tb/PRISM/tools/prism_charuco_calibration_capture.py \
  --output-dir ~/mvs_charuco_data \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --exposure-us 12000 --gain 0 --frame-rate 15
```

（交互式采集 30-40 组图像）

### 第 2 步：标定

```bash
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoCalibrate4Cam.py \
  --dataset-root ~/mvs_charuco_data/CharucoCapture_20260723_144500 \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --aruco-dict DICT_5X5_1000 \
  --output ~/mvs_charuco_data/charuco_4cam_result.json
```

### 第 3 步：可视化和验收

```bash
python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoVisualize4Cam.py \
  --result-json ~/mvs_charuco_data/charuco_4cam_result.json \
  --save-path ~/mvs_charuco_data/extrinsics_view.png \
  --show y
```

检查输出的 3D 视图和基线长度是否符合实际安装尺寸。

### 第 4 步：验证和部署

- 检查 `charuco_4cam_result.json` 质量指标
- 更新 PRISM 项目配置：
  ```bash
  cp ~/mvs_charuco_data/charuco_4cam_result.json \
     /mnt/projects-8tb/PRISM/configs/devices/charuco_4cam_result.json
  ```

---

## 9. 硬件触发与自由运行的区别

| 方面 | 硬件触发 | 自由运行 |
|------|--------|--------|
| **同步精度** | 帧级别 (< 1 ms) | 事后校准 (~10 ms) |
| **是否需要时间对齐** | 否 | 是 (AprilTag) |
| **GPIO 配置** | 需要外部接线 | 无 |
| **实时采集帧率** | 受主相机 fps 限制 | 各自独立 |
| **采集复杂度** | 低 | 中 |
| **标定难度** | 较低 | 中 |

---

## 10. 后续使用

标定完成后，在 PRISM 在线采集中使用结果：

```bash
python3 src/prism/cli/online.py \
  --calib-json /mnt/projects-8tb/PRISM/configs/devices/charuco_4cam_result.json \
  --task-name test_task
```

PRISM 会自动加载内/外参，进行实时 3D 重建。

## 11. CGB-035 + UR3 CB3 联合手眼标定

本节用于使用固定在 **UR3 CB3 末端**的 CGB-035 非对称圆点阵列标定板，联合求解：

```text
base_from_cam0
tcp_from_board
```

这里的 CGB-035 参数为：

| 参数 | 值 |
|------|-----|
| 阵列类型 | 非对称圆点阵列 |
| 列数 | 4 |
| 行数 | 5 |
| 圆心间距 | 35 mm |

脚本会使用 `cam0`、`cam1`、`cam2`、`cam3` 中成功检测到圆点阵列的相机。每个相机的检测结果会使用本文件中的四相机外参转换到 `cam0`；如果同一帧有多台相机检测成功，则选择重投影误差最小的结果。

### 11.1 安装 UR3 控制依赖

在运行机器人控制脚本的 Python 环境中安装 `ur_rtde`：

```bash
pip install ur-rtde
```

电脑必须能通过网络连接 UR3 CB3 控制器，并且机器人 TCP 已在示教器中正确配置。

### 11.2 准备 UR3 运动点

复制 waypoint 示例并根据实际工作空间修改：

```bash
cp configs/deployment/ur3_handeye_waypoints.draft.json \
   configs/deployment/ur3_handeye_waypoints.json
```

waypoint 使用 UR 标准笛卡尔位姿格式：

```text
[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
```

其中位置单位是米，姿态是弧度旋转向量。示例：

```json
{
  "frame": "frame_0000",
  "pose": [0.35, -0.20, 0.30, 2.20, 2.20, 0.00]
}
```

示例坐标仅用于说明格式，不能直接视为你的机器人安全位置。正式执行前必须检查每个点的可达性、碰撞风险和标定板是否在相机视野内。建议准备 15-30 个点，并同时改变位置和 Roll/Pitch/Yaw。

### 11.3 预览运动路径

不加 `--execute` 时只打印 waypoint，不连接机器人，也不会运动：

```bash
python3 tools/collect_ur3_handeye_poses.py \
  --robot-ip 192.168.1.102 \
  --waypoints configs/deployment/ur3_handeye_waypoints.json \
  --output-csv /home/daotan/handeye_robot_poses.csv
```

将 `192.168.1.100` 替换为实际 UR3 控制器 IP。

### 11.4 移动 UR3 并记录实际 TCP 位姿

确认 waypoint 安全后，使用低速执行：

```bash
python3 tools/collect_ur3_handeye_poses.py \
  --robot-ip 192.168.1.102 \
  --waypoints configs/deployment/ur3_handeye_waypoints.json \
  --output-csv /home/daotan/handeye_robot_poses.csv \
  --speed 0.05 \
  --acceleration 0.15 \
  --settle-time 1.5 \
  --execute
```

每到一个姿态，脚本会读取机器人返回的实际 TCP 位姿，并显示：

```text
Capture the cam0 image for frame_0000, then press Enter...
```

此时采集对应的同步相机图片，确认图片已经保存后按 Enter 进入下一点。输出 CSV 包含：

```text
frame,x_m,y_m,z_m,rx_rad,ry_rad,rz_rad,timestamp_unix
```

图片文件名必须和 `frame` 对应，例如：

```text
cam0_DA7914047/frame_0000.png
cam1_DA8165484/frame_0000.png
cam2_DA8165486/frame_0000.png
cam3_DA7914080/frame_0000.png
```

### 11.5 自由驱动采集图像和 TCP 位姿

如果不使用预设 waypoint，可以让 UR3 进入自由驱动模式，手动把标定板移动到不同姿态。启动后先选择四台相机，随后会打开四相机 `2x2` 实时预览。每个画面显示 CGB-035 是否检测成功，窗口底部显示 `valid=N/4`。在预览窗口按 Enter，脚本会保存四台相机的同步图像，并将当前 TCP 位姿写入 CSV；按 `q` 或 `Esc` 结束。

```bash
python3 tools/collect_ur3_handeye_poses.py \
  --robot-ip 192.168.1.102 \
  --output-csv /home/daotan/handeye_robot_poses.csv \
  --camera-output-dir /home/daotan/mvs_charuco_data \
  --camera-exposure-us 12000 \
  --camera-gain 0 \
  --camera-frame-rate 30 \
  --preview-width 640 \
  --freedrive
```

运行环境需要安装 `ur-rtde`。电脑与 UR3 控制器必须能连接 RTDE 端口 `30004`。

启动相机后输入相机索引时，必须按照当前标定文件的顺序选择：

```text
cam0 = DA7914047
cam1 = DA8165484
cam2 = DA8165486
cam3 = DA7914080
```

预览窗口中绿色 `VALID` 表示当前画面完整检测到非对称 `4列x5行` 圆点阵列，红色 `INVALID` 表示当前画面未检测成功。建议至少有一台相机显示 `VALID` 后再按 Enter。每次保存后，终端会显示类似：

```text
saved frame_0000: TCP=['x', 'y', 'z', 'rx', 'ry', 'rz']
```

图片会保存到新建的时间戳目录：

```text
/home/daotan/mvs_charuco_data/CharucoCapture_YYYYMMDD_HHMMSS/
  cam0_DA7914047/frame_0000.png
  cam1_DA8165484/frame_0000.png
  cam2_DA8165486/frame_0000.png
  cam3_DA7914080/frame_0000.png
```

### 11.6 执行多相机联合手眼标定

`--image-root` 应指向包含四个相机目录的采集会话目录：

```bash
python3 tools/calibrate_robot_world_hand_eye_circle_grid.py \
  --image-root /home/daotan/mvs_charuco_data/CharucoCapture_20260819_142703 \
  --robot-csv /home/daotan/handeye_robot_poses.csv \
  --camera-calib configs/devices/charuco_4cam_result.json \
  --camera-key cam0 \
  --output configs/deployment/robot_camera_handeye.json
```

每一帧会遍历四台相机。如果某台相机看不到完整圆点阵列，会尝试其他相机；如果所有相机都失败，该帧会被跳过。输出文件 `robot_camera_handeye.json` 包含 `base_from_cam0`、`tcp_from_board`、有效帧列表、实际使用的相机以及平移/旋转误差。

### 11.7 仅使用 cam0 的兼容方式

如果只准备了 cam0 图片，也可以使用旧模式：

```bash
python3 tools/calibrate_robot_world_hand_eye_circle_grid.py \
  --image-dir /home/daotan/mvs_charuco_data/handeye_cam0 \
  --robot-csv /home/daotan/handeye_robot_poses.csv \
  --camera-calib configs/devices/charuco_4cam_result.json \
  --output configs/deployment/robot_camera_handeye.json
```

该模式不会读取 cam1-cam3。多相机标定应优先使用 `--image-root`。

---

## 参考资料

- MVS 官方标定指南: `/opt/MVS/Samples/64/Python/General/Recording/ChArUco_4Cam_Calibration_Guide.md`
- PRISM 源代码: `/mnt/projects-8tb/PRISM/`
- HIK 官方文档: `/opt/MVS/` 下的 PDF 文档
