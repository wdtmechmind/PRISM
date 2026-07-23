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

---

## 参考资料

- MVS 官方标定指南: `/opt/MVS/Samples/64/Python/General/Recording/ChArUco_4Cam_Calibration_Guide.md`
- PRISM 源代码: `/mnt/projects-8tb/PRISM/`
- HIK 官方文档: `/opt/MVS/` 下的 PDF 文档
