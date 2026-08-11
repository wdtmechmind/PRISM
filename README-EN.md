# PRISM — Multi-Camera Capture and Rigid-Body Reconstruction

PRISM is a research platform for **synchronized multi-camera recording, LED-based 3D triangulation, and 6-DOF rigid-body pose estimation** of a dexterous robotic hand. It integrates four HIK Vision USB3 industrial cameras with an Intel RealSense D435, uses hardware-trigger synchronization for sub-millisecond frame alignment, and provides a complete pipeline from calibration through real-time visualization to offline batch reconstruction and quality reporting.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Setup](#2-hardware-setup)
3. [Software Architecture](#3-software-architecture)
4. [Calibration Pipeline](#4-calibration-pipeline)
5. [Data Collection](#5-data-collection)
6. [LED Detection Backends](#6-led-detection-backends)
7. [Real-Time Reconstruction and Visualization](#7-real-time-reconstruction-and-visualization)
8. [Offline Per-Trial Reconstruction](#8-offline-per-trial-reconstruction)
9. [DexHand Control](#9-dexhand-control)
10. [Trajectory Analysis and Quality Reports](#10-trajectory-analysis-and-quality-reports)
11. [YOLO Dataset Tools](#11-yolo-dataset-tools)
12. [Isaac Sim Replay](#12-isaac-sim-replay)
13. [CLI Reference](#13-cli-reference)
14. [Data Output Structure](#14-data-output-structure)
15. [Configuration](#15-configuration)
16. [Dependencies and Installation](#16-dependencies-and-installation)

---

## 1. System Overview

PRISM operates on the following core pipeline:

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ HIK Cam 0 (master) ─── GPIO Line1 ──► HIK Cam 1/2/3 (slaves)   │
 │              ↓ hardware trigger at 300 fps                       │
 │   per-camera: H.264 MP4 + per-frame timestamp CSV               │
 └─────────────────────────────────────────────────────────────────┘
                         ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │  Online detection loop (per preview frame)                       │
 │    HSV / YOLO / Hybrid LED detection                            │
 │    Multi-view triangulation → 3D LED positions                   │
 │    SVD-based rigid-body pose (roll, pitch, yaw + translation)   │
 │    Live 3D plot (raw frame + corrected frame)                    │
 └─────────────────────────────────────────────────────────────────┘
                         ↓
 ┌─────────────────────────────────────────────────────────────────┐
 │  Offline per-trial reconstruction (prism-reconstruct-trials)     │
 │    Re-reads recorded videos frame-by-frame                       │
 │    Same detector chain as collection run                        │
 │    Produces reproducible trajectory CSVs + rigid pose CSVs      │
 │    Quality report (accuracy_report.json / .md)                  │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Setup

### Cameras

| Role | Device | Interface | Notes |
|------|--------|-----------|-------|
| Master (cam0) | HIK DA8165486 | USB 3.0 | Software trigger → GPIO Line1 strobe out |
| Slave (cam1–3) | HIK USB3 cameras | USB 3.0 | Hardware trigger on GPIO Line0 |
| Depth / color | Intel RealSense D435 | USB 3.0 | 1280×720 @ 30 fps, optional |

**Synchronization**: The master camera fires software triggers at a configurable rate (default 300 fps). Its GPIO Line1 output drives the hardware-trigger input (Line0) of all slave cameras. This achieves frame-level alignment with < 1 ms inter-camera skew — no software-based temporal calibration is needed.

### Raspberry Pi Interface

A Raspberry Pi is optionally connected via serial port (`rpi_port`) to receive I/O events and forward them as timestamped logs. The interface is handled by `src/prism/devices/rpi/io_interface.py`.

---

## 3. Software Architecture

```
src/prism/
├── cli/                      # Entry points
│   ├── collect.py            # prism-collect
│   └── hand_control.py       # prism-hand
├── common/
│   ├── config.py             # YAML config loader with strict key validation
│   ├── console.py            # Rich-based console utilities
│   └── timebase.py           # Timestamp interpolation helpers (pick_nearest, pick_bracket, FpsMeter)
├── devices/
│   ├── cameras/
│   │   ├── mvs_camera.py     # HIK MVS SDK wrappers (enumerate, open, configure, grab)
│   │   ├── highspeed_camera.py  # HikCaptureThread, SoftwareTriggerThread
│   │   └── realsense_camera.py  # RSCaptureThread, undistort maps, intrinsics loader
│   ├── hand/
│   │   └── socket_client.py  # MechHandClient, 17-gesture protocol
│   └── rpi/
│       └── io_interface.py   # Raspberry Pi serial I/O bridge
├── online/
│   ├── session_manager.py    # Main collection loop (argument parsing → camera setup → trial loop)
│   ├── display_server.py     # OpenCV preview compositor (4-cam grid + hand panel)
│   └── trajectory_plotter.py # Live 3D matplotlib plot (raw + corrected frames)
├── reconstruction/
│   ├── calibration.py        # ChArUco JSON loader, corrected-frame transform builder
│   └── realtime_reconstruction.py  # LED detection, triangulation, pose, Kalman fill
├── processing/
│   ├── offline_reconstruct.py # Per-trial video replay and LED re-detection
│   ├── offline_rebuild.py     # Aligned-segment video rebuilder
│   ├── trajectory_analyzer.py # Statistical trajectory comparison and visualization
│   └── led_accuracy.py        # Quantitative accuracy metrics (reproj error, jitter, drift)
└── recording/
    ├── video_recorder.py      # VideoSink: async MP4 writer + timestamp CSV
    └── metadata_writer.py     # task_metadata.yaml, trial metadata, hand log placeholders
```

---

## 4. Calibration Pipeline

PRISM uses two separate calibration procedures.

### 4.1 Multi-Camera Intrinsic + Extrinsic Calibration (ChArUco)

Captures synchronized frames of a ChArUco board from all four HIK cameras and produces a JSON file containing per-camera intrinsics (K, D) and pairwise extrinsics (cam0→camN rotation + translation).

```bash
python3 tools/prism_charuco_calibration_capture.py \
  --output-dir ~/mvs_charuco_data \
  --squares-x 12 --squares-y 9 \
  --square-length-mm 15 --marker-length-mm 11.25 \
  --live y --detect-overlay y
```

The resulting JSON (e.g. `configs/devices/charuco_4cam_result.json`) is the single source of truth for all downstream triangulation and pose-estimation steps.

**Corrected coordinate frame**: `calibration.py` builds a deterministic world frame from the camera centers (centroid as origin, principal axis aligned to camera array span), so that all trajectory data is expressed in a stable, physically meaningful coordinate system.

### 4.2 HSV Threshold Calibration

An interactive single-camera tool for tuning per-color HSV bounds. Click on LED pixels and the tool automatically expands the color range.

```bash
python3 tools/calibrate_hsv_led.py \
  --config configs/collection/default_online.yaml \
  --output configs/collection/hsv_tuned.yaml
```

| Key | Action |
|-----|--------|
| `1`–`4` | Switch active color (red / yellow / blue / green) |
| Left click | Sample pixel, expand HSV bounds |
| Right click | Undo last sample |
| `c` / `r` | Reset current color to initial |
| `w` | Write YAML to `--output` |
| `p` | Print YAML snippet to terminal |

---

## 5. Data Collection

### 5.1 Starting a Collection Session

```bash
scripts/collect_task.sh \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 20 \
  --output-dir data/raw
```

Or directly via the CLI entry point:

```bash
prism-collect \
  --config configs/collection/default_online.yaml \
  --task-name grasp-demo \
  --num-trials 20
```

### 5.2 Session Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Start / stop recording the current trial |
| `p` / `r` | Pause / resume trajectory update |
| `1`–`17` | Send gesture command to DexHand |
| `q` / `Esc` | End session |

### 5.3 What Happens During a Session

1. All four HIK cameras are opened, configured (exposure, gain, frame-rate, trigger mode) and their parameters are verified against readbacks with configurable tolerance.
2. One camera is designated master (software trigger + GPIO strobe output); the rest are hardware-trigger slaves.
3. The optional RealSense D435 starts its own capture thread at a lower frame rate (default 30 fps).
4. For each trial:
   - Pressing `Space` opens a `VideoSink` per camera (async MP4 writer + timestamp CSV).
   - Frames are grabbed in dedicated `HikCaptureThread` threads; each frame is submitted to the video queue and also forwarded to the preview pipeline.
   - LED detection runs on the preview frames; detected positions are triangulated and the rigid-body pose is estimated on every preview frame.
   - The live 3D plotter updates in real-time with two sub-plots: raw world frame and corrected frame.
   - Pressing `Space` again closes the writers, flushes all queued frames, and writes `trial_metadata.yaml`.
5. After all trials, `task_metadata.yaml` is written.
6. The user is optionally prompted to run offline reconstruction immediately.

### 5.4 Camera Parameters

All parameters are configurable via YAML + CLI override:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hik_exposure_us` | 3000 | Exposure time in microseconds |
| `hik_gain` | 12.0 | Analog gain |
| `hik_frame_rate` | 300.0 | Trigger/acquisition rate (fps) |
| `rec_brightness_alpha` | 2.0 | Gamma scale for saved video |
| `rec_brightness_beta` | 20.0 | Brightness offset for saved video |
| `rs_width` / `rs_height` | 1280 × 720 | RealSense resolution |
| `rs_fps` | 30 | RealSense frame rate |

---

## 6. LED Detection Backends

Four colored LEDs (red, yellow, blue, green) are attached to the tracked rigid body. PRISM supports three detection backends, selectable per session or overridden at offline reconstruction time.

### 6.1 HSV Detection (`hsv`)

Converts each frame to HSV, applies per-color thresholds (with hue-wrap support), performs morphological open+close to reduce noise, finds connected components, and selects the largest blob above a minimum area as the LED centroid.

### 6.2 YOLO Detection (`yolo`)

Uses an Ultralytics YOLOv8/v5 model trained on PRISM LED images. The model outputs bounding boxes with class IDs corresponding to LED colors. The centroid is the box center. Confidence threshold, IoU threshold, and inference size are all configurable.

### 6.3 Hybrid Detection (`hybrid`)

YOLO is attempted first. For any color not detected above the confidence threshold, HSV is used as fallback. This combines the robustness of a learned model with the reliability of a rule-based detector.

### 6.4 Backend Configuration

```bash
# CLI flags (apply to both collection and offline reconstruction)
--detector-backend   hsv | yolo | hybrid
--yolo-weights       path/to/best.pt
--yolo-conf          0.25            # confidence threshold
--yolo-iou           0.45            # NMS IoU threshold
--yolo-imgsz         640             # inference image size
```

**Inheritance**: When a task is collected with YOLO or hybrid, the detector parameters are written into `task_metadata.yaml`. Offline reconstruction reads them automatically, so re-processing without explicit flags reproduces the original detector configuration.

---

## 7. Real-Time Reconstruction and Visualization

### 7.1 Triangulation

For each preview frame, detected 2D LED centroids from all cameras with a valid detection are passed to `robust_triangulate`. This performs linear least-squares DLT triangulation, then optionally rejects cameras whose reprojection error exceeds `max_norm_reproj_error`. The result is a 3D world position per color.

### 7.2 Rigid-Body 6-DOF Pose Estimation

Once at least three LED colors are triangulated, `build_body_model` constructs a local body frame from the first three observed LED positions (using SVD to define orthogonal axes). On subsequent frames, `estimate_pose_from_model` solves for the rigid rotation and translation via:

$$H = P_{\text{model}}^T P_{\text{observed}}, \quad [U, \Sigma, V^T] = \text{SVD}(H), \quad R = V U^T$$

with determinant-correction for degenerate cases. The resulting 6-DOF pose (x, y, z, roll, pitch, yaw in ZYX Euler convention) is logged in real-time.

### 7.3 Kalman Fill and Interpolation

The online session manager maintains two parallel trajectory records:
- **Nearest**: each LED position is assigned the wall-clock timestamp of the nearest camera frame.
- **Interpolated**: positions are linearly interpolated to a fixed time grid.

Missing detections in the online display are filled using Kalman prediction to maintain visual continuity, but these filled rows are tagged `mode=predicted` and excluded from accuracy metrics.

### 7.4 Live 3D Plot

`trajectory_plotter.py` renders a real-time matplotlib 3D figure with two panels:

- **Raw World Frame**: LED trajectories and camera positions as calibrated.
- **Corrected Frame**: the same data expressed in a normalized coordinate system (camera-centroid origin, deterministic axis orientation).

The plot is rendered to an off-screen Agg canvas and composited into the OpenCV preview window.

### 7.5 Preview Display

`display_server.py` composites:
- A 2×2 grid of camera thumbnails with LED detections overlaid.
- A hand-control panel showing DexHand connection status, last gesture, last raw command, and the full 17-gesture quick-reference table.
- Session statistics (trial count, frame count, FPS).

---

## 8. Offline Per-Trial Reconstruction

After collection, each trial can be re-reconstructed at full camera frame rate from the saved videos.

```bash
prism-reconstruct-trials data/raw/task_20260723_173238_grasp-demo \
  --calib-json configs/devices/charuco_4cam_result.json
```

### What it does

1. Scans the task directory for `trial_*/cameras/` subdirectories.
2. For each trial, reads all four HIK MP4 files and their `_timestamps.csv` files.
3. Aligns frames across cameras by matching timestamps (within a configurable tolerance, default 8 ms).
4. Runs the configured LED detector on every aligned frame set.
5. Triangulates and estimates 6-DOF pose on every frame — no Kalman fill, only `mode=measured` rows.
6. Writes to `trial_xxxxxx/trajectory/`:
   - `trajectory_led.csv` — per-color 3D positions with dual time axis
   - `rigid_pose_6d.csv` — full 6-DOF pose time series
   - `rigid_6d_frames.png` — multi-panel trajectory plot
   - `accuracy_report.json` / `accuracy_report.md` — quantitative quality metrics

### Dual Time Axis

Every output row carries two time columns:
- `capture_wall_time`: absolute wall-clock time, shared with camera, hand command, and RealSense logs, enabling cross-modal alignment.
- `t_trial`: trial-relative time (0 = recording start), suitable for kinematic analysis and Isaac Sim replay.

### Re-running with a different detector

```bash
prism-reconstruct-trials data/raw/task_xxx \
  --detector-backend hybrid \
  --yolo-weights /path/to/best.pt \
  --yolo-conf 0.3
```

---

## 9. DexHand Control

The `MechHandClient` (`src/prism/devices/hand/socket_client.py`) communicates with the dexterous hand over a TCP socket using the DexHand `@ROG<N>&` command protocol.

### Supported Gestures

| ID | Pose | Description |
|----|------|-------------|
| 1 | `five_grasp` | Five-finger grasp |
| 2 | `five_open` | Five-finger open |
| 3 | `two_grasp_a` | Two-finger grasp (A) |
| 4 | `two_open_a` | Two-finger open (A) |
| 5 | `two_grasp_b` | Two-finger grasp (B) |
| 6 | `two_open_b` | Two-finger open (B) |
| 7 | `three_grasp_a` | Three-finger grasp (A) |
| 8 | `three_open_a` | Three-finger open (A) |
| 9 | `three_grasp_b` | Three-finger grasp (B) |
| 10 | `three_open_b` | Three-finger open (B) |
| 11 | `five_sequence` | Five-finger sequential motion |
| 12 | `thumb_in` | Thumb inward |
| 13 | `thumb_out` | Thumb outward |
| 14 | `index_point` | Index pointing |
| 15 | `index_press` | Index press |
| 16 | `index_single_click` | Index single click |
| 17 | `index_double_click` | Index double click |

Gestures can be triggered from:
- **Keyboard** during collection (keys `1`–`17`).
- **Standalone CLI**: `prism-hand --pose five_grasp --ip 127.0.0.1 --port 60686`.
- **Programmatically** via the `MechHandClient` Python API.

A configurable settle time (`hand_settle_time_s`) is enforced after each command to allow the hand to physically complete the motion before the next command is accepted.

---

## 10. Trajectory Analysis and Quality Reports

### 11.1 Interactive Trajectory Analyzer

```bash
prism-analyze-trajectory data/raw/task_20260723_173238_grasp-demo \
  --output-dir ./traj_plots
```

Or via the shell wrapper:

```bash
scripts/analyze_trajectory.sh data/raw/task_xxx
```

Generates per-trial plots comparing online (preview-rate) and offline (camera-rate) trajectories:
- X/Y/Z time series for each LED color.
- 3D trajectory view.
- Detection mode breakdown (measured vs. predicted).
- Number of cameras contributing to each triangulation.

### 11.2 Quantitative Accuracy Reports

```bash
python3 tools/generate_trajectory_quality_reports.py \
  --task-dir data/raw/task_xxx \
  --write-json --write-md
```

Or for a single trial:

```bash
python3 tools/generate_trajectory_quality_reports.py \
  --trial-dir data/raw/task_xxx/trial_000012 \
  --write-json
```

Each report (JSON + Markdown) contains:
- **Per-color statistics**: detection rate, mean/max reprojection error, position jitter (frame-to-frame displacement), trajectory range.
- **Static-segment analysis**: automatic detection of static intervals; position range and drift are quantified to assess triangulation stability.
- **Rigid-body pose statistics**: translation jitter and angular jitter in roll/pitch/yaw.
- **Overall quality score**: pass/warn/fail for each LED and for the overall rigid-body reconstruction.

---

## 11. YOLO Dataset Tools

### 12.1 Online Annotation (real-time capture + label)

```bash
python3 tools/annotate_led_dataset.py \
  --output-dir data/datasets/led_yolo \
  --camera-indices 0,1,2,3 \
  --auto-from-hsv y \
  --hsv-config configs/collection/default_online.yaml \
  --box-size-px 28
```

Displays a live 4-camera preview. Pressing `c` or `Space` freezes the current synchronized frame set. The user selects a color class and clicks on each LED; HSV auto-labeling can pre-populate boxes. Saving writes YOLO `.txt` annotation files.

| Key | Action |
|-----|--------|
| `c` / `Space` | Freeze current 4-camera frame |
| `1`–`4` | Select color class (red/yellow/blue/green) |
| Left click | Add bounding box |
| Right click | Undo last box for current camera |
| `s` / `Enter` | Save frame and labels |
| `n` | Discard current frame |

### 12.2 Offline Annotation (capture then label)

```bash
# Step 1: capture synchronized image sets
python3 tools/capture_led_photos.py \
  --output-dir data/datasets/led_yolo_raw \
  --camera-indices 0,1,2,3

# Step 2: label offline
python3 tools/annotate_led_dataset_offline.py \
  --dataset-root data/datasets/led_yolo_raw/LedPhoto_YYYYmmdd_HHMMSS \
  --auto-from-hsv y \
  --hsv-config configs/collection/default_online.yaml \
  --box-size-px 28
```

### 12.3 YOLO Training

```bash
python3 tools/train_led_yolo.py \
  --dataset-root data/datasets/led_yolo \
  --weights yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

Converts PRISM annotation format to Ultralytics YOLO dataset structure and launches training. Supports YOLOv8 and YOLOv5 base weights.

### 12.4 Detection Accuracy Evaluation

```bash
python3 tools/eval_led_accuracy.py \
  --trial-dir data/raw/task_xxx/trial_000001
```

Loads `trajectory_led.csv` and `rigid_pose_6d.csv`, filters to `mode=measured` rows, and reports reprojection error distributions, jitter, and static-segment stability — the same metrics used by the quality report generator.

---

## 12. Isaac Sim Replay Pipeline

If no physical cameras are available, you can generate raw-like simulated trial
data first:

```bash
python3 simulation/scripts/collect_sim_trial.py --task-name sim_collect --num-trials 3 --duration-sec 10
```

This writes `trajectory_led.csv`, `rigid_pose_6d.csv`, `hand/sdk_commands.csv`,
and `hand/rpi_commands.csv` under `data/raw/task_*/trial_*/`.

Raw trial trajectories can be converted to the corrected frame, planned for the
AUBO i5 + MechHand robot, and rendered to an Isaac Sim replay video with one
pipeline command.

### End-to-End Replay Video

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_xxx/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

The pipeline writes corrected-frame CSVs, `planned_motion.csv`, and
`planned_motion_overhead.mp4` under `data/processed/simulation/<task>/<trial>/`.
See `simulation/README.md` for the staged commands and optional camera/video
arguments.

---

## 13. CLI Reference

| Command | Description |
|---------|-------------|
| `prism-collect` | Start a collection session (cameras + optional hand + online reconstruction) |
| `prism-reconstruct-trials <task-dir>` | Offline per-trial LED detection, triangulation and pose estimation |
| `prism-rebuild-aligned <input-dir>` | Rebuild time-aligned video segments from raw recordings |
| `prism-analyze-trajectory <task-dir>` | Generate trajectory comparison plots and statistics |
| `prism-hand --pose <name>` | Send a single gesture command to the DexHand |

Shell script wrappers (in `scripts/`):

| Script | Description |
|--------|-------------|
| `collect_task.sh` | Full-featured collection wrapper with post-process prompt |
| `analyze_trajectory.sh` | Trajectory analysis shortcut |
| `rebuild_aligned_segment.sh` | Video rebuild shortcut |
| `restart_rpi_usb.sh` | USB reset utility for RPi recovery |

---

## 14. Data Output Structure

```
data/raw/
└── task_YYYYmmdd_HHMMSS_<task-name>/
    ├── task_metadata.yaml          # task config snapshot, detector params, camera serials
    ├── trajectory_led_nearest.csv  # online: per-color 3D trajectory (nearest-frame time)
    ├── trajectory_led_interp.csv   # online: per-color 3D trajectory (interpolated)
    ├── rigid_pose_6d.csv           # online: 6-DOF pose time series
    ├── time_alignment_log.csv      # cross-camera timestamp alignment diagnostics
    └── trial_000000/
        ├── metadata.yaml           # trial start/end wall time, gesture log
        ├── cameras/
        │   ├── cam0.mp4            # HIK camera 0 video
        │   ├── cam0_timestamps.csv # per-frame: frame_index, capture_wall_time, trial_time, device_frame_num
        │   ├── cam1.mp4 / cam1_timestamps.csv
        │   ├── cam2.mp4 / cam2_timestamps.csv
        │   ├── cam3.mp4 / cam3_timestamps.csv
        │   └── realsense.mp4 / realsense_timestamps.csv (optional)
        └── trajectory/
            ├── trajectory_led.csv          # offline: per-color 3D positions, dual time axis
            ├── rigid_pose_6d.csv           # offline: 6-DOF pose, dual time axis
            ├── rigid_6d_frames.png         # multi-panel trajectory visualization
            ├── accuracy_report.json        # quantitative quality metrics (JSON)
            └── accuracy_report.md          # human-readable quality report (Markdown)
```

### Trajectory CSV Columns (`trajectory_led.csv`)

| Column | Description |
|--------|-------------|
| `t_sec` | Session-relative time in seconds |
| `t_trial` | Trial-relative time (0 = recording start) |
| `capture_wall_time` | Absolute wall-clock UNIX timestamp |
| `frame_index` | Camera frame index |
| `color` | LED color (`red` / `yellow` / `blue` / `green`) |
| `x_m`, `y_m`, `z_m` | 3D world position in meters |
| `mode` | `measured` or `predicted` |
| `num_views` | Number of cameras contributing |
| `max_norm_reproj_err` | Worst normalized reprojection error across cameras |
| `visible_cams` | Bitmask of contributing camera indices |
| `x_smooth_m`, `y_smooth_m`, `z_smooth_m` | Smoothed position |

### Rigid Pose CSV Columns (`rigid_pose_6d.csv`)

| Column | Description |
|--------|-------------|
| `t_sec`, `t_trial`, `capture_wall_time`, `frame_index` | Time axes (same as above) |
| `mode` | `measured` or `predicted` |
| `num_leds_used` | LEDs used in SVD fit |
| `modeled_leds` | LEDs with body-frame model entries |
| `visible_leds` | Currently detected LED count |
| `x_m`, `y_m`, `z_m` | Body origin position in meters |
| `roll_deg`, `pitch_deg`, `yaw_deg` | ZYX Euler angles in degrees |
| `x_smooth_m`, … `yaw_smooth_deg` | Smoothed pose |

---

## 15. Configuration

All collection parameters are stored in YAML files under `configs/collection/`. The loader enforces strict key validation — unknown keys raise an error.

```
configs/
├── collection/
│   ├── default_online.yaml     # Default camera, detector, and hand parameters
│   └── hsv_tuned.yaml          # Per-experiment HSV thresholds
├── devices/
│   ├── charuco_4cam_result.json    # Camera intrinsics + extrinsics (from calibration)
│   └── d435_charuco_intrinsics.json # RealSense D435 intrinsics
└── processing/                 # (reserved for processing configs)
```

Key parameters in `default_online.yaml`:

```yaml
hik_frame_rate: 300.0       # Hardware trigger rate
hik_exposure_us: 3000.0     # Exposure in microseconds
hik_gain: 12.0
detector_backend: hsv        # hsv | yolo | hybrid
calib_json: configs/devices/charuco_4cam_result.json
post_process: ask            # ask | now | never
```

---

## 16. Dependencies and Installation

### Required External SDKs

| SDK | Path | Notes |
|-----|------|-------|
| HIK MVS SDK | `/opt/MVS` | Must set `MVCAM_COMMON_RUNENV` via `source /opt/MVS/bin/set_env_path.sh /opt/MVS` |
| HIK Python binding | `/opt/MVS/Samples/64/Python/MvImport/` | Overridable via `PRISM_MVIMPORT_DIR` |

### Python Environment

Recommended: `~/miniforge3/envs/camera`

### Python Package Installation

```bash
pip install -e .
```

This registers the `prism-collect`, `prism-reconstruct-trials`, `prism-analyze-trajectory`, `prism-rebuild-aligned`, and `prism-hand` CLI commands.

### Core Python Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, triangulation, SVD |
| `opencv-python` | Image processing, video I/O |
| `matplotlib` | Live 3D plot, trajectory visualization |
| `PyYAML` | Config file parsing |
| `rich` | Console formatting and progress display |
| `ultralytics` | YOLOv8 model inference (optional, for `yolo` / `hybrid` backend) |
| `pyrealsense2` | RealSense D435 capture (optional) |

### Isaac Sim Pipeline

Run with the Isaac Sim Python interpreter:

```bash
/isaac-sim/python.sh simulation/scripts/run_replay_pipeline.py \
  --trial-dir data/raw/task_xxx/trial_000001 \
  --calib-json configs/devices/charuco_4cam_result.json
```

---

## Known Limitations

- **RPi serial command reading** is partially implemented; the full closed-loop RPi I/O pipeline is not yet complete.
- **Real-time hand feedback** (streaming finger state back into PRISM logs) is not yet integrated.
- **Temporal calibration tool** (`calibrate_temporal_delay.py`) is deprecated — hardware triggering renders it obsolete.
