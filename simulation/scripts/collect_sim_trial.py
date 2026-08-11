#!/usr/bin/env python3
"""Generate raw-like PRISM trial data from synthetic simulation trajectories.

This script is useful when Isaac replay is available but no real cameras are
connected: it creates task/trial folders under data/raw with trajectory and hand
CSV files that follow the same schema used by real collection outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import sys


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from prism.devices.hand import GESTURE_ID_TO_POSE  # noqa: E402
from simulation.common.geometry import mat_to_euler_zyx_deg, rotation_zyx  # noqa: E402


TRAJ_HEADER = [
    "t_sec", "t_trial", "capture_wall_time", "frame_index", "color",
    "x_m", "y_m", "z_m", "mode", "num_views", "max_norm_reproj_err", "visible_cams",
    "x_smooth_m", "y_smooth_m", "z_smooth_m",
]

RIGID_HEADER = [
    "t_sec", "t_trial", "capture_wall_time", "frame_index", "mode",
    "num_leds_used", "modeled_leds", "visible_leds",
    "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg",
    "x_smooth_m", "y_smooth_m", "z_smooth_m",
    "roll_smooth_deg", "pitch_smooth_deg", "yaw_smooth_deg",
]

HAND_COMMAND_LOG_HEADER = ["t_sec", "wall_time", "trial_time", "action", "command", "status", "message"]
RPI_ENCODER_LOG_HEADER = (
    ["t_sec", "wall_time", "trial_time", "action", "status", "source", "rpi_wall_time", "rpi_monotonic"]
    + ["enc%d_pos" % i for i in range(1, 6)]
    + ["enc%d_angle_deg" % i for i in range(1, 6)]
    + ["enc%d_duty" % i for i in range(1, 6)]
    + ["enc%d_pulse_width_us" % i for i in range(1, 6)]
    + ["enc%d_period_us" % i for i in range(1, 6)]
    + ["message"]
)

LED_ORDER = ["red", "yellow", "blue", "green"]
LED_OFFSETS_BODY = {
    "red": np.array([0.034, 0.016, 0.005], dtype=np.float64),
    "yellow": np.array([0.034, -0.016, 0.005], dtype=np.float64),
    "blue": np.array([-0.022, 0.020, -0.002], dtype=np.float64),
    "green": np.array([-0.022, -0.020, -0.002], dtype=np.float64),
}


def sanitize_task_name(task_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(task_name).strip()).strip("-_.")
    return cleaned or "task"


def write_metadata(path: Path, fields: Sequence[Tuple[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in fields:
            handle.write("%s: %s\n" % (key, json.dumps(value, ensure_ascii=False)))


def _fmt(value: float, digits: int = 6) -> str:
    return ("%%.%df" % int(digits)) % float(value)


def _safe_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for piece in str(raw).split(","):
        text = piece.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value in GESTURE_ID_TO_POSE:
            out.append(value)
    return out


def _build_pattern_pose(
    phase: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = args.center_xyz
    rx = float(args.radius_x)
    ry = float(args.radius_y)
    z_amp = float(args.z_amplitude)

    if args.pattern == "circle":
        x = cx + rx * math.cos(phase)
        y = cy + ry * math.sin(phase)
        z = cz + z_amp * math.sin(2.0 * phase)
        dx = -rx * math.sin(phase)
        dy = ry * math.cos(phase)
    elif args.pattern == "figure8":
        x = cx + rx * math.sin(phase)
        y = cy + 0.5 * ry * math.sin(2.0 * phase)
        z = cz + z_amp * math.cos(phase)
        dx = rx * math.cos(phase)
        dy = ry * math.cos(2.0 * phase)
    elif args.pattern == "line":
        wave = math.sin(phase)
        x = cx + rx * wave
        y = cy + 0.2 * ry * math.sin(3.0 * phase)
        z = cz + z_amp * math.sin(2.5 * phase)
        dx = rx * math.cos(phase)
        dy = 0.6 * ry * math.cos(3.0 * phase)
    else:  # lissajous
        x = cx + rx * math.sin(2.0 * phase + 0.5)
        y = cy + ry * math.sin(3.0 * phase)
        z = cz + z_amp * math.sin(phase)
        dx = 2.0 * rx * math.cos(2.0 * phase + 0.5)
        dy = 3.0 * ry * math.cos(3.0 * phase)

    yaw_deg = math.degrees(math.atan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
    roll_deg = float(args.roll_amplitude_deg) * math.sin(1.2 * phase)
    pitch_deg = float(args.pitch_amplitude_deg) * math.cos(1.4 * phase)
    return np.array([x, y, z], dtype=np.float64), np.array([roll_deg, pitch_deg, yaw_deg], dtype=np.float64)


def _build_times(duration_sec: float, fps: float) -> np.ndarray:
    total = max(2, int(round(max(0.05, duration_sec) * max(1.0, fps))))
    return np.linspace(0.0, float(duration_sec), num=total, endpoint=True, dtype=np.float64)


def _build_gesture_events(
    duration_sec: float,
    interval_sec: float,
    gesture_seq: List[int],
    default_gesture_id: int,
) -> List[Tuple[float, int]]:
    sequence = gesture_seq if gesture_seq else [int(default_gesture_id)]
    if not sequence:
        sequence = [2]
    events: List[Tuple[float, int]] = [(0.0, int(sequence[0]))]
    if interval_sec <= 0.0:
        return events

    t = float(interval_sec)
    idx = 1
    while t < duration_sec + 1e-9:
        events.append((t, int(sequence[idx % len(sequence)])))
        idx += 1
        t += float(interval_sec)
    return events


def _gesture_at_time(events: Sequence[Tuple[float, int]], t_sec: float) -> int:
    gesture_id = int(events[0][1])
    for t_evt, gid in events:
        if t_sec + 1e-12 >= t_evt:
            gesture_id = int(gid)
        else:
            break
    return gesture_id


def _command_from_gesture(gesture_id: int) -> str:
    return "@R%dG0" % max(0, int(gesture_id) - 1)


def generate_trial(
    trial_dir: Path,
    trial_id: int,
    args: argparse.Namespace,
    task_start_wall_time: float,
    timeline_writer: csv.writer,
    rng: np.random.Generator,
) -> Dict[str, float]:
    cameras_dir = trial_dir / "cameras"
    logs_dir = trial_dir / "logs"
    trajectory_dir = trial_dir / "trajectory"
    hand_dir = trial_dir / "hand"
    for path in [cameras_dir, logs_dir, trajectory_dir, hand_dir]:
        path.mkdir(parents=True, exist_ok=True)

    trial_start_wall = float(task_start_wall_time) + float(trial_id - 1) * (float(args.duration_sec) + 0.75)
    t_values = _build_times(float(args.duration_sec), float(args.fps))
    n_frames = len(t_values)
    gesture_events = _build_gesture_events(
        duration_sec=float(args.duration_sec),
        interval_sec=float(args.gesture_interval_sec),
        gesture_seq=_parse_int_list(args.gesture_seq),
        default_gesture_id=int(args.default_gesture_id),
    )

    traj_path = trajectory_dir / "trajectory_led.csv"
    rigid_path = trajectory_dir / "rigid_pose_6d.csv"
    hand_sdk_path = hand_dir / "sdk_commands.csv"
    hand_rpi_path = hand_dir / "rpi_commands.csv"
    hand_feedback_path = hand_dir / "hand_feedback.csv"

    with traj_path.open("w", newline="", encoding="utf-8") as traj_file, \
            rigid_path.open("w", newline="", encoding="utf-8") as rigid_file, \
            hand_sdk_path.open("w", newline="", encoding="utf-8") as hand_sdk_file, \
            hand_rpi_path.open("w", newline="", encoding="utf-8") as hand_rpi_file, \
            hand_feedback_path.open("w", encoding="utf-8"):

        del hand_feedback_path
        traj_writer = csv.writer(traj_file)
        rigid_writer = csv.writer(rigid_file)
        hand_sdk_writer = csv.writer(hand_sdk_file)
        hand_rpi_writer = csv.writer(hand_rpi_file)
        traj_writer.writerow(TRAJ_HEADER)
        rigid_writer.writerow(RIGID_HEADER)
        hand_sdk_writer.writerow(HAND_COMMAND_LOG_HEADER)
        hand_rpi_writer.writerow(RPI_ENCODER_LOG_HEADER)

        last_hand_gesture = None
        for frame_index, t_sec in enumerate(t_values):
            phase = 2.0 * math.pi * (t_sec / max(float(args.duration_sec), 1e-6))
            pos, rpy = _build_pattern_pose(phase, args)

            if args.position_noise_mm > 0.0:
                pos = pos + (float(args.position_noise_mm) / 1000.0) * rng.normal(0.0, 1.0, size=3)
            if args.rpy_noise_deg > 0.0:
                rpy = rpy + float(args.rpy_noise_deg) * rng.normal(0.0, 1.0, size=3)

            rot = rotation_zyx(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            rpy = mat_to_euler_zyx_deg(rot)
            capture_wall = trial_start_wall + float(t_sec)

            rigid_writer.writerow([
                _fmt(t_sec, 6), _fmt(t_sec, 6), _fmt(capture_wall, 6), frame_index, "measured",
                4, "red,yellow,blue,green", "red,yellow,blue,green",
                _fmt(pos[0], 9), _fmt(pos[1], 9), _fmt(pos[2], 9),
                _fmt(rpy[0], 6), _fmt(rpy[1], 6), _fmt(rpy[2], 6),
                _fmt(pos[0], 9), _fmt(pos[1], 9), _fmt(pos[2], 9),
                _fmt(rpy[0], 6), _fmt(rpy[1], 6), _fmt(rpy[2], 6),
            ])

            for color in LED_ORDER:
                point = pos + rot @ LED_OFFSETS_BODY[color]
                if args.position_noise_mm > 0.0:
                    point = point + (float(args.position_noise_mm) / 2000.0) * rng.normal(0.0, 1.0, size=3)
                traj_writer.writerow([
                    _fmt(t_sec, 6), _fmt(t_sec, 6), _fmt(capture_wall, 6), frame_index, color,
                    _fmt(point[0], 9), _fmt(point[1], 9), _fmt(point[2], 9),
                    "measured", 4, _fmt(max(0.0015, float(args.position_noise_mm) * 0.00025), 6), "0,1,2,3",
                    _fmt(point[0], 9), _fmt(point[1], 9), _fmt(point[2], 9),
                ])

            gesture_id = _gesture_at_time(gesture_events, float(t_sec))
            if last_hand_gesture != gesture_id:
                pose_name = GESTURE_ID_TO_POSE.get(int(gesture_id), "unknown")
                command = _command_from_gesture(int(gesture_id))
                hand_sdk_writer.writerow([
                    _fmt(t_sec, 6), _fmt(capture_wall, 6), _fmt(t_sec, 6),
                    "%02d:%s" % (int(gesture_id), pose_name), command, "ok", "simulated",
                ])

                rpi_row = [
                    _fmt(t_sec, 6), _fmt(capture_wall, 6), _fmt(t_sec, 6),
                    "%02d:%s" % (int(gesture_id), pose_name), "ok", "sim:0", _fmt(capture_wall, 6), _fmt(t_sec, 6),
                ]
                rpi_row.extend(["" for _ in range(len(RPI_ENCODER_LOG_HEADER) - len(rpi_row) - 1)])
                rpi_row.append("simulated")
                hand_rpi_writer.writerow(rpi_row)

                timeline_writer.writerow([
                    _fmt((capture_wall - task_start_wall_time), 6),
                    _fmt(capture_wall, 6),
                    trial_id,
                    _fmt(t_sec, 6),
                    "%02d:%s" % (int(gesture_id), pose_name),
                    command,
                    "ok",
                    "simulated",
                    1,
                ])
                last_hand_gesture = gesture_id

    with (logs_dir / "fps_log.csv").open("w", newline="", encoding="utf-8") as fps_file:
        fps_writer = csv.writer(fps_file)
        fps_writer.writerow(["wall_time", "trial_time", "hik0_fps", "hik1_fps", "hik2_fps", "hik3_fps", "rs_fps"])
        fps_writer.writerow([
            _fmt(trial_start_wall, 6), "0.000000",
            _fmt(args.fps, 3), _fmt(args.fps, 3), _fmt(args.fps, 3), _fmt(args.fps, 3), _fmt(args.fps, 3),
        ])

    write_metadata(
        trial_dir / "metadata.yaml",
        [
            ("task_name", args.task_name),
            ("trial_id", trial_id),
            ("status", "stopped"),
            ("start_wall_time", trial_start_wall),
            ("end_wall_time", trial_start_wall + float(args.duration_sec)),
            ("duration_sec", float(args.duration_sec)),
            ("hand_generation", "simulated"),
            ("rpi_sdk_feedback_status", "simulated"),
            ("rpi_event_udp", {"host": "0.0.0.0", "port": 60701}),
            ("rpi_commands_log", "hand/rpi_commands.csv"),
            ("sdk_commands_log", "hand/sdk_commands.csv"),
            ("hand_feedback_log", "hand/hand_feedback.csv"),
            ("frames_written", n_frames * 5),
            ("frames_dropped", 0),
            ("sim_pattern", args.pattern),
        ],
    )
    return {
        "frames": float(n_frames),
        "start_wall": trial_start_wall,
        "end_wall": trial_start_wall + float(args.duration_sec),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="sim_collection")
    parser.add_argument("--output-root", default=str(_REPO_ROOT / "data" / "raw"), help="directory containing generated task_* folders")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pattern", choices=["circle", "figure8", "lissajous", "line"], default="figure8")
    parser.add_argument("--center-xyz", type=float, nargs=3, default=[0.00, 0.00, 0.42], metavar=("X", "Y", "Z"))
    parser.add_argument("--radius-x", type=float, default=0.10)
    parser.add_argument("--radius-y", type=float, default=0.12)
    parser.add_argument("--z-amplitude", type=float, default=0.04)
    parser.add_argument("--roll-amplitude-deg", type=float, default=8.0)
    parser.add_argument("--pitch-amplitude-deg", type=float, default=10.0)
    parser.add_argument("--position-noise-mm", type=float, default=1.2)
    parser.add_argument("--rpy-noise-deg", type=float, default=0.8)
    parser.add_argument("--gesture-interval-sec", type=float, default=1.2)
    parser.add_argument("--default-gesture-id", type=int, default=2)
    parser.add_argument("--gesture-seq", default="2,5,3,4,6", help="comma-separated gesture ids in [1,17]")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.num_trials = max(1, int(args.num_trials))
    args.duration_sec = max(0.2, float(args.duration_sec))
    args.fps = max(1.0, float(args.fps))
    args.default_gesture_id = int(args.default_gesture_id)
    if args.default_gesture_id not in GESTURE_ID_TO_POSE:
        raise SystemExit("default gesture id must be within available ids: %s" % sorted(GESTURE_ID_TO_POSE.keys()))

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_dir = output_root / ("task_%s_%s" % (timestamp, sanitize_task_name(args.task_name)))
    task_dir.mkdir(parents=True, exist_ok=False)

    task_start_wall = time.time()
    task_timeline_path = task_dir / "hand_sdk_commands_timeline.csv"
    rng = np.random.default_rng(int(args.seed))

    write_metadata(
        task_dir / "task_metadata.yaml",
        [
            ("task_name", args.task_name),
            ("session_timestamp", timestamp),
            ("output_root", str(task_dir)),
            ("config", "simulation/scripts/collect_sim_trial.py"),
            ("planned_trials", int(args.num_trials)),
            ("hand_generation", "simulated"),
            ("post_process", "none"),
            ("rpi_port", ""),
            ("rpi_event_udp", {"host": "0.0.0.0", "port": 60701}),
            ("sdk_script", ""),
            ("feedback_port", ""),
            ("rpi_sdk_feedback_status", "simulated"),
            ("hand_cli_socket", {
                "ip": "127.0.0.1",
                "port": 60686,
                "timeout_s": 3.0,
                "settle_time_s": 1.0,
                "auto_connect": "n",
            }),
            ("hik_camera_config", {
                "exposure_us": 5000.0,
                "gain": 0.0,
                "frame_rate": float(args.fps),
            }),
            ("realsense_config", {
                "serial": "",
                "width": 1280,
                "height": 720,
                "fps": float(args.fps),
                "auto_exposure": True,
                "exposure": 0.0,
                "gain": 0.0,
                "brightness": 0.0,
            }),
            ("tracking_detector", {
                "backend": "simulated",
                "yolo_weights": "",
                "yolo_conf": 0.25,
                "yolo_iou": 0.45,
                "yolo_imgsz": 640,
            }),
        ],
    )

    total_frames = 0
    with task_timeline_path.open("w", newline="", encoding="utf-8") as timeline_file:
        timeline_writer = csv.writer(timeline_file)
        timeline_writer.writerow([
            "t_sec", "wall_time", "trial_id", "trial_time", "action", "command", "status", "message", "recording"
        ])
        for trial_idx in range(1, int(args.num_trials) + 1):
            trial_dir = task_dir / ("trial_%06d" % trial_idx)
            summary = generate_trial(
                trial_dir=trial_dir,
                trial_id=trial_idx,
                args=args,
                task_start_wall_time=task_start_wall,
                timeline_writer=timeline_writer,
                rng=rng,
            )
            total_frames += int(summary["frames"])
            print(
                "[sim-collect] trial_%06d frames=%d duration=%.2fs"
                % (trial_idx, int(summary["frames"]), float(args.duration_sec))
            )

    print("[sim-collect] done")
    print("[sim-collect] task dir: %s" % task_dir)
    print("[sim-collect] total frames: %d" % total_frames)
    print("[sim-collect] trials: %d" % int(args.num_trials))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())