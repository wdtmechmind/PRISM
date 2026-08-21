#!/usr/bin/env python3
"""Task-driven simulation collection with raw-like PRISM trial outputs.

This script creates specific task trajectories (currently pick_place), writes
raw-style trial artifacts under data/raw, and can optionally execute the
planned motion in Isaac Sim replay for visual validation/video export.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from prism.devices.hand.socket_client import GESTURE_ID_TO_POSE, POSE_TO_GESTURE_ID  # noqa: E402
from simulation.common.geometry import mat_to_quat_wxyz, rotation_zyx  # noqa: E402


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


def parse_xyz(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    return arr


def default_isaac_python() -> str:
    isaac_python = Path("/isaac-sim/python.sh")
    if isaac_python.exists():
        return str(isaac_python)
    return sys.executable


CAMERA_PRESETS = {
    "overhead_front": {
        "pos": [0.0, 0.2, 0.0],
        "look_at": [0.0, 0.0, -0.9],
    },
    "overhead_back": {
        "pos": [0.0, -0.2, 0.0],
        "look_at": [0.0, 0.0, -0.9],
    },
    "side_left": {
        "pos": [0.25, 0.0, -0.05],
        "look_at": [0.0, 0.0, -0.9],
    },
    "side_right": {
        "pos": [-0.25, 0.0, -0.05],
        "look_at": [0.0, 0.0, -0.9],
    },
    "diag_front_left": {
        "pos": [0.20, 0.18, -0.04],
        "look_at": [0.0, 0.0, -0.9],
    },
    "diag_front_right": {
        "pos": [-0.20, 0.18, -0.04],
        "look_at": [0.0, 0.0, -0.9],
    },
}


def _clip_duration_scale(scale: float) -> float:
    return max(0.2, float(scale))


def sample_trial_namespace(args: argparse.Namespace, rng: np.random.Generator) -> argparse.Namespace:
    trial_ns = argparse.Namespace(**vars(args))

    def jitter_xyz(base_xyz: Sequence[float], xy_jitter: float, z_jitter: float) -> List[float]:
        base = np.asarray(base_xyz, dtype=np.float64).reshape(3)
        base[0] += rng.uniform(-float(xy_jitter), float(xy_jitter))
        base[1] += rng.uniform(-float(xy_jitter), float(xy_jitter))
        base[2] += rng.uniform(-float(z_jitter), float(z_jitter))
        return [float(base[0]), float(base[1]), float(base[2])]

    if bool(args.randomize_per_trial):
        trial_ns.home_xyz = jitter_xyz(args.home_xyz, args.home_jitter_xy, args.home_jitter_z)
        trial_ns.pick_xyz = jitter_xyz(args.pick_xyz, args.pick_jitter_xy, args.pick_jitter_z)
        trial_ns.place_xyz = jitter_xyz(args.place_xyz, args.place_jitter_xy, args.place_jitter_z)

        duration_ratio = max(0.0, float(args.duration_jitter_ratio))
        if duration_ratio > 0.0:
            scale = _clip_duration_scale(1.0 + rng.uniform(-duration_ratio, duration_ratio))
            trial_ns.move_sec = float(args.move_sec) * scale
            trial_ns.approach_sec = float(args.approach_sec) * scale
            trial_ns.grasp_sec = float(args.grasp_sec) * scale
            trial_ns.lift_sec = float(args.lift_sec) * scale
            trial_ns.transfer_sec = float(args.transfer_sec) * scale
            trial_ns.release_sec = float(args.release_sec) * scale

        yaw_jitter = max(0.0, float(args.yaw_jitter_deg))
        if yaw_jitter > 0.0:
            trial_ns.yaw_deg = float(args.yaw_deg) + rng.uniform(-yaw_jitter, yaw_jitter)

    return trial_ns


def build_camera_views(args: argparse.Namespace) -> List[dict]:
    selected = [str(name).strip() for name in (args.camera_preset or []) if str(name).strip()]
    views: List[dict] = []
    for name in selected:
        preset = CAMERA_PRESETS.get(name)
        if preset is None:
            raise SystemExit("unknown --camera-preset: %s (choices: %s)" % (name, ", ".join(sorted(CAMERA_PRESETS.keys()))))
        views.append({
            "name": name,
            "pos": list(float(v) for v in preset["pos"]),
            "look_at": list(float(v) for v in preset["look_at"]),
        })
    if views:
        return views
    return [{
        "name": str(args.camera_name).strip(),
        "pos": list(float(v) for v in args.camera_pos),
        "look_at": list(float(v) for v in args.camera_look_at),
    }]


def build_pick_place_segments(args: argparse.Namespace) -> List[dict]:
    pick = parse_xyz(args.pick_xyz)
    place = parse_xyz(args.place_xyz)
    home = parse_xyz(args.home_xyz)
    z_offset = max(0.02, float(args.approach_offset_z))

    pre_pick = pick + np.array([0.0, 0.0, z_offset], dtype=np.float64)
    pre_place = place + np.array([0.0, 0.0, z_offset], dtype=np.float64)
    lift = pick + np.array([0.0, 0.0, max(z_offset, float(args.lift_extra_z))], dtype=np.float64)

    return [
        {"name": "home", "kind": "move", "start": home, "end": pre_pick, "duration": float(args.move_sec), "pose": "five_open"},
        {"name": "approach_pick", "kind": "move", "start": pre_pick, "end": pick, "duration": float(args.approach_sec), "pose": "five_open"},
        {"name": "grasp", "kind": "hold", "point": pick, "duration": float(args.grasp_sec), "pose": "grasp"},
        {"name": "lift", "kind": "move", "start": pick, "end": lift, "duration": float(args.lift_sec), "pose": "grasp"},
        {"name": "transfer", "kind": "move", "start": lift, "end": pre_place + np.array([0.0, 0.0, float(args.lift_extra_z)], dtype=np.float64), "duration": float(args.transfer_sec), "pose": "grasp"},
        {"name": "approach_place", "kind": "move", "start": pre_place, "end": place, "duration": float(args.approach_sec), "pose": "grasp"},
        {"name": "release", "kind": "hold", "point": place, "duration": float(args.release_sec), "pose": "five_open"},
        {"name": "retreat", "kind": "move", "start": place, "end": pre_place, "duration": float(args.move_sec), "pose": "five_open"},
        {"name": "return_home", "kind": "move", "start": pre_place, "end": home, "duration": float(args.move_sec), "pose": "five_open"},
    ]


def sample_segments(segments: Sequence[dict], fps: float, yaw_deg: float) -> Tuple[List[dict], List[dict]]:
    samples: List[dict] = []
    gesture_events: List[dict] = []
    t_cur = 0.0
    dt = 1.0 / max(1e-6, float(fps))

    for seg in segments:
        pose_name = str(seg["pose"])
        if not gesture_events or gesture_events[-1]["pose"] != pose_name:
            gesture_events.append({"t_sec": t_cur, "pose": pose_name, "phase": str(seg["name"])})

        duration = max(dt, float(seg["duration"]))
        n = max(2, int(round(duration * fps)))
        for i in range(n):
            alpha = float(i) / float(max(1, n - 1))
            if seg["kind"] == "hold":
                pos = np.asarray(seg["point"], dtype=np.float64)
            else:
                a = np.asarray(seg["start"], dtype=np.float64)
                b = np.asarray(seg["end"], dtype=np.float64)
                pos = (1.0 - alpha) * a + alpha * b
            rot = rotation_zyx(180.0, 0.0, float(yaw_deg))
            quat = mat_to_quat_wxyz(rot)
            samples.append({
                "t_sec": t_cur,
                "phase": str(seg["name"]),
                "pose": pose_name,
                "pos": pos,
                "rot": rot,
                "quat": quat,
            })
            t_cur += dt

    return samples, gesture_events


def write_corrected_trajectory(path: Path, samples: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_sec",
        "x_m", "y_m", "z_m",
        "qw", "qx", "qy", "qz",
        "roll_deg", "pitch_deg", "yaw_deg",
        "source_x_m", "source_y_m", "source_z_m",
        "source_roll_deg", "source_pitch_deg", "source_yaw_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in samples:
            pos = item["pos"]
            quat = item["quat"]
            writer.writerow({
                "t_sec": _fmt(item["t_sec"], 9),
                "x_m": _fmt(pos[0], 9), "y_m": _fmt(pos[1], 9), "z_m": _fmt(pos[2], 9),
                "qw": _fmt(quat[0], 12), "qx": _fmt(quat[1], 12), "qy": _fmt(quat[2], 12), "qz": _fmt(quat[3], 12),
                "roll_deg": _fmt(180.0, 9), "pitch_deg": _fmt(0.0, 9), "yaw_deg": _fmt(0.0, 9),
                "source_x_m": _fmt(pos[0], 9), "source_y_m": _fmt(pos[1], 9), "source_z_m": _fmt(pos[2], 9),
                "source_roll_deg": _fmt(180.0, 9), "source_pitch_deg": _fmt(0.0, 9), "source_yaw_deg": _fmt(0.0, 9),
            })


def write_gesture_csv(path: Path, events: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HAND_COMMAND_LOG_HEADER)
        t0 = time.time()
        for event in events:
            pose = str(event["pose"])
            gid = int(POSE_TO_GESTURE_ID.get(pose, 0))
            action = "%02d:%s" % (gid, pose) if gid > 0 else pose
            command = "@ROG<%d>&" % max(0, gid - 1) if gid > 0 else ""
            wall = t0 + float(event["t_sec"])
            writer.writerow([
                _fmt(event["t_sec"], 6),
                _fmt(wall, 6),
                _fmt(event["t_sec"], 6),
                action,
                command,
                "ok",
                "phase=%s" % str(event["phase"]),
            ])


def write_phase_csv(path: Path, samples: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_sec", "phase", "hand_pose"])
        for item in samples:
            writer.writerow([_fmt(item["t_sec"], 6), str(item["phase"]), str(item["pose"])])


def write_trial_raw_like(
    trial_dir: Path,
    samples: Sequence[dict],
    events: Sequence[dict],
    position_noise_mm: float,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    trajectory_dir = trial_dir / "trajectory"
    hand_dir = trial_dir / "hand"
    logs_dir = trial_dir / "logs"
    cameras_dir = trial_dir / "cameras"
    for folder in [trajectory_dir, hand_dir, logs_dir, cameras_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    trial_start = time.time()
    traj_path = trajectory_dir / "trajectory_led.csv"
    rigid_path = trajectory_dir / "rigid_pose_6d.csv"
    hand_sdk_path = hand_dir / "sdk_commands.csv"

    with traj_path.open("w", newline="", encoding="utf-8") as traj_file, \
            rigid_path.open("w", newline="", encoding="utf-8") as rigid_file:
        traj_writer = csv.writer(traj_file)
        rigid_writer = csv.writer(rigid_file)
        traj_writer.writerow(TRAJ_HEADER)
        rigid_writer.writerow(RIGID_HEADER)

        for idx, item in enumerate(samples):
            t_sec = float(item["t_sec"])
            pos = np.asarray(item["pos"], dtype=np.float64)
            rot = np.asarray(item["rot"], dtype=np.float64)
            wall = trial_start + t_sec

            if position_noise_mm > 0.0:
                pos = pos + (float(position_noise_mm) / 1000.0) * rng.normal(0.0, 1.0, size=3)

            rigid_writer.writerow([
                _fmt(t_sec, 6), _fmt(t_sec, 6), _fmt(wall, 6), idx, "measured",
                4, "red,yellow,blue,green", "red,yellow,blue,green",
                _fmt(pos[0], 9), _fmt(pos[1], 9), _fmt(pos[2], 9),
                _fmt(180.0, 6), _fmt(0.0, 6), _fmt(0.0, 6),
                _fmt(pos[0], 9), _fmt(pos[1], 9), _fmt(pos[2], 9),
                _fmt(180.0, 6), _fmt(0.0, 6), _fmt(0.0, 6),
            ])

            for color in LED_ORDER:
                point = pos + rot @ LED_OFFSETS_BODY[color]
                traj_writer.writerow([
                    _fmt(t_sec, 6), _fmt(t_sec, 6), _fmt(wall, 6), idx, color,
                    _fmt(point[0], 9), _fmt(point[1], 9), _fmt(point[2], 9),
                    "measured", 4, _fmt(0.0015, 6), "0,1,2,3",
                    _fmt(point[0], 9), _fmt(point[1], 9), _fmt(point[2], 9),
                ])

    write_gesture_csv(hand_sdk_path, events)
    (hand_dir / "rpi_commands.csv").write_text("", encoding="utf-8")
    (hand_dir / "hand_feedback.csv").write_text("", encoding="utf-8")

    with (logs_dir / "fps_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wall_time", "trial_time", "hik0_fps", "hik1_fps", "hik2_fps", "hik3_fps", "rs_fps"])
        writer.writerow([_fmt(trial_start, 6), "0.000000", "30.000", "30.000", "30.000", "30.000", "30.000"])

    end_trial_time = float(samples[-1]["t_sec"]) if samples else 0.0
    return {
        "trial_start_wall": float(trial_start),
        "trial_end_wall": float(trial_start + end_trial_time),
        "duration_sec": float(end_trial_time),
    }


def write_camera_timestamps(path: Path, samples: Sequence[dict], trial_start_wall: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "capture_wall_time", "trial_time", "device_frame_num"])
        for idx, item in enumerate(samples):
            t_sec = float(item["t_sec"])
            writer.writerow([idx, _fmt(trial_start_wall + t_sec, 6), _fmt(t_sec, 6), idx])


def run_command(label: str, command: List[str], dry_run: bool) -> None:
    print("\n[sim-task] %s" % label)
    print("[sim-task] %s" % " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(_REPO_ROOT), check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-type", choices=["pick_place"], default="pick_place")
    parser.add_argument("--task-name", default="pick_place_sim")
    parser.add_argument("--output-root", default=str(_REPO_ROOT / "data" / "raw"))
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)

    parser.add_argument("--home-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.45], metavar=("X", "Y", "Z"))
    parser.add_argument("--pick-xyz", type=float, nargs=3, default=[0.08, -0.10, 0.30], metavar=("X", "Y", "Z"))
    parser.add_argument("--place-xyz", type=float, nargs=3, default=[-0.10, 0.12, 0.30], metavar=("X", "Y", "Z"))
    parser.add_argument("--approach-offset-z", type=float, default=0.10)
    parser.add_argument("--lift-extra-z", type=float, default=0.12)

    parser.add_argument("--move-sec", type=float, default=1.2)
    parser.add_argument("--approach-sec", type=float, default=0.7)
    parser.add_argument("--grasp-sec", type=float, default=0.5)
    parser.add_argument("--lift-sec", type=float, default=0.8)
    parser.add_argument("--transfer-sec", type=float, default=1.4)
    parser.add_argument("--release-sec", type=float, default=0.5)

    parser.add_argument("--position-noise-mm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    parser.add_argument("--isaac-python", default=default_isaac_python())
    parser.add_argument("--skip-planning", action="store_true", help="only write task/trial collection files, do not run plan_robot_motion")
    parser.add_argument("--execute-replay", action="store_true", help="run replay_planned_motion in Isaac Sim after planning")
    parser.add_argument("--record-video", action="store_true", help="with --execute-replay, export overhead video")
    parser.add_argument("--camera-name", default="sim_overhead", help="recorded camera basename under trial cameras/ when --record-video")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.2, 0.0], metavar=("X", "Y", "Z"),
                        help="camera position for single-view recording")
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, -0.9], metavar=("X", "Y", "Z"),
                        help="camera look-at target for single-view recording")
    parser.add_argument("--camera-preset", action="append", default=None,
                        help="repeatable preset for multi-view recording; one of: %s" % ", ".join(sorted(CAMERA_PRESETS.keys())))
    parser.add_argument("--task-object", dest="task_object", action=argparse.BooleanOptionalAction, default=True,
                        help="render a task object in replay videos")
    parser.add_argument("--task-object-spawn-z", type=float, default=None,
                        help="optional spawn Z for replay task object center; default auto-places on tabletop")
    parser.add_argument("--task-object-spawn-xy-range", type=float, nargs=2, default=[0.18, 0.12], metavar=("X", "Y"),
                        help="uniform spawn range (+/-X, +/-Y) for replay task object")
    parser.add_argument("--task-object-seed", type=int, default=None,
                        help="optional RNG seed for replay task object placement")
    parser.add_argument("--task-object-follow-grasp", dest="task_object_follow_grasp", action=argparse.BooleanOptionalAction, default=False,
                        help="make replay task object follow hand during grasp poses")
    parser.add_argument("--task-object-offset", type=float, nargs=3, default=[0.0, 0.0, -0.06], metavar=("X", "Y", "Z"),
                        help="offset used when --task-object-follow-grasp is enabled")
    parser.add_argument("--task-object-size", type=float, default=0.045,
                        help="cube edge length for replay task object")
    parser.add_argument("--task-object-color", type=float, nargs=3, default=[0.95, 0.35, 0.10], metavar=("R", "G", "B"),
                        help="RGB color for replay task object")
    parser.add_argument("--headless", action="store_true", help="with --execute-replay, run Isaac headless")
    parser.add_argument("--strict-replay", action="store_true", help="fail the whole trial if replay/video capture fails")
    parser.add_argument("--randomize-per-trial", action="store_true", help="randomize waypoints/timing per trial for diverse batch collection")
    parser.add_argument("--home-jitter-xy", type=float, default=0.0, help="uniform jitter range (+/-m) on home x/y")
    parser.add_argument("--home-jitter-z", type=float, default=0.0, help="uniform jitter range (+/-m) on home z")
    parser.add_argument("--pick-jitter-xy", type=float, default=0.0, help="uniform jitter range (+/-m) on pick x/y")
    parser.add_argument("--pick-jitter-z", type=float, default=0.0, help="uniform jitter range (+/-m) on pick z")
    parser.add_argument("--place-jitter-xy", type=float, default=0.0, help="uniform jitter range (+/-m) on place x/y")
    parser.add_argument("--place-jitter-z", type=float, default=0.0, help="uniform jitter range (+/-m) on place z")
    parser.add_argument("--yaw-jitter-deg", type=float, default=0.0, help="uniform jitter range (+/-deg) around yaw")
    parser.add_argument("--duration-jitter-ratio", type=float, default=0.0,
                        help="uniform jitter ratio for segment durations (e.g. 0.2 => scale in [0.8,1.2])")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.num_trials = max(1, int(args.num_trials))
    args.fps = max(2.0, float(args.fps))
    if args.record_video and not args.execute_replay:
        raise SystemExit("--record-video requires --execute-replay")
    if args.duration_jitter_ratio < 0.0:
        raise SystemExit("--duration-jitter-ratio must be >= 0")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    camera_views = build_camera_views(args)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    task_dir = output_root / ("task_%s_%s" % (stamp, sanitize_task_name(args.task_name)))
    task_dir.mkdir(parents=True, exist_ok=False)

    write_metadata(
        task_dir / "task_metadata.yaml",
        [
            ("task_name", args.task_name),
            ("session_timestamp", stamp),
            ("output_root", str(task_dir)),
            ("planned_trials", int(args.num_trials)),
            ("task_type", args.task_type),
            ("generator", "simulation/scripts/collect_sim_task.py"),
            ("fps", float(args.fps)),
            ("pick_xyz", list(float(v) for v in args.pick_xyz)),
            ("place_xyz", list(float(v) for v in args.place_xyz)),
            ("camera_views", camera_views),
            ("randomize_per_trial", bool(args.randomize_per_trial)),
        ],
    )

    timeline_path = task_dir / "hand_sdk_commands_timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as timeline_file:
        timeline_writer = csv.writer(timeline_file)
        timeline_writer.writerow(["t_sec", "wall_time", "trial_id", "trial_time", "action", "command", "status", "message", "recording"])

        for trial_id in range(1, int(args.num_trials) + 1):
            trial_dir = task_dir / ("trial_%06d" % trial_id)
            processed_trial = _REPO_ROOT / "data" / "processed" / "simulation" / task_dir.name / trial_dir.name
            corrected_path = processed_trial / "corrected_trajectory.csv"
            phase_path = processed_trial / "task_phases.csv"
            trial_cfg = sample_trial_namespace(args, rng)

            if args.task_type != "pick_place":
                raise SystemExit("unsupported task type: %s" % args.task_type)
            segments = build_pick_place_segments(trial_cfg)
            samples, events = sample_segments(segments, fps=float(trial_cfg.fps), yaw_deg=float(trial_cfg.yaw_deg))

            write_corrected_trajectory(corrected_path, samples)
            write_phase_csv(phase_path, samples)
            trial_timing = write_trial_raw_like(
                trial_dir=trial_dir,
                samples=samples,
                events=events,
                position_noise_mm=float(args.position_noise_mm),
                seed=int(args.seed) + trial_id,
            )

            gesture_path = trial_dir / "hand" / "sdk_commands.csv"
            planned_path = processed_trial / "planned_motion.csv"
            captured_videos: List[str] = []
            captured_timestamps: List[str] = []

            plan_cmd = [
                args.isaac_python,
                "simulation/scripts/plan_robot_motion.py",
                "--corrected-trajectory", str(corrected_path),
                "--gestures", str(gesture_path),
                "--robot-config", str(Path(args.robot_config).expanduser().resolve()),
                "--out", str(planned_path),
                "--no-use-orientation",
            ]
            if not args.skip_planning:
                try:
                    run_command("plan trial_%06d" % trial_id, plan_cmd, args.dry_run)
                except subprocess.CalledProcessError as exc:
                    raise SystemExit(
                        "planning failed for trial_%06d (exit=%s). "
                        "Try --isaac-python /isaac-sim/python.sh or use --skip-planning to only collect task data."
                        % (trial_id, exc.returncode)
                    ) from exc
            else:
                print("\n[sim-task] planning skipped for trial_%06d (--skip-planning)" % trial_id)

            if args.execute_replay:
                if args.skip_planning:
                    raise SystemExit("--execute-replay requires planning; remove --skip-planning")
                camera_dir = trial_dir / "cameras"
                camera_dir.mkdir(parents=True, exist_ok=True)
                processed_camera_dir = processed_trial / "cameras"
                processed_camera_dir.mkdir(parents=True, exist_ok=True)
                replay_overall_ok = True
                for view in camera_views:
                    view_name = str(view["name"]).strip() or "sim_overhead"
                    video_out = camera_dir / ("%s.mp4" % view_name)
                    timestamp_out = camera_dir / ("%s_timestamps.csv" % view_name)
                    replay_cmd = [
                        args.isaac_python,
                        "simulation/scripts/replay_planned_motion.py",
                        "--planned-motion", str(planned_path),
                        "--robot-config", str(Path(args.robot_config).expanduser().resolve()),
                        "--max-frames", "0",
                        "--stride", "1",
                        "--camera-pos", *(_fmt(float(v), 9) for v in view["pos"]),
                        "--camera-look-at", *(_fmt(float(v), 9) for v in view["look_at"]),
                        ("--task-object" if bool(args.task_object) else "--no-task-object"),
                        "--task-object-size", _fmt(float(args.task_object_size), 9),
                        "--task-object-spawn-xy-range", _fmt(float(args.task_object_spawn_xy_range[0]), 9), _fmt(float(args.task_object_spawn_xy_range[1]), 9),
                        "--task-object-color", _fmt(float(args.task_object_color[0]), 9), _fmt(float(args.task_object_color[1]), 9), _fmt(float(args.task_object_color[2]), 9),
                        ("--task-object-follow-grasp" if bool(args.task_object_follow_grasp) else "--no-task-object-follow-grasp"),
                        "--task-object-offset", _fmt(float(args.task_object_offset[0]), 9), _fmt(float(args.task_object_offset[1]), 9), _fmt(float(args.task_object_offset[2]), 9),
                    ]
                    if args.task_object_spawn_z is not None:
                        replay_cmd.extend(["--task-object-spawn-z", _fmt(float(args.task_object_spawn_z), 9)])
                    if args.task_object_seed is not None:
                        replay_cmd.extend(["--task-object-seed", str(int(args.task_object_seed) + int(trial_id))])
                    if args.record_video:
                        replay_cmd.extend([
                            "--record-video",
                            "--video-path", str(video_out),
                            "--video-fps", _fmt(float(args.fps), 6),
                        ])
                    if args.headless:
                        replay_cmd.extend(["--headless", "--no-preview"])
                    view_ok = True
                    try:
                        run_command("replay trial_%06d view=%s" % (trial_id, view_name), replay_cmd, args.dry_run)
                    except subprocess.CalledProcessError as exc:
                        view_ok = False
                        replay_overall_ok = False
                        if args.strict_replay:
                            raise SystemExit(
                                "replay failed for trial_%06d view=%s (exit=%s). "
                                "Use --strict-replay only when Isaac runtime is stable on this machine."
                                % (trial_id, view_name, exc.returncode)
                            ) from exc
                        print(
                            "[sim-task] WARN: replay failed for trial_%06d view=%s (exit=%s); collection outputs kept."
                            % (trial_id, view_name, exc.returncode)
                        )
                    if args.record_video and view_ok and not args.dry_run and video_out.is_file() and video_out.stat().st_size > 0:
                        write_camera_timestamps(timestamp_out, samples, float(trial_timing["trial_start_wall"]))
                        processed_video_out = processed_camera_dir / video_out.name
                        processed_timestamp_out = processed_camera_dir / timestamp_out.name
                        shutil.copy2(video_out, processed_video_out)
                        shutil.copy2(timestamp_out, processed_timestamp_out)
                        captured_videos.append(str(video_out.relative_to(trial_dir)))
                        captured_timestamps.append(str(timestamp_out.relative_to(trial_dir)))
                    elif args.record_video and not args.dry_run:
                        replay_overall_ok = False
                        print("[sim-task] WARN: camera video not available/empty for trial_%06d view=%s" % (trial_id, view_name))

            base_wall = time.time()
            for event in events:
                pose = str(event["pose"])
                gid = int(POSE_TO_GESTURE_ID.get(pose, 0))
                action = "%02d:%s" % (gid, pose) if gid > 0 else pose
                command = "@ROG<%d>&" % max(0, gid - 1) if gid > 0 else ""
                wall = base_wall + float(event["t_sec"])
                timeline_writer.writerow([
                    _fmt(float(event["t_sec"]) + (trial_id - 1) * 1000.0, 6),
                    _fmt(wall, 6),
                    trial_id,
                    _fmt(float(event["t_sec"]), 6),
                    action,
                    command,
                    "ok",
                    "phase=%s" % str(event["phase"]),
                    1,
                ])

            write_metadata(
                trial_dir / "metadata.yaml",
                [
                    ("task_name", args.task_name),
                    ("task_type", args.task_type),
                    ("trial_id", trial_id),
                    ("status", "stopped"),
                    ("start_wall_time", float(trial_timing["trial_start_wall"])),
                    ("end_wall_time", float(trial_timing["trial_end_wall"])),
                    ("duration_sec", float(trial_timing["duration_sec"])),
                    ("sdk_commands_log", "hand/sdk_commands.csv"),
                    ("camera_video", captured_videos[0] if captured_videos else ""),
                    ("camera_timestamps", captured_timestamps[0] if captured_timestamps else ""),
                    ("camera_videos", captured_videos),
                    ("camera_timestamp_files", captured_timestamps),
                    ("processed_camera_videos", [str((processed_trial / path).relative_to(processed_trial)) for path in captured_videos]),
                    ("processed_camera_timestamps", [str((processed_trial / path.replace(".mp4", "_timestamps.csv")).relative_to(processed_trial)) for path in captured_videos]),
                    ("phase_csv", str(phase_path)),
                    ("corrected_trajectory", str(corrected_path)),
                    ("planned_motion", str(planned_path)),
                    ("trial_home_xyz", [float(v) for v in trial_cfg.home_xyz]),
                    ("trial_pick_xyz", [float(v) for v in trial_cfg.pick_xyz]),
                    ("trial_place_xyz", [float(v) for v in trial_cfg.place_xyz]),
                    ("trial_yaw_deg", float(trial_cfg.yaw_deg)),
                ],
            )

            print(
                "[sim-task] trial_%06d task=%s frames=%d out=%s"
                % (trial_id, args.task_type, len(samples), trial_dir)
            )

    print("\n[sim-task] done")
    print("[sim-task] task dir: %s" % task_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())