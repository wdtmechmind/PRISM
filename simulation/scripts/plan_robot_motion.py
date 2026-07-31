#!/usr/bin/env python3
"""Plan AUBO i5 + MechHand joint targets from corrected-frame trajectory data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from prism.devices.hand.joint_config import ALL_JOINT_NAMES, GESTURE_JOINT_CONFIGS, interpolate_configs  # noqa: E402
from prism.devices.hand.socket_client import GESTURE_ID_TO_POSE, POSE_TO_GESTURE_ID  # noqa: E402
from simulation.common.urdf_kinematics import inverse_transform, load_serial_chain, pose_matrix  # noqa: E402


BLEND_DURATION = 0.25


class GestureTimeline:
    def __init__(self, events: List[Tuple[float, str]], default_pose: str):
        self.events = sorted(events, key=lambda item: item[0])
        self.default_pose = default_pose if default_pose in GESTURE_JOINT_CONFIGS else "five_open"
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.current_pose = self.default_pose
        self.start_cfg = dict(GESTURE_JOINT_CONFIGS[self.default_pose])
        self.target_cfg = dict(GESTURE_JOINT_CONFIGS[self.default_pose])
        self.blend_start = -1.0

    def _eval(self, t_sec: float) -> dict:
        if self.blend_start < 0.0:
            return dict(self.target_cfg)
        alpha = min(1.0, max(0.0, (float(t_sec) - self.blend_start) / BLEND_DURATION))
        return interpolate_configs(self.start_cfg, self.target_cfg, alpha)

    def sample(self, t_sec: float) -> Tuple[str, dict]:
        while self.index < len(self.events) and self.events[self.index][0] <= t_sec:
            _event_t, pose = self.events[self.index]
            if pose in GESTURE_JOINT_CONFIGS and pose != self.current_pose:
                self.start_cfg = self._eval(t_sec)
                self.target_cfg = dict(GESTURE_JOINT_CONFIGS[pose])
                self.blend_start = float(t_sec)
                self.current_pose = pose
            self.index += 1
        return self.current_pose, self._eval(t_sec)


def resolve_default_out(corrected_trajectory: Path) -> Path:
    return corrected_trajectory.parent / "planned_motion.csv"


def load_robot_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "combined_urdf" in data:
        data["combined_urdf"] = str((_REPO_ROOT / data["combined_urdf"]).resolve()) if not os.path.isabs(str(data["combined_urdf"])) else data["combined_urdf"]
    return data


def load_corrected_trajectory(path: Path, max_frames: int = 0, stride: int = 1) -> List[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if stride > 1 and index % stride != 0:
                continue
            rows.append({
                "t_sec": float(row["t_sec"]),
                "pos": np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], dtype=np.float64),
                "quat": np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])], dtype=np.float64),
            })
            if max_frames > 0 and len(rows) >= max_frames:
                break
    if not rows:
        raise SystemExit("no corrected trajectory rows found: %s" % path)
    return rows


def parse_pose_name(row: dict) -> str:
    pose = str(row.get("action", "")).strip()
    if ":" in pose:
        pose = pose.rsplit(":", 1)[1].strip()
    if pose in GESTURE_JOINT_CONFIGS:
        return pose
    command = str(row.get("command", "")).strip()
    if command.startswith("@ROG<") and command.endswith(">&"):
        try:
            rog = int(command.replace("@ROG<", "").replace(">&", ""))
            return GESTURE_ID_TO_POSE.get(rog + 1, "")
        except ValueError:
            return ""
    return ""


def load_gesture_events(path: Optional[Path]) -> List[Tuple[float, str]]:
    if path is None or not path.is_file():
        return []
    events = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        time_col = "trial_time" if "trial_time" in cols else "t_sec"
        for row in reader:
            pose = parse_pose_name(row)
            if not pose:
                continue
            try:
                events.append((float(row[time_col]), pose))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(events, key=lambda item: item[0])


def prefixed_hand_config(config: dict, prefix: str) -> dict:
    return {prefix + name: float(config.get(name, 0.0)) for name in ALL_JOINT_NAMES}


def write_plan(path: Path, rows: List[dict], arm_joints: List[str], hand_joints: List[str]) -> None:
    fieldnames = [
        "t_sec",
        "target_hand_x_m", "target_hand_y_m", "target_hand_z_m",
        "target_hand_qw", "target_hand_qx", "target_hand_qy", "target_hand_qz",
        "target_tool_x_m", "target_tool_y_m", "target_tool_z_m",
        "hand_pose_name", "hand_gesture_id", "hand_rog",
    ] + arm_joints + hand_joints + ["ik_success", "ik_error_m", "ik_rot_error_rad", "ik_iters"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrected-trajectory", required=True)
    parser.add_argument("--gestures", default=None, help="sdk_commands.csv; uses trial_time when available")
    parser.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--use-orientation", action="store_true", help="also minimize corrected-frame orientation during IK")
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    corrected_path = Path(args.corrected_trajectory).expanduser().resolve()
    config = load_robot_config(Path(args.robot_config).expanduser().resolve())
    out_path = Path(args.out).expanduser().resolve() if args.out else resolve_default_out(corrected_path)
    trajectory = load_corrected_trajectory(corrected_path, max_frames=args.max_frames, stride=max(1, args.stride))
    gesture_events = load_gesture_events(Path(args.gestures).expanduser().resolve() if args.gestures else None)

    arm_joints = list(config["arm_joints"])
    hand_joints = list(config["hand_joints"])
    hand_prefix = str(config.get("hand_joint_prefix", "mechhand_"))
    default_pose = str(config.get("default_hand_pose", "five_open"))
    chain = load_serial_chain(config["combined_urdf"], config["base_link"], config["tool_link"], arm_joints)

    world_from_base_cfg = config.get("world_from_base", {}) or {}
    world_from_base = pose_matrix(
        world_from_base_cfg.get("xyz", [0.0, 0.0, 0.0]),
        world_from_base_cfg.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
    )
    base_from_world = inverse_transform(world_from_base)
    tool_to_hand_cfg = config.get("tool_to_hand", {}) or {}
    tool_to_hand = pose_matrix(tool_to_hand_cfg.get("xyz", [0.0, 0.0, 0.0]), tool_to_hand_cfg.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
    hand_to_tool = inverse_transform(tool_to_hand)
    gesture_timeline = GestureTimeline(gesture_events, default_pose=default_pose)

    q = np.asarray(config.get("default_arm_q", [0.0] * len(arm_joints)), dtype=np.float64).reshape(len(arm_joints))
    orientation_weight = float(args.orientation_weight) if args.use_orientation else 0.0
    planned_rows = []
    ok_count = 0
    errors = []
    for item in trajectory:
        hand_target = pose_matrix(item["pos"], item["quat"])
        tool_target = hand_target @ hand_to_tool
        tool_target_base = base_from_world @ tool_target
        q, success, pos_err, rot_err, iters = chain.solve_ik(
            tool_target_base,
            q,
            orientation_weight=orientation_weight,
            max_iters=args.max_iters,
            tolerance=args.tolerance,
        )
        if success:
            ok_count += 1
        errors.append(pos_err)
        pose_name, hand_cfg = gesture_timeline.sample(item["t_sec"])
        hand_targets = prefixed_hand_config(hand_cfg, hand_prefix)
        gesture_id = POSE_TO_GESTURE_ID.get(pose_name, 0)
        row = {
            "t_sec": "%.9f" % item["t_sec"],
            "target_hand_x_m": "%.9f" % item["pos"][0],
            "target_hand_y_m": "%.9f" % item["pos"][1],
            "target_hand_z_m": "%.9f" % item["pos"][2],
            "target_hand_qw": "%.12f" % item["quat"][0],
            "target_hand_qx": "%.12f" % item["quat"][1],
            "target_hand_qy": "%.12f" % item["quat"][2],
            "target_hand_qz": "%.12f" % item["quat"][3],
            "target_tool_x_m": "%.9f" % tool_target[0, 3],
            "target_tool_y_m": "%.9f" % tool_target[1, 3],
            "target_tool_z_m": "%.9f" % tool_target[2, 3],
            "hand_pose_name": pose_name,
            "hand_gesture_id": str(gesture_id),
            "hand_rog": str(gesture_id - 1 if gesture_id else ""),
            "ik_success": "1" if success else "0",
            "ik_error_m": "%.9f" % pos_err,
            "ik_rot_error_rad": "%.9f" % rot_err,
            "ik_iters": str(iters),
        }
        for name, value in zip(arm_joints, q):
            row[name] = "%.9f" % float(value)
        for name in hand_joints:
            row[name] = "%.9f" % float(hand_targets.get(name, 0.0))
        planned_rows.append(row)

    write_plan(out_path, planned_rows, arm_joints, hand_joints)
    err_arr = np.asarray(errors, dtype=np.float64)
    print("[motion-plan] wrote %s" % out_path)
    print("[motion-plan] frames: %d" % len(planned_rows))
    print("[motion-plan] gestures: %d" % len(gesture_events))
    print("[motion-plan] IK success: %d/%d (%.1f%%)" % (ok_count, len(planned_rows), 100.0 * ok_count / max(1, len(planned_rows))))
    print("[motion-plan] IK position error: mean=%.4f m max=%.4f m" % (float(np.mean(err_arr)), float(np.max(err_arr))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
