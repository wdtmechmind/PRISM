#!/usr/bin/env python3
"""Plan AUBO i5 + MechHand joint targets from corrected-frame trajectory data."""

from __future__ import annotations

import argparse
import csv
import json
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


def parse_float_list(raw: Optional[str], expected: int, what: str) -> Optional[np.ndarray]:
    if raw is None:
        return None
    parts = [p for p in raw.replace(",", " ").split() if p]
    if len(parts) != expected:
        raise SystemExit("%s needs %d numbers, got %d" % (what, expected, len(parts)))
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def pick_anchor_index(trajectory: List[dict], anchor_time: Optional[float]) -> int:
    if anchor_time is None:
        return 0
    times = np.asarray([row["t_sec"] for row in trajectory], dtype=np.float64)
    return int(np.argmin(np.abs(times - float(anchor_time))))


def load_base_from_world(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return np.asarray(data["base_from_world"], dtype=np.float64).reshape(4, 4)


def save_base_from_world(path: Path, transform: np.ndarray, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"base_from_world": transform.tolist(), "source": source}, handle, indent=2)


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def wrist_flip(q: np.ndarray) -> np.ndarray:
    """Equivalent spherical-wrist branch: same tool pose, opposite wrist configuration."""
    out = np.asarray(q, dtype=np.float64).copy()
    if out.size < 6:
        return out
    out[3] = wrap_to_pi(out[3] + np.pi)
    out[4] = -out[4]
    out[5] = wrap_to_pi(out[5] + np.pi)
    return out


def solve_ik_retry(chain, target, q_seed, orientation_weight, max_iters, tolerance, rng,
                   max_branch_jump=0.35):
    """Solve IK, re-seeding when the first branch saturates against a joint limit.

    Alternative branches are only accepted if they stay within ``max_branch_jump``
    of the seed: a wrist flip reaches the same tool pose but the arm cannot step
    across it, so continuity outranks solving the frame exactly.
    """
    q_seed = np.asarray(q_seed, dtype=np.float64)
    best = chain.solve_ik(target, q_seed, orientation_weight=orientation_weight,
                          max_iters=max_iters, tolerance=tolerance)
    if best[1]:
        return best + (0,)

    seeds = [wrist_flip(q_seed)]
    for _ in range(6):
        seeds.append(wrap_to_pi(q_seed + rng.normal(scale=0.35, size=len(q_seed))))
    for attempt, seed in enumerate(seeds, start=1):
        cand = chain.solve_ik(target, seed, orientation_weight=orientation_weight,
                              max_iters=max_iters, tolerance=tolerance)
        if float(np.abs(cand[0] - q_seed).max()) > max_branch_jump:
            continue
        if cand[1]:
            return cand + (attempt,)
        if cand[2] < best[2]:
            best = cand
    return best + (len(seeds),)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrected-trajectory", required=True)
    parser.add_argument("--gestures", default=None, help="sdk_commands.csv; uses trial_time when available")
    parser.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--anchor-mode", choices=["relative", "config"], default="relative",
                        help="relative: pin the anchor frame to a known arm pose so the unknown "
                             "cam0-to-base transform cancels; config: use world_from_base as given")
    parser.add_argument("--base-from-world", default=None,
                        help="JSON holding a shared base_from_world 4x4; reuse one across every trial "
                             "so per-trial start-pose diversity is preserved")
    parser.add_argument("--save-base-from-world", default=None,
                        help="write the anchor-derived base_from_world to this JSON for reuse")
    parser.add_argument("--anchor-time", type=float, default=None,
                        help="t_sec of the anchor frame (default: first frame); put it at the grasp "
                             "instant so accuracy peaks where it matters")
    parser.add_argument("--anchor-arm-q", default=None,
                        help="comma-separated arm joint angles (rad) the real arm is jogged to at the anchor")
    parser.add_argument("--use-orientation", dest="use_orientation", default=None, action=argparse.BooleanOptionalAction, help="also minimize corrected-frame orientation during IK")
    parser.add_argument("--orientation-weight", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-branch-jump", type=float, default=0.35,
                        help="rad; reject IK branch switches larger than this to keep the path continuous")
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
    planning_config = config.get("planning", {}) or {}
    use_orientation = bool(planning_config.get("use_orientation", False)) if args.use_orientation is None else bool(args.use_orientation)
    orientation_weight = float(args.orientation_weight if args.orientation_weight is not None else planning_config.get("orientation_weight", 0.35))
    max_iters = int(args.max_iters if args.max_iters is not None else planning_config.get("max_iters", 80))
    tolerance = float(args.tolerance if args.tolerance is not None else planning_config.get("tolerance", 0.005))
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

    if args.base_from_world:
        base_from_world = load_base_from_world(Path(args.base_from_world).expanduser().resolve())
        anchor_source = "shared:%s" % args.base_from_world
        q = np.asarray(config.get("anchor_arm_q", config.get("default_arm_q", [0.0] * len(arm_joints))),
                       dtype=np.float64).reshape(len(arm_joints))
        print("[motion-plan] base_from_world loaded from %s" % args.base_from_world)
    elif args.anchor_mode == "relative":
        q_anchor = parse_float_list(args.anchor_arm_q, len(arm_joints), "--anchor-arm-q")
        if q_anchor is None:
            cfg_anchor = config.get("anchor_arm_q")
            q_anchor = np.asarray(cfg_anchor, dtype=np.float64).reshape(len(arm_joints)) if cfg_anchor else q
        anchor_index = pick_anchor_index(trajectory, args.anchor_time)
        anchor_item = trajectory[anchor_index]
        anchor_hand_world = pose_matrix(anchor_item["pos"], anchor_item["quat"])
        anchor_hand_base = chain.fk(q_anchor) @ tool_to_hand
        # Pinning the anchor this way makes the unknown cam0->base transform drop out.
        base_from_world = anchor_hand_base @ inverse_transform(anchor_hand_world)
        q = np.asarray(q_anchor, dtype=np.float64).copy()
        anchor_source = "anchor:%s@t=%.3f" % (corrected_path.name, anchor_item["t_sec"])
        print("[motion-plan] anchor: frame %d at t=%.3fs, arm q=[%s]"
              % (anchor_index, anchor_item["t_sec"], ", ".join("%.4f" % v for v in q_anchor)))
    else:
        anchor_source = "config:world_from_base"

    if args.save_base_from_world:
        save_base_from_world(Path(args.save_base_from_world).expanduser().resolve(),
                             base_from_world, anchor_source)
        print("[motion-plan] saved base_from_world -> %s" % args.save_base_from_world)

    orientation_weight = orientation_weight if use_orientation else 0.0
    planned_rows = []
    ok_count = 0
    retry_recovered = 0
    errors = []
    rng = np.random.default_rng(0)
    for item in trajectory:
        hand_target = pose_matrix(item["pos"], item["quat"])
        tool_target = hand_target @ hand_to_tool
        tool_target_base = base_from_world @ tool_target
        q, success, pos_err, rot_err, iters, attempts = solve_ik_retry(
            chain, tool_target_base, q, orientation_weight, max_iters, tolerance, rng,
            max_branch_jump=args.max_branch_jump)
        if success:
            ok_count += 1
            if attempts:
                retry_recovered += 1
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
    print("[motion-plan] anchor mode: %s" % anchor_source)
    print("[motion-plan] gestures: %d" % len(gesture_events))
    print("[motion-plan] IK success: %d/%d (%.1f%%)" % (ok_count, len(planned_rows), 100.0 * ok_count / max(1, len(planned_rows))))
    print("[motion-plan] IK recovered by re-seeding: %d" % retry_recovered)
    print("[motion-plan] IK position error: mean=%.4f m max=%.4f m" % (float(np.mean(err_arr)), float(np.max(err_arr))))
    if ok_count < len(planned_rows):
        print("[motion-plan] WARNING: %d frames failed IK -- do not run this on the real arm"
              % (len(planned_rows) - ok_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
