#!/usr/bin/env python3
"""Calibrate UR3 wrist_3 mounting offset from a cached replay trajectory.

The tool replays a cached clearance-aware joint path up to ``--pause-time`` and
then accepts interactive wrist_3-only jog commands. It writes the final delta
from the planned wrist_3 angle to ``wrist_mount_calibration.json``. Real motion
requires ``--execute``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPLAY_PATH = ROOT / "tools" / "replay_ur3_base_trajectory.py"
SPEC = importlib.util.spec_from_file_location("prism_ur3_replay", REPLAY_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--pause-time", type=float, default=None,
                        help="source trajectory pause time; omit to type p + Enter during replay")
    parser.add_argument("--time-scale", type=float, default=35.0)
    parser.add_argument("--rate-hz", type=float, default=125.0)
    parser.add_argument("--ik-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <bundle-dir>/wrist_mount_calibration.json")
    parser.add_argument("--jog-step-deg", type=float, default=1.0)
    parser.add_argument("--jog-speed", type=float, default=0.03)
    parser.add_argument("--jog-acceleration", type=float, default=0.08)
    parser.add_argument("--wrist3-lower-deg", type=float, default=-360.0)
    parser.add_argument("--wrist3-upper-deg", type=float, default=360.0)
    parser.add_argument("--movej-speed", type=float, default=0.05)
    parser.add_argument("--movej-acceleration", type=float, default=0.10)
    parser.add_argument("--servo-lookahead", type=float, default=0.10)
    parser.add_argument("--servo-gain", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    return parser


def save_calibration(path: Path, offset_rad: float, planned_q: np.ndarray,
                     calibrated_q: np.ndarray, pause_time: float, args) -> None:
    data = {
        "schema_version": 1,
        "type": "ur3_wrist3_mount_offset",
        "axis": "wrist_3/local_tool_z",
        "wrist3_offset_rad": float(offset_rad),
        "wrist3_offset_deg": float(math.degrees(offset_rad)),
        "pause_source_time_s": float(pause_time),
        "planned_wrist3_rad": float(planned_q[5]),
        "calibrated_wrist3_rad": float(calibrated_q[5]),
        "bundle_dir": str(args.bundle_dir.expanduser().resolve()),
        "robot_ip": str(args.robot_ip),
        "created_unix_time": time.time(),
        "application": "post-multiply replay TCP orientation by local tool-Z rotation",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    print("saved wrist calibration: %s" % path)
    print("wrist_3 offset: %.6f rad (%.3f deg)" % (offset_rad, math.degrees(offset_rad)))


def main() -> int:
    args = build_parser().parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    trajectory_path = bundle / "base_trajectory.csv"
    cache_path = (args.ik_cache.expanduser().resolve() if args.ik_cache is not None
                  else bundle / "ik_plan_cache.npz")
    output_path = (args.output.expanduser().resolve() if args.output is not None
                   else bundle / "wrist_mount_calibration.json")

    source_trajectory = replay.load_trajectory(trajectory_path)
    cached, cache_error = replay.load_ik_cache(cache_path, source_trajectory, args.robot_ip)
    if cached is None:
        raise SystemExit("valid IK cache required before wrist calibration: %s" % cache_error)
    source_joints, metadata = cached
    source_joints, wrist_turns, wrist_margin = replay.center_periodic_joint_path(
        source_joints, joint_index=5
    )

    print("IK cache: %s" % cache_path)
    if args.pause_time is not None:
        pause_time = float(args.pause_time)
        if pause_time < source_trajectory[0, 0] or pause_time > source_trajectory[-1, 0]:
            raise SystemExit("--pause-time must be within %.3f .. %.3f s" % (
                source_trajectory[0, 0], source_trajectory[-1, 0]))
        print("configured pause source time: %.6f s" % pause_time)
    else:
        pause_time = None
        print("interactive pause: type p + Enter during replay")
    print("cache branch clearances: %s" % metadata.get("branch_clearances_m", []))
    print("wrist_3 periodic shift: %+d x 360 deg; limit margin %.1f deg" % (
        wrist_turns, math.degrees(wrist_margin)))
    if not args.execute:
        print("DRY RUN: no robot connection or motion. Add --execute to calibrate.")
        return 0

    try:
        import rtde_control
        import rtde_receive
    except ImportError as exc:
        raise SystemExit("--execute requires ur_rtde") from exc

    control = None
    receive = None
    try:
        control = rtde_control.RTDEControlInterface(args.robot_ip)
        receive = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        current_q = np.asarray(receive.getActualQ(), dtype=np.float64)
        movej_clearance = replay.interpolated_joint_clearance(current_q, source_joints[0], 0.04)
        if movej_clearance < 0.01:
            raise RuntimeError("initial moveJ path clearance %.1f mm is below 10 mm" % (
                movej_clearance * 1000.0))

        print("moving to cached trajectory start ...", flush=True)
        control.moveJ(source_joints[0].tolist(), args.movej_speed,
                      args.movej_acceleration, False)

        if pause_time is not None:
            pause_index = int(np.searchsorted(source_trajectory[:, 0], pause_time, side="left"))
            pause_index = min(max(pause_index, 0), len(source_trajectory) - 1)
        else:
            pause_index = len(source_trajectory) - 1
        segment_times = source_trajectory[:pause_index + 1, 0]
        duration = float(segment_times[-1] - segment_times[0])
        count = max(2, int(math.ceil(duration * args.time_scale * args.rate_hz)) + 1)
        output_times = np.linspace(segment_times[0], segment_times[-1], count)
        joint_path = replay.resample_joint_trajectory(
            segment_times, source_joints[:pause_index + 1], output_times)
        dt = 1.0 / args.rate_hz
        command_queue = queue.Queue()
        if pause_time is None:
            def read_pause_command():
                while True:
                    try:
                        command_queue.put(input())
                    except EOFError:
                        return
            threading.Thread(target=read_pause_command, daemon=True).start()
            print("replaying trajectory; type p + Enter to pause ...", flush=True)
        else:
            print("replaying to pause (%d servoJ frames) ..." % len(joint_path), flush=True)
        start_wall = time.monotonic()
        paused_output_index = len(joint_path) - 1
        for index, joints in enumerate(joint_path, start=1):
            target_time = index * dt
            while time.monotonic() - start_wall < target_time:
                time.sleep(min(0.002, max(0.0001, target_time - (time.monotonic() - start_wall))))
            control.servoJ(joints.tolist(), 0.0, 0.0, dt,
                           args.servo_lookahead, args.servo_gain)
            if pause_time is None:
                try:
                    command = command_queue.get_nowait().strip().lower()
                except queue.Empty:
                    command = ''
                if command in ('p', 'pause', '') and command:
                    paused_output_index = index - 1
                    break
        control.servoStop(1.0)

        planned_pause_q = joint_path[paused_output_index].copy()
        actual_pause_time = float(output_times[paused_output_index])
        print("paused at source time %.6f s" % actual_pause_time)
        print("planned wrist_3: %.6f rad (%.3f deg)" % (
            planned_pause_q[5], math.degrees(planned_pause_q[5])))

        print("paused. Only wrist_3 will be jogged; joints 1-5 remain fixed.")
        print("Commands: + [deg], - [deg], set <deg>, show, save, cancel")
        target_q = np.asarray(receive.getActualQ(), dtype=np.float64)
        while True:
            if args.pause_time is None:
                print("wrist-calib> ", end="", flush=True)
                raw = command_queue.get().strip().lower()
            else:
                raw = input("wrist-calib> ").strip().lower()
            if raw in ("show", ""):
                delta = wrap_angle(target_q[5] - planned_pause_q[5])
                print("wrist_3=%.6f rad, offset=%.3f deg" % (
                    target_q[5], math.degrees(delta)))
                continue
            if raw in ("cancel", "q", "quit"):
                print("calibration cancelled; no file written")
                return 1
            if raw in ("save", "s"):
                actual_q = np.asarray(receive.getActualQ(), dtype=np.float64)
                offset = wrap_angle(actual_q[5] - planned_pause_q[5])
                save_calibration(
                    output_path, offset, planned_pause_q, actual_q, actual_pause_time, args
                )
                calibrated_trajectory = replay.apply_local_wrist_rotation(
                    source_trajectory, offset
                )
                calibrated_joints, calibrated_turns, calibrated_margin = (
                    replay.center_periodic_joint_path(
                        source_joints, joint_index=5, offset_rad=offset
                    )
                )
                calibrated_cache = bundle / "ik_plan_cache_wrist.npz"
                replay.save_ik_cache(
                    calibrated_cache,
                    calibrated_trajectory,
                    calibrated_joints,
                    args.robot_ip,
                    {
                        "branch_clearances_m": metadata.get("branch_clearances_m", []),
                        "link_radius_m": metadata.get("link_radius_m", 0.04),
                        "min_link_clearance_m": metadata.get("min_link_clearance_m", 0.01),
                        "max_source_joint_step_deg": metadata.get("max_source_joint_step_deg", 30.0),
                        "wrist3_offset_rad": float(offset),
                        "wrist3_periodic_turns": int(calibrated_turns),
                        "wrist3_limit_margin_rad": float(calibrated_margin),
                        "created_unix_time": time.time(),
                    },
                )
                print("saved wrist-adjusted IK cache: %s" % calibrated_cache)
                return 0

            parts = raw.split()
            try:
                if parts[0] in ("+", "-"):
                    magnitude = float(parts[1]) if len(parts) > 1 else args.jog_step_deg
                    delta_deg = magnitude if parts[0] == "+" else -magnitude
                    target_q[5] += math.radians(delta_deg)
                elif parts[0] == "set" and len(parts) == 2:
                    target_q[5] = planned_pause_q[5] + math.radians(float(parts[1]))
                else:
                    print("unknown command")
                    continue
            except ValueError:
                print("invalid angle")
                continue
            lower = math.radians(args.wrist3_lower_deg)
            upper = math.radians(args.wrist3_upper_deg)
            if target_q[5] < lower or target_q[5] > upper:
                print("refusing wrist_3 target %.3f deg outside [%.1f, %.1f] deg" % (
                    math.degrees(target_q[5]),
                    args.wrist3_lower_deg,
                    args.wrist3_upper_deg,
                ))
                target_q = np.asarray(receive.getActualQ(), dtype=np.float64)
                continue
            control.moveJ(target_q.tolist(), args.jog_speed, args.jog_acceleration, False)
            target_q = np.asarray(receive.getActualQ(), dtype=np.float64)
            delta = wrap_angle(target_q[5] - planned_pause_q[5])
            print("offset now %.3f deg" % math.degrees(delta))
    except KeyboardInterrupt:
        print("calibration interrupted; no file written")
        return 130
    finally:
        if control is not None:
            try:
                control.servoStop(1.0)
            except Exception:
                pass
            try:
                control.stopScript()
            except Exception:
                pass
        if receive is not None:
            try:
                receive.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
