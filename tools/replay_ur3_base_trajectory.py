#!/usr/bin/env python3
"""Replay a Base-frame trajectory on a real UR3 and MechHand.

The input bundle is produced by transform_cam_trajectory_to_base.py. The arm
trajectory is streamed with UR RTDE ``servoL`` in UR axis-angle format. Hand SDK
commands from ``sdk_commands.csv`` are sent to localhost:60686 at their
recorded relative times. Hardware execution requires explicit ``--execute``.
Without it, this script only validates and prints the replay plan.
"""

from __future__ import annotations

import argparse
import csv
import socket
import time
from pathlib import Path

import numpy as np


def load_trajectory(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty trajectory: %s" % path)
    required = {"t_sec", "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("trajectory missing columns: %s" % ", ".join(sorted(missing)))
    values = np.asarray([[float(row[key]) for key in ("t_sec", "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad")] for row in rows])
    values = values[np.argsort(values[:, 0])]
    return values


def load_sdk_events(path: Path | None):
    if path is None or not path.is_file():
        return []
    events = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                t = float(row.get("trial_time", row.get("t_sec", "")))
            except (TypeError, ValueError):
                continue
            command = str(row.get("command", "")).strip()
            if command:
                if not command.endswith("&"):
                    command += "&"
                events.append((t, command, str(row.get("action", ""))))
    return sorted(events, key=lambda item: item[0])


def connect_hand(ip: str, port: int, timeout: float):
    sock = socket.create_connection((ip, port), timeout=timeout)
    sock.settimeout(timeout)
    return sock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--hand-ip", default="127.0.0.1")
    parser.add_argument("--hand-port", type=int, default=60686)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--acceleration", type=float, default=0.15)
    parser.add_argument("--servo-lookahead", type=float, default=0.10,
                        help="servoL lookahead time in seconds")
    parser.add_argument("--servo-gain", type=int, default=300,
                        help="servoL proportional gain")
    parser.add_argument("--movej-speed", type=float, default=0.30,
                        help="joint speed used to move to the first trajectory pose")
    parser.add_argument("--movej-acceleration", type=float, default=0.50,
                        help="joint acceleration used to move to the first trajectory pose")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="1.0 original timing; 2.0 twice as slow")
    parser.add_argument("--execute", action="store_true", help="enable real UR3 and hand execution")
    parser.add_argument("--skip-hand", action="store_true")
    parser.add_argument("--hand-timeout", type=float, default=3.0)
    args = parser.parse_args()

    if (args.speed <= 0 or args.acceleration <= 0 or args.time_scale <= 0
            or args.movej_speed <= 0 or args.movej_acceleration <= 0):
        parser.error("speed, acceleration, movej-speed, movej-acceleration and time-scale must be positive")
    bundle = args.bundle_dir.expanduser().resolve()
    trajectory = load_trajectory(bundle / "base_trajectory.csv")
    sdk_events = [] if args.skip_hand else load_sdk_events(bundle / "sdk_commands.csv")
    print("trajectory: %d frames, %.3f s" % (len(trajectory), trajectory[-1, 0] - trajectory[0, 0]))
    print("hand events: %d -> %s:%d" % (len(sdk_events), args.hand_ip, args.hand_port))
    print("start TCP pose: %s" % trajectory[0, 1:])
    if not args.execute:
        print("DRY RUN: no robot motion and no SDK command sent; add --execute")
        return 0

    try:
        import rtde_control
    except ImportError as exc:
        raise SystemExit("--execute requires ur_rtde: pip install ur-rtde") from exc

    control = None
    hand_socket = None
    try:
        control = rtde_control.RTDEControlInterface(args.robot_ip)
        start_pose = [float(value) for value in trajectory[0, 1:]]
        print("computing IK for first trajectory pose ...", flush=True)
        start_joints = control.getInverseKinematics(start_pose)
        if start_joints is None or len(start_joints) != 6:
            raise RuntimeError("UR3 inverse kinematics failed for the first trajectory pose")
        print("moving UR3 with moveJ to trajectory start ...", flush=True)
        movej_ok = control.moveJ(
            [float(value) for value in start_joints],
            args.movej_speed,
            args.movej_acceleration,
            False,
        )
        if movej_ok is False:
            raise RuntimeError("UR3 moveJ to trajectory start failed")
        print("UR3 reached trajectory start; beginning timed servoL replay.", flush=True)
        if sdk_events:
            hand_socket = connect_hand(args.hand_ip, args.hand_port, args.hand_timeout)
        start_wall = time.monotonic()
        event_index = 0
        for row_index, row in enumerate(trajectory):
            target_time = (float(row[0]) - float(trajectory[0, 0])) / args.time_scale
            while time.monotonic() - start_wall < target_time:
                time.sleep(min(0.002, max(0.0001, target_time - (time.monotonic() - start_wall))))
            elapsed = time.monotonic() - start_wall
            while event_index < len(sdk_events):
                event_time = (sdk_events[event_index][0] - float(trajectory[0, 0])) / args.time_scale
                if event_time > elapsed:
                    break
                _, command, action = sdk_events[event_index]
                if hand_socket is not None:
                    hand_socket.sendall(command.encode("utf-8"))
                    print("hand %s -> %s" % (action, command))
                event_index += 1
            pose = [float(value) for value in row[1:]]
            if row_index + 1 < len(trajectory):
                source_dt = float(trajectory[row_index + 1, 0] - row[0])
            elif row_index > 0:
                source_dt = float(row[0] - trajectory[row_index - 1, 0])
            else:
                source_dt = 0.008
            # UR3 CB3 servo loop is normally run at 125 Hz; do not issue
            # commands faster than the controller can consume reliably.
            servo_dt = max(0.008, min(0.1, source_dt / args.time_scale))
            control.servoL(
                pose,
                args.speed,
                args.acceleration,
                servo_dt,
                args.servo_lookahead,
                args.servo_gain,
            )
        while event_index < len(sdk_events):
            _, command, action = sdk_events[event_index]
            if hand_socket is not None:
                hand_socket.sendall(command.encode("utf-8"))
                print("hand %s -> %s" % (action, command))
            event_index += 1
        print("replay complete")
        return 0
    except KeyboardInterrupt:
        print("replay interrupted")
        return 130
    finally:
        if control is not None:
            try:
                control.servoStop(2.0)
            except Exception:
                pass
            try:
                control.stopScript()
            except Exception:
                pass
        if hand_socket is not None:
            hand_socket.close()


if __name__ == "__main__":
    raise SystemExit(main())
