#!/usr/bin/env python3
"""Build and replay one PRISM trial on UR3 with project deployment defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def default_bundle_dir(trial_dir: Path) -> Path:
    trial = trial_dir.expanduser().resolve()
    return ROOT / "data" / "replay_bundles" / "ur3" / trial.parent.name / trial.name


def run(command, label: str, print_only: bool) -> None:
    print("\n[ur3-replay] %s" % label)
    print("[ur3-replay] %s" % " ".join(str(part) for part in command))
    if not print_only:
        subprocess.run(command, cwd=str(ROOT), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", type=Path, default=None,
                        help="default: data/replay_bundles/ur3/<task>/<trial>")
    parser.add_argument("--robot-ip", default="192.168.1.102")
    parser.add_argument("--handeye", type=Path,
                        default=Path("configs/deployment/robot_camera_handeye.json"))
    parser.add_argument("--wrist-calibration", type=Path,
                        default=Path("configs/deployment/ur3_wrist_mount.json"))
    parser.add_argument("--base-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--mount-correction-axis", default="right",
                        choices=("forward", "backward", "left", "right", "up", "down"))
    parser.add_argument("--mount-correction-deg", type=float, default=-45.0)
    parser.add_argument("--min-num-leds", type=int, default=3)
    parser.add_argument("--time-scale", type=float, default=15.0)
    parser.add_argument("--rate-hz", type=float, default=125.0)
    parser.add_argument("--movej-speed", type=float, default=0.05)
    parser.add_argument("--movej-acceleration", type=float, default=0.10)
    parser.add_argument("--min-link-clearance", type=float, default=0.01)
    parser.add_argument("--link-radius", type=float, default=0.04)
    parser.add_argument("--max-source-joint-step-deg", type=float, default=30.0)
    parser.add_argument("--max-joint-speed", type=float, default=1.2)
    parser.add_argument("--hand-ip", default="localhost")
    parser.add_argument("--hand-port", type=int, default=60686)
    parser.add_argument("--skip-hand", action="store_true")
    parser.add_argument("--no-wrist-calibration", action="store_true")
    parser.add_argument("--replan-ik", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true",
                      help="connect and plan/check IK, then exit before motion")
    mode.add_argument("--execute", action="store_true",
                      help="execute real UR3 motion and SDK timeline")
    parser.add_argument("--print-only", action="store_true",
                        help="print both underlying commands without running them")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trial = args.trial_dir.expanduser().resolve()
    if not trial.is_dir():
        raise SystemExit("trial directory not found: %s" % trial)
    rigid = trial / "trajectory" / "rigid_pose_6d.csv"
    if not rigid.is_file():
        raise SystemExit("offline rigid trajectory not found: %s" % rigid)

    bundle = (args.bundle_dir.expanduser().resolve() if args.bundle_dir is not None
              else default_bundle_dir(trial))
    handeye = args.handeye.expanduser().resolve()
    wrist_calibration = args.wrist_calibration.expanduser().resolve()

    build_command = [
        sys.executable,
        str(ROOT / "tools" / "transform_cam_trajectory_to_base.py"),
        "--trial-dir", str(trial),
        "--handeye", str(handeye),
        "--output-dir", str(bundle),
        "--use-smoothed",
        "--min-num-leds", str(args.min_num_leds),
        "--base-offset", *("%.9g" % value for value in args.base_offset),
        "--mount-correction-axis", args.mount_correction_axis,
        "--mount-correction-deg", "%.9g" % args.mount_correction_deg,
    ]
    run(build_command, "1/2 build Base-frame bundle -> %s" % bundle, args.print_only)

    replay_command = [
        sys.executable,
        str(ROOT / "tools" / "replay_ur3_base_trajectory.py"),
        "--bundle-dir", str(bundle),
        "--robot-ip", args.robot_ip,
        "--time-scale", "%.9g" % args.time_scale,
        "--rate-hz", "%.9g" % args.rate_hz,
        "--movej-speed", "%.9g" % args.movej_speed,
        "--movej-acceleration", "%.9g" % args.movej_acceleration,
        "--min-link-clearance", "%.9g" % args.min_link_clearance,
        "--link-radius", "%.9g" % args.link_radius,
        "--max-source-joint-step-deg", "%.9g" % args.max_source_joint_step_deg,
        "--max-joint-speed", "%.9g" % args.max_joint_speed,
    ]
    if args.no_wrist_calibration:
        replay_command.append("--no-wrist-calibration")
    else:
        if not wrist_calibration.is_file():
            raise SystemExit("wrist calibration not found: %s" % wrist_calibration)
        replay_command.extend(["--wrist-calibration", str(wrist_calibration)])
    if args.replan_ik:
        replay_command.append("--replan-ik")
    if args.preflight:
        replay_command.extend(["--skip-hand", "--ik-preflight-only"])
        label = "2/2 IK/clearance preflight (no motion)"
    elif args.execute:
        if args.skip_hand:
            replay_command.append("--skip-hand")
        else:
            replay_command.extend(["--hand-ip", args.hand_ip,
                                   "--hand-port", str(args.hand_port)])
        replay_command.append("--execute")
        label = "2/2 execute UR3 replay"
    else:
        replay_command.append("--skip-hand")
        label = "2/2 local dry-run (no connection, no motion)"
    run(replay_command, label, args.print_only)

    print("\n[ur3-replay] bundle: %s" % bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())