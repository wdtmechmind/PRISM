#!/usr/bin/env python3
"""Run corrected-frame conversion, planning, material application, and replay video export."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]


def default_output_dir(trial_dir: Path) -> Path:
    task_dir = trial_dir.parent.name if trial_dir.parent.name else "trial"
    return _REPO_ROOT / "data" / "processed" / "simulation" / task_dir / trial_dir.name


def default_isaac_python() -> str:
    isaac_python = Path("/isaac-sim/python.sh")
    if isaac_python.exists():
        return str(isaac_python)
    return sys.executable


def resolve_path(value: Optional[str], base: Path = _REPO_ROOT) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def run_step(label: str, command: List[str], dry_run: bool = False) -> None:
    print("\n[pipeline] %s" % label)
    print("[pipeline] %s" % " ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(_REPO_ROOT), check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-dir", required=True, help="raw trial directory containing trajectory/ and optional hand/")
    parser.add_argument("--calib-json", required=True, help="ChArUco calibration JSON for corrected-frame conversion")
    parser.add_argument("--out-dir", default=None, help="processed output dir; defaults to data/processed/simulation/<task>/<trial>")
    parser.add_argument("--rigid-csv", default=None, help="override trajectory/rigid_pose_6d.csv")
    parser.add_argument("--led-csv", default=None, help="override trajectory/trajectory_led.csv")
    parser.add_argument("--gestures", default=None, help="override hand/sdk_commands.csv; default auto-detects under --trial-dir")
    parser.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    parser.add_argument("--isaac-python", default=default_isaac_python(), help="Python launcher for Isaac Sim scripts")
    parser.add_argument("--use-raw", action="store_true", help="use raw pose/LED columns instead of smoothed columns")
    parser.add_argument("--use-orientation", action="store_true", help="ask the planner to minimize orientation too")
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--stride", type=int, default=1, help="planning stride")
    parser.add_argument("--max-frames", type=int, default=0, help="planning max frames; 0 means all frames")
    parser.add_argument("--replay-stride", type=int, default=1)
    parser.add_argument("--replay-max-frames", type=int, default=0, help="replay max frames; 0 means all planned frames")
    parser.add_argument("--video-path", default=None, help="mp4 output path; defaults to <out-dir>/planned_motion_overhead.mp4")
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.2, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, -0.9], metavar=("X", "Y", "Z"))
    parser.add_argument("--keep-frames", action="store_true", help="keep raw PNG frames generated during video export")
    parser.add_argument("--skip-materials", action="store_true", help="do not reapply robot USD material overrides before replay")
    parser.add_argument("--show-preview", action="store_true", help="open the Isaac viewport instead of headless/no-preview replay")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    trial_dir = resolve_path(args.trial_dir)
    calib_json = resolve_path(args.calib_json)
    if trial_dir is None or not trial_dir.is_dir():
        raise SystemExit("trial dir not found: %s" % trial_dir)
    if calib_json is None or not calib_json.is_file():
        raise SystemExit("calibration JSON not found: %s" % calib_json)

    out_dir = resolve_path(args.out_dir) if args.out_dir else default_output_dir(trial_dir)
    assert out_dir is not None
    corrected_trajectory = out_dir / "corrected_trajectory.csv"
    planned_motion = out_dir / "planned_motion.csv"
    video_path = resolve_path(args.video_path) if args.video_path else out_dir / "planned_motion_overhead.mp4"
    assert video_path is not None
    gestures = resolve_path(args.gestures) if args.gestures else trial_dir / "hand" / "sdk_commands.csv"

    correct_cmd = [
        args.isaac_python,
        "simulation/scripts/trajectory_corrector.py",
        "--trial-dir", str(trial_dir),
        "--calib-json", str(calib_json),
        "--out-dir", str(out_dir),
    ]
    if args.rigid_csv:
        correct_cmd.extend(["--rigid-csv", str(resolve_path(args.rigid_csv))])
    if args.led_csv:
        correct_cmd.extend(["--led-csv", str(resolve_path(args.led_csv))])
    if args.use_raw:
        correct_cmd.append("--use-raw")

    plan_cmd = [
        args.isaac_python,
        "simulation/scripts/plan_robot_motion.py",
        "--corrected-trajectory", str(corrected_trajectory),
        "--robot-config", str(resolve_path(args.robot_config)),
        "--out", str(planned_motion),
        "--orientation-weight", str(args.orientation_weight),
        "--max-iters", str(args.max_iters),
        "--tolerance", str(args.tolerance),
        "--stride", str(max(1, args.stride)),
        "--max-frames", str(max(0, args.max_frames)),
    ]
    if gestures.is_file():
        plan_cmd.extend(["--gestures", str(gestures)])
    else:
        print("[pipeline] gesture file not found; planner will use default hand pose: %s" % gestures)
    if args.use_orientation:
        plan_cmd.append("--use-orientation")

    materials_cmd = [args.isaac_python, "simulation/scripts/apply_robot_materials.py"]

    replay_cmd = [
        args.isaac_python,
        "simulation/scripts/replay_planned_motion.py",
        "--planned-motion", str(planned_motion),
        "--robot-config", str(resolve_path(args.robot_config)),
        "--record-video",
        "--video-path", str(video_path),
        "--video-fps", str(args.video_fps),
        "--stride", str(max(1, args.replay_stride)),
        "--max-frames", str(max(0, args.replay_max_frames)),
        "--camera-pos", *("%.9g" % value for value in args.camera_pos),
        "--camera-look-at", *("%.9g" % value for value in args.camera_look_at),
    ]
    if args.keep_frames:
        replay_cmd.append("--keep-frames")
    if not args.show_preview:
        replay_cmd.extend(["--headless", "--no-preview"])

    run_step("1/4 corrected-frame trajectory", correct_cmd, dry_run=args.dry_run)
    run_step("2/4 robot motion planning", plan_cmd, dry_run=args.dry_run)
    if not args.skip_materials:
        run_step("3/4 robot material overrides", materials_cmd, dry_run=args.dry_run)
    else:
        print("\n[pipeline] 3/4 robot material overrides skipped")
    run_step("4/4 planned replay video", replay_cmd, dry_run=args.dry_run)

    print("\n[pipeline] done")
    print("[pipeline] corrected trajectory: %s" % corrected_trajectory)
    print("[pipeline] planned motion: %s" % planned_motion)
    print("[pipeline] replay video: %s" % video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
