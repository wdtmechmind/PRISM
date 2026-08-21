#!/usr/bin/env python3
"""Convert a cam0-frame rigid trajectory to robot Base coordinates.

Input is a PRISM ``rigid_pose_6d.csv`` with position in cam0 and ZYX
roll/pitch/yaw in degrees. The output is a UR-compatible Cartesian trajectory:
``t_sec,x_m,y_m,z_m,rx_rad,ry_rad,rz_rad``. If a trial directory is supplied,
the source ``hand/sdk_commands.csv`` is copied into the output bundle so the
arm and hand replay can use the same recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def rpy_zyx_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def transform_pose(base_from_cam0: np.ndarray, position: np.ndarray, rpy_deg: np.ndarray):
    cam_from_body = np.eye(4, dtype=np.float64)
    cam_from_body[:3, :3] = rpy_zyx_to_matrix(*rpy_deg)
    cam_from_body[:3, 3] = position
    base_from_body = base_from_cam0 @ cam_from_body
    rotation_vector, _ = cv2.Rodrigues(base_from_body[:3, :3])
    return base_from_body[:3, 3], rotation_vector.reshape(3)


def semantic_axis_vector(axis: str) -> np.ndarray:
    vectors = {
        "forward": np.array([1.0, 0.0, 0.0]),
        "backward": np.array([-1.0, 0.0, 0.0]),
        "right": np.array([0.0, 1.0, 0.0]),
        "left": np.array([0.0, -1.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "down": np.array([0.0, 0.0, -1.0]),
    }
    return vectors[axis]


def apply_mount_correction(rows, axis: str, angle_deg: float):
    if not angle_deg:
        return rows
    correction, _ = cv2.Rodrigues(semantic_axis_vector(axis) * np.radians(angle_deg))
    corrected = []
    for row in rows:
        rotation, _ = cv2.Rodrigues(np.asarray(row[4:7], dtype=np.float64))
        rotation_vector, _ = cv2.Rodrigues(rotation @ correction)
        corrected.append((*row[:4], *rotation_vector.reshape(3)))
    return corrected


def resolve_input(args) -> tuple[Path, Path | None]:
    if args.trial_dir:
        trial = args.trial_dir.expanduser().resolve()
        trajectory = trial / "trajectory" / "rigid_pose_6d.csv"
        sdk = trial / "hand" / "sdk_commands.csv"
        return trajectory, sdk if sdk.is_file() else None
    trajectory = args.input_csv.expanduser().resolve()
    return trajectory, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path)
    source.add_argument("--trial-dir", type=Path)
    parser.add_argument("--handeye", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--use-smoothed", action="store_true",
                        help="use x_smooth/y_smooth/z_smooth and smooth RPY columns when available")
    parser.add_argument("--min-num-leds", type=int, default=3)
    parser.add_argument("--base-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--mount-correction-axis", default="right",
                        choices=("forward", "backward", "left", "right", "up", "down"))
    parser.add_argument("--mount-correction-deg", type=float, default=0.0)
    args = parser.parse_args()

    input_csv, sdk_source = resolve_input(args)
    if not input_csv.is_file():
        raise SystemExit("trajectory not found: %s" % input_csv)
    with args.handeye.expanduser().resolve().open("r", encoding="utf-8") as handle:
        handeye = json.load(handle)
    base_from_cam0 = np.asarray(handeye["base_from_cam0"], dtype=np.float64).reshape(4, 4)

    rows = []
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("mode", "measured").strip() not in ("", "measured"):
                continue
            try:
                if int(row.get("num_leds_used", args.min_num_leds)) < args.min_num_leds:
                    continue
                position_keys = ("x_smooth_m", "y_smooth_m", "z_smooth_m") if args.use_smoothed and "x_smooth_m" in row else ("x_m", "y_m", "z_m")
                rpy_keys = ("roll_smooth_deg", "pitch_smooth_deg", "yaw_smooth_deg") if args.use_smoothed and "roll_smooth_deg" in row else ("roll_deg", "pitch_deg", "yaw_deg")
                position = np.asarray([float(row[key]) for key in position_keys], dtype=np.float64)
                rpy = np.asarray([float(row[key]) for key in rpy_keys], dtype=np.float64)
                t = float(row.get("t_trial", row.get("t_sec", "0")))
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(position).all() and np.isfinite(rpy).all() and np.isfinite(t):
                base_position, rotation_vector = transform_pose(base_from_cam0, position, rpy)
                base_position = base_position + np.asarray(args.base_offset, dtype=np.float64)
                rows.append((t, *base_position, *rotation_vector))

    if len(rows) < 2:
        raise SystemExit("fewer than two valid measured trajectory rows")
    rows.sort(key=lambda item: item[0])
    rows = apply_mount_correction(rows, args.mount_correction_axis, args.mount_correction_deg)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "base_trajectory.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_sec", "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad"])
        writer.writerows([["%.6f" % value for value in row] for row in rows])

    sdk_output = None
    if sdk_source is not None:
        sdk_output = output_dir / "sdk_commands.csv"
        shutil.copy2(sdk_source, sdk_output)

    manifest = {
        "source_trajectory": str(input_csv),
        "source_hand_sdk": str(sdk_source) if sdk_source else None,
        "base_from_cam0": base_from_cam0.tolist(),
        "input_frame": "cam0",
        "output_frame": "robot_base",
        "position_units": "m",
        "orientation": "UR axis-angle rotation vector, rad",
        "rows": len(rows),
        "used_smoothed": args.use_smoothed,
        "min_num_leds": args.min_num_leds,
        "base_offset_m": list(args.base_offset),
        "mount_correction": {
            "applied_during_rigid_reconstruction": False,
            "axis_in_semantic_frame": args.mount_correction_axis,
            "angle_deg": args.mount_correction_deg,
            "applied_during_base_conversion": bool(args.mount_correction_deg),
        },
        "handeye_translation_rmse_m": handeye.get("translation_rmse_m"),
        "handeye_rotation_rmse_deg": handeye.get("rotation_rmse_deg"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote %s (%d rows)" % (output_csv, len(rows)))
    if sdk_output:
        print("copied SDK timeline -> %s" % sdk_output)
    else:
        print("warning: no hand/sdk_commands.csv found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
