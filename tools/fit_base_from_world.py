#!/usr/bin/env python3
"""Fit the shared corrected-frame -> robot-base transform used for replay.

The corrected calibration frame is taken to be gravity-aligned (its +Z is up),
so the transform has only 4 free parameters: a yaw about Z and a translation.
Neither is measured -- they are a placement choice, picked so the demonstration
trajectories land inside the arm's comfortable working envelope.

The result is a single JSON shared by every trial, which keeps the per-trial
start-pose diversity that a per-trial anchor would flatten.

Example:
    python tools/fit_base_from_world.py \
        data/processed/simulation/task_*/trial_*/corrected_trajectory.csv \
        --target-xy 0.45 0.0 --target-z 0.30 --yaw-mode auto \
        --out configs/deployment/base_from_world.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def load_positions(paths: List[Path]) -> np.ndarray:
    pts = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cols = reader.fieldnames or []
            prefix = "x_smooth_m" if "x_smooth_m" in cols else "x_m"
            keys = [prefix, prefix.replace("x_", "y_"), prefix.replace("x_", "z_")]
            for row in reader:
                try:
                    pts.append([float(row[k]) for k in keys])
                except (KeyError, TypeError, ValueError):
                    continue
    if not pts:
        raise SystemExit("no positions found in the given trajectories")
    arr = np.asarray(pts, dtype=np.float64)
    return arr[np.isfinite(arr).all(axis=1)]


def corrected_transform(calib_path: Path):
    import sys

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from prism.reconstruction.calibration import (  # noqa: E402
        build_corrected_transform,
        get_camera_centers_world,
        load_calibration,
    )

    cameras = load_calibration(str(calib_path))
    centers = get_camera_centers_world(cameras)
    return build_corrected_transform(cameras, centers), centers


def auto_yaw(points: np.ndarray) -> float:
    """Yaw that aligns the dominant horizontal spread of the data with base +X."""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    if len(xy) < 3:
        return 0.0
    _, _, vh = np.linalg.svd(xy, full_matrices=False)
    axis = vh[0]
    return -float(np.arctan2(axis[1], axis[0]))


def rot_z(yaw_rad: float) -> np.ndarray:
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectories", nargs="+", help="corrected_trajectory.csv paths")
    parser.add_argument("--out", required=True, help="output JSON, consumed by plan_robot_motion --base-from-world")
    parser.add_argument("--yaw-mode", choices=["zero", "auto", "manual"], default="auto")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="yaw for --yaw-mode manual")
    parser.add_argument("--target-xy", type=float, nargs=2, default=[0.45, 0.0],
                        help="where the trajectory centroid should sit in the base XY plane")
    parser.add_argument("--target-z", type=float, default=0.30,
                        help="height of the trajectory centroid above the robot base")
    parser.add_argument("--min-reach", type=float, default=0.25)
    parser.add_argument("--max-reach", type=float, default=0.80)
    parser.add_argument("--calib-json", default=None,
                        help="charuco calibration JSON; reports where each camera lands in the "
                             "base frame so the chosen yaw can be sanity-checked")
    parser.add_argument("--from-cam0-frame", action="store_true",
                        help="inputs are raw cam0-frame reconstruction output (rigid_pose_6d.csv); "
                             "apply the corrected transform first. Requires --calib-json")
    return parser


def report_cameras(centers, corrected, base_from_world: np.ndarray) -> None:
    rot, trans = base_from_world[:3, :3], base_from_world[:3, 3]
    print("[fit-base] camera layout in the robot base frame:")
    for idx in sorted(centers):
        world = corrected["R"] @ (np.asarray(centers[idx], dtype=np.float64) - corrected["origin"])
        pos = rot @ world + trans
        side = "left " if pos[1] > 0 else "right"
        depth = "front" if pos[0] > 0 else "behind"
        print("    cam%s  [%7.3f %7.3f %7.3f]  %s / %s, %.2f m from base"
              % (idx, pos[0], pos[1], pos[2], side, depth, float(np.linalg.norm(pos[:2]))))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = [Path(p).expanduser().resolve() for p in args.trajectories]
    points = load_positions(paths)

    corrected, centers = (None, None)
    if args.calib_json:
        corrected, centers = corrected_transform(Path(args.calib_json).expanduser().resolve())
    if args.from_cam0_frame:
        if corrected is None:
            raise SystemExit("--from-cam0-frame requires --calib-json")
        points = (points - corrected["origin"]) @ corrected["R"].T
        print("[fit-base] applied the corrected transform to cam0-frame inputs")

    if args.yaw_mode == "auto":
        yaw = auto_yaw(points)
    elif args.yaw_mode == "manual":
        yaw = float(np.radians(args.yaw_deg))
    else:
        yaw = 0.0

    rot = rot_z(yaw)
    centroid = points.mean(axis=0)
    target = np.array([args.target_xy[0], args.target_xy[1], args.target_z], dtype=np.float64)
    trans = target - rot @ centroid

    base_from_world = np.eye(4, dtype=np.float64)
    base_from_world[:3, :3] = rot
    base_from_world[:3, 3] = trans

    in_base = points @ rot.T + trans
    reach = np.linalg.norm(in_base, axis=1)
    out_of_range = int(np.sum((reach < args.min_reach) | (reach > args.max_reach)))

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "base_from_world": base_from_world.tolist(),
            "source": "fit_base_from_world",
            "yaw_deg": float(np.degrees(yaw)),
            "target_centroid_base": target.tolist(),
            "trajectories": [str(p) for p in paths],
        }, handle, indent=2)

    print("[fit-base] trajectories: %d, points: %d" % (len(paths), len(points)))
    print("[fit-base] yaw: %.2f deg (%s)" % (float(np.degrees(yaw)), args.yaw_mode))
    print("[fit-base] centroid world -> base: [%s] -> [%s]"
          % (", ".join("%.3f" % v for v in centroid), ", ".join("%.3f" % v for v in target)))
    print("[fit-base] reach from base: min=%.3f m max=%.3f m" % (float(reach.min()), float(reach.max())))
    print("[fit-base] base-frame Z span: %.3f .. %.3f m" % (float(in_base[:, 2].min()), float(in_base[:, 2].max())))
    if out_of_range:
        print("[fit-base] WARNING: %d/%d points outside [%.2f, %.2f] m -- retune --target-xy/--target-z"
              % (out_of_range, len(points), args.min_reach, args.max_reach))
    if float(in_base[:, 2].min()) < 0.0:
        print("[fit-base] WARNING: trajectory dips below the robot base plane")
    if args.calib_json:
        report_cameras(centers, corrected, base_from_world)
    print("[fit-base] wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
