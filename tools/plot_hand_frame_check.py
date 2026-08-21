#!/usr/bin/env python3
"""Visualise the reconstructed hand frame against the camera rig.

Recomputes the hand pose directly from ``trajectory_led.csv`` using the current
LED->hand definition, so the absolute frame can be eyeballed without re-running
video reconstruction. Plots camera positions and optical axes, the hand path,
and the hand's forward/right/up triad sampled along the trajectory.

Example:
    python tools/plot_hand_frame_check.py \
        data/raw/<task>/trial_000001/trajectory/trajectory_led.csv \
        --calib-json configs/devices/charuco_4cam_result.json \
        --out /tmp/hand_frame_check.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from prism.reconstruction.calibration import (  # noqa: E402
    build_corrected_transform,
    get_camera_centers_world,
    load_calibration,
)
from prism.reconstruction.hand_frame import (  # noqa: E402
    BASELINK_FROM_SEMANTIC,
    hand_pose_from_leds,
    load_spec,
    mount_correction,
    semantic_frame_from_leds,
)


def load_frames(path: Path, smoothed: bool = True):
    """Return frames as a list of {color: xyz} ordered by frame index."""
    by_frame = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        pre = "x_smooth_m" if (smoothed and "x_smooth_m" in cols) else "x_m"
        keys = [pre, pre.replace("x_", "y_"), pre.replace("x_", "z_")]
        for row in reader:
            try:
                xyz = np.array([float(row[k]) for k in keys], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(xyz).all():
                by_frame[int(row["frame_index"])][row["color"]] = xyz
    return [by_frame[k] for k in sorted(by_frame)]


def set_equal_aspect(ax, pts):
    span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    r = max(span, 1e-3) / 2.0
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("led_csv")
    p.add_argument("--calib-json", default="configs/devices/charuco_4cam_result.json")
    p.add_argument("--out", default="/tmp/hand_frame_check.png")
    p.add_argument("--triads", type=int, default=24, help="number of hand frames to draw")
    p.add_argument("--axis-len", type=float, default=0.05)
    p.add_argument("--raw", action="store_true", help="use unsmoothed LED columns")
    p.add_argument("--mounting", choices=["robot", "capture"], default="robot",
                   help="draw the hand as mounted on the arm flange, or as held in the capture rig")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    spec = load_spec()
    frames = load_frames(Path(args.led_csv).expanduser().resolve(), smoothed=not args.raw)

    cameras = load_calibration(str(Path(args.calib_json).expanduser().resolve()))
    centers = get_camera_centers_world(cameras)
    corrected = build_corrected_transform(cameras, centers)
    rot_c, org_c = corrected["R"], corrected["origin"]

    poses, skews = [], []
    for pts in frames:
        solved = semantic_frame_from_leds(pts, spec)
        if solved is None:
            continue
        rot_sem, origin, skew = solved
        poses.append((rot_c @ rot_sem, rot_c @ (origin - org_c)))
        skews.append(skew)

    if not poses:
        raise SystemExit("no frame had all four LEDs; cannot build the hand frame")

    pos = np.array([p[1] for p in poses])
    skews = np.asarray(skews)
    corr = mount_correction(spec)
    sem_capture = np.array([p[0] for p in poses])
    sem_robot = np.array([p[0] @ corr for p in poses])
    sem = sem_robot if args.mounting == "robot" else sem_capture
    # Columns of the semantic basis are (forward, LEFT, up); negate to draw right.
    axes_world = np.stack([sem[:, :, 0], -sem[:, :, 1], sem[:, :, 2]], axis=2)

    zhat = np.array([0.0, 0.0, 1.0])
    ang_capture = np.degrees(np.arccos(np.clip(sem_capture[:, :, 2] @ zhat, -1.0, 1.0)))
    ang_robot = np.degrees(np.arccos(np.clip(sem_robot[:, :, 2] @ zhat, -1.0, 1.0)))
    ang_up = ang_robot if args.mounting == "robot" else ang_capture

    print("frames with all 4 LEDs : %d / %d" % (len(poses), len(frames)))
    print("LED axis skew (deg)    : mean %.2f  median %.2f  p95 %.2f  max %.2f"
          % (skews.mean(), np.median(skews), np.percentile(skews, 95), skews.max()))
    print("mount correction       : %.1f deg about %s"
          % (float(spec.get("mount_correction_deg", 0.0)), spec.get("mount_correction_axis", "right")))
    print("back-of-hand vs world +Z (deg):")
    print("    as captured    : mean %5.1f  min %5.1f  max %5.1f"
          % (ang_capture.mean(), ang_capture.min(), ang_capture.max()))
    print("    on arm flange  : mean %5.1f  min %5.1f  max %5.1f"
          % (ang_robot.mean(), ang_robot.min(), ang_robot.max()))
    print("hand position (corrected frame):")
    for i, name in enumerate("xyz"):
        print("    %s: %7.3f .. %7.3f m" % (name, pos[:, i].min(), pos[:, i].max()))

    fig = plt.figure(figsize=(16, 5.6))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="tab:orange", lw=1.6, label="hand path")
    cam_pts = []
    for idx in sorted(centers):
        c = rot_c @ (np.asarray(centers[idx], dtype=np.float64) - org_c)
        look = rot_c @ (cameras[idx]["R"].T @ np.array([0.0, 0.0, 1.0]))
        cam_pts.append(c)
        ax.scatter(*c, color="tab:purple", marker="^", s=70)
        ax.quiver(c[0], c[1], c[2], *(look * 0.30), color="tab:purple", lw=1.0,
                  arrow_length_ratio=0.2, alpha=0.75)
        ax.text(c[0], c[1], c[2] + 0.04, "cam%s" % idx, fontsize=9, color="tab:purple")
    set_equal_aspect(ax, np.vstack([pos, np.array(cam_pts)]))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.view_init(elev=18, azim=-70)
    ax.set_title("rig overview (purple = camera + view direction)")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="0.6", lw=1.2)
    ax.scatter(*pos[0], color="k", marker="o", s=45, label="start")
    ax.scatter(*pos[-1], color="k", marker="X", s=60, label="end")
    step = max(1, len(poses) // max(1, args.triads))
    for i in range(0, len(poses), step):
        o = pos[i]
        for a, (col, lbl) in enumerate(zip("rgb", ["forward", "right", "up"])):
            d = axes_world[i, :, a] * args.axis_len
            ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], color=col, lw=1.8,
                      arrow_length_ratio=0.3, label=lbl if i == 0 else None)
    set_equal_aspect(ax, pos)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.view_init(elev=18, azim=-70)
    ax.set_title("hand frame (%s mounting)\nred=forward  green=right  blue=up" % args.mounting)
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(ang_capture, color="0.6", lw=1.0, label="as captured (rig)")
    ax.plot(ang_robot, color="tab:blue", lw=1.0, label="on arm flange")
    ax.axhline(90, color="0.6", ls="--", lw=1.0)
    ax.set_ylim(0, 185)
    ax.set_xlabel("frame")
    ax.set_ylabel("angle between back-of-hand normal and world +Z (deg)")
    ax.set_title("up-axis, before and after the %.0f deg mount correction"
                 % float(spec.get("mount_correction_deg", 0.0)))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Hand frame check -- corrected frame, +Z up  |  drawn mounting: %s  |  "
                 "LED axis skew mean %.2f deg" % (args.mounting, skews.mean()), fontsize=12)
    fig.tight_layout()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
