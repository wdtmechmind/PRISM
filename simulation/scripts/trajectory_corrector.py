#!/usr/bin/env python3
"""Convert PRISM raw trial trajectories into corrected-frame files.

This is the first stage of the simulation pipeline. It has no Isaac Sim
dependency: it reads raw trial CSV files, applies the ChArUco corrected-frame
transform, writes normalized CSV/JSON artifacts, and renders projection plots
for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from prism.reconstruction.calibration import (  # noqa: E402
    apply_corrected_transform,
    build_corrected_transform,
    get_camera_centers_world,
    load_calibration,
)
from simulation.common.geometry import (  # noqa: E402
    mat_to_euler_zyx_deg,
    mat_to_quat_wxyz,
    rotation_zyx,
)


LED_COLORS = {
    "red": "#d90429",
    "yellow": "#f4b400",
    "blue": "#2979ff",
    "green": "#00a86b",
}


def _pick(colnames: Iterable[str], *candidates: str) -> Optional[str]:
    colset = set(colnames)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def _float_or_nan(value: object) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path], Path]:
    trial_dir = Path(args.trial_dir).expanduser().resolve()
    if not trial_dir.is_dir():
        raise SystemExit("trial dir not found: %s" % trial_dir)

    trajectory_dir = trial_dir / "trajectory"
    rigid_csv = Path(args.rigid_csv).expanduser().resolve() if args.rigid_csv else trajectory_dir / "rigid_pose_6d.csv"
    led_csv = Path(args.led_csv).expanduser().resolve() if args.led_csv else trajectory_dir / "trajectory_led.csv"
    calib_json = Path(args.calib_json).expanduser().resolve()

    if not rigid_csv.is_file():
        raise SystemExit("rigid pose CSV not found: %s" % rigid_csv)
    if not calib_json.is_file():
        raise SystemExit("calibration JSON not found: %s" % calib_json)
    if not led_csv.is_file():
        led_csv = None
    return trial_dir, rigid_csv, led_csv, calib_json


def default_output_dir(trial_dir: Path) -> Path:
    task_dir = trial_dir.parent.name if trial_dir.parent.name else "trial"
    trial_name = trial_dir.name
    return _REPO_ROOT / "data" / "processed" / "simulation" / task_dir / trial_name


def load_rigid_poses(path: Path, prefer_smoothed: bool = True) -> Tuple[List[dict], np.ndarray, np.ndarray, bool, dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        use_smoothed = prefer_smoothed and "x_smooth_m" in cols and "roll_smooth_deg" in cols
        x_col = "x_smooth_m" if use_smoothed else "x_m"
        y_col = "y_smooth_m" if use_smoothed else "y_m"
        z_col = "z_smooth_m" if use_smoothed else "z_m"
        roll_col = "roll_smooth_deg" if use_smoothed else "roll_deg"
        pitch_col = "pitch_smooth_deg" if use_smoothed else "pitch_deg"
        yaw_col = "yaw_smooth_deg" if use_smoothed else "yaw_deg"
        required = ["t_sec", x_col, y_col, z_col, roll_col, pitch_col, yaw_col]
        missing = [col for col in required if col not in cols]
        if missing:
            raise SystemExit("rigid pose CSV is missing columns: %s" % ", ".join(missing))

        rows = []
        positions = []
        rotations = []
        for row in reader:
            t_sec = _float_or_nan(row.get("t_sec"))
            source_pos = np.array([
                _float_or_nan(row.get(x_col)),
                _float_or_nan(row.get(y_col)),
                _float_or_nan(row.get(z_col)),
            ], dtype=np.float64)
            roll = _float_or_nan(row.get(roll_col))
            pitch = _float_or_nan(row.get(pitch_col))
            yaw = _float_or_nan(row.get(yaw_col))
            if not np.isfinite(source_pos).all() or not np.isfinite([t_sec, roll, pitch, yaw]).all():
                continue
            rows.append({
                "t_sec": t_sec,
                "source_pos": source_pos,
                "source_rpy_deg": np.array([roll, pitch, yaw], dtype=np.float64),
            })
            positions.append(source_pos)
            rotations.append(rotation_zyx(roll, pitch, yaw))

    if not rows:
        raise SystemExit("no valid rigid pose rows in %s" % path)

    order = np.argsort([row["t_sec"] for row in rows])
    sorted_rows = [rows[int(index)] for index in order]
    pos_arr = np.asarray(positions, dtype=np.float64)[order]
    rot_arr = np.asarray(rotations, dtype=np.float64)[order]
    columns = {
        "x": x_col,
        "y": y_col,
        "z": z_col,
        "roll": roll_col,
        "pitch": pitch_col,
        "yaw": yaw_col,
    }
    return sorted_rows, pos_arr, rot_arr, use_smoothed, columns


def load_led_rows(path: Optional[Path], prefer_smoothed: bool = True) -> Tuple[List[dict], np.ndarray, bool, dict]:
    if path is None:
        return [], np.zeros((0, 3), dtype=np.float64), False, {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        color_col = _pick(cols, "color")
        t_col = _pick(cols, "t_sec", "t_trial")
        use_smoothed = prefer_smoothed and "x_smooth_m" in cols
        x_col = _pick(cols, "x_smooth_m", "x_m") if use_smoothed else _pick(cols, "x_m", "x_smooth_m")
        y_col = _pick(cols, "y_smooth_m", "y_m") if use_smoothed else _pick(cols, "y_m", "y_smooth_m")
        z_col = _pick(cols, "z_smooth_m", "z_m") if use_smoothed else _pick(cols, "z_m", "z_smooth_m")
        if not (color_col and x_col and y_col and z_col):
            return [], np.zeros((0, 3), dtype=np.float64), use_smoothed, {}

        rows = []
        points = []
        for row in reader:
            point = np.array([
                _float_or_nan(row.get(x_col)),
                _float_or_nan(row.get(y_col)),
                _float_or_nan(row.get(z_col)),
            ], dtype=np.float64)
            if not np.isfinite(point).all():
                continue
            color = str(row.get(color_col, "")).strip()
            if not color:
                continue
            rows.append({
                "t_sec": _float_or_nan(row.get(t_col)) if t_col else float("nan"),
                "color": color,
                "source_pos": point,
            })
            points.append(point)

    columns = {"t": t_col or "", "color": color_col or "", "x": x_col or "", "y": y_col or "", "z": z_col or ""}
    if not rows:
        return [], np.zeros((0, 3), dtype=np.float64), use_smoothed, columns
    return rows, np.asarray(points, dtype=np.float64), use_smoothed, columns


def write_corrected_trajectory(path: Path, rows: List[dict], corrected_pos: np.ndarray, corrected_rot: np.ndarray) -> None:
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
        for index, row in enumerate(rows):
            quat = mat_to_quat_wxyz(corrected_rot[index])
            rpy = mat_to_euler_zyx_deg(corrected_rot[index])
            source_pos = row["source_pos"]
            source_rpy = row["source_rpy_deg"]
            writer.writerow({
                "t_sec": "%.9f" % float(row["t_sec"]),
                "x_m": "%.9f" % float(corrected_pos[index, 0]),
                "y_m": "%.9f" % float(corrected_pos[index, 1]),
                "z_m": "%.9f" % float(corrected_pos[index, 2]),
                "qw": "%.12f" % float(quat[0]),
                "qx": "%.12f" % float(quat[1]),
                "qy": "%.12f" % float(quat[2]),
                "qz": "%.12f" % float(quat[3]),
                "roll_deg": "%.9f" % float(rpy[0]),
                "pitch_deg": "%.9f" % float(rpy[1]),
                "yaw_deg": "%.9f" % float(rpy[2]),
                "source_x_m": "%.9f" % float(source_pos[0]),
                "source_y_m": "%.9f" % float(source_pos[1]),
                "source_z_m": "%.9f" % float(source_pos[2]),
                "source_roll_deg": "%.9f" % float(source_rpy[0]),
                "source_pitch_deg": "%.9f" % float(source_rpy[1]),
                "source_yaw_deg": "%.9f" % float(source_rpy[2]),
            })


def write_corrected_leds(path: Path, rows: List[dict], corrected_points: np.ndarray) -> None:
    fieldnames = ["t_sec", "color", "x_m", "y_m", "z_m", "source_x_m", "source_y_m", "source_z_m"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            source_pos = row["source_pos"]
            t_sec = row["t_sec"]
            writer.writerow({
                "t_sec": "" if math.isnan(float(t_sec)) else "%.9f" % float(t_sec),
                "color": row["color"],
                "x_m": "%.9f" % float(corrected_points[index, 0]),
                "y_m": "%.9f" % float(corrected_points[index, 1]),
                "z_m": "%.9f" % float(corrected_points[index, 2]),
                "source_x_m": "%.9f" % float(source_pos[0]),
                "source_y_m": "%.9f" % float(source_pos[1]),
                "source_z_m": "%.9f" % float(source_pos[2]),
            })


def finite_bounds(*arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    chunks = []
    for array in arrays:
        if array is None or len(array) == 0:
            continue
        finite = array[np.isfinite(array).all(axis=1)]
        if len(finite) > 0:
            chunks.append(finite)
    if not chunks:
        raise SystemExit("no finite corrected points available")
    all_points = np.vstack(chunks)
    return np.min(all_points, axis=0), np.max(all_points, axis=0)


def padded_bounds(mins: np.ndarray, maxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    span = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(span * 0.08, 0.01)
    return mins - pad, maxs + pad


def group_led_points(rows: List[dict], corrected_points: np.ndarray) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[np.ndarray]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["color"]), []).append(corrected_points[index])
    return {color: np.asarray(points, dtype=np.float64) for color, points in grouped.items()}


def plot_projection(ax, axis_a: int, axis_b: int, rigid_points: np.ndarray, led_by_color: Dict[str, np.ndarray], bounds: Tuple[np.ndarray, np.ndarray], title: str) -> None:
    axis_names = ["Xc (m)", "Yc (m)", "Zc (m)"]
    mins, maxs = bounds

    finite_rigid = rigid_points[np.isfinite(rigid_points).all(axis=1)]
    ax.plot(finite_rigid[:, axis_a], finite_rigid[:, axis_b], color="#111111", linewidth=2.1, linestyle="--", alpha=0.82, label="Rigid pose")
    ax.scatter(finite_rigid[0, axis_a], finite_rigid[0, axis_b], color="#111111", s=24, marker="o", label="Start")
    ax.scatter(finite_rigid[-1, axis_a], finite_rigid[-1, axis_b], color="#111111", s=32, marker="x", label="End")

    for color, points in led_by_color.items():
        finite_led = points[np.isfinite(points).all(axis=1)]
        if len(finite_led) == 0:
            continue
        ax.plot(finite_led[:, axis_a], finite_led[:, axis_b], color=LED_COLORS.get(color, "#777777"), linewidth=1.4, alpha=0.85, label="LED %s" % color)

    ax.set_xlim(mins[axis_a], maxs[axis_a])
    ax.set_ylim(mins[axis_b], maxs[axis_b])
    ax.set_xlabel(axis_names[axis_a])
    ax.set_ylabel(axis_names[axis_b])
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def write_preview(path: Path, rigid_points: np.ndarray, led_by_color: Dict[str, np.ndarray], title: str) -> None:
    led_arrays = list(led_by_color.values())
    mins, maxs = finite_bounds(rigid_points, *led_arrays)
    bounds = padded_bounds(mins, maxs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_projection(axes[0], 0, 1, rigid_points, led_by_color, bounds, "XY Projection")
    plot_projection(axes[1], 0, 2, rigid_points, led_by_color, bounds, "XZ Projection")
    plot_projection(axes[2], 1, 2, rigid_points, led_by_color, bounds, "YZ Projection")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        dedup = {}
        for handle, label in zip(handles, labels):
            dedup[label] = handle
        fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=min(6, len(dedup)), frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.92])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_meta(path: Path, meta: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def correct_trial(args: argparse.Namespace) -> dict:
    trial_dir, rigid_csv, led_csv, calib_json = resolve_inputs(args)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_output_dir(trial_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rigid_rows, source_pos, source_rot, rigid_used_smoothed, rigid_columns = load_rigid_poses(rigid_csv, prefer_smoothed=not args.use_raw)
    led_rows, source_led_points, led_used_smoothed, led_columns = load_led_rows(led_csv, prefer_smoothed=not args.use_raw)

    cameras = load_calibration(str(calib_json))
    camera_centers = get_camera_centers_world(cameras)
    corrected_transform = build_corrected_transform(cameras, camera_centers)
    corrected_pos = apply_corrected_transform(source_pos, corrected_transform)
    corrected_led_points = apply_corrected_transform(source_led_points, corrected_transform) if len(source_led_points) else np.zeros((0, 3), dtype=np.float64)
    correction_rotation = np.asarray(corrected_transform["R"], dtype=np.float64).reshape(3, 3)
    corrected_rot = np.asarray([correction_rotation @ rot for rot in source_rot], dtype=np.float64)

    trajectory_path = out_dir / "corrected_trajectory.csv"
    led_path = out_dir / "corrected_leds.csv"
    meta_path = out_dir / "corrected_meta.json"
    preview_path = out_dir / "corrected_preview.png"

    write_corrected_trajectory(trajectory_path, rigid_rows, corrected_pos, corrected_rot)
    write_corrected_leds(led_path, led_rows, corrected_led_points)
    led_by_color = group_led_points(led_rows, corrected_led_points)
    write_preview(preview_path, corrected_pos, led_by_color, "Corrected-frame trial projections\n%s" % trial_dir.name)

    bounds_min, bounds_max = finite_bounds(corrected_pos, corrected_led_points)
    meta = {
        "trial_dir": str(trial_dir),
        "rigid_csv": str(rigid_csv),
        "led_csv": str(led_csv) if led_csv else "",
        "calib_json": str(calib_json),
        "out_dir": str(out_dir),
        "prefer_smoothed": not args.use_raw,
        "rigid_used_smoothed": rigid_used_smoothed,
        "led_used_smoothed": led_used_smoothed,
        "rigid_columns": rigid_columns,
        "led_columns": led_columns,
        "num_rigid_rows": len(rigid_rows),
        "num_led_rows": len(led_rows),
        "corrected_transform": {
            "origin": np.asarray(corrected_transform["origin"], dtype=np.float64).tolist(),
            "R": correction_rotation.tolist(),
        },
        "bounds": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
        },
        "outputs": {
            "corrected_trajectory": str(trajectory_path),
            "corrected_leds": str(led_path),
            "corrected_meta": str(meta_path),
            "corrected_preview": str(preview_path),
        },
    }
    write_meta(meta_path, meta)
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-dir", required=True, help="trial dir containing trajectory/")
    parser.add_argument("--calib-json", required=True, help="ChArUco calibration JSON used to define corrected frame")
    parser.add_argument("--out-dir", default=None, help="output directory (default: data/processed/simulation/<task>/<trial>)")
    parser.add_argument("--rigid-csv", default=None, help="override path to rigid_pose_6d.csv")
    parser.add_argument("--led-csv", default=None, help="override path to trajectory_led.csv")
    parser.add_argument("--use-raw", action="store_true", help="use raw pose/LED columns instead of smoothed columns when available")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    meta = correct_trial(args)
    print("[trajectory-corrector] wrote %s" % meta["outputs"]["corrected_trajectory"])
    print("[trajectory-corrector] wrote %s" % meta["outputs"]["corrected_leds"])
    print("[trajectory-corrector] wrote %s" % meta["outputs"]["corrected_meta"])
    print("[trajectory-corrector] wrote %s" % meta["outputs"]["corrected_preview"])
    print("[trajectory-corrector] rigid rows: %d, LED rows: %d" % (meta["num_rigid_rows"], meta["num_led_rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
