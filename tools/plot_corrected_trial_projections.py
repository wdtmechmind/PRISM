#!/usr/bin/env python3
"""Plot corrected-frame trajectory projections for a PRISM trial.

Given a trial directory, this tool loads trajectory CSVs, remaps points into the
corrected frame defined by the ChArUco calibration, and renders XY / XZ / YZ
plane projections as a PNG.

Usage:
    python tools/plot_corrected_trial_projections.py \
        --trial-dir data/raw/task_xxx/trial_000001 \
        --calib-json configs/devices/charuco_4cam_result.json
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from prism.reconstruction.calibration import apply_corrected_transform, build_corrected_transform, get_camera_centers_world, load_calibration  # noqa: E402


LED_COLORS = {
    'red': '#d90429',
    'yellow': '#f4b400',
    'blue': '#2979ff',
    'green': '#00a86b',
}


def _pick(colnames, *candidates):
    for candidate in candidates:
        if candidate in colnames:
            return candidate
    return None


def _load_led_rows(path, prefer_smoothed=True):
    if not path or not os.path.isfile(path):
        return {}

    rows_by_color = {}
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        color_col = _pick(cols, 'color')
        x_col = _pick(cols, 'x_smooth_m', 'x_m') if prefer_smoothed else _pick(cols, 'x_m', 'x_smooth_m')
        y_col = _pick(cols, 'y_smooth_m', 'y_m') if prefer_smoothed else _pick(cols, 'y_m', 'y_smooth_m')
        z_col = _pick(cols, 'z_smooth_m', 'z_m') if prefer_smoothed else _pick(cols, 'z_m', 'z_smooth_m')
        t_col = _pick(cols, 't_trial', 't_sec')
        if not (color_col and x_col and y_col and z_col):
            return {}

        for row in reader:
            try:
                xyz = np.array([
                    float(row[x_col]),
                    float(row[y_col]),
                    float(row[z_col]),
                ], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(xyz).all():
                continue
            color = str(row.get(color_col, '')).strip()
            if not color:
                continue
            t_value = float(row.get(t_col, 0.0)) if t_col and str(row.get(t_col, '')).strip() else float('nan')
            rows_by_color.setdefault(color, []).append((t_value, xyz))

    out = {}
    for color, items in rows_by_color.items():
        items.sort(key=lambda item: (math.inf if math.isnan(item[0]) else item[0]))
        out[color] = np.asarray([xyz for _t, xyz in items], dtype=np.float64)
    return out


def _load_rigid_rows(path, prefer_smoothed=True):
    if not path or not os.path.isfile(path):
        return None

    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        x_col = _pick(cols, 'x_smooth_m', 'x_m') if prefer_smoothed else _pick(cols, 'x_m', 'x_smooth_m')
        y_col = _pick(cols, 'y_smooth_m', 'y_m') if prefer_smoothed else _pick(cols, 'y_m', 'y_smooth_m')
        z_col = _pick(cols, 'z_smooth_m', 'z_m') if prefer_smoothed else _pick(cols, 'z_m', 'z_smooth_m')
        if not (x_col and y_col and z_col):
            return None

        rows = []
        for row in reader:
            mode = str(row.get('mode', 'measured')).strip()
            if mode not in ('', 'measured'):
                continue
            try:
                xyz = np.array([
                    float(row[x_col]),
                    float(row[y_col]),
                    float(row[z_col]),
                ], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(xyz).all():
                rows.append(xyz)

    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _resolve_trial_inputs(trial_dir, led_csv, rigid_csv, output_path):
    trial_dir = os.path.abspath(os.path.expanduser(trial_dir))
    if not os.path.isdir(trial_dir):
        raise SystemExit('trial dir not found: %s' % trial_dir)

    traj_dir = os.path.join(trial_dir, 'trajectory')
    led_csv = led_csv or os.path.join(traj_dir, 'trajectory_led.csv')
    rigid_csv = rigid_csv or os.path.join(traj_dir, 'rigid_pose_6d.csv')
    if output_path is None:
        output_path = os.path.join(traj_dir, 'corrected_projections.png')
    return trial_dir, led_csv, rigid_csv, os.path.abspath(os.path.expanduser(output_path))


def _build_corrected(trial_points, calib_json):
    cameras = load_calibration(calib_json)
    centers = get_camera_centers_world(cameras)
    corrected = build_corrected_transform(cameras, centers)

    led_corr = {color: apply_corrected_transform(points, corrected) for color, points in trial_points['led'].items()}
    rigid_corr = apply_corrected_transform(trial_points['rigid'], corrected) if trial_points['rigid'] is not None else None
    cam_corr = apply_corrected_transform(np.asarray([centers[k] for k in sorted(centers)], dtype=np.float64), corrected)
    return led_corr, rigid_corr, cam_corr


def _collect_bounds(led_corr, rigid_corr, cam_corr):
    chunks = []
    for points in led_corr.values():
        if points is not None and len(points) > 0:
            finite = points[np.isfinite(points).all(axis=1)]
            if len(finite) > 0:
                chunks.append(finite)
    if rigid_corr is not None and len(rigid_corr) > 0:
        finite = rigid_corr[np.isfinite(rigid_corr).all(axis=1)]
        if len(finite) > 0:
            chunks.append(finite)
    if cam_corr is not None and len(cam_corr) > 0:
        chunks.append(cam_corr[np.isfinite(cam_corr).all(axis=1)])
    if not chunks:
        return None
    all_pts = np.vstack(chunks)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(span * 0.08, 0.01)
    return mins - pad, maxs + pad


def _plot_projection(ax, axis_a, axis_b, led_corr, rigid_corr, cam_corr, bounds, title):
    axis_names = ['Xc (m)', 'Yc (m)', 'Zc (m)']
    mins, maxs = bounds

    for color, points in led_corr.items():
        if points is None or len(points) == 0:
            continue
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) == 0:
            continue
        ax.plot(finite[:, axis_a], finite[:, axis_b], color=LED_COLORS.get(color, '#666666'), linewidth=1.8, alpha=0.9, label='LED %s' % color)
        ax.scatter(finite[0, axis_a], finite[0, axis_b], color=LED_COLORS.get(color, '#666666'), s=24, marker='o')
        ax.scatter(finite[-1, axis_a], finite[-1, axis_b], color=LED_COLORS.get(color, '#666666'), s=28, marker='x')

    if rigid_corr is not None and len(rigid_corr) > 0:
        finite = rigid_corr[np.isfinite(rigid_corr).all(axis=1)]
        if len(finite) > 0:
            ax.plot(finite[:, axis_a], finite[:, axis_b], color='#111111', linewidth=2.2, alpha=0.8, linestyle='--', label='Rigid pose')

    if cam_corr is not None and len(cam_corr) > 0:
        ax.scatter(cam_corr[:, axis_a], cam_corr[:, axis_b], color='#444444', s=34, marker='^', label='Cameras')

    ax.set_xlim(mins[axis_a], maxs[axis_a])
    ax.set_ylim(mins[axis_b], maxs[axis_b])
    ax.set_xlabel(axis_names[axis_a])
    ax.set_ylabel(axis_names[axis_b])
    ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)
    ax.set_aspect('equal', adjustable='box')


def plot_trial_projections(trial_dir, calib_json, led_csv=None, rigid_csv=None, output_path=None, prefer_smoothed=True):
    trial_dir, led_csv, rigid_csv, output_path = _resolve_trial_inputs(trial_dir, led_csv, rigid_csv, output_path)
    calib_json = os.path.abspath(os.path.expanduser(calib_json))
    if not os.path.isfile(calib_json):
        raise SystemExit('calibration json not found: %s' % calib_json)

    led_points = _load_led_rows(led_csv, prefer_smoothed=prefer_smoothed)
    rigid_points = _load_rigid_rows(rigid_csv, prefer_smoothed=prefer_smoothed)
    if not led_points and rigid_points is None:
        raise SystemExit('no usable trajectory data found under %s' % trial_dir)

    led_corr, rigid_corr, cam_corr = _build_corrected({'led': led_points, 'rigid': rigid_points}, calib_json)
    bounds = _collect_bounds(led_corr, rigid_corr, cam_corr)
    if bounds is None:
        raise SystemExit('all trajectory points are invalid after corrected transform')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _plot_projection(axes[0], 0, 1, led_corr, rigid_corr, cam_corr, bounds, 'XY Projection (corrected)')
    _plot_projection(axes[1], 0, 2, led_corr, rigid_corr, cam_corr, bounds, 'XZ Projection (corrected)')
    _plot_projection(axes[2], 1, 2, led_corr, rigid_corr, cam_corr, bounds, 'YZ Projection (corrected)')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        dedup = {}
        for handle, label in zip(handles, labels):
            dedup[label] = handle
        fig.legend(dedup.values(), dedup.keys(), loc='upper center', ncol=min(6, len(dedup)), frameon=False)

    fig.suptitle('Corrected-frame trial projections\n%s' % os.path.basename(trial_dir), fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.90])

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trial-dir', required=True, help='trial dir containing trajectory/')
    parser.add_argument('--calib-json', required=True, help='calibration json used to define corrected frame')
    parser.add_argument('--led-csv', default=None, help='override path to trajectory_led.csv')
    parser.add_argument('--rigid-csv', default=None, help='override path to rigid_pose_6d.csv')
    parser.add_argument('--output', default=None, help='output PNG path (default: trial/trajectory/corrected_projections.png)')
    parser.add_argument('--use-raw', action='store_true', help='use raw xyz columns instead of smoothed columns when available')
    args = parser.parse_args(argv)

    output_path = plot_trial_projections(
        args.trial_dir,
        args.calib_json,
        led_csv=args.led_csv,
        rigid_csv=args.rigid_csv,
        output_path=args.output,
        prefer_smoothed=not args.use_raw,
    )
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())