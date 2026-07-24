"""
Offline per-trial 3D reconstruction from recorded Hik videos.

For each ``trial_%06d`` directory under a task, this reads the 4 Hik camera
videos and their ``*_timestamps.csv``, re-detects the LEDs on every frame,
triangulates them with the ChArUco calibration, and writes per-trial trajectory
and rigid-pose CSVs carrying a DUAL TIME AXIS:

    - ``capture_wall_time``  absolute wall clock (shared with camera timestamps,
                             hand command logs, etc. for cross-modal alignment)
    - ``t_trial``            trial-relative time (0 = trial recording start)

Unlike the online trajectory (computed at preview-loop rate with Kalman
prediction fill), this reconstructs at the camera frame rate using measured
triangulation only, so it is reproducible and re-tunable without recollecting.

CLI:
    prism-reconstruct-trials data/raw/task_2026..._grasp-demo [--calib-json ...]
"""

import argparse
import csv
import glob
import json
import os
import re

import cv2
import numpy as np

from prism.common import console
from prism.common.config import load_yaml_config
from prism.reconstruction.calibration import load_calibration
from prism.reconstruction.realtime_reconstruction import (
    COLOR_ORDER,
    build_body_model,
    detect_all_colors,
    estimate_pose_from_model,
    matrix_to_rpy_zyx,
    robust_triangulate,
    update_body_model,
)

try:
    from prism.processing.led_accuracy import print_accuracy_report
except Exception:  # pragma: no cover - accuracy report is optional
    print_accuracy_report = None


COLOR_TO_PREFIX = {'red': 'r', 'yellow': 'y', 'blue': 'b', 'green': 'g'}

# HSV / detection defaults; overridden by the collection config when present.
HSV_DEFAULTS = {
    'r_h_low': 5, 'r_s_low': 80, 'r_v_low': 80, 'r_h_high': 24, 'r_s_high': 255, 'r_v_high': 255,
    'y_h_low': 25, 'y_s_low': 80, 'y_v_low': 80, 'y_h_high': 45, 'y_s_high': 255, 'y_v_high': 255,
    'b_h_low': 90, 'b_s_low': 80, 'b_v_low': 80, 'b_h_high': 135, 'b_s_high': 255, 'b_v_high': 255,
    'g_h_low': 40, 'g_s_low': 60, 'g_v_low': 60, 'g_h_high': 95, 'g_s_high': 255, 'g_v_high': 255,
    'min_area': 10.0,
    'max_norm_reproj_error': 0.015,
}


TRAJ_HEADER = [
    't_sec', 't_trial', 'capture_wall_time', 'frame_index', 'color',
    'x_m', 'y_m', 'z_m', 'mode', 'num_views', 'max_norm_reproj_err', 'visible_cams',
    'x_smooth_m', 'y_smooth_m', 'z_smooth_m',
]
RIGID_HEADER = [
    't_sec', 't_trial', 'capture_wall_time', 'frame_index', 'mode',
    'num_leds_used', 'modeled_leds', 'visible_leds',
    'x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg',
    'x_smooth_m', 'y_smooth_m', 'z_smooth_m',
    'roll_smooth_deg', 'pitch_smooth_deg', 'yaw_smooth_deg',
]


def _median_filter_1d(y, window):
    """Centered median filter with shrinking edges; removes single-frame spikes."""
    if window <= 1 or len(y) < 2:
        return y
    half = window // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        out[i] = np.median(y[a:b])
    return out


def _moving_average_1d(y, window):
    """Centered moving average with shrinking edges."""
    if window <= 1 or len(y) < 2:
        return y
    half = window // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        out[i] = np.mean(y[a:b])
    return out


def smooth_trajectory(frames, xyz, window, max_gap, despike_window=3):
    """Despike + short-gap interpolate + moving-average smooth a color's XYZ track.

    Works per contiguous segment (frames whose gap to the next measured frame is
    <= ``max_gap`` + 1). Within a segment, intermediate missing frames are filled
    by linear interpolation so the filter window is continuous, then smoothed
    values are sampled back onto the original measured frames. Segments separated
    by long gaps are smoothed independently so we never smooth across a dropout.

    Returns an ndarray aligned 1:1 with the input ``xyz`` rows.
    """
    frames = np.asarray(frames)
    xyz = np.asarray(xyz, dtype=np.float64)
    out = xyz.copy()
    n = len(frames)
    if n < 3 or window <= 1:
        return out

    seg_start = 0
    for i in range(1, n + 1):
        if i == n or (frames[i] - frames[i - 1]) > max_gap + 1:
            seg = slice(seg_start, i)
            fr = frames[seg]
            pts = xyz[seg]
            if len(fr) >= 3:
                dense = np.arange(int(fr[0]), int(fr[-1]) + 1)
                dense_xyz = np.empty((len(dense), 3), dtype=np.float64)
                for a in range(3):
                    col = np.interp(dense, fr, pts[:, a])
                    col = _median_filter_1d(col, despike_window)
                    col = _moving_average_1d(col, window)
                    dense_xyz[:, a] = col
                pos = {int(f): k for k, f in enumerate(dense)}
                for k, f in enumerate(fr):
                    out[seg_start + k] = dense_xyz[pos[int(f)]]
            seg_start = i
    return out


def _quat_from_matrix(rot):
    """Rotation matrix -> unit quaternion [w, x, y, z]."""
    m = np.asarray(rot, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _matrix_from_quat(q):
    """Unit quaternion [w, x, y, z] -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def smooth_rotations(frames, mats, window, max_gap):
    """Smooth a sequence of rotation matrices via windowed quaternion averaging.

    Quaternions are sign-aligned (double-cover safe) before averaging so we never
    average across an antipodal flip. Tracks are split on gaps > ``max_gap`` + 1
    and smoothed per segment. Returns smoothed rotation matrices aligned 1:1 with
    the input ``mats``.
    """
    frames = np.asarray(frames)
    n = len(frames)
    quats = np.array([_quat_from_matrix(m) for m in mats], dtype=np.float64)
    out = [np.asarray(m, dtype=np.float64) for m in mats]
    if n < 3 or window <= 1:
        return out

    half = window // 2
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or (frames[i] - frames[i - 1]) > max_gap + 1:
            seg = list(range(seg_start, i))
            if len(seg) >= 3:
                # Canonicalize signs along the segment for continuity.
                for k in range(1, len(seg)):
                    if np.dot(quats[seg[k]], quats[seg[k - 1]]) < 0.0:
                        quats[seg[k]] = -quats[seg[k]]
                for pos, gi in enumerate(seg):
                    a = max(0, pos - half)
                    b = min(len(seg), pos + half + 1)
                    ref = quats[seg[pos]]
                    acc = np.zeros(4, dtype=np.float64)
                    for wpos in range(a, b):
                        qk = quats[seg[wpos]]
                        if np.dot(qk, ref) < 0.0:
                            qk = -qk
                        acc += qk
                    nrm = np.linalg.norm(acc)
                    if nrm > 1e-12:
                        out[gi] = _matrix_from_quat(acc / nrm)
            seg_start = i
    return out



def read_kv_metadata(path):
    """Parse a ``key: <json-value>`` metadata file into a dict."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, _, raw = line.partition(':')
            key = key.strip()
            raw = raw.strip()
            if not key:
                continue
            try:
                out[key] = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                out[key] = raw
    return out


def resolve_hsv_config(*config_paths):
    """Start from defaults and overlay any HSV / detection keys found in configs."""
    cfg = dict(HSV_DEFAULTS)
    for path in config_paths:
        if not path or not os.path.exists(os.path.expanduser(path)):
            continue
        data = load_yaml_config(path)
        for key in HSV_DEFAULTS:
            if key in data:
                cfg[key] = data[key]
    return cfg


def build_hsv_cfg(cfg):
    out = {}
    for color, pfx in COLOR_TO_PREFIX.items():
        out[color] = (
            (int(cfg['%s_h_low' % pfx]), int(cfg['%s_s_low' % pfx]), int(cfg['%s_v_low' % pfx])),
            (int(cfg['%s_h_high' % pfx]), int(cfg['%s_s_high' % pfx]), int(cfg['%s_v_high' % pfx])),
        )
    return out


def read_timestamps(path):
    """Return the capture_wall_time column as a float ndarray."""
    times = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'capture_wall_time' not in reader.fieldnames:
            raise RuntimeError('missing capture_wall_time in %s' % path)
        for row in reader:
            try:
                times.append(float(row['capture_wall_time']))
            except (ValueError, TypeError):
                continue
    return np.asarray(times, dtype=np.float64)


def discover_hik_streams(cameras_dir):
    """Return {cam_index: {'video', 'ts'}} for hik streams that have both files."""
    streams = {}
    for video in sorted(glob.glob(os.path.join(cameras_dir, 'hik*.mp4'))):
        base = os.path.basename(video)
        m = re.match(r'hik(\d+)_', base)
        if not m:
            continue
        cam_i = int(m.group(1))
        ts_csv = os.path.splitext(video)[0] + '_timestamps.csv'
        if not os.path.isfile(ts_csv):
            console.warning('skip %s (timestamps missing)' % base)
            continue
        try:
            ts = read_timestamps(ts_csv)
        except Exception as exc:
            console.warning('skip %s (%s)' % (base, exc))
            continue
        if len(ts) < 2:
            console.warning('skip %s (too few timestamps)' % base)
            continue
        streams[cam_i] = {'video': video, 'ts': ts}
    return streams


def _visible_str(cam_indices):
    return ','.join('cam%d' % i for i in sorted(cam_indices))


def reconstruct_trial(trial_dir, cameras, hsv_cfg, min_area, max_reproj, tol_s,
                      smooth_window=5, smooth_max_gap=3, despike_window=3):
    """Reconstruct one trial; write trajectory + rigid pose CSVs. Returns paths."""
    cameras_dir = os.path.join(trial_dir, 'cameras')
    streams = discover_hik_streams(cameras_dir)
    if len(streams) < 2:
        console.warning('%s: need >=2 hik streams, found %d; skipping'
                        % (os.path.basename(trial_dir), len(streams)))
        return None, None

    ref_i = min(streams)
    ts_by_cam = {i: streams[i]['ts'] for i in streams}
    caps = {i: cv2.VideoCapture(streams[i]['video']) for i in streams}

    # Prime one decoded frame per camera.
    cur = {}
    idx = {}
    for i in streams:
        ok, frame = caps[i].read()
        cur[i] = frame if ok else None
        idx[i] = 0

    meta = read_kv_metadata(os.path.join(trial_dir, 'metadata.yaml'))
    trial_start = meta.get('start_wall_time')
    if not isinstance(trial_start, (int, float)):
        trial_start = float(ts_by_cam[ref_i][0])

    traj_dir = os.path.join(trial_dir, 'trajectory')
    os.makedirs(traj_dir, exist_ok=True)
    traj_path = os.path.join(traj_dir, 'trajectory_led.csv')
    rigid_path = os.path.join(traj_dir, 'rigid_pose_6d.csv')

    ts_ref = ts_by_cam[ref_i]
    rigid_model = None
    n_measured = 0
    n_frames = 0
    # Accumulate measured LED points per color and rigid poses so we can smooth
    # each track before writing (smoothing needs the whole time series).
    led_rows = {name: [] for name in COLOR_ORDER}
    rigid_rows = []

    j = 0
    while cur[ref_i] is not None and j < len(ts_ref):
        t_ref = float(ts_ref[j])
        t_trial = t_ref - trial_start
        n_frames += 1

        obs = {name: {} for name in COLOR_ORDER}
        det_ref = detect_all_colors(cur[ref_i], hsv_cfg, min_area, {})
        for name in COLOR_ORDER:
            if det_ref[name] is not None:
                obs[name][ref_i] = det_ref[name]

        for i in streams:
            if i == ref_i:
                continue
            ts_i = ts_by_cam[i]
            # Advance this camera to the frame nearest t_ref.
            while idx[i] + 1 < len(ts_i) and abs(ts_i[idx[i] + 1] - t_ref) <= abs(ts_i[idx[i]] - t_ref):
                ok, frame = caps[i].read()
                if not ok:
                    cur[i] = None
                    break
                cur[i] = frame
                idx[i] += 1
            if cur[i] is not None and abs(float(ts_i[idx[i]]) - t_ref) <= tol_s:
                det_i = detect_all_colors(cur[i], hsv_cfg, min_area, {})
                for name in COLOR_ORDER:
                    if det_i[name] is not None:
                        obs[name][i] = det_i[name]

        point_by_color = {}
        for name in COLOR_ORDER:
            if len(obs[name]) < 2:
                continue
            x, errs = robust_triangulate(obs[name], cameras, max_reproj)
            if x is None:
                continue
            point_by_color[name] = np.asarray(x, dtype=np.float64).reshape(3)
            n_measured += 1
            led_rows[name].append({
                'frame': j,
                't_ref': t_ref,
                't_trial': t_trial,
                'xyz': point_by_color[name],
                'num_views': len(obs[name]),
                'max_err': max(errs.values()) if errs else float('nan'),
                'visible': _visible_str(obs[name].keys()),
            })

        visible_names = [n for n in COLOR_ORDER if n in point_by_color]
        if len(point_by_color) >= 3:
            if rigid_model is None:
                rigid_model = build_body_model(point_by_color)
            if rigid_model is not None:
                est = estimate_pose_from_model(rigid_model['model_points'], point_by_color)
                if est is not None:
                    pose_rot, pose_trans = est
                    update_body_model(rigid_model['model_points'], point_by_color, pose_rot, pose_trans)
                    modeled = [n for n in COLOR_ORDER if n in rigid_model['model_points']]
                    used = [n for n in modeled if n in point_by_color]
                    rigid_rows.append({
                        'frame': j,
                        't_ref': t_ref,
                        't_trial': t_trial,
                        'pos': np.asarray(pose_trans, dtype=np.float64).reshape(3),
                        'rot': np.asarray(pose_rot, dtype=np.float64).reshape(3, 3),
                        'num_used': len(used),
                        'modeled': ','.join(modeled),
                        'visible': ','.join(visible_names),
                    })

        ok, frame = caps[ref_i].read()
        cur[ref_i] = frame if ok else None
        j += 1

    for cap in caps.values():
        cap.release()

    # Smooth each color's track (despike + short-gap interp + moving average),
    # then write raw and smoothed positions side by side.
    with open(traj_path, 'w', newline='', encoding='utf-8') as traj_file:
        traj_writer = csv.writer(traj_file)
        traj_writer.writerow(TRAJ_HEADER)
        for name in COLOR_ORDER:
            rows = led_rows[name]
            if not rows:
                continue
            frames = np.array([r['frame'] for r in rows], dtype=np.int64)
            xyz = np.array([r['xyz'] for r in rows], dtype=np.float64)
            xyz_s = smooth_trajectory(frames, xyz, smooth_window, smooth_max_gap, despike_window)
            for r, s in zip(rows, xyz_s):
                traj_writer.writerow([
                    '%.6f' % r['t_trial'], '%.6f' % r['t_trial'], '%.6f' % r['t_ref'], r['frame'], name,
                    '%.9f' % r['xyz'][0], '%.9f' % r['xyz'][1], '%.9f' % r['xyz'][2],
                    'measured', r['num_views'], '%.6f' % r['max_err'], r['visible'],
                    '%.9f' % s[0], '%.9f' % s[1], '%.9f' % s[2],
                ])

    # Smooth rigid pose: position via the LED smoother, orientation via
    # quaternion averaging; write raw + smoothed columns.
    with open(rigid_path, 'w', newline='', encoding='utf-8') as rigid_file:
        rigid_writer = csv.writer(rigid_file)
        rigid_writer.writerow(RIGID_HEADER)
        if rigid_rows:
            r_frames = np.array([r['frame'] for r in rigid_rows], dtype=np.int64)
            r_pos = np.array([r['pos'] for r in rigid_rows], dtype=np.float64)
            r_mats = [r['rot'] for r in rigid_rows]
            pos_s = smooth_trajectory(r_frames, r_pos, smooth_window, smooth_max_gap, despike_window)
            mats_s = smooth_rotations(r_frames, r_mats, smooth_window, smooth_max_gap)
            for r, ps, ms in zip(rigid_rows, pos_s, mats_s):
                roll, pitch, yaw = matrix_to_rpy_zyx(r['rot'])
                rpy = np.degrees([roll, pitch, yaw])
                rolls, pitchs, yaws = matrix_to_rpy_zyx(ms)
                rpy_s = np.degrees([rolls, pitchs, yaws])
                rigid_writer.writerow([
                    '%.6f' % r['t_trial'], '%.6f' % r['t_trial'], '%.6f' % r['t_ref'], r['frame'], 'measured',
                    r['num_used'], r['modeled'], r['visible'],
                    '%.9f' % r['pos'][0], '%.9f' % r['pos'][1], '%.9f' % r['pos'][2],
                    '%.6f' % rpy[0], '%.6f' % rpy[1], '%.6f' % rpy[2],
                    '%.9f' % ps[0], '%.9f' % ps[1], '%.9f' % ps[2],
                    '%.6f' % rpy_s[0], '%.6f' % rpy_s[1], '%.6f' % rpy_s[2],
                ])

    console.success('%s: %d frames, %d measured LED points -> %s'
                    % (os.path.basename(trial_dir), n_frames, n_measured, traj_dir))

    # Render the 6D trajectory as a 3D path with RGB body frames.
    try:
        from prism.processing.trajectory_analyzer import plot_rigid_6d_frames
        plot_rigid_6d_frames(rigid_path, os.path.join(traj_dir, 'rigid_6d_frames.png'))
    except Exception as exc:
        console.warning('6D frame plot failed for %s: %s' % (os.path.basename(trial_dir), exc))

    return traj_path, rigid_path


def _resolve_calib_json(task_dir, calib_json, config_path):
    if calib_json:
        return os.path.abspath(os.path.expanduser(calib_json))
    # Fall back to the config recorded in task metadata, then the given config.
    candidates = []
    task_meta = read_kv_metadata(os.path.join(task_dir, 'task_metadata.yaml'))
    if task_meta.get('config'):
        candidates.append(task_meta['config'])
    if config_path:
        candidates.append(config_path)
    for cfg_path in candidates:
        data = load_yaml_config(cfg_path)
        if data.get('calib_json'):
            return os.path.abspath(os.path.expanduser(data['calib_json']))
    raise RuntimeError('could not resolve calibration json; pass --calib-json explicitly')


def reconstruct_task(task_dir, calib_json=None, config_path=None, tol_ms=8.0,
                     smooth_window=5, smooth_max_gap=3, despike_window=3):
    """Reconstruct every trial under a task directory."""
    task_dir = os.path.abspath(os.path.expanduser(task_dir))
    if not os.path.isdir(task_dir):
        raise RuntimeError('task dir not found: %s' % task_dir)

    calib_path = _resolve_calib_json(task_dir, calib_json, config_path)
    cameras = load_calibration(calib_path)

    task_meta = read_kv_metadata(os.path.join(task_dir, 'task_metadata.yaml'))
    cfg = resolve_hsv_config(task_meta.get('config'), config_path)
    hsv_cfg = build_hsv_cfg(cfg)
    min_area = float(cfg['min_area'])
    max_reproj = float(cfg['max_norm_reproj_error'])
    tol_s = max(0.0, float(tol_ms) / 1000.0)

    trial_dirs = sorted(
        os.path.join(task_dir, name) for name in os.listdir(task_dir)
        if name.startswith('trial_') and os.path.isdir(os.path.join(task_dir, name))
    )
    if not trial_dirs:
        console.warning('no trial_* directories under %s' % task_dir)
        return

    console.rule('Offline per-trial reconstruction')
    console.info('calibration: %s' % calib_path)
    console.info('trials: %d | reproj<=%.4f | assoc tol=%.1f ms | smooth win=%d despike=%d gap<=%d'
                 % (len(trial_dirs), max_reproj, tol_ms, smooth_window, despike_window, smooth_max_gap))

    for trial_dir in trial_dirs:
        traj_path, rigid_path = reconstruct_trial(
            trial_dir, cameras, hsv_cfg, min_area, max_reproj, tol_s,
            smooth_window=smooth_window, smooth_max_gap=smooth_max_gap, despike_window=despike_window)
        if traj_path and print_accuracy_report is not None:
            try:
                console.info('accuracy report for %s:' % os.path.basename(trial_dir))
                print_accuracy_report(
                    traj_path,
                    rigid_path=rigid_path if rigid_path and os.path.isfile(rigid_path) else None,
                )
            except Exception as exc:
                console.warning('accuracy report failed for %s: %s' % (os.path.basename(trial_dir), exc))

    console.done('offline reconstruction complete: %s' % task_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Offline per-trial LED reconstruction from recorded Hik videos.'
    )
    parser.add_argument('task_dir', help='task directory (contains trial_* folders)')
    parser.add_argument('--calib-json', type=str, default=None,
                        help='charuco calibration json; default reads it from task/config metadata')
    parser.add_argument('--config', type=str, default=None,
                        help='collection config yaml to source HSV thresholds from')
    parser.add_argument('--tol-ms', type=float, default=8.0,
                        help='max wall-time gap to associate frames across cameras')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='moving-average window (frames) for jitter smoothing; 1 disables')
    parser.add_argument('--despike-window', type=int, default=3,
                        help='median-filter window (frames) to remove single-frame outliers; 1 disables')
    parser.add_argument('--smooth-max-gap', type=int, default=3,
                        help='max missing-frame gap bridged by interpolation before a track is split')
    args = parser.parse_args(argv)

    reconstruct_task(args.task_dir, calib_json=args.calib_json,
                     config_path=args.config, tol_ms=args.tol_ms,
                     smooth_window=args.smooth_window, smooth_max_gap=args.smooth_max_gap,
                     despike_window=args.despike_window)


if __name__ == '__main__':
    main()
