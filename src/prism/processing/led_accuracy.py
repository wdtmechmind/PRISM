"""
LED 六自由度三角化定位精度分析核心逻辑。

供 session_manager（录制结束后自动调用）和
tools/eval_led_accuracy.py（命令行工具）共同使用。
"""

import csv
import math
import os
from collections import defaultdict
from itertools import combinations

import numpy as np

from prism.common import console

COLORS = ['red', 'yellow', 'blue', 'green']

# ─────────────────────────── I/O helpers ────────────────────────────


def load_traj_csv(path):
    """
    Returns dict: color -> {'t': np.ndarray, 'xyz': (N,3), 'reproj': np.ndarray}
    Only 'measured' rows are kept.
    """
    rows_by_color = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mode', '').strip() != 'measured':
                continue
            color = row['color'].strip()
            try:
                t = float(row['t_sec'])
                x = float(row['x_m'])
                y = float(row['y_m'])
                z = float(row['z_m'])
                reproj = float(row['max_norm_reproj_err']) if row.get('max_norm_reproj_err') else float('nan')
            except (ValueError, KeyError):
                continue
            rows_by_color[color].append((t, x, y, z, reproj))

    result = {}
    for color, rows in rows_by_color.items():
        rows.sort(key=lambda r: r[0])
        arr = np.array(rows, dtype=np.float64)
        result[color] = {
            't': arr[:, 0],
            'xyz': arr[:, 1:4],
            'reproj': arr[:, 4],
        }
    return result


def load_rigid_csv(path):
    """
    Returns dict with arrays: t, xyz (N,3), rpy (N,3)
    Only 'measured' rows.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mode', '').strip() != 'measured':
                continue
            try:
                t = float(row['t_sec'])
                x, y, z = float(row['x_m']), float(row['y_m']), float(row['z_m'])
                ro = float(row['roll_deg'])
                pi = float(row['pitch_deg'])
                ya = float(row['yaw_deg'])
            except (ValueError, KeyError):
                continue
            rows.append((t, x, y, z, ro, pi, ya))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows, dtype=np.float64)
    return {'t': arr[:, 0], 'xyz': arr[:, 1:4], 'rpy': arr[:, 4:7]}


# ─────────────────────────── Metric computations ────────────────────────────


def _synchronize_pair(data, c1, c2):
    """Time-aligned XYZ arrays for two colors (exact match → NN ≤ 100 ms)."""
    t1, xyz1 = data[c1]['t'], data[c1]['xyz']
    t2, xyz2 = data[c2]['t'], data[c2]['xyz']

    set1 = {float(f'{v:.6f}'): i for i, v in enumerate(t1)}
    common_idx1, common_idx2 = [], []
    for i2, v in enumerate(t2):
        key = float(f'{v:.6f}')
        if key in set1:
            common_idx1.append(set1[key])
            common_idx2.append(i2)

    if len(common_idx1) >= 10:
        return xyz1[common_idx1], xyz2[common_idx2]

    idx2_in_t1 = np.searchsorted(t1, t2)
    mask_r = np.clip(idx2_in_t1, 0, len(t1) - 1)
    mask_l = np.clip(idx2_in_t1 - 1, 0, len(t1) - 1)
    dt_r = np.abs(t1[mask_r] - t2)
    dt_l = np.abs(t1[mask_l] - t2)
    use_left = dt_l <= dt_r
    best_idx = np.where(use_left, mask_l, mask_r)
    best_dt = np.minimum(dt_r, dt_l)
    keep = best_dt <= 0.1
    if keep.sum() < 5:
        return None, None
    return xyz1[best_idx[keep]], xyz2[keep]


def rigid_distance_consistency(data):
    """
    For every pair of tracked colors, compute pairwise distance time series.
    Returns list of dicts with stats.
    """
    present = [c for c in COLORS if c in data]
    results = []
    for c1, c2 in combinations(present, 2):
        a1, a2 = _synchronize_pair(data, c1, c2)
        if a1 is None or len(a1) < 10:
            continue
        dist_mm = np.linalg.norm(a1 - a2, axis=1) * 1000.0
        results.append({
            'pair': f'{c1}-{c2}',
            'n': int(len(dist_mm)),
            'mean_mm': float(np.mean(dist_mm)),
            'std_mm': float(np.std(dist_mm, ddof=1)),
            'range_mm': float(np.max(dist_mm) - np.min(dist_mm)),
            'p95_dev_mm': float(np.percentile(np.abs(dist_mm - np.mean(dist_mm)), 95)),
        })
    return results


def reproj_error_stats(data):
    """Statistics of max_norm_reproj_err across all colors."""
    all_err = []
    for d in data.values():
        valid = d['reproj'][np.isfinite(d['reproj'])]
        all_err.append(valid)
    if not all_err:
        return None
    combined = np.concatenate(all_err)
    if len(combined) == 0:
        return None
    return {
        'n': int(len(combined)),
        'mean': float(np.mean(combined)),
        'median': float(np.median(combined)),
        'p95': float(np.percentile(combined, 95)),
        'max': float(np.max(combined)),
    }


def detect_static_window(data, min_frames=20, max_range_mm=3.0):
    """
    Auto-detect a static window (contiguous block where centroid moves < max_range_mm).
    Returns (t_start, t_end) or None.
    """
    t_map = defaultdict(list)
    for d in data.values():
        for i, t in enumerate(d['t']):
            t_map[t].append(d['xyz'][i])

    times = sorted(t_map.keys())
    if len(times) < min_frames:
        return None

    centroids = np.array([np.mean(t_map[t], axis=0) for t in times], dtype=np.float64)

    best_start, best_end, best_std = 0, 0, np.inf
    n = len(times)
    for i in range(n - min_frames + 1):
        window = centroids[i: i + min_frames]
        rng = (np.max(window, axis=0) - np.min(window, axis=0)) * 1000
        if np.max(rng) <= max_range_mm:
            std = float(np.std(np.linalg.norm(window - window.mean(axis=0), axis=1)))
            if std < best_std:
                best_std = std
                best_start = i
                best_end = i + min_frames - 1

    if best_std == np.inf:
        return None
    return float(times[best_start]), float(times[best_end])


def static_position_noise(data, t0, t1):
    """Per-LED XYZ std dev (mm) over window [t0, t1]."""
    results = {}
    for color, d in data.items():
        mask = (d['t'] >= t0) & (d['t'] <= t1)
        xyz_w = d['xyz'][mask]
        if len(xyz_w) < 5:
            continue
        std_xyz = np.std(xyz_w, axis=0, ddof=1) * 1000.0
        results[color] = {
            'n': int(len(xyz_w)),
            'std_x_mm': float(std_xyz[0]),
            'std_y_mm': float(std_xyz[1]),
            'std_z_mm': float(std_xyz[2]),
            'rms_3d_mm': float(np.sqrt(np.sum(std_xyz ** 2))),
        }
    return results


def static_pose_noise(rigid, t0, t1):
    """6DOF pose noise (mm / deg) over window [t0, t1]. Falls back to full data."""
    if rigid is None:
        return None

    def _stats(arr_xyz, arr_rpy):
        std_xyz = np.std(arr_xyz, axis=0, ddof=1) * 1000.0
        std_rpy = np.std(arr_rpy, axis=0, ddof=1)
        return {
            'n': int(len(arr_xyz)),
            'std_x_mm': float(std_xyz[0]),
            'std_y_mm': float(std_xyz[1]),
            'std_z_mm': float(std_xyz[2]),
            'rms_trans_mm': float(np.sqrt(np.sum(std_xyz ** 2))),
            'std_roll_deg': float(std_rpy[0]),
            'std_pitch_deg': float(std_rpy[1]),
            'std_yaw_deg': float(std_rpy[2]),
            'rms_rot_deg': float(np.sqrt(np.sum(std_rpy ** 2))),
        }

    if t0 is not None and t1 is not None:
        mask = (rigid['t'] >= t0) & (rigid['t'] <= t1)
        if mask.sum() >= 5:
            return _stats(rigid['xyz'][mask], rigid['rpy'][mask])

    if len(rigid['t']) >= 5:
        return _stats(rigid['xyz'], rigid['rpy'])
    return None


def tracking_coverage(data):
    """Fraction of unique timestamps where each LED is tracked."""
    t_set = set()
    for d in data.values():
        t_set.update(d['t'].tolist())
    total = len(t_set)
    results = {}
    for color, d in data.items():
        results[color] = {
            'tracked_frames': int(len(d['t'])),
            'coverage_pct': float(len(d['t']) / total * 100) if total > 0 else 0.0,
        }
    return results, total


def _split_runs(t, dt_gap):
    """Split a time array into contiguous [start, end) runs where consecutive dt <= dt_gap."""
    if len(t) == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > dt_gap:
            runs.append((start, i))
            start = i
    runs.append((start, len(t)))
    return runs


def _moving_average(seg, window):
    """Centered moving average along axis 0 with a shrinking window at the edges."""
    n = len(seg)
    half = window // 2
    out = np.empty_like(seg, dtype=np.float64)
    csum = np.concatenate([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)], axis=0)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def frame_consistency(data, smooth_window=5, dt_gap_factor=2.5):
    """
    Frame-to-frame consistency / motion jitter (works during motion, not just static).

    For each LED, temporal smoothness is characterized independently of the true
    motion by separating high-frequency jitter from the low-frequency trajectory:

      - median/p95/max_step_mm : inter-frame displacement magnitude (mm), describes
        how much the LED actually moves between frames (motion magnitude + spikes).
      - jitter_rms_mm / jitter_p95_mm : RMS / P95 of the high-pass residual
        (raw − centered moving average), i.e. deviation from the smooth trajectory.
        This is the frame-to-frame jitter and is largely independent of true motion.
      - accel_rms_mm : RMS of the discrete second difference (jerkiness proxy).

    Only consecutive samples whose dt <= dt_gap_factor * median(dt) are used, so
    tracking dropouts do not inflate the metrics.
    """
    results = {}
    for color in COLORS:
        if color not in data:
            continue
        d = data[color]
        t, xyz = d['t'], d['xyz']
        if len(t) < 2:
            continue
        dt = np.diff(t)
        med_dt = float(np.median(dt)) if len(dt) else 0.0
        dt_gap = med_dt * dt_gap_factor if med_dt > 0 else np.inf

        steps_mm, resid_list, accel_mm = [], [], []
        for s, e in _split_runs(t, dt_gap):
            seg = xyz[s:e]
            n = len(seg)
            if n >= 2:
                steps_mm.append(np.linalg.norm(np.diff(seg, axis=0), axis=1) * 1000.0)
            if n >= 3:
                accel_mm.append(np.linalg.norm(np.diff(seg, n=2, axis=0), axis=1) * 1000.0)
            if n >= smooth_window:
                resid_list.append((seg - _moving_average(seg, smooth_window)) * 1000.0)

        if not steps_mm:
            continue
        steps_all = np.concatenate(steps_mm)
        entry = {
            'n': int(len(t)),
            'median_step_mm': float(np.median(steps_all)),
            'p95_step_mm': float(np.percentile(steps_all, 95)),
            'max_step_mm': float(np.max(steps_all)),
            'jitter_rms_mm': None,
            'jitter_p95_mm': None,
            'accel_rms_mm': None,
        }
        if resid_list:
            resid_all = np.concatenate(resid_list, axis=0)
            resid_norm = np.linalg.norm(resid_all, axis=1)
            entry['jitter_rms_mm'] = float(np.sqrt(np.mean(resid_norm ** 2)))
            entry['jitter_p95_mm'] = float(np.percentile(resid_norm, 95))
        if accel_mm:
            accel_all = np.concatenate(accel_mm)
            entry['accel_rms_mm'] = float(np.sqrt(np.mean(accel_all ** 2)))
        results[color] = entry
    return results


# ─────────────────────────── Report printing ────────────────────────────


def _sep(width=72):
    return '─' * width


def _header(title, width=72):
    pad = max(0, width - len(title) - 4)
    return f'  {title}  {"─" * pad}'


def print_accuracy_report(traj_path, rigid_path=None,
                          static_t0=None, static_t1=None,
                          static_min_frames=20, static_max_range_mm=3.0):
    """
    Load CSVs, compute all metrics, print a formatted report.

    Returns a dict with all computed values (suitable for JSON serialisation).
    """
    data = load_traj_csv(traj_path)
    rigid = load_rigid_csv(rigid_path) if rigid_path and os.path.isfile(rigid_path) else None

    if not data:
        console.warning('精度分析: 轨迹 CSV 无有效 measured 行，跳过。')
        return {}

    # Determine static window
    manual_static = None
    auto_static = None
    if static_t0 is not None and static_t1 is not None:
        manual_static = (static_t0, static_t1)
    else:
        auto_static = detect_static_window(
            data, min_frames=static_min_frames, max_range_mm=static_max_range_mm
        )

    sw_t0, sw_t1 = manual_static if manual_static else (auto_static or (None, None))

    if auto_static:
        sw_src = f'自动检测: t ∈ [{sw_t0:.2f}, {sw_t1:.2f}] s'
    elif manual_static:
        sw_src = f'手动指定: t ∈ [{sw_t0:.2f}, {sw_t1:.2f}] s'
    else:
        sw_src = '未找到合适的静止段'

    print()
    print(_sep())
    print('  PRISM LED 定位精度评估报告')
    print(f'  轨迹文件: {os.path.basename(traj_path)}')
    if rigid_path:
        print(f'  姿态文件: {os.path.basename(rigid_path)}')
    print(_sep())

    # 1. Tracking coverage
    cov, total_frames = tracking_coverage(data)
    print()
    print(_header('1. 追踪覆盖率'))
    print(f'  总时间帧数 (唯一时间戳): {total_frames}')
    for color in COLORS:
        if color in cov:
            c = cov[color]
            print(f'    {color:<8}  {c["tracked_frames"]:>5} 帧   {c["coverage_pct"]:5.1f}%')

    # 2. Reprojection error
    re = reproj_error_stats(data)
    print()
    print(_header('2. 重投影误差 (归一化像素坐标)'))
    if re:
        print(f'  样本数   : {re["n"]}')
        print(f'  均值     : {re["mean"]:.6f}')
        print(f'  中位数   : {re["median"]:.6f}')
        print(f'  P95      : {re["p95"]:.6f}')
        print(f'  最大值   : {re["max"]:.6f}')
    else:
        print('  无有效重投影误差数据')

    # 3. Rigid body distance consistency
    rdc = rigid_distance_consistency(data)
    print()
    print(_header('3. 刚体 LED 间距一致性（定位精度代理指标）'))
    if rdc:
        print(f'  {"LED 对":<16}  {"样本数":>6}  {"均值/mm":>8}  {"标准差/mm":>10}  {"P95偏差/mm":>10}  {"极差/mm":>8}')
        print(f'  {"─"*16}  {"─"*6}  {"─"*8}  {"─"*10}  {"─"*10}  {"─"*8}')
        for r in rdc:
            print(
                f'  {r["pair"]:<16}  {r["n"]:>6}  '
                f'{r["mean_mm"]:>8.2f}  {r["std_mm"]:>10.3f}  '
                f'{r["p95_dev_mm"]:>10.3f}  {r["range_mm"]:>8.3f}'
            )
        stds = [r['std_mm'] for r in rdc]
        mean_std = float(np.mean(stds))
        sigma_led = mean_std / math.sqrt(2)
        print()
        print(f'  平均间距标准差  : {mean_std:.3f} mm')
        print(f'  推算单 LED 噪声 : σ ≈ {sigma_led:.3f} mm  (假设各 LED 噪声独立同分布)')
    else:
        print('  LED 对数不足（需要至少两个 LED 同时追踪）')

    # 4. Static position noise
    print()
    print(_header('4. 静止段 LED 位置噪声'))
    print(f'  静止段: {sw_src}')
    if sw_t0 is not None:
        noise = static_position_noise(data, sw_t0, sw_t1)
        if noise:
            print(f'  {"LED":<8}  {"帧数":>5}  {"σ_X/mm":>7}  {"σ_Y/mm":>7}  {"σ_Z/mm":>7}  {"3D-RMS/mm":>10}')
            print(f'  {"─"*8}  {"─"*5}  {"─"*7}  {"─"*7}  {"─"*7}  {"─"*10}')
            for color in COLORS:
                if color in noise:
                    n = noise[color]
                    print(
                        f'  {color:<8}  {n["n"]:>5}  '
                        f'{n["std_x_mm"]:>7.3f}  {n["std_y_mm"]:>7.3f}  '
                        f'{n["std_z_mm"]:>7.3f}  {n["rms_3d_mm"]:>10.3f}'
                    )
        else:
            print('  静止段内有效帧不足（< 5 帧）')

    # 5. 6DOF pose noise
    print()
    print(_header('5. 刚体 6DOF 姿态噪声'))
    if rigid is None:
        print('  未提供 rigid_pose_6d.csv，跳过')
    else:
        pn = static_pose_noise(rigid, sw_t0, sw_t1)
        if pn:
            win_desc = f't ∈ [{sw_t0:.2f}, {sw_t1:.2f}] s' if sw_t0 else '全程数据'
            print(f'  数据段: {win_desc}   (n={pn["n"]})')
            print()
            print(f'  平移噪声:')
            print(f'    σ_X     = {pn["std_x_mm"]:.3f} mm')
            print(f'    σ_Y     = {pn["std_y_mm"]:.3f} mm')
            print(f'    σ_Z     = {pn["std_z_mm"]:.3f} mm')
            print(f'    3D-RMS  = {pn["rms_trans_mm"]:.3f} mm')
            print()
            print(f'  姿态噪声:')
            print(f'    σ_roll  = {pn["std_roll_deg"]:.4f} deg')
            print(f'    σ_pitch = {pn["std_pitch_deg"]:.4f} deg')
            print(f'    σ_yaw   = {pn["std_yaw_deg"]:.4f} deg')
            print(f'    3D-RMS  = {pn["rms_rot_deg"]:.4f} deg')
        else:
            print('  6DOF 数据不足')

    # 6. Frame-to-frame consistency / motion jitter
    fc = frame_consistency(data)
    print()
    print(_header('6. 帧间一致性 / 运动抖动（全程，含运动段）'))
    if fc:
        print(f'  {"LED":<8}  {"帧数":>5}  {"中位步长/mm":>11}  {"最大步长/mm":>11}  {"抖动RMS/mm":>10}  {"抖动P95/mm":>10}  {"加速度RMS/mm":>12}')
        print(f'  {"─"*8}  {"─"*5}  {"─"*11}  {"─"*11}  {"─"*10}  {"─"*10}  {"─"*12}')
        for color in COLORS:
            if color not in fc:
                continue
            e = fc[color]
            jr = f'{e["jitter_rms_mm"]:.3f}' if e['jitter_rms_mm'] is not None else '  n/a'
            jp = f'{e["jitter_p95_mm"]:.3f}' if e['jitter_p95_mm'] is not None else '  n/a'
            ar = f'{e["accel_rms_mm"]:.3f}' if e['accel_rms_mm'] is not None else '  n/a'
            print(
                f'  {color:<8}  {e["n"]:>5}  '
                f'{e["median_step_mm"]:>11.3f}  {e["max_step_mm"]:>11.3f}  '
                f'{jr:>10}  {jp:>10}  {ar:>12}'
            )
        print()
        print('  抖动RMS = 原始轨迹相对滑动平均的高通残差，反映帧间抖动（基本独立于真实运动）')
    else:
        print('  帧数不足，无法评估帧间一致性')

    print()
    print(_sep())
    print()

    return {
        'traj_csv': traj_path,
        'rigid_csv': rigid_path,
        'static_window': {'t0': sw_t0, 't1': sw_t1, 'source': sw_src},
        'tracking_coverage': cov,
        'total_frames': total_frames,
        'reprojection_error': re,
        'rigid_distance_consistency': rdc,
        'static_position_noise_mm': static_position_noise(data, sw_t0, sw_t1) if sw_t0 else None,
        'pose_noise': static_pose_noise(rigid, sw_t0, sw_t1),
        'frame_consistency': fc,
    }
