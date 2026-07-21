#!/usr/bin/env python3
# ruff: noqa: E402  (sys.path manipulation before imports)
"""
eval_led_accuracy.py — LED 六自由度三角化定位精度评估工具

========= 原理 =========
由于没有外部真值参考（如 OptiTrack），本工具使用两类自洽度量：

  1. 刚体约束一致性（适用任意运动轨迹）
     四个 LED 固定在刚体手上，任意两 LED 间的欧氏距离应为物理常数。
     该距离在时间序列上的标准差即为综合定位噪声的代理指标。
     std(dist_ij) ≤ sqrt(2) * sigma_single_LED（假设各 LED 独立同分布噪声）

  2. 静止段位置噪声
     当手保持静止时，每个 LED 坐标的时序标准差直接反映定位噪声（mm 量级）。
     静止段可自动检测，也可通过 --static-t0 / --static-t1 手动指定。

  3. 重投影误差分布
     max_norm_reproj_err（归一化像素坐标下，最大相机重投影残差）的统计。

  4. 6DOF 姿态噪声（需提供 rigid_pose_6d.csv）
     静止段内 XYZ 平移（mm）和 RPY 姿态（deg）的标准差。

========= 用法 =========
  # 仅分析轨迹
  python tools/eval_led_accuracy.py \\
      data/raw/task_xxx/trajectory_led_nearest.csv

  # 同时分析 6DOF 姿态
  python tools/eval_led_accuracy.py \\
      data/raw/task_xxx/trajectory_led_nearest.csv \\
      --rigid data/raw/task_xxx/rigid_pose_6d.csv

  # 手动指定静止段（秒）
  python tools/eval_led_accuracy.py \\
      data/raw/task_xxx/trajectory_led_nearest.csv \\
      --static-t0 2.0 --static-t1 8.0

  # 保存结果到 JSON
  python tools/eval_led_accuracy.py \\
      data/raw/task_xxx/trajectory_led_nearest.csv \\
      --rigid data/raw/task_xxx/rigid_pose_6d.csv \\
      --output results.json
"""

import argparse
import json
import os
import sys

# Allow running directly from the repo root without installing the package.
_repo_src = os.path.join(os.path.dirname(__file__), '..', 'src')
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from prism.processing.led_accuracy import print_accuracy_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description='PRISM LED 六自由度定位精度评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('traj_csv', help='trajectory_led_nearest.csv 或 trajectory_led_interp.csv')
    parser.add_argument('--rigid', metavar='CSV', default=None,
                        help='rigid_pose_6d.csv（可选，用于 6DOF 姿态噪声）')
    parser.add_argument('--static-t0', type=float, default=None,
                        help='静止段起始时间（秒），不指定则自动检测')
    parser.add_argument('--static-t1', type=float, default=None,
                        help='静止段结束时间（秒）')
    parser.add_argument('--static-min-frames', type=int, default=20,
                        help='自动检测静止段的最小帧数 (默认 20)')
    parser.add_argument('--static-max-range-mm', type=float, default=3.0,
                        help='自动检测静止段的质心最大位移范围 mm (默认 3.0)')
    parser.add_argument('--output', metavar='JSON', default=None,
                        help='将结果保存为 JSON 文件')
    args = parser.parse_args()

    if not os.path.isfile(args.traj_csv):
        print(f'错误: 找不到文件 {args.traj_csv}', file=sys.stderr)
        sys.exit(1)

    rigid_path = args.rigid
    if rigid_path and not os.path.isfile(rigid_path):
        print(f'警告: 找不到 rigid_pose_6d.csv: {rigid_path}', file=sys.stderr)
        rigid_path = None

    result = print_accuracy_report(
        args.traj_csv,
        rigid_path=rigid_path,
        static_t0=args.static_t0,
        static_t1=args.static_t1,
        static_min_frames=args.static_min_frames,
        static_max_range_mm=args.static_max_range_mm,
    )

    if args.output and result:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'结果已保存: {args.output}')


if __name__ == '__main__':
    main()



def synchronize_pair(data, c1, c2):
    """
    Return time-aligned XYZ arrays for two colors.
    Uses exact timestamp intersection (trajectories may differ in length/timing).
    Falls back to nearest-neighbor within 100 ms if no exact match.
    """
    t1, xyz1 = data[c1]['t'], data[c1]['xyz']
    t2, xyz2 = data[c2]['t'], data[c2]['xyz']

    # Build set of common timestamps
    set1 = {float(f'{v:.6f}'): i for i, v in enumerate(t1)}
    common_idx1, common_idx2 = [], []
    for i2, v in enumerate(t2):
        key = float(f'{v:.6f}')
        if key in set1:
            common_idx1.append(set1[key])
            common_idx2.append(i2)

    if len(common_idx1) >= 10:
        return xyz1[common_idx1], xyz2[common_idx2]

    # Nearest-neighbour fallback (≤ 100 ms)
    idx2_in_t1 = np.searchsorted(t1, t2)
    mask_r = np.clip(idx2_in_t1, 0, len(t1) - 1)
    mask_l = np.clip(idx2_in_t1 - 1, 0, len(t1) - 1)
    dt_r = np.abs(t1[mask_r] - t2)
    dt_l = np.abs(t1[mask_l] - t2)
    use_left = dt_l <= dt_r
    best_idx = np.where(use_left, mask_l, mask_r)
    best_dt = np.minimum(dt_r, dt_l)
    keep = best_dt <= 0.1  # 100 ms
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
        a1, a2 = synchronize_pair(data, c1, c2)
        if a1 is None or len(a1) < 10:
            continue
        dist_m = np.linalg.norm(a1 - a2, axis=1)
        dist_mm = dist_m * 1000.0
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
    for color, d in data.items():
        err = d['reproj']
        valid = err[np.isfinite(err)]
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
    Auto-detect a static window: find the contiguous block of timestamps
    where the centroid of all visible LEDs moves < max_range_mm.

    Returns (t_start, t_end) or None.
    """
    # Collect all unique timestamps with centroid
    t_map = defaultdict(list)
    for color, d in data.items():
        for i, t in enumerate(d['t']):
            t_map[t].append(d['xyz'][i])

    times = sorted(t_map.keys())
    if len(times) < min_frames:
        return None

    centroids = np.array(
        [np.mean(t_map[t], axis=0) for t in times], dtype=np.float64
    )

    best_start, best_end, best_std = 0, 0, np.inf
    n = len(times)
    # Sliding window of size min_frames
    for i in range(n - min_frames + 1):
        window = centroids[i: i + min_frames]
        rng = (np.max(window, axis=0) - np.min(window, axis=0)) * 1000  # mm
        if np.max(rng) <= max_range_mm:
            std = float(np.std(np.linalg.norm(window - window.mean(axis=0), axis=1)))
            if std < best_std:
                best_std = std
                best_start = i
                best_end = i + min_frames - 1

    if best_end == 0 and best_start == 0:
        return None
    return float(times[best_start]), float(times[best_end])


def static_position_noise(data, t0, t1):
    """
    For each LED, compute XYZ std dev over the time window [t0, t1].
    Returns dict: color -> stats dict.
    """
    results = {}
    for color, d in data.items():
        mask = (d['t'] >= t0) & (d['t'] <= t1)
        xyz_w = d['xyz'][mask]
        if len(xyz_w) < 5:
            continue
        std_xyz = np.std(xyz_w, axis=0, ddof=1) * 1000.0  # mm
        results[color] = {
            'n': int(len(xyz_w)),
            'std_x_mm': float(std_xyz[0]),
            'std_y_mm': float(std_xyz[1]),
            'std_z_mm': float(std_xyz[2]),
            'rms_3d_mm': float(np.sqrt(np.sum(std_xyz ** 2))),
        }
    return results


def static_pose_noise(rigid, t0, t1):
    """
    6DOF pose noise over time window [t0, t1] from rigid_pose_6d.
    Falls back to full dataset if window has fewer than 5 frames.
    """
    if rigid is None:
        return None

    def _stats(arr_xyz, arr_rpy):
        std_xyz = np.std(arr_xyz, axis=0, ddof=1) * 1000.0  # mm
        std_rpy = np.std(arr_rpy, axis=0, ddof=1)           # deg
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

    # Fall back to whole dataset
    if len(rigid['t']) >= 5:
        return _stats(rigid['xyz'], rigid['rpy'])
    return None


def tracking_coverage(data, all_t=None):
    """Fraction of unique timestamps where each LED is tracked."""
    if all_t is None:
        t_set = set()
        for d in data.values():
            t_set.update(d['t'].tolist())
        total = len(t_set)
    else:
        total = len(all_t)

    results = {}
    for color, d in data.items():
        results[color] = {
            'tracked_frames': int(len(d['t'])),
            'coverage_pct': float(len(d['t']) / total * 100) if total > 0 else 0.0,
        }
    return results, total


# ─────────────────────────── Report formatting ────────────────────────────


def _sep(width=72):
    return '─' * width


def _header(title, width=72):
    pad = max(0, width - len(title) - 4)
    return f'  {title}  {"─" * pad}'


def print_report(traj_path, rigid_path, data, rigid, static_window, manual_static, args):
    print()
    print(_sep())
    print('  PRISM LED 定位精度评估报告')
    print(f'  轨迹文件: {os.path.basename(traj_path)}')
    if rigid_path:
        print(f'  姿态文件: {os.path.basename(rigid_path)}')
    print(_sep())

    # ── 1. Tracking coverage ──
    cov, total_frames = tracking_coverage(data)
    print()
    print(_header('1. 追踪覆盖率'))
    print(f'  总时间帧数 (唯一时间戳): {total_frames}')
    for color in COLORS:
        if color in cov:
            c = cov[color]
            print(f'    {color:<8}  {c["tracked_frames"]:>5} 帧   {c["coverage_pct"]:5.1f}%')

    # ── 2. Reprojection error ──
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

    # ── 3. Rigid body distance consistency ──
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

    # ── 4. Static window ──
    sw_t0, sw_t1 = None, None
    if manual_static:
        sw_t0, sw_t1 = manual_static
        src = f'手动指定: t ∈ [{sw_t0:.2f}, {sw_t1:.2f}] s'
    elif static_window:
        sw_t0, sw_t1 = static_window
        src = f'自动检测: t ∈ [{sw_t0:.2f}, {sw_t1:.2f}] s'
    else:
        src = '未找到合适的静止段'

    print()
    print(_header('4. 静止段 LED 位置噪声'))
    print(f'  静止段: {src}')

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

    # ── 5. 6DOF pose noise ──
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

    print()
    print(_sep())
    print()
    return {
        'traj_csv': traj_path,
        'rigid_csv': rigid_path,
        'static_window': {'t0': sw_t0, 't1': sw_t1, 'source': src},
        'tracking_coverage': cov,
        'total_frames': total_frames,
        'reprojection_error': re,
        'rigid_distance_consistency': rdc,
        'static_position_noise_mm': static_position_noise(data, sw_t0, sw_t1) if sw_t0 else None,
        'pose_noise': static_pose_noise(rigid, sw_t0, sw_t1),
    }



