#!/usr/bin/env python3
# ruff: noqa: E402  (sys.path manipulation before imports)
"""
track_apriltag_trajectory.py — 用标定好的 AprilTag rig 追踪刚体 6DOF 轨迹。

========= 原理 =========
对每一帧（4 相机时间对齐后）：
  1. 检测所有可见 tag，用 solvePnP + 相机外参求出每个 tag 在世界坐标系（cam0）
     的姿态（多相机看到同一 tag 时加权平均）。
  2. 每个已标定的可见 tag 用 rig 里的固定相对姿态"投票"出一个刚体位姿
     T_world_body = T_world_tag @ inv(T_body_tag)。
  3. 稳健融合所有投票（平移中位数剔野点 + 四元数加权平均）得到单一刚体位姿。

只要看到 1 个已标定 tag 就能恢复完整位姿，看到多个则更稳更准——这正好应对
"tag 方向分散、经常只看得到几个"的情况。

输出（写到每个 trial 的 trajectory/ 下，与 LED 管线格式一致，可直接用
eval_led_accuracy.py / trajectory_analyzer 分析）：
  - rigid_pose_6d_apriltag.csv   刚体 6DOF 位姿（含 smoothed 列）
  - tag_world_positions.csv      每个 tag 的世界坐标（诊断用）

========= 用法 =========
  python tools/track_apriltag_trajectory.py \\
      data/raw/task_xxx \\
      --rig configs/devices/apriltag_rig.json \\
      --calib-json configs/devices/charuco_4cam_result.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

_repo_src = os.path.join(os.path.dirname(__file__), '..', 'src')
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from prism.common import console
from prism.processing.offline_reconstruct import smooth_rotations, smooth_trajectory
from prism.reconstruction.apriltag_tracking import (
    estimate_body_pose,
    iter_trial_frames,
    make_detector,
    tag_world_poses,
)
from prism.reconstruction.calibration import load_calibration
from prism.reconstruction.realtime_reconstruction import matrix_to_rpy_zyx

DEFAULT_CALIB = 'configs/devices/charuco_4cam_result.json'
DEFAULT_RIG = 'configs/devices/apriltag_rig.json'

RIGID_HEADER = [
    't_sec', 't_trial', 'capture_wall_time', 'frame_index', 'mode',
    'num_leds_used', 'modeled_leds', 'visible_leds',
    'x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg',
    'x_smooth_m', 'y_smooth_m', 'z_smooth_m',
    'roll_smooth_deg', 'pitch_smooth_deg', 'yaw_smooth_deg',
]
TAG_HEADER = ['t_sec', 't_trial', 'capture_wall_time', 'frame_index',
              'tag_id', 'x_m', 'y_m', 'z_m', 'num_cams', 'reproj_px', 'mode']


def _load_rig(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tags = {}
    for tid_str, info in data['tags'].items():
        tags[int(tid_str)] = {
            'T_body_tag': np.asarray(info['T_body_tag'], dtype=np.float64),
        }
    return {'reference_tag': data.get('reference_tag'), 'tags': tags,
            'tag_size_m': float(data['tag_size_m'])}


def _list_trials(task_dir):
    return sorted(
        os.path.join(task_dir, name) for name in os.listdir(task_dir)
        if name.startswith('trial_') and os.path.isdir(os.path.join(task_dir, name))
    )


def track_trial(trial_dir, cameras, detector, rig, tag_size, tol_s,
                max_reproj_px, smooth_window, smooth_max_gap, despike_window):
    rows = []
    tag_rows = []
    for j, t_ref, t_trial, frames in iter_trial_frames(trial_dir, tol_s=tol_s):
        poses = tag_world_poses(frames, cameras, detector, tag_size,
                                max_reproj_px=max_reproj_px)
        for tid, info in poses.items():
            t = info['T'][:3, 3]
            tag_rows.append([
                '%.6f' % t_trial, '%.6f' % t_trial, '%.6f' % t_ref, j, tid,
                '%.9f' % t[0], '%.9f' % t[1], '%.9f' % t[2],
                info['num_cams'], '%.4f' % info['reproj_px'], 'measured',
            ])
        est = estimate_body_pose(poses, rig)
        if est is None:
            continue
        rows.append({
            'frame': j, 't_ref': t_ref, 't_trial': t_trial,
            'pos': est['t'], 'rot': est['R'],
            'num_used': est['num_tags'],
            'tags': ','.join(str(x) for x in sorted(est['tags'])),
        })

    traj_dir = os.path.join(trial_dir, 'trajectory')
    os.makedirs(traj_dir, exist_ok=True)
    rigid_path = os.path.join(traj_dir, 'rigid_pose_6d_apriltag.csv')
    tag_path = os.path.join(traj_dir, 'tag_world_positions.csv')

    modeled = ','.join(str(t) for t in sorted(rig['tags']))
    with open(rigid_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(RIGID_HEADER)
        if rows:
            r_frames = np.array([r['frame'] for r in rows], dtype=np.int64)
            r_pos = np.array([r['pos'] for r in rows], dtype=np.float64)
            r_mats = [r['rot'] for r in rows]
            pos_s = smooth_trajectory(r_frames, r_pos, smooth_window,
                                      smooth_max_gap, despike_window)
            mats_s = smooth_rotations(r_frames, r_mats, smooth_window, smooth_max_gap)
            for r, ps, ms in zip(rows, pos_s, mats_s):
                roll, pitch, yaw = np.degrees(matrix_to_rpy_zyx(r['rot']))
                rolls, pitchs, yaws = np.degrees(matrix_to_rpy_zyx(ms))
                w.writerow([
                    '%.6f' % r['t_trial'], '%.6f' % r['t_trial'], '%.6f' % r['t_ref'],
                    r['frame'], 'measured', r['num_used'], modeled, r['tags'],
                    '%.9f' % r['pos'][0], '%.9f' % r['pos'][1], '%.9f' % r['pos'][2],
                    '%.6f' % roll, '%.6f' % pitch, '%.6f' % yaw,
                    '%.9f' % ps[0], '%.9f' % ps[1], '%.9f' % ps[2],
                    '%.6f' % rolls, '%.6f' % pitchs, '%.6f' % yaws,
                ])

    with open(tag_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(TAG_HEADER)
        w.writerows(tag_rows)

    return rigid_path, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description='用 AprilTag rig 追踪刚体 6DOF 轨迹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('task_dir', help='包含 trial_* 的任务目录')
    parser.add_argument('--rig', default=DEFAULT_RIG, help='apriltag_rig.json')
    parser.add_argument('--calib-json', default=DEFAULT_CALIB, help='4 相机标定 JSON')
    parser.add_argument('--tag-size', type=float, default=None,
                        help='tag 黑边边长（米）；默认用 rig JSON 里的值')
    parser.add_argument('--tol-ms', type=float, default=8.0,
                        help='跨相机帧关联的最大时间差 (ms)')
    parser.add_argument('--max-reproj-px', type=float, default=3.0,
                        help='单 tag PnP 重投影误差上限 (像素)')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='滑动平均窗口 (帧)，1 关闭')
    parser.add_argument('--despike-window', type=int, default=3,
                        help='中值滤波窗口 (帧)，1 关闭')
    parser.add_argument('--smooth-max-gap', type=int, default=3,
                        help='插值可跨越的最大缺帧数')
    args = parser.parse_args()

    task_dir = os.path.abspath(os.path.expanduser(args.task_dir))
    if not os.path.isdir(task_dir):
        console.warning('任务目录不存在: %s' % task_dir)
        sys.exit(1)
    rig_path = os.path.expanduser(args.rig)
    if not os.path.isfile(rig_path):
        console.warning('找不到 rig: %s（先运行 calibrate_apriltag_rig.py）' % rig_path)
        sys.exit(1)

    rig = _load_rig(rig_path)
    tag_size = args.tag_size if args.tag_size is not None else rig['tag_size_m']
    cameras = load_calibration(os.path.expanduser(args.calib_json))
    detector = make_detector()
    tol_s = max(0.0, args.tol_ms / 1000.0)

    trials = _list_trials(task_dir)
    if not trials:
        console.warning('未找到 trial_* 目录: %s' % task_dir)
        sys.exit(1)

    console.rule('AprilTag 轨迹追踪')
    console.info('rig: %s | 参考 tag=%s | tag_size=%.4f m'
                 % (rig_path, rig['reference_tag'], tag_size))
    console.info('已建模 tag: %s' % ', '.join(str(t) for t in sorted(rig['tags'])))

    for trial in trials:
        rigid_path, n = track_trial(
            trial, cameras, detector, rig, tag_size, tol_s, args.max_reproj_px,
            args.smooth_window, args.smooth_max_gap, args.despike_window)
        console.info('  %s: %d 帧有效位姿 -> %s'
                     % (os.path.basename(trial), n, os.path.basename(rigid_path)))

    console.done('AprilTag 追踪完成: %s' % task_dir)


if __name__ == '__main__':
    main()
