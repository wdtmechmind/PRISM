#!/usr/bin/env python3
# ruff: noqa: E402  (sys.path manipulation before imports)
"""
calibrate_apriltag_rig.py — 标定贴在同一刚体上的多个 AprilTag 的相对位置。

========= 原理 =========
刚体上贴了若干 tag36h11，方向分散，任一时刻只能看到其中几个。利用已标定的
4 相机内外参：
  1. 每个可见 tag 用 solvePnP 求出它在各相机中的 6DOF 姿态，再用外参换算到
     世界坐标系（cam0）。多相机看到同一 tag 时取加权平均。
  2. 逐帧累积任意两个共同可见 tag 之间的相对变换 inv(T_a)@T_b 并平均。
  3. 以参考 tag 为根做"最大生成树"，让每个 tag 得到相对刚体（参考 tag）坐标
     系的固定姿态——即使某个 tag 从未与参考 tag 同时出现，也能经中间 tag 链接。

结果写入 JSON（默认 configs/devices/apriltag_rig.json），供
track_apriltag_trajectory.py 追踪整段轨迹使用。

========= 用法 =========
  # 用一段专门"翻转刚体让所有 tag 都露出"的录制来标定
  python tools/calibrate_apriltag_rig.py \\
      data/raw/task_xxx_calib \\
      --tag-size 0.01 \\
      --calib-json configs/devices/charuco_4cam_result.json \\
      --output configs/devices/apriltag_rig.json

  # 指定参考 tag（默认自动选出现次数最多的）
  python tools/calibrate_apriltag_rig.py data/raw/task_xxx_calib \\
      --tag-size 0.01 --reference-tag 0

注意:
  * --tag-size 必须等于 tag 黑色边框正方形的实际边长（米），用卡尺量。
    该值错误会让多相机对同一 tag 的世界位置不一致（见输出的 cam_spread 诊断）。
  * 同一刚体上的 tag ID 必须唯一；重复 ID 的 tag 会被自动丢弃并告警。
"""

import argparse
import json
import os
import sys

import numpy as np

_repo_src = os.path.join(os.path.dirname(__file__), '..', 'src')
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from prism.common import console
from prism.reconstruction.apriltag_tracking import (
    build_tag_rig,
    iter_trial_frames,
    make_detector,
    tag_world_poses,
)
from prism.reconstruction.calibration import load_calibration

DEFAULT_CALIB = 'configs/devices/charuco_4cam_result.json'
DEFAULT_OUTPUT = 'configs/devices/apriltag_rig.json'


def _list_trials(task_dir):
    return sorted(
        os.path.join(task_dir, name) for name in os.listdir(task_dir)
        if name.startswith('trial_') and os.path.isdir(os.path.join(task_dir, name))
    )


def main():
    parser = argparse.ArgumentParser(
        description='标定刚体上多个 AprilTag 的相对位置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('task_dir', help='包含 trial_* 的任务目录（建议用专门的标定录制）')
    parser.add_argument('--tag-size', type=float, required=True,
                        help='tag 黑边正方形实际边长（米），例如 0.01')
    parser.add_argument('--calib-json', default=DEFAULT_CALIB,
                        help='4 相机 charuco 标定 JSON')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='输出 rig JSON 路径')
    parser.add_argument('--reference-tag', type=int, default=None,
                        help='参考 tag ID（默认自动选出现最多的）')
    parser.add_argument('--tol-ms', type=float, default=8.0,
                        help='跨相机帧关联的最大时间差 (ms)')
    parser.add_argument('--max-reproj-px', type=float, default=3.0,
                        help='单 tag PnP 重投影误差上限 (像素)')
    parser.add_argument('--min-pair-frames', type=int, default=5,
                        help='一对 tag 建立连接所需的最少共见帧数')
    parser.add_argument('--stride', type=int, default=1,
                        help='每隔多少帧处理一次（加速标定）')
    args = parser.parse_args()

    task_dir = os.path.abspath(os.path.expanduser(args.task_dir))
    if not os.path.isdir(task_dir):
        console.warning('任务目录不存在: %s' % task_dir)
        sys.exit(1)

    cameras = load_calibration(os.path.expanduser(args.calib_json))
    detector = make_detector()
    tol_s = max(0.0, args.tol_ms / 1000.0)

    trials = _list_trials(task_dir)
    if not trials:
        console.warning('未找到 trial_* 目录: %s' % task_dir)
        sys.exit(1)

    console.rule('AprilTag rig 标定')
    console.info('标定源: %s (%d trials)' % (os.path.basename(task_dir), len(trials)))
    console.info('tag_size=%.4f m | 相机标定=%s' % (args.tag_size, args.calib_json))

    frame_poses = []
    tag_hits = {}
    spread_by_tag = {}
    for trial in trials:
        n_used = 0
        for j, _t_ref, _t_trial, frames in iter_trial_frames(trial, tol_s=tol_s):
            if args.stride > 1 and (j % args.stride) != 0:
                continue
            poses = tag_world_poses(frames, cameras, detector, args.tag_size,
                                    max_reproj_px=args.max_reproj_px)
            if not poses:
                continue
            frame_poses.append(poses)
            n_used += 1
            for tid, info in poses.items():
                tag_hits[tid] = tag_hits.get(tid, 0) + 1
                if info['num_cams'] > 1:
                    spread_by_tag.setdefault(tid, []).append(info['cam_spread_mm'])
        console.info('  %s: 采集 %d 帧含 tag' % (os.path.basename(trial), n_used))

    if not frame_poses:
        console.warning('没有检测到任何 tag，检查 tag family / tag-size / 光照')
        sys.exit(1)

    console.rule('检测统计')
    for tid in sorted(tag_hits):
        spr = spread_by_tag.get(tid)
        spr_txt = ('多相机一致性 %.2f mm' % float(np.mean(spr))) if spr else '单相机'
        console.info('  tag %-3d  出现 %5d 帧   %s' % (tid, tag_hits[tid], spr_txt))

    all_spread = [s for v in spread_by_tag.values() for s in v]
    if all_spread:
        med = float(np.median(all_spread))
        console.info('多相机位置一致性中位数: %.2f mm' % med)
        if med > 8.0:
            console.warning('一致性偏大 (> 8 mm)，--tag-size 可能不准，请复核实测边长')

    rig = build_tag_rig(frame_poses, reference_tag=args.reference_tag,
                        min_pair_frames=args.min_pair_frames)
    if rig['reference_tag'] is None:
        console.warning('无法建立 rig（可见 tag 太少）')
        sys.exit(1)

    console.rule('rig 结果')
    console.info('参考 tag（刚体原点）: %d' % rig['reference_tag'])
    console.info('%-6s  %-10s  %-12s  %-10s' % ('tag', 'link_from', '共见帧', '位置标准差'))
    for tid in sorted(rig['tags']):
        info = rig['tags'][tid]
        link = '(root)' if info['link_from'] is None else str(info['link_from'])
        t = info['T_body_tag'][:3, 3]
        console.info('%-6d  %-10s  %-12d  %.3f mm   body_xyz=(%.3f, %.3f, %.3f) m'
                     % (tid, link, info['pair_frames'], info['trans_std_mm'],
                        t[0], t[1], t[2]))
    if rig['skipped']:
        console.warning('未能连接（从未与已连接 tag 共见）: %s'
                        % ', '.join(str(t) for t in rig['skipped']))

    out = {
        'family': 'tag36h11',
        'tag_size_m': float(args.tag_size),
        'reference_tag': rig['reference_tag'],
        'source_task': task_dir,
        'calib_json': os.path.abspath(os.path.expanduser(args.calib_json)),
        'tags': {
            str(tid): {
                'T_body_tag': rig['tags'][tid]['T_body_tag'].tolist(),
                'link_from': rig['tags'][tid]['link_from'],
                'pair_frames': rig['tags'][tid]['pair_frames'],
                'trans_std_mm': rig['tags'][tid]['trans_std_mm'],
            }
            for tid in rig['tags']
        },
        'skipped_tags': rig['skipped'],
    }
    out_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    console.success('rig 已保存: %s' % out_path)


if __name__ == '__main__':
    main()
