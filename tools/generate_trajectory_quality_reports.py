#!/usr/bin/env python3
"""Generate trajectory quality reports from offline reconstruction outputs.

This tool is intentionally decoupled from post-processing. It scans either a
single PRISM trial directory or a full task directory, then runs the same
quality evaluation previously invoked during offline reconstruction.

Examples:
    python3 tools/generate_trajectory_quality_reports.py \
        --trial-dir data/raw/task_xxx/trial_000001

    python3 tools/generate_trajectory_quality_reports.py \
        --task-dir data/raw/task_xxx \
        --write-json --write-md
"""

import argparse
import json
import os
import sys

_repo_src = os.path.join(os.path.dirname(__file__), '..', 'src')
if os.path.isdir(_repo_src) and _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from prism.common import console  # noqa: E402
from prism.processing.led_accuracy import print_accuracy_report  # noqa: E402


def _iter_trial_dirs(task_dir):
    return sorted(
        os.path.join(task_dir, name)
        for name in os.listdir(task_dir)
        if name.startswith('trial_') and os.path.isdir(os.path.join(task_dir, name))
    )


def _resolve_targets(task_dir=None, trial_dir=None):
    if bool(task_dir) == bool(trial_dir):
        raise SystemExit('pass exactly one of --task-dir or --trial-dir')

    if trial_dir:
        trial_dir = os.path.abspath(os.path.expanduser(trial_dir))
        if not os.path.isdir(trial_dir):
            raise SystemExit('trial dir not found: %s' % trial_dir)
        return [trial_dir]

    task_dir = os.path.abspath(os.path.expanduser(task_dir))
    if not os.path.isdir(task_dir):
        raise SystemExit('task dir not found: %s' % task_dir)
    trial_dirs = _iter_trial_dirs(task_dir)
    if not trial_dirs:
        raise SystemExit('no trial_* directories under %s' % task_dir)
    return trial_dirs


def _default_output_paths(trial_dir, write_json, write_md):
    traj_dir = os.path.join(trial_dir, 'trajectory')
    json_path = os.path.join(traj_dir, 'accuracy_report.json') if write_json else None
    md_path = os.path.join(traj_dir, 'accuracy_report.md') if write_md else None
    return json_path, md_path


def _report_inputs(trial_dir):
    traj_dir = os.path.join(trial_dir, 'trajectory')
    traj_path = os.path.join(traj_dir, 'trajectory_led.csv')
    rigid_path = os.path.join(traj_dir, 'rigid_pose_6d.csv')
    if not os.path.isfile(traj_path):
        return None, None
    if not os.path.isfile(rigid_path):
        rigid_path = None
    return traj_path, rigid_path


def generate_reports(trial_dirs, write_json=False, write_md=True,
                     static_t0=None, static_t1=None,
                     static_min_frames=20, static_max_range_mm=3.0):
    generated = []
    skipped = []

    for trial_dir in trial_dirs:
        traj_path, rigid_path = _report_inputs(trial_dir)
        if traj_path is None:
            skipped.append({'trial_dir': trial_dir, 'reason': 'trajectory_led.csv missing'})
            console.warning('skip %s: trajectory_led.csv missing' % os.path.basename(trial_dir))
            continue

        json_path, md_path = _default_output_paths(trial_dir, write_json, write_md)
        console.rule('Trajectory Quality Report: %s' % os.path.basename(trial_dir))
        result = print_accuracy_report(
            traj_path,
            rigid_path=rigid_path,
            static_t0=static_t0,
            static_t1=static_t1,
            static_min_frames=static_min_frames,
            static_max_range_mm=static_max_range_mm,
            md_path=md_path,
        )
        if write_json and result:
            with open(json_path, 'w', encoding='utf-8') as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            console.saved('精度报告 (json): %s' % json_path)

        generated.append({
            'trial_dir': trial_dir,
            'traj_csv': traj_path,
            'rigid_csv': rigid_path,
            'json_report': json_path,
            'md_report': md_path,
        })

    return {'generated': generated, 'skipped': skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate offline trajectory quality reports for a PRISM trial or task directory.'
    )
    parser.add_argument('--task-dir', default=None,
                        help='task directory containing trial_* folders')
    parser.add_argument('--trial-dir', default=None,
                        help='single trial directory to analyze')
    parser.add_argument('--write-json', action='store_true',
                        help='write accuracy_report.json beside each trajectory CSV')
    parser.add_argument('--no-md', action='store_true',
                        help='disable markdown output (default writes accuracy_report.md)')
    parser.add_argument('--static-t0', type=float, default=None,
                        help='manual static-window start time in seconds')
    parser.add_argument('--static-t1', type=float, default=None,
                        help='manual static-window end time in seconds')
    parser.add_argument('--static-min-frames', type=int, default=20,
                        help='minimum frames for auto-detected static window')
    parser.add_argument('--static-max-range-mm', type=float, default=3.0,
                        help='maximum centroid motion range for auto static-window detection')
    args = parser.parse_args(argv)

    trial_dirs = _resolve_targets(task_dir=args.task_dir, trial_dir=args.trial_dir)
    summary = generate_reports(
        trial_dirs,
        write_json=args.write_json,
        write_md=not args.no_md,
        static_t0=args.static_t0,
        static_t1=args.static_t1,
        static_min_frames=args.static_min_frames,
        static_max_range_mm=args.static_max_range_mm,
    )
    console.done('trajectory quality report generation complete: generated=%d skipped=%d'
                 % (len(summary['generated']), len(summary['skipped'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())