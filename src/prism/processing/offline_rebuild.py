import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np

from prism.recording.video_recorder import make_ffmpeg_safe_fps


def parse_yes_no(raw_text):
    return str(raw_text).strip().lower() in ['y', 'yes', '1', 'true']


def read_timestamps_csv(path):
    timestamps = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError('empty csv: %s' % path)
        if 'capture_wall_time' not in reader.fieldnames:
            raise RuntimeError('missing capture_wall_time in %s' % path)
        for row in reader:
            try:
                timestamps.append(float(row['capture_wall_time']))
            except Exception:
                continue

    if len(timestamps) < 2:
        raise RuntimeError('not enough timestamps in %s' % path)
    return np.asarray(timestamps, dtype=np.float64)


def resolve_stream_dir(input_dir):
    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    if not os.path.isdir(input_dir):
        raise RuntimeError('input dir not found: %s' % input_dir)

    cameras_dir = os.path.join(input_dir, 'cameras')
    if os.path.isdir(cameras_dir):
        return cameras_dir, input_dir
    return input_dir, input_dir


def discover_streams(input_dir, include_rs):
    stream_dir, output_parent = resolve_stream_dir(input_dir)
    ts_files = sorted(glob.glob(os.path.join(stream_dir, '*_timestamps.csv')))
    streams = []
    for ts_csv in ts_files:
        base = os.path.basename(ts_csv)
        name = base[:-len('_timestamps.csv')]
        if (not include_rs) and name.startswith('realsense'):
            continue

        video_path = os.path.join(stream_dir, name + '.mp4')
        if not os.path.isfile(video_path):
            print('warning: skip %s (video missing)' % name)
            continue

        try:
            timestamps = read_timestamps_csv(ts_csv)
        except Exception as ex:
            print('warning: skip %s (%s)' % (name, str(ex)))
            continue

        duration = timestamps[-1] - timestamps[0]
        fps_est = (len(timestamps) - 1) / duration if duration > 1e-9 else 0.0
        streams.append({
            'name': name,
            'video': video_path,
            'ts_csv': ts_csv,
            'ts': timestamps,
            'capture_duration': duration,
            'capture_fps': fps_est,
        })

    if not streams:
        raise RuntimeError('no valid streams found in %s' % stream_dir)
    return streams, stream_dir, output_parent


def choose_target_fps(streams, target_fps, only_hik_for_auto=True):
    if target_fps and target_fps > 0:
        return make_ffmpeg_safe_fps(target_fps, default=120.0)

    fps_pool = []
    for stream in streams:
        if only_hik_for_auto and (not stream['name'].startswith('hik')):
            continue
        if stream['capture_fps'] > 1.0:
            fps_pool.append(stream['capture_fps'])

    if not fps_pool:
        fps_pool = [stream['capture_fps'] for stream in streams if stream['capture_fps'] > 1.0]
    if not fps_pool:
        return 120.0

    return make_ffmpeg_safe_fps(min(fps_pool), default=120.0)


def build_common_timeline(streams, target_fps, mode):
    starts = [stream['ts'][0] for stream in streams]
    ends = [stream['ts'][-1] for stream in streams]

    if mode == 'overlap':
        t0 = max(starts)
        t1 = min(ends)
    elif mode == 'union':
        t0 = min(starts)
        t1 = max(ends)
    else:
        raise RuntimeError('unsupported mode: %s' % mode)

    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise RuntimeError('invalid common time range: [%.6f, %.6f]' % (t0, t1))

    n_frames = int(math.floor((t1 - t0) * target_fps)) + 1
    n_frames = max(1, n_frames)
    timeline = t0 + np.arange(n_frames, dtype=np.float64) / float(target_fps)
    return timeline, t0, t1


def nearest_mapping(timestamps, timeline):
    pos = np.searchsorted(timestamps, timeline, side='left')
    right = np.clip(pos, 0, len(timestamps) - 1)
    left = np.clip(pos - 1, 0, len(timestamps) - 1)

    dt_right = np.abs(timestamps[right] - timeline)
    dt_left = np.abs(timestamps[left] - timeline)
    use_left = dt_left <= dt_right
    indices = np.where(use_left, left, right).astype(np.int64)
    dt_ms = (timestamps[indices] - timeline) * 1000.0
    return indices, dt_ms


def save_mapping_csv(path, timeline, indices, timestamps, dt_ms):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['out_frame_index', 'target_time', 'src_frame_index', 'src_capture_time', 'dt_ms'])
        for i in range(len(timeline)):
            writer.writerow([
                i,
                '%.6f' % timeline[i],
                int(indices[i]),
                '%.6f' % timestamps[indices[i]],
                '%.3f' % dt_ms[i],
            ])


def rebuild_one_stream(stream, out_video, out_map_csv, timeline, target_fps, codec):
    cap = cv2.VideoCapture(stream['video'])
    if not cap.isOpened():
        raise RuntimeError('failed to open video: %s' % stream['video'])

    source_timestamps = stream['ts']
    cap_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_source = len(source_timestamps)
    if cap_count > 0:
        n_source = min(n_source, cap_count)
    if n_source < 2:
        cap.release()
        raise RuntimeError('not enough source frames: %s' % stream['name'])

    source_timestamps = source_timestamps[:n_source]
    indices, dt_ms = nearest_mapping(source_timestamps, timeline)
    indices = np.clip(indices, 0, n_source - 1)
    save_mapping_csv(out_map_csv, timeline, indices, source_timestamps, dt_ms)

    counts = np.bincount(indices, minlength=n_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError('failed to read first frame: %s' % stream['video'])
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_video, fourcc, float(target_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('failed to open writer: %s' % out_video)

    written = 0
    source_read = 0
    for source_i in range(n_source):
        ok, frame = cap.read()
        if not ok:
            break
        source_read += 1
        for _ in range(int(counts[source_i])):
            writer.write(frame)
            written += 1

    cap.release()
    writer.release()

    return {
        'name': stream['name'],
        'src_frames': int(n_source),
        'src_read': int(source_read),
        'out_frames': int(written),
        'capture_duration_s': float(stream['capture_duration']),
        'capture_fps': float(stream['capture_fps']),
        'mean_abs_dt_ms': float(np.mean(np.abs(dt_ms))),
        'max_abs_dt_ms': float(np.max(np.abs(dt_ms))),
    }


def write_summary_csv(path, summary, target_fps, timeline, t0, t1, time_range):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'stream', 'src_frames', 'src_read', 'out_frames',
            'capture_duration_s', 'capture_fps', 'mean_abs_dt_ms', 'max_abs_dt_ms',
            'target_fps', 'timeline_frames', 'timeline_start', 'timeline_end', 'time_range',
        ])
        for item in summary:
            writer.writerow([
                item['name'], item['src_frames'], item['src_read'], item['out_frames'],
                '%.6f' % item['capture_duration_s'], '%.6f' % item['capture_fps'],
                '%.6f' % item['mean_abs_dt_ms'], '%.6f' % item['max_abs_dt_ms'],
                '%.6f' % target_fps, len(timeline), '%.6f' % t0, '%.6f' % t1, time_range,
            ])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Offline rebuild aligned constant-frame-rate videos from PRISM trial/segment timestamps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_dir', nargs='?', default='',
                        help='PRISM trial folder, cameras folder, or legacy segment folder')
    parser.add_argument('--segment-dir', type=str, default='',
                        help='backward-compatible alias for input_dir')
    parser.add_argument('--output-dir', type=str, default='',
                        help='output folder; default is <trial-or-segment>/aligned_offline')
    parser.add_argument('--target-fps', type=float, default=0.0,
                        help='target CFR fps; <=0 means auto(min measured Hik fps)')
    parser.add_argument('--time-range', type=str, default='overlap', choices=['overlap', 'union'],
                        help='overlap uses strict common range; union covers full range with edge duplication')
    parser.add_argument('--include-rs', type=str, default='y', help='include RealSense stream y/n')
    parser.add_argument('--codec', type=str, default='mp4v', help='fourcc codec, e.g. mp4v')
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.strip() or args.segment_dir.strip()
    if not input_dir:
        parser.error('input_dir or --segment-dir is required')

    include_rs = parse_yes_no(args.include_rs)
    streams, stream_dir, output_parent = discover_streams(input_dir, include_rs=include_rs)
    out_dir = args.output_dir.strip()
    if out_dir == '':
        out_dir = os.path.join(output_parent, 'aligned_offline')
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    target_fps = choose_target_fps(streams, args.target_fps)
    timeline, t0, t1 = build_common_timeline(streams, target_fps, args.time_range)

    print('input: %s' % os.path.abspath(os.path.expanduser(input_dir)))
    print('stream_dir: %s' % stream_dir)
    print('streams: %s' % ', '.join([stream['name'] for stream in streams]))
    print('target_fps: %.3f' % target_fps)
    print('time_range: %s [%.6f, %.6f], out_frames=%d' % (args.time_range, t0, t1, len(timeline)))

    summary = []
    for stream in streams:
        out_video = os.path.join(out_dir, stream['name'] + '_aligned.mp4')
        out_map = os.path.join(out_dir, stream['name'] + '_aligned_map.csv')
        stat = rebuild_one_stream(stream, out_video, out_map, timeline, target_fps, args.codec)
        summary.append(stat)
        print('rebuilt %-24s src=%d out=%d mean|dt|=%.3fms max|dt|=%.3fms' % (
            stream['name'], stat['src_frames'], stat['out_frames'],
            stat['mean_abs_dt_ms'], stat['max_abs_dt_ms'],
        ))

    summary_csv = os.path.join(out_dir, 'alignment_summary.csv')
    write_summary_csv(summary_csv, summary, target_fps, timeline, t0, t1, args.time_range)
    print('done. output: %s' % out_dir)
    print('summary: %s' % summary_csv)


if __name__ == '__main__':
    main()
