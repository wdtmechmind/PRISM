import json
import os
import re


def sanitize_task_name(task_name):
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '-', task_name.strip()).strip('-_.')
    return cleaned or 'task'


def resolve_output_root(output_arg, timestamp, task_name='task'):
    session_name = 'task_%s_%s' % (timestamp, sanitize_task_name(task_name))
    if output_arg.strip():
        base = os.path.abspath(os.path.expanduser(output_arg.strip()))
        session_dir = os.path.join(base, session_name)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    preferred = os.path.abspath(session_name)
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except OSError:
        home_fallback = os.path.join(os.path.expanduser('~'), session_name)
        os.makedirs(home_fallback, exist_ok=True)
        print('warning: no write permission in current directory, fallback to: %s' % home_fallback)
        return home_fallback


def _write_metadata(path, fields):
    with open(path, 'w', encoding='utf-8') as f:
        for key, value in fields:
            f.write('%s: %s\n' % (key, json.dumps(value, ensure_ascii=False)))


def write_task_metadata(session_dir, args, timestamp):
    path = os.path.join(session_dir, 'task_metadata.yaml')
    _write_metadata(path, [
        ('task_name', args.task_name),
        ('session_timestamp', timestamp),
        ('output_root', session_dir),
        ('config', args.config),
        ('planned_trials', args.num_trials),
        ('hand_generation', args.hand_generation),
        ('post_process', args.post_process),
        ('rpi_port', args.rpi_port),
        ('sdk_script', args.sdk_script),
        ('feedback_port', args.feedback_port),
        ('rpi_sdk_feedback_status', 'partial_cli_socket'),
        ('hand_cli_socket', {
            'ip': getattr(args, 'hand_ip', ''),
            'port': getattr(args, 'hand_port', 0),
            'timeout_s': getattr(args, 'hand_timeout_s', 0.0),
            'settle_time_s': getattr(args, 'hand_settle_time_s', 0.0),
            'auto_connect': getattr(args, 'hand_auto_connect', 'n'),
        }),
        ('hik_camera_config', {
            'exposure_us': args.hik_exposure_us,
            'gain': args.hik_gain,
            'frame_rate': args.hik_frame_rate,
        }),
        ('realsense_config', {
            'serial': args.rs_serial,
            'width': args.rs_width,
            'height': args.rs_height,
            'fps': args.rs_fps,
            'auto_exposure': args.rs_auto_exposure,
            'exposure': args.rs_exposure,
            'gain': args.rs_gain,
            'brightness': args.rs_brightness,
        }),
    ])
    return path


def write_trial_metadata(trial_dir, fields):
    path = os.path.join(trial_dir, 'metadata.yaml')
    _write_metadata(path, fields)
    return path


def touch_placeholder_hand_logs(trial_dir):
    hand_dir = os.path.join(trial_dir, 'hand')
    os.makedirs(hand_dir, exist_ok=True)
    for name in ['rpi_commands.csv', 'sdk_commands.csv', 'hand_feedback.csv']:
        open(os.path.join(hand_dir, name), 'w', encoding='utf-8').close()
