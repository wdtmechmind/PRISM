import argparse
import csv
import json
import os
import signal
import sys
import time

import cv2
import numpy as np

from prism.common import console
from prism.common.config import load_yaml_config, merge_defaults
from prism.common.timebase import pick_nearest
from prism.devices.cameras.highspeed_camera import HikCaptureThread
from prism.devices.cameras.mvs_camera import (
    MvCamera,
    UsbCameraGrabber,
    compare_and_check_readbacks,
    enumerate_usb_devices,
    parse_indices,
)
from prism.devices.cameras.realsense_camera import (
    RSCaptureThread,
    RealSenseColorGrabber,
    build_undistort_maps,
    load_rs_intrinsics,
)
from prism.devices.hand import GESTURE_ID_TO_NAME, GESTURE_ID_TO_POSE, GESTURE_TABLE, MechHandClient
from prism.online.display_server import draw_preview
from prism.online.preview_backend import create_preview_backend
from prism.online.trajectory_plotter import Live3DPlotter
from prism.reconstruction.calibration import (
    build_corrected_transform,
    get_camera_centers_world,
    load_calibration,
)
from prism.reconstruction.realtime_reconstruction import (
    COLOR_BRG,
    COLOR_ORDER,
    advance_tracking,
    build_body_model,
    build_observations_interp,
    build_observations_nearest,
    estimate_pose_from_model,
    make_track_state,
    matrix_to_rpy_zyx,
    update_body_model,
)
from prism.recording.metadata_writer import (
    resolve_output_root,
    touch_placeholder_hand_logs,
    write_task_metadata,
    write_trial_metadata,
)
from prism.recording.video_recorder import VideoSink


def parse_yes_no(raw_text):
    return raw_text.strip().lower() in ['y', 'yes', '1', 'true']


DEFAULT_ONLINE_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'configs', 'collection', 'default_online.yaml')
)

DEFAULT_CLI_VALUES = {
    'task_name': 'dexhand_task',
    'num_trials': 0,
    'output_dir': '~/prism_data/raw',
    'hand_generation': 'none',
    'post_process': 'ask',
    'rpi_port': '',
    'sdk_script': '',
    'feedback_port': '',
    'hand_ip': '127.0.0.1',
    'hand_port': 60686,
    'hand_timeout_s': 3.0,
    'hand_settle_time_s': 1.0,
    'hand_auto_connect': 'n',
    'calib_json': '',
    'hik_exposure_us': 3000.0,
    'hik_gain': 0.0,
    'hik_frame_rate': 300.0,
    'rec_brightness_alpha': 2.0,
    'rec_brightness_beta': 20.0,
    'strict_param_check': 'y',
    'param_tolerance': 1e-3,
    'writer_queue': 512,
    'rs_serial': '',
    'rs_calib_json': '',
    'rs_undistort': 'y',
    'rs_width': 1280,
    'rs_height': 720,
    'rs_fps': 30,
    'rs_auto_exposure': 'n',
    'rs_exposure': 260.0,
    'rs_gain': 64.0,
    'rs_brightness': 0.0,
    'r_h_low': 5,
    'r_s_low': 80,
    'r_v_low': 80,
    'r_h_high': 24,
    'r_s_high': 255,
    'r_v_high': 255,
    'y_h_low': 25,
    'y_s_low': 80,
    'y_v_low': 80,
    'y_h_high': 45,
    'y_s_high': 255,
    'y_v_high': 255,
    'b_h_low': 90,
    'b_s_low': 80,
    'b_v_low': 80,
    'b_h_high': 135,
    'b_s_high': 255,
    'b_v_high': 255,
    'g_h_low': 40,
    'g_s_low': 60,
    'g_v_low': 60,
    'g_h_high': 95,
    'g_s_high': 255,
    'g_v_high': 255,
    'min_area': 10.0,
    'max_norm_reproj_error': 0.015,
    'max_traj_points': 5000,
    'max_predict_frames': 6,
    'viz_3d': 'y',
    'rigid_axis_len': 0.03,
    'track_every': 1,
    'frame_buffer': 15,
    'preview_target_w': 600,
    'preview_window_width': 1920,
    'preview_window_height': 1080,
    'ui_backend': 'opencv',
    'temporal_calib': 'y',
    'temporal_calib_tag_id': 0,
    'temporal_calib_tag_delay_s': 3.0,
    'temporal_calib_tag_display_s': 1.5,
}


def build_arg_parser(defaults=None, config_path=DEFAULT_ONLINE_CONFIG):
    defaults = defaults or DEFAULT_CLI_VALUES
    parser = argparse.ArgumentParser(
        description='PRISM online collection pipeline: task -> multiple trials -> optional post-processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    pipeline = parser.add_argument_group('online task pipeline')
    pipeline.add_argument('--config', type=str, default=config_path,
                          help='YAML config file used for CLI defaults; command-line flags override it')
    pipeline.add_argument('--task-name', type=str, default=defaults['task_name'],
                          help='task name used in the raw data folder name and metadata')
    pipeline.add_argument('-n', '--num-trials', type=int, default=defaults['num_trials'],
                          help='planned number of trials; 0 means unlimited until q/ESC')
    pipeline.add_argument('--output-dir', type=str, default=defaults['output_dir'],
                          help='raw task output root folder')
    pipeline.add_argument('--hand-generation', type=str, default=defaults['hand_generation'], choices=['none', 'gen2', 'gen3'],
                          help='hand model metadata only; RPi/SDK/feedback integration is not implemented yet')
    pipeline.add_argument('--post-process', type=str, default=defaults['post_process'], choices=['ask', 'now', 'later'],
                          help='what to do after online collection finishes')
    pipeline.add_argument('--rpi-port', type=str, default=defaults['rpi_port'],
                          help='reserved for future RPi serial command input; currently left blank')
    pipeline.add_argument('--sdk-script', type=str, default=defaults['sdk_script'],
                          help='reserved for future dexterous-hand SDK command bridge; currently left blank')
    pipeline.add_argument('--feedback-port', type=str, default=defaults['feedback_port'],
                          help='reserved for future hand feedback input; currently left blank')
    pipeline.add_argument('--hand-ip', type=str, default=defaults['hand_ip'],
                          help='mech hand TCP controller IP for in-session pose control')
    pipeline.add_argument('--hand-port', type=int, default=defaults['hand_port'],
                          help='mech hand TCP controller port for in-session pose control')
    pipeline.add_argument('--hand-timeout-s', type=float, default=defaults['hand_timeout_s'],
                          help='socket timeout for mech hand connection/send')
    pipeline.add_argument('--hand-settle-time-s', type=float, default=defaults['hand_settle_time_s'],
                          help='delay after each hand command so motion can complete')
    pipeline.add_argument('--hand-auto-connect', type=str, default=defaults['hand_auto_connect'],
                          help='connect to hand at startup (y/n); otherwise connect on first hand key')

    device = parser.add_argument_group('device configuration')
    device.add_argument('--calib-json', type=str, default=defaults['calib_json'], help='path to charuco_4cam_result.json')

    hik = parser.add_argument_group('Hik high-speed cameras')
    hik.add_argument('--hik-exposure-us', type=float, default=defaults['hik_exposure_us'])
    hik.add_argument('--hik-gain', type=float, default=defaults['hik_gain'])
    hik.add_argument('--hik-frame-rate', type=float, default=defaults['hik_frame_rate'],
                     help='requested Hik acquisition fps (continuous mode); real fps shown live')
    hik.add_argument('--rec-brightness-alpha', type=float, default=defaults['rec_brightness_alpha'])
    hik.add_argument('--rec-brightness-beta', type=float, default=defaults['rec_brightness_beta'])
    hik.add_argument('--strict-param-check', type=str, default=defaults['strict_param_check'])
    hik.add_argument('--param-tolerance', type=float, default=defaults['param_tolerance'])
    hik.add_argument('--writer-queue', type=int, default=defaults['writer_queue'],
                     help='max buffered frames per stream before dropping (protects capture cadence)')

    rs = parser.add_argument_group('RealSense camera')
    rs.add_argument('--rs-serial', type=str, default=defaults['rs_serial'])
    rs.add_argument('--rs-calib-json', type=str, default=defaults['rs_calib_json'])
    rs.add_argument('--rs-undistort', type=str, default=defaults['rs_undistort'])
    rs.add_argument('--rs-width', type=int, default=defaults['rs_width'])
    rs.add_argument('--rs-height', type=int, default=defaults['rs_height'])
    rs.add_argument('--rs-fps', type=int, default=defaults['rs_fps'])
    rs.add_argument('--rs-auto-exposure', type=str, default=defaults['rs_auto_exposure'])
    rs.add_argument('--rs-exposure', type=float, default=defaults['rs_exposure'])
    rs.add_argument('--rs-gain', type=float, default=defaults['rs_gain'])
    rs.add_argument('--rs-brightness', type=float, default=defaults['rs_brightness'])

    tracking = parser.add_argument_group('online trajectory tracking')
    tracking.add_argument('--r-h-low', type=int, default=defaults['r_h_low'])
    tracking.add_argument('--r-s-low', type=int, default=defaults['r_s_low'])
    tracking.add_argument('--r-v-low', type=int, default=defaults['r_v_low'])
    tracking.add_argument('--r-h-high', type=int, default=defaults['r_h_high'])
    tracking.add_argument('--r-s-high', type=int, default=defaults['r_s_high'])
    tracking.add_argument('--r-v-high', type=int, default=defaults['r_v_high'])

    tracking.add_argument('--y-h-low', type=int, default=defaults['y_h_low'])
    tracking.add_argument('--y-s-low', type=int, default=defaults['y_s_low'])
    tracking.add_argument('--y-v-low', type=int, default=defaults['y_v_low'])
    tracking.add_argument('--y-h-high', type=int, default=defaults['y_h_high'])
    tracking.add_argument('--y-s-high', type=int, default=defaults['y_s_high'])
    tracking.add_argument('--y-v-high', type=int, default=defaults['y_v_high'])

    tracking.add_argument('--b-h-low', type=int, default=defaults['b_h_low'])
    tracking.add_argument('--b-s-low', type=int, default=defaults['b_s_low'])
    tracking.add_argument('--b-v-low', type=int, default=defaults['b_v_low'])
    tracking.add_argument('--b-h-high', type=int, default=defaults['b_h_high'])
    tracking.add_argument('--b-s-high', type=int, default=defaults['b_s_high'])
    tracking.add_argument('--b-v-high', type=int, default=defaults['b_v_high'])

    tracking.add_argument('--g-h-low', type=int, default=defaults['g_h_low'])
    tracking.add_argument('--g-s-low', type=int, default=defaults['g_s_low'])
    tracking.add_argument('--g-v-low', type=int, default=defaults['g_v_low'])
    tracking.add_argument('--g-h-high', type=int, default=defaults['g_h_high'])
    tracking.add_argument('--g-s-high', type=int, default=defaults['g_s_high'])
    tracking.add_argument('--g-v-high', type=int, default=defaults['g_v_high'])

    tracking.add_argument('--min-area', type=float, default=defaults['min_area'])
    tracking.add_argument('--max-norm-reproj-error', type=float, default=defaults['max_norm_reproj_error'])
    tracking.add_argument('--max-traj-points', type=int, default=defaults['max_traj_points'])
    tracking.add_argument('--max-predict-frames', type=int, default=defaults['max_predict_frames'])
    tracking.add_argument('--viz-3d', type=str, default=defaults['viz_3d'])
    tracking.add_argument('--rigid-axis-len', type=float, default=defaults['rigid_axis_len'],
                          help='axis length in meters for rigid-body frame visualization')
    tracking.add_argument('--track-every', type=int, default=defaults['track_every'],
                          help='run LED tracking every N preview iterations (raise to lighten CPU)')
    tracking.add_argument('--frame-buffer', type=int, default=defaults['frame_buffer'],
                          help='per-camera timestamped frame buffer length for time-based association')
    tracking.add_argument('--preview-target-w', type=int, default=defaults['preview_target_w'],
                          help='render width per camera cell in unified preview (larger is clearer but heavier)')
    tracking.add_argument('--preview-window-width', type=int, default=defaults['preview_window_width'],
                          help='initial unified preview window width in pixels')
    tracking.add_argument('--preview-window-height', type=int, default=defaults['preview_window_height'],
                          help='initial unified preview window height in pixels')
    tracking.add_argument('--ui-backend', type=str, default=defaults['ui_backend'], choices=['opencv', 'qt'],
                          help='preview UI backend; qt keeps keyboard/CLI control and avoids OpenCV window path')

    tcalib = parser.add_argument_group('temporal delay calibration (runs once after cameras open)')
    tcalib.add_argument('--temporal-calib', type=str, default=defaults['temporal_calib'],
                        help='run AprilTag temporal calibration at session start (y/n, default: y)')
    tcalib.add_argument('--skip-temporal-calib', action='store_true',
                        help='shorthand for --temporal-calib n')
    tcalib.add_argument('--temporal-calib-tag-id', type=int,
                        default=defaults['temporal_calib_tag_id'],
                        help='AprilTag ID to flash; -1 = any (default: 0)')
    tcalib.add_argument('--temporal-calib-tag-delay-s', type=float,
                        default=defaults['temporal_calib_tag_delay_s'],
                        help='seconds between cameras-ready and tag appearing on screen (default: 3)')
    tcalib.add_argument('--temporal-calib-tag-display-s', type=float,
                        default=defaults['temporal_calib_tag_display_s'],
                        help='seconds to keep tag visible on screen (default: 1.5)')
    return parser


class SessionManager(object):
    def __init__(self, args):
        self.args = args
        self.strict_param_check = parse_yes_no(args.strict_param_check)
        self.rs_auto_exposure = parse_yes_no(args.rs_auto_exposure)
        self.rs_use_undistort = parse_yes_no(args.rs_undistort)
        self.use_viz_3d = parse_yes_no(args.viz_3d)
        self.hand_auto_connect = parse_yes_no(args.hand_auto_connect)
        self.planned_trials = max(0, int(args.num_trials))

        self.hsv_cfg = {
            'red': ((args.r_h_low, args.r_s_low, args.r_v_low), (args.r_h_high, args.r_s_high, args.r_v_high)),
            'yellow': ((args.y_h_low, args.y_s_low, args.y_v_low), (args.y_h_high, args.y_s_high, args.y_v_high)),
            'blue': ((args.b_h_low, args.b_s_low, args.b_v_low), (args.b_h_high, args.b_s_high, args.b_v_high)),
            'green': ((args.g_h_low, args.g_s_low, args.g_v_low), (args.g_h_high, args.g_s_high, args.g_v_high)),
        }

        self.cameras = None
        self.camera_centers = None
        self.corrected_transform = None
        self.output_root = None
        self.task_timestamp = None
        self.traj_near_csv_path = None
        self.traj_interp_csv_path = None
        self.pose_csv_path = None
        self.align_csv_path = None
        self.rs_undistort_maps = None

        self.recorders = []
        self.hik_threads = []
        self.hik_serials = []
        self.rs_thread = None
        self.rs_grabber = None
        self.plotter3d = None

        self.track_near = make_track_state()
        self.track_interp = make_track_state()
        self.rigid_model = None
        self.pose_history = []
        self.pose_rot_history = []
        self.pose_valid_prev = False

        self.paused = False
        self.task_complete = False
        self.last_time = None
        self.t0 = time.time()

        self.recording = False
        self.trial_id = 0
        self.trial_start_wall = None
        self.current_trial_dir = None
        self.active_sinks = []
        self.ts_csv_file = None
        self.ts_csv_writer = None
        self.frame_idx = 0

        self.hand_client = None
        self.hand_connected = False
        self.hand_cmd_file = None
        self.hand_cmd_writer = None
        self.hand_sdk_cmd_file = None
        self.hand_sdk_cmd_writer = None
        self.hand_task_cmd_path = None
        self.hand_task_cmd_file = None
        self.hand_task_cmd_writer = None
        self.hand_last_action = 'none'
        self.hand_last_command = ''
        self.hand_last_t_sec = float('nan')
        self.hand_total_commands = 0
        self.hand_digit_buffer = ''
        self.hand_digit_last_wall = 0.0
        self.hand_digit_timeout_s = 0.65
        self.hand_ui_regions = []
        self.hand_click_gesture = None
        self.preview_backend = None
        self.window_name = 'DexHand HighFps Capture'

    def prepare_session(self):
        args = self.args
        self.cameras = load_calibration(os.path.expanduser(args.calib_json))
        self.camera_centers = get_camera_centers_world(self.cameras)
        self.corrected_transform = build_corrected_transform(self.cameras, self.camera_centers)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.task_timestamp = timestamp
        self.output_root = resolve_output_root(os.path.expanduser(args.output_dir), timestamp, args.task_name)
        self.traj_near_csv_path = os.path.join(self.output_root, 'trajectory_led_nearest.csv')
        self.traj_interp_csv_path = os.path.join(self.output_root, 'trajectory_led_interp.csv')
        self.pose_csv_path = os.path.join(self.output_root, 'rigid_pose_6d.csv')
        self.align_csv_path = os.path.join(self.output_root, 'time_alignment_log.csv')
        self.hand_task_cmd_path = os.path.join(self.output_root, 'hand_sdk_commands_timeline.csv')
        self.hand_task_cmd_file = open(self.hand_task_cmd_path, 'w', newline='', encoding='utf-8')
        self.hand_task_cmd_writer = csv.writer(self.hand_task_cmd_file)
        self.hand_task_cmd_writer.writerow([
            't_sec', 'wall_time', 'trial_id', 'trial_time', 'action', 'command', 'status', 'message', 'recording'
        ])
        task_metadata_path = write_task_metadata(self.output_root, args, timestamp)
        console.rule('PRISM Online Collection')
        console.saved('output folder: %s' % self.output_root)
        console.saved('task metadata: %s' % task_metadata_path)
        console.info('task: %s | planned trials: %s | hand: %s'
                     % (args.task_name, self.planned_trials if self.planned_trials > 0 else 'unlimited', args.hand_generation))
        if args.rpi_port or args.sdk_script or args.feedback_port:
            console.warning('RPi command bridge, SDK forwarding, and hand feedback logging are placeholders for now.')
        console.info('hand pose hotkeys enabled via socket target %s:%d'
                     % (args.hand_ip, int(args.hand_port)))

        if args.rs_calib_json.strip():
            rs_K, rs_D, _rs_calib_wh, rs_data = load_rs_intrinsics(os.path.expanduser(args.rs_calib_json))
            rs_json_out = os.path.join(self.output_root, 'realsense_intrinsics.json')
            with open(rs_json_out, 'w', encoding='utf-8') as f:
                json.dump(rs_data, f, ensure_ascii=False, indent=2)
            console.saved('realsense intrinsics saved to session: %s' % rs_json_out)
            if self.rs_use_undistort:
                map1, map2, _ = build_undistort_maps(rs_K, rs_D, args.rs_width, args.rs_height)
                self.rs_undistort_maps = (map1, map2)
                console.success('realsense color undistortion enabled.')
        elif self.rs_use_undistort:
            console.warning('--rs-undistort requested but --rs-calib-json not provided; undistortion disabled.')

    def open_devices(self):
        args = self.args
        MvCamera.MV_CC_Initialize()
        print('SDKVersion[0x%x]' % MvCamera.MV_CC_GetSDKVersion())

        usb_devices = enumerate_usb_devices()
        if len(usb_devices) < 4:
            raise RuntimeError('found %d USB cameras, need 4 Hik cameras' % len(usb_devices))

        print('Found %d USB cameras:' % len(usb_devices))
        for i, _, model, serial in usb_devices:
            print('  [%d] model=%s, serial=%s' % (i, model, serial))

        selected_text = input('please input 4 Hik camera indices (blank means first 4 cameras): ').strip()
        if selected_text == '':
            selected = list(range(4))
        else:
            selected = parse_indices(selected_text, len(usb_devices), 4)

        selected_infos = [usb_devices[idx] for idx in selected]
        readbacks = []
        for cam_i, (_, dev_info, model, serial) in enumerate(selected_infos):
            serial_safe = serial if serial else ('cam%d' % cam_i)
            self.hik_serials.append(serial_safe)
            print('selected hik%d: model=%s serial=%s' % (cam_i, model, serial_safe))

            rec = UsbCameraGrabber(dev_info, serial_safe, model)
            rb = rec.open_and_prepare(
                use_hardware_trigger=False,
                use_software_trigger=False,
                exposure_us=args.hik_exposure_us,
                gain=args.hik_gain,
                frame_rate=args.hik_frame_rate,
            )
            self.recorders.append(rec)
            readbacks.append(rb)

        compare_and_check_readbacks(readbacks, args.hik_exposure_us, 'exposure_us', args.param_tolerance, self.strict_param_check)
        compare_and_check_readbacks(readbacks, args.hik_gain, 'gain', args.param_tolerance, self.strict_param_check)
        compare_and_check_readbacks(readbacks, args.hik_frame_rate, 'frame_rate', args.param_tolerance, self.strict_param_check)

        self.rs_grabber = RealSenseColorGrabber(
            serial=args.rs_serial if args.rs_serial.strip() else None,
            width=args.rs_width,
            height=args.rs_height,
            fps=args.rs_fps,
            auto_exposure=self.rs_auto_exposure,
            exposure=args.rs_exposure,
            gain=args.rs_gain,
            brightness=args.rs_brightness,
        )
        self.rs_grabber.open_and_prepare()

        for cam_i, rec in enumerate(self.recorders):
            th = HikCaptureThread(rec, cam_i, self.hik_serials[cam_i], timeout_ms=200, buffer_len=args.frame_buffer)
            th.start()
            self.hik_threads.append(th)

        self.rs_thread = RSCaptureThread(self.rs_grabber, self.rs_undistort_maps, args.rs_width, args.rs_height,
                                         timeout_ms=1000, buffer_len=max(5, args.frame_buffer))
        self.rs_thread.start()

        self.hand_client = MechHandClient(
            ip=args.hand_ip,
            port=args.hand_port,
            timeout_s=args.hand_timeout_s,
            settle_time_s=args.hand_settle_time_s,
        )
        if self.hand_auto_connect:
            self._connect_hand()

    def _connect_hand(self):
        if self.hand_client is None:
            return False
        if self.hand_connected:
            return True
        try:
            self.hand_client.connect()
            self.hand_connected = True
            console.success('hand connected: %s:%d' % (self.args.hand_ip, int(self.args.hand_port)))
            return True
        except Exception as exc:
            console.warning('hand connect failed (%s:%d): %s'
                            % (self.args.hand_ip, int(self.args.hand_port), str(exc)))
            self.hand_connected = False
            return False

    def _log_hand_command(self, wall_time_sec, trial_time_sec, action, command, status, message):
        t_sec = wall_time_sec - self.t0

        if self.hand_cmd_writer is not None:
            self.hand_cmd_writer.writerow([
                '%.6f' % t_sec,
                '%.6f' % wall_time_sec,
                '%.6f' % trial_time_sec,
                action,
                command,
                status,
                message,
            ])
        if self.hand_sdk_cmd_writer is not None:
            self.hand_sdk_cmd_writer.writerow([
                '%.6f' % t_sec,
                '%.6f' % wall_time_sec,
                '%.6f' % trial_time_sec,
                action,
                command,
                status,
                message,
            ])
        if self.hand_task_cmd_writer is not None:
            self.hand_task_cmd_writer.writerow([
                '%.6f' % t_sec,
                '%.6f' % wall_time_sec,
                self.trial_id if self.recording else 0,
                '%.6f' % trial_time_sec,
                action,
                command,
                status,
                message,
                int(bool(self.recording)),
            ])

        self.hand_last_action = action
        self.hand_last_command = command
        self.hand_last_t_sec = t_sec
        self.hand_total_commands += 1

        if self.hand_cmd_file is not None:
            self.hand_cmd_file.flush()
        if self.hand_sdk_cmd_file is not None:
            self.hand_sdk_cmd_file.flush()
        if self.hand_task_cmd_file is not None:
            self.hand_task_cmd_file.flush()

    def _send_hand_gesture(self, gesture_id):
        pose_name = GESTURE_ID_TO_POSE.get(int(gesture_id), 'unknown')
        gesture_name = GESTURE_ID_TO_NAME.get(int(gesture_id), 'unknown')
        action_text = '%02d:%s' % (int(gesture_id), pose_name)
        cmd_preview = ''
        try:
            if not self._connect_hand():
                now = time.time()
                trial_elapsed = (now - self.trial_start_wall) if (self.recording and self.trial_start_wall is not None) else 0.0
                self._log_hand_command(now, trial_elapsed, action_text, cmd_preview, 'connect_failed', 'connect_failed')
                return
            sent = self.hand_client.send_gesture(int(gesture_id))
            cmd_preview = sent
            now = time.time()
            trial_elapsed = (now - self.trial_start_wall) if (self.recording and self.trial_start_wall is not None) else 0.0
            self._log_hand_command(now, trial_elapsed, action_text, sent, 'ok', gesture_name)
            console.step('hand pose sent: [%d] %s -> %s' % (int(gesture_id), gesture_name, sent))
        except Exception as exc:
            self.hand_connected = False
            now = time.time()
            trial_elapsed = (now - self.trial_start_wall) if (self.recording and self.trial_start_wall is not None) else 0.0
            self._log_hand_command(now, trial_elapsed, action_text, cmd_preview, 'error', str(exc))
            console.warning('hand pose send failed: [%d] %s (%s)' % (int(gesture_id), gesture_name, str(exc)))

    def _submit_hand_digit_buffer(self):
        if not self.hand_digit_buffer:
            return
        try:
            gesture_id = int(self.hand_digit_buffer)
        except ValueError:
            self.hand_digit_buffer = ''
            return
        self.hand_digit_buffer = ''
        if 1 <= gesture_id <= 17:
            self._send_hand_gesture(gesture_id)
        else:
            console.warning('invalid hand gesture id: %d (valid range: 1-17)' % gesture_id)

    def _on_digit_key(self, digit_char, now_wall):
        if digit_char < '0' or digit_char > '9':
            return

        self.hand_digit_last_wall = now_wall

        # 2..9 are unambiguous single gestures; 1 may be 1 or 10..17.
        if not self.hand_digit_buffer:
            if digit_char == '1':
                self.hand_digit_buffer = '1'
                self.hand_digit_last_wall = now_wall
                return
            if digit_char == '0':
                return
            self.hand_digit_buffer = digit_char
            self._submit_hand_digit_buffer()
            return

        # Only a leading '1' should be buffered for two-digit gesture ids.
        if self.hand_digit_buffer == '1':
            if '0' <= digit_char <= '7':
                self.hand_digit_buffer = '1' + digit_char
                self._submit_hand_digit_buffer()
                return

            # 18/19 are invalid; treat as separate key presses.
            self._submit_hand_digit_buffer()
            if digit_char != '0':
                self.hand_digit_buffer = digit_char
                self._submit_hand_digit_buffer()
            return

        # Fallback safety path.
        self._submit_hand_digit_buffer()
        if digit_char != '0':
            self.hand_digit_buffer = digit_char
            self._submit_hand_digit_buffer()

    def _on_mouse_event(self, event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for region in self.hand_ui_regions:
            x1 = int(region.get('x1', -1))
            y1 = int(region.get('y1', -1))
            x2 = int(region.get('x2', -1))
            y2 = int(region.get('y2', -1))
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.hand_click_gesture = int(region.get('gesture_id', 0))
                return

    def _estimate_preview_target_w(self, hik_latest, rs_latest):
        requested = max(320, int(self.args.preview_target_w))

        ratios = []
        for frame in hik_latest:
            if frame is not None:
                h, w = frame.shape[:2]
                if w > 0:
                    ratios.append(float(h) / float(w))
        if rs_latest is not None:
            rh, rw = rs_latest.shape[:2]
            if rw > 0:
                ratios.append(float(rh) / float(rw))
        if not ratios:
            ratios = [720.0 / 1280.0]

        max_ratio = max(ratios)
        if self.preview_backend is not None:
            window_w, window_h = self.preview_backend.get_window_size()
        else:
            window_w = max(960, int(self.args.preview_window_width))
            window_h = max(640, int(self.args.preview_window_height))

        # Grid has 2 columns and 3 rows based on max cell height.
        max_by_h = int((window_h - 24) / max(1e-6, 3.0 * max_ratio))

        # Keep room for right-side control panel so total width does not exceed window width.
        min_panel_w = 480
        max_by_w = int((window_w - min_panel_w) / 2.0)

        fitted = min(requested, max_by_h, max_by_w)
        return max(320, fitted)

    def start_recording(self):
        args = self.args
        if self.recording:
            return
        if self.planned_trials > 0 and self.trial_id >= self.planned_trials:
            console.warning('planned trial count reached (%d). press q/ESC to finish task.' % self.planned_trials)
            return

        self.trial_id += 1
        trial_dir = os.path.join(self.output_root, 'trial_%06d' % self.trial_id)
        cameras_dir = os.path.join(trial_dir, 'cameras')
        logs_dir = os.path.join(trial_dir, 'logs')
        os.makedirs(cameras_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, 'trajectory'), exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        touch_placeholder_hand_logs(trial_dir)
        self.current_trial_dir = trial_dir

        hand_log_path = os.path.join(trial_dir, 'hand', 'rpi_commands.csv')
        self.hand_cmd_file = open(hand_log_path, 'w', newline='', encoding='utf-8')
        self.hand_cmd_writer = csv.writer(self.hand_cmd_file)
        self.hand_cmd_writer.writerow(['t_sec', 'wall_time', 'trial_time', 'action', 'command', 'status', 'message'])

        sdk_log_path = os.path.join(trial_dir, 'hand', 'sdk_commands.csv')
        self.hand_sdk_cmd_file = open(sdk_log_path, 'w', newline='', encoding='utf-8')
        self.hand_sdk_cmd_writer = csv.writer(self.hand_sdk_cmd_file)
        self.hand_sdk_cmd_writer.writerow(['t_sec', 'wall_time', 'trial_time', 'action', 'command', 'status', 'message'])

        self.active_sinks = []
        for cam_i in range(4):
            fps = self.hik_threads[cam_i].fps_meter.fps()
            if fps < 1.0:
                fps = float(args.hik_frame_rate)
            path = os.path.join(cameras_dir, 'hik%d_%s.mp4' % (cam_i, self.hik_serials[cam_i]))
            sink = VideoSink(path, fps, max_queue=args.writer_queue,
                             brightness_alpha=args.rec_brightness_alpha,
                             brightness_beta=args.rec_brightness_beta)
            self.hik_threads[cam_i].set_sink(sink)
            self.active_sinks.append(sink)

        rs_fps = self.rs_thread.fps_meter.fps()
        if rs_fps < 1.0:
            rs_fps = float(args.rs_fps)
        rs_path = os.path.join(cameras_dir, 'realsense_color.mp4')
        rs_sink = VideoSink(rs_path, rs_fps, max_queue=args.writer_queue)
        self.rs_thread.set_sink(rs_sink)
        self.active_sinks.append(rs_sink)

        self.ts_csv_file = open(os.path.join(logs_dir, 'fps_log.csv'), 'w', newline='', encoding='utf-8')
        self.ts_csv_writer = csv.writer(self.ts_csv_file)
        self.ts_csv_writer.writerow(['wall_time', 'trial_time', 'hik0_fps', 'hik1_fps', 'hik2_fps', 'hik3_fps', 'rs_fps'])

        self.trial_start_wall = time.time()
        write_trial_metadata(trial_dir, [
            ('task_name', args.task_name),
            ('trial_id', self.trial_id),
            ('status', 'recording'),
            ('start_wall_time', self.trial_start_wall),
            ('hand_generation', args.hand_generation),
            ('rpi_sdk_feedback_status', 'partial_cli_socket'),
            ('rpi_commands_log', hand_log_path),
            ('sdk_commands_log', os.path.join('hand', 'sdk_commands.csv')),
            ('hand_feedback_log', ''),
        ])

        self.recording = True
        console.step('trial started: trial_%06d (hik_fps~%s, rs_fps~%.1f)'
                 % (self.trial_id, ['%.1f' % s.fps for s in self.active_sinks[:4]], rs_fps))

    def stop_recording(self):
        if not self.recording:
            return
        for th in self.hik_threads:
            th.set_sink(None)
        self.rs_thread.set_sink(None)

        total_written = 0
        total_dropped = 0
        for sink in self.active_sinks:
            sink.close()
            total_written += sink.written
            total_dropped += sink.dropped
        self.active_sinks = []

        if self.ts_csv_file is not None:
            self.ts_csv_file.flush()
            self.ts_csv_file.close()
            self.ts_csv_file = None
            self.ts_csv_writer = None

        if self.hand_cmd_file is not None:
            self.hand_cmd_file.flush()
            self.hand_cmd_file.close()
            self.hand_cmd_file = None
            self.hand_cmd_writer = None

        if self.hand_sdk_cmd_file is not None:
            self.hand_sdk_cmd_file.flush()
            self.hand_sdk_cmd_file.close()
            self.hand_sdk_cmd_file = None
            self.hand_sdk_cmd_writer = None

        self.recording = False
        stop_wall = time.time()
        if self.current_trial_dir is not None:
            write_trial_metadata(self.current_trial_dir, [
                ('task_name', self.args.task_name),
                ('trial_id', self.trial_id),
                ('status', 'stopped'),
                ('start_wall_time', self.trial_start_wall),
                ('end_wall_time', stop_wall),
                ('duration_sec', stop_wall - self.trial_start_wall if self.trial_start_wall is not None else 0.0),
                ('hand_generation', self.args.hand_generation),
                ('rpi_sdk_feedback_status', 'partial_cli_socket'),
                ('rpi_commands_log', os.path.join('hand', 'rpi_commands.csv')),
                ('sdk_commands_log', os.path.join('hand', 'sdk_commands.csv')),
                ('hand_feedback_log', ''),
                ('frames_written', total_written),
                ('frames_dropped', total_dropped),
            ])
        self.trial_start_wall = None
        self.current_trial_dir = None
        console.success('trial stopped: trial_%06d (written=%d, dropped=%d)'
                        % (self.trial_id, total_written, total_dropped))
        if total_dropped > 0:
            console.warning('%d frames dropped (disk/encoding too slow). Lower fps/resolution or raise --writer-queue.'
                            % total_dropped)
        if self.planned_trials > 0 and self.trial_id >= self.planned_trials:
            self.task_complete = True
            console.done('planned trial count reached (%d). finishing online collection.' % self.planned_trials)

    def run_loop(self):
        mouse_cb = self._on_mouse_event if self.args.ui_backend == 'opencv' else None
        self.preview_backend = create_preview_backend(
            self.args.ui_backend,
            self.window_name,
            max(640, int(self.args.preview_window_width)),
            max(480, int(self.args.preview_window_height)),
            mouse_callback=mouse_cb,
        )
        self.preview_backend.open()
        self.plotter3d = Live3DPlotter(
            enabled=self.use_viz_3d,
            camera_centers=self.camera_centers,
            corrected_transform=self.corrected_transform,
        )

        console.info('preview started. click preview window to focus keys.')
        console.info('keys: SPACE start/stop current trial, p pause tracking, r resume tracking, '
                     'click hand gesture buttons or type hand gesture id 1-17 '
                     '(for 10-17 type two digits quickly), '
                     'Enter submit pending 1, Backspace clear pending input, q/ESC finish task')
        if self.args.ui_backend == 'qt':
            console.info('ui backend: qt (keyboard/CLI gesture control only; click-to-send disabled).')

        with open(self.traj_near_csv_path, 'w', newline='', encoding='utf-8') as near_file, \
                open(self.traj_interp_csv_path, 'w', newline='', encoding='utf-8') as interp_file, \
                open(self.pose_csv_path, 'w', newline='', encoding='utf-8') as pose_file, \
                open(self.align_csv_path, 'w', newline='', encoding='utf-8') as align_file:
            near_writer = csv.writer(near_file)
            interp_writer = csv.writer(interp_file)
            pose_writer = csv.writer(pose_file)
            align_writer = csv.writer(align_file)
            header = ['t_sec', 'color', 'x_m', 'y_m', 'z_m', 'mode', 'num_views', 'max_norm_reproj_err', 'visible_cams']
            near_writer.writerow(header)
            interp_writer.writerow(header)
            pose_writer.writerow(['t_sec', 'mode', 'num_leds_used', 'modeled_leds', 'visible_leds',
                                  'x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg'])
            align_writer.writerow([
                't_ref_sec', 'hik0_ts', 'hik1_ts', 'hik2_ts', 'hik3_ts',
                'hik_spread_ms', 'rs_ts', 'rs_offset_ms',
            ])

            while True:
                if not self._run_once(near_file, interp_file, pose_file, align_file,
                                      near_writer, interp_writer, pose_writer, align_writer):
                    break

    def _run_once(self, near_file, interp_file, pose_file, align_file,
                  near_writer, interp_writer, pose_writer, align_writer):
        args = self.args
        now = time.time()
        dt = 1.0 / 30.0 if self.last_time is None else max(1e-3, now - self.last_time)
        self.last_time = now
        self.frame_idx += 1

        hik_latest = [th.get_latest() for th in self.hik_threads]
        rs_latest = self.rs_thread.get_latest()
        hik_fps = [th.fps_meter.fps() for th in self.hik_threads]
        rs_fps_now = self.rs_thread.fps_meter.fps()

        do_track = (not self.paused) and (self.frame_idx % max(1, int(args.track_every)) == 0)

        obs_near = {name: {} for name in COLOR_ORDER}
        obs_interp = {name: {} for name in COLOR_ORDER}
        point_near = {name: None for name in COLOR_ORDER}
        mode_near = {name: 'none' for name in COLOR_ORDER}
        point_interp = {name: None for name in COLOR_ORDER}
        mode_interp = {name: 'none' for name in COLOR_ORDER}

        ref_time = now
        hik_spread_ms = 0.0
        rs_offset_ms = float('nan')
        rs_aligned = rs_latest

        if do_track:
            buffers = [th.get_buffer() for th in self.hik_threads]
            latest_ts = [b[-1][0] for b in buffers if b]
            if latest_ts:
                t_ref = min(latest_ts)
                ref_time = t_ref
                det_cache = {}
                obs_near = build_observations_nearest(buffers, t_ref, self.hsv_cfg, args.min_area, det_cache)
                obs_interp = build_observations_interp(buffers, t_ref, self.hsv_cfg, args.min_area, det_cache)

                hik_sel_ts = []
                for b in buffers:
                    sel = pick_nearest(b, t_ref) if b else None
                    hik_sel_ts.append(sel[0] if sel is not None else float('nan'))
                valid_ts = [x for x in hik_sel_ts if x == x]
                if len(valid_ts) >= 2:
                    hik_spread_ms = (max(valid_ts) - min(valid_ts)) * 1000.0

                rs_buf = self.rs_thread.get_buffer()
                rs_sel = pick_nearest(rs_buf, t_ref) if rs_buf else None
                if rs_sel is not None:
                    rs_offset_ms = (rs_sel[0] - t_ref) * 1000.0
                    rs_aligned = rs_sel[1]

                align_writer.writerow(
                    ['%.6f' % (t_ref - self.t0)] +
                    ['%.6f' % v for v in hik_sel_ts] +
                    ['%.3f' % hik_spread_ms,
                     ('%.6f' % rs_sel[0]) if rs_sel is not None else '',
                     ('%.3f' % rs_offset_ms) if rs_offset_ms == rs_offset_ms else '']
                )

        if self.paused:
            point_near, mode_near = advance_tracking(
                self.track_near, obs_near, self.cameras, args, dt, ref_time, self.t0, near_writer, True)
            point_interp, mode_interp = advance_tracking(
                self.track_interp, obs_interp, self.cameras, args, dt, ref_time, self.t0, interp_writer, True)
        elif do_track:
            point_near, mode_near = advance_tracking(
                self.track_near, obs_near, self.cameras, args, dt, ref_time, self.t0, near_writer, False)
            point_interp, mode_interp = advance_tracking(
                self.track_interp, obs_interp, self.cameras, args, dt, ref_time, self.t0, interp_writer, False)
            near_file.flush()
            interp_file.flush()
            align_file.flush()

        pose_mode = 'none'
        pose_xyz = None
        pose_rpy_deg = None
        pose_rot = None
        pose_now = None
        pose_used_names = []
        modeled_names = []
        visible_names = []

        valid_points = {name: point_near[name] for name in COLOR_ORDER if point_near[name] is not None}
        visible_names = [name for name in COLOR_ORDER if name in valid_points]
        if len(valid_points) >= 3:
            if self.rigid_model is None:
                self.rigid_model = build_body_model(valid_points)
                if self.rigid_model is not None:
                    console.success('rigid model initialized from LEDs: %s.'
                                    % ','.join(self.rigid_model['model_points'].keys()))

            if self.rigid_model is not None:
                modeled_names = [name for name in COLOR_ORDER if name in self.rigid_model['model_points']]
                est = estimate_pose_from_model(self.rigid_model['model_points'], valid_points)
                if est is not None:
                    pose_rot, pose_xyz = est
                    pose_used_names = [name for name in modeled_names if name in valid_points]
                    added_model_names = update_body_model(self.rigid_model['model_points'], valid_points, pose_rot, pose_xyz)
                    if added_model_names:
                        console.success('added LEDs to rigid model: %s.' % ','.join(added_model_names))
                        modeled_names = [name for name in COLOR_ORDER if name in self.rigid_model['model_points']]
                    roll, pitch, yaw = matrix_to_rpy_zyx(pose_rot)
                    pose_rpy_deg = np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))
                    pose_now = np.array([
                        pose_xyz[0], pose_xyz[1], pose_xyz[2],
                        pose_rpy_deg[0], pose_rpy_deg[1], pose_rpy_deg[2],
                    ], dtype=np.float64)
                    pose_mode = 'measured'

        if pose_now is not None:
            self.pose_history.append(pose_now.tolist())
            self.pose_rot_history.append(pose_rot.reshape(9).tolist())
            self.pose_valid_prev = True
            if len(self.pose_history) > args.max_traj_points:
                self.pose_history = self.pose_history[-args.max_traj_points:]
            if len(self.pose_rot_history) > args.max_traj_points:
                self.pose_rot_history = self.pose_rot_history[-args.max_traj_points:]
            pose_writer.writerow([
                '%.6f' % (ref_time - self.t0), pose_mode, len(pose_used_names),
                ','.join(modeled_names), ','.join(visible_names),
                '%.9f' % pose_now[0], '%.9f' % pose_now[1], '%.9f' % pose_now[2],
                '%.6f' % pose_now[3], '%.6f' % pose_now[4], '%.6f' % pose_now[5],
            ])
            pose_file.flush()
        elif self.pose_valid_prev and pose_mode in ('none', 'paused'):
            self.pose_history.append([float('nan')] * 6)
            self.pose_rot_history.append([float('nan')] * 9)
            if len(self.pose_history) > args.max_traj_points:
                self.pose_history = self.pose_history[-args.max_traj_points:]
            if len(self.pose_rot_history) > args.max_traj_points:
                self.pose_rot_history = self.pose_rot_history[-args.max_traj_points:]
            self.pose_valid_prev = False

        trial_elapsed = (now - self.trial_start_wall) if (self.recording and self.trial_start_wall is not None) else 0.0
        if self.recording and self.ts_csv_writer is not None:
            self.ts_csv_writer.writerow(['%.6f' % now, '%.6f' % trial_elapsed] +
                                        ['%.3f' % f for f in hik_fps] + ['%.3f' % rs_fps_now])

        self.plotter3d.update(
            self.track_near['traj_points'], point_near,
            mode_text='NEAREST | ' + ' | '.join(['%s:%s' % (n, mode_near[n]) for n in COLOR_ORDER] + ['pose:%s' % pose_mode]),
            pose_t=pose_xyz,
            pose_R=pose_rot,
            pose_history=self.pose_history,
            pose_rot_history=self.pose_rot_history,
            rigid_axis_len=args.rigid_axis_len,
        )

        hand_info = {
            'connected': self.hand_connected,
            'last_action': self.hand_last_action,
            'last_command': self.hand_last_command,
            'last_t_sec': self.hand_last_t_sec,
            'total_commands': self.hand_total_commands,
            'pending_digits': self.hand_digit_buffer,
            'gesture_hints': [(gid, name) for gid, _pose, name in GESTURE_TABLE],
        }

        effective_target_w = self._estimate_preview_target_w(hik_latest, rs_aligned)

        window_w, _window_h = self.preview_backend.get_window_size() if self.preview_backend is not None else (
            max(640, int(args.preview_window_width)),
            max(480, int(args.preview_window_height)),
        )

        draw_out = draw_preview(
            hik_latest,
            rs_aligned,
            obs_near,
            hik_fps,
            rs_fps_now,
            self.recording,
            self.trial_id,
            trial_elapsed,
            target_w=effective_target_w,
            traj_image=self.plotter3d.get_latest_frame(),
            hand_info=hand_info,
            traj_error=self.plotter3d.get_latest_error(),
            return_ui_meta=(self.args.ui_backend == 'opencv'),
            composite_target_w=max(640, int(window_w)),
        )
        if self.args.ui_backend == 'opencv':
            grid, ui_meta = draw_out
            self.hand_ui_regions = ui_meta.get('gesture_regions', [])
        else:
            grid = draw_out
            self.hand_ui_regions = []

        cv2.putText(grid, 'hik sync spread=%.1f ms | rs offset=%s ms' % (
            hik_spread_ms, ('%.1f' % rs_offset_ms) if rs_offset_ms == rs_offset_ms else 'n/a'),
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2, cv2.LINE_AA)

        y0 = 110
        for name in COLOR_ORDER:
            txt = '%s: near=%s interp=%s views=%d' % (
                name, mode_near[name], mode_interp[name], len(obs_near[name]))
            cv2.putText(grid, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_BRG[name], 2, cv2.LINE_AA)
            y0 += 30

        if pose_xyz is not None and pose_rpy_deg is not None:
            cv2.putText(grid, 'rigid6d xyz=(%.3f, %.3f, %.3f)m' % (pose_xyz[0], pose_xyz[1], pose_xyz[2]),
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += 30
            cv2.putText(grid, 'rpy=(%.2f, %.2f, %.2f) deg' % (pose_rpy_deg[0], pose_rpy_deg[1], pose_rpy_deg[2]),
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(grid, 'rigid6d: need >=3 modeled LEDs (visible=%d)' % len(visible_names),
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2, cv2.LINE_AA)

        self.preview_backend.show_frame(grid)

        key = self.preview_backend.poll_key()
        if key in (ord('p'), ord('P')):
            self.paused = True
        elif key in (ord('r'), ord('R')):
            self.paused = False
        elif key == ord(' '):
            if self.recording:
                self.stop_recording()
                if self.task_complete:
                    return False
            else:
                self.start_recording()
        elif key in (8, 127):
            self.hand_digit_buffer = ''
        elif key in (10, 13):
            self._submit_hand_digit_buffer()
        elif ord('0') <= key <= ord('9'):
            self._on_digit_key(chr(key), now)
        elif key in (ord('q'), ord('Q'), 27):
            return False

        if self.hand_click_gesture is not None:
            gid = int(self.hand_click_gesture)
            self.hand_click_gesture = None
            if 1 <= gid <= 17:
                self._send_hand_gesture(gid)

        if self.hand_digit_buffer and (now - self.hand_digit_last_wall) > self.hand_digit_timeout_s:
            self._submit_hand_digit_buffer()
        return True

    def handle_post_process_choice(self):
        choice = self.args.post_process
        if choice == 'ask':
            choice = 'now' if console.ask_yes_no('online collection finished. run offline post-processing now?') else 'later'

        if choice == 'now':
            console.warning('offline post-processing entrypoint is not implemented yet; raw data is ready at: %s' % self.output_root)
        else:
            console.saved('post-processing deferred. raw data is ready at: %s' % self.output_root)

    def close(self):
        """
        Comprehensive cleanup that ensures all threads, files, and resources are properly closed.
        Uses longer timeouts and handles cleanup failures gracefully.
        """
        console.info('[cleanup] starting graceful shutdown...')
        
        # Stop any active recording
        if self.recording:
            try:
                console.info('[cleanup] stopping active recording...')
                self.stop_recording()
            except Exception as e:
                console.warning('[cleanup] error stopping recording: %s' % e)

        # Stop VideoSink threads (these have their own worker threads)
        if self.active_sinks:
            console.info('[cleanup] closing %d active video sinks...' % len(self.active_sinks))
            for sink in self.active_sinks:
                try:
                    sink.close()
                except Exception as e:
                    console.warning('[cleanup] error closing sink: %s' % e)
            self.active_sinks = []

        # Stop camera capture threads
        console.info('[cleanup] stopping %d HIK camera threads...' % len(self.hik_threads))
        for th in self.hik_threads:
            try:
                th.stop()
            except Exception as e:
                console.warning('[cleanup] error stopping HIK thread: %s' % e)
        
        if self.rs_thread is not None:
            console.info('[cleanup] stopping RealSense thread...')
            try:
                self.rs_thread.stop()
            except Exception as e:
                console.warning('[cleanup] error stopping RS thread: %s' % e)

        # Wait for camera threads with extended timeout (5 sec each)
        console.info('[cleanup] waiting for camera threads to finish...')
        for i, th in enumerate(self.hik_threads):
            try:
                if th.is_alive():
                    console.info('[cleanup] waiting for HIK thread %d (timeout=5.0s)...' % i)
                    th.join(timeout=5.0)
                    if th.is_alive():
                        console.warning('[cleanup] HIK thread %d did not finish in time' % i)
            except Exception as e:
                console.warning('[cleanup] error joining HIK thread %d: %s' % (i, e))
        
        if self.rs_thread is not None:
            try:
                if self.rs_thread.is_alive():
                    console.info('[cleanup] waiting for RS thread (timeout=5.0s)...')
                    self.rs_thread.join(timeout=5.0)
                    if self.rs_thread.is_alive():
                        console.warning('[cleanup] RS thread did not finish in time')
            except Exception as e:
                console.warning('[cleanup] error joining RS thread: %s' % e)

        # Close camera grabbers
        console.info('[cleanup] closing camera grabbers...')
        for i, rec in enumerate(self.recorders):
            try:
                rec.stop_and_close()
            except Exception as e:
                console.warning('[cleanup] error closing HIK grabber %d: %s' % (i, e))
        
        if self.rs_grabber is not None:
            try:
                self.rs_grabber.stop_and_close()
            except Exception as e:
                console.warning('[cleanup] error closing RS grabber: %s' % e)

        # Close visualization and UI
        console.info('[cleanup] closing UI and visualization...')
        if self.plotter3d is not None:
            try:
                self.plotter3d.close()
            except Exception as e:
                console.warning('[cleanup] error closing 3D plotter: %s' % e)
        
        if self.preview_backend is not None:
            try:
                self.preview_backend.close()
            except Exception as e:
                console.warning('[cleanup] error closing preview backend: %s' % e)
            self.preview_backend = None

        # Close hand-related files
        console.info('[cleanup] closing hand command files...')
        if self.hand_task_cmd_file is not None:
            try:
                self.hand_task_cmd_file.flush()
                self.hand_task_cmd_file.close()
            except Exception as e:
                console.warning('[cleanup] error closing hand task cmd file: %s' % e)
            self.hand_task_cmd_file = None
            self.hand_task_cmd_writer = None

        # Close hand connection
        if self.hand_client is not None:
            try:
                self.hand_client.close()
            except Exception as e:
                console.warning('[cleanup] error closing hand client: %s' % e)

        # Close UI windows
        if self.args.ui_backend == 'opencv':
            try:
                console.info('[cleanup] destroying OpenCV windows...')
                cv2.destroyAllWindows()
            except Exception as e:
                console.warning('[cleanup] error destroying windows: %s' % e)

        # Finalize camera SDK
        try:
            console.info('[cleanup] finalizing camera SDK...')
            MvCamera.MV_CC_Finalize()
        except Exception as e:
            console.warning('[cleanup] error finalizing SDK: %s' % e)
        
        console.success('[cleanup] shutdown complete')

    def _run_temporal_calib(self):
        """Run AprilTag temporal calibration using the already-open camera threads."""
        args = self.args
        if getattr(args, 'skip_temporal_calib', False) or not parse_yes_no(args.temporal_calib):
            console.info('temporal calibration skipped (--skip-temporal-calib)')
            return
        try:
            from prism.recording.temporal_calibration import run_session_temporal_calibration
            run_session_temporal_calibration(
                hik_threads=self.hik_threads,
                rs_thread=self.rs_thread,
                hik_serials=self.hik_serials,
                rs_serial=args.rs_serial.strip() or None,
                output_dir=self.output_root,
                tag_id=args.temporal_calib_tag_id,
                tag_delay_s=args.temporal_calib_tag_delay_s,
                tag_display_s=args.temporal_calib_tag_display_s,
            )
        except Exception as exc:
            console.warning('temporal calibration failed: %s' % exc)
            console.warning('continuing without temporal offsets.')

    def run(self):
        try:
            self.prepare_session()
            self.open_devices()
            self._run_temporal_calib()
            self.run_loop()
        finally:
            self.close()

        console.saved('trajectory (nearest) csv saved: %s' % self.traj_near_csv_path)
        console.saved('trajectory (interp)  csv saved: %s' % self.traj_interp_csv_path)
        if self.hand_task_cmd_path:
            console.saved('hand sdk timeline csv saved: %s' % self.hand_task_cmd_path)
        self._analyze_accuracy()
        self.handle_post_process_choice()
        console.done('done.')

    def _analyze_accuracy(self):
        """Print a LED localization accuracy report after the session ends."""
        try:
            from prism.processing.led_accuracy import print_accuracy_report
        except Exception as ex:
            console.warning('精度分析模块导入失败，跳过: %s' % ex)
            return
        try:
            console.info('正在计算 LED 定位精度...')
            print_accuracy_report(
                self.traj_near_csv_path,
                rigid_path=self.pose_csv_path if os.path.isfile(self.pose_csv_path) else None,
            )
        except Exception as ex:
            console.warning('精度分析失败，跳过: %s' % ex)


def main(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=DEFAULT_ONLINE_CONFIG)
    config_args, _ = config_parser.parse_known_args(argv)
    config_values = load_yaml_config(config_args.config)
    defaults = merge_defaults(DEFAULT_CLI_VALUES, config_values)

    parser = build_arg_parser(defaults, config_args.config)
    args = parser.parse_args(argv)
    if not args.calib_json:
        parser.error('--calib-json is required unless calib_json is set in --config')
    
    # Create session manager
    session_manager = SessionManager(args)
    
    # Set up signal handlers for graceful shutdown
    def handle_shutdown_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        console.warning('[signal] received %s, initiating graceful shutdown...' % sig_name)
        session_manager.close()
        sys.exit(0)
    
    # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM (system termination)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    try:
        return session_manager.run()
    except KeyboardInterrupt:
        console.warning('[signal] keyboard interrupt detected, shutting down...')
        session_manager.close()
        return 1
    except Exception as e:
        console.error('[error] unexpected error during session: %s' % e)
        console.warning('[error] forcing cleanup...')
        try:
            session_manager.close()
        except Exception as cleanup_error:
            console.warning('[error] cleanup error: %s' % cleanup_error)
        raise


if __name__ == '__main__':
    main()
