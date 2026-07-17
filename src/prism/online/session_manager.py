import argparse
import csv
import json
import os
import time

import cv2

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
from prism.online.display_server import draw_preview
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
    build_observations_interp,
    build_observations_nearest,
    make_track_state,
)
from prism.recording.metadata_writer import resolve_output_root
from prism.recording.video_recorder import VideoSink


def parse_yes_no(raw_text):
    return raw_text.strip().lower() in ['y', 'yes', '1', 'true']


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='PRISM native multi-threaded capture: 4 Hik cameras (continuous) + RealSense'
    )
    parser.add_argument('--calib-json', type=str, required=True, help='path to charuco_4cam_result.json')
    parser.add_argument('--output-dir', type=str, default='~/mvs_dexhand_capture_mt', help='output root folder')

    parser.add_argument('--hik-exposure-us', type=float, default=3000.0)
    parser.add_argument('--hik-gain', type=float, default=0.0)
    parser.add_argument('--hik-frame-rate', type=float, default=300.0,
                        help='requested Hik acquisition fps (continuous mode); real fps shown live')
    parser.add_argument('--rec-brightness-alpha', type=float, default=2.0)
    parser.add_argument('--rec-brightness-beta', type=float, default=20.0)
    parser.add_argument('--strict-param-check', type=str, default='y')
    parser.add_argument('--param-tolerance', type=float, default=1e-3)
    parser.add_argument('--writer-queue', type=int, default=512,
                        help='max buffered frames per stream before dropping (protects capture cadence)')

    parser.add_argument('--rs-serial', type=str, default='')
    parser.add_argument('--rs-calib-json', type=str, default='')
    parser.add_argument('--rs-undistort', type=str, default='y')
    parser.add_argument('--rs-width', type=int, default=1280)
    parser.add_argument('--rs-height', type=int, default=720)
    parser.add_argument('--rs-fps', type=int, default=30)
    parser.add_argument('--rs-auto-exposure', type=str, default='n')
    parser.add_argument('--rs-exposure', type=float, default=260.0)
    parser.add_argument('--rs-gain', type=float, default=64.0)
    parser.add_argument('--rs-brightness', type=float, default=0.0)

    parser.add_argument('--y-h-low', type=int, default=15)
    parser.add_argument('--y-s-low', type=int, default=80)
    parser.add_argument('--y-v-low', type=int, default=80)
    parser.add_argument('--y-h-high', type=int, default=40)
    parser.add_argument('--y-s-high', type=int, default=255)
    parser.add_argument('--y-v-high', type=int, default=255)

    parser.add_argument('--b-h-low', type=int, default=90)
    parser.add_argument('--b-s-low', type=int, default=80)
    parser.add_argument('--b-v-low', type=int, default=80)
    parser.add_argument('--b-h-high', type=int, default=135)
    parser.add_argument('--b-s-high', type=int, default=255)
    parser.add_argument('--b-v-high', type=int, default=255)

    parser.add_argument('--g-h-low', type=int, default=40)
    parser.add_argument('--g-s-low', type=int, default=60)
    parser.add_argument('--g-v-low', type=int, default=60)
    parser.add_argument('--g-h-high', type=int, default=95)
    parser.add_argument('--g-s-high', type=int, default=255)
    parser.add_argument('--g-v-high', type=int, default=255)

    parser.add_argument('--min-area', type=float, default=10.0)
    parser.add_argument('--max-norm-reproj-error', type=float, default=0.015)
    parser.add_argument('--max-traj-points', type=int, default=5000)
    parser.add_argument('--max-predict-frames', type=int, default=6)
    parser.add_argument('--viz-3d', type=str, default='y')
    parser.add_argument('--track-every', type=int, default=1,
                        help='run LED tracking every N preview iterations (raise to lighten CPU)')
    parser.add_argument('--frame-buffer', type=int, default=15,
                        help='per-camera timestamped frame buffer length for time-based association')
    return parser


class SessionManager(object):
    def __init__(self, args):
        self.args = args
        self.strict_param_check = parse_yes_no(args.strict_param_check)
        self.rs_auto_exposure = parse_yes_no(args.rs_auto_exposure)
        self.rs_use_undistort = parse_yes_no(args.rs_undistort)
        self.use_viz_3d = parse_yes_no(args.viz_3d)

        self.hsv_cfg = {
            'yellow': ((args.y_h_low, args.y_s_low, args.y_v_low), (args.y_h_high, args.y_s_high, args.y_v_high)),
            'blue': ((args.b_h_low, args.b_s_low, args.b_v_low), (args.b_h_high, args.b_s_high, args.b_v_high)),
            'green': ((args.g_h_low, args.g_s_low, args.g_v_low), (args.g_h_high, args.g_s_high, args.g_v_high)),
        }

        self.cameras = None
        self.camera_centers = None
        self.corrected_transform = None
        self.output_root = None
        self.traj_near_csv_path = None
        self.traj_interp_csv_path = None
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

        self.paused = False
        self.last_time = None
        self.t0 = time.time()

        self.recording = False
        self.segment_id = 0
        self.segment_start_wall = None
        self.active_sinks = []
        self.ts_csv_file = None
        self.ts_csv_writer = None
        self.frame_idx = 0

    def prepare_session(self):
        args = self.args
        self.cameras = load_calibration(os.path.expanduser(args.calib_json))
        self.camera_centers = get_camera_centers_world(self.cameras)
        self.corrected_transform = build_corrected_transform(self.cameras, self.camera_centers)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.output_root = resolve_output_root(os.path.expanduser(args.output_dir), timestamp)
        self.traj_near_csv_path = os.path.join(self.output_root, 'trajectory_3color_nearest.csv')
        self.traj_interp_csv_path = os.path.join(self.output_root, 'trajectory_3color_interp.csv')
        self.align_csv_path = os.path.join(self.output_root, 'time_alignment_log.csv')
        print('output folder: %s' % self.output_root)

        if args.rs_calib_json.strip():
            rs_K, rs_D, _rs_calib_wh, rs_data = load_rs_intrinsics(os.path.expanduser(args.rs_calib_json))
            rs_json_out = os.path.join(self.output_root, 'realsense_intrinsics.json')
            with open(rs_json_out, 'w', encoding='utf-8') as f:
                json.dump(rs_data, f, ensure_ascii=False, indent=2)
            print('realsense intrinsics saved to session: %s' % rs_json_out)
            if self.rs_use_undistort:
                map1, map2, _ = build_undistort_maps(rs_K, rs_D, args.rs_width, args.rs_height)
                self.rs_undistort_maps = (map1, map2)
                print('realsense color undistortion enabled.')
        elif self.rs_use_undistort:
            print('warning: --rs-undistort requested but --rs-calib-json not provided; undistortion disabled.')

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

    def start_recording(self):
        args = self.args
        if self.recording:
            return
        self.segment_id += 1
        seg_dir = os.path.join(self.output_root, 'segment_%03d' % self.segment_id)
        os.makedirs(seg_dir, exist_ok=True)

        self.active_sinks = []
        for cam_i in range(4):
            fps = self.hik_threads[cam_i].fps_meter.fps()
            if fps < 1.0:
                fps = float(args.hik_frame_rate)
            path = os.path.join(seg_dir, 'hik%d_%s.mp4' % (cam_i, self.hik_serials[cam_i]))
            sink = VideoSink(path, fps, max_queue=args.writer_queue,
                             brightness_alpha=args.rec_brightness_alpha,
                             brightness_beta=args.rec_brightness_beta)
            self.hik_threads[cam_i].set_sink(sink)
            self.active_sinks.append(sink)

        rs_fps = self.rs_thread.fps_meter.fps()
        if rs_fps < 1.0:
            rs_fps = float(args.rs_fps)
        rs_path = os.path.join(seg_dir, 'realsense_color.mp4')
        rs_sink = VideoSink(rs_path, rs_fps, max_queue=args.writer_queue)
        self.rs_thread.set_sink(rs_sink)
        self.active_sinks.append(rs_sink)

        self.ts_csv_file = open(os.path.join(seg_dir, 'fps_log.csv'), 'w', newline='', encoding='utf-8')
        self.ts_csv_writer = csv.writer(self.ts_csv_file)
        self.ts_csv_writer.writerow(['wall_time', 'seg_time', 'hik0_fps', 'hik1_fps', 'hik2_fps', 'hik3_fps', 'rs_fps'])

        self.segment_start_wall = time.time()
        self.recording = True
        print('recording started: segment_%03d (hik_fps~%s, rs_fps~%.1f)'
              % (self.segment_id, ['%.1f' % s.fps for s in self.active_sinks[:4]], rs_fps))

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

        self.recording = False
        self.segment_start_wall = None
        print('recording stopped: segment_%03d (written=%d, dropped=%d)'
              % (self.segment_id, total_written, total_dropped))
        if total_dropped > 0:
            print('warning: %d frames dropped (disk/encoding too slow). Lower fps/resolution or raise --writer-queue.'
                  % total_dropped)

    def run_loop(self):
        cv2.namedWindow('DexHand HighFps Capture', cv2.WINDOW_NORMAL)
        self.plotter3d = Live3DPlotter(
            enabled=self.use_viz_3d,
            camera_centers=self.camera_centers,
            corrected_transform=self.corrected_transform,
        )

        print('preview started. click preview window to focus keys.')
        print('keys: SPACE start/stop recording, p pause tracking, r resume tracking, q/ESC quit')

        with open(self.traj_near_csv_path, 'w', newline='', encoding='utf-8') as near_file, \
                open(self.traj_interp_csv_path, 'w', newline='', encoding='utf-8') as interp_file, \
                open(self.align_csv_path, 'w', newline='', encoding='utf-8') as align_file:
            near_writer = csv.writer(near_file)
            interp_writer = csv.writer(interp_file)
            align_writer = csv.writer(align_file)
            header = ['t_sec', 'color', 'x_m', 'y_m', 'z_m', 'mode', 'num_views', 'max_norm_reproj_err', 'visible_cams']
            near_writer.writerow(header)
            interp_writer.writerow(header)
            align_writer.writerow([
                't_ref_sec', 'hik0_ts', 'hik1_ts', 'hik2_ts', 'hik3_ts',
                'hik_spread_ms', 'rs_ts', 'rs_offset_ms',
            ])

            while True:
                if not self._run_once(near_file, interp_file, align_file, near_writer, interp_writer, align_writer):
                    break

    def _run_once(self, near_file, interp_file, align_file, near_writer, interp_writer, align_writer):
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

        segment_elapsed = (now - self.segment_start_wall) if (self.recording and self.segment_start_wall is not None) else 0.0
        if self.recording and self.ts_csv_writer is not None:
            self.ts_csv_writer.writerow(['%.6f' % now, '%.6f' % segment_elapsed] +
                                        ['%.3f' % f for f in hik_fps] + ['%.3f' % rs_fps_now])

        grid = draw_preview(hik_latest, rs_aligned, obs_near, hik_fps, rs_fps_now,
                            self.recording, self.segment_id, segment_elapsed)

        cv2.putText(grid, 'hik sync spread=%.1f ms | rs offset=%s ms' % (
            hik_spread_ms, ('%.1f' % rs_offset_ms) if rs_offset_ms == rs_offset_ms else 'n/a'),
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2, cv2.LINE_AA)

        y0 = 110
        for name in COLOR_ORDER:
            txt = '%s: near=%s interp=%s views=%d' % (
                name, mode_near[name], mode_interp[name], len(obs_near[name]))
            cv2.putText(grid, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_BRG[name], 2, cv2.LINE_AA)
            y0 += 30

        cv2.imshow('DexHand HighFps Capture', grid)

        self.plotter3d.update(
            self.track_near['traj_points'], point_near,
            mode_text='NEAREST | ' + ' | '.join(['%s:%s' % (n, mode_near[n]) for n in COLOR_ORDER]),
        )

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('p'), ord('P')):
            self.paused = True
        elif key in (ord('r'), ord('R')):
            self.paused = False
        elif key == ord(' '):
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key in (ord('q'), ord('Q'), 27):
            return False
        return True

    def close(self):
        if self.recording:
            try:
                self.stop_recording()
            except Exception:
                pass
        for th in self.hik_threads:
            th.stop()
        if self.rs_thread is not None:
            self.rs_thread.stop()
        for th in self.hik_threads:
            th.join(timeout=2.0)
        if self.rs_thread is not None:
            self.rs_thread.join(timeout=2.0)
        for rec in self.recorders:
            try:
                rec.stop_and_close()
            except Exception:
                pass
        if self.rs_grabber is not None:
            try:
                self.rs_grabber.stop_and_close()
            except Exception:
                pass
        if self.plotter3d is not None:
            self.plotter3d.close()
        cv2.destroyAllWindows()
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass

    def run(self):
        try:
            self.prepare_session()
            self.open_devices()
            self.run_loop()
        finally:
            self.close()

        print('trajectory (nearest) csv saved: %s' % self.traj_near_csv_path)
        print('trajectory (interp)  csv saved: %s' % self.traj_interp_csv_path)
        print('done.')


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return SessionManager(args).run()


if __name__ == '__main__':
    main()
