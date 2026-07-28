#!/usr/bin/env python3
"""Capture-first tool: collect synchronized 4-camera LED photos without labeling.

Output layout:
  <output-dir>/LedPhoto_<timestamp>/images/<serial>/set_000000.png
  <output-dir>/LedPhoto_<timestamp>/meta.json

Keys:
  SPACE / ENTER: save one synchronized 4-camera frame set
  q / ESC: quit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_mv_candidates = []
if os.environ.get('PRISM_MVIMPORT_DIR'):
    _mv_candidates.append(os.environ['PRISM_MVIMPORT_DIR'])
_mv_candidates.append('/opt/MVS/Samples/64/Python/MvImport')
for _p in _mv_candidates:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism.devices.cameras.mvs_camera import MvCamera, UsbCameraGrabber, enumerate_usb_devices, parse_indices  # noqa: E402
from prism.devices.cameras.highspeed_camera import HikCaptureThread, SoftwareTriggerThread  # noqa: E402


MASTER_SERIAL = 'DA8165486'


def _resize_with_quality(img, out_w, out_h):
    in_h, in_w = img.shape[:2]
    if out_w <= 0 or out_h <= 0:
        return img
    if out_w == in_w and out_h == in_h:
        return img
    if out_w < in_w or out_h < in_h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


class LedPhotoCapture(object):
    def __init__(self, output_dir, exposure_us, gain, frame_rate, camera_indices=None, camera_serials=None):
        out_root = Path(output_dir).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
        self.session_dir = out_root / ('LedPhoto_%s' % time.strftime('%Y%m%d_%H%M%S'))
        self.images_dir = self.session_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.exposure_us = float(exposure_us)
        self.gain = float(gain)
        self.frame_rate = float(frame_rate)
        self.camera_indices = list(camera_indices or [])
        self.camera_serials = list(camera_serials or [])

        self.cameras = []
        self.capture_threads = []
        self.trigger_thread = None
        self.master_cam_index = 0
        self.camera_dir_names = []
        self.saved_count = 0

    def _select_4_devices(self):
        MvCamera.MV_CC_Initialize()
        usb_devices = enumerate_usb_devices()
        if len(usb_devices) < 4:
            raise RuntimeError('Found %d USB cameras, need at least 4 Hik cameras' % len(usb_devices))

        print('Found %d USB cameras:' % len(usb_devices))
        for i, _, model, serial in usb_devices:
            print('  [%d] model=%s serial=%s' % (i, model, serial))

        selected = []
        if self.camera_serials:
            if len(self.camera_serials) != 4:
                raise RuntimeError('camera-serials requires exactly 4 serials separated by commas')
            serial_to_dev = {serial: info for info in usb_devices for serial in [info[3]]}
            for serial in self.camera_serials:
                if serial not in serial_to_dev:
                    raise RuntimeError('camera serial not found: %s' % serial)
                selected.append(serial_to_dev[serial])
        elif self.camera_indices:
            if len(self.camera_indices) != 4:
                raise RuntimeError('camera-indices requires exactly 4 values')
            for idx in self.camera_indices:
                if idx < 0 or idx >= len(usb_devices):
                    raise RuntimeError('camera index out of range: %d' % idx)
                selected.append(usb_devices[idx])
        else:
            raw = input('camera indices for 4-camera capture (blank=0,1,2,3): ').strip()
            chosen = list(range(4)) if raw == '' else parse_indices(raw, len(usb_devices), 4)
            selected = [usb_devices[idx] for idx in chosen]

        if len(selected) != 4:
            raise RuntimeError('need exactly 4 cameras, got %d' % len(selected))
        return selected

    def open_cameras(self):
        selected = self._select_4_devices()
        selected_serials = [serial for _, _, _, serial in selected]
        if MASTER_SERIAL in selected_serials:
            self.master_cam_index = selected_serials.index(MASTER_SERIAL)
        else:
            self.master_cam_index = 0
            print('warning: master serial %s not found in selection; using cam0 as master' % MASTER_SERIAL)

        used_names = set()
        for cam_i, (_, dev_info, model, serial) in enumerate(selected):
            serial_safe = serial if serial else ('cam%d' % cam_i)
            print('Initializing hik%d: model=%s serial=%s' % (cam_i, model, serial_safe))

            cam = UsbCameraGrabber(dev_info, serial_safe, model)
            cam.open_and_prepare(
                exposure_us=self.exposure_us,
                gain=self.gain,
                frame_rate=self.frame_rate,
                trigger_source='Software' if cam_i == self.master_cam_index else 'Line0',
                gpio_output_line=1 if cam_i == self.master_cam_index else None,
            )
            self.cameras.append((cam_i, cam, serial_safe))

            dir_name = serial_safe.replace('/', '_').replace('\\', '_').replace(' ', '_') or ('cam%d' % cam_i)
            if dir_name in used_names:
                dir_name = '%s_cam%d' % (dir_name, cam_i)
            used_names.add(dir_name)
            self.camera_dir_names.append(dir_name)

        self.capture_threads = [None] * len(self.cameras)
        for cam_i, cam, serial in self.cameras:
            th = HikCaptureThread(cam, cam_i, serial, timeout_ms=1000, buffer_len=3)
            th.start()
            self.capture_threads[cam_i] = th

        master_cam = self.cameras[self.master_cam_index][1]
        self.trigger_thread = SoftwareTriggerThread(master_cam, fps=self.frame_rate)
        self.trigger_thread.start()
        print('Master software trigger started at %.1f fps' % self.frame_rate)

    def close_cameras(self):
        if self.trigger_thread is not None:
            try:
                self.trigger_thread.stop()
                self.trigger_thread.join(timeout=1.0)
            except Exception:
                pass
            self.trigger_thread = None
        for th in self.capture_threads:
            if th is not None:
                try:
                    th.stop()
                    th.join(timeout=2.0)
                except Exception:
                    pass
        self.capture_threads = []
        for _, cam, _ in self.cameras:
            try:
                cam.stop_and_close()
            except Exception:
                pass
        self.cameras = []
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass

    def _latest_frames(self):
        out = []
        for cam_i in range(4):
            th = self.capture_threads[cam_i] if cam_i < len(self.capture_threads) else None
            frame = th.get_latest() if th is not None else None
            out.append(frame.copy() if frame is not None else None)
        return out

    def _save_set(self, frames):
        set_id = 'set_%06d' % self.saved_count
        for cam_i, frame in enumerate(frames):
            cam_dir = self.images_dir / self.camera_dir_names[cam_i]
            cam_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cam_dir / (set_id + '.png')), frame)
        self.saved_count += 1
        print('saved %s (4 cams)' % set_id)

    def _build_mosaic(self, frames, target_w=640):
        tiles = []
        for cam_i, frame in enumerate(frames):
            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, 'cam%d waiting...' % cam_i, (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
            serial = self.cameras[cam_i][2] if cam_i < len(self.cameras) else 'unknown'
            label = '%s%s' % (serial, ' [MASTER]' if cam_i == self.master_cam_index else '')
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 36), (20, 20, 20), -1)
            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            h, w = frame.shape[:2]
            s = target_w / float(w)
            tiles.append(_resize_with_quality(frame, target_w, max(1, int(round(h * s)))))

        max_h = max(t.shape[0] for t in tiles)
        max_w = max(t.shape[1] for t in tiles)
        pads = []
        for t in tiles:
            h, w = t.shape[:2]
            if h != max_h or w != max_w:
                t = cv2.copyMakeBorder(t, 0, max_h - h, 0, max_w - w, cv2.BORDER_CONSTANT, value=0)
            pads.append(t)

        top = np.hstack([pads[0], pads[1]])
        bottom = np.hstack([pads[2], pads[3]])
        mosaic = np.vstack([top, bottom])
        txt = 'saved=%d | SPACE/ENTER save set | q quit' % self.saved_count
        cv2.rectangle(mosaic, (0, 0), (mosaic.shape[1] - 1, 40), (28, 28, 28), -1)
        cv2.putText(mosaic, txt, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
        return mosaic

    def save_meta(self):
        meta = {
            'camera_serials': [serial for _, _, serial in self.cameras],
            'camera_dir_names': self.camera_dir_names,
            'master_serial': MASTER_SERIAL,
            'master_cam_index': int(self.master_cam_index),
            'capture': {
                'exposure_us': self.exposure_us,
                'gain': self.gain,
                'frame_rate': self.frame_rate,
            },
            'saved_sets': int(self.saved_count),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.session_dir / 'meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def run(self):
        win = 'PRISM 4-Camera Photo Capture'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print('Capture ready: SPACE/ENTER save one synchronized set, q/ESC quit.')
        try:
            while True:
                frames = self._latest_frames()
                mosaic = self._build_mosaic(frames, target_w=640)
                cv2.imshow(win, mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord(' '), 10, 13):
                    if any(f is None for f in frames):
                        print('skip save: some camera frames are not ready')
                        continue
                    self._save_set(frames)
        finally:
            cv2.destroyWindow(win)
            self.save_meta()
            self.close_cameras()


def parse_args():
    p = argparse.ArgumentParser(description='Capture synchronized 4-camera LED photos')
    p.add_argument('--output-dir', type=str, default='data/datasets/led_yolo_raw', help='photo dataset output root')
    p.add_argument('--camera-indices', type=str, default='', help='4 camera indices like 0,1,2,3')
    p.add_argument('--camera-serials', type=str, default='', help='4 camera serials like a,b,c,d')
    p.add_argument('--exposure-us', type=float, default=3000.0)
    p.add_argument('--gain', type=float, default=12.0)
    p.add_argument('--frame-rate', type=float, default=30.0)
    return p.parse_args()


def main(argv=None):
    args = parse_args()
    camera_indices = []
    if args.camera_indices.strip():
        camera_indices = [int(x.strip()) for x in args.camera_indices.split(',') if x.strip()]
    camera_serials = []
    if args.camera_serials.strip():
        camera_serials = [x.strip() for x in args.camera_serials.split(',') if x.strip()]

    app = LedPhotoCapture(
        output_dir=args.output_dir,
        exposure_us=args.exposure_us,
        gain=args.gain,
        frame_rate=args.frame_rate,
        camera_indices=camera_indices,
        camera_serials=camera_serials,
    )
    app.open_cameras()
    app.run()


if __name__ == '__main__':
    main()