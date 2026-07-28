#!/usr/bin/env python3
"""Interactive 4-camera live LED annotation tool.

The tool opens the PRISM 4 Hik cameras, shows a 2x2 live preview, and lets you
freeze a synchronized frame set, click LED centers in each camera view, and
save YOLO-format labels per camera.

Keys:
    c / space  freeze the current 4-camera frame set
    1..4       select class: red / yellow / blue / green
    left click add a box centered at the click in the hovered camera tile
    right click undo last box for the hovered camera tile
    [ / ]      shrink / grow box size
    s / enter  save frozen frame set and labels
    n          discard frozen frame set and continue
    q / esc    quit
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
from prism.common.config import load_yaml_config  # noqa: E402
from prism.reconstruction.realtime_reconstruction import detect_led_hsv  # noqa: E402


CLASS_NAMES = ['red', 'yellow', 'blue', 'green']
CLASS_KEYS = {
    ord('1'): 'red',
    ord('2'): 'yellow',
    ord('3'): 'blue',
    ord('4'): 'green',
}
CLASS_BGR = {
    'red': (0, 0, 255),
    'yellow': (0, 220, 255),
    'blue': (255, 120, 0),
    'green': (0, 220, 0),
}
MASTER_SERIAL = 'DA8165486'
COLOR_TO_PREFIX = {'red': 'r', 'yellow': 'y', 'blue': 'b', 'green': 'g'}


def _build_hsv_cfg_from_config(config_path):
    cfg = load_yaml_config(config_path)
    hsv_cfg = {}
    for color, pfx in COLOR_TO_PREFIX.items():
        hsv_cfg[color] = (
            (int(cfg['%s_h_low' % pfx]), int(cfg['%s_s_low' % pfx]), int(cfg['%s_v_low' % pfx])),
            (int(cfg['%s_h_high' % pfx]), int(cfg['%s_s_high' % pfx]), int(cfg['%s_v_high' % pfx])),
        )
    min_area = float(cfg.get('min_area', 10.0))
    return hsv_cfg, min_area


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


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


class LiveLedAnnotator(object):
    def __init__(self, output_dir, class_names, box_size_px=28, exposure_us=None, gain=None,
                 frame_rate=30.0, camera_indices=None, camera_serials=None,
                 auto_from_hsv=False, hsv_config_path=''):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_dir / ('LedAnnot_%s' % time.strftime('%Y%m%d_%H%M%S'))
        self.images_dir = self.session_dir / 'images'
        self.labels_dir = self.session_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = list(class_names)
        self.box_size_px = int(box_size_px)
        self.exposure_us = exposure_us
        self.gain = gain
        self.frame_rate = float(frame_rate)
        self.camera_indices = list(camera_indices or [-1, -1, -1, -1])
        self.camera_serials = list(camera_serials or ['', '', '', ''])
        self.auto_from_hsv = bool(auto_from_hsv)
        self.hsv_config_path = hsv_config_path
        self.hsv_cfg = None
        self.hsv_min_area = 10.0

        self.current_class = self.class_names[0]
        self.freeze_frames = None
        self.freeze_stamp = None
        self.boxes = {i: [] for i in range(4)}
        self.saved_count = 0

        self.cameras = []
        self.capture_threads = []
        self.trigger_thread = None
        self.master_cam_index = 0
        self.camera_dir_names = []

        self.preview_geom = []
        self._hover_cam = 0

        if self.auto_from_hsv:
            if not self.hsv_config_path:
                raise RuntimeError('auto-from-hsv enabled but --hsv-config is empty')
            self.hsv_cfg, self.hsv_min_area = _build_hsv_cfg_from_config(self.hsv_config_path)

    def _select_4_devices(self):
        MvCamera.MV_CC_Initialize()
        usb_devices = enumerate_usb_devices()
        if len(usb_devices) < 4:
            raise RuntimeError('Found %d USB cameras, need at least 4 Hik cameras' % len(usb_devices))

        print('Found %d USB cameras:' % len(usb_devices))
        for i, _, model, serial in usb_devices:
            print('  [%d] model=%s serial=%s' % (i, model, serial))

        selected = []
        if any(self.camera_serials):
            wanted = [s for s in self.camera_serials if s]
            if len(wanted) != 4:
                raise RuntimeError('camera-serials requires exactly 4 serials separated by commas')
            serial_to_dev = {serial: (idx, dev_info, model, serial) for idx, dev_info, model, serial in usb_devices}
            for serial in wanted:
                if serial not in serial_to_dev:
                    raise RuntimeError('camera serial not found: %s' % serial)
                selected.append(serial_to_dev[serial])
        elif all(idx >= 0 for idx in self.camera_indices):
            for idx in self.camera_indices:
                if idx >= len(usb_devices):
                    raise RuntimeError('camera index out of range: %d' % idx)
                selected.append(usb_devices[idx])
        else:
            raw = input('camera indices for 4-camera annotation (blank=0,1,2,3): ').strip()
            chosen = list(range(4)) if raw == '' else parse_indices(raw, len(usb_devices), 4)
            selected = [usb_devices[idx] for idx in chosen]

        if len(selected) != 4:
            raise RuntimeError('need exactly 4 cameras, got %d' % len(selected))
        return selected

    def open_cameras(self):
        selected = self._select_4_devices()
        self.cameras = []
        self.camera_dir_names = []
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
            dir_name = serial_safe.replace('/', '_').replace('\\', '_').replace(' ', '_')
            if not dir_name:
                dir_name = 'cam%d' % cam_i
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
        frames = []
        for cam_i in range(4):
            th = self.capture_threads[cam_i] if cam_i < len(self.capture_threads) else None
            frame = th.get_latest() if th is not None else None
            frames.append(frame.copy() if frame is not None else None)
        return frames

    def _freeze_latest(self):
        frames = self._latest_frames()
        if any(frame is None for frame in frames):
            return False
        self.freeze_frames = frames
        self.freeze_stamp = time.strftime('%Y%m%d_%H%M%S')
        self.boxes = {i: [] for i in range(4)}
        if self.auto_from_hsv:
            self._auto_label_from_hsv()
        return True

    def _auto_label_from_hsv(self):
        if self.freeze_frames is None or self.hsv_cfg is None:
            return
        for cam_i, frame in enumerate(self.freeze_frames):
            h, w = frame.shape[:2]
            for color in self.class_names:
                low, high = self.hsv_cfg[color]
                pt, _area = detect_led_hsv(frame, low, high, self.hsv_min_area)
                if pt is None:
                    continue
                x, y = int(round(pt[0])), int(round(pt[1]))
                half = self.box_size_px // 2
                x1 = _clamp(x - half, 0, w - 1)
                y1 = _clamp(y - half, 0, h - 1)
                x2 = _clamp(x + half, 0, w - 1)
                y2 = _clamp(y + half, 0, h - 1)
                if x2 <= x1:
                    x2 = min(w - 1, x1 + 1)
                if y2 <= y1:
                    y2 = min(h - 1, y1 + 1)
                self.boxes[cam_i].append({
                    'cls': color,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                })

    def _add_box(self, cam_i, x, y):
        if self.freeze_frames is None:
            if not self._freeze_latest():
                return
        frame = self.freeze_frames[cam_i]
        h, w = frame.shape[:2]
        half = self.box_size_px // 2
        x1 = _clamp(x - half, 0, w - 1)
        y1 = _clamp(y - half, 0, h - 1)
        x2 = _clamp(x + half, 0, w - 1)
        y2 = _clamp(y + half, 0, h - 1)
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        # Keep only one instance per color in each camera view.
        self.boxes[cam_i] = [b for b in self.boxes[cam_i] if b.get('cls') != self.current_class]
        self.boxes[cam_i].append({
            'cls': self.current_class,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
        })

    def _undo_box(self, cam_i):
        if self.boxes.get(cam_i):
            self.boxes[cam_i].pop()

    def _clear_camera_boxes(self, cam_i):
        self.boxes[cam_i] = []

    def _clear_all_boxes(self):
        self.boxes = {i: [] for i in range(4)}

    def _remove_current_class_box(self, cam_i):
        self.boxes[cam_i] = [b for b in self.boxes[cam_i] if b.get('cls') != self.current_class]

    def _save_current(self):
        if self.freeze_frames is None:
            if not self._freeze_latest():
                return False

        set_id = 'set_%06d' % self.saved_count
        meta_path = self.session_dir / 'meta.json'

        for cam_i, frame in enumerate(self.freeze_frames):
            cam_dir = self.images_dir / self.camera_dir_names[cam_i]
            label_dir = self.labels_dir / self.camera_dir_names[cam_i]
            cam_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            image_path = cam_dir / (set_id + '.png')
            label_path = label_dir / (set_id + '.txt')
            cv2.imwrite(str(image_path), frame)

            h, w = frame.shape[:2]
            lines = []
            for box in self.boxes.get(cam_i, []):
                cls_idx = self.class_names.index(box['cls'])
                x_c = ((box['x1'] + box['x2']) * 0.5) / float(w)
                y_c = ((box['y1'] + box['y2']) * 0.5) / float(h)
                bw = max(1.0, float(box['x2'] - box['x1'])) / float(w)
                bh = max(1.0, float(box['y2'] - box['y1'])) / float(h)
                lines.append('%d %.6f %.6f %.6f %.6f' % (cls_idx, x_c, y_c, bw, bh))

            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                if lines:
                    f.write('\n')

        if not meta_path.exists():
            payload = {
                'class_names': self.class_names,
                'box_size_px': self.box_size_px,
                'auto_from_hsv': self.auto_from_hsv,
                'hsv_config': self.hsv_config_path,
                'camera_serials': [serial for _, _, serial in self.cameras],
                'camera_dir_names': self.camera_dir_names,
                'camera_mapping': [
                    {
                        'logical_index': cam_i,
                        'dir_name': self.camera_dir_names[cam_i],
                        'serial': serial,
                    }
                    for cam_i, _, serial in self.cameras
                ],
                'capture': {
                    'exposure_us': self.exposure_us,
                    'gain': self.gain,
                    'frame_rate': self.frame_rate,
                },
                'layout': '4-camera synchronized preview',
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        print('saved %s for 4 cameras' % set_id)
        self.saved_count += 1
        self.freeze_frames = None
        self.freeze_stamp = None
        self.boxes = {i: [] for i in range(4)}
        return True

    def _draw_tile(self, frame, cam_i):
        view = frame.copy()
        h, w = view.shape[:2]
        for box in self.boxes.get(cam_i, []):
            color = CLASS_BGR[box['cls']]
            cv2.rectangle(view, (box['x1'], box['y1']), (box['x2'], box['y2']), color, 2)
            cv2.putText(view, box['cls'], (box['x1'], max(16, box['y1'] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        serial = self.cameras[cam_i][2] if cam_i < len(self.cameras) else 'unknown'
        label = '%s%s' % (serial, ' [MASTER]' if cam_i == self.master_cam_index else '')
        status = 'freeze' if self.freeze_frames is not None else 'live'
        cv2.rectangle(view, (0, 0), (w - 1, 36), (20, 20, 20), -1)
        cv2.putText(view, '%s | %s | class=%s | box=%dpx' % (label, status, self.current_class, self.box_size_px),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        return view

    def _build_mosaic(self, target_w=640):
        frames = self.freeze_frames if self.freeze_frames is not None else self._latest_frames()
        tiles = []
        self.preview_geom = []
        tile_meta = []

        for cam_i in range(4):
            frame = frames[cam_i]
            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, 'cam%d waiting...' % cam_i, (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

            tile = self._draw_tile(frame, cam_i)
            src_h, src_w = tile.shape[:2]
            scale = target_w / float(src_w)
            draw_h = max(1, int(round(src_h * scale)))
            tile = _resize_with_quality(tile, target_w, draw_h)
            tiles.append(tile)
            tile_meta.append({
                'src_w': src_w,
                'src_h': src_h,
                'draw_w': target_w,
                'draw_h': draw_h,
            })

        max_h = max(tile.shape[0] for tile in tiles)
        max_w = max(tile.shape[1] for tile in tiles)
        padded = []
        for tile in tiles:
            h, w = tile.shape[:2]
            if h != max_h or w != max_w:
                tile = cv2.copyMakeBorder(tile, 0, max_h - h, 0, max_w - w,
                                          cv2.BORDER_CONSTANT, value=0)
            padded.append(tile)

        top = np.hstack([padded[0], padded[1]])
        bottom = np.hstack([padded[2], padded[3]])
        mosaic = np.vstack([top, bottom])

        self.preview_geom = []
        for cam_i, tile in enumerate(padded):
            row = 0 if cam_i < 2 else 1
            col = cam_i % 2
            x0 = col * max_w
            y0 = row * max_h
            meta = tile_meta[cam_i]
            self.preview_geom.append({
                'cam_i': cam_i,
                'tile_x0': x0,
                'tile_y0': y0,
                'tile_x1': x0 + max_w,
                'tile_y1': y0 + max_h,
                'draw_w': int(meta['draw_w']),
                'draw_h': int(meta['draw_h']),
                'src_w': int(meta['src_w']),
                'src_h': int(meta['src_h']),
            })

        banner = 'saved=%d | freeze=%s | auto_hsv=%s | c freeze | click replace | v drop-color | s save | n discard | x clear-cam | X clear-all | [ ] size | 1-4 class | q quit' % (
            self.saved_count, 'yes' if self.freeze_frames is not None else 'no'
            , 'on' if self.auto_from_hsv else 'off'
        )
        cv2.rectangle(mosaic, (0, 0), (mosaic.shape[1] - 1, 40), (28, 28, 28), -1)
        cv2.putText(mosaic, banner, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
        return mosaic

    def _mouse_to_camera(self, x, y):
        for geom in self.preview_geom:
            x0 = geom['tile_x0']
            y0 = geom['tile_y0']
            x1 = geom['tile_x1']
            y1 = geom['tile_y1']
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue

            # Ignore clicks in bottom/right padding outside the actual drawn image.
            lx = x - x0
            ly = y - y0
            draw_w = geom['draw_w']
            draw_h = geom['draw_h']
            if lx < 0 or ly < 0 or lx >= draw_w or ly >= draw_h:
                return geom['cam_i'], None, None

            src_w = geom['src_w']
            src_h = geom['src_h']
            src_x = int(round((float(lx) / float(max(1, draw_w))) * float(src_w - 1)))
            src_y = int(round((float(ly) / float(max(1, draw_h))) * float(src_h - 1)))
            src_x = _clamp(src_x, 0, src_w - 1)
            src_y = _clamp(src_y, 0, src_h - 1)
            return geom['cam_i'], src_x, src_y
        return None, None, None

    def run(self):
        win = 'PRISM 4-Camera Live LED Annotator'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, _param):
            cam_i, lx, ly = self._mouse_to_camera(x, y)
            if cam_i is None:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                self._hover_cam = cam_i
                if lx is None or ly is None:
                    return
                self._add_box(cam_i, lx, ly)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._hover_cam = cam_i
                self._undo_box(cam_i)

        cv2.setMouseCallback(win, on_mouse)

        print('4-camera live annotator ready.')
        print('  c/space freeze synchronized frame set, click inside each tile to set/replace current-color box, s save, n discard, q quit')
        print('  v removes current color box in hovered camera (use for occlusion / not visible)')
        if self.auto_from_hsv:
            print('  auto HSV pre-label enabled from: %s' % self.hsv_config_path)

        try:
            while True:
                mosaic = self._build_mosaic(target_w=640)
                cv2.imshow(win, mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('c'), ord('C'), ord(' ')):
                    self._freeze_latest()
                elif key in CLASS_KEYS:
                    self.current_class = CLASS_KEYS[key]
                elif key in (ord('['),):
                    self.box_size_px = max(6, self.box_size_px - 2)
                elif key in (ord(']'),):
                    self.box_size_px = min(200, self.box_size_px + 2)
                elif key in (ord('s'), ord('S'), 10, 13):
                    self._save_current()
                elif key in (ord('n'), ord('N')):
                    self.freeze_frames = None
                    self.freeze_stamp = None
                    self.boxes = {i: [] for i in range(4)}
                elif key == ord('x'):
                    self._clear_camera_boxes(self._hover_cam)
                elif key == ord('X'):
                    self._clear_all_boxes()
                elif key in (ord('v'), ord('V')):
                    self._remove_current_class_box(self._hover_cam)
                elif key in (ord('a'), ord('A')):
                    if self.freeze_frames is not None and self.auto_from_hsv:
                        self._clear_all_boxes()
                        self._auto_label_from_hsv()
        finally:
            cv2.destroyWindow(win)
            self.close_cameras()


def parse_args():
    p = argparse.ArgumentParser(description='PRISM 4-camera live LED annotation tool')
    p.add_argument('--output-dir', type=str, default='data/datasets/led_yolo', help='dataset output root')
    p.add_argument('--camera-indices', type=str, default='', help='4 camera indices like 0,1,2,3')
    p.add_argument('--camera-serials', type=str, default='', help='4 camera serials like a,b,c,d')
    p.add_argument('--exposure-us', type=float, default=3000.0)
    p.add_argument('--gain', type=float, default=12.0)
    p.add_argument('--frame-rate', type=float, default=30.0)
    p.add_argument('--box-size-px', type=int, default=28, help='fixed YOLO box side length in pixels')
    p.add_argument('--auto-from-hsv', type=str, default='y', choices=['y', 'n'],
                   help='pre-label frozen frames with HSV detections before manual correction')
    p.add_argument('--hsv-config', type=str,
                   default=str(ROOT / 'configs' / 'collection' / 'default_online.yaml'),
                   help='config YAML providing r/y/b/g HSV ranges for auto pre-label')
    return p.parse_args()


def main(argv=None):
    args = parse_args()
    camera_indices = []
    if args.camera_indices.strip():
        camera_indices = [int(x.strip()) for x in args.camera_indices.split(',') if x.strip()]
        if len(camera_indices) != 4:
            raise RuntimeError('--camera-indices must contain exactly 4 values')
    camera_serials = []
    if args.camera_serials.strip():
        camera_serials = [x.strip() for x in args.camera_serials.split(',') if x.strip()]
        if len(camera_serials) != 4:
            raise RuntimeError('--camera-serials must contain exactly 4 values')

    annot = LiveLedAnnotator(
        output_dir=args.output_dir,
        class_names=CLASS_NAMES,
        box_size_px=args.box_size_px,
        exposure_us=args.exposure_us,
        gain=args.gain,
        frame_rate=args.frame_rate,
        camera_indices=camera_indices,
        camera_serials=camera_serials,
        auto_from_hsv=(args.auto_from_hsv == 'y'),
        hsv_config_path=args.hsv_config,
    )
    annot.open_cameras()
    annot.run()


if __name__ == '__main__':
    main()