#!/usr/bin/env python3
"""Offline annotator for pre-captured 4-camera synchronized image sets.

Expected input layout:
  <dataset-root>/images/<serial>/set_000000.png
  <dataset-root>/images/<serial>/set_000001.png
  ...

Labels are written to:
  <dataset-root>/labels/<serial>/set_000000.txt
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism.common.config import load_yaml_config  # noqa: E402
from prism.reconstruction.realtime_reconstruction import detect_led_hsv  # noqa: E402


CLASS_NAMES = ['red', 'yellow', 'blue', 'green']
CLASS_KEYS = {ord('1'): 'red', ord('2'): 'yellow', ord('3'): 'blue', ord('4'): 'green'}
CLASS_BGR = {
    'red': (0, 0, 255),
    'yellow': (0, 220, 255),
    'blue': (255, 120, 0),
    'green': (0, 220, 0),
}
COLOR_TO_PREFIX = {'red': 'r', 'yellow': 'y', 'blue': 'b', 'green': 'g'}


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


def _build_hsv_cfg(path):
    cfg = load_yaml_config(path)
    hsv_cfg = {}
    for color, pfx in COLOR_TO_PREFIX.items():
        hsv_cfg[color] = (
            (int(cfg['%s_h_low' % pfx]), int(cfg['%s_s_low' % pfx]), int(cfg['%s_v_low' % pfx])),
            (int(cfg['%s_h_high' % pfx]), int(cfg['%s_s_high' % pfx]), int(cfg['%s_v_high' % pfx])),
        )
    min_area = float(cfg.get('min_area', 10.0))
    return hsv_cfg, min_area


class OfflineAnnotator(object):
    def __init__(self, dataset_root, box_size_px=28, auto_from_hsv=True, hsv_config=''):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.images_root = self.dataset_root / 'images'
        self.labels_root = self.dataset_root / 'labels'
        self.labels_root.mkdir(parents=True, exist_ok=True)
        self.box_size_px = int(box_size_px)
        self.auto_from_hsv = bool(auto_from_hsv)
        self.hsv_config = hsv_config

        self.hsv_cfg = None
        self.hsv_min_area = 10.0
        if self.auto_from_hsv:
            if not self.hsv_config:
                raise RuntimeError('auto-from-hsv enabled but --hsv-config is empty')
            self.hsv_cfg, self.hsv_min_area = _build_hsv_cfg(self.hsv_config)

        self.current_class = CLASS_NAMES[0]
        self._hover_cam = 0
        self.preview_geom = []

        self.cam_names = []
        self.set_ids = []
        self.index = 0
        self.frames = []
        self.boxes = {i: [] for i in range(4)}

    def _discover(self):
        if not self.images_root.is_dir():
            raise RuntimeError('missing images directory: %s' % self.images_root)

        cams = sorted([p.name for p in self.images_root.iterdir() if p.is_dir()])
        if len(cams) != 4:
            raise RuntimeError('expected 4 camera dirs under %s, found %d' % (self.images_root, len(cams)))
        self.cam_names = cams

        common = None
        for cam in self.cam_names:
            stems = set(p.stem for p in (self.images_root / cam).glob('*.png'))
            common = stems if common is None else (common & stems)
        self.set_ids = sorted(common or [])
        if not self.set_ids:
            raise RuntimeError('no common set_*.png found across all camera dirs')

        print('discovered %d synchronized sets' % len(self.set_ids))

    def _label_path(self, cam_i, set_id):
        cam = self.cam_names[cam_i]
        d = self.labels_root / cam
        d.mkdir(parents=True, exist_ok=True)
        return d / (set_id + '.txt')

    def _image_path(self, cam_i, set_id):
        return self.images_root / self.cam_names[cam_i] / (set_id + '.png')

    def _load_existing_labels(self, cam_i, set_id, w, h):
        path = self._label_path(cam_i, set_id)
        out = []
        if not path.is_file():
            return out
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_idx = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:]]
                except Exception:
                    continue
                if cls_idx < 0 or cls_idx >= len(CLASS_NAMES):
                    continue
                x1 = int(round((xc - bw / 2.0) * w))
                y1 = int(round((yc - bh / 2.0) * h))
                x2 = int(round((xc + bw / 2.0) * w))
                y2 = int(round((yc + bh / 2.0) * h))
                out.append({
                    'cls': CLASS_NAMES[cls_idx],
                    'x1': _clamp(x1, 0, w - 1),
                    'y1': _clamp(y1, 0, h - 1),
                    'x2': _clamp(x2, 0, w - 1),
                    'y2': _clamp(y2, 0, h - 1),
                })
        return out

    def _auto_label(self):
        for cam_i, frame in enumerate(self.frames):
            h, w = frame.shape[:2]
            auto_boxes = []
            for color in CLASS_NAMES:
                low, high = self.hsv_cfg[color]
                pt, _ = detect_led_hsv(frame, low, high, self.hsv_min_area)
                if pt is None:
                    continue
                x = int(round(pt[0]))
                y = int(round(pt[1]))
                half = self.box_size_px // 2
                x1 = _clamp(x - half, 0, w - 1)
                y1 = _clamp(y - half, 0, h - 1)
                x2 = _clamp(x + half, 0, w - 1)
                y2 = _clamp(y + half, 0, h - 1)
                auto_boxes.append({'cls': color, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            self.boxes[cam_i] = auto_boxes

    def _load_set(self, idx):
        self.index = max(0, min(idx, len(self.set_ids) - 1))
        set_id = self.set_ids[self.index]
        frames = []
        for cam_i in range(4):
            p = self._image_path(cam_i, set_id)
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError('failed to read %s' % p)
            frames.append(img)
        self.frames = frames

        self.boxes = {i: [] for i in range(4)}
        loaded_any = False
        for cam_i in range(4):
            h, w = self.frames[cam_i].shape[:2]
            b = self._load_existing_labels(cam_i, set_id, w, h)
            if b:
                loaded_any = True
            self.boxes[cam_i] = b

        if (not loaded_any) and self.auto_from_hsv:
            self._auto_label()

    def _save_set(self):
        set_id = self.set_ids[self.index]
        for cam_i, frame in enumerate(self.frames):
            h, w = frame.shape[:2]
            lines = []
            for box in self.boxes.get(cam_i, []):
                cls_idx = CLASS_NAMES.index(box['cls'])
                xc = ((box['x1'] + box['x2']) * 0.5) / float(w)
                yc = ((box['y1'] + box['y2']) * 0.5) / float(h)
                bw = max(1.0, float(box['x2'] - box['x1'])) / float(w)
                bh = max(1.0, float(box['y2'] - box['y1'])) / float(h)
                lines.append('%d %.6f %.6f %.6f %.6f' % (cls_idx, xc, yc, bw, bh))
            path = self._label_path(cam_i, set_id)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                if lines:
                    f.write('\n')
        print('saved labels for %s' % set_id)

    def _add_box(self, cam_i, x, y):
        frame = self.frames[cam_i]
        h, w = frame.shape[:2]
        half = self.box_size_px // 2
        x1 = _clamp(x - half, 0, w - 1)
        y1 = _clamp(y - half, 0, h - 1)
        x2 = _clamp(x + half, 0, w - 1)
        y2 = _clamp(y + half, 0, h - 1)
        self.boxes[cam_i] = [b for b in self.boxes[cam_i] if b.get('cls') != self.current_class]
        self.boxes[cam_i].append({'cls': self.current_class, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    def _undo(self, cam_i):
        if self.boxes.get(cam_i):
            self.boxes[cam_i].pop()

    def _remove_current(self, cam_i):
        self.boxes[cam_i] = [b for b in self.boxes[cam_i] if b.get('cls') != self.current_class]

    def _clear_cam(self, cam_i):
        self.boxes[cam_i] = []

    def _clear_all(self):
        self.boxes = {i: [] for i in range(4)}

    def _draw_tile(self, frame, cam_i):
        view = frame.copy()
        for box in self.boxes.get(cam_i, []):
            color = CLASS_BGR[box['cls']]
            cv2.rectangle(view, (box['x1'], box['y1']), (box['x2'], box['y2']), color, 2)
            cv2.putText(view, box['cls'], (box['x1'], max(16, box['y1'] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        label = self.cam_names[cam_i]
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, 36), (20, 20, 20), -1)
        cv2.putText(view, '%s | class=%s' % (label, self.current_class), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
        return view

    def _build_mosaic(self, target_w=640):
        tiles = []
        self.preview_geom = []
        meta = []
        for cam_i in range(4):
            tile = self._draw_tile(self.frames[cam_i], cam_i)
            sh, sw = tile.shape[:2]
            s = target_w / float(sw)
            dh = max(1, int(round(sh * s)))
            tile = _resize_with_quality(tile, target_w, dh)
            tiles.append(tile)
            meta.append({'src_w': sw, 'src_h': sh, 'draw_w': target_w, 'draw_h': dh})

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

        for cam_i in range(4):
            row = 0 if cam_i < 2 else 1
            col = cam_i % 2
            x0 = col * max_w
            y0 = row * max_h
            self.preview_geom.append({
                'cam_i': cam_i,
                'tile_x0': x0,
                'tile_y0': y0,
                'tile_x1': x0 + max_w,
                'tile_y1': y0 + max_h,
                'draw_w': int(meta[cam_i]['draw_w']),
                'draw_h': int(meta[cam_i]['draw_h']),
                'src_w': int(meta[cam_i]['src_w']),
                'src_h': int(meta[cam_i]['src_h']),
            })

        set_id = self.set_ids[self.index]
        banner = 'set=%s (%d/%d) | s save | ] next(auto-save) | [ prev(auto-save) | 1-4 class | click replace | v drop-color | x clear-cam | X clear-all | a auto-hsv | q quit' % (
            set_id, self.index + 1, len(self.set_ids)
        )
        cv2.rectangle(mosaic, (0, 0), (mosaic.shape[1] - 1, 40), (28, 28, 28), -1)
        cv2.putText(mosaic, banner, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
        return mosaic

    def _mouse_to_src(self, x, y):
        for g in self.preview_geom:
            if not (g['tile_x0'] <= x < g['tile_x1'] and g['tile_y0'] <= y < g['tile_y1']):
                continue
            lx = x - g['tile_x0']
            ly = y - g['tile_y0']
            if lx < 0 or ly < 0 or lx >= g['draw_w'] or ly >= g['draw_h']:
                return g['cam_i'], None, None
            sx = int(round((float(lx) / float(max(1, g['draw_w']))) * float(g['src_w'] - 1)))
            sy = int(round((float(ly) / float(max(1, g['draw_h']))) * float(g['src_h'] - 1)))
            return g['cam_i'], _clamp(sx, 0, g['src_w'] - 1), _clamp(sy, 0, g['src_h'] - 1)
        return None, None, None

    def run(self):
        self._discover()
        self._load_set(0)

        win = 'PRISM Offline 4-Camera Annotator'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, flags, _param):
            cam_i, sx, sy = self._mouse_to_src(x, y)
            if cam_i is None:
                return
            self._hover_cam = cam_i
            if event == cv2.EVENT_LBUTTONDOWN:
                if sx is None or sy is None:
                    return
                self._add_box(cam_i, sx, sy)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._undo(cam_i)

        cv2.setMouseCallback(win, on_mouse)
        print('Offline annotator ready: s save | ] next(auto-save) | [ prev(auto-save) | a auto-hsv | q quit')

        try:
            while True:
                mosaic = self._build_mosaic(target_w=640)
                cv2.imshow(win, mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in CLASS_KEYS:
                    self.current_class = CLASS_KEYS[key]
                elif key in (ord('s'), ord('S'), 10, 13):
                    self._save_set()
                elif key == ord(']'):
                    if self.index + 1 < len(self.set_ids):
                        self._save_set()
                        self._load_set(self.index + 1)
                elif key == ord('['):
                    if self.index - 1 >= 0:
                        self._save_set()
                        self._load_set(self.index - 1)
                elif key in (ord('x'),):
                    self._clear_cam(self._hover_cam)
                elif key in (ord('X'),):
                    self._clear_all()
                elif key in (ord('v'), ord('V')):
                    self._remove_current(self._hover_cam)
                elif key in (ord('a'), ord('A')):
                    if self.auto_from_hsv:
                        self._clear_all()
                        self._auto_label()
        finally:
            cv2.destroyWindow(win)


def parse_args():
    p = argparse.ArgumentParser(description='Offline 4-camera LED annotator')
    p.add_argument('--dataset-root', type=str, default='data/datasets/led_yolo_raw', help='capture-first dataset root')
    p.add_argument('--box-size-px', type=int, default=28)
    p.add_argument('--auto-from-hsv', type=str, default='y', choices=['y', 'n'])
    p.add_argument('--hsv-config', type=str,
                   default=str(ROOT / 'configs' / 'collection' / 'default_online.yaml'))
    return p.parse_args()


def main(argv=None):
    args = parse_args()
    app = OfflineAnnotator(
        dataset_root=args.dataset_root,
        box_size_px=args.box_size_px,
        auto_from_hsv=(args.auto_from_hsv == 'y'),
        hsv_config=args.hsv_config,
    )
    app.run()


if __name__ == '__main__':
    main()