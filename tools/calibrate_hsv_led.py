#!/usr/bin/env python3
"""
PRISM interactive HSV calibrator (click-to-tune).

Shows a live 3-panel view for one Hik camera:
    [ original | HSV mask | detection overlay ]

You tune each LED colour by CLICKING on the LED in the image: every click
samples the pixel's HSV value and expands that colour's HSV range to cover it
(plus an adjustable margin). Detection of the current colour is drawn live so
you can immediately see the effect.

The camera is driven by its own software trigger, so this tool works standalone
with a single camera (it does not need the full multi-camera trigger rig).

Keys:
    1 / 2 / 3 / 4 : switch active colour (red / yellow / blue / green)
    left click    : sample clicked pixel and grow active colour's HSV range
    right click   : undo the last sample for the active colour
    c             : clear all samples for the active colour (revert to config)
    r             : reset active colour to config default (same as clear)
    w             : save tuned YAML to --output
    p             : print YAML snippet to the terminal
    q / ESC       : quit

Mouse clicks work on any of the three panels (they share image coordinates).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

_mv_candidates = []
if os.environ.get('PRISM_MVIMPORT_DIR'):
    _mv_candidates.append(os.environ['PRISM_MVIMPORT_DIR'])
_mv_candidates.append('/opt/MVS/Samples/64/Python/MvImport')
for _p in _mv_candidates:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism.devices.cameras.mvs_camera import MvCamera, UsbCameraGrabber, enumerate_usb_devices  # noqa: E402
from prism.devices.cameras.highspeed_camera import SoftwareTriggerThread  # noqa: E402
from prism.reconstruction.realtime_reconstruction import build_hsv_mask, detect_led_hsv  # noqa: E402


DEFAULT_CONFIG = os.path.join(ROOT, 'configs', 'collection', 'default_online.yaml')
COLOR_ORDER = ['red', 'yellow', 'blue', 'green']
COLOR_TO_PREFIX = {
    'red': 'r',
    'yellow': 'y',
    'blue': 'b',
    'green': 'g',
}
COLOR_KEY_MAP = {
    ord('1'): 'red',
    ord('2'): 'yellow',
    ord('3'): 'blue',
    ord('4'): 'green',
}
# BGR colours used to label each active colour on screen.
COLOR_BGR = {
    'red': (0, 0, 255),
    'yellow': (0, 220, 255),
    'blue': (255, 120, 0),
    'green': (0, 220, 0),
}


def parse_args():
    p = argparse.ArgumentParser(description='Interactive click-to-tune HSV calibration for PRISM LEDs.')
    p.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                   help='source config yaml to load initial HSV values and camera settings')
    p.add_argument('--camera-index', type=int, default=-1,
                   help='USB camera index to use; -1 means choose interactively')
    p.add_argument('--hik-exposure-us', type=float, default=None)
    p.add_argument('--hik-gain', type=float, default=None)
    p.add_argument('--preview-fps', type=float, default=30.0,
                   help='software-trigger rate driving the live preview')
    p.add_argument('--min-area', type=float, default=None,
                   help='initial min area for blob filtering')
    p.add_argument('--patch', type=int, default=5,
                   help='side length (px) of the neighbourhood averaged on each click')
    p.add_argument('--start-color', type=str, default='yellow', choices=COLOR_ORDER)
    p.add_argument('--output', type=str,
                   default=os.path.join(ROOT, 'configs', 'collection', 'hsv_tuned.yaml'),
                   help='yaml path for saved tuned parameters')
    return p.parse_args()


def load_cfg(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def config_bounds(cfg):
    """Read per-colour HSV bounds from a config dict."""
    bounds = {}
    for color in COLOR_ORDER:
        pfx = COLOR_TO_PREFIX[color]
        bounds[color] = {
            'h_low': int(cfg.get('%s_h_low' % pfx, 0)),
            's_low': int(cfg.get('%s_s_low' % pfx, 0)),
            'v_low': int(cfg.get('%s_v_low' % pfx, 0)),
            'h_high': int(cfg.get('%s_h_high' % pfx, 179)),
            's_high': int(cfg.get('%s_s_high' % pfx, 255)),
            'v_high': int(cfg.get('%s_v_high' % pfx, 255)),
        }
    return bounds


def _clamp(value, lo, hi):
    return max(lo, min(hi, int(round(value))))


def _hue_range_from_samples(hues, margin):
    """Return (h_low, h_high) covering all sampled hues on the 0..179 circle.

    If the tightest covering arc crosses the 0/179 boundary (typical for red),
    the returned range has h_low > h_high, which build_hsv_mask/detect_led_hsv
    interpret as a wrap-around range.
    """
    hs = sorted(int(h) % 180 for h in hues)
    if len(hs) == 1:
        h_low = hs[0] - margin
        h_high = hs[0] + margin
    else:
        # Find the largest empty gap on the hue circle; the covering arc is its
        # complement.
        largest_gap = -1
        gap_at = 0
        for i in range(len(hs)):
            nxt = hs[(i + 1) % len(hs)]
            gap = (nxt - hs[i]) % 180
            if gap > largest_gap:
                largest_gap = gap
                gap_at = i
        # Arc runs from the point after the largest gap, forward to the point
        # before it (wrapping through 0 as needed).
        h_low = hs[(gap_at + 1) % len(hs)] - margin
        h_high = hs[gap_at] + margin

    h_low %= 180
    h_high %= 180
    return int(h_low), int(h_high)


def bounds_from_samples(samples, hm, sm, vm):
    """Compute HSV bounds covering all clicked samples plus per-channel margins."""
    hues = [h for h, _s, _v in samples]
    svals = [s for _h, s, _v in samples]
    vvals = [v for _h, _s, v in samples]

    h_low, h_high = _hue_range_from_samples(hues, hm)
    s_low = _clamp(min(svals) - sm, 0, 255)
    s_high = _clamp(max(svals) + sm, 0, 255)
    v_low = _clamp(min(vvals) - vm, 0, 255)
    v_high = _clamp(max(vvals) + vm, 0, 255)
    return {
        'h_low': h_low, 's_low': s_low, 'v_low': v_low,
        'h_high': h_high, 's_high': s_high, 'v_high': v_high,
    }


def make_yaml_payload(bounds_by_color, min_area):
    out = {'min_area': float(min_area)}
    for color in COLOR_ORDER:
        pfx = COLOR_TO_PREFIX[color]
        b = bounds_by_color[color]
        out['%s_h_low' % pfx] = int(b['h_low'])
        out['%s_s_low' % pfx] = int(b['s_low'])
        out['%s_v_low' % pfx] = int(b['v_low'])
        out['%s_h_high' % pfx] = int(b['h_high'])
        out['%s_s_high' % pfx] = int(b['s_high'])
        out['%s_v_high' % pfx] = int(b['v_high'])
    return out


def save_yaml(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=False, sort_keys=False)


def _noop(_value):
    return None


def pick_camera_index(all_devs, chosen):
    print('Found %d USB cameras:' % len(all_devs))
    for i, _, model, serial in all_devs:
        print('  [%d] model=%s serial=%s' % (i, model, serial))
    if chosen >= 0:
        if chosen >= len(all_devs):
            raise RuntimeError('camera index out of range: %d' % chosen)
        return chosen
    raw = input('camera index for HSV tuning (default 0): ').strip()
    if raw == '':
        return 0
    idx = int(raw)
    if idx < 0 or idx >= len(all_devs):
        raise RuntimeError('camera index out of range: %d' % idx)
    return idx


class HsvCalibrator:
    """Holds tuning state and handles mouse clicks that sample HSV values."""

    def __init__(self, cfg, active, min_area, patch):
        self.config_bounds = config_bounds(cfg)
        self.samples = {color: [] for color in COLOR_ORDER}
        self.active = active
        self.min_area = float(min_area)
        self.patch = max(1, int(patch))
        # Latest HSV frame + its size, updated every loop for click sampling.
        self.cur_hsv = None
        self.frame_w = 0
        self.frame_h = 0
        # Margins (updated from trackbars each loop).
        self.hm = 8
        self.sm = 40
        self.vm = 40
        self.last_click = None  # (x_img, y) for on-screen marker

    def effective_bounds(self, color):
        if self.samples[color]:
            return bounds_from_samples(self.samples[color], self.hm, self.sm, self.vm)
        return dict(self.config_bounds[color])

    def all_bounds(self):
        return {color: self.effective_bounds(color) for color in COLOR_ORDER}

    def _sample_hsv_at(self, x_img, y):
        half = self.patch // 2
        x0 = _clamp(x_img - half, 0, self.frame_w - 1)
        x1 = _clamp(x_img + half, 0, self.frame_w - 1)
        y0 = _clamp(y - half, 0, self.frame_h - 1)
        y1 = _clamp(y + half, 0, self.frame_h - 1)
        patch = self.cur_hsv[y0:y1 + 1, x0:x1 + 1, :].reshape(-1, 3)
        h = int(np.median(patch[:, 0]))
        s = int(np.median(patch[:, 1]))
        v = int(np.median(patch[:, 2]))
        return h, s, v

    def on_mouse(self, event, x, y, _flags, _param):
        if self.cur_hsv is None or self.frame_w == 0:
            return
        # Panels are stacked horizontally; map click back to image coordinates.
        x_img = x % self.frame_w
        if x_img >= self.frame_w or y >= self.frame_h:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hsv = self._sample_hsv_at(x_img, y)
            self.samples[self.active].append(hsv)
            self.last_click = (x_img, y)
            print('sampled %s HSV=%s (samples=%d)'
                  % (self.active, hsv, len(self.samples[self.active])))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.samples[self.active]:
                removed = self.samples[self.active].pop()
                print('undo %s sample %s (remaining=%d)'
                      % (self.active, removed, len(self.samples[self.active])))

    def clear_active(self):
        self.samples[self.active] = []
        print('cleared samples for %s (reverted to config)' % self.active)


def _draw_hud(overlay, calib, bounds, area, pt):
    active = calib.active
    label_color = COLOR_BGR.get(active, (255, 255, 255))
    cv2.putText(overlay, 'color=%s  samples=%d  min_area=%.0f' % (
        active, len(calib.samples[active]), calib.min_area),
        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)
    cv2.putText(overlay, 'H[%d,%d] S[%d,%d] V[%d,%d]' % (
        bounds['h_low'], bounds['h_high'], bounds['s_low'],
        bounds['s_high'], bounds['v_low'], bounds['v_high']),
        (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if pt is not None:
        cv2.circle(overlay, (int(round(pt[0])), int(round(pt[1]))), 9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, 'det area=%.1f' % area, (16, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, 'no detection', (16, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, 'click LED to tune | 1-4 colour | right-click undo | w save | q quit',
                (16, overlay.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    exposure_us = args.hik_exposure_us if args.hik_exposure_us is not None else float(cfg.get('hik_exposure_us', 3000.0))
    gain = args.hik_gain if args.hik_gain is not None else float(cfg.get('hik_gain', 0.0))
    min_area = args.min_area if args.min_area is not None else float(cfg.get('min_area', 10.0))

    calib = HsvCalibrator(cfg, args.start_color, min_area, args.patch)

    MvCamera.MV_CC_Initialize()
    grabber = None
    trigger = None
    try:
        devs = enumerate_usb_devices()
        if not devs:
            raise RuntimeError('no Hik USB cameras found')

        cam_idx = pick_camera_index(devs, args.camera_index)
        _, dev_info, model, serial = devs[cam_idx]
        serial_safe = serial if serial else ('cam%d' % cam_idx)
        print('using camera index=%d serial=%s model=%s' % (cam_idx, serial_safe, model))

        grabber = UsbCameraGrabber(dev_info, serial_safe, model)
        # Software trigger so this single camera streams standalone.
        grabber.open_and_prepare(
            exposure_us=exposure_us,
            gain=gain,
            frame_rate=args.preview_fps,
            trigger_source='Software',
        )
        trigger = SoftwareTriggerThread(grabber, fps=max(1.0, args.preview_fps))
        trigger.start()
        print('software trigger started at %.1f fps' % args.preview_fps)

        win = 'PRISM HSV Calibrator (click LED to tune)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1800, 720)
        cv2.createTrackbar('H_margin', win, calib.hm, 40, _noop)
        cv2.createTrackbar('S_margin', win, calib.sm, 128, _noop)
        cv2.createTrackbar('V_margin', win, calib.vm, 128, _noop)
        cv2.createTrackbar('MinArea', win, max(1, min(1000, int(round(min_area)))), 1000, _noop)
        cv2.setMouseCallback(win, calib.on_mouse)

        print(__doc__)

        last_print = 0.0
        while True:
            try:
                img_bgr, frame_num = grabber.grab_one_bgr(timeout_ms=1500)
            except Exception as exc:
                # Show a placeholder so the window stays responsive if a frame drops.
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'waiting for frames... (%s)' % exc, (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, blank)
                if (cv2.waitKey(30) & 0xFF) in (27, ord('q')):
                    break
                continue

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            calib.cur_hsv = hsv
            calib.frame_h, calib.frame_w = hsv.shape[:2]

            calib.hm = cv2.getTrackbarPos('H_margin', win)
            calib.sm = cv2.getTrackbarPos('S_margin', win)
            calib.vm = cv2.getTrackbarPos('V_margin', win)
            calib.min_area = float(cv2.getTrackbarPos('MinArea', win))

            bounds = calib.effective_bounds(calib.active)
            low = (bounds['h_low'], bounds['s_low'], bounds['v_low'])
            high = (bounds['h_high'], bounds['s_high'], bounds['v_high'])

            mask = build_hsv_mask(hsv, low, high)
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            pt, area = detect_led_hsv(img_bgr, low, high, calib.min_area)
            overlay = img_bgr.copy()
            if calib.last_click is not None:
                cx, cy = calib.last_click
                cv2.drawMarker(overlay, (cx, cy), COLOR_BGR.get(calib.active, (255, 255, 255)),
                               cv2.MARKER_CROSS, 18, 2)
            _draw_hud(overlay, calib, bounds, area, pt)

            board = np.hstack([img_bgr, mask_vis, overlay])
            cv2.imshow(win, board)

            now = time.time()
            if now - last_print > 8.0:
                print('active=%s H[%d,%d] S[%d,%d] V[%d,%d] min_area=%.0f samples=%d' % (
                    calib.active, low[0], high[0], low[1], high[1], low[2], high[2],
                    calib.min_area, len(calib.samples[calib.active])))
                last_print = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in COLOR_KEY_MAP:
                calib.active = COLOR_KEY_MAP[key]
                calib.last_click = None
            elif key in (ord('c'), ord('r')):
                calib.clear_active()
                calib.last_click = None
            elif key == ord('p'):
                payload = make_yaml_payload(calib.all_bounds(), calib.min_area)
                print(yaml.safe_dump(payload, allow_unicode=False, sort_keys=False))
            elif key == ord('w'):
                payload = make_yaml_payload(calib.all_bounds(), calib.min_area)
                save_yaml(args.output, payload)
                print('saved tuned hsv config to %s' % args.output)

    finally:
        if trigger is not None:
            try:
                trigger.stop()
                trigger.join(timeout=1.0)
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if grabber is not None:
            try:
                grabber.stop_and_close()
            except Exception:
                pass
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass


if __name__ == '__main__':
    main()
