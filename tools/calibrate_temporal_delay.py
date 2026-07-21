#!/usr/bin/env python3
"""
Standalone temporal delay calibration tool for HIK + RealSense cameras.

Opens the cameras, shows the AprilTag fullscreen on this monitor, detects the
tag appearance in each stream, and saves per-camera temporal offsets.

This is the standalone version for use OUTSIDE a recording session.
When running prism-collect, temporal calibration is built-in -- see
--temporal-calib / --skip-temporal-calib.

Usage
-----
    python tools/calibrate_temporal_delay.py \\
        --hik-serials 00DA2318497 00DA2318498 00DA2318499 00DA2318500 \\
        --rs-serial 234522070717 \\
        --fps 30 --exposure-us 8000 --gain 10 \\
        --tag-family tag36h11 --tag-id 0 \\
        --output configs/devices/camera_delay_calib.json

Output JSON
-----------
    aligned_time = capture_wall_time - offsets_s[serial]

NOTE: HIK cameras are free-running; offsets change every power-cycle.
Re-run at the start of each recording session, or use the built-in
prism-collect integration which does this automatically.
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from prism.recording.temporal_calibration import (  # noqa: E402
    FAMILY_TO_DICT,
    CalibSink,
    build_detector,
    generate_tag_image,
    show_tag_on_screen,
    find_tag_appearance,
)


# ─────────────────────── Per-camera capture threads ──────────────────────────

class _HikBufferThread(threading.Thread):
    def __init__(self, grabber, serial, timeout_ms=300):
        super().__init__(daemon=True)
        self.grabber = grabber
        self.serial = serial
        self.timeout_ms = int(timeout_ms)
        self.sink = CalibSink()
        self.stop_flag = threading.Event()
        self.error = None

    def run(self):
        while not self.stop_flag.is_set():
            try:
                img_rgb, _fnum = self.grabber.grab_one_rgb(timeout_ms=self.timeout_ms)
            except Exception as exc:
                if not self.stop_flag.is_set():
                    self.error = exc
                break
            frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self.sink.submit(frame_bgr, time.time())

    def stop(self):
        self.stop_flag.set()


class _RsBufferThread(threading.Thread):
    def __init__(self, rs_grabber, timeout_ms=1000):
        super().__init__(daemon=True)
        self.rs_grabber = rs_grabber
        self.timeout_ms = int(timeout_ms)
        self.sink = CalibSink()
        self.stop_flag = threading.Event()
        self.error = None

    def run(self):
        while not self.stop_flag.is_set():
            try:
                img_bgr = self.rs_grabber.grab_color_bgr(timeout_ms=self.timeout_ms)
            except Exception as exc:
                if not self.stop_flag.is_set():
                    self.error = exc
                break
            if img_bgr is None:
                continue
            self.sink.submit(img_bgr, time.time())

    def stop(self):
        self.stop_flag.set()


# ────────────────────────────── Camera helpers ───────────────────────────────

def _open_hik(serial, fps, exposure_us, gain):
    from prism.devices.cameras.mvs_camera import UsbCameraGrabber, enumerate_usb_devices
    for _idx, dev_info, model, dev_serial in enumerate_usb_devices():
        if dev_serial == serial:
            g = UsbCameraGrabber(dev_info, dev_serial, model)
            g.open_and_prepare(use_hardware_trigger=False, use_software_trigger=False,
                               exposure_us=exposure_us, gain=gain, frame_rate=fps)
            print(f"  [{serial}] opened  model={model}")
            return g
    from prism.devices.cameras.mvs_camera import enumerate_usb_devices as _e
    raise RuntimeError("HIK camera not found: %r\nAvailable: %s"
                       % (serial, [s for _, _, _, s in _e()]))


# ────────────────────────────────── CLI ──────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hik-serials", nargs="+", metavar="SERIAL", required=True)
    p.add_argument("--rs-serial", default=None, metavar="SERIAL",
                   help="RealSense serial; omit to skip RealSense")

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--exposure-us", type=float, default=8000.0)
    p.add_argument("--gain", type=float, default=10.0)

    p.add_argument("--tag-family", default="tag36h11", choices=list(FAMILY_TO_DICT))
    p.add_argument("--tag-id", type=int, default=0,
                   help="Tag ID to detect; -1 = any (default: 0)")
    p.add_argument("--tag-delay", type=float, default=4.0,
                   help="Seconds before tag appears on screen (default: 4)")
    p.add_argument("--tag-display-s", type=float, default=1.5,
                   help="Seconds to keep tag visible (default: 1.5)")
    p.add_argument("--min-consecutive", type=int, default=2)
    p.add_argument("--duration", type=float, default=None,
                   help="Total duration; default = tag-delay + tag-display-s + 1.5")

    p.add_argument("--reference-serial", default=None)
    p.add_argument("--output", default="configs/devices/camera_delay_calib.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────── Main ───────────────────────────────────

def main():
    args = _parse_args()
    os.chdir(_REPO_ROOT)

    total = args.duration or (args.tag_delay + args.tag_display_s + 1.5)
    detect_fn = build_detector(args.tag_family)
    tag_img = generate_tag_image(args.tag_family, max(0, args.tag_id))

    print("=" * 60)
    print("  Temporal delay calibration (screen-flash / AprilTag)")
    print("=" * 60)
    print(f"  HIK serials  : {args.hik_serials}")
    print(f"  RS serial    : {args.rs_serial or '(not used)'}")
    print(f"  Tag          : {args.tag_family}  id={args.tag_id}")
    print(f"  Tag appears  : +{args.tag_delay:.0f}s  visible={args.tag_display_s:.1f}s")
    print(f"  Total        : {total:.1f}s")
    print()

    # Open cameras
    from prism.devices.cameras.mvs_camera import MvCamera
    MvCamera.MV_CC_Initialize()

    hik_grabbers = {}
    for serial in args.hik_serials:
        print(f"Opening HIK {serial} ...")
        hik_grabbers[serial] = _open_hik(serial, args.fps, args.exposure_us, args.gain)

    rs_grabber = None
    if args.rs_serial:
        from prism.devices.cameras.realsense_camera import RealSenseColorGrabber
        print(f"Opening RealSense {args.rs_serial} ...")
        rs_grabber = RealSenseColorGrabber(serial=args.rs_serial, fps=30)
        rs_grabber.open_and_prepare()
        print(f"  [{args.rs_serial}] opened")

    hik_threads = {s: _HikBufferThread(g, s) for s, g in hik_grabbers.items()}
    rs_thread = _RsBufferThread(rs_grabber) if rs_grabber else None

    print()
    for t in hik_threads.values():
        t.start()
    if rs_thread:
        rs_thread.start()

    print("Recording started.  Point ALL cameras at this screen.")
    print(f"AprilTag appears in ~{args.tag_delay:.0f}s -- screen goes black when done.")
    print()

    t_start = time.time()
    while time.time() - t_start < args.tag_delay:
        remaining = args.tag_delay - (time.time() - t_start)
        print(f"\r  Tag in {remaining:.1f}s ...", end="", flush=True)
        time.sleep(0.05)
    print()

    print("  Showing AprilTag ...")
    show_tag_on_screen(tag_img, display_s=args.tag_display_s)
    print("  Tag hidden.")

    tail = max(0.5, total - (time.time() - t_start))
    time.sleep(tail)

    for t in hik_threads.values():
        t.stop()
    if rs_thread:
        rs_thread.stop()
    for t in hik_threads.values():
        t.join(timeout=3.0)
    if rs_thread:
        rs_thread.join(timeout=3.0)
    for g in hik_grabbers.values():
        g.stop_and_close()
    if rs_grabber:
        rs_grabber.stop_and_close()

    # Detect
    print()
    all_serials = list(args.hik_serials) + ([args.rs_serial] if args.rs_serial else [])
    all_threads = list(hik_threads.values()) + ([rs_thread] if rs_thread else [])

    event_times = {}
    for serial, thread in zip(all_serials, all_threads):
        frames = thread.sink.get_frames()
        n = len(frames)
        fps_actual = (n - 1) / (frames[-1][0] - frames[0][0]) if n > 1 else 0.0
        event_t, event_idx = find_tag_appearance(
            frames, detect_fn, args.tag_id, args.min_consecutive
        )
        if event_t is None:
            print(f"  [{serial}]  {n} frames @ {fps_actual:.1f} fps  -- tag NOT detected")
        else:
            print(f"  [{serial}]  {n} frames @ {fps_actual:.1f} fps  -- tag at frame {event_idx}")
            event_times[serial] = event_t

    if len(event_times) < 2:
        print("\nERROR: tag detected in fewer than 2 cameras.")
        sys.exit(1)

    ref = args.reference_serial or args.hik_serials[0]
    if ref not in event_times:
        ref = next(iter(event_times))

    offsets = {s: round(t - event_times[ref], 6) for s, t in event_times.items()}

    print()
    print(f"Reference: {ref}")
    for s, off in offsets.items():
        label = "  <-- reference" if s == ref else ""
        print(f"  {s}:  {off * 1000:+8.3f} ms{label}")
    print(f"\nUncertainty: +/-{500.0 / args.fps:.0f} ms at {args.fps:.0f} fps")

    calib = {
        "reference_serial": ref,
        "offsets_s": offsets,
        "fps_at_calibration": args.fps,
        "tag_family": args.tag_family,
        "tag_id": args.tag_id,
        "note": "aligned_time = capture_wall_time - offsets_s[serial]",
    }

    if not args.dry_run:
        out = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        print(f"Saved: {out}")
    else:
        print("[dry-run]", json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
