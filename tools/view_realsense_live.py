#!/usr/bin/env python3
"""Live color preview for a RealSense camera used by PRISM.

Keys:
    S       save the current displayed frame
    F       toggle fullscreen
    Q/Esc   quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prism.devices.cameras.realsense_camera import (  # noqa: E402
    RealSenseColorGrabber,
    build_undistort_maps,
    load_rs_intrinsics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="RealSense serial; default uses the first device")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--exposure", type=float, default=260.0)
    parser.add_argument("--gain", type=float, default=64.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--calib-json", type=Path,
                        default=Path("configs/devices/d435_charuco_intrinsics.json"))
    parser.add_argument("--undistort", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/realsense_snapshots"))
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height and fps must be positive")

    calibration_path = args.calib_json.expanduser().resolve()
    undistort_maps = None
    if args.undistort:
        if not calibration_path.is_file():
            raise SystemExit("RealSense calibration not found: %s" % calibration_path)
        intrinsic, distortion, calibrated_size, _ = load_rs_intrinsics(str(calibration_path))
        if calibrated_size is not None and calibrated_size != (args.width, args.height):
            raise SystemExit(
                "calibration size %s does not match requested stream %dx%d"
                % (calibrated_size, args.width, args.height)
            )
        map1, map2, _ = build_undistort_maps(
            intrinsic, distortion, args.width, args.height
        )
        undistort_maps = (map1, map2)

    grabber = RealSenseColorGrabber(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        auto_exposure=args.auto_exposure,
        exposure=None if args.auto_exposure else args.exposure,
        gain=None if args.auto_exposure else args.gain,
        brightness=args.brightness,
    )
    window = "PRISM RealSense Live | S=save F=fullscreen Q=quit"
    output_dir = args.output_dir.expanduser().resolve()
    fullscreen = False
    frame_count = 0
    fps_value = 0.0
    fps_start = time.monotonic()

    try:
        readback = grabber.open_and_prepare()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        window_height = max(480, int(args.window_width * args.height / float(args.width)))
        cv2.resizeWindow(window, max(640, args.window_width), window_height)
        print("RealSense live preview started: %dx%d @ %d FPS" % (
            args.width, args.height, args.fps
        ))
        print("settings: %s" % readback)

        while True:
            frame = grabber.grab_color_bgr(timeout_ms=args.timeout_ms)
            if frame is None:
                continue
            if undistort_maps is not None:
                frame = cv2.remap(
                    frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR
                )

            frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_start
            if elapsed >= 0.5:
                fps_value = frame_count / elapsed
                frame_count = 0
                fps_start = now

            shown = frame.copy()
            status = "%dx%d | %.1f FPS | %s exposure" % (
                shown.shape[1], shown.shape[0], fps_value,
                "auto" if args.auto_exposure else "manual",
            )
            cv2.putText(shown, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(shown, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (80, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / ("realsense_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
                cv2.imwrite(str(output), frame)
                print("saved %s" % output)
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, mode)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        grabber.stop_and_close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())