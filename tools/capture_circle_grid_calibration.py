#!/usr/bin/env python3
"""Capture synchronized four-camera images of the CGB-035 circle grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
from collect_ur3_handeye_poses import (  # noqa: E402
    CGB035_COLUMNS,
    CGB035_ROWS,
    circle_grid_preview,
    tile_camera_preview,
)
from prism_charuco_calibration_capture import CalibrationSession  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture synchronized Hik images for CGB-035 camera calibration"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("~/mvs_circle_grid_data"))
    parser.add_argument("--exposure-us", type=int, default=12000)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--frame-rate", type=float, default=15.0)
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--min-valid-cameras", type=int, default=3)
    return parser.parse_args()


def save_capture(session: CalibrationSession, frames: dict[str, np.ndarray], frame_id: int) -> None:
    for serial, image in frames.items():
        camera_index = session._find_cam_index(serial)
        camera_dir = session.session_dir / ("cam%d_%s" % (camera_index, serial))
        camera_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(camera_dir / ("frame_%04d.png" % frame_id)), image)


def main() -> int:
    args = parse_args()
    if not 1 <= args.min_valid_cameras <= 4:
        raise SystemExit("--min-valid-cameras must be between 1 and 4")

    session = CalibrationSession(
        output_dir=args.output_dir,
        squares_x=12,
        squares_y=9,
        square_length_mm=15.0,
        marker_length_mm=11.25,
        aruco_dict="DICT_5X5_1000",
        exposure_us=args.exposure_us,
        gain=args.gain,
        frame_rate=args.frame_rate,
    )
    unused_session_dir = session.session_dir
    session.session_dir = session.output_dir / ("CircleGridCapture_%s" % time.strftime("%Y%m%d_%H%M%S"))
    session.session_dir.mkdir(parents=True, exist_ok=True)
    if unused_session_dir != session.session_dir and unused_session_dir.exists():
        unused_session_dir.rmdir()
    metadata = {
        "target": {"type": "asymmetric_circle_grid", "columns": CGB035_COLUMNS,
                   "rows": CGB035_ROWS, "spacing_mm": 35.0},
        "exposure_us": args.exposure_us,
        "gain": args.gain,
        "frame_rate": args.frame_rate,
    }
    session.initialize_cameras()
    session.start_streaming()
    window_name = "PRISM CGB-035 Camera Calibration | Enter=capture | Q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.preview_width * 2, int(args.preview_width * 0.75) * 2)
    frame_id = 0
    try:
        print("live preview: green VALID means CGB-035 was detected")
        print("Enter/Space=capture synchronized set; q/ESC=finish")
        while True:
            cells = []
            valid_count = 0
            latest = {}
            for cam_i, _, serial in session.cameras:
                thread = session.capture_threads[cam_i]
                frame = thread.get_latest() if thread is not None else None
                if frame is None:
                    blank = np.zeros((360, 480, 3), dtype=np.uint8)
                    cv2.putText(blank, "cam%d %s: NO FRAME" % (cam_i, serial),
                                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cells.append(blank)
                    continue
                latest[serial] = frame.copy()
                overlay, valid = circle_grid_preview(frame, "cam%d %s" % (cam_i, serial))
                valid_count += int(valid)
                cells.append(overlay)
            preview = tile_camera_preview(cells, args.preview_width)
            color = (0, 255, 0) if valid_count >= args.min_valid_cameras else (0, 200, 255)
            cv2.putText(preview, "captured=%d | valid=%d/4 | need>=%d" %
                        (frame_id, valid_count, args.min_valid_cameras),
                        (12, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key not in (10, 13, 32) or valid_count < args.min_valid_cameras:
                continue
            if len(latest) != len(session.cameras):
                print("capture skipped: one or more cameras have no frame")
                continue
            save_capture(session, latest, frame_id)
            frame_id += 1
            print("captured frame set %d (%d/%d cameras valid)" %
                  (frame_id - 1, valid_count, len(session.cameras)))
    finally:
        session.cleanup()
        cv2.destroyAllWindows()
        metadata["num_frames"] = frame_id
        metadata["camera_serials"] = [serial for _, _, serial in session.cameras]
        (session.session_dir / "calibration_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    print("wrote %d synchronized frame sets to %s" % (frame_id, session.session_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())