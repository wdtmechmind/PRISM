#!/usr/bin/env python3
"""Move a UR3 CB3 through hand-eye poses and record actual TCP poses.

The robot pose format is UR's Cartesian pose:
[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad].
The calibration script expects the resulting CSV columns ``x_m/y_m/z_m`` and
``rx_rad/ry_rad/rz_rad`` as ``base_from_tcp``.

Waypoint JSON format::

    {
      "poses": [
        {"frame": "frame_0000", "pose": [0.35, -0.20, 0.30, 2.20, 2.20, 0.0]},
        {"frame": "frame_0001", "pose": [0.40, -0.15, 0.35, 2.10, 2.25, 0.20]}
      ]
    }

By default this script only validates and prints the waypoints. Real motion
requires ``--execute`` and ``ur_rtde``. At each stopped pose, capture the cam0
image using your camera capture process, then press Enter to continue.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import socket
import time
from pathlib import Path

import cv2
import numpy as np


_CIRCLE_GRID_BLOB_DETECTOR = None
CGB035_COLUMNS = 4
CGB035_ROWS = 5
CGB035_PATTERN_CANDIDATES = (
    (4, 5, True),
    (5, 4, True),
    (4, 5, False),
    (5, 4, False),
)


def circle_grid_blob_detector():
    """Detect the white blobs used by the CGB-035 board on a dark plate."""
    global _CIRCLE_GRID_BLOB_DETECTOR
    if _CIRCLE_GRID_BLOB_DETECTOR is None:
        params = cv2.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = 5.0
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 150.0
        params.maxArea = 5000.0
        params.filterByCircularity = True
        params.minCircularity = 0.65
        params.filterByConvexity = True
        params.minConvexity = 0.80
        params.filterByInertia = True
        params.minInertiaRatio = 0.50
        _CIRCLE_GRID_BLOB_DETECTOR = cv2.SimpleBlobDetector_create(params)
    return _CIRCLE_GRID_BLOB_DETECTOR


def find_cgb035_grid(gray: np.ndarray):
    """Find CGB-035 with polarity, preprocessing, and orientation fallbacks."""
    detector = circle_grid_blob_detector()
    blob_count = len(detector.detect(gray))
    images = (gray, cv2.equalizeHist(gray))
    for image in images:
        for columns, rows, asymmetric in CGB035_PATTERN_CANDIDATES:
            pattern_flag = cv2.CALIB_CB_ASYMMETRIC_GRID if asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID
            for flags in (pattern_flag, pattern_flag | cv2.CALIB_CB_CLUSTERING):
                found, centers = cv2.findCirclesGrid(
                    image, (columns, rows), flags=flags, blobDetector=detector
                )
                if found:
                    return found, centers, (columns, rows, asymmetric), blob_count
    return False, None, None, blob_count


def load_waypoints(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    poses = data.get("poses", data) if isinstance(data, dict) else data
    if not isinstance(poses, list) or not poses:
        raise ValueError("waypoint JSON must contain a non-empty 'poses' list")
    output = []
    for index, item in enumerate(poses):
        if not isinstance(item, dict) or "pose" not in item:
            raise ValueError("waypoint %d must contain a pose list" % index)
        pose = np.asarray(item["pose"], dtype=np.float64).reshape(-1)
        if pose.size != 6 or not np.isfinite(pose).all():
            raise ValueError("waypoint %d pose must contain six finite numbers" % index)
        frame = str(item.get("frame", "frame_%04d" % index))
        output.append({"frame": frame, "pose": pose.tolist()})
    return output


def validate_waypoints(waypoints: list[dict]) -> None:
    positions = np.asarray([item["pose"][:3] for item in waypoints], dtype=np.float64)
    if len(positions) >= 2 and float(np.ptp(positions, axis=0).max()) < 1e-3:
        print("warning: waypoints have almost no translation variation")
    rotations = np.asarray([item["pose"][3:] for item in waypoints], dtype=np.float64)
    if len(rotations) >= 2 and float(np.ptp(rotations, axis=0).max()) < 0.05:
        print("warning: waypoints have almost no rotation variation")


def write_pose_row(writer, frame: str, pose: list[float], timestamp: float) -> None:
    writer.writerow([frame, "%.9f" % pose[0], "%.9f" % pose[1], "%.9f" % pose[2],
                     "%.9f" % pose[3], "%.9f" % pose[4], "%.9f" % pose[5],
                     "%.6f" % timestamp])


def read_robot_state(receive) -> dict[str, object]:
    state = {}
    for method_name in ("isProtectiveStopped", "isEmergencyStopped", "getRobotMode", "getRobotStatus"):
        method = getattr(receive, method_name, None)
        if method is not None:
            try:
                state[method_name] = method()
            except Exception as exc:
                state[method_name] = "unavailable: %s" % exc
    return state


def check_rtde_port(robot_ip: str, timeout_s: float = 3.0) -> None:
    try:
        with socket.create_connection((robot_ip, 30004), timeout=timeout_s):
            return
    except OSError as exc:
        raise RuntimeError(
            "cannot reach UR RTDE port 30004 at %s: %s. "
            "Check the robot IP, Ethernet connection, and that the controller is running."
            % (robot_ip, exc)
        ) from exc


def circle_grid_preview(frame: np.ndarray, camera_label: str) -> tuple[np.ndarray, bool]:
    """Draw CGB-035 asymmetric-grid validity on one live camera frame."""
    overlay = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, centers, pattern, blob_count = find_cgb035_grid(gray)
    if found and centers is not None:
        cv2.drawChessboardCorners(overlay, (CGB035_COLUMNS, CGB035_ROWS), centers, found)
    color = (0, 220, 0) if found else (0, 0, 255)
    if found:
        columns, rows, asymmetric = pattern
        status = "VALID %dx%d %s" % (columns, rows, "ASYM" if asymmetric else "SYM")
    else:
        status = "INVALID blobs=%d" % blob_count
    cv2.putText(overlay, "%s: %s" % (camera_label, status), (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return overlay, bool(found)


def tile_camera_preview(cells: list[np.ndarray], preview_width: int) -> np.ndarray:
    """Resize four camera cells and arrange them in a 2x2 preview."""
    normalized = []
    for cell in cells:
        height, width = cell.shape[:2]
        scale = preview_width / float(width)
        normalized.append(cv2.resize(cell, (preview_width, max(1, int(height * scale)))))
    cell_height = max(cell.shape[0] for cell in normalized)
    padded = []
    for cell in normalized:
        if cell.shape[0] < cell_height:
            cell = cv2.copyMakeBorder(
                cell, 0, cell_height - cell.shape[0], 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        padded.append(cell)
    while len(padded) < 4:
        padded.append(np.zeros_like(padded[0]))
    return np.vstack((np.hstack((padded[0], padded[1])),
                      np.hstack((padded[2], padded[3]))))


def run_freedrive(args, rtde_control, rtde_receive) -> int:
    """Capture synchronized Hik frames and the current TCP on each Enter."""
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from prism_charuco_calibration_capture import CalibrationSession
    except Exception as exc:
        raise SystemExit(
            "freedrive mode requires the PRISM MVS camera dependencies: %s" % exc
        ) from exc

    print("checking RTDE connection to %s:30004 ..." % args.robot_ip, flush=True)
    check_rtde_port(args.robot_ip)
    print("connecting RTDE control interface ...", flush=True)
    control = rtde_control.RTDEControlInterface(args.robot_ip)
    print("connecting RTDE receive interface ...", flush=True)
    receive = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    session = CalibrationSession(
        output_dir=args.camera_output_dir,
        squares_x=12,
        squares_y=9,
        square_length_mm=15.0,
        marker_length_mm=11.25,
        aruco_dict="DICT_5X5_1000",
        exposure_us=args.camera_exposure_us,
        gain=args.camera_gain,
        frame_rate=args.camera_frame_rate,
    )
    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    teach_mode_started = False
    try:
        session.initialize_cameras()
        session.start_streaming()
        print("starting UR3 freedrive mode; physically guide the robot to each pose")
        control.teachMode()
        teach_mode_started = True
        window_name = "PRISM UR3 Hand-Eye | Enter=capture | Q=quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.preview_width * 2, int(args.preview_width * 0.82) * 2)
        print("live preview started: Enter=capture image+TCP, q/ESC=quit")
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "frame", "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad",
                "timestamp_unix",
            ])
            while True:
                cells = []
                valid_count = 0
                for cam_i, _, serial in session.cameras:
                    thread = session.capture_threads[cam_i]
                    frame = thread.get_latest() if thread is not None else None
                    if frame is None:
                        blank = np.zeros((360, 480, 3), dtype=np.uint8)
                        cv2.putText(blank, "cam%d %s: NO FRAME" % (cam_i, serial),
                                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cells.append(blank)
                        continue
                    overlay, valid = circle_grid_preview(frame, "cam%d %s" % (cam_i, serial))
                    valid_count += int(valid)
                    cells.append(overlay)

                if cells:
                    preview = tile_camera_preview(cells, args.preview_width)
                    cv2.putText(preview, "valid=%d/%d | Enter=capture | Q=quit" %
                                (valid_count, len(cells)),
                                (12, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key not in (10, 13):
                    continue

                frame_name = "frame_%04d" % session.frame_count
                actual_pose = [float(value) for value in receive.getActualTCPPose()]
                captured = session.capture_current()
                if not captured:
                    print("capture failed; TCP pose was not written for %s" % frame_name)
                    continue
                write_pose_row(writer, frame_name, actual_pose, time.time())
                handle.flush()
                print("saved %s: valid cameras=%d/%d, TCP=%s" % (
                    frame_name, valid_count, len(cells),
                    ["%.6f" % value for value in actual_pose]
                ))
        cv2.destroyWindow(window_name)
        print("wrote %s" % output_csv)
        return 0
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if teach_mode_started:
            try:
                control.endTeachMode()
            except Exception as exc:
                print("cleanup endTeachMode skipped: %s" % exc)
        try:
            session.cleanup()
        finally:
            try:
                control.stopScript()
            except Exception as exc:
                print("cleanup stopScript skipped: %s" % exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True, help="UR3 controller IP")
    parser.add_argument("--waypoints", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--speed", type=float, default=0.10, help="linear speed in m/s")
    parser.add_argument("--acceleration", type=float, default=0.30, help="linear acceleration in m/s^2")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--position-tolerance", type=float, default=0.005,
                        help="abort if actual TCP position misses target by this many metres")
    parser.add_argument("--execute", action="store_true", help="connect and move the real robot")
    parser.add_argument("--freedrive", action="store_true",
                        help="guide the robot by hand; Enter captures four cameras and TCP")
    parser.add_argument("--camera-output-dir", type=Path, default=Path("/home/daotan/mvs_charuco_data"))
    parser.add_argument("--camera-exposure-us", type=float, default=12000.0)
    parser.add_argument("--camera-gain", type=float, default=0.0)
    parser.add_argument("--camera-frame-rate", type=float, default=30.0)
    parser.add_argument("--preview-width", type=int, default=640,
                        help="width of each live camera preview cell in pixels")
    parser.add_argument("--no-prompt", action="store_true", help="do not wait for image capture confirmation")
    args = parser.parse_args()

    if args.freedrive and args.waypoints is not None:
        parser.error("--freedrive does not use --waypoints")
    if not args.freedrive and args.waypoints is None:
        parser.error("--waypoints is required unless --freedrive is used")

    if args.speed <= 0.0 or args.acceleration <= 0.0 or args.settle_time < 0.0 or args.position_tolerance <= 0.0:
        parser.error("speed, acceleration, settle-time, and position-tolerance must be positive/non-negative")
    try:
        import rtde_control
        import rtde_receive
    except ImportError as exc:
        raise SystemExit(
            "--execute/--freedrive requires ur_rtde; install it in the active robot-control environment: "
            "pip install ur-rtde"
        ) from exc

    if args.freedrive:
        return run_freedrive(args, rtde_control, rtde_receive)

    waypoints = load_waypoints(args.waypoints.expanduser().resolve())
    validate_waypoints(waypoints)

    print("Loaded %d UR3 waypoints" % len(waypoints))
    for item in waypoints:
        print("  %-16s %s" % (item["frame"], ["%.5f" % value for value in item["pose"]]))
    if not args.execute:
        print("Preview only: no robot connection and no movement. Add --execute to run.")
        return 0

    print("checking RTDE connection to %s:30004 ..." % args.robot_ip, flush=True)
    check_rtde_port(args.robot_ip)
    print("connecting RTDE control interface ...", flush=True)
    control = rtde_control.RTDEControlInterface(args.robot_ip)
    print("connecting RTDE receive interface ...", flush=True)
    receive = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    print("RTDE connected; robot motion will start next.", flush=True)
    args.output_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.expanduser().resolve().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad", "timestamp_unix"])
        try:
            for index, item in enumerate(waypoints):
                frame = item["frame"]
                print("\n[%d/%d] moving to %s" % (index + 1, len(waypoints), frame), flush=True)
                target_pose = np.asarray(item["pose"], dtype=np.float64)
                move_result = control.moveL(item["pose"], args.speed, args.acceleration)
                time.sleep(args.settle_time)
                actual_pose = [float(value) for value in receive.getActualTCPPose()]
                print("actual TCP: %s" % ["%.6f" % value for value in actual_pose])
                position_error = float(np.linalg.norm(
                    np.asarray(actual_pose[:3], dtype=np.float64) - target_pose[:3]
                ))
                state = read_robot_state(receive)
                if move_result is False or position_error > args.position_tolerance:
                    raise RuntimeError(
                        "robot did not reach %s: position error=%.3f mm, state=%s. "
                        "Check protective stop, waypoint reachability, TCP, and collision."
                        % (frame, position_error * 1000.0, state)
                    )
                if state.get("isProtectiveStopped") is True or state.get("isEmergencyStopped") is True:
                    raise RuntimeError("robot stopped before %s: state=%s" % (frame, state))
                if not args.no_prompt:
                    input("Capture the cam0 image for %s, then press Enter... " % frame)
                write_pose_row(writer, frame, actual_pose, time.time())
                handle.flush()
        finally:
            try:
                control.stopL(1.0)
            except Exception as exc:
                print("cleanup stopL skipped: %s" % exc)
            try:
                control.stopScript()
            except Exception as exc:
                print("cleanup stopScript skipped: %s" % exc)
    print("wrote %s" % args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
