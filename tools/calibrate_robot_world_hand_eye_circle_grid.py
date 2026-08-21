#!/usr/bin/env python3
"""Jointly calibrate a fixed cam0, robot base, and a moving Circle Grid.

The target is a CGB-035 asymmetric 5 x 4 circle grid with 35 mm center spacing.
The target is assumed to be rigidly attached to the robot TCP, but its TCP
offset is unknown. Each robot row must describe the same stopped pose as the
corresponding image.

Example robot CSV columns::

    frame,x_m,y_m,z_m,rx_rad,ry_rad,rz_rad

``[x_m, y_m, z_m]`` is TCP in robot base coordinates. ``[rx_rad, ry_rad,
rz_rad]`` is an axis-angle rotation vector for base_from_tcp. Degree columns
``rx_deg,ry_deg,rz_deg`` are accepted instead.

The output contains ``base_from_cam0`` and ``tcp_from_board``. The board frame
has its origin at the top-left grid center, +X along columns, +Y along rows,
and +Z out of the board according to the object-point convention below.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


_CIRCLE_GRID_BLOB_DETECTOR = None
CGB035_COLUMNS = 4
CGB035_ROWS = 5


def circle_grid_blob_detector():
    """Detect white CGB-035 circles on its dark background."""
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


def find_cgb035_grid(gray: np.ndarray, asymmetric: bool):
    """Find CGB-035 circles with a white-blob detector and robust flags."""
    detector = circle_grid_blob_detector()
    base_flag = cv2.CALIB_CB_ASYMMETRIC_GRID if asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID
    for flags in (base_flag, base_flag | cv2.CALIB_CB_CLUSTERING):
        found, centers = cv2.findCirclesGrid(
            gray, (CGB035_COLUMNS, CGB035_ROWS), flags=flags, blobDetector=detector
        )
        if found:
            return found, centers
    return False, None


def make_object_points(columns: int, rows: int, spacing_m: float, asymmetric: bool) -> np.ndarray:
    return np.asarray(
                [[(2 * column + (row % 2 if asymmetric else 0)) * spacing_m,
                    row * spacing_m, 0.0]
                 for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )


def pose_from_rotation_vector(position: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rotation_vector.reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position.reshape(3)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def read_robot_poses(path: Path) -> dict[str, np.ndarray]:
    poses = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"frame", "x_m", "y_m", "z_m"}
        missing = required - fields
        if missing:
            raise ValueError("robot CSV missing columns: %s" % ", ".join(sorted(missing)))

        if {"rx_rad", "ry_rad", "rz_rad"}.issubset(fields):
            rotation_columns = ("rx_rad", "ry_rad", "rz_rad")
            angle_scale = 1.0
        elif {"rx_deg", "ry_deg", "rz_deg"}.issubset(fields):
            rotation_columns = ("rx_deg", "ry_deg", "rz_deg")
            angle_scale = np.pi / 180.0
        else:
            raise ValueError(
                "robot CSV needs rx_rad/ry_rad/rz_rad or rx_deg/ry_deg/rz_deg"
            )

        for row in reader:
            frame = str(row["frame"]).strip()
            if not frame:
                continue
            position = np.asarray([float(row[key]) for key in ("x_m", "y_m", "z_m")])
            rotation_vector = np.asarray([float(row[key]) for key in rotation_columns]) * angle_scale
            poses[frame] = pose_from_rotation_vector(position, rotation_vector)
    return poses


def read_camera_calibrations(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[int]]], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    calibrations = {}
    for camera_key, camera in data["intrinsics"].items():
        calibrations[camera_key] = (
            np.asarray(camera["K"], dtype=np.float64),
            np.asarray(camera["D"], dtype=np.float64).reshape(-1, 1),
            [int(camera["image_size"][0]), int(camera["image_size"][1])],
        )

    camera_from_cam0 = {"cam0": np.eye(4, dtype=np.float64)}
    for camera_index in (1, 2, 3):
        key = "cam0_to_cam%d" % camera_index
        extrinsic = data["multi_camera"]["extrinsics"].get(key)
        if extrinsic is None:
            raise ValueError("camera calibration missing %s" % key)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(extrinsic["R"], dtype=np.float64)
        transform[:3, 3] = np.asarray(extrinsic["t"], dtype=np.float64).reshape(3)
        camera_from_cam0["cam%d" % camera_index] = transform
    return calibrations, camera_from_cam0


def find_camera_dirs(image_root: Path) -> dict[str, Path]:
    directories = {}
    for camera_index in range(4):
        matches = sorted(image_root.glob("cam%d*" % camera_index))
        matches = [path for path in matches if path.is_dir()]
        if matches:
            directories["cam%d" % camera_index] = matches[0]
    return directories


def detect_board_pose(
    image: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    asymmetric: bool,
) -> tuple[np.ndarray, float] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_CLUSTERING
    if asymmetric:
        flags |= cv2.CALIB_CB_ASYMMETRIC_GRID
    else:
        flags |= cv2.CALIB_CB_SYMMETRIC_GRID

    found, centers = find_cgb035_grid(gray, asymmetric)
    if not found or centers is None or len(centers) != len(object_points):
        return None

    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        centers.reshape(-1, 2),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    pose = pose_from_rotation_vector(
        translation_vector.reshape(3), rotation_vector.reshape(3)
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    reprojection_error = float(np.mean(np.linalg.norm(
        projected.reshape(-1, 2) - centers.reshape(-1, 2), axis=1
    )))
    return pose, reprojection_error


def frame_name(frame: str, extension: str) -> str:
    path = Path(frame)
    return path.name if path.suffix else path.name + extension


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(relative)
    return float(np.degrees(np.linalg.norm(rotation_vector)))


def solve_joint(
    cam_from_board: list[np.ndarray],
    base_from_tcp: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(cv2, "calibrateRobotWorldHandEye"):
        raise RuntimeError("this OpenCV build lacks calibrateRobotWorldHandEye")

    # OpenCV solves A X = Z B. For this fixed-camera/moving-board setup we
    # use the algebraic assignment A=base_from_tcp, X=tcp_from_board,
    # Z=base_from_cam0, B=cam_from_board. The parameter names in OpenCV's
    # documentation describe the complementary eye-in-hand arrangement, but
    # the equation is the same.
    r_world2cam = [pose[:3, :3] for pose in base_from_tcp]
    t_world2cam = [pose[:3, 3].reshape(3, 1) for pose in base_from_tcp]
    r_base2gripper = [pose[:3, :3] for pose in cam_from_board]
    t_base2gripper = [pose[:3, 3].reshape(3, 1) for pose in cam_from_board]
    tcp_from_board_r, tcp_from_board_t, base_from_cam_r, base_from_cam_t = cv2.calibrateRobotWorldHandEye(
        r_world2cam,
        t_world2cam,
        r_base2gripper,
        t_base2gripper,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    tcp_from_board = np.eye(4, dtype=np.float64)
    tcp_from_board[:3, :3] = np.asarray(tcp_from_board_r, dtype=np.float64)
    tcp_from_board[:3, 3] = np.asarray(tcp_from_board_t, dtype=np.float64).reshape(3)
    base_from_cam = np.eye(4, dtype=np.float64)
    base_from_cam[:3, :3] = np.asarray(base_from_cam_r, dtype=np.float64)
    base_from_cam[:3, 3] = np.asarray(base_from_cam_t, dtype=np.float64).reshape(3)
    return base_from_cam, tcp_from_board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image-root", type=Path,
                             help="capture directory containing cam0_*, cam1_*, cam2_*, cam3_*")
    image_group.add_argument("--image-dir", type=Path,
                             help="legacy cam0-only image directory")
    parser.add_argument("--robot-csv", required=True, type=Path)
    parser.add_argument("--camera-calib", required=True, type=Path, help="charuco_4cam_result.json")
    parser.add_argument("--camera-key", default="cam0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-extension", default=".png")
    parser.add_argument("--symmetric", action="store_true",
                        help="override the default asymmetric CGB-035 pattern")
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()

    if args.min_samples < 3:
        parser.error("--min-samples must be at least 3")
    object_points = make_object_points(CGB035_COLUMNS, CGB035_ROWS, 0.035, asymmetric=not args.symmetric)
    calibrations, camera_from_cam0 = read_camera_calibrations(args.camera_calib)
    if args.camera_key not in calibrations:
        raise RuntimeError("camera calibration does not contain %s" % args.camera_key)
    if args.image_root:
        camera_dirs = find_camera_dirs(args.image_root.expanduser().resolve())
    else:
        camera_dirs = {args.camera_key: args.image_dir.expanduser().resolve()}
        print("warning: --image-dir enables cam0-only processing; use --image-root for all cameras")
    if not camera_dirs:
        raise RuntimeError("no camera directories found")
    robot_poses = read_robot_poses(args.robot_csv)

    board_poses = []
    base_poses = []
    sample_names = []
    sample_sources = []
    for frame, robot_pose in robot_poses.items():
        candidates = []
        for camera_key, image_dir in camera_dirs.items():
            if camera_key not in calibrations:
                print("skip %s/%s: no calibration entry" % (frame, camera_key))
                continue
            image_path = image_dir / frame_name(frame, args.image_extension)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            camera_matrix, distortion, image_size = calibrations[camera_key]
            if [image.shape[1], image.shape[0]] != image_size:
                raise RuntimeError(
                    "image size mismatch for %s: got %s, calibration expects %s"
                    % (image_path, [image.shape[1], image.shape[0]], image_size)
                )
            detected = detect_board_pose(
                image, object_points, camera_matrix, distortion, not args.symmetric
            )
            if detected is None:
                continue
            pose_camera_from_board, reprojection_error = detected
            pose_cam0_from_board = inverse_transform(camera_from_cam0[camera_key]) @ pose_camera_from_board
            candidates.append((reprojection_error, camera_key, pose_cam0_from_board))

        if not candidates:
            print("skip %s: Circle Grid not detected in any camera" % frame)
            continue
        candidates.sort(key=lambda candidate: candidate[0])
        reprojection_error, source_camera, board_pose = candidates[0]
        sample_names.append(frame)
        sample_sources.append({"frame": frame, "camera": source_camera,
                               "reprojection_error_px": reprojection_error,
                               "candidates": [candidate[1] for candidate in candidates]})
        board_poses.append(board_pose)
        base_poses.append(robot_pose)
        print("use %s for %s (reprojection %.3f px; candidates=%s)"
              % (source_camera, frame, reprojection_error,
                 ",".join(candidate[1] for candidate in candidates)))

    if len(board_poses) < args.min_samples:
        raise RuntimeError("only %d valid samples; need at least %d" % (len(board_poses), args.min_samples))

    base_from_cam0, tcp_from_board = solve_joint(board_poses, base_poses)
    translation_errors = []
    rotation_errors = []
    for board_pose, base_pose in zip(board_poses, base_poses):
        predicted_cam_from_board = inverse_transform(base_from_cam0) @ base_pose @ tcp_from_board
        translation_errors.append(float(np.linalg.norm(
            predicted_cam_from_board[:3, 3] - board_pose[:3, 3]
        )))
        rotation_errors.append(rotation_error_deg(predicted_cam_from_board, board_pose))

    result = {
        "target": {"type": "asymmetric_circle_grid", "columns": CGB035_COLUMNS, "rows": CGB035_ROWS, "spacing_mm": 35.0},
        "camera_frame": args.camera_key,
        "camera_calibration": str(args.camera_calib.resolve()),
        "board_frame": "origin=top_left_circle, +X=columns, +Y=rows, +Z=solvePnP_normal",
        "base_from_cam0": base_from_cam0.tolist(),
        "tcp_from_board": tcp_from_board.tolist(),
        "num_samples": len(sample_names),
        "samples": sample_names,
        "sample_sources": sample_sources,
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(translation_errors)))),
        "translation_max_m": float(max(translation_errors)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotation_errors)))),
        "rotation_max_deg": float(max(rotation_errors)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("valid samples: %d" % len(sample_names))
    print("translation RMSE: %.3f mm (max %.3f mm)" % (result["translation_rmse_m"] * 1000, result["translation_max_m"] * 1000))
    print("rotation RMSE: %.3f deg (max %.3f deg)" % (result["rotation_rmse_deg"], result["rotation_max_deg"]))
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())