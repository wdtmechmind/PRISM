#!/usr/bin/env python3
"""Export all calibrated camera poses in the robot Base frame.

Inputs:
  - charuco_4cam_result.json: intrinsics and cam0 -> camN transforms
  - robot_camera_handeye.json: base_from_cam0

The output contains ``base_from_camN`` for every camera and camera centers in
robot Base coordinates. It is intended as a replay/deployment companion file;
it does not alter the original camera calibration JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def load_transform(data: dict, key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=np.float64).reshape(4, 4)


def camera_transform_from_cam0(calibration: dict, camera_index: int) -> np.ndarray:
    if camera_index == 0:
        return np.eye(4, dtype=np.float64)
    item = calibration["multi_camera"]["extrinsics"]["cam0_to_cam%d" % camera_index]
    transform_cam_i_from_cam0 = np.eye(4, dtype=np.float64)
    transform_cam_i_from_cam0[:3, :3] = np.asarray(item["R"], dtype=np.float64)
    transform_cam_i_from_cam0[:3, 3] = np.asarray(item["t"], dtype=np.float64).reshape(3)
    return inverse_transform(transform_cam_i_from_cam0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-calib", required=True, type=Path)
    parser.add_argument("--handeye", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.camera_calib.expanduser().resolve().open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    with args.handeye.expanduser().resolve().open("r", encoding="utf-8") as handle:
        handeye = json.load(handle)

    base_from_cam0 = load_transform(handeye, "base_from_cam0")
    cameras = {}
    serials = {}
    for camera_index in range(4):
        camera_key = "cam%d" % camera_index
        base_from_cam = base_from_cam0 @ camera_transform_from_cam0(calibration, camera_index)
        intrinsics = calibration["intrinsics"][camera_key]
        folder = str(intrinsics.get("folder", ""))
        serial = Path(folder.rstrip("/")).name.split("_", 1)[1] if "_" in Path(folder.rstrip("/")).name else ""
        serials[camera_key] = serial
        cameras[camera_key] = {
            "serial": serial,
            "base_from_camera": base_from_cam.tolist(),
            "camera_center_in_base_m": base_from_cam[:3, 3].tolist(),
            "image_size": intrinsics.get("image_size"),
            "K": intrinsics.get("K"),
            "D": intrinsics.get("D"),
            "source_extrinsic": "identity" if camera_index == 0 else "inverse(cam0_to_cam%d)" % camera_index,
        }

    output_data = {
        "frame_convention": "base_from_camera maps camera-frame points into robot Base coordinates",
        "reference_camera": "cam0",
        "camera_serials": serials,
        "base_from_cam0": base_from_cam0.tolist(),
        "cameras": cameras,
        "source_camera_calibration": str(args.camera_calib.expanduser().resolve()),
        "source_handeye_calibration": str(args.handeye.expanduser().resolve()),
        "handeye_samples": handeye.get("num_samples"),
        "handeye_translation_rmse_m": handeye.get("translation_rmse_m"),
        "handeye_rotation_rmse_deg": handeye.get("rotation_rmse_deg"),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)
    print("wrote %s" % output_path)
    for camera_key, item in cameras.items():
        print("%s serial=%s center_base_m=[%s]" % (
            camera_key,
            item["serial"],
            ", ".join("%.6f" % value for value in item["camera_center_in_base_m"]),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
