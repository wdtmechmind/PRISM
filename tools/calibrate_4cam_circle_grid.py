#!/usr/bin/env python3
"""Calibrate four Hik cameras from synchronized CGB-035 circle-grid images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from collect_ur3_handeye_poses import CGB035_COLUMNS, CGB035_ROWS, find_cgb035_grid


def object_points() -> np.ndarray:
    return np.asarray(
        [[(2 * column + row % 2) * 0.035, row * 0.035, 0.0]
         for row in range(CGB035_ROWS) for column in range(CGB035_COLUMNS)],
        dtype=np.float32,
    )


def camera_dirs(root: Path) -> dict[str, Path]:
    return {
        "cam%d" % index: matches[0]
        for index in range(4)
        if (matches := sorted(path for path in root.glob("cam%d*" % index) if path.is_dir()))
    }


def image_points(directory: Path, extension: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str], tuple[int, int]]:
    points = []
    objects = []
    names = []
    size = None
    grid_object_points = object_points()
    for image_path in sorted(directory.glob("*" + extension)):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        current_size = (image.shape[1], image.shape[0])
        if size is None:
            size = current_size
        if current_size != size:
            raise RuntimeError("image size mismatch in %s" % directory)
        found, centers, _, _ = find_cgb035_grid(image)
        if found and centers is not None and len(centers) == len(grid_object_points):
            objects.append(grid_object_points.copy())
            points.append(centers.astype(np.float32))
            names.append(image_path.stem)
    if size is None:
        raise RuntimeError("no readable images in %s" % directory)
    return objects, points, names, size


def calibrate_camera(objects, points, image_size):
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(objects, points, image_size, None, None)
    return float(rms), matrix, distortion.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path,
                        help="CircleGridCapture_YYYYMMDD_HHMMSS directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-extension", default=".png")
    parser.add_argument("--min-valid-images", type=int, default=10)
    args = parser.parse_args()
    if args.min_valid_images < 3:
        parser.error("--min-valid-images must be at least 3")

    directories = camera_dirs(args.image_root.expanduser().resolve())
    if len(directories) != 4:
        raise RuntimeError("expected four camera directories, found: %s" % sorted(directories))

    observations = {}
    result = {
        "board": {"type": "asymmetric_circle_grid", "columns": CGB035_COLUMNS,
                  "rows": CGB035_ROWS, "spacing_mm": 35.0},
        "intrinsics": {},
        "multi_camera": {"reference_camera": "cam0", "extrinsics": {}},
    }
    for camera_key, directory in directories.items():
        objects, points, names, size = image_points(directory, args.image_extension)
        if len(points) < args.min_valid_images:
            raise RuntimeError("%s has only %d valid images; need at least %d" %
                               (camera_key, len(points), args.min_valid_images))
        rms, matrix, distortion = calibrate_camera(objects, points, size)
        observations[camera_key] = {"objects": objects, "points": points, "names": names,
                                    "size": size, "K": matrix, "D": distortion}
        result["intrinsics"][camera_key] = {
            "image_size": list(size), "num_images_total": len(list(directory.glob("*" + args.image_extension))),
            "num_images_valid": len(points), "rms": rms,
            "K": matrix.tolist(), "D": distortion.tolist(),
            "folder": str(directory),
        }
        print("%s: %d/%d valid, RMS %.4f px" %
              (camera_key, len(points), result["intrinsics"][camera_key]["num_images_total"], rms))

    reference = observations["cam0"]
    for camera_index in range(1, 4):
        camera_key = "cam%d" % camera_index
        target = observations[camera_key]
        common = sorted(set(reference["names"]) & set(target["names"]))
        reference_by_name = {name: index for index, name in enumerate(reference["names"])}
        target_by_name = {name: index for index, name in enumerate(target["names"])}
        object_sets = [reference["objects"][reference_by_name[name]] for name in common]
        points0 = [reference["points"][reference_by_name[name]] for name in common]
        pointsi = [target["points"][target_by_name[name]] for name in common]
        if len(common) < args.min_valid_images:
            raise RuntimeError("cam0/%s have only %d common valid images" % (camera_key, len(common)))
        _, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
            object_sets, points0, pointsi, reference["K"], reference["D"],
            target["K"], target["D"], reference["size"], flags=cv2.CALIB_FIX_INTRINSIC,
        )
        result["multi_camera"]["extrinsics"]["cam0_to_%s" % camera_key] = {
            "R": rotation.tolist(), "t": translation.reshape(3).tolist(),
            "num_common_images": len(common),
        }
        print("cam0 -> %s: %d common images" % (camera_key, len(common)))

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote %s" % args.output.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())