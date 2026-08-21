#!/usr/bin/env python3
"""Visualize YOLO detections while interactively tuning confidence.

Supports a synchronized capture directory containing cam0_*/.../cam3_* image
folders, or a single image directory. The confidence slider reruns YOLO on the
current frame. This is intended for checking false positives and missed LEDs,
not for producing reconstruction CSV files.

Keys:
    left/right or A/D  previous/next synchronized frame
    S                  save the current 2x2 visualization
    Q/Esc              quit
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "blue": (255, 120, 0),
    "green": (0, 220, 0),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_images(root: Path) -> dict[str, list[Path]]:
    directories = {}
    camera_dirs = sorted(path for path in root.glob("cam[0-3]_*") if path.is_dir())
    if camera_dirs:
        for path in camera_dirs:
            camera_key = path.name.split("_", 1)[0]
            directories[camera_key] = path
    else:
        directories["cam0"] = root

    result = {}
    for camera_key, directory in directories.items():
        result[camera_key] = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return result


def load_frame(images: dict[str, list[Path]], index: int) -> dict[str, np.ndarray]:
    frames = {}
    for camera_key, paths in images.items():
        if index >= len(paths):
            continue
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is not None:
            frames[camera_key] = image
    return frames


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def annotate(image: np.ndarray, result, names) -> tuple[np.ndarray, dict[str, int]]:
    output = image.copy()
    counts = {name: 0 for name in COLORS}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return output, counts
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
        label = class_name(names, int(class_id)).lower()
        color = COLORS.get(label, (255, 255, 255))
        if label in counts:
            counts[label] += 1
        x0, y0, x1, y1 = [int(round(value)) for value in coordinates]
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        text = "%s %.3f" % (label, float(confidence))
        text_y = max(20, y0 - 7)
        cv2.putText(output, text, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    return output, counts


def tile(cells: list[np.ndarray], width: int) -> np.ndarray:
    resized = []
    for cell in cells:
        height, source_width = cell.shape[:2]
        scale = width / float(source_width)
        resized.append(cv2.resize(cell, (width, max(1, int(height * scale)))))
    if not resized:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    height = max(cell.shape[0] for cell in resized)
    padded = [cv2.copyMakeBorder(cell, 0, height - cell.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
              for cell in resized]
    while len(padded) < 4:
        padded.append(np.zeros_like(padded[0]))
    return np.vstack((np.hstack((padded[0], padded[1])),
                      np.hstack((padded[2], padded[3]))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path,
                        help="capture folder containing cam0_*, cam1_*, ...")
    parser.add_argument("--yolo-weights", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_conf_debug"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("需要 ultralytics：pip install ultralytics") from exc

    images = discover_images(args.image_root.expanduser().resolve())
    if not images or not any(images.values()):
        raise SystemExit("没有找到图片目录或图片文件")
    frame_count = max(len(paths) for paths in images.values())
    index = max(0, min(args.start_index, frame_count - 1))
    model = YOLO(str(args.yolo_weights.expanduser().resolve()))
    names = getattr(model, "names", {})
    window = "YOLO confidence tuner | A/D=frame S=save Q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.preview_width * 2, int(args.preview_width * 0.82) * 2)

    confidence = 50
    cv2.createTrackbar("conf x0.01", window, confidence, 99, lambda value: None)
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    try:
        while True:
            confidence = max(1, cv2.getTrackbarPos("conf x0.01", window)) / 100.0
            frames = load_frame(images, index)
            cells = []
            summaries = []
            for camera_key in ("cam0", "cam1", "cam2", "cam3"):
                image = frames.get(camera_key)
                if image is None:
                    continue
                predict_kwargs = {"conf": confidence, "imgsz": args.imgsz, "verbose": False}
                if args.device is not None:
                    predict_kwargs["device"] = args.device
                result = model.predict(image, **predict_kwargs)[0]
                annotated, counts = annotate(image, result, names)
                total = sum(counts.values())
                cv2.putText(annotated, "%s | conf=%.2f | detections=%d" %
                            (camera_key, confidence, total), (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cells.append(annotated)
                summaries.append("%s:%d" % (camera_key, total))

            preview = tile(cells, args.preview_width)
            cv2.putText(preview, "frame=%d/%d | %s | A/D or arrows: frame | S: save | Q: quit" %
                        (index, frame_count - 1, " ".join(summaries)),
                        (12, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D"), 83):
                index = min(frame_count - 1, index + 1)
            elif key in (ord("a"), ord("A"), 81):
                index = max(0, index - 1)
            elif key in (ord("s"), ord("S")):
                output_name = "frame_%04d_conf_%03d.png" % (
                    index, int(round(confidence * 100))
                )
                output = args.output_dir.expanduser().resolve() / output_name
                cv2.imwrite(str(output), preview)
                print("saved %s" % output)
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
