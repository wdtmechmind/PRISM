#!/usr/bin/env python3
"""Live four-camera YOLO confidence tuner for PRISM Hik cameras.

The cameras use the existing PRISM hardware-trigger arrangement. The window
shows cam0..cam3, YOLO boxes, class/confidence labels, and per-camera counts.
Move the ``conf x0.01`` slider while the cameras are live.

Keys:
    S       save the current 2x2 preview
    Q/Esc   quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for candidate in (os.environ.get("PRISM_MVIMPORT_DIR", ""),
                  "/opt/MVS/Samples/64/Python/MvImport"):
    if candidate and os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from prism.devices.cameras.mvs_camera import (  # noqa: E402
    MvCamera,
    UsbCameraGrabber,
    enumerate_usb_devices,
    parse_indices,
)
from prism.devices.cameras.highspeed_camera import (  # noqa: E402
    HikCaptureThread,
    SoftwareTriggerThread,
)

COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 255),
    "blue": (255, 120, 0),
    "green": (0, 220, 0),
}
MASTER_SERIAL = "DA8165486"


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
        cv2.putText(output, "%s %.3f" % (label, float(confidence)),
                    (x0, max(20, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    return output, counts


def tile(cells: list[np.ndarray], width: int) -> np.ndarray:
    if not cells:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    resized = []
    for cell in cells:
        height, source_width = cell.shape[:2]
        scale = width / float(source_width)
        resized.append(cv2.resize(cell, (width, max(1, int(height * scale)))))
    height = max(cell.shape[0] for cell in resized)
    padded = [cv2.copyMakeBorder(cell, 0, height - cell.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
              for cell in resized]
    while len(padded) < 4:
        padded.append(np.zeros_like(padded[0]))
    return np.vstack((np.hstack((padded[0], padded[1])),
                      np.hstack((padded[2], padded[3]))))


def select_devices(camera_serials: str | None):
    devices = enumerate_usb_devices()
    if len(devices) < 4:
        raise RuntimeError("found %d USB cameras, need at least 4" % len(devices))
    print("Found %d USB cameras:" % len(devices))
    for index, _, model, serial in devices:
        print("  [%d] model=%s serial=%s" % (index, model, serial))

    if camera_serials:
        wanted = [item.strip() for item in camera_serials.split(",") if item.strip()]
        if len(wanted) != 4:
            raise ValueError("--camera-serials requires four comma-separated serials")
        by_serial = {serial: item for item in devices for serial in [item[3]]}
        missing = [serial for serial in wanted if serial not in by_serial]
        if missing:
            raise RuntimeError("camera serial not found: %s" % ", ".join(missing))
        return [by_serial[serial] for serial in wanted]

    raw = input("camera indices for cam0..cam3 (blank=first 4): ").strip()
    indices = list(range(4)) if not raw else parse_indices(raw, len(devices), 4)
    return [devices[index] for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_conf_live"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera-serials", default=None,
                        help="four serials in cam0,cam1,cam2,cam3 order")
    parser.add_argument("--exposure-us", type=float, default=3000.0)
    parser.add_argument("--gain", type=float, default=12.0)
    parser.add_argument("--frame-rate", type=float, default=30.0)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("需要 ultralytics：pip install ultralytics") from exc

    model = YOLO(str(args.yolo_weights.expanduser().resolve()))
    names = getattr(model, "names", {})
    MvCamera.MV_CC_Initialize()
    cameras = []
    threads = []
    trigger = None
    window = "Live YOLO confidence | S=save Q=quit"
    try:
        selected = select_devices(args.camera_serials)
        serials = [item[3] for item in selected]
        master_index = serials.index(MASTER_SERIAL) if MASTER_SERIAL in serials else 0
        if MASTER_SERIAL not in serials:
            print("warning: master %s not selected; using cam0 as master" % MASTER_SERIAL)

        for cam_i, (_, device_info, model_name, serial) in enumerate(selected):
            grabber = UsbCameraGrabber(device_info, serial, model_name)
            grabber.open_and_prepare(
                exposure_us=args.exposure_us,
                gain=args.gain,
                frame_rate=args.frame_rate,
                trigger_source="Software" if cam_i == master_index else "Line0",
                gpio_output_line=1 if cam_i == master_index else None,
            )
            cameras.append((cam_i, grabber, serial))

        threads = [None] * 4
        for cam_i, grabber, serial in cameras:
            thread = HikCaptureThread(grabber, cam_i, serial, timeout_ms=1000, buffer_len=3)
            thread.start()
            threads[cam_i] = thread
        trigger = SoftwareTriggerThread(cameras[master_index][1], fps=args.frame_rate)
        trigger.start()
        print("hardware-trigger live stream started")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.preview_width * 2, int(args.preview_width * 0.82) * 2)
        cv2.createTrackbar("conf x0.01", window, 50, 99, lambda value: None)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        last_preview = None
        while True:
            confidence = max(1, cv2.getTrackbarPos("conf x0.01", window)) / 100.0
            frames = []
            available = []
            for cam_i in range(4):
                frame = threads[cam_i].get_latest() if threads[cam_i] else None
                if frame is None:
                    frame = np.zeros((540, 720, 3), dtype=np.uint8)
                    cv2.putText(frame, "cam%d: NO FRAME" % cam_i, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    available.append(cam_i)
                frames.append(frame)

            predict_kwargs = {"conf": confidence, "imgsz": args.imgsz, "verbose": False}
            if args.device is not None:
                predict_kwargs["device"] = args.device
            results = model.predict(frames, **predict_kwargs)
            cells = []
            summaries = []
            for cam_i, (frame, result) in enumerate(zip(frames, results)):
                annotated, counts = annotate(frame, result, names)
                total = sum(counts.values())
                cv2.putText(annotated, "cam%d %s | conf=%.2f | n=%d" %
                            (cam_i, serials[cam_i], confidence, total), (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cells.append(annotated)
                summaries.append("cam%d:%d" % (cam_i, total))
            preview = tile(cells, args.preview_width)
            cv2.putText(preview, "conf=%.2f | %s | S=save Q=quit" %
                        (confidence, " ".join(summaries)), (12, preview.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            last_preview = preview
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")) and last_preview is not None:
                output = output_dir / "live_%s_conf_%03d.png" % (
                    time.strftime("%Y%m%d_%H%M%S"), int(round(confidence * 100))
                )
                cv2.imwrite(str(output), last_preview)
                print("saved %s" % output)
        return 0
    finally:
        cv2.destroyAllWindows()
        if trigger is not None:
            trigger.stop()
            trigger.join(timeout=1.0)
        for thread in threads:
            if thread is not None:
                thread.stop()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=2.0)
        for _, camera, _ in cameras:
            try:
                camera.stop_and_close()
            except Exception:
                pass
        MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    raise SystemExit(main())
