#!/usr/bin/env python3
"""Interactive calibration for MechHand V3 angle mapping from RPi UDP stream.

This tool calibrates source angles used by tools/teleop_mechhand_v3_from_rpi.py:
- mapping.J1..J5.source_min
- mapping.J1..J5.source_max
- mapping.J1..J5.invert

It listens to prism.rpi_hand_event.v1 UDP packets and records two user-defined
positions per joint:
1) source angle for target_min_deg
2) source angle for target_max_deg

Unlike tools/calibrate_encoder_pwm.py, this script does NOT touch encoder PWM
calibration JSON files under configs/devices/.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]


@dataclass
class JointSpec:
    name: str
    source_channel: int
    source_min: float
    source_max: float
    target_min_deg: float
    target_max_deg: float
    invert: bool


def resolve_config_path(value: str) -> Path:
    raw = str(value or "").strip()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = Path.cwd() / path
    if candidate.is_file():
        return candidate.resolve()
    fallback = REPO_ROOT / "configs" / "deployment" / path
    if fallback.is_file():
        return fallback.resolve()
    return candidate.resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _channel_value(payload: dict, channel: int):
    value = payload.get(channel)
    if value is not None:
        return value
    return payload.get(str(channel))


def parse_encoder_angles(event: dict) -> Dict[int, float]:
    out: Dict[int, float] = {}
    angles = event.get("angles") or {}
    for channel in range(1, 6):
        val = _safe_float(_channel_value(angles, channel))
        if val is None:
            val = _safe_float(event.get("enc%d_angle_deg" % channel))
        if val is None:
            val = _safe_float(event.get("enc%d_angle" % channel))
        if val is None:
            continue
        out[channel] = val
    return out


def build_joint_specs(mapping_cfg: dict) -> List[JointSpec]:
    order = ["J1", "J2", "J3", "J4", "J5"]
    specs: List[JointSpec] = []
    for i, name in enumerate(order, start=1):
        item = mapping_cfg.get(name) or {}
        specs.append(
            JointSpec(
                name=name,
                source_channel=int(item.get("source_channel", i)),
                source_min=float(item.get("source_min", item.get("source_min_deg", 0.0))),
                source_max=float(item.get("source_max", item.get("source_max_deg", 360.0))),
                target_min_deg=float(item.get("target_min_deg", -90.0)),
                target_max_deg=float(item.get("target_max_deg", 90.0)),
                invert=bool(item.get("invert", False)),
            )
        )
    return specs


def capture_channel_median(
    sock: socket.socket,
    channel: int,
    capture_sec: float,
    max_wait_sec: float,
    min_samples: int,
) -> Tuple[float, int]:
    values: List[float] = []
    deadline = time.monotonic() + max_wait_sec
    end_capture = time.monotonic() + capture_sec

    while time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except BlockingIOError:
            continue

        try:
            event = json.loads(data.decode("utf-8"))
        except Exception:
            continue

        if not isinstance(event, dict):
            continue

        angles = parse_encoder_angles(event)
        val = angles.get(channel)
        if val is None:
            continue
        values.append(float(val))

        now = time.monotonic()
        if now >= end_capture and len(values) >= min_samples:
            break

    if not values:
        raise RuntimeError("no UDP angle samples for Enc%d" % channel)

    median = float(statistics.median(values))
    return median, len(values)


def prompt_choice() -> str:
    text = input("Press ENTER to capture, s to skip this joint, q to quit: ").strip().lower()
    if text in {"s", "q"}:
        return text
    return ""


def update_mapping_for_joint(mapping_cfg: dict, spec: JointSpec, src_for_target_min: float, src_for_target_max: float) -> None:
    item = mapping_cfg.setdefault(spec.name, {})

    if src_for_target_max >= src_for_target_min:
        item["source_min"] = float(src_for_target_min)
        item["source_max"] = float(src_for_target_max)
        item["invert"] = False
    else:
        item["source_min"] = float(src_for_target_max)
        item["source_max"] = float(src_for_target_min)
        item["invert"] = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "deployment" / "mechhand_v3_teleop.yaml"),
        help="teleop YAML config to read/update",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional output YAML path; default overwrites --config",
    )
    parser.add_argument("--event-host", default=None, help="UDP bind host override")
    parser.add_argument("--event-port", type=int, default=None, help="UDP bind port override")
    parser.add_argument("--recv-timeout-sec", type=float, default=0.02, help="socket recv timeout")
    parser.add_argument("--capture-sec", type=float, default=0.7, help="capture window per sample")
    parser.add_argument("--max-wait-sec", type=float, default=8.0, help="max wait per sample")
    parser.add_argument("--min-samples", type=int, default=10, help="minimum sample count per capture")
    parser.add_argument("--min-span-deg", type=float, default=8.0, help="warn when span is below this")
    parser.add_argument("--no-backup", action="store_true", help="do not create backup before overwrite")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg_path = resolve_config_path(args.config)
    if not cfg_path.is_file():
        raise SystemExit("config not found: %s" % cfg_path)

    cfg = load_yaml(cfg_path)
    mapping_cfg = cfg.setdefault("mapping", {})
    udp_cfg = cfg.get("rpi_udp") or {}

    host = str(args.event_host or udp_cfg.get("host", "0.0.0.0"))
    port = int(args.event_port or udp_cfg.get("port", 60701))

    out_path = resolve_config_path(args.output) if args.output else cfg_path

    specs = build_joint_specs(mapping_cfg)

    print("[calib-map] config:", cfg_path)
    print("[calib-map] output:", out_path)
    print("[calib-map] listening UDP %s:%d" % (host, port))
    print("[calib-map] This updates mapping source_min/source_max/invert only.")
    print("[calib-map] It does NOT modify configs/devices/encoder_pwm_calibration*.json")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(max(0.001, float(args.recv_timeout_sec)))
    sock.bind((host, port))

    updated = 0
    try:
        for spec in specs:
            print("\n=== %s (Enc%d) ===" % (spec.name, spec.source_channel))
            print(
                "Current mapping: src=[%.2f, %.2f] invert=%s -> tgt=[%.1f, %.1f] deg"
                % (
                    spec.source_min,
                    spec.source_max,
                    str(spec.invert),
                    spec.target_min_deg,
                    spec.target_max_deg,
                )
            )

            print(
                "Move exoskeleton to the pose that should map to %s target_min_deg (%.1f deg)."
                % (spec.name, spec.target_min_deg)
            )
            choice = prompt_choice()
            if choice == "q":
                break
            if choice == "s":
                print("skip", spec.name)
                continue

            low_val, low_n = capture_channel_median(
                sock=sock,
                channel=spec.source_channel,
                capture_sec=float(args.capture_sec),
                max_wait_sec=float(args.max_wait_sec),
                min_samples=max(1, int(args.min_samples)),
            )
            print("Captured target_min source angle: %.3f (samples=%d)" % (low_val, low_n))

            print(
                "Move exoskeleton to the pose that should map to %s target_max_deg (%.1f deg)."
                % (spec.name, spec.target_max_deg)
            )
            choice = prompt_choice()
            if choice == "q":
                break
            if choice == "s":
                print("skip", spec.name)
                continue

            high_val, high_n = capture_channel_median(
                sock=sock,
                channel=spec.source_channel,
                capture_sec=float(args.capture_sec),
                max_wait_sec=float(args.max_wait_sec),
                min_samples=max(1, int(args.min_samples)),
            )
            print("Captured target_max source angle: %.3f (samples=%d)" % (high_val, high_n))

            span = abs(high_val - low_val)
            if span < float(args.min_span_deg):
                print(
                    "WARNING: span %.2f deg < min-span-deg %.2f."
                    " Calibration may be too narrow."
                    % (span, float(args.min_span_deg))
                )

            update_mapping_for_joint(mapping_cfg, spec, low_val, high_val)
            item = mapping_cfg[spec.name]
            print(
                "Updated %s -> source_min=%.3f source_max=%.3f invert=%s"
                % (spec.name, float(item["source_min"]), float(item["source_max"]), str(bool(item["invert"])))
            )
            updated += 1

        if updated == 0:
            print("[calib-map] no joint updated, nothing written.")
            return 0

        print("\n[calib-map] %d joint mappings updated." % updated)
        answer = input("Write changes to %s ? [y/N]: " % out_path).strip().lower()
        if answer not in {"y", "yes"}:
            print("[calib-map] canceled by user.")
            return 0

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path == cfg_path and (not args.no_backup):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = cfg_path.with_suffix(cfg_path.suffix + ".bak." + ts)
            backup.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
            print("[calib-map] backup created:", backup)

        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

        print("[calib-map] saved:", out_path)
        return 0
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
