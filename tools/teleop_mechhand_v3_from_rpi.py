#!/usr/bin/env python3
"""Real-hardware teleop: RPi encoder UDP stream -> MechHand V3 daemon RM commands.

This script receives RPi event packets (schema prism.rpi_hand_event.v1), filters
Enc1..Enc5 angles, maps to J1..J5 (SDK order), and sends @RM<...>& commands.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

import yaml

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism.devices.hand.v3_daemon_client import MechHandV3CommandError, MechHandV3DaemonClient, degrees_to_tenths


def resolve_repo_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


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


def _median(values: Deque[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    m = n // 2
    if n % 2 == 1:
        return ordered[m]
    return 0.5 * (ordered[m - 1] + ordered[m])


def filter_angles(
    raw_angles: Dict[int, float],
    histories: Dict[int, Deque[float]],
    prev_filtered: Dict[int, float],
    median_window: int,
    input_deadband_deg: float,
) -> Dict[int, float]:
    out = dict(prev_filtered)
    for ch, value in raw_angles.items():
        history = histories.setdefault(ch, deque(maxlen=max(1, int(median_window))))
        history.append(float(value))
        filtered = _median(history)
        prev = prev_filtered.get(ch)
        if prev is not None and abs(filtered - prev) < input_deadband_deg:
            out[ch] = float(prev)
        else:
            out[ch] = float(filtered)
    return out


@dataclass
class JointMapSpec:
    source_channel: int
    source_min: float
    source_max: float
    target_min_deg: float
    target_max_deg: float
    invert: bool


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def map_channel_to_joint_deg(angle_deg: float, spec: JointMapSpec) -> float:
    lo = float(spec.source_min)
    hi = float(spec.source_max)
    if hi <= lo:
        return float(spec.target_min_deg)

    p = (float(angle_deg) - lo) / (hi - lo)
    p = max(0.0, min(1.0, p))
    if spec.invert:
        p = 1.0 - p
    return float(spec.target_min_deg) + p * (float(spec.target_max_deg) - float(spec.target_min_deg))


def map_channel_delta_to_joint_deg(delta_source_deg: float, spec: JointMapSpec) -> float:
    lo = float(spec.source_min)
    hi = float(spec.source_max)
    span = hi - lo
    if span <= 1e-9:
        return 0.0
    gain = (float(spec.target_max_deg) - float(spec.target_min_deg)) / span
    if spec.invert:
        gain = -gain
    return float(delta_source_deg) * gain


def unwrap_delta_deg(curr: float, prev: float, wrap_deg: float) -> float:
    d = float(curr) - float(prev)
    if wrap_deg <= 0.0:
        return d
    half = 0.5 * wrap_deg
    while d > half:
        d -= wrap_deg
    while d < -half:
        d += wrap_deg
    return d

def apply_joint_limits(
    desired: Dict[str, float],
    prev: Dict[str, float],
    joint_deadband_deg: float,
    max_joint_speed_deg_s: float,
    dt: float,
) -> Dict[str, float]:
    out = dict(prev)
    deadband = max(0.0, float(joint_deadband_deg))
    max_speed = max(0.0, float(max_joint_speed_deg_s))
    dt_sec = max(0.0, float(dt))
    for name, target in desired.items():
        t = float(target)
        old = float(prev.get(name, t))
        if abs(t - old) < deadband:
            t = old
        if max_speed > 0.0 and dt_sec > 0.0:
            max_step = max_speed * dt_sec
            delta = t - old
            if delta > max_step:
                t = old + max_step
            elif delta < -max_step:
                t = old - max_step
        out[name] = t
    return out


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_range_calibration(
    sock: socket.socket,
    duration_sec: float,
    recv_timeout: float,
) -> Dict[int, Dict[str, float]]:
    """Collect encoder packets for *duration_sec* seconds and return per-channel
    {min, max} ranges.  Returns an empty dict if no packets arrive.
    """
    ranges: Dict[int, Dict[str, float]] = {}
    end_t = time.monotonic() + duration_sec
    next_print = time.monotonic() + 1.0
    packet_count = 0

    print("[calib] move each finger through its FULL range now...")
    while True:
        now = time.monotonic()
        remaining = end_t - now
        if remaining <= 0.0:
            break
        if now >= next_print:
            print("[calib] %.0fs remaining, packets=%d" % (remaining, packet_count))
            next_print += 1.0
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
        if not angles:
            continue
        packet_count += 1
        for ch, val in angles.items():
            if ch not in ranges:
                ranges[ch] = {"min": val, "max": val}
            else:
                if val < ranges[ch]["min"]:
                    ranges[ch]["min"] = val
                if val > ranges[ch]["max"]:
                    ranges[ch]["max"] = val

    print("[calib] done. packets=%d" % packet_count)
    for ch in sorted(ranges):
        r = ranges[ch]
        print("[calib] Enc%d  min=%.2f  max=%.2f  span=%.2f" % (ch, r["min"], r["max"], r["max"] - r["min"]))
    if not ranges:
        print("[calib] WARNING: no packets received, keeping YAML source ranges")
    return ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "deployment" / "mechhand_v3_teleop.yaml"))
    parser.add_argument("--event-host", default=None)
    parser.add_argument("--event-port", type=int, default=None)
    parser.add_argument("--daemon-host", default=None)
    parser.add_argument("--daemon-port", type=int, default=None)
    parser.add_argument("--daemon-timeout-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="do not send RM commands to daemon")
    parser.add_argument("--print-every", type=float, default=2.0, help="status log rate in Hz")
    parser.add_argument(
        "--calibrate-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        help="range calibration duration in seconds before teleop starts (default: 0, disabled)",
    )
    return parser.parse_args()


def _values_from_named_list(data: Sequence[int], fallback: int) -> List[int]:
    if len(data) == 5:
        return [int(v) for v in data]
    return [int(fallback)] * 5


def main() -> int:
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    if not cfg_path.is_file():
        raise SystemExit("config not found: %s" % cfg_path)
    cfg = load_config(cfg_path)

    udp_cfg = cfg.get("rpi_udp") or {}
    daemon_cfg = cfg.get("daemon") or {}
    control_cfg = cfg.get("control") or {}
    mapping_cfg = cfg.get("mapping") or {}

    event_host = str(args.event_host or udp_cfg.get("host", "0.0.0.0"))
    event_port = int(args.event_port or udp_cfg.get("port", 60701))
    recv_timeout = max(0.001, float(control_cfg.get("recv_timeout_sec", 0.01)))

    daemon_host = str(args.daemon_host or daemon_cfg.get("host", "127.0.0.1"))
    daemon_port = int(args.daemon_port or daemon_cfg.get("port", 8080))
    daemon_timeout_s = float(args.daemon_timeout_s or daemon_cfg.get("timeout_s", 1.5))

    command_rate_hz = max(0.1, float(control_cfg.get("command_rate_hz", 25.0)))
    min_send_interval_sec = max(0.0, float(control_cfg.get("min_send_interval_sec", 0.0)))
    idle_timeout_s = max(0.05, float(control_cfg.get("idle_timeout_sec", 0.25)))
    send_on_change_only = bool(control_cfg.get("send_on_change_only", True))
    keepalive_sec = max(0.0, float(control_cfg.get("keepalive_sec", 1.0)))
    command_trigger_delta_deg = max(0.0, float(control_cfg.get("command_trigger_delta_deg", 0.0)))
    command_trigger_delta_tenths = int(round(command_trigger_delta_deg * 10.0))
    median_window = max(1, int(control_cfg.get("median_window", 5)))
    input_deadband_deg = max(0.0, float(control_cfg.get("input_deadband_deg", 0.6)))
    joint_deadband_deg = max(0.0, float(control_cfg.get("joint_deadband_deg", 0.6)))
    max_joint_speed_deg_s = max(0.0, float(control_cfg.get("max_joint_speed_deg_s", 200.0)))
    input_mode = str(control_cfg.get("input_mode", "absolute")).strip().lower()
    if input_mode not in {"absolute", "incremental"}:
        raise SystemExit("invalid control.input_mode: %s (expected absolute|incremental)" % input_mode)
    incremental_wrap_deg = max(0.0, float(control_cfg.get("incremental_wrap_deg", 360.0)))
    incremental_deadband_deg = max(0.0, float(control_cfg.get("incremental_deadband_deg", 0.5)))
    incremental_max_step_deg = max(0.0, float(control_cfg.get("incremental_max_step_deg", 3.0)))
    incremental_jump_guard_deg = max(0.0, float(control_cfg.get("incremental_jump_guard_deg", 120.0)))
    busy_mode = bool(control_cfg.get("busy_mode", True))
    complete_pos_tol_deg = max(0.0, float(control_cfg.get("complete_pos_tol_deg", 2.0)))
    complete_vel_tol_deg_s = max(0.0, float(control_cfg.get("complete_vel_tol_deg_s", 3.0)))
    complete_stable_sec = max(0.0, float(control_cfg.get("complete_stable_sec", 0.25)))
    busy_timeout_sec = max(0.1, float(control_cfg.get("busy_timeout_sec", 3.0)))
    append_retry_delay_sec = max(0.01, float(control_cfg.get("append_retry_delay_sec", 0.15)))
    device_timeout_retry_delay_sec = max(0.05, float(control_cfg.get("device_timeout_retry_delay_sec", 0.6)))
    max_consecutive_device_timeouts = max(1, int(control_cfg.get("max_consecutive_device_timeouts", 8)))

    order = ["J1", "J2", "J3", "J4", "J5"]
    joint_specs: Dict[str, JointMapSpec] = {}
    for name in order:
        item = mapping_cfg.get(name) or {}
        joint_specs[name] = JointMapSpec(
            source_channel=int(item.get("source_channel", 1)),
            source_min=float(item.get("source_min", item.get("source_min_deg", 0.0))),
            source_max=float(item.get("source_max", item.get("source_max_deg", 360.0))),
            target_min_deg=float(item.get("target_min_deg", -90.0)),
            target_max_deg=float(item.get("target_max_deg", 90.0)),
            invert=bool(item.get("invert", False)),
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(recv_timeout)
    sock.bind((event_host, event_port))

    if args.calibrate_sec > 0.0:
        print("[calib] starting %.0f-second range calibration on UDP %s:%d" % (args.calibrate_sec, event_host, event_port))
        calib_ranges = run_range_calibration(sock, duration_sec=args.calibrate_sec, recv_timeout=recv_timeout)
        for name in order:
            spec = joint_specs[name]
            ch_range = calib_ranges.get(spec.source_channel)
            if ch_range is None:
                continue
            span = ch_range["max"] - ch_range["min"]
            if span < 1.0:
                print("[calib] Enc%d span=%.2f too small, skipping" % (spec.source_channel, span))
                continue
            spec.source_min = ch_range["min"]
            spec.source_max = ch_range["max"]
            print("[calib] %s <- Enc%d  source_min=%.2f  source_max=%.2f" % (name, spec.source_channel, spec.source_min, spec.source_max))
        print("[calib] calibration applied, starting teleop...")

    print("[hw-teleop] listening UDP %s:%d" % (event_host, event_port))
    print("[hw-teleop] daemon target %s:%d dry_run=%s" % (daemon_host, daemon_port, bool(args.dry_run)))
    print(
        "[hw-teleop] command_rate_hz=%.2f min_send_interval_sec=%.3f idle_timeout_s=%.3f send_on_change_only=%s keepalive_sec=%.2f"
        % (command_rate_hz, min_send_interval_sec, idle_timeout_s, bool(send_on_change_only), keepalive_sec)
    )
    print(
        "[hw-teleop] trigger_delta_deg=%.2f (tenths=%d)"
        % (command_trigger_delta_deg, command_trigger_delta_tenths)
    )
    print(
        "[hw-teleop] input_mode=%s incr_wrap=%.1f incr_deadband=%.2f incr_max_step=%.2f incr_jump_guard=%.1f"
        % (
            input_mode,
            incremental_wrap_deg,
            incremental_deadband_deg,
            incremental_max_step_deg,
            incremental_jump_guard_deg,
        )
    )
    print(
        "[hw-teleop] busy_mode=%s complete_pos_tol_deg=%.2f complete_vel_tol_deg_s=%.2f complete_stable_sec=%.2f busy_timeout_sec=%.2f"
        % (bool(busy_mode), complete_pos_tol_deg, complete_vel_tol_deg_s, complete_stable_sec, busy_timeout_sec)
    )
    print(
        "[hw-teleop] retry append=%.3fs device_timeout=%.3fs max_device_timeouts=%d"
        % (append_retry_delay_sec, device_timeout_retry_delay_sec, max_consecutive_device_timeouts)
    )

    client: Optional[MechHandV3DaemonClient] = None
    if not args.dry_run:
        client = MechHandV3DaemonClient(host=daemon_host, port=daemon_port, timeout_s=daemon_timeout_s)
        client.connect()
        if not client.query_connected():
            raise RuntimeError("daemon reachable but hand is not connected (CC=0)")

        speed = _values_from_named_list(daemon_cfg.get("speed", [60, 60, 60, 60, 60]), 60)
        force = _values_from_named_list(daemon_cfg.get("force", [50, 50, 50, 50, 50]), 50)
        runtime = _values_from_named_list(daemon_cfg.get("running_time", [30, 30, 30, 30, 30]), 30)
        client.set_speed(speed)
        client.set_force(force)
        client.set_running_time(runtime)
        print("[hw-teleop] daemon configured: SV=%s SF=%s SRT=%s" % (speed, force, runtime))

    histories: Dict[int, Deque[float]] = {}
    last_filtered: Dict[int, float] = {}
    last_event_time = 0.0
    last_send = 0.0
    retry_after = 0.0
    last_keepalive = 0.0
    last_print = 0.0
    event_count = 0
    send_count = 0
    last_sent_tenths: Optional[List[int]] = None
    consecutive_device_timeouts = 0
    prev_measured_tenths: Optional[List[int]] = None
    prev_measured_time = 0.0

    busy = False
    busy_started = 0.0
    stable_since: Optional[float] = None
    inflight_tenths: Optional[List[int]] = None
    queued_tenths: Optional[List[int]] = None

    desired_deg = {name: 0.0 for name in order}
    applied_deg = {name: 0.0 for name in order}
    prev_input_angles: Dict[int, float] = {}
    last_apply_time = time.monotonic()

    try:
        while True:
            now = time.monotonic()

            for _ in range(16):
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    break
                except BlockingIOError:
                    break

                try:
                    event = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue

                raw = parse_encoder_angles(event)
                if len(raw) < 5:
                    continue

                last_filtered = filter_angles(
                    raw_angles=raw,
                    histories=histories,
                    prev_filtered=last_filtered,
                    median_window=median_window,
                    input_deadband_deg=input_deadband_deg,
                )
                last_event_time = now
                event_count += 1

            if now - last_event_time > idle_timeout_s:
                if now - last_print > (1.0 / max(0.1, float(args.print_every))):
                    last_print = now
                    print("[hw-teleop] waiting fresh encoder stream... events=%d sends=%d" % (event_count, send_count))
                continue

            if len(last_filtered) < 5:
                continue

            if input_mode == "absolute":
                for name in order:
                    spec = joint_specs[name]
                    source_angle = last_filtered.get(spec.source_channel)
                    if source_angle is None:
                        continue
                    desired_deg[name] = map_channel_to_joint_deg(source_angle, spec)
            else:
                for name in order:
                    spec = joint_specs[name]
                    ch = spec.source_channel
                    source_angle = last_filtered.get(ch)
                    if source_angle is None:
                        continue
                    prev_angle = prev_input_angles.get(ch)
                    prev_input_angles[ch] = float(source_angle)
                    if prev_angle is None:
                        continue

                    delta_source = unwrap_delta_deg(source_angle, prev_angle, incremental_wrap_deg)
                    if incremental_jump_guard_deg > 0.0 and abs(delta_source) > incremental_jump_guard_deg:
                        continue
                    if abs(delta_source) < incremental_deadband_deg:
                        continue

                    delta_joint = map_channel_delta_to_joint_deg(delta_source, spec)
                    if incremental_max_step_deg > 0.0:
                        delta_joint = clamp(delta_joint, -incremental_max_step_deg, incremental_max_step_deg)
                    desired_deg[name] = clamp(
                        desired_deg[name] + delta_joint,
                        spec.target_min_deg,
                        spec.target_max_deg,
                    )

            dt = now - last_apply_time
            last_apply_time = now
            applied_deg = apply_joint_limits(
                desired=desired_deg,
                prev=applied_deg,
                joint_deadband_deg=joint_deadband_deg,
                max_joint_speed_deg_s=max_joint_speed_deg_s,
                dt=dt,
            )

            values_tenths = [degrees_to_tenths(applied_deg[name], clamp=True) for name in order]

            measured_tenths: Optional[List[int]] = None
            measured_vel_tenths_s = [0.0] * 5
            if len(last_filtered) >= 5:
                measured_deg = []
                for name in order:
                    spec = joint_specs[name]
                    source_angle = last_filtered.get(spec.source_channel)
                    if source_angle is None:
                        measured_deg = []
                        break
                    measured_deg.append(map_channel_to_joint_deg(source_angle, spec))
                if len(measured_deg) == 5:
                    measured_tenths = [degrees_to_tenths(v, clamp=True) for v in measured_deg]
                    if prev_measured_tenths is not None and prev_measured_time > 0.0:
                        dtm = max(1e-6, now - prev_measured_time)
                        measured_vel_tenths_s = [
                            abs((measured_tenths[i] - prev_measured_tenths[i]) / dtm) for i in range(5)
                        ]
                    prev_measured_tenths = list(measured_tenths)
                    prev_measured_time = now

            was_busy = busy
            if busy and inflight_tenths is not None and measured_tenths is not None:
                pos_ok = all(abs(measured_tenths[i] - inflight_tenths[i]) <= int(round(complete_pos_tol_deg * 10.0)) for i in range(5))
                vel_ok = all(v <= (complete_vel_tol_deg_s * 10.0) for v in measured_vel_tenths_s)
                if pos_ok and vel_ok:
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= complete_stable_sec:
                        busy = False
                        inflight_tenths = None
                        stable_since = None
                else:
                    stable_since = None

            if busy and (now - busy_started) >= busy_timeout_sec:
                print("[hw-teleop] WARN busy timeout, releasing gate")
                busy = False
                inflight_tenths = None
                stable_since = None
            just_released_busy = was_busy and (not busy)

            changed = (last_sent_tenths is None) or (values_tenths != last_sent_tenths)
            trigger_changed = changed
            if (last_sent_tenths is not None) and (command_trigger_delta_tenths > 0):
                max_abs_delta = max(abs(values_tenths[i] - last_sent_tenths[i]) for i in range(5))
                trigger_changed = max_abs_delta >= command_trigger_delta_tenths
            keepalive_due = keepalive_sec > 0.0 and (now - last_keepalive) >= keepalive_sec
            should_queue = trigger_changed or (not send_on_change_only) or keepalive_due

            if should_queue:
                # Latest-only policy: overwrite pending action with the newest one.
                queued_tenths = list(values_tenths)

            if queued_tenths is None:
                continue

            if busy_mode and busy:
                continue

            if now < retry_after:
                continue

            send_interval = max(1.0 / command_rate_hz, min_send_interval_sec)
            if (not just_released_busy) and ((now - last_send) < send_interval):
                continue

            if args.dry_run:
                to_send = list(queued_tenths)
            else:
                assert client is not None
                to_send = list(queued_tenths)
                try:
                    client.move_rm_tenths(*to_send)
                except MechHandV3CommandError as exc:
                    text = str(exc)
                    if "append_task_failed" in text:
                        busy = True if busy_mode else False
                        busy_started = now
                        retry_after = now + append_retry_delay_sec
                        queued_tenths = None
                        print("[hw-teleop] append_task_failed, backoff %.3fs" % append_retry_delay_sec)
                        continue
                    if "device_timeout" in text:
                        consecutive_device_timeouts += 1
                        retry_after = now + device_timeout_retry_delay_sec
                        busy = True if busy_mode else False
                        busy_started = now
                        stable_since = None
                        inflight_tenths = None
                        queued_tenths = None
                        print(
                            "[hw-teleop] device_timeout, backoff %.3fs (consecutive=%d)"
                            % (device_timeout_retry_delay_sec, consecutive_device_timeouts)
                        )
                        if consecutive_device_timeouts >= max_consecutive_device_timeouts:
                            raise RuntimeError(
                                "too many consecutive device_timeout errors (%d); check mapping range/speed/SRT"
                                % consecutive_device_timeouts
                            )
                        continue
                    raise
                consecutive_device_timeouts = 0
                if busy_mode:
                    busy = True
                    busy_started = now
                    stable_since = None
                    inflight_tenths = list(to_send)
                values_tenths = list(to_send)

            last_sent_tenths = list(values_tenths)
            queued_tenths = None
            send_count += 1
            last_send = now
            last_keepalive = now

            if now - last_print > (1.0 / max(0.1, float(args.print_every))):
                last_print = now
                info = " ".join("%s=%d" % (name, values_tenths[i]) for i, name in enumerate(order))
                print("[hw-teleop] events=%d sends=%d %s" % (event_count, send_count, info))

    except KeyboardInterrupt:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass
        if client is not None:
            client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
