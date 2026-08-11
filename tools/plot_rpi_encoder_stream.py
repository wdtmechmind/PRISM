#!/usr/bin/env python3
"""Live visualization of RPi encoder angles over UDP.

Listens for prism.rpi_hand_event.v1 style packets and plots Enc1..Enc5 angles
versus time in a rolling window.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="UDP bind host")
    parser.add_argument("--port", type=int, default=60701, help="UDP bind port")
    parser.add_argument("--window-sec", type=float, default=20.0, help="Rolling time window in seconds")
    parser.add_argument("--max-points", type=int, default=6000, help="Ring buffer size")
    parser.add_argument("--refresh-ms", type=int, default=50, help="Plot refresh interval in milliseconds")
    parser.add_argument("--print-hz", type=float, default=1.0, help="Status print rate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.window_sec <= 0:
        raise SystemExit("--window-sec must be > 0")
    if args.max_points < 10:
        raise SystemExit("--max-points must be >= 10")
    if args.refresh_ms < 10:
        raise SystemExit("--refresh-ms must be >= 10")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((args.host, args.port))

    print("[plot-rpi] listening UDP %s:%d" % (args.host, args.port))
    print("[plot-rpi] window_sec=%.1f max_points=%d refresh_ms=%d" % (args.window_sec, args.max_points, args.refresh_ms))

    start_t = time.monotonic()
    t_hist: Deque[float] = deque(maxlen=args.max_points)
    angle_hist: Dict[int, Deque[float]] = {ch: deque(maxlen=args.max_points) for ch in range(1, 6)}
    last_values = {ch: float("nan") for ch in range(1, 6)}

    event_count = 0
    last_packet_t = 0.0
    last_print_t = 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
    lines: Dict[int, any] = {}
    for ch, color in zip(range(1, 6), colors):
        (line,) = ax.plot([], [], lw=1.8, color=color, label="Enc%d" % ch)
        lines[ch] = line

    ax.set_title("RPi Encoder Angles (Live UDP Stream)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    status_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
    )

    def _drain_socket(now: float) -> None:
        nonlocal event_count, last_packet_t
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break

            try:
                event = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(event, dict):
                continue

            angles = parse_encoder_angles(event)
            if not angles:
                continue

            t_rel = now - start_t
            t_hist.append(t_rel)
            for ch in range(1, 6):
                val = angles.get(ch, last_values[ch])
                if val == val:
                    last_values[ch] = val
                angle_hist[ch].append(val)

            event_count += 1
            last_packet_t = now

    def _trim_window() -> None:
        if not t_hist:
            return
        t_right = t_hist[-1]
        t_left = max(0.0, t_right - args.window_sec)
        while t_hist and t_hist[0] < t_left:
            t_hist.popleft()
            for ch in range(1, 6):
                if angle_hist[ch]:
                    angle_hist[ch].popleft()

    def _update_status(now: float) -> None:
        nonlocal last_print_t
        age = now - last_packet_t if last_packet_t > 0.0 else float("inf")
        if age == float("inf"):
            stream_state = "no data yet"
        elif age > 1.0:
            stream_state = "stale (%.2fs)" % age
        else:
            stream_state = "fresh (%.2fs)" % age

        latest = " ".join("E%d=%s" % (ch, ("%.2f" % last_values[ch] if last_values[ch] == last_values[ch] else "nan")) for ch in range(1, 6))
        status_text.set_text("events=%d  stream=%s\n%s" % (event_count, stream_state, latest))

        if now - last_print_t >= (1.0 / max(0.1, float(args.print_hz))):
            last_print_t = now
            print("[plot-rpi] events=%d stream=%s %s" % (event_count, stream_state, latest))

    def _animate(_frame_index: int):
        now = time.monotonic()
        _drain_socket(now)
        _trim_window()
        _update_status(now)

        if not t_hist:
            return list(lines.values()) + [status_text]

        x = list(t_hist)
        y_min = float("inf")
        y_max = float("-inf")
        for ch in range(1, 6):
            y = list(angle_hist[ch])
            lines[ch].set_data(x, y)
            for v in y:
                if v == v:
                    y_min = min(y_min, v)
                    y_max = max(y_max, v)

        x_right = x[-1]
        x_left = max(0.0, x_right - args.window_sec)
        ax.set_xlim(x_left, x_right + 1e-6)

        if y_min < y_max:
            pad = max(2.0, 0.08 * (y_max - y_min))
            ax.set_ylim(y_min - pad, y_max + pad)

        return list(lines.values()) + [status_text]

    def _on_close(_event) -> None:
        try:
            sock.close()
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", _on_close)
    _ani = FuncAnimation(fig, _animate, interval=args.refresh_ms, blit=False)
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
