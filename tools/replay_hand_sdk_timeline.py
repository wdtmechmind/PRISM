#!/usr/bin/env python3
"""Replay a timestamped hand SDK timeline to the MechHand controller (V2 / ROG-only SDK).

Input is a CSV produced during collection, e.g.
  data/raw/<task>/hand_sdk_commands_timeline.csv   (has trial_id column)
  data/raw/<task>/trial_XXXXXX/hand/sdk_commands.csv

Expected columns: t_sec, wall_time, [trial_id], [trial_time], action, command
Only ``@ROG<n>&`` style raw commands are sent (V2 SDK has no joint-level RM).

Example:
  python tools/replay_hand_sdk_timeline.py \
      --csv data/raw/task_20260814_100103_test_collection/hand_sdk_commands_timeline.csv \
      --trial-id 1 --ip 127.0.0.1 --port 60686
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism.devices.hand import MechHandClient  # noqa: E402
from prism.devices.hand.socket_client import POSE_TO_COMMAND, POSE_ALIASES  # noqa: E402


@dataclass
class TimedCommand:
    t_sec: float
    command: str
    action: str


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pose_from_action(action: str) -> str:
    """Collection actions look like ``01:grasp`` / ``02:five_open``."""
    name = str(action or "").strip()
    if ":" in name:
        name = name.split(":", 1)[1]
    return name.strip()


def _command_from_row(row: dict) -> Optional[str]:
    raw = str(row.get("command") or "").strip()
    if raw:
        return raw
    pose = _pose_from_action(row.get("action", ""))
    canonical = POSE_ALIASES.get(pose, pose)
    return POSE_TO_COMMAND.get(canonical)


def load_timeline(
    csv_path: Path,
    time_column: str,
    trial_id: Optional[int],
) -> List[TimedCommand]:
    rows: List[TimedCommand] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("empty csv: %s" % csv_path)
        if time_column not in reader.fieldnames:
            raise SystemExit(
                "time column %r not found; available: %s" % (time_column, ", ".join(reader.fieldnames))
            )

        for row in reader:
            if trial_id is not None:
                row_trial = _safe_float(row.get("trial_id"))
                if row_trial is None or int(row_trial) != int(trial_id):
                    continue

            t = _safe_float(row.get(time_column))
            if t is None:
                continue

            command = _command_from_row(row)
            if not command:
                continue

            rows.append(TimedCommand(t_sec=t, command=command, action=str(row.get("action") or "")))

    rows.sort(key=lambda item: item.t_sec)
    if not rows:
        raise SystemExit("no replayable rows found in %s" % csv_path)

    origin = rows[0].t_sec
    for item in rows:
        item.t_sec -= origin
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="path to the timestamped SDK command CSV")
    parser.add_argument("--time-column", default="t_sec", choices=["t_sec", "wall_time", "trial_time"])
    parser.add_argument("--trial-id", type=int, default=None, help="only replay rows of this trial_id")
    parser.add_argument("--ip", default="127.0.0.1", help="MechHand controller IP")
    parser.add_argument("--port", type=int, default=60686, help="MechHand controller TCP port")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor (>1 is faster)")
    parser.add_argument("--start-delay-s", type=float, default=0.0, help="wait before the first command")
    parser.add_argument("--max-gap-s", type=float, default=0.0, help="cap idle time between commands (0=off)")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule without connecting")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        candidate = Path.cwd() / csv_path
        csv_path = candidate if candidate.is_file() else (REPO_ROOT / args.csv)
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit("csv not found: %s" % csv_path)

    speed = float(args.speed)
    if speed <= 0.0:
        raise SystemExit("--speed must be > 0")

    timeline = load_timeline(csv_path, args.time_column, args.trial_id)
    print("[replay] %s rows=%d duration=%.2fs speed=%.2fx" % (csv_path, len(timeline), timeline[-1].t_sec, speed))

    if args.max_gap_s > 0.0:
        compressed = 0.0
        prev_src = 0.0
        for item in timeline:
            gap = min(item.t_sec - prev_src, args.max_gap_s)
            prev_src = item.t_sec
            compressed += gap
            item.t_sec = compressed

    if args.dry_run:
        for item in timeline:
            print("[dry-run] t=%8.3f  %-20s %s" % (item.t_sec, item.action, item.command))
        return 0

    # settle_time_s=0 so scheduling is driven purely by the timeline.
    client = MechHandClient(ip=args.ip, port=args.port, timeout_s=args.timeout_s, settle_time_s=0.0)
    try:
        client.connect()
        print("[replay] connected to %s:%d" % (args.ip, args.port))

        if args.start_delay_s > 0.0:
            time.sleep(args.start_delay_s)

        t0 = time.monotonic()
        for index, item in enumerate(timeline):
            target = t0 + item.t_sec / speed
            while True:
                remaining = target - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.02))

            sent = client.send_raw(item.command)
            elapsed = time.monotonic() - t0
            print(
                "[replay] %3d/%d t=%7.3f (actual %7.3f) %-20s %s"
                % (index + 1, len(timeline), item.t_sec / speed, elapsed, item.action, sent)
            )
    except KeyboardInterrupt:
        print("\n[replay] interrupted")
        return 130
    finally:
        client.close()

    print("[replay] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
