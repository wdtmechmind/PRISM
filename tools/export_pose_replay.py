#!/usr/bin/env python3
"""Export a Cartesian end-effector trajectory for replay on any arm.

Emits the MechHand ``base_link`` pose in the robot base frame, so the arm-side
machine runs its own IK. No URDF, no arm model, no joint angles.

Poses are given both as quaternions and as UR-style rotation vectors, so a UR
controller can consume ``[x, y, z, rx, ry, rz]`` directly.

Example:
    python tools/export_pose_replay.py <trial>/corrected_trajectory.csv \
        --base-from-world configs/deployment/base_from_world_<task>.json \
        --gestures data/raw/<task>/<trial>/hand/sdk_commands.csv \
        --max-reach 0.45 --min-reach 0.18 \
        --out-dir replay_bundle_<trial>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from prism.devices.hand.socket_client import GESTURE_ID_TO_POSE, POSE_TO_GESTURE_ID  # noqa: E402
from prism.reconstruction.hand_frame import load_spec  # noqa: E402


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """UR-style rotation vector: axis scaled by angle."""
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    angle = 2.0 * float(np.arccos(np.clip(q[0], -1.0, 1.0)))
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return q[1:] / s * angle


def align_signs(quats: np.ndarray) -> np.ndarray:
    """Remove the q/-q ambiguity so interpolation and averaging stay continuous."""
    out = quats.copy()
    for i in range(1, len(out)):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    return out


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1.0, 1.0))
    th = th0 * u
    q2 = q1 - q0 * d
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def resample(t: np.ndarray, pos: np.ndarray, quat: np.ndarray, grid: np.ndarray):
    p = np.column_stack([np.interp(grid, t, pos[:, i]) for i in range(3)])
    q = np.empty((len(grid), 4), dtype=np.float64)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    for k, (g, i) in enumerate(zip(grid, idx)):
        span = t[i + 1] - t[i]
        u = 0.0 if span <= 0 else float((g - t[i]) / span)
        q[k] = slerp(quat[i], quat[i + 1], min(max(u, 0.0), 1.0))
    return p, q


def smooth(pos: np.ndarray, quat: np.ndarray, win: int):
    if win <= 1:
        return pos, quat
    kern = np.ones(win) / win
    pad = win // 2
    sp = np.empty_like(pos)
    for i in range(3):
        sp[:, i] = np.convolve(np.pad(pos[:, i], (pad, pad), mode="edge"), kern, mode="valid")[:len(pos)]
    sq = np.empty_like(quat)
    padded = np.pad(quat, ((pad, pad), (0, 0)), mode="edge")
    for i in range(len(quat)):
        w = padded[i:i + win]
        avg = w.mean(axis=0)
        n = np.linalg.norm(avg)
        sq[i] = quat[i] if n < 1e-9 else avg / n
    return sp, sq


def load_trajectory(path: Path):
    t, pos, quat = [], [], []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                p = [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]
                q = [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]
                ts = float(r["t_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(p).all() and np.isfinite(q).all():
                t.append(ts)
                pos.append(p)
                quat.append(q)
    if not t:
        raise SystemExit("no usable rows in %s" % path)
    q = np.asarray(quat, dtype=np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.asarray(t), np.asarray(pos), align_signs(q)


def load_gestures(path: Optional[Path]):
    if path is None or not path.is_file():
        return []
    events = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        tcol = "trial_time" if "trial_time" in cols else "t_sec"
        for r in reader:
            pose = str(r.get("action", "")).strip()
            if ":" in pose:
                pose = pose.rsplit(":", 1)[1].strip()
            gid = POSE_TO_GESTURE_ID.get(pose)
            if gid is None:
                cmd = str(r.get("command", "")).strip()
                if cmd.startswith("@ROG<") and cmd.endswith(">&"):
                    try:
                        gid = int(cmd[5:-2]) + 1
                        pose = GESTURE_ID_TO_POSE.get(gid, "")
                    except ValueError:
                        gid = None
            if gid is None:
                continue
            try:
                events.append((float(r[tcol]), gid, pose))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(events)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corrected_trajectory")
    p.add_argument("--base-from-world", required=True)
    p.add_argument("--gestures", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rate-hz", type=float, default=125.0)
    p.add_argument("--smooth-s", type=float, default=0.08)
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="0.5 replays at half speed and halves Cartesian rates")
    p.add_argument("--scale", type=float, default=1.0,
                   help="shrink the motion about its centroid to fit a shorter-reach arm")
    p.add_argument("--min-reach", type=float, default=0.18)
    p.add_argument("--max-reach", type=float, default=0.45)
    p.add_argument("--max-linear-speed", type=float, default=0.25, help="m/s")
    p.add_argument("--max-angular-speed", type=float, default=1.0, help="rad/s")
    p.add_argument("--force", action="store_true", help="export despite failed checks (NOT for hardware)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t, pos_w, quat_w = load_trajectory(Path(args.corrected_trajectory).expanduser().resolve())
    bfw = np.asarray(json.load(Path(args.base_from_world).expanduser().resolve().open())["base_from_world"],
                     dtype=np.float64).reshape(4, 4)
    rot_bw, trans_bw = bfw[:3, :3], bfw[:3, 3]

    pos = (rot_bw @ pos_w.T).T + trans_bw
    quat = align_signs(np.array([mat_to_quat(rot_bw @ quat_to_mat(q)) for q in quat_w]))

    if args.scale != 1.0:
        centre = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
        pos = centre + (pos - centre) * float(args.scale)

    grid = np.arange(float(t[0]), float(t[-1]), 1.0 / args.rate_hz) if args.rate_hz > 0 else t
    p_out, q_out = resample(t, pos, quat, grid)
    win = max(1, int(round(args.smooth_s * args.rate_hz))) if args.rate_hz > 0 else 1
    p_out, q_out = smooth(p_out, q_out, win)
    q_out = align_signs(q_out)
    t_out = grid[0] + (grid - grid[0]) / float(args.time_scale)

    reach = np.linalg.norm(p_out, axis=1)
    dt = np.diff(t_out)
    dt[dt <= 0] = np.nan
    lin = np.nanmax(np.linalg.norm(np.diff(p_out, axis=0), axis=1) / dt)
    dots = np.abs(np.sum(q_out[1:] * q_out[:-1], axis=1)).clip(-1.0, 1.0)
    ang = np.nanmax(2.0 * np.arccos(dots) / dt)

    problems = []
    n_far = int((reach > args.max_reach).sum())
    n_near = int((reach < args.min_reach).sum())
    if n_far:
        problems.append("%d/%d points beyond max reach %.2f m (max %.3f)"
                        % (n_far, len(reach), args.max_reach, reach.max()))
    if n_near:
        problems.append("%d/%d points inside min reach %.2f m (min %.3f)"
                        % (n_near, len(reach), args.min_reach, reach.min()))
    if lin > args.max_linear_speed:
        problems.append("peak linear speed %.3f m/s exceeds %.3f" % (lin, args.max_linear_speed))
    if ang > args.max_angular_speed:
        problems.append("peak angular speed %.3f rad/s exceeds %.3f" % (ang, args.max_angular_speed))

    centre = (p_out.max(axis=0) + p_out.min(axis=0)) / 2.0
    envelope = float(np.linalg.norm(p_out - centre, axis=1).max())

    print("[export] frames %d -> %d @ %.0f Hz, duration %.1f s (time scale %.2fx, spatial scale %.2fx)"
          % (len(t), len(t_out), args.rate_hz, t_out[-1] - t_out[0], args.time_scale, args.scale))
    print("[export] reach from base: %.3f .. %.3f m   (allowed %.2f .. %.2f)"
          % (reach.min(), reach.max(), args.min_reach, args.max_reach))
    print("[export] motion envelope radius: %.3f m" % envelope)
    print("[export] peak linear %.3f m/s, peak angular %.3f rad/s" % (lin, ang))

    if problems and not args.force:
        print("[export] REFUSING to export:")
        for pr in problems:
            print("    - %s" % pr)
        print("[export] retune --scale / the base_from_world placement, or pass --force for simulation")
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_gestures(Path(args.gestures).expanduser().resolve() if args.gestures else None)
    ev_t = np.asarray([e[0] for e in events]) if events else np.zeros(0)

    with (out_dir / "pose_trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
                    "rx", "ry", "rz", "hand_gesture_id", "hand_rog", "hand_pose_name"])
        for i in range(len(t_out)):
            rv = quat_to_rotvec(q_out[i])
            gid, pose = "", ""
            if len(ev_t):
                k = int(np.searchsorted(ev_t, grid[i], side="right")) - 1
                if k >= 0:
                    gid, pose = events[k][1], events[k][2]
            w.writerow(["%.6f" % t_out[i]] + ["%.6f" % v for v in p_out[i]]
                       + ["%.9f" % v for v in q_out[i]] + ["%.9f" % v for v in rv]
                       + [gid, (gid - 1) if gid != "" else "", pose])

    spec = load_spec()
    manifest = {
        "source_trajectory": str(Path(args.corrected_trajectory).resolve()),
        "frame": {
            "pose_of": "MechHand base_link (the arm-flange mount face)",
            "expressed_in": "robot base frame",
            "position_units": "metres",
            "orientation": "quaternion (qw,qx,qy,qz) and UR rotation vector (rx,ry,rz)",
            "tcp_note": "set the controller TCP to the MechHand base_link offset from the flange; "
                        "if the hand bolts straight onto the flange the TCP offset is zero",
            "mount_correction_deg": spec.get("mount_correction_deg"),
            "mount_correction_axis": spec.get("mount_correction_axis"),
        },
        "trajectory": {
            "frames": len(t_out),
            "rate_hz": args.rate_hz,
            "duration_s": round(float(t_out[-1] - t_out[0]), 3),
            "time_scale": args.time_scale,
            "spatial_scale": args.scale,
            "smoothing_window_s": args.smooth_s,
            "start_pose": {"xyz": [round(float(v), 6) for v in p_out[0]],
                           "quat_wxyz": [round(float(v), 9) for v in q_out[0]],
                           "rotvec": [round(float(v), 9) for v in quat_to_rotvec(q_out[0])],
                           "note": "move here first and confirm before streaming"},
            "reach_m": {"min": round(float(reach.min()), 4), "max": round(float(reach.max()), 4)},
            "envelope_radius_m": round(envelope, 4),
            "peak_linear_speed_m_s": round(float(lin), 4),
            "peak_angular_speed_rad_s": round(float(ang), 4),
        },
        "hand": {
            "gesture_events": [{"t_sec": round(e[0], 3), "gesture_id": e[1], "rog": e[1] - 1, "pose": e[2]}
                               for e in events],
            "protocol": "MechHand socket, raw command @ROG<gesture_id-1>&",
        },
        "checks": {"problems": problems},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[export] gesture events: %d" % len(events))
    print("[export] wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
