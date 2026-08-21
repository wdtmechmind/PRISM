#!/usr/bin/env python3
"""Package a planned motion into a self-contained bundle for real-robot replay.

Produces everything the arm-side machine needs, plus the safety checks that must
pass before anything is streamed to hardware: IK success, joint limits, and
per-joint speed.

Example:
    python tools/export_robot_replay.py <trial>/planned_motion.csv \
        --base-from-world configs/deployment/base_from_world_<task>.json \
        --out-dir /tmp/replay_bundle
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from prism.devices.hand.socket_client import GESTURE_ID_TO_POSE  # noqa: E402
from prism.reconstruction.hand_frame import load_spec  # noqa: E402


def joint_limits(urdf_path: Path, names: List[str]) -> dict:
    root = ET.parse(str(urdf_path)).getroot()
    out = {}
    for j in root.findall("joint"):
        if j.get("name") not in names:
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        out[j.get("name")] = {
            "lower": float(lim.get("lower", "-3.14")),
            "upper": float(lim.get("upper", "3.14")),
            "velocity": float(lim.get("velocity", "3.0")),
        }
    return out


def resample_and_smooth(t: np.ndarray, q: np.ndarray, rate_hz: float, smooth_s: float):
    """Uniformly resample, then moving-average, so vision jitter is not commanded.

    The reconstruction runs near 300 Hz where per-frame noise dominates the
    numerical derivative; the arm only needs the underlying motion.
    """
    if rate_hz <= 0:
        return t, q, 1
    grid = np.arange(float(t[0]), float(t[-1]), 1.0 / rate_hz)
    out = np.column_stack([np.interp(grid, t, q[:, i]) for i in range(q.shape[1])])
    win = max(1, int(round(smooth_s * rate_hz)))
    if win > 1:
        kern = np.ones(win) / win
        pad = win // 2
        smoothed = np.empty_like(out)
        for i in range(out.shape[1]):
            padded = np.pad(out[:, i], (pad, pad), mode="edge")
            smoothed[:, i] = np.convolve(padded, kern, mode="valid")[:len(out)]
        out = smoothed
    return grid, out, win


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("planned_motion")
    p.add_argument("--base-from-world", required=True)
    p.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-joint-speed", type=float, default=None,
                   help="rad/s cap; default is --speed-margin times the URDF velocity limit")
    p.add_argument("--speed-margin", type=float, default=0.5,
                   help="fraction of the URDF velocity limit allowed")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="stretch the timeline; 0.5 replays at half speed and halves joint rates")
    p.add_argument("--rate-hz", type=float, default=125.0,
                   help="resample the joint path to this rate; 0 keeps the source sampling")
    p.add_argument("--smooth-s", type=float, default=0.08,
                   help="moving-average window in seconds applied after resampling")
    p.add_argument("--allow-ik-failures", action="store_true",
                   help="export even though some frames failed IK (NOT for hardware)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan_path = Path(args.planned_motion).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    cfg = yaml.safe_load(Path(args.robot_config).expanduser().resolve().read_text(encoding="utf-8"))
    arm_joints = list(cfg["arm_joints"])

    rows = list(csv.DictReader(plan_path.open("r", newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("no rows in %s" % plan_path)

    t = np.asarray([float(r["t_sec"]) for r in rows], dtype=np.float64)
    q = np.asarray([[float(r[j]) for j in arm_joints] for r in rows], dtype=np.float64)
    ik_ok = np.asarray([int(r.get("ik_success", "1")) for r in rows], dtype=int)
    ik_err = np.asarray([float(r.get("ik_error_m", "0")) for r in rows], dtype=np.float64)

    problems = []
    n_fail = int((ik_ok == 0).sum())
    if n_fail:
        problems.append("%d/%d frames failed IK" % (n_fail, len(rows)))

    urdf = Path(cfg["combined_urdf"])
    if not urdf.is_absolute():
        urdf = _REPO_ROOT / urdf
    limits = joint_limits(urdf, arm_joints)
    limit_hits = {}
    for i, name in enumerate(arm_joints):
        if name not in limits:
            continue
        lo, hi = limits[name]["lower"], limits[name]["upper"]
        n_out = int(((q[:, i] < lo) | (q[:, i] > hi)).sum())
        if n_out:
            limit_hits[name] = n_out
            problems.append("%s: %d frames outside [%.3f, %.3f]" % (name, n_out, lo, hi))

    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    raw_peak = np.nanmax(np.abs(np.diff(q, axis=0)) / dt[:, None], axis=0)

    gesture_src = [(float(r["t_sec"]), r.get("hand_gesture_id", ""), r.get("hand_rog", ""),
                    r.get("hand_pose_name", "")) for r in rows]
    t_out, q_out, win = resample_and_smooth(t, q, args.rate_hz, args.smooth_s)
    if args.time_scale != 1.0:
        t_out = t_out[0] + (t_out - t_out[0]) / float(args.time_scale)
    dt_out = np.diff(t_out)
    dt_out[dt_out <= 0] = np.nan
    speed = np.abs(np.diff(q_out, axis=0)) / dt_out[:, None]
    max_speed = np.nanmax(speed, axis=0)
    caps = {n: (args.max_joint_speed if args.max_joint_speed
                else args.speed_margin * limits.get(n, {"velocity": 3.0})["velocity"])
            for n in arm_joints}
    fast = {arm_joints[i]: float(max_speed[i]) for i in range(len(arm_joints))
            if max_speed[i] > caps[arm_joints[i]]}
    for name, v in fast.items():
        problems.append("%s: peak speed %.2f rad/s exceeds %.2f (URDF limit %.1f)"
                        % (name, v, caps[name], limits.get(name, {"velocity": 3.0})["velocity"]))

    gestures = []
    last = None
    for r in rows:
        gid = r.get("hand_gesture_id", "")
        if gid and gid != last:
            gestures.append({"t_sec": float(r["t_sec"]), "gesture_id": int(gid),
                             "rog": r.get("hand_rog", ""), "pose": r.get("hand_pose_name", "")})
            last = gid

    if problems and not args.allow_ik_failures:
        print("[export] REFUSING to export, unresolved problems:")
        for p in problems:
            print("    - %s" % p)
        print("[export] fix the plan, or re-run with --allow-ik-failures for simulation only")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "joint_trajectory.csv"
    src_t = np.asarray([g[0] for g in gesture_src], dtype=np.float64)
    with traj_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec"] + arm_joints + ["hand_gesture_id", "hand_rog", "hand_pose_name"])
        for i in range(len(t_out)):
            k = int(np.searchsorted(src_t, t_out[i], side="right")) - 1
            k = min(max(k, 0), len(gesture_src) - 1)
            w.writerow(["%.6f" % t_out[i]] + ["%.9f" % v for v in q_out[i]]
                       + [gesture_src[k][1], gesture_src[k][2], gesture_src[k][3]])

    shutil.copy(Path(args.base_from_world).expanduser().resolve(), out_dir / "base_from_world.json")
    spec = load_spec()

    manifest = {
        "source_plan": str(plan_path),
        "robot": {
            "model": "AUBO i5 + MechHand (left hand)",
            "arm_joint_order": arm_joints,
            "joint_units": "radians",
            "time_column": "t_sec",
            "time_units": "seconds, relative to trajectory start",
            "urdf": str(urdf),
        },
        "start_pose": {
            "arm_q": [round(float(v), 9) for v in q_out[0]],
            "note": "jog the arm here and confirm before streaming; the trajectory assumes this exact start",
        },
        "end_pose": {"arm_q": [round(float(v), 9) for v in q_out[-1]]},
        "trajectory": {
            "frames": len(t_out),
            "duration_s": round(float(t_out[-1] - t_out[0]), 3),
            "rate_hz": round(float(args.rate_hz), 2) if args.rate_hz > 0 else "source",
            "resampled_from_frames": len(rows),
            "smoothing_window_s": args.smooth_s,
            "time_scale": args.time_scale,
            "speed_cap_rad_s": {k: round(float(v), 3) for k, v in caps.items()},
            "peak_joint_speed_rad_s": {arm_joints[i]: round(float(max_speed[i]), 4)
                                       for i in range(len(arm_joints))},
            "peak_joint_speed_before_smoothing_rad_s": {arm_joints[i]: round(float(raw_peak[i]), 2)
                                                        for i in range(len(arm_joints))},
        },
        "hand": {
            "gesture_events": gestures,
            "gesture_id_to_pose": {str(k): v for k, v in sorted(GESTURE_ID_TO_POSE.items())},
            "protocol": "MechHand V2/V3 socket, raw command @ROG<gesture_id-1>&",
        },
        "frame_convention": {
            "led_forward": spec["forward_from"],
            "led_right": spec["right_from"],
            "mount_correction_deg": spec.get("mount_correction_deg"),
            "mount_correction_axis": spec.get("mount_correction_axis"),
            "note": "poses were converted from capture-rig mounting to arm-flange mounting",
        },
        "checks": {
            "ik_success": "%d/%d" % (int((ik_ok == 1).sum()), len(rows)),
            "ik_error_m_mean": round(float(ik_err.mean()), 6),
            "ik_error_m_max": round(float(ik_err.max()), 6),
            "joint_limit_violations": limit_hits,
            "problems": problems,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[export] frames %d -> %d after resampling to %s Hz (smooth %.0f ms)"
          % (len(rows), len(t_out), manifest["trajectory"]["rate_hz"], args.smooth_s * 1000))
    print("[export] duration %.2f s (time scale %.2fx)" % (manifest["trajectory"]["duration_s"], args.time_scale))
    print("[export] IK success %s, max error %.4f m" % (manifest["checks"]["ik_success"], ik_err.max()))
    print("[export] peak joint speed: %s"
          % ", ".join("%s=%.2f" % (arm_joints[i], max_speed[i]) for i in range(len(arm_joints))))
    print("[export] gesture events: %d" % len(gestures))
    print("[export] start pose (jog here first): [%s]"
          % ", ".join("%.4f" % v for v in q_out[0]))
    if problems:
        print("[export] WARNING exported with problems: %s" % "; ".join(problems))
    print("[export] wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
