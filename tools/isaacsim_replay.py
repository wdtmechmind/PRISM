#!/usr/bin/env python3
"""Isaac Sim 5.0 kinematic replay of PRISM rigid 6D pose demonstrations.

This is a placeholder replay: the dex hand base 6D pose from
``rigid_pose_6d.csv`` is played back in Isaac Sim on a stand-in geometry
(a semi-transparent palm box plus an RGB axis triad), the LED trajectories
are drawn as colored polylines, and hand gesture commands are logged on the
timeline. No hand URDF/USD and no gesture->joint mapping are required yet;
swap the placeholder for the real hand articulation once those are available.

Pose convention (must match PRISM reconstruction):
    RPY are ZYX Euler angles in degrees, R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
    columns of R are the body axes expressed in the world frame. The world
    frame is the charuco calibration frame, already Z-up (matches Isaac Sim).

Run with the Isaac Sim bundled Python (do NOT use the project conda env):

    /isaac-sim/python.sh tools/isaacsim_replay.py \
        --trial-dir data/raw/task_20260723_180201_grasp-demo/trial_000001

    /isaac-sim/python.sh tools/isaacsim_replay.py \
        --rigid data/raw/task_20260723_180201_grasp-demo/rigid_pose_6d.csv \
        --speed 1.0 --loop
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np

# --- LED name -> display color (RGB, 0..1) -------------------------------
LED_COLORS = {
    "red": (1.0, 0.05, 0.05),
    "yellow": (1.0, 0.9, 0.05),
    "blue": (0.1, 0.3, 1.0),
    "green": (0.1, 0.9, 0.2),
}


# =========================================================================
# Data loading (pure Python / numpy, no Isaac imports here)
# =========================================================================
def _rotation_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll); columns are body axes in world."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (world = R @ local) -> unit quaternion [w, x, y, z]."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> rotation matrix (world = R @ local)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Spherical linear interpolation between two [w,x,y,z] quaternions."""
    d = float(np.dot(q0, q1))
    if d < 0.0:  # take shortest path
        q1 = -q1
        d = -d
    if d > 0.9995:  # nearly parallel -> linear
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = math.acos(max(-1.0, min(1.0, d)))
    theta = theta0 * a
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return s0 * q0 + s1 * q1


def _pick(colnames, *candidates):
    for c in candidates:
        if c in colnames:
            return c
    return None


def load_poses(path: str, prefer_smoothed: bool = True):
    """Load rigid 6D poses -> (t[N], pos[N,3], quat[N,4])."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        use_sm = prefer_smoothed and ("x_smooth_m" in cols) and ("roll_smooth_deg" in cols)
        cx = "x_smooth_m" if use_sm else "x_m"
        cy = "y_smooth_m" if use_sm else "y_m"
        cz = "z_smooth_m" if use_sm else "z_m"
        cr = "roll_smooth_deg" if use_sm else "roll_deg"
        cp = "pitch_smooth_deg" if use_sm else "pitch_deg"
        cyaw = "yaw_smooth_deg" if use_sm else "yaw_deg"
        ts, pos, quat = [], [], []
        for row in reader:
            try:
                t = float(row["t_sec"])
                p = [float(row[cx]), float(row[cy]), float(row[cz])]
                r = _rotation_zyx(float(row[cr]), float(row[cp]), float(row[cyaw]))
            except (KeyError, ValueError, TypeError):
                continue
            if any(math.isnan(v) for v in p):
                continue
            ts.append(t)
            pos.append(p)
            quat.append(_mat_to_quat(r))
    if not ts:
        raise RuntimeError(f"no valid pose rows in {path}")
    order = np.argsort(ts)
    t_arr = np.asarray(ts)[order]
    pos_arr = np.asarray(pos)[order]
    quat_arr = np.asarray(quat)[order]
    return t_arr, pos_arr, quat_arr, use_sm


def load_leds(path: str):
    """Load long-format LED trajectory -> {color: (M,3) points ordered by t}."""
    if not path or not os.path.isfile(path):
        return {}
    rows_by_color: dict[str, list[tuple[float, list[float]]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        cx = _pick(cols, "x_smooth_m", "x_m")
        cy = _pick(cols, "y_smooth_m", "y_m")
        cz = _pick(cols, "z_smooth_m", "z_m")
        ccolor = _pick(cols, "color")
        if not (cx and cy and cz and ccolor):
            return {}
        for row in reader:
            try:
                t = float(row.get("t_sec", 0.0))
                p = [float(row[cx]), float(row[cy]), float(row[cz])]
            except (ValueError, TypeError):
                continue
            if any(math.isnan(v) for v in p):
                continue
            rows_by_color.setdefault(row[ccolor], []).append((t, p))
    out = {}
    for color, items in rows_by_color.items():
        items.sort(key=lambda kv: kv[0])
        out[color] = np.asarray([p for _, p in items], dtype=np.float64)
    return out


def load_gestures(path: str):
    """Load gesture/command events -> sorted list of (t_sec, label)."""
    if not path or not os.path.isfile(path):
        return []
    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        ct = _pick(cols, "t_sec")
        if not ct:
            return []
        label_cols = [c for c in ("action", "command", "gesture_id", "message") if c in cols]
        for row in reader:
            try:
                t = float(row[ct])
            except (ValueError, TypeError, KeyError):
                continue
            parts = [f"{c}={row[c]}" for c in label_cols if str(row.get(c, "")).strip()]
            if not parts:
                continue
            events.append((t, " ".join(parts)))
    events.sort(key=lambda kv: kv[0])
    return events


def resolve_inputs(args):
    rigid, led, gestures = args.rigid, args.led, args.gestures
    if args.trial_dir:
        d = args.trial_dir
        rigid = rigid or os.path.join(d, "trajectory", "rigid_pose_6d.csv")
        led = led or os.path.join(d, "trajectory", "trajectory_led.csv")
        gestures = gestures or os.path.join(d, "hand", "sdk_commands.csv")
    if not rigid or not os.path.isfile(rigid):
        raise SystemExit(f"rigid pose CSV not found: {rigid!r} (use --rigid or --trial-dir)")
    return rigid, led, gestures


# =========================================================================
# Scene sampling helpers
# =========================================================================
def sample_pose(t_arr, pos_arr, quat_arr, t):
    """Interpolate (pos, quat) at time t; clamp outside the recorded range."""
    if t <= t_arr[0]:
        return pos_arr[0], quat_arr[0]
    if t >= t_arr[-1]:
        return pos_arr[-1], quat_arr[-1]
    i = int(np.searchsorted(t_arr, t) - 1)
    i = max(0, min(i, len(t_arr) - 2))
    t0, t1 = t_arr[i], t_arr[i + 1]
    a = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
    pos = pos_arr[i] * (1.0 - a) + pos_arr[i + 1] * a
    quat = _slerp(quat_arr[i], quat_arr[i + 1], a)
    return pos, quat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial-dir", help="trial dir containing trajectory/ and hand/ (fills defaults)")
    ap.add_argument("--rigid", help="path to rigid_pose_6d.csv")
    ap.add_argument("--led", help="path to trajectory_led.csv (long format)")
    ap.add_argument("--gestures", help="path to sdk_commands.csv / gesture timeline CSV")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--fps", type=float, default=60.0, help="render frame-rate cap (0 = uncapped)")
    ap.add_argument("--loop", action="store_true", help="loop playback")
    ap.add_argument("--headless", action="store_true", help="run without a window")
    ap.add_argument("--no-leds", action="store_true", help="do not draw LED trajectories")
    ap.add_argument("--no-smoothed", action="store_true", help="use raw (unsmoothed) pose columns")
    args = ap.parse_args()

    rigid_path, led_path, gest_path = resolve_inputs(args)
    t_arr, pos_arr, quat_arr, used_sm = load_poses(rigid_path, prefer_smoothed=not args.no_smoothed)
    leds = {} if args.no_leds else load_leds(led_path)
    gestures = load_gestures(gest_path)

    dur = float(t_arr[-1] - t_arr[0])
    print(f"[replay] poses: {len(t_arr)} samples, {dur:.2f}s, smoothed={used_sm}")
    print(f"[replay] LED colors: {sorted(leds.keys()) or 'none'}")
    print(f"[replay] gesture events: {len(gestures)}")

    # --- Start Isaac Sim before importing omni/pxr -----------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2500.0)

    def make_box(path, size_xyz, color, opacity=1.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)  # unit cube [-0.5, 0.5]; scaled by xform
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cube.CreateDisplayOpacityAttr([float(opacity)])
        xf = UsdGeom.Xformable(cube)
        xf.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
        return cube

    def make_axis_triad(root_path, length=0.12, thick=0.01):
        """RGB axis boxes under an Xform root; returns the root Xform prim."""
        root = UsdGeom.Xform.Define(stage, root_path)
        specs = [
            ("X", (length, thick, thick), (length / 2, 0, 0), (1.0, 0.1, 0.1)),
            ("Y", (thick, length, thick), (0, length / 2, 0), (0.1, 1.0, 0.1)),
            ("Z", (thick, thick, length), (0, 0, length / 2), (0.2, 0.4, 1.0)),
        ]
        for name, size, off, color in specs:
            cube = UsdGeom.Cube.Define(stage, f"{root_path}/axis_{name}")
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(*off))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
        return root

    # World-origin reference triad (static)
    make_axis_triad("/World/OriginFrame", length=0.2, thick=0.006)

    # Hand placeholder: palm box + body axis triad under one movable Xform
    hand_xform = UsdGeom.Xform.Define(stage, "/World/Hand")
    hand_op = hand_xform.MakeMatrixXform()  # single transform op we set each frame
    make_box("/World/Hand/palm", (0.09, 0.06, 0.02), (0.6, 0.6, 0.65), opacity=0.55)
    make_axis_triad("/World/Hand/body_frame", length=0.1, thick=0.008)

    # LED trajectories as linear polylines
    for color, pts in leds.items():
        if len(pts) < 2:
            continue
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/LED_{color}")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
        curve.CreateWidthsAttr([0.004] * len(pts))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        rgb = LED_COLORS.get(color, (0.8, 0.8, 0.8))
        curve.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def set_hand(pos, quat):
        r = _quat_to_mat(quat)
        rt = r.T  # USD uses row-vector convention: 3x3 block = R^T
        m = Gf.Matrix4d(
            rt[0, 0], rt[0, 1], rt[0, 2], 0.0,
            rt[1, 0], rt[1, 1], rt[1, 2], 0.0,
            rt[2, 0], rt[2, 1], rt[2, 2], 0.0,
            float(pos[0]), float(pos[1]), float(pos[2]), 1.0,
        )
        hand_op.Set(m)

    # Warm up a few frames so the scene is visible before playback timing
    set_hand(pos_arr[0], quat_arr[0])
    for _ in range(5):
        simulation_app.update()

    t0 = float(t_arr[0])
    frame_dt = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    print("[replay] playing... (close the window or Ctrl-C to stop)")
    while simulation_app.is_running():
        start_wall = time.time()
        next_frame = time.time()
        g_idx = 0
        while simulation_app.is_running():
            now = t0 + (time.time() - start_wall) * args.speed
            pos, quat = sample_pose(t_arr, pos_arr, quat_arr, now)
            set_hand(pos, quat)
            while g_idx < len(gestures) and gestures[g_idx][0] <= now:
                gt, label = gestures[g_idx]
                print(f"[gesture] t={gt:.3f}s  {label}")
                g_idx += 1
            simulation_app.update()
            # Cap render rate so we do not busy-spin and saturate the CPU.
            if frame_dt > 0.0:
                next_frame += frame_dt
                sleep_t = next_frame - time.time()
                if sleep_t > 0.0:
                    time.sleep(sleep_t)
                else:
                    next_frame = time.time()
            if now >= t_arr[-1]:
                break
        if not args.loop:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
